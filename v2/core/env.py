"""Trading environment for the session-level sequential agent.

Wraps the replay loop as a gym-like step interface. Each episode is one
trading day. The environment uses a frozen TradingModel as a feature
extractor and the existing simulator for trade execution.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from v2.core.chain_data import (
    extract_contract_series,
    load_sidecar_cached,
    padded_snapshot,
)
from v2.core.policy import DEFAULT_POLICY, DecisionPolicy
from v2.core.schema import TradeIntent
from v2.core.simulator import simulate_trade


# Actions
ACT_HOLD = 0       # no-trade when flat, hold when in position
ACT_ENTER_CALL = 1
ACT_ENTER_PUT = 2
ACT_EXIT = 3
NUM_ACTIONS = 4

SESSION_STATE_DIM = 12

BARS_PER_DAY = 390


@dataclass
class Observation:
    """What the agent sees at each step."""
    context: np.ndarray          # (d_model,) from frozen encoder
    contract_scores: np.ndarray  # (max_contracts,) from frozen scorer
    valid_mask: np.ndarray       # (max_contracts,) bool
    is_put: np.ndarray           # (max_contracts,) bool
    session_state: np.ndarray    # (SESSION_STATE_DIM,)


@dataclass
class StepInfo:
    """Extra info returned by step()."""
    day: str = ""
    bar: int = 0
    action_taken: int = 0
    action_valid: bool = True
    trade_opened: bool = False
    trade_closed: bool = False
    trade_pnl: float = 0.0
    exit_reason: str = ""
    in_position: bool = False


class TradingEnv:
    """One-day 0DTE trading environment.

    Usage:
        env = TradingEnv(data, policy, sidecar_dir, model, device)
        obs = env.reset(day)
        while True:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
    """

    def __init__(
        self,
        data: dict,
        policy: DecisionPolicy | None = None,
        sidecar_dir: str = "v2/data_sidecars",
        model: torch.nn.Module | None = None,
        device: str = "cpu",
        lookback: int = 30,
    ):
        self.data = data
        self.features = data["X"]
        self.dates = data["dates"]
        self.bar_of_day = data["bar_of_day"]
        self.policy = policy or DEFAULT_POLICY
        self.sidecar_dir = sidecar_dir
        self.model = model
        self.device = device
        self.lookback = lookback
        self.max_contracts = int(data.get("metadata", {}).get("max_contracts_per_bar", 100))

        # Build day → global bar index mapping
        self._day_bars: dict[str, list[int]] = {}
        for gi, d in enumerate(self.dates):
            self._day_bars.setdefault(d, []).append(gi)

        # Episode state (set by reset)
        self._day: str = ""
        self._sidecar: dict = {}
        self._eligible_bars: list[tuple[int, int]] = []  # (global_bar, local_bar)
        self._step_idx: int = 0
        self._done: bool = True

        # Position state
        self._in_position: bool = False
        self._position_entry_bar: int = -1
        self._position_entry_price: float = 0.0
        self._position_contract_idx: int = -1
        self._position_side: int = 0  # 0=flat, 1=call, -1=put
        self._position_mfe: float = 0.0
        self._position_mae: float = 0.0
        self._position_series: np.ndarray = np.array([])
        self._position_intent: TradeIntent | None = None

        # Session state
        self._day_pnl: float = 0.0
        self._num_trades: int = 0
        self._num_stops: int = 0
        self._last_stop_bar: int = -100
        self._starting_equity: float = 0.0
        self._max_drawdown: float = 0.0
        self._peak_equity: float = 0.0
        self._cumulative_equity: float = 0.0

    def reset(self, day: str) -> Observation:
        """Start a new trading day episode."""
        self._day = day
        self._sidecar = load_sidecar_cached(os.path.join(self.sidecar_dir, f"{day}.pt"))

        # Find eligible bars for this day
        global_bars = self._day_bars.get(day, [])
        self._eligible_bars = []
        for gi in global_bars:
            local_bar = int(self.bar_of_day[gi])
            if self.policy.no_trade_before_bar <= local_bar < self.policy.no_trade_after_bar:
                if gi >= self.lookback:
                    self._eligible_bars.append((gi, local_bar))

        self._step_idx = 0
        self._done = len(self._eligible_bars) == 0

        # Reset position state
        self._in_position = False
        self._position_entry_bar = -1
        self._position_entry_price = 0.0
        self._position_contract_idx = -1
        self._position_side = 0
        self._position_mfe = 0.0
        self._position_mae = 0.0
        self._position_series = np.array([])
        self._position_intent = None

        # Reset session state
        self._day_pnl = 0.0
        self._num_trades = 0
        self._num_stops = 0
        self._last_stop_bar = -100
        self._starting_equity = self.policy.starting_equity
        self._cumulative_equity = self._starting_equity
        self._peak_equity = self._starting_equity
        self._max_drawdown = 0.0

        if self._done:
            return self._empty_obs()
        return self._build_obs()

    def step(self, action: int) -> tuple[Observation, float, bool, StepInfo]:
        """Execute one action, advance to next bar."""
        if self._done:
            return self._empty_obs(), 0.0, True, StepInfo()

        gi, local_bar = self._eligible_bars[self._step_idx]
        info = StepInfo(day=self._day, bar=local_bar, action_taken=action)
        reward = 0.0

        # --- Cooldown check ---
        if gi - self._last_stop_bar < self.policy.cooldown_bars:
            action = ACT_HOLD  # forced hold during cooldown

        # --- Loss cap check ---
        if self._day_pnl < 0 and abs(self._day_pnl) / self._starting_equity >= self.policy.daily_loss_cap_pct:
            action = ACT_HOLD  # forced hold after loss cap

        # --- Execute action ---
        if self._in_position:
            # Update mark-to-market
            unrealized = self._get_unrealized(local_bar)
            if action == ACT_EXIT:
                # Agent-initiated exit
                reward, exit_pnl, exit_reason = self._close_position(local_bar)
                info.trade_closed = True
                info.trade_pnl = exit_pnl
                info.exit_reason = exit_reason
            else:
                # Hold — check if simulator would have exited (stop/TP/trailing/max_hold)
                exited, exit_pnl, exit_reason = self._check_policy_exit(local_bar)
                if exited:
                    reward = exit_pnl  # policy-forced exit
                    info.trade_closed = True
                    info.trade_pnl = exit_pnl
                    info.exit_reason = exit_reason
                else:
                    # Still holding — reward is mark-to-market delta
                    prev_unrealized = self._get_unrealized(local_bar - 1) if local_bar > self._position_entry_bar + 1 else 0.0
                    reward = (unrealized - prev_unrealized) * 0.01  # small shaping

                    # Decay-aware holding penalty for stagnating losers
                    bars_held = local_bar - self._position_entry_bar
                    if (bars_held > 5
                            and unrealized < -0.03
                            and self._position_mfe < 0.02):
                        reward -= 0.002 * (bars_held - 5)
        else:
            if action in (ACT_ENTER_CALL, ACT_ENTER_PUT):
                # Try to open position
                is_put = (action == ACT_ENTER_PUT)
                opened = self._open_position(local_bar, is_put)
                info.trade_opened = opened
                if not opened:
                    info.action_valid = False
            else:
                # Flat, no action — small opportunity cost
                reward = -0.0001

        info.in_position = self._in_position

        # Advance
        self._step_idx += 1
        if self._step_idx >= len(self._eligible_bars):
            # End of session — close any open position at EOD
            if self._in_position:
                eod_reward, eod_pnl, eod_reason = self._close_position_eod()
                reward += eod_reward
                info.trade_closed = True
                info.trade_pnl += eod_pnl
                info.exit_reason = eod_reason

            # Episode terminal reward
            if self._day_pnl > 0:
                reward += 0.01 * (self._day_pnl / self._starting_equity)
            reward -= 0.05 * self._max_drawdown
            self._done = True
        else:
            self._done = False

        obs = self._build_obs() if not self._done else self._empty_obs()
        return obs, reward, self._done, info

    def _build_obs(self) -> Observation:
        """Build observation for current step."""
        gi, local_bar = self._eligible_bars[self._step_idx]

        # Market context from frozen model
        window = self.features[gi - self.lookback: gi].numpy()
        contracts_np, _, _ = padded_snapshot(self._sidecar, local_bar, self.max_contracts)

        if self.model is not None:
            with torch.no_grad():
                x = torch.from_numpy(window).unsqueeze(0).float().to(self.device)
                c = torch.from_numpy(contracts_np).unsqueeze(0).float().to(self.device)
                out = self.model(x, c)
                context = out["context"][0].cpu().numpy()
                scores = out["contract_scores"][0].cpu().numpy()
                valid = out["valid_mask"][0].cpu().numpy()
                is_put = out["is_put"][0].cpu().numpy()
        else:
            # No model — return zeros (for testing)
            context = np.zeros(96, dtype=np.float32)
            scores = np.zeros(self.max_contracts, dtype=np.float32)
            valid = contracts_np[:, 0] > 0.5
            is_put = contracts_np[:, 2] > 0.5

        # Session state vector
        session = np.zeros(SESSION_STATE_DIM, dtype=np.float32)
        session[0] = local_bar / BARS_PER_DAY  # bar_of_session_norm
        session[1] = 1.0 if self._in_position else 0.0
        if self._in_position:
            bars_held = local_bar - self._position_entry_bar
            session[2] = bars_held / max(self.policy.max_hold_bars, 1)
            session[3] = self._get_unrealized(local_bar)
            session[4] = self._position_mfe
            session[5] = self._position_mae
        session[6] = self._day_pnl / max(self._starting_equity, 1.0)
        session[7] = self._num_trades / 5.0
        session[8] = self._num_stops / 2.0
        bars_since_trade = gi - self._last_stop_bar if self._last_stop_bar > 0 else 100
        session[9] = min(bars_since_trade / 10.0, 1.0)
        session[10] = float(np.max(scores[valid])) if valid.any() else 0.0
        session[11] = float(self._position_side)

        return Observation(
            context=context,
            contract_scores=scores,
            valid_mask=valid,
            is_put=is_put,
            session_state=session,
        )

    def _empty_obs(self) -> Observation:
        return Observation(
            context=np.zeros(96, dtype=np.float32),
            contract_scores=np.zeros(self.max_contracts, dtype=np.float32),
            valid_mask=np.zeros(self.max_contracts, dtype=bool),
            is_put=np.zeros(self.max_contracts, dtype=bool),
            session_state=np.zeros(SESSION_STATE_DIM, dtype=np.float32),
        )

    def _open_position(self, local_bar: int, is_put: bool) -> bool:
        """Try to open a position on the best-scored contract of the chosen side."""
        gi, _ = self._eligible_bars[self._step_idx]
        contracts_np, _, contract_indices = padded_snapshot(
            self._sidecar, local_bar, self.max_contracts
        )

        # Find best valid contract on the chosen side
        valid = contracts_np[:, 0] > 0.5
        side_match = (contracts_np[:, 2] > 0.5) == is_put
        eligible = valid & side_match

        if not eligible.any():
            return False

        # Use model scores if available, else use first eligible
        if self.model is not None:
            with torch.no_grad():
                x = torch.from_numpy(
                    self.features[gi - self.lookback: gi].numpy()
                ).unsqueeze(0).float().to(self.device)
                c = torch.from_numpy(contracts_np).unsqueeze(0).float().to(self.device)
                out = self.model(x, c)
                scores = out["contract_scores"][0].cpu().numpy()
        else:
            scores = np.zeros(self.max_contracts)

        scores[~eligible] = -np.inf
        best_row = int(np.argmax(scores))
        cidx = int(contract_indices[best_row])
        if cidx < 0:
            return False

        mid = float(contracts_np[best_row, 3])
        if mid < self.policy.min_contract_mid:
            return False

        # Store position info
        right = "P" if is_put else "C"
        strike = float(self._sidecar["contract_strike"][cidx])
        self._position_series = extract_contract_series(self._sidecar, cidx)["mid"].astype(np.float32)
        self._position_entry_bar = local_bar
        self._position_entry_price = mid
        self._position_contract_idx = cidx
        self._position_side = -1 if is_put else 1
        self._position_mfe = 0.0
        self._position_mae = 0.0

        self._position_intent = TradeIntent(
            trade=True,
            expiry=self._sidecar["expiry"],
            strike=strike,
            right=right,
            qty=self.policy.qty,
            entry_ref_price=mid,
            stop_price=mid * (1.0 - self.policy.stop_pct),
            take_profit_price=mid * (1.0 + self.policy.target_pct),
            max_hold_bars=self.policy.max_hold_bars,
            exit_policy=self.policy.exit_policy,
            bar_index=local_bar,
            contract_index=cidx,
        )

        self._in_position = True
        self._num_trades += 1
        return True

    def _get_unrealized(self, bar: int) -> float:
        """Get unrealized P&L fraction at a given bar."""
        if not self._in_position or bar >= len(self._position_series):
            return 0.0
        current = float(self._position_series[bar])
        if not np.isfinite(current) or current <= 0:
            return 0.0
        pnl = (current - self._position_entry_price) / self._position_entry_price
        if pnl > self._position_mfe:
            self._position_mfe = pnl
        if pnl < self._position_mae:
            self._position_mae = pnl
        return pnl

    def _close_position(self, local_bar: int) -> tuple[float, float, str]:
        """Close position at next bar (market order)."""
        exit_bar = local_bar + 1
        if exit_bar >= len(self._position_series):
            exit_bar = len(self._position_series) - 1
        exit_price = float(self._position_series[exit_bar])
        if not np.isfinite(exit_price) or exit_price <= 0:
            exit_price = self._position_entry_price

        raw_pnl = (exit_price - self._position_entry_price) / self._position_entry_price
        # Estimate spread cost
        spread_cost = 0.02  # conservative 2% round-trip
        net_pnl = raw_pnl - spread_cost

        self._apply_trade_result(net_pnl, "AGENT_EXIT")
        return net_pnl, net_pnl, "AGENT_EXIT"

    def _close_position_eod(self) -> tuple[float, float, str]:
        """Close position at end of day."""
        eod_bar = min(BARS_PER_DAY - 1, len(self._position_series) - 1)
        exit_price = float(self._position_series[eod_bar])
        if not np.isfinite(exit_price) or exit_price <= 0:
            exit_price = self._position_entry_price

        raw_pnl = (exit_price - self._position_entry_price) / self._position_entry_price
        spread_cost = 0.02
        net_pnl = raw_pnl - spread_cost

        self._apply_trade_result(net_pnl, "EOD")
        return net_pnl, net_pnl, "EOD"

    def _check_policy_exit(self, local_bar: int) -> tuple[bool, float, str]:
        """Check if the policy's stop/TP/trailing would force an exit."""
        if not self._in_position:
            return False, 0.0, ""

        unrealized = self._get_unrealized(local_bar)
        bars_held = local_bar - self._position_entry_bar

        # Min hold
        if bars_held < 2:
            return False, 0.0, ""

        # Stop loss
        if unrealized <= -self.policy.stop_pct:
            pnl = unrealized - 0.02  # spread cost
            self._apply_trade_result(pnl, "STOP_LOSS")
            return True, pnl, "STOP_LOSS"

        # Take profit
        if unrealized >= self.policy.target_pct:
            pnl = unrealized - 0.02
            self._apply_trade_result(pnl, "TAKE_PROFIT")
            return True, pnl, "TAKE_PROFIT"

        # Max hold
        if bars_held >= self.policy.max_hold_bars:
            pnl = unrealized - 0.02
            self._apply_trade_result(pnl, "MAX_HOLD")
            return True, pnl, "MAX_HOLD"

        return False, 0.0, ""

    def _apply_trade_result(self, net_pnl: float, exit_reason: str) -> None:
        """Update session state after a trade closes."""
        dollar_pnl = net_pnl * self._position_entry_price * self.policy.contract_multiplier * self.policy.qty
        self._day_pnl += dollar_pnl
        self._cumulative_equity += dollar_pnl

        if self._cumulative_equity > self._peak_equity:
            self._peak_equity = self._cumulative_equity
        dd = (self._peak_equity - self._cumulative_equity) / max(self._peak_equity, 1.0)
        if dd > self._max_drawdown:
            self._max_drawdown = dd

        if exit_reason == "STOP_LOSS":
            gi, _ = self._eligible_bars[self._step_idx]
            self._last_stop_bar = gi
            self._num_stops += 1

        self._in_position = False
        self._position_side = 0
        self._position_intent = None

    @property
    def episode_days(self) -> list[str]:
        """All unique trading days available."""
        return sorted(set(self.dates))
