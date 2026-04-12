"""Exact-chain day environment for v3 pure RL."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from v2.core.chain_data import describe_contract
from v3.core.data import EpisodeMarket
from v3.core.execution import (
    ExecutionConfig,
    buy_fill_price,
    commission_dollars,
    map_entry_fracs,
    risk_budget_to_qty,
    sell_fill_price,
    time_stop_to_bar,
)
from v3.core.schema import (
    AgentAction,
    BARS_PER_DAY,
    EpisodeSummary,
    ExecutionResult,
    PolicyObservation,
    PositionState,
    SESSION_STATE_FEATURE_NAMES,
    StepTrace,
    TRADE_STATE_FEATURE_NAMES,
    TradeRecord,
    zero_contract_features,
    zero_trade_state,
)


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping config for PPO."""

    drawdown_penalty: float = 0.10
    action_penalty: float = 0.0002
    resize_penalty: float = 0.0001
    terminal_pnl_bonus: float = 0.10
    terminal_drawdown_penalty: float = 0.10
    trade_exploration_bonus: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExactChainEnv:
    """One trading-day exact-chain environment."""

    def __init__(
        self,
        episode: EpisodeMarket,
        *,
        execution_config: ExecutionConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.episode = episode
        self.execution_config = execution_config or ExecutionConfig()
        self.reward_config = reward_config or RewardConfig()
        self.position = PositionState()
        self.current_trade_meta: dict[str, Any] | None = None
        self.trade_records: list[TradeRecord] = []
        self.local_bar = 0
        self.cash = self.execution_config.starting_equity
        self.equity = self.execution_config.starting_equity
        self.peak_equity = self.execution_config.starting_equity
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.realized_day_pnl = 0.0
        self.total_reward = 0.0
        self.turnover_contracts = 0
        self.exposure_bars = 0
        self.action_confidences: list[float] = []
        self.action_rewards: list[float] = []
        self.done = False

    def reset(self) -> PolicyObservation:
        self.position = PositionState()
        self.current_trade_meta = None
        self.trade_records = []
        self.local_bar = 0
        self.cash = self.execution_config.starting_equity
        self.equity = self.execution_config.starting_equity
        self.peak_equity = self.execution_config.starting_equity
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.realized_day_pnl = 0.0
        self.total_reward = 0.0
        self.turnover_contracts = 0
        self.exposure_bars = 0
        self.action_confidences = []
        self.action_rewards = []
        self.done = False
        return self.current_observation()

    def current_observation(self) -> PolicyObservation:
        snapshot, contract_indices, valid_mask = self.episode.snapshot(self.local_bar)
        if self.position.in_position:
            self._sync_position_mark(self.local_bar)
            pos_contract = self.episode.contract_features_for_position(self.position.contract_index, self.local_bar)
        else:
            pos_contract = zero_contract_features(snapshot.shape[-1])
        return PolicyObservation(
            date=self.episode.date,
            local_bar=self.local_bar,
            global_bar=int(self.episode.global_indices[self.local_bar]),
            spot_price=float(self.episode.spot_prices[self.local_bar]),
            context_1m=self.episode.context_1m_window(self.local_bar),
            context_5m=self.episode.context_5m_window(self.local_bar),
            session_state=self.episode.session_state(self.local_bar),
            trade_state=self._trade_state_vector(self.local_bar),
            contract_snapshot=snapshot,
            contract_indices=contract_indices,
            valid_mask=valid_mask,
            position=self._position_copy(),
            realized_day_pnl=float(self.realized_day_pnl),
            cash=float(self.cash),
            equity=float(self.equity),
            peak_equity=float(self.peak_equity),
            drawdown=float(self.current_drawdown),
            position_contract_features=pos_contract,
        )

    def step(self, action: AgentAction) -> tuple[PolicyObservation, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode already completed")

        notes = list(action.validate())
        equity_before = self._mark_equity(self.local_bar)
        drawdown_before = self.current_drawdown
        next_bar = min(self.local_bar + 1, self.episode.n_bars - 1)
        opened_this_transition = False

        if self.position.in_position:
            self._sync_position_mark(self.local_bar)

        if not self.position.in_position:
            execution = self._handle_flat_action(action, next_bar, notes)
            opened_this_transition = execution.executed and execution.event == "open"
        else:
            execution = self._handle_live_action(action, next_bar, notes)

        if self.position.in_position and not opened_this_transition:
            auto = self._apply_auto_exit(next_bar)
            if auto is not None:
                execution = auto

        self.local_bar = next_bar
        if self.position.in_position:
            self._sync_position_mark(self.local_bar)
            self.exposure_bars += 1

        if self.local_bar >= self.episode.n_bars - 1:
            if self.position.in_position:
                execution = self._close_position(self.local_bar, reason="EOD", validation_notes=notes)
            self.done = True

        equity_after = self._mark_equity(self.local_bar)
        self._update_equity_state(equity_after)

        reward = (equity_after - equity_before) / max(self.execution_config.starting_equity, 1e-6)
        if opened_this_transition and self.reward_config.trade_exploration_bonus > 0:
            reward += self.reward_config.trade_exploration_bonus
        if execution.executed:
            reward -= self.reward_config.action_penalty
        if execution.qty_delta != 0 and execution.event in {"add", "reduce"}:
            reward -= self.reward_config.resize_penalty * abs(execution.qty_delta)
        reward -= self.reward_config.drawdown_penalty * max(0.0, self.current_drawdown - drawdown_before)
        if self.done:
            reward += self.reward_config.terminal_pnl_bonus * (
                self.realized_day_pnl / max(self.execution_config.starting_equity, 1e-6)
            )
            reward -= self.reward_config.terminal_drawdown_penalty * self.max_drawdown

        self.total_reward += reward
        self.action_confidences.append(float(action.confidence))
        self.action_rewards.append(float(reward))

        execution.reward = float(reward)
        execution.cash = float(self.cash)
        execution.equity = float(self.equity)
        execution.drawdown = float(self.current_drawdown)
        execution.unrealized_pnl = float(self._unrealized_pnl())

        info: dict[str, Any] = {
            "execution": execution,
            "trade_records": [t.to_dict() for t in self.trade_records],
        }
        if self.done:
            info["summary"] = self.summary().to_dict()
        return self.current_observation(), float(reward), self.done, info

    def summary(self) -> EpisodeSummary:
        winning = sum(1 for t in self.trade_records if t.pnl_dollars > 0)
        avg_conf = float(np.mean(self.action_confidences)) if self.action_confidences else 0.0
        return EpisodeSummary(
            date=self.episode.date,
            steps=self.local_bar + 1,
            total_reward=float(self.total_reward),
            starting_equity=float(self.execution_config.starting_equity),
            ending_equity=float(self.equity),
            realized_pnl=float(self.realized_day_pnl),
            max_drawdown=float(self.max_drawdown),
            total_trades=len(self.trade_records),
            winning_trades=winning,
            turnover_contracts=int(self.turnover_contracts),
            exposure_bars=int(self.exposure_bars),
            avg_action_confidence=avg_conf,
        )

    def build_step_trace(
        self,
        *,
        obs: PolicyObservation,
        action: AgentAction,
        execution: ExecutionResult,
        pointer_scores: list[float],
        sampled_action: dict[str, Any],
    ) -> StepTrace:
        return StepTrace(
            date=obs.date,
            local_bar=obs.local_bar,
            global_bar=obs.global_bar,
            in_position=obs.position.in_position,
            action_type=action.action_type,
            manage_mode=action.manage_mode,
            exit_style=action.exit_style,
            selected_contract_row=action.contract_row,
            selected_contract_index=execution.contract_index,
            pointer_scores=pointer_scores,
            valid_mask=[int(x) for x in obs.valid_mask.astype(int).tolist()],
            context_summary={
                "spot_price": float(obs.spot_price),
                "vix_regime": float(obs.context_1m[-1, 14]),
                "minutes_to_close": float(obs.context_1m[-1, 13]),
                "ret_12": float(obs.context_1m[-1, 1]),
                "ema_cross": float(obs.context_1m[-1, 9]),
                "call_put_flow_ratio": float(obs.context_1m[-1, 41]),
                "drawdown": float(obs.drawdown),
                "equity": float(obs.equity),
            },
            session_memory=self._named_state(obs.session_state, SESSION_STATE_FEATURE_NAMES),
            trade_memory=self._named_state(obs.trade_state, TRADE_STATE_FEATURE_NAMES),
            selected_contract_features=self._trace_contract_features(obs, action, execution),
            sampled_action=sampled_action,
            execution=execution.to_dict(),
            realized_day_pnl=float(self.realized_day_pnl),
            equity=float(self.equity),
            drawdown=float(self.current_drawdown),
        )

    def _handle_flat_action(self, action: AgentAction, next_bar: int, notes: list[str]) -> ExecutionResult:
        if action.action_type != "OPEN":
            return self._empty_execution(next_bar, action, event="flat", notes=notes)

        snapshot, contract_indices, valid_mask = self.episode.snapshot(self.local_bar)
        if action.contract_row < 0 or action.contract_row >= len(contract_indices):
            notes.append("invalid_contract_row")
            return self._empty_execution(next_bar, action, event="reject", notes=notes)
        if not bool(valid_mask[action.contract_row]) or int(contract_indices[action.contract_row]) < 0:
            notes.append("contract_not_valid")
            return self._empty_execution(next_bar, action, event="reject", notes=notes)

        contract_idx = int(contract_indices[action.contract_row])
        series = self.episode.contract_series(contract_idx)
        mid = float(series["mid"][next_bar])
        bid = float(series["bid"][next_bar]) if np.isfinite(series["bid"][next_bar]) else None
        ask = float(series["ask"][next_bar]) if np.isfinite(series["ask"][next_bar]) else None
        if not np.isfinite(mid) or mid <= 0:
            notes.append("missing_next_bar_price")
            return self._empty_execution(next_bar, action, event="reject", notes=notes, contract_idx=contract_idx)

        risk_budget, stop_frac, target_frac, time_stop_frac = map_entry_fracs(
            risk_budget_raw=action.risk_budget_frac,
            stop_raw=action.stop_frac,
            target_raw=action.target_frac,
            time_stop_raw=action.time_stop_frac,
            config=self.execution_config,
        )
        qty = risk_budget_to_qty(
            risk_budget_frac=risk_budget,
            equity=self.equity,
            cash=self.cash,
            entry_mid=mid,
            bid=bid,
            ask=ask,
            stop_frac=stop_frac,
            config=self.execution_config,
        )
        if qty <= 0:
            notes.append("qty_zero_after_risk_budget")
            return self._empty_execution(next_bar, action, event="reject", notes=notes, contract_idx=contract_idx)

        fill = buy_fill_price(mid, bid, ask, qty, self.execution_config)
        total_cost = fill * qty * self.execution_config.contract_multiplier + commission_dollars(qty, self.execution_config)
        if total_cost > self.cash:
            notes.append("insufficient_cash")
            return self._empty_execution(next_bar, action, event="reject", notes=notes, contract_idx=contract_idx)

        contract = describe_contract(self.episode.sidecar, contract_idx)
        self.cash -= total_cost
        self.realized_day_pnl -= commission_dollars(qty, self.execution_config)
        self.turnover_contracts += qty
        max_hold_bar = time_stop_to_bar(current_bar=next_bar, time_stop_frac=time_stop_frac, bars_per_day=self.episode.n_bars)
        entry_anchor = self._entry_anchor_features(next_bar)
        self.position = PositionState(
            in_position=True,
            contract_index=contract_idx,
            snapshot_row=int(action.contract_row),
            expiry=contract.expiry,
            strike=float(contract.strike),
            right=contract.right,
            qty=qty,
            avg_entry_price=fill,
            current_mid_price=mid,
            stop_price=max(0.01, fill * (1.0 - stop_frac)),
            target_price=fill * (1.0 + target_frac),
            max_hold_bar=max_hold_bar,
            exit_style=action.exit_style,
            opened_bar=next_bar,
            last_update_bar=next_bar,
            last_adjust_bar=next_bar,
            cumulative_realized_pnl=-commission_dollars(qty, self.execution_config),
            risk_budget_frac=risk_budget,
            entry_vwap_dist=entry_anchor["entry_vwap_dist"],
            entry_session_range_position=entry_anchor["entry_session_range_position"],
            entry_call_put_flow_ratio=entry_anchor["entry_call_put_flow_ratio"],
        )
        self.current_trade_meta = {
            "date": self.episode.date,
            "expiry": contract.expiry,
            "strike": float(contract.strike),
            "right": contract.right,
            "contract_index": contract_idx,
            "entry_bar": next_bar,
            "entry_price": fill,
            "max_qty": qty,
            "adjustments": 0,
        }
        return ExecutionResult(
            date=self.episode.date,
            local_bar=next_bar,
            action_type=action.action_type,
            executed=True,
            event="open",
            reason_codes=("open",),
            contract_index=contract_idx,
            snapshot_row=int(action.contract_row),
            qty_delta=qty,
            resulting_qty=qty,
            fill_price=fill,
            stop_price=self.position.stop_price,
            target_price=self.position.target_price,
            max_hold_bar=max_hold_bar,
            realized_pnl=float(self.realized_day_pnl),
            confidence=float(action.confidence),
            validation_notes=tuple(notes),
        )

    def _handle_live_action(self, action: AgentAction, next_bar: int, notes: list[str]) -> ExecutionResult:
        if action.action_type in {"NOOP", "HOLD"} or action.manage_mode == "HOLD":
            self.position.last_update_bar = next_bar
            return self._empty_execution(next_bar, action, event="hold", notes=notes, contract_idx=self.position.contract_index)
        if action.action_type == "CLOSE" or action.manage_mode == "CLOSE":
            return self._close_position(next_bar, reason="MODEL_EXIT", validation_notes=notes)
        if action.action_type == "ADJUST" or action.manage_mode == "ADJUST":
            return self._adjust_position(action, next_bar, notes)
        notes.append("invalid_live_action")
        return self._empty_execution(next_bar, action, event="reject", notes=notes, contract_idx=self.position.contract_index)

    def _adjust_position(self, action: AgentAction, next_bar: int, notes: list[str]) -> ExecutionResult:
        contract_idx = self.position.contract_index
        series = self.episode.contract_series(contract_idx)
        mid = float(series["mid"][next_bar])
        bid = float(series["bid"][next_bar]) if np.isfinite(series["bid"][next_bar]) else None
        ask = float(series["ask"][next_bar]) if np.isfinite(series["ask"][next_bar]) else None
        if not np.isfinite(mid) or mid <= 0:
            notes.append("missing_adjust_price")
            return self._empty_execution(next_bar, action, event="reject", notes=notes, contract_idx=contract_idx)

        _, stop_frac, target_frac, time_stop_frac = map_entry_fracs(
            risk_budget_raw=0.0,
            stop_raw=action.updated_stop_frac,
            target_raw=action.updated_target_frac,
            time_stop_raw=action.updated_time_stop_frac,
            config=self.execution_config,
        )
        self.position.stop_price = max(0.01, mid * (1.0 - stop_frac))
        self.position.target_price = mid * (1.0 + target_frac)
        self.position.max_hold_bar = time_stop_to_bar(
            current_bar=next_bar,
            time_stop_frac=time_stop_frac,
            bars_per_day=self.episode.n_bars,
        )
        self.position.last_update_bar = next_bar
        self.position.last_adjust_bar = next_bar
        qty_delta = 0
        fill_price = mid
        event = "adjust"

        if action.size_delta_frac > 0:
            qty_delta = risk_budget_to_qty(
                risk_budget_frac=abs(float(action.size_delta_frac)) * self.execution_config.max_risk_budget_frac,
                equity=self.equity,
                cash=self.cash,
                entry_mid=mid,
                bid=bid,
                ask=ask,
                stop_frac=stop_frac,
                config=self.execution_config,
            )
            if qty_delta > 0:
                fill_price = buy_fill_price(mid, bid, ask, qty_delta, self.execution_config)
                total_cost = fill_price * qty_delta * self.execution_config.contract_multiplier + commission_dollars(qty_delta, self.execution_config)
                if total_cost <= self.cash:
                    old_qty = self.position.qty
                    self.cash -= total_cost
                    self.position.avg_entry_price = (
                        self.position.avg_entry_price * old_qty + fill_price * qty_delta
                    ) / max(old_qty + qty_delta, 1)
                    self.position.qty += qty_delta
                    self.position.cumulative_realized_pnl -= commission_dollars(qty_delta, self.execution_config)
                    self.realized_day_pnl -= commission_dollars(qty_delta, self.execution_config)
                    self.turnover_contracts += qty_delta
                    self.position.add_count += 1
                    event = "add"
                else:
                    notes.append("insufficient_cash_for_add")
                    qty_delta = 0
        elif action.size_delta_frac < 0:
            qty_delta = -max(1, int(np.ceil(abs(float(action.size_delta_frac)) * self.position.qty)))
            qty_to_sell = min(self.position.qty, abs(qty_delta))
            fill_price = sell_fill_price(mid, bid, ask, qty_to_sell, self.execution_config)
            proceeds = fill_price * qty_to_sell * self.execution_config.contract_multiplier - commission_dollars(qty_to_sell, self.execution_config)
            self.cash += proceeds
            realized = (
                (fill_price - self.position.avg_entry_price)
                * qty_to_sell
                * self.execution_config.contract_multiplier
                - commission_dollars(qty_to_sell, self.execution_config)
            )
            self.position.qty -= qty_to_sell
            self.position.cumulative_realized_pnl += realized
            self.realized_day_pnl += realized
            self.turnover_contracts += qty_to_sell
            self.position.reduce_count += 1
            qty_delta = -qty_to_sell
            event = "reduce"
            if self.position.qty <= 0:
                return self._finalize_flatten(
                    next_bar,
                    fill_price=fill_price,
                    qty_delta=qty_delta,
                    exit_reason="MODEL_EXIT",
                    event="close",
                    validation_notes=notes,
                )

        if self.current_trade_meta is not None and event in {"add", "reduce", "adjust"}:
            self.current_trade_meta["adjustments"] += 1
            self.current_trade_meta["max_qty"] = max(self.current_trade_meta["max_qty"], self.position.qty)

        return ExecutionResult(
            date=self.episode.date,
            local_bar=next_bar,
            action_type=action.action_type,
            executed=True,
            event=event,
            reason_codes=(event,),
            contract_index=contract_idx,
            snapshot_row=self.position.snapshot_row,
            qty_delta=qty_delta,
            resulting_qty=self.position.qty,
            fill_price=fill_price,
            stop_price=self.position.stop_price,
            target_price=self.position.target_price,
            max_hold_bar=self.position.max_hold_bar,
            realized_pnl=float(self.realized_day_pnl),
            confidence=float(action.confidence),
            validation_notes=tuple(notes),
        )

    def _apply_auto_exit(self, next_bar: int) -> ExecutionResult | None:
        if not self.position.in_position:
            return None
        self.position.last_update_bar = next_bar
        series = self.episode.contract_series(self.position.contract_index)
        mid = float(series["mid"][next_bar])
        if not np.isfinite(mid) or mid <= 0:
            return None
        style = self.position.exit_style
        if next_bar >= self.position.max_hold_bar:
            return self._close_position(next_bar, reason="TIME_STOP", validation_notes=[])
        if style in {"STATIC", "MODEL_EXIT"} and mid <= self.position.stop_price:
            return self._close_position(next_bar, reason="STOP", validation_notes=[])
        if style == "STATIC" and mid >= self.position.target_price:
            return self._close_position(next_bar, reason="TARGET", validation_notes=[])
        return None

    def _close_position(self, local_bar: int, *, reason: str, validation_notes: list[str]) -> ExecutionResult:
        series = self.episode.contract_series(self.position.contract_index)
        mid = float(series["mid"][local_bar])
        bid = float(series["bid"][local_bar]) if np.isfinite(series["bid"][local_bar]) else None
        ask = float(series["ask"][local_bar]) if np.isfinite(series["ask"][local_bar]) else None
        qty = self.position.qty
        fill = sell_fill_price(mid, bid, ask, qty, self.execution_config)
        proceeds = fill * qty * self.execution_config.contract_multiplier - commission_dollars(qty, self.execution_config)
        self.cash += proceeds
        realized = (
            (fill - self.position.avg_entry_price)
            * qty
            * self.execution_config.contract_multiplier
            - commission_dollars(qty, self.execution_config)
        )
        self.position.cumulative_realized_pnl += realized
        self.realized_day_pnl += realized
        self.turnover_contracts += qty
        return self._finalize_flatten(
            local_bar,
            fill_price=fill,
            qty_delta=-qty,
            exit_reason=reason,
            event="close",
            validation_notes=validation_notes,
        )

    def _finalize_flatten(
        self,
        local_bar: int,
        *,
        fill_price: float,
        qty_delta: int,
        exit_reason: str,
        event: str,
        validation_notes: list[str],
    ) -> ExecutionResult:
        position = self.position
        if self.current_trade_meta is not None:
            self.trade_records.append(
                TradeRecord(
                    date=self.episode.date,
                    expiry=position.expiry,
                    strike=position.strike,
                    right=position.right,
                    contract_index=position.contract_index,
                    entry_bar=self.current_trade_meta["entry_bar"],
                    exit_bar=local_bar,
                    max_qty=self.current_trade_meta["max_qty"],
                    entry_price=self.current_trade_meta["entry_price"],
                    exit_price=fill_price,
                    pnl_dollars=position.cumulative_realized_pnl,
                    pnl_pct=position.cumulative_realized_pnl
                    / max(self.current_trade_meta["entry_price"] * self.current_trade_meta["max_qty"] * self.execution_config.contract_multiplier, 1e-6),
                    exit_reason=exit_reason,
                    adjustments=self.current_trade_meta["adjustments"],
                )
            )
        result = ExecutionResult(
            date=self.episode.date,
            local_bar=local_bar,
            action_type="CLOSE" if exit_reason == "MODEL_EXIT" else event.upper(),
            executed=True,
            event=event,
            reason_codes=(exit_reason,),
            contract_index=position.contract_index,
            snapshot_row=position.snapshot_row,
            qty_delta=qty_delta,
            resulting_qty=0,
            fill_price=fill_price,
            stop_price=position.stop_price,
            target_price=position.target_price,
            max_hold_bar=position.max_hold_bar,
            realized_pnl=float(self.realized_day_pnl),
            confidence=0.0,
            validation_notes=tuple(validation_notes),
        )
        self.position = PositionState()
        self.current_trade_meta = None
        return result

    def _sync_position_mark(self, local_bar: int) -> None:
        if not self.position.in_position:
            return
        series = self.episode.contract_series(self.position.contract_index)
        mid = float(series["mid"][local_bar])
        if np.isfinite(mid) and mid > 0:
            self.position.current_mid_price = mid
        self.position.last_update_bar = max(self.position.last_update_bar, local_bar)
        unrealized = self._current_unrealized_pnl_at_bar(local_bar)
        self.position.peak_unrealized_pnl = max(self.position.peak_unrealized_pnl, unrealized)
        self.position.trough_unrealized_pnl = min(self.position.trough_unrealized_pnl, unrealized)
        self.position.mfe_dollars = max(self.position.mfe_dollars, max(0.0, unrealized))
        self.position.mae_dollars = max(self.position.mae_dollars, max(0.0, -unrealized))

    def _unrealized_pnl(self) -> float:
        return self._current_unrealized_pnl_at_bar(self.local_bar)

    def _current_unrealized_pnl_at_bar(self, local_bar: int) -> float:
        if not self.position.in_position:
            return 0.0
        series = self.episode.contract_series(self.position.contract_index)
        bid = float(series["bid"][local_bar]) if np.isfinite(series["bid"][local_bar]) else None
        ask = float(series["ask"][local_bar]) if np.isfinite(series["ask"][local_bar]) else None
        liquidation_price = sell_fill_price(
            self.position.current_mid_price,
            bid,
            ask,
            self.position.qty,
            self.execution_config,
        )
        liquidation_value = (
            liquidation_price * self.position.qty * self.execution_config.contract_multiplier
            - commission_dollars(self.position.qty, self.execution_config)
        )
        entry_cost = self.position.avg_entry_price * self.position.qty * self.execution_config.contract_multiplier
        return liquidation_value - entry_cost

    def _mark_equity(self, local_bar: int) -> float:
        if self.position.in_position:
            self._sync_position_mark(local_bar)
            series = self.episode.contract_series(self.position.contract_index)
            bid = float(series["bid"][local_bar]) if np.isfinite(series["bid"][local_bar]) else None
            ask = float(series["ask"][local_bar]) if np.isfinite(series["ask"][local_bar]) else None
            liquidation_price = sell_fill_price(
                self.position.current_mid_price,
                bid,
                ask,
                self.position.qty,
                self.execution_config,
            )
            liquidation_value = (
                liquidation_price * self.position.qty * self.execution_config.contract_multiplier
                - commission_dollars(self.position.qty, self.execution_config)
            )
            self.equity = self.cash + liquidation_value
        else:
            self.equity = self.cash
        return float(self.equity)

    def _update_equity_state(self, equity: float) -> None:
        self.equity = float(equity)
        self.peak_equity = max(self.peak_equity, self.equity)
        if self.peak_equity > 0:
            self.current_drawdown = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

    def _position_copy(self) -> PositionState:
        return PositionState(**self.position.to_dict())

    def _empty_execution(
        self,
        local_bar: int,
        action: AgentAction,
        *,
        event: str,
        notes: list[str],
        contract_idx: int = -1,
    ) -> ExecutionResult:
        return ExecutionResult(
            date=self.episode.date,
            local_bar=local_bar,
            action_type=action.action_type,
            executed=False,
            event=event,
            reason_codes=tuple(notes) if notes else (event,),
            contract_index=contract_idx,
            snapshot_row=action.contract_row,
            qty_delta=0,
            resulting_qty=self.position.qty,
            fill_price=0.0,
            stop_price=self.position.stop_price,
            target_price=self.position.target_price,
            max_hold_bar=self.position.max_hold_bar,
            realized_pnl=float(self.realized_day_pnl),
            confidence=float(action.confidence),
            validation_notes=tuple(notes),
        )

    def _trace_contract_features(
        self,
        obs: PolicyObservation,
        action: AgentAction,
        execution: ExecutionResult,
    ) -> dict[str, float]:
        if action.contract_row >= 0 and action.contract_row < len(obs.contract_snapshot):
            feat = obs.contract_snapshot[action.contract_row]
            return {
                "mid": float(feat[3]),
                "spread_fraction": float(feat[4]),
                "delta": float(feat[8]),
                "gamma": float(feat[9]),
                "theta": float(feat[10]),
                "moneyness_pct": float(feat[11]),
                "distance_points": float(feat[12]),
                "quality_flag": float(feat[14]),
            }
        if execution.contract_index >= 0 and obs.position.in_position:
            feat = obs.position_contract_features
            return {
                "mid": float(feat[3]),
                "spread_fraction": float(feat[4]),
                "delta": float(feat[8]),
                "gamma": float(feat[9]),
                "theta": float(feat[10]),
                "moneyness_pct": float(feat[11]),
                "distance_points": float(feat[12]),
                "quality_flag": float(feat[14]),
            }
        return {}

    def _trade_state_vector(self, local_bar: int) -> np.ndarray:
        if not self.position.in_position:
            return zero_trade_state()
        pos = self.position
        denom = max(1.0, self.equity)
        current_mid = max(pos.current_mid_price, 1e-6)
        unrealized = self._current_unrealized_pnl_at_bar(local_bar)
        return np.asarray(
            [
                1.0,
                pos.bars_held / float(BARS_PER_DAY),
                max(0, local_bar - pos.last_adjust_bar) / float(BARS_PER_DAY) if pos.last_adjust_bar >= 0 else 0.0,
                float(pos.qty) / 25.0,
                float(pos.risk_budget_frac),
                pos.avg_entry_price / max(self.episode.spot_prices[local_bar], 1.0),
                pos.current_mid_price / max(self.episode.spot_prices[local_bar], 1.0),
                (pos.current_mid_price - pos.stop_price) / current_mid,
                (pos.target_price - pos.current_mid_price) / current_mid,
                max(0.0, pos.max_hold_bar - local_bar) / float(BARS_PER_DAY),
                unrealized / denom,
                pos.cumulative_realized_pnl / denom,
                pos.mfe_dollars / denom,
                pos.mae_dollars / denom,
                pos.peak_unrealized_pnl / denom,
                pos.trough_unrealized_pnl / denom,
                max(0, pos.opened_bar) / float(BARS_PER_DAY),
                min(pos.add_count, 10) / 10.0,
                min(pos.reduce_count, 10) / 10.0,
                pos.entry_vwap_dist,
                pos.entry_session_range_position,
                pos.entry_call_put_flow_ratio,
            ],
            dtype=np.float32,
        )

    def _named_state(self, values: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
        return {name: float(values[i]) for i, name in enumerate(names)}

    def _entry_anchor_features(self, local_bar: int) -> dict[str, float]:
        global_bar = int(self.episode.global_indices[local_bar])
        row = self.episode.context_source[global_bar]
        return {
            "entry_vwap_dist": float(row[6]),
            "entry_session_range_position": float(row[17]),
            "entry_call_put_flow_ratio": float(row[41]),
        }
