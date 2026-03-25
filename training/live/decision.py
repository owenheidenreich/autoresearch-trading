from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from training.prepare import (
    ACTION_DO_NOTHING,
    ACTION_EXIT,
    BARS_PER_DAY,
    NO_TRADE_BEFORE_BAR,
    STOP_LOSS_PCT,
    STOP_COOLDOWN_BARS,
    compute_dynamic_stop,
)
from training.live.contracts import (
    FEATURE_CONTRACT_VERSION,
    DecisionIntent,
    ExecutionState,
    RiskUpdateIntent,
)
from training.live.resolver import SPXWContractResolver

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

from replay import load_model  # noqa: E402

# Phase D imports removed — value head is built into the model, no separate exit policy needed


@dataclass
class InferenceResult:
    action: int
    confidence: float
    gate_trade_prob: float
    direction_probs: list[float]
    reason_codes: list[str]


class ModelDecisionEngine:
    """Turns model outputs into executable intents (entry and risk updates)."""

    def __init__(
        self,
        model: torch.nn.Module,
        lookback: int,
        device: str = "cpu",
        min_trade_prob: float = 0.55,
        max_qty: int = 1,
        num_features: int = 60,
        feature_contract_version: str = FEATURE_CONTRACT_VERSION,
        specialist_models: dict[str, torch.nn.Module] | None = None,
    ) -> None:
        self.model = model
        self.lookback = lookback
        self.device = device
        self.min_trade_prob = float(min_trade_prob)
        self.max_qty = max(1, int(max_qty))
        self.num_features = int(num_features)
        self.feature_contract_version = feature_contract_version
        self._has_position_proj = hasattr(model, 'position_proj')
        self._has_value_head = hasattr(model, 'value_head')
        # Phase B: specialist models (keyed by regime name)
        self._specialists = specialist_models or {}
        # Position tracking for gate head context
        self._in_trade = False
        self._bars_held = 0
        self._unrealized_pnl = 0.0
        self._account_health = 1.0  # account_balance / starting_capital
        self._loss_streak_frac = 0.0  # consecutive_losses / threshold
        self._entry_confidence = 0.0  # confidence at entry time
        self._best_pnl = 0.0  # best unrealized P&L since entry
        self._bars_since_high = 0  # bars since best P&L
        self._entry_stop_distance = 0.35  # stop distance set at entry

    def update_position_state(self, in_trade: bool, bars_held: int = 0,
                               unrealized_pnl: float = 0.0,
                               account_health: float = 1.0,
                               loss_streak_frac: float = 0.0) -> None:
        """Called by service.py each bar to keep position state in sync."""
        self._in_trade = in_trade
        self._bars_held = bars_held
        self._unrealized_pnl = unrealized_pnl
        self._account_health = account_health
        self._loss_streak_frac = loss_streak_frac
        # Track best P&L for exit policy
        if in_trade:
            if unrealized_pnl > self._best_pnl:
                self._best_pnl = unrealized_pnl
                self._bars_since_high = 0
            else:
                self._bars_since_high += 1
        else:
            self._best_pnl = 0.0
            self._bars_since_high = 0

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        train_py_path: str | None = None,
        device: str = "cpu",
        min_trade_prob: float = 0.55,
        max_qty: int = 1,
        specialist_dir: str | None = None,
    ) -> "ModelDecisionEngine":
        model, lookback, config, _ = load_model(model_path, device=device, train_py_path=train_py_path)
        ckpt_contract = str(config.get("feature_contract_version", FEATURE_CONTRACT_VERSION))
        ckpt_num_features = int(config.get("num_features", _infer_model_num_features(model)))
        if ckpt_contract != FEATURE_CONTRACT_VERSION:
            raise RuntimeError(
                f"Model feature_contract_version={ckpt_contract} "
                f"does not match required {FEATURE_CONTRACT_VERSION}"
            )

        # Phase D: value head is built into the model — no separate checkpoint
        if hasattr(model, 'value_head'):
            print(f"  Value head: available (Phase D exit intelligence)")
        else:
            print(f"  Value head: not available (gate-only exits)")

        # Phase B: Load specialist models
        specialists = {}
        if specialist_dir is None:
            specialist_dir = os.path.dirname(model_path)
        for spec_name in ("morning", "midday", "afternoon", "highvol"):
            spec_path = os.path.join(specialist_dir, f"best_model_{spec_name}.pt")
            if os.path.exists(spec_path):
                try:
                    spec_model, _, _, _ = load_model(spec_path, device=device, train_py_path=train_py_path)
                    specialists[spec_name] = spec_model
                    print(f"  Specialist loaded: {spec_name} ({spec_path})")
                except Exception as e:
                    print(f"  WARNING: Failed to load {spec_name} specialist: {e}")

        return cls(
            model,
            lookback,
            device=device,
            min_trade_prob=min_trade_prob,
            max_qty=max_qty,
            num_features=ckpt_num_features,
            feature_contract_version=ckpt_contract,
            specialist_models=specialists if specialists else None,
        )

    def _select_model(self, feature_window: np.ndarray) -> tuple[torch.nn.Module, str]:
        """Phase B meta-selector: pick specialist or generalist based on regime.

        Returns (model, source_name).
        """
        if not self._specialists:
            return self.model, "generalist"

        # Determine time-of-day regime from minutes_to_close feature (idx 19)
        if feature_window.shape[1] > 19:
            # minutes_to_close is log(minutes_remaining + 1), normalized
            # Approximate bar_of_day from raw value
            mtc_raw = float(feature_window[-1, 19])
            # Feature is z-scored, so use regime buckets instead:
            # morning: bars 0-120 (9:30-11:30)
            # midday: bars 120-240 (11:30-13:30)
            # afternoon: bars 240-390 (13:30-16:00)
            # We can estimate from VIX regime too
        else:
            return self.model, "generalist"

        # Use bar_of_day from minutes_to_close if we can reconstruct it
        # Simpler: check if vix_regime indicates high vol
        vix_regime = float(feature_window[-1, 24]) if feature_window.shape[1] > 24 else 0.0

        # Time-based specialist selection
        # We need bar_of_day which isn't directly in features. Use time_sin/time_cos (idx 20,21)
        if feature_window.shape[1] > 21:
            time_sin = float(feature_window[-1, 20])
            time_cos = float(feature_window[-1, 21])
            # session_progress = atan2(sin, cos) / (2*pi), maps to [0, 1]
            progress = (math.atan2(time_sin, time_cos) / (2 * math.pi)) % 1.0
            bar_of_day_est = int(progress * BARS_PER_DAY)
        else:
            return self.model, "generalist"

        # Check high-vol first (takes priority)
        if vix_regime > 0 and "highvol" in self._specialists:
            spec_model = self._specialists["highvol"]
            # Compare confidence: run both, pick higher confidence
            return self._compare_confidence(feature_window, spec_model, "highvol")

        # Time-of-day routing
        if bar_of_day_est < 120 and "morning" in self._specialists:
            return self._compare_confidence(feature_window, self._specialists["morning"], "morning")
        elif bar_of_day_est < 240 and "midday" in self._specialists:
            return self._compare_confidence(feature_window, self._specialists["midday"], "midday")
        elif "afternoon" in self._specialists:
            return self._compare_confidence(feature_window, self._specialists["afternoon"], "afternoon")

        return self.model, "generalist"

    def _compare_confidence(self, feature_window: np.ndarray,
                             specialist: torch.nn.Module, name: str
                             ) -> tuple[torch.nn.Module, str]:
        """Run both generalist and specialist, return higher-confidence one."""
        x = torch.tensor(feature_window[-self.lookback:], dtype=torch.float32, device=self.device).unsqueeze(0)
        pos_state = self._build_position_state()

        with torch.no_grad():
            g_gate, g_dir = self.model(x, position_state=pos_state)
            s_gate, s_dir = specialist(x, position_state=pos_state)

            g_conf = float(torch.softmax(g_gate, dim=-1)[0, 1] * torch.softmax(g_dir, dim=-1)[0].max())
            s_conf = float(torch.softmax(s_gate, dim=-1)[0, 1] * torch.softmax(s_dir, dim=-1)[0].max())

        if s_conf > g_conf * 1.05:  # specialist must beat generalist by 5%
            return specialist, name
        return self.model, "generalist"

    def _build_position_state(self) -> torch.Tensor | None:
        if not self._has_position_proj:
            return None
        _ps_dim = getattr(self.model, 'POSITION_STATE_DIM', 7)
        pos_state = torch.zeros(1, _ps_dim, device=self.device)
        if self._in_trade:
            pos_state[0, 0] = 1.0
            pos_state[0, 1] = min(self._bars_held / BARS_PER_DAY, 1.0)
            pos_state[0, 2] = float(np.tanh(self._unrealized_pnl * 5.0))
            if _ps_dim >= 7:
                pos_state[0, 5] = float(np.tanh(self._best_pnl * 2.0))
                pos_state[0, 6] = min(self._bars_since_high / BARS_PER_DAY, 1.0)
        pos_state[0, 3] = self._account_health
        pos_state[0, 4] = self._loss_streak_frac
        return pos_state

    def infer(self, feature_window: np.ndarray) -> InferenceResult:
        if feature_window.ndim != 2:
            return InferenceResult(
                action=ACTION_DO_NOTHING,
                confidence=0.0,
                gate_trade_prob=0.0,
                direction_probs=[],
                reason_codes=["invalid_feature_shape"],
            )
        if feature_window.shape[0] < self.lookback:
            return InferenceResult(
                action=ACTION_DO_NOTHING,
                confidence=0.0,
                gate_trade_prob=0.0,
                direction_probs=[],
                reason_codes=["insufficient_lookback"],
            )
        if feature_window.shape[1] < self.num_features:
            return InferenceResult(
                action=ACTION_DO_NOTHING,
                confidence=0.0,
                gate_trade_prob=0.0,
                direction_probs=[],
                reason_codes=["feature_dim_too_small"],
            )
        reason_codes: list[str] = []
        if feature_window.shape[1] > self.num_features:
            feature_window = feature_window[:, : self.num_features]
            reason_codes.append("feature_dim_truncated")

        # Phase B: Meta-selector picks specialist or generalist
        selected_model, model_source = self._select_model(feature_window)

        x = torch.tensor(feature_window[-self.lookback:], dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0)
        pos_state = self._build_position_state()
        with torch.no_grad():
            _out = selected_model(x, position_state=pos_state)
            gate_logits, dir_logits = _out[0], _out[1]
            gate_probs = torch.softmax(gate_logits, dim=-1)[0].detach().cpu().numpy()
            dir_probs = torch.softmax(dir_logits, dim=-1)[0].detach().cpu().numpy()

        if model_source != "generalist":
            reason_codes.append(f"specialist:{model_source}")

        gate_trade_prob = float(gate_probs[1])
        gate_action = int(torch.argmax(gate_logits, dim=-1).item())  # 0=NO_TRADE, 1=TRADE
        best_dir = int(np.argmax(dir_probs))
        best_dir_prob = float(dir_probs[best_dir])
        confidence = gate_trade_prob * best_dir_prob

        # Argmax gate — same as training evaluate_trades(). No hardcoded threshold.
        if gate_action == 0:  # NO_TRADE
            return InferenceResult(
                action=ACTION_DO_NOTHING,
                confidence=confidence,
                gate_trade_prob=gate_trade_prob,
                direction_probs=[float(x) for x in dir_probs],
                reason_codes=["gate_no_trade", *reason_codes],
            )
        action = best_dir + 1
        return InferenceResult(
            action=action,
            confidence=confidence,
            gate_trade_prob=gate_trade_prob,
            direction_probs=[float(x) for x in dir_probs],
            reason_codes=["trade_signal", *reason_codes],
        )

    def _position_size(self, confidence: float) -> int:
        if self.max_qty <= 1:
            return 1
        return max(1, min(self.max_qty, int(round(1 + confidence * (self.max_qty - 1)))))

    def build_entry_intent(
        self,
        inference: InferenceResult,
        resolver: SPXWContractResolver,
        spx_price: float,
        latest_features: np.ndarray,
        bar_of_day: int = 999,
    ) -> DecisionIntent | None:
        if inference.action in (ACTION_DO_NOTHING, ACTION_EXIT):
            return None
        # Gate confidence filter — reject low-confidence entries
        if inference.gate_trade_prob < self.min_trade_prob:
            return None
        # Pre-10am block — matches training evaluate_trades()
        if bar_of_day < NO_TRADE_BEFORE_BAR:
            return None
        contract = resolver.resolve(inference.action, spx_price)
        entry_mid = resolver.quote_mid(contract) or 1.0

        # Dynamic stop from gate confidence + market features.
        # Model's gate head (NO_TRADE while holding) is the primary exit.
        from training.prepare import _FEAT_IDX
        # latest_features is 1D (single row from LiveFeatureSnapshot.latest_raw_row)
        _iv_val = float(latest_features[_FEAT_IDX['atm_iv']]) if len(latest_features) > _FEAT_IDX['atm_iv'] else 0.0
        _vix_val = float(latest_features[_FEAT_IDX['vix_regime']]) if len(latest_features) > _FEAT_IDX['vix_regime'] else 0.0
        stop_pct = compute_dynamic_stop(inference.gate_trade_prob, _iv_val, _vix_val)

        stop_px = float(entry_mid * (1.0 - stop_pct))
        # Set TP very wide (5x entry) — effectively no hardcoded TP.
        # OCO bracket still needs a value, but model exit should fire first.
        take_profit_px = float(entry_mid * 6.0)
        qty = self._position_size(inference.confidence)
        # Use LMT at ask (mid + small buffer) — IBKR rejects MKT orders on
        # SPXW due to worst-case margin calculation.
        entry_limit = round(entry_mid * 1.05, 2)  # 5% above mid
        return DecisionIntent(
            action=inference.action,
            contract=contract,
            qty=qty,
            entry_order="LMT",
            entry_limit_price=entry_limit,
            stop_price=stop_px,
            take_profit_price=take_profit_px,
            confidence=float(inference.confidence),
            reason_codes=list(inference.reason_codes),
            reference_price=float(entry_mid),
            metadata={
                "gate_trade_prob": inference.gate_trade_prob,
                "direction_probs": inference.direction_probs,
                "entry_limit_price": entry_limit,
            },
        )

    def build_risk_update_intent(
        self,
        state: ExecutionState,
        current_option_mid: float | None,
        latest_features: np.ndarray,
        feature_window: np.ndarray | None = None,
    ) -> RiskUpdateIntent | None:
        """Phase D: Value head provides exit intelligence.

        Queries the model's value head to predict remaining P&L.
        Returns RiskUpdateIntent to exit when value drops below threshold.
        """
        if not self._in_trade or current_option_mid is None:
            return None
        if self._bars_held < 2:
            return None

        # Value head exit: requires feature_window for model forward pass
        _has_value_head = hasattr(self.model, 'value_head')
        if not _has_value_head or feature_window is None:
            return None
        if feature_window.shape[0] < self.lookback:
            return None

        x = torch.tensor(
            feature_window[-self.lookback:, :self.num_features],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        pos_state = self._build_position_state()

        with torch.no_grad():
            _, _, value_pred = self.model(x, position_state=pos_state, return_value=True)
            value = float(value_pred[0].item())

        # Exit when value head predicts low remaining upside
        _threshold = 0.02  # VALUE_EXIT_THRESHOLD
        if value < _threshold:
            return RiskUpdateIntent(
                position_id=state.position_id,
                new_stop_price=current_option_mid * 1.01,  # above current → triggers flatten
                reason_codes=["value_exit", f"value_pred={value:.4f}"],
            )

        return None

    def set_entry_context(self, confidence: float, stop_distance: float) -> None:
        """Called at entry time to record context for exit policy."""
        self._entry_confidence = confidence
        self._entry_stop_distance = stop_distance
        self._best_pnl = 0.0
        self._bars_since_high = 0


def _safe(v: Any, default: float) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _feat_at(arr: np.ndarray, idx: int) -> float:
    try:
        if idx < 0 or idx >= int(arr.shape[0]):
            return float("nan")
        return float(arr[idx])
    except Exception:
        return float("nan")


def _infer_model_num_features(model: torch.nn.Module) -> int:
    try:
        gate_net = model.feature_gate.gate_net
        return int(gate_net[0].in_features)
    except Exception:
        return 60
