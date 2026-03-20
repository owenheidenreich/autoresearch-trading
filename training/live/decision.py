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
    ) -> None:
        self.model = model
        self.lookback = lookback
        self.device = device
        self.min_trade_prob = float(min_trade_prob)
        self.max_qty = max(1, int(max_qty))
        self.num_features = int(num_features)
        self.feature_contract_version = feature_contract_version
        self._has_position_proj = hasattr(model, 'position_proj')
        # Position tracking for gate head context
        self._in_trade = False
        self._bars_held = 0
        self._unrealized_pnl = 0.0
        self._account_health = 1.0  # account_balance / starting_capital
        self._loss_streak_frac = 0.0  # consecutive_losses / threshold

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

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        train_py_path: str | None = None,
        device: str = "cpu",
        min_trade_prob: float = 0.55,
        max_qty: int = 1,
    ) -> "ModelDecisionEngine":
        model, lookback, config, _ = load_model(model_path, device=device, train_py_path=train_py_path)
        ckpt_contract = str(config.get("feature_contract_version", FEATURE_CONTRACT_VERSION))
        ckpt_num_features = int(config.get("num_features", _infer_model_num_features(model)))
        if ckpt_contract != FEATURE_CONTRACT_VERSION:
            raise RuntimeError(
                f"Model feature_contract_version={ckpt_contract} "
                f"does not match required {FEATURE_CONTRACT_VERSION}"
            )
        return cls(
            model,
            lookback,
            device=device,
            min_trade_prob=min_trade_prob,
            max_qty=max_qty,
            num_features=ckpt_num_features,
            feature_contract_version=ckpt_contract,
        )

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

        x = torch.tensor(feature_window[-self.lookback:], dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0)
        # Build position state tensor matching training's evaluate_trades()
        pos_state = None
        if self._has_position_proj:
            pos_state = torch.zeros(1, 5, device=self.device)
            if self._in_trade:
                pos_state[0, 0] = 1.0
                pos_state[0, 1] = min(self._bars_held / BARS_PER_DAY, 1.0)
                pos_state[0, 2] = float(np.tanh(self._unrealized_pnl * 5.0))
            pos_state[0, 3] = self._account_health
            pos_state[0, 4] = self._loss_streak_frac
        with torch.no_grad():
            gate_logits, dir_logits = self.model(x, position_state=pos_state)
            gate_probs = torch.softmax(gate_logits, dim=-1)[0].detach().cpu().numpy()
            dir_probs = torch.softmax(dir_logits, dim=-1)[0].detach().cpu().numpy()

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
        # Pre-10am block — matches training evaluate_trades()
        if bar_of_day < NO_TRADE_BEFORE_BAR:
            return None
        contract = resolver.resolve(inference.action, spx_price)
        entry_mid = resolver.quote_mid(contract) or 1.0

        # Emergency stop loss only — no hardcoded profit target.
        # Model's gate head (NO_TRADE while holding) is the primary exit.
        stop_pct = STOP_LOSS_PCT           # 0.30

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
    ) -> RiskUpdateIntent | None:
        """Fixed risk management matching training — no modulation for now."""
        # No risk updates — fixed stop/TP set at entry, matching training eval
        return None


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
