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
        feature_contract_version: str = FEATURE_CONTRACT_VERSION,
    ) -> None:
        self.model = model
        self.lookback = lookback
        self.device = device
        self.min_trade_prob = float(min_trade_prob)
        self.max_qty = max(1, int(max_qty))
        self.feature_contract_version = feature_contract_version

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
            feature_contract_version=ckpt_contract,
        )

    def infer(self, feature_window: np.ndarray) -> InferenceResult:
        if feature_window.shape[0] < self.lookback:
            return InferenceResult(
                action=ACTION_DO_NOTHING,
                confidence=0.0,
                gate_trade_prob=0.0,
                direction_probs=[],
                reason_codes=["insufficient_lookback"],
            )
        x = torch.tensor(feature_window[-self.lookback:], dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0)
        with torch.no_grad():
            gate_logits, dir_logits = self.model(x)
            gate_probs = torch.softmax(gate_logits, dim=-1)[0].detach().cpu().numpy()
            dir_probs = torch.softmax(dir_logits, dim=-1)[0].detach().cpu().numpy()

        gate_trade_prob = float(gate_probs[1])
        best_dir = int(np.argmax(dir_probs))
        best_dir_prob = float(dir_probs[best_dir])
        confidence = gate_trade_prob * best_dir_prob
        if gate_trade_prob < self.min_trade_prob:
            return InferenceResult(
                action=ACTION_DO_NOTHING,
                confidence=confidence,
                gate_trade_prob=gate_trade_prob,
                direction_probs=[float(x) for x in dir_probs],
                reason_codes=["gate_below_threshold"],
            )
        action = best_dir + 1
        return InferenceResult(
            action=action,
            confidence=confidence,
            gate_trade_prob=gate_trade_prob,
            direction_probs=[float(x) for x in dir_probs],
            reason_codes=["trade_signal"],
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
    ) -> DecisionIntent | None:
        if inference.action in (ACTION_DO_NOTHING, ACTION_EXIT):
            return None
        contract = resolver.resolve(inference.action, spx_price)
        entry_mid = resolver.quote_mid(contract) or 1.0

        # Model-driven risk profile from confidence and Greeks context.
        gamma_theta_ratio = _safe(latest_features[59], 1.0)
        stop_pct = np.clip(0.30 - 0.14 * inference.confidence - 0.03 * (gamma_theta_ratio - 1.0), 0.08, 0.30)
        take_profit_pct = np.clip(0.30 + 0.45 * inference.confidence + 0.05 * max(gamma_theta_ratio - 1.0, 0.0), 0.20, 1.25)

        stop_px = float(entry_mid * (1.0 - stop_pct))
        take_profit_px = float(entry_mid * (1.0 + take_profit_pct))
        qty = self._position_size(inference.confidence)
        return DecisionIntent(
            action=inference.action,
            contract=contract,
            qty=qty,
            entry_order="MKT",
            stop_price=stop_px,
            take_profit_price=take_profit_px,
            confidence=float(inference.confidence),
            reason_codes=list(inference.reason_codes),
            reference_price=float(entry_mid),
            metadata={
                "gate_trade_prob": inference.gate_trade_prob,
                "direction_probs": inference.direction_probs,
                "gamma_theta_ratio": gamma_theta_ratio,
            },
        )

    def build_risk_update_intent(
        self,
        state: ExecutionState,
        current_option_mid: float | None,
        latest_features: np.ndarray,
    ) -> RiskUpdateIntent | None:
        if current_option_mid is None or current_option_mid <= 0:
            return None
        if state.entry_price_reference is None or state.entry_price_reference <= 0:
            return None
        pnl = (current_option_mid / state.entry_price_reference) - 1.0
        confidence = _safe(latest_features[59], 1.0)

        new_stop = state.current_stop
        if pnl >= 0.25:
            new_stop = max(new_stop, state.entry_price_reference)
        if pnl >= 0.40:
            new_stop = max(new_stop, state.entry_price_reference * (1.0 + 0.10))
        if pnl >= 0.60:
            new_stop = max(new_stop, state.entry_price_reference * (1.0 + 0.20))

        target_boost = 0.12 + 0.15 * min(confidence, 2.0)
        new_tp = max(state.current_take_profit, current_option_mid * (1.0 + target_boost))

        if new_stop <= state.current_stop and new_tp <= state.current_take_profit:
            return None
        return RiskUpdateIntent(
            position_id=state.position_id,
            new_stop_price=float(new_stop),
            new_take_profit_price=float(new_tp),
            reason_codes=["ratchet_up_only", f"pnl={pnl:.3f}"],
        )


def _safe(v: Any, default: float) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default
