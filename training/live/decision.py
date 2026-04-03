from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

from training.prepare import (
    ACTION_DO_NOTHING,
    ACTION_BUY_CALL_ATM,
    ACTION_BUY_CALL_OTM5,
    ACTION_BUY_CALL_OTM10,
    ACTION_BUY_PUT_ATM,
    ACTION_BUY_PUT_OTM5,
    ACTION_BUY_PUT_OTM10,
    BARS_PER_DAY,
    NO_TRADE_BEFORE_BAR,
    NUM_FEATURES,
    STARTING_CAPITAL,
    DYNAMIC_STOP_MIN,
    DYNAMIC_STOP_MAX,
    compute_dynamic_stop,
    _FEAT_IDX,
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

# v17 thresholds (backward compat)
V17_TRADE_PROB_THRESHOLD = 0.5
V17_MIN_PREDICTED_MOVE = 0.001
V17_EXIT_SIGNAL_THRESHOLD = 0.6

# v18 direction class to action mapping
_V18_DIR_CLS_TO_ACTION = [
    ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5, ACTION_BUY_CALL_OTM10,
    ACTION_BUY_PUT_ATM, ACTION_BUY_PUT_OTM5, ACTION_BUY_PUT_OTM10,
]


@dataclass
class InferenceResult:
    action: int
    confidence: float
    trade_prob: float
    pred_return_30: float
    pred_conf: float
    exit_signal: float
    reason_codes: list[str]
    # v18 risk params (zero for v17)
    risk_stop_distance: float = 0.0
    risk_target_distance: float = 0.0
    risk_conviction: float = 0.0


class ModelDecisionEngine:
    """Turns model outputs into executable intents. Supports v17 and v18."""

    def __init__(
        self,
        model: torch.nn.Module,
        lookback: int,
        device: str = "cpu",
        min_trade_prob: float = 0.55,
        max_qty: int = 1,
        num_features: int = 39,
        feature_contract_version: str = FEATURE_CONTRACT_VERSION,
        model_version: str = "v17",
    ) -> None:
        self.model = model
        self.lookback = lookback
        self.device = device
        self.min_trade_prob = float(min_trade_prob)
        self.max_qty = max(1, int(max_qty))
        self.num_features = int(num_features)
        self.feature_contract_version = feature_contract_version
        self.model_version = model_version
        self._is_v18 = hasattr(model, 'gate_head')
        # Account tracking
        self._account_balance = STARTING_CAPITAL
        self._daily_pnl_frac = 0.0
        self._win_rate_20 = 0.5
        self._consecutive_losses = 0
        self._peak_balance = STARTING_CAPITAL
        # Position tracking
        self._in_trade = False
        self._bars_held = 0
        self._unrealized_pnl = 0.0
        self._entry_confidence = 0.0
        self._best_pnl = 0.0
        self._bars_since_high = 0

    def update_position_state(self, in_trade: bool, bars_held: int = 0,
                               unrealized_pnl: float = 0.0,
                               account_health: float = 1.0,
                               loss_streak_frac: float = 0.0) -> None:
        """Called by service.py each bar to keep position state in sync."""
        self._in_trade = in_trade
        self._bars_held = bars_held
        self._unrealized_pnl = unrealized_pnl
        self._consecutive_losses = int(loss_streak_frac * 5)
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
    ) -> "ModelDecisionEngine":
        model, lookback, config, _ = load_model(model_path, device=device, train_py_path=train_py_path)
        ckpt_contract = str(config.get("feature_contract_version", FEATURE_CONTRACT_VERSION))
        ckpt_num_features = int(config.get("num_features", NUM_FEATURES))
        model_version = config.get("model_version", "v17")
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
            model_version=model_version,
        )

    def _build_account_state(self) -> torch.Tensor:
        """Build v17 5-dim account state tensor (only used for v17 models)."""
        _as_dim = getattr(self.model, 'ACCOUNT_STATE_DIM', 5)
        acct = torch.zeros(1, _as_dim, device=self.device)
        acct[0, 0] = self._account_balance / STARTING_CAPITAL
        acct[0, 1] = min(self._consecutive_losses / 5.0, 1.0)
        acct[0, 2] = self._daily_pnl_frac
        acct[0, 3] = self._win_rate_20
        if _as_dim >= 5:
            self._peak_balance = max(self._peak_balance, self._account_balance)
            acct[0, 4] = (self._account_balance - self._peak_balance) / max(self._peak_balance, 1.0)
        return acct

    def infer(self, feature_window: np.ndarray) -> InferenceResult:
        _no_trade = InferenceResult(
            action=ACTION_DO_NOTHING, confidence=0.0, trade_prob=0.0,
            pred_return_30=0.0, pred_conf=0.0, exit_signal=0.0,
            reason_codes=[],
        )
        if feature_window.ndim != 2:
            _no_trade.reason_codes = ["invalid_feature_shape"]
            return _no_trade
        if feature_window.shape[0] < self.lookback:
            _no_trade.reason_codes = ["insufficient_lookback"]
            return _no_trade
        if feature_window.shape[1] < self.num_features:
            _no_trade.reason_codes = ["feature_dim_too_small"]
            return _no_trade

        reason_codes: list[str] = []
        if feature_window.shape[1] > self.num_features:
            feature_window = feature_window[:, :self.num_features]
            reason_codes.append("feature_dim_truncated")

        x = torch.tensor(feature_window[-self.lookback:], dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0)

        with torch.no_grad():
            if self._is_v18:
                # v18: TradingModel(x) -> (market_pred, entry_gate, risk_params, exit_signal, dir_logits)
                market_pred, entry_gate, risk_params, exit_sig, dir_logits = self.model(x)
                trade_prob = float(entry_gate[0].item())
                exit_signal = float(exit_sig[0].item())
                pred_30 = float(market_pred[0, 1].item())
                stop_dist = float(risk_params[0, 0].item())
                target_dist = float(risk_params[0, 1].item())
                conviction = float(risk_params[0, 2].item())

                # Direction from 6-class head
                dir_cls = int(dir_logits[0].argmax().item())
                if trade_prob > self.min_trade_prob:
                    action = _V18_DIR_CLS_TO_ACTION[dir_cls]
                    reason_codes.append("trade_signal")
                    reason_codes.append(f"dir_cls={dir_cls}")
                else:
                    action = ACTION_DO_NOTHING
                    reason_codes.append("no_trade")

                return InferenceResult(
                    action=action,
                    confidence=trade_prob,
                    trade_prob=trade_prob,
                    pred_return_30=pred_30,
                    pred_conf=conviction,
                    exit_signal=exit_signal,
                    reason_codes=reason_codes,
                    risk_stop_distance=stop_dist,
                    risk_target_distance=target_dist,
                    risk_conviction=conviction,
                )
            else:
                # v17: PredictionModel(x, account_state) -> (pred_returns, pred_conf, action_out)
                acct_state = self._build_account_state()
                pred_returns, pred_conf, action_out = self.model(x, account_state=acct_state)
                trade_prob = float(action_out[0, 0].item())
                exit_signal = float(action_out[0, 2].item())
                pred_30 = float(pred_returns[0, 1].item())
                conf = float(pred_conf[0].item()) if pred_conf.dim() > 0 else float(pred_conf.item())

                if trade_prob > V17_TRADE_PROB_THRESHOLD and abs(pred_30) > V17_MIN_PREDICTED_MOVE:
                    if pred_30 > 0:
                        action = ACTION_BUY_CALL_ATM
                    else:
                        action = ACTION_BUY_PUT_ATM
                    reason_codes.append("trade_signal")
                else:
                    action = ACTION_DO_NOTHING
                    reason_codes.append("no_trade")

                return InferenceResult(
                    action=action,
                    confidence=trade_prob,
                    trade_prob=trade_prob,
                    pred_return_30=pred_30,
                    pred_conf=conf,
                    exit_signal=exit_signal,
                    reason_codes=reason_codes,
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
        feature_window: np.ndarray | None = None,
    ) -> DecisionIntent | None:
        if inference.action == ACTION_DO_NOTHING:
            return None
        if inference.trade_prob < self.min_trade_prob:
            return None
        if bar_of_day < NO_TRADE_BEFORE_BAR:
            return None
        contract = resolver.resolve(inference.action, spx_price)
        entry_mid = resolver.quote_mid(contract) or 1.0

        # v18: use learned stop distance; v17: formula
        if self._is_v18 and inference.risk_stop_distance > 0:
            _atr_feat = float(latest_features[_FEAT_IDX['atr_14']]) if len(latest_features) > _FEAT_IDX['atr_14'] else 0.01
            stop_pct = min(max(inference.risk_stop_distance * _atr_feat, DYNAMIC_STOP_MIN), DYNAMIC_STOP_MAX)
        else:
            _iv_val = float(latest_features[_FEAT_IDX['atm_iv']]) if len(latest_features) > _FEAT_IDX['atm_iv'] else 0.0
            _vix_val = float(latest_features[_FEAT_IDX['vix_regime']]) if len(latest_features) > _FEAT_IDX['vix_regime'] else 0.0
            stop_pct = compute_dynamic_stop(inference.trade_prob, _iv_val, _vix_val)

        stop_px = float(entry_mid * (1.0 - stop_pct))
        take_profit_px = float(entry_mid * 6.0)
        qty = self._position_size(inference.confidence)
        entry_limit = round(entry_mid * 1.05, 2)
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
                "trade_prob": inference.trade_prob,
                "pred_return_30": inference.pred_return_30,
                "pred_conf": inference.pred_conf,
                "entry_limit_price": entry_limit,
                "risk_stop_distance": inference.risk_stop_distance,
                "risk_target_distance": inference.risk_target_distance,
                "risk_conviction": inference.risk_conviction,
            },
        )

    def build_risk_update_intent(
        self,
        state: ExecutionState,
        current_option_mid: float | None,
        latest_features: np.ndarray,
        feature_window: np.ndarray | None = None,
    ) -> RiskUpdateIntent | None:
        """Risk management: trailing stop + model exit signal.

        Trailing stop tiers (based on unrealized P&L):
          +30% -> move stop to entry (breakeven)
          +50% -> move stop to +25%
          +80% -> move stop to +50%
          +120% -> move stop to +80%

        Model exit: v18 uses exit_head directly, v17 uses action_out exit_signal.
        """
        if not self._in_trade or current_option_mid is None:
            return None

        entry_px = state.fill_price or state.entry_price_reference
        if entry_px is None or entry_px <= 0:
            return None

        unrealized_pct = (current_option_mid - entry_px) / entry_px
        reason_codes: list[str] = []
        new_stop = None

        # Trailing stop tiers
        _tiers = [
            (1.20, 0.80),
            (0.80, 0.50),
            (0.50, 0.25),
            (0.30, 0.00),
        ]
        for trigger_pct, lock_pct in _tiers:
            if unrealized_pct >= trigger_pct:
                locked_stop = entry_px * (1.0 + lock_pct)
                if locked_stop > state.current_stop:
                    new_stop = locked_stop
                    reason_codes.append(f"trailing_stop_lock_{int(lock_pct*100)}pct")
                break

        if new_stop is not None:
            return RiskUpdateIntent(
                position_id=state.position_id,
                new_stop_price=new_stop,
                reason_codes=reason_codes + [f"unrealized={unrealized_pct:.2%}"],
            )

        # Model exit signal (requires 2+ bars held)
        if self._bars_held < 2 or feature_window is None:
            return None
        if feature_window.shape[0] < self.lookback:
            return None

        x = torch.tensor(
            feature_window[-self.lookback:, :self.num_features],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if self._is_v18:
                _, _, _, exit_sig, _ = self.model(x)
                exit_signal = float(exit_sig[0].item())
            else:
                acct_state = self._build_account_state()
                _, _, action_out = self.model(x, account_state=acct_state)
                exit_signal = float(action_out[0, 2].item())

        if exit_signal > V17_EXIT_SIGNAL_THRESHOLD:
            return RiskUpdateIntent(
                position_id=state.position_id,
                new_stop_price=current_option_mid * 1.01,
                reason_codes=["model_exit", f"exit_signal={exit_signal:.4f}"],
            )

        return None

    def set_account_state(self, balance: float, daily_pnl_frac: float = 0.0,
                          win_rate_20: float = 0.0) -> None:
        """Update account state from service (real IBKR balance)."""
        self._account_balance = balance
        self._daily_pnl_frac = daily_pnl_frac
        self._win_rate_20 = win_rate_20

    def set_entry_context(self, confidence: float, stop_distance: float) -> None:
        """Called at entry time to record context for exit policy."""
        self._entry_confidence = confidence
        self._best_pnl = 0.0
        self._bars_since_high = 0
