"""Replay baselines for the v3 pure-RL environment."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from v2.core.features import _FEAT_IDX
from v3.core.execution import ExecutionConfig, range_to_squashed
from v3.core.schema import AgentAction, PolicyObservation
from v3.model import observation_to_tensors


def _encode_entry_action(
    *,
    contract_row: int,
    risk_budget_frac: float,
    stop_frac: float,
    target_frac: float,
    time_stop_frac: float,
    exit_style: str,
    config: ExecutionConfig,
    confidence: float,
) -> AgentAction:
    return AgentAction(
        action_type="OPEN",
        contract_row=contract_row,
        exit_style=exit_style,
        risk_budget_frac=range_to_squashed(risk_budget_frac, 0.0, config.max_risk_budget_frac),
        stop_frac=range_to_squashed(stop_frac, config.min_stop_frac, config.max_stop_frac),
        target_frac=range_to_squashed(target_frac, config.min_target_frac, config.max_target_frac),
        time_stop_frac=range_to_squashed(time_stop_frac, config.min_time_stop_frac, config.max_time_stop_frac),
        confidence=confidence,
    )


def _select_contract_row(obs: PolicyObservation, *, right: str | None, target_abs_delta: float) -> int:
    best_row = -1
    best_score = float("inf")
    for row, valid in enumerate(obs.valid_mask):
        if not bool(valid):
            continue
        feat = obs.contract_snapshot[row]
        is_put = feat[2] > 0.5
        if right == "C" and is_put:
            continue
        if right == "P" and not is_put:
            continue
        abs_delta = abs(float(feat[8]))
        distance = abs(float(feat[12]))
        score = abs(abs_delta - target_abs_delta) + 0.01 * distance
        if score < best_score:
            best_row = row
            best_score = score
    return best_row


class BaseReplayAgent:
    name = "base"

    def reset(self) -> None:
        pass

    def act(self, obs: PolicyObservation) -> AgentAction:
        raise NotImplementedError


class NoTradeAgent(BaseReplayAgent):
    name = "NoTrade"

    def act(self, obs: PolicyObservation) -> AgentAction:
        return AgentAction.noop()


@dataclass
class ATMFixedAgent(BaseReplayAgent):
    execution_config: ExecutionConfig
    name: str = "ATM-Fixed"

    def act(self, obs: PolicyObservation) -> AgentAction:
        if obs.position.in_position:
            return AgentAction(action_type="HOLD", manage_mode="HOLD", confidence=0.5)
        ret_12 = float(obs.context_1m[-1, _FEAT_IDX["ret_12"]])
        if abs(ret_12) < 0.10:
            return AgentAction.noop()
        side = "C" if ret_12 >= 0 else "P"
        row = _select_contract_row(obs, right=side, target_abs_delta=0.50)
        if row < 0:
            return AgentAction.noop()
        remaining = max(1, 390 - obs.local_bar - 1)
        return _encode_entry_action(
            contract_row=row,
            risk_budget_frac=0.01,
            stop_frac=0.30,
            target_frac=0.50,
            time_stop_frac=min(1.0, 120.0 / remaining),
            exit_style="STATIC",
            config=self.execution_config,
            confidence=min(1.0, abs(ret_12)),
        )


@dataclass
class SimpleRulesFixedAgent(BaseReplayAgent):
    execution_config: ExecutionConfig
    name: str = "SimpleRules-Fixed"

    def act(self, obs: PolicyObservation) -> AgentAction:
        if obs.position.in_position:
            return AgentAction(action_type="HOLD", manage_mode="HOLD", confidence=0.5)
        ema = float(obs.context_1m[-1, _FEAT_IDX["ema_cross"]])
        vwap = float(obs.context_1m[-1, _FEAT_IDX["vwap_dist"]])
        flow = float(obs.context_1m[-1, _FEAT_IDX["call_put_flow_ratio"]])
        rule_strength = max(abs(ema), abs(vwap), abs(flow - 0.5) * 2.0)
        if rule_strength < 0.20:
            return AgentAction.noop()
        if ema >= 0 and vwap >= 0 and flow >= 0.50:
            side = "C"
        elif ema <= 0 and vwap <= 0 and flow <= 0.50:
            side = "P"
        else:
            return AgentAction.noop()
        row = _select_contract_row(obs, right=side, target_abs_delta=0.35)
        if row < 0:
            return AgentAction.noop()
        remaining = max(1, 390 - obs.local_bar - 1)
        return _encode_entry_action(
            contract_row=row,
            risk_budget_frac=0.0125,
            stop_frac=0.25,
            target_frac=0.75,
            time_stop_frac=min(1.0, 90.0 / remaining),
            exit_style="STATIC",
            config=self.execution_config,
            confidence=min(1.0, rule_strength),
        )


class V2BestStaticAgent(BaseReplayAgent):
    name = "V2-Best-Static"

    def __init__(self, execution_config: ExecutionConfig) -> None:
        self.execution_config = execution_config
        self.model = None
        self.policy = None
        self.available = False
        try:
            from v2.replay import load_best_model

            self.model, self.policy, _ = load_best_model(device="cpu")
            self.available = True
        except Exception:
            self.available = False

    def act(self, obs: PolicyObservation) -> AgentAction:
        if not self.available:
            return AgentAction.noop()
        if obs.position.in_position:
            return AgentAction(action_type="HOLD", manage_mode="HOLD", confidence=0.5)
        with torch.no_grad():
            tensors = observation_to_tensors(obs, torch.device("cpu"))
            out = self.model(tensors["context_1m"], tensors["contracts"])
            scores = out["contract_scores"][0]
            no_trade = float(out["no_trade_score"][0].item())
            scores = scores.clone()
            scores[~tensors["contract_mask"][0]] = -1e9
            row = int(scores.argmax().item()) if torch.isfinite(scores).any() else -1
            best = float(scores[row].item()) if row >= 0 else -1e9
            if row < 0 or best <= no_trade:
                return AgentAction.noop()
        remaining = max(1, 390 - obs.local_bar - 1)
        stop = float(getattr(self.policy, "stop_pct", 0.30))
        target = float(getattr(self.policy, "target_pct", 0.50))
        hold = int(getattr(self.policy, "max_hold_bars", 120))
        confidence = min(1.0, max(0.0, (best - no_trade + 1.0) / 3.0))
        return _encode_entry_action(
            contract_row=row,
            risk_budget_frac=0.01,
            stop_frac=stop,
            target_frac=target,
            time_stop_frac=min(1.0, hold / remaining),
            exit_style="STATIC",
            config=self.execution_config,
            confidence=confidence,
        )
