"""Per-bar decision trace for harness observability.

Every eligible bar produces a trace record capturing what the model saw,
what it decided, what the oracle would have done, and the realized outcome.
This is the "dense feedback signal" that turns a single experiment score
into an actionable dataset for harness improvement.
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass, field, fields, asdict
from typing import Any

import numpy as np


@dataclass
class DecisionTrace:
    """One per-bar decision record."""

    # --- Identity ---
    date: str = ""
    bar_of_day: int = 0
    global_bar_idx: int = 0

    # --- Market context at decision time ---
    spot_price: float = 0.0
    vix_regime: float = 0.0
    minutes_to_close: int = 0

    # --- Model outputs ---
    gate_logit: float = 0.0
    gate_threshold: float = 0.0
    gate_pass: bool = False
    best_contract_score: float = float("-inf")
    top_5_scores: str = ""            # serialized as "s1;s2;s3;s4;s5"
    top_5_strikes: str = ""           # "k1;k2;k3;k4;k5"
    top_5_rights: str = ""            # "C;P;C;C;P"
    n_valid_contracts: int = 0

    # --- Decision ---
    decision: str = ""                # "trade" or "no_trade"
    skip_reason: str = ""             # in_position, cooldown, loss_cap, gate, equity_zero

    # --- Selected contract (if trade) ---
    selected_strike: float = 0.0
    selected_right: str = ""
    selected_mid: float = 0.0
    selected_contract_idx: int = -1

    # --- Oracle (what should have been picked) ---
    oracle_contract_idx: int = -1
    oracle_strike: float = 0.0
    oracle_right: str = ""
    oracle_pnl: float = float("nan")
    label_quality: float = 0.0        # top_label - second_label margin
    bar_is_labelable: bool = False

    # --- Realized outcome (filled after trade sim, NaN if no trade) ---
    model_pnl: float = float("nan")
    delta_pnl: float = float("nan")   # model_pnl - oracle_pnl
    exit_reason: str = ""
    bars_held: int = 0


def _extract_oracle(
    contract_labels: np.ndarray,
    contract_features: np.ndarray,
    contract_indices: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[int, float, float, str, float, bool]:
    """Extract oracle contract info from sidecar labels.

    Returns:
        (oracle_contract_idx, oracle_strike, oracle_pnl, oracle_right,
         label_quality, bar_is_labelable)
    """
    finite_mask = np.isfinite(contract_labels) & valid_mask
    if not finite_mask.any():
        return -1, 0.0, float("nan"), "", 0.0, False

    valid_labels = contract_labels.copy()
    valid_labels[~finite_mask] = -np.inf

    best_row = int(np.argmax(valid_labels))
    oracle_pnl = float(contract_labels[best_row])
    oracle_contract_idx = int(contract_indices[best_row])
    oracle_strike = float(contract_features[best_row, 1])  # strike field
    oracle_right = "P" if contract_features[best_row, 2] > 0.5 else "C"

    # Label quality: margin between top and second-best
    sorted_valid = np.sort(valid_labels[finite_mask])[::-1]
    if len(sorted_valid) >= 2:
        label_quality = float(sorted_valid[0] - sorted_valid[1])
    else:
        label_quality = float(sorted_valid[0]) if len(sorted_valid) == 1 else 0.0

    return oracle_contract_idx, oracle_strike, oracle_pnl, oracle_right, label_quality, True


def _extract_top_k(
    contract_scores: np.ndarray,
    valid_mask: np.ndarray,
    contract_features: np.ndarray,
    k: int = 5,
) -> tuple[str, str, str, int]:
    """Extract top-k scored contracts.

    Returns (scores_str, strikes_str, rights_str, n_valid).
    """
    scores = contract_scores.copy()
    scores[~valid_mask] = -np.inf
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return "", "", "", 0

    top_k = min(k, n_valid)
    top_indices = np.argsort(scores)[::-1][:top_k]

    top_scores = [f"{scores[i]:.4f}" for i in top_indices]
    top_strikes = [f"{contract_features[i, 1]:.0f}" for i in top_indices]
    top_rights = ["P" if contract_features[i, 2] > 0.5 else "C" for i in top_indices]

    return (
        ";".join(top_scores),
        ";".join(top_strikes),
        ";".join(top_rights),
        n_valid,
    )


def build_trace_for_bar(
    *,
    date: str,
    bar_of_day: int,
    global_bar_idx: int,
    spot_price: float,
    vix_regime: float,
    gate_logit: float,
    gate_threshold: float,
    contract_scores: np.ndarray,
    valid_mask: np.ndarray,
    contract_features: np.ndarray,
    contract_labels: np.ndarray,
    contract_indices: np.ndarray,
    decision: str,
    skip_reason: str = "",
    selected_strike: float = 0.0,
    selected_right: str = "",
    selected_mid: float = 0.0,
    selected_contract_idx: int = -1,
) -> DecisionTrace:
    """Build a DecisionTrace for one bar (before trade outcome is known)."""
    scores_str, strikes_str, rights_str, n_valid = _extract_top_k(
        contract_scores, valid_mask, contract_features,
    )

    best_score = float(contract_scores[valid_mask].max()) if valid_mask.any() else float("-inf")

    oracle_idx, oracle_strike, oracle_pnl, oracle_right, lq, labelable = _extract_oracle(
        contract_labels, contract_features, contract_indices, valid_mask,
    )

    mtc = 390 - bar_of_day

    return DecisionTrace(
        date=date,
        bar_of_day=bar_of_day,
        global_bar_idx=global_bar_idx,
        spot_price=spot_price,
        vix_regime=vix_regime,
        minutes_to_close=mtc,
        gate_logit=gate_logit,
        gate_threshold=gate_threshold,
        gate_pass=gate_logit > gate_threshold,
        best_contract_score=best_score,
        top_5_scores=scores_str,
        top_5_strikes=strikes_str,
        top_5_rights=rights_str,
        n_valid_contracts=n_valid,
        decision=decision,
        skip_reason=skip_reason,
        selected_strike=selected_strike,
        selected_right=selected_right,
        selected_mid=selected_mid,
        selected_contract_idx=selected_contract_idx,
        oracle_contract_idx=oracle_idx,
        oracle_strike=oracle_strike,
        oracle_right=oracle_right,
        oracle_pnl=oracle_pnl,
        label_quality=lq,
        bar_is_labelable=labelable,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_FIELD_NAMES = [f.name for f in fields(DecisionTrace)]


def save_traces(traces: list[DecisionTrace], path: str) -> None:
    """Write traces to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELD_NAMES)
        writer.writeheader()
        for t in traces:
            writer.writerow(asdict(t))


def load_traces(path: str) -> list[DecisionTrace]:
    """Read traces from CSV."""
    traces = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = DecisionTrace()
            for k, v in row.items():
                if not hasattr(t, k):
                    continue
                field_obj = {f.name: f for f in fields(DecisionTrace)}[k]
                ftype = field_obj.type
                if ftype == "int":
                    setattr(t, k, int(v))
                elif ftype == "float":
                    setattr(t, k, float(v))
                elif ftype == "bool":
                    setattr(t, k, v.lower() in ("true", "1"))
                else:
                    setattr(t, k, v)
            traces.append(t)
    return traces


# ---------------------------------------------------------------------------
# Summary / analysis
# ---------------------------------------------------------------------------

def trace_summary(traces: list[DecisionTrace]) -> dict[str, Any]:
    """Aggregate trace statistics for quick diagnosis."""
    if not traces:
        return {"n_bars": 0}

    n_bars = len(traces)
    decisions = Counter(t.decision for t in traces)
    skip_reasons = Counter(t.skip_reason for t in traces if t.skip_reason)

    traded = [t for t in traces if t.decision == "trade"]
    n_traded = len(traded)

    # Gate accuracy: of labelable bars, how often does model agree with oracle on trade/no-trade?
    labelable = [t for t in traces if t.bar_is_labelable]
    gate_correct = 0
    for t in labelable:
        oracle_says_trade = not np.isnan(t.oracle_pnl) and t.oracle_pnl > 0.04
        model_says_trade = t.decision == "trade"
        if oracle_says_trade == model_says_trade:
            gate_correct += 1
    gate_accuracy = gate_correct / len(labelable) if labelable else 0.0

    # Selection accuracy: of traded bars, how often did model pick the oracle contract?
    selection_match = sum(
        1 for t in traded
        if t.selected_contract_idx == t.oracle_contract_idx and t.oracle_contract_idx >= 0
    )
    selection_accuracy = selection_match / n_traded if n_traded else 0.0

    # P&L gap analysis
    traded_with_pnl = [t for t in traded if np.isfinite(t.model_pnl)]
    if traded_with_pnl:
        model_pnls = [t.model_pnl for t in traded_with_pnl]
        oracle_pnls = [t.oracle_pnl for t in traded_with_pnl if np.isfinite(t.oracle_pnl)]
        delta_pnls = [t.delta_pnl for t in traded_with_pnl if np.isfinite(t.delta_pnl)]
        avg_model_pnl = float(np.mean(model_pnls))
        avg_oracle_pnl = float(np.mean(oracle_pnls)) if oracle_pnls else float("nan")
        avg_delta = float(np.mean(delta_pnls)) if delta_pnls else float("nan")
    else:
        avg_model_pnl = avg_oracle_pnl = avg_delta = float("nan")

    # Label quality distribution
    lq_values = [t.label_quality for t in labelable if t.bar_is_labelable]
    noisy_bars_pct = sum(1 for lq in lq_values if lq < 0.01) / len(lq_values) if lq_values else 0.0

    # Exit reason breakdown (traded bars only)
    exit_reasons = Counter(t.exit_reason for t in traded if t.exit_reason)

    return {
        "n_bars": n_bars,
        "n_traded": n_traded,
        "n_no_trade": decisions.get("no_trade", 0),
        "trade_rate": n_traded / n_bars if n_bars else 0.0,
        "skip_reasons": dict(skip_reasons),
        "gate_accuracy": gate_accuracy,
        "selection_accuracy": selection_accuracy,
        "avg_model_pnl": avg_model_pnl,
        "avg_oracle_pnl": avg_oracle_pnl,
        "avg_delta_pnl": avg_delta,
        "noisy_bars_pct": noisy_bars_pct,
        "avg_label_quality": float(np.mean(lq_values)) if lq_values else 0.0,
        "exit_reasons": dict(exit_reasons),
    }


def print_trace_summary(traces: list[DecisionTrace]) -> None:
    """Print a human-readable trace summary."""
    s = trace_summary(traces)
    print(f"\n{'=' * 60}")
    print(f"  DECISION TRACE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Bars evaluated:     {s['n_bars']}")
    print(f"  Traded:             {s['n_traded']}  ({s['trade_rate']:.1%})")
    print(f"  No-trade:           {s['n_no_trade']}")
    print(f"  Gate accuracy:      {s['gate_accuracy']:.1%}")
    print(f"  Selection accuracy: {s['selection_accuracy']:.1%}")
    print(f"  Avg model P&L:      {s['avg_model_pnl']:.4f}")
    print(f"  Avg oracle P&L:     {s['avg_oracle_pnl']:.4f}")
    print(f"  Avg delta (gap):    {s['avg_delta_pnl']:.4f}")
    print(f"  Noisy bars (<0.01): {s['noisy_bars_pct']:.1%}")
    print(f"  Avg label quality:  {s['avg_label_quality']:.4f}")
    if s.get("skip_reasons"):
        print(f"  Skip reasons:       {s['skip_reasons']}")
    if s.get("exit_reasons"):
        print(f"  Exit reasons:       {s['exit_reasons']}")
