"""Build label-uplift, sampler, null, and lifecycle attribution diagnostics.

This is a provisional offline packet for the Protocol101 fair-contract review.
It does not train, tune thresholds, contact brokers/vendors, download data,
change defaults, promote a candidate, or enable paper-submit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from v4.model.supervised_pilot import (
    SELECTION_MODE_STABLE_ABS_OFFSET_20,
    DecisionCandidates,
    Trade,
    entry_filter_mask,
    load_decisions,
    metrics_for_trades,
)
from v4.scripts.run_protocol101_fair_contract_gate_null_baselines import (
    empirical_lower_tail_p_value,
    empirical_p_value,
    eligible_grouped_decisions,
    null_result,
    percentile_summary,
    prediction_for_selected_keys,
    simulate_gate_only,
    stress_trades,
    write_csv,
    write_json,
)
from v4.scripts.run_protocol101_fair_contract_training_runner import (
    POLICY_META,
    load_json,
    paths_by_split,
)


DEFAULT_DESIGN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_expanded_jul_dec2025_128_q1_design/summary.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_uplift_lifecycle_attribution"
)
DEFAULT_ATTEMPT130_SELECTED = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_aggregate_edge_loss_diagnostic_attempt130/"
    "selected_trade_attribution.csv"
)
DEFAULT_ATTEMPT131_SELECTED = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_aggregate_edge_loss_diagnostic_attempt131/"
    "selected_trade_attribution.csv"
)
DEFAULT_ATTEMPT130_SCORES = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_aggregate_edge_loss_diagnostic_attempt130/"
    "candidate_scores.csv"
)
DEFAULT_ATTEMPT131_SCORES = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_aggregate_edge_loss_diagnostic_attempt131/"
    "candidate_scores.csv"
)
BASE_GATE = "put_near_after_0940_vwap_m2_10"
NEAR_VWAP_GATE = "put_near_after_0940_vwap_m2_10_near_vwap"
PREMIUM_GATE = "put_near_after_0940_vwap_m2_10_premium_gte_7_5"


@dataclass(frozen=True)
class SelectedValue:
    split: str
    session: str
    decision_time: str
    contract_id: str
    gate: str
    v_sel: float
    v_band: float
    v_policy1: float
    v_policy2: float
    entry_ask: float
    gate_age_bucket: str
    time_bucket: str
    momentum5_side: str
    momentum15_side: str
    vwap_gap_bucket: str
    premium_bucket: str
    return_on_premium: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy-index", type=int, choices=sorted(POLICY_META), default=0)
    parser.add_argument("--selection-mode", default=SELECTION_MODE_STABLE_ABS_OFFSET_20)
    parser.add_argument("--max-trades-per-session", type=int, default=3)
    parser.add_argument("--max-daily-loss", type=float, default=500.0)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--stress-per-trade", type=float, default=20.0)
    parser.add_argument("--null-seeds", type=int, default=1000)
    parser.add_argument("--null-seed-start", type=int, default=0)
    parser.add_argument("--attempt130-selected", type=Path, default=DEFAULT_ATTEMPT130_SELECTED)
    parser.add_argument("--attempt131-selected", type=Path, default=DEFAULT_ATTEMPT131_SELECTED)
    parser.add_argument("--attempt130-scores", type=Path, default=DEFAULT_ATTEMPT130_SCORES)
    parser.add_argument("--attempt131-scores", type=Path, default=DEFAULT_ATTEMPT131_SCORES)
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def side(value: float, *, flat_band: float = 0.0, prefix: str) -> str:
    if not math.isfinite(value):
        return f"{prefix}_unknown"
    if value > flat_band:
        return f"{prefix}_pos"
    if value < -flat_band:
        return f"{prefix}_neg"
    return f"{prefix}_flat"


def time_bucket(value: datetime) -> str:
    local = value.astimezone(ZoneInfo("America/New_York"))
    minute = local.hour * 60 + local.minute
    if minute < 10 * 60:
        return "open_0931_0959"
    if minute < 11 * 60 + 30:
        return "morning_1000_1129"
    if minute < 13 * 60 + 30:
        return "midday_1130_1329"
    if minute < 15 * 60:
        return "afternoon_1330_1459"
    return "late_1500_1530"


def vwap_gap_bucket(value: float) -> str:
    if not math.isfinite(value):
        return "vwap_gap_unknown"
    if value < -2.0:
        return "vwap_gap_lt_m2"
    if value < 0.0:
        return "vwap_gap_m2_0"
    if value < 2.0:
        return "vwap_gap_0_2"
    if value < 10.0:
        return "vwap_gap_2_10"
    return "vwap_gap_gte_10"


def premium_bucket(value: float) -> str:
    if not math.isfinite(value):
        return "premium_unknown"
    if value < 3.0:
        return "premium_lt_3"
    if value < 7.5:
        return "premium_3_7_5"
    if value < 15.0:
        return "premium_7_5_15"
    return "premium_gte_15"


def gate_age_bucket(age: int) -> str:
    if age <= 0:
        return "gate_age_inactive"
    if age == 1:
        return "gate_age_1"
    if age <= 3:
        return "gate_age_2_3"
    if age <= 10:
        return "gate_age_4_10"
    return "gate_age_gt_10"


def selected_key(session: str, decision_time: str) -> tuple[str, str]:
    return (str(session), str(decision_time))


def load_raw_rows(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(paths):
        session = path.name.removesuffix(".pkl")
        with path.open("rb") as handle:
            out[session] = pickle.load(handle)
    return out


def decision_from_raw_row(session: str, row: dict[str, Any], *, policy_index: int) -> DecisionCandidates | None:
    from v4.model.supervised_pilot import decision_candidates_from_row

    return decision_candidates_from_row(session=session, row=row, policy_index=policy_index)


def stable_selected_index(decision: DecisionCandidates, *, gate: str) -> int | None:
    allowed = entry_filter_mask(decision, gate)
    if len(allowed) != len(decision.labels):
        return None
    if not np.any(allowed):
        return None
    offsets = np.asarray(decision.offsets, dtype=np.float32)
    rights = np.asarray(decision.rights, dtype=object)
    eligible = np.flatnonzero(allowed)
    return min(
        (int(idx) for idx in eligible),
        key=lambda idx: (
            abs(abs(float(offsets[idx])) - 20.0),
            abs(float(offsets[idx])),
            float(offsets[idx]),
            str(rights[idx]),
            idx,
        ),
    )


def value_from_decision(
    *,
    split: str,
    session: str,
    row: dict[str, Any],
    decision0: DecisionCandidates,
    decision1: DecisionCandidates,
    decision2: DecisionCandidates,
    gate: str,
    age: int,
) -> SelectedValue | None:
    idx = stable_selected_index(decision0, gate=gate)
    if idx is None:
        return None
    contract_id = ""
    if decision0.contract_ids is not None:
        contract_id = str(decision0.contract_ids[idx])
    entry_ask = float("nan")
    if decision0.entry_asks is not None and idx < len(decision0.entry_asks):
        entry_ask = safe_float(decision0.entry_asks[idx])
    allowed = entry_filter_mask(decision0, gate)
    band_values = np.asarray(decision0.labels, dtype=float)[np.asarray(allowed, dtype=bool)]
    v_band = float(np.median(band_values)) if len(band_values) else float("nan")
    market_last = np.asarray(row["market_window"], dtype=np.float32)[-1]
    spx_close = safe_float(market_last[0])
    spx_vwap = safe_float(market_last[2])
    v_sel = safe_float(decision0.labels[idx])
    return SelectedValue(
        split=split,
        session=session,
        decision_time=decision0.decision_time.isoformat(),
        contract_id=contract_id,
        gate=gate,
        v_sel=v_sel,
        v_band=v_band,
        v_policy1=safe_float(decision1.labels[idx]) if idx < len(decision1.labels) else float("nan"),
        v_policy2=safe_float(decision2.labels[idx]) if idx < len(decision2.labels) else float("nan"),
        entry_ask=entry_ask,
        gate_age_bucket=gate_age_bucket(age),
        time_bucket=time_bucket(decision0.decision_time),
        momentum5_side=side(safe_float(market_last[5]), prefix="mom5"),
        momentum15_side=side(safe_float(market_last[6]), prefix="mom15"),
        vwap_gap_bucket=vwap_gap_bucket(spx_close - spx_vwap),
        premium_bucket=premium_bucket(entry_ask),
        return_on_premium=(
            float(v_sel) / max(entry_ask * 100.0, 1.0)
            if math.isfinite(v_sel) and math.isfinite(entry_ask) and entry_ask > 0
            else float("nan")
        ),
    )


def build_value_maps(
    *,
    split: str,
    raw0: dict[str, list[dict[str, Any]]],
    raw1: dict[str, list[dict[str, Any]]],
    raw2: dict[str, list[dict[str, Any]]],
    gates: Sequence[str],
) -> dict[str, dict[tuple[str, str], SelectedValue]]:
    out: dict[str, dict[tuple[str, str], SelectedValue]] = {gate: {} for gate in gates}
    age_by_gate: dict[str, int] = {gate: 0 for gate in gates}
    for session in sorted(raw0):
        age_by_gate = {gate: 0 for gate in gates}
        rows0 = raw0[session]
        rows1 = raw1.get(session) or []
        rows2 = raw2.get(session) or []
        for i, row0 in enumerate(rows0):
            if i >= len(rows1) or i >= len(rows2):
                continue
            decision0 = decision_from_raw_row(session, row0, policy_index=0)
            decision1 = decision_from_raw_row(session, rows1[i], policy_index=1)
            decision2 = decision_from_raw_row(session, rows2[i], policy_index=2)
            if decision0 is None or decision1 is None or decision2 is None:
                for gate in gates:
                    age_by_gate[gate] = 0
                continue
            key = selected_key(session, decision0.decision_time.isoformat())
            for gate in gates:
                active = stable_selected_index(decision0, gate=gate) is not None
                age_by_gate[gate] = age_by_gate[gate] + 1 if active else 0
                if not active:
                    continue
                value = value_from_decision(
                    split=split,
                    session=session,
                    row=row0,
                    decision0=decision0,
                    decision1=decision1,
                    decision2=decision2,
                    gate=gate,
                    age=age_by_gate[gate],
                )
                if value is not None:
                    out[gate][key] = value
    return out


def stats(values: Sequence[float], *, sessions: Sequence[str] | None = None) -> dict[str, Any]:
    arr = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "winsor600_mean": 0.0,
            "trim10_mean": 0.0,
            "median": 0.0,
            "share_positive": 0.0,
            "sum": 0.0,
            "top5pct_contribution_share": 0.0,
            "mean_without_top_session": 0.0,
        }
    winsor = np.clip(arr, -600.0, 600.0)
    sorted_arr = np.sort(arr)
    trim_n = int(math.floor(0.10 * len(sorted_arr)))
    trimmed = sorted_arr[trim_n:-trim_n] if trim_n and len(sorted_arr) > 2 * trim_n else sorted_arr
    top_n = max(int(math.ceil(0.05 * len(sorted_arr))), 1)
    top_sum = float(np.sort(arr)[-top_n:].sum())
    total = float(arr.sum())
    mean_without_top_session = float(np.mean(arr))
    if sessions is not None and len(sessions) == len(values):
        by_session: dict[str, float] = defaultdict(float)
        for session, value in zip(sessions, values):
            if math.isfinite(float(value)):
                by_session[str(session)] += float(value)
        if len(by_session) > 1:
            top_session = max(by_session, key=lambda key: by_session[key])
            kept = [float(value) for session, value in zip(sessions, values) if str(session) != str(top_session)]
            mean_without_top_session = float(np.mean(kept)) if kept else 0.0
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "winsor600_mean": float(np.mean(winsor)),
        "trim10_mean": float(np.mean(trimmed)) if len(trimmed) else 0.0,
        "median": float(np.median(arr)),
        "share_positive": float(np.mean(arr > 0.0)),
        "sum": total,
        "top5pct_contribution_share": float(top_sum / total) if abs(total) > 1e-9 else 0.0,
        "mean_without_top_session": mean_without_top_session,
    }


def bootstrap_winsor_mean_ci(
    values: Sequence[float],
    sessions: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    pairs = [
        (str(session), float(value))
        for session, value in zip(sessions, values)
        if math.isfinite(float(value))
    ]
    if not pairs:
        return 0.0, 0.0
    by_session: dict[str, list[float]] = defaultdict(list)
    for session, value in pairs:
        by_session[session].append(value)
    session_keys = sorted(by_session)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        sampled: list[float] = []
        for session in rng.choice(session_keys, size=len(session_keys), replace=True):
            sampled.extend(by_session[str(session)])
        if sampled:
            draws.append(float(np.mean(np.clip(np.asarray(sampled, dtype=float), -600.0, 600.0))))
    if not draws:
        return 0.0, 0.0
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _component_values(
    all_values: Sequence[SelectedValue],
    selected_values: Sequence[SelectedValue],
) -> tuple[float, float]:
    all_v_sel = np.asarray([value.v_sel for value in all_values], dtype=float)
    all_v_band = np.asarray([value.v_band for value in all_values], dtype=float)
    sel_v_sel = np.asarray([value.v_sel for value in selected_values], dtype=float)
    sel_v_band = np.asarray([value.v_band for value in selected_values], dtype=float)
    if len(all_v_sel) == 0 or len(sel_v_sel) == 0:
        return 0.0, 0.0
    e = float(np.nanmean(sel_v_band) - np.nanmean(all_v_band))
    c = float((np.nanmean(sel_v_sel) - np.nanmean(sel_v_band)) - (np.nanmean(all_v_sel) - np.nanmean(all_v_band)))
    return e, c


def bootstrap_component_ci(
    all_values: Sequence[SelectedValue],
    selected_values: Sequence[SelectedValue],
    *,
    n_boot: int = 1000,
    seed: int = 43,
) -> dict[str, float]:
    if not all_values or not selected_values:
        return {
            "bootstrap_ci_E_low": 0.0,
            "bootstrap_ci_E_high": 0.0,
            "bootstrap_ci_C_low": 0.0,
            "bootstrap_ci_C_high": 0.0,
        }
    sessions = sorted({value.session for value in all_values} | {value.session for value in selected_values})
    all_by_session: dict[str, list[SelectedValue]] = defaultdict(list)
    selected_by_session: dict[str, list[SelectedValue]] = defaultdict(list)
    for value in all_values:
        all_by_session[value.session].append(value)
    for value in selected_values:
        selected_by_session[value.session].append(value)
    rng = np.random.default_rng(seed)
    e_draws: list[float] = []
    c_draws: list[float] = []
    for _ in range(n_boot):
        boot_all: list[SelectedValue] = []
        boot_selected: list[SelectedValue] = []
        for session in rng.choice(sessions, size=len(sessions), replace=True):
            boot_all.extend(all_by_session[str(session)])
            boot_selected.extend(selected_by_session[str(session)])
        if boot_all and boot_selected:
            e, c = _component_values(boot_all, boot_selected)
            e_draws.append(e)
            c_draws.append(c)
    if not e_draws:
        return {
            "bootstrap_ci_E_low": 0.0,
            "bootstrap_ci_E_high": 0.0,
            "bootstrap_ci_C_low": 0.0,
            "bootstrap_ci_C_high": 0.0,
        }
    return {
        "bootstrap_ci_E_low": float(np.percentile(e_draws, 2.5)),
        "bootstrap_ci_E_high": float(np.percentile(e_draws, 97.5)),
        "bootstrap_ci_C_low": float(np.percentile(c_draws, 2.5)),
        "bootstrap_ci_C_high": float(np.percentile(c_draws, 97.5)),
    }


def value_rows_for_keys(
    value_map: dict[tuple[str, str], SelectedValue],
    keys: Iterable[tuple[str, str]],
) -> list[SelectedValue]:
    return [value_map[key] for key in keys if key in value_map]


def selected_keys_from_trades(trades: Sequence[Trade]) -> set[tuple[str, str]]:
    return {(str(trade.session), str(trade.decision_time)) for trade in trades}


def selected_keys_from_csv(path: Path, *, split: str) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split") or "") != split:
                continue
            keys.add(selected_key(str(row.get("session") or ""), str(row.get("decision_time") or "")))
    return keys


def session_counts_from_keys(keys: Iterable[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for session, _time in keys:
        counts[str(session)] += 1
    return dict(counts)


def slot_schedule_keys(value_map: dict[tuple[str, str], SelectedValue]) -> set[tuple[str, str]]:
    windows = (
        ("win_0940_1030", 9 * 60 + 40, 10 * 60 + 30),
        ("win_1030_1200", 10 * 60 + 30, 12 * 60),
        ("win_1300_1500", 13 * 60, 15 * 60),
    )
    by_session_window: dict[tuple[str, str], tuple[str, str]] = {}
    for key, value in sorted(value_map.items(), key=lambda item: (item[0][0], item[0][1])):
        dt = datetime.fromisoformat(value.decision_time).astimezone(ZoneInfo("America/New_York"))
        minute = dt.hour * 60 + dt.minute
        for name, start, end in windows:
            if start <= minute < end:
                by_session_window.setdefault((value.session, name), key)
                break
    return set(by_session_window.values())


def topk_parallel_keys_from_scores(
    *,
    score_csv: Path,
    value_map: dict[tuple[str, str], SelectedValue],
    session_counts: dict[str, int],
) -> set[tuple[str, str]]:
    contract_by_key = {key: value.contract_id for key, value in value_map.items()}
    score_by_key: dict[tuple[str, str], float] = {}
    if not score_csv.exists():
        return set()
    wanted_sessions = set(session_counts)
    with score_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            session = str(row.get("session") or "")
            if session not in wanted_sessions:
                continue
            key = selected_key(session, str(row.get("decision_time") or ""))
            if key not in contract_by_key:
                continue
            if str(row.get("contract_id") or "") != contract_by_key[key]:
                continue
            score_by_key[key] = safe_float(row.get("score"))
    by_session: dict[str, list[tuple[tuple[str, str], float]]] = defaultdict(list)
    for key, score in score_by_key.items():
        if math.isfinite(score):
            by_session[key[0]].append((key, score))
    selected: set[tuple[str, str]] = set()
    for session, items in by_session.items():
        k = int(session_counts.get(session, 0))
        for key, _score in sorted(items, key=lambda item: item[1], reverse=True)[:k]:
            selected.add(key)
    return selected


def simulate_masked_keys(
    decisions: Sequence[DecisionCandidates],
    *,
    keys: set[tuple[str, str]],
    gate: str,
    cooldown_minutes: int,
    strategy: str,
    max_trades_per_session: int,
    max_daily_loss: float,
    starting_cash: float,
    stress_per_trade: float,
) -> list[Trade]:
    from v4.model.supervised_pilot import simulate_model_policy

    return simulate_model_policy(
        decisions,
        prediction_for_selected_keys(decisions, keys),
        threshold=0.0,
        cooldown_minutes=cooldown_minutes,
        strategy=strategy,
        entry_filter=gate,
        min_score_margin=0.0,
        max_score_ceiling=0.0,
        max_trades_per_session=max_trades_per_session,
        max_daily_loss=max_daily_loss,
        selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
        starting_cash=starting_cash,
        cash_pnl_adjustment=-float(stress_per_trade),
    )


def strategy_stats_row(
    *,
    split: str,
    gate: str,
    strategy: str,
    values: Sequence[SelectedValue],
) -> dict[str, Any]:
    v_sel = [value.v_sel for value in values]
    rop = [value.return_on_premium for value in values]
    base = stats(v_sel, sessions=[value.session for value in values])
    rop_stats = stats(rop, sessions=[value.session for value in values])
    ci_low, ci_high = bootstrap_winsor_mean_ci(
        v_sel,
        [value.session for value in values],
    )
    return {
        "fold_id": "march2026_current_split",
        "evidence_grade": "provisional_current_march_splits_only",
        "split": split,
        "gate": gate,
        "strategy": strategy,
        **{f"v_sel_{key}": value for key, value in base.items()},
        "v_sel_winsor600_mean_boot_ci_low": ci_low,
        "v_sel_winsor600_mean_boot_ci_high": ci_high,
        "return_on_premium_mean": rop_stats["mean"],
        "return_on_premium_median": rop_stats["median"],
    }


def stratum_rows(
    *,
    split: str,
    gate: str,
    strategy: str,
    values: Sequence[SelectedValue],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    strata = {
        "gate_age": [value.gate_age_bucket for value in values],
        "time_bucket": [value.time_bucket for value in values],
        "momentum5_side": [value.momentum5_side for value in values],
        "momentum15_side": [value.momentum15_side for value in values],
        "vwap_gap_bucket": [value.vwap_gap_bucket for value in values],
        "premium_bucket": [value.premium_bucket for value in values],
    }
    for key, labels in strata.items():
        grouped: dict[str, list[SelectedValue]] = defaultdict(list)
        for label, value in zip(labels, values):
            grouped[str(label)].append(value)
        for label, group in grouped.items():
            row = strategy_stats_row(split=split, gate=gate, strategy=strategy, values=group)
            row["stratum_key"] = key
            row["stratum_value"] = label
            out.append(row)
    return out


def attribution_row(
    *,
    split: str,
    gate: str,
    strategy: str,
    all_values: Sequence[SelectedValue],
    selected_values: Sequence[SelectedValue],
    parallel_values: Sequence[SelectedValue] | None,
    train_alt_policy: int,
) -> dict[str, Any]:
    all_v_sel = np.asarray([value.v_sel for value in all_values], dtype=float)
    all_v_band = np.asarray([value.v_band for value in all_values], dtype=float)
    sel_v_sel = np.asarray([value.v_sel for value in selected_values], dtype=float)
    sel_v_band = np.asarray([value.v_band for value in selected_values], dtype=float)
    if len(all_v_sel) == 0 or len(sel_v_sel) == 0:
        g = e = c = identity = mean_sel = 0.0
    else:
        g = float(np.nanmean(all_v_sel))
        e = float(np.nanmean(sel_v_band) - np.nanmean(all_v_band))
        c = float((np.nanmean(sel_v_sel) - np.nanmean(sel_v_band)) - (np.nanmean(all_v_sel) - np.nanmean(all_v_band)))
        identity = g + e + c
        mean_sel = float(np.nanmean(sel_v_sel))
    p1_delta = [value.v_policy1 - value.v_sel for value in selected_values]
    p2_delta = [value.v_policy2 - value.v_sel for value in selected_values]
    serial_sampling_delta = 0.0
    if parallel_values:
        serial_sampling_delta = float(np.nanmean([value.v_sel for value in parallel_values]) - np.nanmean(sel_v_sel))
    row = {
        "fold_id": "march2026_current_split",
        "evidence_grade": "provisional_current_march_splits_only",
        "split": split,
        "gate": gate,
        "strategy": strategy,
        "n_all_gated": int(len(all_values)),
        "n_selected": int(len(selected_values)),
        "gate_value_G": g,
        "entry_uplift_E": e,
        "selection_delta_C": c,
        "identity_check_mean_V_sel": identity,
        "selected_mean_V_sel": mean_sel,
        "identity_error": float(identity - mean_sel),
        "lifecycle_delta_policy1": float(np.nanmean(p1_delta)) if p1_delta else 0.0,
        "lifecycle_delta_policy2": float(np.nanmean(p2_delta)) if p2_delta else 0.0,
        "train_designated_alt_policy": int(train_alt_policy),
        "train_designated_lifecycle_delta": (
            float(np.nanmean(p1_delta))
            if train_alt_policy == 1 and p1_delta
            else float(np.nanmean(p2_delta))
            if train_alt_policy == 2 and p2_delta
            else 0.0
        ),
        "serial_sampling_delta_A": serial_sampling_delta,
    }
    row.update(bootstrap_component_ci(all_values, selected_values))
    return row


def choose_train_alt_policy(
    *,
    train_values_by_gate: dict[str, dict[tuple[str, str], SelectedValue]],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for gate, value_map in train_values_by_gate.items():
        values = list(value_map.values())
        if not values:
            out[gate] = 1
            continue
        p1 = float(np.nanmean([value.v_policy1 - value.v_sel for value in values]))
        p2 = float(np.nanmean([value.v_policy2 - value.v_sel for value in values]))
        out[gate] = 1 if p1 >= p2 else 2
    return out


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    design = load_json(args.design)
    manifest = load_json(Path(design["allowed_data"]["canonical_manifest"]))
    paths, blockers = paths_by_split(design, manifest)
    if blockers:
        raise SystemExit(f"split path blockers: {blockers}")
    policy_name, cooldown_minutes = POLICY_META[int(args.policy_index)]
    gates = [BASE_GATE, NEAR_VWAP_GATE, PREMIUM_GATE]
    split_names = ["validation", "diagnostic_test"]

    raw_by_policy: dict[int, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for policy_index in (0, 1, 2):
        raw_by_policy[policy_index] = {
            split: load_raw_rows(paths[split])
            for split in ["train", "validation", "diagnostic_test"]
        }
    decisions_by_split = {
        split: load_decisions(paths[split], policy_index=0)
        for split in split_names
    }

    value_maps: dict[str, dict[str, dict[tuple[str, str], SelectedValue]]] = {}
    for split in ["train", "validation", "diagnostic_test"]:
        value_maps[split] = build_value_maps(
            split=split,
            raw0=raw_by_policy[0][split],
            raw1=raw_by_policy[1][split],
            raw2=raw_by_policy[2][split],
            gates=gates,
        )
    train_alt_policy = choose_train_alt_policy(train_values_by_gate=value_maps["train"])

    strategy_rows: list[dict[str, Any]] = []
    stratum_output_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": "Protocol101UpliftLifecycleAttributionV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "design": str(args.design),
        "selected_feature_contract": design.get("selected_feature_contract"),
        "policy_index": int(args.policy_index),
        "policy_name": policy_name,
        "cooldown_minutes": int(cooldown_minutes),
        "selection_mode": str(args.selection_mode),
        "max_trades_per_session": int(args.max_trades_per_session),
        "max_daily_loss": float(args.max_daily_loss),
        "starting_cash": float(args.starting_cash),
        "stress_per_trade": float(args.stress_per_trade),
        "null_seeds": int(args.null_seeds),
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "model_training_executed": False,
        "threshold_selection_executed": False,
        "paid_data_downloaded": False,
        "recorder_days_used_for_selection": False,
        "evidence_grade": "provisional_current_march_splits_only",
        "train_designated_alt_policy_by_gate": train_alt_policy,
        "results": {},
    }

    attempt_specs = {
        "attempt130": {
            "gate": NEAR_VWAP_GATE,
            "selected_csv": args.attempt130_selected,
            "score_csv": args.attempt130_scores,
        },
        "attempt131": {
            "gate": BASE_GATE,
            "selected_csv": args.attempt131_selected,
            "score_csv": args.attempt131_scores,
        },
    }

    for gate in gates:
        summary["results"][gate] = {}
        for split in split_names:
            decisions = decisions_by_split[split]
            value_map = value_maps[split][gate]
            all_values = list(value_map.values())
            grouped_eligible = eligible_grouped_decisions(
                decisions,
                entry_filter=gate,
                selection_mode=str(args.selection_mode),
            )
            gate_trades = simulate_gate_only(
                decisions,
                entry_filter=gate,
                selection_mode=str(args.selection_mode),
                cooldown_minutes=cooldown_minutes,
                max_trades_per_session=int(args.max_trades_per_session),
                max_daily_loss=float(args.max_daily_loss),
                starting_cash=float(args.starting_cash),
                stress_per_trade=float(args.stress_per_trade),
                strategy=f"gate_first_eligible_{gate}",
            )
            gate_keys = selected_keys_from_trades(gate_trades)
            slot_keys = slot_schedule_keys(value_map)
            slot_trades = simulate_masked_keys(
                decisions,
                keys=slot_keys,
                gate=gate,
                cooldown_minutes=cooldown_minutes,
                strategy=f"slot_schedule_3win_{gate}",
                max_trades_per_session=int(args.max_trades_per_session),
                max_daily_loss=float(args.max_daily_loss),
                starting_cash=float(args.starting_cash),
                stress_per_trade=float(args.stress_per_trade),
            )
            strategies: dict[str, set[tuple[str, str]]] = {
                "all_gated_parallel": set(value_map),
                "gate_first_eligible": gate_keys,
                "slot_schedule_3win": selected_keys_from_trades(slot_trades),
            }
            for attempt_name, spec in attempt_specs.items():
                if spec["gate"] != gate:
                    continue
                serial_keys = selected_keys_from_csv(Path(spec["selected_csv"]), split=split)
                counts = session_counts_from_keys(serial_keys)
                topk_keys = topk_parallel_keys_from_scores(
                    score_csv=Path(spec["score_csv"]),
                    value_map=value_map,
                    session_counts=counts,
                )
                strategies[f"{attempt_name}_serial"] = serial_keys
                strategies[f"{attempt_name}_topk_parallel"] = topk_keys
                null = null_result(
                    decisions,
                    grouped_eligible=grouped_eligible,
                    entry_filter=gate,
                    selection_mode=str(args.selection_mode),
                    cooldown_minutes=cooldown_minutes,
                    max_trades_per_session=int(args.max_trades_per_session),
                    max_daily_loss=float(args.max_daily_loss),
                    starting_cash=float(args.starting_cash),
                    stress_per_trade=float(args.stress_per_trade),
                    seed_start=int(args.null_seed_start),
                    null_seeds=int(args.null_seeds),
                    match_counts=counts,
                    strategy_prefix=f"null_m_{attempt_name}_{gate}",
                )
                null_values = null.pop("_stressed_total_pnl_values")
                null.pop("rows")
                selected_values = value_rows_for_keys(value_map, serial_keys)
                candidate_stressed_pnl = sum(value.v_sel - float(args.stress_per_trade) for value in selected_values)
                null_rows.append(
                    {
                        "split": split,
                        "gate": gate,
                        "candidate": attempt_name,
                        "candidate_trade_count": len(selected_values),
                        "candidate_stressed_total_pnl": candidate_stressed_pnl,
                        "upper_tail_p_value": empirical_p_value(
                            candidate_value=candidate_stressed_pnl,
                            null_values=null_values,
                        ),
                        "lower_tail_p_value": empirical_lower_tail_p_value(
                            candidate_value=candidate_stressed_pnl,
                            null_values=null_values,
                        ),
                        **{
                            f"null_m_total_pnl_{key}": value
                            for key, value in null["stressed_total_pnl_distribution"].items()
                        },
                    }
                )
            for strategy, keys in strategies.items():
                selected_values = value_rows_for_keys(value_map, keys)
                strategy_rows.append(
                    strategy_stats_row(
                        split=split,
                        gate=gate,
                        strategy=strategy,
                        values=selected_values,
                    )
                )
                stratum_output_rows.extend(
                    stratum_rows(
                        split=split,
                        gate=gate,
                        strategy=strategy,
                        values=selected_values,
                    )
                )
                parallel_values = None
                if strategy.endswith("_serial"):
                    topk_name = strategy.replace("_serial", "_topk_parallel")
                    if topk_name in strategies:
                        parallel_values = value_rows_for_keys(value_map, strategies[topk_name])
                attribution_rows.append(
                    attribution_row(
                        split=split,
                        gate=gate,
                        strategy=strategy,
                        all_values=all_values,
                        selected_values=selected_values,
                        parallel_values=parallel_values,
                        train_alt_policy=train_alt_policy.get(gate, 1),
                    )
                )
            summary["results"][gate][split] = {
                "all_gated_minutes": len(all_values),
                "gate_first_eligible_raw_metrics": metrics_for_trades(gate_trades),
                "gate_first_eligible_stressed_metrics": metrics_for_trades(
                    stress_trades(gate_trades, stress_per_trade=float(args.stress_per_trade))
                ),
                "slot_schedule_raw_metrics": metrics_for_trades(slot_trades),
                "slot_schedule_stressed_metrics": metrics_for_trades(
                    stress_trades(slot_trades, stress_per_trade=float(args.stress_per_trade))
                ),
            }

    write_csv(out_dir / "strategy_uplift_rows.csv", strategy_rows)
    write_csv(out_dir / "stratum_uplift_rows.csv", stratum_output_rows)
    write_csv(out_dir / "lifecycle_attribution_rows.csv", attribution_rows)
    write_csv(out_dir / "candidate_null_m_rows.csv", null_rows)
    write_json(out_dir / "summary.json", summary)
    report_lines = [
        "# Protocol101 Label Uplift And Lifecycle Attribution Packet",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "Offline provisional diagnostic only. No training, threshold tuning, broker calls, paid downloads, "
        "paper-submit, default changes, promotion changes, or recorder-day selection use occurred.",
        "",
        "## Gate / Sampler Summary",
        "",
        "| Gate | Split | First-Eligible PnL | Slot PnL | All-Gated Winsor Mean |",
        "|---|---:|---:|---:|---:|",
    ]
    strategy_by_key = {(row["gate"], row["split"], row["strategy"]): row for row in strategy_rows}
    for gate in gates:
        for split in split_names:
            first = strategy_by_key.get((gate, split, "gate_first_eligible"), {})
            slot = strategy_by_key.get((gate, split, "slot_schedule_3win"), {})
            allg = strategy_by_key.get((gate, split, "all_gated_parallel"), {})
            report_lines.append(
                f"| `{gate}` | {split} | ${float(first.get('v_sel_sum', 0.0)) - float(args.stress_per_trade) * int(first.get('v_sel_n', 0)):,.0f} | "
                f"${float(slot.get('v_sel_sum', 0.0)) - float(args.stress_per_trade) * int(slot.get('v_sel_n', 0)):,.0f} | "
                f"${float(allg.get('v_sel_winsor600_mean', 0.0)):,.2f} |"
            )
    report_lines.extend(
        [
            "",
            "## Candidate-Matched Null Rows",
            "",
            "| Candidate | Gate | Split | Candidate PnL | Null-M p95 | Upper p | Lower p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in null_rows:
        report_lines.append(
            f"| {row['candidate']} | `{row['gate']}` | {row['split']} | "
            f"${float(row['candidate_stressed_total_pnl']):,.0f} | "
            f"${float(row['null_m_total_pnl_p95']):,.0f} | "
            f"{float(row['upper_tail_p_value']):.4f} | {float(row['lower_tail_p_value']):.4f} |"
        )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
