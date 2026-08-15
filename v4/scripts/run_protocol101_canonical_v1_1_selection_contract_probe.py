"""Protocol101 canonical v1.1 selection-contract probe.

Offline-only burned-day audit. This proposes and tests a transfer-stable slot
selection contract for the canonical v1 quote-derived feature battery without
changing runtime contracts or training on the 15-month corpus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol101_canonical_v1_l0_l2_design_audit import (
    FEATURE_FAMILY,
    TRACE_PREFIX,
    build_slot_frame,
)
from v4.scripts.run_protocol101_canonical_v1_l1_l3_probe_audit import (
    DEFAULT_GROUP1_DEFINITION,
    DEFAULT_GROUP1_DIR,
    DEFAULT_L0_L2_DIR,
    DEFAULT_OUT_DIR as L1_L3_OUT_DIR,
    GROUP1_NON_VIX_FEATURES,
    L3_SUBSETS,
    add_group1_features,
    add_plane_returns,
    assign_terciles,
    finite,
    finite_series,
    l1_score,
    l3_feature_columns,
    load_authorized_features,
    matrix,
    minute_bucket,
    pair_slot_frame,
    train_l3_model,
)


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_RECON_DIR = BASE_AUDIT / "protocol101_canonical_v1_probe_rejection_reconciliation_attempt001"
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_canonical_v1_1_selection_contract_probe_attempt002"
SCHEMA_VERSION = "Protocol101CanonicalV11SelectionContractProbeAttempt002"
CONTRACT = "protocol101-live-v2-microstructure-masked"
TRANSFORM = "mask_vendor_sensitive_option_quote_greek_microstructure"
PRIMARY_K = 2.0
DIAGNOSTIC_KS = [1.0, 3.0]
SCORE_EPS = 1e-12
ACTION_AGREEMENT_MIN = 0.98
ACTION_AGREEMENT_REJECT_BELOW = 0.90
MUTUAL_CONFIDENT_SLOT_AGREEMENT_MIN = 0.99
MIN_SELECTED_SLOT_GATE_N = 30
TOTAL_DISAGREEMENT_REDUCTION_MIN = 0.60
MATERIAL_TRUE_REORDER_REDUCTION_MIN = 0.45
FEE_MATERIALITY_DOLLARS = 3.0
THRESHOLD_ADJACENT_ABS = 0.05
MIN_CONCENTRATION_N = 30


L1_PROBE_FAMILIES = {
    "random_with_guards_null": "control",
    "vwap_side_alignment": "G1",
    "spx_momentum_following": "G1",
    "spx_mean_reversion": "G1",
    "option_mid_momentum": "C",
    "straddle_mid_expansion": "D",
    "put_call_ratio_skew": "D",
    "internal_iv_expansion_compression": "E",
    "internal_delta_geometry": "E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l0-l2-dir", type=Path, default=DEFAULT_L0_L2_DIR)
    parser.add_argument("--l1-l3-dir", type=Path, default=L1_L3_OUT_DIR)
    parser.add_argument("--reconciliation-dir", type=Path, default=DEFAULT_RECON_DIR)
    parser.add_argument("--group1-dir", type=Path, default=DEFAULT_GROUP1_DIR)
    parser.add_argument("--group1-definition", type=Path, default=DEFAULT_GROUP1_DEFINITION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-prefix", default=TRACE_PREFIX)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def right_idx(value: Any) -> int:
    return 0 if str(value) == "C" else 1 if str(value) == "P" else 99


def add_static_indices(paired: pd.DataFrame) -> pd.DataFrame:
    frame = paired.copy()
    frame["strike_float"] = pd.to_numeric(frame["strike_historical"], errors="coerce")
    frame["strike_idx_rank"] = (
        frame.groupby(["session_date", "decision_minute_et"])["strike_float"]
        .rank(method="dense", ascending=True)
        .astype("Int64")
        - 1
    )
    frame["right_idx"] = frame["right"].map(right_idx).astype(int)
    return frame


def wilson_ci(successes: int, total: int, z: float = 1.96) -> dict[str, Any]:
    if total <= 0:
        return {"count": int(successes), "n": int(total), "share": None, "ci_low": None, "ci_high": None}
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return {
        "count": int(successes),
        "n": int(total),
        "share": float(p),
        "ci_low": float(max(0.0, center - half)),
        "ci_high": float(min(1.0, center + half)),
        "n_status": "sufficient_n" if total >= MIN_CONCENTRATION_N else "insufficient_n_descriptive_only",
    }


def load_attempt001_l1_rows(l1_summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in l1_summary.get("results", []):
        rows.append({key: value for key, value in row.items() if key != "threshold_rule"})
    return pd.DataFrame(rows)


def load_baseline_disagreements(recon_dir: Path) -> pd.DataFrame:
    path = recon_dir / "full_population_disagreement_attribution.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def baseline_counts(recon_summary: dict[str, Any], baseline_disagreements: pd.DataFrame) -> dict[str, int]:
    if baseline_disagreements.empty:
        material_true = int(recon_summary.get("material_non_threshold_instability_count") or 0)
        material_disagreements = int(recon_summary.get("economically_material_slot_swap_count") or 0)
    else:
        material_true = int(
            (
                baseline_disagreements["true_score_reordering"].fillna(False)
                & baseline_disagreements["selected_slot_swap_economically_material"].fillna(False)
            ).sum()
        )
        material_disagreements = int(
            baseline_disagreements["selected_slot_swap_economically_material"].fillna(False).sum()
        )
    return {
        "total_disagreements": int(recon_summary.get("total_disagreements") or len(baseline_disagreements)),
        "true_score_reordering_count": int(recon_summary.get("true_score_reordering_count") or 0),
        "material_true_score_reordering_count": material_true,
        "material_disagreement_count": material_disagreements,
    }


def build_paired_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    admitted, _, _ = load_authorized_features(args.l0_l2_dir, args.group1_dir)
    slot_frame, readiness = build_slot_frame(args.trace_prefix)
    if readiness.get("blockers") or slot_frame.empty:
        return pd.DataFrame(), admitted, {"blockers": readiness.get("blockers") or ["empty_slot_frame"], "trace_readiness": readiness}
    slot_frame = add_group1_features(slot_frame, args.trace_prefix, args.group1_definition)
    slot_frame = add_plane_returns(slot_frame)
    paired = pair_slot_frame(slot_frame, admitted + GROUP1_NON_VIX_FEATURES)
    paired = add_static_indices(assign_terciles(paired))
    paired["time_bucket"] = paired["decision_minute_et"].map(minute_bucket)
    return paired, admitted, {"blockers": [], "trace_readiness": readiness}


def abs_offset_band(values: pd.Series) -> pd.Series:
    numeric = finite_series(values).abs()
    band = np.floor(numeric / 10.0)
    band = band.where(numeric.notna(), np.nan)
    return band.clip(upper=9)


def band_aggregate_score(paired: pd.DataFrame, raw_scores: pd.Series, plane: str) -> pd.Series:
    suffix = f"_{plane}"
    work = paired[["session_date", "decision_minute_et", "right"]].copy()
    work["band"] = abs_offset_band(paired[f"B.ladder.abs_offset{suffix}"])
    work["raw_score"] = finite_series(raw_scores)
    grouped = work.groupby(["session_date", "decision_minute_et", "right", "band"], dropna=False)["raw_score"].transform("mean")
    return finite_series(grouped)


def l1_v11_score(paired: pd.DataFrame, probe: str, plane: str, thresholds: dict[str, Any]) -> pd.Series:
    raw = l1_score(paired, probe, plane, thresholds)
    if probe in {"option_mid_momentum", "internal_iv_expansion_compression"}:
        return band_aggregate_score(paired, raw, plane)
    return raw


def epsilon_policy(
    *,
    l1_summary: dict[str, Any],
    previous_l3: pd.DataFrame,
    recon_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    drift_map = dict(recon_summary.get("score_drift_p95_by_probe") or {})
    eps: dict[str, dict[str, Any]] = {}
    l1_zero_keep = {
        "random_with_guards_null",
        "vwap_side_alignment",
        "spx_momentum_following",
        "spx_mean_reversion",
    }
    for probe in L1_PROBE_FAMILIES:
        value = float(drift_map.get(probe, 0.0) or 0.0)
        source = "reconciliation_score_drift_p95"
        if probe in l1_zero_keep and value == 0.0:
            source = "l1_zero_drift_kept_zero_per_policy"
        if probe == "option_mid_momentum":
            value = float(drift_map["option_mid_momentum"])
            source = "original_option_mid_momentum_reconciliation_p95_for_band_aggregate_probe"
        if probe == "internal_iv_expansion_compression":
            value = float(drift_map["internal_iv_expansion_compression"])
            source = "original_internal_iv_reconciliation_p95_for_band_aggregate_probe"
        eps[probe] = {"epsilon": value, "source": source}
    for row in previous_l3.itertuples(index=False):
        probe = str(row.probe)
        value = float(drift_map.get(probe, 0.0) or 0.0)
        source = "reconciliation_score_drift_p95"
        if value == 0.0:
            value = float(getattr(row, "score_drift_max_abs"))
            source = "attempt001_l3_score_drift_max_abs_floor_because_p95_zero"
        eps[probe] = {"epsilon": value, "source": source}
    return eps


def write_selection_contract(
    *,
    out_dir: Path,
    l1_summary: dict[str, Any],
    previous_l3: pd.DataFrame,
    epsilon: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    l1_thresholds = {
        probe: rule.get("threshold")
        for probe, rule in (l1_summary.get("thresholds") or {}).items()
    }
    l3_thresholds = {str(row.probe): float(row.threshold) for row in previous_l3.itertuples(index=False)}
    contract = {
        "schema_version": "Protocol101CanonicalV11SelectionContractV1",
        "attempt_id": out_dir.name,
        "contract": CONTRACT,
        "transform": TRANSFORM,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "deterministic_candidate_ordering": [
            "highest_score",
            "lowest_strike_idx",
            "right_idx_C0_P1",
        ],
        "slot_margin_gate": {
            "primary_k": PRIMARY_K,
            "diagnostic_k": DIAGNOSTIC_KS,
            "confident_rule": "top2_score_margin > k * frozen_epsilon",
            "otherwise": "no_confident_slot_preference",
        },
        "fallback_rules_compared": [
            "nearest_atm_among_top_noise_equivalent_set_then_lowest_strike_idx_then_right_idx",
            "formal_ordering_among_top_noise_equivalent_set_lowest_strike_idx_then_right_idx",
        ],
        "routing_fallback_selection_policy": "choose the fallback rule with higher cross-plane fallback-slot agreement; tie goes to formal ordering; never choose by PnL",
        "unchanged_action_thresholds": {
            "l1": l1_thresholds,
            "l3": l3_thresholds,
        },
        "feature_families": {
            "canonical_v1_families_unchanged": True,
            "family_c_d_e_not_removed": True,
            "audit_local_l1_repairs": {
                "option_mid_momentum": "band-aggregate score by session/minute/right/10-point abs-offset band before selection",
                "internal_iv_expansion_compression": "band-aggregate score by session/minute/right/10-point abs-offset band before selection",
            },
            "per_slot_features_preserved_as_inputs_or_diagnostics": [
                "C.mid.logret_5m",
                "E.bs.iv",
            ],
        },
        "frozen_epsilons": epsilon,
        "forbidden_changes": {
            "threshold_optimization": True,
            "runtime_contract_modification": True,
            "paper_submit": True,
            "broker_api_calls": True,
            "paid_downloads": True,
            "promotion_or_default_changes": True,
        },
    }
    path = out_dir / "selection_contract_v1_1.json"
    write_json(path, contract)
    return contract, sha256_path(path)


def write_preregistration(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    contract_hash: str,
) -> None:
    prereg = {
        "schema_version": "Protocol101CanonicalV11SelectionContractPreregistrationV1",
        "attempt_id": out_dir.name,
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "selection_contract_path": str(out_dir / "selection_contract_v1_1.json"),
        "selection_contract_sha256": contract_hash,
        "inputs": {
            "l0_l2_dir": str(args.l0_l2_dir),
            "l1_l3_dir": str(args.l1_l3_dir),
            "reconciliation_dir": str(args.reconciliation_dir),
            "group1_dir": str(args.group1_dir),
        },
        "scope": "offline burned-day design/probe audit only",
        "same_burned_days": ["2026-06-30", "2026-07-01", "2026-07-02"],
        "primary_k": PRIMARY_K,
        "diagnostic_k": DIAGNOSTIC_KS,
        "side_effects_forbidden": {
            "fifteen_month_training": True,
            "uplift_cv": True,
            "threshold_optimization": True,
            "broker_api_calls": True,
            "paid_downloads": True,
            "paper_submit": True,
            "promotion_default_runtime_launchd_edits": True,
            "real_money_path_changes": True,
        },
    }
    write_json(out_dir / "preregistration.json", prereg)


def score_population_rows(
    *,
    paired: pd.DataFrame,
    kind: str,
    probe: str,
    feature_family: str,
    threshold: float,
    epsilon: float,
    historical_scores: pd.Series | np.ndarray,
    ibkr_scores: pd.Series | np.ndarray,
    subset: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    base_cols = [
        "session_date",
        "decision_minute_et",
        "strike_key",
        "right",
        "strike_float",
        "strike_idx_rank",
        "right_idx",
        "time_bucket",
        "realized_5m_spx_vol_tercile",
        "opportunity_tercile",
        "B.ladder.abs_offset_historical",
        "B.ladder.abs_offset_ibkr",
        "plane_pnl_15m_dollars_historical",
        "plane_pnl_15m_dollars_ibkr",
    ]
    hist = paired[base_cols].copy()
    hist["plane"] = "historical"
    hist["score"] = pd.Series(historical_scores, index=paired.index).to_numpy(dtype=float)
    hist["selected_pnl"] = hist["plane_pnl_15m_dollars_historical"]
    hist["abs_offset"] = hist["B.ladder.abs_offset_historical"]
    live = paired[base_cols].copy()
    live["plane"] = "ibkr"
    live["score"] = pd.Series(ibkr_scores, index=paired.index).to_numpy(dtype=float)
    live["selected_pnl"] = live["plane_pnl_15m_dollars_ibkr"]
    live["abs_offset"] = live["B.ladder.abs_offset_ibkr"]
    out = pd.concat([hist, live], ignore_index=True)
    out["kind"] = kind
    out["probe"] = probe
    out["subset"] = subset
    out["model"] = model
    out["feature_family"] = feature_family
    out["threshold"] = threshold
    out["epsilon"] = epsilon
    out["slot_id"] = out["strike_key"].astype(str) + "|" + out["right"].astype(str)
    return out


def top_noise_set(group: pd.DataFrame, top_score: float, k: float, epsilon: float) -> pd.DataFrame:
    return group[group["score"] >= top_score - (k * epsilon) - SCORE_EPS]


def choose_fallback(candidates: pd.DataFrame, rule: str) -> pd.Series:
    if candidates.empty:
        raise ValueError("empty fallback candidates")
    if rule == "nearest_atm":
        sort_cols = ["abs_offset", "strike_idx_rank", "right_idx"]
        ascending = [True, True, True]
    elif rule == "formal_ordering":
        sort_cols = ["strike_idx_rank", "right_idx"]
        ascending = [True, True]
    else:
        raise KeyError(rule)
    return candidates.sort_values(sort_cols, ascending=ascending, kind="mergesort").iloc[0]


def choose_top(group: pd.DataFrame) -> pd.Series:
    return group.sort_values(["score", "strike_idx_rank", "right_idx"], ascending=[False, True, True], kind="mergesort").iloc[0]


def select_decisions(population: pd.DataFrame, *, k: float, fallback_rule: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["kind", "probe", "subset", "model", "feature_family", "plane", "session_date", "decision_minute_et"]
    for keys, group in population.groupby(group_cols, sort=False, dropna=False):
        kind, probe, subset, model, feature_family, plane, session, minute = keys
        finite_group = group[group["score"].notna()]
        base = {
            "kind": kind,
            "probe": probe,
            "subset": subset,
            "model": model,
            "feature_family": feature_family,
            "plane": plane,
            "session_date": session,
            "decision_minute_et": minute,
            "k": k,
            "fallback_rule": fallback_rule,
        }
        if finite_group.empty:
            rows.append({**base, "action": "wait", "confidence_status": "no_enter", "selection_reason": "no_finite_scores"})
            continue
        top = choose_top(finite_group)
        ordered_scores = finite_group["score"].sort_values(ascending=False).to_numpy(dtype=float)
        top_score = float(ordered_scores[0])
        second_best = float(ordered_scores[1]) if len(ordered_scores) > 1 else None
        top2_margin = float(top_score - second_best) if second_best is not None else None
        threshold = float(top["threshold"])
        epsilon = float(top["epsilon"])
        action = "enter" if math.isfinite(threshold) and top_score >= threshold else "wait"
        top_tie = finite_group[np.isclose(finite_group["score"], top_score, atol=SCORE_EPS, rtol=0.0)]
        top_set = top_noise_set(finite_group, top_score, k, epsilon)
        confident = bool(action == "enter" and top2_margin is not None and top2_margin > (k * epsilon) + SCORE_EPS)
        nearest = choose_fallback(top_set, "nearest_atm")
        formal = choose_fallback(top_set, "formal_ordering")
        selected = top if confident else (nearest if fallback_rule == "nearest_atm" else formal)
        confidence_status = "no_enter"
        selection_reason = "top_score_below_threshold"
        if action == "enter" and confident:
            confidence_status = "confident"
            selection_reason = "top2_margin_gt_k_epsilon"
        elif action == "enter":
            confidence_status = "fallback"
            selection_reason = "no_confident_slot_preference"
        rows.append(
            {
                **base,
                "action": action,
                "confidence_status": confidence_status,
                "selection_reason": selection_reason,
                "selected_slot": selected["slot_id"] if action == "enter" else None,
                "selected_side": selected["right"] if action == "enter" else None,
                "selected_strike": finite(selected["strike_float"]) if action == "enter" else None,
                "selected_score": finite(selected["score"]) if action == "enter" else None,
                "selected_pnl": finite(selected["selected_pnl"]) if action == "enter" else None,
                "selected_abs_offset": finite(selected["abs_offset"]) if action == "enter" else None,
                "threshold": threshold,
                "epsilon": epsilon,
                "k_epsilon": k * epsilon,
                "top_score": top_score,
                "second_best_score": second_best,
                "top2_score_margin": top2_margin,
                "top2_margin_minus_k_epsilon": (top2_margin - (k * epsilon)) if top2_margin is not None else None,
                "margin_epsilon_ratio": (top2_margin / epsilon) if epsilon > 0 and top2_margin is not None else (math.inf if top2_margin and top2_margin > 0 else 0.0),
                "threshold_distance": top_score - threshold if math.isfinite(threshold) else None,
                "top_score_tie_set_size": int(len(top_tie)),
                "top_noise_equivalent_set_size": int(len(top_set)),
                "nearest_atm_fallback_slot": nearest["slot_id"],
                "formal_ordering_fallback_slot": formal["slot_id"],
                "fallback_slot": selected["slot_id"] if action == "enter" and not confident else None,
                "no_confident_slot_preference": bool(action == "enter" and not confident),
                "time_bucket": selected["time_bucket"],
                "realized_5m_spx_vol_tercile": selected["realized_5m_spx_vol_tercile"],
                "opportunity_tercile": selected["opportunity_tercile"],
            }
        )
    return pd.DataFrame.from_records(rows)


def merge_plane_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    hist = decisions[decisions["plane"] == "historical"]
    live = decisions[decisions["plane"] == "ibkr"]
    key_cols = ["kind", "probe", "subset", "model", "feature_family", "session_date", "decision_minute_et", "k", "fallback_rule"]
    merged = hist.merge(live, on=key_cols, suffixes=("_historical", "_ibkr"), how="inner", validate="one_to_one")
    merged["action_agree"] = merged["action_historical"] == merged["action_ibkr"]
    merged["both_enter"] = (merged["action_historical"] == "enter") & (merged["action_ibkr"] == "enter")
    merged["enter_either"] = (merged["action_historical"] == "enter") | (merged["action_ibkr"] == "enter")
    merged["selected_slot_agree"] = merged["both_enter"] & (
        merged["selected_slot_historical"] == merged["selected_slot_ibkr"]
    )
    merged["same_action_different_slot"] = merged["both_enter"] & ~merged["selected_slot_agree"].fillna(False)
    merged["mutually_confident_enter"] = merged["both_enter"] & (
        merged["confidence_status_historical"] == "confident"
    ) & (merged["confidence_status_ibkr"] == "confident")
    merged["split_confidence"] = merged["both_enter"] & (
        (
            (merged["confidence_status_historical"] == "confident")
            & (merged["confidence_status_ibkr"] == "fallback")
        )
        | (
            (merged["confidence_status_historical"] == "fallback")
            & (merged["confidence_status_ibkr"] == "confident")
        )
    )
    merged["both_fallback"] = merged["both_enter"] & (
        merged["confidence_status_historical"] == "fallback"
    ) & (merged["confidence_status_ibkr"] == "fallback")
    merged["disagreement"] = ~(
        merged["action_agree"] & ((~merged["both_enter"]) | merged["selected_slot_agree"].fillna(False))
    )
    return merged


def trade_set_jaccard(merged: pd.DataFrame) -> float:
    hist_set = set(
        zip(
            merged.loc[merged["action_historical"] == "enter", "session_date"],
            merged.loc[merged["action_historical"] == "enter", "decision_minute_et"],
            merged.loc[merged["action_historical"] == "enter", "selected_slot_historical"],
            strict=False,
        )
    )
    live_set = set(
        zip(
            merged.loc[merged["action_ibkr"] == "enter", "session_date"],
            merged.loc[merged["action_ibkr"] == "enter", "decision_minute_et"],
            merged.loc[merged["action_ibkr"] == "enter", "selected_slot_ibkr"],
            strict=False,
        )
    )
    union = hist_set | live_set
    return len(hist_set & live_set) / len(union) if union else 1.0


def margin_bucket(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "missing"
    numeric = float(value)
    if not math.isfinite(numeric):
        return ">5x"
    if numeric <= 1.0:
        return "<=1x"
    if numeric <= 2.0:
        return "1-2x"
    if numeric <= 3.0:
        return "2-3x"
    if numeric <= 5.0:
        return "3-5x"
    return ">5x"


def enrich_disagreement_classes(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["action_flip"] = out["action_historical"] != out["action_ibkr"]
    out["threshold_adjacent_action_flip"] = out["action_flip"] & (
        (pd.to_numeric(out["threshold_distance_historical"], errors="coerce").abs() < THRESHOLD_ADJACENT_ABS)
        | (pd.to_numeric(out["threshold_distance_ibkr"], errors="coerce").abs() < THRESHOLD_ADJACENT_ABS)
    )
    out["non_threshold_action_flip"] = out["action_flip"] & ~out["threshold_adjacent_action_flip"]
    out["both_confident_slot_disagreement"] = out["mutually_confident_enter"] & out["same_action_different_slot"]
    out["residual_tail_divergence"] = out["both_confident_slot_disagreement"] & (
        (pd.to_numeric(out["top2_score_margin_historical"], errors="coerce") > (2.0 * pd.to_numeric(out["epsilon_historical"], errors="coerce")))
        & (pd.to_numeric(out["top2_score_margin_ibkr"], errors="coerce") > (2.0 * pd.to_numeric(out["epsilon_ibkr"], errors="coerce")))
    )
    out["true_score_reordering"] = out["both_confident_slot_disagreement"] & (
        (pd.to_numeric(out["top_score_tie_set_size_historical"], errors="coerce") == 1)
        & (pd.to_numeric(out["top_score_tie_set_size_ibkr"], errors="coerce") == 1)
    )
    out["near_tie_or_tie_set_membership"] = out["same_action_different_slot"] & ~out["residual_tail_divergence"]
    out["pnl_delta"] = pd.to_numeric(out["selected_pnl_ibkr"], errors="coerce") - pd.to_numeric(
        out["selected_pnl_historical"], errors="coerce"
    )
    out["pnl_delta_missing"] = out["pnl_delta"].isna() & out["disagreement"]
    out["economically_material"] = out["disagreement"] & out["pnl_delta"].abs().gt(FEE_MATERIALITY_DOLLARS)
    ratio = pd.concat(
        [
            pd.to_numeric(out["margin_epsilon_ratio_historical"], errors="coerce").replace([np.inf, -np.inf], np.inf),
            pd.to_numeric(out["margin_epsilon_ratio_ibkr"], errors="coerce").replace([np.inf, -np.inf], np.inf),
        ],
        axis=1,
    ).min(axis=1)
    out["min_margin_epsilon_ratio"] = ratio
    out["margin_epsilon_ratio_bucket"] = [margin_bucket(value) for value in ratio]
    out["split_confidence_material"] = out["split_confidence"] & out["economically_material"]
    return out


def summarize_probe(merged: pd.DataFrame) -> dict[str, Any]:
    n = len(merged)
    both_enter = merged["both_enter"]
    disagreements = merged[merged["disagreement"]]
    mutual = merged["mutually_confident_enter"]
    split = merged["split_confidence"]
    selected_slot_agreement = float(merged.loc[both_enter, "selected_slot_agree"].mean()) if bool(both_enter.any()) else 1.0
    mutual_slot_n = int(mutual.sum())
    mutual_slot_agreement = (
        float(merged.loc[mutual, "selected_slot_agree"].mean()) if mutual_slot_n else 1.0
    )
    mutual_gate_status = "descriptive_only_low_n" if mutual_slot_n < MIN_SELECTED_SLOT_GATE_N else (
        "pass" if mutual_slot_agreement >= MUTUAL_CONFIDENT_SLOT_AGREEMENT_MIN else "fail"
    )
    top_vol = int((disagreements["realized_5m_spx_vol_tercile_historical"] == "top").sum()) if len(disagreements) else 0
    top_opp = int((disagreements["opportunity_tercile_historical"] == "top").sum()) if len(disagreements) else 0
    return {
        "kind": merged["kind"].iloc[0] if n else None,
        "probe": merged["probe"].iloc[0] if n else None,
        "subset": merged["subset"].iloc[0] if n else None,
        "model": merged["model"].iloc[0] if n else None,
        "feature_family": merged["feature_family"].iloc[0] if n else None,
        "fallback_rule": merged["fallback_rule"].iloc[0] if n else None,
        "minutes": int(n),
        "historical_actions": int((merged["action_historical"] == "enter").sum()),
        "ibkr_actions": int((merged["action_ibkr"] == "enter").sum()),
        "action_agreement": float(merged["action_agree"].mean()) if n else None,
        "selected_slot_agreement": selected_slot_agreement,
        "mutually_confident_enter_n": mutual_slot_n,
        "selected_slot_agreement_mutually_confident_enter": mutual_slot_agreement,
        "selected_slot_gate_status": mutual_gate_status,
        "trade_set_jaccard": trade_set_jaccard(merged),
        "disagreements": int(len(disagreements)),
        "threshold_adjacent_action_flips": int(merged["threshold_adjacent_action_flip"].sum()),
        "non_threshold_action_flips": int(merged["non_threshold_action_flip"].sum()),
        "no_confident_slot_preference_count": int(
            (merged["no_confident_slot_preference_historical"].fillna(False)).sum()
            + (merged["no_confident_slot_preference_ibkr"].fillna(False)).sum()
        ),
        "fallback_slot_count": int((merged["confidence_status_historical"] == "fallback").sum() + (merged["confidence_status_ibkr"] == "fallback").sum()),
        "both_confident_count": int(merged["mutually_confident_enter"].sum()),
        "both_fallback_count": int(merged["both_fallback"].sum()),
        "split_confidence_count": int(merged["split_confidence"].sum()),
        "split_confidence_selected_slot_agreement": float(merged.loc[split, "selected_slot_agree"].mean()) if bool(split.any()) else 1.0,
        "split_confidence_material_count": int(merged["split_confidence_material"].sum()),
        "material_pnl_delta_count": int(merged["economically_material"].sum()),
        "true_score_reordering_count": int(merged["true_score_reordering"].sum()),
        "material_true_score_reordering_count": int((merged["true_score_reordering"] & merged["economically_material"]).sum()),
        "residual_tail_divergence_count": int(merged["residual_tail_divergence"].sum()),
        "near_tie_tie_set_membership_count": int(merged["near_tie_or_tie_set_membership"].sum()),
        "top_volatility_concentration": wilson_ci(top_vol, len(disagreements)),
        "top_opportunity_concentration": wilson_ci(top_opp, len(disagreements)),
    }


def summarize_all(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in merged.groupby(["kind", "probe", "subset", "model"], dropna=False, sort=False):
        rows.append(summarize_probe(group))
    return pd.DataFrame(rows)


def summarize_fallback_agreement(nearest_pairs: pd.DataFrame, formal_pairs: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    for rule, pairs in (("nearest_atm", nearest_pairs), ("formal_ordering", formal_pairs)):
        fallback = pairs[pairs["both_enter"] & pairs["both_fallback"]].copy()
        rows.append(
            {
                "scope": "overall",
                "fallback_rule": rule,
                "both_fallback_minutes": int(len(fallback)),
                "fallback_slot_agreement": float(fallback["selected_slot_agree"].mean()) if len(fallback) else 1.0,
            }
        )
        for probe, group in fallback.groupby("probe", dropna=False):
            rows.append(
                {
                    "scope": str(probe),
                    "fallback_rule": rule,
                    "both_fallback_minutes": int(len(group)),
                    "fallback_slot_agreement": float(group["selected_slot_agree"].mean()) if len(group) else 1.0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "fallback_slot_agreement.csv", index=False)
    overall = out[out["scope"] == "overall"]
    nearest = float(overall[overall["fallback_rule"] == "nearest_atm"]["fallback_slot_agreement"].iloc[0])
    formal = float(overall[overall["fallback_rule"] == "formal_ordering"]["fallback_slot_agreement"].iloc[0])
    chosen = "nearest_atm" if nearest > formal else "formal_ordering"
    return out, chosen


def write_confidence_reports(merged: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    no_conf = []
    confidence = []
    margin = []
    for keys, group in merged.groupby(["kind", "probe", "subset", "model"], dropna=False, sort=False):
        kind, probe, subset, model = keys
        for plane in ("historical", "ibkr"):
            no_conf.append(
                {
                    "kind": kind,
                    "probe": probe,
                    "subset": subset,
                    "model": model,
                    "plane": plane,
                    "no_confident_slot_preference_count": int(group[f"no_confident_slot_preference_{plane}"].fillna(False).sum()),
                    "fallback_count": int((group[f"confidence_status_{plane}"] == "fallback").sum()),
                    "enter_count": int((group[f"action_{plane}"] == "enter").sum()),
                }
            )
        enter_either = group[group["enter_either"]]
        confidence.append(
            {
                "kind": kind,
                "probe": probe,
                "subset": subset,
                "model": model,
                "enter_either_minutes": int(len(enter_either)),
                "both_confident": int(group["mutually_confident_enter"].sum()),
                "both_fallback": int(group["both_fallback"].sum()),
                "split_confidence": int(group["split_confidence"].sum()),
                "confidence_status_agreement_rate": float(
                    (
                        group["confidence_status_historical"]
                        == group["confidence_status_ibkr"]
                    ).mean()
                ) if len(group) else None,
                "selected_slot_agreement_on_split_confidence": float(group.loc[group["split_confidence"], "selected_slot_agree"].mean())
                if bool(group["split_confidence"].any())
                else 1.0,
                "split_confidence_material_count": int(group["split_confidence_material"].sum()),
            }
        )
        split = group[group["split_confidence"]]
        margin.append(
            {
                "kind": kind,
                "probe": probe,
                "subset": subset,
                "model": model,
                "split_confidence_minutes": int(len(split)),
                "split_margin_minus_k_epsilon_min": finite(
                    pd.concat(
                        [
                            pd.to_numeric(split["top2_margin_minus_k_epsilon_historical"], errors="coerce"),
                            pd.to_numeric(split["top2_margin_minus_k_epsilon_ibkr"], errors="coerce"),
                        ],
                        ignore_index=True,
                    ).min()
                )
                if len(split)
                else None,
                "split_margin_minus_k_epsilon_median": finite(
                    pd.concat(
                        [
                            pd.to_numeric(split["top2_margin_minus_k_epsilon_historical"], errors="coerce"),
                            pd.to_numeric(split["top2_margin_minus_k_epsilon_ibkr"], errors="coerce"),
                        ],
                        ignore_index=True,
                    ).median()
                )
                if len(split)
                else None,
            }
        )
    no_conf_df = pd.DataFrame(no_conf)
    conf_df = pd.DataFrame(confidence)
    margin_df = pd.DataFrame(margin)
    no_conf_df.to_csv(out_dir / "no_confident_slot_preference_report.csv", index=False)
    conf_df.to_csv(out_dir / "confidence_status_agreement.csv", index=False)
    return no_conf_df, conf_df, margin_df


def write_economic_materiality(merged: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, group in [("overall", merged)] + [(str(probe), g) for probe, g in merged.groupby("probe", dropna=False)]:
        disag = group[group["disagreement"]]
        pnl = pd.to_numeric(disag["pnl_delta"], errors="coerce")
        rows.append(
            {
                "scope": scope,
                "disagreements": int(len(disag)),
                "material_pnl_delta_count": int(disag["economically_material"].sum()),
                "missing_pnl_count": int(disag["pnl_delta_missing"].sum()),
                "median_pnl_delta": finite(pnl.dropna().median()) if len(pnl.dropna()) else None,
                "p95_abs_pnl_delta": finite(pnl.dropna().abs().quantile(0.95)) if len(pnl.dropna()) else None,
                "true_score_reordering_count": int(disag["true_score_reordering"].sum()),
                "material_true_score_reordering_count": int((disag["true_score_reordering"] & disag["economically_material"]).sum()),
                "split_confidence_material_count": int(disag["split_confidence_material"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "economic_materiality_attempt002.csv", index=False)
    return out


def write_ratio_bucket_report(merged: pd.DataFrame, split_margin: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    true_reorder = merged[merged["true_score_reordering"]].copy()
    rows: list[dict[str, Any]] = []
    for row in split_margin.to_dict("records"):
        rows.append(
            {
                "report_type": "split_confidence_margin_distance",
                **row,
            }
        )
    for bucket in ["<=1x", "1-2x", "2-3x", "3-5x", ">5x", "missing"]:
        group = true_reorder[true_reorder["margin_epsilon_ratio_bucket"] == bucket]
        rows.append(
            {
                "report_type": "true_score_reordering_margin_epsilon_bucket",
                "bucket": bucket,
                "true_score_reordering_count": int(len(group)),
                "material_true_score_reordering_count": int(group["economically_material"].sum()) if len(group) else 0,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "margin_epsilon_ratio_report.csv", index=False)
    return out


def write_diagnostic_k_sensitivity(primary_pairs: pd.DataFrame, fallback_rule: str, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in DIAGNOSTIC_KS:
        pairs = primary_pairs.copy()
        hist_confident = (pairs["action_historical"] == "enter") & (
            pd.to_numeric(pairs["top2_score_margin_historical"], errors="coerce")
            > k * pd.to_numeric(pairs["epsilon_historical"], errors="coerce")
        )
        ibkr_confident = (pairs["action_ibkr"] == "enter") & (
            pd.to_numeric(pairs["top2_score_margin_ibkr"], errors="coerce")
            > k * pd.to_numeric(pairs["epsilon_ibkr"], errors="coerce")
        )
        hist_fallback = (pairs["action_historical"] == "enter") & ~hist_confident
        ibkr_fallback = (pairs["action_ibkr"] == "enter") & ~ibkr_confident
        both_confident = pairs["both_enter"] & hist_confident & ibkr_confident
        both_fallback = pairs["both_enter"] & hist_fallback & ibkr_fallback
        split_confidence = pairs["both_enter"] & (
            (hist_confident & ibkr_fallback) | (hist_fallback & ibkr_confident)
        )
        rows.append(
            {
                "k": k,
                "fallback_rule": fallback_rule,
                "fallback_rule_scope": "chosen_primary_k2_fallback_rule",
                "diagnostic_method": "confidence_status_sensitivity_on_primary_k2_selected_pairs",
                "total_disagreements_primary_k2": int(pairs["disagreement"].sum()),
                "both_confident": int(both_confident.sum()),
                "both_fallback": int(both_fallback.sum()),
                "split_confidence": int(split_confidence.sum()),
                "split_confidence_material_count_primary_selected_pairs": int(
                    (split_confidence & pairs["economically_material"]).sum()
                ),
                "residual_tail_divergence_primary_k2": int(pairs["residual_tail_divergence"].sum()),
                "route_use": "diagnostic_only_not_used_for_attempt002_routing",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "diagnostic_k_sensitivity.csv", index=False)
    return out


def build_populations(
    *,
    paired: pd.DataFrame,
    admitted: list[str],
    l1_summary: dict[str, Any],
    previous_l3: pd.DataFrame,
    epsilon: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    thresholds = l1_summary["thresholds"]
    populations: list[pd.DataFrame] = []
    for probe, family in L1_PROBE_FAMILIES.items():
        threshold = float(thresholds[probe]["threshold"])
        eps = float(epsilon[probe]["epsilon"])
        populations.append(
            score_population_rows(
                paired=paired,
                kind="L1",
                probe=probe,
                feature_family=family,
                threshold=threshold,
                epsilon=eps,
                historical_scores=l1_v11_score(paired, probe, "historical", thresholds),
                ibkr_scores=l1_v11_score(paired, probe, "ibkr", thresholds),
            )
        )
    train_mask = paired["y_15m_conservative_return_ibkr"].notna()
    y = (paired.loc[train_mask, "y_15m_conservative_return_ibkr"].astype(float) > 0).astype(int).to_numpy()
    for subset_name in L3_SUBSETS:
        cols = l3_feature_columns(subset_name, admitted)
        for model_name in ("logistic", "bounded_hgb"):
            probe = f"{subset_name}_{model_name}"
            threshold_row = previous_l3[previous_l3["probe"] == probe]
            if threshold_row.empty:
                continue
            threshold = float(threshold_row.iloc[0]["threshold"])
            eps = float(epsilon[probe]["epsilon"])
            model = train_l3_model(model_name)
            model.fit(matrix(paired.loc[train_mask], cols, "_historical"), y)
            hist_scores = model.predict_proba(matrix(paired, cols, "_historical"))[:, 1]
            ibkr_scores = model.predict_proba(matrix(paired, cols, "_ibkr"))[:, 1]
            populations.append(
                score_population_rows(
                    paired=paired,
                    kind="L3",
                    probe=probe,
                    feature_family="+".join(L3_SUBSETS[subset_name]) + "+G1",
                    threshold=threshold,
                    epsilon=eps,
                    historical_scores=hist_scores,
                    ibkr_scores=ibkr_scores,
                    subset=subset_name,
                    model=model_name,
                )
            )
    return pd.concat(populations, ignore_index=True)


def before_after(
    *,
    attempt001_l1: pd.DataFrame,
    attempt001_l3: pd.DataFrame,
    attempt002: pd.DataFrame,
    baseline_disagreements: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    prior = pd.concat([attempt001_l1, attempt001_l3], ignore_index=True, sort=False)
    prior = prior.rename(
        columns={
            "action_agreement": "attempt001_action_agreement",
            "selected_slot_agreement": "attempt001_selected_slot_agreement",
            "trade_set_jaccard": "attempt001_trade_set_jaccard",
            "disagreements": "attempt001_disagreements",
        }
    )
    prior_counts = (
        baseline_disagreements.groupby("probe", dropna=False)
        .agg(
            attempt001_threshold_adjacent_action_flips=("threshold_adjacent_action_flip", "sum"),
            attempt001_non_threshold_action_flips=("non_threshold_action_flip", "sum"),
            attempt001_true_score_reordering_count=("true_score_reordering", "sum"),
            attempt001_near_tie_tie_set_membership_count=("near_tie", "sum"),
        )
        .reset_index()
    ) if not baseline_disagreements.empty else pd.DataFrame(columns=["probe"])
    merged = attempt002.merge(
        prior[
            [
                col
                for col in [
                    "probe",
                    "attempt001_action_agreement",
                    "attempt001_selected_slot_agreement",
                    "attempt001_trade_set_jaccard",
                    "attempt001_disagreements",
                ]
                if col in prior.columns
            ]
        ],
        on="probe",
        how="left",
    ).merge(prior_counts, on="probe", how="left")
    merged["disagreement_reduction_fraction"] = (
        (merged["attempt001_disagreements"] - merged["disagreements"]) / merged["attempt001_disagreements"].replace(0, np.nan)
    )
    merged.to_csv(out_dir / "before_after_reconciliation.csv", index=False)
    return merged


def route_attempt(
    *,
    l1_rows: pd.DataFrame,
    l3_rows: pd.DataFrame,
    chosen_pairs: pd.DataFrame,
    baseline: dict[str, int],
    fallback_rule: str,
) -> dict[str, Any]:
    all_rows = pd.concat([l1_rows[l1_rows["probe"] != "random_with_guards_null"], l3_rows], ignore_index=True)
    action_gate_failures = all_rows[all_rows["action_agreement"] < ACTION_AGREEMENT_MIN]["probe"].tolist()
    action_reject_failures = all_rows[all_rows["action_agreement"] < ACTION_AGREEMENT_REJECT_BELOW]["probe"].tolist()
    slot_gate_failures = all_rows[all_rows["selected_slot_gate_status"] == "fail"]["probe"].tolist()
    total_disagreements = int(chosen_pairs["disagreement"].sum())
    baseline_total = max(1, baseline["total_disagreements"])
    total_reduction = (baseline_total - total_disagreements) / baseline_total
    material_true = int((chosen_pairs["true_score_reordering"] & chosen_pairs["economically_material"]).sum())
    baseline_material_true = max(1, baseline["material_true_score_reordering_count"])
    material_true_reduction = (baseline_material_true - material_true) / baseline_material_true
    material_disagreements = int(chosen_pairs["economically_material"].sum())
    material_reduction = baseline["material_disagreement_count"] - material_disagreements
    split_material = int(chosen_pairs["split_confidence_material"].sum())
    split_material_status = "material" if split_material >= 30 or split_material > material_reduction else "descriptive_only"
    residual_tail = int(chosen_pairs["residual_tail_divergence"].sum())
    gates = {
        "action_agreement_gate_pass": not action_gate_failures,
        "selected_slot_mutually_confident_gate_pass": not slot_gate_failures,
        "total_disagreement_reduction": total_reduction,
        "total_disagreement_reduction_gate_pass": total_reduction >= TOTAL_DISAGREEMENT_REDUCTION_MIN,
        "material_true_score_reordering_reduction": material_true_reduction,
        "material_true_score_reordering_reduction_gate_pass": material_true_reduction >= MATERIAL_TRUE_REORDER_REDUCTION_MIN,
        "split_confidence_material_status": split_material_status,
        "split_confidence_material_count": split_material,
        "residual_tail_divergence_count": residual_tail,
    }
    if action_reject_failures and material_true_reduction < MATERIAL_TRUE_REORDER_REDUCTION_MIN:
        decision = "canonical_v1_1_rejected_features_unstable_after_margin_gate"
    elif all(
        [
            gates["action_agreement_gate_pass"],
            gates["selected_slot_mutually_confident_gate_pass"],
            gates["total_disagreement_reduction_gate_pass"],
            gates["material_true_score_reordering_reduction_gate_pass"],
            split_material_status != "material",
        ]
    ):
        decision = "canonical_v1_1_selection_contract_pass"
    else:
        decision = "canonical_v1_1_selection_contract_repair_needed"
    return {
        "schema_version": "Protocol101CanonicalV11SelectionContractRoutingV1",
        "attempt_id": DEFAULT_OUT_DIR.name,
        "routing_decision": decision,
        "highest_allowed_claim": "canonical v1.1 selection-contract probe complete",
        "chosen_fallback_rule": fallback_rule,
        "baseline_counts": baseline,
        "attempt002_counts": {
            "total_disagreements": total_disagreements,
            "material_disagreement_count": material_disagreements,
            "material_true_score_reordering_count": material_true,
            "residual_tail_divergence_count": residual_tail,
            "split_confidence_material_count": split_material,
        },
        "gates": gates,
        "gate_failures": {
            "action_agreement": action_gate_failures,
            "selected_slot_mutually_confident": slot_gate_failures,
        },
        "low_n_policy": "mutually confident selected-slot denominators below 30 are descriptive_only and do not pass/fail the whole run alone",
        "side_effect_policy": {
            "fifteen_month_training_executed": False,
            "uplift_cv_executed": False,
            "threshold_optimization_executed": False,
            "diagnostic_transfer_probe_training_executed": True,
            "broker_endpoint_called": False,
            "paid_data_download": False,
            "paper_submit_allowed": False,
            "promotion_or_default_changed": False,
            "runtime_flags_edited": False,
            "launchd_changed": False,
            "real_money_path_changed": False,
        },
        "do_not_claim": [
            "paper readiness",
            "training readiness",
            "profitability",
            "runtime contract updated",
        ],
    }


def write_report(out_dir: Path, routing: dict[str, Any]) -> None:
    lines = [
        "# Protocol101 Canonical v1.1 Selection-Contract Probe Attempt002",
        "",
        f"- Routing decision: `{routing['routing_decision']}`",
        f"- Chosen fallback rule: `{routing['chosen_fallback_rule']}`",
        "- Highest allowed claim: `canonical v1.1 selection-contract probe complete`",
        "- Scope: offline burned-day design/probe audit only",
        "",
        "## Gates",
        "",
        f"- Action agreement gate pass: `{routing['gates']['action_agreement_gate_pass']}`",
        f"- Mutually confident selected-slot gate pass: `{routing['gates']['selected_slot_mutually_confident_gate_pass']}`",
        f"- Total disagreement reduction: `{routing['gates']['total_disagreement_reduction']}`",
        f"- Material true-score-reordering reduction: `{routing['gates']['material_true_score_reordering_reduction']}`",
        f"- Split-confidence material status: `{routing['gates']['split_confidence_material_status']}`",
        f"- Residual tail divergences: `{routing['gates']['residual_tail_divergence_count']}`",
        "",
        "## Artifacts",
        "",
        "- `selection_contract_v1_1.json`",
        "- `preregistration.json`",
        "- `l1_attempt002_results.csv`",
        "- `l3_attempt002_results.csv`",
        "- `before_after_reconciliation.csv`",
        "- `no_confident_slot_preference_report.csv`",
        "- `fallback_slot_agreement.csv`",
        "- `confidence_status_agreement.csv`",
        "- `margin_epsilon_ratio_report.csv`",
        "- `diagnostic_k_sensitivity.csv`",
        "- `economic_materiality_attempt002.csv`",
        "- `routing_decision.json`",
        "- `progress.json`",
        "",
        "## Side Effects",
        "",
        "- 15-month training: `false`",
        "- uplift CV: `false`",
        "- threshold optimization: `false`",
        "- broker/API calls: `false`",
        "- paid downloads: `false`",
        "- paper-submit: `false`",
        "- promotion/default/runtime/launchd edits: `false`",
        "",
        "This packet does not declare paper readiness.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} exists; pass --force to overwrite")
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "schema_version": "Protocol101CanonicalV11SelectionContractProgressV1",
        "attempt_id": out_dir.name,
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "fifteen_month_training_executed": False,
        "uplift_cv_executed": False,
        "threshold_optimization_executed": False,
        "broker_endpoint_called": False,
        "paid_data_download": False,
        "paper_submit_allowed": False,
        "promotion_or_default_changed": False,
        "runtime_flags_edited": False,
        "launchd_changed": False,
        "real_money_path_changed": False,
    }
    write_json(out_dir / "progress.json", progress)
    input_readiness = {"blockers": []}
    required = [
        args.l1_l3_dir / "l1_summary.json",
        args.l1_l3_dir / "l3_transfer_probe_results.csv",
        args.reconciliation_dir / "attribution_summary.json",
        args.reconciliation_dir / "full_population_disagreement_attribution.csv",
    ]
    for path in required:
        if not path.exists():
            input_readiness["blockers"].append(f"missing_{path.name}")
    if input_readiness["blockers"]:
        routing = {
            "schema_version": "Protocol101CanonicalV11SelectionContractRoutingV1",
            "attempt_id": out_dir.name,
            "routing_decision": "canonical_v1_1_probe_artifact_requires_rerun",
            "blockers": input_readiness["blockers"],
        }
        write_json(out_dir / "input_readiness.json", input_readiness)
        write_json(out_dir / "routing_decision.json", routing)
        progress["status"] = "complete_requires_rerun"
        progress["routing_decision"] = routing["routing_decision"]
        progress["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_json(out_dir / "progress.json", progress)
        return
    l1_summary = load_json(args.l1_l3_dir / "l1_summary.json")
    previous_l3 = pd.read_csv(args.l1_l3_dir / "l3_transfer_probe_results.csv")
    recon_summary = load_json(args.reconciliation_dir / "attribution_summary.json")
    baseline_disagreements = load_baseline_disagreements(args.reconciliation_dir)
    epsilon = epsilon_policy(l1_summary=l1_summary, previous_l3=previous_l3, recon_summary=recon_summary)
    _, contract_hash = write_selection_contract(
        out_dir=out_dir,
        l1_summary=l1_summary,
        previous_l3=previous_l3,
        epsilon=epsilon,
    )
    write_preregistration(out_dir=out_dir, args=args, contract_hash=contract_hash)
    paired, admitted, readiness = build_paired_frame(args)
    input_readiness.update(readiness)
    write_json(out_dir / "input_readiness.json", input_readiness)
    if readiness["blockers"]:
        routing = {
            "schema_version": "Protocol101CanonicalV11SelectionContractRoutingV1",
            "attempt_id": out_dir.name,
            "routing_decision": "canonical_v1_1_probe_artifact_requires_rerun",
            "blockers": readiness["blockers"],
        }
        write_json(out_dir / "routing_decision.json", routing)
        progress["status"] = "complete_requires_rerun"
        progress["routing_decision"] = routing["routing_decision"]
        progress["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_json(out_dir / "progress.json", progress)
        return
    progress["status"] = "building_probe_populations"
    write_json(out_dir / "progress.json", progress)
    population = build_populations(
        paired=paired,
        admitted=admitted,
        l1_summary=l1_summary,
        previous_l3=previous_l3,
        epsilon=epsilon,
    )
    progress["status"] = "selecting_fallback_rules"
    write_json(out_dir / "progress.json", progress)
    nearest_pairs = enrich_disagreement_classes(merge_plane_decisions(select_decisions(population, k=PRIMARY_K, fallback_rule="nearest_atm")))
    formal_pairs = enrich_disagreement_classes(merge_plane_decisions(select_decisions(population, k=PRIMARY_K, fallback_rule="formal_ordering")))
    _, chosen_rule = summarize_fallback_agreement(nearest_pairs, formal_pairs, out_dir)
    chosen_pairs = nearest_pairs if chosen_rule == "nearest_atm" else formal_pairs
    l1_rows = summarize_all(chosen_pairs[chosen_pairs["kind"] == "L1"])
    l3_rows = summarize_all(chosen_pairs[chosen_pairs["kind"] == "L3"])
    l1_rows.to_csv(out_dir / "l1_attempt002_results.csv", index=False)
    l3_rows.to_csv(out_dir / "l3_attempt002_results.csv", index=False)
    _, _, split_margin = write_confidence_reports(chosen_pairs, out_dir)
    write_ratio_bucket_report(chosen_pairs, split_margin, out_dir)
    write_diagnostic_k_sensitivity(chosen_pairs, chosen_rule, out_dir)
    write_economic_materiality(chosen_pairs, out_dir)
    attempt001_l1 = load_attempt001_l1_rows(l1_summary)
    before_after(
        attempt001_l1=attempt001_l1,
        attempt001_l3=previous_l3,
        attempt002=pd.concat([l1_rows, l3_rows], ignore_index=True),
        baseline_disagreements=baseline_disagreements,
        out_dir=out_dir,
    )
    routing = route_attempt(
        l1_rows=l1_rows,
        l3_rows=l3_rows,
        chosen_pairs=chosen_pairs,
        baseline=baseline_counts(recon_summary, baseline_disagreements),
        fallback_rule=chosen_rule,
    )
    write_json(out_dir / "routing_decision.json", routing)
    write_report(out_dir, routing)
    progress["status"] = "complete"
    progress["routing_decision"] = routing["routing_decision"]
    progress["chosen_fallback_rule"] = chosen_rule
    progress["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(out_dir / "progress.json", progress)


if __name__ == "__main__":
    main()
