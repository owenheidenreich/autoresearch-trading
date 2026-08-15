"""Protocol101 canonical v1 L1/L3 rejection reconciliation.

Offline-only attribution pass. It reruns the existing burned-day probe scoring
to materialize full populations, formalizes deterministic selection, and
separates true transfer instability from score-collision, tie-set, threshold,
low-margin, and low-n concentration effects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from v4.scripts.run_protocol101_canonical_v1_l0_l2_design_audit import (
    FEATURE_FAMILY,
    build_slot_frame,
)


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_IN_DIR = L1_L3_OUT_DIR
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_canonical_v1_probe_rejection_reconciliation_attempt001"
TRACE_PREFIX = "protocol101_live_v2_candidate_universe_parity_source_aligned"
SCHEMA_VERSION = "Protocol101CanonicalV1ProbeRejectionReconciliationAttempt001"
SCORE_EPS = 1e-12
THRESHOLD_ADJACENT_ABS = 0.05
FEE_MATERIALITY_DOLLARS = 3.0
MIN_CONCENTRATION_N = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l0-l2-dir", type=Path, default=DEFAULT_L0_L2_DIR)
    parser.add_argument("--l1-l3-dir", type=Path, default=DEFAULT_IN_DIR)
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


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def right_idx(value: Any) -> int:
    return 0 if str(value) == "C" else 1 if str(value) == "P" else 99


def nullable_int(value: Any, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return int(numeric)


def key_part(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


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


def score_population_rows(
    *,
    paired: pd.DataFrame,
    probe: str,
    kind: str,
    feature_family: str,
    threshold: float,
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
        "C.mid.mid_tick_q_historical",
        "C.mid.mid_tick_q_ibkr",
        "plane_pnl_15m_dollars_historical",
        "plane_pnl_15m_dollars_ibkr",
    ]
    hist = paired[base_cols].copy()
    hist["plane"] = "historical"
    hist["score"] = pd.Series(historical_scores, index=paired.index).to_numpy(dtype=float)
    hist["selected_pnl"] = hist["plane_pnl_15m_dollars_historical"]
    hist["mid_tick_q"] = hist["C.mid.mid_tick_q_historical"]
    live = paired[base_cols].copy()
    live["plane"] = "ibkr"
    live["score"] = pd.Series(ibkr_scores, index=paired.index).to_numpy(dtype=float)
    live["selected_pnl"] = live["plane_pnl_15m_dollars_ibkr"]
    live["mid_tick_q"] = live["C.mid.mid_tick_q_ibkr"]
    out = pd.concat([hist, live], ignore_index=True)
    out["kind"] = kind
    out["probe"] = probe
    out["subset"] = subset
    out["model"] = model
    out["feature_family"] = feature_family
    out["threshold"] = threshold
    out["slot_id"] = out["strike_key"].astype(str) + "|" + out["right"].astype(str)
    return out


def select_decisions(population: pd.DataFrame, *, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sort_cols = ["score"]
    ascending = [False]
    if method == "deterministic":
        sort_cols = ["score", "strike_idx_rank", "right_idx"]
        ascending = [False, True, True]
    for keys, group in population.groupby(["kind", "probe", "subset", "model", "feature_family", "plane", "session_date", "decision_minute_et"], sort=False, dropna=False):
        kind, probe, subset, model, feature_family, plane, session, minute = keys
        finite_group = group[group["score"].notna()].copy()
        if finite_group.empty:
            rows.append(
                {
                    "kind": kind,
                    "probe": probe,
                    "subset": subset,
                    "model": model,
                    "feature_family": feature_family,
                    "plane": plane,
                    "session_date": session,
                    "decision_minute_et": minute,
                    "method": method,
                    "action": "wait",
                    "selection_reason": "no_finite_scores",
                }
            )
            continue
        if method == "current_idxmax":
            selected = finite_group.loc[finite_group["score"].idxmax()]
        else:
            selected = finite_group.sort_values(sort_cols, ascending=ascending, kind="mergesort").iloc[0]
        ordered_scores = finite_group["score"].sort_values(ascending=False).to_numpy(dtype=float)
        top_score = float(ordered_scores[0])
        second_best = float(ordered_scores[1]) if len(ordered_scores) > 1 else None
        top2_margin = float(top_score - second_best) if second_best is not None else None
        top_tie = finite_group[np.isclose(finite_group["score"], top_score, atol=SCORE_EPS, rtol=0.0)]
        threshold = float(selected["threshold"])
        action = "enter" if math.isfinite(threshold) and float(selected["score"]) >= threshold else "wait"
        rows.append(
            {
                "kind": kind,
                "probe": probe,
                "subset": subset,
                "model": model,
                "feature_family": feature_family,
                "plane": plane,
                "session_date": session,
                "decision_minute_et": minute,
                "method": method,
                "action": action,
                "selected_slot": selected["slot_id"] if action == "enter" else None,
                "selected_strike": finite(selected["strike_float"]) if action == "enter" else None,
                "selected_side": selected["right"] if action == "enter" else None,
                "selected_score": float(selected["score"]),
                "selected_pnl": finite(selected["selected_pnl"]) if action == "enter" else None,
                "threshold": threshold,
                "threshold_distance": float(selected["score"]) - threshold if math.isfinite(threshold) else None,
                "selected_strike_idx": int(selected["strike_idx_rank"]) if not pd.isna(selected["strike_idx_rank"]) else None,
                "selected_right_idx": int(selected["right_idx"]),
                "top_score": top_score,
                "second_best_score": second_best,
                "top2_score_margin": top2_margin,
                "top_score_tie_set_size": int(len(top_tie)),
                "top_score_tie_set_slots": ";".join(sorted(str(item) for item in top_tie["slot_id"].tolist())),
                "selected_mid_tick_q": finite(selected["mid_tick_q"]) if action == "enter" else None,
                "time_bucket": selected["time_bucket"],
                "realized_5m_spx_vol_tercile": selected["realized_5m_spx_vol_tercile"],
                "opportunity_tercile": selected["opportunity_tercile"],
                "selection_rule": "highest_score_then_lowest_strike_idx_then_right_idx_C0_P1"
                if method == "deterministic"
                else "pandas_idxmax_first_row_in_current_population_order",
            }
        )
    return pd.DataFrame.from_records(rows)


def merge_plane_decisions(decisions: pd.DataFrame, *, method: str) -> pd.DataFrame:
    use = decisions[decisions["method"] == method]
    hist = use[use["plane"] == "historical"]
    live = use[use["plane"] == "ibkr"]
    key_cols = ["kind", "probe", "subset", "model", "feature_family", "session_date", "decision_minute_et"]
    merged = hist.merge(live, on=key_cols, suffixes=("_historical", "_ibkr"), how="inner", validate="one_to_one")
    merged["method"] = method
    merged["action_agree"] = merged["action_historical"] == merged["action_ibkr"]
    both_enter = (merged["action_historical"] == "enter") & (merged["action_ibkr"] == "enter")
    merged["both_enter"] = both_enter
    merged["selected_slot_agree"] = both_enter & (merged["selected_slot_historical"] == merged["selected_slot_ibkr"])
    merged["side_agree"] = both_enter & (merged["selected_side_historical"] == merged["selected_side_ibkr"])
    merged["disagreement"] = ~(
        merged["action_agree"] & ((~both_enter) | merged["selected_slot_agree"].fillna(False))
    )
    return merged


def agreement_summary(merged: pd.DataFrame) -> dict[str, Any]:
    both_enter = merged["both_enter"]
    return {
        "minutes": int(len(merged)),
        "action_agreement": float(merged["action_agree"].mean()) if len(merged) else None,
        "selected_slot_agreement": float(merged.loc[both_enter, "selected_slot_agree"].mean()) if bool(both_enter.any()) else 1.0,
        "side_agreement": float(merged.loc[both_enter, "side_agree"].mean()) if bool(both_enter.any()) else 1.0,
        "disagreements": int(merged["disagreement"].sum()),
    }


def deterministic_reconciliation(current: pd.DataFrame, deterministic: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key = ["kind", "probe", "subset", "model", "feature_family"]
    all_keys = deterministic[key].drop_duplicates()
    det_pairs = merge_plane_decisions(deterministic, method="deterministic")
    cur_pairs = merge_plane_decisions(current, method="current_idxmax")
    for item in all_keys.itertuples(index=False):
        mask_det = np.logical_and.reduce([(det_pairs[col].fillna("") == ("" if pd.isna(value) else value)) for col, value in zip(key, item, strict=False)])
        mask_cur = np.logical_and.reduce([(cur_pairs[col].fillna("") == ("" if pd.isna(value) else value)) for col, value in zip(key, item, strict=False)])
        det_subset = det_pairs[mask_det]
        cur_subset = cur_pairs[mask_cur]
        row = {col: None if pd.isna(value) else value for col, value in zip(key, item, strict=False)}
        row.update({f"before_{k}": v for k, v in agreement_summary(cur_subset).items()})
        row.update({f"after_{k}": v for k, v in agreement_summary(det_subset).items()})
        # Current-vs-deterministic reproducibility by plane.
        for plane in ("historical", "ibkr"):
            cur_plane = current[current["plane"] == plane]
            det_plane = deterministic[deterministic["plane"] == plane]
            plane_keys = key + ["session_date", "decision_minute_et"]
            joined = cur_plane.merge(det_plane, on=plane_keys, suffixes=("_current", "_deterministic"), how="inner")
            mask_join = np.logical_and.reduce([(joined[f"{col}"].fillna("") == ("" if pd.isna(value) else value)) for col, value in zip(key, item, strict=False)])
            joined = joined[mask_join]
            row[f"{plane}_current_vs_deterministic_action_agreement"] = float(
                (joined["action_current"] == joined["action_deterministic"]).mean()
            ) if len(joined) else None
            both_enter = (joined["action_current"] == "enter") & (joined["action_deterministic"] == "enter")
            row[f"{plane}_current_vs_deterministic_slot_agreement"] = float(
                (joined.loc[both_enter, "selected_slot_current"] == joined.loc[both_enter, "selected_slot_deterministic"]).mean()
            ) if bool(both_enter.any()) else 1.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "deterministic_selection_reconciliation.csv", index=False)
    return out


def p95_score_drift_by_probe(population: pd.DataFrame, previous_l3: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    key_cols = ["kind", "probe", "subset", "model", "feature_family", "session_date", "decision_minute_et", "slot_id"]
    hist = population[population["plane"] == "historical"]
    live = population[population["plane"] == "ibkr"]
    merged = hist.merge(live, on=key_cols, suffixes=("_historical", "_ibkr"), how="inner")
    for keys, group in merged.groupby(["kind", "probe", "subset", "model"], dropna=False):
        kind, probe, subset, model = keys
        label = probe
        if kind == "L3":
            prior = previous_l3[previous_l3["probe"] == probe]
            if not prior.empty and pd.notna(prior.iloc[0].get("score_drift_p95_abs")):
                out[str(label)] = float(prior.iloc[0]["score_drift_p95_abs"])
                continue
        drift = (group["score_ibkr"].astype(float) - group["score_historical"].astype(float)).abs()
        out[str(label)] = float(drift.quantile(0.95)) if len(drift.dropna()) else 0.0
    return out


def build_population_lookup(population: pd.DataFrame) -> dict[tuple[str, ...], dict[str, Any]]:
    key_cols = ["kind", "probe", "subset", "model", "plane", "session_date", "decision_minute_et", "slot_id"]
    value_cols = ["C.mid.mid_tick_q_historical", "C.mid.mid_tick_q_ibkr"]
    lookup: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in population[key_cols + value_cols].to_dict("records"):
        key = tuple(key_part(row[col]) for col in key_cols)
        lookup.setdefault(key, row)
    return lookup


def selected_slot_row(lookup: dict[tuple[str, ...], dict[str, Any]], decision: pd.Series, plane: str) -> dict[str, Any] | None:
    slot = decision.get(f"selected_slot_{plane}")
    if not isinstance(slot, str) or not slot:
        return None
    key = (
        key_part(decision.get("kind")),
        key_part(decision.get("probe")),
        key_part(decision.get("subset")),
        key_part(decision.get("model")),
        plane,
        key_part(decision.get("session_date")),
        key_part(decision.get("decision_minute_et")),
        key_part(slot),
    )
    return lookup.get(key)


def classify_disagreements(deterministic_pairs: pd.DataFrame, population: pd.DataFrame, drift_p95: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    selected = deterministic_pairs.copy()
    population_lookup = build_population_lookup(population)
    for _, decision in selected.iterrows():
        probe = str(decision["probe"])
        noise = float(drift_p95.get(probe, 0.0) or 0.0)
        selected_disagreement = bool(decision["disagreement"])
        for plane in ("historical", "ibkr"):
            tie_rows.append(
                {
                    "kind": decision["kind"],
                    "probe": decision["probe"],
                    "subset": decision.get("subset"),
                    "model": decision.get("model"),
                    "feature_family": decision["feature_family"],
                    "plane": plane,
                    "session_date": decision["session_date"],
                    "decision_minute_et": decision["decision_minute_et"],
                    "time_bucket": decision.get(f"time_bucket_{plane}"),
                    "realized_5m_spx_vol_tercile": decision.get(f"realized_5m_spx_vol_tercile_{plane}"),
                    "opportunity_tercile": decision.get(f"opportunity_tercile_{plane}"),
                    "action": decision[f"action_{plane}"],
                    "selected_slot": decision.get(f"selected_slot_{plane}"),
                    "selected_score": decision.get(f"selected_score_{plane}"),
                    "second_best_score": decision.get(f"second_best_score_{plane}"),
                    "top2_score_margin": decision.get(f"top2_score_margin_{plane}"),
                    "score_drift_p95_for_probe": noise,
                    "near_tie": pd.notna(decision.get(f"top2_score_margin_{plane}"))
                    and float(decision.get(f"top2_score_margin_{plane}")) <= noise + SCORE_EPS,
                    "exact_score_tie": nullable_int(decision.get(f"top_score_tie_set_size_{plane}")) > 1,
                    "top_score_tie_set_size": decision.get(f"top_score_tie_set_size_{plane}"),
                    "top_score_tie_set_slots": decision.get(f"top_score_tie_set_slots_{plane}"),
                    "selected_slot_disagreement": selected_disagreement,
                }
            )
        if not selected_disagreement:
            continue
        action_flip = bool(decision["action_historical"] != decision["action_ibkr"])
        both_enter = bool(decision["both_enter"])
        same_action_different_slot = bool(
            both_enter and decision.get("selected_slot_historical") != decision.get("selected_slot_ibkr")
        )
        threshold_adjacent = False
        if action_flip:
            for plane in ("historical", "ibkr"):
                dist = decision.get(f"threshold_distance_{plane}")
                if pd.notna(dist) and abs(float(dist)) < THRESHOLD_ADJACENT_ABS:
                    threshold_adjacent = True
        exact_tie = nullable_int(decision.get("top_score_tie_set_size_historical")) > 1 or nullable_int(
            decision.get("top_score_tie_set_size_ibkr")
        ) > 1
        near_tie = any(
            pd.notna(decision.get(f"top2_score_margin_{plane}"))
            and float(decision.get(f"top2_score_margin_{plane}")) <= noise + SCORE_EPS
            for plane in ("historical", "ibkr")
        )
        hist_sel = selected_slot_row(population_lookup, decision, "historical")
        live_sel = selected_slot_row(population_lookup, decision, "ibkr")
        mid_diff_small = False
        if hist_sel is not None:
            h_mid_delta = abs(float(hist_sel.get("C.mid.mid_tick_q_ibkr", np.nan)) - float(hist_sel.get("C.mid.mid_tick_q_historical", np.nan)))
            mid_diff_small = mid_diff_small or (math.isfinite(h_mid_delta) and h_mid_delta <= 0.05 + SCORE_EPS)
        if live_sel is not None:
            l_mid_delta = abs(float(live_sel.get("C.mid.mid_tick_q_ibkr", np.nan)) - float(live_sel.get("C.mid.mid_tick_q_historical", np.nan)))
            mid_diff_small = mid_diff_small or (math.isfinite(l_mid_delta) and l_mid_delta <= 0.05 + SCORE_EPS)
        hist_top = set(str(decision.get("top_score_tie_set_slots_historical") or "").split(";")) - {""}
        live_top = set(str(decision.get("top_score_tie_set_slots_ibkr") or "").split(";")) - {""}
        hist_slot = decision.get("selected_slot_historical")
        live_slot = decision.get("selected_slot_ibkr")
        tie_membership_change = bool(
            same_action_different_slot
            and exact_tie
            and mid_diff_small
            and ((hist_slot in hist_top and hist_slot not in live_top) or (live_slot in live_top and live_slot not in hist_top))
        )
        true_reordering = bool(
            same_action_different_slot
            and nullable_int(decision.get("top_score_tie_set_size_historical")) == 1
            and nullable_int(decision.get("top_score_tie_set_size_ibkr")) == 1
        )
        pnl_delta = None
        missing_pnl = False
        if same_action_different_slot or action_flip:
            hp = finite(decision.get("selected_pnl_historical"))
            lp = finite(decision.get("selected_pnl_ibkr"))
            if hp is None or lp is None:
                missing_pnl = True
            else:
                pnl_delta = lp - hp
        economic_material = pnl_delta is not None and abs(float(pnl_delta)) > FEE_MATERIALITY_DOLLARS
        adjacent_strike = False
        if same_action_different_slot and pd.notna(decision.get("selected_strike_historical")) and pd.notna(decision.get("selected_strike_ibkr")):
            adjacent_strike = abs(float(decision["selected_strike_historical"]) - float(decision["selected_strike_ibkr"])) <= 5.0 + SCORE_EPS
        primary = "same_action_different_slot"
        if action_flip:
            primary = "threshold_adjacent_action_flip" if threshold_adjacent else "non_threshold_action_flip"
        elif tie_membership_change:
            primary = "cross_plane_tie_set_membership_change"
        elif exact_tie:
            primary = "exact_score_tie"
        elif near_tie:
            primary = "near_tie"
        elif true_reordering:
            primary = "true_score_reordering"
        rows.append(
            {
                "kind": decision["kind"],
                "probe": decision["probe"],
                "subset": decision.get("subset"),
                "model": decision.get("model"),
                "feature_family": decision["feature_family"],
                "session_date": decision["session_date"],
                "decision_minute_et": decision["decision_minute_et"],
                "time_bucket": decision.get("time_bucket_historical"),
                "realized_5m_spx_vol_tercile": decision.get("realized_5m_spx_vol_tercile_historical"),
                "opportunity_tercile": decision.get("opportunity_tercile_historical"),
                "action_historical": decision["action_historical"],
                "action_ibkr": decision["action_ibkr"],
                "selected_slot_historical": hist_slot,
                "selected_slot_ibkr": live_slot,
                "selected_score_historical": decision.get("selected_score_historical"),
                "selected_score_ibkr": decision.get("selected_score_ibkr"),
                "threshold_distance_historical": decision.get("threshold_distance_historical"),
                "threshold_distance_ibkr": decision.get("threshold_distance_ibkr"),
                "top2_margin_historical": decision.get("top2_score_margin_historical"),
                "top2_margin_ibkr": decision.get("top2_score_margin_ibkr"),
                "score_drift_p95_for_probe": noise,
                "historical_top_tie_size": decision.get("top_score_tie_set_size_historical"),
                "ibkr_top_tie_size": decision.get("top_score_tie_set_size_ibkr"),
                "action_flip": action_flip,
                "same_action_different_slot": same_action_different_slot,
                "exact_score_tie": exact_tie,
                "near_tie": near_tie,
                "threshold_adjacent_action_flip": bool(action_flip and threshold_adjacent),
                "non_threshold_action_flip": bool(action_flip and not threshold_adjacent),
                "selected_slot_swap_economically_small": bool(same_action_different_slot and pnl_delta is not None and not economic_material),
                "selected_slot_swap_economically_material": bool(same_action_different_slot and economic_material),
                "cross_plane_tie_set_membership_change": tie_membership_change,
                "true_score_reordering": true_reordering,
                "mid_one_tick_or_less_for_involved_slot": mid_diff_small,
                "selected_pnl_historical": decision.get("selected_pnl_historical"),
                "selected_pnl_ibkr": decision.get("selected_pnl_ibkr"),
                "selected_pnl_delta_ibkr_minus_historical": pnl_delta,
                "selected_pnl_missing": missing_pnl,
                "adjacent_strike_swap": adjacent_strike,
                "same_side_swap": bool(same_action_different_slot and decision.get("selected_side_historical") == decision.get("selected_side_ibkr")),
                "opposite_side_swap": bool(same_action_different_slot and decision.get("selected_side_historical") != decision.get("selected_side_ibkr")),
                "primary_class": primary,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(tie_rows)


def margin_percentiles(tie_rows: pd.DataFrame) -> pd.DataFrame:
    out = tie_rows.copy()
    out["top2_margin_percentile_within_probe_plane"] = np.nan
    for _, idx in out.groupby(["probe", "plane"], dropna=False).groups.items():
        margins = out.loc[idx, "top2_score_margin"].astype(float)
        out.loc[idx, "top2_margin_percentile_within_probe_plane"] = margins.rank(pct=True)
    return out


def aggregate_margin_analysis(tie_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("by_probe", ["probe"]),
        ("by_family", ["feature_family"]),
        ("by_day", ["session_date"]),
        ("by_volatility_tercile", ["realized_5m_spx_vol_tercile"]),
        ("by_opportunity_tercile", ["opportunity_tercile"]),
        ("by_probe_family", ["probe", "feature_family"]),
    ]
    for group_type, cols in group_specs:
        grouped = [((), tie_rows)] if not cols else tie_rows.groupby(cols, dropna=False)
        for key, group in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            base = {"group_type": group_type, "n": int(len(group))}
            for col, value in zip(cols, key, strict=False):
                base[col] = value
            margins = pd.to_numeric(group["top2_score_margin"], errors="coerce").dropna()
            base.update(
                {
                    "tie_set_size_1": int((group["top_score_tie_set_size"] == 1).sum()),
                    "tie_set_size_gt1": int((group["top_score_tie_set_size"] > 1).sum()),
                    "near_tie_count": int(group["near_tie"].fillna(False).sum()),
                    "selected_slot_disagreement_count": int(group["selected_slot_disagreement"].fillna(False).sum()),
                    "top2_margin_median": float(margins.median()) if len(margins) else None,
                    "top2_margin_p05": float(margins.quantile(0.05)) if len(margins) else None,
                    "top2_margin_p95": float(margins.quantile(0.95)) if len(margins) else None,
                }
            )
            rows.append(base)
    return pd.DataFrame(rows)


def concentration_claims(disagreements: pd.DataFrame) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    scopes = [("overall", None), ("by_probe", "probe"), ("by_family", "feature_family")]
    for scope_name, group_col in scopes:
        grouped = [(None, disagreements)] if group_col is None else disagreements.groupby(group_col, dropna=False)
        for group_value, group in grouped:
            n = int(len(group))
            prefix = {"scope": scope_name, "scope_value": None if group_col is None else group_value}
            for col, top_value, claim_name in (
                ("realized_5m_spx_vol_tercile", "top", "top_volatility_tercile"),
                ("opportunity_tercile", "top", "top_opportunity_tercile"),
            ):
                count = int((group[col] == top_value).sum()) if n else 0
                row = {**prefix, "claim": claim_name, **wilson_ci(count, n)}
                claims.append(row)
            for bucket, count in group["time_bucket"].value_counts(dropna=False).to_dict().items():
                row = {**prefix, "claim": f"time_bucket::{bucket}", **wilson_ci(int(count), n)}
                claims.append(row)
    return {
        "schema_version": "Protocol101CanonicalV1ConcentrationClaimsWithCIV1",
        "minimum_n_for_rejection_evidence": MIN_CONCENTRATION_N,
        "rule": "claims with denominator < 30 are marked insufficient_n_descriptive_only and are not rejection evidence",
        "claims": claims,
    }


def economic_materiality(disagreements: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    swap = disagreements[disagreements["same_action_different_slot"]].copy()
    rows: list[dict[str, Any]] = []
    grouped = [("overall", swap)] + [(str(probe), group) for probe, group in swap.groupby("probe", dropna=False)]
    for name, group in grouped:
        pnl = pd.to_numeric(group["selected_pnl_delta_ibkr_minus_historical"], errors="coerce")
        rows.append(
            {
                "scope": name,
                "slot_swaps": int(len(group)),
                "median_pnl_delta": float(pnl.dropna().median()) if len(pnl.dropna()) else None,
                "p95_abs_pnl_delta": float(pnl.dropna().abs().quantile(0.95)) if len(pnl.dropna()) else None,
                "missing_pnl_count": int(group["selected_pnl_missing"].sum()) if len(group) else 0,
                "economically_small_count": int(group["selected_slot_swap_economically_small"].sum()) if len(group) else 0,
                "economically_material_count": int(group["selected_slot_swap_economically_material"].sum()) if len(group) else 0,
                "adjacent_strike_swap_count": int(group["adjacent_strike_swap"].sum()) if len(group) else 0,
                "same_side_swap_count": int(group["same_side_swap"].sum()) if len(group) else 0,
                "opposite_side_swap_count": int(group["opposite_side_swap"].sum()) if len(group) else 0,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "economic_materiality.csv", index=False)
    return out


def repair_candidates(summary: dict[str, Any]) -> dict[str, Any]:
    route = summary["routing_decision"]
    candidates = [
        {
            "candidate": "formal_deterministic_selection_rule_for_future_harnesses",
            "assessment": "recommended_as_documented_invariant",
            "implement_now": False,
            "rationale": "Current harness is deterministic via idxmax/current row order, but explicit score/strike_idx/right_idx ordering is clearer and auditable.",
        },
        {
            "candidate": "remove_or_coarsen_family_C_per_slot_mid_momentum",
            "assessment": "strong_candidate" if summary["family_c_instability"] else "not_primary",
            "implement_now": False,
            "rationale": "Family C option-mid momentum produced the only L1 action-agreement failure if family C instability remains after reconciliation.",
        },
        {
            "candidate": "replace_per_slot_mid_momentum_with_band_level_aggregate_momentum",
            "assessment": "strong_candidate" if summary["family_c_instability"] else "watch",
            "implement_now": False,
            "rationale": "Band-level aggregation may reduce one-tick slot-score collision and top-slot membership churn.",
        },
        {
            "candidate": "keep_family_D_composites",
            "assessment": "supported" if not summary["family_d_independently_unstable"] else "needs_repair",
            "implement_now": False,
            "rationale": "Family D logistic transfer passed; bounded-HGB D instability is primarily selected-slot collision/low-n concentration if no material true reordering remains.",
        },
        {
            "candidate": "keep_family_E_internal_greeks_if_not_independently_unstable",
            "assessment": "supported" if not summary["family_e_independently_unstable"] else "needs_repair",
            "implement_now": False,
            "rationale": "Family E logistic action transfer passed; HGB slot churn should be treated separately from IV/delta/gamma rejection.",
        },
        {
            "candidate": "use_logistic_linear_models_first",
            "assessment": "recommended",
            "implement_now": False,
            "rationale": "Logistic probes were materially more stable than bounded-HGB when C/D/E were present.",
        },
        {
            "candidate": "require_larger_score_or_threshold_margins_before_action",
            "assessment": "recommended" if summary["threshold_adjacent_action_flip_count"] else "optional",
            "implement_now": False,
            "rationale": "Threshold-adjacent action flips are probe-design fragility, not proof of irreducible feature mismatch.",
        },
        {
            "candidate": "require_larger_selected_slot_score_margins_before_slot_selection",
            "assessment": "recommended" if summary["near_tie_or_tie_collision_count"] else "optional",
            "implement_now": False,
            "rationale": "Low top-2 margins relative to measured cross-plane score drift mean selected slot is not a stable semantic decision.",
        },
        {
            "candidate": "inject_measured_divergence_noise_in_future_training",
            "assessment": "recommended_for_future_design",
            "implement_now": False,
            "rationale": "Future training should learn under the measured canonical feature drift rather than assume zero drift.",
        },
        {
            "candidate": "mark_high_volatility_minutes_as_stress_regime",
            "assessment": "diagnostic_only",
            "implement_now": False,
            "rationale": "High-volatility concentration claims require minimum-n discipline and should not be used to discard minutes blindly.",
        },
    ]
    return {
        "schema_version": "Protocol101CanonicalV1RepairCandidatesV1",
        "routing_decision_considered": route,
        "candidates": candidates,
        "rules": [
            "analytical candidates only; none implemented",
            "no feature contract or threshold changes",
            "do not declare quote features rejected unless material non-threshold instability remains after reconciliation",
        ],
    }


def decide_routing(disagreements: pd.DataFrame, deterministic_recon: pd.DataFrame) -> dict[str, Any]:
    total = int(len(disagreements))
    if total == 0:
        decision = "canonical_v1_probe_artifact_score_collision_confirmed"
    threshold_adjacent = int(disagreements["threshold_adjacent_action_flip"].sum()) if total else 0
    non_threshold = int(disagreements["non_threshold_action_flip"].sum()) if total else 0
    material_swaps = int(disagreements["selected_slot_swap_economically_material"].sum()) if total else 0
    true_reordering = int(disagreements["true_score_reordering"].sum()) if total else 0
    collision = int((disagreements["exact_score_tie"] | disagreements["near_tie"] | disagreements["cross_plane_tie_set_membership_change"]).sum()) if total else 0
    family_c = bool(len(disagreements[(disagreements["feature_family"].astype(str).str.contains("C"))]))
    family_d_ind = bool(len(disagreements[(disagreements["feature_family"].astype(str).str.fullmatch("D")) & disagreements["true_score_reordering"]]))
    family_e_ind = bool(len(disagreements[(disagreements["feature_family"].astype(str).str.fullmatch("E")) & disagreements["true_score_reordering"]]))
    material_non_threshold = int(
        (
            (disagreements["non_threshold_action_flip"])
            | (disagreements["true_score_reordering"] & disagreements["selected_slot_swap_economically_material"])
        ).sum()
    ) if total else 0
    # Stable A/B and A/B+D/E logistic probes mean the next safe direction is to narrow
    # rather than reject all canonical quote-derived features.
    ab_rows = deterministic_recon[deterministic_recon["probe"].astype(str).str.contains("AB_plus_group1")]
    stable_ab = bool((ab_rows.get("after_action_agreement", pd.Series(dtype=float)) >= 0.98).all()) if len(ab_rows) else False
    if material_non_threshold > 0 and true_reordering > collision:
        decision = "canonical_v1_rejected_quote_features_unstable"
    elif family_c or stable_ab:
        decision = "canonical_v1_narrow_to_stable_subsets"
    elif collision >= max(1, int(0.50 * total)):
        decision = "canonical_v1_probe_artifact_score_collision_confirmed"
    else:
        decision = "canonical_v1_repair_candidate_identified"
    return {
        "routing_decision": decision,
        "total_disagreements": total,
        "threshold_adjacent_action_flip_count": threshold_adjacent,
        "non_threshold_action_flip_count": non_threshold,
        "economically_material_slot_swap_count": material_swaps,
        "true_score_reordering_count": true_reordering,
        "near_tie_or_tie_collision_count": collision,
        "material_non_threshold_instability_count": material_non_threshold,
        "family_c_instability": family_c,
        "family_d_independently_unstable": family_d_ind,
        "family_e_independently_unstable": family_e_ind,
        "stable_ab_or_g1_subsets_observed": stable_ab,
    }


def write_report(out_dir: Path, routing: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# Protocol101 Canonical v1 Probe Rejection Reconciliation",
        "",
        f"- Routing decision: `{routing['routing_decision']}`",
        "- Highest allowed claim: `canonical v1 L1/L3 rejection reconciliation complete`",
        "- Scope: offline burned-day reconciliation only",
        "",
        "## Deterministic Selection",
        "",
        "- Current harness rule: pandas `idxmax` chooses the first row with the highest score in current population order.",
        "- Formal reconciliation rule: highest score, then lowest `strike_idx`, then `right_idx` with `C=0`, `P=1`.",
        "- This is a formalization step; the prior result is not described as nondeterministic.",
        "",
        "## Attribution",
        "",
        f"- Total full-population disagreements: `{summary['total_disagreements']}`",
        f"- Threshold-adjacent action flips: `{summary['threshold_adjacent_action_flip_count']}`",
        f"- Non-threshold action flips: `{summary['non_threshold_action_flip_count']}`",
        f"- True score reorderings: `{summary['true_score_reordering_count']}`",
        f"- Tie/near-tie/collision count: `{summary['near_tie_or_tie_collision_count']}`",
        f"- Economically material slot swaps: `{summary['economically_material_slot_swap_count']}`",
        f"- Material non-threshold instability count: `{summary['material_non_threshold_instability_count']}`",
        "",
        "## Interpretation",
        "",
        "- Low-n concentration claims are marked descriptive-only when denominator < 30 and are not used as rejection evidence.",
        "- Same-action equal-score or near-tie selected-slot swaps are treated as score-collision fragility, not true transfer instability.",
        "- Canonical training is not declared ready by this packet.",
        "",
        "## Artifacts",
        "",
        "- `attribution_summary.json`",
        "- `deterministic_selection_reconciliation.csv`",
        "- `full_population_disagreement_attribution.csv`",
        "- `top_score_tie_set_analysis.csv`",
        "- `top2_margin_analysis.csv`",
        "- `concentration_claims_with_ci.json`",
        "- `economic_materiality.csv`",
        "- `repair_candidates.json`",
        "- `routing_decision.json`",
        "- `progress.json`",
        "",
        "## Side Effects",
        "",
        "- strategy_training_executed: `false`",
        "- uplift_cv_executed: `false`",
        "- threshold_optimization_executed: `false`",
        "- broker_endpoint_called: `false`",
        "- paid_data_download: `false`",
        "- paper_submit_allowed: `false`",
        "- promotion/default/runtime/launchd edits: `false`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} exists; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "schema_version": "Protocol101CanonicalV1ProbeReconciliationProgressV1",
        "attempt_id": out_dir.name,
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "strategy_training_executed": False,
        "uplift_cv_executed": False,
        "threshold_optimization_executed": False,
        "broker_endpoint_called": False,
        "paid_data_download": False,
        "paper_submit_allowed": False,
        "promotion_or_default_changed": False,
        "runtime_flags_edited": False,
        "launchd_changed": False,
    }
    write_json(out_dir / "progress.json", progress)
    input_readiness: dict[str, Any] = {"blockers": []}
    for required in ("l1_summary.json", "l3_transfer_probe_results.csv", "routing_decision.json"):
        if not (args.l1_l3_dir / required).exists():
            input_readiness["blockers"].append(f"missing_l1_l3_{required}")
    if not (args.l0_l2_dir / "per_feature_verdicts.csv").exists():
        input_readiness["blockers"].append("missing_l0_l2_per_feature_verdicts")
    if input_readiness["blockers"]:
        routing = {
            "schema_version": "Protocol101CanonicalV1ProbeReconciliationRoutingV1",
            "attempt_id": out_dir.name,
            "routing_decision": "canonical_v1_probe_artifact_requires_rerun",
            "blockers": input_readiness["blockers"],
        }
        write_json(out_dir / "attribution_summary.json", {"input_readiness": input_readiness})
        write_json(out_dir / "routing_decision.json", routing)
        write_report(out_dir, routing, {"total_disagreements": 0, "threshold_adjacent_action_flip_count": 0, "non_threshold_action_flip_count": 0, "true_score_reordering_count": 0, "near_tie_or_tie_collision_count": 0, "economically_material_slot_swap_count": 0, "material_non_threshold_instability_count": 0})
        progress["status"] = "complete_requires_rerun"
        progress["routing_decision"] = routing["routing_decision"]
        progress["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_json(out_dir / "progress.json", progress)
        return
    l1_summary = load_json(args.l1_l3_dir / "l1_summary.json")
    previous_l3 = pd.read_csv(args.l1_l3_dir / "l3_transfer_probe_results.csv")
    admitted, _, _ = load_authorized_features(args.l0_l2_dir, args.group1_dir)
    slot_frame, readiness = build_slot_frame(args.trace_prefix)
    input_readiness["trace_readiness"] = readiness
    if readiness.get("blockers") or slot_frame.empty:
        input_readiness["blockers"].extend(readiness.get("blockers") or ["empty_slot_frame"])
    write_json(out_dir / "input_readiness.json", input_readiness)
    if input_readiness["blockers"]:
        routing = {
            "schema_version": "Protocol101CanonicalV1ProbeReconciliationRoutingV1",
            "attempt_id": out_dir.name,
            "routing_decision": "canonical_v1_probe_artifact_requires_rerun",
            "blockers": input_readiness["blockers"],
        }
        write_json(out_dir / "routing_decision.json", routing)
        progress["status"] = "complete_requires_rerun"
        progress["routing_decision"] = routing["routing_decision"]
        progress["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_json(out_dir / "progress.json", progress)
        return
    slot_frame = add_group1_features(slot_frame, args.trace_prefix, args.group1_definition)
    slot_frame = add_plane_returns(slot_frame)
    feature_cols = admitted + GROUP1_NON_VIX_FEATURES
    paired = pair_slot_frame(slot_frame, feature_cols)
    paired = add_static_indices(assign_terciles(paired))
    paired["time_bucket"] = paired["decision_minute_et"].map(minute_bucket)
    populations: list[pd.DataFrame] = []
    thresholds = l1_summary["thresholds"]
    probe_families = {
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
    for probe, family in probe_families.items():
        threshold = float(thresholds[probe]["threshold"])
        populations.append(
            score_population_rows(
                paired=paired,
                probe=probe,
                kind="L1",
                feature_family=family,
                threshold=threshold,
                historical_scores=l1_score(paired, probe, "historical", thresholds),
                ibkr_scores=l1_score(paired, probe, "ibkr", thresholds),
            )
        )
    train_mask = paired["y_15m_conservative_return_ibkr"].notna()
    y = (paired.loc[train_mask, "y_15m_conservative_return_ibkr"].astype(float) > 0).astype(int).to_numpy()
    progress["status"] = "running_l3_diagnostic_rescore"
    write_json(out_dir / "progress.json", progress)
    for subset_name in L3_SUBSETS:
        cols = l3_feature_columns(subset_name, admitted)
        for model_name in ("logistic", "bounded_hgb"):
            probe = f"{subset_name}_{model_name}"
            threshold_row = previous_l3[previous_l3["probe"] == probe]
            if threshold_row.empty:
                continue
            threshold = float(threshold_row.iloc[0]["threshold"])
            model = train_l3_model(model_name)
            model.fit(matrix(paired.loc[train_mask], cols, "_historical"), y)
            hist_scores = model.predict_proba(matrix(paired, cols, "_historical"))[:, 1]
            live_scores = model.predict_proba(matrix(paired, cols, "_ibkr"))[:, 1]
            populations.append(
                score_population_rows(
                    paired=paired,
                    probe=probe,
                    kind="L3",
                    feature_family="+".join(L3_SUBSETS[subset_name]) + "+G1",
                    threshold=threshold,
                    historical_scores=hist_scores,
                    ibkr_scores=live_scores,
                    subset=subset_name,
                    model=model_name,
                )
            )
    population = pd.concat(populations, ignore_index=True)
    population.to_parquet(out_dir / "full_population_probe_scores.parquet", index=False)
    current = select_decisions(population, method="current_idxmax")
    deterministic = select_decisions(population, method="deterministic")
    det_recon = deterministic_reconciliation(current, deterministic, out_dir)
    det_pairs = merge_plane_decisions(deterministic, method="deterministic")
    drift_p95 = p95_score_drift_by_probe(population, previous_l3)
    disagreements, tie_rows = classify_disagreements(det_pairs, population, drift_p95)
    disagreements.to_csv(out_dir / "full_population_disagreement_attribution.csv", index=False)
    tie_rows = margin_percentiles(tie_rows)
    tie_rows.to_csv(out_dir / "top_score_tie_set_analysis.csv", index=False)
    margin_analysis = aggregate_margin_analysis(tie_rows)
    margin_analysis.to_csv(out_dir / "top2_margin_analysis.csv", index=False)
    claims = concentration_claims(disagreements)
    write_json(out_dir / "concentration_claims_with_ci.json", claims)
    econ = economic_materiality(disagreements, out_dir)
    summary = decide_routing(disagreements, det_recon)
    summary.update(
        {
            "schema_version": "Protocol101CanonicalV1ProbeReconciliationAttributionSummaryV1",
            "attempt_id": out_dir.name,
            "highest_allowed_claim": "canonical v1 L1/L3 rejection reconciliation complete",
            "input_l1_l3_routing_decision": load_json(args.l1_l3_dir / "routing_decision.json").get("routing_decision"),
            "paired_minutes": int(paired[["session_date", "decision_minute_et"]].drop_duplicates().shape[0]),
            "paired_slot_rows": int(len(paired)),
            "full_population_probe_score_rows": int(len(population)),
            "deterministic_selection_summary": {
                "current_rule": "pandas_idxmax_first_row_in_current_population_order",
                "formalized_rule": "highest_score_then_lowest_strike_idx_then_right_idx_C0_P1",
                "prior_result_described_as_nondeterministic": False,
            },
            "low_n_concentration_rule": "denominator < 30 is descriptive only and not rejection evidence",
            "score_drift_p95_by_probe": drift_p95,
        }
    )
    write_json(out_dir / "attribution_summary.json", summary)
    repairs = repair_candidates(summary)
    write_json(out_dir / "repair_candidates.json", repairs)
    routing = {
        "schema_version": "Protocol101CanonicalV1ProbeReconciliationRoutingV1",
        "attempt_id": out_dir.name,
        "routing_decision": summary["routing_decision"],
        "highest_allowed_claim": "canonical v1 L1/L3 rejection reconciliation complete",
        "source_rejection_packet": str(args.l1_l3_dir),
        "summary": summary,
        "do_not_claim": [
            "training-ready",
            "paper readiness",
            "profitability",
            "canonical quote features rejected without material post-reconciliation instability",
        ],
        "side_effect_policy": {
            "strategy_training_executed": False,
            "uplift_cv_executed": False,
            "threshold_optimization_executed": False,
            "broker_endpoint_called": False,
            "paid_data_download": False,
            "paper_submit_allowed": False,
            "promotion_or_default_changed": False,
            "runtime_flags_edited": False,
            "launchd_changed": False,
        },
    }
    write_json(out_dir / "routing_decision.json", routing)
    write_report(out_dir, routing, summary)
    progress["status"] = "complete"
    progress["routing_decision"] = routing["routing_decision"]
    progress["full_population_probe_score_rows"] = int(len(population))
    progress["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(out_dir / "progress.json", progress)


if __name__ == "__main__":
    main()
