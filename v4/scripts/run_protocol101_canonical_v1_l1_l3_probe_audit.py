"""Protocol101 canonical minute-game v1 L1/L3 probe audit.

Offline-only design audit on burned paired recorder days. It consumes the
canonical v1 L0/L2 audit packet, derives authorized Group 1 non-VIX context
features from existing paired traces, runs fixed-policy transfer probes, and
trains small diagnostic transfer probes. It does not train production models,
tune strategy thresholds, call brokers, download paid data, or touch runtime
state.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v4.scripts.run_protocol101_canonical_v1_l0_l2_design_audit import (
    FEATURE_FAMILY,
    FEATURE_NAMES,
    RANDOM_STATE,
    SESSIONS,
    TRACE_PREFIX,
    build_slot_frame,
    finite,
    load_trace_rows,
    trace_paths,
)
from v4.scripts.run_protocol101_feature_recovery_group1_parity_audit import (
    safe_div,
    token_vectors,
    value as group1_value,
)


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_L0_L2_DIR = BASE_AUDIT / "protocol101_canonical_v1_l0_l2_design_audit_attempt001"
DEFAULT_GROUP1_DIR = BASE_AUDIT / "protocol101_live_v2_group1_index_context_parity_resolution"
DEFAULT_GROUP1_DEFINITION = (
    BASE_AUDIT
    / "protocol101_live_v2_feature_recovery_group1_stable_index_context_plan"
    / "feature_group_definition.json"
)
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_canonical_v1_l1_l3_probe_audit_attempt001"
CONTRACT = "protocol101-live-v2-microstructure-masked"
TRANSFORM = "mask_vendor_sensitive_option_quote_greek_microstructure"
SCHEMA_VERSION = "Protocol101CanonicalV1L1L3ProbeAuditAttempt001"
EXCLUDED_PRE_WINDOW_MINUTE = "09:31"
ACTION_AGREEMENT_PASS_MIN = 0.98
ACTION_AGREEMENT_REJECT_BELOW = 0.90
TOP_TERCILE_CONCENTRATION_MAX = 0.50
TOP_TERCILE_REJECT_ABOVE = 0.67

GROUP1_NON_VIX_FEATURES = [
    "spx_vwap_gap_points",
    "spx_vwap_gap_bps",
    "spx_vwap_gap_over_session_range",
    "session_range_bps",
    "momentum_5m_bps",
    "momentum_15m_bps",
    "momentum_5m_over_session_range",
    "momentum_15m_over_session_range",
    "omar_clipped_neg3_pos3",
    "vwap_side_alignment_flag",
    "omar_side_alignment_flag",
    "momentum15_side_alignment_flag",
]
FORBIDDEN_FEATURE_NAMES = [
    "vix_change_5m",
    "vix_change_15m",
    "vix_change_5m_bps",
    "vix_change_15m_bps",
    "raw_bid",
    "raw_ask",
    "raw_spread",
    "raw_bid_size",
    "raw_ask_size",
    "raw_quote_age_ms",
    "option_ohlcv_volume",
    "stat_open_interest",
    "vendor_greeks",
    "distance_to_guard_boundary",
    "sub_minute_fields",
]
L3_SUBSETS = {
    "AB_plus_group1_nonvix": ["A", "B"],
    "AB_C_plus_group1_nonvix": ["A", "B", "C"],
    "AB_D_plus_group1_nonvix": ["A", "B", "D"],
    "AB_E_plus_group1_nonvix": ["A", "B", "E"],
    "AB_CDE_plus_group1_nonvix": ["A", "B", "C", "D", "E"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l0-l2-dir", type=Path, default=DEFAULT_L0_L2_DIR)
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


def minute_bucket(minute_et: str) -> str:
    hour, minute = minute_et.split(":")
    total = int(hour) * 60 + int(minute)
    if total < 10 * 60 + 30:
        return "open_0932_1029"
    if total < 12 * 60:
        return "morning_1030_1159"
    if total < 14 * 60:
        return "midday_1200_1359"
    if total < 15 * 60:
        return "afternoon_1400_1459"
    return "late_1500_1530"


def hash_unit_interval(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def probe_policy_definition(admitted_features: list[str], group1_features: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "Protocol101CanonicalV1L1L3ProbePolicyDefinitionV1",
        "attempt_id": DEFAULT_OUT_DIR.name,
        "contract": CONTRACT,
        "transform": TRANSFORM,
        "authorized_feature_inputs": {
            "canonical_l0_l2_admitted_features": admitted_features,
            "group1_non_vix_features": group1_features,
        },
        "forbidden_feature_inputs": FORBIDDEN_FEATURE_NAMES,
        "row_rule": {
            "score_window": "09:32-15:30 ET",
            "exclude_09_31_from_scoring": True,
            "static_slots_per_row_required": 42,
        },
        "l1_fixed_policy_probes": {
            "random_with_guards_null": {
                "score": "deterministic sha256 uniform per session/minute/slot",
                "threshold": 0.98,
                "gating": "diagnostic null; not subject to 98% non-random pass gate",
            },
            "vwap_side_alignment": {
                "score": "vwap_side_alignment_flag * abs(spx_vwap_gap_over_session_range)",
                "threshold": "historical-plane q70 over positive finite scores",
            },
            "spx_momentum_following": {
                "score": "momentum15_side_alignment_flag * abs(momentum_15m_over_session_range)",
                "threshold": "historical-plane q70 over positive finite scores",
            },
            "spx_mean_reversion": {
                "score": "(1 - vwap_side_alignment_flag) * abs(spx_vwap_gap_over_session_range)",
                "threshold": "historical-plane q70 over positive finite scores",
            },
            "option_mid_momentum": {
                "score": "C.mid.logret_5m",
                "threshold": "historical-plane q80 over positive finite scores",
            },
            "straddle_mid_expansion": {
                "score": "D.near_atm.straddle_mid_spot_bps",
                "threshold": "historical-plane q80 over finite scores",
            },
            "put_call_ratio_skew": {
                "score": "abs(log(D.near_atm.put_call_mid_ratio))",
                "threshold": "historical-plane q80 over finite scores",
            },
            "internal_iv_expansion_compression": {
                "score": "abs(E.bs.iv - historical-plane median E.bs.iv)",
                "threshold": "historical-plane q80 over finite centered scores",
            },
            "internal_delta_geometry": {
                "score": "-abs(abs(E.bs.delta) - 0.35) - abs(B.ladder.offset)/100",
                "threshold": "historical-plane q80 over finite scores",
            },
        },
        "l1_gates": {
            "non_random_action_agreement_min": ACTION_AGREEMENT_PASS_MIN,
            "reject_if_non_random_action_agreement_below": ACTION_AGREEMENT_REJECT_BELOW,
            "top_vol_or_opportunity_tercile_disagreement_share_max": TOP_TERCILE_CONCENTRATION_MAX,
            "reject_if_top_vol_or_opportunity_share_above": TOP_TERCILE_REJECT_ABOVE,
        },
        "l3_transfer_probes": {
            "diagnostic_models": {
                "logistic": {"family": "logistic", "max_iter": 1000, "C": 0.5},
                "bounded_hgb": {"family": "HistGradientBoostingClassifier", "max_depth": 3, "max_trees": 160},
            },
            "target": "sign(y), where y is the L0 15m forward conservative IBKR-plane return",
            "score_threshold": "historical-plane q95 of diagnostic model scores for that subset/model",
            "feature_subsets": L3_SUBSETS,
        },
        "l3_gates": {
            "action_agreement_min_per_subset_model": ACTION_AGREEMENT_PASS_MIN,
            "reject_if_action_agreement_below": ACTION_AGREEMENT_REJECT_BELOW,
            "top_vol_or_opportunity_tercile_disagreement_share_max": TOP_TERCILE_CONCENTRATION_MAX,
            "reject_if_top_vol_or_opportunity_share_above": TOP_TERCILE_REJECT_ABOVE,
            "score_correlation_and_score_drift_are_diagnostic_only": True,
        },
        "routing_decisions": [
            "canonical_v1_probe_pass",
            "canonical_v1_probe_repair_needed",
            "canonical_v1_probe_rejected_transfer_unstable",
            "canonical_v1_probe_insufficient_artifacts",
        ],
    }


def preregistration(
    out_dir: Path,
    policy_path: Path,
    policy_hash: str,
    l0_l2_dir: Path,
    group1_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "Protocol101CanonicalV1L1L3ProbePreregistrationV1",
        "attempt_id": out_dir.name,
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "input_l0_l2_dir": str(l0_l2_dir),
        "input_group1_resolution_dir": str(group1_dir),
        "probe_policy_definition_path": str(policy_path),
        "probe_policy_definition_sha256": policy_hash,
        "sessions": list(SESSIONS),
        "trace_prefix": TRACE_PREFIX,
        "l3_label": "15m forward conservative IBKR-plane return and sign(y)",
        "comparisons_started_after_preregistration": True,
        "scope": "design audit on burned paired recorder days only",
        "side_effects_forbidden": {
            "full_strategy_training": True,
            "production_model_training": True,
            "threshold_optimization": True,
            "uplift_cv_on_15mo_corpus": True,
            "broker_api_calls": True,
            "paid_downloads": True,
            "paper_submit": True,
            "promotion_default_runtime_launchd_edits": True,
        },
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_authorized_features(l0_l2_dir: Path, group1_dir: Path) -> tuple[list[str], list[str], list[str]]:
    verdicts = pd.read_csv(l0_l2_dir / "per_feature_verdicts.csv")
    admitted = sorted(str(row.feature) for row in verdicts.itertuples(index=False) if bool(row.admitted))
    group1_summary = load_json(group1_dir / "summary.json")
    group1_features = list(group1_summary.get("eligible_subset_features") or [])
    missing = [name for name in GROUP1_NON_VIX_FEATURES if name not in group1_features]
    extras = [name for name in group1_features if name not in GROUP1_NON_VIX_FEATURES]
    if missing or extras:
        raise RuntimeError(f"group1 feature set mismatch missing={missing} extras={extras}")
    return admitted, group1_features, missing


def write_preregistration(out_dir: Path, l0_l2_dir: Path, group1_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    admitted, group1_features, _ = load_authorized_features(l0_l2_dir, group1_dir)
    policy_path = out_dir / "probe_policy_definition.json"
    policy = probe_policy_definition(admitted, group1_features)
    policy["attempt_id"] = out_dir.name
    write_json(policy_path, policy)
    policy_hash = sha256_path(policy_path)
    prereg = preregistration(out_dir, policy_path, policy_hash, l0_l2_dir, group1_dir)
    write_json(out_dir / "preregistration.json", prereg)
    return prereg


def impute_zero(value: float | None) -> tuple[float, bool]:
    if value is None or not math.isfinite(float(value)):
        return 0.0, True
    return float(value), False


def derive_group1_context_features(row: dict[str, Any], slot: dict[str, Any], definition: dict[str, Any]) -> dict[str, float]:
    vectors = token_vectors(row)
    vector = next(iter(vectors.values()), None)
    if vector is None:
        vector = []
    guards = definition["deterministic_guards"]
    min_denominator = float(guards["minimum_denominator_abs"])
    range_floor = float(guards["session_range_floor_points"])
    omar_min = float(guards["omar_clip_min"])
    omar_max = float(guards["omar_clip_max"])
    spx, _ = impute_zero(group1_value(vector, "market_last.spx_close"))
    vwap, _ = impute_zero(group1_value(vector, "market_last.spx_vwap"))
    omar, _ = impute_zero(group1_value(vector, "market_last.omar"))
    session_range, _ = impute_zero(group1_value(vector, "market_last.session_range"))
    momentum_5m, _ = impute_zero(group1_value(vector, "market_last.momentum_5m"))
    momentum_15m, _ = impute_zero(group1_value(vector, "market_last.momentum_15m"))
    gap = spx - vwap
    right = str(slot.get("right") or "")
    return {
        "spx_vwap_gap_points": gap,
        "spx_vwap_gap_bps": safe_div(gap, spx, min_denominator) * 10_000.0,
        "spx_vwap_gap_over_session_range": safe_div(gap, session_range, range_floor),
        "session_range_bps": safe_div(session_range, spx, min_denominator) * 10_000.0,
        "momentum_5m_bps": safe_div(momentum_5m, spx, min_denominator) * 10_000.0,
        "momentum_15m_bps": safe_div(momentum_15m, spx, min_denominator) * 10_000.0,
        "momentum_5m_over_session_range": safe_div(momentum_5m, session_range, range_floor),
        "momentum_15m_over_session_range": safe_div(momentum_15m, session_range, range_floor),
        "omar_clipped_neg3_pos3": min(max(omar, omar_min), omar_max),
        "vwap_side_alignment_flag": float((right == "C" and gap > 0.0) or (right == "P" and gap < 0.0)),
        "omar_side_alignment_flag": float((right == "C" and omar > 0.0) or (right == "P" and omar < 0.0)),
        "momentum15_side_alignment_flag": float(
            (right == "C" and momentum_15m > 0.0) or (right == "P" and momentum_15m < 0.0)
        ),
    }


def group1_feature_frame(trace_prefix: str, definition_path: Path) -> pd.DataFrame:
    definition = load_json(definition_path)
    records: list[dict[str, Any]] = []
    for session in SESSIONS:
        live_path, historical_path = trace_paths(trace_prefix, session)
        for source, path in (("ibkr", live_path), ("historical", historical_path)):
            rows = load_trace_rows(path)
            for ts, row in sorted(rows.items()):
                trace = row.get("candidate_filter_trace")
                if not isinstance(trace, list):
                    continue
                for slot in trace:
                    strike = finite(slot.get("strike"))
                    right = str(slot.get("right") or "")
                    rec = {
                        "source": source,
                        "session_date": session,
                        "decision_ts_utc": ts.isoformat(),
                        "strike_key": f"{strike:.3f}" if strike is not None else "",
                        "right": right,
                    }
                    rec.update(derive_group1_context_features(row, slot, definition))
                    records.append(rec)
    return pd.DataFrame.from_records(records)


def add_group1_features(slot_frame: pd.DataFrame, trace_prefix: str, definition_path: Path) -> pd.DataFrame:
    group1 = group1_feature_frame(trace_prefix, definition_path)
    key = ["source", "session_date", "decision_ts_utc", "strike_key", "right"]
    return slot_frame.merge(group1, on=key, how="left", validate="one_to_one")


def add_plane_returns(slot_frame: pd.DataFrame) -> pd.DataFrame:
    frame = slot_frame.sort_values(["source", "session_date", "strike_key", "right", "decision_ts_utc"]).copy()
    grouped = frame.groupby(["source", "session_date", "strike_key", "right"], sort=False)
    frame["plane_exit_bid_t_plus_15m"] = grouped["raw_bid"].shift(-15)
    mask = frame["raw_ask"].notna() & frame["plane_exit_bid_t_plus_15m"].notna() & (frame["raw_ask"] > 0)
    frame["plane_y_15m_return"] = np.nan
    frame["plane_pnl_15m_dollars"] = np.nan
    frame.loc[mask, "plane_y_15m_return"] = (
        (frame.loc[mask, "plane_exit_bid_t_plus_15m"] - frame.loc[mask, "raw_ask"]) / frame.loc[mask, "raw_ask"]
    )
    frame.loc[mask, "plane_pnl_15m_dollars"] = (
        frame.loc[mask, "plane_exit_bid_t_plus_15m"] - frame.loc[mask, "raw_ask"]
    ) * 100.0
    return frame


def pair_slot_frame(slot_frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    score = slot_frame[slot_frame["score_row"]].copy()
    key_cols = ["session_date", "decision_minute_et", "decision_ts_utc", "strike_key", "right"]
    keep = key_cols + [
        "contract_id",
        "strike",
        "raw_bid",
        "raw_ask",
        "raw_spread",
        "plane_y_15m_return",
        "plane_pnl_15m_dollars",
        "y_15m_conservative_return",
        "realized_5m_spx_vol",
    ]
    keep += feature_cols
    historical = score[score["source"] == "historical"][keep]
    ibkr = score[score["source"] == "ibkr"][keep]
    paired = historical.merge(
        ibkr,
        on=key_cols,
        how="inner",
        suffixes=("_historical", "_ibkr"),
        validate="one_to_one",
    )
    paired["time_bucket"] = paired["decision_minute_et"].map(minute_bucket)
    paired["pair_slot_id"] = (
        paired["session_date"].astype(str)
        + "|"
        + paired["decision_minute_et"].astype(str)
        + "|"
        + paired["strike_key"].astype(str)
        + "|"
        + paired["right"].astype(str)
    )
    return paired


def assign_terciles(paired: pd.DataFrame) -> pd.DataFrame:
    frame = paired.copy()
    minute = (
        frame.groupby(["session_date", "decision_minute_et"], as_index=False)
        .agg(
            realized_5m_spx_vol=("realized_5m_spx_vol_ibkr", "max"),
            opportunity=("y_15m_conservative_return_ibkr", "max"),
        )
    )
    for col in ("realized_5m_spx_vol", "opportunity"):
        values = minute[col].replace([np.inf, -np.inf], np.nan)
        q1 = values.quantile(1 / 3)
        q2 = values.quantile(2 / 3)
        labels = []
        for value in values:
            if pd.isna(value) or not math.isfinite(float(value)):
                labels.append("missing")
            elif value <= q1:
                labels.append("low")
            elif value <= q2:
                labels.append("mid")
            else:
                labels.append("top")
        minute[col + "_tercile"] = labels
    frame = frame.merge(
        minute[["session_date", "decision_minute_et", "realized_5m_spx_vol_tercile", "opportunity_tercile"]],
        on=["session_date", "decision_minute_et"],
        how="left",
        validate="many_to_one",
    )
    return frame


def finite_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def quantile_threshold(values: pd.Series, q: float, positive_only: bool = False) -> float:
    series = finite_series(values).dropna()
    if positive_only:
        series = series[series > 0]
    if series.empty:
        return float("inf")
    return float(series.quantile(q))


def iv_center_threshold(paired: pd.DataFrame) -> tuple[float, float]:
    iv = finite_series(paired["E.bs.iv_historical"]).dropna()
    if iv.empty:
        return 0.0, float("inf")
    center = float(iv.median())
    threshold = float((iv - center).abs().quantile(0.80))
    return center, threshold


def l1_score(paired: pd.DataFrame, probe: str, plane: str, thresholds: dict[str, Any]) -> pd.Series:
    suffix = f"_{plane}"
    if probe == "random_with_guards_null":
        raw = pd.Series(
            [
                hash_unit_interval(probe, row.session_date, row.decision_minute_et, row.strike_key, row.right)
                for row in paired.itertuples(index=False)
            ],
            index=paired.index,
        )
        suffix = f"_{plane}"
        mid = finite_series(paired[f"C.mid.mid_tick_q{suffix}"])
        guarded = raw.where(mid.between(0.5, 35.0))
        return guarded
    if probe == "vwap_side_alignment":
        return finite_series(paired[f"vwap_side_alignment_flag{suffix}"]) * finite_series(
            paired[f"spx_vwap_gap_over_session_range{suffix}"]
        ).abs()
    if probe == "spx_momentum_following":
        return finite_series(paired[f"momentum15_side_alignment_flag{suffix}"]) * finite_series(
            paired[f"momentum_15m_over_session_range{suffix}"]
        ).abs()
    if probe == "spx_mean_reversion":
        return (1.0 - finite_series(paired[f"vwap_side_alignment_flag{suffix}"])) * finite_series(
            paired[f"spx_vwap_gap_over_session_range{suffix}"]
        ).abs()
    if probe == "option_mid_momentum":
        return finite_series(paired[f"C.mid.logret_5m{suffix}"])
    if probe == "straddle_mid_expansion":
        return finite_series(paired[f"D.near_atm.straddle_mid_spot_bps{suffix}"])
    if probe == "put_call_ratio_skew":
        ratio = finite_series(paired[f"D.near_atm.put_call_mid_ratio{suffix}"])
        return np.log(ratio.where(ratio > 0)).abs()
    if probe == "internal_iv_expansion_compression":
        center = float(thresholds[probe]["center"])
        return (finite_series(paired[f"E.bs.iv{suffix}"]) - center).abs()
    if probe == "internal_delta_geometry":
        delta = finite_series(paired[f"E.bs.delta{suffix}"]).abs()
        offset = finite_series(paired[f"B.ladder.offset{suffix}"]).abs()
        return -((delta - 0.35).abs()) - (offset / 100.0)
    raise KeyError(probe)


def resolve_l1_thresholds(paired: pd.DataFrame) -> dict[str, Any]:
    thresholds: dict[str, Any] = {
        "random_with_guards_null": {"threshold": 0.98, "rule": "fixed deterministic null threshold"},
    }
    center, iv_threshold = iv_center_threshold(paired)
    probe_rules = {
        "vwap_side_alignment": (0.70, True),
        "spx_momentum_following": (0.70, True),
        "spx_mean_reversion": (0.70, True),
        "option_mid_momentum": (0.80, True),
        "straddle_mid_expansion": (0.80, False),
        "put_call_ratio_skew": (0.80, False),
        "internal_delta_geometry": (0.80, False),
    }
    for probe, (q, positive_only) in probe_rules.items():
        scores = l1_score(paired, probe, "historical", {"internal_iv_expansion_compression": {"center": center}})
        thresholds[probe] = {
            "threshold": quantile_threshold(scores, q, positive_only=positive_only),
            "quantile": q,
            "positive_only": positive_only,
            "source": "historical_plane",
        }
    thresholds["internal_iv_expansion_compression"] = {
        "threshold": iv_threshold,
        "center": center,
        "quantile": 0.80,
        "source": "historical_plane_centered_abs_deviation",
    }
    return thresholds


def decisions_from_scores(
    paired: pd.DataFrame,
    scores: pd.Series,
    threshold: float,
    *,
    plane: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    work = paired[["session_date", "decision_minute_et", "strike_key", "right", f"plane_pnl_15m_dollars_{plane}"]].copy()
    work["score"] = finite_series(scores)
    for (session, minute), group in work.groupby(["session_date", "decision_minute_et"], sort=False):
        eligible = group[group["score"].notna()]
        if eligible.empty or not math.isfinite(float(threshold)):
            records.append({"session_date": session, "decision_minute_et": minute, "action": "wait"})
            continue
        idx = eligible["score"].idxmax()
        best = eligible.loc[idx]
        if float(best["score"]) >= float(threshold):
            records.append(
                {
                    "session_date": session,
                    "decision_minute_et": minute,
                    "action": "enter",
                    "selected_slot": f"{best['strike_key']}|{best['right']}",
                    "selected_strike": best["strike_key"],
                    "selected_side": best["right"],
                    "selected_score": float(best["score"]),
                    "selected_pnl": finite(best[f"plane_pnl_15m_dollars_{plane}"]),
                    "threshold_distance": float(best["score"]) - float(threshold),
                }
            )
        else:
            records.append(
                {
                    "session_date": session,
                    "decision_minute_et": minute,
                    "action": "wait",
                    "selected_score": float(best["score"]),
                    "threshold_distance": float(best["score"]) - float(threshold),
                }
            )
    frame = pd.DataFrame.from_records(records)
    for col in (
        "selected_slot",
        "selected_strike",
        "selected_side",
        "selected_score",
        "selected_pnl",
        "threshold_distance",
    ):
        if col not in frame.columns:
            frame[col] = np.nan
    return frame


def summarize_decisions(
    *,
    kind: str,
    probe_name: str,
    historical_decisions: pd.DataFrame,
    ibkr_decisions: pd.DataFrame,
    minute_context: pd.DataFrame,
    feature_family: str,
    random_probe: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = historical_decisions.merge(
        ibkr_decisions,
        on=["session_date", "decision_minute_et"],
        how="inner",
        suffixes=("_historical", "_ibkr"),
        validate="one_to_one",
    ).merge(minute_context, on=["session_date", "decision_minute_et"], how="left", validate="one_to_one")
    merged["action_agree"] = merged["action_historical"] == merged["action_ibkr"]
    both_enter = (merged["action_historical"] == "enter") & (merged["action_ibkr"] == "enter")
    merged["selected_slot_agree"] = both_enter & (merged["selected_slot_historical"] == merged["selected_slot_ibkr"])
    merged["side_agree"] = both_enter & (merged["selected_side_historical"] == merged["selected_side_ibkr"])
    merged["disagreement"] = ~(
        merged["action_agree"] & ((~both_enter) | merged["selected_slot_agree"].fillna(False))
    )
    hist_set = set(
        zip(
            merged.loc[merged["action_historical"] == "enter", "session_date"],
            merged.loc[merged["action_historical"] == "enter", "decision_minute_et"],
            merged.loc[merged["action_historical"] == "enter", "selected_slot_historical"],
        )
    )
    ibkr_set = set(
        zip(
            merged.loc[merged["action_ibkr"] == "enter", "session_date"],
            merged.loc[merged["action_ibkr"] == "enter", "decision_minute_et"],
            merged.loc[merged["action_ibkr"] == "enter", "selected_slot_ibkr"],
        )
    )
    union = hist_set | ibkr_set
    jaccard = len(hist_set & ibkr_set) / len(union) if union else 1.0
    action_agreement = float(merged["action_agree"].mean())
    selected_slot_agreement = float(merged.loc[both_enter, "selected_slot_agree"].mean()) if both_enter.any() else 1.0
    side_agreement = float(merged.loc[both_enter, "side_agree"].mean()) if both_enter.any() else 1.0
    disagreements = merged[merged["disagreement"]]
    top_vol_share = float((disagreements["realized_5m_spx_vol_tercile"] == "top").mean()) if len(disagreements) else 0.0
    top_opp_share = float((disagreements["opportunity_tercile"] == "top").mean()) if len(disagreements) else 0.0
    day_pnl = (
        merged.groupby("session_date", as_index=False)
        .agg(
            historical_day_pnl=("selected_pnl_historical", "sum"),
            ibkr_day_pnl=("selected_pnl_ibkr", "sum"),
        )
        .fillna(0.0)
    )
    if len(day_pnl) >= 2 and day_pnl["historical_day_pnl"].std() > 0 and day_pnl["ibkr_day_pnl"].std() > 0:
        day_corr = float(day_pnl["historical_day_pnl"].corr(day_pnl["ibkr_day_pnl"]))
    else:
        day_corr = None
    threshold_dist = pd.concat(
        [
            finite_series(merged.get("threshold_distance_historical", pd.Series(dtype=float))),
            finite_series(merged.get("threshold_distance_ibkr", pd.Series(dtype=float))),
        ],
        ignore_index=True,
    ).dropna()
    route = "pass"
    if (not random_probe and action_agreement < ACTION_AGREEMENT_REJECT_BELOW) or max(top_vol_share, top_opp_share) > TOP_TERCILE_REJECT_ABOVE:
        route = "reject"
    elif (not random_probe and action_agreement < ACTION_AGREEMENT_PASS_MIN) or max(top_vol_share, top_opp_share) > TOP_TERCILE_CONCENTRATION_MAX:
        route = "repair"
    summary = {
        "kind": kind,
        "probe": probe_name,
        "feature_family": feature_family,
        "minutes": int(len(merged)),
        "historical_actions": int((merged["action_historical"] == "enter").sum()),
        "ibkr_actions": int((merged["action_ibkr"] == "enter").sum()),
        "action_agreement": action_agreement,
        "selected_slot_agreement": selected_slot_agreement,
        "trade_set_jaccard": jaccard,
        "side_agreement": side_agreement,
        "disagreements": int(len(disagreements)),
        "top_vol_tercile_disagreement_share": top_vol_share,
        "top_opportunity_tercile_disagreement_share": top_opp_share,
        "mean_pnl_delta_ibkr_minus_historical": finite((merged["selected_pnl_ibkr"] - merged["selected_pnl_historical"]).mean()),
        "day_pnl_correlation_descriptive": day_corr,
        "threshold_distance_abs_p05": finite(threshold_dist.abs().quantile(0.05)) if not threshold_dist.empty else None,
        "route": route,
        "failure_family": feature_family if route != "pass" else None,
    }
    example_cols = [
        "session_date",
        "decision_minute_et",
        "time_bucket",
        "realized_5m_spx_vol_tercile",
        "opportunity_tercile",
        "action_historical",
        "action_ibkr",
        "selected_slot_historical",
        "selected_slot_ibkr",
        "selected_score_historical",
        "selected_score_ibkr",
        "selected_pnl_historical",
        "selected_pnl_ibkr",
        "threshold_distance_historical",
        "threshold_distance_ibkr",
    ]
    examples = disagreements.assign(kind=kind, probe=probe_name, feature_family=feature_family)
    keep = ["kind", "probe", "feature_family"] + [col for col in example_cols if col in examples.columns]
    return summary, examples[keep].head(100)


def minute_context(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby(["session_date", "decision_minute_et"], as_index=False)
        .agg(
            realized_5m_spx_vol_tercile=("realized_5m_spx_vol_tercile", "first"),
            opportunity_tercile=("opportunity_tercile", "first"),
            time_bucket=("time_bucket", "first"),
        )
    )


def run_l1(paired: pd.DataFrame, out_dir: Path) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    thresholds = resolve_l1_thresholds(paired)
    context = minute_context(paired)
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
    rows: list[dict[str, Any]] = []
    examples: list[pd.DataFrame] = []
    for probe, family in probe_families.items():
        threshold = float(thresholds[probe]["threshold"])
        hist_scores = l1_score(paired, probe, "historical", thresholds)
        ibkr_scores = l1_score(paired, probe, "ibkr", thresholds)
        hist = decisions_from_scores(paired, hist_scores, threshold, plane="historical")
        ibkr = decisions_from_scores(paired, ibkr_scores, threshold, plane="ibkr")
        summary, ex = summarize_decisions(
            kind="L1",
            probe_name=probe,
            historical_decisions=hist,
            ibkr_decisions=ibkr,
            minute_context=context,
            feature_family=family,
            random_probe=probe == "random_with_guards_null",
        )
        summary["threshold"] = threshold
        summary["threshold_rule"] = thresholds[probe]
        rows.append(summary)
        if not ex.empty:
            examples.append(ex)
    with (out_dir / "l1_fixed_policy_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row if key != "threshold_rule"}))
        writer.writeheader()
        for row in rows:
            copy = {key: value for key, value in row.items() if key != "threshold_rule"}
            writer.writerow(copy)
    l1_summary = {
        "schema_version": "Protocol101CanonicalV1L1SummaryV1",
        "thresholds": thresholds,
        "results": rows,
        "pass": all(row["route"] == "pass" for row in rows if row["probe"] != "random_with_guards_null"),
    }
    write_json(out_dir / "l1_summary.json", l1_summary)
    return rows, pd.concat(examples, ignore_index=True) if examples else pd.DataFrame(), thresholds


def l3_feature_columns(subset_name: str, admitted_features: list[str]) -> list[str]:
    families = set(L3_SUBSETS[subset_name])
    columns = [name for name in admitted_features if FEATURE_FAMILY.get(name) in families]
    columns.extend(GROUP1_NON_VIX_FEATURES)
    return list(dict.fromkeys(columns))


def matrix(frame: pd.DataFrame, cols: list[str], suffix: str = "") -> pd.DataFrame:
    use = [f"{col}{suffix}" for col in cols]
    out = frame[use].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out.columns = cols
    return out


def train_l3_model(model_name: str) -> Any:
    if model_name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.5, solver="lbfgs", random_state=RANDOM_STATE),
        )
    if model_name == "bounded_hgb":
        return HistGradientBoostingClassifier(
            max_iter=160,
            max_depth=3,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        )
    raise KeyError(model_name)


def decision_from_model_scores(
    paired: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    *,
    plane: str,
) -> pd.DataFrame:
    return decisions_from_scores(paired, pd.Series(scores, index=paired.index), threshold, plane=plane)


def score_stats(hist_scores: np.ndarray, ibkr_scores: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(hist_scores) & np.isfinite(ibkr_scores)
    if int(mask.sum()) < 2:
        corr = None
    elif np.std(hist_scores[mask]) == 0 or np.std(ibkr_scores[mask]) == 0:
        corr = None
    else:
        corr = float(np.corrcoef(hist_scores[mask], ibkr_scores[mask])[0, 1])
    drift = pd.Series(ibkr_scores[mask] - hist_scores[mask])
    return {
        "score_correlation": corr,
        "score_drift_median": finite(drift.median()) if not drift.empty else None,
        "score_drift_p95_abs": finite(drift.abs().quantile(0.95)) if not drift.empty else None,
        "score_drift_max_abs": finite(drift.abs().max()) if not drift.empty else None,
    }


def run_l3(paired: pd.DataFrame, admitted_features: list[str], out_dir: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    train_mask = paired["y_15m_conservative_return_ibkr"].notna()
    y = (paired.loc[train_mask, "y_15m_conservative_return_ibkr"].astype(float) > 0).astype(int).to_numpy()
    context = minute_context(paired)
    rows: list[dict[str, Any]] = []
    examples: list[pd.DataFrame] = []
    for subset_name in L3_SUBSETS:
        cols = l3_feature_columns(subset_name, admitted_features)
        for model_name in ("logistic", "bounded_hgb"):
            model = train_l3_model(model_name)
            x_train = matrix(paired.loc[train_mask], cols, "_historical")
            if len(np.unique(y)) < 2:
                summary = {
                    "kind": "L3",
                    "probe": f"{subset_name}_{model_name}",
                    "subset": subset_name,
                    "model": model_name,
                    "route": "insufficient",
                    "reason": "single_class_label",
                }
                rows.append(summary)
                continue
            model.fit(x_train, y)
            hist_scores = model.predict_proba(matrix(paired, cols, "_historical"))[:, 1]
            ibkr_scores = model.predict_proba(matrix(paired, cols, "_ibkr"))[:, 1]
            threshold = float(pd.Series(hist_scores).quantile(0.95))
            hist = decision_from_model_scores(paired, hist_scores, threshold, plane="historical")
            ibkr = decision_from_model_scores(paired, ibkr_scores, threshold, plane="ibkr")
            summary, ex = summarize_decisions(
                kind="L3",
                probe_name=f"{subset_name}_{model_name}",
                historical_decisions=hist,
                ibkr_decisions=ibkr,
                minute_context=context,
                feature_family="+".join(L3_SUBSETS[subset_name]) + "+G1",
            )
            summary.update(
                {
                    "subset": subset_name,
                    "model": model_name,
                    "feature_count": len(cols),
                    "threshold": threshold,
                    "threshold_rule": "historical-plane q95 diagnostic score",
                    **score_stats(hist_scores, ibkr_scores),
                }
            )
            rows.append(summary)
            if not ex.empty:
                examples.append(ex)
    with (out_dir / "l3_transfer_probe_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    l3_summary = {
        "schema_version": "Protocol101CanonicalV1L3SummaryV1",
        "target": "sign(y), with y = 15m forward conservative IBKR-plane return",
        "models": ["logistic", "bounded_hgb"],
        "subsets": L3_SUBSETS,
        "results": rows,
        "pass": all(row.get("route") == "pass" for row in rows),
    }
    write_json(out_dir / "l3_summary.json", l3_summary)
    return rows, pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()


def route(l1_rows: list[dict[str, Any]], l3_rows: list[dict[str, Any]], admitted_features: list[str]) -> dict[str, Any]:
    all_rows = [row for row in l1_rows if row.get("probe") != "random_with_guards_null"] + l3_rows
    insufficient = [row for row in all_rows if row.get("route") == "insufficient"]
    rejected = [row for row in all_rows if row.get("route") == "reject"]
    repair = [row for row in all_rows if row.get("route") == "repair"]
    if insufficient:
        decision = "canonical_v1_probe_insufficient_artifacts"
    elif rejected:
        decision = "canonical_v1_probe_rejected_transfer_unstable"
    elif repair:
        decision = "canonical_v1_probe_repair_needed"
    else:
        decision = "canonical_v1_probe_pass"
    return {
        "schema_version": "Protocol101CanonicalV1L1L3RoutingDecisionV1",
        "attempt_id": DEFAULT_OUT_DIR.name,
        "routing_decision": decision,
        "authorized_next_step_if_pass": "canonical_v1_training_design_on_15mo_corpus_using_admitted_features_noise_injection_conservative_fill_labels_and_G1_G9_gates"
        if decision == "canonical_v1_probe_pass"
        else None,
        "highest_allowed_claim": "canonical v1 L1/L3 probe audit complete",
        "admitted_canonical_features": admitted_features,
        "authorized_group1_non_vix_features": GROUP1_NON_VIX_FEATURES,
        "l1_probe_routes": {row["probe"]: row.get("route") for row in l1_rows},
        "l3_probe_routes": {row["probe"]: row.get("route") for row in l3_rows},
        "repair_items": [
            {
                "kind": row.get("kind"),
                "probe": row.get("probe"),
                "feature_family": row.get("feature_family"),
                "action_agreement": row.get("action_agreement"),
                "top_vol_tercile_disagreement_share": row.get("top_vol_tercile_disagreement_share"),
                "top_opportunity_tercile_disagreement_share": row.get("top_opportunity_tercile_disagreement_share"),
                "route": row.get("route"),
            }
            for row in repair + rejected + insufficient
        ],
        "claims_not_made": {
            "profitability": True,
            "paper_readiness": True,
            "promotion_readiness": True,
            "real_money_readiness": True,
            "training_executed_on_15mo_corpus": True,
        },
        "side_effect_policy": {
            "full_strategy_training_executed": False,
            "production_model_training_executed": False,
            "diagnostic_transfer_probe_training_executed": True,
            "threshold_optimization_executed": False,
            "uplift_cv_on_15mo_corpus_executed": False,
            "broker_endpoint_called": False,
            "paid_data_download": False,
            "paper_submit_allowed": False,
            "promotion_or_default_changed": False,
            "runtime_flags_edited": False,
            "launchd_changed": False,
            "real_money_path_changed": False,
        },
    }


def write_report(out_dir: Path, routing: dict[str, Any], l1_rows: list[dict[str, Any]], l3_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Protocol101 Canonical Minute-Game v1 L1/L3 Probe Audit",
        "",
        f"- Routing decision: `{routing['routing_decision']}`",
        "- Highest allowed claim: `canonical v1 L1/L3 probe audit complete`",
        f"- Contract: `{CONTRACT}`",
        f"- Transform: `{TRANSFORM}`",
        "- Scope: burned paired recorder days only (`2026-06-30`, `2026-07-01`, `2026-07-02`)",
        "",
        "## L1 Fixed Policy Probes",
        "",
    ]
    for row in l1_rows:
        lines.append(
            f"- `{row['probe']}`: route=`{row.get('route')}`, action_agreement=`{row.get('action_agreement')}`, "
            f"slot_agreement=`{row.get('selected_slot_agreement')}`, jaccard=`{row.get('trade_set_jaccard')}`"
        )
    lines.extend(["", "## L3 Transfer Probes", ""])
    for row in l3_rows:
        lines.append(
            f"- `{row['probe']}`: route=`{row.get('route')}`, action_agreement=`{row.get('action_agreement')}`, "
            f"score_corr=`{row.get('score_correlation')}`, score_p95_abs_drift=`{row.get('score_drift_p95_abs')}`"
        )
    lines.extend(
        [
            "",
            "## Routing",
            "",
            f"- Decision: `{routing['routing_decision']}`",
            f"- Repair items: `{len(routing['repair_items'])}`",
            "",
            "If pass, this authorizes only the next planning step: canonical v1 training design on the 15-month corpus using admitted features, measured divergence noise injection, conservative fill labels, and G1-G9 gates.",
            "",
            "## Side Effects",
            "",
            "- full_strategy_training_executed: `false`",
            "- production_model_training_executed: `false`",
            "- diagnostic_transfer_probe_training_executed: `true`",
            "- threshold_optimization_executed: `false`",
            "- uplift_cv_on_15mo_corpus_executed: `false`",
            "- broker_endpoint_called: `false`",
            "- paid_data_download: `false`",
            "- paper_submit_allowed: `false`",
            "- promotion/default/runtime/launchd edits: `false`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_disagreement_examples(out_dir: Path, l1_examples: pd.DataFrame, l3_examples: pd.DataFrame) -> None:
    examples = pd.concat([l1_examples, l3_examples], ignore_index=True) if not l1_examples.empty or not l3_examples.empty else pd.DataFrame()
    if examples.empty:
        examples = pd.DataFrame(
            columns=[
                "kind",
                "probe",
                "feature_family",
                "session_date",
                "decision_minute_et",
                "action_historical",
                "action_ibkr",
                "selected_slot_historical",
                "selected_slot_ibkr",
            ]
        )
    examples.to_csv(out_dir / "disagreement_examples.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} exists; pass --force to overwrite audit artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg = write_preregistration(out_dir, args.l0_l2_dir, args.group1_dir)
    progress = {
        "schema_version": "Protocol101CanonicalV1L1L3ProgressV1",
        "attempt_id": out_dir.name,
        "status": "preregistered",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "full_strategy_training_executed": False,
        "production_model_training_executed": False,
        "diagnostic_transfer_probe_training_executed": False,
        "threshold_optimization_executed": False,
        "uplift_cv_on_15mo_corpus_executed": False,
        "broker_endpoint_called": False,
        "paid_data_download": False,
        "paper_submit_allowed": False,
        "promotion_or_default_changed": False,
        "runtime_flags_edited": False,
        "launchd_changed": False,
        "real_money_path_changed": False,
    }
    write_json(out_dir / "progress.json", progress)
    admitted_features, group1_features, _ = load_authorized_features(args.l0_l2_dir, args.group1_dir)
    l0_routing = load_json(args.l0_l2_dir / "routing_decision.json")
    group1_summary = load_json(args.group1_dir / "summary.json")
    blockers: list[str] = []
    if l0_routing.get("routing_decision") != "canonical_v1_design_pass":
        blockers.append("l0_l2_not_design_pass")
    if not group1_summary.get("subset_eligibility"):
        blockers.append("group1_non_vix_subset_not_eligible")
    slot_frame, readiness = build_slot_frame(args.trace_prefix)
    if readiness.get("blockers"):
        blockers.extend(readiness["blockers"])
    if slot_frame.empty:
        blockers.append("empty_slot_frame")
    input_readiness = {
        "schema_version": "Protocol101CanonicalV1L1L3InputReadinessV1",
        "blockers": blockers,
        "l0_l2_routing_decision": l0_routing.get("routing_decision"),
        "group1_subset_eligibility": group1_summary.get("subset_eligibility"),
        "canonical_admitted_feature_count": len(admitted_features),
        "group1_non_vix_feature_count": len(group1_features),
        "trace_readiness": readiness,
    }
    write_json(out_dir / "input_readiness.json", input_readiness)
    if blockers:
        routing = {
            "schema_version": "Protocol101CanonicalV1L1L3RoutingDecisionV1",
            "attempt_id": out_dir.name,
            "routing_decision": "canonical_v1_probe_insufficient_artifacts",
            "blockers": blockers,
            "highest_allowed_claim": "canonical v1 L1/L3 probe audit complete",
        }
        write_json(out_dir / "routing_decision.json", routing)
        write_report(out_dir, routing, [], [])
        progress["status"] = "complete_insufficient_artifacts"
        progress["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_json(out_dir / "progress.json", progress)
        return
    slot_frame = add_group1_features(slot_frame, args.trace_prefix, args.group1_definition)
    slot_frame = add_plane_returns(slot_frame)
    feature_cols = admitted_features + group1_features
    paired = pair_slot_frame(slot_frame, feature_cols)
    paired = assign_terciles(paired)
    progress["status"] = "running_l1"
    write_json(out_dir / "progress.json", progress)
    l1_rows, l1_examples, _ = run_l1(paired, out_dir)
    progress["status"] = "running_l3"
    progress["diagnostic_transfer_probe_training_executed"] = True
    write_json(out_dir / "progress.json", progress)
    l3_rows, l3_examples = run_l3(paired, admitted_features, out_dir)
    write_disagreement_examples(out_dir, l1_examples, l3_examples)
    routing = route(l1_rows, l3_rows, admitted_features)
    routing["attempt_id"] = out_dir.name
    routing["probe_policy_definition_sha256"] = sha256_path(out_dir / "probe_policy_definition.json")
    routing["preregistration_sha256"] = sha256_path(out_dir / "preregistration.json")
    routing["paired_slot_rows"] = int(len(paired))
    routing["paired_minutes"] = int(paired[["session_date", "decision_minute_et"]].drop_duplicates().shape[0])
    write_json(out_dir / "routing_decision.json", routing)
    write_report(out_dir, routing, l1_rows, l3_rows)
    progress["status"] = "complete"
    progress["routing_decision"] = routing["routing_decision"]
    progress["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(out_dir / "progress.json", progress)


if __name__ == "__main__":
    main()
