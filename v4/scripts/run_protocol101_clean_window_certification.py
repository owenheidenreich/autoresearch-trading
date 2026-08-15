"""Certify Protocol101 transfer on causal clean windows instead of whole days.

The input sessions are already-opened confirmation/development evidence. This
runner preserves canonical v1.4 features, probes, models, thresholds, epsilons,
and selection semantics. It changes only the evidence unit: recovered, stale,
misaligned, incomplete, and post-interruption warm-up minutes are quarantined.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v4.scripts import run_protocol101_canonical_v1_4_rehearsal_battery as battery
from v4.scripts import run_protocol101_canonical_v1_4_near_atm_band_restriction_probe as v14
from v4.scripts import run_protocol101_canonical_v1_l0_l2_design_audit as l0l2


AUDIT = Path("v4/audit/autoresearch")
DEFAULT_OUT_DIR = (
    AUDIT / "protocol101_canonical_v1_4_clean_window_certification_attempt001"
)
SESSIONS = (
    "2026-07-15",
    "2026-07-17",
    "2026-07-20",
    "2026-07-21",
    "2026-07-22",
)
TRACE_PREFIX_BY_SESSION = {
    "2026-07-15": "protocol101_canonical_early_confirmation",
    "2026-07-17": "protocol101_canonical_early_confirmation",
    "2026-07-20": "protocol101_canonical_rehearsal",
    "2026-07-21": "protocol101_canonical_early_confirmation",
    "2026-07-22": "protocol101_canonical_early_confirmation",
}
SEALED_CAPTURE_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture_sealed"
OPEN_CAPTURE_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture"
NY = ZoneInfo("America/New_York")

SPX_DRIFT_MAX_POINTS = 0.75
QUOTE_AGE_MAX_MS = 90_000.0
WARMUP_MINUTES = 15
MIN_WINDOW_MINUTES = 30
MIN_DATES = 3
MIN_TOTAL_MINUTES = 600
MIN_TIME_BUCKET_MINUTES = 60
MAX_DATE_SHARE = 0.50
ACTION_AGREEMENT_MIN = 0.98
SLOT_AGREEMENT_MIN = 0.99
SLOT_GATE_MIN_N = 30
L2_AUC_MAX = 0.55
MATERIAL_REORDERING_ALLOWANCE = 1

TIME_BUCKETS = {
    "morning": ("09:32", "10:30"),
    "midday": ("10:30", "14:00"),
    "afternoon": ("14:00", "15:31"),
}

GROUP1_FEATURES = list(v14.GROUP1_NON_VIX_FEATURES)
FAMILY_FEATURES = {
    "AB_index_context_geometry": [
        *[
            name
            for name in l0l2.FEATURE_NAMES
            if l0l2.FEATURE_FAMILY[name] in {"A", "B"}
        ],
        *GROUP1_FEATURES,
    ],
    "C_option_price": [
        name for name in l0l2.FEATURE_NAMES if l0l2.FEATURE_FAMILY[name] == "C"
    ],
    "D_composite": [
        name for name in l0l2.FEATURE_NAMES if l0l2.FEATURE_FAMILY[name] == "D"
    ],
    "E_internal_greek": [
        name for name in l0l2.FEATURE_NAMES if l0l2.FEATURE_FAMILY[name] == "E"
    ],
}
FAMILY_ELIGIBILITY = {
    "AB_index_context_geometry": "context_eligible",
    "C_option_price": "option_eligible",
    "D_composite": "option_eligible",
    "E_internal_greek": "option_eligible",
}
PROBE_ELIGIBILITY = {
    "vwap_side_alignment": "context_eligible",
    "spx_momentum_following": "context_eligible",
    "spx_mean_reversion": "context_eligible",
    "option_mid_momentum": "option_eligible",
    "straddle_mid_expansion": "option_eligible",
    "put_call_ratio_skew": "option_eligible",
    "internal_iv_expansion_compression": "option_eligible",
    "internal_delta_geometry": "option_eligible",
    "AB_plus_group1_nonvix_logistic": "context_eligible",
    "AB_plus_group1_nonvix_bounded_hgb": "context_eligible",
    "AB_C_plus_group1_nonvix_logistic": "context_eligible",
    "AB_C_plus_group1_nonvix_bounded_hgb": "context_eligible",
    "AB_D_plus_group1_nonvix_logistic": "context_eligible",
    "AB_D_plus_group1_nonvix_bounded_hgb": "context_eligible",
    "AB_E_plus_group1_nonvix_logistic": "context_eligible",
    "AB_E_plus_group1_nonvix_bounded_hgb": "context_eligible",
    "AB_CDE_plus_group1_nonvix_logistic": "context_eligible",
    "AB_CDE_plus_group1_nonvix_bounded_hgb": "context_eligible",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preregister", "run"), required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n"
    )


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_dir(session: str, plane: str) -> Path:
    prefix = TRACE_PREFIX_BY_SESSION[session]
    slug = session.replace("-", "_")
    return AUDIT / f"{prefix}_{plane}_replay_{slug}"


def trace_path(session: str, plane: str) -> Path:
    return trace_dir(session, plane) / "decision_traces.jsonl"


def capture_root(session: str) -> Path:
    return OPEN_CAPTURE_ROOT if session == "2026-07-20" else SEALED_CAPTURE_ROOT


def capture_dir(session: str) -> Path:
    return capture_root(session) / session / f"protocol101-recorder-{session}"


def quality_path(session: str) -> Path:
    return capture_dir(session) / "ibkr_capture_quality.json"


def frozen_inputs() -> list[Path]:
    paths = [path for path, _expected in battery.FROZEN.values()]
    paths.extend(
        path
        for session in SESSIONS
        for path in (trace_path(session, "historical"), trace_path(session, "ibkr_capture"))
    )
    paths.extend(quality_path(session) for session in SESSIONS)
    return paths


def preregistration_payload() -> dict[str, Any]:
    inventory = []
    for path in frozen_inputs():
        inventory.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_path(path) if path.exists() else None,
            }
        )
    return {
        "schema_version": "Protocol101CleanWindowCertificationPreregistrationV1",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "evidence_grade": "retrospective_opened_evidence_clean_window_certification",
        "sessions": list(SESSIONS),
        "trace_prefix_by_session": TRACE_PREFIX_BY_SESSION,
        "frozen_contract_sha256": battery.FROZEN["contract"][1],
        "window_policy": {
            "spx_drift_max_points": SPX_DRIFT_MAX_POINTS,
            "quote_age_max_ms": QUOTE_AGE_MAX_MS,
            "required_slots_per_minute": 42,
            "raw_checkpoint_only": True,
            "recovered_checkpoint_quarantined": True,
            "atm_alignment_required": True,
            "live_market_data_required": True,
            "consecutive_healthy_warmup_minutes": WARMUP_MINUTES,
            "minimum_eligible_window_minutes": MIN_WINDOW_MINUTES,
            "context_policy": (
                "opening_context_ready and clean prefix; once the cumulative "
                "context path is interrupted, later context rows are quarantined"
            ),
        },
        "coverage_gates": {
            "minimum_distinct_dates": MIN_DATES,
            "minimum_total_minutes": MIN_TOTAL_MINUTES,
            "minimum_minutes_per_time_bucket": MIN_TIME_BUCKET_MINUTES,
            "maximum_single_date_share": MAX_DATE_SHARE,
            "time_buckets_et": TIME_BUCKETS,
        },
        "frozen_transfer_gates": {
            "l0": l0l2.canonical_feature_definition()["l0_thresholds"],
            "l2_source_auc_max": L2_AUC_MAX,
            "action_agreement_min": ACTION_AGREEMENT_MIN,
            "selected_slot_agreement_min": SLOT_AGREEMENT_MIN,
            "selected_slot_gate_min_n": SLOT_GATE_MIN_N,
            "material_true_reordering_allowance": MATERIAL_REORDERING_ALLOWANCE,
        },
        "feature_families": FAMILY_FEATURES,
        "family_eligibility": FAMILY_ELIGIBILITY,
        "probe_eligibility": PROBE_ELIGIBILITY,
        "input_inventory": inventory,
        "highest_allowed_claim": (
            "clean-window transfer certification pass/fail for governed "
            "offline hill-climbing readiness; never paper readiness"
        ),
        "forbidden_actions": {
            "model_training": False,
            "threshold_tuning": False,
            "broker_api_calls": False,
            "paper_submit": False,
            "paid_downloads": False,
            "promotion_or_default_changes": False,
            "runtime_or_launchd_changes": False,
            "real_money_changes": False,
        },
    }


def preregister(out_dir: Path, *, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preregistration.json"
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to replace")
    payload = preregistration_payload()
    write_json(path, payload)
    write_json(
        out_dir / "preregistration_hash.json",
        {"path": str(path), "sha256": sha256_path(path)},
    )
    write_json(
        out_dir / "progress.json",
        {
            "status": "preregistered",
            "preregistration_sha256": sha256_path(path),
            "comparisons_executed": False,
        },
    )
    print(json.dumps({"status": "preregistered", "sha256": sha256_path(path)}, indent=2))


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_trace_payloads(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        payload = row.get("payload") or row
        timestamp = pd.Timestamp(payload.get("decision_ts") or row.get("decision_ts"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        minute = timestamp.tz_convert(NY).strftime("%H:%M")
        rows[minute] = payload
    return rows


def recovered_decision_minutes(session: str) -> set[str]:
    quality = json.loads(quality_path(session).read_text())
    recovered = (quality.get("evidence") or {}).get("recovered_checkpoint_minutes") or []
    return {
        (pd.Timestamp(value) + pd.Timedelta(minutes=1)).strftime("%H:%M")
        for value in recovered
    }


def time_bucket(minute: str) -> str:
    for name, (start, end) in TIME_BUCKETS.items():
        if start <= minute < end:
            return name
    return "outside"


def trace_minute_quality(session: str) -> pd.DataFrame:
    rows = read_trace_payloads(trace_path(session, "ibkr_capture"))
    recovered = recovered_decision_minutes(session)
    records = []
    for minute, payload in sorted(rows.items()):
        slots = payload.get("candidate_filter_trace")
        slot_count = len(slots) if isinstance(slots, list) else 0
        quote_complete = slot_count == 42
        live_market_data = slot_count == 42
        max_quote_age = 0.0
        if quote_complete:
            for slot in slots:
                bid = finite(slot.get("bid"))
                ask = finite(slot.get("ask"))
                mid = finite(slot.get("mid"))
                age = finite(slot.get("quote_age_ms"))
                quote_complete = quote_complete and all(
                    value is not None and value > 0 for value in (bid, ask, mid)
                )
                quote_complete = quote_complete and age is not None
                if age is not None:
                    max_quote_age = max(max_quote_age, age)
                market_data_name = str(slot.get("market_data_type_name") or "").lower()
                market_data_type = slot.get("market_data_type")
                live_market_data = live_market_data and (
                    market_data_name == "live" or market_data_type == 1
                )
        records.append(
            {
                "session_date": session,
                "decision_minute_et": minute,
                "trace_slot_count": slot_count,
                "quotes_complete": bool(quote_complete),
                "live_market_data": bool(live_market_data),
                "max_quote_age_ms": max_quote_age if slot_count else np.nan,
                "quote_fresh": bool(
                    quote_complete and max_quote_age <= QUOTE_AGE_MAX_MS
                ),
                "recovered_checkpoint": minute in recovered,
                "raw_checkpoint": minute not in recovered,
                "opening_context_ready": bool(payload.get("opening_context_ready")),
            }
        )
    return pd.DataFrame(records)


def apply_window_policy(minute: pd.DataFrame) -> pd.DataFrame:
    """Apply causal quarantine, warm-up, and minimum-window rules."""
    out = minute.copy()
    out["_decision_ts"] = pd.to_datetime(
        out["session_date"].astype(str)
        + " "
        + out["decision_minute_et"].astype(str),
        format="%Y-%m-%d %H:%M",
    )
    out = out.sort_values(["session_date", "_decision_ts"], kind="stable").reset_index(
        drop=True
    )
    out["healthy_streak"] = 0
    out["option_segment_id"] = 0
    out["option_preliminary_eligible"] = False
    out["context_prefix_clean"] = False

    for _session, indices in out.groupby("session_date", sort=False).groups.items():
        positions = list(indices)
        streak = 0
        segment = 0
        prefix_clean = True
        previous_ts: pd.Timestamp | None = None
        for position in positions:
            timestamp = out.at[position, "_decision_ts"]
            if time_bucket(str(out.at[position, "decision_minute_et"])) == "outside":
                # The 09:31 row is causal history for the first decision, not a
                # decision minute. It must not start warm-up or poison context.
                out.at[position, "healthy_streak"] = 0
                out.at[position, "option_segment_id"] = segment
                out.at[position, "option_preliminary_eligible"] = False
                out.at[position, "context_prefix_clean"] = False
                previous_ts = None
                continue
            consecutive = (
                previous_ts is not None
                and timestamp - previous_ts == pd.Timedelta(minutes=1)
            )
            base_healthy = bool(out.at[position, "base_healthy"])
            if not base_healthy or (previous_ts is not None and not consecutive):
                streak = 0
                segment += 1
            if base_healthy:
                streak = streak + 1 if consecutive else 1
            prefix_clean = (
                prefix_clean
                and base_healthy
                and bool(out.at[position, "opening_context_ready"])
            )
            out.at[position, "healthy_streak"] = streak
            out.at[position, "option_segment_id"] = segment
            out.at[position, "option_preliminary_eligible"] = (
                base_healthy and streak > WARMUP_MINUTES
            )
            out.at[position, "context_prefix_clean"] = prefix_clean
            previous_ts = timestamp

    preliminary = out[out["option_preliminary_eligible"]]
    sizes = preliminary.groupby(
        ["session_date", "option_segment_id"], sort=False
    ).size()
    qualified = {
        key for key, size in sizes.items() if int(size) >= MIN_WINDOW_MINUTES
    }
    out["option_eligible"] = [
        bool(preliminary_flag and (session, segment) in qualified)
        for session, segment, preliminary_flag in zip(
            out["session_date"],
            out["option_segment_id"],
            out["option_preliminary_eligible"],
            strict=True,
        )
    ]

    context_candidates = out[out["context_prefix_clean"]]
    context_counts = context_candidates.groupby("session_date").size()
    context_qualified = {
        session
        for session, count in context_counts.items()
        if int(count) >= MIN_WINDOW_MINUTES
    }
    out["context_eligible"] = [
        bool(clean and session in context_qualified)
        for session, clean in zip(
            out["session_date"], out["context_prefix_clean"], strict=True
        )
    ]
    out["time_bucket"] = out["decision_minute_et"].map(time_bucket)
    out["warmup_remaining"] = np.maximum(
        0, WARMUP_MINUTES + 1 - pd.to_numeric(out["healthy_streak"])
    )
    return out.drop(columns=["_decision_ts"])


def build_frames() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    args = SimpleNamespace(
        l0_l2_dir=battery.l1l3.DEFAULT_L0_L2_DIR,
        group1_dir=battery.l1l3.DEFAULT_GROUP1_DIR,
        group1_definition=battery.l1l3.DEFAULT_GROUP1_DEFINITION,
    )
    frames = []
    admitted_reference: list[str] | None = None
    readiness: dict[str, Any] = {"sessions": {}, "blockers": []}
    for session in SESSIONS:
        prefix = TRACE_PREFIX_BY_SESSION[session]
        frame, admitted, status = battery.build_frame(args, prefix, (session,))
        readiness["sessions"][session] = status
        if status.get("blockers"):
            readiness["blockers"].extend(
                f"{session}:{blocker}" for blocker in status["blockers"]
            )
        if admitted_reference is None:
            admitted_reference = admitted
        elif admitted != admitted_reference:
            readiness["blockers"].append(f"{session}:admitted_feature_mismatch")
        if not frame.empty:
            frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        admitted_reference or [],
        readiness,
    )


def minute_manifest(eval_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        eval_frame.groupby(["session_date", "decision_minute_et"], as_index=False)
        .agg(
            paired_slot_count=("pair_slot_id", "size"),
            spx_historical=("A.context.spx_for_ladder_historical", "median"),
            spx_ibkr=("A.context.spx_for_ladder_ibkr", "median"),
            atm_historical=("A.context.atm_strike_historical", "median"),
            atm_ibkr=("A.context.atm_strike_ibkr", "median"),
        )
    )
    grouped["spx_abs_drift_points"] = (
        grouped["spx_historical"] - grouped["spx_ibkr"]
    ).abs()
    grouped["atm_aligned"] = grouped["atm_historical"] == grouped["atm_ibkr"]
    quality = pd.concat(
        [trace_minute_quality(session) for session in SESSIONS], ignore_index=True
    )
    minute = quality.merge(
        grouped, on=["session_date", "decision_minute_et"], how="left"
    )
    minute["paired_slot_count"] = minute["paired_slot_count"].fillna(0).astype(int)
    minute["pair_complete"] = minute["paired_slot_count"] == 42
    minute["spx_aligned"] = (
        minute["spx_abs_drift_points"].notna()
        & (minute["spx_abs_drift_points"] <= SPX_DRIFT_MAX_POINTS)
    )
    minute["base_healthy"] = (
        minute["raw_checkpoint"]
        & minute["pair_complete"]
        & minute["quotes_complete"]
        & minute["quote_fresh"]
        & minute["live_market_data"]
        & minute["atm_aligned"]
        & minute["spx_aligned"]
    )
    reasons = []
    for row in minute.itertuples(index=False):
        failed = []
        for column, label in (
            ("raw_checkpoint", "recovered"),
            ("pair_complete", "incomplete_pair"),
            ("quotes_complete", "incomplete_quotes"),
            ("quote_fresh", "stale_quote"),
            ("live_market_data", "not_live"),
            ("atm_aligned", "atm_mismatch"),
            ("spx_aligned", "spx_drift"),
        ):
            if not bool(getattr(row, column)):
                failed.append(label)
        reasons.append(",".join(failed))
    minute["quarantine_reasons"] = reasons
    return apply_window_policy(minute)


def window_manifest(minute: pd.DataFrame) -> pd.DataFrame:
    records = []
    for family, eligibility in (
        ("option_path", "option_eligible"),
        ("index_context", "context_eligible"),
    ):
        eligible = minute[minute[eligibility]].copy()
        if eligible.empty:
            continue
        eligible["_ts"] = pd.to_datetime(
            eligible["session_date"] + " " + eligible["decision_minute_et"]
        )
        for session, group in eligible.groupby("session_date", sort=True):
            group = group.sort_values("_ts")
            split = group["_ts"].diff().ne(pd.Timedelta(minutes=1)).cumsum()
            for index, window in group.groupby(split, sort=True):
                bucket_counts = (
                    window["time_bucket"].value_counts().sort_index().to_dict()
                )
                records.append(
                    {
                        "family": family,
                        "session_date": session,
                        "window_id": f"{family}:{session}:{int(index)}",
                        "start_minute_et": window.iloc[0]["decision_minute_et"],
                        "end_minute_et": window.iloc[-1]["decision_minute_et"],
                        "eligible_minutes": len(window),
                        "time_bucket_counts": json.dumps(bucket_counts, sort_keys=True),
                    }
                )
    return pd.DataFrame(records)


def allowed_keys(minute: pd.DataFrame, eligibility: str) -> set[tuple[str, str]]:
    rows = minute.loc[
        minute[eligibility], ["session_date", "decision_minute_et"]
    ].astype(str)
    return set(map(tuple, rows.itertuples(index=False, name=None)))


def frame_for_keys(
    frame: pd.DataFrame, keys: set[tuple[str, str]]
) -> pd.DataFrame:
    row_keys = list(
        zip(
            frame["session_date"].astype(str),
            frame["decision_minute_et"].astype(str),
            strict=True,
        )
    )
    return frame[[key in keys for key in row_keys]].copy()


def coverage_summary(minute: pd.DataFrame, eligibility: str) -> dict[str, Any]:
    eligible = minute[minute[eligibility]]
    per_date = eligible.groupby("session_date").size().astype(int).to_dict()
    per_bucket = eligible.groupby("time_bucket").size().astype(int).to_dict()
    total = int(len(eligible))
    dates = len(per_date)
    max_share = max(per_date.values(), default=0) / total if total else 1.0
    failures = []
    if dates < MIN_DATES:
        failures.append(f"dates:{dates}<{MIN_DATES}")
    if total < MIN_TOTAL_MINUTES:
        failures.append(f"minutes:{total}<{MIN_TOTAL_MINUTES}")
    for bucket in TIME_BUCKETS:
        count = int(per_bucket.get(bucket, 0))
        if count < MIN_TIME_BUCKET_MINUTES:
            failures.append(f"{bucket}:{count}<{MIN_TIME_BUCKET_MINUTES}")
    if max_share > MAX_DATE_SHARE:
        failures.append(f"max_date_share:{max_share:.4f}>{MAX_DATE_SHARE:.4f}")
    return {
        "eligibility": eligibility,
        "total_minutes": total,
        "distinct_dates": dates,
        "per_date": per_date,
        "per_time_bucket": per_bucket,
        "maximum_date_share": max_share,
        "pass": not failures,
        "failures": failures,
    }


def l0_by_family(
    eval_frame: pd.DataFrame, minute: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tables = []
    summaries = []
    for family, features in FAMILY_FEATURES.items():
        eligibility = FAMILY_ELIGIBILITY[family]
        subset = frame_for_keys(eval_frame, allowed_keys(minute, eligibility))
        if subset.empty:
            summaries.append(
                {"family": family, "eligibility": eligibility, "pass": False, "failures": ["no_rows"]}
            )
            continue
        table, summary = battery.compact_l0(subset, features)
        table.insert(0, "family", family)
        tables.append(table)
        summaries.append(
            {
                "family": family,
                "eligibility": eligibility,
                "slot_rows": len(subset),
                "pass": not summary["failing"],
                **summary,
            }
        )
    return (
        pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(),
        summaries,
    )


def l2_family_auc(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    if frame.empty or frame["session_date"].nunique() < MIN_DATES:
        return {
            "status": "insufficient_dates",
            "dates": int(frame["session_date"].nunique()) if not frame.empty else 0,
            "pass": False,
        }
    historical = frame[[f"{feature}_historical" for feature in features]].apply(
        pd.to_numeric, errors="coerce"
    )
    ibkr = frame[[f"{feature}_ibkr" for feature in features]].apply(
        pd.to_numeric, errors="coerce"
    )
    historical.columns = features
    ibkr.columns = features
    X = pd.concat([historical, ibkr], ignore_index=True).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    y = np.concatenate(
        [np.zeros(len(historical), dtype=int), np.ones(len(ibkr), dtype=int)]
    )
    groups = np.concatenate(
        [
            frame["session_date"].astype(str).to_numpy(),
            frame["session_date"].astype(str).to_numpy(),
        ]
    )
    logo = LeaveOneGroupOut()
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=101),
        ),
        "bounded_hgb": HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, random_state=101
        ),
    }
    result: dict[str, Any] = {
        "status": "complete",
        "dates": int(frame["session_date"].nunique()),
        "rows_per_plane": len(frame),
        "folds": {},
    }
    all_pass = True
    for name, model in models.items():
        predictions = np.full(len(y), np.nan)
        fold_rows = []
        for train, test in logo.split(X, y, groups):
            model.fit(X.iloc[train], y[train])
            score = model.predict_proba(X.iloc[test])[:, 1]
            predictions[test] = score
            auc = float(roc_auc_score(y[test], score))
            fold_rows.append(
                {"held_out_date": str(groups[test][0]), "auc": auc, "rows": len(test)}
            )
        pooled = float(roc_auc_score(y, predictions))
        result[f"{name}_pooled_auc"] = pooled
        result["folds"][name] = fold_rows
        all_pass = all_pass and pooled <= L2_AUC_MAX
    result["pass"] = bool(all_pass)
    return result


def l2_by_family(
    eval_frame: pd.DataFrame, minute: pd.DataFrame
) -> list[dict[str, Any]]:
    rows = []
    for family, features in FAMILY_FEATURES.items():
        eligibility = FAMILY_ELIGIBILITY[family]
        subset = frame_for_keys(eval_frame, allowed_keys(minute, eligibility))
        rows.append(
            {
                "family": family,
                "eligibility": eligibility,
                **l2_family_auc(subset, features),
            }
        )
    return rows


def build_frozen_decisions(eval_frame: pd.DataFrame, admitted: list[str]) -> pd.DataFrame:
    args = SimpleNamespace(
        l0_l2_dir=battery.l1l3.DEFAULT_L0_L2_DIR,
        group1_dir=battery.l1l3.DEFAULT_GROUP1_DIR,
        group1_definition=battery.l1l3.DEFAULT_GROUP1_DEFINITION,
    )
    contract = json.loads(battery.FROZEN["contract"][0].read_text())
    l1_summary = json.loads((battery.ATTEMPT001_DIR / "l1_summary.json").read_text())
    thresholds = battery.frozen_l1_thresholds(contract, l1_summary)
    l3_reference = battery.build_l3_reference(
        contract, battery.ATTEMPT001_DIR / "l3_transfer_probe_results.csv"
    )
    train_frame, _features, readiness = battery.build_frame(
        args, battery.BURNED_PREFIX, battery.BURNED
    )
    if readiness.get("blockers"):
        raise RuntimeError(f"burned-frame blockers: {readiness['blockers']}")
    population = battery.build_populations_cross_day(
        eval_frame=eval_frame,
        train_frame=train_frame,
        admitted=admitted,
        l1_thresholds=thresholds,
        previous_l3=l3_reference,
        epsilon=contract["frozen_epsilons"],
    )
    decisions = v14.select_decisions(
        population, k=2.0, fallback_rule="score_independent_nearest_atm"
    )
    return v14.enrich_disagreement_classes(v14.merge_plane_decisions(decisions))


def decision_results(
    merged: pd.DataFrame, minute: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rows = []
    per_date_rows = []
    failures = []
    for probe, eligibility in PROBE_ELIGIBILITY.items():
        keys = allowed_keys(minute, eligibility)
        subset = frame_for_keys(merged, keys)
        subset = subset[subset["probe"] == probe]
        if subset.empty:
            failures.append(f"{probe}:no_eligible_rows")
            continue
        summary = v14.summarize_all(subset)
        row = summary.iloc[0].to_dict()
        row["eligibility"] = eligibility
        rows.append(row)
        action_agreement = float(row["action_agreement"])
        if action_agreement < ACTION_AGREEMENT_MIN:
            failures.append(
                f"{probe}:pooled_action:{action_agreement:.4f}<{ACTION_AGREEMENT_MIN:.4f}"
            )
        mutually_confident = int(row["mutually_confident_enter_n"])
        slot_agreement = float(
            row["selected_slot_agreement_mutually_confident_enter"]
        )
        if mutually_confident >= SLOT_GATE_MIN_N and slot_agreement < SLOT_AGREEMENT_MIN:
            failures.append(
                f"{probe}:slot:{slot_agreement:.4f}<{SLOT_AGREEMENT_MIN:.4f}"
            )
        material = int(
            (
                (subset["true_score_reordering"] == True)  # noqa: E712
                & (subset["economically_material"] == True)  # noqa: E712
            ).sum()
        )
        if material > MATERIAL_REORDERING_ALLOWANCE:
            failures.append(
                f"{probe}:material_reorderings:{material}>"
                f"{MATERIAL_REORDERING_ALLOWANCE}"
            )
        for session, group in subset.groupby("session_date", sort=True):
            agreement = float(group["action_agree"].mean())
            per_date_rows.append(
                {
                    "probe": probe,
                    "eligibility": eligibility,
                    "session_date": session,
                    "minutes": len(group),
                    "action_agreement": agreement,
                    "gating": len(group) >= 30,
                    "pass": len(group) < 30 or agreement >= ACTION_AGREEMENT_MIN,
                }
            )
            if len(group) >= 30 and agreement < ACTION_AGREEMENT_MIN:
                failures.append(
                    f"{probe}:{session}:action:{agreement:.4f}<"
                    f"{ACTION_AGREEMENT_MIN:.4f}"
                )
    return pd.DataFrame(rows), pd.DataFrame(per_date_rows), sorted(set(failures))


def run(out_dir: Path) -> None:
    prereg_path = out_dir / "preregistration.json"
    prereg_hash_path = out_dir / "preregistration_hash.json"
    if not prereg_path.exists() or not prereg_hash_path.exists():
        raise SystemExit("preregistration missing; run --mode preregister first")
    pinned = json.loads(prereg_hash_path.read_text())
    if sha256_path(prereg_path) != pinned["sha256"]:
        raise SystemExit("preregistration hash mismatch")
    progress = {
        "status": "running",
        "preregistration_sha256": pinned["sha256"],
        "model_training_executed": False,
        "threshold_tuning_executed": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
    }
    write_json(out_dir / "progress.json", progress)

    eval_frame, admitted, readiness = build_frames()
    if readiness["blockers"]:
        routing = {
            "routing_decision": "clean_window_insufficient_artifacts",
            "confident_stop_recorder_and_begin_hill_climbing": False,
            "blockers": readiness["blockers"],
        }
        write_json(out_dir / "routing_decision.json", routing)
        return

    minute = minute_manifest(eval_frame)
    minute.to_csv(out_dir / "minute_eligibility.csv", index=False)
    windows = window_manifest(minute)
    windows.to_csv(out_dir / "clean_window_manifest.csv", index=False)
    coverage = {
        eligibility: coverage_summary(minute, eligibility)
        for eligibility in ("option_eligible", "context_eligible")
    }
    write_json(out_dir / "coverage_summary.json", coverage)

    l0_table, l0_summaries = l0_by_family(eval_frame, minute)
    l0_table.to_csv(out_dir / "l0_feature_family_results.csv", index=False)
    write_json(out_dir / "l0_feature_family_summary.json", l0_summaries)

    l2_summaries = l2_by_family(eval_frame, minute)
    write_json(out_dir / "l2_feature_family_summary.json", l2_summaries)

    merged = build_frozen_decisions(eval_frame, admitted)
    merged.to_parquet(out_dir / "frozen_decisions_merged.parquet", index=False)
    decision_table, per_date, decision_failures = decision_results(merged, minute)
    decision_table.to_csv(out_dir / "decision_transfer_results.csv", index=False)
    per_date.to_csv(out_dir / "decision_transfer_by_date.csv", index=False)

    failures = []
    for eligibility, item in coverage.items():
        failures.extend(f"coverage:{eligibility}:{value}" for value in item["failures"])
    for item in l0_summaries:
        if not item["pass"]:
            failures.extend(
                f"l0:{item['family']}:{value}"
                for value in item.get("failing", item.get("failures", []))
            )
    for item in l2_summaries:
        if not item["pass"]:
            failures.append(f"l2:{item['family']}:{item['status']}")
    failures.extend(f"decision:{value}" for value in decision_failures)

    coverage_failures = [item for item in failures if item.startswith("coverage:")]
    if not failures:
        route = "clean_window_certification_pass_hill_climbing_unblocked"
        confident = True
    elif len(coverage_failures) == len(failures):
        route = "clean_window_insufficient_evidence"
        confident = False
    else:
        route = "clean_window_transfer_failed_repair_before_hill_climbing"
        confident = False
    routing = {
        "schema_version": "Protocol101CleanWindowCertificationRoutingV1",
        "routing_decision": route,
        "confident_stop_recorder_and_begin_hill_climbing": confident,
        "failures": failures,
        "coverage": coverage,
        "l0_feature_families": l0_summaries,
        "l2_feature_families": l2_summaries,
        "decision_failure_count": len(decision_failures),
        "decision_failures": decision_failures,
        "preregistration_sha256": pinned["sha256"],
        "frozen_contract_sha256": battery.FROZEN["contract"][1],
        "no_paper_readiness_claim": True,
        "side_effects": {
            "model_training_executed": False,
            "threshold_tuning_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_download": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "real_money_path_changed": False,
        },
    }
    write_json(out_dir / "routing_decision.json", routing)

    lines = [
        "# Protocol101 Clean-Window Certification",
        "",
        f"- Routing: `{route}`",
        f"- Confident recorder-stop / governed-hill-climb answer: `{confident}`",
        f"- Option-window minutes: `{coverage['option_eligible']['total_minutes']}` "
        f"across `{coverage['option_eligible']['distinct_dates']}` dates",
        f"- Context-window minutes: `{coverage['context_eligible']['total_minutes']}` "
        f"across `{coverage['context_eligible']['distinct_dates']}` dates",
        f"- Failures: `{len(failures)}`",
        "",
        "## Coverage",
        "",
    ]
    for eligibility, item in coverage.items():
        lines.append(
            f"- `{eligibility}`: pass=`{item['pass']}`, per-date="
            f"`{item['per_date']}`, buckets=`{item['per_time_bucket']}`"
        )
    lines.extend(["", "## Feature families", ""])
    for item in l0_summaries:
        lines.append(
            f"- L0 `{item['family']}`: pass=`{item['pass']}`, "
            f"failing=`{item.get('failing', item.get('failures', []))}`"
        )
    for item in l2_summaries:
        lines.append(
            f"- L2 `{item['family']}`: pass=`{item['pass']}`, "
            f"status=`{item['status']}`"
        )
    lines.extend(
        [
            "",
            "## Decision transfer",
            "",
            f"- Decision failures: `{decision_failures}`",
            "",
            "This is a clean-window hill-climbing-readiness result, not paper readiness.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    progress["status"] = "complete"
    progress["routing_decision"] = route
    write_json(out_dir / "progress.json", progress)
    print(
        json.dumps(
            {
                "routing_decision": route,
                "confident_stop_recorder_and_begin_hill_climbing": confident,
                "option_minutes": coverage["option_eligible"]["total_minutes"],
                "context_minutes": coverage["context_eligible"]["total_minutes"],
                "failure_count": len(failures),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.mode == "preregister":
        preregister(args.out_dir, force=args.force)
        return
    run(args.out_dir)


if __name__ == "__main__":
    main()
