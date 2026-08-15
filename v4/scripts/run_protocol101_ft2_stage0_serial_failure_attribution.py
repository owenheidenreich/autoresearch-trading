"""Frozen Stage-0 P5/HGB serial failure accounting attribution."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ATTEMPT002 = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_ft2_stage0_p5_hold_exit_feasibility_attempt002"
)
OUTPUT_DIR = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_ft2_stage0_serial_failure_attribution_attempt001"
)
SIMULATOR_PATH = ROOT / "v4/model/protocol101_serial_simulator_v5.py"
GOAL_PATH = (
    ROOT
    / "v4/docs/protocol101/training/history/"
    "closed_stage1_graph_and_goals_2026_07_28/goals/"
    "PROTOCOL101_FT2_STAGE0_SERIAL_FAILURE_ATTRIBUTION_GOAL_2026_07_28.md"
)
GOAL_SHA256 = "b86521454fe51f6143cc795bd063a01c4ead641536aa8c8f350310935c14d1f8"
EVIDENCE_GRADE = "analysis_only_non_promotable"
EXPECTED_TERMINAL_DECISION = "stop_no_preliminary_exit_signal"
EXPECTED_DELTA = -4_190.0
IDENTITY_COLUMNS = (
    "session",
    "decision_time_ns",
    "contract_id",
    "canonical_strike_slot",
)
EXIT_REASON_NAMES = {
    0: "invalid",
    1: "stop_loss",
    2: "take_profit",
    3: "max_hold",
    4: "forced_flat",
    5: "no_bid_stop",
}
NY = ZoneInfo("America/New_York")


class AttributionBlocked(RuntimeError):
    """Raised when frozen evidence cannot support the requested attribution."""

    def __init__(self, route: str, message: str) -> None:
        super().__init__(message)
        self.route = route


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_checksum_manifest(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise AttributionBlocked(
                "attribution_blocked_artifact_mismatch",
                f"invalid checksum manifest line {line_number}",
            )
        records.append((parts[0], parts[1].lstrip("*")))
    if not records:
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            "attempt002 checksum manifest is empty",
        )
    return records


def validate_checksum_manifest(
    manifest_path: Path, base_dir: Path
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for expected, relative_name in parse_checksum_manifest(manifest_path):
        target = base_dir / relative_name
        actual = sha256_file(target) if target.is_file() else None
        results.append(
            {
                "path": relative_name,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "present": target.is_file(),
                "matches": actual == expected,
            }
        )
    failures = [item for item in results if not item["matches"]]
    if failures:
        names = ", ".join(item["path"] for item in failures)
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            f"attempt002 checksum validation failed: {names}",
        )
    return results


def simulator_fee_default(path: Path) -> float:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "SerialSimulatorV5Config":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if not isinstance(item.target, ast.Name):
                continue
            if item.target.id != "campaign_round_trip_fee_dollars":
                continue
            value = ast.literal_eval(item.value)
            return float(value)
    raise AttributionBlocked(
        "attribution_blocked_artifact_mismatch",
        "simulator-v5 fee default not found",
    )


def identity_tuple(row: Any, slot_column: str = "canonical_strike_slot") -> tuple[Any, ...]:
    return (
        str(row.session),
        int(row.decision_time_ns),
        str(row.contract_id),
        int(getattr(row, slot_column)),
    )


def identity_dict(identity: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(IDENTITY_COLUMNS, identity, strict=True))


def ensure_unique_identities(
    frame: pd.DataFrame,
    *,
    label: str,
    slot_column: str = "canonical_strike_slot",
) -> None:
    duplicate_mask = frame.duplicated(
        ["session", "decision_time_ns", "contract_id", slot_column],
        keep=False,
    )
    if duplicate_mask.any():
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            f"duplicate {label} identities: {int(duplicate_mask.sum())} rows",
        )


def minute_of_day_et(timestamp_ns: int) -> int:
    dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=NY)
    return dt.hour * 60 + dt.minute


def time_of_day_label(timestamp_ns: int) -> str:
    minute = minute_of_day_et(timestamp_ns)
    if minute < 10 * 60 + 30:
        return "09:32-10:29_ET"
    if minute < 14 * 60:
        return "10:30-13:59_ET"
    return "14:00-15:30_ET"


def number_summary(values: Iterable[float]) -> dict[str, Any]:
    series = pd.Series(list(values), dtype="float64").dropna()
    if series.empty:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "max": None,
            "sum": 0.0,
        }
    return {
        "count": int(series.size),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
        "sum": float(series.sum()),
    }


def money_sum(values: Iterable[float]) -> float:
    cents = sum(int(round(float(value) * 100.0)) for value in values)
    return cents / 100.0


def exact_frequency(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return dict(sorted(counts.items()))


def route_from_contributions(
    common_exit_contribution: float,
    entry_stream_contribution: float,
) -> str:
    if common_exit_contribution >= 0 and entry_stream_contribution < 0:
        return "entry_timing_and_reentry_is_binding"
    if common_exit_contribution < 0 and entry_stream_contribution >= 0:
        return "one_step_exit_target_is_binding"
    if common_exit_contribution < 0 and entry_stream_contribution < 0:
        return "both_entry_and_exit_are_binding"
    return "attribution_inconclusive"


def classify_skipped_intents(
    intents: pd.DataFrame,
    admitted: pd.DataFrame,
    *,
    daily_loss_fraction: float,
) -> pd.DataFrame:
    admitted_ids = {
        identity_tuple(row) for row in admitted.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for session, session_intents in intents.groupby("session", sort=True):
        trades = admitted[admitted["session"].eq(session)].sort_values(
            "decision_time_ns"
        )
        stop_time: int | None = None
        if not trades.empty:
            start_equity = float(trades.iloc[0]["session_start_equity"])
            crossed = trades[
                trades["session_realized_pnl_after"]
                <= -(daily_loss_fraction * start_equity)
            ]
            if not crossed.empty:
                stop_time = int(crossed.iloc[0]["label_realized_exit_time_ns"])
        occupancy = [
            (
                int(row.decision_time_ns),
                int(row.label_realized_exit_time_ns),
            )
            for row in trades.itertuples(index=False)
        ]
        for row in session_intents.sort_values("decision_time_ns").itertuples(
            index=False
        ):
            identity = identity_tuple(row, "canonical_slot")
            if identity in admitted_ids:
                continue
            decision_time = int(row.decision_time_ns)
            if stop_time is not None and decision_time >= stop_time:
                reason = "daily_loss_stop"
            elif any(start < decision_time < end for start, end in occupancy):
                reason = "overlap"
            elif any(decision_time == start for start, _ in occupancy):
                reason = "admitted_identity_mismatch"
            else:
                reason = "unclassified"
            records.append(
                {
                    **identity_dict(identity),
                    "reason": reason,
                    "time_of_day_et": time_of_day_label(decision_time),
                }
            )
    return pd.DataFrame.from_records(records)


def fee_equation_error(frame: pd.DataFrame, fee: float) -> pd.Series:
    expected = (
        (
            frame["label_executable_exit_bid"].astype(float)
            - frame["entry_ask"].astype(float)
        )
        * 100.0
        - fee
    )
    return (frame["stressed_pnl"].astype(float) - expected).abs()


def prepare_inputs() -> dict[str, Any]:
    required = {
        "decision": ATTEMPT002 / "decision.json",
        "pilot_results": ATTEMPT002 / "pilot_results.json",
        "serial_replays": ATTEMPT002 / "serial_replays.parquet",
        "serial_receipts": ATTEMPT002 / "serial_identity_receipts.json",
        "episode_replays": ATTEMPT002 / "episode_replays.parquet",
        "predictions": ATTEMPT002 / "predictions.parquet",
        "checksums": ATTEMPT002 / "hashes.sha256",
        "simulator": SIMULATOR_PATH,
        "goal": GOAL_PATH,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            f"required frozen inputs missing: {missing}",
        )
    if sha256_file(GOAL_PATH) != GOAL_SHA256:
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            "Goal hash does not match owner-supplied SHA-256",
        )
    checksum_results = validate_checksum_manifest(
        required["checksums"], ATTEMPT002
    )
    decision = json.loads(required["decision"].read_text(encoding="utf-8"))
    pilot = json.loads(required["pilot_results"].read_text(encoding="utf-8"))
    receipts = json.loads(
        required["serial_receipts"].read_text(encoding="utf-8")
    )
    serial = pd.read_parquet(required["serial_replays"])
    episodes = pd.read_parquet(required["episode_replays"])
    predictions = pd.read_parquet(required["predictions"])
    return {
        "required": required,
        "checksum_results": checksum_results,
        "decision": decision,
        "pilot": pilot,
        "receipts": receipts,
        "serial": serial,
        "episodes": episodes,
        "prediction_schema": {
            "rows": int(len(predictions)),
            "columns": list(predictions.columns),
        },
        "fee": simulator_fee_default(SIMULATOR_PATH),
    }


def verify_frozen_result(inputs: dict[str, Any]) -> dict[str, Any]:
    decision = inputs["decision"]
    pilot = inputs["pilot"]
    receipts = inputs["receipts"]
    serial = inputs["serial"]
    episodes = inputs["episodes"]
    fee = float(inputs["fee"])
    if decision.get("terminal_decision") != EXPECTED_TERMINAL_DECISION:
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            "attempt002 terminal decision does not match frozen requirement",
        )
    selected = serial[serial["comparator"].isin(["original_p5", "real_hgb"])]
    expected_columns = {
        *IDENTITY_COLUMNS,
        "comparator",
        "stressed_pnl",
        "entry_ask",
        "premium_at_risk",
        "premium_plus_fee_required",
        "label_realized_exit_time_ns",
        "label_source_exit_quote_time_ns",
        "label_executable_exit_bid",
        "label_exit_reason_code",
        "cash_after",
        "session_start_equity",
        "session_realized_pnl_after",
    }
    absent = sorted(expected_columns - set(selected.columns))
    if absent:
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            f"serial replay fields missing: {absent}",
        )
    p5 = selected[selected["comparator"].eq("original_p5")].copy()
    hgb = selected[selected["comparator"].eq("real_hgb")].copy()
    ensure_unique_identities(p5, label="P5 admitted trade")
    ensure_unique_identities(hgb, label="HGB admitted trade")
    p5_pnl = money_sum(p5["stressed_pnl"])
    hgb_pnl = money_sum(hgb["stressed_pnl"])
    delta = hgb_pnl - p5_pnl
    frozen_delta = float(
        pilot["decision_detail"]["ledger_B_real_lifts"]["original_p5"]
    )
    decision_delta = float(
        decision["decision_detail"]["ledger_B_real_lifts"]["original_p5"]
    )
    if not (
        math.isclose(delta, EXPECTED_DELTA, abs_tol=1e-9)
        and math.isclose(delta, frozen_delta, abs_tol=1e-9)
        and math.isclose(delta, decision_delta, abs_tol=1e-9)
    ):
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            f"frozen Ledger-B delta mismatch: {delta}",
        )
    if p5["session"].nunique() != 5 or hgb["session"].nunique() != 5:
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            "expected exactly five validation sessions",
        )
    validation = episodes[episodes["role"].eq("nested_validation")].copy()
    p5_intents = validation[
        validation["comparator"].eq("original_p5")
    ].sort_values(["session", "decision_time_ns"])
    hgb_intents = validation[
        validation["comparator"].eq("real_hgb")
    ].sort_values(["session", "decision_time_ns"])
    ensure_unique_identities(
        p5_intents, label="P5 ordered intent", slot_column="canonical_slot"
    )
    ensure_unique_identities(
        hgb_intents, label="HGB ordered intent", slot_column="canonical_slot"
    )
    p5_order = [
        identity_tuple(row, "canonical_slot")
        for row in p5_intents.itertuples(index=False)
    ]
    hgb_order = [
        identity_tuple(row, "canonical_slot")
        for row in hgb_intents.itertuples(index=False)
    ]
    receipt_p5 = receipts["comparators"]["original_p5"]
    receipt_hgb = receipts["comparators"]["real_hgb"]
    if (
        p5_order != hgb_order
        or len(p5_order) != 1_354
        or receipt_p5["candidate_stream_hash"]
        != receipt_hgb["candidate_stream_hash"]
        or receipt_p5["entry_intents"] != receipt_hgb["entry_intents"]
    ):
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            "ordered P5 intent streams changed between comparators",
        )
    max_fee_error = float(fee_equation_error(selected, fee).max())
    reserve_error = (
        selected["premium_plus_fee_required"]
        - selected["premium_at_risk"]
        - fee
    ).abs()
    if max_fee_error > 1e-9 or float(reserve_error.max()) > 1e-9:
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            "fee accounting does not reproduce simulator-v5 records",
        )
    if (
        selected["label_realized_exit_time_ns"].isna().any()
        or selected["label_source_exit_quote_time_ns"].isna().any()
    ):
        raise AttributionBlocked(
            "attribution_blocked_artifact_mismatch",
            "two-clock exit fields are incomplete",
        )
    input_hashes = {
        key: sha256_file(path)
        for key, path in inputs["required"].items()
    }
    return {
        "status": "pass",
        "evidence_grade": EVIDENCE_GRADE,
        "goal_sha256": input_hashes["goal"],
        "input_sha256": input_hashes,
        "attempt002_checksum_count": len(inputs["checksum_results"]),
        "attempt002_checksums_all_valid": True,
        "attempt002_checksum_results": inputs["checksum_results"],
        "terminal_decision": decision["terminal_decision"],
        "validation_sessions": sorted(p5["session"].unique().tolist()),
        "validation_session_count": int(p5["session"].nunique()),
        "ordered_p5_intent_count": len(p5_order),
        "ordered_p5_intents_unchanged": True,
        "candidate_stream_hash": receipt_p5["candidate_stream_hash"],
        "duplicate_admitted_trade_identities": 0,
        "duplicate_ordered_intent_identities": 0,
        "p5_serial_pnl": p5_pnl,
        "hgb_serial_pnl": hgb_pnl,
        "reproduced_hgb_minus_p5_delta": delta,
        "frozen_hgb_minus_p5_delta": frozen_delta,
        "simulator_v5_fee_dollars": fee,
        "fee_fields_present": True,
        "fee_equation_max_abs_error": max_fee_error,
        "two_clock_fields_present": True,
        "prediction_artifact_schema": inputs["prediction_schema"],
    }


def decomposition_frame(
    p5: pd.DataFrame, hgb: pd.DataFrame, fee: float
) -> pd.DataFrame:
    p5_by_id = {
        identity_tuple(row): row for row in p5.itertuples(index=False)
    }
    hgb_by_id = {
        identity_tuple(row): row for row in hgb.itertuples(index=False)
    }
    all_ids = sorted(set(p5_by_id) | set(hgb_by_id))
    records: list[dict[str, Any]] = []
    for identity in all_ids:
        p5_row = p5_by_id.get(identity)
        hgb_row = hgb_by_id.get(identity)
        if p5_row is not None and hgb_row is not None:
            classification = "common"
        elif p5_row is not None:
            classification = "p5_only"
        else:
            classification = "hgb_only"
        base = p5_row if p5_row is not None else hgb_row
        assert base is not None
        record: dict[str, Any] = {
            **identity_dict(identity),
            "classification": classification,
            "right": str(base.right),
            "time_of_day_et": time_of_day_label(int(base.decision_time_ns)),
            "entry_ask": float(base.entry_ask),
            "premium_at_risk": float(base.premium_at_risk),
            "p5_admitted": p5_row is not None,
            "hgb_admitted": hgb_row is not None,
        }
        for prefix, row in (("p5", p5_row), ("hgb", hgb_row)):
            record[f"{prefix}_exit_time_ns"] = (
                int(row.label_realized_exit_time_ns)
                if row is not None
                else None
            )
            record[f"{prefix}_source_exit_quote_time_ns"] = (
                int(row.label_source_exit_quote_time_ns)
                if row is not None
                else None
            )
            record[f"{prefix}_exit_reason_code"] = (
                int(row.label_exit_reason_code) if row is not None else None
            )
            record[f"{prefix}_exit_reason"] = (
                EXIT_REASON_NAMES.get(int(row.label_exit_reason_code), "unknown")
                if row is not None
                else None
            )
            record[f"{prefix}_holding_minutes"] = (
                (
                    int(row.label_realized_exit_time_ns)
                    - int(row.decision_time_ns)
                )
                / 60_000_000_000
                if row is not None
                else None
            )
            record[f"{prefix}_fee_adjusted_pnl"] = (
                float(row.stressed_pnl) if row is not None else None
            )
            record[f"{prefix}_fee_dollars"] = fee if row is not None else 0.0
        record["common_exit_pnl_delta"] = (
            money_sum([hgb_row.stressed_pnl, -p5_row.stressed_pnl])
            if p5_row is not None and hgb_row is not None
            else 0.0
        )
        record["entry_stream_pnl_component"] = (
            money_sum([hgb_row.stressed_pnl])
            if classification == "hgb_only"
            else money_sum([-p5_row.stressed_pnl])
            if classification == "p5_only"
            else 0.0
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def per_session_attribution(
    sessions: list[str],
    p5: pd.DataFrame,
    hgb: pd.DataFrame,
    decomposition: pd.DataFrame,
    p5_skipped: pd.DataFrame,
    hgb_skipped: pd.DataFrame,
    fee: float,
    daily_loss_fraction: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for session in sessions:
        p5_s = p5[p5["session"].eq(session)].sort_values("decision_time_ns")
        hgb_s = hgb[hgb["session"].eq(session)].sort_values("decision_time_ns")
        d_s = decomposition[decomposition["session"].eq(session)]
        common = d_s[d_s["classification"].eq("common")]
        p5_only = d_s[d_s["classification"].eq("p5_only")]
        hgb_only = d_s[d_s["classification"].eq("hgb_only")]
        p5_first = p5_s.iloc[0]
        hgb_first = hgb_s.iloc[0]
        hgb_gaps = (
            hgb_s["decision_time_ns"].iloc[1:].reset_index(drop=True)
            - hgb_s["label_realized_exit_time_ns"]
            .iloc[:-1]
            .reset_index(drop=True)
        ) / 60_000_000_000
        p5_start = float(p5_s.iloc[0]["session_start_equity"])
        hgb_start = float(hgb_s.iloc[0]["session_start_equity"])
        p5_stop = bool(
            (
                p5_s["session_realized_pnl_after"]
                <= -(daily_loss_fraction * p5_start)
            ).any()
        )
        hgb_stop = bool(
            (
                hgb_s["session_realized_pnl_after"]
                <= -(daily_loss_fraction * hgb_start)
            ).any()
        )
        records.append(
            {
                "session": session,
                "p5_trade_count": int(len(p5_s)),
                "hgb_trade_count": int(len(hgb_s)),
                "common_entry_count": int(len(common)),
                "p5_only_entry_count": int(len(p5_only)),
                "hgb_only_entry_count": int(len(hgb_only)),
                "p5_total_pnl": money_sum(p5_s["stressed_pnl"]),
                "hgb_total_pnl": money_sum(hgb_s["stressed_pnl"]),
                "hgb_minus_p5_total_pnl": money_sum(
                    [*hgb_s["stressed_pnl"], *(-p5_s["stressed_pnl"])]
                ),
                "common_exit_contribution": money_sum(
                    common["common_exit_pnl_delta"]
                ),
                "entry_stream_contribution": money_sum(
                    [
                        *hgb_only["hgb_fee_adjusted_pnl"].dropna(),
                        *(-p5_only["p5_fee_adjusted_pnl"].dropna()),
                    ]
                ),
                "accounting_residual": money_sum(
                    [
                        *hgb_s["stressed_pnl"],
                        *(-p5_s["stressed_pnl"]),
                        *(-common["common_exit_pnl_delta"]),
                        *(-hgb_only["hgb_fee_adjusted_pnl"].dropna()),
                        *p5_only["p5_fee_adjusted_pnl"].dropna(),
                    ]
                ),
                "p5_first_trade_identity": json.dumps(
                    identity_dict(identity_tuple(p5_first)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "hgb_first_trade_identity": json.dumps(
                    identity_dict(identity_tuple(hgb_first)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "p5_first_trade_pnl": float(p5_first["stressed_pnl"]),
                "hgb_first_trade_pnl": float(hgb_first["stressed_pnl"]),
                "p5_second_and_later_pnl": money_sum(
                    p5_s.iloc[1:]["stressed_pnl"]
                ),
                "hgb_second_and_later_pnl": money_sum(
                    hgb_s.iloc[1:]["stressed_pnl"]
                ),
                "p5_mean_holding_minutes": float(
                    (
                        p5_s["label_realized_exit_time_ns"]
                        - p5_s["decision_time_ns"]
                    ).mean()
                    / 60_000_000_000
                ),
                "hgb_mean_holding_minutes": float(
                    (
                        hgb_s["label_realized_exit_time_ns"]
                        - hgb_s["decision_time_ns"]
                    ).mean()
                    / 60_000_000_000
                ),
                "hgb_mean_minutes_exit_to_next_entry": (
                    float(hgb_gaps.mean()) if not hgb_gaps.empty else None
                ),
                "p5_common_fees": float(len(common) * fee),
                "hgb_common_fees": float(len(common) * fee),
                "p5_only_fees": float(len(p5_only) * fee),
                "hgb_only_fees": float(len(hgb_only) * fee),
                "p5_skipped_intents": int(
                    p5_skipped["session"].eq(session).sum()
                ),
                "hgb_skipped_intents": int(
                    hgb_skipped["session"].eq(session).sum()
                ),
                "p5_daily_stop_activated": p5_stop,
                "hgb_daily_stop_activated": hgb_stop,
                "p5_minimum_realized_equity": float(
                    min(p5_start, p5_s["cash_after"].min())
                ),
                "hgb_minimum_realized_equity": float(
                    min(hgb_start, hgb_s["cash_after"].min())
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def comparator_occupancy(
    comparator: str,
    admitted: pd.DataFrame,
    skipped: pd.DataFrame,
    fee: float,
    daily_loss_fraction: float,
) -> dict[str, Any]:
    admitted = admitted.sort_values(["session", "decision_time_ns"]).copy()
    admitted["holding_minutes"] = (
        admitted["label_realized_exit_time_ns"]
        - admitted["decision_time_ns"]
    ) / 60_000_000_000
    gaps: list[float] = []
    session_details: dict[str, Any] = {}
    for session, trades in admitted.groupby("session", sort=True):
        trades = trades.sort_values("decision_time_ns")
        start_equity = float(trades.iloc[0]["session_start_equity"])
        threshold = daily_loss_fraction * start_equity
        crossed = trades[
            trades["session_realized_pnl_after"] <= -threshold
        ]
        activation_time = (
            int(crossed.iloc[0]["label_realized_exit_time_ns"])
            if not crossed.empty
            else None
        )
        local_gaps = (
            trades["decision_time_ns"].iloc[1:].reset_index(drop=True)
            - trades["label_realized_exit_time_ns"]
            .iloc[:-1]
            .reset_index(drop=True)
        ) / 60_000_000_000
        gaps.extend(float(value) for value in local_gaps)
        skipped_session = skipped[skipped["session"].eq(session)]
        session_details[session] = {
            "admitted_trade_count": int(len(trades)),
            "admitted_trade_identities": [
                identity_dict(identity_tuple(row))
                for row in trades.itertuples(index=False)
            ],
            "first_trade_identity": identity_dict(
                identity_tuple(trades.iloc[0])
            ),
            "first_trade_pnl": float(trades.iloc[0]["stressed_pnl"]),
            "first_trade_only_pnl": float(trades.iloc[0]["stressed_pnl"]),
            "second_and_later_pnl": float(
                trades.iloc[1:]["stressed_pnl"].sum()
            ),
            "holding_minutes": [
                float(value) for value in trades["holding_minutes"]
            ],
            "exit_to_next_entry_minutes": [
                float(value) for value in local_gaps
            ],
            "fees_dollars": float(len(trades) * fee),
            "skipped_intent_count": int(len(skipped_session)),
            "skipped_reason_counts": exact_frequency(
                skipped_session["reason"]
            ),
            "skipped_intent_times_ns": [
                int(value)
                for value in skipped_session["decision_time_ns"].tolist()
            ],
            "skipped_intent_time_of_day_counts": exact_frequency(
                skipped_session["time_of_day_et"]
            ),
            "daily_stop_activated": activation_time is not None,
            "daily_stop_activation_time_ns": activation_time,
            "daily_loss_threshold_dollars": float(threshold),
            "minimum_realized_equity": float(
                min(start_equity, trades["cash_after"].min())
            ),
            "exit_action_counts": {
                EXIT_REASON_NAMES.get(int(key), f"unknown_{key}"): int(value)
                for key, value in trades["label_exit_reason_code"]
                .value_counts()
                .sort_index()
                .items()
            },
            "elapsed_minutes_distribution": number_summary(
                trades["holding_minutes"]
            ),
        }
    return {
        "comparator": comparator,
        "admitted_trade_count": int(len(admitted)),
        "total_pnl": money_sum(admitted["stressed_pnl"]),
        "first_trade_only_pnl": money_sum(
            admitted.groupby("session", sort=True).head(1)["stressed_pnl"]
        ),
        "second_and_later_pnl": money_sum(
            admitted.drop(
                admitted.groupby("session", sort=True).head(1).index
            )["stressed_pnl"]
        ),
        "holding_minutes_distribution": number_summary(
            admitted["holding_minutes"]
        ),
        "exit_to_next_entry_minutes_distribution": number_summary(gaps),
        "total_fees_dollars": float(len(admitted) * fee),
        "skipped_intent_count": int(len(skipped)),
        "skipped_reason_counts": exact_frequency(skipped["reason"]),
        "skipped_time_of_day_counts": exact_frequency(
            skipped["time_of_day_et"]
        ),
        "daily_stop_sessions": [
            session
            for session, detail in session_details.items()
            if detail["daily_stop_activated"]
        ],
        "minimum_realized_equity": float(
            min(
                min(detail["minimum_realized_equity"] for detail in session_details.values()),
                admitted.iloc[0]["session_start_equity"],
            )
        ),
        "exit_action_counts": {
            EXIT_REASON_NAMES.get(int(key), f"unknown_{key}"): int(value)
            for key, value in admitted["label_exit_reason_code"]
            .value_counts()
            .sort_index()
            .items()
        },
        "elapsed_minutes_exact_counts": exact_frequency(
            admitted["holding_minutes"]
        ),
        "sessions": session_details,
    }


def side_only_breakdown(
    decomposition: pd.DataFrame, classification: str
) -> dict[str, Any]:
    side = decomposition[decomposition["classification"].eq(classification)]
    pnl_column = (
        "hgb_fee_adjusted_pnl"
        if classification == "hgb_only"
        else "p5_fee_adjusted_pnl"
    )
    return {
        "classification": classification,
        "count": int(len(side)),
        "total_fee_adjusted_pnl": money_sum(side[pnl_column].dropna()),
        "by_right": {
            str(key): {
                "count": int(len(group)),
                "pnl": money_sum(group[pnl_column].dropna()),
            }
            for key, group in side.groupby("right", sort=True)
        },
        "by_time_of_day_et": {
            str(key): {
                "count": int(len(group)),
                "pnl": money_sum(group[pnl_column].dropna()),
            }
            for key, group in side.groupby("time_of_day_et", sort=True)
        },
        "entry_premium_dollars": number_summary(side["premium_at_risk"]),
        "entry_ask": number_summary(side["entry_ask"]),
    }


def generate_report(
    route: str,
    reconciliation: dict[str, Any],
    occupancy: dict[str, Any],
) -> str:
    common = reconciliation["common_exit_contribution"]
    stream = reconciliation["entry_stream_contribution"]
    total = reconciliation["hgb_minus_p5_total"]
    p5 = occupancy["comparators"]["original_p5"]
    hgb = occupancy["comparators"]["real_hgb"]
    return f"""# Protocol101 FT2 Stage-0 Serial Failure Attribution

## Result

Terminal route: `{route}`

Highest allowed claim: Frozen Stage-0 serial failure accounting attribution complete.

## Exact Accounting

- Frozen P5 serial PnL: `${reconciliation["p5_total_pnl"]:,.2f}`
- Frozen HGB serial PnL: `${reconciliation["hgb_total_pnl"]:,.2f}`
- Reproduced HGB-minus-P5 delta: `${total:,.2f}`
- Common-entry exit contribution: `${common:,.2f}`
- Entry-stream/re-entry contribution: `${stream:,.2f}`
- Accounting residual: `${reconciliation["accounting_residual"]:,.2f}`

The HGB lost on both mechanisms. On the five entries shared with P5, its
one-step exit policy produced `${common:,.2f}` less PnL. Its early exits then
reopened the account and admitted ten HGB-only replacement entries whose net
contribution was `${stream:,.2f}`. The two components sum exactly to the
frozen `-$4,190` disadvantage.

## Occupancy

- P5 admitted trades: `{p5["admitted_trade_count"]}`
- HGB admitted trades: `{hgb["admitted_trade_count"]}`
- P5 first-trade PnL: `${p5["first_trade_only_pnl"]:,.2f}`
- HGB first-trade PnL: `${hgb["first_trade_only_pnl"]:,.2f}`
- HGB second-and-later PnL: `${hgb["second_and_later_pnl"]:,.2f}`
- P5 daily-stop sessions: `{len(p5["daily_stop_sessions"])}`
- HGB daily-stop sessions: `{len(hgb["daily_stop_sessions"])}`

P5 held one trade per session to forced flat. HGB used short stop/target exits,
re-entered when the account became available, and then reached the daily-loss
stop in every validation session. P5 crossed the loss threshold at final
forced-flat on two sessions, but this blocked zero later intents; HGB's
intraday crossings blocked 1,328. These are accounting observations, not
statistical proof and not a repaired strategy proposal.

## Boundaries

This run used only frozen attempt002 artifacts and simulator-v5 source. It did
not fit, refit, rescore, tune, select, promote, inspect protected evidence,
contact a broker, submit orders, or change runtime state.
"""


def write_packet(inputs: dict[str, Any], verification: dict[str, Any]) -> str:
    serial = inputs["serial"]
    episodes = inputs["episodes"]
    receipts = inputs["receipts"]
    fee = float(inputs["fee"])
    selected = serial[serial["comparator"].isin(["original_p5", "real_hgb"])]
    p5 = selected[selected["comparator"].eq("original_p5")].copy()
    hgb = selected[selected["comparator"].eq("real_hgb")].copy()
    validation = episodes[episodes["role"].eq("nested_validation")]
    intents = validation[validation["comparator"].eq("original_p5")].copy()
    daily_loss_fraction = 0.05
    p5_skipped = classify_skipped_intents(
        intents, p5, daily_loss_fraction=daily_loss_fraction
    )
    hgb_skipped = classify_skipped_intents(
        intents, hgb, daily_loss_fraction=daily_loss_fraction
    )
    for comparator, skipped in (
        ("original_p5", p5_skipped),
        ("real_hgb", hgb_skipped),
    ):
        expected = receipts["comparators"][comparator]["skipped"]
        actual = Counter(skipped["reason"])
        for reason in expected:
            if int(actual.get(reason, 0)) != int(expected[reason]):
                raise AttributionBlocked(
                    "attribution_blocked_artifact_mismatch",
                    f"{comparator} skipped reason mismatch for {reason}",
                )
        unexpected = {
            key: value for key, value in actual.items() if key not in expected
        }
        if unexpected:
            raise AttributionBlocked(
                "attribution_blocked_artifact_mismatch",
                f"{comparator} has unclassified skipped intents: {unexpected}",
            )
    decomposition = decomposition_frame(p5, hgb, fee)
    sessions = sorted(intents["session"].unique().tolist())
    session_frame = per_session_attribution(
        sessions,
        p5,
        hgb,
        decomposition,
        p5_skipped,
        hgb_skipped,
        fee,
        daily_loss_fraction,
    )
    common_exit = money_sum(decomposition["common_exit_pnl_delta"])
    hgb_only_pnl = money_sum(
        decomposition.loc[
            decomposition["classification"].eq("hgb_only"),
            "hgb_fee_adjusted_pnl",
        ].dropna()
    )
    p5_only_pnl = money_sum(
        decomposition.loc[
            decomposition["classification"].eq("p5_only"),
            "p5_fee_adjusted_pnl",
        ].dropna()
    )
    entry_stream = money_sum([hgb_only_pnl, -p5_only_pnl])
    p5_total = money_sum(p5["stressed_pnl"])
    hgb_total = money_sum(hgb["stressed_pnl"])
    total_delta = money_sum([hgb_total, -p5_total])
    residual = money_sum([total_delta, -common_exit, -entry_stream])
    reconciliation = {
        "evidence_grade": EVIDENCE_GRADE,
        "identity_equation": (
            "HGB total - P5 total = common exit delta + "
            "HGB-only PnL - P5-only PnL"
        ),
        "p5_total_pnl": p5_total,
        "hgb_total_pnl": hgb_total,
        "hgb_minus_p5_total": total_delta,
        "common_entry_count": int(
            decomposition["classification"].eq("common").sum()
        ),
        "p5_only_entry_count": int(
            decomposition["classification"].eq("p5_only").sum()
        ),
        "hgb_only_entry_count": int(
            decomposition["classification"].eq("hgb_only").sum()
        ),
        "common_exit_contribution": common_exit,
        "hgb_only_entry_pnl": hgb_only_pnl,
        "p5_only_entry_pnl": p5_only_pnl,
        "entry_stream_contribution": entry_stream,
        "accounting_residual": residual,
        "reconciles_exactly": math.isclose(residual, 0.0, abs_tol=1e-9),
        "session_residual_max_abs": float(
            session_frame["accounting_residual"].abs().max()
        ),
        "fees": {
            "fee_per_trade_dollars": fee,
            "p5_common": float(
                decomposition.loc[
                    decomposition["classification"].eq("common"),
                    "p5_fee_dollars",
                ].sum()
            ),
            "hgb_common": float(
                decomposition.loc[
                    decomposition["classification"].eq("common"),
                    "hgb_fee_dollars",
                ].sum()
            ),
            "p5_only": float(
                decomposition.loc[
                    decomposition["classification"].eq("p5_only"),
                    "p5_fee_dollars",
                ].sum()
            ),
            "hgb_only": float(
                decomposition.loc[
                    decomposition["classification"].eq("hgb_only"),
                    "hgb_fee_dollars",
                ].sum()
            ),
        },
    }
    if not reconciliation["reconciles_exactly"]:
        raise AttributionBlocked(
            "attribution_blocked_accounting_residual",
            f"pooled accounting residual is {residual}",
        )
    if reconciliation["session_residual_max_abs"] > 1e-9:
        raise AttributionBlocked(
            "attribution_blocked_accounting_residual",
            "per-session accounting does not reconcile",
        )
    route = route_from_contributions(common_exit, entry_stream)
    occupancy = {
        "evidence_grade": EVIDENCE_GRADE,
        "daily_loss_fraction_of_session_start_equity": daily_loss_fraction,
        "comparators": {
            "original_p5": comparator_occupancy(
                "original_p5",
                p5,
                p5_skipped,
                fee,
                daily_loss_fraction,
            ),
            "real_hgb": comparator_occupancy(
                "real_hgb",
                hgb,
                hgb_skipped,
                fee,
                daily_loss_fraction,
            ),
        },
        "side_only_entry_breakdown": {
            "p5_only": side_only_breakdown(decomposition, "p5_only"),
            "hgb_only": side_only_breakdown(decomposition, "hgb_only"),
        },
    }
    routing = {
        "terminal_route": route,
        "evidence_grade": EVIDENCE_GRADE,
        "common_exit_contribution": common_exit,
        "entry_stream_contribution": entry_stream,
        "sign_rule_applied": True,
        "signs_are_accounting_not_statistical_proof": True,
        "training_executed": False,
        "rescoring_executed": False,
        "tuning_executed": False,
        "protected_evidence_inspected": False,
        "broker_or_paper_activity": False,
        "runtime_or_promotion_changed": False,
        "highest_allowed_claim": (
            "Frozen Stage-0 serial failure accounting attribution complete."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT_DIR / "input_verification.json", verification)
    decomposition.to_parquet(
        OUTPUT_DIR / "trade_identity_decomposition.parquet", index=False
    )
    session_frame.to_csv(
        OUTPUT_DIR / "session_attribution.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    write_json(OUTPUT_DIR / "occupancy_and_reentry.json", occupancy)
    write_json(
        OUTPUT_DIR / "accounting_reconciliation.json", reconciliation
    )
    write_json(OUTPUT_DIR / "routing_decision.json", routing)
    (OUTPUT_DIR / "report.md").write_text(
        generate_report(route, reconciliation, occupancy),
        encoding="utf-8",
    )
    hash_targets = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.name != "hashes.sha256"
    )
    (OUTPUT_DIR / "hashes.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in hash_targets
        ),
        encoding="utf-8",
    )
    return route


def write_blocked_packet(
    route: str, message: str, inputs: dict[str, Any] | None
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    verification = {
        "status": "blocked",
        "evidence_grade": EVIDENCE_GRADE,
        "terminal_route": route,
        "blocker": message,
    }
    if inputs is not None:
        verification["attempt002_checksum_results"] = inputs.get(
            "checksum_results", []
        )
    write_json(OUTPUT_DIR / "input_verification.json", verification)
    pd.DataFrame().to_parquet(
        OUTPUT_DIR / "trade_identity_decomposition.parquet", index=False
    )
    pd.DataFrame().to_csv(OUTPUT_DIR / "session_attribution.csv", index=False)
    write_json(
        OUTPUT_DIR / "occupancy_and_reentry.json",
        {"status": "blocked", "blocker": message},
    )
    write_json(
        OUTPUT_DIR / "accounting_reconciliation.json",
        {"status": "blocked", "blocker": message},
    )
    write_json(
        OUTPUT_DIR / "routing_decision.json",
        {
            "terminal_route": route,
            "evidence_grade": EVIDENCE_GRADE,
            "blocker": message,
        },
    )
    (OUTPUT_DIR / "report.md").write_text(
        "# Protocol101 FT2 Stage-0 Serial Failure Attribution\n\n"
        f"Terminal route: `{route}`\n\n{message}\n",
        encoding="utf-8",
    )
    paths = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.name != "hashes.sha256"
    )
    (OUTPUT_DIR / "hashes.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the dedicated attribution output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if OUTPUT_DIR.exists():
        if not args.force:
            raise SystemExit(f"output directory already exists: {OUTPUT_DIR}")
        import shutil

        shutil.rmtree(OUTPUT_DIR)
    inputs: dict[str, Any] | None = None
    try:
        inputs = prepare_inputs()
        verification = verify_frozen_result(inputs)
        route = write_packet(inputs, verification)
    except AttributionBlocked as exc:
        write_blocked_packet(exc.route, str(exc), inputs)
        route = exc.route
    print(json.dumps({"terminal_route": route, "output_dir": str(OUTPUT_DIR)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
