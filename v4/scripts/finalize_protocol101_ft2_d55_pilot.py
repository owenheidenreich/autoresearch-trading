"""Seal and independently review the completed FT2-D55 pilot evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol101_ft2_d55_acquisition import (
    APPROVAL_TEXT,
    AUTHORITY_PATH,
    AUTHORITY_SHA256,
    DATASET,
    DEFAULT_OUT_DIR,
    DEFAULT_RAW_ROOT,
    GOAL,
    HARD_CAP_USD,
    SCHEMA,
    canonical_sha256,
    sha256_file,
    utc_now,
    write_json,
)


OUT_DIR = DEFAULT_OUT_DIR
RAW_ROOT = DEFAULT_RAW_ROOT
RECEIPT_PATH = OUT_DIR / "receipt.json"
REVIEW_PATH = OUT_DIR / "delta_scoped_review.json"
MEASUREMENT_OUTPUTS = (
    "per_session_metrics.csv",
    "per_premium_band.csv",
    "per_moneyness_band.csv",
    "per_premium_moneyness_band.csv",
    "floor_slippage_by_band.csv",
    "entry_signal_exploratory.csv",
)
SOURCE_FILES = (
    Path("v4/scripts/run_protocol101_ft2_d55_acquisition.py"),
    Path("v4/scripts/run_protocol101_ft2_d55_measurements.py"),
    Path("v4/scripts/finalize_protocol101_ft2_d55_pilot.py"),
    Path("v4/tests/test_protocol101_ft2_d55_acquisition.py"),
    Path("v4/tests/test_protocol101_ft2_d55_measurements.py"),
)


class Review:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, evidence: Any) -> None:
        if not condition:
            raise RuntimeError(f"D55 review failed: {name}: {evidence}")
        self.checks.append(
            {"name": name, "outcome": "PASS", "evidence": evidence}
        )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    review = Review()
    review.check(
        "authority_hash",
        sha256_file(AUTHORITY_PATH) == AUTHORITY_SHA256,
        {"path": str(AUTHORITY_PATH), "sha256": AUTHORITY_SHA256},
    )
    plan_path = OUT_DIR / "acquisition_plan.json"
    cost_path = OUT_DIR / "cost_receipt.json"
    acquisition_path = OUT_DIR / "acquisition_summary.json"
    failures_path = OUT_DIR / "failed_download_attempts.json"
    batch_path = OUT_DIR / "batch_fallback_jobs.json"
    spec_path = OUT_DIR / "measurement_spec.json"
    results_path = OUT_DIR / "distortion_results.json"
    report_path = OUT_DIR / "report.md"
    plan = load_json(plan_path)
    cost = load_json(cost_path)
    acquisition = load_json(acquisition_path)
    failures = load_json(failures_path)
    batch = load_json(batch_path)
    spec = load_json(spec_path)
    results = load_json(results_path)

    sessions = plan["sessions"]
    dates = [str(row["date"]) for row in sessions]
    review.check(
        "deterministic_selection",
        len(dates) == 30
        and len(set(dates)) == 30
        and dates == acquisition["selected_dates"]
        and dates == results["selected_dates"],
        {"dates": dates, "indices": plan["selection"]["indices"]},
    )
    review.check(
        "exact_paid_scope",
        plan["request"]["dataset"] == DATASET
        and plan["request"]["schema"] == SCHEMA
        and cost["request"]["dataset"] == DATASET
        and cost["request"]["schema"] == SCHEMA
        and acquisition["dataset"] == DATASET
        and acquisition["schema"] == SCHEMA,
        {
            "dataset": DATASET,
            "schema": SCHEMA,
            "symbol_sessions": plan["request"]["symbol_session_count"],
        },
    )
    estimated_total = float(cost["estimated_total_usd"])
    failure_upper = float(failures["billable_cost_upper_bound_usd"])
    worst_case = estimated_total + failure_upper
    review.check(
        "cost_gate_and_reconciliation",
        cost["gate"] == "PASS"
        and estimated_total <= HARD_CAP_USD
        and close(acquisition["estimated_cost_usd"], estimated_total)
        and close(
            acquisition["failed_attempt_billable_cost_upper_bound_usd"],
            failure_upper,
        )
        and close(
            acquisition["worst_case_estimated_spend_usd"], worst_case
        )
        and worst_case <= HARD_CAP_USD,
        {
            "estimated_total_usd": estimated_total,
            "failed_attempt_billable_cost_upper_bound_usd": failure_upper,
            "worst_case_estimated_spend_usd": worst_case,
            "hard_cap_usd": HARD_CAP_USD,
            "actual_vendor_invoice_cost_usd": acquisition[
                "actual_vendor_invoice_cost_usd"
            ],
        },
    )

    cost_by_date = {
        str(row["date"]): row for row in cost["session_estimates"]
    }
    raw_receipts: list[dict[str, Any]] = []
    total_rows = 0
    total_dbn_bytes = 0
    total_parquet_bytes = 0
    total_symbols = 0
    total_manifest_cost = 0.0
    for plan_row in sessions:
        session = str(plan_row["date"])
        directory = RAW_ROOT / session
        manifest_path = directory / "manifest.json"
        manifest = load_json(manifest_path)
        symbols = list(plan_row["symbols"])
        dbn = Path(manifest["files"]["dbn"]["path"])
        parquet = Path(manifest["files"]["parquet"]["path"])
        review.check(
            f"raw_session_{session}",
            manifest["date"] == session
            and manifest["request"]["dataset"] == DATASET
            and manifest["request"]["schema"] == SCHEMA
            and manifest["request"]["symbols"] == symbols
            and manifest["request"]["symbols_sha256"]
            == canonical_sha256(symbols)
            and manifest["request"]["symbols_sha256"]
            == plan_row["symbols_sha256"]
            and manifest["files"]["dbn"]["sha256"] == sha256_file(dbn)
            and manifest["files"]["parquet"]["sha256"]
            == sha256_file(parquet)
            and manifest["files"]["dbn"]["bytes"] == dbn.stat().st_size
            and manifest["files"]["parquet"]["bytes"]
            == parquet.stat().st_size
            and close(
                manifest["request"]["estimated_cost_usd"],
                cost_by_date[session]["cost_estimate_usd"],
            ),
            {
                "symbol_count": len(symbols),
                "rows": manifest["download"]["rows"],
                "dbn_bytes": dbn.stat().st_size,
                "parquet_bytes": parquet.stat().st_size,
            },
        )
        total_rows += int(manifest["download"]["rows"])
        total_dbn_bytes += dbn.stat().st_size
        total_parquet_bytes += parquet.stat().st_size
        total_symbols += len(symbols)
        total_manifest_cost += float(
            manifest["request"]["estimated_cost_usd"]
        )
        raw_receipts.append(
            {
                "sequence": int(plan_row["sequence"]),
                "selection_index": int(plan_row["selection_index"]),
                "date": session,
                "symbols": symbols,
                "symbol_count": len(symbols),
                "symbols_sha256": plan_row["symbols_sha256"],
                "normalized_source": {
                    "path": plan_row["normalized_path"],
                    "sha256": plan_row["normalized_sha256"],
                },
                "estimated_cost_usd": float(
                    manifest["request"]["estimated_cost_usd"]
                ),
                "delivery": manifest["request"].get(
                    "delivery", "historical_streaming"
                ),
                "rows": int(manifest["download"]["rows"]),
                "manifest": file_record(manifest_path),
                "dbn": file_record(dbn),
                "parquet": file_record(parquet),
            }
        )
    review.check(
        "raw_totals",
        total_rows == int(acquisition["rows"])
        and total_symbols == int(acquisition["requested_symbol_sessions"])
        and total_symbols == int(acquisition["returned_symbols"])
        and total_dbn_bytes == int(acquisition["bytes"]["dbn"])
        and total_parquet_bytes == int(acquisition["bytes"]["parquet"])
        and close(total_manifest_cost, estimated_total),
        {
            "sessions": len(raw_receipts),
            "symbols": total_symbols,
            "rows": total_rows,
            "dbn_bytes": total_dbn_bytes,
            "parquet_bytes": total_parquet_bytes,
            "estimated_cost_usd": total_manifest_cost,
        },
    )
    raw_dates = sorted(
        path.name for path in RAW_ROOT.iterdir() if path.is_dir()
    )
    review.check(
        "raw_root_date_scope",
        raw_dates == sorted(dates),
        {"raw_root": str(RAW_ROOT), "dates": raw_dates},
    )
    review.check(
        "failure_scope",
        all(
            row["dataset"] == DATASET
            and row["schema"] == SCHEMA
            and str(row["date"]) in dates
            for row in failures["attempts"]
        ),
        {
            "attempts": len(failures["attempts"]),
            "billable_cost_upper_bound_usd": failure_upper,
        },
    )
    review.check(
        "batch_fallback_scope",
        all(
            row["dataset"] == DATASET
            and row["schema"] == SCHEMA
            and str(row["date"]) in dates
            and row["symbols_sha256"]
            == next(
                plan_row["symbols_sha256"]
                for plan_row in sessions
                if plan_row["date"] == row["date"]
            )
            for row in batch["jobs"]
        ),
        {
            "jobs": [
                {"date": row["date"], "job_id": row["job_id"]}
                for row in batch["jobs"]
            ]
        },
    )

    measurement_receipts: list[dict[str, Any]] = []
    for session in dates:
        manifest_path = (
            OUT_DIR / "session_measurements" / session / "manifest.json"
        )
        manifest = load_json(manifest_path)
        outputs = manifest["outputs"]
        valid = manifest["goal"] == GOAL and manifest["session"] == session
        for value in outputs.values():
            path = Path(value["path"])
            valid = (
                valid
                and path.exists()
                and value["sha256"] == sha256_file(path)
                and int(value["bytes"]) == path.stat().st_size
            )
        review.check(
            f"measurement_session_{session}",
            valid,
            manifest["rows"],
        )
        measurement_receipts.append(
            {
                "date": session,
                "manifest": file_record(manifest_path),
                "rows": manifest["rows"],
                "outputs": outputs,
            }
        )

    per_session = pd.read_csv(OUT_DIR / "per_session_metrics.csv")
    per_premium = pd.read_csv(OUT_DIR / "per_premium_band.csv")
    per_moneyness = pd.read_csv(OUT_DIR / "per_moneyness_band.csv")
    per_cross = pd.read_csv(OUT_DIR / "per_premium_moneyness_band.csv")
    floor_band = pd.read_csv(OUT_DIR / "floor_slippage_by_band.csv")
    overall = results["overall"]
    review.check(
        "distortion_population_reconciliation",
        int(per_session["contract_minutes"].sum())
        == int(overall["contract_minutes"])
        and int(per_premium["contract_minutes"].sum())
        == int(overall["contract_minutes"])
        and int(per_moneyness["contract_minutes"].sum())
        == int(overall["contract_minutes"])
        and int(per_cross["contract_minutes"].sum())
        == int(overall["contract_minutes"])
        and int(per_session["one_second_stream_rows"].sum())
        == int(overall["one_second_stream_rows"])
        and int(per_session["one_second_executable_rows"].sum())
        == int(overall["one_second_executable_rows"]),
        {
            "contract_minutes": int(overall["contract_minutes"]),
            "stream_rows": int(overall["one_second_stream_rows"]),
            "executable_rows": int(overall["one_second_executable_rows"]),
        },
    )
    for floor_row in results["floor_response"]:
        ratio = float(floor_row["floor_ratio"])
        subset = floor_band[
            np.isclose(
                pd.to_numeric(floor_band["floor_ratio"], errors="coerce"),
                ratio,
            )
        ]
        review.check(
            f"floor_population_{ratio:.2f}",
            int(subset["eligible_contract_minutes"].sum())
            == int(floor_row["eligible_contract_minutes"])
            and int(subset["crossed_contract_minutes"].sum())
            == int(floor_row["crossed_contract_minutes"]),
            {
                "eligible": int(floor_row["eligible_contract_minutes"]),
                "crossed": int(floor_row["crossed_contract_minutes"]),
            },
        )
    review.check(
        "measurement_route",
        results["sessions"] == 30
        and results["verdict"]["minute_resolution_exit_floor_validation"]
        == "NOT_TRUSTWORTHY_WITHOUT_HIGHRES_VALIDATION_OR_ADJUSTMENT"
        and results["highest_allowed_claim"]
        == (
            "the minute-vs-1-second exit-realism distortion is measured; "
            "the owner can decide substrate and exit-data strategy"
        )
        and results["next"] == "STOP_FOR_OWNER_DECISION"
        and results["side_effects"]["model_training_or_fitting"] is False
        and results["side_effects"]["broker_contacted"] is False
        and results["side_effects"]["signed_contract_modified"] is False
        and results["side_effects"]["graph_modified"] is False
        and results["side_effects"]["census_modified"] is False,
        {
            "verdict": results["verdict"][
                "minute_resolution_exit_floor_validation"
            ],
            "next": results["next"],
        },
    )

    deliverables = {
        path.name: file_record(path)
        for path in (
            plan_path,
            cost_path,
            acquisition_path,
            failures_path,
            batch_path,
            spec_path,
            results_path,
            report_path,
            *(OUT_DIR / name for name in MEASUREMENT_OUTPUTS),
        )
    }
    sources = {
        str(path): file_record(path)
        for path in SOURCE_FILES
    }
    receipt = {
        "schema_version": "Protocol101FT2D55ReceiptV1",
        "goal": GOAL,
        "sealed_at_utc": utc_now(),
        "authority": {
            "path": str(AUTHORITY_PATH),
            "sha256": AUTHORITY_SHA256,
        },
        "owner_authorization": {
            "date": "2026-07-30",
            "exact_approval_text": APPROVAL_TEXT,
            "paid_scope": {
                "dataset": DATASET,
                "schema": SCHEMA,
                "sessions": 30,
                "symbol_sessions": total_symbols,
                "other_paid_pulls": False,
            },
        },
        "selected_dates": dates,
        "raw_sessions": raw_receipts,
        "cost": {
            "estimated_total_usd": estimated_total,
            "failed_attempt_billable_cost_upper_bound_usd": failure_upper,
            "worst_case_estimated_spend_usd": worst_case,
            "hard_cap_usd": HARD_CAP_USD,
            "actual_vendor_invoice_cost_usd": acquisition[
                "actual_vendor_invoice_cost_usd"
            ],
            "reconciliation_delta_usd": total_manifest_cost
            - estimated_total,
        },
        "acquisition_totals": {
            "rows": total_rows,
            "dbn_bytes": total_dbn_bytes,
            "parquet_bytes": total_parquet_bytes,
            "requested_symbol_sessions": total_symbols,
            "returned_symbols": int(acquisition["returned_symbols"]),
            "failed_stream_attempts": len(failures["attempts"]),
            "batch_fallback_jobs": [
                {"date": row["date"], "job_id": row["job_id"]}
                for row in batch["jobs"]
            ],
        },
        "measurement": {
            "sessions": results["sessions"],
            "contract_minutes": int(overall["contract_minutes"]),
            "verdict": results["verdict"],
            "substrate_recommendation": results[
                "substrate_recommendation"
            ],
            "session_receipts": measurement_receipts,
        },
        "deliverables": deliverables,
        "implementation_sources": sources,
        "side_effects": {
            "paid_download_performed": True,
            "broker_contacted": False,
            "model_training_or_fitting": False,
            "protected_data_accessed": False,
            "runtime_or_launchd_modified": False,
            "promotion_or_paper_default_modified": False,
            "signed_contract_modified": False,
            "graph_modified": False,
            "census_modified": False,
        },
        "highest_allowed_claim": results["highest_allowed_claim"],
        "next": "STOP_FOR_OWNER_DECISION",
    }
    write_json(RECEIPT_PATH, receipt)
    review.check(
        "receipt_scope",
        len(receipt["raw_sessions"]) == 30
        and sum(row["symbol_count"] for row in receipt["raw_sessions"])
        == total_symbols
        and receipt["next"] == "STOP_FOR_OWNER_DECISION",
        {
            "receipt": str(RECEIPT_PATH),
            "sha256": sha256_file(RECEIPT_PATH),
        },
    )
    review_payload = {
        "schema_version": "Protocol101FT2D55DeltaScopedReviewV1",
        "goal": GOAL,
        "reviewed_at_utc": utc_now(),
        "outcome": "PASS",
        "checks": review.checks,
        "receipt": file_record(RECEIPT_PATH),
        "highest_allowed_claim": receipt["highest_allowed_claim"],
        "next": "STOP_FOR_OWNER_DECISION",
    }
    write_json(REVIEW_PATH, review_payload)
    print(
        json.dumps(
            {
                "outcome": "PASS",
                "checks": len(review.checks),
                "receipt": str(RECEIPT_PATH),
                "receipt_sha256": sha256_file(RECEIPT_PATH),
                "review": str(REVIEW_PATH),
                "review_sha256": sha256_file(REVIEW_PATH),
                "next": "STOP_FOR_OWNER_DECISION",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
