"""Write the pre-outcome declaration for the one-trade action-value labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256


DATASET_RECEIPT = Path(
    "v4/audit/autoresearch/causal_day_dataset_settlement_validated_2026_08_14/receipt.json"
)
CANDIDATES = Path("/Volumes/AR_TRADING_DATA/derived/causal_day_trader_v2/candidates.parquet")
OUT_ROOT = Path("/Volumes/AR_TRADING_DATA/derived/causal_day_action_advantage_v2")
EVIDENCE_DIR = Path(
    "v4/audit/autoresearch/causal_day_action_advantage_2026_08_14_attempt002"
)
IMPLEMENTATION = (
    Path("v5/research/causal_day_action_advantage.py"),
    Path("v5/ops/build_causal_day_action_advantage.py"),
    Path("v5/ops/write_causal_day_action_advantage_declaration.py"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or OUT_ROOT.exists() or EVIDENCE_DIR.exists():
        raise RuntimeError("refusing to overwrite declaration, output, or evidence")
    receipt = json.loads(DATASET_RECEIPT.read_text())
    expected = receipt.get("receipt_sha256")
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned_receipt)).hexdigest() != expected:
        raise RuntimeError("dataset receipt self-hash mismatch")
    candidate_info = receipt["outputs"]["candidates"]
    if Path(candidate_info["path"]) != CANDIDATES or file_sha256(CANDIDATES) != candidate_info["sha256"]:
        raise RuntimeError("candidate artifact differs from settlement-complete receipt")
    payload = {
        "schema_version": "v5.causal-day-action-advantage-declaration.v2",
        "declared_on": "2026-08-14",
        "supersedes": {
            "path": "v5/work/entry-exit-attribution/ACTION_ADVANTAGE_DECLARATION_V1.json",
            "receipt_sha256": "3cf17e53754245e78ab0a43c9b174520f8670f5a881ae46b84ee7938661d2cb3",
            "status": "FAILED_SAFE_BEFORE_OUTPUT",
            "defect": "V1 assumed every decision minute had an eligible contract row; 1,964 of 79,218 minute states are correctly WAIT-only",
            "outcome_metric_computed": False,
            "output_or_evidence_written": False,
        },
        "purpose": "freeze a dense executable Q(enter) versus Q(wait) target before reading its values or fitting",
        "family": {"members": 1, "multiplicity_spent": 1, "nearby_retries_forbidden": True},
        "population": {
            "sessions": 243,
            "candidate_rows": 698231,
            "decision_minutes_per_session": 326,
            "decision_minutes": 79218,
            "structurally_measured_wait_only_minutes": 1964,
            "first_session": "2025-08-01",
            "last_session": "2026-07-30",
            "reserved_sessions_forbidden_on_or_after": "2026-08-06",
        },
        "label": {
            "name": "serial_action_advantage_120m",
            "trade_cap": 1,
            "position_cap": 1,
            "starting_equity_usd": 10000.0,
            "q_enter": "the existing net_bid_120m_usd: ask entry, first bid at/after 120m or validated settlement, measured fee included",
            "q_wait": "max(0, every q_enter at a strictly later eligible minute in the same session)",
            "a_enter": "q_enter - q_wait",
            "minutes_without_candidates": "retain the decision minute with no Q(enter) actions and a valid Q(wait); never drop the state",
            "primary_oracle_tie_break": "earliest global-maximum minute, then lexicographically first contract_id and trade_id",
            "mid_values": "parallel diagnostics only; not an alternate training target",
            "best_price_or_mfe_forbidden": True,
        },
        "diagnostics_declared_before_values": [
            "fixed quantiles of q_enter_bid, q_wait_bid, a_enter_bid, q_enter_mid and session oracle values",
            "positive q_enter and positive advantage row counts",
            "positive best-advantage minute count",
            "primary oracle counts by the six predeclared clock bands",
            "primary oracle counts by call and put",
        ],
        "diagnostic_consequence": "integrity and scale assessment only; no diagnostic may change horizon, cap, architecture, loss, subgroup, or operating law",
        "future_fit_if_separately_authorized": {
            "architecture": "compact_interaction_entry",
            "computed_parameters": 48,
            "action_vector": "WAIT plus every currently eligible affordable contract",
            "loss": "one frozen robust joint regression of q_wait and q_enter after training-prefix scaling",
            "inference": "argmax over the action vector; no external threshold or current-day rank",
            "chronology": "the existing five OOF folds",
            "controls": "identically fitted session-surface shuffled null and outcome-blind composition-matched control",
            "kill_conditions": [
                "mid_to_mid_gross_positive",
                "beats_composition_matched_control",
                "beats_shuffled_label_null",
                "per_feature_timestamp_audit",
                "no_post_entry_slot_filter",
                "chronological_out_of_sample",
                "no_reserved_sessions",
            ],
        },
        "forbidden": [
            "fit a model during this label build",
            "read or choose an alternative horizon",
            "change the one-trade cap after diagnostics",
            "select a time, side, premium, delta, or payoff subgroup",
            "use the oracle action as a causal feature",
            "open reserved sessions",
        ],
        "immutable_input": {
            "dataset_receipt": {"path": str(DATASET_RECEIPT), "sha256": file_sha256(DATASET_RECEIPT)},
            "candidates": {"path": str(CANDIDATES), "sha256": file_sha256(CANDIDATES)},
        },
        "implementation_hashes": {str(path): file_sha256(path) for path in IMPLEMENTATION},
        "outputs": {"root": str(OUT_ROOT), "evidence_dir": str(EVIDENCE_DIR)},
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(payload["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
