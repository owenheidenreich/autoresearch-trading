"""Audit every activated causal-day deliverable without redefining completion.

The audit is intentionally allowed to conclude ``NOT_COMPLETE``.  It separates
implemented model-free machinery from the fitted, out-of-fold evidence the
owner's goal actually requires.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256


RECEIPTS = {
    "coverage": Path(
        "v4/audit/autoresearch/causal_day_trader_coverage_2026_08_14_attempt002/receipt.json"
    ),
    "settlement": Path(
        "v4/audit/autoresearch/causal_day_terminal_settlement_2026_08_14/receipt.json"
    ),
    "dataset": Path(
        "v4/audit/autoresearch/causal_day_dataset_settlement_validated_2026_08_14/receipt.json"
    ),
    "atlas": Path(
        "v4/audit/autoresearch/causal_day_atlas_settlement_validated_2026_08_14/receipt.json"
    ),
    "observation": Path(
        "v4/audit/autoresearch/causal_day_observation_contract_2026_08_14/receipt.json"
    ),
    "simulator": Path(
        "v4/audit/autoresearch/causal_day_simulator_2026_08_14_attempt005/receipt.json"
    ),
    "architectures": Path(
        "v4/audit/autoresearch/causal_day_architecture_interfaces_2026_08_14_attempt003/receipt.json"
    ),
    "replays": Path(
        "v4/audit/autoresearch/causal_day_replay_plumbing_2026_08_14_attempt005/receipt.json"
    ),
    "model_block": Path(
        "v4/audit/autoresearch/causal_day_model_block_2026_08_14_attempt003/receipt.json"
    ),
}


def _load_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = value.pop("receipt_sha256")
    actual = hashlib.sha256(canonical_json(value)).hexdigest()
    if actual != expected:
        raise RuntimeError(f"receipt self-hash mismatch: {path}")
    value["receipt_sha256"] = expected
    return value


def run(out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {out_dir}")
    loaded = {name: _load_receipt(path) for name, path in RECEIPTS.items()}
    coverage = loaded["coverage"]
    dataset = loaded["dataset"]
    atlas = loaded["atlas"]
    observation = loaded["observation"]
    simulator = loaded["simulator"]
    architectures = loaded["architectures"]
    replays = loaded["replays"]
    block = loaded["model_block"]

    if coverage["sessions"]["included_for_episode_build"] != 243:
        raise RuntimeError("coverage no longer proves 243 complete episodes")
    if dataset["rows"]["candidates"] != 698_231:
        raise RuntimeError("dataset candidate population moved")
    if atlas["rows"]["all_minutes"] != 79_218 or atlas["sessions"] != 243:
        raise RuntimeError("atlas is not complete over the included entry-minute grid")
    if observation["assertions"]["policy_chain_equals_source_live_two_sided_count"] is not True:
        raise RuntimeError("whole-chain observation is not proven")
    if simulator["known_answer_test_count"] != 16 or simulator["smoke_cells"] != 18:
        raise RuntimeError("simulator proof count moved")
    if architectures["fit_performed"] or replays["fit_performed"] or block["fit_performed"]:
        raise RuntimeError("completion audit expected an unfitted foundation")
    if not all(value["current_fit_blockers"] for value in architectures["family"].values()):
        raise RuntimeError("an architecture unexpectedly has no recorded fit blocker")

    requirements = [
        {
            "id": "D1",
            "requirement": "data coverage receipt with exact sessions, clocks, chain breadth and first decisions",
            "status": "PROVEN",
            "evidence": ["coverage", "observation"],
        },
        {
            "id": "D2",
            "requirement": "deterministic causal $10,000 simulator with account, touch fills and no overlap",
            "status": "PROVEN",
            "evidence": ["simulator", "settlement", "observation"],
        },
        {
            "id": "D3",
            "requirement": "out-of-fold joint versus shared-four-head architecture comparison and conditional independent control",
            "status": "NOT_ACHIEVED_FIT_BLOCKED",
            "evidence": ["architectures", "model_block"],
        },
        {
            "id": "D4",
            "requirement": "every-day 10:00/13:30 atlas plus honest complete time-of-day surface",
            "status": "PROVEN_MODEL_FREE",
            "evidence": ["atlas", "dataset"],
        },
        {
            "id": "D5",
            "requirement": "model win/loss/abstention replays with probabilities and behavioral attribution",
            "status": "NOT_ACHIEVED_PLUMBING_ONLY",
            "evidence": ["replays", "model_block"],
        },
        {
            "id": "D6",
            "requirement": "complete declared policy economics and risk at bid/mid with corrected confidence",
            "status": "NOT_ACHIEVED_PLUMBING_ONLY",
            "evidence": ["replays", "model_block"],
        },
        {
            "id": "D7",
            "requirement": "durable log, status, do-not-retest and evidence records",
            "status": "PROVEN",
            "evidence": [
                "v5/work/entry-exit-attribution/LOG.md",
                "v5/STATUS.md",
                "v5/research/history/DO_NOT_RETEST.md",
                "v5/evidence/INDEX.md",
            ],
        },
        {
            "id": "V1",
            "requirement": "strict chronological walk-forward, fold-only transforms and out-of-fold exits",
            "status": "NOT_ACHIEVED_FIT_BLOCKED",
            "evidence": ["architectures", "model_block"],
        },
        {
            "id": "V2",
            "requirement": "matched random entries, clocks, trails, simple exits and full declared family",
            "status": "NOT_ACHIEVED_FIT_BLOCKED",
            "evidence": ["model_block"],
        },
        {
            "id": "V3",
            "requirement": "positive/null conclusion about whether the causal whole-day trader is profitable",
            "status": "NOT_ACHIEVED_NO_POLICY",
            "evidence": ["model_block"],
        },
    ]
    achieved = [item["id"] for item in requirements if item["status"].startswith("PROVEN")]
    remaining = [item["id"] for item in requirements if not item["status"].startswith("PROVEN")]
    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-completion-audit.v1",
        "created_on": "2026-08-14",
        "objective": "activated causal time-aware 0DTE day-trader goal",
        "overall_status": "NOT_COMPLETE_FIT_DEPENDENT_DELIVERABLES_MISSING",
        "requirements": requirements,
        "proven_requirement_ids": achieved,
        "remaining_requirement_ids": remaining,
        "remaining_work": [
            "fit and score the declared chronological shallow comparison",
            "fit sequence candidates only if the neural sample rules also open",
            "run matched entry and exit controls on each model's own out-of-fold trades",
            "export model-selected wins, losses and abstentions with behavioral attribution",
            "report the complete bid/mid family and reach the empirical one-versus-four conclusion",
        ],
        "blocking_condition": {
            "general": [
                "G1 is UNDERPOWERED and STATUS prohibits option-model fitting until it passes",
                "the later quote-priced selective-entry do-not-retest row closes another long-side fit on this corpus",
            ],
            "neural_additional": [
                "243 sessions are below the frozen 1,140-session floor",
                "the 1,296-parameter shared candidate requires 25,920 sessions at 20 sessions per parameter",
            ],
        },
        "fit_performed": False,
        "profitability_claim": None,
        "architecture_winner": None,
        "receipt_bindings": {
            name: {
                "path": str(RECEIPTS[name]),
                "file_sha256": file_sha256(RECEIPTS[name]),
                "receipt_sha256": loaded[name]["receipt_sha256"],
            }
            for name in RECEIPTS
        },
        "record_hashes": {
            str(path): file_sha256(path)
            for path in (
                Path("v5/work/entry-exit-attribution/GOAL.md"),
                Path("v5/work/entry-exit-attribution/PLAN.md"),
                Path("v5/work/entry-exit-attribution/LOG.md"),
                Path("v5/STATUS.md"),
                Path("v5/research/history/DO_NOT_RETEST.md"),
                Path("v5/evidence/INDEX.md"),
            )
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=False)
    path = out_dir / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
