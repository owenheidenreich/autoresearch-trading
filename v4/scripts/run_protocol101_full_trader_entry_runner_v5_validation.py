"""No-fit validation for the Protocol101 fresh entry-runner v5 core."""
from __future__ import annotations

import argparse
import ast
import inspect
import json
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v4.model import protocol101_scoped_stage1_hgb as core
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as runner
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner_v2 as durable


DEFAULT_OUT = (
    Path("v4/audit/autoresearch")
    / "protocol101_full_trader_entry_runner_v5_core_repair_attempt001"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )


def _called_names(function: Any) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def validate_call_graph() -> dict[str, Any]:
    unit_calls = _called_names(core.run_hgb_unit_v5)
    threshold_calls = _called_names(core.choose_threshold_v5)
    fee_calls = _called_names(core.replay_candidates_v5_at_fee)
    replay_calls = _called_names(core.replay_candidates_v5)
    runner_calls = _called_names(runner.run_fresh_hypothesis)
    durable_calls = _called_names(durable.run_fresh_hypothesis)
    predicates = {
        "fresh_runner_calls_repaired_loader": (
            "load_repaired_decisions" in runner_calls
        ),
        "fresh_runner_calls_v5_unit": "run_hgb_unit_v5" in runner_calls,
        "durable_runner_calls_repaired_loader": (
            "load_repaired_decisions" in durable_calls
        ),
        "durable_runner_calls_v5_unit": (
            "run_hgb_unit_v5" in durable_calls
        ),
        "unit_uses_v5_threshold": "choose_threshold_v5" in unit_calls,
        "unit_uses_v5_selection": "selection_rows_v5" in unit_calls,
        "unit_uses_v5_primary_replay": (
            "replay_candidates_v5" in unit_calls
        ),
        "unit_uses_v5_fee_replay": (
            "replay_candidates_v5_at_fee" in unit_calls
        ),
        "threshold_uses_v5_replay": (
            "replay_candidates_v5" in threshold_calls
        ),
        "fee_uses_v5_replay": "replay_candidates_v5" in fee_calls,
        "v5_replay_terminates_in_v5_simulator": (
            "simulate_serial_candidates_v5" in replay_calls
        ),
        "fresh_runner_avoids_legacy_loader": (
            "load_decisions" not in runner_calls
        ),
        "fresh_runner_avoids_legacy_unit": (
            "run_hgb_unit" not in runner_calls
        ),
        "durable_runner_avoids_legacy_loader": (
            "load_decisions" not in durable_calls
        ),
        "durable_runner_avoids_legacy_unit": (
            "run_hgb_unit" not in durable_calls
        ),
    }
    return {
        "schema_version": "Protocol101FreshRunnerCallGraphValidationV1",
        "status": "PASS" if all(predicates.values()) else "FAIL",
        "predicates": predicates,
        "calls": {
            "fresh_runner": sorted(runner_calls),
            "durable_runner": sorted(durable_calls),
            "v5_unit": sorted(unit_calls),
            "v5_threshold": sorted(threshold_calls),
            "v5_fee": sorted(fee_calls),
            "v5_replay": sorted(replay_calls),
        },
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dry_args = Namespace(
        mode="dry-run",
        out_dir=runner.DEFAULT_FRESH_OUT,
        readiness=runner.DEFAULT_READINESS,
        hypothesis="H0",
        owner_approved_plumbing_smoke=False,
        owner_approved_offline_training=False,
        force=False,
        smoke_rows_per_session=20,
    )
    plan = runner.fresh_runner_plan(dry_args)
    call_graph = validate_call_graph()
    write_json(args.out_dir / "dry_run" / "runner_plan.json", plan)
    write_json(args.out_dir / "fresh_runner_call_graph.json", call_graph)
    status = (
        "PASS"
        if plan["status"]
        == "v5_core_ready_pending_independent_acceptance"
        and not plan["core_blockers"]
        and call_graph["status"] == "PASS"
        else "FAIL"
    )
    result = {
        "schema_version": "Protocol101FreshRunnerV5NoFitValidationV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "runner_plan_status": plan["status"],
        "core_blockers": plan["core_blockers"],
        "deferred_stack_blockers": plan["deferred_stack_blockers"],
        "call_graph_status": call_graph["status"],
        "model_fit_executed": False,
        "model_score_executed": False,
        "economic_replay_executed": False,
        "seed_45_accessed": False,
        "protected_or_sealed_data_accessed": False,
        "broker_or_paper_submit": False,
    }
    write_json(args.out_dir / "no_fit_validation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
