"""Project the evidence value of the same-game SPXW 0DTE CBBO backfill.

This analysis opens no new market outcome, fits nothing and contacts nobody.
It combines already-recorded inventory, effective-sample and built-parameter
receipts under a predeclared arithmetic law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256


SCHEMA = "v5.spxw-0dte-backfill-value.v1"


def _verify_self_hash(payload: dict[str, Any], *, field: str = "self_hash") -> None:
    expected = str(payload[field])
    body = dict(payload)
    body.pop(field)
    actual = hashlib.sha256(canonical_json(body)).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{field} mismatch: {actual} != {expected}")


def project_value(
    *,
    current_complete_sessions: int,
    current_quote_sessions: int,
    additional_nonempty_sessions: int,
    effective_ns: list[float],
    parameter_counts: dict[str, int],
    closed_architectures: set[str],
    observations_per_parameter: int = 20,
) -> dict[str, Any]:
    if not (
        0 < current_complete_sessions <= current_quote_sessions
        and additional_nonempty_sessions > 0
        and observations_per_parameter > 0
        and effective_ns
        and parameter_counts
    ):
        raise RuntimeError("backfill-value inputs are empty or invalid")
    if any(value <= 0 for value in effective_ns) or any(
        value <= 0 for value in parameter_counts.values()
    ):
        raise RuntimeError("effective sample sizes and parameter counts must be positive")

    projected_additional = math.floor(
        additional_nonempty_sessions * current_complete_sessions / current_quote_sessions
    )
    scenarios = {
        "observed_completeness": current_complete_sessions + projected_additional,
        "all_nonempty_complete": current_complete_sessions + additional_nonempty_sessions,
    }
    effective_per_session = {
        "low": min(effective_ns) / current_complete_sessions,
        "high": max(effective_ns) / current_complete_sessions,
    }
    scenario_rows: dict[str, Any] = {}
    for name, total in scenarios.items():
        projected_low = effective_per_session["low"] * total
        projected_high = effective_per_session["high"] * total
        scenario_rows[name] = {
            "projected_complete_sessions": total,
            "evidence_multiplier": total / current_complete_sessions,
            "projected_effective_n_design_effect": [projected_low, projected_high],
            "conservative_parameter_budget": [
                math.floor(projected_low / observations_per_parameter),
                math.floor(projected_high / observations_per_parameter),
            ],
        }

    support: dict[str, Any] = {}
    for architecture, parameters in sorted(parameter_counts.items()):
        required = [
            math.ceil(parameters * observations_per_parameter / effective_per_session["high"]),
            math.ceil(parameters * observations_per_parameter / effective_per_session["low"]),
        ]
        support[architecture] = {
            "parameters": parameters,
            "closed_on_current_corpus": architecture in closed_architectures,
            "complete_sessions_required_design_effect": required,
            "supported_observed_completeness": all(
                parameters <= budget
                for budget in scenario_rows["observed_completeness"]["conservative_parameter_budget"]
            ),
            "supported_all_nonempty_complete": all(
                parameters <= budget
                for budget in scenario_rows["all_nonempty_complete"]["conservative_parameter_budget"]
            ),
        }

    open_counts = {
        name: count for name, count in parameter_counts.items() if name not in closed_architectures
    }
    smallest_open = min(open_counts, key=open_counts.get)
    worst_budget = scenario_rows["observed_completeness"]["conservative_parameter_budget"][0]
    best_budget = scenario_rows["all_nonempty_complete"]["conservative_parameter_budget"][1]
    return {
        "projected_additional_complete_sessions": projected_additional,
        "scenarios": scenario_rows,
        "architecture_support": support,
        "smallest_open_existing_architecture": {
            "name": smallest_open,
            "parameters": open_counts[smallest_open],
            "supported_under_full_conservative_range": support[smallest_open][
                "supported_all_nonempty_complete"
            ],
        },
        "new_architecture_requirement": {
            "parameters_at_most_for_full_conservative_support": worst_budget,
            "parameters_at_most_under_optimistic_conservative_case": best_budget,
        },
        "decision": (
            "BACKFILL_MULTIPLIES_EVIDENCE_BUT_NEEDS_A_NEW_SMALLER_SHARED_LIFECYCLE_DESIGN"
            if not support[smallest_open]["supported_all_nonempty_complete"]
            else "BACKFILL_SUPPORTS_AN_EXISTING_OPEN_ARCHITECTURE"
        ),
    }


def run(*, declaration_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("refusing to overwrite backfill-value receipt")
    declaration = json.loads(declaration_path.read_text())
    _verify_self_hash(declaration)
    inputs = declaration["inputs"]
    cost = json.loads(Path(inputs["backfill_cost_receipt"]).read_text())
    effective = json.loads(Path(inputs["effective_sample_receipt"]).read_text())
    architecture = json.loads(Path(inputs["architecture_declaration"]).read_text())
    economics = json.loads(Path(inputs["compact_economics_receipt"]).read_text())

    current_complete = int(effective["sessions"])
    current_quote = 251
    design_effect_ns = [float(row["effective_n_design_effect"]) for row in effective["by_label"]]
    counts = {str(k): int(v) for k, v in architecture["architecture"]["canonical_parameter_counts"].items()}
    closed = {str(economics["fit_gate"]["architecture"])}
    result = project_value(
        current_complete_sessions=current_complete,
        current_quote_sessions=current_quote,
        additional_nonempty_sessions=int(cost["estimate"]["additional_sessions"]),
        effective_ns=design_effect_ns,
        parameter_counts=counts,
        closed_architectures=closed,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_on": "2026-08-14",
        "declaration": {"path": str(declaration_path), "sha256": file_sha256(declaration_path)},
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(Path(path))}
            for name, path in inputs.items()
        },
        "scope": declaration["scope"],
        "result": result,
        "integrity": {
            "fit_performed": False,
            "new_economics_read": False,
            "vendor_contacted": False,
            "data_downloaded": False,
            "money_spent": False,
            "reserved_sessions_used": False,
        },
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(declaration_path=args.declaration, output_path=args.output)
    print(json.dumps(payload["result"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
