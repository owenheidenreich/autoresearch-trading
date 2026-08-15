"""Record the built compact shared lifecycle design without fitting it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_architectures import ARCHITECTURE_NAMES
from v5.research.causal_day_compact_shared_lifecycle import (
    ARCHITECTURE_NAME,
    LADDER_CONTEXT_FEATURES,
    POSITION_INDICES,
    computed_parameter_count,
)


SCHEMA = "v5.compact-shared-lifecycle-design.v1"


def build_receipt(*, design_path: Path, value_path: Path) -> dict[str, Any]:
    value = json.loads(value_path.read_text())
    parameters = computed_parameter_count()
    budget = int(
        value["result"]["new_architecture_requirement"][
            "parameters_at_most_for_full_conservative_support"
        ]
    )
    if parameters > budget:
        raise RuntimeError(f"built count {parameters} exceeds projected budget {budget}")
    if ARCHITECTURE_NAME in ARCHITECTURE_NAMES:
        raise RuntimeError("design-only architecture unexpectedly entered the active fit family")
    return {
        "schema_version": SCHEMA,
        "created_on": "2026-08-14",
        "question": "Can one shared entry/exit representation fit the worst projected conservative backfill budget?",
        "scope": {
            "traded_position": "one long SPXW 0DTE call or put only",
            "context": "SPXW and optional SPX only",
            "actions": ["WAIT", "BUY_ONE", "HOLD", "SELL_SAME_CONTRACT"],
        },
        "architecture": {
            "name": ARCHITECTURE_NAME,
            "built_parameter_count": parameters,
            "parameter_count_source": "computed_parameter_count() over built CompactSharedLifecyclePolicy",
            "projected_worst_conservative_budget": budget,
            "budget_headroom": budget - parameters,
            "shared_representation": True,
            "independent_specialists": False,
            "whole_ladder_context_features": list(LADDER_CONTEXT_FEATURES),
            "position_features_used": len(POSITION_INDICES),
            "opening_regime_owned_exit": True,
        },
        "fit_boundary": {
            "registered_in_active_architecture_family": False,
            "fit_permitted": False,
            "reason": "design-only prerequisite; future evidence and owner gate required",
        },
        "evidence": {
            "design": {"path": str(design_path), "sha256": file_sha256(design_path)},
            "backfill_value": {"path": str(value_path), "sha256": file_sha256(value_path)},
            "implementation": {
                "path": "v5/research/causal_day_compact_shared_lifecycle.py",
                "sha256": file_sha256(Path("v5/research/causal_day_compact_shared_lifecycle.py")),
            },
            "test": "v5/tests/test_causal_day_compact_shared_lifecycle.py",
        },
        "integrity": {
            "fit_performed": False,
            "economics_read": False,
            "vendor_contacted": False,
            "data_downloaded": False,
            "money_spent": False,
            "reserved_sessions_used": False,
        },
    }


def run(*, design_path: Path, value_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("refusing to overwrite compact lifecycle design receipt")
    payload = build_receipt(design_path=design_path, value_path=value_path)
    payload["implementation_sha256"] = file_sha256(Path(__file__))
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--value", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(design_path=args.design, value_path=args.value, output_path=args.output)
    print(json.dumps(payload["architecture"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
