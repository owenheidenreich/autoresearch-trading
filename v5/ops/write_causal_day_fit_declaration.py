"""Generate the width-3 Job-39 fit declaration from built models.

Counts are never literals in the artifact: this command builds every declared
architecture through :func:`computed_parameter_counts` and writes the values it
measured.  The width-8 V3 file remains immutable history.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_architectures import (
    computed_parameter_counts,
    declared_dimensions,
)
from v5.research.causal_day_magnitude import HORIZONS, PERMITTED_ARCHITECTURES
from v5.research.causal_day_policy_gate import (
    CONSERVATIVE_EFFECTIVE_OBSERVATIONS,
    MEASURED_EFFECTIVE_OBSERVATIONS,
    REOPENED_CORPUS,
    REOPENED_LABEL,
    REQUIRED_KILL_CONDITIONS,
    load_reopening,
)
from v5.research.knobs import frozen_value


DEFAULT_V3 = Path("v5/work/entry-exit-attribution/DECLARATION_V3.json")
RERULING = Path("v5/governance/CAUSAL_DAY_FIT_RERULING_2026_08_14.md")
IMPLEMENTATIONS = (
    Path("v5/research/causal_day_policy_gate.py"),
    Path("v5/research/causal_day_magnitude.py"),
    Path("v5/research/causal_day_fit_cache.py"),
    Path("v5/ops/build_causal_day_fit_cache.py"),
    Path("v5/ops/train_causal_day_magnitude.py"),
    Path("v5/ops/evaluate_causal_day_magnitude.py"),
    Path("v5/ops/audit_causal_day_fit_features.py"),
    Path("v5/ops/write_causal_day_fit_declaration.py"),
)


def _verified_v3(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError("DECLARATION_V3 self-hash mismatch")
    if value.get("schema_version") != "v5.causal-day-trader-fit-declaration.v3":
        raise RuntimeError("expected immutable DECLARATION_V3 input")
    return value


def build_declaration(v3_path: Path = DEFAULT_V3) -> dict:
    v3 = _verified_v3(v3_path)
    value = copy.deepcopy(v3)
    reopening = load_reopening()
    counts = computed_parameter_counts()
    dimensions = declared_dimensions()
    per_parameter = int(frozen_value("minimum_sessions_per_neural_parameter"))
    budget = MEASURED_EFFECTIVE_OBSERVATIONS // per_parameter
    conservative_budget = CONSERVATIVE_EFFECTIVE_OBSERVATIONS // per_parameter

    permitted = {name: counts[name] for name in PERMITTED_ARCHITECTURES}
    if any(count > budget for count in permitted.values()):
        raise RuntimeError("a permitted built architecture exceeds the measured budget")
    if counts["four_independent"] <= budget:
        raise RuntimeError("the conditional independent architecture unexpectedly fits")

    value["schema_version"] = "v5.causal-day-trader-fit-declaration.v4"
    value["supersedes_declaration_sha256"] = file_sha256(v3_path)
    value["reopening"].update(
        {
            "sha256": reopening.document_sha256,
            "label": REOPENED_LABEL,
            "corpus": REOPENED_CORPUS,
            "horizons_minutes": list(HORIZONS),
            "required_kill_conditions": list(REQUIRED_KILL_CONDITIONS),
            "reruling_path": str(RERULING),
            "reruling_sha256": file_sha256(RERULING),
        }
    )
    value["evidence_budget"] = {
        "measured_effective_observations_generous": MEASURED_EFFECTIVE_OBSERVATIONS,
        "measured_effective_observations_design_effect_range": [590, 1_016],
        "minimum_observations_per_parameter": per_parameter,
        "trainable_parameter_budget_generous": budget,
        "trainable_parameter_budget_design_effect_range": [29, conservative_budget],
        "ruling": "The generous route authorizes comparison; the conservative route admits no architecture and must accompany every result",
    }
    value["architectures"] = {
        "hidden_size": dimensions.hidden_size,
        "count_source": "computed_parameter_counts() over canonical built models and declared tensorizer dimensions",
        "permitted": permitted,
        "refused": {
            "four_independent": {
                "parameters": counts["four_independent"],
                "reason": "Exceeds the measured 377-parameter ceiling and remains conditional on prior shared-head specialization",
            }
        },
        "horizon_models": "One canonical scalar contract-head model per architecture and declared horizon; no uncounted multi-output head",
        "exit_heads_during_entry_screen": "Present in the built parameter count but not optimized until the entry screen survives all signed kill conditions",
    }
    value["label"]["loss"] = (
        "Per-horizon smooth-L1 beta 0.25 over every eligible contract; each horizon "
        "uses its own canonical scalar contract-head model"
    )
    value["shuffled_label_null"]["law"] = (
        "Within each newly eligible training block, permute the complete three-horizon outcome "
        "vector among candidates with the same option side and rounded 5-point moneyness node; "
        "each horizon model consumes its corresponding permuted component"
    )
    value["cache_reuse"] = {
        "root": "/Volumes/AR_TRADING_DATA/derived/causal_day_magnitude_cache_v3",
        "law": "Reuse the completed model-free whole-prefix/whole-chain cache after verifying its immutable receipt and population; width does not enter cached tensors or labels",
        "v3_remains_history": True,
    }
    value["implementation_hashes"] = {
        str(path): file_sha256(path) for path in IMPLEMENTATIONS
    }
    value["forbidden"] = list(
        dict.fromkeys(
            [
                *value["forbidden"],
                "hidden size other than frozen width 3",
                "uncounted multi-output magnitude head",
            ]
        )
    )
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return value


def run(*, out_path: Path, v3_path: Path = DEFAULT_V3) -> dict:
    if out_path.exists():
        raise RuntimeError(f"refusing to overwrite declaration: {out_path}")
    value = build_declaration(v3_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(out_path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v3", type=Path, default=DEFAULT_V3)
    args = parser.parse_args()
    run(out_path=args.out, v3_path=args.v3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
