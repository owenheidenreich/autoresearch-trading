#!/usr/bin/env python3
"""T6 regression: FT2-08 validator, terminal fixtures, and SE fixture."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FT208 = (
    REPO
    / "v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract"
)
FT211 = (
    REPO
    / "v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract"
)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected object: {path}")
    return payload


def load_checker() -> Any:
    path = HERE / "check_cross_contract_consistency_v3.py"
    spec = importlib.util.spec_from_file_location("consistency_v3_t6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "t6_regression_output.json",
    )
    args = parser.parse_args()
    process = subprocess.run(
        [sys.executable, str(FT208 / "validate_contract_v3.py")],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    validation = load(FT208 / "validation.json")
    checker = load_checker()

    terminal_fixture = load(FT211 / "worked_terminal_examples.json")
    terminal_results: list[dict[str, Any]] = []
    for example in terminal_fixture["examples"]:
        terminal, triggered = checker.terminal_resolve(example)
        terminal_results.append(
            {
                "id": example["id"],
                "observed_terminal": terminal,
                "expected_terminal": example["expected_terminal"],
                "observed_predicates": triggered,
                "expected_predicates": example.get(
                    "expected_triggered_predicates", []
                ),
                "pass": (
                    terminal == example["expected_terminal"]
                    and triggered
                    == example.get("expected_triggered_predicates", [])
                ),
            }
        )

    bootstrap = load(FT211 / "bootstrap_spec.json")
    fixture = bootstrap["worked_studentization_example"]
    observed = list(map(float, fixture["observed_series"]))
    _, _, observed_variance = checker.delete_block_values(
        observed,
        int(fixture["block_length_L"]),
    )
    observed_se = math.sqrt(observed_variance)
    observed_t = sum(observed) / len(observed) / observed_se
    centered = list(map(float, fixture["centered_outer_resample"]))
    _, _, star_variance = checker.delete_block_values(
        centered,
        int(fixture["block_length_L"]),
    )
    star_se = math.sqrt(star_variance)
    star_t = sum(centered) / len(centered) / star_se
    se_pass = all(
        (
            math.isclose(
                observed_se,
                float(fixture["SE_h"]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ),
            math.isclose(
                observed_t,
                float(fixture["observed_t_h"]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ),
            math.isclose(
                star_se,
                float(fixture["SE_h_star_b"]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ),
            math.isclose(
                star_t,
                float(fixture["t_h_star_b"]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ),
        )
    )
    assertions = {
        "FT2_08_validator_process_exit_zero": process.returncode == 0,
        "FT2_08_exactly_22_checks_green": (
            validation["passed"] is True
            and len(validation["checks"]) == 22
            and all(validation["checks"].values())
        ),
        "three_terminal_examples_recomputed": (
            len(terminal_results) == 3
            and all(item["pass"] for item in terminal_results)
        ),
        "SE_worked_example_recomputed": se_pass,
    }
    payload = {
        "schema_version": "Protocol101FT2ScopedRoundRegressionT6OutputV1",
        "assertions": assertions,
        "ft2_08_validation_schema": validation["schema_version"],
        "ft2_08_check_count": len(validation["checks"]),
        "ft2_08_stdout": process.stdout.strip(),
        "ft2_08_stderr": process.stderr.strip(),
        "terminal_results": terminal_results,
        "SE_fixture": {
            "SE_h": observed_se,
            "t_h": observed_t,
            "SE_h_star_b": star_se,
            "t_h_star_b": star_t,
        },
        "outcome": "pass" if all(assertions.values()) else "fail",
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "outcome": payload["outcome"],
                "assertions": assertions,
            }
        )
    )
    return 0 if payload["outcome"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
