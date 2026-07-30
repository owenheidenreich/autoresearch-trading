#!/usr/bin/env python3
"""T3/T4 reconciliation and D48 transition-audit completeness tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CENSUS = (
    REPO
    / "v4/audit/autoresearch/protocol101_ft2_05_opportunity_census"
)
LAW = (
    REPO
    / "v4/audit/autoresearch/"
    "protocol101_ft2_08_data_tensor_label_contract/"
    "intent_fill_recheck_law.json"
)
LAW_HASH = (
    "5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0"
)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "t3_t4_census_v4_output.json",
    )
    args = parser.parse_args()
    result = load(CENSUS / "census_results.json")
    impact = load(CENSUS / "v3_v4_impact.json")
    audit = load(CENSUS / "d48_reference_transition_audit.json")
    row_audit_path = CENSUS / "d48_reference_transition_audit.csv"
    row_audit = pd.read_csv(
        row_audit_path,
        usecols=[
            "session",
            "decision_time_ns",
            "contract_id",
            "transition_v2_v3",
            "transition_v3_v4",
        ],
    )

    rejection = result["fill_recheck_rejection_denominators"]
    numerator = int(rejection["numerator"]["row_count"])
    governed_denominator = int(rejection["governed_rows"]["denominator"])
    conditional_denominator = int(
        rejection["conditional_on_reaching_recheck"]["denominator"]
    )
    transition = audit["transition_counts"]
    reason_counts = audit["v2_to_v3_removed_reason_counts"]
    reconciliation = audit["reconciliation"]

    t3 = {
        "canonical_law_hash": (
            sha256(LAW) == LAW_HASH
            and result["input_contracts"][
                "intent_fill_recheck_law_sha256"
            ]
            == LAW_HASH
        ),
        "exact_session_and_governed_row_counts": (
            int(result["session_count"]) == 45
            and int(result["label_rows_all_governed_candidates"])
            == int(audit["governed_row_count"])
            == 460_937
        ),
        "v4_reference_universe_equals_time_t_intent": (
            int(result["label_rows_d48_reference"])
            == int(result["intent_eligible_fee3_rows"])
            == 128_758
            and impact["expected_identity"][
                "v4_d48_reference_equals_time_t_intent_count"
            ]
            is True
        ),
        "v3_v4_impact_complete": (
            int(
                impact["scalar_changes"]["label_rows_d48_reference"][
                    "v3"
                ]
            )
            == 121_553
            and int(
                impact["scalar_changes"]["label_rows_d48_reference"][
                    "v4"
                ]
            )
            == 128_758
            and int(
                impact["scalar_changes"]["label_rows_d48_reference"][
                    "v4_minus_v3"
                ]
            )
            == 7_205
            and len(impact["semantic_changes"]) >= 4
            and impact["expected_identity"][
                "v3_to_v4_reference_delta_equals_recheck_rejections"
            ]
            is True
        ),
        "both_rejection_denominators_exact": (
            numerator == 7_205
            and governed_denominator == 460_937
            and conditional_denominator == 128_758
            and math.isclose(
                float(rejection["governed_rows"]["rate"]),
                7_205 / 460_937,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                float(
                    rejection[
                        "conditional_on_reaching_recheck"
                    ]["rate"]
                ),
                7_205 / 128_758,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ),
    }
    t4 = {
        "row_audit_identity_complete_and_unique": (
            len(row_audit) == 460_937
            and len(
                row_audit[
                    ["session", "decision_time_ns", "contract_id"]
                ].drop_duplicates()
            )
            == 460_937
            and int(audit["unique_identity_count"]) == 460_937
        ),
        "each_transition_step_sums_exactly": (
            sum(map(int, transition["v2_to_v3"].values())) == 460_937
            and sum(map(int, transition["v3_to_v4"].values())) == 460_937
            and reconciliation["each_step_sums_to_governed_rows"]
            is True
        ),
        "v2_v3_all_removed_rows_accounted": (
            int(transition["v2_to_v3"]["removed"]) == 37_759
            and int(reason_counts["time_t_intent_ineligible"]) == 34_862
            and int(
                reason_counts[
                    "time_t_intent_eligible_tplus1_recheck_rejected"
                ]
            )
            == 2_897
            and reconciliation["every_removed_row_accounted"] is True
        ),
        "aggregate_identity_and_overlap_distinguished": (
            int(
                reconciliation[
                    "v2_minus_v4_net_intent_universe_change"
                ]
            )
            == 30_554
            and int(
                reconciliation["v4_minus_v3_recheck_population"]
            )
            == 7_205
            and reconciliation[
                "aggregate_identity_v2_minus_v3_equals_net_intent_plus_recheck"
            ]
            is True
            and int(
                audit["three_version_membership_overlap"][
                    "v2_0_v3_0_v4_1"
                ]
            )
            == 4_308
            and int(
                audit["three_version_membership_overlap"][
                    "v2_1_v3_0_v4_1"
                ]
            )
            == 2_897
        ),
        "v3_v4_restores_exact_rejection_population": (
            int(transition["v3_to_v4"]["added"]) == 7_205
            and int(transition["v3_to_v4"]["removed"]) == 0
            and reconciliation[
                "v3_to_v4_additions_equal_v4_recheck_rejections"
            ]
            is True
        ),
        "economic_and_family_distributions_nonempty": all(
            int(
                audit["distributions"][step][cohort]["row_count"]
            )
            > 0
            and any(
                int(item["valid_count"]) > 0
                for item in audit["distributions"][step][cohort][
                    "economic"
                ].values()
            )
            and any(
                int(item["valid_count"]) > 0
                for item in audit["distributions"][step][cohort][
                    "label_families"
                ].values()
            )
            and any(
                audit["distributions"][step][cohort][
                    "categorical"
                ][dimension]
                for dimension in (
                    "premium_band",
                    "moneyness_band",
                    "market_phase",
                )
            )
            for step, cohort in (
                ("v2_to_v3", "removed"),
                ("v2_to_v3", "retained"),
                ("v3_to_v4", "added"),
                ("v3_to_v4", "retained"),
            )
        ),
    }
    assertions = {**{f"T3_{k}": v for k, v in t3.items()}, **{f"T4_{k}": v for k, v in t4.items()}}
    payload = {
        "schema_version": "Protocol101FT2ScopedRoundCensusV4TestOutputV1",
        "T3": t3,
        "T4": t4,
        "assertion_count": len(assertions),
        "failed_assertions": sorted(
            key for key, passed in assertions.items() if not passed
        ),
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
                "failed_assertions": payload["failed_assertions"],
            }
        )
    )
    return 0 if payload["outcome"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
