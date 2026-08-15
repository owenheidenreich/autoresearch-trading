from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts import (
    run_protocol101_d1_shuffled_profit_attribution as attribution,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_d1_shuffled_profit_causal_attribution_attempt009"
)
OUTPUT_ROOT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_d1_shuffled_profit_causal_attribution_"
    "inference_correction_attempt011"
)


class CorrectionError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_source_hashes() -> int:
    hash_path = SOURCE_ROOT / "hashes.sha256"
    lines = [
        line.strip()
        for line in hash_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for line in lines:
        digest, relative = line.split("  ", maxsplit=1)
        path = SOURCE_ROOT / relative
        if not path.is_file():
            raise CorrectionError(f"missing source artifact:{relative}")
        if attribution.sha256_path(path) != digest:
            raise CorrectionError(f"source hash mismatch:{relative}")
    return len(lines)


def _paired_mean_contrast(
    sessions: pd.DataFrame,
    *,
    left: str,
    right: str,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    keys = ["model_seed", "fold", "session"]
    left_mean = (
        sessions[
            (sessions["control"] == left)
            & sessions["model_seed"].isin(seeds)
        ]
        .groupby(keys, as_index=False)["net_pnl"]
        .mean()
        .rename(columns={"net_pnl": "left_pnl"})
    )
    right_mean = (
        sessions[
            (sessions["control"] == right)
            & sessions["model_seed"].isin(seeds)
        ]
        .groupby(keys, as_index=False)["net_pnl"]
        .mean()
        .rename(columns={"net_pnl": "right_pnl"})
    )
    if left_mean.duplicated(keys).any() or right_mean.duplicated(keys).any():
        raise CorrectionError(f"duplicate paired mean axis:{left}:{right}")
    left_keys = set(
        map(tuple, left_mean[keys].itertuples(index=False, name=None))
    )
    right_keys = set(
        map(tuple, right_mean[keys].itertuples(index=False, name=None))
    )
    if left_keys != right_keys:
        raise CorrectionError(f"paired mean session loss:{left}:{right}")
    paired = left_mean.merge(
        right_mean,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(left_mean):
        raise CorrectionError(f"paired mean row loss:{left}:{right}")
    paired["difference"] = paired["left_pnl"] - paired["right_pnl"]
    return paired


def _corrected_contrasts(
    sessions: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], str]:
    shared_indexes = attribution._shared_bootstrap_indexes(
        int(sessions["session"].nunique())
    )
    shared_indexes_sha256 = hashlib.sha256(
        shared_indexes.tobytes(order="C")
    ).hexdigest()
    contrast_specs = {
        "F_minus_C_model_slot_beyond_random_at_model_times": (
            "F",
            "C",
            attribution.D1_SEEDS,
            True,
        ),
        "F_minus_E_threshold_optimization": (
            "F",
            "E",
            attribution.D1_SEEDS,
            False,
        ),
        "F_minus_E_common_threshold_budget_choice": (
            "F",
            "E_common",
            attribution.D1_SEEDS,
            False,
        ),
        "F_minus_M_F_current_D1_margin_matched_increment": (
            "F",
            "M_F",
            attribution.D1_SEEDS,
            True,
        ),
        "E_minus_G_weak_shuffle_residual_structure": (
            "E",
            "G",
            attribution.D1_SEEDS,
            False,
        ),
        "E_common_minus_G_common_weak_shuffle_structure": (
            "E_common",
            "G_common",
            attribution.D1_SEEDS,
            False,
        ),
        "G_minus_C_G_strong_shuffle_slot_residual": (
            "G",
            "C_G",
            attribution.D1_SEEDS,
            True,
        ),
        "G_minus_B_strong_shuffle_total_increment": (
            "G",
            "B",
            attribution.D1_SEEDS,
            True,
        ),
        "G_minus_M_G_valid_negative_control_increment": (
            "G",
            "M_G",
            attribution.D1_SEEDS,
            True,
        ),
        "H_minus_B_real_total_real_model_increment": (
            "H",
            "B_real",
            attribution.REAL_SEEDS,
            True,
        ),
        "H_minus_C_real_real_slot_increment": (
            "H",
            "C_real",
            attribution.REAL_SEEDS,
            True,
        ),
        "H_minus_M_real_margin_matched_real_increment": (
            "H",
            "M_real",
            attribution.REAL_SEEDS,
            True,
        ),
        "H_minus_I_real_score_direction": (
            "H",
            "I",
            attribution.REAL_SEEDS,
            False,
        ),
    }
    contrasts: dict[str, dict[str, Any]] = {}
    for name, spec in contrast_specs.items():
        paired = attribution._session_contrast(
            sessions,
            left=spec[0],
            right=spec[1],
            seeds=spec[2],
            random_right=spec[3],
        )
        contrasts[name] = attribution._paired_block_bootstrap(
            paired,
            bootstrap_indexes=shared_indexes,
        )

    for name, left, right, seeds in (
        (
            "C_minus_B_timing_given_random_slots",
            "C",
            "B",
            attribution.D1_SEEDS,
        ),
        (
            "D_minus_B_model_like_contract_profile",
            "D",
            "B",
            attribution.D1_SEEDS,
        ),
        (
            "C_G_minus_B_strong_shuffle_timing_residual",
            "C_G",
            "B",
            attribution.D1_SEEDS,
        ),
        (
            "D_G_minus_B_strong_shuffle_contract_profile_residual",
            "D_G",
            "B",
            attribution.D1_SEEDS,
        ),
        (
            "C_real_minus_B_real_real_timing_increment",
            "C_real",
            "B_real",
            attribution.REAL_SEEDS,
        ),
    ):
        paired = _paired_mean_contrast(
            sessions,
            left=left,
            right=right,
            seeds=seeds,
        )
        contrasts[name] = attribution._paired_block_bootstrap(
            paired,
            bootstrap_indexes=shared_indexes,
        )

    keys = ["model_seed", "fold", "session"]
    means = (
        sessions[
            sessions["control"].isin(["B", "C", "D", "F"])
            & sessions["model_seed"].isin(attribution.D1_SEEDS)
        ]
        .groupby(["control", *keys], as_index=False)["net_pnl"]
        .mean()
    )
    pivot = means.pivot_table(
        index=keys,
        columns="control",
        values="net_pnl",
    ).reset_index()
    if (
        set(("B", "C", "D", "F")) - set(pivot.columns)
        or pivot[["B", "C", "D", "F"]].isna().any().any()
        or len(pivot)
        != len(attribution.D1_SEEDS) * len(attribution.FOLDS) * 45
    ):
        raise CorrectionError("difference-in-differences grid incomplete")
    pivot["difference"] = pivot["F"] - pivot["C"] - pivot["D"] + pivot["B"]
    contrasts["F_minus_C_minus_D_plus_B_interaction"] = (
        attribution._paired_block_bootstrap(
            pivot,
            bootstrap_indexes=shared_indexes,
        )
    )
    if set(contrasts) != set(attribution.PRIMARY_CONTRASTS):
        raise CorrectionError("corrected contrast family differs")
    attribution._holm_adjust(contrasts)
    return contrasts, shared_indexes_sha256


def _write_report(
    causal: dict[str, Any],
    trust: dict[str, Any],
    old_paired: dict[str, Any],
    corrected_paired: dict[str, Any],
) -> None:
    old_strong = old_paired["primary_contrasts"][
        "G_minus_B_strong_shuffle_total_increment"
    ]
    new_strong = corrected_paired["primary_contrasts"][
        "G_minus_B_strong_shuffle_total_increment"
    ]
    lines = [
        "# Protocol101 D1 Attribution Inference Correction",
        "",
        f"Terminal route: `{trust['status']}`",
        "",
        "## Corrected Answer",
        "",
        causal["causal_conclusion"],
        "",
        "## What Changed",
        "",
        "- The 1,734,075-row frozen session ledger and every model/control "
        "output from attempt009 were reused byte-for-byte.",
        "- All 19 contrasts now use one shared 20,000-replicate moving-block "
        "bootstrap index matrix, as preregistered.",
        "- The mislabeled G-minus-M_G component now uses its actual contrast.",
        "- The report distinguishes a failed equivalence certificate from a "
        "statistically demonstrated positive increment.",
        "",
        "## Stability",
        "",
        f"- Original G-minus-B interval: "
        f"[${float(old_strong['ci_p2_5']):,.2f}, "
        f"${float(old_strong['ci_p97_5']):,.2f}].",
        f"- Corrected G-minus-B interval: "
        f"[${float(new_strong['ci_p2_5']):,.2f}, "
        f"${float(new_strong['ci_p97_5']):,.2f}].",
        f"- Revised D1 pass: "
        f"`{str(trust['D1_revised_negative_control_pass']).lower()}`.",
        f"- Real H2/P5 increment credible: "
        f"`{str(trust['real_H2_P5_increment_credible']).lower()}`.",
        "",
        "## Safety",
        "",
        "- No campaign model was refit.",
        "- No threshold, replay, economics, or control stream changed.",
        "- No G9, protected holdout, HOLD/EXIT, broker, paper, or live work "
        "occurred.",
        "",
    ]
    (OUTPUT_ROOT / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise CorrectionError(f"output already exists:{OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    source_hash_count = _verify_source_hashes()
    old_paired = _read_json(SOURCE_ROOT / "paired_control_results.json")
    implementation = _read_json(SOURCE_ROOT / "implementation_audit.json")
    session_path = SOURCE_ROOT / "session_level_results.parquet"
    if (
        attribution.sha256_path(session_path)
        != old_paired["session_level_results"]["sha256"]
    ):
        raise CorrectionError("session ledger hash mismatch")
    sessions = pd.read_parquet(session_path)
    corrected_paired = copy.deepcopy(old_paired)
    corrected_paired["schema_version"] = (
        "Protocol101D1PairedControlResultsV1SharedBootstrapCorrection"
    )
    (
        corrected_paired["primary_contrasts"],
        shared_indexes_sha256,
    ) = _corrected_contrasts(sessions)
    corrected_paired["correction"] = {
        "source_attempt": str(SOURCE_ROOT.relative_to(ROOT)),
        "source_hash_count": source_hash_count,
        "source_hashes_sha256": attribution.sha256_path(
            SOURCE_ROOT / "hashes.sha256"
        ),
        "session_ledger_reused_byte_identically": True,
        "models_refit": False,
        "controls_replayed": False,
        "shared_resample_across_all_19_contrasts": True,
    }
    corrected_paired["results_sha256"] = attribution.stable_hash(
        {
            key: value
            for key, value in corrected_paired.items()
            if key != "results_sha256"
        }
    )
    causal = attribution._causal_attribution(
        corrected_paired,
        implementation,
    )
    revised = attribution._revised_d1_specification(corrected_paired)
    trust = attribution._entry_trust_decision(corrected_paired, causal)
    mechanical_validation = {
        "schema_version": "Protocol101D1MechanicalValidationV1",
        "status": "complete_pending_external_review",
        "findings_repaired": [
            "shared bootstrap matrix was not used across contrasts",
            "G-minus-M_G component referenced G-minus-B",
            "failed equivalence was narrated as demonstrated material difference",
        ],
        "producer_hashes": {
            "attribution_runner_sha256": attribution.sha256_path(
                ROOT
                / "v4/scripts/"
                "run_protocol101_d1_shuffled_profit_attribution.py"
            ),
            "correction_script_sha256": attribution.sha256_path(Path(__file__)),
            "focused_test_sha256": attribution.sha256_path(
                ROOT
                / "v4/tests/"
                "test_protocol101_d1_shuffled_profit_attribution.py"
            ),
            "shared_bootstrap_index_matrix_sha256": (
                shared_indexes_sha256
            ),
        },
        "source_control_grid_complete": True,
        "source_session_rows": int(len(sessions)),
        "source_duplicate_session_axes": int(
            sessions.duplicated(
                ["control", "model_seed", "draw", "fold", "session"]
            ).sum()
        ),
    }
    mechanical_validation["validation_sha256"] = attribution.stable_hash(
        mechanical_validation
    )
    plan = {
        "schema_version": "Protocol101D1InferenceCorrectionPlanV1",
        "status": "complete",
        "scope": "derived inference and reporting only",
        "source_attempt": str(SOURCE_ROOT.relative_to(ROOT)),
        "source_hashes_sha256": attribution.sha256_path(
            SOURCE_ROOT / "hashes.sha256"
        ),
        "frozen_session_ledger_sha256": attribution.sha256_path(session_path),
        "shared_bootstrap_seed": "random_generator(9,0)",
        "bootstrap_replicates": attribution.BOOTSTRAP_REPLICATES,
        "bootstrap_block_sessions": attribution.BOOTSTRAP_BLOCK_SESSIONS,
        "contrast_count": len(attribution.PRIMARY_CONTRASTS),
        "producer_hashes": mechanical_validation["producer_hashes"],
        "no_economic_or_model_recomputation": True,
    }
    plan["plan_sha256"] = attribution.stable_hash(plan)

    _write_json(OUTPUT_ROOT / "correction_plan.json", plan)
    _write_json(
        OUTPUT_ROOT / "paired_control_results_corrected.json",
        corrected_paired,
    )
    _write_json(
        OUTPUT_ROOT / "causal_attribution_corrected.json",
        causal,
    )
    _write_json(
        OUTPUT_ROOT / "revised_d1_specification_corrected.json",
        revised,
    )
    _write_json(
        OUTPUT_ROOT / "entry_model_trust_decision_corrected.json",
        trust,
    )
    _write_json(
        OUTPUT_ROOT / "mechanical_validation.json",
        mechanical_validation,
    )
    _write_report(causal, trust, old_paired, corrected_paired)

    hash_lines = []
    for path in sorted(OUTPUT_ROOT.iterdir()):
        if path.is_file() and path.name != "hashes.sha256":
            hash_lines.append(
                f"{attribution.sha256_path(path)}  {path.name}"
            )
    (OUTPUT_ROOT / "hashes.sha256").write_text(
        "\n".join(hash_lines) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
