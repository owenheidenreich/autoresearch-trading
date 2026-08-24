from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v5.ops.measure_selector_quality_requirement import (
    SelectorRequirementRunError,
    canonical_json,
    run,
)
from v5.research import selector_quality_requirement as requirement


ROOT = Path(__file__).resolve().parents[2]
PNL_CSV = (
    ROOT
    / "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15"
    / "pnl_sweep_2026_08_23.csv"
)


def _tiny_frame(*, calibration: int = 30, later: int = 30) -> pd.DataFrame:
    calibration_sessions = pd.date_range(
        "2025-07-01", periods=calibration, freq="D"
    ).strftime("%Y-%m-%d")
    later_sessions = pd.date_range(
        "2025-08-01", periods=later, freq="D"
    ).strftime("%Y-%m-%d")
    calibration_pnl = np.linspace(-30.0, 30.0, calibration)
    later_pnl = np.linspace(-24.0, 36.0, later)
    return pd.DataFrame(
        {
            "session": [*calibration_sessions, *later_sessions],
            "start": "09:35",
            "offset": 0.0,
            "pnl": np.concatenate([calibration_pnl, later_pnl]),
            "why": "horizon",
        },
        columns=requirement.REQUIRED_COLUMNS,
    )


def _tiny_config(**overrides: object) -> requirement.StudyConfig:
    values: dict[str, object] = {
        "selector_draws": 32,
        "bootstrap_reps": 100,
        "declared_rhos": (0.0, 0.05),
        "selection_rates": (1.0,),
        "inverse_step": 0.10,
        "inverse_max": 0.50,
    }
    values.update(overrides)
    return requirement.StudyConfig(**values)


def test_strict_input_qc_refuses_population_schema_and_reserved_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-04", "2024-01-02", "2024-01-03", "2024-01-05"],
            "start": ["09:35"] * 4,
            "offset": [0.0] * 4,
            "pnl": [4.0, -2.0, 1.0, -1.0],
            "why": ["horizon", "stop", "horizon", "stop"],
        },
        columns=requirement.REQUIRED_COLUMNS,
    )
    monkeypatch.setattr(requirement, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(requirement, "EXPECTED_SESSIONS", 4)
    monkeypatch.setattr(requirement, "EXPECTED_SESSION_MIN", "2024-01-02")
    monkeypatch.setattr(requirement, "EXPECTED_SESSION_MAX", "2024-01-05")
    monkeypatch.setattr(requirement, "EXPECTED_REASON_COUNTS", {"horizon": 2, "stop": 2})
    monkeypatch.setattr(requirement, "EXPECTED_CELL_COUNTS", {("09:35", 0.0): 4})

    checked = requirement.validate_pnl_table(frame, strict_population=True)
    assert checked["session"].tolist() == sorted(frame["session"])

    with pytest.raises(requirement.SelectorRequirementError, match="row-count drift"):
        requirement.validate_pnl_table(frame.iloc[:-1], strict_population=True)
    with pytest.raises(requirement.SelectorRequirementError, match="schema drift"):
        requirement.validate_pnl_table(
            frame[["session", "start", "pnl", "offset", "why"]],
            strict_population=True,
        )
    duplicate = frame.copy()
    duplicate.loc[3, ["session", "why"]] = ["2024-01-04", "stop"]
    with pytest.raises(requirement.SelectorRequirementError, match="duplicate"):
        requirement.validate_pnl_table(duplicate, strict_population=True)
    reserved = frame.copy()
    reserved.loc[0, "session"] = requirement.RESERVATION_START
    with pytest.raises(requirement.SelectorRequirementError, match="confirmation-reserved"):
        requirement.validate_pnl_table(reserved, strict_population=True)


def test_nominal_rho_uses_common_noise_and_rho_zero_is_outcome_independent() -> None:
    rng = np.random.default_rng(7)
    z = np.linspace(-2.0, 2.0, 200)
    z = (z - z.mean()) / z.std(ddof=0)
    alternate_outcome = np.sin(np.linspace(0.0, 8.0, 200))
    noise = requirement._normal_noise(rng, draws=64, n=len(z))

    zero = requirement.construct_scores(z, noise, 0.0)
    alternate_zero = requirement.construct_scores(alternate_outcome, noise, 0.0)
    rho = 0.40
    scores = requirement.construct_scores(z, noise, rho)

    assert np.array_equal(zero, noise)
    assert np.array_equal(alternate_zero, zero)
    assert np.allclose(
        (scores - rho * z[None, :]) / np.sqrt(1.0 - rho**2),
        zero,
    )
    pearson, spearman = requirement._correlations(scores, z)
    assert pearson.mean() == pytest.approx(rho, abs=0.04)
    assert np.max(np.abs(pearson - rho)) > 0.01
    assert np.isfinite(spearman).all()


def test_fixed_source_seam_and_calibration_transform_do_not_read_later_pnl() -> None:
    config = _tiny_config()
    frame = _tiny_frame()
    context = requirement._make_context(frame, config)
    changed = frame.copy()
    changed.loc[changed["session"] >= requirement.LATER_START, "pnl"] += 10_000.0
    changed_context = requirement._make_context(changed, config)

    assert context.train_n == 30
    assert context.sessions[context.train_n - 1] == "2025-07-30"
    assert context.sessions[context.train_n] == "2025-08-01"
    assert np.array_equal(context.train_z, changed_context.train_z)
    assert np.array_equal(context.train_noise, changed_context.train_noise)
    assert context.train_mean == changed_context.train_mean
    assert context.train_std == changed_context.train_std
    assert not np.array_equal(context.later_z, changed_context.later_z)


def test_later_selection_uses_the_calibration_threshold_not_later_pnl() -> None:
    train_scores = np.asarray(
        [[0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0]]
    )
    later_scores = np.asarray([[1.9, 2.0, 4.0], [29.9, 30.0, 31.0]])
    first = requirement._selection(
        train_scores,
        later_scores,
        np.asarray([10.0, 20.0, 30.0]),
        0.50,
        in_sample=False,
    )
    changed_pnl = requirement._selection(
        train_scores,
        later_scores,
        np.asarray([-9_999.0, 20.0, 30.0]),
        0.50,
        in_sample=False,
    )

    assert np.array_equal(first.threshold, np.asarray([2.0, 30.0]))
    assert np.array_equal(first.raw_weights, np.asarray([0.0, 1.0, 1.0]))
    assert np.array_equal(changed_pnl.threshold, first.threshold)
    assert np.array_equal(changed_pnl.raw_weights, first.raw_weights)


def test_drop_best_removes_one_selected_session_without_refill() -> None:
    mask = np.asarray(
        [
            [True, True, False, False],
            [False, True, True, True],
            [False, False, True, False],
        ],
        dtype=bool,
    )
    pnl = np.asarray([100.0, 1.0, 50.0, 2.0])

    dropped = requirement._drop_best(mask, pnl)

    assert np.array_equal(
        dropped,
        np.asarray(
            [
                [False, True, False, False],
                [False, True, False, True],
                [False, False, False, False],
            ],
            dtype=bool,
        ),
    )
    assert np.array_equal(dropped.sum(axis=1), mask.sum(axis=1) - 1)
    assert not np.any(dropped & ~mask)


def test_session_bootstrap_and_weighted_metrics_keep_whole_session_units() -> None:
    first = requirement._bootstrap_counts(np.random.default_rng(11), 100, 3)
    second = requirement._bootstrap_counts(np.random.default_rng(11), 100, 3)
    assert np.array_equal(first, second)
    assert np.array_equal(first.sum(axis=1), np.full(100, 3.0))

    pnl = np.asarray([10.0, -2.0, 4.0])
    weights = np.asarray([1.0, 0.5, 0.0])
    session_counts = np.asarray(
        [[1.0, 1.0, 1.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
    )
    metrics = requirement._weighted_bootstrap(
        pnl, weights, session_counts, confidence=0.95
    )

    assert metrics["mean_pnl_per_selected_ticket"] == pytest.approx(6.0)
    assert metrics["mean_pnl_per_calendar_session"] == pytest.approx(3.0)
    assert metrics["realized_selection_rate"] == pytest.approx(0.5)
    assert metrics["profitable_share"] == pytest.approx(2.0 / 3.0)
    assert metrics["bootstrap_ticket_values"].shape == (3,)


def test_failed_rho_zero_floor_aborts_before_any_nonzero_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pnl_csv = tmp_path / "pnl.csv"
    _tiny_frame().to_csv(pnl_csv, index=False)
    real_construct = requirement.construct_scores
    constructed_rhos: list[float] = []

    def recording_construct(z: np.ndarray, noise: np.ndarray, rho: float) -> np.ndarray:
        constructed_rhos.append(rho)
        return real_construct(z, noise, rho)

    monkeypatch.setattr(requirement, "construct_scores", recording_construct)
    monkeypatch.setattr(requirement, "_sanity_row", lambda *args, **kwargs: {"passes": False})

    with pytest.raises(requirement.SelectorRequirementError, match="sanity floor failed"):
        requirement.run_analysis(
            pnl_csv,
            config=_tiny_config(declared_rhos=(0.0, 0.20)),
            strict_population=False,
        )

    assert constructed_rhos == [0.0, 0.0]


def test_public_result_labels_the_oracle_as_noncausal_and_unknown_oof(
    tmp_path: Path,
) -> None:
    pnl_csv = tmp_path / "pnl.csv"
    primary = _tiny_frame()
    qc_only_cell = primary.copy()
    qc_only_cell["start"] = "11:30"
    qc_only_cell["offset"] = 10.0
    pd.concat([primary, qc_only_cell], ignore_index=True).to_csv(pnl_csv, index=False)

    result = requirement.run_analysis(
        pnl_csv,
        config=_tiny_config(),
        strict_population=False,
    )

    assert set(result.requirement_curve["partition"]) == {
        "calibration_in_sample_oracle",
        "later_outcome_conditioned_oracle",
    }
    assert set(result.requirement_curve["start"]) == {"09:35"}
    assert set(result.requirement_curve["offset"]) == {0.0}
    assert len(result.input_manifest) == 2
    assert result.input_manifest["analyzed_for_selector_curve"].sum() == 1
    assert result.requirement_curve["constructed_score_uses_same_partition_outcome"].all()
    assert not result.requirement_curve["predictive_signal_evidence"].any()
    assert set(result.requirement_curve["genuine_oof_generalization_gap"]) == {"UNKNOWN"}
    assert result.requirement_curve["individually_executable_under_two_ticket_law"].all()
    assert set(result.requirement_curve["ticket_dollar_eligibility"]) == {
        "UNKNOWN_ASK_NOT_IN_PNL_TABLE"
    }
    assert set(result.requirement_curve["joint_cell_executability"]) == {
        "UNKNOWN_NOT_EVALUATED"
    }
    assert {
        "achieved_pearson_mean",
        "achieved_pearson_median",
        "achieved_spearman_mean",
        "achieved_spearman_median",
    }.issubset(result.correlation_diagnostics.columns)
    assert set(result.correlation_diagnostics["interval_type"]) == {
        "fixed-session selector-world construction range; not a confidence interval"
    }
    assert result.metadata["model_fit"] is False
    assert result.metadata["alpha_spent"] is False
    assert result.metadata["genuine_outcome_blind_oof"] is False
    assert result.metadata["chronology"]["later_pnl_used_to_set_threshold"] is False
    assert result.metadata["chronology"]["later_pnl_used_inside_each_later_constructed_score"] is True
    assert result.metadata["uncertainty"]["common_noise_across_rhos"] is True
    assert result.metadata["adoption"] == "ADOPT NOTHING"
    assert "No signal exists" in result.readable_tables.splitlines()[2]


def test_inverse_summary_is_the_first_sustained_simultaneous_clear() -> None:
    config = _tiny_config(
        selector_draws=64,
        bootstrap_reps=120,
        selection_rates=(0.50,),
    )
    context = requirement._make_context(_tiny_frame(), config)

    summaries, trace, critical = requirement._inverse_for_context(context, config)

    assert np.isfinite(critical)
    assert len(summaries) == 1
    rows = sorted(trace, key=lambda row: row["nominal_gaussian_rho"])
    assert all(
        row["clears_zero_after_drop_best"]
        == (row["drop_best_simultaneous_one_sided_95_lcb"] > 0.0)
        for row in rows
    )
    passes = [row["clears_zero_after_drop_best"] for row in rows]
    sustained = next(
        (index for index, passed in enumerate(passes) if passed and all(passes[index:])),
        None,
    )
    summary = summaries[0]
    assert summary["simultaneous_across_rates_and_rhos"] is True
    assert summary["genuine_outcome_blind_requirement"] == "UNKNOWN"
    if sustained is None:
        assert summary["minimum_nominal_rho_grid"] is None
        assert summary["clears_by_grid_maximum"] is False
    else:
        assert summary["minimum_nominal_rho_grid"] == rows[sustained][
            "nominal_gaussian_rho"
        ]
        assert summary["previous_nominal_rho_grid"] == (
            rows[sustained - 1]["nominal_gaussian_rho"] if sustained else None
        )
        assert all(row["clears_zero_after_drop_best"] for row in rows[sustained:])


@pytest.mark.owned_data
@pytest.mark.skipif(not PNL_CSV.is_file(), reason="owned P&L evidence is unavailable")
def test_exact_claim_reproduction_is_constructible_but_not_predictive() -> None:
    frame = requirement.load_pnl_table(PNL_CSV, strict_population=True)
    primary = frame[
        (frame["start"] == requirement.PRIMARY_CELL[0])
        & (frame["offset"] == requirement.PRIMARY_CELL[1])
    ]

    audit = requirement._claim_audit(primary, _tiny_config())
    reconstruction = audit["exact_nonunique_reconstruction"]

    assert audit["asserted_claim"]["baseline_mean_pnl"] == pytest.approx(
        2.0123416838355417
    )
    assert reconstruction["mean_pnl"] == pytest.approx(43.46890568, abs=1e-8)
    assert reconstruction["drop_best_mean_pnl"] == pytest.approx(29.03208, abs=1e-5)
    assert reconstruction["achieved_pearson"] == pytest.approx(0.05030779, abs=1e-8)
    assert reconstruction["achieved_spearman"] == pytest.approx(0.04830189, abs=1e-8)
    assert reconstruction["mean_pnl_ci_95_low"] < 0.0 < reconstruction["mean_pnl_ci_95_high"]
    assert (
        reconstruction["drop_best_mean_pnl_ci_95_low"]
        < 0.0
        < reconstruction["drop_best_mean_pnl_ci_95_high"]
    )
    assert audit["genuine_outcome_blind_oof_survival"].startswith("UNKNOWN")
    assert audit["predictive_signal_evidence"] is False
    assert audit["verdict"].startswith("VERIFIED_CONSTRUCTIBLE_BUT_REFUTED")


def test_wrapper_refuses_a_preexisting_attempt_without_mutating_it(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()
    sentinel = output_dir / "belongs-to-an-earlier-run.txt"
    sentinel.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(SelectorRequirementRunError, match="refusing to overwrite"):
        run(tmp_path / "missing.csv", output_dir, repo_root=tmp_path / "fake-repo")

    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert list(output_dir.iterdir()) == [sentinel]


def test_wrapper_refuses_a_dangling_attempt_symlink_without_mutation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "attempt"
    missing_target = tmp_path / "missing-target"
    output_dir.symlink_to(missing_target, target_is_directory=True)

    with pytest.raises(SelectorRequirementRunError, match="refusing to overwrite"):
        run(PNL_CSV, output_dir, repo_root=ROOT)

    assert output_dir.is_symlink()
    assert output_dir.readlink() == missing_target


def test_wrapper_preserves_a_self_hashed_failure_after_attempt_creation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "attempt"

    receipt = run(
        tmp_path / "missing.csv",
        output_dir,
        repo_root=tmp_path / "fake-repo",
    )

    assert receipt["status"] == "FAIL"
    assert receipt["exception_type"] == "SelectorRequirementRunError"
    assert "analysis source is missing" in receipt["exception"]
    assert (output_dir / "measure_selector_quality_requirement.py").is_file()
    assert (output_dir / "run.log").read_text(encoding="utf-8").startswith("FAIL\n")
    on_disk = json.loads(
        (output_dir / "failure_receipt.json").read_text(encoding="utf-8")
    )
    signature = on_disk.pop("receipt_sha256")
    assert signature == hashlib.sha256(canonical_json(on_disk)).hexdigest()
    assert receipt["receipt_sha256"] == signature


def test_wrapper_preserves_both_sources_when_input_qc_fails(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "attempt"

    receipt = run(tmp_path / "missing.csv", output_dir, repo_root=ROOT)

    assert receipt["status"] == "FAIL"
    assert "P&L CSV is missing" in receipt["exception"]
    assert (output_dir / "measure_selector_quality_requirement.py").is_file()
    assert (output_dir / "selector_quality_requirement.py").is_file()
    assert receipt["artifacts"]["archived_wrapper"]["sha256"]
    assert receipt["artifacts"]["archived_analysis_module"]["sha256"]
    assert receipt["artifacts"]["run_log"]["sha256"]
