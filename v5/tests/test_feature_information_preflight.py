"""Phase 4a must be able to find an edge, and must not invent one."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v5.research.feature_information_preflight import (
    LIFT_BAR_PP,
    PARAMETER_CEILING,
    PLANTED_LIFT_PP,
    PROBE_FEATURES,
    PROBE_PARAMETERS,
    SELECTIONS_PER_SESSION,
    PreflightError,
    ProbeResult,
    _top_per_session,
    apply_verdict,
    cluster_bootstrap_lift,
    plant_known_answer,
    run_probe,
    wilson_interval,
)

PRIMARY_LABEL = "first_touch_50pct_before_loss_30pct_60m"


def _frame(sessions: int = 60, per_session: int = 40, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = sessions * per_session
    data = {name: rng.normal(size=rows) for name in PROBE_FEATURES}
    data["session"] = np.repeat([f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(sessions)], per_session)
    data[PRIMARY_LABEL] = (rng.random(rows) < 0.30).astype(float)
    return pd.DataFrame(data)


def _folds(frame: pd.DataFrame, splits: int = 4):
    sessions = sorted(frame["session"].unique())
    chunks = np.array_split(np.array(sessions, dtype=object), splits + 1)
    return tuple(
        (tuple(np.concatenate(chunks[:i]).tolist()), tuple(chunks[i].tolist()))
        for i in range(1, splits + 1)
    )


# ------------------------------------------------------------------ structure


def test_the_probe_stays_inside_its_declared_parameter_ceiling() -> None:
    """The ceiling is what makes a null result a statement about the features."""

    assert PROBE_PARAMETERS == len(PROBE_FEATURES) + 1 == 25
    assert PROBE_PARAMETERS <= PARAMETER_CEILING


def test_a_probe_over_the_ceiling_is_refused() -> None:
    frame = _frame(sessions=10, per_session=10)
    frame["extra_a"] = 0.0
    frame["extra_b"] = 0.0
    with pytest.raises(PreflightError, match="over the declared"):
        run_probe(
            frame,
            _folds(frame, 2),
            label_column=PRIMARY_LABEL,
            seed=1,
            name="too_wide",
            features=tuple(PROBE_FEATURES) + ("extra_a", "extra_b"),
        )


def test_a_missing_declared_feature_is_refused_rather_than_skipped() -> None:
    frame = _frame(sessions=10, per_session=10).drop(columns=["smile_residual"])
    with pytest.raises(PreflightError, match="missing declared features"):
        run_probe(frame, _folds(frame, 2), label_column=PRIMARY_LABEL, seed=1, name="short")


def test_the_operating_rate_takes_the_best_two_of_each_session() -> None:
    sessions = np.array(["a", "a", "a", "b", "b"])
    scores = np.array([0.1, 0.9, 0.5, -1.0, -2.0])
    selected = _top_per_session(sessions, scores, SELECTIONS_PER_SESSION)
    # Session "a" gives up its two best; session "b" only has two, so it gives
    # both -- the rate is a ceiling per session, not a quota to be filled.
    assert list(selected) == [False, True, True, True, True]


def test_a_session_shorter_than_the_operating_rate_is_not_padded() -> None:
    sessions = np.array(["a", "b", "b", "b"])
    scores = np.array([5.0, 1.0, 2.0, 3.0])
    selected = _top_per_session(sessions, scores, SELECTIONS_PER_SESSION)
    assert selected.sum() == 3
    assert selected[0]


# ------------------------------------------------------------------ intervals


def test_wilson_matches_a_known_value() -> None:
    # The published 95% Wilson score interval for 40/100, uncorrected.
    low, high = wilson_interval(40, 100)
    assert low == pytest.approx(0.3094, abs=1e-3)
    assert high == pytest.approx(0.4980, abs=1e-3)


def test_wilson_on_no_trials_is_nan_rather_than_a_confident_zero() -> None:
    assert all(np.isnan(v) for v in wilson_interval(0, 0))


def test_the_bootstrap_preserves_the_within_session_pairing() -> None:
    """The session is the unit -- and here that makes it tighter, not wider.

    Clustering is normally reached for because ignoring it understates an
    interval. This statistic is a within-session paired contrast (a session's own
    selections against its own base rate), so a session's regime cancels inside
    it. Resampling sessions keeps that pairing; resampling rows breaks it and
    adds variance. Asserted so nobody later "corrects" this into a row bootstrap
    on the belief that wider must mean safer.
    """

    rng = np.random.default_rng(3)
    n_sessions, per_session = 60, 10
    sessions = np.repeat([f"s{i}" for i in range(n_sessions)], per_session)
    good = rng.random(n_sessions) < 0.35
    label = np.concatenate(
        [rng.random(per_session) < (0.65 if flag else 0.10) for flag in good]
    ).astype(float)
    selected = np.zeros(len(sessions), dtype=bool)
    selected[::per_session] = True
    selected[1::per_session] = True

    low, high = cluster_bootstrap_lift(sessions, selected, label, resamples=4000, seed=1)
    observed = float(label[selected].mean() - label.mean())
    assert low <= observed <= high
    assert low < 0.0 < high, "no selection edge was planted, so zero must be inside"

    rows = np.random.default_rng(1)
    n = len(label)
    draws = []
    for _ in range(4000):
        index = rows.integers(0, n, n)
        chosen = selected[index]
        if chosen.any():
            draws.append(label[index][chosen].mean() - label[index].mean())
    row_width = float(np.quantile(draws, 0.975) - np.quantile(draws, 0.025))
    assert (high - low) < row_width


def test_the_bootstrap_declines_to_answer_on_a_single_session() -> None:
    sessions = np.array(["a", "a", "a"])
    selected = np.array([True, False, False])
    label = np.array([1.0, 0.0, 1.0])
    assert all(np.isnan(v) for v in cluster_bootstrap_lift(sessions, selected, label, seed=1))


# ------------------------------------------------------------------ behaviour


def test_the_probe_finds_a_planted_edge() -> None:
    """Power, not correctness: if it cannot find 20pp, it certifies nothing."""

    frame = _frame(sessions=120, per_session=40, seed=7)
    planted = plant_known_answer(frame, label_column=PRIMARY_LABEL, seed=11)
    result = run_probe(planted, _folds(planted), label_column=PRIMARY_LABEL, seed=5, name="planted")
    assert result.lift * 100.0 > 10.0


def test_the_probe_reports_no_edge_on_noise() -> None:
    """The control that stops a powered probe from being a lift generator."""

    frame = _frame(sessions=120, per_session=40, seed=21)
    result = run_probe(frame, _folds(frame), label_column=PRIMARY_LABEL, seed=5, name="noise")
    assert abs(result.lift) * 100.0 < 8.0
    assert result.bootstrap_lift[0] < 0.0 < result.bootstrap_lift[1]


def test_the_plant_preserves_the_base_rate_it_was_built_from() -> None:
    """A plant that also moved the base rate would flatter the lift."""

    frame = _frame(sessions=150, per_session=40, seed=13)
    planted = plant_known_answer(frame, label_column=PRIMARY_LABEL, seed=17)
    assert planted[PRIMARY_LABEL].mean() == pytest.approx(frame[PRIMARY_LABEL].mean(), abs=0.02)


def test_every_fold_trains_strictly_before_it_scores() -> None:
    frame = _frame(sessions=50, per_session=10)
    for train, holdout in _folds(frame):
        assert max(train) < min(holdout)
        assert not set(train) & set(holdout)


def test_a_fold_with_one_label_class_is_skipped_not_fitted() -> None:
    frame = _frame(sessions=40, per_session=10, seed=2)
    first = sorted(frame["session"].unique())[:8]
    frame.loc[frame["session"].isin(first), PRIMARY_LABEL] = 0.0
    result = run_probe(frame, _folds(frame), label_column=PRIMARY_LABEL, seed=1, name="degenerate")
    assert result.folds >= 1


# ------------------------------------------------------------------ the verdict


def _result(*, lift: float, upper: float, lower: float = 0.0) -> ProbeResult:
    return ProbeResult(
        label="x", rows=1000, sessions=100, folds=4, base_rate=0.30,
        precision=0.30 + lift, lift=lift, selected=200,
        wilson_precision=(0.0, 0.0), wilson_lift_upper=upper,
        bootstrap_lift=(lower, upper), lift_upper=upper, parameters=25,
    )


def test_a_powered_probe_below_the_bar_stops_the_job() -> None:
    verdict = apply_verdict(
        _result(lift=0.005, upper=0.02),
        _result(lift=PLANTED_LIFT_PP / 100.0, upper=0.25, lower=0.15),
    )
    assert verdict["verdict"] == "STOP_FEATURES_INSUFFICIENT"
    assert verdict["plant_recovered"] is True


def test_a_powered_probe_above_the_bar_proceeds() -> None:
    verdict = apply_verdict(
        _result(lift=0.05, upper=0.08),
        _result(lift=PLANTED_LIFT_PP / 100.0, upper=0.25, lower=0.15),
    )
    assert verdict["verdict"] == "PROCEED"


def test_an_unrecovered_plant_routes_the_decision_to_the_owner() -> None:
    """Not a pass and not a fail: an agent may not call this one."""

    verdict = apply_verdict(
        _result(lift=0.001, upper=0.01),
        _result(lift=0.02, upper=0.05, lower=-0.01),
    )
    assert verdict["verdict"] == "ADVISORY_UNDERPOWERED"
    assert verdict["plant_recovered"] is False


def test_the_bar_is_exactly_the_declared_one() -> None:
    """A silent edit to the bar would be the cheapest way to fake a pass."""

    declaration = json.loads(
        Path("v5/work/lifecycle-training/PHASE_4A_DECLARATION_V1.json").read_text()
    )
    assert declaration["verdict_rule"]["bar_pp"] == LIFT_BAR_PP
    assert declaration["known_answer_twin"]["planted_lift_pp"] == PLANTED_LIFT_PP
    assert declaration["probe"]["parameters"] == PROBE_PARAMETERS


def test_the_declaration_is_internally_intact() -> None:
    """Nobody edited the declaration's content after it was published."""

    import hashlib
    from v5.ops.build_causal_day_dataset import canonical_json

    source = Path("v5/work/lifecycle-training/PHASE_4A_DECLARATION_V1.json")
    declaration = json.loads(source.read_text())
    semantic = {k: v for k, v in declaration.items() if k != "declaration_sha256"}
    assert declaration["declaration_sha256"] == hashlib.sha256(canonical_json(semantic)).hexdigest()


def test_the_closed_phase_can_no_longer_be_re_run_as_the_same_experiment() -> None:
    """Phase 4a is historically sealed, and this is the mechanism that seals it.

    `implementation_hashes` records the code that produced the published result.
    It is a historical record and must never be edited to match later code. Since
    the phase closed, `lifecycle_episode_adapter.py` has legitimately moved -- a
    stale-candle defect in held batches was fixed and a prefix mode added on
    2026-08-20 -- so the declaration now refuses. That refusal is correct: it
    means nobody can re-run 4a against changed code and call it the same
    experiment. Phase 4a's own numbers are unaffected, and Phase 4b reproduced
    its headline bit-for-bit before the adapter changed.
    """

    from v5.ops.run_feature_information_preflight import verify_declaration

    source = Path("v5/work/lifecycle-training/PHASE_4A_DECLARATION_V1.json")
    with pytest.raises(PreflightError, match="has changed since the declaration"):
        verify_declaration(source)


def test_a_tampered_implementation_hash_is_refused(tmp_path: Path) -> None:
    import hashlib

    from v5.ops.build_causal_day_dataset import canonical_json
    from v5.ops.run_feature_information_preflight import verify_declaration

    source = Path("v5/work/lifecycle-training/PHASE_4A_DECLARATION_V1.json")
    tampered = json.loads(source.read_text())
    tampered["implementation_hashes"] = {
        "v5/research/feature_information_preflight.py": "0" * 64
    }
    semantic = {k: v for k, v in tampered.items() if k != "declaration_sha256"}
    tampered["declaration_sha256"] = hashlib.sha256(canonical_json(semantic)).hexdigest()
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(PreflightError, match="has changed since the declaration"):
        verify_declaration(path)


def test_a_declaration_whose_own_hash_is_wrong_is_refused(tmp_path: Path) -> None:
    from v5.ops.run_feature_information_preflight import verify_declaration

    source = Path("v5/work/lifecycle-training/PHASE_4A_DECLARATION_V1.json")
    edited = json.loads(source.read_text())
    edited["verdict_rule"]["bar_pp"] = 0.1
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(edited))
    with pytest.raises(PreflightError, match="does not match its own content"):
        verify_declaration(path)
