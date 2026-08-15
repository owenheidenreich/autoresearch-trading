from __future__ import annotations

from copy import deepcopy

from v4.scripts import run_protocol101_entry_objective_audit as audit


CONTROLLED_METRICS = (
    "return_5m",
    "return_15m",
    "breakeven_within_15m",
)


def _metric(*, positive: bool, selected: bool = False) -> dict:
    sign = 1.0 if positive else -1.0
    payload = {
        "rank_correlation": sign * 0.02,
        "session_paired_top_minus_bottom": sign * 0.01,
        "session_paired_ci": (
            [0.001, 0.02] if positive else [-0.02, 0.01]
        ),
    }
    if selected:
        payload["seed_stability"] = {
            str(seed): {"rank_correlation": sign * 0.02}
            for seed in (42, 43, 44)
        }
    else:
        payload["fold_stability"] = {
            str(fold): {"rank_correlation": sign * 0.02}
            for fold in range(1, 6)
        }
    return payload


def _within(*, positive: bool) -> dict:
    return {
        "mean_within_decision_rank_correlation": 0.02 if positive else 0.0,
        "session_mean_ci": [0.001, 0.03] if positive else [-0.01, 0.01],
    }


def _path_results(*, controlled_rows: set[str]) -> dict:
    full_candidate_p5 = {}
    selected_intent_rows = {}
    for row_id in ("H0/P5", "H1/P5", "H2/P5", "H3/P5"):
        controlled = row_id in controlled_rows
        metrics = {
            name: _metric(positive=controlled)
            for name in CONTROLLED_METRICS
        }
        metrics["current_fixed_exit_rop"] = {
            "rank_correlation": 0.01,
        }
        full_candidate_p5[row_id] = {
            "analysis": {"metrics": metrics},
            "within_decision_controls": {
                name: _within(positive=controlled)
                for name in CONTROLLED_METRICS
            },
        }
        selected_intent_rows[row_id] = {
            "analysis": {
                "metrics": {
                    name: _metric(positive=False, selected=True)
                    for name in CONTROLLED_METRICS
                }
            }
        }
    return {
        "full_candidate_p5": full_candidate_p5,
        "selected_intent_rows": selected_intent_rows,
    }


def _economic(*, credible: bool = False) -> dict:
    return {
        "fixed_p5_heuristic_absolute_pnl": 100.0,
        "paired_model_contribution": {"credible": credible},
    }


def test_route_b_requires_controlled_candidate_signal() -> None:
    path_results = _path_results(controlled_rows={"H2/P5", "H3/P5"})

    route, evidence = audit.recommend_route(path_results, _economic())

    assert route == "Route B"
    assert evidence["controlled_predictive_hypotheses"] == 2
    assert evidence["thresholded_predictive_hypotheses"] == 0
    assert evidence["per_hypothesis"]["H2/P5"][
        "controlled_candidate_metric_pass_count"
    ] == 3


def test_route_d_freezes_profitable_p5_without_controlled_hgb_signal() -> None:
    path_results = _path_results(controlled_rows=set())

    route, evidence = audit.recommend_route(path_results, _economic())

    assert route == "Route D"
    assert evidence["controlled_predictive_hypotheses"] == 0
    assert evidence["model_increment_credible"] is False


def test_candidate_signs_alone_do_not_trigger_route_b() -> None:
    path_results = _path_results(controlled_rows={"H2/P5", "H3/P5"})
    weakened = deepcopy(path_results)
    for row_id in ("H2/P5", "H3/P5"):
        for metric_name in CONTROLLED_METRICS:
            weakened["full_candidate_p5"][row_id]["within_decision_controls"][
                metric_name
            ]["session_mean_ci"] = [-0.001, 0.03]

    route, evidence = audit.recommend_route(weakened, _economic())

    assert route == "Route D"
    assert evidence["controlled_predictive_hypotheses"] == 0


def test_route_helpers_fail_closed_on_incomplete_stability() -> None:
    metric = _metric(positive=True)
    metric["fold_stability"].pop("5")
    selected = _metric(positive=True, selected=True)
    selected["seed_stability"].pop("44")

    assert audit._all_fold_correlations_positive(metric) is False
    assert audit._all_seed_correlations_positive(selected) is False
    assert audit._strictly_positive_ci([0.0, 1.0]) is False
    assert audit._strictly_positive_ci([0.001, 1.0]) is True
