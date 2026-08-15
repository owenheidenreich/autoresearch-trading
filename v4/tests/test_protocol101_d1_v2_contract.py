from __future__ import annotations

from copy import deepcopy

from v4.model.protocol101_d1_v2_contract import (
    CANDIDATE_MULTIPLICITY_FAMILY_SHA256,
    CANDIDATE_MULTIPLICITY_ROW_IDS,
    aggregate_exposure,
    evaluate_candidate_incremental_edge,
    evaluate_d1_v2,
    exposure_match,
)


def _row(
    *,
    intents: int = 100,
    trades: int = 40,
    calls: int = 20,
    puts: int = 20,
    premium: float = 20.0,
    holding: float = 30.0,
    pnl: float = 10_000.0,
) -> dict:
    return {
        "entry_intents": intents,
        "trades": trades,
        "call_put_distribution": {"C": calls, "P": puts},
        "slot_distribution": {"10": 20, "15": 10, "20": 10},
        "executed_premium_at_risk": {"mean": premium},
        "executed_holding_minutes": {"mean": holding},
        "net_pnl": pnl,
    }


def _passing_match() -> dict:
    exposure = aggregate_exposure([_row()])
    return exposure_match(
        exposure,
        deepcopy(exposure),
        exact_opportunity_set_preserved=True,
        random_selector_receipts_complete=True,
    )


def test_positive_absolute_shuffled_pnl_is_not_a_gate() -> None:
    result = evaluate_d1_v2(
        exposure_matches=[_passing_match()],
        effect_fee_adjusted_pnl=1_000.0,
        ci_lower=-5_000.0,
        ci_upper=7_000.0,
        multiplicity_adjusted_p=0.50,
        absolute_shuffled_pnl=250_000.0,
        equivalence_bound_dollars=3_000.0,
    )
    assert result["pass"] is True
    assert result["absolute_pnl_is_a_gate"] is False


def test_positive_adjusted_shuffled_increment_fails() -> None:
    result = evaluate_d1_v2(
        exposure_matches=[_passing_match()],
        effect_fee_adjusted_pnl=25_000.0,
        ci_lower=5_000.0,
        ci_upper=45_000.0,
        multiplicity_adjusted_p=0.01,
        absolute_shuffled_pnl=250_000.0,
        equivalence_bound_dollars=3_000.0,
    )
    assert result["pass"] is False
    assert result["status"] == "POSITIVE_SHUFFLED_MODEL_ARTIFACT"


def test_exposure_mismatch_fails_closed_without_claiming_artifact() -> None:
    left = aggregate_exposure([_row()])
    right = aggregate_exposure([_row(trades=20, calls=18, puts=2)])
    mismatch = exposure_match(
        left,
        right,
        exact_opportunity_set_preserved=False,
        random_selector_receipts_complete=False,
    )
    result = evaluate_d1_v2(
        exposure_matches=[mismatch],
        effect_fee_adjusted_pnl=0.0,
        ci_lower=-1.0,
        ci_upper=1.0,
        multiplicity_adjusted_p=1.0,
        absolute_shuffled_pnl=100_000.0,
        equivalence_bound_dollars=3_000.0,
    )
    assert result["pass"] is False
    assert result["status"] == "INSUFFICIENT_MATCHED_CONTROL"
    assert result["positive_artifact_detected"] is False


def test_three_dollar_equivalence_is_diagnostic_only() -> None:
    common = dict(
        exposure_matches=[_passing_match()],
        effect_fee_adjusted_pnl=0.0,
        ci_lower=-50_000.0,
        ci_upper=50_000.0,
        multiplicity_adjusted_p=1.0,
        absolute_shuffled_pnl=200_000.0,
    )
    narrow = evaluate_d1_v2(**common, equivalence_bound_dollars=3.0)
    wide = evaluate_d1_v2(**common, equivalence_bound_dollars=1_000_000.0)
    assert narrow["pass"] == wide["pass"] is True
    assert narrow["equivalence_diagnostic"]["changes_gate"] is False


def test_real_candidate_requires_strictly_positive_lower_ci_and_adjusted_p() -> None:
    passed = evaluate_candidate_incremental_edge(
        row_id="H2/P5",
        exposure_matches=[_passing_match()],
        effect_fee_adjusted_pnl=10_000.0,
        ci_lower=1.0,
        ci_upper=20_000.0,
        multiplicity_adjusted_p=0.05,
        multiplicity_family_size=28,
        multiplicity_family_ids=CANDIDATE_MULTIPLICITY_ROW_IDS,
        multiplicity_family_sha256=CANDIDATE_MULTIPLICITY_FAMILY_SHA256,
        multiplicity_receipt_complete=True,
        costs_included=True,
        evidence_complete=True,
    )
    zero = evaluate_candidate_incremental_edge(
        row_id="H2/P5",
        exposure_matches=[_passing_match()],
        effect_fee_adjusted_pnl=10_000.0,
        ci_lower=0.0,
        ci_upper=20_000.0,
        multiplicity_adjusted_p=0.05,
        multiplicity_family_size=28,
        multiplicity_family_ids=CANDIDATE_MULTIPLICITY_ROW_IDS,
        multiplicity_family_sha256=CANDIDATE_MULTIPLICITY_FAMILY_SHA256,
        multiplicity_receipt_complete=True,
        costs_included=True,
        evidence_complete=True,
    )
    unadjusted = evaluate_candidate_incremental_edge(
        row_id="H2/P5",
        exposure_matches=[_passing_match()],
        effect_fee_adjusted_pnl=10_000.0,
        ci_lower=1.0,
        ci_upper=20_000.0,
        multiplicity_adjusted_p=0.051,
        multiplicity_family_size=28,
        multiplicity_family_ids=CANDIDATE_MULTIPLICITY_ROW_IDS,
        multiplicity_family_sha256=CANDIDATE_MULTIPLICITY_FAMILY_SHA256,
        multiplicity_receipt_complete=True,
        costs_included=True,
        evidence_complete=True,
    )
    assert passed["pass"] is True
    assert zero["pass"] is False
    assert unadjusted["pass"] is False


def test_real_candidate_requires_complete_28_row_multiplicity_receipt() -> None:
    result = evaluate_candidate_incremental_edge(
        row_id="H2/P5",
        exposure_matches=[_passing_match()],
        effect_fee_adjusted_pnl=10_000.0,
        ci_lower=1.0,
        ci_upper=20_000.0,
        multiplicity_adjusted_p=0.01,
        multiplicity_family_size=5,
        multiplicity_family_ids=CANDIDATE_MULTIPLICITY_ROW_IDS[:5],
        multiplicity_family_sha256=CANDIDATE_MULTIPLICITY_FAMILY_SHA256,
        multiplicity_receipt_complete=True,
        costs_included=True,
        evidence_complete=True,
    )
    assert result["pass"] is False
    assert "campaign_multiplicity_receipt_incomplete" in result["blockers"]


def test_real_candidate_rejects_unhashed_multiplicity_family_identity() -> None:
    result = evaluate_candidate_incremental_edge(
        row_id="H2/P5",
        exposure_matches=[_passing_match()],
        effect_fee_adjusted_pnl=10_000.0,
        ci_lower=1.0,
        ci_upper=20_000.0,
        multiplicity_adjusted_p=0.01,
        multiplicity_family_size=28,
        multiplicity_family_ids=CANDIDATE_MULTIPLICITY_ROW_IDS,
        multiplicity_family_sha256="0" * 64,
        multiplicity_receipt_complete=True,
        costs_included=True,
        evidence_complete=True,
    )
    assert result["pass"] is False
    assert "campaign_multiplicity_receipt_incomplete" in result["blockers"]


def test_missing_candidate_specific_evidence_fails_closed() -> None:
    result = evaluate_candidate_incremental_edge(
        row_id="H0/P5",
        exposure_matches=[],
        effect_fee_adjusted_pnl=None,
        ci_lower=None,
        ci_upper=None,
        multiplicity_adjusted_p=None,
        multiplicity_family_size=None,
        multiplicity_family_ids=None,
        multiplicity_family_sha256=None,
        multiplicity_receipt_complete=False,
        costs_included=True,
        evidence_complete=False,
    )
    assert result["pass"] is False
    assert "incremental_evidence_incomplete" in result["blockers"]
    assert "matched_random_exposure_incomplete" in result["blockers"]
