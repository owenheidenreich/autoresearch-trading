from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from v4.research.lean_autoresearch import harness as H
from v4.research.lean_autoresearch import search_s4_s5 as S


def _synthetic_context() -> dict:
    names = ("spx_close", "vix_close", "session_range")
    spx = 5000.0 * np.exp(np.arange(30, dtype=float) * 0.001)
    vix = 15.0 + np.arange(30, dtype=float) * 0.1
    session_range = 20.0 + np.arange(30, dtype=float)
    return {
        "decision_time": "2026-02-03T15:00:00+00:00",
        "source_context_time": "2026-02-03T15:00:00+00:00",
        "market_feature_names": names,
        "market_window": np.column_stack([spx, vix, session_range]),
        "future_label": 999_999.0,
    }


def _fold(
    *,
    rho: float,
    econ: float,
    shuffled_rho: float = 0.0,
    shuffled_econ: float = -1.0,
    signrev_econ: float = -2.0,
) -> dict:
    return {
        "oof_rho": rho,
        "econ": econ,
        "shuffled_rho": shuffled_rho,
        "shuffled_econ": shuffled_econ,
        "signrev_econ": signrev_econ,
        "n_enter": 100,
        "n_sessions": 25,
    }


def test_s4_s5_feature_composition_and_exact_causal_formulas() -> None:
    context = _synthetic_context()
    observed = H.causal_volatility_time_features(context)
    spx = context["market_window"][:, 0]
    ranges = context["market_window"][:, 2]

    assert H.FEATURE_SETS["S4"] == H.SIGNED17 + H.VOLATILITY_TIME
    assert H.FEATURE_SETS["S5"] == (
        H.SIGNED17 + H.MICROSTRUCTURE_GREEKS + H.VOLATILITY_TIME
    )
    assert observed["vix_level"] == pytest.approx(17.9)
    assert observed["vix_change_5m"] == pytest.approx(0.5)
    assert observed["vix_change_15m"] == pytest.approx(1.5)
    assert observed["minute_of_session"] == 30.0
    assert observed["minutes_to_1555"] == 355.0
    assert observed["spx_rv_5m_annualized_pct"] == pytest.approx(
        np.sqrt(0.001**2 * 252 * 390) * 100
    )
    assert observed["spx_rv_15m_annualized_pct"] == pytest.approx(
        np.sqrt(0.001**2 * 252 * 390) * 100
    )
    assert observed["vix_minus_spx_rv15_pct_points"] == pytest.approx(
        17.9 - observed["spx_rv_15m_annualized_pct"]
    )
    assert observed["range_expansion_5m_bps"] == pytest.approx(
        (ranges[-1] - ranges[-6]) / spx[-1] * 1e4
    )
    assert observed["range_expansion_15m_bps"] == pytest.approx(
        (ranges[-1] - ranges[-16]) / spx[-1] * 1e4
    )


def test_mutate_future_and_labels_cannot_change_s4_s5_features() -> None:
    context = _synthetic_context()
    expected = H.causal_volatility_time_features(context)
    mutated = dict(context)
    mutated["future_label"] = -999_999.0
    mutated["labels_net_pnl"] = np.full((21, 2, 7), 1e12)
    mutated["future_best_exit"] = "oracle"
    assert H.causal_volatility_time_features(mutated) == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"source_context_time": "2026-02-03T15:00:01+00:00"}, "future context"),
        ({"decision_time": "2026-02-03T15:00:00"}, "timezone-aware"),
        ({"decision_time": "2026-02-03T21:00:00+00:00"}, "outside"),
    ],
)
def test_s4_s5_context_fails_closed_on_clock_violations(mutation, message) -> None:
    context = _synthetic_context()
    context.update(mutation)
    with pytest.raises(ValueError, match=message):
        H.causal_volatility_time_features(context)


def test_official_context_applies_v32_bar_open_plus_60(monkeypatch) -> None:
    times = pd.date_range("2026-02-03T14:30:00Z", periods=31, freq="min")
    spx = pd.DataFrame({"event_time": times, "close": 5000.0 + np.arange(31)})
    vix = pd.DataFrame({"event_time": times, "close": 18.0 + np.arange(31) / 10})

    def fake_frame(session: str, symbol: str) -> pd.DataFrame:
        assert session == "2026-02-03"
        return spx if symbol == "SPX" else vix

    monkeypatch.setattr(H, "_official_index_frame", fake_frame)
    entry = {
        "decision_time": "2026-02-03T15:00:00Z",
        "source_context_time": "2026-02-03T15:00:00Z",
    }
    context = H._official_volatility_time_context(entry, session="2026-02-03")
    assert context["decision_time"] == pd.Timestamp("2026-02-03T15:01:00Z")
    assert context["source_context_time"] == pd.Timestamp("2026-02-03T15:01:00Z")
    assert context["market_window"][-1, 0] == 5030.0
    assert context["market_window"][-1, 1] == 21.0
    assert H.causal_volatility_time_features(context)["minute_of_session"] == 31.0


def test_grid_uses_exact_fixed_budget_and_unique_paired_controls() -> None:
    candidates = S.build_candidate_grid()
    controls = S.build_control_grid()
    assert len(candidates) == S.CANDIDATE_TRIALS == 72
    assert len(controls) == S.PAIRED_CONTROL_TRIALS == 72
    assert len({S._sha256(item) for item in candidates + controls}) == S.BUDGET == 144
    assert {item["feature_set"] for item in candidates} == {"S4", "S5"}
    assert {item["feature_set"] for item in controls} == {"S0", "S3"}
    assert all(
        S.paired_control_config(candidate) == control
        for candidate, control in zip(candidates, controls, strict=True)
    )


def test_guards_require_incremental_forward_stable_feature_attribution() -> None:
    candidate = {"per_fold": [_fold(rho=0.10, econ=10.0) for _ in range(5)]}
    baseline = {"per_fold": [_fold(rho=0.05, econ=5.0) for _ in range(5)]}
    passing = S.apply_guards(candidate, baseline)
    assert passing["passed"] is True
    assert all(passing["guards"].values())

    flat_baseline = {"per_fold": [_fold(rho=0.10, econ=10.0) for _ in range(5)]}
    rejected = S.apply_guards(candidate, flat_baseline)
    assert rejected["passed"] is False
    assert rejected["g1_negative_controls_passed"] is True
    assert rejected["g1_paired_attribution_passed"] is False
    assert rejected["reasons"] == ["G1_controls_and_feature_attribution"]


def test_sign_reversed_economics_is_binding_in_g1() -> None:
    candidate = {
        "per_fold": [
            _fold(rho=0.10, econ=10.0, signrev_econ=11.0) for _ in range(5)
        ]
    }
    baseline = {"per_fold": [_fold(rho=0.05, econ=5.0) for _ in range(5)]}
    observed = S.apply_guards(candidate, baseline)
    assert observed["g1_negative_controls_passed"] is False
    assert observed["passed"] is False


def test_holdout_loader_requires_exact_self_hashed_receipt(monkeypatch) -> None:
    sessions = list(H._SA["protected_holdout_primary_non_degraded_29"])
    semantic = {
        "schema_version": "lean_autoresearch.holdout_access_receipt.v2",
        "open_count": 1,
        "sessions": sessions,
        "sessions_sha256": hashlib.sha256(("\n".join(sessions) + "\n").encode()).hexdigest(),
        "preregistration_sha256": "a" * 64,
        "winner_seal_sha256": "b" * 64,
    }
    receipt = dict(semantic)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    calls = []

    def fake_loader(session, capability):
        calls.append((session, capability))
        return pd.DataFrame({"session": [session]})

    monkeypatch.setattr(H, "_load_session_impl", fake_loader)
    loaded = H.load_holdout_frame(access_receipt=receipt)
    assert loaded["session"].tolist() == sessions
    assert len(calls) == len(sessions)
    assert all(capability is H._HOLDOUT_CAPABILITY for _, capability in calls)

    tampered = dict(receipt)
    tampered["open_count"] = 2
    with pytest.raises(RuntimeError, match="invalid lean holdout capability"):
        H.load_holdout_frame(access_receipt=tampered)


def test_search_loader_refuses_every_protected_session() -> None:
    for session in H.HOLDOUT:
        with pytest.raises(RuntimeError, match="HOLDOUT SEALED"):
            H.refuse_if_holdout(session)
    assert set(S._development_sessions()).isdisjoint(H.HOLDOUT)


def test_bootstrap_lcb_is_deterministic_and_conservative(monkeypatch) -> None:
    monkeypatch.setattr(S, "BOOTSTRAP_DRAWS", 5_000)
    positive = np.asarray([1.0, 2.0, 3.0, 4.0])
    first = S._bootstrap_lcb(positive)
    second = S._bootstrap_lcb(positive)
    assert first == second
    assert first > 0
    assert S._bootstrap_lcb(np.asarray([-5.0, 1.0, 2.0])) < 0
