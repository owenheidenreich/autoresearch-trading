"""The corpus-to-trainer adapter: keying, causality, and the frozen exit law.

The tests that need the built corpus are skipped when the SSD is not mounted;
everything structural runs everywhere.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES
from v5.research.causal_day_architectures import ArchitectureDimensions
from v5.research.causal_day_chain_state_lifecycle import ChainStateLifecyclePolicy
from v5.research.causal_day_tensorizer import (
    ACCOUNT_FEATURES,
    CANDLE_FEATURES,
    CLOCK_FEATURES,
    POSITION_FEATURES,
    _candle_feature_frame,
)
from v5.research.chain_internal_features import CHAIN_STATE_FEATURES
from v5.research.lifecycle_episode_adapter import (
    CONTRACT_FEATURES,
    SCALED_CONTRACT_FEATURES,
    CorpusIndex,
    CorpusSession,
    EpisodeAdapterError,
    FeatureStatistics,
    SellPathMatrix,
    _forward_fill,
    _session_features,
    build_episode,
)

CORPUS_ROOT = Path("/Volumes/AR_TRADING_DATA/lifecycle_corpus_2022-06-01_2026-07-31")
BACKFILL_QUOTES = Path("/Volumes/AR_TRADING_DATA/lifecycle_repaired_2022-06-01_2025-07-31")
OWNED_QUOTES = Path(
    "/Users/och/.autoresearch-trading/pathd_2025-08-01_2026-07-31/aligned/normalized"
)
BUILD_RECEIPT = Path(
    "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/corpus_build_receipt_attempt1.json"
)
BACKFILL_SESSION = "2022-06-01"
OWNED_SESSION = "2026-07-30"
NO_TRADE_SESSION = "2025-04-09"

PRIMARY_LABEL = "first_touch_50pct_before_loss_30pct_60m"


def _corpus_available() -> bool:
    return (
        CORPUS_ROOT.is_dir()
        and BACKFILL_QUOTES.is_dir()
        and OWNED_QUOTES.is_dir()
        and BUILD_RECEIPT.is_file()
    )


needs_corpus = pytest.mark.skipif(not _corpus_available(), reason="built corpus not mounted")


def _index() -> CorpusIndex:
    return CorpusIndex.from_receipt(
        BUILD_RECEIPT,
        corpus_root=CORPUS_ROOT,
        quote_roots={"owned": OWNED_QUOTES, "backfill": BACKFILL_QUOTES},
    )


# --------------------------------------------------------------------------- #
# Structural tests
# --------------------------------------------------------------------------- #


def _matrix(minutes: int = 8, contracts: int = 3) -> SellPathMatrix:
    sale = np.arange(minutes * contracts, dtype=float).reshape(minutes, contracts)
    quote_column = np.array([[0, 1, -1], [2, -1, 1]])
    return SellPathMatrix(
        sale_per_share=sale,
        quote_column=quote_column,
        entry_ask_usd=np.full((2, 3), 100.0),
        minute_row=np.array([0, 2]),
        horizon=3,
    )


def test_a_sell_path_belongs_to_one_contract_not_one_minute() -> None:
    """The defect this key exists to prevent: two contracts, one minute."""

    paths = _matrix()
    first = paths[(0, 0)]
    second = paths[(0, 1)]
    assert first.shape == second.shape
    assert not np.allclose(first, second)


def test_an_infeasible_pair_is_absent_rather_than_plausible() -> None:
    paths = _matrix()
    assert paths.get((0, 2)) is None
    assert paths.get((1, 1)) is None
    with pytest.raises(KeyError):
        paths[(0, 2)]
    assert set(paths) == {(0, 0), (0, 1), (1, 0), (1, 2)}
    assert len(paths) == 4


def test_the_path_starts_after_the_entry_minute_and_charges_the_round_trip() -> None:
    paths = _matrix()
    # Entry at row 0, horizon 3 -> held minutes 1, 2, 3 of the sale matrix.
    expected = np.array([3.0, 6.0, 9.0]) * 100.0 - 100.0 - 3.08
    np.testing.assert_allclose(paths[(0, 0)], expected)


def test_identity_statistics_change_nothing() -> None:
    stats = FeatureStatistics.identity()
    values = np.array([[1.0, -2.0, 3.0]])
    padded = np.tile(values, (1, len(CANDLE_FEATURES) // 3 + 1))[:, : len(CANDLE_FEATURES)]
    np.testing.assert_allclose(stats.apply_candles(padded), padded)


def test_indicator_and_points_columns_are_never_centred_or_scaled() -> None:
    """`is_call` and `moneyness_itm_points` are the two reordering channels.

    Centring `is_call` would quietly change what `direction(state) * is_call`
    means, and `moneyness_itm_points` is denominated in the index points the
    risk law is written in.
    """

    for name in ("is_call", "moneyness_itm_points"):
        assert name in CONTRACT_FEATURES
        assert name not in SCALED_CONTRACT_FEATURES
    assert not any(name.endswith("_observed") for name in SCALED_CONTRACT_FEATURES)


def test_forward_fill_marks_to_the_last_seen_quote() -> None:
    marks = np.array([np.nan, 2.0, np.nan, np.nan, 5.0])
    np.testing.assert_allclose(_forward_fill(marks, seed=1.0), [1.0, 2.0, 2.0, 2.0, 5.0])


def test_the_index_skips_failed_sessions_and_names_an_unknown_era(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "session": "2024-01-02",
                        "classification": "BUILT",
                        "era": "backfill",
                        "settlement_source": "parity_close",
                        "settlement_spx": 4700.0,
                    },
                    {"session": "2024-01-03", "classification": "FAILED", "reason": "x"},
                ]
            }
        )
    )
    index = CorpusIndex.from_receipt(
        receipt, corpus_root=tmp_path, quote_roots={"backfill": tmp_path}
    )
    assert index.sessions == ("2024-01-02",)
    with pytest.raises(EpisodeAdapterError, match="no quote root declared"):
        CorpusIndex.from_receipt(receipt, corpus_root=tmp_path, quote_roots={"owned": tmp_path})


def test_an_empty_receipt_is_refused_rather_than_producing_an_empty_corpus(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"sessions": []}))
    with pytest.raises(EpisodeAdapterError, match="no built session"):
        CorpusIndex.from_receipt(receipt, corpus_root=tmp_path, quote_roots={"owned": tmp_path})


# --------------------------------------------------------------------------- #
# Real-corpus tests
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def statistics() -> FeatureStatistics:
    if not _corpus_available():  # pragma: no cover - skipped downstream
        pytest.skip("built corpus not mounted")
    return FeatureStatistics.fit(_index(), [BACKFILL_SESSION])


@needs_corpus
@pytest.mark.parametrize("session", [BACKFILL_SESSION, OWNED_SESSION])
def test_the_terminal_sale_reproduces_the_pinned_builder_exactly(
    session: str, statistics: FeatureStatistics
) -> None:
    """The whole exit law, checked against the frozen column it must agree with.

    This module derives the first-later-bid / validated-settlement value from
    the raw quote file independently of `attach_candidate_outcomes`. The last
    element of a 60-minute sell path is the clock exit, so it must equal the
    pinned `net_bid_60m_usd` for that candidate -- on every candidate, not on a
    sample.
    """

    row = _index()[session]
    episode, artifacts = build_episode(row, statistics)
    candidates = pd.read_parquet(
        row.table("candidates"), columns=["entry_minute", "contract_id", "net_bid_60m_usd"]
    )
    pinned = {
        (str(minute), str(contract)): float(value)
        for minute, contract, value in candidates.itertuples(index=False)
    }
    checked = 0
    for index, minute in enumerate(artifacts.decision_minutes):
        for column, contract in enumerate(artifacts.contract_ids[index]):
            expected = pinned.get((minute, contract))
            if expected is None:
                continue
            path = episode.sell_paths.get((index, column))
            assert path is not None
            assert float(path[-1]) == pytest.approx(expected, abs=1e-6)
            checked += 1
    assert checked == len(pinned) > 3_000


@needs_corpus
def test_a_sell_path_survives_the_contract_leaving_the_near_atm_band(
    statistics: FeatureStatistics,
) -> None:
    """The reason paths come from the quote file, asserted rather than assumed.

    The corpus `ladder` table is the +/-25-point band, and a bought contract
    leaves it precisely when the trade is working. Measured on four sessions,
    29.1% of exits -- and 18.3% of winners' exits -- land on a minute where the
    contract is absent from that table. Every path must still be complete and
    finite.
    """

    row = _index()[BACKFILL_SESSION]
    episode, artifacts = build_episode(row, statistics)
    ladder = pd.read_parquet(row.table("ladder"), columns=["minute", "contract_id"])
    present = set(zip(ladder["minute"].astype(str), ladder["contract_id"].astype(str)))

    departures = 0
    for index, minute in enumerate(artifacts.decision_minutes):
        entry_row = QUOTE_MINUTES.index(minute)
        for column, contract in enumerate(artifacts.contract_ids[index]):
            path = episode.sell_paths.get((index, column))
            if path is None:
                continue
            assert len(path) == 60
            assert np.isfinite(path).all()
            exit_minute = QUOTE_MINUTES[min(entry_row + 60, len(QUOTE_MINUTES) - 1)]
            if (exit_minute, contract) not in present:
                departures += 1
    assert departures > 0, "no contract left the band; the control proves nothing here"


@needs_corpus
def test_whole_session_candle_features_equal_the_per_minute_prefix_build(
    statistics: FeatureStatistics,
) -> None:
    """Computing once and slicing must be identical to rebuilding per minute.

    Every chart channel is an expanding or rolling function of its own past, so
    this holds by construction -- and is asserted, because if a non-causal
    channel were ever added the slice would start disagreeing with the rebuild.
    """

    row = _index()[BACKFILL_SESSION]
    features = _session_features(row)
    candles = pd.read_parquet(row.table("candles"))
    for cut in (5, 60, 200, len(candles)):
        rebuilt = _candle_feature_frame(candles.iloc[:cut])
        np.testing.assert_allclose(
            rebuilt.loc[:, list(CANDLE_FEATURES)].to_numpy(float),
            features.candle_values[:cut],
            rtol=0.0,
            atol=0.0,
        )


@needs_corpus
def test_the_entry_batch_is_unchanged_when_the_future_is_mutated(
    tmp_path: Path, statistics: FeatureStatistics
) -> None:
    """The project's standing causality control, run on the real adapter.

    A copy of the corpus is perturbed after 11:00 -- candles and ladder both --
    and every decision minute at or before 11:00 must be bitwise identical.
    """

    row = _index()[BACKFILL_SESSION]
    root = tmp_path / "corpus"
    for table in ("candles", "ladder", "candidates"):
        destination = root / table
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy(row.table(table), destination / f"{row.session}.parquet")

    mutated = CorpusSession(
        session=row.session,
        era=row.era,
        settlement_source=row.settlement_source,
        settlement_spx=row.settlement_spx,
        quote_path=row.quote_path,
        corpus_root=root,
    )
    before, artifacts = build_episode(mutated, statistics, with_paths=False)

    candles = pd.read_parquet(mutated.table("candles"))
    later = candles["knowable_at"].astype(str) > "11:00"
    for column in ("open", "high", "low", "close", "volume"):
        candles.loc[later, column] = candles.loc[later, column] * 1.37 + 11.0
    candles.to_parquet(mutated.table("candles"), index=False)

    ladder = pd.read_parquet(mutated.table("ladder"))
    later = ladder["minute"].astype(str) > "11:00"
    for column in ("bid", "ask", "mid", "bid_size", "ask_size", "self_iv"):
        ladder.loc[later, column] = ladder.loc[later, column] * 3.0
    ladder.to_parquet(mutated.table("ladder"), index=False)

    after, _ = build_episode(mutated, statistics, with_paths=False)
    keep = [i for i, minute in enumerate(artifacts.decision_minutes) if minute <= "11:00"]
    assert len(keep) > 50
    index = torch.tensor(keep)
    for name in ("candles", "ladder", "chain", "clock", "account", "position"):
        torch.testing.assert_close(
            getattr(before.entry_batch, name)[index],
            getattr(after.entry_batch, name)[index],
            rtol=0.0,
            atol=0.0,
        )
    assert torch.equal(
        before.entry_batch.entry_action_mask[index], after.entry_batch.entry_action_mask[index]
    )


@needs_corpus
def test_a_no_trade_session_is_a_full_episode_not_a_missing_one(
    statistics: FeatureStatistics,
) -> None:
    """2025-04-09 held zero affordable contracts. It is still a trading day.

    The $2,000 ticket cap admitted nothing that session. Under the signed risk
    law that is a legitimate all-day WAIT, so the episode carries every decision
    minute with no feasible action, and the WAIT head still sees it.
    """

    row = _index()[NO_TRADE_SESSION]
    episode, artifacts = build_episode(row, statistics)
    assert len(artifacts.decision_minutes) > 300
    assert artifacts.feasible_actions == 0
    assert len(episode.sell_paths) == 0
    assert not torch.isfinite(episode.entry_value_usd).any()
    assert not episode.entry_batch.entry_action_mask.any()

    model = ChainStateLifecyclePolicy()
    with torch.no_grad():
        scores = model(episode.entry_batch)
    assert torch.isfinite(scores.abstain_logits).all()


@needs_corpus
def test_the_batch_satisfies_the_architecture_contract(
    statistics: FeatureStatistics,
) -> None:
    row = _index()[BACKFILL_SESSION]
    episode, _ = build_episode(row, statistics, with_paths=False)
    dimensions = ArchitectureDimensions(
        candle_features=len(CANDLE_FEATURES),
        ladder_features=len(CONTRACT_FEATURES),
        account_features=ACCOUNT_FEATURES,
        position_features=POSITION_FEATURES,
        clock_features=CLOCK_FEATURES,
        hidden_size=3,
    )
    episode.entry_batch.validate(dimensions)
    assert episode.entry_batch.chain.shape[1] == len(CHAIN_STATE_FEATURES)
    assert torch.isfinite(episode.entry_batch.chain).all()


@needs_corpus
def test_a_held_batch_has_exactly_one_row_per_sell_path_minute(
    statistics: FeatureStatistics,
) -> None:
    """`train_exit_head` pairs one batch with one path; a mismatch is silent."""

    row = _index()[BACKFILL_SESSION]
    episode, _ = build_episode(row, statistics)
    key = next(iter(episode.sell_paths))
    path = episode.sell_paths[key]
    batch = episode.held_batch_builder(*key)
    assert batch.batch_size == len(path)
    assert all(role.endswith("_exit") for role in batch.roles)
    assert not batch.entry_action_mask.any()
    assert torch.isfinite(batch.position).all()
    # Minutes held must advance one per row, never restart.
    minutes_held = batch.position[:, 7] * 120.0
    np.testing.assert_allclose(
        minutes_held.numpy(), np.arange(1, len(path) + 1), rtol=0.0, atol=1e-4
    )


@needs_corpus
def test_a_target_exists_exactly_where_the_pinned_label_is_known(
    statistics: FeatureStatistics,
) -> None:
    """Unknown labels stay feasible and stay unsupervised; nothing is dropped."""

    row = _index()[BACKFILL_SESSION]
    _, artifacts = build_episode(row, statistics)
    known = np.isfinite(artifacts.label)
    valued = np.isfinite(artifacts.entry_value_usd)
    np.testing.assert_array_equal(known, valued)
    assert known.sum() + artifacts.unknown_targets == artifacts.feasible_actions


@needs_corpus
def test_building_the_same_session_twice_is_bitwise_identical(
    statistics: FeatureStatistics,
) -> None:
    row = _index()[BACKFILL_SESSION]
    first, _ = build_episode(row, statistics, with_paths=False)
    second, _ = build_episode(row, statistics, with_paths=False)
    for name in ("candles", "ladder", "chain", "clock", "account", "position"):
        assert torch.equal(getattr(first.entry_batch, name), getattr(second.entry_batch, name))
    assert torch.equal(
        torch.nan_to_num(first.entry_value_usd, nan=-1.0),
        torch.nan_to_num(second.entry_value_usd, nan=-1.0),
    )


@needs_corpus
def test_statistics_fit_only_on_the_sessions_they_are_given() -> None:
    index = _index()
    stats = FeatureStatistics.fit(index, [BACKFILL_SESSION, "2022-06-02"])
    assert stats.sessions == (BACKFILL_SESSION, "2022-06-02")
    assert np.isfinite(stats.chain_fill).all()
    assert (stats.contract_scale > 0.0).all()
    with pytest.raises(EpisodeAdapterError, match="at least one session"):
        FeatureStatistics.fit(index, [])
