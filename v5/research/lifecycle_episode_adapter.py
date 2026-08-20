"""Construct `SessionEpisode` records from the built two-era corpus.

This is the last structural gap between the corpus and the trainer: nothing else
in `v5` builds a `SessionEpisode`, so the trainer has only ever run on synthetic
known-answer episodes.

**Three things here are load-bearing, and each exists because of a measured
defect rather than a preference.**

1. **Sell paths are read from the raw quote file, never from the corpus `ladder`
   table.** `build_session` stores `ladder_state(..., whole_live_chain=False)` --
   the near-ATM +/-25-point band -- and its own docstring says so: "It is not the
   policy observation. The simulator and future policy tensorization must use
   `full_ladder_state`". A contract bought 10 points out of the money leaves that
   band as soon as the index moves far enough, which is exactly what happens on
   the trades that win. Measured on four sessions spanning both eras, **29.1% of
   candidate exits, and 18.3% of the exits belonging to winning trades, fall on a
   minute where the bought contract is absent from the corpus ladder table.**
   Building sell paths from that table would silently truncate the winners and
   read out as "the exit adds nothing". The quote file carries the whole chain,
   so the exit matrix is built from it.

2. **`sell_paths` is keyed `(decision_index, ladder_column)`.** A sale path
   belongs to the exact contract bought; the minute-only form throws no error and
   trains the exit head on some other instrument's price history.

3. **A session with zero affordable contracts is a full episode, not an empty
   one.** 2025-04-09 and 2025-04-10 hold zero candidates: the $2,000 ticket cap
   correctly admitted nothing on those days. They are legitimate no-trade days
   under the signed risk law, so they appear here with every decision minute
   present, every entry action masked off, and every entry target unknown. The
   WAIT head still receives them, and a serial simulator still learns that the
   account sat flat. Dropping them would quietly delete two days from the
   chronology; crashing on them would strand the build.

**Execution law.** The exit value at a held minute is the first executable bid at
or after that minute, or -- when no later executable bid exists -- the validated
cash settlement at the terminal intrinsic. That is the pinned builder's law, and
`test_lifecycle_episode_adapter.py` asserts that this module's independently
derived terminal sale value reproduces the pinned `net_bid_60m_usd` column
exactly, on real sessions, for every candidate.

**Scaling and imputation are fitted, never assumed.** `FeatureStatistics.fit`
reads a declared session list -- the chronological training prefix -- and nothing
later. `build_episode` requires one, so no caller can accidentally train on
unscaled or NaN-bearing features.

**Why the private imports.** `causal_day_tensorizer.py` and
`causal_day_architectures.py` are pinned by the sealed declarations' reseal
guard, and `build_causal_day_dataset.py` by `PREACQUISITION_SEMANTIC_FREEZE_V1`.
Promoting their helpers to public names would change those files' digests and
break the guards that prove the sealed declarations still describe the code that
ran. Importing the existing private helpers keeps the semantics byte-identical.

Design reference: `work/lifecycle-training/LEARNING_CONTENT_DESIGN_2026_08_16.md`
deliverable E.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from v5.ops.audit_causal_day_coverage import (
    CONTRACT_MULTIPLIER,
    FIRST_DECISION_MINUTE,
    LAST_ENTRY_MINUTE,
    QUOTE_MINUTES,
    live_two_sided,
)
from v5.ops.build_causal_day_dataset import (
    QUOTE_COLUMNS,
    _wide_quote_matrix,
    prepare_quotes,
    regime_for_minute,
)
from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD
from v5.research.causal_day_architectures import CausalPolicyBatch
from v5.research.causal_day_chain_state_lifecycle import ChainPolicyBatch
from v5.research.causal_day_tensorizer import (
    ACCOUNT_FEATURES,
    CANDLE_FEATURES,
    CLOCK_FEATURES,
    LADDER_FEATURES,
    POSITION_FEATURES,
    STARTING_EQUITY_USD,
    AccountObservation,
    PositionObservation,
    _account_vector,
    _candle_feature_frame,
    _clock_vector,
    _ladder_feature_frame,
    _position_vector,
)
from v5.research.chain_internal_features import (
    CHAIN_STATE_FEATURES,
    CONTRACT_CHAIN_FEATURES,
    chain_state,
    contract_chain_features,
)
from v5.research.lifecycle_trainer import SessionEpisode

SCHEMA_VERSION = "v5.lifecycle-episode-adapter.v1"
CORPUS_TABLES = ("candles", "ladder", "candidates")

#: Owner ruling, 2026-08-19: the active policy is SPXW/SPX-only with no futures
#: input, so the tape must be SPX-derived. The superseded ES-tape corpus is still
#: on disk beside the current one and is a valid corpus in every other respect --
#: it would build episodes that look entirely normal. This is checked here rather
#: than left to the caller passing the right path, because a path is exactly the
#: kind of thing that gets typed wrong once and noticed mid-fit.
REQUIRED_TAPE_SOURCE = "spx_parity_spot"
DEFAULT_HORIZON_MINUTES = 60

#: How much of the candle prefix an episode carries.
#:
#: Both declared members read **only the last completed candle** --
#: `last = candle_mask.sum(dim=1) - 1`, then `candles[arange, last]` -- in the
#: frozen baseline and in the chain-state member alike. Carrying the full 330-row
#: prefix therefore costs 7.7 MB per episode for rows the architecture provably
#: ignores, and 405 prefix episodes will not sit in memory at once because of it.
#:
#: `"last"` stores the one candle the model reads. It is **not** an approximation
#: for this architecture family: `test_a_last_candle_episode_scores_identically`
#: asserts the member's output is bitwise identical either way. It would be an
#: approximation for a future architecture that reads the sequence, so `"full"`
#: stays the default and the fit declaration records which was used.
CANDLE_PREFIX_MODES = ("full", "last")
DEFAULT_TRADE_CAP = 2

#: Member P of the semantic freeze, mapped to executable dollars: sell at the
#: first touch of +50%, stop at -30%, otherwise the horizon clock -- each resolved
#: through the first-later-bid law, ask in and bid out, fees charged.
PRIMARY_LABEL = "first_touch_50pct_before_loss_30pct_60m"
GAIN_MINUTE_COLUMN = "first_touch_50pct_minute_60m"
LOSS_MINUTE_COLUMN = "first_touch_loss_30pct_minute_50pct_60m"

#: The full ladder feature vector this adapter emits: the tensorizer's pinned
#: channels followed by the two per-contract chain-internal terms.
CONTRACT_FEATURES = LADDER_FEATURES + CONTRACT_CHAIN_FEATURES

#: Columns that are centred and scaled by fitted statistics. Everything absent
#: from this set is passed through untouched, and deliberately:
#:
#: * `is_call` and every `*_observed` flag are indicators. `direction(state) *
#:   is_call` is one of the architecture's two reordering channels and its
#:   "zero for puts" reading is the semantics the member is built around;
#:   centring it would silently change what that channel means.
#: * `moneyness_itm_points` feeds the other channel and is denominated in index
#:   points, which is the unit the risk law and the entry band are written in.
SCALED_CONTRACT_FEATURES = tuple(
    name
    for name in CONTRACT_FEATURES
    if name not in {"is_call", "moneyness_itm_points"} and not name.endswith("_observed")
)


class EpisodeAdapterError(RuntimeError):
    """The corpus cannot produce the declared episode."""


# --------------------------------------------------------------------------- #
# Corpus index
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CorpusSession:
    """One session's location and the settlement law that governs it."""

    session: str
    era: str
    settlement_source: str
    settlement_spx: float
    quote_path: Path
    corpus_root: Path

    def table(self, name: str) -> Path:
        if name not in CORPUS_TABLES:
            raise EpisodeAdapterError(f"unknown corpus table: {name}")
        return self.corpus_root / name / f"{self.session}.parquet"


@dataclass(frozen=True)
class CorpusIndex:
    """Session -> location, era, settlement. Sorted order is chronological."""

    rows: tuple[CorpusSession, ...]

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(row.session for row in self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[CorpusSession]:
        return iter(self.rows)

    def __getitem__(self, session: str) -> CorpusSession:
        for row in self.rows:
            if row.session == session:
                return row
        raise EpisodeAdapterError(f"session not in the corpus index: {session}")

    @staticmethod
    def from_receipt(
        receipt_path: Path,
        *,
        corpus_root: Path,
        quote_roots: Mapping[str, Path],
    ) -> "CorpusIndex":
        """Build the index from the corpus build receipt.

        Only `BUILT` rows are indexed. A `FAILED` session has no tables and must
        not silently become a gap in the chronology, so it is left out here and
        reported by the receipt that refused it.
        """

        payload = json.loads(Path(receipt_path).read_text())
        rows: list[CorpusSession] = []
        for entry in payload.get("sessions", []):
            if entry.get("classification") != "BUILT":
                continue
            era = str(entry["era"])
            if era not in quote_roots:
                raise EpisodeAdapterError(f"no quote root declared for era {era!r}")
            session = str(entry["session"])
            rows.append(
                CorpusSession(
                    session=session,
                    era=era,
                    settlement_source=str(entry["settlement_source"]),
                    settlement_spx=float(entry["settlement_spx"]),
                    quote_path=Path(quote_roots[era]) / f"databento_spxw_0dte_{session}.parquet",
                    corpus_root=Path(corpus_root),
                )
            )
        if not rows:
            raise EpisodeAdapterError(f"{receipt_path}: receipt certifies no built session")
        return CorpusIndex(tuple(sorted(rows, key=lambda row: row.session)))


# --------------------------------------------------------------------------- #
# Fitted feature statistics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureStatistics:
    """Imputation fills and standardisation constants, fitted on a session list.

    Fitted from the chronological training prefix and nothing later. The fill is
    the prefix median and the scale is the prefix standard deviation; a column
    with no finite prefix value or no variance keeps a fill of 0.0 and a scale of
    1.0 rather than dividing by nothing.
    """

    sessions: tuple[str, ...]
    candle_centre: np.ndarray
    candle_scale: np.ndarray
    contract_fill: np.ndarray
    contract_centre: np.ndarray
    contract_scale: np.ndarray
    chain_fill: np.ndarray
    chain_centre: np.ndarray
    chain_scale: np.ndarray

    @staticmethod
    def identity() -> "FeatureStatistics":
        """Neutral statistics for tests; imputes nothing and scales nothing.

        Any real feature frame carrying a NaN will be refused downstream under
        these, which is the point: an untransformed episode must not train.
        """

        return FeatureStatistics(
            sessions=(),
            candle_centre=np.zeros(len(CANDLE_FEATURES)),
            candle_scale=np.ones(len(CANDLE_FEATURES)),
            contract_fill=np.zeros(len(CONTRACT_FEATURES)),
            contract_centre=np.zeros(len(CONTRACT_FEATURES)),
            contract_scale=np.ones(len(CONTRACT_FEATURES)),
            chain_fill=np.zeros(len(CHAIN_STATE_FEATURES)),
            chain_centre=np.zeros(len(CHAIN_STATE_FEATURES)),
            chain_scale=np.ones(len(CHAIN_STATE_FEATURES)),
        )

    @staticmethod
    def fit(index: CorpusIndex, sessions: Sequence[str]) -> "FeatureStatistics":
        """Fit on the declared sessions only, reading no quote file.

        Statistics need the corpus tables alone, so fitting never touches the
        exit matrix and cannot see a price path.
        """

        ordered = sorted({str(session) for session in sessions})
        if not ordered:
            raise EpisodeAdapterError("feature statistics require at least one session")
        candles: list[np.ndarray] = []
        contracts: list[np.ndarray] = []
        chains: list[np.ndarray] = []
        for session in ordered:
            frames = _session_features(index[session])
            candles.append(frames.candle_values)
            contracts.append(frames.contract_values)
            chains.append(frames.chain_values)

        candle_all = np.concatenate(candles, axis=0)
        contract_all = np.concatenate(contracts, axis=0)
        chain_all = np.concatenate(chains, axis=0)

        candle_centre, candle_scale = _centre_scale(candle_all, CANDLE_FEATURES, CANDLE_FEATURES)
        contract_centre, contract_scale = _centre_scale(
            contract_all, CONTRACT_FEATURES, SCALED_CONTRACT_FEATURES
        )
        chain_centre, chain_scale = _centre_scale(
            chain_all, CHAIN_STATE_FEATURES, CHAIN_STATE_FEATURES
        )
        return FeatureStatistics(
            sessions=tuple(ordered),
            candle_centre=candle_centre,
            candle_scale=candle_scale,
            contract_fill=_median_fill(contract_all),
            contract_centre=contract_centre,
            contract_scale=contract_scale,
            chain_fill=_median_fill(chain_all),
            chain_centre=chain_centre,
            chain_scale=chain_scale,
        )

    def apply_candles(self, values: np.ndarray) -> np.ndarray:
        return (values - self.candle_centre) / self.candle_scale

    def apply_contracts(self, values: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(values), values, self.contract_fill)
        return (filled - self.contract_centre) / self.contract_scale

    def apply_chain(self, values: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(values), values, self.chain_fill)
        return (filled - self.chain_centre) / self.chain_scale


def _median_fill(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        fill = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    return np.where(np.isfinite(fill), fill, 0.0)


def _centre_scale(
    values: np.ndarray, names: Sequence[str], scaled: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.where(np.isfinite(values), values, np.nan)
    with np.errstate(all="ignore"):
        centre = np.nanmean(clean, axis=0)
        scale = np.nanstd(clean, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    passthrough = np.asarray([name not in set(scaled) for name in names])
    centre = np.where(passthrough, 0.0, centre)
    scale = np.where(passthrough, 1.0, scale)
    return centre, scale


# --------------------------------------------------------------------------- #
# Per-session feature frames
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SessionFeatures:
    """Whole-session causal feature frames, computed once and sliced per minute.

    Every channel here is an expanding or rolling function of its own past --
    `cummax`, `expanding().median()`, `rolling(...).std()`, `pct_change`, and the
    session's first open -- so row *t* of the whole-session frame equals row *t*
    of any prefix ending at *t*. Computing once and slicing is therefore not an
    approximation of the per-minute build; `test_whole_session_features_equal_the_
    per_minute_build` asserts the two agree on a real session.
    """

    session: str
    candle_minutes: tuple[str, ...]
    candle_values: np.ndarray
    ladder: pd.DataFrame
    contract_values: np.ndarray
    chain_minutes: tuple[str, ...]
    chain_values: np.ndarray


def assert_declared_tape(row: CorpusSession) -> None:
    """Refuse a corpus whose tape is not the one the owner ruled for.

    An untagged corpus is refused by name rather than assumed innocent: the
    corpus built before the ruling carries no `tape_source` column at all, and
    "no stamp" is precisely the state that means ES.
    """

    available = set(pq.ParquetFile(row.table("candles")).schema_arrow.names)
    if "tape_source" not in available:
        raise EpisodeAdapterError(
            f"{row.session}: corpus carries no tape_source stamp, so it predates the "
            "2026-08-19 SPX-tape ruling; rebuild it with build_parity_spot_candles"
        )
    stamps = set(pd.read_parquet(row.table("candles"), columns=["tape_source"])["tape_source"])
    if stamps != {REQUIRED_TAPE_SOURCE}:
        raise EpisodeAdapterError(
            f"{row.session}: tape_source is {sorted(stamps)}, not {REQUIRED_TAPE_SOURCE}; "
            "the active policy takes no futures input"
        )


def _session_features(row: CorpusSession) -> SessionFeatures:
    assert_declared_tape(row)
    candles = pd.read_parquet(row.table("candles"))
    ladder = pd.read_parquet(row.table("ladder"))
    if candles.empty:
        raise EpisodeAdapterError(f"{row.session}: no candles in the corpus")
    if ladder.empty:
        raise EpisodeAdapterError(f"{row.session}: no ladder in the corpus")

    candle_frame = _candle_feature_frame(candles)
    candle_values = candle_frame.loc[:, list(CANDLE_FEATURES)].to_numpy(float)

    ladder = ladder.sort_values(
        ["minute", "right", "strike", "contract_id"]
    ).reset_index(drop=True)
    pinned = _ladder_feature_frame(ladder)
    chain_contract = contract_chain_features(ladder)
    contract_values = np.column_stack(
        [pinned.loc[:, list(LADDER_FEATURES)].to_numpy(float),
         chain_contract.loc[:, list(CONTRACT_CHAIN_FEATURES)].to_numpy(float)]
    )

    state = chain_state(ladder)
    return SessionFeatures(
        session=row.session,
        candle_minutes=tuple(candles["knowable_at"].astype(str)),
        candle_values=candle_values,
        ladder=ladder,
        contract_values=contract_values,
        chain_minutes=tuple(state["minute"].astype(str)),
        chain_values=state.loc[:, list(CHAIN_STATE_FEATURES)].to_numpy(float),
    )


# --------------------------------------------------------------------------- #
# Exit matrix
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExitMatrix:
    """Per-share sale value for every (minute, contract) under the frozen law.

    `sale_per_share[i, c]` is what a sale requested at minute *i* actually
    realises: the first executable bid at or after *i*, or the validated cash
    settlement's terminal intrinsic when the chain never quotes that contract
    again. It is finite everywhere by construction -- every session in this
    corpus carries a settlement, official or parity -- which is what lets the
    exit head's HOLD/SELL targets be built without inventing a value.

    `mark_per_share` is the contemporaneous two-sided mid, NaN where the contract
    is not live. It marks a held position; it never executes anything.
    """

    minutes: tuple[str, ...]
    contracts: tuple[str, ...]
    sale_per_share: np.ndarray
    mark_per_share: np.ndarray
    settlement_spx: float


def build_exit_matrix(row: CorpusSession) -> ExitMatrix:
    raw = pd.read_parquet(row.quote_path, columns=list(QUOTE_COLUMNS))
    quotes = prepare_quotes(raw, row.session)
    contracts = sorted(quotes["contract_id"].unique())
    if not contracts:
        raise EpisodeAdapterError(f"{row.session}: quote file carries no contract")

    # The pinned exit-side liveness law: two-sided, and a bid someone will
    # actually take. Identical to `attach_candidate_outcomes`.
    exit_valid = live_two_sided(quotes) & quotes["bid_size"].ge(1.0)
    bid = _wide_quote_matrix(quotes, contracts, "bid", exit_valid)
    mark = _wide_quote_matrix(quotes, contracts, "mid", exit_valid)

    reference = (
        quotes.drop_duplicates("contract_id")
        .set_index("contract_id")
        .reindex(contracts)
    )
    strike = pd.to_numeric(reference["strike"], errors="coerce").to_numpy(float)
    is_call = reference["right"].astype(str).eq("C").to_numpy(bool)
    settlement = float(row.settlement_spx)
    if not np.isfinite(settlement) or settlement <= 0.0:
        raise EpisodeAdapterError(f"{row.session}: settlement SPX must be finite and positive")
    terminal_intrinsic = np.where(
        is_call,
        np.maximum(0.0, settlement - strike),
        np.maximum(0.0, strike - settlement),
    )

    n_minutes, n_contracts = bid.shape
    next_live = np.full((n_minutes + 1, n_contracts), n_minutes, dtype=np.int32)
    for i in range(n_minutes - 1, -1, -1):
        next_live[i] = np.where(np.isfinite(bid[i]), i, next_live[i + 1])
    chosen = next_live[:n_minutes]
    executable = chosen < n_minutes
    safe = np.minimum(chosen, n_minutes - 1)
    sale = np.where(
        executable,
        np.take_along_axis(bid, safe, axis=0),
        np.broadcast_to(terminal_intrinsic, bid.shape),
    )
    if not np.isfinite(sale).all():
        raise EpisodeAdapterError(
            f"{row.session}: sale matrix is not finite; the settlement branch failed"
        )
    return ExitMatrix(
        minutes=tuple(QUOTE_MINUTES),
        contracts=tuple(contracts),
        sale_per_share=sale,
        mark_per_share=mark,
        settlement_spx=settlement,
    )


class SellPathMatrix(Mapping):
    """`(decision_index, ladder_column)` -> the executable sale path, on demand.

    Slicing the exit matrix lazily rather than materialising every path keeps the
    episode small: a session offers roughly 6,500 feasible (minute, contract)
    pairs and a fit selects a few dozen. `.get` returns `None` for an infeasible
    pair, which is what the trainer already checks for.
    """

    def __init__(
        self,
        *,
        sale_per_share: np.ndarray,
        quote_column: np.ndarray,
        entry_ask_usd: np.ndarray,
        minute_row: np.ndarray,
        horizon: int,
    ) -> None:
        self._sale = sale_per_share
        self._quote_column = quote_column
        self._entry_ask = entry_ask_usd
        self._minute_row = minute_row
        self._horizon = int(horizon)
        self._n_minutes = int(sale_per_share.shape[0])

    def _span(self, decision: int) -> tuple[int, int]:
        start = int(self._minute_row[decision]) + 1
        stop = min(int(self._minute_row[decision]) + self._horizon, self._n_minutes - 1)
        return start, stop

    def __getitem__(self, key: tuple[int, int]) -> np.ndarray:
        decision, column = key
        if not (0 <= decision < self._quote_column.shape[0]):
            raise KeyError(key)
        if not (0 <= column < self._quote_column.shape[1]):
            raise KeyError(key)
        quote_column = int(self._quote_column[decision, column])
        if quote_column < 0:
            raise KeyError(key)
        start, stop = self._span(decision)
        if stop < start + 1:
            raise KeyError(key)
        path = self._sale[start : stop + 1, quote_column] * CONTRACT_MULTIPLIER
        return path - float(self._entry_ask[decision, column]) - FEES_PER_ROUND_TRIP_USD

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for decision, column in zip(*np.nonzero(self._quote_column >= 0), strict=True):
            yield int(decision), int(column)

    def __len__(self) -> int:
        return int((self._quote_column >= 0).sum())


# --------------------------------------------------------------------------- #
# Episode construction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EpisodeArtifacts:
    """Everything the episode needed, kept for diagnostics and the simulator."""

    row: CorpusSession
    decision_minutes: tuple[str, ...]
    contract_ids: tuple[tuple[str, ...], ...]
    entry_action_mask: np.ndarray
    entry_value_usd: np.ndarray
    label: np.ndarray
    feasible_actions: int
    unknown_targets: int


def _decision_minutes(features: SessionFeatures) -> list[str]:
    """Every entry-window minute the ladder actually quotes.

    All of them, not only the ones holding an affordable contract: a minute where
    nothing was affordable is a minute the policy had to answer WAIT on, and the
    two zero-candidate sessions are made entirely of those.
    """

    available = set(features.ladder["minute"].astype(str))
    knowable = set(features.candle_minutes)
    return [
        minute
        for minute in QUOTE_MINUTES
        if FIRST_DECISION_MINUTE <= minute <= LAST_ENTRY_MINUTE
        and minute in available
        and minute in knowable
    ]


def build_episode(
    row: CorpusSession,
    statistics: FeatureStatistics,
    *,
    horizon: int = DEFAULT_HORIZON_MINUTES,
    trade_cap: int = DEFAULT_TRADE_CAP,
    with_paths: bool = True,
    candle_prefix: str = "full",
) -> tuple[SessionEpisode, EpisodeArtifacts]:
    """One session as a trainer-ready episode plus its audit artifacts.

    `statistics` is required rather than defaulted: an episode built without
    fitted imputation carries NaNs into the first forward pass, and an episode
    built with statistics fitted on its own future is the leak this project
    already paid for once.
    """

    if candle_prefix not in CANDLE_PREFIX_MODES:
        raise EpisodeAdapterError(
            f"candle_prefix must be one of {CANDLE_PREFIX_MODES}, got {candle_prefix!r}"
        )
    features = _session_features(row)
    candidates = pd.read_parquet(row.table("candidates"))
    minutes = _decision_minutes(features)
    if not minutes:
        raise EpisodeAdapterError(f"{row.session}: no quoted minute inside the entry window")

    candle_row = {minute: i for i, minute in enumerate(features.candle_minutes)}
    chain_row = {minute: i for i, minute in enumerate(features.chain_minutes)}
    ladder_positions = features.ladder.groupby(
        features.ladder["minute"].astype(str), sort=False
    ).indices
    quote_row = {minute: i for i, minute in enumerate(QUOTE_MINUTES)}

    width = max(len(ladder_positions[minute]) for minute in minutes)
    n_decisions = len(minutes)
    full_prefix = candle_prefix == "full"
    max_prefix = (max(candle_row[minute] for minute in minutes) + 1) if full_prefix else 1
    # Held minutes run past the last decision minute, so a held batch needs room
    # for the whole session's candles rather than the entry window's.
    held_prefix = len(features.candle_values) if full_prefix else 1

    candles = np.zeros((n_decisions, max_prefix, len(CANDLE_FEATURES)), dtype=np.float32)
    candle_mask = np.zeros((n_decisions, max_prefix), dtype=bool)
    ladder = np.zeros((n_decisions, width, len(CONTRACT_FEATURES)), dtype=np.float32)
    ladder_mask = np.zeros((n_decisions, width), dtype=bool)
    entry_mask = np.zeros((n_decisions, width), dtype=bool)
    chain = np.zeros((n_decisions, len(CHAIN_STATE_FEATURES)), dtype=np.float32)
    clock = np.zeros((n_decisions, CLOCK_FEATURES), dtype=np.float32)
    account = np.zeros((n_decisions, ACCOUNT_FEATURES), dtype=np.float32)
    position = np.zeros((n_decisions, POSITION_FEATURES), dtype=np.float32)
    entry_value = np.full((n_decisions, width), np.nan, dtype=np.float64)
    label = np.full((n_decisions, width), np.nan, dtype=np.float64)
    entry_ask = np.full((n_decisions, width), np.nan, dtype=np.float64)
    quote_column = np.full((n_decisions, width), -1, dtype=np.int64)
    minute_row = np.asarray([quote_row[minute] for minute in minutes], dtype=np.int64)
    roles: list[str] = []
    contract_ids: list[tuple[str, ...]] = []

    start_account = AccountObservation(
        cash_usd=STARTING_EQUITY_USD,
        realised_pnl_usd=0.0,
        trades_opened=0,
        trade_cap=trade_cap,
        breaker_triggered=False,
    )
    start_account_vector = _account_vector(start_account)

    scaled_candles = statistics.apply_candles(features.candle_values)
    scaled_contracts = statistics.apply_contracts(features.contract_values)
    scaled_chain = statistics.apply_chain(features.chain_values)

    matrix = build_exit_matrix(row) if with_paths else None
    exit_column = (
        {contract: i for i, contract in enumerate(matrix.contracts)} if matrix else {}
    )

    candidate_index = _candidate_index(candidates)
    unknown = 0
    for i, minute in enumerate(minutes):
        if full_prefix:
            prefix = candle_row[minute] + 1
            candles[i, :prefix] = scaled_candles[:prefix]
            candle_mask[i, :prefix] = True
        else:
            candles[i, 0] = scaled_candles[candle_row[minute]]
            candle_mask[i, 0] = True

        positions = ladder_positions[minute]
        count = len(positions)
        ladder[i, :count] = scaled_contracts[positions]
        ladder_mask[i, :count] = True
        ids = tuple(features.ladder["contract_id"].astype(str).to_numpy()[positions])
        contract_ids.append(ids)

        eligible = (
            features.ladder["entry_eligible"].fillna(False).astype(bool).to_numpy()[positions]
        )
        entry_mask[i, :count] = eligible

        chain[i] = scaled_chain[chain_row[minute]] if minute in chain_row else np.nan
        clock[i] = _clock_vector(minute)
        account[i] = start_account_vector
        roles.append(f"{regime_for_minute(minute)}_entry")

        for column, contract in enumerate(ids):
            record = candidate_index.get((minute, contract))
            if record is None:
                continue
            entry_ask[i, column] = record.entry_ask_usd
            if matrix is not None and contract in exit_column:
                quote_column[i, column] = exit_column[contract]
            label[i, column] = record.label
            if not np.isfinite(record.label):
                unknown += 1
                continue
            entry_value[i, column] = _bracket_value(
                record,
                matrix=matrix,
                entry_row=minute_row[i],
                horizon=horizon,
                exit_column=exit_column,
            )

    if not np.isfinite(chain).all():
        raise EpisodeAdapterError(
            f"{row.session}: a decision minute has no chain-internal state; "
            "imputation is fitted, never invented"
        )

    batch = ChainPolicyBatch(
        candles=torch.from_numpy(candles),
        candle_mask=torch.from_numpy(candle_mask),
        ladder=torch.from_numpy(ladder),
        ladder_mask=torch.from_numpy(ladder_mask),
        entry_action_mask=torch.from_numpy(entry_mask),
        account=torch.from_numpy(account),
        position=torch.from_numpy(position),
        clock=torch.from_numpy(clock),
        roles=tuple(roles),
        chain=torch.from_numpy(chain),
    )

    sell_paths: Mapping[tuple[int, int], np.ndarray] = {}
    builder = None
    if matrix is not None:
        sell_paths = SellPathMatrix(
            sale_per_share=matrix.sale_per_share,
            quote_column=quote_column,
            entry_ask_usd=entry_ask,
            minute_row=minute_row,
            horizon=horizon,
        )
        builder = _HeldBatchBuilder(
            statistics=statistics,
            features=features,
            matrix=matrix,
            minutes=tuple(minutes),
            minute_row=minute_row,
            contract_ids=tuple(contract_ids),
            entry_ask_usd=entry_ask,
            quote_column=quote_column,
            candle_row=candle_row,
            chain_row=chain_row,
            ladder_positions=ladder_positions,
            scaled_candles=scaled_candles,
            scaled_contracts=scaled_contracts,
            scaled_chain=scaled_chain,
            width=width,
            max_prefix=held_prefix,
            full_prefix=full_prefix,
            horizon=horizon,
            trade_cap=trade_cap,
        )

    episode = SessionEpisode(
        session=row.session,
        entry_batch=batch,
        entry_value_usd=torch.tensor(entry_value, dtype=torch.float32),
        sell_paths=sell_paths,
        held_batch_builder=builder,
    )
    artifacts = EpisodeArtifacts(
        row=row,
        decision_minutes=tuple(minutes),
        contract_ids=tuple(contract_ids),
        entry_action_mask=entry_mask,
        entry_value_usd=entry_value,
        label=label,
        feasible_actions=int(entry_mask.sum()),
        unknown_targets=unknown,
    )
    return episode, artifacts


@dataclass(frozen=True)
class _Candidate:
    contract_id: str
    entry_ask_usd: float
    strike: float
    is_call: bool
    moneyness_itm_points: float
    entry_mid_usd: float
    label: float
    gain_minute: float
    loss_minute: float


def _candidate_index(candidates: pd.DataFrame) -> dict[tuple[str, str], _Candidate]:
    """`(minute, contract)` -> the pinned builder's candidate record."""

    if candidates.empty:
        return {}
    required = {
        "entry_minute",
        "contract_id",
        "entry_ask_usd",
        "entry_mid_usd",
        "strike",
        "is_call",
        "moneyness_itm_points",
        PRIMARY_LABEL,
        GAIN_MINUTE_COLUMN,
        LOSS_MINUTE_COLUMN,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise EpisodeAdapterError(f"candidates missing columns: {missing}")
    out: dict[tuple[str, str], _Candidate] = {}
    for record in candidates.itertuples(index=False):
        out[(str(record.entry_minute), str(record.contract_id))] = _Candidate(
            contract_id=str(record.contract_id),
            entry_ask_usd=float(record.entry_ask_usd),
            strike=float(record.strike),
            is_call=bool(record.is_call),
            moneyness_itm_points=float(record.moneyness_itm_points),
            entry_mid_usd=float(record.entry_mid_usd),
            label=float(getattr(record, PRIMARY_LABEL)),
            gain_minute=float(getattr(record, GAIN_MINUTE_COLUMN)),
            loss_minute=float(getattr(record, LOSS_MINUTE_COLUMN)),
        )
    return out


def _bracket_value(
    record: _Candidate,
    *,
    matrix: ExitMatrix | None,
    entry_row: int,
    horizon: int,
    exit_column: Mapping[str, int],
) -> float:
    """Member P in executable dollars: +50% target, -30% stop, else the clock.

    The bracket triggers on the quoted mid, exactly as the pinned label does, and
    the resulting sale is filled at the first executable bid at or after the
    trigger minute. Ask in, bid out, one round trip of fees.
    """

    if matrix is None:
        return float("nan")
    offsets = [value for value in (record.gain_minute, record.loss_minute) if np.isfinite(value)]
    offset = int(min(offsets)) if offsets else horizon
    exit_row = min(entry_row + offset, matrix.sale_per_share.shape[0] - 1)
    column = exit_column.get(record.contract_id)
    if column is None:
        return float("nan")
    sale = float(matrix.sale_per_share[exit_row, column])
    return sale * CONTRACT_MULTIPLIER - record.entry_ask_usd - FEES_PER_ROUND_TRIP_USD


# --------------------------------------------------------------------------- #
# Held-state batches
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _HeldBatchBuilder:
    """`(decision_index, ladder_column)` -> one batch row per held minute.

    The rows line up one-for-one with `sell_paths[(i, c)]`, because
    `train_exit_head` pairs a trajectory's path with a single batch and would
    otherwise mis-align silently.
    """

    statistics: FeatureStatistics
    features: SessionFeatures
    matrix: ExitMatrix
    minutes: tuple[str, ...]
    minute_row: np.ndarray
    contract_ids: tuple[tuple[str, ...], ...]
    entry_ask_usd: np.ndarray
    quote_column: np.ndarray
    candle_row: Mapping[str, int]
    chain_row: Mapping[str, int]
    ladder_positions: Mapping[str, np.ndarray]
    scaled_candles: np.ndarray
    scaled_contracts: np.ndarray
    scaled_chain: np.ndarray
    width: int
    max_prefix: int
    full_prefix: bool
    horizon: int
    trade_cap: int

    def __call__(self, decision: int, column: int) -> CausalPolicyBatch:
        quote_column = int(self.quote_column[decision, column])
        if quote_column < 0:
            raise EpisodeAdapterError(
                f"{self.features.session}: ({decision}, {column}) is not a feasible entry"
            )
        entry_row = int(self.minute_row[decision])
        start = entry_row + 1
        stop = min(entry_row + self.horizon, self.matrix.sale_per_share.shape[0] - 1)
        held_minutes = [
            minute
            for minute in QUOTE_MINUTES[start : stop + 1]
            if minute in self.candle_row and minute in self.ladder_positions
        ]
        if len(held_minutes) != stop - start + 1:
            raise EpisodeAdapterError(
                f"{self.features.session}: held window {start}-{stop} is not fully quoted"
            )
        entry_ask = float(self.entry_ask_usd[decision, column])
        contract = self.contract_ids[decision][column]
        reference = self.features.ladder[
            self.features.ladder["contract_id"].astype(str).eq(contract)
        ].iloc[0]
        strike = float(reference["strike"])
        is_call = bool(reference["is_call"])
        entry_minute = self.minutes[decision]
        origin = regime_for_minute(entry_minute)
        entry_mid = float(reference["mid"])

        marks = self.matrix.mark_per_share[start : stop + 1, quote_column]
        marked = _forward_fill(marks, seed=entry_mid)
        unrealised = marked * CONTRACT_MULTIPLIER - entry_ask
        favourable = np.maximum.accumulate(unrealised)
        adverse = np.minimum.accumulate(unrealised)

        rows = len(held_minutes)
        candles = np.zeros((rows, self.max_prefix, len(CANDLE_FEATURES)), dtype=np.float32)
        candle_mask = np.zeros((rows, self.max_prefix), dtype=bool)
        ladder = np.zeros((rows, self.width, len(CONTRACT_FEATURES)), dtype=np.float32)
        ladder_mask = np.zeros((rows, self.width), dtype=bool)
        chain = np.zeros((rows, len(CHAIN_STATE_FEATURES)), dtype=np.float32)
        clock = np.zeros((rows, CLOCK_FEATURES), dtype=np.float32)
        account = np.zeros((rows, ACCOUNT_FEATURES), dtype=np.float32)
        position = np.zeros((rows, POSITION_FEATURES), dtype=np.float32)
        roles: list[str] = []

        held_account = _account_vector(
            AccountObservation(
                cash_usd=STARTING_EQUITY_USD - entry_ask,
                realised_pnl_usd=0.0,
                trades_opened=1,
                trade_cap=self.trade_cap,
                breaker_triggered=False,
            )
        )
        for i, minute in enumerate(held_minutes):
            # Defect fixed 2026-08-20: this previously clamped the prefix to the
            # entry window's width and then stored `scaled_candles[:prefix]`, so a
            # held minute after 15:00 silently received the 15:00 candle as its
            # "latest completed" bar. The model reads the last unmasked row, so it
            # was marking a held position against a stale tape with no error.
            if self.full_prefix:
                prefix = self.candle_row[minute] + 1
                candles[i, :prefix] = self.scaled_candles[:prefix]
                candle_mask[i, :prefix] = True
            else:
                candles[i, 0] = self.scaled_candles[self.candle_row[minute]]
                candle_mask[i, 0] = True
            positions = self.ladder_positions[minute][: self.width]
            count = len(positions)
            ladder[i, :count] = self.scaled_contracts[positions]
            ladder_mask[i, :count] = True
            chain[i] = self.scaled_chain[self.chain_row[minute]]
            clock[i] = _clock_vector(minute)
            account[i] = held_account
            spot = float(
                pd.to_numeric(
                    self.features.ladder.iloc[positions]["underlying_price"], errors="coerce"
                ).median()
            )
            position[i] = _position_vector(
                PositionObservation(
                    right="C" if is_call else "P",
                    strike=strike,
                    origin_regime=origin,
                    entry_ask_usd=entry_ask,
                    entry_moneyness_itm_points=float(reference["moneyness_itm_points"]),
                    unrealised_pnl_usd=float(unrealised[i]),
                    maximum_favourable_usd=float(favourable[i]),
                    maximum_adverse_usd=float(adverse[i]),
                    minutes_held=i + 1,
                ),
                spot,
            )
            roles.append(f"{regime_for_minute(minute)}_exit")

        return ChainPolicyBatch(
            candles=torch.from_numpy(candles),
            candle_mask=torch.from_numpy(candle_mask),
            ladder=torch.from_numpy(ladder),
            ladder_mask=torch.from_numpy(ladder_mask),
            # An exit role may never expose an entry action; the gate enforces it
            # and the simulator owns the one-position-at-a-time law.
            entry_action_mask=torch.zeros((rows, self.width), dtype=torch.bool),
            account=torch.from_numpy(account),
            position=torch.from_numpy(position),
            clock=torch.from_numpy(clock),
            roles=tuple(roles),
            chain=torch.from_numpy(chain),
        )


def _forward_fill(values: np.ndarray, *, seed: float) -> np.ndarray:
    """Mark to the most recent observed mid, seeded by the entry mid.

    A live blotter cannot mark to a quote that is not there, and it does not
    invent one either -- it keeps showing the last price it saw. Seeding with the
    entry mid means the first held minute always has a mark.
    """

    out = np.asarray(values, dtype=np.float64).copy()
    last = float(seed)
    for i, value in enumerate(out):
        if np.isfinite(value):
            last = float(value)
        else:
            out[i] = last
    return out
