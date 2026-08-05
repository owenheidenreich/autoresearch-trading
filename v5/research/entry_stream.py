"""The frozen entry stream: the artifact that crosses from entry to exit search.

Rung 5 produces entries; rung 6 fits an exit policy on them.  The ordering is
only real if the entry stream is frozen *before* the exit search starts and the
final trade ledger is provably the same entries — otherwise an exit search can
quietly drop entries whose exits look bad, which is entry re-optimization
through the back door and invisible to every per-trade consistency check.

An entry stream is one row per (session, decision boundary): out-of-fold score,
side, abstain flag, and fold id.  Freezing it produces a content-addressed lock
naming the stream hash, the model specification hash, and the feature ledger
hash.  ``assert_trades_match_entry_stream`` then requires the trade ledger's
(session, decision_time, side) rows to equal the stream's non-abstained rows
exactly — no additions, no omissions, no alterations.

Model-free and network-free.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from typing import Any, Mapping

import pandas as pd


ENTRY_STREAM_SCHEMA_VERSION = "v5.entry-stream-lock.v1"
REQUIRED_ENTRY_COLUMNS = ("session", "decision_time", "side", "abstain", "score", "fold")
TRADABLE_SIDES = frozenset({"CALL", "PUT", "LONG", "SHORT"})


class EntryStreamError(RuntimeError):
    """The entry stream, its lock, or a trade ledger failed the freeze contract."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _hash_payload(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def normalize_entry_stream(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonically order an entry stream frame."""

    missing = set(REQUIRED_ENTRY_COLUMNS) - set(frame.columns)
    if missing:
        raise EntryStreamError(f"entry stream missing columns: {sorted(missing)}")
    result = frame.loc[:, list(REQUIRED_ENTRY_COLUMNS)].copy()
    if not len(result):
        raise EntryStreamError("entry stream is empty")
    result["session"] = result["session"].astype(str)
    decision = pd.to_datetime(result["decision_time"], utc=True, errors="coerce")
    if decision.isna().any():
        raise EntryStreamError("entry stream has invalid decision timestamps")
    result["decision_time"] = decision.map(lambda value: value.isoformat())
    if not result["abstain"].map(lambda value: isinstance(value, (bool,))).all():
        raise EntryStreamError("entry stream abstain flags must be booleans")
    result["side"] = result["side"].astype(str).str.upper()
    tradable = result.loc[~result["abstain"], "side"]
    if not tradable.isin(TRADABLE_SIDES).all():
        raise EntryStreamError(
            "non-abstained entry rows must carry a tradable side "
            f"({sorted(TRADABLE_SIDES)})"
        )
    score = pd.to_numeric(result["score"], errors="coerce")
    if score.isna().any():
        raise EntryStreamError("entry stream scores must be finite numbers")
    result["score"] = score.astype(float)
    result["fold"] = result["fold"].astype(str)
    if result.duplicated(["session", "decision_time"]).any():
        raise EntryStreamError(
            "entry stream has duplicate (session, decision_time) rows; "
            "a serial one-account policy decides once per boundary"
        )
    return result.sort_values(
        ["session", "decision_time"], kind="stable"
    ).reset_index(drop=True)


def entry_stream_sha256(frame: pd.DataFrame) -> str:
    """Content hash of the canonical CSV form of a validated entry stream."""

    canonical = normalize_entry_stream(frame)
    text = canonical.to_csv(index=False, lineterminator="\n", float_format="%.10g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EntryStreamLock:
    schema_version: str
    entry_stream_sha256: str
    model_spec_sha256: str
    feature_ledger_sha256: str
    frozen_on: str
    row_count: int
    non_abstained_count: int
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_valid(self) -> "EntryStreamLock":
        if self.schema_version != ENTRY_STREAM_SCHEMA_VERSION:
            raise EntryStreamError(f"unknown entry lock schema: {self.schema_version}")
        if _hash_payload(self.to_dict()) != self.receipt_sha256:
            raise EntryStreamError("entry lock self-hash mismatch; the bytes were edited")
        for name in ("entry_stream_sha256", "model_spec_sha256", "feature_ledger_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise EntryStreamError(f"entry lock field {name} is not a sha256 digest")
        try:
            datetime.strptime(self.frozen_on, "%Y-%m-%d")
        except ValueError as exc:
            raise EntryStreamError(
                f"entry lock has an invalid freeze date: {self.frozen_on}"
            ) from exc
        if not 0 <= self.non_abstained_count <= self.row_count:
            raise EntryStreamError("entry lock row counts are inconsistent")
        return self


def freeze_entry_stream(
    frame: pd.DataFrame,
    *,
    model_spec_sha256: str,
    feature_ledger_sha256: str,
    frozen_on: str,
) -> EntryStreamLock:
    canonical = normalize_entry_stream(frame)
    unsigned = {
        "schema_version": ENTRY_STREAM_SCHEMA_VERSION,
        "entry_stream_sha256": entry_stream_sha256(canonical),
        "model_spec_sha256": str(model_spec_sha256),
        "feature_ledger_sha256": str(feature_ledger_sha256),
        "frozen_on": str(frozen_on),
        "row_count": int(len(canonical)),
        "non_abstained_count": int((~canonical["abstain"]).sum()),
    }
    return EntryStreamLock(
        **unsigned, receipt_sha256=_hash_payload(unsigned)
    ).assert_valid()


def assert_trades_match_entry_stream(
    trades: pd.DataFrame,
    entry_stream: pd.DataFrame,
    *,
    lock: EntryStreamLock,
) -> None:
    """Refuse a trade ledger that is not exactly the frozen non-abstained entries.

    ``trades`` needs ``session``, ``decision_time``, and ``side`` columns — the
    candidate packet's ledger already carries all three.
    """

    lock.assert_valid()
    canonical = normalize_entry_stream(entry_stream)
    if entry_stream_sha256(canonical) != lock.entry_stream_sha256:
        raise EntryStreamError(
            "entry stream does not match its lock; it was altered after freezing"
        )
    required = {"session", "decision_time", "side"}
    missing = required - set(trades.columns)
    if missing:
        raise EntryStreamError(f"trade ledger missing columns: {sorted(missing)}")

    def _keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
        decision = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
        if decision.isna().any():
            raise EntryStreamError("trade ledger has invalid decision timestamps")
        return set(
            zip(
                frame["session"].astype(str),
                decision.map(lambda value: value.isoformat()),
                frame["side"].astype(str).str.upper(),
            )
        )

    expected = _keys(canonical.loc[~canonical["abstain"]])
    if len(trades) != len(expected) or trades.duplicated(["session", "decision_time"]).any():
        raise EntryStreamError(
            f"trade ledger has {len(trades)} entries; the frozen stream has "
            f"{len(expected)} non-abstained rows — entries were added, dropped, "
            "or duplicated after the freeze"
        )
    actual = _keys(trades)
    if actual != expected:
        dropped = sorted(expected - actual)[:3]
        invented = sorted(actual - expected)[:3]
        raise EntryStreamError(
            "trade ledger entries differ from the frozen entry stream; "
            f"missing={dropped} unexpected={invented}"
        )
