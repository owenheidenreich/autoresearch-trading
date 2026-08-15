"""Shared contracts for the owner-signed Protocol101 Stage-1 repair.

This module contains no fitting, scoring, replay economics, gate aggregation,
or protected-data access.  It pins the additive processed-row schema, typed
identity failures, the model-alpha firewall, and diagnostic quote-age report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, Mapping, Sequence, TypeVar
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROTOCOL101_CONTRACT_ID = "PROTOCOL101_CANONICAL_V1_STAGE1"
LEGACY_PROCESSED_ROW_SCHEMA = "Protocol101LegacyProcessedRowV1"
TWO_CLOCK_PROCESSED_ROW_SCHEMA = (
    "Protocol101ScopedCanonicalStage1ProcessedRowV2TwoClockExit"
)
EXIT_QUOTE_AGE_REPORT_SCHEMA = "Protocol101ExitQuoteAgeReportV1"
INT64_MISSING = int(np.iinfo(np.int64).min)
NY = ZoneInfo("America/New_York")


class ExitReason(IntEnum):
    INVALID = 0
    STOP_LOSS = 1
    TAKE_PROFIT = 2
    MAX_HOLD = 3
    FORCED_FLAT = 4
    NO_BID_STOP = 5


class InvalidReason(IntEnum):
    NONE = 0
    NO_CAUSAL_FUTURE_QUOTE_AT_OR_BEFORE_DEADLINE = 1
    NONFINITE_OR_NONPOSITIVE_ENTRY_ASK = 2
    NONFINITE_EXECUTABLE_EXIT_BID = 3
    SOURCE_QUOTE_NOT_STRICTLY_AFTER_ENTRY = 4
    EXIT_CLOCK_CROSSES_SESSION = 5
    DUPLICATE_PATH_QUOTE_IDENTITY = 6
    AXIS_OR_POLICY_ALIGNMENT_FAILURE = 7
    DEADLINE_BEFORE_OR_AT_ENTRY = 8


TERMINAL_EXIT_REASONS = frozenset(
    {
        ExitReason.STOP_LOSS,
        ExitReason.TAKE_PROFIT,
        ExitReason.MAX_HOLD,
        ExitReason.FORCED_FLAT,
        ExitReason.NO_BID_STOP,
    }
)


@dataclass(frozen=True)
class TwoClockLabel:
    """One fixed-policy label plus its pricing and occupancy clocks."""

    net_pnl: float
    mid_pnl: float
    realized_exit_time_ns: int
    source_exit_quote_time_ns: int
    exit_quote_age_ms: float
    exit_reason_code: int
    executable_exit_bid: float
    policy_deadline_ns: int
    invalid_reason_code: int

    @property
    def valid(self) -> bool:
        try:
            reason = ExitReason(int(self.exit_reason_code))
        except ValueError:
            return False
        return (
            reason in TERMINAL_EXIT_REASONS
            and int(self.invalid_reason_code) == int(InvalidReason.NONE)
        )

    @classmethod
    def invalid(
        cls,
        reason: InvalidReason,
        *,
        policy_deadline_ns: int = INT64_MISSING,
    ) -> "TwoClockLabel":
        # The signed v2 schema requires every timestamp, including the policy
        # deadline, to use INT64_MIN whenever the cell is invalid.
        del policy_deadline_ns
        return cls(
            net_pnl=float("nan"),
            mid_pnl=float("nan"),
            realized_exit_time_ns=INT64_MISSING,
            source_exit_quote_time_ns=INT64_MISSING,
            exit_quote_age_ms=float("nan"),
            exit_reason_code=int(ExitReason.INVALID),
            executable_exit_bid=float("nan"),
            policy_deadline_ns=INT64_MISSING,
            invalid_reason_code=int(reason),
        )


class Protocol101RegimenRepairError(RuntimeError):
    """Base error carrying a stable machine-readable blocker code."""

    blocker_code = "P101_REPAIR_CONTRACT_ERROR"

    def __init__(
        self,
        message: str,
        *,
        canonical_key: Any = None,
        first_source_locator: Any = None,
        duplicate_source_locator: Any = None,
        observed_count: int | None = None,
        boundary: str = "",
    ) -> None:
        super().__init__(message)
        self.payload = {
            "blocker_code": self.blocker_code,
            "canonical_key": canonical_key,
            "first_source_locator": first_source_locator,
            "duplicate_source_locator": duplicate_source_locator,
            "observed_count": observed_count,
            "boundary": boundary,
        }


class Protocol101IdentityContractError(Protocol101RegimenRepairError):
    blocker_code = "P101_ID_CONTRACT_ERROR"


class Protocol101DuplicateSessionMembershipError(
    Protocol101IdentityContractError
):
    blocker_code = "P101_ID_DUPLICATE_SESSION_MEMBERSHIP"


class Protocol101SessionRoleOverlapError(Protocol101IdentityContractError):
    blocker_code = "P101_ID_SESSION_ROLE_OVERLAP"


class Protocol101ValidationFoldOverlapError(Protocol101IdentityContractError):
    blocker_code = "P101_ID_VALIDATION_FOLD_OVERLAP"


class Protocol101DuplicateDecisionIdentityError(
    Protocol101IdentityContractError
):
    blocker_code = "P101_ID_DUPLICATE_DECISION"


class Protocol101DuplicateContractIdentityError(
    Protocol101IdentityContractError
):
    blocker_code = "P101_ID_DUPLICATE_CONTRACT"


class Protocol101DuplicateCanonicalSlotError(Protocol101IdentityContractError):
    blocker_code = "P101_ID_DUPLICATE_CANONICAL_SLOT"


class Protocol101DuplicatePathQuoteIdentityError(
    Protocol101IdentityContractError
):
    blocker_code = "P101_ID_DUPLICATE_PATH_QUOTE_IDENTITY"


class Protocol101PolicyAxisAlignmentError(Protocol101IdentityContractError):
    blocker_code = "P101_ID_POLICY_AXIS_MISALIGNED"


class Protocol101AlphaFirewallError(Protocol101RegimenRepairError):
    blocker_code = "P101_ALPHA_FIREWALL_VIOLATION"


def utc_timestamp_ns(value: Any) -> int:
    """Normalize a timezone-aware timestamp to signed UTC nanoseconds."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise Protocol101RegimenRepairError(
            "timezone-naive timestamp",
            canonical_key=str(value),
            boundary="timestamp_normalization",
        )
    return int(timestamp.tz_convert("UTC").value)


def ny_session_for_ns(value: int) -> str:
    return (
        pd.Timestamp(int(value), unit="ns", tz="UTC")
        .tz_convert("America/New_York")
        .date()
        .isoformat()
    )


def render_utc_ns(value: int) -> str:
    """Render canonical UTC nanoseconds with exactly nine digits and ``Z``."""

    timestamp = pd.Timestamp(int(value), unit="ns", tz="UTC")
    return (
        timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        + (
            f".{int(timestamp.microsecond) * 1_000 + int(timestamp.nanosecond):09d}Z"
        )
    )


ErrorT = TypeVar("ErrorT", bound=Protocol101IdentityContractError)


def _assert_unique(
    rows: Iterable[tuple[Any, Any]],
    *,
    error_type: type[ErrorT],
    boundary: str,
) -> int:
    first: dict[Any, Any] = {}
    count = 0
    for key, locator in rows:
        count += 1
        if key in first:
            raise error_type(
                f"{error_type.blocker_code}: duplicate identity {key!r}",
                canonical_key=key,
                first_source_locator=first[key],
                duplicate_source_locator=locator,
                observed_count=2,
                boundary=boundary,
            )
        first[key] = locator
    return count


def assert_manifest_session_identities(
    included_sessions: Sequence[Mapping[str, Any]],
    *,
    boundary: str = "manifest load before any dataset load",
) -> dict[str, int]:
    count = _assert_unique(
        (
            (
                str(row.get("session") or ""),
                {"row_index": index, "session": row.get("session")},
            )
            for index, row in enumerate(included_sessions)
        ),
        error_type=Protocol101DuplicateSessionMembershipError,
        boundary=boundary,
    )
    return {
        "rows_seen": count,
        "unique_identities": count,
        "duplicate_identities": 0,
        "role_overlaps": 0,
    }


def assert_fold_session_identities(
    folds: Sequence[Mapping[str, Any]],
    *,
    attempt_id: str,
    boundary: str = "session/fold manifest load",
) -> dict[str, int]:
    membership_rows: list[tuple[tuple[str, str, str, str], dict[str, Any]]] = []
    validation_fold: dict[str, tuple[str, dict[str, Any]]] = {}
    seen_role_by_fold_session: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    role_overlaps = 0
    for fold_index, fold in enumerate(folds):
        fold_id = str(fold.get("fold_id") or f"fold_{fold_index + 1}")
        role_values = [
            ("train", fold.get("train_sessions") or []),
            ("fit", fold.get("fit_sessions") or []),
            ("calibration", fold.get("calibration_sessions") or []),
            ("validation", fold.get("validation_sessions") or []),
            (
                "embargo",
                [
                    value.get("session")
                    if isinstance(value, Mapping)
                    else value
                    for value in (fold.get("embargoed_sessions") or [])
                ],
            ),
        ]
        for role, values in role_values:
            for position, session_value in enumerate(values):
                session = str(session_value)
                locator = {
                    "fold_id": fold_id,
                    "role": role,
                    "position": position,
                    "session": session,
                }
                membership_rows.append(
                    ((str(attempt_id), fold_id, role, session), locator)
                )
                fold_session_key = (fold_id, session)
                previous_role = seen_role_by_fold_session.get(fold_session_key)
                if previous_role is not None and previous_role[0] != role:
                    role_overlaps += 1
                    raise Protocol101SessionRoleOverlapError(
                        (
                            "P101_ID_SESSION_ROLE_OVERLAP: session appears in "
                            "multiple roles inside one fold"
                        ),
                        canonical_key=(str(attempt_id), fold_id, session),
                        first_source_locator=previous_role[1],
                        duplicate_source_locator=locator,
                        observed_count=2,
                        boundary=boundary,
                    )
                seen_role_by_fold_session[fold_session_key] = (role, locator)
                if role == "validation":
                    previous_fold = validation_fold.get(session)
                    if previous_fold is not None and previous_fold[0] != fold_id:
                        raise Protocol101ValidationFoldOverlapError(
                            (
                                "P101_ID_VALIDATION_FOLD_OVERLAP: validation "
                                "session appears in multiple outer folds"
                            ),
                            canonical_key=(str(attempt_id), session),
                            first_source_locator=previous_fold[1],
                            duplicate_source_locator=locator,
                            observed_count=2,
                            boundary=boundary,
                        )
                    validation_fold[session] = (fold_id, locator)
    count = _assert_unique(
        membership_rows,
        error_type=Protocol101DuplicateSessionMembershipError,
        boundary=boundary,
    )
    return {
        "rows_seen": count,
        "unique_identities": count,
        "duplicate_identities": 0,
        "role_overlaps": role_overlaps,
    }


def assert_decision_row_identities(
    row: Mapping[str, Any],
    *,
    split: str,
    session: str,
    boundary: str = "canonical decision construction",
) -> dict[str, int]:
    decision_time_ns = utc_timestamp_ns(row.get("decision_time"))
    contract_ids = np.asarray(row.get("contract_ids"), dtype=object)
    offsets = np.asarray(row.get("strike_offsets"), dtype=float)
    rights = tuple(str(value) for value in (row.get("rights") or ()))
    if (
        contract_ids.ndim != 2
        or contract_ids.shape != (len(offsets), len(rights))
        or len(rights) != len(set(rights))
        or set(rights) != {"C", "P"}
    ):
        raise Protocol101PolicyAxisAlignmentError(
            "contract/right axes are missing or misaligned",
            canonical_key=(split, session, decision_time_ns),
            boundary=boundary,
        )
    if len(offsets) != len(set(float(value) for value in offsets)):
        duplicate = next(
            value
            for value in offsets
            if list(offsets).count(value) > 1
        )
        raise Protocol101DuplicateCanonicalSlotError(
            "duplicate canonical strike offset",
            canonical_key=(split, session, decision_time_ns, float(duplicate)),
            observed_count=2,
            boundary=boundary,
        )
    contract_rows: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    slot_rows: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for strike_idx in range(contract_ids.shape[0]):
        for right_idx in range(contract_ids.shape[1]):
            contract_id = str(contract_ids[strike_idx, right_idx] or "")
            right = rights[right_idx]
            locator = {
                "strike_index": strike_idx,
                "right_index": right_idx,
                "contract_id": contract_id,
            }
            if contract_id:
                contract_rows.append(
                    (
                        (
                            str(split),
                            str(session),
                            decision_time_ns,
                            contract_id,
                        ),
                        locator,
                    )
                )
            slot_rows.append(
                (
                    (
                        str(split),
                        str(session),
                        decision_time_ns,
                        strike_idx,
                        right,
                    ),
                    locator,
                )
            )
    _assert_unique(
        contract_rows,
        error_type=Protocol101DuplicateContractIdentityError,
        boundary=boundary,
    )
    _assert_unique(
        slot_rows,
        error_type=Protocol101DuplicateCanonicalSlotError,
        boundary=boundary,
    )

    if row.get("processed_row_schema_version") == TWO_CLOCK_PROCESSED_ROW_SCHEMA:
        policy_axis = np.asarray(row.get("label_policy_index"), dtype=np.uint8)
        expected = np.arange(7, dtype=np.uint8)
        if policy_axis.ndim != 1 or not np.array_equal(policy_axis, expected):
            raise Protocol101PolicyAxisAlignmentError(
                "two-clock policy axis is unordered or misaligned",
                canonical_key=(str(split), str(session), decision_time_ns),
                boundary=boundary,
            )
        expected_shape = (len(offsets), len(rights), len(policy_axis))
        for name in (
            "labels_net_pnl",
            "labels_mid_pnl",
            "label_realized_exit_time_ns",
            "label_source_exit_quote_time_ns",
            "label_exit_quote_age_ms",
            "label_exit_reason_code",
            "label_executable_exit_bid",
            "label_policy_deadline_ns",
            "label_invalid_reason_code",
        ):
            if np.asarray(row.get(name)).shape != expected_shape:
                raise Protocol101PolicyAxisAlignmentError(
                    f"{name} shape is not aligned to [strike,right,policy]",
                    canonical_key=(str(split), str(session), decision_time_ns),
                    boundary=boundary,
                )
    return {
        "rows_seen": len(contract_rows),
        "unique_identities": len(contract_rows),
        "duplicate_identities": 0,
        "role_overlaps": 0,
    }


def assert_processed_row_identities(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    session: str,
    boundary: str = "processed-row load before target construction",
) -> dict[str, int]:
    decision_rows = []
    contract_count = 0
    for row_index, row in enumerate(rows):
        decision_time_ns = utc_timestamp_ns(row.get("decision_time"))
        decision_rows.append(
            (
                (str(split), str(session), decision_time_ns),
                {"row_index": row_index, "decision_time_ns": decision_time_ns},
            )
        )
        result = assert_decision_row_identities(
            row,
            split=split,
            session=session,
            boundary=boundary,
        )
        contract_count += result["rows_seen"]
    count = _assert_unique(
        decision_rows,
        error_type=Protocol101DuplicateDecisionIdentityError,
        boundary=boundary,
    )
    return {
        "rows_seen": count,
        "unique_identities": count,
        "duplicate_identities": 0,
        "role_overlaps": 0,
        "contract_identities": contract_count,
    }


FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "labels_net_pnl",
        "labels_mid_pnl",
        "label_realized_exit_time_ns",
        "label_source_exit_quote_time_ns",
        "label_exit_quote_age_ms",
        "label_exit_reason_code",
        "label_executable_exit_bid",
        "label_policy_deadline_ns",
        "label_policy_index",
        "label_invalid_reason_code",
    }
)
FORBIDDEN_ALPHA_TOKENS = (
    "future",
    "path",
    "label",
    "pnl",
    "mfe",
    "mae",
    "exit_quote",
    "exit_time",
    "realized_exit",
    "policy_deadline",
)


def assert_alpha_feature_names(
    feature_names: Sequence[str],
    *,
    allowed_feature_sets: Iterable[Sequence[str]],
    boundary: str = "model matrix construction",
) -> tuple[str, ...]:
    """Require one explicit signed allowlist; wildcard discovery is forbidden."""

    names = tuple(str(value) for value in feature_names)
    allowed = {tuple(str(value) for value in values) for values in allowed_feature_sets}
    bad = sorted(
        {
            name
            for name in names
            if name in FORBIDDEN_MODEL_FIELDS
            or any(token in name.lower() for token in FORBIDDEN_ALPHA_TOKENS)
            or any(wildcard in name for wildcard in ("*", "?", "[", "]"))
        }
    )
    if bad or names not in allowed:
        raise Protocol101AlphaFirewallError(
            "model feature names are not one exact signed Stage-1 allowlist",
            canonical_key={"feature_names": names, "forbidden": bad},
            boundary=boundary,
        )
    return names


def _time_of_day_bucket(value_ns: int) -> str:
    local = pd.Timestamp(int(value_ns), unit="ns", tz="UTC").tz_convert(
        "America/New_York"
    )
    minute = local.hour * 60 + local.minute
    if minute < 10 * 60 + 30:
        return "open"
    if minute < 14 * 60:
        return "midday"
    return "late"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_exit_quote_age_report(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate diagnostic-only quote age without any eligibility threshold."""

    grouped: dict[tuple[int, str, str], dict[str, Any]] = {}
    total = 0
    for record in records:
        total += 1
        policy = int(record["policy_index"])
        session = str(record["session"])
        valid = bool(record.get("valid"))
        clock = (
            record.get("label_realized_exit_time_ns")
            if valid
            else record.get("label_policy_deadline_ns")
        )
        if clock in (None, INT64_MISSING):
            bucket = "unknown"
        else:
            bucket = _time_of_day_bucket(int(clock))
        group = grouped.setdefault(
            (policy, session, bucket),
            {"valid_ages": [], "invalid_count": 0},
        )
        age = _finite_float(record.get("label_exit_quote_age_ms"))
        if valid:
            if age is None or age < 0.0:
                raise Protocol101RegimenRepairError(
                    "valid quote-age record is missing, nonfinite, or negative",
                    canonical_key=(policy, session, bucket),
                    boundary="quote-age report construction",
                )
            group["valid_ages"].append(age)
        else:
            group["invalid_count"] += 1

    rows: list[dict[str, Any]] = []
    for (policy, session, bucket), values in sorted(grouped.items()):
        ages = np.asarray(values["valid_ages"], dtype=np.float64)
        valid_count = int(len(ages))
        row = {
            "policy_index": policy,
            "session": session,
            "time_of_day_bucket": bucket,
            "valid_count": valid_count,
            "invalid_count": int(values["invalid_count"]),
            "zero_age_share": (
                float(np.mean(ages == 0.0)) if valid_count else None
            ),
            "minimum_ms": float(np.min(ages)) if valid_count else None,
            "mean_ms": float(np.mean(ages)) if valid_count else None,
            "p50_ms": float(np.quantile(ages, 0.50)) if valid_count else None,
            "p90_ms": float(np.quantile(ages, 0.90)) if valid_count else None,
            "p95_ms": float(np.quantile(ages, 0.95)) if valid_count else None,
            "p99_ms": float(np.quantile(ages, 0.99)) if valid_count else None,
            "maximum_ms": float(np.max(ages)) if valid_count else None,
        }
        rows.append(row)
    return {
        "schema_version": EXIT_QUOTE_AGE_REPORT_SCHEMA,
        "contract_id": PROTOCOL101_CONTRACT_ID,
        "status": "diagnostic_only",
        "gate": False,
        "rejection_threshold_ms": None,
        "dimensions": ["policy", "session", "realized_exit_time_of_day"],
        "record_count": total,
        "groups": rows,
    }
