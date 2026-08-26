"""Fail-closed Job-52 paid-resume controls for the frozen Job-51 CMBP scope.

The Job-51 V1 module and evidence remain immutable.  This overlay reuses its
scope reconstruction, volume guard, durable journal primitive, DBN iterator,
and decoder, while applying the owner's lifetime USD 1.50-per-session and USD
32.00-all-session committed-quote ceilings.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import stat
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from v5.research import cmbp_tier0 as base


JOB_ID = 52
TARGET_JOB_ID = 51
CONTRACT_ARTIFACT = "JOB52_CMBP_TIER0_PAID_RESUME_PROGRAM_CONTRACT_V1"
READINESS_ARTIFACT = "JOB52_CMBP_TIER0_PAID_RESUME_LOCAL_READINESS_RECEIPT_V1"
READINESS_STATUS = "JOB52_CMBP_TIER0_PAID_RESUME_LOCAL_READY_ONLY"
SESSION_QC_ARTIFACT = "JOB52_CMBP_TIER0_PAID_SESSION_QC_V1"
AGGREGATE_RECEIPT_ARTIFACT = "JOB52_CMBP_TIER0_PAID_ACQUISITION_QC_RECEIPT_V1"
ATTEMPT_STOP_ARTIFACT = "JOB52_CMBP_TIER0_PAID_ATTEMPT_STOP_V1"
EXPECTED_PROGRAM_CONTRACT_SHA256 = "094733cb8e6145214161cf4fd3765b3667cc8fabac53eea9abc08796d00600cf"
EXPECTED_PROGRAM_CONTRACT_FILE_SHA256 = "befa3f3fa2e5d7f5d4b667da78061a2a5def7c1b4802a8c6b7a2cf801af277ac"
EXPECTED_PLAN_FILE_SHA256 = "7eed19e91e6e094778ba7d596ff24dc716557742cf9bda17cbbea7351254b0ba"
EXPECTED_LEGACY_ATTEMPT_ID = "a5bd7111-0ac5-4992-8308-32fdded55134"
EXPECTED_LEGACY_STOP_RECEIPT_FILE_SHA256 = "9be49967b29e2f447a5af5f43569d57741911b47cd0c0d23ca89e35fab454d93"
EXPECTED_LEGACY_STOP_RECEIPT_SHA256 = "b886c418e5661f3589abf68475f09ccbe5dcffd89de1e8210711e990ffa673e3"
PER_SESSION_LIFETIME_CAP_USD = Decimal("1.50")
TOTAL_COMMITTED_CAP_USD = Decimal("32.00")
EXPECTED_ZSTANDARD_VERSION = "0.25.0"
EXPECTED_CERTIFI_VERSION = "2025.11.12"
EXPECTED_IDNA_VERSION = "3.13"
EXPECTED_CHARSET_NORMALIZER_VERSION = "3.4.7"
EXPECTED_FOCUSED_TEST_COUNT = 208
EXPECTED_FOCUSED_TEST_IDENTITY_SHA256 = "f773980a8cbdc1ea26df9174767685135ee5a435893cec34d6d0c52adaf5ecfd"
PAID_AUTHORITY = "OWNER_CURRENT_CONVERSATION_CAPPED_PAID_RESUME"
PAID_ATTEMPT_STOP_NAME = "ATTEMPT_STOP_PAID_V1.json"
PAID_SESSION_QC_NAME = "SESSION_QC_V2.json"
PAID_AGGREGATE_NAME = "JOB52_CMBP_TIER0_PAID_ACQUISITION_QC_RECEIPT_V1.json"
PAID_ATTEMPT_ANCHOR_NAME = "JOB52_ATTEMPT_SET_ANCHOR_V1.json"
PAID_LOCK_BINDING_NAME = "RUN_LOCK_JOB52_BINDING_V1.json"
SHA256_RE = base.SHA256_RE
Tier0Error = base.Tier0Error
SessionRequest = base.SessionRequest
VolumeIdentity = base.VolumeIdentity
AttemptJournal = base.AttemptJournal


@dataclass(frozen=True)
class PaidScopeBundle:
    """Exact Job-51 scope plus the independently frozen Job-52 overlay."""

    base_bundle: base.ScopeBundle
    contract: Mapping[str, Any]
    contract_file_sha256: str
    plan_file_sha256: str

    @property
    def sessions(self) -> tuple[SessionRequest, ...]:
        return self.base_bundle.sessions


def _paths(repo_root: Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    return {
        "plan": root / "v5/work/cmbp-tier0-paid-resume/PLAN.md",
        "contract": root / "v5/work/cmbp-tier0-paid-resume/PROGRAM_CONTRACT_V1.json",
        "readiness": root / "v5/work/cmbp-tier0-paid-resume/LOCAL_READINESS_RECEIPT_V1.json",
        "test_report": root / "v5/work/cmbp-tier0-paid-resume/TEST_RESULTS_V1.xml",
        "legacy_stop": root / "v5/work/cmbp-tier0-acquisition/JOB51_ZERO_COST_GATE_STOP_RECEIPT_V1.json",
    }


def _private_regular(path: Path, *, status: str) -> os.stat_result:
    try:
        metadata = Path(path).lstat()
    except FileNotFoundError as exc:
        raise Tier0Error(f"required file is absent: {Path(path).name}", status=status) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise Tier0Error(f"required file is unsafe: {Path(path).name}", status=status)
    return metadata


def load_paid_scope_bundle(repo_root: Path) -> PaidScopeBundle:
    """Reconstruct immutable Job-51 evidence and the exact paid overlay."""

    root = Path(repo_root).resolve()
    paths = _paths(root)
    for key in ("plan", "contract", "legacy_stop"):
        _private_regular(paths[key], status="STOP_PAID_CONTRACT_DRIFT")
    base_bundle = base.load_scope_bundle(root)
    if base.file_sha256(paths["plan"]) != EXPECTED_PLAN_FILE_SHA256:
        raise Tier0Error("Job-52 paid plan raw hash drifted", status="STOP_PAID_CONTRACT_DRIFT")
    contract = base.strict_json(paths["contract"])
    contract_file_sha = base.file_sha256(paths["contract"])
    if (
        contract.get("artifact_type") != CONTRACT_ARTIFACT
        or contract.get("schema_version") != "v5.job52-cmbp-tier0-paid-resume-program-contract.v1"
        or contract.get("job_id") != JOB_ID
        or contract.get("target_job_id") != TARGET_JOB_ID
        or contract.get("contract_sha256") != EXPECTED_PROGRAM_CONTRACT_SHA256
        or base.self_hash(contract, "contract_sha256") != EXPECTED_PROGRAM_CONTRACT_SHA256
        or contract_file_sha != EXPECTED_PROGRAM_CONTRACT_FILE_SHA256
        or contract.get("plan_file_sha256") != EXPECTED_PLAN_FILE_SHA256
    ):
        raise Tier0Error("Job-52 paid contract identity/hash drifted", status="STOP_PAID_CONTRACT_DRIFT")
    authority = contract.get("authority", {})
    commitment = contract.get("commitment_law", {})
    scope = contract.get("scope", {})
    if (
        authority.get("owner_authorized_in_current_conversation") is not True
        or authority.get("per_session_lifetime_committed_quote_cap_usd") != "1.50"
        or authority.get("total_committed_quote_cap_usd") != "32.00"
        or commitment.get("maximum_per_session_lifetime_usd") != "1.50"
        or commitment.get("maximum_total_usd") != "32.00"
        or commitment.get("actual_vendor_invoice_cost_usd") != "UNKNOWN"
        or commitment.get("prior_job51_quote_committed_usd") != "0"
        or commitment.get("prior_job51_quote_observed_usd") != "0.950392448902"
        or scope.get("semantic_sha256") != base.EXPECTED_SCOPE_SHA256
        or scope.get("raw_file_sha256") != base.EXPECTED_SCOPE_FILE_SHA256
        or scope.get("session_count") != base.EXPECTED_SESSION_COUNT
        or scope.get("total_record_count") != base.EXPECTED_RECORD_COUNT
        or scope.get("total_session_symbols") != base.EXPECTED_SESSION_SYMBOLS
    ):
        raise Tier0Error("Job-52 paid authority/scope drifted", status="STOP_PAID_CONTRACT_DRIFT")
    legacy = base.strict_json(paths["legacy_stop"])
    if (
        base.file_sha256(paths["legacy_stop"]) != EXPECTED_LEGACY_STOP_RECEIPT_FILE_SHA256
        or legacy.get("receipt_sha256") != EXPECTED_LEGACY_STOP_RECEIPT_SHA256
        or base.self_hash(legacy, "receipt_sha256") != EXPECTED_LEGACY_STOP_RECEIPT_SHA256
        or legacy.get("attempt", {}).get("attempt_id") != EXPECTED_LEGACY_ATTEMPT_ID
        or legacy.get("cost_boundary", {}).get("observed_sdk_quote_usd") != "0.950392448902"
        or legacy.get("call_accounting", {}).get("time_series_call_starts") != 0
    ):
        raise Tier0Error("Job-51 legacy stop receipt drifted", status="STOP_PAID_CONTRACT_DRIFT")
    return PaidScopeBundle(
        base_bundle=base_bundle,
        contract=contract,
        contract_file_sha256=contract_file_sha,
        plan_file_sha256=base.file_sha256(paths["plan"]),
    )


def normalize_paid_quote(value: Any) -> str:
    """Return the exact finite unsigned numeric quote text used for accounting."""

    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise Tier0Error("fresh SDK cost quote is not numeric", status="STOP_PAID_QUOTE_INVALID")
    if isinstance(value, float) and not math.isfinite(value):
        raise Tier0Error("fresh SDK cost quote is nonfinite", status="STOP_PAID_QUOTE_INVALID")
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal.is_finite() or decimal.is_signed() or decimal < 0:
        raise Tier0Error("fresh SDK cost quote is negative, signed zero, or nonfinite", status="STOP_PAID_QUOTE_INVALID")
    return format(decimal, "f")


def _exact_nonnegative_add(left: Decimal, right: Decimal) -> Decimal:
    """Add nonnegative Decimals without dependence on the process Decimal context."""

    if not left.is_finite() or not right.is_finite() or left.is_signed() or right.is_signed():
        raise Tier0Error("paid commitment operand is invalid", status="STOP_PAID_BUDGET_INVALID")
    left_tuple = left.as_tuple()
    right_tuple = right.as_tuple()
    exponent = min(left_tuple.exponent, right_tuple.exponent)

    def coefficient(value: Decimal, value_tuple: Any) -> int:
        digits = int("".join(str(item) for item in value_tuple.digits) or "0")
        return digits * (10 ** (value_tuple.exponent - exponent))

    total = coefficient(left, left_tuple) + coefficient(right, right_tuple)
    digits = tuple(int(item) for item in str(total)) if total else (0,)
    return Decimal((0, digits, exponent))


def project_paid_commitment(
    *,
    committed_total_usd: Decimal,
    committed_session_usd: Decimal,
    fresh_quote_usd: Decimal,
) -> dict[str, Any]:
    """Pure, precision-independent projection for both owner caps."""

    for value in (committed_total_usd, committed_session_usd, fresh_quote_usd):
        if not isinstance(value, Decimal) or not value.is_finite() or value.is_signed() or value < 0:
            raise Tier0Error("paid commitment projection operand is invalid", status="STOP_PAID_BUDGET_INVALID")
    session_after = _exact_nonnegative_add(committed_session_usd, fresh_quote_usd)
    total_after = _exact_nonnegative_add(committed_total_usd, fresh_quote_usd)
    return {
        "committed_quote_session_after_usd": session_after,
        "committed_quote_total_after_usd": total_after,
        "per_session_lifetime_cap_pass": session_after <= PER_SESSION_LIFETIME_CAP_USD,
        "total_cap_pass": total_after <= TOTAL_COMMITTED_CAP_USD,
    }


@dataclass
class PaidBudgetState:
    """In-memory mirror of disk-reconstructed, irrevocable start commitments."""

    committed_total_usd: Decimal = Decimal("0")
    committed_by_session_usd: dict[str, Decimal] = field(default_factory=dict)
    commitment_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.committed_total_usd, Decimal):
            self.committed_total_usd = Decimal(str(self.committed_total_usd))
        if (
            not self.committed_total_usd.is_finite()
            or self.committed_total_usd.is_signed()
            or self.committed_total_usd < 0
        ):
            raise Tier0Error("paid budget total is invalid", status="STOP_PAID_BUDGET_INVALID")
        self.committed_by_session_usd = {
            str(session): value if isinstance(value, Decimal) else Decimal(str(value))
            for session, value in self.committed_by_session_usd.items()
        }
        reconstructed = Decimal("0")
        for session, value in self.committed_by_session_usd.items():
            if not session or not value.is_finite() or value.is_signed() or value < 0:
                raise Tier0Error("paid budget state contains an invalid session commitment", status="STOP_PAID_BUDGET_INVALID")
            if value > PER_SESSION_LIFETIME_CAP_USD:
                raise Tier0Error("paid budget state exceeds a session cap", status="STOP_PAID_SESSION_CAP")
            reconstructed = _exact_nonnegative_add(reconstructed, value)
        if reconstructed != self.committed_total_usd or self.committed_total_usd > TOTAL_COMMITTED_CAP_USD:
            raise Tier0Error("paid budget state total does not reconcile", status="STOP_PAID_TOTAL_CAP")
        if isinstance(self.commitment_count, bool) or not isinstance(self.commitment_count, int) or self.commitment_count < 0:
            raise Tier0Error("paid budget commitment count is invalid", status="STOP_PAID_BUDGET_INVALID")

    def projection(self, session: str, quote: Decimal) -> dict[str, Any]:
        current_session = self.committed_by_session_usd.get(session, Decimal("0"))
        return project_paid_commitment(
            committed_total_usd=self.committed_total_usd,
            committed_session_usd=current_session,
            fresh_quote_usd=quote,
        )

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any]) -> "PaidBudgetState":
        by_session = summary.get("committed_quote_by_session_usd")
        total = summary.get("committed_quote_total_usd")
        count = summary.get("timeseries_call_starts")
        if not isinstance(by_session, dict) or not isinstance(total, str):
            raise Tier0Error("paid attempt summary budget fields are invalid", status="STOP_PAID_BUDGET_INVALID")
        if isinstance(count, bool) or not isinstance(count, int):
            raise Tier0Error("paid attempt summary start count is invalid", status="STOP_PAID_BUDGET_INVALID")
        try:
            parsed = {str(session): Decimal(str(value)) for session, value in by_session.items()}
            parsed_total = Decimal(total)
        except Exception as exc:  # noqa: BLE001
            raise Tier0Error("paid attempt summary decimal is invalid", status="STOP_PAID_BUDGET_INVALID") from exc
        return cls(
            committed_total_usd=parsed_total,
            committed_by_session_usd=parsed,
            commitment_count=count,
        )

    def commit(self, session: str, quote: Decimal, projection: Mapping[str, Any]) -> None:
        expected = self.projection(session, quote)
        if dict(projection) != expected or not expected["per_session_lifetime_cap_pass"] or not expected["total_cap_pass"]:
            raise Tier0Error("paid commitment cannot be applied", status="STOP_PAID_BUDGET_INVALID")
        self.committed_by_session_usd[session] = expected["committed_quote_session_after_usd"]
        self.committed_total_usd = expected["committed_quote_total_after_usd"]
        self.commitment_count += 1


def _safe_quote_observation(value: Any) -> str:
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return repr(value)
    return type(value).__name__


def _paid_start_payload(
    *,
    request: SessionRequest,
    output_path: Path,
    quote_text: str,
    cost_result_record_hash: str,
    state: PaidBudgetState,
    projection: Mapping[str, Any],
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    ordinal: int,
) -> dict[str, Any]:
    session_before = state.committed_by_session_usd.get(request.session, Decimal("0"))
    session_after = projection["committed_quote_session_after_usd"]
    total_after = projection["committed_quote_total_after_usd"]
    return {
        "method": "timeseries.get_range",
        "parameters": {**request.market_parameters, "stype_out": base.EXPECTED_STYPE_OUT, "limit": None},
        "output_relative": output_path.name,
        "fresh_quote_usd": quote_text,
        "immediately_preceding_quote_usd": quote_text,
        "cost_result_record_hash": cost_result_record_hash,
        "commitment_index": state.commitment_count + 1,
        "ordinal": ordinal,
        "committed_quote_session_before_usd": format(session_before, "f"),
        "committed_quote_session_after_usd": format(session_after, "f"),
        "committed_quote_session_usd": format(session_after, "f"),
        "committed_quote_total_before_usd": format(state.committed_total_usd, "f"),
        "committed_quote_total_after_usd": format(total_after, "f"),
        "committed_quote_total_usd": format(total_after, "f"),
        "per_session_lifetime_cap_usd": "1.50",
        "total_cap_usd": "32.00",
        "program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "readiness_receipt_file_sha256": readiness_receipt_file_sha256,
    }


def acquire_paid_session_bytes(
    client: Any,
    request: SessionRequest,
    *,
    output_path: Path,
    journal: Any,
    pre_pair_gate: Callable[[], None],
    budget_state: PaidBudgetState,
    readiness_receipt_sha256: str = "UNSEALED_TEST_ONLY",
    readiness_receipt_file_sha256: str = "UNSEALED_TEST_ONLY",
    ordinal: int = 0,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Freshly quote, durably commit, and stream one exact frozen request."""

    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise Tier0Error("session output path already exists", status="STOP_EXTERNAL_PATH")
    if progress is not None:
        progress(f"{request.session}: requesting fresh capped cost quote")
    pre_pair_gate()
    request_sha = request.market_request_sha256
    journal.append(
        "COST_CALL_START",
        session=request.session,
        request_sha256=request_sha,
        payload={"method": "metadata.get_cost", "parameters": request.market_parameters},
    )
    try:
        observed = client.metadata.get_cost(**request.market_parameters)
    except Exception as exc:  # noqa: BLE001 - vendor text may contain secrets
        journal.append(
            "COST_CALL_ERROR",
            session=request.session,
            request_sha256=request_sha,
            payload={"error_class": type(exc).__name__, "actual_vendor_invoice_cost_usd": "UNKNOWN"},
        )
        raise Tier0Error("fresh paid SDK cost request failed", status="STOP_VENDOR_COST_CALL") from exc

    session_before = budget_state.committed_by_session_usd.get(request.session, Decimal("0"))
    try:
        quote_text = normalize_paid_quote(observed)
        quote_decimal = Decimal(quote_text)
    except Tier0Error:
        journal.append(
            "COST_CALL_RESULT",
            session=request.session,
            request_sha256=request_sha,
            payload={
                "observed_sdk_quote": _safe_quote_observation(observed),
                "quote_valid": False,
                "commitment_count_before": budget_state.commitment_count,
                "committed_quote_session_before_usd": format(session_before, "f"),
                "committed_quote_total_before_usd": format(budget_state.committed_total_usd, "f"),
                "per_session_lifetime_cap_usd": "1.50",
                "total_cap_usd": "32.00",
                "per_session_lifetime_cap_pass": False,
                "total_cap_pass": False,
                "time_series_start_permitted": False,
                "actual_vendor_invoice_cost_usd": "UNKNOWN",
            },
        )
        raise

    projection = budget_state.projection(request.session, quote_decimal)
    price_gate_pass = bool(projection["per_session_lifetime_cap_pass"] and projection["total_cap_pass"])
    cost_result = journal.append(
        "COST_CALL_RESULT",
        session=request.session,
        request_sha256=request_sha,
        payload={
            "observed_sdk_quote_usd": quote_text,
            "quote_valid": True,
            "commitment_count_before": budget_state.commitment_count,
            "committed_quote_session_before_usd": format(session_before, "f"),
            "committed_quote_session_projected_usd": format(projection["committed_quote_session_after_usd"], "f"),
            "committed_quote_total_before_usd": format(budget_state.committed_total_usd, "f"),
            "committed_quote_total_projected_usd": format(projection["committed_quote_total_after_usd"], "f"),
            "per_session_lifetime_cap_usd": "1.50",
            "total_cap_usd": "32.00",
            "per_session_lifetime_cap_pass": projection["per_session_lifetime_cap_pass"],
            "total_cap_pass": projection["total_cap_pass"],
            "time_series_start_permitted": price_gate_pass,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
        },
    )
    if not projection["per_session_lifetime_cap_pass"]:
        raise Tier0Error("session lifetime committed quote would exceed USD 1.50", status="STOP_PAID_SESSION_CAP")
    if not projection["total_cap_pass"]:
        raise Tier0Error("total committed quote would exceed USD 32.00", status="STOP_PAID_TOTAL_CAP")

    # Nothing except these two required durable records may intervene between
    # the fresh quote result and the irrevocable time-series start commitment.
    start = journal.append(
        "TIMESERIES_CALL_START",
        session=request.session,
        request_sha256=request_sha,
        payload=_paid_start_payload(
            request=request,
            output_path=output_path,
            quote_text=quote_text,
            cost_result_record_hash=str(cost_result["record_hash"]),
            state=budget_state,
            projection=projection,
            readiness_receipt_sha256=readiness_receipt_sha256,
            readiness_receipt_file_sha256=readiness_receipt_file_sha256,
            ordinal=ordinal,
        ),
    )
    budget_state.commit(request.session, quote_decimal, projection)
    result_context = {
        "fresh_quote_usd": quote_text,
        "commitment_index": budget_state.commitment_count,
        "timeseries_start_record_hash": start["record_hash"],
        "committed_quote_session_usd": format(budget_state.committed_by_session_usd[request.session], "f"),
        "committed_quote_total_usd": format(budget_state.committed_total_usd, "f"),
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
    }
    try:
        client.timeseries.get_range(
            **request.market_parameters,
            stype_out=base.EXPECTED_STYPE_OUT,
            limit=None,
            path=output_path,
        )
    except Exception as exc:  # noqa: BLE001 - redact vendor exception text
        journal.append(
            "TIMESERIES_CALL_ERROR",
            session=request.session,
            request_sha256=request_sha,
            payload={**result_context, "error_class": type(exc).__name__},
        )
        raise Tier0Error("paid time-series stream failed", status="STOP_VENDOR_TIMESERIES_CALL") from exc
    output_metadata = output_path.lstat() if output_path.exists() or output_path.is_symlink() else None
    parent_metadata = output_path.parent.lstat()
    if (
        output_metadata is None
        or stat.S_ISLNK(output_metadata.st_mode)
        or not stat.S_ISREG(output_metadata.st_mode)
        or output_metadata.st_nlink != 1
        or output_metadata.st_dev != parent_metadata.st_dev
        or output_metadata.st_size <= 0
    ):
        journal.append(
            "TIMESERIES_CALL_ERROR",
            session=request.session,
            request_sha256=request_sha,
            payload={**result_context, "error_class": "MissingOrEmptyOutput"},
        )
        raise Tier0Error("paid time-series stream produced no safe file", status="STOP_VENDOR_TIMESERIES_CALL")
    descriptor = os.open(output_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    base.fsync_directory(output_path.parent)
    journal.append(
        "TIMESERIES_CALL_RESULT",
        session=request.session,
        request_sha256=request_sha,
        payload={
            **result_context,
            "compressed_bytes": output_metadata.st_size,
            "dbn_file_sha256": base.file_sha256(output_path),
        },
    )
    if progress is not None:
        progress(f"{request.session}: capped DBN stream durably recorded")
    return quote_text


class PaidRunLock:
    """Flock the persistent Job-51 lock without rewriting its V1 bytes."""

    def __init__(self, job_root: Path, *, volume: VolumeIdentity) -> None:
        self.path = Path(job_root) / "RUN_LOCK_V1"
        self.fd: int | None = None
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except FileNotFoundError as exc:
            raise Tier0Error("persistent Job-51 run lock is absent", status="STOP_CONCURRENT_RUNNER") from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_dev != volume.st_dev:
            os.close(descriptor)
            raise Tier0Error("persistent Job-51 run lock is unsafe", status="STOP_CONCURRENT_RUNNER")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise Tier0Error("another Job-51/52 runner holds the lock", status="STOP_CONCURRENT_RUNNER") from exc
            raise
        expected = base.canonical_json_bytes(
            {
                "artifact_type": "JOB51_RUN_LOCK_V1",
                "readiness_sha256": "851ae1de70308a19fad4160bf33464a3e0a2c79bba15085bde80aea5ea99ff03",
                "scope_sha256": base.EXPECTED_SCOPE_SHA256,
            }
        ) + b"\n"
        os.lseek(descriptor, 0, os.SEEK_SET)
        actual = os.read(descriptor, max(len(expected) + 1, metadata.st_size + 1))
        if actual != expected:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise Tier0Error("persistent Job-51 run-lock binding drifted", status="STOP_CONCURRENT_RUNNER")
        self.fd = descriptor

    def close(self) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    def __enter__(self) -> "PaidRunLock":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def ensure_paid_lock_binding(
    job_root: Path,
    *,
    volume: VolumeIdentity,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
) -> dict[str, Any]:
    """Bind paid authority beside—without rewriting—the shared V1 flock file."""

    if SHA256_RE.fullmatch(readiness_receipt_sha256) is None or SHA256_RE.fullmatch(readiness_receipt_file_sha256) is None:
        raise Tier0Error("paid lock readiness identity is invalid", status="STOP_PAID_LOCK_BINDING")
    path = Path(job_root) / PAID_LOCK_BINDING_NAME
    receipt: dict[str, Any] = {
        "artifact_type": "JOB52_CMBP_TIER0_PAID_RUN_LOCK_BINDING_V1",
        "schema_version": "v5.job52-cmbp-tier0-paid-run-lock-binding.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "paid_program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "paid_readiness_receipt_sha256": readiness_receipt_sha256,
        "paid_readiness_receipt_file_sha256": readiness_receipt_file_sha256,
        "shared_flock_file": "RUN_LOCK_V1",
        "shared_flock_bytes_mutated": False,
    }
    receipt["binding_sha256"] = base.self_hash(receipt, "binding_sha256")
    if not path.exists() and not path.is_symlink():
        attempts_root = Path(job_root) / "attempts"
        paid_attempts_exist = False
        if attempts_root.exists() and not attempts_root.is_symlink():
            paid_attempts_exist = any(
                child.name != EXPECTED_LEGACY_ATTEMPT_ID
                for child in attempts_root.iterdir()
            )
        if paid_attempts_exist or (Path(job_root) / PAID_ATTEMPT_ANCHOR_NAME).exists():
            raise Tier0Error(
                "paid lock binding disappeared after Job-52 state existed",
                status="STOP_PAID_LOCK_BINDING",
            )
        base.write_canonical_exclusive(path, receipt)
    _private_regular(path, status="STOP_PAID_LOCK_BINDING")
    if path.stat().st_dev != volume.st_dev:
        raise Tier0Error("paid lock binding is cross-device", status="STOP_PAID_LOCK_BINDING")
    actual = base.strict_json(path)
    if actual != receipt:
        raise Tier0Error("paid lock binding drifted", status="STOP_PAID_LOCK_BINDING")
    return actual


def _anchor_receipt(
    *,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    paid_attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "artifact_type": "JOB52_CMBP_TIER0_PAID_ATTEMPT_SET_ANCHOR_V1",
        "schema_version": "v5.job52-cmbp-tier0-paid-attempt-set-anchor.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "paid_program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "paid_readiness_receipt_sha256": readiness_receipt_sha256,
        "paid_readiness_receipt_file_sha256": readiness_receipt_file_sha256,
        "legacy_attempt_id": EXPECTED_LEGACY_ATTEMPT_ID,
        "paid_attempt_count": len(paid_attempts),
        "paid_attempts": sorted(
            (dict(item) for item in paid_attempts),
            key=lambda item: item["paid_attempt_ordinal"],
        ),
        "threat_boundary": "DETECTS_MISSING_OR_ROLLED_BACK_SINGLE_DESTINATION_COMPONENT; COORDINATED_REWRITE_REQUIRES_OUTSIDE_ANCHOR",
    }
    receipt["anchor_sha256"] = base.self_hash(receipt, "anchor_sha256")
    return receipt


def _validate_paid_attempt_lineage(
    entries: Sequence[Mapping[str, Any]],
    *,
    status: str = "STOP_PAID_ATTEMPT_SET",
) -> list[Mapping[str, Any]]:
    """Return attempts in their durable, globally contiguous creation order."""

    ordered = sorted(entries, key=lambda item: item.get("paid_attempt_ordinal", -1))
    if [item.get("paid_attempt_ordinal") for item in ordered] != list(range(1, len(ordered) + 1)):
        raise Tier0Error("paid attempt ordinals are not unique and contiguous", status=status)
    previous: Mapping[str, Any] | None = None
    for item in ordered:
        expected_previous_id = None if previous is None else previous.get("attempt_id")
        expected_previous_hash = None if previous is None else previous.get("header_record_hash")
        if (
            item.get("previous_paid_attempt_id") != expected_previous_id
            or item.get("previous_paid_attempt_header_record_hash") != expected_previous_hash
        ):
            raise Tier0Error("paid attempt lineage is broken", status=status)
        previous = item
    return ordered


def validate_paid_attempt_anchor(
    job_root: Path,
    *,
    volume: VolumeIdentity,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    allow_initialize: bool = False,
    repair_header_only: bool = False,
) -> dict[str, Any]:
    """Detect disappearance of a whole paid attempt before budget reconstruction."""

    job_root = Path(job_root)
    anchor_path = job_root / PAID_ATTEMPT_ANCHOR_NAME
    attempts_root = job_root / "attempts"
    actual_entries: dict[str, dict[str, Any]] = {}
    for attempt_dir in sorted(attempts_root.iterdir()):
        if attempt_dir.name == EXPECTED_LEGACY_ATTEMPT_ID:
            continue
        if attempt_dir.is_symlink() or not attempt_dir.is_dir() or base.UUID4_RE.fullmatch(attempt_dir.name) is None:
            raise Tier0Error("paid attempt set contains an unsafe entry", status="STOP_PAID_ATTEMPT_SET")
        if attempt_dir.stat().st_dev != volume.st_dev:
            raise Tier0Error("paid attempt set contains a cross-device entry", status="STOP_PAID_ATTEMPT_SET")
        verified = base.verify_attempt_journal(attempt_dir)
        records = verified["records"]
        header = _validate_paid_header(
            records[0],
            attempt_dir=attempt_dir,
            expected_readiness_sha256=readiness_receipt_sha256,
            expected_readiness_file_sha256=readiness_receipt_file_sha256,
        )
        actual_entries[attempt_dir.name] = {
            "attempt_id": attempt_dir.name,
            "header_record_hash": records[0]["record_hash"],
            "paid_readiness_receipt_sha256": header["paid_readiness_receipt_sha256"],
            "paid_attempt_ordinal": header["paid_attempt_ordinal"],
            "previous_paid_attempt_id": header["previous_paid_attempt_id"],
            "previous_paid_attempt_header_record_hash": header[
                "previous_paid_attempt_header_record_hash"
            ],
        }
    _validate_paid_attempt_lineage(list(actual_entries.values()))

    if not anchor_path.exists() and not anchor_path.is_symlink():
        if not allow_initialize:
            raise Tier0Error("paid attempt-set anchor is absent", status="STOP_PAID_ATTEMPT_SET")
        non_header_only = []
        for attempt_id in actual_entries:
            records = base.verify_attempt_journal(attempts_root / attempt_id)["records"]
            if len(records) != 1:
                non_header_only.append(attempt_id)
        if non_header_only:
            raise Tier0Error("paid anchor vanished after a paid attempt advanced", status="STOP_PAID_ATTEMPT_SET")
        initial = _anchor_receipt(
            readiness_receipt_sha256=readiness_receipt_sha256,
            readiness_receipt_file_sha256=readiness_receipt_file_sha256,
            paid_attempts=list(actual_entries.values()),
        )
        base.write_canonical_exclusive(anchor_path, initial)

    _private_regular(anchor_path, status="STOP_PAID_ATTEMPT_SET")
    if anchor_path.stat().st_dev != volume.st_dev:
        raise Tier0Error("paid attempt-set anchor is cross-device", status="STOP_PAID_ATTEMPT_SET")
    anchor = base.strict_json(anchor_path)
    if anchor.get("anchor_sha256") != base.self_hash(anchor, "anchor_sha256"):
        raise Tier0Error("paid attempt-set anchor self-hash mismatch", status="STOP_PAID_ATTEMPT_SET")
    expected_base = _anchor_receipt(
        readiness_receipt_sha256=readiness_receipt_sha256,
        readiness_receipt_file_sha256=readiness_receipt_file_sha256,
        paid_attempts=anchor.get("paid_attempts", []) if isinstance(anchor.get("paid_attempts"), list) else [],
    )
    if anchor != expected_base:
        raise Tier0Error("paid attempt-set anchor fields drifted", status="STOP_PAID_ATTEMPT_SET")
    anchored_entries = {
        str(item.get("attempt_id")): item
        for item in anchor["paid_attempts"]
        if isinstance(item, dict)
    }
    if len(anchored_entries) != len(anchor["paid_attempts"]):
        raise Tier0Error("paid attempt-set anchor contains duplicate/malformed entries", status="STOP_PAID_ATTEMPT_SET")
    _validate_paid_attempt_lineage(list(anchored_entries.values()))
    missing = set(anchored_entries) - set(actual_entries)
    extra = set(actual_entries) - set(anchored_entries)
    if missing:
        raise Tier0Error("a paid attempt directory disappeared", status="STOP_PAID_ATTEMPT_SET")
    if extra:
        if not repair_header_only:
            raise Tier0Error("an unanchored paid attempt exists", status="STOP_PAID_ATTEMPT_SET")
        for attempt_id in extra:
            records = base.verify_attempt_journal(attempts_root / attempt_id)["records"]
            if len(records) != 1:
                raise Tier0Error("an advanced paid attempt is absent from the anchor", status="STOP_PAID_ATTEMPT_SET")
        repaired = _anchor_receipt(
            readiness_receipt_sha256=readiness_receipt_sha256,
            readiness_receipt_file_sha256=readiness_receipt_file_sha256,
            paid_attempts=list(actual_entries.values()),
        )
        base.write_canonical_replace(anchor_path, repaired)
        anchor = repaired
        anchored_entries = {item["attempt_id"]: item for item in anchor["paid_attempts"]}
    if anchored_entries != actual_entries:
        raise Tier0Error("paid attempt-set anchor/header binding drifted", status="STOP_PAID_ATTEMPT_SET")
    return anchor


_PAID_EVENTS = frozenset(
    {
        "ATTEMPT_START",
        "CLIENT_CONSTRUCTED",
        "COST_CALL_START",
        "COST_CALL_RESULT",
        "COST_CALL_ERROR",
        "TIMESERIES_CALL_START",
        "TIMESERIES_CALL_RESULT",
        "TIMESERIES_CALL_ERROR",
        "SESSION_PUBLISHED",
        "SESSION_REUSED",
        "SESSION_RECOVERED",
        "ATTEMPT_SEALED_FOR_AGGREGATE",
        "ATTEMPT_STOP",
    }
)


def _validate_marker_semantics(attempt_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Close the payload/timestamp gap intentionally left by the V1 generic verifier."""

    marker_dir = Path(attempt_dir) / "call-markers"
    expected_fields = {
        "artifact_type",
        "schema_version",
        "attempt_id",
        "sequence",
        "event",
        "session",
        "request_sha256",
        "journal_record_hash",
        "recorded_at_utc",
        "payload",
        "marker_sha256",
    }
    for path in sorted(marker_dir.iterdir()):
        marker = base.strict_json(path)
        sequence = marker.get("sequence")
        if set(marker) != expected_fields or isinstance(sequence, bool) or not isinstance(sequence, int):
            raise Tier0Error("paid call marker field population drifted", status="STOP_PAID_JOURNAL_INVALID")
        if not (0 <= sequence < len(records)):
            raise Tier0Error("paid call marker sequence is invalid", status="STOP_PAID_JOURNAL_INVALID")
        source = records[sequence]
        exact_projection = {
            "artifact_type": base.CALL_MARKER_ARTIFACT,
            "schema_version": "v5.job51-vendor-call-marker.v1",
            "attempt_id": source["attempt_id"],
            "sequence": source["sequence"],
            "event": source["event"],
            "session": source["session"],
            "request_sha256": source["request_sha256"],
            "journal_record_hash": source["record_hash"],
            "recorded_at_utc": source["recorded_at_utc"],
            "payload": source["payload"],
        }
        if {key: marker[key] for key in exact_projection} != exact_projection:
            raise Tier0Error("paid call marker does not exactly mirror its journal record", status="STOP_PAID_JOURNAL_INVALID")


def _validate_paid_attempt_tree(
    attempt_dir: Path,
    *,
    allowed_sessions: set[str],
    legacy: bool,
) -> dict[str, list[str]]:
    children = sorted(path.name for path in attempt_dir.iterdir())
    expected_core = {
        "ACQUISITION_JOURNAL_V1.jsonl",
        "JOURNAL_WATERMARK_V1.json",
        "call-markers",
        "sessions",
    }
    allowed = set(expected_core)
    allowed.add("ATTEMPT_STOP_V1.json" if legacy else PAID_ATTEMPT_STOP_NAME)
    if not expected_core <= set(children) or any(name not in allowed for name in children):
        raise Tier0Error("attempt directory contains an unexpected child", status="STOP_PAID_JOURNAL_INVALID")
    sessions_dir = attempt_dir / "sessions"
    sessions_metadata = sessions_dir.lstat()
    if (
        stat.S_ISLNK(sessions_metadata.st_mode)
        or not stat.S_ISDIR(sessions_metadata.st_mode)
        or sessions_metadata.st_dev != attempt_dir.stat().st_dev
    ):
        raise Tier0Error("attempt staging directory is unsafe", status="STOP_PAID_JOURNAL_INVALID")
    staging: dict[str, list[str]] = {}
    for child in sorted(sessions_dir.iterdir()):
        suffix = ".bundle.part"
        child_metadata = child.lstat()
        if (
            stat.S_ISLNK(child_metadata.st_mode)
            or not stat.S_ISDIR(child_metadata.st_mode)
            or child_metadata.st_dev != attempt_dir.stat().st_dev
            or not child.name.endswith(suffix)
        ):
            raise Tier0Error("attempt staging contains an unsafe child", status="STOP_PAID_JOURNAL_INVALID")
        session = child.name[: -len(suffix)]
        if session not in allowed_sessions:
            raise Tier0Error("attempt staging contains an out-of-scope session", status="STOP_PAID_SCOPE_WIDENING")
        names = sorted(path.name for path in child.iterdir())
        if legacy:
            if session != "2025-08-22" or names:
                raise Tier0Error("legacy Job-51 staging population drifted", status="STOP_PAID_LEGACY_DRIFT")
            staging[session] = names
            continue
        if any(name not in {"data.cmbp-1.dbn.zst", PAID_SESSION_QC_NAME} for name in names):
            raise Tier0Error("attempt staging bundle contains an unexpected file", status="STOP_PAID_JOURNAL_INVALID")
        for item in child.iterdir():
            metadata = item.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_dev != attempt_dir.stat().st_dev
            ):
                raise Tier0Error("attempt staging bundle contains an unsafe file", status="STOP_PAID_JOURNAL_INVALID")
        staging[session] = names
    return staging


def _validate_legacy_attempt(job_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    attempt_dir = Path(job_root) / "attempts" / EXPECTED_LEGACY_ATTEMPT_ID
    if not attempt_dir.is_dir() or attempt_dir.is_symlink():
        raise Tier0Error("exact Job-51 legacy attempt is absent", status="STOP_PAID_LEGACY_DRIFT")
    legacy_staging = _validate_paid_attempt_tree(
        attempt_dir, allowed_sessions={"2025-08-22"}, legacy=True
    )
    verified = base.verify_attempt_journal(attempt_dir)
    records = verified["records"]
    _validate_marker_semantics(attempt_dir, records)
    expected = contract["legacy_stop_evidence"]
    stop_path = attempt_dir / "ATTEMPT_STOP_V1.json"
    stop = base.strict_json(stop_path)
    if (
        legacy_staging != {"2025-08-22": []}
        or verified["journal_file_sha256"] != expected["journal_file_sha256"]
        or verified["watermark_file_sha256"] != expected["watermark_file_sha256"]
        or verified["marker_files"] != expected["marker_file_sha256"]
        or base.file_sha256(stop_path) != expected["attempt_stop_file_sha256"]
        or stop.get("stop_sha256") != expected["attempt_stop_sha256"]
        or base.self_hash(stop, "stop_sha256") != expected["attempt_stop_sha256"]
        or len(records) != 5
        or records[3].get("event") != "COST_CALL_RESULT"
        or records[3].get("record_hash") != expected["cost_result_record_hash"]
        or records[3].get("payload") != {"observed_sdk_quote": "0.950392448902", "zero_gate_pass": False}
        or records[4].get("event") != "ATTEMPT_STOP"
        or records[4].get("record_hash") != expected["journal_terminal_head"]
        or any(record.get("event") == "TIMESERIES_CALL_START" for record in records)
    ):
        raise Tier0Error("Job-51 legacy attempt bytes/semantics drifted", status="STOP_PAID_LEGACY_DRIFT")
    return {
        "attempt_id": EXPECTED_LEGACY_ATTEMPT_ID,
        "observed_sdk_quote_usd": "0.950392448902",
        "committed_quote_usd": "0",
        "cost_call_starts": 1,
        "timeseries_call_starts": 0,
        "journal_file_sha256": verified["journal_file_sha256"],
        "watermark_file_sha256": verified["watermark_file_sha256"],
        "marker_files": verified["marker_files"],
        "attempt_stop_file_sha256": base.file_sha256(stop_path),
    }


def _validate_paid_header(
    header: Mapping[str, Any],
    *,
    attempt_dir: Path,
    expected_readiness_sha256: str | None,
    expected_readiness_file_sha256: str | None,
) -> Mapping[str, Any]:
    payload = header.get("payload")
    expected_keys = {
        "scope_sha256",
        "scope_file_sha256",
        "job51_program_contract_sha256",
        "paid_program_contract_sha256",
        "paid_program_contract_file_sha256",
        "paid_readiness_receipt_sha256",
        "paid_readiness_receipt_file_sha256",
        "legacy_stop_receipt_sha256",
        "legacy_stop_receipt_file_sha256",
        "volume_identity",
        "authority",
        "actual_vendor_invoice_cost_usd",
        "paid_attempt_ordinal",
        "previous_paid_attempt_id",
        "previous_paid_attempt_header_record_hash",
    }
    if header.get("event") != "ATTEMPT_START" or header.get("sequence") != 0 or not isinstance(payload, dict):
        raise Tier0Error("paid attempt header is absent", status="STOP_PAID_JOURNAL_INVALID")
    if set(payload) != expected_keys:
        raise Tier0Error("paid attempt header field population drifted", status="STOP_PAID_JOURNAL_INVALID")
    expected_scalars = {
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "job51_program_contract_sha256": base.EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "legacy_stop_receipt_sha256": EXPECTED_LEGACY_STOP_RECEIPT_SHA256,
        "legacy_stop_receipt_file_sha256": EXPECTED_LEGACY_STOP_RECEIPT_FILE_SHA256,
        "authority": PAID_AUTHORITY,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
    }
    for field_name, value in expected_scalars.items():
        if payload.get(field_name) != value:
            raise Tier0Error(f"paid attempt header {field_name} drifted", status="STOP_PAID_JOURNAL_INVALID")
    paid_attempt_ordinal = payload.get("paid_attempt_ordinal")
    previous_attempt_id = payload.get("previous_paid_attempt_id")
    previous_header_hash = payload.get("previous_paid_attempt_header_record_hash")
    if (
        isinstance(paid_attempt_ordinal, bool)
        or not isinstance(paid_attempt_ordinal, int)
        or paid_attempt_ordinal < 1
    ):
        raise Tier0Error("paid attempt ordinal is invalid", status="STOP_PAID_JOURNAL_INVALID")
    if paid_attempt_ordinal == 1:
        if previous_attempt_id is not None or previous_header_hash is not None:
            raise Tier0Error("first paid attempt has a predecessor", status="STOP_PAID_JOURNAL_INVALID")
    elif (
        base.UUID4_RE.fullmatch(str(previous_attempt_id)) is None
        or SHA256_RE.fullmatch(str(previous_header_hash)) is None
    ):
        raise Tier0Error("paid attempt predecessor identity is invalid", status="STOP_PAID_JOURNAL_INVALID")
    for field_name, expected_value in (
        ("paid_readiness_receipt_sha256", expected_readiness_sha256),
        ("paid_readiness_receipt_file_sha256", expected_readiness_file_sha256),
    ):
        actual = payload.get(field_name)
        if SHA256_RE.fullmatch(str(actual)) is None or (expected_value is not None and actual != expected_value):
            raise Tier0Error(f"paid attempt header {field_name} drifted", status="STOP_PAID_JOURNAL_INVALID")
    volume = payload.get("volume_identity")
    volume_fields = {
        "mount_point",
        "volume_uuid",
        "device_identifier",
        "filesystem",
        "bus_protocol",
        "st_dev",
        "free_bytes",
        "total_bytes",
    }
    if not isinstance(volume, dict) or set(volume) != volume_fields:
        raise Tier0Error("paid attempt volume identity is malformed", status="STOP_PAID_JOURNAL_INVALID")
    if (
        not isinstance(volume.get("mount_point"), str)
        or not volume.get("mount_point")
        or volume.get("volume_uuid") != base.EXPECTED_VOLUME_UUID
        or volume.get("filesystem") != base.EXPECTED_FILESYSTEM
        or not isinstance(volume.get("device_identifier"), str)
        or not volume.get("device_identifier")
        or not isinstance(volume.get("bus_protocol"), str)
        or not volume.get("bus_protocol")
        or isinstance(volume.get("st_dev"), bool)
        or not isinstance(volume.get("st_dev"), int)
        or volume.get("st_dev") != attempt_dir.stat().st_dev
        or isinstance(volume.get("free_bytes"), bool)
        or not isinstance(volume.get("free_bytes"), int)
        or volume.get("free_bytes") < 0
        or isinstance(volume.get("total_bytes"), bool)
        or not isinstance(volume.get("total_bytes"), int)
        or volume.get("total_bytes") < volume.get("free_bytes")
    ):
        raise Tier0Error("paid attempt volume identity drifted", status="STOP_PAID_JOURNAL_INVALID")
    return payload


def build_paid_attempt_header_payload(
    *,
    volume: VolumeIdentity,
    readiness_receipt_sha256: str,
    readiness_receipt_file_sha256: str,
    paid_attempt_ordinal: int,
    previous_paid_attempt_id: str | None,
    previous_paid_attempt_header_record_hash: str | None,
) -> dict[str, Any]:
    """Build the exact Job-52 header projection later revalidated from disk."""

    if (
        SHA256_RE.fullmatch(readiness_receipt_sha256) is None
        or SHA256_RE.fullmatch(readiness_receipt_file_sha256) is None
        or isinstance(paid_attempt_ordinal, bool)
        or not isinstance(paid_attempt_ordinal, int)
        or paid_attempt_ordinal < 1
    ):
        raise Tier0Error("paid attempt header inputs are invalid", status="STOP_PAID_JOURNAL_INVALID")
    if paid_attempt_ordinal == 1:
        if previous_paid_attempt_id is not None or previous_paid_attempt_header_record_hash is not None:
            raise Tier0Error("first paid attempt cannot have a predecessor", status="STOP_PAID_JOURNAL_INVALID")
    elif (
        base.UUID4_RE.fullmatch(str(previous_paid_attempt_id)) is None
        or SHA256_RE.fullmatch(str(previous_paid_attempt_header_record_hash)) is None
    ):
        raise Tier0Error("paid attempt predecessor inputs are invalid", status="STOP_PAID_JOURNAL_INVALID")
    return {
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "job51_program_contract_sha256": base.EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "paid_readiness_receipt_sha256": readiness_receipt_sha256,
        "paid_readiness_receipt_file_sha256": readiness_receipt_file_sha256,
        "legacy_stop_receipt_sha256": EXPECTED_LEGACY_STOP_RECEIPT_SHA256,
        "legacy_stop_receipt_file_sha256": EXPECTED_LEGACY_STOP_RECEIPT_FILE_SHA256,
        "volume_identity": asdict(volume),
        "authority": PAID_AUTHORITY,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "paid_attempt_ordinal": paid_attempt_ordinal,
        "previous_paid_attempt_id": previous_paid_attempt_id,
        "previous_paid_attempt_header_record_hash": previous_paid_attempt_header_record_hash,
    }


def _numeric_cost_payload(state: PaidBudgetState, session: str, quote_text: str) -> dict[str, Any]:
    quote = Decimal(quote_text)
    projection = state.projection(session, quote)
    return {
        "observed_sdk_quote_usd": quote_text,
        "quote_valid": True,
        "commitment_count_before": state.commitment_count,
        "committed_quote_session_before_usd": format(
            state.committed_by_session_usd.get(session, Decimal("0")), "f"
        ),
        "committed_quote_session_projected_usd": format(
            projection["committed_quote_session_after_usd"], "f"
        ),
        "committed_quote_total_before_usd": format(state.committed_total_usd, "f"),
        "committed_quote_total_projected_usd": format(
            projection["committed_quote_total_after_usd"], "f"
        ),
        "per_session_lifetime_cap_usd": "1.50",
        "total_cap_usd": "32.00",
        "per_session_lifetime_cap_pass": projection["per_session_lifetime_cap_pass"],
        "total_cap_pass": projection["total_cap_pass"],
        "time_series_start_permitted": bool(
            projection["per_session_lifetime_cap_pass"] and projection["total_cap_pass"]
        ),
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
    }


def _validate_paid_stop(attempt_dir: Path, records: Sequence[Mapping[str, Any]], header: Mapping[str, Any]) -> str | None:
    path = Path(attempt_dir) / PAID_ATTEMPT_STOP_NAME
    terminal = records[-1]
    if terminal.get("event") != "ATTEMPT_STOP":
        if path.exists() or path.is_symlink():
            raise Tier0Error("paid stop receipt exists without terminal stop", status="STOP_PAID_JOURNAL_INVALID")
        return None
    _private_regular(path, status="STOP_PAID_JOURNAL_INVALID")
    receipt = base.strict_json(path)
    expected_fields = {
        "artifact_type",
        "schema_version",
        "job_id",
        "target_job_id",
        "attempt_id",
        "scope_sha256",
        "paid_program_contract_sha256",
        "paid_readiness_receipt_sha256",
        "status",
        "error_class",
        "actual_vendor_invoice_cost_usd",
        "journal_terminal_sequence",
        "journal_terminal_head",
        "stop_sha256",
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("artifact_type") != ATTEMPT_STOP_ARTIFACT
        or receipt.get("schema_version") != "v5.job52-cmbp-tier0-paid-attempt-stop.v1"
        or receipt.get("job_id") != JOB_ID
        or receipt.get("target_job_id") != TARGET_JOB_ID
        or receipt.get("attempt_id") != attempt_dir.name
        or receipt.get("scope_sha256") != base.EXPECTED_SCOPE_SHA256
        or receipt.get("paid_program_contract_sha256") != EXPECTED_PROGRAM_CONTRACT_SHA256
        or receipt.get("paid_readiness_receipt_sha256") != header["paid_readiness_receipt_sha256"]
        or receipt.get("status") != terminal.get("payload", {}).get("status")
        or receipt.get("error_class") != terminal.get("payload", {}).get("error_class")
        or receipt.get("actual_vendor_invoice_cost_usd") != "UNKNOWN"
        or receipt.get("journal_terminal_sequence") != terminal.get("sequence")
        or receipt.get("journal_terminal_head") != terminal.get("record_hash")
        or receipt.get("stop_sha256") != base.self_hash(receipt, "stop_sha256")
    ):
        raise Tier0Error("paid stop receipt does not bind its terminal journal", status="STOP_PAID_JOURNAL_INVALID")
    return base.file_sha256(path)


def summarize_paid_attempts(
    job_root: Path,
    allowed_requests: Sequence[SessionRequest],
    *,
    require_legacy_stop: bool = False,
    expected_readiness_sha256: str | None = None,
    expected_readiness_file_sha256: str | None = None,
    require_client_constructed: bool = False,
    expected_sdk_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Independently rebuild every quote commitment from durable start records."""

    job_root = Path(job_root)
    attempts_root = job_root / "attempts"
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise Tier0Error("paid attempts root is absent or unsafe", status="STOP_PAID_JOURNAL_INVALID")
    job_root_dev = job_root.stat().st_dev
    if attempts_root.stat().st_dev != job_root_dev:
        raise Tier0Error("paid attempts root is cross-device", status="STOP_PAID_JOURNAL_INVALID")
    if require_client_constructed and SHA256_RE.fullmatch(str(expected_sdk_identity_sha256)) is None:
        raise Tier0Error("expected paid SDK identity is absent", status="STOP_PAID_JOURNAL_INVALID")
    allowed = {request.session: request for request in allowed_requests}
    if len(allowed) != len(allowed_requests):
        raise Tier0Error("allowed paid request population has duplicate sessions", status="STOP_PAID_SCOPE_WIDENING")
    ordinal_by_session = {request.session: index for index, request in enumerate(allowed_requests, start=1)}
    attempt_children = sorted(attempts_root.iterdir())
    if any(path.is_symlink() or not path.is_dir() or base.UUID4_RE.fullmatch(path.name) is None for path in attempt_children):
        raise Tier0Error("paid attempts root contains an unexpected entry", status="STOP_PAID_JOURNAL_INVALID")

    repo_root = Path(__file__).resolve().parents[2]
    contract = load_paid_scope_bundle(repo_root).contract
    legacy_summary: dict[str, Any] | None = None
    paid_verified: list[dict[str, Any]] = []
    for attempt_dir in attempt_children:
        if attempt_dir.stat().st_dev != job_root_dev:
            raise Tier0Error("paid attempt directory is cross-device", status="STOP_PAID_JOURNAL_INVALID")
        if attempt_dir.name == EXPECTED_LEGACY_ATTEMPT_ID:
            legacy_summary = _validate_legacy_attempt(job_root, contract)
            continue
        staging = _validate_paid_attempt_tree(attempt_dir, allowed_sessions=set(allowed), legacy=False)
        verified = base.verify_attempt_journal(attempt_dir)
        records = verified["records"]
        _validate_marker_semantics(attempt_dir, records)
        header = _validate_paid_header(
            records[0],
            attempt_dir=attempt_dir,
            expected_readiness_sha256=expected_readiness_sha256,
            expected_readiness_file_sha256=expected_readiness_file_sha256,
        )
        paid_verified.append(
            {
                "attempt_dir": attempt_dir,
                "verified": verified,
                "records": records,
                "header": header,
                "staging": staging,
            }
        )
    if require_legacy_stop and legacy_summary is None:
        raise Tier0Error("frozen Job-51 legacy attempt population is absent", status="STOP_PAID_LEGACY_DRIFT")

    lineage_entries = [
        {
            "attempt_id": item["attempt_dir"].name,
            "header_record_hash": item["records"][0]["record_hash"],
            "paid_attempt_ordinal": item["header"]["paid_attempt_ordinal"],
            "previous_paid_attempt_id": item["header"]["previous_paid_attempt_id"],
            "previous_paid_attempt_header_record_hash": item["header"][
                "previous_paid_attempt_header_record_hash"
            ],
        }
        for item in paid_verified
    ]
    ordered_lineage = _validate_paid_attempt_lineage(
        lineage_entries,
        status="STOP_PAID_JOURNAL_INVALID",
    )
    paid_by_id = {item["attempt_dir"].name: item for item in paid_verified}
    paid_verified = [paid_by_id[str(item["attempt_id"])] for item in ordered_lineage]

    cost_starts = cost_results = cost_errors = timeseries_starts = timeseries_results = 0
    failed_timeseries = pending_timeseries = pending_cost_calls = uncommitted_passing = 0
    start_pairs: list[dict[str, Any]] = []
    result_pairs: list[dict[str, Any]] = []
    cost_observations: list[dict[str, Any]] = []
    attempt_summaries: list[dict[str, Any]] = []
    session_attempt_counts: dict[str, int] = {}
    terminal_failures: list[dict[str, Any]] = []
    published_sessions: set[str] = set()
    completed_session_events: set[str] = set()
    publication_records: list[dict[str, Any]] = []
    aggregate_seal_records: list[dict[str, Any]] = []
    global_recovery_barrier: Mapping[str, Any] | None = None

    for item in paid_verified:
        records = item["records"]
        attempt_dir = item["attempt_dir"]
        header = item["header"]
        pending_cost: Mapping[str, Any] | None = None
        pending_start: Mapping[str, Any] | None = None
        awaiting_publication: Mapping[str, Any] | None = None
        last_cost_result: Mapping[str, Any] | None = None
        paired_cost_sequences: set[int] = set()
        client_seen = False
        call_error_seen = False
        per_attempt = {"cost_starts": 0, "cost_results": 0, "timeseries_starts": 0, "timeseries_results": 0}
        for index, record in enumerate(records):
            event = record.get("event")
            if event not in _PAID_EVENTS:
                raise Tier0Error("paid attempt contains an unknown event", status="STOP_PAID_JOURNAL_INVALID")
            if index == 0:
                continue
            if global_recovery_barrier is not None and event != "ATTEMPT_STOP":
                barrier_payload = global_recovery_barrier.get("payload", {})
                record_payload = record.get("payload", {})
                exact_recovery = (
                    event == "SESSION_RECOVERED"
                    and record.get("session") == global_recovery_barrier.get("session")
                    and record.get("request_sha256") == global_recovery_barrier.get("request_sha256")
                    and record_payload.get("source_attempt_id") == global_recovery_barrier.get("attempt_id")
                    and record_payload.get("source_timeseries_result_sequence")
                    == global_recovery_barrier.get("sequence")
                    and record_payload.get("source_timeseries_result_record_hash")
                    == global_recovery_barrier.get("record_hash")
                    and record_payload.get("compressed_bytes") == barrier_payload.get("compressed_bytes")
                )
                if not exact_recovery:
                    raise Tier0Error(
                        "paid execution continued before locally recovering a successful stream",
                        status="STOP_PAID_JOURNAL_INVALID",
                    )
            if event == "ATTEMPT_START":
                raise Tier0Error("paid attempt contains a duplicate header", status="STOP_PAID_JOURNAL_INVALID")
            if event == "ATTEMPT_STOP" and index != len(records) - 1:
                raise Tier0Error("paid attempt stop is not terminal", status="STOP_PAID_JOURNAL_INVALID")
            if event == "ATTEMPT_SEALED_FOR_AGGREGATE" and index != len(records) - 1:
                if index != len(records) - 2 or records[-1].get("event") != "ATTEMPT_STOP":
                    raise Tier0Error("paid aggregate seal has an invalid successor", status="STOP_PAID_JOURNAL_INVALID")
            if call_error_seen and event != "ATTEMPT_STOP":
                raise Tier0Error(
                    "paid attempt continued after a vendor-call error",
                    status="STOP_PAID_JOURNAL_INVALID",
                )
            if awaiting_publication is not None and event not in {"SESSION_PUBLISHED", "ATTEMPT_STOP"}:
                raise Tier0Error(
                    "paid attempt began another action before publishing its completed session",
                    status="STOP_PAID_JOURNAL_INVALID",
                )
            if event == "CLIENT_CONSTRUCTED":
                if client_seen or any(count for count in per_attempt.values()):
                    raise Tier0Error("paid client construction event is misplaced", status="STOP_PAID_JOURNAL_INVALID")
                payload = record.get("payload")
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"credential_source", "sdk_identity_sha256"}
                    or payload.get("credential_source") not in {"process_environment", "repository_root_dotenv"}
                    or SHA256_RE.fullmatch(str(payload.get("sdk_identity_sha256"))) is None
                    or (
                        expected_sdk_identity_sha256 is not None
                        and payload.get("sdk_identity_sha256") != expected_sdk_identity_sha256
                    )
                ):
                    raise Tier0Error("paid client construction payload drifted", status="STOP_PAID_JOURNAL_INVALID")
                client_seen = True
                continue
            if event in base.CALL_EVENTS:
                if require_client_constructed and not client_seen:
                    raise Tier0Error("paid vendor call precedes client construction", status="STOP_PAID_JOURNAL_INVALID")
                request = allowed.get(str(record.get("session")))
                if request is None or record.get("request_sha256") != request.market_request_sha256:
                    raise Tier0Error("paid attempt contains an out-of-scope call", status="STOP_PAID_SCOPE_WIDENING")
                if request.session in completed_session_events:
                    raise Tier0Error(
                        "paid vendor call follows durable session completion",
                        status="STOP_PAID_JOURNAL_INVALID",
                    )
            if event == "COST_CALL_START":
                if pending_cost is not None or pending_start is not None or last_cost_result is not None:
                    raise Tier0Error("paid cost call overlaps prior call state", status="STOP_PAID_JOURNAL_INVALID")
                request = allowed[str(record["session"])]
                if record.get("payload") != {"method": "metadata.get_cost", "parameters": request.market_parameters}:
                    raise Tier0Error("paid cost request payload widened", status="STOP_PAID_SCOPE_WIDENING")
                pending_cost = record
                cost_starts += 1
                per_attempt["cost_starts"] += 1
            elif event == "COST_CALL_RESULT":
                if (
                    pending_cost is None
                    or pending_cost.get("session") != record.get("session")
                    or pending_cost.get("request_sha256") != record.get("request_sha256")
                ):
                    raise Tier0Error("paid cost result lacks a matching start", status="STOP_PAID_JOURNAL_INVALID")
                pending_cost = None
                last_cost_result = record
                cost_results += 1
                per_attempt["cost_results"] += 1
                cost_observations.append({"record": record, "attempt": item})
            elif event == "COST_CALL_ERROR":
                if (
                    pending_cost is None
                    or pending_cost.get("session") != record.get("session")
                    or pending_cost.get("request_sha256") != record.get("request_sha256")
                    or record.get("payload")
                    != {
                        "error_class": record.get("payload", {}).get("error_class"),
                        "actual_vendor_invoice_cost_usd": "UNKNOWN",
                    }
                    or not isinstance(record.get("payload", {}).get("error_class"), str)
                    or not record.get("payload", {}).get("error_class")
                ):
                    raise Tier0Error("paid cost error payload is invalid", status="STOP_PAID_JOURNAL_INVALID")
                pending_cost = None
                cost_errors += 1
                call_error_seen = True
            elif event == "TIMESERIES_CALL_START":
                if (
                    last_cost_result is None
                    or last_cost_result.get("sequence") != record.get("sequence", -2) - 1
                    or last_cost_result.get("session") != record.get("session")
                    or last_cost_result.get("request_sha256") != record.get("request_sha256")
                    or pending_start is not None
                ):
                    raise Tier0Error("paid time-series start lacks an adjacent quote", status="STOP_PAID_JOURNAL_INVALID")
                start_pairs.append({"cost": last_cost_result, "start": record, "attempt": item})
                paired_cost_sequences.add(int(last_cost_result["sequence"]))
                last_cost_result = None
                pending_start = record
                timeseries_starts += 1
                per_attempt["timeseries_starts"] += 1
                session = str(record["session"])
                session_attempt_counts[session] = session_attempt_counts.get(session, 0) + 1
            elif event in {"TIMESERIES_CALL_RESULT", "TIMESERIES_CALL_ERROR"}:
                if (
                    pending_start is None
                    or pending_start.get("session") != record.get("session")
                    or pending_start.get("request_sha256") != record.get("request_sha256")
                ):
                    raise Tier0Error("paid time-series terminal lacks a matching start", status="STOP_PAID_JOURNAL_INVALID")
                result_pairs.append({"start": pending_start, "terminal": record, "attempt": item})
                if event == "TIMESERIES_CALL_RESULT":
                    timeseries_results += 1
                    per_attempt["timeseries_results"] += 1
                    awaiting_publication = record
                else:
                    failed_timeseries += 1
                    call_error_seen = True
                pending_start = None
            elif event in {"SESSION_PUBLISHED", "SESSION_REUSED", "SESSION_RECOVERED"}:
                if pending_cost is not None or pending_start is not None or last_cost_result is not None:
                    raise Tier0Error("paid publication occurred with open call state", status="STOP_PAID_JOURNAL_INVALID")
                request = allowed.get(str(record.get("session")))
                if request is None or record.get("request_sha256") != request.market_request_sha256:
                    raise Tier0Error("paid publication cites an out-of-scope session", status="STOP_PAID_SCOPE_WIDENING")
                if event == "SESSION_PUBLISHED":
                    payload = record.get("payload", {})
                    if (
                        awaiting_publication is None
                        or payload.get("source_attempt_id") != awaiting_publication.get("attempt_id")
                        or payload.get("source_timeseries_result_sequence") != awaiting_publication.get("sequence")
                        or payload.get("source_timeseries_result_record_hash")
                        != awaiting_publication.get("record_hash")
                    ):
                        raise Tier0Error(
                            "paid publication does not immediately discharge its completed stream",
                            status="STOP_PAID_JOURNAL_INVALID",
                        )
                    awaiting_publication = None
                elif awaiting_publication is not None:
                    raise Tier0Error(
                        "paid completed stream used the wrong publication mode",
                        status="STOP_PAID_JOURNAL_INVALID",
                    )
                if event == "SESSION_RECOVERED" and global_recovery_barrier is not None:
                    global_recovery_barrier = None
                publication_records.append({"record": record, "request": request, "attempt": item})
                published_sessions.add(request.session)
                completed_session_events.add(request.session)
            elif event == "ATTEMPT_SEALED_FOR_AGGREGATE":
                if pending_cost is not None or pending_start is not None or last_cost_result is not None:
                    raise Tier0Error("paid aggregate seal occurred with open call state", status="STOP_PAID_JOURNAL_INVALID")
                aggregate_seal_records.append({"record": record, "attempt": item})
            elif event == "ATTEMPT_STOP":
                payload = record.get("payload")
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"status", "error_class"}
                    or not isinstance(payload.get("status"), str)
                    or not payload["status"]
                    or not isinstance(payload.get("error_class"), str)
                    or not payload["error_class"]
                ):
                    raise Tier0Error("paid attempt stop payload drifted", status="STOP_PAID_JOURNAL_INVALID")
            elif event not in {"CLIENT_CONSTRUCTED"}:
                raise Tier0Error("paid journal event state is invalid", status="STOP_PAID_JOURNAL_INVALID")
        if pending_start is not None:
            pending_timeseries += 1
        if pending_cost is not None:
            pending_cost_calls += 1
        if last_cost_result is not None:
            payload = last_cost_result.get("payload", {})
            if payload.get("quote_valid") is True and payload.get("time_series_start_permitted") is True:
                uncommitted_passing += 1
        if awaiting_publication is not None:
            if global_recovery_barrier is not None:
                raise Tier0Error(
                    "multiple successful streams await local recovery",
                    status="STOP_PAID_JOURNAL_INVALID",
                )
            global_recovery_barrier = awaiting_publication
        stop_file_sha = _validate_paid_stop(attempt_dir, records, header)
        attempt_summaries.append(
            {
                "attempt_id": attempt_dir.name,
                "paid_attempt_ordinal": header["paid_attempt_ordinal"],
                **per_attempt,
                "terminal_event": records[-1].get("event"),
                "journal_file_sha256": item["verified"]["journal_file_sha256"],
                "watermark_file_sha256": item["verified"]["watermark_file_sha256"],
                "marker_files": item["verified"]["marker_files"],
                "attempt_stop_file_sha256": stop_file_sha,
            }
        )

    # Commitment index, not UUID or wall-clock order, is the sole global order.
    indexed: dict[int, dict[str, Any]] = {}
    for pair in start_pairs:
        index = pair["start"].get("payload", {}).get("commitment_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 1 or index in indexed:
            raise Tier0Error("paid commitment index is absent, duplicate, or invalid", status="STOP_PAID_JOURNAL_INVALID")
        indexed[index] = pair
    if sorted(indexed) != list(range(1, len(indexed) + 1)):
        raise Tier0Error("paid commitment indices are not contiguous", status="STOP_PAID_JOURNAL_INVALID")
    physical_commitment_indices = [
        pair["start"].get("payload", {}).get("commitment_index") for pair in start_pairs
    ]
    if physical_commitment_indices != list(range(1, len(start_pairs) + 1)):
        raise Tier0Error(
            "paid commitment indices do not match attempt-lineage/journal order",
            status="STOP_PAID_JOURNAL_INVALID",
        )

    state = PaidBudgetState()
    snapshots: list[PaidBudgetState] = [PaidBudgetState()]
    commitments: list[dict[str, Any]] = []
    for index in range(1, len(indexed) + 1):
        pair = indexed[index]
        cost = pair["cost"]
        start = pair["start"]
        start_attempt_ordinal = int(pair["attempt"]["header"]["paid_attempt_ordinal"])
        request = allowed[str(start["session"])]
        payload = cost.get("payload", {})
        quote_text = payload.get("observed_sdk_quote_usd")
        if not isinstance(quote_text, str):
            raise Tier0Error("paid start quote is not an exact decimal string", status="STOP_PAID_JOURNAL_INVALID")
        try:
            if normalize_paid_quote(Decimal(quote_text)) != quote_text:
                raise ValueError("noncanonical")
        except Exception as exc:  # noqa: BLE001
            raise Tier0Error("paid start quote is malformed", status="STOP_PAID_JOURNAL_INVALID") from exc
        expected_cost = _numeric_cost_payload(state, request.session, quote_text)
        if payload != expected_cost or payload["time_series_start_permitted"] is not True:
            raise Tier0Error("paid start quote arithmetic/gate payload is false", status="STOP_PAID_JOURNAL_INVALID")
        if payload["commitment_count_before"] != index - 1:
            raise Tier0Error("paid quote commitment count/order is false", status="STOP_PAID_JOURNAL_INVALID")
        projection = state.projection(request.session, Decimal(quote_text))
        start_payload = start.get("payload", {})
        ordinal = ordinal_by_session[request.session]
        expected_start = _paid_start_payload(
            request=request,
            output_path=Path(str(start_payload.get("output_relative", ""))),
            quote_text=quote_text,
            cost_result_record_hash=str(cost["record_hash"]),
            state=state,
            projection=projection,
            readiness_receipt_sha256=pair["attempt"]["header"]["paid_readiness_receipt_sha256"],
            readiness_receipt_file_sha256=pair["attempt"]["header"]["paid_readiness_receipt_file_sha256"],
            ordinal=ordinal,
        )
        if start_payload != expected_start or start_payload.get("output_relative") != "data.cmbp-1.dbn.zst":
            raise Tier0Error("paid start request/commitment payload drifted", status="STOP_PAID_JOURNAL_INVALID")
        state.commit(request.session, Decimal(quote_text), projection)
        snapshots.append(
            PaidBudgetState(
                committed_total_usd=state.committed_total_usd,
                committed_by_session_usd=dict(state.committed_by_session_usd),
                commitment_count=state.commitment_count,
            )
        )
        commitments.append(
            {
                "commitment_index": index,
                "attempt_id": start["attempt_id"],
                "paid_attempt_ordinal": start_attempt_ordinal,
                "session": request.session,
                "request_sha256": request.market_request_sha256,
                "fresh_quote_usd": quote_text,
                "timeseries_start_sequence": start["sequence"],
                "timeseries_start_record_hash": start["record_hash"],
                "committed_quote_session_usd": format(state.committed_by_session_usd[request.session], "f"),
                "committed_quote_total_usd": format(state.committed_total_usd, "f"),
            }
        )

    paired_cost_hashes = {pair["cost"]["record_hash"] for pair in start_pairs}
    physical_start_keys = [
        (
            int(pair["attempt"]["header"]["paid_attempt_ordinal"]),
            int(pair["start"]["sequence"]),
        )
        for pair in start_pairs
    ]
    quote_observation_receipts: list[dict[str, Any]] = []
    for observation in cost_observations:
        record = observation["record"]
        payload = record.get("payload", {})
        quote_observation_receipts.append(
            {
                "attempt_id": record["attempt_id"],
                "paid_attempt_ordinal": observation["attempt"]["header"]["paid_attempt_ordinal"],
                "session": record["session"],
                "request_sha256": record["request_sha256"],
                "cost_result_sequence": record["sequence"],
                "cost_result_record_hash": record["record_hash"],
                "quote_valid": payload.get("quote_valid"),
                "observed_sdk_quote": payload.get("observed_sdk_quote_usd", payload.get("observed_sdk_quote")),
                "commitment_count_before": payload.get("commitment_count_before"),
                "time_series_start_permitted": payload.get("time_series_start_permitted"),
                "committed_to_time_series_start": record["record_hash"] in paired_cost_hashes,
            }
        )
        if record["record_hash"] in paired_cost_hashes:
            continue
        count_before = payload.get("commitment_count_before")
        if isinstance(count_before, bool) or not isinstance(count_before, int) or not (0 <= count_before <= len(indexed)):
            raise Tier0Error("uncommitted quote has an invalid commitment count", status="STOP_PAID_JOURNAL_INVALID")
        observation_key = (
            int(observation["attempt"]["header"]["paid_attempt_ordinal"]),
            int(record["sequence"]),
        )
        physical_prior_start_count = sum(1 for key in physical_start_keys if key < observation_key)
        if count_before != physical_prior_start_count:
            raise Tier0Error(
                "uncommitted quote commitment count does not match physical lineage order",
                status="STOP_PAID_JOURNAL_INVALID",
            )
        prior = snapshots[count_before]
        session = str(record["session"])
        if payload.get("quote_valid") is True:
            quote_text = payload.get("observed_sdk_quote_usd")
            if not isinstance(quote_text, str) or payload != _numeric_cost_payload(prior, session, quote_text):
                raise Tier0Error("uncommitted paid quote arithmetic is false", status="STOP_PAID_JOURNAL_INVALID")
            passing = payload["time_series_start_permitted"] is True
        else:
            expected_invalid_fields = {
                "observed_sdk_quote",
                "quote_valid",
                "commitment_count_before",
                "committed_quote_session_before_usd",
                "committed_quote_total_before_usd",
                "per_session_lifetime_cap_usd",
                "total_cap_usd",
                "per_session_lifetime_cap_pass",
                "total_cap_pass",
                "time_series_start_permitted",
                "actual_vendor_invoice_cost_usd",
            }
            if (
                set(payload) != expected_invalid_fields
                or not isinstance(payload.get("observed_sdk_quote"), str)
                or payload.get("quote_valid") is not False
                or payload.get("committed_quote_session_before_usd")
                != format(prior.committed_by_session_usd.get(session, Decimal("0")), "f")
                or payload.get("committed_quote_total_before_usd") != format(prior.committed_total_usd, "f")
                or payload.get("per_session_lifetime_cap_usd") != "1.50"
                or payload.get("total_cap_usd") != "32.00"
                or payload.get("per_session_lifetime_cap_pass") is not False
                or payload.get("total_cap_pass") is not False
                or payload.get("time_series_start_permitted") is not False
                or payload.get("actual_vendor_invoice_cost_usd") != "UNKNOWN"
            ):
                raise Tier0Error("malformed paid quote refusal payload drifted", status="STOP_PAID_JOURNAL_INVALID")
            passing = False
        if not passing:
            terminal_failures.append(
                {
                    "attempt_id": record["attempt_id"],
                    "paid_attempt_ordinal": observation["attempt"]["header"]["paid_attempt_ordinal"],
                    "session": session,
                    "request_sha256": record["request_sha256"],
                    "cost_result_sequence": record["sequence"],
                    "commitment_count_before": count_before,
                    "quote_valid": payload.get("quote_valid"),
                    "observed_sdk_quote": payload.get("observed_sdk_quote_usd", payload.get("observed_sdk_quote")),
                }
            )
            if len(indexed) > count_before:
                raise Tier0Error("a time-series start follows a terminal paid quote failure", status="STOP_PAID_AUTHORITY_TERMINAL")

    if terminal_failures:
        first_terminal_ordinal = min(int(item["paid_attempt_ordinal"]) for item in terminal_failures)
        for item in paid_verified:
            attempt_ordinal = int(item["header"]["paid_attempt_ordinal"])
            if attempt_ordinal > first_terminal_ordinal:
                raise Tier0Error(
                    "a paid attempt exists after a terminal quote failure",
                    status="STOP_PAID_AUTHORITY_TERMINAL",
                )

    start_by_hash = {pair["start"]["record_hash"]: pair for pair in start_pairs}
    terminal_by_start: set[str] = set()
    classified_data_paths: set[str] = set()
    recoverable_staging: list[dict[str, Any]] = []
    partial_or_failed_staging: list[dict[str, Any]] = []
    timeseries_result_records: list[dict[str, Any]] = []
    for pair in result_pairs:
        start = pair["start"]
        terminal = pair["terminal"]
        start_payload = start["payload"]
        context = {
            "fresh_quote_usd": start_payload["fresh_quote_usd"],
            "commitment_index": start_payload["commitment_index"],
            "timeseries_start_record_hash": start["record_hash"],
            "committed_quote_session_usd": start_payload["committed_quote_session_after_usd"],
            "committed_quote_total_usd": start_payload["committed_quote_total_after_usd"],
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
        }
        payload = terminal.get("payload", {})
        if terminal["event"] == "TIMESERIES_CALL_RESULT":
            if (
                not isinstance(payload, dict)
                or set(payload) != set(context) | {"compressed_bytes", "dbn_file_sha256"}
                or {key: payload.get(key) for key in context} != context
                or isinstance(payload.get("compressed_bytes"), bool)
                or not isinstance(payload.get("compressed_bytes"), int)
                or payload["compressed_bytes"] <= 0
                or SHA256_RE.fullmatch(str(payload.get("dbn_file_sha256"))) is None
            ):
                raise Tier0Error("paid time-series result payload drifted", status="STOP_PAID_JOURNAL_INVALID")
            timeseries_result_records.append(terminal)
        else:
            if (
                not isinstance(payload, dict)
                or set(payload) != set(context) | {"error_class"}
                or {key: payload.get(key) for key in context} != context
                or not isinstance(payload.get("error_class"), str)
                or not payload["error_class"]
            ):
                raise Tier0Error("paid time-series error payload drifted", status="STOP_PAID_JOURNAL_INVALID")
        terminal_by_start.add(start["record_hash"])

        attempt_dir = pair["attempt"]["attempt_dir"]
        staging_dir = attempt_dir / "sessions" / f"{start['session']}.bundle.part"
        data_path = staging_dir / "data.cmbp-1.dbn.zst"
        if data_path.exists() or data_path.is_symlink():
            metadata = data_path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise Tier0Error("paid staging data file is unsafe", status="STOP_PAID_JOURNAL_INVALID")
            staged = {
                "attempt_id": start["attempt_id"],
                "session": start["session"],
                "request_sha256": start["request_sha256"],
                "data_path": str(data_path),
                "compressed_bytes": metadata.st_size,
                "file_sha256": base.file_sha256(data_path),
                "timeseries_start_record_hash": start["record_hash"],
                "timeseries_terminal_record_hash": terminal["record_hash"],
                "timeseries_terminal_sequence": terminal["sequence"],
            }
            classified_data_paths.add(str(data_path))
            if terminal["event"] == "TIMESERIES_CALL_RESULT":
                if (
                    staged["compressed_bytes"] != payload["compressed_bytes"]
                    or staged["file_sha256"] != payload["dbn_file_sha256"]
                ):
                    raise Tier0Error("result-bound paid staging file drifted", status="STOP_PAID_JOURNAL_INVALID")
                recoverable_staging.append(staged)
            else:
                partial_or_failed_staging.append(staged)

    starts_by_attempt_session: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for pair in start_pairs:
        key = (str(pair["start"]["attempt_id"]), str(pair["start"]["session"]))
        starts_by_attempt_session.setdefault(key, []).append(pair["start"])
    for item in paid_verified:
        attempt_dir = item["attempt_dir"]
        for session, names in item["staging"].items():
            key = (attempt_dir.name, session)
            starts = starts_by_attempt_session.get(key, [])
            data_path = attempt_dir / "sessions" / f"{session}.bundle.part" / "data.cmbp-1.dbn.zst"
            qc_path = data_path.parent / PAID_SESSION_QC_NAME
            if "data.cmbp-1.dbn.zst" in names and not starts:
                raise Tier0Error("paid staging data has no time-series start", status="STOP_PAID_JOURNAL_INVALID")
            if PAID_SESSION_QC_NAME in names and not starts:
                raise Tier0Error("paid staging QC has no time-series source", status="STOP_PAID_JOURNAL_INVALID")
            if len(starts) > 1:
                raise Tier0Error("one paid attempt staged the same session more than once", status="STOP_PAID_JOURNAL_INVALID")
            if "data.cmbp-1.dbn.zst" in names and str(data_path) not in classified_data_paths:
                start = starts[0]
                metadata = data_path.lstat()
                partial_or_failed_staging.append(
                    {
                        "attempt_id": start["attempt_id"],
                        "session": start["session"],
                        "request_sha256": start["request_sha256"],
                        "data_path": str(data_path),
                        "compressed_bytes": metadata.st_size,
                        "file_sha256": base.file_sha256(data_path),
                        "timeseries_start_record_hash": start["record_hash"],
                        "timeseries_terminal_record_hash": None,
                        "timeseries_terminal_sequence": None,
                    }
                )
            if qc_path.exists() and str(data_path) not in {
                item["data_path"] for item in recoverable_staging
            }:
                raise Tier0Error("paid staging QC is not result-bound", status="STOP_PAID_JOURNAL_INVALID")

    pending_timeseries = sum(1 for pair in start_pairs if pair["start"]["record_hash"] not in terminal_by_start)
    result_lookup = {
        (str(record["attempt_id"]), int(record["sequence"])): record
        for record in timeseries_result_records
    }
    attempt_ordinal_by_id = {
        item["attempt_dir"].name: int(item["header"]["paid_attempt_ordinal"])
        for item in paid_verified
    }
    publication_seen_sessions: set[str] = set()
    publication_seen_by_attempt: set[tuple[str, str]] = set()
    for item in publication_records:
        record = item["record"]
        request = item["request"]
        payload = record.get("payload")
        expected_fields = {
            "ordinal",
            "publication_mode",
            "session_qc_sha256",
            "decoded_records",
            "compressed_bytes",
            "source_attempt_id",
            "source_timeseries_result_sequence",
            "source_timeseries_result_record_hash",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise Tier0Error("paid publication payload field population drifted", status="STOP_PAID_JOURNAL_INVALID")
        source_key = (str(payload.get("source_attempt_id")), payload.get("source_timeseries_result_sequence"))
        source = result_lookup.get(source_key) if isinstance(source_key[1], int) and not isinstance(source_key[1], bool) else None
        publication_key = (
            int(item["attempt"]["header"]["paid_attempt_ordinal"]),
            int(record["sequence"]),
        )
        source_order_key = (
            attempt_ordinal_by_id.get(str(payload.get("source_attempt_id")), -1),
            int(payload.get("source_timeseries_result_sequence", -1)),
        )
        attempt_session_key = (str(record["attempt_id"]), request.session)
        final_dir = job_root / "sessions" / request.session
        if (
            payload.get("ordinal") != ordinal_by_session[request.session]
            or payload.get("publication_mode") != record["event"]
            or SHA256_RE.fullmatch(str(payload.get("session_qc_sha256"))) is None
            or payload.get("decoded_records") != request.expected_record_count
            or isinstance(payload.get("compressed_bytes"), bool)
            or not isinstance(payload.get("compressed_bytes"), int)
            or payload["compressed_bytes"] <= 0
            or source is None
            or source.get("record_hash") != payload.get("source_timeseries_result_record_hash")
            or source.get("session") != request.session
            or source.get("request_sha256") != request.market_request_sha256
            or source.get("payload", {}).get("compressed_bytes") != payload.get("compressed_bytes")
            or (record["event"] == "SESSION_PUBLISHED" and payload.get("source_attempt_id") != record["attempt_id"])
            or (record["event"] == "SESSION_RECOVERED" and payload.get("source_attempt_id") == record["attempt_id"])
            or source_order_key >= publication_key
            or attempt_session_key in publication_seen_by_attempt
            or final_dir.is_symlink()
            or not final_dir.is_dir()
            or final_dir.stat().st_dev != job_root_dev
        ):
            raise Tier0Error("paid publication/source binding drifted", status="STOP_PAID_JOURNAL_INVALID")
        if record["event"] in {"SESSION_PUBLISHED", "SESSION_RECOVERED"}:
            if request.session in publication_seen_sessions:
                raise Tier0Error(
                    "paid first-publication mode follows earlier publication evidence",
                    status="STOP_PAID_JOURNAL_INVALID",
                )
        elif request.session not in publication_seen_sessions:
            raise Tier0Error(
                "paid reuse lacks earlier publication evidence",
                status="STOP_PAID_JOURNAL_INVALID",
            )
        publication_seen_sessions.add(request.session)
        publication_seen_by_attempt.add(attempt_session_key)

    for item in aggregate_seal_records:
        record = item["record"]
        payload = record.get("payload")
        count = payload.get("commitment_count") if isinstance(payload, dict) else None
        if isinstance(count, bool) or not isinstance(count, int) or not (0 <= count < len(snapshots)):
            raise Tier0Error("paid aggregate seal commitment count is invalid", status="STOP_PAID_JOURNAL_INVALID")
        snapshot = snapshots[count]
        expected_payload = {
            "published_or_reused_sessions": base.EXPECTED_SESSION_COUNT,
            "decoded_records": base.EXPECTED_RECORD_COUNT,
            "commitment_count": count,
            "committed_quote_total_usd": format(snapshot.committed_total_usd, "f"),
            "paid_readiness_receipt_sha256": item["attempt"]["header"]["paid_readiness_receipt_sha256"],
        }
        if payload != expected_payload or len(indexed) > count:
            raise Tier0Error("paid aggregate seal payload/order drifted", status="STOP_PAID_JOURNAL_INVALID")
    if len(aggregate_seal_records) > 1:
        raise Tier0Error("paid attempt population contains multiple aggregate seals", status="STOP_PAID_JOURNAL_INVALID")
    if aggregate_seal_records:
        seal_item = aggregate_seal_records[0]
        seal_record = seal_item["record"]
        seal_ordinal = int(seal_item["attempt"]["header"]["paid_attempt_ordinal"])
        latest_attempt_ordinal = max(
            (int(item["header"]["paid_attempt_ordinal"]) for item in paid_verified),
            default=0,
        )
        seal_key = (seal_ordinal, int(seal_record["sequence"]))
        seal_epoch_publications = [
            item
            for item in publication_records
            if int(item["attempt"]["header"]["paid_attempt_ordinal"]) == seal_ordinal
        ]
        publication_sessions = [str(item["record"]["session"]) for item in seal_epoch_publications]
        if (
            seal_ordinal != latest_attempt_ordinal
            or len(seal_epoch_publications) != base.EXPECTED_SESSION_COUNT
            or len(set(publication_sessions)) != base.EXPECTED_SESSION_COUNT
            or set(publication_sessions) != set(allowed)
            or any(
                (int(item["attempt"]["header"]["paid_attempt_ordinal"]), int(item["record"]["sequence"]))
                >= seal_key
                for item in seal_epoch_publications
            )
        ):
            raise Tier0Error(
                "paid aggregate seal is not the unique terminal seal after all exact publications",
                status="STOP_PAID_JOURNAL_INVALID",
            )
    committed_by_session = {
        session: format(value, "f") for session, value in sorted(state.committed_by_session_usd.items())
    }
    return {
        "legacy_attempt": legacy_summary,
        "paid_attempt_count": len(paid_verified),
        "cost_call_starts": cost_starts,
        "cost_call_results": cost_results,
        "cost_call_errors": cost_errors,
        "timeseries_call_starts": timeseries_starts,
        "timeseries_call_results": timeseries_results,
        "failed_timeseries_starts": failed_timeseries,
        "pending_timeseries_starts": pending_timeseries,
        "pending_cost_call_starts": pending_cost_calls,
        "uncommitted_passing_quote_count": uncommitted_passing,
        "session_timeseries_attempt_counts": dict(sorted(session_attempt_counts.items())),
        "duplicate_session_attempts": {
            session: count for session, count in sorted(session_attempt_counts.items()) if count > 1
        },
        "committed_quote_total_usd": format(state.committed_total_usd, "f"),
        "committed_quote_by_session_usd": committed_by_session,
        "per_session_lifetime_cap_usd": "1.50",
        "total_committed_quote_cap_usd": "32.00",
        "commitments": commitments,
        "quote_observations": quote_observation_receipts,
        "all_quote_observations": (
            ([{
                "attempt_id": EXPECTED_LEGACY_ATTEMPT_ID,
                "session": "2025-08-22",
                "observed_sdk_quote": "0.950392448902",
                "quote_valid_under_job51_zero_gate": False,
                "committed_to_time_series_start": False,
            }] if legacy_summary is not None else [])
            + quote_observation_receipts
        ),
        "terminal_authority_failure_observed": bool(terminal_failures),
        "terminal_authority_failures": terminal_failures,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "timeseries_result_records": timeseries_result_records,
        "timeseries_start_records": [pair["start"] for pair in start_pairs],
        "recoverable_staging": recoverable_staging,
        "partial_or_failed_staging": partial_or_failed_staging,
        "successful_stream_recovery_required": global_recovery_barrier is not None,
        "successful_stream_recovery_record": (
            dict(global_recovery_barrier) if global_recovery_barrier is not None else None
        ),
        "published_sessions_in_journals": sorted(published_sessions),
        "publication_records": [item["record"] for item in publication_records],
        "aggregate_seal_count": len(aggregate_seal_records),
        "aggregate_seal_attempt_id": (
            aggregate_seal_records[0]["record"]["attempt_id"] if aggregate_seal_records else None
        ),
        "attempts": attempt_summaries,
    }


def _paid_junit_counts(path: Path, *, enforce_frozen_population: bool = True) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:  # noqa: BLE001
        raise Tier0Error("paid focused JUnit report is unreadable", status="STOP_PAID_LOCAL_TESTS") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        raise Tier0Error("paid focused JUnit report has no suite", status="STOP_PAID_LOCAL_TESTS")
    counts = {
        name: sum(int(float(suite.attrib.get(name, "0"))) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    cases = list(root.findall(".//testcase"))
    identities = [
        {"classname": classname, "name": name}
        for classname, name in sorted(
            (str(case.attrib.get("classname", "")), str(case.attrib.get("name", ""))) for case in cases
        )
    ]
    names = sorted(item["name"] for item in identities if item["name"])
    classnames = sorted(set(item["classname"] for item in identities))
    identity_sha = base.json_sha256(identities)
    allowed = (
        "v5.tests.test_cmbp_stream",
        "v5.tests.test_cmbp_tier0",
        "v5.tests.test_cmbp_tier0_paid",
    )
    if (
        counts["tests"] < 12
        or counts["failures"]
        or counts["errors"]
        or counts["skipped"]
        or len(identities) != counts["tests"]
        or not classnames
        or any(not classname.startswith(allowed) for classname in classnames)
    ):
        raise Tier0Error("paid focused tests are not a clean exact-target pass", status="STOP_PAID_LOCAL_TESTS")
    if enforce_frozen_population and (
        EXPECTED_FOCUSED_TEST_COUNT <= 0
        or counts["tests"] != EXPECTED_FOCUSED_TEST_COUNT
        or identity_sha != EXPECTED_FOCUSED_TEST_IDENTITY_SHA256
    ):
        raise Tier0Error("paid focused test population/identity drifted", status="STOP_PAID_LOCAL_TESTS")
    return {
        **counts,
        "test_names": names,
        "test_classnames": classnames,
        "test_case_identity_sha256": identity_sha,
    }


def paid_sdk_identity() -> dict[str, Any]:
    """Bind Job-51 SDK evidence plus compression and TLS trust inputs."""

    identity = base.sdk_identity()
    try:
        zstandard_distribution = importlib.metadata.distribution("zstandard")
    except importlib.metadata.PackageNotFoundError as exc:
        raise Tier0Error("zstandard is not installed", status="STOP_SDK_DRIFT") from exc
    if zstandard_distribution.version != EXPECTED_ZSTANDARD_VERSION:
        raise Tier0Error("pinned zstandard version drifted", status="STOP_SDK_DRIFT")

    import zstandard
    import zstandard.backend_c as zstandard_backend

    if getattr(zstandard, "backend", None) != "cext":
        raise Tier0Error("zstandard active backend drifted", status="STOP_SDK_DRIFT")
    module_files: dict[str, dict[str, str]] = {}
    for name, module in {
        "zstandard.__init__": zstandard,
        "zstandard.backend_c": zstandard_backend,
    }.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise Tier0Error(f"compression identity unavailable: {name}", status="STOP_SDK_DRIFT")
        unresolved = Path(raw_path)
        if unresolved.is_symlink():
            raise Tier0Error(f"compression identity unsafe: {name}", status="STOP_SDK_DRIFT")
        path = unresolved.resolve()
        if not path.is_file():
            raise Tier0Error(f"compression identity unsafe: {name}", status="STOP_SDK_DRIFT")
        module_files[name] = {"path": str(path), "file_sha256": base.file_sha256(path)}

    manifest: dict[str, str] = {}
    for relative in zstandard_distribution.files or ():
        relative_text = str(relative)
        located = Path(zstandard_distribution.locate_file(relative))
        if located.is_symlink():
            raise Tier0Error("zstandard distribution contains a symlink", status="STOP_SDK_DRIFT")
        located = located.resolve()
        if not located.is_file():
            raise Tier0Error("zstandard distribution file is absent", status="STOP_SDK_DRIFT")
        manifest[relative_text] = base.file_sha256(located)
    if not manifest:
        raise Tier0Error("zstandard distribution manifest is empty", status="STOP_SDK_DRIFT")

    try:
        certifi_distribution = importlib.metadata.distribution("certifi")
    except importlib.metadata.PackageNotFoundError as exc:
        raise Tier0Error("certifi is not installed", status="STOP_SDK_DRIFT") from exc
    if certifi_distribution.version != EXPECTED_CERTIFI_VERSION:
        raise Tier0Error("pinned certifi version drifted", status="STOP_SDK_DRIFT")

    import certifi
    import certifi.core as certifi_core
    import requests.certs as requests_certs

    certifi_modules: dict[str, dict[str, str]] = {}
    for name, module in {
        "certifi.__init__": certifi,
        "certifi.core": certifi_core,
        "requests.certs": requests_certs,
    }.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise Tier0Error(f"TLS trust identity unavailable: {name}", status="STOP_SDK_DRIFT")
        unresolved = Path(raw_path)
        if unresolved.is_symlink():
            raise Tier0Error(f"TLS trust identity unsafe: {name}", status="STOP_SDK_DRIFT")
        path = unresolved.resolve()
        if not path.is_file():
            raise Tier0Error(f"TLS trust identity unsafe: {name}", status="STOP_SDK_DRIFT")
        certifi_modules[name] = {"path": str(path), "file_sha256": base.file_sha256(path)}

    ca_unresolved = Path(certifi.where())
    if ca_unresolved.is_symlink():
        raise Tier0Error("certifi CA bundle is unsafe", status="STOP_SDK_DRIFT")
    ca_path = ca_unresolved.resolve()
    requests_ca_path = Path(requests_certs.where()).resolve()
    if not ca_path.is_file() or requests_ca_path != ca_path:
        raise Tier0Error("Requests default CA bundle drifted", status="STOP_SDK_DRIFT")

    certifi_manifest: dict[str, str] = {}
    for relative in certifi_distribution.files or ():
        relative_text = str(relative)
        located = Path(certifi_distribution.locate_file(relative))
        if located.is_symlink():
            raise Tier0Error("certifi distribution contains a symlink", status="STOP_SDK_DRIFT")
        located = located.resolve()
        if not located.is_file():
            raise Tier0Error("certifi distribution file is absent", status="STOP_SDK_DRIFT")
        certifi_manifest[relative_text] = base.file_sha256(located)
    if not certifi_manifest:
        raise Tier0Error("certifi distribution manifest is empty", status="STOP_SDK_DRIFT")

    try:
        idna_distribution = importlib.metadata.distribution("idna")
        charset_distribution = importlib.metadata.distribution("charset-normalizer")
    except importlib.metadata.PackageNotFoundError as exc:
        raise Tier0Error("Requests runtime dependency is absent", status="STOP_SDK_DRIFT") from exc
    if (
        idna_distribution.version != EXPECTED_IDNA_VERSION
        or charset_distribution.version != EXPECTED_CHARSET_NORMALIZER_VERSION
    ):
        raise Tier0Error("pinned Requests runtime dependency drifted", status="STOP_SDK_DRIFT")

    import charset_normalizer
    import charset_normalizer.api as charset_api
    import charset_normalizer.md as charset_native
    import idna
    import idna.core as idna_core
    import idna.idnadata as idna_data

    http_dependency_modules: dict[str, dict[str, str]] = {}
    for name, module in {
        "idna.__init__": idna,
        "idna.core": idna_core,
        "idna.idnadata": idna_data,
        "charset_normalizer.__init__": charset_normalizer,
        "charset_normalizer.api": charset_api,
        "charset_normalizer.md": charset_native,
    }.items():
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise Tier0Error(f"HTTP dependency identity unavailable: {name}", status="STOP_SDK_DRIFT")
        unresolved = Path(raw_path)
        if unresolved.is_symlink():
            raise Tier0Error(f"HTTP dependency identity unsafe: {name}", status="STOP_SDK_DRIFT")
        path = unresolved.resolve()
        if not path.is_file():
            raise Tier0Error(f"HTTP dependency identity unsafe: {name}", status="STOP_SDK_DRIFT")
        http_dependency_modules[name] = {"path": str(path), "file_sha256": base.file_sha256(path)}

    http_dependency_distributions: dict[str, dict[str, Any]] = {}
    for distribution_name, distribution in {
        "idna": idna_distribution,
        "charset-normalizer": charset_distribution,
    }.items():
        dependency_manifest: dict[str, str] = {}
        for relative in distribution.files or ():
            relative_text = str(relative)
            located = Path(distribution.locate_file(relative))
            if located.is_symlink():
                raise Tier0Error(
                    f"{distribution_name} distribution contains a symlink",
                    status="STOP_SDK_DRIFT",
                )
            located = located.resolve()
            if not located.is_file():
                raise Tier0Error(
                    f"{distribution_name} distribution file is absent",
                    status="STOP_SDK_DRIFT",
                )
            dependency_manifest[relative_text] = base.file_sha256(located)
        if not dependency_manifest:
            raise Tier0Error(
                f"{distribution_name} distribution manifest is empty",
                status="STOP_SDK_DRIFT",
            )
        http_dependency_distributions[distribution_name] = {
            "version": distribution.version,
            "file_count": len(dependency_manifest),
            "manifest_sha256": base.json_sha256(dependency_manifest),
            "files": dependency_manifest,
        }

    return {
        "job51_sdk_identity": identity,
        "job52_compression_identity": {
            "active_backend": "cext",
            "version": zstandard_distribution.version,
            "files": module_files,
            "distribution": {
                "file_count": len(manifest),
                "manifest_sha256": base.json_sha256(manifest),
                "files": manifest,
            },
        },
        "job52_transport_trust_identity": {
            "version": certifi_distribution.version,
            "modules": certifi_modules,
            "default_ca_bundle": {
                "path": str(ca_path),
                "file_sha256": base.file_sha256(ca_path),
            },
            "distribution": {
                "file_count": len(certifi_manifest),
                "manifest_sha256": base.json_sha256(certifi_manifest),
                "files": certifi_manifest,
            },
        },
        "job52_http_dependency_identity": {
            "modules": http_dependency_modules,
            "distributions": http_dependency_distributions,
        },
    }


def build_paid_readiness_receipt(repo_root: Path, *, test_report_path: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    bundle = load_paid_scope_bundle(root)
    contract = bundle.contract
    bound_files: dict[str, str] = {}
    required = contract.get("required_bound_files")
    if not isinstance(required, list) or len(required) != len(set(required)):
        raise Tier0Error("paid contract bound-file list is malformed", status="STOP_PAID_CONTRACT_DRIFT")
    for relative in required:
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise Tier0Error("paid contract contains an unsafe bound path", status="STOP_PAID_CONTRACT_DRIFT")
        path = root / relative
        _private_regular(path, status="STOP_PAID_CONTRACT_DRIFT")
        bound_files[relative] = base.file_sha256(path)
    report = Path(os.path.abspath(test_report_path))
    expected_report = Path(os.path.abspath(_paths(root)["test_report"]))
    if report != expected_report:
        raise Tier0Error("paid JUnit path is not canonical", status="STOP_PAID_LOCAL_TESTS")
    _private_regular(report, status="STOP_PAID_LOCAL_TESTS")
    legacy = base.strict_json(_paths(root)["legacy_stop"])
    receipt: dict[str, Any] = {
        "artifact_type": READINESS_ARTIFACT,
        "schema_version": "v5.job52-cmbp-tier0-paid-resume-local-readiness.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "status": READINESS_STATUS,
        "status_meaning": "LOCAL_BUILD_AND_TEST_SEAL_ONLY; NO_VENDOR_OR_ACQUISITION_RESULT",
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "paid_program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "paid_plan_file_sha256": EXPECTED_PLAN_FILE_SHA256,
        "job51_program_contract_sha256": base.EXPECTED_PROGRAM_CONTRACT_SHA256,
        "job51_program_contract_file_sha256": base.EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "job51_local_readiness_sha256": "851ae1de70308a19fad4160bf33464a3e0a2c79bba15085bde80aea5ea99ff03",
        "job51_local_readiness_file_sha256": "9e308c7f4d3cd6da97c46df54c1a61104e5cbfad0d365b3661ab2e8e93f166aa",
        "legacy_stop_receipt_sha256": legacy["receipt_sha256"],
        "legacy_stop_receipt_file_sha256": base.file_sha256(_paths(root)["legacy_stop"]),
        "money_boundary": {
            "per_session_lifetime_committed_quote_cap_usd": "1.50",
            "total_committed_quote_cap_usd": "32.00",
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
        },
        "bound_files": bound_files,
        "focused_test_report": {
            "path": str(report.relative_to(root)),
            "file_sha256": base.file_sha256(report),
            "runner": "./.venv/bin/python -m pytest",
            "runner_flags": [
                "-c",
                "/dev/null",
                "--rootdir=<REPOSITORY_ROOT>",
                "-p",
                "no:cacheprovider",
                "-q",
            ],
            "test_targets": [
                "v5/tests/test_cmbp_stream.py",
                "v5/tests/test_cmbp_tier0.py",
                "v5/tests/test_cmbp_tier0_paid.py",
            ],
            "environment_allowlist": [
                "PATH",
                "LANG",
                "LC_ALL",
                "PYTHONHASHSEED",
                "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            ],
            "pytest_plugin_autoload_disabled": True,
            "repository_and_environment_pytest_addopts_ignored": True,
            "report_published_atomically_after_subprocess_exit_zero": True,
            **_paid_junit_counts(report),
        },
        "sdk_identity": paid_sdk_identity(),
        "integrity": {
            "credential_read": False,
            "authenticated_client_constructed": False,
            "external_calls": 0,
            "metadata_calls": 0,
            "timeseries_calls": 0,
            "data_downloaded": False,
            "acquisition_initiated": False,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "models_fit": 0,
            "broker_calls": 0,
            "orders_submitted": 0,
        },
        "authorization_effect": "LOCAL_READINESS_ONLY; EXECUTION_REMAINS_BOUND_TO_CURRENT_OWNER_CAP_AUTHORITY",
    }
    receipt["receipt_sha256"] = base.self_hash(receipt, "receipt_sha256")
    return receipt


def validate_paid_readiness_receipt(repo_root: Path, path: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    canonical = Path(os.path.abspath(_paths(root)["readiness"]))
    target = Path(os.path.abspath(canonical if path is None else path))
    if target != canonical:
        raise Tier0Error("paid readiness path is not canonical", status="STOP_PAID_READINESS")
    _private_regular(target, status="STOP_PAID_READINESS")
    receipt = base.strict_json(target)
    if (
        receipt.get("artifact_type") != READINESS_ARTIFACT
        or receipt.get("schema_version") != "v5.job52-cmbp-tier0-paid-resume-local-readiness.v1"
        or receipt.get("status") != READINESS_STATUS
        or receipt.get("receipt_sha256") != base.self_hash(receipt, "receipt_sha256")
    ):
        raise Tier0Error("paid readiness identity/self-hash drifted", status="STOP_PAID_READINESS")
    reconstructed = build_paid_readiness_receipt(root, test_report_path=_paths(root)["test_report"])
    if receipt != reconstructed:
        raise Tier0Error("paid readiness does not reconstruct exactly", status="STOP_PAID_READINESS")
    if receipt.get("integrity") != {
        "credential_read": False,
        "authenticated_client_constructed": False,
        "external_calls": 0,
        "metadata_calls": 0,
        "timeseries_calls": 0,
        "data_downloaded": False,
        "acquisition_initiated": False,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "models_fit": 0,
        "broker_calls": 0,
        "orders_submitted": 0,
    }:
        raise Tier0Error("paid readiness is not precredential/local-only", status="STOP_PAID_READINESS")
    return receipt


def seal_paid_readiness(repo_root: Path) -> Path:
    root = Path(repo_root).resolve()
    paths = _paths(root)
    receipt = build_paid_readiness_receipt(root, test_report_path=paths["test_report"])
    if paths["readiness"].exists() or paths["readiness"].is_symlink():
        raise Tier0Error("paid readiness V1 already exists and is immutable", status="STOP_PAID_READINESS")
    base.write_canonical_exclusive(paths["readiness"], receipt)
    validate_paid_readiness_receipt(root, paths["readiness"])
    return paths["readiness"]


def validate_paid_job_tree(
    job_root: Path,
    *,
    volume: VolumeIdentity,
    allowed_requests: Sequence[SessionRequest],
    allow_missing_paid_controls: bool = False,
) -> None:
    """Refuse unexpected root, receipt, session, and control entries."""

    job_root = Path(job_root)
    base.ensure_nofollow_directory(job_root, volume=volume)
    allowed_root = {
        "attempts",
        "sessions",
        "receipts",
        "RUN_LOCK_V1",
        PAID_LOCK_BINDING_NAME,
        PAID_ATTEMPT_ANCHOR_NAME,
    }
    actual_root = {path.name for path in job_root.iterdir()}
    if any(name not in allowed_root for name in actual_root):
        raise Tier0Error("Job-51/52 root contains an unexpected entry", status="STOP_PAID_EXTERNAL_TREE")
    required = {"attempts", "sessions", "receipts", "RUN_LOCK_V1"}
    if not required <= actual_root:
        raise Tier0Error("Job-51/52 root is incomplete", status="STOP_PAID_EXTERNAL_TREE")
    if not allow_missing_paid_controls and not {PAID_LOCK_BINDING_NAME, PAID_ATTEMPT_ANCHOR_NAME} <= actual_root:
        raise Tier0Error("Job-52 root controls are incomplete", status="STOP_PAID_EXTERNAL_TREE")
    for directory_name in ("attempts", "sessions", "receipts"):
        base.ensure_nofollow_directory(job_root / directory_name, volume=volume)
    for file_name in actual_root & {"RUN_LOCK_V1", PAID_LOCK_BINDING_NAME, PAID_ATTEMPT_ANCHOR_NAME}:
        metadata = _private_regular(job_root / file_name, status="STOP_PAID_EXTERNAL_TREE")
        if metadata.st_dev != volume.st_dev:
            raise Tier0Error("Job-52 root control is cross-device", status="STOP_PAID_EXTERNAL_TREE")
    expected_sessions = {request.session for request in allowed_requests}
    for child in (job_root / "sessions").iterdir():
        if child.is_symlink() or not child.is_dir() or child.name not in expected_sessions:
            raise Tier0Error("paid final sessions root contains an unexpected entry", status="STOP_PAID_EXTERNAL_TREE")
    # The exact legacy state stopped before its first time-series request, so a
    # Job-51 aggregate receipt can never legitimately exist in this tree.
    allowed_receipts = {PAID_AGGREGATE_NAME}
    for child in (job_root / "receipts").iterdir():
        if child.name not in allowed_receipts:
            raise Tier0Error("paid receipts root contains an unexpected entry", status="STOP_PAID_EXTERNAL_TREE")
        metadata = _private_regular(child, status="STOP_PAID_EXTERNAL_TREE")
        if metadata.st_dev != volume.st_dev:
            raise Tier0Error("paid terminal receipt is cross-device", status="STOP_PAID_EXTERNAL_TREE")


def _request_projection(request: SessionRequest) -> dict[str, Any]:
    return {
        "market_parameters": request.market_parameters,
        "market_request_sha256": request.market_request_sha256,
        "cost_request_sha256": request.cost_request_sha256,
        "record_count_request_sha256": request.record_count_request_sha256,
        "symbology_request_sha256": request.symbology_request_sha256,
        "expected_record_count": request.expected_record_count,
        "expected_mappings": [
            {"instrument_id": instrument_id, "raw_symbol": symbol}
            for instrument_id, symbol in request.expected_mappings
        ],
    }


def _validate_decoder_summary(decoder: Mapping[str, Any], request: SessionRequest) -> None:
    from v5.research.cmbp_stream import CmbpStreamSummary

    expected_decoder_fields = set(CmbpStreamSummary.__dataclass_fields__)
    if set(decoder) != expected_decoder_fields:
        raise Tier0Error(
            f"{request.session}: paid decoder field population drifted",
            status="STOP_PAID_SESSION_QC",
        )
    invariants = {
        "artifact_type": "JOB51_CMBP_STREAM_SUMMARY_V1",
        "session": request.session,
        "source_kind": "historical",
        "window_start_ns": base._timestamp_ns(request.start),
        "window_end_ns": base._timestamp_ns(request.end),
        "cmbp1_records": request.expected_record_count,
        "expected_cmbp1_records": request.expected_record_count,
        "record_count_reconciled": True,
        "mapping_reconciled": True,
        "all_expected_instruments_seen": True,
        "all_causal_priors_strict": True,
        "tied_receive_priors_excluded": True,
        "global_receive_regressions": 0,
        "classification_reconciled": True,
    }
    for field_name, value in invariants.items():
        if decoder.get(field_name) != value:
            raise Tier0Error(f"{request.session}: paid decoder invariant {field_name} drifted", status="STOP_PAID_SESSION_QC")
    count_fields = (
        "mapping_records",
        "total_records_accepted",
        "cmbp1_records",
        "expected_cmbp1_records",
        "trade_records",
        "strict_prior_trades",
        "tied_prior_trades_excluded",
        "no_prior_trades",
        "signed_trades",
        "at_bid_trades",
        "at_ask_trades",
        "inside_trades",
        "outside_trades",
        "ambiguous_trades",
        "undefined_trade_price_excluded",
        "missing_prior_book_excluded",
        "undefined_prior_book_excluded",
        "locked_prior_book_excluded",
        "crossed_prior_book_excluded",
        "trade_bad_ts_recv_excluded",
        "prior_bad_ts_recv_excluded",
        "prior_maybe_bad_book_excluded",
        "global_receive_ties",
        "global_receive_regressions",
        "global_event_regressions",
        "instrument_event_regressions",
        "disconnect_count",
        "reconnect_count",
        "gap_count",
        "book_state_clear_count",
        "gaps_with_known_bounds",
        "total_known_gap_ns",
        "system_records",
        "heartbeat_records",
        "rows_with_flags",
        "unknown_flag_rows",
    )
    if any(
        isinstance(decoder.get(field_name), bool)
        or not isinstance(decoder.get(field_name), int)
        or decoder[field_name] < 0
        for field_name in count_fields
    ):
        raise Tier0Error(f"{request.session}: paid decoder counter is invalid", status="STOP_PAID_SESSION_QC")
    expected_mapping = _request_projection(request)["expected_mappings"]
    expected_seen = [instrument_id for instrument_id, _symbol in request.expected_mappings]
    if (
        decoder.get("expected_mappings") != expected_mapping
        or decoder.get("active_mappings") != expected_mapping
        or decoder.get("seen_instrument_ids") != expected_seen
    ):
        raise Tier0Error(f"{request.session}: paid decoder mapping summary drifted", status="STOP_PAID_SESSION_QC")
    trade_total = int(decoder["trade_records"])
    exclusion_fields = (
        "undefined_trade_price_excluded",
        "missing_prior_book_excluded",
        "undefined_prior_book_excluded",
        "locked_prior_book_excluded",
        "crossed_prior_book_excluded",
        "trade_bad_ts_recv_excluded",
        "prior_bad_ts_recv_excluded",
        "prior_maybe_bad_book_excluded",
    )
    exclusions = sum(int(decoder[field_name]) for field_name in exclusion_fields)
    if trade_total != int(decoder["strict_prior_trades"]) + int(decoder["tied_prior_trades_excluded"]) + int(decoder["no_prior_trades"]):
        raise Tier0Error(f"{request.session}: paid causal trade counts do not reconcile", status="STOP_PAID_SESSION_QC")
    if int(decoder["signed_trades"]) != int(decoder["at_bid_trades"]) + int(decoder["at_ask_trades"]):
        raise Tier0Error(f"{request.session}: paid signed counts do not reconcile", status="STOP_PAID_SESSION_QC")
    if trade_total != sum(
        int(decoder[field_name])
        for field_name in ("at_bid_trades", "at_ask_trades", "inside_trades", "outside_trades", "ambiguous_trades")
    ):
        raise Tier0Error(f"{request.session}: paid classifications do not reconcile", status="STOP_PAID_SESSION_QC")
    if int(decoder["strict_prior_trades"]) != (
        int(decoder["at_bid_trades"])
        + int(decoder["at_ask_trades"])
        + int(decoder["inside_trades"])
        + int(decoder["outside_trades"])
        + exclusions
    ):
        raise Tier0Error(f"{request.session}: paid strict-prior counts do not reconcile", status="STOP_PAID_SESSION_QC")
    if int(decoder["ambiguous_trades"]) != int(decoder["tied_prior_trades_excluded"]) + int(decoder["no_prior_trades"]) + exclusions:
        raise Tier0Error(f"{request.session}: paid ambiguous counts do not reconcile", status="STOP_PAID_SESSION_QC")
    if int(decoder["total_records_accepted"]) != int(decoder["cmbp1_records"]) + int(decoder["mapping_records"]) + int(decoder["system_records"]):
        raise Tier0Error(f"{request.session}: paid accepted-record counts do not reconcile", status="STOP_PAID_SESSION_QC")
    if (
        decoder.get("disconnect_count") != 0
        or decoder.get("reconnect_count") != 0
        or decoder.get("gap_count") != 0
        or decoder.get("book_state_clear_count") != 0
        or decoder.get("gaps_with_known_bounds") != 0
        or decoder.get("total_known_gap_ns") != 0
        or decoder.get("max_known_gap_ns") is not None
        or decoder.get("explicit_connection_telemetry") != "UNKNOWN"
    ):
        raise Tier0Error(f"{request.session}: paid historical connection telemetry drifted", status="STOP_PAID_SESSION_QC")
    first_recv = decoder.get("first_ts_recv")
    last_recv = decoder.get("last_ts_recv")
    if (
        isinstance(first_recv, bool)
        or not isinstance(first_recv, int)
        or isinstance(last_recv, bool)
        or not isinstance(last_recv, int)
        or not (base._timestamp_ns(request.start) <= first_recv <= last_recv < base._timestamp_ns(request.end))
    ):
        raise Tier0Error(f"{request.session}: paid decoder receive bounds drifted", status="STOP_PAID_SESSION_QC")

    max_stream_silence = decoder.get("max_stream_silence_ns")
    receive_span_ns = last_recv - first_recv
    if (
        isinstance(max_stream_silence, bool)
        or not isinstance(max_stream_silence, int)
        or max_stream_silence < 0
        or max_stream_silence > receive_span_ns
    ):
        raise Tier0Error(
            f"{request.session}: paid stream-silence maximum is invalid",
            status="STOP_PAID_SESSION_QC",
        )
    silences = decoder.get("instrument_silences")
    if not isinstance(silences, list):
        raise Tier0Error(
            f"{request.session}: paid instrument-silence population is invalid",
            status="STOP_PAID_SESSION_QC",
        )
    expected_instrument_ids = {instrument_id for instrument_id, _symbol in request.expected_mappings}
    seen_silence_ids: list[int] = []
    silence_values: list[int] = []
    for item in silences:
        if not isinstance(item, dict) or set(item) != {"instrument_id", "max_silence_ns"}:
            raise Tier0Error(
                f"{request.session}: paid instrument-silence entry drifted",
                status="STOP_PAID_SESSION_QC",
            )
        instrument_id = item["instrument_id"]
        silence_ns = item["max_silence_ns"]
        if (
            isinstance(instrument_id, bool)
            or not isinstance(instrument_id, int)
            or instrument_id not in expected_instrument_ids
            or instrument_id in seen_silence_ids
            or isinstance(silence_ns, bool)
            or not isinstance(silence_ns, int)
            or silence_ns < 0
            or silence_ns > receive_span_ns
        ):
            raise Tier0Error(
                f"{request.session}: paid instrument-silence value is invalid",
                status="STOP_PAID_SESSION_QC",
            )
        seen_silence_ids.append(instrument_id)
        silence_values.append(silence_ns)
    if seen_silence_ids != sorted(seen_silence_ids):
        raise Tier0Error(
            f"{request.session}: paid instrument-silence order drifted",
            status="STOP_PAID_SESSION_QC",
        )
    expected_max_instrument = max(silence_values, default=None)
    if decoder.get("max_instrument_silence_ns") != expected_max_instrument:
        raise Tier0Error(
            f"{request.session}: paid instrument-silence maximum does not reconcile",
            status="STOP_PAID_SESSION_QC",
        )

    def named_sum(field_name: str, expected_names: set[str] | None = None) -> int:
        values = decoder.get(field_name)
        if not isinstance(values, list):
            raise Tier0Error(f"{request.session}: paid decoder {field_name} is invalid", status="STOP_PAID_SESSION_QC")
        names: set[str] = set()
        total = 0
        for item in values:
            if not isinstance(item, dict) or set(item) != {"name", "count"}:
                raise Tier0Error(f"{request.session}: paid decoder named counter drifted", status="STOP_PAID_SESSION_QC")
            name, count = item["name"], item["count"]
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise Tier0Error(f"{request.session}: paid decoder named counter is invalid", status="STOP_PAID_SESSION_QC")
            names.add(name)
            total += count
        if expected_names is not None and names != expected_names:
            raise Tier0Error(f"{request.session}: paid decoder named population drifted", status="STOP_PAID_SESSION_QC")
        return total

    if (
        named_sum("action_counts") != request.expected_record_count
        or named_sum("side_counts") != request.expected_record_count
        or named_sum("flag_value_counts") != request.expected_record_count
        or named_sum("system_code_counts") != int(decoder["system_records"])
    ):
        raise Tier0Error(f"{request.session}: paid decoder named counts do not reconcile", status="STOP_PAID_SESSION_QC")
    named_sum(
        "flag_counts",
        {"LAST", "TOB", "SNAPSHOT", "MBP", "BAD_TS_RECV", "MAYBE_BAD_BOOK", "PUBLISHER_SPECIFIC"},
    )
    if int(decoder["heartbeat_records"]) > int(decoder["system_records"]):
        raise Tier0Error(f"{request.session}: paid heartbeat count is impossible", status="STOP_PAID_SESSION_QC")


def build_paid_session_qc(
    *,
    request: SessionRequest,
    data_path: Path,
    metadata_summary: Mapping[str, Any],
    decoder_summary: Mapping[str, Any],
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    start_record: Mapping[str, Any],
    result_record: Mapping[str, Any],
) -> dict[str, Any]:
    start_payload = start_record.get("payload", {})
    result_payload = result_record.get("payload", {})
    if (
        start_record.get("event") != "TIMESERIES_CALL_START"
        or result_record.get("event") != "TIMESERIES_CALL_RESULT"
        or start_record.get("attempt_id") != result_record.get("attempt_id")
        or start_record.get("session") != request.session
        or result_record.get("session") != request.session
        or start_record.get("request_sha256") != request.market_request_sha256
        or result_record.get("request_sha256") != request.market_request_sha256
        or result_payload.get("timeseries_start_record_hash") != start_record.get("record_hash")
    ):
        raise Tier0Error("paid session QC source pair is invalid", status="STOP_PAID_SESSION_QC")
    receipt: dict[str, Any] = {
        "artifact_type": SESSION_QC_ARTIFACT,
        "schema_version": "v5.job52-cmbp-tier0-paid-session-qc.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "status": "JOB52_CMBP_TIER0_PAID_SESSION_QC_PASS",
        "session": request.session,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "paid_program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "paid_readiness_receipt_sha256": readiness["receipt_sha256"],
        "paid_readiness_receipt_file_sha256": readiness_file_sha256,
        "request": _request_projection(request),
        "cost_commitment": {
            "fresh_observed_sdk_quote_usd": start_payload["fresh_quote_usd"],
            "commitment_index": start_payload["commitment_index"],
            "committed_quote_session_before_usd": start_payload["committed_quote_session_before_usd"],
            "committed_quote_session_after_usd": start_payload["committed_quote_session_after_usd"],
            "committed_quote_total_before_usd": start_payload["committed_quote_total_before_usd"],
            "committed_quote_total_after_usd": start_payload["committed_quote_total_after_usd"],
            "per_session_lifetime_cap_usd": "1.50",
            "total_cap_usd": "32.00",
            "acquisition_initiated": True,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "quote_is_atomic_invoice_lock": False,
        },
        "raw_dbn": {
            "file_name": data_path.name,
            "file_sha256": base.file_sha256(data_path),
            "compressed_bytes": data_path.stat().st_size,
        },
        "dbn_metadata": dict(metadata_summary),
        "decoder": dict(decoder_summary),
        "source_attempt": {
            "attempt_id": start_record["attempt_id"],
            "timeseries_start_sequence": start_record["sequence"],
            "timeseries_start_record_hash": start_record["record_hash"],
            "timeseries_result_sequence": result_record["sequence"],
            "timeseries_result_record_hash": result_record["record_hash"],
        },
        "qc_law": {
            "streaming_no_full_session_pandas": True,
            "causal_clock": "ts_recv",
            "strict_prior_inequality": "prior.ts_recv < trade.ts_recv",
            "receive_time_ties_excluded": True,
            "historical_connection_telemetry": "UNKNOWN_NOT_PRESENT_IN_FILE_TRANSPORT",
        },
    }
    receipt["session_qc_sha256"] = base.self_hash(receipt, "session_qc_sha256")
    return receipt


def validate_paid_session_bundle(
    bundle_dir: Path,
    *,
    request: SessionRequest,
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    volume: VolumeIdentity,
    source_record_lookup: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    base.ensure_nofollow_directory(bundle_dir, volume=volume)
    if sorted(path.name for path in bundle_dir.iterdir()) != [PAID_SESSION_QC_NAME, "data.cmbp-1.dbn.zst"]:
        raise Tier0Error(f"{request.session}: paid bundle file population drifted", status="STOP_PAID_SESSION_QC")
    data_path = bundle_dir / "data.cmbp-1.dbn.zst"
    qc_path = bundle_dir / PAID_SESSION_QC_NAME
    for path in (data_path, qc_path):
        metadata = _private_regular(path, status="STOP_PAID_SESSION_QC")
        if metadata.st_dev != volume.st_dev:
            raise Tier0Error(f"{request.session}: paid bundle file is cross-device", status="STOP_PAID_SESSION_QC")
    receipt = base.strict_json(qc_path)
    expected_fields = {
        "artifact_type",
        "schema_version",
        "job_id",
        "target_job_id",
        "status",
        "session",
        "scope_sha256",
        "scope_file_sha256",
        "paid_program_contract_sha256",
        "paid_program_contract_file_sha256",
        "paid_readiness_receipt_sha256",
        "paid_readiness_receipt_file_sha256",
        "request",
        "cost_commitment",
        "raw_dbn",
        "dbn_metadata",
        "decoder",
        "source_attempt",
        "qc_law",
        "session_qc_sha256",
    }
    expected_scalars = {
        "artifact_type": SESSION_QC_ARTIFACT,
        "schema_version": "v5.job52-cmbp-tier0-paid-session-qc.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "status": "JOB52_CMBP_TIER0_PAID_SESSION_QC_PASS",
        "session": request.session,
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "paid_program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "paid_readiness_receipt_sha256": readiness["receipt_sha256"],
        "paid_readiness_receipt_file_sha256": readiness_file_sha256,
    }
    if set(receipt) != expected_fields or any(receipt.get(key) != value for key, value in expected_scalars.items()):
        raise Tier0Error(f"{request.session}: paid session QC identity drifted", status="STOP_PAID_SESSION_QC")
    if receipt.get("session_qc_sha256") != base.self_hash(receipt, "session_qc_sha256"):
        raise Tier0Error(f"{request.session}: paid session QC self-hash mismatch", status="STOP_PAID_SESSION_QC")
    if receipt.get("request") != _request_projection(request):
        raise Tier0Error(f"{request.session}: paid session request projection drifted", status="STOP_PAID_SESSION_QC")
    expected_raw = {
        "file_name": data_path.name,
        "file_sha256": base.file_sha256(data_path),
        "compressed_bytes": data_path.stat().st_size,
    }
    if receipt.get("raw_dbn") != expected_raw:
        raise Tier0Error(f"{request.session}: paid raw DBN binding drifted", status="STOP_PAID_SESSION_QC")
    cost = receipt.get("cost_commitment")
    expected_cost_fields = {
        "fresh_observed_sdk_quote_usd",
        "commitment_index",
        "committed_quote_session_before_usd",
        "committed_quote_session_after_usd",
        "committed_quote_total_before_usd",
        "committed_quote_total_after_usd",
        "per_session_lifetime_cap_usd",
        "total_cap_usd",
        "acquisition_initiated",
        "actual_vendor_invoice_cost_usd",
        "quote_is_atomic_invoice_lock",
    }
    if not isinstance(cost, dict) or set(cost) != expected_cost_fields:
        raise Tier0Error(f"{request.session}: paid cost commitment fields drifted", status="STOP_PAID_SESSION_QC")
    try:
        quote = Decimal(str(cost["fresh_observed_sdk_quote_usd"]))
        session_before = Decimal(str(cost["committed_quote_session_before_usd"]))
        session_after = Decimal(str(cost["committed_quote_session_after_usd"]))
        total_before = Decimal(str(cost["committed_quote_total_before_usd"]))
        total_after = Decimal(str(cost["committed_quote_total_after_usd"]))
    except Exception as exc:  # noqa: BLE001
        raise Tier0Error(f"{request.session}: paid cost decimal is invalid", status="STOP_PAID_SESSION_QC") from exc
    if (
        normalize_paid_quote(quote) != cost["fresh_observed_sdk_quote_usd"]
        or _exact_nonnegative_add(session_before, quote) != session_after
        or _exact_nonnegative_add(total_before, quote) != total_after
        or session_after > PER_SESSION_LIFETIME_CAP_USD
        or total_after > TOTAL_COMMITTED_CAP_USD
        or isinstance(cost.get("commitment_index"), bool)
        or not isinstance(cost.get("commitment_index"), int)
        or cost["commitment_index"] < 1
        or cost.get("per_session_lifetime_cap_usd") != "1.50"
        or cost.get("total_cap_usd") != "32.00"
        or cost.get("acquisition_initiated") is not True
        or cost.get("actual_vendor_invoice_cost_usd") != "UNKNOWN"
        or cost.get("quote_is_atomic_invoice_lock") is not False
    ):
        raise Tier0Error(f"{request.session}: paid cost commitment arithmetic/law drifted", status="STOP_PAID_SESSION_QC")
    expected_qc_law = {
        "streaming_no_full_session_pandas": True,
        "causal_clock": "ts_recv",
        "strict_prior_inequality": "prior.ts_recv < trade.ts_recv",
        "receive_time_ties_excluded": True,
        "historical_connection_telemetry": "UNKNOWN_NOT_PRESENT_IN_FILE_TRANSPORT",
    }
    if receipt.get("qc_law") != expected_qc_law:
        raise Tier0Error(f"{request.session}: paid QC law drifted", status="STOP_PAID_SESSION_QC")
    decoder = receipt.get("decoder")
    if not isinstance(decoder, dict):
        raise Tier0Error(f"{request.session}: paid decoder summary is absent", status="STOP_PAID_SESSION_QC")
    _validate_decoder_summary(decoder, request)
    source = receipt.get("source_attempt")
    source_fields = {
        "attempt_id",
        "timeseries_start_sequence",
        "timeseries_start_record_hash",
        "timeseries_result_sequence",
        "timeseries_result_record_hash",
    }
    if (
        not isinstance(source, dict)
        or set(source) != source_fields
        or base.UUID4_RE.fullmatch(str(source.get("attempt_id"))) is None
        or any(
            isinstance(source.get(field_name), bool)
            or not isinstance(source.get(field_name), int)
            or source[field_name] < 0
            for field_name in ("timeseries_start_sequence", "timeseries_result_sequence")
        )
        or SHA256_RE.fullmatch(str(source.get("timeseries_start_record_hash"))) is None
        or SHA256_RE.fullmatch(str(source.get("timeseries_result_record_hash"))) is None
    ):
        raise Tier0Error(f"{request.session}: paid source locator is invalid", status="STOP_PAID_SESSION_QC")
    if source_record_lookup is not None:
        start = source_record_lookup.get((source["attempt_id"], source["timeseries_start_sequence"]))
        result = source_record_lookup.get((source["attempt_id"], source["timeseries_result_sequence"]))
        if (
            start is None
            or result is None
            or start.get("record_hash") != source["timeseries_start_record_hash"]
            or result.get("record_hash") != source["timeseries_result_record_hash"]
            or start.get("event") != "TIMESERIES_CALL_START"
            or result.get("event") != "TIMESERIES_CALL_RESULT"
            or start.get("session") != request.session
            or result.get("session") != request.session
            or start.get("request_sha256") != request.market_request_sha256
            or result.get("request_sha256") != request.market_request_sha256
            or result.get("payload", {}).get("timeseries_start_record_hash") != start.get("record_hash")
            or result.get("payload", {}).get("dbn_file_sha256") != expected_raw["file_sha256"]
            or result.get("payload", {}).get("compressed_bytes") != expected_raw["compressed_bytes"]
        ):
            raise Tier0Error(f"{request.session}: paid source journal binding drifted", status="STOP_PAID_SESSION_QC")
        reconstructed = build_paid_session_qc(
            request=request,
            data_path=data_path,
            metadata_summary=receipt["dbn_metadata"],
            decoder_summary=decoder,
            readiness=readiness,
            readiness_file_sha256=readiness_file_sha256,
            start_record=start,
            result_record=result,
        )
        if receipt != reconstructed:
            raise Tier0Error(f"{request.session}: paid session QC does not reconstruct", status="STOP_PAID_SESSION_QC")
    import databento

    store = databento.DBNStore.from_file(data_path)
    if base.validate_dbn_metadata(store, request) != receipt.get("dbn_metadata"):
        raise Tier0Error(f"{request.session}: paid DBN metadata drifted", status="STOP_PAID_SESSION_QC")
    return receipt


def validate_paid_existing_session_population(
    job_root: Path,
    *,
    bundle: PaidScopeBundle,
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    volume: VolumeIdentity,
    source_record_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    published_sessions_in_journals: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    sessions_root = Path(job_root) / "sessions"
    base.ensure_nofollow_directory(sessions_root, volume=volume)
    expected = {request.session: request for request in bundle.sessions}
    result: dict[str, dict[str, Any]] = {}
    for child in sorted(sessions_root.iterdir()):
        if child.is_symlink() or not child.is_dir() or child.name not in expected:
            raise Tier0Error("paid final session population contains an unexpected entry", status="STOP_PAID_SESSION_QC")
        result[child.name] = validate_paid_session_bundle(
            child,
            request=expected[child.name],
            readiness=readiness,
            readiness_file_sha256=readiness_file_sha256,
            volume=volume,
            source_record_lookup=source_record_lookup,
        )
    missing_published = set(published_sessions_in_journals) - set(result)
    if missing_published:
        raise Tier0Error("a durably published paid session bundle disappeared", status="STOP_PAID_SESSION_QC")
    return result


def build_paid_aggregate_receipt(
    job_root: Path,
    *,
    bundle: PaidScopeBundle,
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    volume: VolumeIdentity,
) -> dict[str, Any]:
    job_root = Path(job_root)
    validate_paid_job_tree(
        job_root,
        volume=volume,
        allowed_requests=bundle.sessions,
        allow_missing_paid_controls=False,
    )
    anchor = validate_paid_attempt_anchor(
        job_root,
        volume=volume,
        readiness_receipt_sha256=str(readiness["receipt_sha256"]),
        readiness_receipt_file_sha256=readiness_file_sha256,
    )
    summary = summarize_paid_attempts(
        job_root,
        bundle.sessions,
        require_legacy_stop=True,
        expected_readiness_sha256=str(readiness["receipt_sha256"]),
        expected_readiness_file_sha256=readiness_file_sha256,
        require_client_constructed=True,
        expected_sdk_identity_sha256=base.json_sha256(readiness["sdk_identity"]),
    )
    if summary["terminal_authority_failure_observed"]:
        raise Tier0Error("a terminal paid quote failure exists", status="STOP_PAID_AUTHORITY_TERMINAL")
    if Decimal(summary["committed_quote_total_usd"]) > TOTAL_COMMITTED_CAP_USD:
        raise Tier0Error("paid aggregate total cap was exceeded", status="STOP_PAID_TOTAL_CAP")
    if any(Decimal(value) > PER_SESSION_LIFETIME_CAP_USD for value in summary["committed_quote_by_session_usd"].values()):
        raise Tier0Error("paid aggregate session cap was exceeded", status="STOP_PAID_SESSION_CAP")
    if summary["cost_call_starts"] != (
        summary["cost_call_results"] + summary["cost_call_errors"] + summary["pending_cost_call_starts"]
    ):
        raise Tier0Error("paid aggregate cost-call accounting does not reconcile", status="STOP_PAID_AGGREGATE_QC")
    if summary["timeseries_call_starts"] != (
        summary["timeseries_call_results"]
        + summary["failed_timeseries_starts"]
        + summary["pending_timeseries_starts"]
    ):
        raise Tier0Error("paid aggregate time-series accounting does not reconcile", status="STOP_PAID_AGGREGATE_QC")
    if summary["aggregate_seal_count"] != 1:
        raise Tier0Error("paid aggregate lacks a durable attempt seal", status="STOP_PAID_AGGREGATE_QC")
    source_records = summary["timeseries_start_records"] + summary["timeseries_result_records"]
    source_lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in source_records:
        key = (str(record["attempt_id"]), int(record["sequence"]))
        if key in source_lookup:
            raise Tier0Error("paid source record locator is duplicated", status="STOP_PAID_AGGREGATE_QC")
        source_lookup[key] = record
    existing = validate_paid_existing_session_population(
        job_root,
        bundle=bundle,
        readiness=readiness,
        readiness_file_sha256=readiness_file_sha256,
        volume=volume,
        source_record_lookup=source_lookup,
        published_sessions_in_journals=summary["published_sessions_in_journals"],
    )
    expected_names = [request.session for request in bundle.sessions]
    if sorted(existing) != expected_names or summary["published_sessions_in_journals"] != expected_names:
        raise Tier0Error("paid aggregate final/publication population is incomplete", status="STOP_PAID_AGGREGATE_QC")
    publications_by_session: dict[str, list[Mapping[str, Any]]] = {}
    for record in summary["publication_records"]:
        if record.get("attempt_id") == summary["aggregate_seal_attempt_id"]:
            publications_by_session.setdefault(str(record["session"]), []).append(record)

    decoder_fields = (
        "trade_records",
        "strict_prior_trades",
        "tied_prior_trades_excluded",
        "no_prior_trades",
        "signed_trades",
        "at_bid_trades",
        "at_ask_trades",
        "inside_trades",
        "outside_trades",
        "ambiguous_trades",
        "global_receive_ties",
        "global_receive_regressions",
        "global_event_regressions",
        "instrument_event_regressions",
        "disconnect_count",
        "reconnect_count",
        "gap_count",
        "book_state_clear_count",
        "gaps_with_known_bounds",
        "total_known_gap_ns",
        "system_records",
        "heartbeat_records",
    )
    decoder_totals = {field_name: 0 for field_name in decoder_fields}
    max_stream_silence_ns: int | None = None
    max_instrument_silence_ns: int | None = None
    files: list[dict[str, Any]] = []
    total_records = total_symbols = total_bytes = 0
    source_attempt_ids: set[str] = set()
    for request in bundle.sessions:
        qc = existing[request.session]
        publications = publications_by_session.get(request.session, [])
        matching_publications = [
            record
            for record in publications
            if record.get("payload", {}).get("session_qc_sha256") == qc["session_qc_sha256"]
            and record.get("payload", {}).get("compressed_bytes") == qc["raw_dbn"]["compressed_bytes"]
            and record.get("payload", {}).get("source_attempt_id")
            == qc["source_attempt"]["attempt_id"]
            and record.get("payload", {}).get("source_timeseries_result_sequence")
            == qc["source_attempt"]["timeseries_result_sequence"]
            and record.get("payload", {}).get("source_timeseries_result_record_hash")
            == qc["source_attempt"]["timeseries_result_record_hash"]
        ]
        if not matching_publications:
            raise Tier0Error(f"{request.session}: paid final lacks a matching publication record", status="STOP_PAID_AGGREGATE_QC")
        total_records += request.expected_record_count
        total_symbols += len(request.symbols)
        total_bytes += int(qc["raw_dbn"]["compressed_bytes"])
        source_attempt_ids.add(str(qc["source_attempt"]["attempt_id"]))
        decoder = qc["decoder"]
        for field_name in decoder_fields:
            decoder_totals[field_name] += int(decoder[field_name])
        stream_silence = decoder.get("max_stream_silence_ns")
        if isinstance(stream_silence, int) and not isinstance(stream_silence, bool):
            max_stream_silence_ns = stream_silence if max_stream_silence_ns is None else max(max_stream_silence_ns, stream_silence)
        instrument_silence = decoder.get("max_instrument_silence_ns")
        if isinstance(instrument_silence, int) and not isinstance(instrument_silence, bool):
            max_instrument_silence_ns = (
                instrument_silence
                if max_instrument_silence_ns is None
                else max(max_instrument_silence_ns, instrument_silence)
            )
        bundle_dir = job_root / "sessions" / request.session
        files.append(
            {
                "session": request.session,
                "relative_bundle": f"sessions/{request.session}",
                "dbn_file_sha256": qc["raw_dbn"]["file_sha256"],
                "compressed_bytes": qc["raw_dbn"]["compressed_bytes"],
                "session_qc_sha256": qc["session_qc_sha256"],
                "session_qc_file_sha256": base.file_sha256(bundle_dir / PAID_SESSION_QC_NAME),
                "decoded_records": request.expected_record_count,
                "symbol_count": len(request.symbols),
                "source_attempt_id": qc["source_attempt"]["attempt_id"],
                "source_commitment_index": qc["cost_commitment"]["commitment_index"],
                "fresh_observed_sdk_quote_usd": qc["cost_commitment"]["fresh_observed_sdk_quote_usd"],
                "committed_quote_session_after_usd": qc["cost_commitment"]["committed_quote_session_after_usd"],
                "committed_quote_total_after_usd": qc["cost_commitment"]["committed_quote_total_after_usd"],
                "decoder_summary_sha256": base.json_sha256(decoder),
                "trade_records": decoder["trade_records"],
                "strict_prior_trades": decoder["strict_prior_trades"],
                "tied_prior_trades_excluded": decoder["tied_prior_trades_excluded"],
                "signed_trades": decoder["signed_trades"],
                "global_receive_regressions": decoder["global_receive_regressions"],
                "max_stream_silence_ns": decoder["max_stream_silence_ns"],
            }
        )
    if total_records != base.EXPECTED_RECORD_COUNT or total_symbols != base.EXPECTED_SESSION_SYMBOLS:
        raise Tier0Error("paid aggregate census totals do not reconcile", status="STOP_PAID_AGGREGATE_QC")
    known_attempt_ids = {item["attempt_id"] for item in summary["attempts"]}
    if not source_attempt_ids <= known_attempt_ids:
        raise Tier0Error("paid final cites an absent source attempt", status="STOP_PAID_AGGREGATE_QC")
    anchor_path = job_root / PAID_ATTEMPT_ANCHOR_NAME
    lock_binding_path = job_root / PAID_LOCK_BINDING_NAME
    receipt: dict[str, Any] = {
        "artifact_type": AGGREGATE_RECEIPT_ARTIFACT,
        "schema_version": "v5.job52-cmbp-tier0-paid-acquisition-qc-receipt.v1",
        "job_id": JOB_ID,
        "target_job_id": TARGET_JOB_ID,
        "status": "JOB52_CMBP_TIER0_PAID_ACQUISITION_AND_QC_PASS",
        "status_meaning": "EXACT_FROZEN_RAW_ACQUISITION_AND_STREAMING_QC_ONLY; NO_STRATEGY_OR_TRADING_CLAIM",
        "scope_sha256": base.EXPECTED_SCOPE_SHA256,
        "scope_file_sha256": base.EXPECTED_SCOPE_FILE_SHA256,
        "paid_program_contract_sha256": EXPECTED_PROGRAM_CONTRACT_SHA256,
        "paid_program_contract_file_sha256": EXPECTED_PROGRAM_CONTRACT_FILE_SHA256,
        "paid_readiness_receipt_sha256": readiness["receipt_sha256"],
        "paid_readiness_receipt_file_sha256": readiness_file_sha256,
        "job51_zero_gate_stop_receipt_sha256": EXPECTED_LEGACY_STOP_RECEIPT_SHA256,
        "job51_zero_gate_stop_receipt_file_sha256": EXPECTED_LEGACY_STOP_RECEIPT_FILE_SHA256,
        "volume_identity": asdict(volume),
        "completeness": {
            "expected_sessions": base.EXPECTED_SESSION_COUNT,
            "published_sessions": len(files),
            "expected_records": base.EXPECTED_RECORD_COUNT,
            "decoded_records": total_records,
            "expected_session_symbol_memberships": base.EXPECTED_SESSION_SYMBOLS,
            "mapped_session_symbol_memberships": total_symbols,
            "compressed_bytes": total_bytes,
            "per_session_census_reconciliation": True,
        },
        "cost_boundary": {
            "money_measure": "COMMITTED_QUOTED_EXPOSURE_NOT_SPEND_OR_INVOICE",
            "per_session_lifetime_cap_usd": "1.50",
            "total_committed_quote_cap_usd": "32.00",
            "maximum_scope_reachable_commitment_usd": "31.50",
            "committed_quote_total_usd": summary["committed_quote_total_usd"],
            "committed_quote_by_session_usd": summary["committed_quote_by_session_usd"],
            "commitments": summary["commitments"],
            "all_quote_observations": summary["all_quote_observations"],
            "prior_job51_observed_quote_usd": "0.950392448902",
            "prior_job51_committed_quote_usd": "0",
            "terminal_authority_failure_observed": False,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "quote_is_atomic_invoice_lock": False,
            "acquisition_initiated": True,
        },
        "call_accounting": {
            "paid_attempt_count": summary["paid_attempt_count"],
            "paid_cost_call_starts": summary["cost_call_starts"],
            "paid_cost_call_results": summary["cost_call_results"],
            "paid_cost_call_errors": summary["cost_call_errors"],
            "paid_pending_cost_calls": summary["pending_cost_call_starts"],
            "paid_timeseries_call_starts": summary["timeseries_call_starts"],
            "paid_timeseries_call_results": summary["timeseries_call_results"],
            "paid_failed_timeseries_starts": summary["failed_timeseries_starts"],
            "paid_pending_timeseries_starts": summary["pending_timeseries_starts"],
            "session_timeseries_attempt_counts": summary["session_timeseries_attempt_counts"],
            "duplicate_session_attempts": summary["duplicate_session_attempts"],
            "uncommitted_passing_quote_count": summary["uncommitted_passing_quote_count"],
            "out_of_scope_calls": 0,
        },
        "causal_ordering": {
            "clock": "ts_recv",
            "strict_prior_inequality": "prior.ts_recv < trade.ts_recv",
            "receive_time_ties_excluded": True,
            "trade_records": decoder_totals["trade_records"],
            "strict_prior_trades": decoder_totals["strict_prior_trades"],
            "tied_prior_trades_excluded": decoder_totals["tied_prior_trades_excluded"],
            "no_prior_trades": decoder_totals["no_prior_trades"],
            "signed_trades": decoder_totals["signed_trades"],
            "at_bid_trades": decoder_totals["at_bid_trades"],
            "at_ask_trades": decoder_totals["at_ask_trades"],
            "inside_trades": decoder_totals["inside_trades"],
            "outside_trades": decoder_totals["outside_trades"],
            "ambiguous_trades": decoder_totals["ambiguous_trades"],
            "global_receive_ties": decoder_totals["global_receive_ties"],
            "global_receive_regressions": decoder_totals["global_receive_regressions"],
            "global_event_regressions_diagnostic": decoder_totals["global_event_regressions"],
            "instrument_event_regressions_diagnostic": decoder_totals["instrument_event_regressions"],
        },
        "gaps_and_reconnects": {
            "historical_transport_telemetry": "UNKNOWN_NOT_PRESENT_IN_DBN_FILES",
            "explicit_decoder_gap_reconnect_controls": True,
            "live_connection_exercised": False,
            "disconnect_count": decoder_totals["disconnect_count"],
            "reconnect_count": decoder_totals["reconnect_count"],
            "gap_count": decoder_totals["gap_count"],
            "book_state_clear_count": decoder_totals["book_state_clear_count"],
            "gaps_with_known_bounds": decoder_totals["gaps_with_known_bounds"],
            "total_known_gap_ns": decoder_totals["total_known_gap_ns"],
            "max_stream_silence_ns": max_stream_silence_ns,
            "max_instrument_silence_ns": max_instrument_silence_ns,
            "system_records": decoder_totals["system_records"],
            "heartbeat_records": decoder_totals["heartbeat_records"],
        },
        "attempt_evidence": {
            "legacy_attempt": summary["legacy_attempt"],
            "paid_attempts": summary["attempts"],
            "attempt_set_anchor_sha256": anchor["anchor_sha256"],
            "attempt_set_anchor_file_sha256": base.file_sha256(anchor_path),
            "paid_lock_binding_file_sha256": base.file_sha256(lock_binding_path),
            "threat_boundary": anchor["threat_boundary"],
        },
        "files": files,
    }
    receipt["receipt_sha256"] = base.self_hash(receipt, "receipt_sha256")
    return receipt


def validate_paid_aggregate_receipt(
    job_root: Path,
    *,
    bundle: PaidScopeBundle,
    readiness: Mapping[str, Any],
    readiness_file_sha256: str,
    volume: VolumeIdentity,
) -> dict[str, Any]:
    """Validate an existing terminal receipt by exact local reconstruction."""

    path = Path(job_root) / "receipts" / PAID_AGGREGATE_NAME
    metadata = _private_regular(path, status="STOP_PAID_ALREADY_TERMINAL")
    if metadata.st_dev != volume.st_dev:
        raise Tier0Error("paid aggregate receipt is cross-device", status="STOP_PAID_ALREADY_TERMINAL")
    receipt = base.strict_json(path)
    if (
        receipt.get("artifact_type") != AGGREGATE_RECEIPT_ARTIFACT
        or receipt.get("schema_version") != "v5.job52-cmbp-tier0-paid-acquisition-qc-receipt.v1"
        or receipt.get("receipt_sha256") != base.self_hash(receipt, "receipt_sha256")
    ):
        raise Tier0Error("paid aggregate receipt identity drifted", status="STOP_PAID_ALREADY_TERMINAL")
    recorded_volume_payload = receipt.get("volume_identity")
    expected_volume_fields = {
        "mount_point",
        "volume_uuid",
        "device_identifier",
        "filesystem",
        "bus_protocol",
        "st_dev",
        "free_bytes",
        "total_bytes",
    }
    if not isinstance(recorded_volume_payload, dict) or set(recorded_volume_payload) != expected_volume_fields:
        raise Tier0Error("paid aggregate recorded volume is malformed", status="STOP_PAID_ALREADY_TERMINAL")
    string_volume_fields = (
        "mount_point",
        "volume_uuid",
        "device_identifier",
        "filesystem",
        "bus_protocol",
    )
    integer_volume_fields = ("st_dev", "free_bytes", "total_bytes")
    if (
        any(not isinstance(recorded_volume_payload.get(name), str) or not recorded_volume_payload[name] for name in string_volume_fields)
        or any(
            isinstance(recorded_volume_payload.get(name), bool)
            or not isinstance(recorded_volume_payload.get(name), int)
            or recorded_volume_payload[name] < 0
            for name in integer_volume_fields
        )
        or any(not isinstance(getattr(volume, name), str) or not getattr(volume, name) for name in string_volume_fields)
        or any(
            isinstance(getattr(volume, name), bool)
            or not isinstance(getattr(volume, name), int)
            or getattr(volume, name) < 0
            for name in integer_volume_fields
        )
    ):
        raise Tier0Error("paid aggregate volume field types are invalid", status="STOP_PAID_ALREADY_TERMINAL")
    try:
        recorded_volume = VolumeIdentity(**recorded_volume_payload)
    except Exception as exc:  # noqa: BLE001
        raise Tier0Error("paid aggregate recorded volume is invalid", status="STOP_PAID_ALREADY_TERMINAL") from exc
    for field_name in (
        "mount_point",
        "volume_uuid",
        "device_identifier",
        "filesystem",
        "bus_protocol",
        "st_dev",
        "total_bytes",
    ):
        if getattr(recorded_volume, field_name) != getattr(volume, field_name):
            raise Tier0Error("paid aggregate volume identity changed", status="STOP_PAID_ALREADY_TERMINAL")
    reconstructed = build_paid_aggregate_receipt(
        job_root,
        bundle=bundle,
        readiness=readiness,
        readiness_file_sha256=readiness_file_sha256,
        volume=recorded_volume,
    )
    if receipt != reconstructed:
        raise Tier0Error("paid aggregate receipt does not reconstruct", status="STOP_PAID_ALREADY_TERMINAL")
    return receipt
