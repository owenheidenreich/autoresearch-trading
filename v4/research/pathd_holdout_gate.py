"""Atomic, one-shot protected-holdout transaction for Path-D research.

This module is intentionally isolated from model fitting and evaluation code.  It
owns the irreversible access receipt, holds a non-blocking filesystem lease for
the lifetime of the one authorized evaluation, and never offers a reset, repair,
path-override, or reopen operation.

The protected dataset/evaluator implementations are imported only at the instant
they are needed.  Consequently this module remains importable before those future
research modules exist and fails closed if they have not been installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Callable, Mapping, TYPE_CHECKING
import uuid

from v4.research import pathd_entry_exit as _foundation

if TYPE_CHECKING:  # pragma: no cover - the future module is deliberately optional.
    from v4.research.pathd_entry_dataset import EntryEvidenceDatasetV1
    from v4.scripts.run_pathd_entry_exit_research import (
        ProtectedHoldoutEvaluationV1,
    )


AUDIT_ROOT = _foundation.AUDIT_ROOT
PRE_HOLDOUT_PACKET_PATH = _foundation.PRE_HOLDOUT_PACKET_PATH
PROTECTED_HOLDOUT_ROOT = _foundation.PROTECTED_HOLDOUT_ROOT
HOLDOUT_LOCK_PATH = _foundation.HOLDOUT_LOCK_PATH
HOLDOUT_ACCESS_RECEIPT_PATH = _foundation.HOLDOUT_ACCESS_RECEIPT_PATH
HOLDOUT_RESULT_PATH = _foundation.HOLDOUT_RESULT_PATH
HOLDOUT_SEAL_RECEIPT_PATH = _foundation.HOLDOUT_SEAL_RECEIPT_PATH
HOLDOUT_ABORT_RECEIPT_PATH = _foundation.HOLDOUT_ABORT_RECEIPT_PATH

PREREGISTRATION_PATH = _foundation.PREREG_PATH
SESSION_ASSIGNMENTS_PATH = _foundation.SESSION_PATH
CORPUS_INTEGRITY_RECEIPT_PATH = _foundation.CORPUS_INTEGRITY_RECEIPT_PATH
LINEAGE_RECEIPT_PATH = _foundation.LINEAGE_IMPLEMENTATION_RECEIPT_PATH
MACHINERY_RECEIPT_PATH = _foundation.ENTRY_MACHINERY_RECEIPT_PATH
OUTER_RESULT_RECEIPT_PATHS = tuple(
    _foundation.ENTRY_FOLD_ARTIFACT_ROOT
    / f"fold_{outer_fold}"
    / "outer_result_receipt.json"
    for outer_fold in range(1, 6)
)
OUTER_ENTRY_EXIT_ARTIFACTS_PATH = AUDIT_ROOT / "outer_entry_exit_artifacts.json"
FOUR_BOX_PACKET_PATH = AUDIT_ROOT / "four_box_packet.json"
GUARD_PANEL_PATH = AUDIT_ROOT / "guard_panel.json"
OUTER_ACCEPTANCE_PACKET_PATH = AUDIT_ROOT / "outer_acceptance_packet.json"
FULL_FIT_ENTRY_ARTIFACTS_PATH = AUDIT_ROOT / "full_fit_entry_artifacts.json"
FULL_FIT_EXIT_ARTIFACTS_PATH = AUDIT_ROOT / "full_fit_exit_artifacts.json"
HOLDOUT_TRACE_PATH = _foundation.HOLDOUT_TRACE_PATH
HOLDOUT_EVALUATOR_RECEIPT_PATH = _foundation.HOLDOUT_EVALUATOR_RECEIPT_PATH

PACKET_SCHEMA = "pathd.complete_pre_holdout_packet.v1"
ACCESS_SCHEMA = "pathd.protected_holdout_access.v1"
RESULT_SCHEMA = "pathd.protected_holdout_result.v1"
SEAL_SCHEMA = "pathd.protected_holdout_seal.v1"
ABORT_SCHEMA = "pathd.protected_holdout_abort.v1"
EVALUATION_SCHEMA = "pathd.protected_holdout_evaluation.v1"
EVALUATOR_RECEIPT_SCHEMA = "pathd.protected_holdout_evaluator_receipt.v1"
EVALUATOR_SOURCE_PATH = (
    _foundation.REPO_ROOT / "v4/scripts/run_pathd_entry_exit_research.py"
)

HOLDOUT_GUARD_KEYS = (
    "action_conditioned_calibration",
    "fee4_survival",
    "floor_ablation",
    "latency_survival",
    "minimum_power",
    "protected_access_count",
)
HOLDOUT_SURVIVAL_VIOLATION_KEYS = (
    "d48_breach",
    "d49_breach",
    "duplicate_terminal_consumption",
    "nonflat_close",
    "overlap",
    "quantity_breach",
    "unaffordable_fill",
)
OWNER_FACING_METRIC_NAMES = (
    "action_conditioned_calibration",
    "daily_positive_fraction",
    "distinct_trajectory_count",
    "entry_and_exit_churn",
    "entry_and_exit_no_fill_rate",
    "equity_curve",
    "exposure_seconds",
    "four_bucket_pickles_counts_shares_and_pnl",
    "guard_panel",
    "maximum_drawdown",
    "mean_session_pnl",
    "median_session_pnl",
    "minimum_power_ledger",
    "moneyness_concentration",
    "occupancy_fraction",
    "one_second_row_count",
    "paired_session_pnl",
    "premium_band_concentration",
    "profit_factor",
    "regime_concentration",
    "side_concentration",
    "skipped_opportunity_count_and_value",
    "time_bucket_concentration",
    "total_pnl",
    "unique_session_count",
    "worst_session_pnl",
)

UNOPENED = "UNOPENED"
OPENING_ACTIVE = "OPENING_ACTIVE"
BURNED_INCOMPLETE = "BURNED_INCOMPLETE"
RESULT_DURABLE_PENDING_ABORT = "RESULT_DURABLE_PENDING_ABORT"
SEALED = "SEALED"
ABORTED = "ABORTED"
CORRUPT_BURNED = "CORRUPT_BURNED"

ABORT_REASON_CODES = frozenset(
    {
        "OPERATOR_ABORT",
        "DATASET_DECODE_FAILED",
        "EVALUATION_VALIDATION_FAILED",
        "RESULT_COMMIT_FAILED",
        "RESULT_SEAL_FAILED",
        "CRASH_RECOVERY_INCOMPLETE",
        "CORRUPT_BURNED",
    }
)

_HEX64 = re.compile(r"[0-9a-f]{64}")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T[^\s]+Z")
_SESSION = re.compile(r"\d{4}-\d{2}-\d{2}")
_TRANSACTION_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)

_EXPECTED_ROOT_NAMES = frozenset(
    {
        HOLDOUT_LOCK_PATH.name,
        HOLDOUT_ACCESS_RECEIPT_PATH.name,
        HOLDOUT_RESULT_PATH.name,
        HOLDOUT_SEAL_RECEIPT_PATH.name,
        HOLDOUT_ABORT_RECEIPT_PATH.name,
        HOLDOUT_TRACE_PATH.name,
        HOLDOUT_EVALUATOR_RECEIPT_PATH.name,
    }
)

PRE_HOLDOUT_PREREQUISITE_CHECK_NAMES = (
    "all_five_outer_results_semantically_valid",
    "outer_entry_exit_artifacts_semantically_valid",
    "four_box_packet_semantically_valid",
    "guard_panel_all_pass",
    "outer_acceptance_all_pass",
    "full_fit_entry_artifacts_semantically_valid",
    "full_fit_exit_artifacts_semantically_valid",
    "current_source_receipts_and_dependency_closure_valid",
    "protected_holdout_access_count_zero",
)


class ProtectedHoldoutError(RuntimeError):
    """Fail-closed protected-holdout transaction error."""


class _LeaseBusy(ProtectedHoldoutError):
    pass


@dataclass(frozen=True)
class ProtectedHoldoutStateV1:
    SCHEMA_VERSION = "pathd.protected_holdout_state.v1"

    schema_version: str
    state: str
    holdout_open_count: int
    transaction_id: str | None
    detail: str
    access_receipt_sha256: str | None
    result_sha256: str | None
    terminal_receipt_sha256: str | None


_AUTH_CONSTRUCTOR_TOKEN = object()


class ActiveProtectedHoldoutAuthorizationV1:
    """Opaque process-local capability that owns the live filesystem lease."""

    __slots__ = (
        "_transaction_token_sha256",
        "_lock_fd",
        "_originating_pid",
        "_terminal",
        "_decode_attempted",
        "_dataset",
        "_dataset_sha256",
        "_source_receipts_root_sha256",
        "_semantic",
    )

    def __init__(
        self,
        constructor_token: object,
        *,
        transaction_token_sha256: str,
        lock_fd: int,
        semantic: Mapping[str, Any],
    ) -> None:
        if constructor_token is not _AUTH_CONSTRUCTOR_TOKEN:
            raise TypeError("protected holdout authorizations have a private constructor")
        self._transaction_token_sha256 = transaction_token_sha256
        self._lock_fd = lock_fd
        self._originating_pid = os.getpid()
        self._terminal = False
        self._decode_attempted = False
        self._dataset = None
        self._dataset_sha256 = None
        self._source_receipts_root_sha256 = None
        self._semantic = MappingProxyType(dict(semantic))

    def __reduce__(self) -> Any:
        raise TypeError("protected holdout authorization is process-local")

    def __getstate__(self) -> Any:
        raise TypeError("protected holdout authorization is non-serializable")

    def __repr__(self) -> str:
        return (
            "ActiveProtectedHoldoutAuthorizationV1("
            f"transaction_id={self.transaction_id!r}, active={not self._terminal})"
        )

    @property
    def transaction_id(self) -> str:
        return str(self._semantic["transaction_id"])

    @property
    def holdout_open_count(self) -> int:
        return 1

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(self._semantic["sessions"])

    @property
    def sessions_sha256_newline(self) -> str:
        return str(self._semantic["sessions_sha256_newline"])

    @property
    def primary_sessions_sha256_newline(self) -> str:
        return str(self._semantic["primary_sessions_sha256_newline"])

    @property
    def preregistration_sha256(self) -> str:
        return str(self._semantic["preregistration_sha256"])

    @property
    def session_assignments_sha256(self) -> str:
        return str(self._semantic["session_assignments_sha256"])

    @property
    def source_hash_policy_sha256(self) -> str:
        return str(self._semantic["source_hash_policy_sha256"])

    @property
    def corpus_integrity_receipt_sha256(self) -> str:
        return str(self._semantic["corpus_integrity_receipt_sha256"])

    @property
    def lineage_receipt_sha256(self) -> str:
        return str(self._semantic["lineage_receipt_sha256"])

    @property
    def machinery_receipt_sha256(self) -> str:
        return str(self._semantic["machinery_receipt_sha256"])

    @property
    def fit_environment_sha256(self) -> str:
        return str(self._semantic["fit_environment_sha256"])

    @property
    def preopen_packet_sha256(self) -> str:
        return str(self._semantic["preopen_packet_sha256"])

    @property
    def artifact_root_sha256(self) -> str:
        return str(self._semantic["artifact_root_sha256"])

    @property
    def box_d_policy_id(self) -> str:
        return str(self._semantic["box_d_policy_id"])

    @property
    def comparator_policy_id(self) -> str:
        return str(self._semantic["comparator_policy_id"])

    @property
    def access_receipt_sha256(self) -> str:
        return str(self._semantic["access_receipt_sha256"])

    @property
    def authorization_sha256(self) -> str:
        return str(self._semantic["authorization_sha256"])


def _strict_json_value(value: Any, *, path: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtectedHoldoutError(f"nonfinite JSON number at {path}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _strict_json_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ProtectedHoldoutError(f"non-string JSON key at {path}")
            _strict_json_value(item, path=f"{path}.{key}")
        return
    raise ProtectedHoldoutError(
        f"non-JSON value at {path}: {type(value).__name__}"
    )


def _canonical_json_bytes(payload: Any) -> bytes:
    _strict_json_value(payload)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtectedHoldoutError("payload is not strict JSON") from exc
    # Parse once before touching the filesystem.  This also rejects encoding bugs.
    if json.loads(encoded) != payload:
        raise ProtectedHoldoutError("canonical JSON round-trip drift")
    return encoded


def _stable_hash(value: Any) -> str:
    return _foundation.stable_hash(value)


def _is_hex64(value: Any) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _is_utc_timestamp(value: Any) -> bool:
    if type(value) is not str or _UTC_TIMESTAMP.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _is_session(value: Any) -> bool:
    if type(value) is not str or _SESSION.fullmatch(value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _path_label(path: Path) -> str:
    try:
        return _foundation.repo_path_label(path)
    except (OSError, ValueError):
        return str(path)


def _canonical_directory(path: Path) -> Path:
    if not os.path.lexists(path):
        raise ProtectedHoldoutError(f"required directory is absent: {path}")
    try:
        info = path.lstat()
    except OSError as exc:
        raise ProtectedHoldoutError(f"cannot stat directory: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ProtectedHoldoutError(f"directory is not a regular non-symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProtectedHoldoutError(f"cannot resolve directory: {path}") from exc
    absolute = Path(os.path.abspath(path))
    if resolved != absolute:
        raise ProtectedHoldoutError(f"directory path is not canonical: {path}")
    return resolved


def _canonical_regular_file(path: Path, *, require_mode_600: bool = False) -> Path:
    if not os.path.lexists(path):
        raise ProtectedHoldoutError(f"required file is absent: {path}")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ProtectedHoldoutError(f"cannot stat file: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProtectedHoldoutError(f"file is not a regular non-symlink: {path}")
    if require_mode_600 and stat.S_IMODE(before.st_mode) != 0o600:
        raise ProtectedHoldoutError(f"semantic file mode drift: {path}")
    resolved = path.resolve(strict=True)
    if resolved != Path(os.path.abspath(path)):
        raise ProtectedHoldoutError(f"file path is not canonical: {path}")
    return resolved


def _read_regular_bytes(path: Path, *, require_mode_600: bool = False) -> bytes:
    resolved = _canonical_regular_file(path, require_mode_600=require_mode_600)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(resolved, flags)
    try:
        opened = os.fstat(fd)
        current = resolved.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
        ):
            raise ProtectedHoldoutError(f"file changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _sha256_regular(path: Path, *, require_mode_600: bool = False) -> str:
    import hashlib

    return hashlib.sha256(
        _read_regular_bytes(path, require_mode_600=require_mode_600)
    ).hexdigest()


def _read_json_regular(path: Path) -> dict[str, Any]:
    raw = _read_regular_bytes(path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtectedHoldoutError(f"invalid JSON file: {path}") from exc
    if type(value) is not dict:
        raise ProtectedHoldoutError(f"JSON file is not an object: {path}")
    _strict_json_value(value)
    return value


def _read_canonical_semantic_json(path: Path) -> dict[str, Any]:
    raw = _read_regular_bytes(path, require_mode_600=True)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtectedHoldoutError(f"invalid semantic JSON file: {path}") from exc
    if type(value) is not dict or _canonical_json_bytes(value) != raw:
        raise ProtectedHoldoutError(f"semantic JSON is not canonical: {path}")
    return value


def write_json_exclusive_durable(path: Path, payload: Any) -> None:
    """Exclusive-create one canonical strict-JSON file and durably sync it.

    This primitive never creates parents, writes temporary files, replaces,
    truncates, unlinks, or retries an occupied semantic path.
    """

    target = Path(path)
    data = _canonical_json_bytes(payload)
    parent = _canonical_directory(target.parent)
    if target.name in {"", ".", ".."} or target.parent.resolve(strict=True) != parent:
        raise ProtectedHoldoutError("invalid exclusive JSON target")
    if os.path.lexists(target):
        raise ProtectedHoldoutError(f"write-once path is already occupied: {target}")

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ProtectedHoldoutError("platform lacks required O_NOFOLLOW/O_DIRECTORY")
    parent_fd = os.open(parent, os.O_RDONLY | directory | nofollow)
    fd: int | None = None
    try:
        try:
            fd = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError as exc:
            raise ProtectedHoldoutError(
                f"write-once path is already occupied: {target}"
            ) from exc
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ProtectedHoldoutError("exclusive JSON target is not regular")
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise ProtectedHoldoutError("short write to exclusive JSON target")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.fsync(parent_fd)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def _ensure_protected_root_safe() -> None:
    """Descriptor-safe wrapper used instead of the terse mkdir helper above."""

    parent = _canonical_directory(PROTECTED_HOLDOUT_ROOT.parent)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    created = False
    try:
        if not os.path.lexists(PROTECTED_HOLDOUT_ROOT):
            try:
                os.mkdir(PROTECTED_HOLDOUT_ROOT.name, 0o700, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                pass
        if created:
            # The directory entry itself must survive power loss before the lock or
            # access receipt can be created inside it.
            os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    root = _canonical_directory(PROTECTED_HOLDOUT_ROOT)
    if stat.S_IMODE(root.lstat().st_mode) != 0o700:
        raise ProtectedHoldoutError("protected holdout directory mode drift")


def _unexpected_root_entries() -> list[str]:
    if not os.path.lexists(PROTECTED_HOLDOUT_ROOT):
        return []
    _canonical_directory(PROTECTED_HOLDOUT_ROOT)
    return sorted(
        entry.name
        for entry in PROTECTED_HOLDOUT_ROOT.iterdir()
        if entry.name not in _EXPECTED_ROOT_NAMES
    )


def _open_lock(*, create: bool) -> int | None:
    if create:
        _ensure_protected_root_safe()
    elif not os.path.lexists(PROTECTED_HOLDOUT_ROOT):
        return None
    else:
        _canonical_directory(PROTECTED_HOLDOUT_ROOT)

    if not os.path.lexists(HOLDOUT_LOCK_PATH) and not create:
        return None
    flags = os.O_RDWR | os.O_NOFOLLOW
    if create:
        flags |= os.O_CREAT
    try:
        fd = os.open(HOLDOUT_LOCK_PATH, flags, 0o600)
    except OSError as exc:
        raise ProtectedHoldoutError("protected holdout lock is unavailable") from exc
    try:
        opened = os.fstat(fd)
        current = HOLDOUT_LOCK_PATH.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise ProtectedHoldoutError("protected holdout lock identity drift")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise _LeaseBusy("protected holdout transaction lease is active") from exc
            raise
        return fd
    except BaseException:
        os.close(fd)
        raise


def _close_lock(fd: int | None) -> None:
    if fd is None or fd < 0:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], owner: str) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise ProtectedHoldoutError(
            f"{owner} field drift (missing={missing}, extra={extra})"
        )


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, owner: str
) -> None:
    observed = value.get(field)
    if not _is_hex64(observed):
        raise ProtectedHoldoutError(f"{owner} {field} is malformed")
    semantic = dict(value)
    semantic.pop(field, None)
    if observed != _stable_hash(semantic):
        raise ProtectedHoldoutError(f"{owner} self hash drift")


_ARTIFACT_ROW_KEYS = {"path", "sha256"}


def _validate_artifact_row(value: Any, *, owner: str) -> dict[str, str]:
    if type(value) is not dict:
        raise ProtectedHoldoutError(f"{owner} artifact reference is not an object")
    _require_exact_keys(value, _ARTIFACT_ROW_KEYS, owner)
    label = value.get("path")
    digest = value.get("sha256")
    if type(label) is not str or not label or not _is_hex64(digest):
        raise ProtectedHoldoutError(f"{owner} artifact reference is malformed")
    if "\\" in label:
        raise ProtectedHoldoutError(f"{owner} artifact path is not POSIX-canonical")
    raw_path = Path(label)
    if raw_path.is_absolute():
        if str(raw_path) != label:
            raise ProtectedHoldoutError(f"{owner} artifact path is not canonical")
        path = raw_path
    else:
        pure = PurePosixPath(label)
        if (
            pure.is_absolute()
            or any(part in ("", ".", "..") for part in pure.parts)
            or pure.as_posix() != label
        ):
            raise ProtectedHoldoutError(f"{owner} artifact path is not canonical")
        path = _foundation.REPO_ROOT / pure
    resolved = _canonical_regular_file(path)
    audit_root = _canonical_directory(AUDIT_ROOT)
    if not resolved.is_relative_to(audit_root):
        raise ProtectedHoldoutError(
            f"{owner} artifact is outside the fixed pre-holdout audit root"
        )
    if os.path.lexists(PROTECTED_HOLDOUT_ROOT):
        protected_root = _canonical_directory(PROTECTED_HOLDOUT_ROOT)
        if resolved.is_relative_to(protected_root):
            raise ProtectedHoldoutError(
                f"{owner} artifact entered the protected transaction root"
            )
    if resolved == Path(os.path.abspath(PRE_HOLDOUT_PACKET_PATH)):
        raise ProtectedHoldoutError(f"{owner} recursively references the packet")
    observed = _sha256_regular(path)
    if observed != digest:
        raise ProtectedHoldoutError(f"{owner} artifact byte hash drift: {label}")
    return {"path": label, "sha256": digest}


def _validate_artifact_list(value: Any, *, owner: str) -> list[dict[str, str]]:
    if type(value) is not list or not value:
        raise ProtectedHoldoutError(f"{owner} must be a nonempty artifact list")
    return [
        _validate_artifact_row(row, owner=f"{owner}[{index}]")
        for index, row in enumerate(value)
    ]


_PACKET_KEYS = {
    "schema_version",
    "status",
    "frozen_at_utc",
    "preregistration_sha256",
    "session_assignments_sha256",
    "source_hash_policy_sha256",
    "corpus_integrity_receipt_sha256",
    "lineage_receipt_sha256",
    "machinery_receipt_sha256",
    "fit_environment_sha256",
    "outer_result_receipts",
    "outer_entry_exit_artifacts",
    "four_box_packet",
    "guard_panel",
    "outer_acceptance_packet",
    "full_fit_entry_artifacts",
    "full_fit_exit_artifacts",
    "selected_box_d_policy_id",
    "selected_comparator_policy_id",
    "artifact_rows",
    "artifact_root_sha256",
    "prerequisite_checks",
    "protected_sessions_sha256_newline",
    "primary_sessions_sha256_newline",
    "holdout_open_count",
    "quarantine_labels",
    "claim_boundary",
    "holdout_caveat",
    "packet_sha256",
}


def _assert_preregistration_current(preregistration_sha256: str) -> None:
    receipt = _foundation.assert_preregistration_frozen()
    if receipt.get("preregistration_sha256") != preregistration_sha256:
        raise ProtectedHoldoutError("frozen preregistration receipt hash drift")
    payload = _foundation.read_json(_foundation.PREREG_PATH)
    _foundation._verify_immutable_sources(payload)
    lineage_receipt = _foundation.assert_lineage_implementation_frozen()
    machinery_receipt = _foundation._assert_entry_machinery_frozen(
        receipt, payload, lineage_receipt
    )
    _foundation._validated_corpus_integrity_receipt(
        payload, receipt, machinery_receipt
    )
    _foundation._verify_authorized_dependency_closure(
        payload, allow_missing_future=False
    )
    _foundation.assert_entry_fit_environment_current()
    _foundation.assert_research_foundation_stable()


def _validate_outer_result_semantics(path: Path, outer_fold: int) -> dict[str, Any]:
    if path != OUTER_RESULT_RECEIPT_PATHS[outer_fold - 1]:
        raise ProtectedHoldoutError("outer-result fixed path drift")
    payload = _foundation.read_json(_foundation.PREREG_PATH)
    assignments = _foundation.read_json(_foundation.SESSION_PATH)
    try:
        receipt = _foundation._validate_outer_result_receipt_one(
            payload, assignments, outer_fold
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise ProtectedHoldoutError(
            f"outer fold {outer_fold} result is not semantically frozen"
        ) from exc
    if (
        type(receipt) is not dict
        or receipt.get("status") != "FROZEN_PRIMARY_OUTER_RESULT"
        or receipt.get("outer_fold") != outer_fold
    ):
        raise ProtectedHoldoutError(f"outer fold {outer_fold} receipt semantic drift")
    return receipt


def _registered_prepacket_artifact_validators() -> dict[str, Callable[[Path], Any]]:
    # Static local imports keep this module importable before the future runner is
    # built while remaining visible to the frozen dependency-closure audit.
    from v4.scripts.run_pathd_entry_exit_research import (
        validate_four_box_packet_for_holdout,
        validate_full_fit_entry_artifacts_for_holdout,
        validate_full_fit_exit_artifacts_for_holdout,
        validate_guard_panel_for_holdout,
        validate_outer_acceptance_packet_for_holdout,
        validate_outer_entry_exit_artifacts_for_holdout,
    )

    return {
        "outer_entry_exit_artifacts": validate_outer_entry_exit_artifacts_for_holdout,
        "four_box_packet": validate_four_box_packet_for_holdout,
        "guard_panel": validate_guard_panel_for_holdout,
        "outer_acceptance_packet": validate_outer_acceptance_packet_for_holdout,
        "full_fit_entry_artifacts": validate_full_fit_entry_artifacts_for_holdout,
        "full_fit_exit_artifacts": validate_full_fit_exit_artifacts_for_holdout,
    }


def _fixed_named_artifact_paths() -> dict[str, Path]:
    return {
        "outer_entry_exit_artifacts": OUTER_ENTRY_EXIT_ARTIFACTS_PATH,
        "four_box_packet": FOUR_BOX_PACKET_PATH,
        "guard_panel": GUARD_PANEL_PATH,
        "outer_acceptance_packet": OUTER_ACCEPTANCE_PACKET_PATH,
        "full_fit_entry_artifacts": FULL_FIT_ENTRY_ARTIFACTS_PATH,
        "full_fit_exit_artifacts": FULL_FIT_EXIT_ARTIFACTS_PATH,
    }


def _validate_named_prepacket_artifacts(
    packet: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    validators = _registered_prepacket_artifact_validators()
    paths = _fixed_named_artifact_paths()
    if set(validators) != set(paths):
        raise ProtectedHoldoutError("registered prepacket validator set drift")
    expected_status = {
        "outer_entry_exit_artifacts": "FROZEN_COMPLETE",
        "four_box_packet": "FROZEN_COMPLETE",
        "guard_panel": "PASS",
        "outer_acceptance_packet": "PASS",
        "full_fit_entry_artifacts": "FROZEN_COMPLETE",
        "full_fit_exit_artifacts": "FROZEN_COMPLETE",
    }
    rows: list[dict[str, str]] = []
    validated_records: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        row = _validate_artifact_row(packet.get(name), owner=name)
        if row["path"] != _path_label(path):
            raise ProtectedHoldoutError(f"{name} fixed path drift")
        try:
            record = validators[name](path)
        except BaseException as exc:
            raise ProtectedHoldoutError(f"{name} semantic validation failed") from exc
        if (
            type(record) is not dict
            or record.get("artifact_kind") != name
            or record.get("status") != expected_status[name]
            or record.get("artifact_sha256") != row["sha256"]
        ):
            raise ProtectedHoldoutError(f"{name} semantic validator result drift")
        _strict_json_value(record)
        rows.append(row)
        validated_records[name] = dict(record)
    acceptance = validated_records["outer_acceptance_packet"]
    four_box = validated_records["four_box_packet"]
    if (
        type(acceptance.get("selected_box_d_policy_id")) is not str
        or type(acceptance.get("selected_comparator_policy_id")) is not str
        or acceptance.get("selected_box_d_policy_id")
        != four_box.get("selected_box_d_policy_id")
        or packet.get("selected_box_d_policy_id")
        != acceptance.get("selected_box_d_policy_id")
        or packet.get("selected_comparator_policy_id")
        != acceptance.get("selected_comparator_policy_id")
    ):
        raise ProtectedHoldoutError("prepacket selected policy identity drift")
    return rows, validated_records


def _frozen_sessions() -> tuple[tuple[str, ...], tuple[str, ...]]:
    assignments = _read_json_regular(SESSION_ASSIGNMENTS_PATH)
    values = assignments.get("protected_holdout_30")
    if (
        type(values) is not list
        or len(values) != 30
        or len(values) != len(set(values))
        or values != sorted(values)
        or any(not _is_session(value) for value in values)
    ):
        raise ProtectedHoldoutError("protected holdout session assignment drift")
    protected = tuple(values)
    degraded = set(_foundation.OPRA_DEGRADED_DATES)
    primary = tuple(value for value in protected if value not in degraded)
    if len(primary) != 29:
        raise ProtectedHoldoutError("protected primary-session assignment drift")
    return protected, primary


def _validate_complete_pre_holdout_packet(
    packet: Mapping[str, Any], *, physical_path: Path | None = None
) -> dict[str, Any]:
    if type(packet) is not dict:
        raise ProtectedHoldoutError("complete pre-holdout packet is not an object")
    _strict_json_value(packet)
    _require_exact_keys(packet, _PACKET_KEYS, "complete pre-holdout packet")
    if (
        packet.get("schema_version") != PACKET_SCHEMA
        or packet.get("status") != "FROZEN_PRE_HOLDOUT"
        or not _is_utc_timestamp(packet.get("frozen_at_utc"))
        or type(packet.get("holdout_open_count")) is not int
        or packet.get("holdout_open_count") != 0
        or packet.get("quarantine_labels") != list(_foundation.QUARANTINE_LABELS)
        or packet.get("claim_boundary") != _foundation.CLAIM_BOUNDARY
        or packet.get("holdout_caveat") != _foundation.HOLDOUT_CAVEAT
    ):
        raise ProtectedHoldoutError("complete pre-holdout packet semantic drift")
    for field in (
        "preregistration_sha256",
        "session_assignments_sha256",
        "source_hash_policy_sha256",
        "corpus_integrity_receipt_sha256",
        "lineage_receipt_sha256",
        "machinery_receipt_sha256",
        "fit_environment_sha256",
        "artifact_root_sha256",
        "protected_sessions_sha256_newline",
        "primary_sessions_sha256_newline",
        "packet_sha256",
    ):
        if not _is_hex64(packet.get(field)):
            raise ProtectedHoldoutError(f"complete packet {field} is malformed")
    for field in ("selected_box_d_policy_id", "selected_comparator_policy_id"):
        if type(packet.get(field)) is not str or not packet[field].strip():
            raise ProtectedHoldoutError(f"complete packet {field} is malformed")

    preregistration_sha256 = _sha256_regular(PREREGISTRATION_PATH)
    if packet["preregistration_sha256"] != preregistration_sha256:
        raise ProtectedHoldoutError("complete packet preregistration hash drift")
    _assert_preregistration_current(preregistration_sha256)
    if packet["session_assignments_sha256"] != _sha256_regular(
        SESSION_ASSIGNMENTS_PATH
    ):
        raise ProtectedHoldoutError("complete packet session assignment hash drift")
    preregistration = _read_json_regular(PREREGISTRATION_PATH)
    if packet["source_hash_policy_sha256"] != _stable_hash(
        preregistration.get("source_hash_policy")
    ):
        raise ProtectedHoldoutError("complete packet source-policy hash drift")
    receipt_bindings = (
        ("corpus_integrity_receipt_sha256", CORPUS_INTEGRITY_RECEIPT_PATH),
        ("lineage_receipt_sha256", LINEAGE_RECEIPT_PATH),
        ("machinery_receipt_sha256", MACHINERY_RECEIPT_PATH),
    )
    for field, path in receipt_bindings:
        if packet[field] != _sha256_regular(path):
            raise ProtectedHoldoutError(f"complete packet {field} drift")
    if packet["fit_environment_sha256"] != _foundation.stable_hash(
        _foundation.assert_entry_fit_environment_current()
    ):
        raise ProtectedHoldoutError("complete packet fit environment drift")

    protected, primary = _frozen_sessions()
    if packet["protected_sessions_sha256_newline"] != _foundation.canonical_session_hash(
        protected
    ):
        raise ProtectedHoldoutError("complete packet protected-session hash drift")
    if packet["primary_sessions_sha256_newline"] != _foundation.canonical_session_hash(
        primary
    ):
        raise ProtectedHoldoutError("complete packet primary-session hash drift")

    checks = packet.get("prerequisite_checks")
    if checks != {
        name: "PASS" for name in PRE_HOLDOUT_PREREQUISITE_CHECK_NAMES
    }:
        raise ProtectedHoldoutError("complete packet prerequisite checks did not all PASS")

    outer = _validate_artifact_list(
        packet.get("outer_result_receipts"), owner="outer_result_receipts"
    )
    if len(outer) != 5 or [row["path"] for row in outer] != [
        _path_label(path) for path in OUTER_RESULT_RECEIPT_PATHS
    ]:
        raise ProtectedHoldoutError("complete packet outer-result identity drift")
    for outer_fold, path in enumerate(OUTER_RESULT_RECEIPT_PATHS, start=1):
        _validate_outer_result_semantics(path, outer_fold)
    groups: list[dict[str, str]] = list(outer)
    named_rows, _semantic_records = _validate_named_prepacket_artifacts(packet)
    groups.extend(named_rows)
    artifact_rows = _validate_artifact_list(
        packet.get("artifact_rows"), owner="artifact_rows"
    )
    paths = [row["path"] for row in artifact_rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ProtectedHoldoutError("complete packet artifact rows are not unique/sorted")
    by_path: dict[str, dict[str, str]] = {}
    for row in groups:
        prior = by_path.setdefault(row["path"], row)
        if prior != row:
            raise ProtectedHoldoutError("conflicting packet artifact references")
    if artifact_rows != [by_path[path] for path in sorted(by_path)]:
        raise ProtectedHoldoutError(
            "complete packet artifact rows are not the exact referenced closure"
        )
    if packet["artifact_root_sha256"] != _stable_hash(artifact_rows):
        raise ProtectedHoldoutError("complete packet artifact root drift")
    _validate_self_hash(packet, field="packet_sha256", owner="complete packet")
    if physical_path is not None:
        observed = _read_canonical_semantic_json(physical_path)
        if observed != packet:
            raise ProtectedHoldoutError("complete packet physical byte drift")
    return dict(packet)


def _freeze_complete_pre_holdout_packet(packet: dict[str, Any], /) -> dict[str, Any]:
    """Internal write-once packet writer for the eventual canonical runner."""

    _assert_protected_holdout_unopened()
    validated = _validate_complete_pre_holdout_packet(packet)
    write_json_exclusive_durable(PRE_HOLDOUT_PACKET_PATH, validated)
    return _validate_complete_pre_holdout_packet(
        _read_canonical_semantic_json(PRE_HOLDOUT_PACKET_PATH),
        physical_path=PRE_HOLDOUT_PACKET_PATH,
    )


_ACCESS_KEYS = {
    "schema_version",
    "status",
    "transaction_id",
    "creator_pid",
    "opened_at_utc",
    "holdout_open_count",
    "preregistration_sha256",
    "session_assignments_sha256",
    "source_hash_policy_sha256",
    "corpus_integrity_receipt_sha256",
    "lineage_receipt_sha256",
    "machinery_receipt_sha256",
    "fit_environment_sha256",
    "preopen_packet_path",
    "preopen_packet_sha256",
    "artifact_root_sha256",
    "protected_sessions",
    "protected_sessions_sha256_newline",
    "primary_sessions_sha256_newline",
    "box_d_policy_id",
    "comparator_policy_id",
    "claim_sha256",
}


def _validate_access_receipt(
    receipt: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any]:
    if type(receipt) is not dict:
        raise ProtectedHoldoutError("holdout access receipt is not an object")
    _require_exact_keys(receipt, _ACCESS_KEYS, "holdout access receipt")
    if (
        receipt.get("schema_version") != ACCESS_SCHEMA
        or receipt.get("status") != "OPENING"
        or type(receipt.get("creator_pid")) is not int
        or receipt["creator_pid"] <= 0
        or not _is_utc_timestamp(receipt.get("opened_at_utc"))
        or type(receipt.get("holdout_open_count")) is not int
        or receipt.get("holdout_open_count") != 1
        or type(receipt.get("transaction_id")) is not str
        or _TRANSACTION_ID.fullmatch(receipt["transaction_id"]) is None
        or receipt.get("preopen_packet_path") != _path_label(PRE_HOLDOUT_PACKET_PATH)
    ):
        raise ProtectedHoldoutError("holdout access receipt semantic drift")
    protected, primary = _frozen_sessions()
    expected = {
        "preregistration_sha256": packet["preregistration_sha256"],
        "session_assignments_sha256": packet["session_assignments_sha256"],
        "source_hash_policy_sha256": packet["source_hash_policy_sha256"],
        "corpus_integrity_receipt_sha256": packet[
            "corpus_integrity_receipt_sha256"
        ],
        "lineage_receipt_sha256": packet["lineage_receipt_sha256"],
        "machinery_receipt_sha256": packet["machinery_receipt_sha256"],
        "fit_environment_sha256": packet["fit_environment_sha256"],
        "preopen_packet_sha256": _sha256_regular(
            PRE_HOLDOUT_PACKET_PATH, require_mode_600=True
        ),
        "artifact_root_sha256": packet["artifact_root_sha256"],
        "protected_sessions": list(protected),
        "protected_sessions_sha256_newline": _foundation.canonical_session_hash(
            protected
        ),
        "primary_sessions_sha256_newline": _foundation.canonical_session_hash(primary),
        "box_d_policy_id": packet["selected_box_d_policy_id"],
        "comparator_policy_id": packet["selected_comparator_policy_id"],
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise ProtectedHoldoutError("holdout access receipt packet binding drift")
    _validate_self_hash(receipt, field="claim_sha256", owner="holdout access receipt")
    return dict(receipt)


def _authorization_semantic(
    access: Mapping[str, Any], *, access_receipt_sha256: str
) -> dict[str, Any]:
    authorization_sha256 = _stable_hash(
        {
            "claim_sha256": access["claim_sha256"],
            "access_receipt_sha256": access_receipt_sha256,
        }
    )
    return {
        "transaction_id": access["transaction_id"],
        "holdout_open_count": 1,
        "sessions": tuple(access["protected_sessions"]),
        "sessions_sha256_newline": access["protected_sessions_sha256_newline"],
        "primary_sessions_sha256_newline": access[
            "primary_sessions_sha256_newline"
        ],
        "preregistration_sha256": access["preregistration_sha256"],
        "session_assignments_sha256": access["session_assignments_sha256"],
        "source_hash_policy_sha256": access["source_hash_policy_sha256"],
        "corpus_integrity_receipt_sha256": access[
            "corpus_integrity_receipt_sha256"
        ],
        "lineage_receipt_sha256": access["lineage_receipt_sha256"],
        "machinery_receipt_sha256": access["machinery_receipt_sha256"],
        "fit_environment_sha256": access["fit_environment_sha256"],
        "preopen_packet_sha256": access["preopen_packet_sha256"],
        "artifact_root_sha256": access["artifact_root_sha256"],
        "box_d_policy_id": access["box_d_policy_id"],
        "comparator_policy_id": access["comparator_policy_id"],
        "access_receipt_sha256": access_receipt_sha256,
        "authorization_sha256": authorization_sha256,
    }


def _active_authorization_for_access(access_sha256: str) -> bool:
    for auth in list(_ACTIVE_AUTH_OBJECTS.values()):
        if (
            not auth._terminal
            and auth._originating_pid == os.getpid()
            and auth.access_receipt_sha256 == access_sha256
            and auth._lock_fd >= 0
        ):
            try:
                os.fstat(auth._lock_fd)
            except OSError:
                continue
            return True
    return False


# A strong reference is intentional: abandoning the opaque capability without a
# terminal receipt must not silently release its lease inside a live process.
_ACTIVE_AUTH_OBJECTS: dict[int, ActiveProtectedHoldoutAuthorizationV1] = {}


def _semantic_presence() -> dict[str, bool]:
    return {
        "access": os.path.lexists(HOLDOUT_ACCESS_RECEIPT_PATH),
        "result": os.path.lexists(HOLDOUT_RESULT_PATH),
        "seal": os.path.lexists(HOLDOUT_SEAL_RECEIPT_PATH),
        "abort": os.path.lexists(HOLDOUT_ABORT_RECEIPT_PATH),
        "trace": os.path.lexists(HOLDOUT_TRACE_PATH),
        "evaluator": os.path.lexists(HOLDOUT_EVALUATOR_RECEIPT_PATH),
    }


def _state(
    state: str,
    *,
    open_count: int,
    detail: str,
    transaction_id: str | None = None,
    access_sha256: str | None = None,
    result_sha256: str | None = None,
    terminal_sha256: str | None = None,
) -> ProtectedHoldoutStateV1:
    return ProtectedHoldoutStateV1(
        schema_version="pathd.protected_holdout_state.v1",
        state=state,
        holdout_open_count=open_count,
        transaction_id=transaction_id,
        detail=detail,
        access_receipt_sha256=access_sha256,
        result_sha256=result_sha256,
        terminal_receipt_sha256=terminal_sha256,
    )


def _read_and_validate_packet() -> dict[str, Any]:
    packet = _read_canonical_semantic_json(PRE_HOLDOUT_PACKET_PATH)
    return _validate_complete_pre_holdout_packet(
        packet, physical_path=PRE_HOLDOUT_PACKET_PATH
    )


def _result_envelope_fields(packet: Mapping[str, Any]) -> dict[str, Any]:
    preregistration = _read_json_regular(PREREGISTRATION_PATH)
    try:
        plan_sha256 = preregistration["binding_plan"]["sha256"]
        fill_law_hash = preregistration["fill_law"]["fill_law_hash"]
    except (KeyError, TypeError) as exc:
        raise ProtectedHoldoutError("preregistration result envelope is malformed") from exc
    if not _is_hex64(plan_sha256) or not _is_hex64(fill_law_hash):
        raise ProtectedHoldoutError("preregistration result-envelope hashes drifted")
    return {
        "quarantine_labels": list(_foundation.QUARANTINE_LABELS),
        "claim_boundary": _foundation.CLAIM_BOUNDARY,
        "holdout_caveat": _foundation.HOLDOUT_CAVEAT,
        "plan_sha256": plan_sha256,
        "preregistration_sha256": packet["preregistration_sha256"],
        "fill_law_hash": fill_law_hash,
        "holdout_open_count": 1,
    }


_HOLDOUT_PAYLOAD_KEYS = {
    "schema_version",
    "verdict",
    "protected_session_count",
    "primary_non_degraded_session_count",
    "completed_box_d_trades_on_primary_29",
    "box_d_net_pnl_micros_on_primary_29",
    "box_d_minus_comparator_paired_net_pnl_micros_on_primary_29",
    "guards",
    "survival_violation_counts",
    "owner_facing_metrics",
    "degraded_sensitivity",
    "pass_criteria_recomputed",
}


def _validate_holdout_payload(payload: Any) -> dict[str, Any]:
    if type(payload) is not dict:
        raise ProtectedHoldoutError("protected holdout payload is not an object")
    _strict_json_value(payload)
    _require_exact_keys(payload, _HOLDOUT_PAYLOAD_KEYS, "protected holdout payload")
    for field, expected in (
        ("protected_session_count", 30),
        ("primary_non_degraded_session_count", 29),
    ):
        if type(payload.get(field)) is not int or payload[field] != expected:
            raise ProtectedHoldoutError(f"protected payload {field} drift")
    for field in (
        "completed_box_d_trades_on_primary_29",
        "box_d_net_pnl_micros_on_primary_29",
        "box_d_minus_comparator_paired_net_pnl_micros_on_primary_29",
    ):
        if type(payload.get(field)) is not int:
            raise ProtectedHoldoutError(f"protected payload {field} must be exact int")
    if payload["completed_box_d_trades_on_primary_29"] < 0:
        raise ProtectedHoldoutError("protected payload trade count is negative")

    guards = payload.get("guards")
    if (
        type(guards) is not dict
        or set(guards) != set(HOLDOUT_GUARD_KEYS)
        or any(type(value) is not bool for value in guards.values())
    ):
        raise ProtectedHoldoutError("protected payload guard panel drift")
    survival = payload.get("survival_violation_counts")
    if (
        type(survival) is not dict
        or set(survival) != set(HOLDOUT_SURVIVAL_VIOLATION_KEYS)
        or any(type(value) is not int or value < 0 for value in survival.values())
    ):
        raise ProtectedHoldoutError("protected payload survival ledger drift")
    metrics = payload.get("owner_facing_metrics")
    if type(metrics) is not dict or set(metrics) != set(OWNER_FACING_METRIC_NAMES):
        raise ProtectedHoldoutError("protected payload owner-facing metric coverage drift")
    degraded = payload.get("degraded_sensitivity")
    if (
        type(degraded) is not dict
        or set(degraded) != {"session", "excluded_from_primary", "metrics"}
        or degraded.get("session") != "2026-07-31"
        or degraded.get("excluded_from_primary") is not True
        or type(degraded.get("metrics")) is not dict
    ):
        raise ProtectedHoldoutError("protected degraded sensitivity was not quarantined")

    criteria = {
        "primary_non_degraded_sessions_at_least_25": payload[
            "primary_non_degraded_session_count"
        ]
        >= 25,
        "completed_box_d_trades_on_primary_29_at_least_50": payload[
            "completed_box_d_trades_on_primary_29"
        ] >= 50,
        "box_d_net_pnl_on_primary_29_positive": payload[
            "box_d_net_pnl_micros_on_primary_29"
        ] > 0,
        "paired_comparator_delta_on_primary_29_positive": payload[
            "box_d_minus_comparator_paired_net_pnl_micros_on_primary_29"
        ]
        > 0,
        "all_guards_pass": all(guards.values()),
        "all_survival_violations_zero": all(value == 0 for value in survival.values()),
    }
    if payload.get("pass_criteria_recomputed") != criteria:
        raise ProtectedHoldoutError("protected payload pass criteria were caller-authored")
    if all(criteria.values()):
        verdict = "PASS"
    elif (
        not criteria["primary_non_degraded_sessions_at_least_25"]
        or not criteria["completed_box_d_trades_on_primary_29_at_least_50"]
    ):
        verdict = "insufficient_evidence"
    elif not criteria["all_guards_pass"] or not criteria[
        "all_survival_violations_zero"
    ]:
        verdict = "owner_decision_required"
    else:
        verdict = "no_genuine_signal"
    if payload.get("verdict") != verdict:
        raise ProtectedHoldoutError("protected payload verdict does not recompute")
    return dict(payload)


_EVALUATOR_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "transaction_id",
    "authorization_sha256",
    "dataset_sha256",
    "source_receipts_root_sha256",
    "machinery_receipt_sha256",
    "evaluator_source_path",
    "evaluator_source_sha256",
    "trace_path",
    "trace_artifact_sha256",
    "trace_root_sha256",
    "payload_sha256",
    "receipt_sha256",
}


def _validate_evaluator_receipt(
    *,
    transaction_id: str,
    authorization_sha256: str,
    dataset_sha256: str,
    source_receipts_root_sha256: str,
    machinery_receipt_sha256: str,
    payload_sha256: str,
) -> dict[str, Any]:
    receipt = _read_canonical_semantic_json(HOLDOUT_EVALUATOR_RECEIPT_PATH)
    _require_exact_keys(receipt, _EVALUATOR_RECEIPT_KEYS, "holdout evaluator receipt")
    trace_sha256 = _sha256_regular(HOLDOUT_TRACE_PATH, require_mode_600=True)
    expected = {
        "schema_version": EVALUATOR_RECEIPT_SCHEMA,
        "status": "EVALUATION_COMPLETE",
        "transaction_id": transaction_id,
        "authorization_sha256": authorization_sha256,
        "dataset_sha256": dataset_sha256,
        "source_receipts_root_sha256": source_receipts_root_sha256,
        "machinery_receipt_sha256": machinery_receipt_sha256,
        "evaluator_source_path": _path_label(EVALUATOR_SOURCE_PATH),
        "evaluator_source_sha256": _sha256_regular(EVALUATOR_SOURCE_PATH),
        "trace_path": _path_label(HOLDOUT_TRACE_PATH),
        "trace_artifact_sha256": trace_sha256,
        # The fixed v1 trace root is the SHA-256 of the exact immutable JSONL bytes.
        "trace_root_sha256": trace_sha256,
        "payload_sha256": payload_sha256,
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise ProtectedHoldoutError("holdout evaluator receipt binding drift")
    _validate_self_hash(
        receipt, field="receipt_sha256", owner="holdout evaluator receipt"
    )
    return receipt


_RESULT_KEYS = {
    "schema_version",
    "quarantine_labels",
    "claim_boundary",
    "holdout_caveat",
    "plan_sha256",
    "preregistration_sha256",
    "fill_law_hash",
    "holdout_open_count",
    "authorization_sha256",
    "access_receipt_sha256",
    "preopen_packet_sha256",
    "artifact_root_sha256",
    "fit_environment_sha256",
    "sessions_sha256_newline",
    "primary_sessions_sha256_newline",
    "dataset_sha256",
    "source_receipts_root_sha256",
    "box_d_policy_id",
    "comparator_policy_id",
    "trace_root_sha256",
    "trace_artifact_sha256",
    "evaluator_receipt_sha256",
    "payload_sha256",
    "payload",
}


def _validate_result(
    result: Mapping[str, Any], access: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any]:
    if type(result) is not dict:
        raise ProtectedHoldoutError("protected holdout result is not an object")
    _strict_json_value(result)
    _require_exact_keys(result, _RESULT_KEYS, "protected holdout result")
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ProtectedHoldoutError("protected holdout result schema drift")
    envelope = _result_envelope_fields(packet)
    if any(result.get(field) != value for field, value in envelope.items()):
        raise ProtectedHoldoutError("protected holdout result envelope drift")
    access_sha256 = _sha256_regular(
        HOLDOUT_ACCESS_RECEIPT_PATH, require_mode_600=True
    )
    authorization = _authorization_semantic(
        access, access_receipt_sha256=access_sha256
    )
    expected = {
        "authorization_sha256": authorization["authorization_sha256"],
        "access_receipt_sha256": access_sha256,
        "preopen_packet_sha256": access["preopen_packet_sha256"],
        "artifact_root_sha256": access["artifact_root_sha256"],
        "fit_environment_sha256": access["fit_environment_sha256"],
        "sessions_sha256_newline": access["protected_sessions_sha256_newline"],
        "primary_sessions_sha256_newline": access[
            "primary_sessions_sha256_newline"
        ],
        "box_d_policy_id": access["box_d_policy_id"],
        "comparator_policy_id": access["comparator_policy_id"],
    }
    if any(result.get(field) != value for field, value in expected.items()):
        raise ProtectedHoldoutError("protected holdout result binding drift")
    for field in (
        "dataset_sha256",
        "source_receipts_root_sha256",
        "trace_root_sha256",
        "trace_artifact_sha256",
        "evaluator_receipt_sha256",
        "payload_sha256",
    ):
        if not _is_hex64(result.get(field)):
            raise ProtectedHoldoutError(f"protected result {field} is malformed")
    payload = _validate_holdout_payload(result.get("payload"))
    if result["payload_sha256"] != _stable_hash(payload):
        raise ProtectedHoldoutError("protected result payload hash drift")
    evaluator_receipt = _validate_evaluator_receipt(
        transaction_id=access["transaction_id"],
        authorization_sha256=authorization["authorization_sha256"],
        dataset_sha256=result["dataset_sha256"],
        source_receipts_root_sha256=result["source_receipts_root_sha256"],
        machinery_receipt_sha256=access["machinery_receipt_sha256"],
        payload_sha256=result["payload_sha256"],
    )
    if (
        result["trace_artifact_sha256"]
        != evaluator_receipt["trace_artifact_sha256"]
        or result["trace_root_sha256"] != evaluator_receipt["trace_root_sha256"]
        or result["evaluator_receipt_sha256"]
        != _sha256_regular(
            HOLDOUT_EVALUATOR_RECEIPT_PATH, require_mode_600=True
        )
    ):
        raise ProtectedHoldoutError("protected result trace/evaluator binding drift")
    return dict(result)


_SEAL_KEYS = {
    "schema_version",
    "status",
    "transaction_id",
    "sealed_at_utc",
    "holdout_open_count",
    "access_receipt_path",
    "access_receipt_sha256",
    "authorization_sha256",
    "preopen_packet_sha256",
    "artifact_root_sha256",
    "fit_environment_sha256",
    "result_path",
    "result_sha256",
    "trace_path",
    "trace_artifact_sha256",
    "evaluator_receipt_path",
    "evaluator_receipt_sha256",
    "dataset_sha256",
    "sessions_sha256_newline",
    "primary_sessions_sha256_newline",
    "box_d_policy_id",
    "comparator_policy_id",
    "payload_sha256",
    "quarantine_labels",
    "claim_boundary",
    "holdout_caveat",
    "receipt_sha256",
}


def _seal_receipt_payload(
    access: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    semantic = {
        "schema_version": SEAL_SCHEMA,
        "status": "SEALED",
        "transaction_id": access["transaction_id"],
        "sealed_at_utc": _now_utc(),
        "holdout_open_count": 1,
        "access_receipt_path": _path_label(HOLDOUT_ACCESS_RECEIPT_PATH),
        "access_receipt_sha256": result["access_receipt_sha256"],
        "authorization_sha256": result["authorization_sha256"],
        "preopen_packet_sha256": result["preopen_packet_sha256"],
        "artifact_root_sha256": result["artifact_root_sha256"],
        "fit_environment_sha256": result["fit_environment_sha256"],
        "result_path": _path_label(HOLDOUT_RESULT_PATH),
        "result_sha256": _sha256_regular(HOLDOUT_RESULT_PATH, require_mode_600=True),
        "trace_path": _path_label(HOLDOUT_TRACE_PATH),
        "trace_artifact_sha256": result["trace_artifact_sha256"],
        "evaluator_receipt_path": _path_label(HOLDOUT_EVALUATOR_RECEIPT_PATH),
        "evaluator_receipt_sha256": result["evaluator_receipt_sha256"],
        "dataset_sha256": result["dataset_sha256"],
        "sessions_sha256_newline": result["sessions_sha256_newline"],
        "primary_sessions_sha256_newline": result[
            "primary_sessions_sha256_newline"
        ],
        "box_d_policy_id": result["box_d_policy_id"],
        "comparator_policy_id": result["comparator_policy_id"],
        "payload_sha256": result["payload_sha256"],
        "quarantine_labels": list(_foundation.QUARANTINE_LABELS),
        "claim_boundary": _foundation.CLAIM_BOUNDARY,
        "holdout_caveat": _foundation.HOLDOUT_CAVEAT,
    }
    return {**semantic, "receipt_sha256": _stable_hash(semantic)}


def _validate_seal(
    seal: Mapping[str, Any],
    access: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if type(seal) is not dict:
        raise ProtectedHoldoutError("protected holdout seal is not an object")
    _require_exact_keys(seal, _SEAL_KEYS, "protected holdout seal")
    if (
        seal.get("schema_version") != SEAL_SCHEMA
        or seal.get("status") != "SEALED"
        or seal.get("transaction_id") != access["transaction_id"]
        or not _is_utc_timestamp(seal.get("sealed_at_utc"))
        or type(seal.get("holdout_open_count")) is not int
        or seal.get("holdout_open_count") != 1
    ):
        raise ProtectedHoldoutError("protected holdout seal semantic drift")
    expected = {
        "access_receipt_path": _path_label(HOLDOUT_ACCESS_RECEIPT_PATH),
        "access_receipt_sha256": result["access_receipt_sha256"],
        "authorization_sha256": result["authorization_sha256"],
        "preopen_packet_sha256": result["preopen_packet_sha256"],
        "artifact_root_sha256": result["artifact_root_sha256"],
        "fit_environment_sha256": result["fit_environment_sha256"],
        "result_path": _path_label(HOLDOUT_RESULT_PATH),
        "result_sha256": _sha256_regular(
            HOLDOUT_RESULT_PATH, require_mode_600=True
        ),
        "trace_path": _path_label(HOLDOUT_TRACE_PATH),
        "trace_artifact_sha256": result["trace_artifact_sha256"],
        "evaluator_receipt_path": _path_label(HOLDOUT_EVALUATOR_RECEIPT_PATH),
        "evaluator_receipt_sha256": result["evaluator_receipt_sha256"],
        "dataset_sha256": result["dataset_sha256"],
        "sessions_sha256_newline": result["sessions_sha256_newline"],
        "primary_sessions_sha256_newline": result[
            "primary_sessions_sha256_newline"
        ],
        "box_d_policy_id": result["box_d_policy_id"],
        "comparator_policy_id": result["comparator_policy_id"],
        "payload_sha256": result["payload_sha256"],
        "quarantine_labels": list(_foundation.QUARANTINE_LABELS),
        "claim_boundary": _foundation.CLAIM_BOUNDARY,
        "holdout_caveat": _foundation.HOLDOUT_CAVEAT,
    }
    if any(seal.get(field) != value for field, value in expected.items()):
        raise ProtectedHoldoutError("protected holdout seal binding drift")
    _validate_self_hash(seal, field="receipt_sha256", owner="protected holdout seal")
    return dict(seal)


_ABORT_KEYS = {
    "schema_version",
    "status",
    "transaction_id",
    "aborted_at_utc",
    "holdout_open_count",
    "access_receipt_path",
    "access_receipt_sha256",
    "authorization_sha256",
    "preopen_packet_sha256",
    "artifact_root_sha256",
    "result_path",
    "result_sha256",
    "trace_path",
    "trace_artifact_sha256",
    "evaluator_receipt_path",
    "evaluator_receipt_sha256",
    "reason_code",
    "corpus_reopened",
    "artifacts_rewritten",
    "quarantine_labels",
    "claim_boundary",
    "holdout_caveat",
    "receipt_sha256",
}


def _existing_result_hash_or_none() -> str | None:
    if not os.path.lexists(HOLDOUT_RESULT_PATH):
        return None
    try:
        return _sha256_regular(HOLDOUT_RESULT_PATH, require_mode_600=True)
    except ProtectedHoldoutError:
        return None


def _existing_semantic_hash_or_none(path: Path) -> str | None:
    if not os.path.lexists(path):
        return None
    try:
        return _sha256_regular(path, require_mode_600=True)
    except ProtectedHoldoutError:
        return None


def _abort_receipt_payload(
    access: Mapping[str, Any], *, reason_code: str
) -> dict[str, Any]:
    if reason_code not in ABORT_REASON_CODES:
        raise ProtectedHoldoutError("unsupported protected holdout abort reason")
    access_sha256 = _sha256_regular(
        HOLDOUT_ACCESS_RECEIPT_PATH, require_mode_600=True
    )
    authorization = _authorization_semantic(
        access, access_receipt_sha256=access_sha256
    )
    semantic = {
        "schema_version": ABORT_SCHEMA,
        "status": "ABORTED",
        "transaction_id": access["transaction_id"],
        "aborted_at_utc": _now_utc(),
        "holdout_open_count": 1,
        "access_receipt_path": _path_label(HOLDOUT_ACCESS_RECEIPT_PATH),
        "access_receipt_sha256": access_sha256,
        "authorization_sha256": authorization["authorization_sha256"],
        "preopen_packet_sha256": access["preopen_packet_sha256"],
        "artifact_root_sha256": access["artifact_root_sha256"],
        "result_path": _path_label(HOLDOUT_RESULT_PATH),
        "result_sha256": _existing_result_hash_or_none(),
        "trace_path": _path_label(HOLDOUT_TRACE_PATH),
        "trace_artifact_sha256": _existing_semantic_hash_or_none(
            HOLDOUT_TRACE_PATH
        ),
        "evaluator_receipt_path": _path_label(HOLDOUT_EVALUATOR_RECEIPT_PATH),
        "evaluator_receipt_sha256": _existing_semantic_hash_or_none(
            HOLDOUT_EVALUATOR_RECEIPT_PATH
        ),
        "reason_code": reason_code,
        "corpus_reopened": False,
        "artifacts_rewritten": False,
        "quarantine_labels": list(_foundation.QUARANTINE_LABELS),
        "claim_boundary": _foundation.CLAIM_BOUNDARY,
        "holdout_caveat": _foundation.HOLDOUT_CAVEAT,
    }
    return {**semantic, "receipt_sha256": _stable_hash(semantic)}


def _validate_abort(
    abort: Mapping[str, Any], access: Mapping[str, Any]
) -> dict[str, Any]:
    if type(abort) is not dict:
        raise ProtectedHoldoutError("protected holdout abort is not an object")
    _require_exact_keys(abort, _ABORT_KEYS, "protected holdout abort")
    access_sha256 = _sha256_regular(
        HOLDOUT_ACCESS_RECEIPT_PATH, require_mode_600=True
    )
    authorization = _authorization_semantic(
        access, access_receipt_sha256=access_sha256
    )
    if os.path.lexists(HOLDOUT_RESULT_PATH):
        current_result_sha256 = _sha256_regular(
            HOLDOUT_RESULT_PATH, require_mode_600=True
        )
    else:
        current_result_sha256 = None
    if (
        abort.get("schema_version") != ABORT_SCHEMA
        or abort.get("status") != "ABORTED"
        or abort.get("transaction_id") != access["transaction_id"]
        or not _is_utc_timestamp(abort.get("aborted_at_utc"))
        or type(abort.get("holdout_open_count")) is not int
        or abort.get("holdout_open_count") != 1
        or abort.get("reason_code") not in ABORT_REASON_CODES
        or abort.get("corpus_reopened") is not False
        or abort.get("artifacts_rewritten") is not False
    ):
        raise ProtectedHoldoutError("protected holdout abort semantic drift")
    expected = {
        "access_receipt_path": _path_label(HOLDOUT_ACCESS_RECEIPT_PATH),
        "access_receipt_sha256": access_sha256,
        "authorization_sha256": authorization["authorization_sha256"],
        "preopen_packet_sha256": access["preopen_packet_sha256"],
        "artifact_root_sha256": access["artifact_root_sha256"],
        "result_path": _path_label(HOLDOUT_RESULT_PATH),
        "result_sha256": current_result_sha256,
        "trace_path": _path_label(HOLDOUT_TRACE_PATH),
        "trace_artifact_sha256": _existing_semantic_hash_or_none(
            HOLDOUT_TRACE_PATH
        ),
        "evaluator_receipt_path": _path_label(HOLDOUT_EVALUATOR_RECEIPT_PATH),
        "evaluator_receipt_sha256": _existing_semantic_hash_or_none(
            HOLDOUT_EVALUATOR_RECEIPT_PATH
        ),
        "quarantine_labels": list(_foundation.QUARANTINE_LABELS),
        "claim_boundary": _foundation.CLAIM_BOUNDARY,
        "holdout_caveat": _foundation.HOLDOUT_CAVEAT,
    }
    if any(abort.get(field) != value for field, value in expected.items()):
        raise ProtectedHoldoutError("protected holdout abort binding drift")
    _validate_self_hash(abort, field="receipt_sha256", owner="protected holdout abort")
    return dict(abort)


def inspect_protected_holdout_state() -> ProtectedHoldoutStateV1:
    """Inspect fixed paths without mutating or acquiring protected data."""

    presence = _semantic_presence()
    open_count = 1 if presence["access"] else 0
    try:
        if os.path.lexists(PROTECTED_HOLDOUT_ROOT):
            unexpected = _unexpected_root_entries()
            if unexpected:
                raise ProtectedHoldoutError(
                    f"unexpected protected-root entries: {unexpected}"
                )
        if os.path.lexists(HOLDOUT_LOCK_PATH):
            _canonical_regular_file(HOLDOUT_LOCK_PATH, require_mode_600=True)

        if not any(presence.values()):
            if os.path.lexists(PRE_HOLDOUT_PACKET_PATH):
                _read_and_validate_packet()
            return _state(
                UNOPENED,
                open_count=0,
                detail="no protected holdout access receipt exists",
            )
        if not presence["access"]:
            raise ProtectedHoldoutError(
                "reserved result/seal/abort path exists before access receipt"
            )
        packet = _read_and_validate_packet()
        access = _validate_access_receipt(
            _read_canonical_semantic_json(HOLDOUT_ACCESS_RECEIPT_PATH), packet
        )
        access_sha256 = _sha256_regular(
            HOLDOUT_ACCESS_RECEIPT_PATH, require_mode_600=True
        )
        transaction_id = access["transaction_id"]
        if presence["seal"] and presence["abort"]:
            raise ProtectedHoldoutError("both terminal holdout receipts exist")
        if presence["seal"]:
            if not presence["result"]:
                raise ProtectedHoldoutError("holdout seal exists without result")
            result = _validate_result(
                _read_canonical_semantic_json(HOLDOUT_RESULT_PATH), access, packet
            )
            seal = _validate_seal(
                _read_canonical_semantic_json(HOLDOUT_SEAL_RECEIPT_PATH),
                access,
                result,
            )
            return _state(
                SEALED,
                open_count=1,
                detail="protected holdout result is durably sealed",
                transaction_id=transaction_id,
                access_sha256=access_sha256,
                result_sha256=_sha256_regular(
                    HOLDOUT_RESULT_PATH, require_mode_600=True
                ),
                terminal_sha256=_sha256_regular(
                    HOLDOUT_SEAL_RECEIPT_PATH, require_mode_600=True
                ),
            )
        if presence["abort"]:
            if presence["result"]:
                _validate_result(
                    _read_canonical_semantic_json(HOLDOUT_RESULT_PATH), access, packet
                )
            abort = _validate_abort(
                _read_canonical_semantic_json(HOLDOUT_ABORT_RECEIPT_PATH), access
            )
            return _state(
                ABORTED,
                open_count=1,
                detail=f"protected holdout transaction aborted: {abort['reason_code']}",
                transaction_id=transaction_id,
                access_sha256=access_sha256,
                result_sha256=_existing_result_hash_or_none(),
                terminal_sha256=_sha256_regular(
                    HOLDOUT_ABORT_RECEIPT_PATH, require_mode_600=True
                ),
            )
        if presence["result"]:
            _validate_result(
                _read_canonical_semantic_json(HOLDOUT_RESULT_PATH), access, packet
            )
            return _state(
                RESULT_DURABLE_PENDING_ABORT,
                open_count=1,
                detail=(
                    "durable result exists without an in-process seal and must be "
                    "permanently burned without corpus access"
                ),
                transaction_id=transaction_id,
                access_sha256=access_sha256,
                result_sha256=_sha256_regular(
                    HOLDOUT_RESULT_PATH, require_mode_600=True
                ),
            )
        if _active_authorization_for_access(access_sha256):
            return _state(
                OPENING_ACTIVE,
                open_count=1,
                detail="originating process-local authorization owns the lease",
                transaction_id=transaction_id,
                access_sha256=access_sha256,
            )
        try:
            fd = _open_lock(create=False)
        except _LeaseBusy:
            return _state(
                OPENING_ACTIVE,
                open_count=1,
                detail="originating transaction lease is active",
                transaction_id=transaction_id,
                access_sha256=access_sha256,
            )
        if fd is None:
            raise ProtectedHoldoutError("access receipt exists without transaction lock")
        _close_lock(fd)
        return _state(
            BURNED_INCOMPLETE,
            open_count=1,
            detail="access receipt exists but originating lease ended",
            transaction_id=transaction_id,
            access_sha256=access_sha256,
        )
    except ProtectedHoldoutError as exc:
        return _state(
            CORRUPT_BURNED,
            open_count=open_count,
            detail=str(exc),
        )


def _assert_protected_holdout_unopened() -> None:
    """Block every pre-holdout writer as soon as any reserved path is occupied."""

    if os.path.lexists(HOLDOUT_ACCESS_RECEIPT_PATH):
        raise ProtectedHoldoutError(
            "protected holdout is already opened; pre-holdout writes are forbidden"
        )
    if any(
        os.path.lexists(path)
        for path in (
            HOLDOUT_RESULT_PATH,
            HOLDOUT_SEAL_RECEIPT_PATH,
            HOLDOUT_ABORT_RECEIPT_PATH,
            HOLDOUT_TRACE_PATH,
            HOLDOUT_EVALUATOR_RECEIPT_PATH,
        )
    ):
        raise ProtectedHoldoutError(
            "protected holdout reserved path is occupied; writes are forbidden"
        )
    state = inspect_protected_holdout_state()
    if state.state not in {UNOPENED}:
        raise ProtectedHoldoutError(
            f"protected holdout is not safely unopened: {state.state}"
        )


def _validate_live_authorization(
    authorization: ActiveProtectedHoldoutAuthorizationV1,
    transaction_token: str,
    *,
    allow_result: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(authorization) is not ActiveProtectedHoldoutAuthorizationV1:
        raise ProtectedHoldoutError("untrusted protected holdout authorization type")
    if (
        authorization._originating_pid != os.getpid()
        or authorization._terminal
        or authorization._lock_fd < 0
        or type(transaction_token) is not str
        or authorization._transaction_token_sha256
        != _stable_hash({"protected_holdout_transaction_token": transaction_token})
        or _ACTIVE_AUTH_OBJECTS.get(id(authorization)) is not authorization
    ):
        raise ProtectedHoldoutError("protected holdout authorization is inactive")
    try:
        opened = os.fstat(authorization._lock_fd)
        current = HOLDOUT_LOCK_PATH.lstat()
    except OSError as exc:
        raise ProtectedHoldoutError("protected holdout lease is no longer live") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or opened.st_dev != current.st_dev
        or opened.st_ino != current.st_ino
    ):
        raise ProtectedHoldoutError("protected holdout lease identity drift")
    if _unexpected_root_entries():
        raise ProtectedHoldoutError("unexpected file entered protected holdout root")
    presence = _semantic_presence()
    if not presence["access"] or presence["seal"] or presence["abort"]:
        raise ProtectedHoldoutError("protected holdout transaction is terminal or corrupt")
    if presence["result"] and not allow_result:
        raise ProtectedHoldoutError("protected holdout result path is already occupied")
    packet = _read_and_validate_packet()
    access = _validate_access_receipt(
        _read_canonical_semantic_json(HOLDOUT_ACCESS_RECEIPT_PATH), packet
    )
    access_sha256 = _sha256_regular(
        HOLDOUT_ACCESS_RECEIPT_PATH, require_mode_600=True
    )
    semantic = _authorization_semantic(access, access_receipt_sha256=access_sha256)
    if dict(authorization._semantic) != semantic:
        raise ProtectedHoldoutError("protected holdout authorization binding drift")
    return access, packet


def _registered_dataset_api() -> tuple[type[Any], Callable[..., Any], Callable[..., Any]]:
    try:
        from v4.research.pathd_entry_dataset import (
            EntryEvidenceDatasetV1,
            load_authorized_protected_holdout_dataset,
            validate_protected_holdout_dataset,
        )
    except (ImportError, AttributeError) as exc:
        raise ProtectedHoldoutError(
            "registered protected dataset loader is unavailable"
        ) from exc
    dataset_type = EntryEvidenceDatasetV1
    loader = load_authorized_protected_holdout_dataset
    validator = validate_protected_holdout_dataset
    if not isinstance(dataset_type, type) or not callable(loader) or not callable(validator):
        raise ProtectedHoldoutError("registered protected dataset API is malformed")
    return dataset_type, loader, validator


def _dataset_field(dataset: Any, name: str) -> Any:
    try:
        return getattr(dataset, name)
    except AttributeError as exc:
        raise ProtectedHoldoutError(f"protected dataset lacks {name}") from exc


def _source_receipts_root(dataset: Any) -> str:
    receipts = _dataset_field(dataset, "source_receipts")
    if isinstance(receipts, tuple):
        receipts = list(receipts)
    if type(receipts) is not list or not receipts:
        raise ProtectedHoldoutError("protected dataset source receipts are malformed")
    normalized: list[Any] = []
    for receipt in receipts:
        if hasattr(receipt, "to_dict") and callable(receipt.to_dict):
            receipt = receipt.to_dict()
        elif hasattr(receipt, "__dataclass_fields__"):
            from dataclasses import asdict

            receipt = asdict(receipt)
        if type(receipt) is not dict:
            raise ProtectedHoldoutError("protected dataset source receipt is unsealed")
        _strict_json_value(receipt)
        normalized.append(receipt)
    return _stable_hash(normalized)


def _revalidate_authorized_dataset(
    authorization: ActiveProtectedHoldoutAuthorizationV1,
    transaction_token: str,
) -> Any:
    _validate_live_authorization(authorization, transaction_token)
    dataset = authorization._dataset
    if dataset is None:
        raise ProtectedHoldoutError("protected dataset has not been decoded")
    dataset_type, _loader, validator = _registered_dataset_api()
    if type(dataset) is not dataset_type:
        raise ProtectedHoldoutError("protected dataset concrete type drift")
    validated = validator(dataset, authorization=authorization)
    if validated is not dataset:
        raise ProtectedHoldoutError("protected dataset validator identity drift")
    if (
        _dataset_field(dataset, "schema_version")
        != "pathd.entry_evidence_dataset.v1"
        or _dataset_field(dataset, "authorization_sha256")
        != authorization.authorization_sha256
        or _dataset_field(dataset, "role") != "protected_holdout_once"
        or tuple(_dataset_field(dataset, "sessions")) != authorization.sessions
        or _dataset_field(dataset, "sessions_sha256_newline")
        != authorization.sessions_sha256_newline
        or _dataset_field(dataset, "dataset_sha256")
        != authorization._dataset_sha256
        or _source_receipts_root(dataset)
        != authorization._source_receipts_root_sha256
    ):
        raise ProtectedHoldoutError("protected dataset authorization seal drift")
    return dataset


def _load_authorized_protected_holdout(
    authorization: ActiveProtectedHoldoutAuthorizationV1,
    transaction_token: str,
    /,
) -> EntryEvidenceDatasetV1:
    """Decode only for the token held by the atomic execute call."""

    _validate_live_authorization(authorization, transaction_token)
    if authorization._decode_attempted:
        raise ProtectedHoldoutError("protected holdout decode is one-use")
    authorization._decode_attempted = True
    try:
        dataset_type, loader, _validator = _registered_dataset_api()
        # Revalidation immediately before this call ensures the access receipt and
        # every packet-bound artifact hash are durable/current before any decoder runs.
        _validate_live_authorization(authorization, transaction_token)
        dataset = loader(authorization)
        if type(dataset) is not dataset_type:
            raise ProtectedHoldoutError("protected dataset concrete type drift")
        if (
            _dataset_field(dataset, "schema_version")
            != "pathd.entry_evidence_dataset.v1"
            or _dataset_field(dataset, "authorization_sha256")
            != authorization.authorization_sha256
            or _dataset_field(dataset, "role") != "protected_holdout_once"
            or tuple(_dataset_field(dataset, "sessions")) != authorization.sessions
            or _dataset_field(dataset, "sessions_sha256_newline")
            != authorization.sessions_sha256_newline
            or not _is_hex64(_dataset_field(dataset, "dataset_sha256"))
        ):
            raise ProtectedHoldoutError("protected dataset authorization seal drift")
        authorization._dataset = dataset
        authorization._dataset_sha256 = _dataset_field(dataset, "dataset_sha256")
        authorization._source_receipts_root_sha256 = _source_receipts_root(dataset)
        _revalidate_authorized_dataset(authorization, transaction_token)
        return dataset
    except BaseException:
        try:
            _abort_active(
                authorization,
                transaction_token,
                reason_code="DATASET_DECODE_FAILED",
            )
        except BaseException:
            _deactivate_authorization(authorization)
        raise


def _registered_evaluator_api() -> tuple[
    type[Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]
]:
    # Static local imports are part of the frozen source dependency closure.
    from v4.scripts.run_pathd_entry_exit_research import (
        ProtectedHoldoutEvaluationV1,
        reconstruct_protected_holdout_evaluation_from_trace,
        run_fixed_protected_holdout_evaluator,
        validate_protected_holdout_evaluation,
    )

    if (
        not isinstance(ProtectedHoldoutEvaluationV1, type)
        or not callable(run_fixed_protected_holdout_evaluator)
        or not callable(validate_protected_holdout_evaluation)
        or not callable(reconstruct_protected_holdout_evaluation_from_trace)
    ):
        raise ProtectedHoldoutError("registered protected evaluator API is malformed")
    return (
        ProtectedHoldoutEvaluationV1,
        run_fixed_protected_holdout_evaluator,
        validate_protected_holdout_evaluation,
        reconstruct_protected_holdout_evaluation_from_trace,
    )


def _validated_evaluation(
    authorization: ActiveProtectedHoldoutAuthorizationV1,
    evaluation: ProtectedHoldoutEvaluationV1,
) -> dict[str, Any]:
    evaluation_type, _runner, validator, reconstructor = _registered_evaluator_api()
    if type(evaluation) is not evaluation_type:
        raise ProtectedHoldoutError("untrusted protected holdout evaluation type")
    value = validator(
        evaluation,
        authorization=authorization,
        dataset=authorization._dataset,
    )
    if type(value) is not dict:
        raise ProtectedHoldoutError("protected evaluator validator returned no seal")
    reconstructed = reconstructor(authorization, authorization._dataset)
    if type(reconstructed) is not dict or reconstructed != value:
        raise ProtectedHoldoutError(
            "protected evaluator result does not reconstruct from typed trace"
        )
    _strict_json_value(value)
    expected_keys = {
        "schema_version",
        "authorization_sha256",
        "access_receipt_sha256",
        "preopen_packet_sha256",
        "artifact_root_sha256",
        "fit_environment_sha256",
        "sessions_sha256_newline",
        "primary_sessions_sha256_newline",
        "dataset_sha256",
        "source_receipts_root_sha256",
        "box_d_policy_id",
        "comparator_policy_id",
        "trace_root_sha256",
        "trace_artifact_sha256",
        "evaluator_receipt_sha256",
        "payload_sha256",
        "payload",
    }
    _require_exact_keys(value, expected_keys, "protected evaluator seal")
    expected = {
        "schema_version": EVALUATION_SCHEMA,
        "authorization_sha256": authorization.authorization_sha256,
        "access_receipt_sha256": authorization.access_receipt_sha256,
        "preopen_packet_sha256": authorization.preopen_packet_sha256,
        "artifact_root_sha256": authorization.artifact_root_sha256,
        "fit_environment_sha256": authorization.fit_environment_sha256,
        "sessions_sha256_newline": authorization.sessions_sha256_newline,
        "primary_sessions_sha256_newline": authorization.primary_sessions_sha256_newline,
        "dataset_sha256": authorization._dataset_sha256,
        "source_receipts_root_sha256": authorization._source_receipts_root_sha256,
        "box_d_policy_id": authorization.box_d_policy_id,
        "comparator_policy_id": authorization.comparator_policy_id,
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise ProtectedHoldoutError("protected evaluation authorization binding drift")
    payload = _validate_holdout_payload(value.get("payload"))
    if not _is_hex64(value.get("payload_sha256")) or value[
        "payload_sha256"
    ] != _stable_hash(payload):
        raise ProtectedHoldoutError("protected evaluation payload binding drift")
    evaluator_receipt = _validate_evaluator_receipt(
        transaction_id=authorization.transaction_id,
        authorization_sha256=authorization.authorization_sha256,
        dataset_sha256=authorization._dataset_sha256,
        source_receipts_root_sha256=authorization._source_receipts_root_sha256,
        machinery_receipt_sha256=authorization.machinery_receipt_sha256,
        payload_sha256=value["payload_sha256"],
    )
    if (
        value.get("trace_root_sha256") != evaluator_receipt["trace_root_sha256"]
        or value.get("trace_artifact_sha256")
        != evaluator_receipt["trace_artifact_sha256"]
        or value.get("evaluator_receipt_sha256")
        != _sha256_regular(
            HOLDOUT_EVALUATOR_RECEIPT_PATH, require_mode_600=True
        )
    ):
        raise ProtectedHoldoutError("protected evaluator trace receipt drift")
    return value


def _holdout_result_from_evaluation(
    authorization: ActiveProtectedHoldoutAuthorizationV1,
    evaluation: ProtectedHoldoutEvaluationV1,
    transaction_token: str,
) -> dict[str, Any]:
    access, packet = _validate_live_authorization(
        authorization, transaction_token
    )
    _revalidate_authorized_dataset(authorization, transaction_token)
    value = _validated_evaluation(authorization, evaluation)
    result = {
        "schema_version": RESULT_SCHEMA,
        **_result_envelope_fields(packet),
        **{
            field: value[field]
            for field in (
                "authorization_sha256",
                "access_receipt_sha256",
                "preopen_packet_sha256",
                "artifact_root_sha256",
                "fit_environment_sha256",
                "sessions_sha256_newline",
                "primary_sessions_sha256_newline",
                "dataset_sha256",
                "source_receipts_root_sha256",
                "box_d_policy_id",
                "comparator_policy_id",
                "trace_root_sha256",
                "trace_artifact_sha256",
                "evaluator_receipt_sha256",
                "payload_sha256",
                "payload",
            )
        },
    }
    return _validate_result(result, access, packet)


def _execute_protected_holdout_transaction(
    authorization: ActiveProtectedHoldoutAuthorizationV1,
    transaction_token: str,
    /,
) -> dict[str, Any]:
    """Decode, evaluate, and terminate under one private transaction token."""

    try:
        _validate_live_authorization(authorization, transaction_token)
        if authorization._decode_attempted:
            raise ProtectedHoldoutError(
                "protected holdout execution is already consumed"
            )
        dataset = _load_authorized_protected_holdout(
            authorization, transaction_token
        )
        _revalidate_authorized_dataset(authorization, transaction_token)
        evaluation_type, evaluator, _validator, _reconstructor = _registered_evaluator_api()
        evaluation = evaluator(authorization, dataset)
        if type(evaluation) is not evaluation_type:
            raise ProtectedHoldoutError(
                "fixed protected evaluator returned an untrusted type"
            )
        return _seal_protected_holdout_result(
            authorization, evaluation, transaction_token
        )
    except BaseException:
        if (
            _ACTIVE_AUTH_OBJECTS.get(id(authorization)) is authorization
            and not authorization._terminal
        ):
            try:
                _abort_active(
                    authorization,
                    transaction_token,
                    reason_code="EVALUATION_VALIDATION_FAILED",
                )
            except BaseException:
                _deactivate_authorization(authorization)
        raise


def _deactivate_authorization(
    authorization: ActiveProtectedHoldoutAuthorizationV1,
) -> None:
    if type(authorization) is not ActiveProtectedHoldoutAuthorizationV1:
        return
    _ACTIVE_AUTH_OBJECTS.pop(id(authorization), None)
    authorization._terminal = True
    authorization._transaction_token_sha256 = None
    fd = authorization._lock_fd
    authorization._lock_fd = -1
    _close_lock(fd)


def _seal_protected_holdout_result(
    authorization: ActiveProtectedHoldoutAuthorizationV1,
    evaluation: ProtectedHoldoutEvaluationV1,
    transaction_token: str,
    /,
) -> dict[str, Any]:
    """Durably commit the count-one result and terminal seal receipt."""

    _validate_live_authorization(authorization, transaction_token)
    result_created = False
    try:
        result = _holdout_result_from_evaluation(
            authorization, evaluation, transaction_token
        )
        # Revalidate all packet/artifact/access bindings at the last possible point.
        access, packet = _validate_live_authorization(
            authorization, transaction_token
        )
        _revalidate_authorized_dataset(authorization, transaction_token)
        _validate_result(result, access, packet)
        write_json_exclusive_durable(HOLDOUT_RESULT_PATH, result)
        result_created = True
        persisted_result = _validate_result(
            _read_canonical_semantic_json(HOLDOUT_RESULT_PATH), access, packet
        )
        seal = _seal_receipt_payload(access, persisted_result)
        write_json_exclusive_durable(HOLDOUT_SEAL_RECEIPT_PATH, seal)
        persisted_seal = _validate_seal(
            _read_canonical_semantic_json(HOLDOUT_SEAL_RECEIPT_PATH),
            access,
            persisted_result,
        )
        _deactivate_authorization(authorization)
        return persisted_seal
    except BaseException:
        if not result_created:
            try:
                _abort_active(
                    authorization,
                    transaction_token,
                    reason_code="RESULT_COMMIT_FAILED",
                )
            except BaseException:
                _deactivate_authorization(authorization)
        else:
            # An in-process seal failure permanently disqualifies this result.
            # Recovery never decodes the corpus and may only write a burn receipt.
            _deactivate_authorization(authorization)
        raise


def execute_protected_holdout_once() -> dict[str, Any]:
    """Atomically begin, decode, evaluate, and seal the one protected holdout."""

    transaction_token = uuid.uuid4().hex + uuid.uuid4().hex
    lock_fd = _open_lock(create=True)
    assert lock_fd is not None
    access_created = False
    try:
        presence = _semantic_presence()
        unexpected = _unexpected_root_entries()
        if unexpected or any(presence.values()):
            raise ProtectedHoldoutError(
                "protected holdout cannot open: a reserved path is occupied"
            )
        packet = _read_and_validate_packet()
        protected, primary = _frozen_sessions()
        semantic = {
            "schema_version": ACCESS_SCHEMA,
            "status": "OPENING",
            "transaction_id": str(uuid.uuid4()),
            "creator_pid": os.getpid(),
            "opened_at_utc": _now_utc(),
            "holdout_open_count": 1,
            "preregistration_sha256": packet["preregistration_sha256"],
            "session_assignments_sha256": packet["session_assignments_sha256"],
            "source_hash_policy_sha256": packet["source_hash_policy_sha256"],
            "corpus_integrity_receipt_sha256": packet[
                "corpus_integrity_receipt_sha256"
            ],
            "lineage_receipt_sha256": packet["lineage_receipt_sha256"],
            "machinery_receipt_sha256": packet["machinery_receipt_sha256"],
            "fit_environment_sha256": packet["fit_environment_sha256"],
            "preopen_packet_path": _path_label(PRE_HOLDOUT_PACKET_PATH),
            "preopen_packet_sha256": _sha256_regular(
                PRE_HOLDOUT_PACKET_PATH, require_mode_600=True
            ),
            "artifact_root_sha256": packet["artifact_root_sha256"],
            "protected_sessions": list(protected),
            "protected_sessions_sha256_newline": _foundation.canonical_session_hash(
                protected
            ),
            "primary_sessions_sha256_newline": _foundation.canonical_session_hash(
                primary
            ),
            "box_d_policy_id": packet["selected_box_d_policy_id"],
            "comparator_policy_id": packet["selected_comparator_policy_id"],
        }
        receipt = {**semantic, "claim_sha256": _stable_hash(semantic)}
        write_json_exclusive_durable(HOLDOUT_ACCESS_RECEIPT_PATH, receipt)
        access_created = True
        persisted = _validate_access_receipt(
            _read_canonical_semantic_json(HOLDOUT_ACCESS_RECEIPT_PATH), packet
        )
        access_sha256 = _sha256_regular(
            HOLDOUT_ACCESS_RECEIPT_PATH, require_mode_600=True
        )
        authorization_semantic = _authorization_semantic(
            persisted, access_receipt_sha256=access_sha256
        )
        authorization = ActiveProtectedHoldoutAuthorizationV1(
            _AUTH_CONSTRUCTOR_TOKEN,
            transaction_token_sha256=_stable_hash(
                {"protected_holdout_transaction_token": transaction_token}
            ),
            lock_fd=lock_fd,
            semantic=authorization_semantic,
        )
        _ACTIVE_AUTH_OBJECTS[id(authorization)] = authorization
    except BaseException:
        _close_lock(lock_fd)
        if access_created:
            # The durable access receipt deliberately remains and burns the attempt.
            pass
        raise
    return _execute_protected_holdout_transaction(
        authorization, transaction_token
    )


def _abort_active(
    authorization: ActiveProtectedHoldoutAuthorizationV1,
    transaction_token: str,
    *,
    reason_code: str,
) -> dict[str, Any]:
    access, _packet = _validate_live_authorization(
        authorization, transaction_token, allow_result=True
    )
    if os.path.lexists(HOLDOUT_SEAL_RECEIPT_PATH):
        raise ProtectedHoldoutError("cannot abort a sealed protected holdout")
    abort = _abort_receipt_payload(access, reason_code=reason_code)
    write_json_exclusive_durable(HOLDOUT_ABORT_RECEIPT_PATH, abort)
    persisted = _validate_abort(
        _read_canonical_semantic_json(HOLDOUT_ABORT_RECEIPT_PATH), access
    )
    _deactivate_authorization(authorization)
    return persisted


def _abort_protected_holdout(
    authorization: ActiveProtectedHoldoutAuthorizationV1,
    transaction_token: str,
    *,
    reason_code: str,
) -> dict[str, Any]:
    """Private pre-decode abort for atomic-runner failure injection/tests."""

    _validate_live_authorization(authorization, transaction_token)
    if authorization._decode_attempted:
        raise ProtectedHoldoutError(
            "pre-decode abort is forbidden after protected decode starts"
        )
    if reason_code != "OPERATOR_ABORT":
        raise ProtectedHoldoutError(
            "private pre-decode abort requires OPERATOR_ABORT"
        )
    return _abort_active(
        authorization, transaction_token, reason_code=reason_code
    )


def _recover_abort(access: Mapping[str, Any], *, reason_code: str) -> None:
    if not os.path.lexists(HOLDOUT_ABORT_RECEIPT_PATH):
        write_json_exclusive_durable(
            HOLDOUT_ABORT_RECEIPT_PATH,
            _abort_receipt_payload(access, reason_code=reason_code),
        )


def recover_protected_holdout_after_crash() -> ProtectedHoldoutStateV1:
    """Permanently burn every unterminated attempt; never decode or seal."""

    try:
        lock_fd = _open_lock(create=False)
    except _LeaseBusy:
        return inspect_protected_holdout_state()
    if lock_fd is None:
        return inspect_protected_holdout_state()
    try:
        presence = _semantic_presence()
        if not presence["access"]:
            return inspect_protected_holdout_state()
        try:
            packet = _read_and_validate_packet()
            access = _validate_access_receipt(
                _read_canonical_semantic_json(HOLDOUT_ACCESS_RECEIPT_PATH), packet
            )
        except ProtectedHoldoutError:
            return inspect_protected_holdout_state()
        if presence["seal"] or presence["abort"]:
            return inspect_protected_holdout_state()
        if presence["result"]:
            _recover_abort(access, reason_code="CORRUPT_BURNED")
        else:
            _recover_abort(access, reason_code="CRASH_RECOVERY_INCOMPLETE")
    finally:
        _close_lock(lock_fd)
    return inspect_protected_holdout_state()


__all__ = [
    "ABORTED",
    "BURNED_INCOMPLETE",
    "CORRUPT_BURNED",
    "OPENING_ACTIVE",
    "RESULT_DURABLE_PENDING_ABORT",
    "SEALED",
    "UNOPENED",
    "ProtectedHoldoutError",
    "ProtectedHoldoutStateV1",
    "execute_protected_holdout_once",
    "inspect_protected_holdout_state",
    "recover_protected_holdout_after_crash",
    "write_json_exclusive_durable",
]
