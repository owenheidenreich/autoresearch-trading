"""Irreversible one-shot evidence access transactions for Path-D research.

The evidence gate is deliberately small and append-only.  An access receipt is
durably created with ``O_EXCL`` before a dataset loader may claim the sole decode
capability.  A successful run appends a dataset receipt, result, and terminal
result receipt while retaining the live lease.  Any lease loss before that
terminal receipt permanently burns the scope, even if dataset/result bytes are
already durable; recovery never recreates an authorization, decodes evidence,
or seals a result.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterator
import uuid

from v4.research import pathd_entry_exit as _foundation


REPO_ROOT = _foundation.REPO_ROOT
ENTRY_FOLD_ARTIFACT_ROOT = _foundation.ENTRY_FOLD_ARTIFACT_ROOT
PREREG_PATH = _foundation.PREREG_PATH
SESSION_PATH = _foundation.SESSION_PATH
PLAN_PATH = _foundation.PLAN_PATH
CORPUS_INTEGRITY_RECEIPT_PATH = _foundation.CORPUS_INTEGRITY_RECEIPT_PATH
LINEAGE_IMPLEMENTATION_RECEIPT_PATH = _foundation.LINEAGE_IMPLEMENTATION_RECEIPT_PATH
ENTRY_MACHINERY_RECEIPT_PATH = _foundation.ENTRY_MACHINERY_RECEIPT_PATH
CORPUS_ROOT = _foundation.CORPUS_ROOT

_prepare_entry_evidence_authorization_claim = (
    _foundation._prepare_entry_evidence_authorization_claim
)

_ROLES = frozenset(
    {
        "nested_validation",
        "outer_test_primary",
        "outer_test_shortened_diagnostic",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T[^\s]+Z")
_UUID4 = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)

UNOPENED = "UNOPENED"
OPEN_ACTIVE = "OPEN_ACTIVE"
OPEN_PENDING_RESULT = "OPEN_PENDING_RESULT"
DURABLE_UNSEALED_BURN_REQUIRED = "DURABLE_UNSEALED_BURN_REQUIRED"
SEALED = "SEALED"
BURNED = "BURNED"
INVALID_SKIPPED = "INVALID_SKIPPED"


class EntryEvidenceGateError(RuntimeError):
    """Fail-closed evidence transaction error."""


@dataclass(frozen=True)
class _ScopePaths:
    role: str
    outer_fold: int
    inner_fold: int | None
    lock: Path
    preopen: Path
    access: Path
    dataset_receipt: Path
    result: Path
    result_receipt: Path
    burned_receipt: Path
    skip_receipt: Path | None


@dataclass
class _ActiveCapability:
    pid: int
    lock_fd: int
    authorization: _foundation.FrozenEvidenceAuthorization
    authorization_sha256: str
    state: str


_ACTIVE_CAPABILITIES: dict[int, _ActiveCapability] = {}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value, newline=False)).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise EntryEvidenceGateError("entry evidence blocked: non-string JSON key")
        return {key: _jsonable(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_jsonable(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise EntryEvidenceGateError("entry evidence blocked: non-finite JSON number")
        return value
    raise EntryEvidenceGateError(
        f"entry evidence blocked: unsupported JSON value {type(value).__name__}"
    )


def _canonical_bytes(value: Any, *, newline: bool = True) -> bytes:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EntryEvidenceGateError(
                f"entry evidence blocked: duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EntryEvidenceGateError(
        f"entry evidence blocked: non-finite JSON constant {value!r}"
    )


def _read_bytes_nofollow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EntryEvidenceGateError(
            f"entry evidence blocked: cannot open fixed artifact {path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise EntryEvidenceGateError(
                f"entry evidence blocked: fixed artifact is not regular: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_canonical_json(path: Path) -> dict[str, Any]:
    raw = _read_bytes_nofollow(path)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntryEvidenceGateError(
            f"entry evidence blocked: malformed canonical JSON {path}"
        ) from exc
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise EntryEvidenceGateError(
            f"entry evidence blocked: noncanonical JSON bytes {path}"
        )
    return value


def _sha256_nofollow(path: Path) -> str:
    return hashlib.sha256(_read_bytes_nofollow(path)).hexdigest()


def _ensure_directory(path: Path) -> None:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise EntryEvidenceGateError(
            "entry evidence blocked: transaction path escaped repository root"
        ) from exc
    current = REPO_ROOT
    if not current.exists() or current.is_symlink() or not current.is_dir():
        raise EntryEvidenceGateError(
            "entry evidence blocked: repository root is not a trusted directory"
        )
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            parent_fd = os.open(
                current.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.mkdir(current, 0o700)
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise EntryEvidenceGateError(
                f"entry evidence blocked: untrusted transaction directory {current}"
            )


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> str:
    _ensure_directory(path.parent)
    raw = _canonical_bytes(value)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise EntryEvidenceGateError(
            f"entry evidence blocked: immutable artifact already exists: {path}"
        ) from exc
    except OSError as exc:
        raise EntryEvidenceGateError(
            f"entry evidence blocked: exclusive artifact creation failed: {path}"
        ) from exc
    complete = False
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
        complete = True
    finally:
        os.close(descriptor)
    if not complete:
        raise EntryEvidenceGateError(
            f"entry evidence blocked: durable artifact write failed: {path}"
        )
    _fsync_parent(path)
    return hashlib.sha256(raw).hexdigest()


def _path_label(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise EntryEvidenceGateError(
            "entry evidence blocked: fixed path is outside repository root"
        ) from exc


def _validate_identity(
    *, role: str, outer_fold: int | None, inner_fold: int | None
) -> tuple[int, int | None]:
    if role not in _ROLES:
        raise EntryEvidenceGateError(
            "entry evidence blocked: unsupported raw evidence role"
        )
    if type(outer_fold) is not int or not 1 <= outer_fold <= 5:
        raise EntryEvidenceGateError("entry evidence blocked: invalid outer fold")
    if role == "nested_validation":
        if type(inner_fold) is not int or not 1 <= inner_fold <= 4:
            raise EntryEvidenceGateError("entry evidence blocked: invalid inner fold")
    elif inner_fold is not None:
        raise EntryEvidenceGateError(
            "entry evidence blocked: non-nested evidence cannot name an inner fold"
        )
    return outer_fold, inner_fold


def _scope_paths(
    *, role: str, outer_fold: int | None, inner_fold: int | None
) -> _ScopePaths:
    outer, inner = _validate_identity(
        role=role, outer_fold=outer_fold, inner_fold=inner_fold
    )
    fold_dir = ENTRY_FOLD_ARTIFACT_ROOT / f"fold_{outer}"
    lock_dir = ENTRY_FOLD_ARTIFACT_ROOT / ".evidence_gate_locks"
    if role == "nested_validation":
        stem = f"nested_inner_{inner}"
        return _ScopePaths(
            role=role,
            outer_fold=outer,
            inner_fold=inner,
            lock=lock_dir / f"fold_{outer}_{stem}.lock",
            preopen=fold_dir / f"{stem}_preopen_receipt.json",
            access=fold_dir / f"{stem}_access_receipt.json",
            dataset_receipt=fold_dir / f"{stem}_dataset_receipt.json",
            result=fold_dir / f"{stem}_result.json",
            result_receipt=fold_dir / f"{stem}_result_receipt.json",
            burned_receipt=fold_dir / f"{stem}_burned_receipt.json",
            skip_receipt=fold_dir / f"{stem}_skip_receipt.json",
        )
    if role == "outer_test_primary":
        return _ScopePaths(
            role=role,
            outer_fold=outer,
            inner_fold=None,
            lock=lock_dir / f"fold_{outer}_outer_primary.lock",
            preopen=fold_dir / "preopen_receipt.json",
            access=fold_dir / "outer_primary_access_receipt.json",
            dataset_receipt=fold_dir / "outer_primary_dataset_receipt.json",
            result=fold_dir / "outer_primary_result.json",
            result_receipt=fold_dir / "outer_result_receipt.json",
            burned_receipt=fold_dir / "outer_primary_burned_receipt.json",
            skip_receipt=None,
        )
    return _ScopePaths(
        role=role,
        outer_fold=outer,
        inner_fold=None,
        lock=lock_dir / f"fold_{outer}_shortened.lock",
        preopen=fold_dir / "shortened_preopen_receipt.json",
        access=fold_dir / "shortened_access_receipt.json",
        dataset_receipt=fold_dir / "shortened_dataset_receipt.json",
        result=fold_dir / "shortened_result.json",
        result_receipt=fold_dir / "shortened_result_receipt.json",
        burned_receipt=fold_dir / "shortened_burned_receipt.json",
        skip_receipt=None,
    )


def _acquire_lock(path: Path) -> int:
    _ensure_directory(path.parent)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise EntryEvidenceGateError(
            "entry evidence blocked: cannot open fixed transaction lock"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EntryEvidenceGateError(
                "entry evidence blocked: transaction lock is not regular"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise EntryEvidenceGateError(
                    "entry evidence blocked: evidence transaction is already active"
                ) from exc
            raise
        _fsync_parent(path)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _release_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def _scope_lease(path: Path) -> Iterator[int]:
    descriptor = _acquire_lock(path)
    try:
        yield descriptor
    finally:
        _release_lock(descriptor)


def _claim_to_json(claim: dict[str, Any]) -> dict[str, Any]:
    expected = {
        field.name
        for field in fields(_foundation.FrozenEvidenceAuthorization)
    } - {
        "access_scope_id",
        "access_receipt_path",
        "access_receipt_sha256",
        "transaction_id",
    }
    transitional_foundation_fields = {
        "foundation_generation_sha256",
        "foundation_stability_receipt_sha256",
    } & expected
    # Synthetic pre-correction fixtures may omit both optional generation fields.
    # A corrected official claim carries both; accepting exactly zero-or-both keeps
    # the transition explicit and rejects partially bound foundation identity.
    if set(claim) not in {
        frozenset(expected),
        frozenset(expected - transitional_foundation_fields),
    }:
        raise EntryEvidenceGateError(
            "entry evidence blocked: authorization claim schema drift"
        )
    result = _jsonable(claim)
    if type(result) is not dict:
        raise EntryEvidenceGateError("entry evidence blocked: malformed claim")
    if (
        type(result.get("sessions")) is not list
        or any(type(item) is not str for item in result["sessions"])
        or type(result.get("open_gate_receipts_sha256")) is not list
        or any(
            type(item) is not str or _HEX64.fullmatch(item) is None
            for item in result["open_gate_receipts_sha256"]
        )
    ):
        raise EntryEvidenceGateError(
            "entry evidence blocked: malformed authorization session claim"
        )
    for key in (
        "sessions_sha256_newline",
        "preregistration_sha256",
        "session_assignments_sha256",
        "source_hash_policy_sha256",
        "corpus_integrity_receipt_sha256",
        "lineage_receipt_sha256",
        "machinery_receipt_sha256",
        "fit_environment_sha256",
    ):
        if type(result.get(key)) is not str or _HEX64.fullmatch(result[key]) is None:
            raise EntryEvidenceGateError(
                f"entry evidence blocked: malformed authorization digest {key}"
            )
    for key in (
        "foundation_generation_sha256",
        "foundation_stability_receipt_sha256",
    ):
        if key in result and (
            type(result[key]) is not str or _HEX64.fullmatch(result[key]) is None
        ):
            raise EntryEvidenceGateError(
                f"entry evidence blocked: malformed authorization digest {key}"
            )
    return result


def _scope_id(paths: _ScopePaths, *, claim: dict[str, Any]) -> str:
    """Bind a role/fold scope to the exact frozen campaign generation.

    The generation digest is carried by the immutable authorization claim.  The
    preregistration digest is retained only as a transition fallback for an
    authorization dataclass that predates the corrected foundation-generation
    field; neither branch reads mutable global foundation state.
    """

    claim_json = _claim_to_json(claim)
    foundation_generation_sha256 = claim_json.get(
        "foundation_generation_sha256",
        claim_json["preregistration_sha256"],
    )
    return _stable_hash(
        {
            "schema_version": "pathd.entry_evidence_access_scope.v2",
            "foundation_generation_sha256": foundation_generation_sha256,
            "role": paths.role,
            "outer_fold": paths.outer_fold,
            "inner_fold": paths.inner_fold,
        }
    )


def _access_schema(role: str) -> tuple[str, str, str]:
    if role == "nested_validation":
        return (
            "pathd.entry_nested_access_receipt.v1",
            "OPENED_NESTED_BEFORE_SOURCE_DECODE",
            "validation_access_count",
        )
    if role == "outer_test_primary":
        return (
            "pathd.entry_outer_primary_access_receipt.v1",
            "OPENED_OUTER_PRIMARY_BEFORE_SOURCE_DECODE",
            "outer_evidence_access_count",
        )
    return (
        "pathd.entry_shortened_diagnostic_access_receipt.v1",
        "OPENED_SHORTENED_DIAGNOSTIC_BEFORE_SOURCE_DECODE",
        "diagnostic_access_count",
    )


def _self_hash(value: dict[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in value:
        raise EntryEvidenceGateError("entry evidence blocked: duplicate receipt digest")
    result = dict(value)
    result["receipt_sha256"] = _stable_hash(value)
    return result


def _validate_self_hash(value: dict[str, Any]) -> None:
    digest = value.get("receipt_sha256")
    semantic = dict(value)
    semantic.pop("receipt_sha256", None)
    if type(digest) is not str or _HEX64.fullmatch(digest) is None:
        raise EntryEvidenceGateError("entry evidence blocked: malformed receipt digest")
    if digest != _stable_hash(semantic):
        raise EntryEvidenceGateError("entry evidence blocked: receipt digest drift")


def _access_receipt(
    paths: _ScopePaths, *, claim: dict[str, Any], transaction_id: str
) -> dict[str, Any]:
    schema, status_value, count_key = _access_schema(paths.role)
    claim_json = _claim_to_json(claim)
    return _self_hash(
        {
            "schema_version": schema,
            "status": status_value,
            "created_at_utc": _now_utc(),
            "role": paths.role,
            "outer_fold": paths.outer_fold,
            "inner_fold": paths.inner_fold,
            "access_scope_id": _scope_id(paths, claim=claim_json),
            "transaction_id": transaction_id,
            "authorization_claim": claim_json,
            "authorization_claim_sha256": _stable_hash(claim_json),
            "access_receipt_path": _path_label(paths.access),
            count_key: 1,
            "source_decode_started": False,
            "holdout_open_count": 0,
        }
    )


def _read_access_receipt(paths: _ScopePaths) -> dict[str, Any]:
    value = _read_canonical_json(paths.access)
    schema, status_value, count_key = _access_schema(paths.role)
    expected = {
        "schema_version",
        "status",
        "created_at_utc",
        "role",
        "outer_fold",
        "inner_fold",
        "access_scope_id",
        "transaction_id",
        "authorization_claim",
        "authorization_claim_sha256",
        "access_receipt_path",
        count_key,
        "source_decode_started",
        "holdout_open_count",
        "receipt_sha256",
    }
    claim = value.get("authorization_claim")
    if (
        set(value) != expected
        or value.get("schema_version") != schema
        or value.get("status") != status_value
        or type(value.get("created_at_utc")) is not str
        or _UTC.fullmatch(value["created_at_utc"]) is None
        or value.get("role") != paths.role
        or value.get("outer_fold") != paths.outer_fold
        or value.get("inner_fold") != paths.inner_fold
        or value.get("access_scope_id") != _scope_id(paths, claim=claim)
        or type(value.get("transaction_id")) is not str
        or _UUID4.fullmatch(value["transaction_id"]) is None
        or type(claim) is not dict
        or value.get("authorization_claim_sha256") != _stable_hash(claim)
        or value.get("access_receipt_path") != _path_label(paths.access)
        or type(value.get(count_key)) is not int
        or value.get(count_key) != 1
        or value.get("source_decode_started") is not False
        or type(value.get("holdout_open_count")) is not int
        or value.get("holdout_open_count") != 0
    ):
        raise EntryEvidenceGateError(
            "entry evidence blocked: invalid durable access receipt"
        )
    _claim_to_json(claim)
    _validate_self_hash(value)
    return value


def _authorization_from_access(
    paths: _ScopePaths, access: dict[str, Any]
) -> _foundation.FrozenEvidenceAuthorization:
    claim = access["authorization_claim"]
    optional_foundation_identity = {
        key: claim[key]
        for key in (
            "foundation_generation_sha256",
            "foundation_stability_receipt_sha256",
        )
        if key in claim
        and key
        in {field.name for field in fields(_foundation.FrozenEvidenceAuthorization)}
    }
    return _foundation.FrozenEvidenceAuthorization(
        role=claim["role"],
        outer_fold=claim["outer_fold"],
        inner_fold=claim["inner_fold"],
        sessions=tuple(claim["sessions"]),
        sessions_sha256_newline=claim["sessions_sha256_newline"],
        preregistration_sha256=claim["preregistration_sha256"],
        session_assignments_sha256=claim["session_assignments_sha256"],
        source_hash_policy_sha256=claim["source_hash_policy_sha256"],
        corpus_integrity_receipt_sha256=claim[
            "corpus_integrity_receipt_sha256"
        ],
        lineage_receipt_sha256=claim["lineage_receipt_sha256"],
        machinery_receipt_sha256=claim["machinery_receipt_sha256"],
        fit_environment_sha256=claim["fit_environment_sha256"],
        open_gate_receipts_sha256=tuple(claim["open_gate_receipts_sha256"]),
        access_scope_id=access["access_scope_id"],
        access_receipt_path=_path_label(paths.access),
        access_receipt_sha256=_sha256_nofollow(paths.access),
        transaction_id=access["transaction_id"],
        **optional_foundation_identity,
    )


def _gate_paths(paths: _ScopePaths) -> tuple[Path, ...]:
    fold_dir = ENTRY_FOLD_ARTIFACT_ROOT / f"fold_{paths.outer_fold}"
    if paths.role == "nested_validation":
        prior: list[Path] = []
        assert paths.inner_fold is not None
        for inner in range(1, paths.inner_fold):
            result = fold_dir / f"nested_inner_{inner}_result_receipt.json"
            skip = fold_dir / f"nested_inner_{inner}_skip_receipt.json"
            if result.exists() == skip.exists():
                raise EntryEvidenceGateError(
                    "entry evidence blocked: prior nested terminal receipt ambiguity"
                )
            prior.append(result if result.exists() else skip)
        return tuple([*prior, paths.preopen])
    if paths.role == "outer_test_primary":
        return tuple(
            [
                ENTRY_FOLD_ARTIFACT_ROOT / f"fold_{fold}" / "outer_result_receipt.json"
                for fold in range(1, paths.outer_fold)
            ]
            + [paths.preopen]
        )
    return (
        fold_dir / "outer_result_receipt.json",
        paths.preopen,
    )


def _validate_claim_foundation(
    paths: _ScopePaths, authorization: _foundation.FrozenEvidenceAuthorization
) -> None:
    claim = authorization.to_dict()
    claim.pop("access_scope_id")
    claim.pop("access_receipt_path")
    claim.pop("access_receipt_sha256")
    claim.pop("transaction_id")
    claim_json = _claim_to_json(claim)
    if (
        claim_json["role"] != paths.role
        or claim_json["outer_fold"] != paths.outer_fold
        or claim_json["inner_fold"] != paths.inner_fold
        or authorization.access_scope_id
        != _scope_id(paths, claim=claim_json)
        or authorization.access_receipt_path != _path_label(paths.access)
    ):
        raise EntryEvidenceGateError("entry evidence blocked: claim identity drift")
    from v4.research.pathd_holdout_gate import _assert_protected_holdout_unopened

    _assert_protected_holdout_unopened()
    stability_receipt = _foundation.assert_research_foundation_stable()
    if (
        claim_json.get("foundation_generation_sha256")
        != stability_receipt.get("foundation_generation_sha256")
        or claim_json.get("foundation_stability_receipt_sha256")
        != _sha256_nofollow(_foundation.FOUNDATION_STABILITY_RECEIPT_PATH)
    ):
        raise EntryEvidenceGateError(
            "entry evidence blocked: global foundation-stability identity drift"
        )
    preregistration_receipt = _foundation.assert_preregistration_frozen()
    payload = _foundation.read_json(PREREG_PATH)
    _foundation._verify_immutable_sources(payload)
    lineage_receipt = _foundation.assert_lineage_implementation_frozen()
    machinery_receipt = _foundation._assert_entry_machinery_frozen(
        preregistration_receipt, payload, lineage_receipt
    )
    _foundation._verify_authorized_dependency_closure(payload)
    _foundation._validated_corpus_integrity_receipt(
        payload, preregistration_receipt, machinery_receipt
    )
    if (
        preregistration_receipt.get("preregistration_sha256")
        != claim_json["preregistration_sha256"]
    ):
        raise EntryEvidenceGateError(
            "entry evidence blocked: preregistration receipt identity drift"
        )
    fixed_hashes = {
        "preregistration_sha256": PREREG_PATH,
        "session_assignments_sha256": SESSION_PATH,
        "corpus_integrity_receipt_sha256": CORPUS_INTEGRITY_RECEIPT_PATH,
        "lineage_receipt_sha256": LINEAGE_IMPLEMENTATION_RECEIPT_PATH,
        "machinery_receipt_sha256": ENTRY_MACHINERY_RECEIPT_PATH,
    }
    for key, path in fixed_hashes.items():
        if not path.exists() or _sha256_nofollow(path) != claim_json[key]:
            raise EntryEvidenceGateError(
                f"entry evidence blocked: frozen claim artifact drift: {key}"
            )
    if _stable_hash(payload["source_hash_policy"]) != claim_json[
        "source_hash_policy_sha256"
    ]:
        raise EntryEvidenceGateError("entry evidence blocked: source policy drift")
    assignments = _foundation.read_json(SESSION_PATH)
    expected_sessions = _foundation._resolve_frozen_evidence_sessions(
        assignments,
        role=paths.role,
        outer_fold=paths.outer_fold,
        inner_fold=paths.inner_fold,
    )
    if (
        tuple(expected_sessions) != authorization.sessions
        or _foundation.canonical_session_hash(expected_sessions)
        != authorization.sessions_sha256_newline
    ):
        raise EntryEvidenceGateError("entry evidence blocked: evidence sessions drift")
    gates = _gate_paths(paths)
    if any(not path.exists() for path in gates):
        raise EntryEvidenceGateError("entry evidence blocked: open-gate receipt absent")
    if tuple(_sha256_nofollow(path) for path in gates) != (
        authorization.open_gate_receipts_sha256
    ):
        raise EntryEvidenceGateError("entry evidence blocked: open-gate hash drift")
    environment = _foundation.assert_entry_fit_environment_current()
    if _stable_hash(environment) != authorization.fit_environment_sha256:
        raise EntryEvidenceGateError("entry evidence blocked: fit environment drift")


def _assert_scope_order(paths: _ScopePaths) -> None:
    if paths.role == "nested_validation":
        assert paths.inner_fold is not None
        fold_dir = paths.preopen.parent
        for inner in range(1, paths.inner_fold):
            result = fold_dir / f"nested_inner_{inner}_result_receipt.json"
            skip = fold_dir / f"nested_inner_{inner}_skip_receipt.json"
            if result.exists() == skip.exists():
                raise EntryEvidenceGateError(
                    "entry evidence blocked: nested blocks must terminate in order"
                )
        for inner in range(paths.inner_fold + 1, 5):
            prefix = f"nested_inner_{inner}_"
            if fold_dir.exists() and any(
                item.name.startswith(prefix) for item in fold_dir.iterdir()
            ):
                raise EntryEvidenceGateError(
                    "entry evidence blocked: later nested block exists prematurely"
                )
    elif paths.role == "outer_test_primary":
        fold_dir = paths.preopen.parent
        for inner in range(1, 5):
            result = fold_dir / f"nested_inner_{inner}_result_receipt.json"
            skip = fold_dir / f"nested_inner_{inner}_skip_receipt.json"
            if result.exists() == skip.exists():
                raise EntryEvidenceGateError(
                    "entry evidence blocked: outer primary requires four nested terminals"
                )
    else:
        primary = paths.preopen.parent / "outer_result_receipt.json"
        if not primary.exists():
            raise EntryEvidenceGateError(
                "entry evidence blocked: shortened diagnostic precedes primary result"
            )


def _assert_no_later_artifacts(paths: _ScopePaths) -> None:
    """Prevent outcomes opened later in the frozen chronology from leaking back."""

    if paths.role == "nested_validation":
        assert paths.inner_fold is not None
        fold_dir = paths.preopen.parent
        for inner in range(paths.inner_fold + 1, 5):
            prefix = f"nested_inner_{inner}_"
            if fold_dir.exists() and any(
                item.name.startswith(prefix) for item in fold_dir.iterdir()
            ):
                raise EntryEvidenceGateError(
                    "entry evidence blocked: later nested artifact appeared after open"
                )
        future_outer_names = {
            "preopen_receipt.json",
            "outer_primary_access_receipt.json",
            "outer_primary_dataset_receipt.json",
            "outer_primary_result.json",
            "outer_result_receipt.json",
            "outer_primary_burned_receipt.json",
            "negative_control_panel.json",
            "control_replay_result.json",
        }
        if fold_dir.exists() and any(
            item.name in future_outer_names or item.name.startswith("shortened_")
            for item in fold_dir.iterdir()
        ):
            raise EntryEvidenceGateError(
                "entry evidence blocked: outer/diagnostic artifact appeared during nested evidence"
            )
    elif paths.role == "outer_test_primary":
        fold_dir = paths.preopen.parent
        if fold_dir.exists() and any(
            item.name.startswith("shortened_") for item in fold_dir.iterdir()
        ):
            raise EntryEvidenceGateError(
                "entry evidence blocked: shortened diagnostic appeared before primary seal"
            )
    for outer in range(paths.outer_fold + 1, 6):
        later_dir = ENTRY_FOLD_ARTIFACT_ROOT / f"fold_{outer}"
        if later_dir.exists() and any(item.is_file() for item in later_dir.iterdir()):
            raise EntryEvidenceGateError(
                "entry evidence blocked: later outer-fold artifact appeared after open"
            )


def _outcome_paths(paths: _ScopePaths) -> tuple[Path, ...]:
    values = [
        paths.access,
        paths.dataset_receipt,
        paths.result,
        paths.result_receipt,
        paths.burned_receipt,
    ]
    if paths.skip_receipt is not None:
        values.append(paths.skip_receipt)
    if paths.role == "outer_test_primary":
        values.extend(
            [
                paths.preopen.parent / "negative_control_panel.json",
                paths.preopen.parent / "control_replay_result.json",
            ]
        )
    return tuple(values)


def _assert_unopened(paths: _ScopePaths) -> None:
    existing = [path.name for path in _outcome_paths(paths) if path.exists()]
    if existing:
        raise EntryEvidenceGateError(
            f"entry evidence blocked: scope is not pristine: {sorted(existing)}"
        )


def begin_entry_evidence_once(
    *,
    role: str,
    outer_fold: int | None = None,
    inner_fold: int | None = None,
) -> _foundation.FrozenEvidenceAuthorization:
    """Open exactly one frozen evidence scope and retain its live lease."""

    paths = _scope_paths(
        role=role, outer_fold=outer_fold, inner_fold=inner_fold
    )
    lock_fd = _acquire_lock(paths.lock)
    keep_lock = False
    try:
        _assert_scope_order(paths)
        _assert_no_later_artifacts(paths)
        _assert_unopened(paths)
        claim = _prepare_entry_evidence_authorization_claim(
            role=role, outer_fold=outer_fold, inner_fold=inner_fold
        )
        claim_json = _claim_to_json(claim)
        transaction_id = str(uuid.uuid4())
        access = _access_receipt(
            paths, claim=claim_json, transaction_id=transaction_id
        )
        _write_exclusive_json(paths.access, access)
        authorization = _authorization_from_access(paths, access)
        authorization_sha = _stable_hash(authorization.to_dict())
        capability_key = id(authorization)
        if capability_key in _ACTIVE_CAPABILITIES:
            raise EntryEvidenceGateError(
                "entry evidence blocked: process capability collision"
            )
        _ACTIVE_CAPABILITIES[capability_key] = _ActiveCapability(
            pid=os.getpid(),
            lock_fd=lock_fd,
            authorization=authorization,
            authorization_sha256=authorization_sha,
            state="ISSUED_BEFORE_DECODE",
        )
        keep_lock = True
        return authorization
    finally:
        if not keep_lock:
            _release_lock(lock_fd)


def read_frozen_entry_evidence_authorization(
    *,
    role: str,
    outer_fold: int | None = None,
    inner_fold: int | None = None,
) -> _foundation.FrozenEvidenceAuthorization:
    """Read an already-open authorization without creating a live capability."""

    paths = _scope_paths(
        role=role, outer_fold=outer_fold, inner_fold=inner_fold
    )
    access = _read_access_receipt(paths)
    authorization = _authorization_from_access(paths, access)
    _validate_claim_foundation(paths, authorization)
    return authorization


def _require_active_capability_before_scope_io(
    authorization: _foundation.FrozenEvidenceAuthorization,
    /,
) -> _ActiveCapability:
    """Authenticate the exact in-process capability without touching fixed paths."""

    capability = _ACTIVE_CAPABILITIES.get(id(authorization))
    if (
        capability is None
        or capability.pid != os.getpid()
        or capability.authorization is not authorization
    ):
        raise EntryEvidenceGateError(
            "entry evidence blocked: no active process-local decode capability"
        )
    authorization_sha = _stable_hash(authorization.to_dict())
    if capability.authorization_sha256 != authorization_sha:
        raise EntryEvidenceGateError(
            "entry evidence blocked: no active process-local decode capability"
        )
    if capability.state not in {"ISSUED_BEFORE_DECODE", "DECODE_CLAIMED"}:
        raise EntryEvidenceGateError(
            "entry evidence blocked: evidence capability is not active"
        )
    return capability


def validate_entry_evidence_access(
    authorization: _foundation.FrozenEvidenceAuthorization,
    /,
) -> _foundation.FrozenEvidenceAuthorization:
    """Revalidate one live capability without consuming another dataset decode."""

    if type(authorization) is not _foundation.FrozenEvidenceAuthorization:
        raise EntryEvidenceGateError(
            "entry evidence blocked: untrusted evidence authorization type"
        )
    _require_active_capability_before_scope_io(authorization)
    paths = _scope_paths(
        role=authorization.role,
        outer_fold=authorization.outer_fold,
        inner_fold=authorization.inner_fold,
    )
    try:
        current = read_frozen_entry_evidence_authorization(
            role=authorization.role,
            outer_fold=authorization.outer_fold,
            inner_fold=authorization.inner_fold,
        )
    except BaseException:
        try:
            _write_burned(
                paths,
                reason="FROZEN_FOUNDATION_DRIFT_BEFORE_DECODE",
                transaction_id=authorization.transaction_id,
                access_scope_id=authorization.access_scope_id,
            )
        finally:
            _release_transaction_capabilities(authorization.transaction_id)
        raise
    if current != authorization:
        try:
            _write_burned(
                paths,
                reason="AUTHORIZATION_DRIFT_BEFORE_DECODE",
                transaction_id=authorization.transaction_id,
                access_scope_id=authorization.access_scope_id,
            )
        finally:
            _release_transaction_capabilities(authorization.transaction_id)
        raise EntryEvidenceGateError(
            "entry evidence blocked: frozen evidence authorization drift"
        )
    try:
        _assert_no_later_artifacts(paths)
    except BaseException:
        try:
            _write_burned(
                paths,
                reason="LATER_EVIDENCE_APPEARED_DURING_ACTIVE_TRANSACTION",
                transaction_id=authorization.transaction_id,
                access_scope_id=authorization.access_scope_id,
            )
        finally:
            _release_transaction_capabilities(authorization.transaction_id)
        raise
    if paths.burned_receipt.exists() or paths.result_receipt.exists():
        _release_transaction_capabilities(authorization.transaction_id)
        raise EntryEvidenceGateError(
            "entry evidence blocked: evidence transaction is already terminal"
        )
    return authorization


def claim_entry_evidence_decode_once(
    authorization: _foundation.FrozenEvidenceAuthorization,
    /,
) -> _foundation.FrozenEvidenceAuthorization:
    """Private loader seam: consume the sole raw-dataset decode permission."""

    current = validate_entry_evidence_access(authorization)
    capability = _ACTIVE_CAPABILITIES.get(id(authorization))
    if (
        capability is None
        or capability.authorization is not authorization
        or capability.pid != os.getpid()
    ):
        raise EntryEvidenceGateError(
            "entry evidence blocked: no active process-local decode capability"
        )
    if capability.state != "ISSUED_BEFORE_DECODE":
        raise EntryEvidenceGateError(
            "entry evidence blocked: evidence dataset decode already consumed"
        )
    capability.state = "DECODE_CLAIMED"
    return current


def _release_capability(capability_key: int) -> None:
    capability = _ACTIVE_CAPABILITIES.pop(capability_key, None)
    if capability is not None:
        _release_lock(capability.lock_fd)


def _release_transaction_capabilities(transaction_id: str) -> None:
    keys = [
        key
        for key, capability in _ACTIVE_CAPABILITIES.items()
        if capability.authorization.transaction_id == transaction_id
    ]
    for key in keys:
        _release_capability(key)


def _prepare_invalid_nested_skip_claim(
    *, outer_fold: int, inner_fold: int
) -> dict[str, Any]:
    from v4.research.pathd_holdout_gate import _assert_protected_holdout_unopened

    _assert_protected_holdout_unopened()
    _foundation.assert_research_foundation_stable()
    prereg_receipt = _foundation.assert_preregistration_frozen()
    payload = _foundation.read_json(PREREG_PATH)
    assignments = _foundation.read_json(SESSION_PATH)
    _foundation._verify_immutable_sources(payload)
    lineage = _foundation.assert_lineage_implementation_frozen()
    machinery = _foundation._assert_entry_machinery_frozen(
        prereg_receipt, payload, lineage
    )
    _foundation._verify_authorized_dependency_closure(payload)
    _foundation._validated_corpus_integrity_receipt(
        payload, prereg_receipt, machinery
    )
    role = _foundation._nested_role_record(assignments, outer_fold, inner_fold)
    if role.get("calibration_valid") is not False:
        raise EntryEvidenceGateError(
            "entry evidence blocked: calibration-valid nested block cannot skip"
        )
    for prior in range(1, inner_fold):
        _foundation._validate_nested_block_receipt_one(
            payload, assignments, outer_fold, prior
        )
    return {
        "payload": payload,
        "role": role,
        "preregistration_sha256": prereg_receipt["preregistration_sha256"],
        "session_assignments_sha256": _sha256_nofollow(SESSION_PATH),
        "source_hash_policy_sha256": _stable_hash(payload["source_hash_policy"]),
    }


def seal_invalid_nested_block_skip(
    *, outer_fold: int, inner_fold: int
) -> dict[str, Any]:
    """Seal one preregistered calibration-invalid nested block without decode."""

    paths = _scope_paths(
        role="nested_validation", outer_fold=outer_fold, inner_fold=inner_fold
    )
    with _scope_lease(paths.lock):
        _assert_scope_order(paths)
        _assert_no_later_artifacts(paths)
        _assert_unopened(paths)
        prefix = f"nested_inner_{inner_fold}_"
        if paths.preopen.parent.exists() and any(
            item.name.startswith(prefix) for item in paths.preopen.parent.iterdir()
        ):
            raise EntryEvidenceGateError(
                "entry evidence blocked: invalid nested block has decode/fit artifacts"
            )
        prepared = _prepare_invalid_nested_skip_claim(
            outer_fold=outer_fold, inner_fold=inner_fold
        )
        role = prepared["role"]
        payload = prepared["payload"]
        receipt = _self_hash(
            {
                "schema_version": "pathd.entry_nested_insufficient_calibration_skip.v1",
                "status": "SKIPPED_INSUFFICIENT_CALIBRATION",
                "frozen_at_utc": _now_utc(),
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "preregistration_sha256": prepared["preregistration_sha256"],
                "session_assignments_sha256": prepared[
                    "session_assignments_sha256"
                ],
                "source_hash_policy_sha256": prepared[
                    "source_hash_policy_sha256"
                ],
                "weights_sessions_sha256_newline": role[
                    "model_fit_sha256_newline"
                ],
                "calibration_sessions_sha256_newline": role[
                    "calibration_sha256_newline"
                ],
                "validation_sessions_sha256_newline": role[
                    "validation_sha256_newline"
                ],
                "calibration_session_count": len(role["calibration"]),
                "calibration_minimum_sessions": role[
                    "calibration_minimum_sessions"
                ],
                "reason": "CALIBRATION_SESSIONS_LT_10",
                "validation_access_count": 0,
                "dataset_opened": False,
                "result_generated": False,
                "quarantine_labels": list(_foundation.QUARANTINE_LABELS),
                "claim_boundary": _foundation.CLAIM_BOUNDARY,
                "holdout_caveat": _foundation.HOLDOUT_CAVEAT,
                "plan_sha256": payload["binding_plan"]["sha256"],
                "fill_law_hash": payload["fill_law"]["fill_law_hash"],
                "holdout_open_count": 0,
            }
        )
        _write_exclusive_json(paths.skip_receipt, receipt)  # type: ignore[arg-type]
        return receipt


def _validate_exact_dataset(
    dataset: Any, *, authorization: _foundation.FrozenEvidenceAuthorization
) -> None:
    from v4.research.pathd_entry_dataset import (
        EntryEvidenceDatasetV1,
        validate_entry_evidence_dataset,
    )

    if type(dataset) is not EntryEvidenceDatasetV1:
        raise EntryEvidenceGateError(
            "entry evidence blocked: result requires exact EntryEvidenceDatasetV1"
        )
    validate_entry_evidence_dataset(dataset, authorization=authorization)


def _validate_exact_evaluation(
    evaluation: Any,
    *,
    authorization: _foundation.FrozenEvidenceAuthorization,
    dataset: Any,
) -> None:
    from v4.scripts.run_pathd_entry_exit_research import (
        EntryNestedFamilyEvaluationV1,
        EntryOuterPrimaryResultV1,
        EntryPolicyEvaluationV1,
        validate_entry_nested_family_evaluation,
        validate_entry_outer_primary_result,
        validate_entry_policy_evaluation,
    )

    if authorization.role == "nested_validation":
        if type(evaluation) is not EntryNestedFamilyEvaluationV1:
            raise EntryEvidenceGateError(
                "entry evidence blocked: nested result type drift"
            )
        validate_entry_nested_family_evaluation(
            evaluation, authorization=authorization, dataset=dataset
        )
    elif authorization.role == "outer_test_primary":
        if type(evaluation) is not EntryOuterPrimaryResultV1:
            raise EntryEvidenceGateError(
                "entry evidence blocked: outer result type drift"
            )
        validate_entry_outer_primary_result(
            evaluation, authorization=authorization, dataset=dataset
        )
    else:
        if type(evaluation) is not EntryPolicyEvaluationV1:
            raise EntryEvidenceGateError(
                "entry evidence blocked: shortened diagnostic type drift"
            )
        validate_entry_policy_evaluation(
            evaluation, authorization=authorization
        )


def _dataset_receipt(
    dataset: Any, *, authorization: _foundation.FrozenEvidenceAuthorization
) -> dict[str, Any]:
    source_receipts = _jsonable(dataset.source_receipts)
    if type(source_receipts) is not list:
        raise EntryEvidenceGateError(
            "entry evidence blocked: evidence source receipts are malformed"
        )
    sessions = list(authorization.sessions)
    grouped: dict[str, list[str]] = {session: [] for session in sessions}
    ordered: list[str] = []
    for example in dataset.examples:
        session = getattr(getattr(example, "model_input", None), "session", None)
        digest = getattr(example, "canonical_sha256", None)
        if (
            type(session) is not str
            or session not in grouped
            or type(digest) is not str
            or _HEX64.fullmatch(digest) is None
        ):
            raise EntryEvidenceGateError(
                "entry evidence blocked: evidence example identity is malformed"
            )
        grouped[session].append(digest)
        ordered.append(digest)
    if any(not grouped[session] for session in sessions):
        raise EntryEvidenceGateError(
            "entry evidence blocked: dataset omits an authorization session"
        )
    if [row.get("session") for row in source_receipts] != sessions:
        raise EntryEvidenceGateError(
            "entry evidence blocked: source receipt session order drift"
        )
    context = _result_context()
    authorization_sha = _stable_hash(authorization.to_dict())
    receipt = _self_hash(
        {
            "schema_version": "pathd.entry_evidence_dataset_receipt.v1",
            "status": "SEALED_BEFORE_RESULT",
            "created_at_utc": _now_utc(),
            "role": authorization.role,
            "outer_fold": authorization.outer_fold,
            "inner_fold": authorization.inner_fold,
            "authorization_sha256": authorization_sha,
            "sessions": sessions,
            "sessions_sha256_newline": authorization.sessions_sha256_newline,
            "source_receipts": source_receipts,
            "source_receipts_root_sha256": _stable_hash(source_receipts),
            "session_example_hashes": [
                {"session": session, "ordered_example_hashes": grouped[session]}
                for session in sessions
            ],
            "ordered_example_hashes": ordered,
            "dataset_sha256": dataset.dataset_sha256,
            "quarantine_labels": list(_foundation.QUARANTINE_LABELS),
            "claim_boundary": _foundation.CLAIM_BOUNDARY,
            "holdout_caveat": _foundation.HOLDOUT_CAVEAT,
            "plan_sha256": context["plan_sha256"],
            "preregistration_sha256": context["preregistration_sha256"],
            "fill_law_hash": context["fill_law_hash"],
            "holdout_open_count": 0,
        }
    )
    return receipt


def _rehash_dataset_source_files(dataset_receipt: dict[str, Any]) -> None:
    """Rehash every sealed source byte without decoding any market-data row."""

    source_receipts = dataset_receipt.get("source_receipts")
    if type(source_receipts) is not list:
        raise EntryEvidenceGateError(
            "entry evidence blocked: source receipt list is malformed"
        )
    try:
        raw_root_metadata = CORPUS_ROOT.lstat()
        corpus_root = CORPUS_ROOT.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EntryEvidenceGateError(
            "entry evidence blocked: frozen corpus root is absent"
        ) from exc
    if stat.S_ISLNK(raw_root_metadata.st_mode) or not corpus_root.is_dir():
        raise EntryEvidenceGateError(
            "entry evidence blocked: frozen corpus root is untrusted"
        )
    seen: dict[str, tuple[int, str]] = {}
    for receipt in source_receipts:
        source_files = receipt.get("source_files") if type(receipt) is dict else None
        if type(source_files) is not list:
            raise EntryEvidenceGateError(
                "entry evidence blocked: source file receipt is malformed"
            )
        for row in source_files:
            if (
                type(row) is not dict
                or set(row) != {"relative_path", "bytes", "sha256"}
                or type(row.get("relative_path")) is not str
                or not row["relative_path"]
                or row["relative_path"].startswith("/")
                or "\\" in row["relative_path"]
                or any(part in ("", ".", "..") for part in row["relative_path"].split("/"))
                or type(row.get("bytes")) is not int
                or row["bytes"] < 0
                or type(row.get("sha256")) is not str
                or _HEX64.fullmatch(row["sha256"]) is None
            ):
                raise EntryEvidenceGateError(
                    "entry evidence blocked: noncanonical source file receipt"
                )
            relative = row["relative_path"]
            identity = (row["bytes"], row["sha256"])
            if relative in seen and seen[relative] != identity:
                raise EntryEvidenceGateError(
                    "entry evidence blocked: conflicting source file receipts"
                )
            seen[relative] = identity
    for relative, (expected_bytes, expected_sha) in sorted(seen.items()):
        candidate = corpus_root
        for part in relative.split("/"):
            candidate = candidate / part
            try:
                metadata = candidate.lstat()
            except FileNotFoundError as exc:
                raise EntryEvidenceGateError(
                    f"entry evidence blocked: sealed source is absent: {relative}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise EntryEvidenceGateError(
                    f"entry evidence blocked: sealed source is symlinked: {relative}"
                )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(corpus_root)
        except (FileNotFoundError, ValueError) as exc:
            raise EntryEvidenceGateError(
                f"entry evidence blocked: sealed source escaped corpus: {relative}"
            ) from exc
        metadata = resolved.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != expected_bytes
            or _sha256_nofollow(resolved) != expected_sha
        ):
            raise EntryEvidenceGateError(
                f"entry evidence blocked: sealed source bytes drifted: {relative}"
            )


def _result_context() -> dict[str, str]:
    payload = _foundation.read_json(PREREG_PATH)
    return {
        "plan_sha256": payload["binding_plan"]["sha256"],
        "preregistration_sha256": _sha256_nofollow(PREREG_PATH),
        "fill_law_hash": payload["fill_law"]["fill_law_hash"],
    }


def _build_result_document(
    evaluation: Any,
    *,
    authorization: _foundation.FrozenEvidenceAuthorization,
    dataset: Any,
) -> dict[str, Any]:
    if authorization.role != "outer_test_primary":
        value = _jsonable(evaluation)
        if type(value) is not dict:
            raise EntryEvidenceGateError("entry evidence blocked: result is not an object")
        if value.get("holdout_caveat") != _foundation.HOLDOUT_CAVEAT:
            raise EntryEvidenceGateError(
                "entry evidence blocked: result payload holdout caveat drift"
            )
        return value
    from v4.path_d.execution.research_fill_law import (
        research_fill_law_from_preregistration,
    )
    from v4.path_d.execution.research_replay import (
        ResearchResultEnvelopeV1,
        seal_research_result,
    )

    payload = _jsonable(evaluation)
    fill_law = research_fill_law_from_preregistration(
        _foundation.read_json(PREREG_PATH)
    )
    envelope = seal_research_result(
        authorization=authorization,
        dataset=dataset,
        fill_law=fill_law,
        payload=payload,
    )
    if type(envelope) is not ResearchResultEnvelopeV1:
        raise EntryEvidenceGateError(
            "entry evidence blocked: result envelope type drift"
        )
    value = _jsonable(envelope)
    if type(value) is not dict:
        raise EntryEvidenceGateError("entry evidence blocked: envelope is not an object")
    return value


def _validate_result_document_binding(
    value: dict[str, Any],
    *,
    authorization: _foundation.FrozenEvidenceAuthorization,
    dataset_sha256: str,
) -> None:
    authorization_sha = _stable_hash(authorization.to_dict())
    if authorization.role == "nested_validation":
        expected = set(
            _foundation.entry_future_api_contract()["dataclass_fields"][
                "EntryNestedFamilyEvaluationV1"
            ]
        )
        semantic = dict(value)
        digest = semantic.pop("result_sha256", None)
        if (
            set(value) != expected
            or value.get("schema_version")
            != "pathd.entry_nested_family_evaluation.v1"
            or value.get("holdout_caveat") != _foundation.HOLDOUT_CAVEAT
            or value.get("outer_fold") != authorization.outer_fold
            or value.get("inner_fold") != authorization.inner_fold
            or value.get("authorization_sha256") != authorization_sha
            or value.get("dataset_sha256") != dataset_sha256
            or value.get("access_receipt_sha256")
            != authorization.access_receipt_sha256
            or digest != _stable_hash(semantic)
        ):
            raise EntryEvidenceGateError(
                "entry evidence blocked: nested result binding drift"
            )
    elif authorization.role == "outer_test_primary":
        envelope_fields = set(
            _foundation.entry_future_api_contract()["dataclass_fields"][
                "ResearchResultEnvelopeV1"
            ]
        )
        payload = value.get("payload")
        if (
            set(value) != envelope_fields
            or type(payload) is not dict
            or value.get("authorization_sha256") != authorization_sha
            or value.get("sessions_sha256_newline")
            != authorization.sessions_sha256_newline
            or value.get("dataset_sha256") != dataset_sha256
            or value.get("payload_sha256") != _stable_hash(payload)
            or payload.get("schema_version")
            != "pathd.entry_outer_primary_result.v1"
            or payload.get("authorization_sha256") != authorization_sha
            or payload.get("dataset_sha256") != dataset_sha256
            or payload.get("access_receipt_sha256")
            != authorization.access_receipt_sha256
        ):
            raise EntryEvidenceGateError(
                "entry evidence blocked: outer result binding drift"
            )
    else:
        expected = set(
            _foundation.entry_future_api_contract()["dataclass_fields"][
                "EntryPolicyEvaluationV1"
            ]
        )
        semantic = dict(value)
        digest = semantic.pop("result_sha256", None)
        if (
            set(value) != expected
            or value.get("schema_version") != "pathd.entry_policy_evaluation.v1"
            or value.get("holdout_caveat") != _foundation.HOLDOUT_CAVEAT
            or value.get("evidence_role")
            != "outer_test_shortened_diagnostic"
            or value.get("authorization_sha256") != authorization_sha
            or value.get("dataset_sha256") != dataset_sha256
            or digest != _stable_hash(semantic)
        ):
            raise EntryEvidenceGateError(
                "entry evidence blocked: shortened result binding drift"
            )


def _result_receipt(
    paths: _ScopePaths,
    *,
    authorization: _foundation.FrozenEvidenceAuthorization,
    dataset_receipt: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    context = _result_context()
    authorization_sha = _stable_hash(authorization.to_dict())
    common = {
        "frozen_at_utc": _now_utc(),
        "outer_fold": paths.outer_fold,
        "preopen_receipt_sha256": _sha256_nofollow(paths.preopen),
        "access_receipt_path": authorization.access_receipt_path,
        "access_receipt_sha256": authorization.access_receipt_sha256,
        "dataset_receipt_path": _path_label(paths.dataset_receipt),
        "dataset_receipt_sha256": _sha256_nofollow(paths.dataset_receipt),
        "source_receipts_root_sha256": dataset_receipt[
            "source_receipts_root_sha256"
        ],
        "result_path": _path_label(paths.result),
        "result_sha256": _sha256_nofollow(paths.result),
        "quarantine_labels": list(_foundation.QUARANTINE_LABELS),
        "claim_boundary": _foundation.CLAIM_BOUNDARY,
        "holdout_caveat": _foundation.HOLDOUT_CAVEAT,
        "plan_sha256": context["plan_sha256"],
        "fill_law_hash": context["fill_law_hash"],
        "holdout_open_count": 0,
    }
    if paths.role == "nested_validation":
        value = {
            "schema_version": "pathd.entry_nested_result_receipt.v1",
            "status": "FROZEN_NESTED_VALIDATION_RESULT",
            **common,
            "inner_fold": paths.inner_fold,
            "validation_access_count": 1,
            "authorization_sha256": authorization_sha,
            "dataset_sha256": dataset_receipt["dataset_sha256"],
            "hgb_evaluation_sha256": result["hgb"]["result_sha256"],
            "neural_evaluation_sha256": result["neural"]["result_sha256"],
        }
        return _self_hash(value)
    if paths.role == "outer_test_primary":
        payload = result["payload"]
        preopen = _foundation.read_json(paths.preopen)
        negative_panel = paths.preopen.parent / "negative_control_panel.json"
        replay_result = paths.preopen.parent / "control_replay_result.json"
        if not negative_panel.exists() or not replay_result.exists():
            raise EntryEvidenceGateError(
                "entry evidence blocked: outer post-open controls are absent"
            )
        value = {
            "schema_version": "pathd.entry_outer_result_receipt.v1",
            "status": "FROZEN_PRIMARY_OUTER_RESULT",
            "frozen_at_utc": common["frozen_at_utc"],
            "outer_fold": paths.outer_fold,
            "preregistration_sha256": context["preregistration_sha256"],
            "preopen_receipt_path": _path_label(paths.preopen),
            "preopen_receipt_sha256": common["preopen_receipt_sha256"],
            "access_receipt_path": authorization.access_receipt_path,
            "access_receipt_sha256": authorization.access_receipt_sha256,
            "outer_evidence_access_count": 1,
            "primary_sessions_sha256_newline": authorization.sessions_sha256_newline,
            "primary_authorization_sha256": authorization_sha,
            "dataset_receipt_path": common["dataset_receipt_path"],
            "dataset_receipt_sha256": common["dataset_receipt_sha256"],
            "primary_dataset_sha256": dataset_receipt["dataset_sha256"],
            "source_receipts_root_sha256": common[
                "source_receipts_root_sha256"
            ],
            "result_path": common["result_path"],
            "result_sha256": common["result_sha256"],
            "hgb_evaluation_sha256": payload["hgb"]["result_sha256"],
            "neural_evaluation_sha256": payload["neural"]["result_sha256"],
            "selected_family": payload["selected_family"],
            "negative_control_panel_path": _path_label(negative_panel),
            "negative_control_panel_sha256": _sha256_nofollow(negative_panel),
            "control_replay_result_path": _path_label(replay_result),
            "control_replay_result_sha256": _sha256_nofollow(replay_result),
            "quarantine_labels": common["quarantine_labels"],
            "holdout_caveat": common["holdout_caveat"],
            "claim_boundary": common["claim_boundary"],
            "plan_sha256": common["plan_sha256"],
            "fill_law_hash": common["fill_law_hash"],
            "holdout_open_count": 0,
        }
        if preopen.get("selected_family") != value["selected_family"]:
            raise EntryEvidenceGateError(
                "entry evidence blocked: selected family changed after preopen"
            )
        return _self_hash(value)
    value = {
        "schema_version": "pathd.entry_shortened_diagnostic_result_receipt.v1",
        "status": "FROZEN_SHORTENED_DIAGNOSTIC_RESULT",
        **common,
        "diagnostic_access_count": 1,
        "authorization_sha256": authorization_sha,
        "dataset_sha256": dataset_receipt["dataset_sha256"],
        "evaluation_sha256": result["result_sha256"],
    }
    return _self_hash(value)


def _burn_scope_id(
    paths: _ScopePaths, *, access_scope_id: str | None
) -> str:
    if access_scope_id is not None:
        if (
            type(access_scope_id) is not str
            or _HEX64.fullmatch(access_scope_id) is None
        ):
            raise EntryEvidenceGateError(
                "entry evidence blocked: malformed burn access scope"
            )
        return access_scope_id
    if paths.access.exists():
        try:
            candidate = _read_canonical_json(paths.access).get("access_scope_id")
        except EntryEvidenceGateError:
            candidate = None
        if type(candidate) is str and _HEX64.fullmatch(candidate) is not None:
            return candidate
    # A corrupt/missing durable access has no trustworthy foundation claim.  Its
    # burn still receives a campaign-path-bound quarantine identity, never the
    # role/fold-only identity used by the pre-correction implementation.
    return _stable_hash(
        {
            "schema_version": "pathd.entry_evidence_unbound_burn_scope.v1",
            "access_receipt_path": _path_label(paths.access),
            "role": paths.role,
            "outer_fold": paths.outer_fold,
            "inner_fold": paths.inner_fold,
        }
    )


def _write_burned(
    paths: _ScopePaths,
    *,
    reason: str,
    transaction_id: str | None,
    access_scope_id: str | None = None,
) -> dict[str, Any]:
    if paths.burned_receipt.exists():
        value = _read_canonical_json(paths.burned_receipt)
        _validate_self_hash(value)
        return value
    value = _self_hash(
        {
            "schema_version": "pathd.entry_evidence_burned_receipt.v1",
            "status": "BURNED_NO_REOPEN",
            "burned_at_utc": _now_utc(),
            "role": paths.role,
            "outer_fold": paths.outer_fold,
            "inner_fold": paths.inner_fold,
            "access_scope_id": _burn_scope_id(
                paths, access_scope_id=access_scope_id
            ),
            "transaction_id": transaction_id,
            "reason": reason,
            "access_receipt_sha256": (
                _sha256_nofollow(paths.access) if paths.access.exists() else None
            ),
            "dataset_receipt_sha256": (
                _sha256_nofollow(paths.dataset_receipt)
                if paths.dataset_receipt.exists()
                else None
            ),
            "result_sha256": (
                _sha256_nofollow(paths.result) if paths.result.exists() else None
            ),
            "access_count": 1 if paths.access.exists() else 0,
            "reopen_permitted": False,
            "holdout_open_count": 0,
        }
    )
    _write_exclusive_json(paths.burned_receipt, value)
    return value


def seal_entry_evidence_result(
    authorization: _foundation.FrozenEvidenceAuthorization,
    /,
    *,
    dataset: Any,
    evaluation: Any,
) -> dict[str, Any]:
    """Seal the exact decoded dataset and typed result, then revoke capability."""

    if type(authorization) is not _foundation.FrozenEvidenceAuthorization:
        raise EntryEvidenceGateError(
            "entry evidence blocked: untrusted evidence authorization type"
        )
    paths = _scope_paths(
        role=authorization.role,
        outer_fold=authorization.outer_fold,
        inner_fold=authorization.inner_fold,
    )
    authorization_sha = _stable_hash(authorization.to_dict())
    capability_key = id(authorization)
    capability = _ACTIVE_CAPABILITIES.get(capability_key)
    if (
        capability is None
        or capability.pid != os.getpid()
        or capability.authorization is not authorization
        or capability.state != "DECODE_CLAIMED"
    ):
        raise EntryEvidenceGateError(
            "entry evidence blocked: result lacks the consumed live capability"
        )
    try:
        current = read_frozen_entry_evidence_authorization(
            role=authorization.role,
            outer_fold=authorization.outer_fold,
            inner_fold=authorization.inner_fold,
        )
    except BaseException:
        _write_burned(
            paths,
            reason="FROZEN_FOUNDATION_DRIFT_AFTER_DECODE",
            transaction_id=authorization.transaction_id,
            access_scope_id=authorization.access_scope_id,
        )
        _release_capability(capability_key)
        raise
    if current != authorization:
        _write_burned(
            paths,
            reason="AUTHORIZATION_DRIFT_AFTER_OPEN",
            transaction_id=authorization.transaction_id,
            access_scope_id=authorization.access_scope_id,
        )
        _release_capability(capability_key)
        raise EntryEvidenceGateError(
            "entry evidence blocked: authorization drift after open"
        )
    try:
        _assert_no_later_artifacts(paths)
    except BaseException:
        _write_burned(
            paths,
            reason="LATER_EVIDENCE_APPEARED_AFTER_OPEN",
            transaction_id=authorization.transaction_id,
            access_scope_id=authorization.access_scope_id,
        )
        _release_capability(capability_key)
        raise
    try:
        _validate_exact_dataset(dataset, authorization=authorization)
        _validate_exact_evaluation(
            evaluation, authorization=authorization, dataset=dataset
        )
        dataset_receipt = _dataset_receipt(
            dataset, authorization=authorization
        )
        _rehash_dataset_source_files(dataset_receipt)
        result = _build_result_document(
            evaluation, authorization=authorization, dataset=dataset
        )
        _validate_result_document_binding(
            result,
            authorization=authorization,
            dataset_sha256=dataset.dataset_sha256,
        )
        if authorization.role == "outer_test_shortened_diagnostic":
            _foundation._validate_policy_evaluation_dataset_membership(
                result, dataset_receipt
            )
    except BaseException:
        _write_burned(
            paths,
            reason="INVALID_DATASET_OR_TYPED_EVALUATION",
            transaction_id=authorization.transaction_id,
            access_scope_id=authorization.access_scope_id,
        )
        _release_capability(capability_key)
        raise
    try:
        _write_exclusive_json(paths.dataset_receipt, dataset_receipt)
        _write_exclusive_json(paths.result, result)
        receipt = _result_receipt(
            paths,
            authorization=authorization,
            dataset_receipt=dataset_receipt,
            result=result,
        )
        _write_exclusive_json(paths.result_receipt, receipt)
    finally:
        _release_capability(capability_key)
    return receipt


def _validate_shortened_result_receipt(
    paths: _ScopePaths,
    *,
    authorization: _foundation.FrozenEvidenceAuthorization,
) -> dict[str, Any]:
    receipt = _read_canonical_json(paths.result_receipt)
    _validate_self_hash(receipt)
    dataset_receipt = _foundation._validate_entry_evidence_dataset_receipt(
        paths.dataset_receipt, authorization=authorization
    )
    result = _read_canonical_json(paths.result)
    _validate_result_document_binding(
        result,
        authorization=authorization,
        dataset_sha256=dataset_receipt["dataset_sha256"],
    )
    _foundation._validate_policy_evaluation_dataset_membership(
        result, dataset_receipt
    )
    expected = _result_receipt(
        paths,
        authorization=authorization,
        dataset_receipt=dataset_receipt,
        result=result,
    )
    semantic_expected = dict(expected)
    semantic_expected["frozen_at_utc"] = receipt.get("frozen_at_utc")
    semantic_expected.pop("receipt_sha256", None)
    semantic_receipt = dict(receipt)
    semantic_receipt.pop("receipt_sha256", None)
    if semantic_receipt != semantic_expected:
        raise EntryEvidenceGateError(
            "entry evidence blocked: shortened result receipt drift"
        )
    return receipt


def _validate_terminal_result(
    paths: _ScopePaths,
    *,
    authorization: _foundation.FrozenEvidenceAuthorization,
) -> dict[str, Any]:
    canonical_receipt = _read_canonical_json(paths.result_receipt)
    _validate_self_hash(canonical_receipt)
    if paths.role == "nested_validation":
        payload = _foundation.read_json(PREREG_PATH)
        assignments = _foundation.read_json(SESSION_PATH)
        return _foundation._validate_nested_result_receipt_one(
            payload, assignments, paths.outer_fold, paths.inner_fold
        )
    if paths.role == "outer_test_primary":
        payload = _foundation.read_json(PREREG_PATH)
        assignments = _foundation.read_json(SESSION_PATH)
        return _foundation._validate_outer_result_receipt_one(
            payload, assignments, paths.outer_fold
        )
    return _validate_shortened_result_receipt(
        paths, authorization=authorization
    )


def inspect_entry_evidence_state(
    *,
    role: str,
    outer_fold: int | None = None,
    inner_fold: int | None = None,
) -> dict[str, Any]:
    """Inspect immutable transaction artifacts without opening evidence."""

    paths = _scope_paths(
        role=role, outer_fold=outer_fold, inner_fold=inner_fold
    )
    skip = paths.skip_receipt is not None and paths.skip_receipt.exists()
    access = paths.access.exists()
    burned = paths.burned_receipt.exists()
    terminal = paths.result_receipt.exists()
    dataset = paths.dataset_receipt.exists()
    result = paths.result.exists()
    if skip:
        if access or burned or terminal or dataset or result:
            raise EntryEvidenceGateError(
                "entry evidence blocked: skipped block has evidence artifacts"
            )
        value = _read_canonical_json(paths.skip_receipt)  # type: ignore[arg-type]
        _validate_self_hash(value)
        state = INVALID_SKIPPED
        transaction_id = None
    elif burned:
        value = _read_canonical_json(paths.burned_receipt)
        _validate_self_hash(value)
        state = BURNED
        transaction_id = value.get("transaction_id")
    elif terminal:
        authorization = read_frozen_entry_evidence_authorization(
            role=role, outer_fold=outer_fold, inner_fold=inner_fold
        )
        _validate_terminal_result(paths, authorization=authorization)
        state = SEALED
        transaction_id = authorization.transaction_id
    elif access:
        authorization = read_frozen_entry_evidence_authorization(
            role=role, outer_fold=outer_fold, inner_fold=inner_fold
        )
        authorization_sha = _stable_hash(authorization.to_dict())
        capability = next(
            (
                item
                for item in _ACTIVE_CAPABILITIES.values()
                if item.authorization_sha256 == authorization_sha
                and item.pid == os.getpid()
            ),
            None,
        )
        if capability is not None:
            state = OPEN_ACTIVE
        elif dataset and result:
            state = DURABLE_UNSEALED_BURN_REQUIRED
        else:
            state = OPEN_PENDING_RESULT
        transaction_id = authorization.transaction_id
    else:
        if dataset or result:
            raise EntryEvidenceGateError(
                "entry evidence blocked: result artifact exists without access"
            )
        state = UNOPENED
        transaction_id = None
    return {
        "schema_version": "pathd.entry_evidence_state.v1",
        "state": state,
        "role": role,
        "outer_fold": paths.outer_fold,
        "inner_fold": paths.inner_fold,
        "access_count": 1 if access else 0,
        "transaction_id": transaction_id,
        "access_receipt_sha256": (
            _sha256_nofollow(paths.access) if access else None
        ),
        "dataset_receipt_sha256": (
            _sha256_nofollow(paths.dataset_receipt) if dataset else None
        ),
        "result_sha256": _sha256_nofollow(paths.result) if result else None,
        "terminal_receipt_sha256": (
            _sha256_nofollow(paths.result_receipt)
            if terminal
            else _sha256_nofollow(paths.burned_receipt)
            if burned
            else _sha256_nofollow(paths.skip_receipt)  # type: ignore[arg-type]
            if skip
            else None
        ),
    }


def recover_entry_evidence_after_crash(
    *,
    role: str,
    outer_fold: int | None = None,
    inner_fold: int | None = None,
) -> dict[str, Any]:
    """Validate a completed live seal; otherwise permanently burn the scope."""

    paths = _scope_paths(
        role=role, outer_fold=outer_fold, inner_fold=inner_fold
    )
    with _scope_lease(paths.lock):
        if paths.skip_receipt is not None and paths.skip_receipt.exists():
            return inspect_entry_evidence_state(
                role=role, outer_fold=outer_fold, inner_fold=inner_fold
            )
        if not paths.access.exists():
            if paths.dataset_receipt.exists() or paths.result.exists():
                _write_burned(
                    paths,
                    reason="RESULT_WITHOUT_DURABLE_ACCESS",
                    transaction_id=None,
                    access_scope_id=None,
                )
            return inspect_entry_evidence_state(
                role=role, outer_fold=outer_fold, inner_fold=inner_fold
            )
        transaction_id: str | None = None
        access_scope_id: str | None = None
        try:
            access = _read_access_receipt(paths)
            transaction_id = access["transaction_id"]
            access_scope_id = access["access_scope_id"]
            authorization = _authorization_from_access(paths, access)
            _validate_claim_foundation(paths, authorization)
        except BaseException:
            _write_burned(
                paths,
                reason="CORRUPT_DURABLE_ACCESS",
                transaction_id=transaction_id,
                access_scope_id=access_scope_id,
            )
            return inspect_entry_evidence_state(
                role=role, outer_fold=outer_fold, inner_fold=inner_fold
            )
        if paths.result_receipt.exists():
            try:
                _validate_terminal_result(paths, authorization=authorization)
            except BaseException:
                _write_burned(
                    paths,
                    reason="CORRUPT_TERMINAL_RESULT",
                    transaction_id=transaction_id,
                    access_scope_id=authorization.access_scope_id,
                )
            return inspect_entry_evidence_state(
                role=role, outer_fold=outer_fold, inner_fold=inner_fold
            )
        if paths.burned_receipt.exists():
            return inspect_entry_evidence_state(
                role=role, outer_fold=outer_fold, inner_fold=inner_fold
            )
        _write_burned(
            paths,
            reason=(
                "LEASE_LOST_WITH_DURABLE_UNSEALED_RESULT"
                if paths.dataset_receipt.exists() or paths.result.exists()
                else "LEASE_LOST_AFTER_ACCESS"
            ),
            transaction_id=transaction_id,
            access_scope_id=authorization.access_scope_id,
        )
        return inspect_entry_evidence_state(
            role=role, outer_fold=outer_fold, inner_fold=inner_fold
        )
