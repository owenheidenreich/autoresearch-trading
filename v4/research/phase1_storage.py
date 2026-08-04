"""Storage contract for the one-second Path-D exit-model campaign.

The module deliberately has no vendor, broker, holdout, or model imports.  It
provides one safe place for resolving research roots, checking an external
APFS volume, and checksum-verifying a corpus relocation before callers switch
their configured data root.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
from typing import Iterable, Mapping


DATA_ROOT_ENV = "AR_TRADING_DATA_ROOT"
SCRATCH_ROOT_ENV = "AR_TRADING_SCRATCH_ROOT"
ARTIFACT_ROOT_ENV = "AR_TRADING_ARTIFACT_ROOT"
MINIMUM_FREE_FRACTION = 0.25
PHASE1_MAX_ALLOCATED_BYTES = 150_000_000_000
MANIFEST_SCHEMA = "pathd.phase1.corpus-manifest.v1"
PHASE1_DIRECTORIES = (
    "vendor",
    "canonical",
    "oof",
    "exit_features",
    "exit_labels",
    "scratch",
    "artifacts",
    "reports",
)


class StorageContractError(RuntimeError):
    """A storage prerequisite failed closed."""


@dataclass(frozen=True)
class ResearchRoots:
    data_root: Path
    scratch_root: Path
    artifact_root: Path

    @classmethod
    def resolve(
        cls,
        *,
        data_root: Path | None = None,
        scratch_root: Path | None = None,
        artifact_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "ResearchRoots":
        env = os.environ if environ is None else environ

        def one(explicit: Path | None, variable: str) -> Path:
            value = explicit if explicit is not None else env.get(variable)
            if value is None or not str(value).strip():
                raise StorageContractError(
                    f"missing research root: pass the CLI option or set {variable}"
                )
            result = Path(value).expanduser().resolve(strict=False)
            if _is_cloud_managed(result):
                raise StorageContractError(f"research root may not be cloud-managed: {result}")
            return result

        return cls(
            data_root=one(data_root, DATA_ROOT_ENV),
            scratch_root=one(scratch_root, SCRATCH_ROOT_ENV),
            artifact_root=one(artifact_root, ARTIFACT_ROOT_ENV),
        )


@dataclass(frozen=True)
class ManifestEntry:
    relative_path: str
    bytes: int
    sha256: str


def _is_cloud_managed(path: Path) -> bool:
    lowered = path.as_posix().lower()
    markers = (
        "/library/mobile documents/",
        "/library/cloudstorage/",
        "/icloud drive/",
    )
    return any(marker in lowered for marker in markers)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise StorageContractError(f"corpus may not contain symlinks: {path}")
        if path.is_file():
            yield path


def build_manifest(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise StorageContractError(f"corpus root is not a directory: {root}")
    entries = tuple(
        ManifestEntry(
            relative_path=path.relative_to(root).as_posix(),
            bytes=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path in _files(root)
    )
    semantic: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "file_count": len(entries),
        "total_bytes": sum(entry.bytes for entry in entries),
        "entries": [asdict(entry) for entry in entries],
    }
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return {**semantic, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def verify_manifest(root: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    observed = build_manifest(root)
    if dict(manifest) != observed:
        expected_entries = {
            str(row["relative_path"]): row
            for row in manifest.get("entries", [])  # type: ignore[union-attr]
        }
        observed_entries = {
            str(row["relative_path"]): row
            for row in observed["entries"]  # type: ignore[union-attr]
        }
        missing = sorted(set(expected_entries) - set(observed_entries))
        extra = sorted(set(observed_entries) - set(expected_entries))
        changed = sorted(
            key
            for key in set(expected_entries) & set(observed_entries)
            if expected_entries[key] != observed_entries[key]
        )
        raise StorageContractError(
            f"corpus checksum mismatch: missing={missing[:5]} extra={extra[:5]} "
            f"changed={changed[:5]}"
        )
    return observed


def _filesystem_mountpoint(path: Path) -> Path:
    """Resolve an existing path to the root of its backing filesystem."""

    candidate = path.expanduser().resolve(strict=True)
    if candidate.is_file():
        candidate = candidate.parent
    device = candidate.stat().st_dev
    while candidate.parent != candidate:
        parent = candidate.parent
        if parent.stat().st_dev != device:
            break
        candidate = parent
    return candidate


def _diskutil_info(path: Path) -> dict[str, object]:
    mountpoint = _filesystem_mountpoint(path)
    command = ["diskutil", "info", "-plist", str(mountpoint)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise StorageContractError(f"unable to inspect storage volume for {path}") from exc
    return plistlib.loads(completed.stdout)


def _apple_volume_is_encrypted(info: Mapping[str, object]) -> bool:
    """Accept the equivalent diskutil plist keys emitted across macOS versions."""

    flags = [info[key] for key in ("Encrypted", "Encryption") if key in info]
    return bool(flags) and all(flag is True for flag in flags)


def validate_apple_volume_info(
    info: Mapping[str, object], *, expected_volume_name: str = "AR_TRADING_DATA"
) -> None:
    personality = str(info.get("FilesystemType") or info.get("FileSystemPersonality") or "")
    if "apfs" not in personality.lower():
        raise StorageContractError(f"research volume must use APFS, observed {personality!r}")
    if not _apple_volume_is_encrypted(info):
        raise StorageContractError("research volume must be encrypted APFS")
    name = str(info.get("VolumeName") or "")
    if name != expected_volume_name:
        raise StorageContractError(
            f"research volume name must be {expected_volume_name!r}, observed {name!r}"
        )
    internal = info.get("Internal")
    if internal is not False:
        raise StorageContractError("research data root must be on an external volume")


def validate_capacity(path: Path, *, incoming_bytes: int = 0) -> dict[str, float | int]:
    usage = shutil.disk_usage(path)
    projected_free = usage.free - incoming_bytes
    if projected_free < 0:
        raise StorageContractError("insufficient free bytes for requested operation")
    projected_fraction = projected_free / usage.total
    if projected_fraction < MINIMUM_FREE_FRACTION:
        raise StorageContractError(
            f"operation would leave {projected_fraction:.1%} free; minimum is "
            f"{MINIMUM_FREE_FRACTION:.0%}"
        )
    return {
        "total_bytes": usage.total,
        "free_bytes_before": usage.free,
        "incoming_bytes": incoming_bytes,
        "free_bytes_after": projected_free,
        "free_fraction_after": projected_fraction,
    }


def validate_phase1_allocation(path: Path, *, incoming_bytes: int = 0) -> dict[str, int]:
    """Fail closed if the complete Phase-1 tree would exceed its frozen cap."""

    allocated = 0
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise StorageContractError(f"Phase-1 roots may not contain symlinks: {candidate}")
        if candidate.is_file():
            allocated += candidate.stat().st_size
    projected = allocated + int(incoming_bytes)
    if projected > PHASE1_MAX_ALLOCATED_BYTES:
        raise StorageContractError(
            f"Phase-1 allocation would be {projected} bytes; cap is "
            f"{PHASE1_MAX_ALLOCATED_BYTES} bytes"
        )
    return {
        "allocated_bytes": allocated,
        "incoming_bytes": int(incoming_bytes),
        "projected_bytes": projected,
        "cap_bytes": PHASE1_MAX_ALLOCATED_BYTES,
    }


def preflight_roots(
    roots: ResearchRoots,
    *,
    expected_volume_name: str = "AR_TRADING_DATA",
    incoming_bytes: int = 0,
    create: bool = False,
    volume_info: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if create:
        for root in (roots.data_root, roots.scratch_root, roots.artifact_root):
            root.mkdir(parents=True, exist_ok=True)
        for name in PHASE1_DIRECTORIES:
            (roots.data_root / name).mkdir(parents=True, exist_ok=True)
    for root in (roots.data_root, roots.scratch_root, roots.artifact_root):
        if not root.is_dir():
            raise StorageContractError(f"research root does not exist: {root}")
    info = _diskutil_info(roots.data_root) if volume_info is None else dict(volume_info)
    validate_apple_volume_info(info, expected_volume_name=expected_volume_name)
    data_device = info.get("DeviceIdentifier")
    for root in (roots.scratch_root, roots.artifact_root):
        other = _diskutil_info(root) if volume_info is None else info
        if other.get("DeviceIdentifier") != data_device:
            raise StorageContractError("all Phase-1 roots must be on the approved volume")
    capacity = validate_capacity(roots.data_root, incoming_bytes=incoming_bytes)
    allocation = validate_phase1_allocation(roots.data_root, incoming_bytes=incoming_bytes)
    return {
        "status": "PASS",
        "roots": {name: str(value) for name, value in asdict(roots).items()},
        "volume_name": info.get("VolumeName"),
        "filesystem": info.get("FilesystemType") or info.get("FileSystemPersonality"),
        "encrypted": _apple_volume_is_encrypted(info),
        "external": info.get("Internal") is False,
        "capacity": capacity,
        "phase1_allocation": allocation,
    }


def relocate_corpus(
    source: Path,
    destination: Path,
    *,
    expected_volume_name: str = "AR_TRADING_DATA",
    volume_info: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Copy and verify without deleting or changing the source corpus."""

    source = source.resolve(strict=True)
    destination = destination.expanduser().resolve(strict=False)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise StorageContractError(f"destination must be absent or empty: {destination}")
    manifest = build_manifest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = _diskutil_info(destination.parent) if volume_info is None else dict(volume_info)
    validate_apple_volume_info(info, expected_volume_name=expected_volume_name)
    validate_capacity(destination.parent, incoming_bytes=int(manifest["total_bytes"]))
    validate_phase1_allocation(destination.parent.parent, incoming_bytes=int(manifest["total_bytes"]))
    destination.mkdir(exist_ok=True)
    incomplete = destination.parent / f".{destination.name}.relocation-incomplete"
    if incomplete.exists():
        raise StorageContractError(f"prior relocation is incomplete: {incomplete}")
    incomplete.write_text("verification pending\n", encoding="utf-8")
    for row in manifest["entries"]:  # type: ignore[union-attr]
        relative = Path(str(row["relative_path"]))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    verified = verify_manifest(destination, manifest)
    incomplete.unlink()
    manifest_path = destination.parent / f"{destination.name}.manifest.json"
    if manifest_path.exists():
        raise StorageContractError(f"refusing to overwrite manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(verified, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "COPIED_AND_VERIFIED",
        "source_preserved": True,
        "source": str(source),
        "destination": str(destination),
        "manifest_path": str(manifest_path),
        "manifest_sha256": verified["manifest_sha256"],
        "file_count": verified["file_count"],
        "total_bytes": verified["total_bytes"],
    }
