from __future__ import annotations

from pathlib import Path

import pytest

from v4.research import phase1_storage as storage


EXTERNAL_APFS = {
    "FilesystemType": "apfs",
    "Encrypted": True,
    "VolumeName": "AR_TRADING_DATA",
    "Internal": False,
    "DeviceIdentifier": "disk9s1",
}


def test_roots_require_explicit_cli_or_environment(tmp_path: Path) -> None:
    with pytest.raises(storage.StorageContractError, match="AR_TRADING_DATA_ROOT"):
        storage.ResearchRoots.resolve(environ={})
    roots = storage.ResearchRoots.resolve(
        environ={
            storage.DATA_ROOT_ENV: str(tmp_path / "data"),
            storage.SCRATCH_ROOT_ENV: str(tmp_path / "scratch"),
            storage.ARTIFACT_ROOT_ENV: str(tmp_path / "artifacts"),
        }
    )
    assert roots.data_root == (tmp_path / "data").resolve()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"FilesystemType": "exfat"}, "APFS"),
        ({"Encrypted": False}, "encrypted"),
        ({"VolumeName": "UNTITLED"}, "volume name"),
        ({"Internal": True}, "external"),
    ],
)
def test_volume_contract_fails_closed(change: dict[str, object], message: str) -> None:
    info = {**EXTERNAL_APFS, **change}
    with pytest.raises(storage.StorageContractError, match=message):
        storage.validate_apple_volume_info(info)


def test_manifest_detects_mutation_and_relocation_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.bin").write_bytes(b"abc")
    (source / "nested").mkdir()
    (source / "nested/b.bin").write_bytes(b"def")
    manifest = storage.build_manifest(source)
    assert manifest["file_count"] == 2
    storage.verify_manifest(source, manifest)
    (source / "a.bin").write_bytes(b"changed")
    with pytest.raises(storage.StorageContractError, match="checksum mismatch"):
        storage.verify_manifest(source, manifest)
    (source / "a.bin").write_bytes(b"abc")

    monkeypatch.setattr(
        storage,
        "validate_capacity",
        lambda path, incoming_bytes=0: {"incoming_bytes": incoming_bytes},
    )
    destination = tmp_path / "volume" / "pathd"
    result = storage.relocate_corpus(
        source, destination, volume_info=EXTERNAL_APFS
    )
    assert result["status"] == "COPIED_AND_VERIFIED"
    assert (source / "a.bin").read_bytes() == b"abc"
    storage.verify_manifest(destination, storage.build_manifest(source))


def test_low_disk_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    usage = storage.shutil._ntuple_diskusage(total=1_000, used=800, free=200)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: usage)
    with pytest.raises(storage.StorageContractError, match="minimum is 25%"):
        storage.validate_capacity(tmp_path)

