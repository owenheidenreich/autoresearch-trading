"""Research provenance block.

Every admissible result (CPU audit, screen, CV run, final train) writes a JSON
side-car containing these seven fields. Any mismatch against the current
code/dataset marks the result non-comparable mechanically — not by human memory.

Field mapping (see ``build_provenance``):

    dataset_fingerprint    -> ``data.pt`` metadata
    sidecar_schema_version -> first sidecar ``schema_version``
    feature_set_id         -> hash of ``ACTIVE_FEATURE_NAMES``
    label_mode             -> opp-label fn name + threshold constants
    cost_model_version     -> fingerprint of simulator/features source
    split_recipe           -> walkforward generator hash + screen mode
    git_commit             -> ``git rev-parse HEAD`` at write time

The module is import-light and deliberately free of project imports so it can be
used from any tier (analysis, train, replay, ops).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SIMULATOR_PY = PROJECT_ROOT / "v2" / "core" / "simulator.py"
_FEATURES_PY = PROJECT_ROOT / "v2" / "core" / "features.py"
_WALKFORWARD_PY = PROJECT_ROOT / "v2" / "core" / "walkforward.py"

PROVENANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Provenance:
    dataset_fingerprint: str
    sidecar_schema_version: str
    feature_set_id: str
    label_mode: str
    cost_model_version: str
    split_recipe: str
    git_commit: str
    research_tier: bool = False
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _hash_json(obj: Any) -> str:
    return _hash_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _feature_set_id(feature_names: list[str]) -> str:
    return _hash_json(list(feature_names))


def _file_bytes(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    return b"missing:" + str(path).encode()


def _cost_model_version() -> str:
    return _hash_bytes(b"||".join([_file_bytes(_SIMULATOR_PY), _file_bytes(_FEATURES_PY)]))


def _split_recipe(*, screen_mode: str, n_folds: int) -> str:
    return _hash_json({
        "walkforward_src": _hash_bytes(_file_bytes(_WALKFORWARD_PY)),
        "screen_mode": screen_mode,
        "n_folds": int(n_folds),
    })


def _label_mode_fingerprint(label_mode: str, thresholds: dict[str, Any] | None) -> str:
    clean = dict(sorted((thresholds or {}).items()))
    return _hash_json({"mode": str(label_mode), "thresholds": clean})


def build_provenance(
    *,
    data_path: str | os.PathLike,
    feature_names: list[str],
    label_mode: str,
    label_thresholds: dict[str, Any] | None = None,
    screen_mode: str = "unknown",
    n_folds: int = 0,
    research_tier: bool = False,
    dataset_fingerprint: str | None = None,
    sidecar_schema_version: str | None = None,
) -> Provenance:
    """Build a Provenance block. Reads ``data.pt`` only if dataset fields are not given.

    The caller may pass ``dataset_fingerprint`` / ``sidecar_schema_version``
    directly to avoid re-loading the manifest (the training harness already has
    these at hand)."""
    if dataset_fingerprint is None or sidecar_schema_version is None:
        import torch  # deferred to keep import-cost low for callers that pass both
        data = torch.load(str(data_path), map_location="cpu", weights_only=False)
        md = data.get("metadata", {}) or {}
        if dataset_fingerprint is None:
            dataset_fingerprint = str(md.get("dataset_fingerprint", "unknown"))
        if sidecar_schema_version is None:
            sidecar_schema_version = str(md.get("schema_version", "unknown"))

    return Provenance(
        dataset_fingerprint=str(dataset_fingerprint),
        sidecar_schema_version=str(sidecar_schema_version),
        feature_set_id=_feature_set_id(feature_names),
        label_mode=_label_mode_fingerprint(label_mode, label_thresholds),
        cost_model_version=_cost_model_version(),
        split_recipe=_split_recipe(screen_mode=screen_mode, n_folds=n_folds),
        git_commit=_git_commit(),
        research_tier=bool(research_tier),
    )


def write_provenance(
    path: str | os.PathLike,
    provenance: Provenance,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write provenance + optional ``extra`` metrics/notes to a JSON file."""
    payload = {
        "provenance": provenance.to_dict(),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra is not None:
        payload["extra"] = extra
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def load_provenance(path: str | os.PathLike) -> Provenance:
    with open(path) as f:
        payload = json.load(f)
    block = payload.get("provenance", payload)
    return Provenance(
        dataset_fingerprint=block["dataset_fingerprint"],
        sidecar_schema_version=block["sidecar_schema_version"],
        feature_set_id=block["feature_set_id"],
        label_mode=block["label_mode"],
        cost_model_version=block["cost_model_version"],
        split_recipe=block["split_recipe"],
        git_commit=block["git_commit"],
        research_tier=bool(block.get("research_tier", False)),
        schema_version=int(block.get("schema_version", PROVENANCE_SCHEMA_VERSION)),
    )


_ALL_FIELDS: tuple[str, ...] = (
    "dataset_fingerprint",
    "sidecar_schema_version",
    "feature_set_id",
    "label_mode",
    "cost_model_version",
    "split_recipe",
    "git_commit",
    "research_tier",
    "schema_version",
)


def compare_provenance(
    a: Provenance,
    b: Provenance,
    *,
    ignore: tuple[str, ...] = ("git_commit",),
) -> list[str]:
    """Return the list of field names that differ. ``git_commit`` is ignored by
    default so a pure recompile does not void comparability — but all substantive
    mismatches (data, label, cost, schema, split) are always surfaced."""
    return [f for f in _ALL_FIELDS if f not in ignore and getattr(a, f) != getattr(b, f)]


def assert_comparable(
    a: Provenance,
    b: Provenance,
    *,
    ignore: tuple[str, ...] = ("git_commit",),
) -> None:
    diffs = compare_provenance(a, b, ignore=ignore)
    if diffs:
        raise RuntimeError(
            "Provenance mismatch — results are not comparable. "
            f"Fields: {', '.join(diffs)}"
        )


# --------------------------------------------------------------------------
# Self-test: python3 -m v2.core.provenance runs a mismatch-detection check.
# --------------------------------------------------------------------------
def _self_test() -> int:
    base = Provenance(
        dataset_fingerprint="6162cf3d83db3586",
        sidecar_schema_version="v5_exact_chain_v2_slice",
        feature_set_id="aaaabbbbccccdddd",
        label_mode="deadbeefdeadbeef",
        cost_model_version="ffffffffffffffff",
        split_recipe="1111111111111111",
        git_commit="sha-a",
        research_tier=True,
    )
    same = Provenance(**{**base.to_dict(), "git_commit": "sha-b"})
    diffs = compare_provenance(base, same)
    assert diffs == [], f"expected git_commit ignored but got diffs={diffs}"

    label_flipped = Provenance(**{**base.to_dict(), "label_mode": "0000000000000000"})
    try:
        assert_comparable(base, label_flipped)
    except RuntimeError as exc:
        assert "label_mode" in str(exc), f"mismatch message missing label_mode: {exc}"
    else:
        raise AssertionError("assert_comparable should have raised on label_mode change")

    research_flipped = Provenance(**{**base.to_dict(), "research_tier": False})
    try:
        assert_comparable(base, research_flipped)
    except RuntimeError as exc:
        assert "research_tier" in str(exc), f"mismatch message missing research_tier: {exc}"
    else:
        raise AssertionError("assert_comparable should have raised on research_tier change")

    print("provenance self-test OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_self_test())
