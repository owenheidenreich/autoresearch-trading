from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_PROTOCOL_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts")
DEFAULT_SEQUENCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_077_q4start_lifecycle_sequence_dataset")
DEFAULT_REPRODUCTION_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_082_q4start_artifact_reproduction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--reproduction-dir", type=Path, default=DEFAULT_REPRODUCTION_DIR)
    parser.add_argument("--promotion-dir", type=Path, default=Path("v4/promotion"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _artifact_entry(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    files = {}
    for name, file_path in manifest["files"].items():
        files[name] = _file_record(Path(file_path))
    return {
        "fold": manifest["fold"],
        "test_split": manifest["test_split"],
        "seed": int(manifest["seed"]),
        "sequence_mode": manifest["sequence_mode"],
        "calibrate_threshold": bool(manifest["calibrate_threshold"]),
        "selected_override_threshold": manifest["selected_override_threshold"],
        "feature_count": len(manifest["feature_columns"]),
        "train_splits": manifest.get("train_splits", []),
        "validation_source": manifest.get("validation_source"),
        "train_trades": int(manifest.get("train_trades", 0)),
        "validation_trades": int(manifest.get("validation_trades", 0)),
        "test_trades": int(manifest.get("test_trades", 0)),
        "files": files,
    }


def _deployment_manifest(protocol_dir: Path) -> dict[str, Any]:
    artifact_root = protocol_dir / "model_artifacts"
    manifest_paths = sorted(artifact_root.glob("*/seed_*/manifest.json"))
    if not manifest_paths:
        raise SystemExit(f"no model artifact manifests found under {artifact_root}")
    artifacts = [_artifact_entry(path) for path in manifest_paths]
    folds = sorted({row["fold"] for row in artifacts})
    seed_count_per_fold = {
        fold: len({row["seed"] for row in artifacts if row["fold"] == fold})
        for fold in folds
    }
    file_count = sum(len(row["files"]) for row in artifacts)
    return {
        "protocol_id": "protocol_081",
        "manifest_type": "deployment_artifact_pointer",
        "created_at": str(date.today()),
        "status": "persisted_research_challenger_not_paper_or_live_approved",
        "artifact_root": str(artifact_root),
        "artifact_run": protocol_dir.name,
        "fold_count": len(folds),
        "folds": folds,
        "seed_count_per_fold": seed_count_per_fold,
        "artifact_bundle_count": len(artifacts),
        "file_count": file_count,
        "model_class": "v4.scripts.run_protocol061_sequence_lifecycle_model.LifecycleSequenceModel",
        "scaler_class": "v4.model.supervised_pilot.FeatureScaler",
        "selection_note": (
            "These are persisted fold/seed research artifacts for Protocol 081. "
            "A live inference ensemble/router is not approved yet."
        ),
        "promotion_constraints": [
            "no paid data download without explicit user approval",
            "no broker-connected paper trading until no-order shadow-feed parity passes",
            "entry quote timestamp and live stale-quote checks must be retained before paper trading",
            "Protocol 054 fallback routing must remain explicit until replaced by a separately validated lifecycle model",
        ],
        "artifacts": artifacts,
    }


def _freeze_manifest(
    *,
    protocol_dir: Path,
    sequence_dir: Path,
    reproduction_dir: Path,
    deployment_manifest_path: Path,
) -> dict[str, Any]:
    frozen = {
        "report_json": protocol_dir / "report.json",
        "selected_trades": protocol_dir / "selected_trades_sequence_exits.json",
        "sequence_dataset_report": sequence_dir / "report.json",
        "artifact_reproduction_summary": reproduction_dir / "summary.json",
        "deployment_artifacts": deployment_manifest_path,
    }
    missing = [str(path) for path in frozen.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing frozen artifact inputs: {missing}")
    return {
        "protocol_id": "protocol_081",
        "freeze_type": "research_candidate_freeze",
        "frozen_at": str(date.today()),
        "status": "frozen_challenger_not_paper_or_live_approved",
        "baseline_protocol": "protocol_054",
        "entry_protocol": "protocol_051",
        "model_protocol": {
            "source_protocol": "protocol_081",
            "sequence_mode": "protocol054_residual_recovery_penalty",
            "calibrate_threshold": True,
            "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "epochs": 12,
            "batch_size": 256,
            "hidden_dim": 72,
            "validation_days": 10,
            "decision_rule": (
                "mandatory hard stop/target first; otherwise exit before Protocol 054 only when "
                "predicted residual value exceeds validation-selected threshold plus deterministic epsilon; "
                "if no override fires, use frozen Protocol 054 exit"
            ),
        },
        "frozen_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in frozen.items()
        },
        "promotion_constraints": [
            "no entry-side knobs while Protocol 081 is frozen",
            "no paid data download without explicit user approval",
            "paper/live approval requires live-data parity, order-state accounting, conservative slippage, and high-resolution path audit evidence",
        ],
    }


def main() -> int:
    args = parse_args()
    args.promotion_dir.mkdir(parents=True, exist_ok=True)
    deployment_path = args.promotion_dir / "PROTOCOL_081_DEPLOYMENT_ARTIFACTS.json"
    deployment = _deployment_manifest(args.protocol_dir)
    deployment_path.write_text(json.dumps(deployment, indent=2, sort_keys=True) + "\n")

    freeze_path = args.promotion_dir / "PROTOCOL_081_FREEZE.json"
    freeze = _freeze_manifest(
        protocol_dir=args.protocol_dir,
        sequence_dir=args.sequence_dir,
        reproduction_dir=args.reproduction_dir,
        deployment_manifest_path=deployment_path,
    )
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "deployment_manifest": str(deployment_path),
        "freeze_manifest": str(freeze_path),
        "artifact_bundle_count": deployment["artifact_bundle_count"],
        "file_count": deployment["file_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
