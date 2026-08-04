"""Command surface for Phase-1 one-second exit-model preparation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.research.phase1_exit_model import (
    OOFEntryReceiptV1,
    build_trajectory_tables,
    load_completed_spx_context,
    load_exact_cbbo_path,
    sensitivity_label_path,
    stable_hash,
    validate_oof_entry_receipt,
    train_exit_campaign,
    write_sensitivity_label_partition,
    write_trajectory_partition,
)
from v4.research.pathd_phase1_entry import (
    build_causal_entry_session,
    development_sessions,
    load_entry_dataset,
    train_entry_campaign,
    write_causal_entry_session,
)
from v4.research.pathd_phase1_replay import run_four_box_replay
from v4.research.phase1_storage import (
    ARTIFACT_ROOT_ENV,
    DATA_ROOT_ENV,
    SCRATCH_ROOT_ENV,
    ResearchRoots,
    StorageContractError,
    preflight_roots,
    relocate_corpus,
)


def _receipt(path: Path) -> OOFEntryReceiptV1:
    payload = json.loads(path.read_text())
    receipt = OOFEntryReceiptV1(**payload)
    validate_oof_entry_receipt(receipt)
    return receipt


def _roots(args: argparse.Namespace) -> ResearchRoots:
    return ResearchRoots.resolve(
        data_root=args.data_root,
        scratch_root=args.scratch_root,
        artifact_root=args.artifact_root,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--volume-name", default="AR_TRADING_DATA")


def _corpus(roots: ResearchRoots) -> Path:
    preferred = roots.data_root / "vendor" / "pathd_2025-08-01_2026-07-31"
    legacy = roots.data_root / "pathd_2025-08-01_2026-07-31"
    for candidate in (preferred, legacy):
        if candidate.is_dir():
            return candidate
    raise StorageContractError(f"verified Path-D corpus is not present under {roots.data_root}")


def _frozen_lag(path: Path) -> int:
    lag = json.loads(path.read_text())
    semantic = dict(lag)
    expected = semantic.pop("receipt_sha256", None)
    if (
        expected != stable_hash(semantic)
        or
        lag.get("schema_version") != "pathd.shared-emission-lag.v1"
        or lag.get("status") != "FROZEN"
        or lag.get("theta_sample_count", 0) < 5
        or lag.get("frozen_emission_lag_ms") != 2_336
    ):
        raise StorageContractError("shared emission-lag receipt is not the frozen 2,336 ms law")
    return int(lag["frozen_emission_lag_ms"])


def _build_trajectory_bundle(
    task: tuple[str, str, str, int, bool]
) -> dict[str, int]:
    """Build one receipt's baseline and sensitivity partitions."""

    receipt_path, corpus_text, scratch_text, lag_ms, resume = task
    receipt = _receipt(Path(receipt_path))
    corpus = Path(corpus_text)
    scratch_root = Path(scratch_text)
    feature_path = scratch_root / "exit_features" / f"session={receipt.session}" / f"{receipt.trajectory_id}.parquet"
    label_path = scratch_root / "exit_labels" / f"session={receipt.session}" / f"{receipt.trajectory_id}.parquet"
    baseline_complete = feature_path.is_file() and label_path.is_file()
    if feature_path.exists() != label_path.exists():
        raise StorageContractError(f"partial trajectory: {receipt.trajectory_id}")
    sensitivity_specs = tuple(
        (fee, latency)
        for fee in (1.5, 2.0)
        for latency in (0, 1, 2, 5)
    )
    sensitivity_paths = {
        (fee, latency): sensitivity_label_path(
            scratch_root,
            session=receipt.session,
            trajectory_id=receipt.trajectory_id,
            fee_per_side_dollars=fee,
            latency_seconds=latency,
        )
        for fee, latency in sensitivity_specs
    }
    if baseline_complete and not resume:
        raise StorageContractError(f"partial or duplicate trajectory: {receipt.trajectory_id}")
    if baseline_complete and all(path.is_file() for path in sensitivity_paths.values()):
        return {"complete": 0, "skipped": 1, "sensitivity_written": 0}
    cbbo = load_exact_cbbo_path(
        corpus / "raw/databento/opra_spxw_cbbo_1s" / f"{receipt.session}.cbbo-1s.parquet",
        receipt,
    )
    spx = load_completed_spx_context(
        corpus / "raw/index/spx_1m" / f"{receipt.session}.official_spx.parquet",
        emission_lag_ms=lag_ms,
    )
    complete = 0
    sensitivity_written = 0
    if not baseline_complete:
        features, labels = build_trajectory_tables(receipt, cbbo, spx)
        write_trajectory_partition(
            features,
            labels,
            scratch_root=scratch_root,
            session=receipt.session,
            trajectory_id=receipt.trajectory_id,
        )
        complete = 1
    for fee, latency in sensitivity_specs:
        if sensitivity_paths[(fee, latency)].is_file() and resume:
            continue
        _, sensitivity_labels = build_trajectory_tables(
            receipt,
            cbbo,
            spx,
            fee_per_side_dollars=fee,
            latency_seconds=latency,
        )
        write_sensitivity_label_partition(
            sensitivity_labels,
            scratch_root=scratch_root,
            session=receipt.session,
            trajectory_id=receipt.trajectory_id,
            fee_per_side_dollars=fee,
            latency_seconds=latency,
        )
        sensitivity_written += 1
    return {
        "complete": complete,
        "skipped": 0,
        "sensitivity_written": sensitivity_written,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        epilog=(
            f"Defaults: {DATA_ROOT_ENV}, {SCRATCH_ROOT_ENV}, {ARTIFACT_ROOT_ENV}. "
            "No user-specific paths are hard coded."
        )
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    preflight = sub.add_parser("storage-preflight")
    _common(preflight)
    preflight.add_argument("--create-roots", action="store_true")
    relocate = sub.add_parser("relocate-corpus")
    _common(relocate)
    relocate.add_argument("--source", type=Path, required=True)
    relocate.add_argument("--destination-name", default="pathd_2025-08-01_2026-07-31")
    validate = sub.add_parser("validate-entry-receipt")
    validate.add_argument("--receipt", type=Path, required=True)
    build = sub.add_parser("build-trajectory")
    _common(build)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--emission-lag-receipt", type=Path, required=True)
    build.add_argument("--fee-per-side", type=float, choices=(1.5, 2.0), default=1.5)
    build.add_argument("--latency-seconds", type=int, choices=(0, 1, 2, 5), default=1)
    materialize = sub.add_parser("materialize-entry")
    _common(materialize)
    materialize.add_argument("--session", action="append", default=[])
    materialize.add_argument("--resume", action="store_true")
    train_entry = sub.add_parser("train-entry")
    _common(train_entry)
    build_all = sub.add_parser("build-trajectories")
    _common(build_all)
    build_all.add_argument("--entry-campaign", type=Path, required=True)
    build_all.add_argument("--emission-lag-receipt", type=Path, required=True)
    build_all.add_argument("--resume", action="store_true")
    build_all.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    train_exit = sub.add_parser("train-exit")
    _common(train_exit)
    train_exit.add_argument("--entry-campaign", type=Path, required=True)
    replay = sub.add_parser("replay")
    _common(replay)
    replay.add_argument("--entry-campaign", type=Path, required=True)
    replay.add_argument("--exit-campaign", type=Path, required=True)
    status = sub.add_parser("status")
    _common(status)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "validate-entry-receipt":
        print(json.dumps(asdict(_receipt(args.receipt)), indent=2, sort_keys=True))
        return 0
    roots = _roots(args)
    if args.stage == "storage-preflight":
        print(json.dumps(preflight_roots(
            roots, expected_volume_name=args.volume_name, create=args.create_roots
        ), indent=2, sort_keys=True))
        return 0
    if args.stage == "relocate-corpus":
        preflight_roots(roots, expected_volume_name=args.volume_name, create=True)
        destination = roots.data_root / "vendor" / args.destination_name
        print(json.dumps(relocate_corpus(
            args.source, destination, expected_volume_name=args.volume_name
        ), indent=2, sort_keys=True))
        return 0
    if args.stage == "build-trajectory":
        preflight_roots(roots, expected_volume_name=args.volume_name)
        receipt = _receipt(args.receipt)
        lag_ms = _frozen_lag(args.emission_lag_receipt)
        corpus = _corpus(roots)
        cbbo_path = corpus / "raw/databento/opra_spxw_cbbo_1s" / f"{receipt.session}.cbbo-1s.parquet"
        spx_path = corpus / "raw/index/spx_1m" / f"{receipt.session}.official_spx.parquet"
        cbbo = load_exact_cbbo_path(cbbo_path, receipt)
        spx = load_completed_spx_context(
            spx_path, emission_lag_ms=lag_ms
        )
        features, labels = build_trajectory_tables(
            receipt,
            cbbo,
            spx,
            fee_per_side_dollars=args.fee_per_side,
            latency_seconds=args.latency_seconds,
        )
        result: dict[str, Any] = write_trajectory_partition(
            features,
            labels,
            scratch_root=roots.scratch_root,
            session=receipt.session,
            trajectory_id=receipt.trajectory_id,
        )
        result.update({"feature_rows": len(features), "label_rows": len(labels)})
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.stage == "materialize-entry":
        preflight_roots(roots, expected_volume_name=args.volume_name)
        corpus = _corpus(roots)
        available = list(development_sessions(corpus))
        sessions = sorted(set(map(str, args.session))) if args.session else available
        unknown = sorted(set(sessions) - set(available))
        if unknown:
            raise StorageContractError(f"requested sessions are not in the corpus: {unknown[:5]}")
        canonical = roots.data_root / "canonical"
        canonical.mkdir(parents=True, exist_ok=True)
        written: list[dict[str, Any]] = []
        skipped: list[str] = []
        for session in sessions:
            target = canonical / "entry_v2" / f"session={session}"
            if target.exists():
                if not args.resume:
                    raise StorageContractError(f"entry partition already exists: {target}")
                skipped.append(session)
                continue
            states, labels, receipt = build_causal_entry_session(corpus, session)
            written.append(
                write_causal_entry_session(
                    states, labels, receipt, canonical_root=canonical
                )
            )
        print(json.dumps({
            "status": "COMPLETE",
            "requested_sessions": len(sessions),
            "written_sessions": len(written),
            "skipped_sessions": skipped,
            "partitions": written,
        }, indent=2, sort_keys=True))
        return 0
    if args.stage == "train-entry":
        preflight_roots(roots, expected_volume_name=args.volume_name)
        dataset = load_entry_dataset(roots.data_root / "canonical")
        allowed = set(development_sessions(_corpus(roots)))
        observed = set(dataset["session"].astype(str))
        if observed != allowed:
            raise StorageContractError(
                f"entry dataset must contain exactly 215 development sessions; "
                f"missing={len(allowed-observed)} extra_or_firewall={len(observed-allowed)}"
            )
        print(json.dumps(train_entry_campaign(
            dataset, artifact_root=roots.artifact_root
        ), indent=2, sort_keys=True))
        return 0
    if args.stage == "build-trajectories":
        preflight_roots(roots, expected_volume_name=args.volume_name)
        lag_ms = _frozen_lag(args.emission_lag_receipt)
        campaign = json.loads(args.entry_campaign.read_text())
        index = pd.read_parquet(campaign["trajectory_index_path"])
        corpus = _corpus(roots)
        complete = 0
        skipped = 0
        sensitivity_written = 0
        sensitivity_specs = tuple(
            (fee, latency)
            for fee in (1.5, 2.0)
            for latency in (0, 1, 2, 5)
        )
        tasks = [
            (
                str(row["receipt_path"]),
                str(corpus),
                str(roots.scratch_root),
                lag_ms,
                bool(args.resume),
            )
            for row in index.sort_values(["session", "trajectory_id"]).to_dict("records")
        ]
        if args.workers == 1:
            results = map(_build_trajectory_bundle, tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            results = executor.map(_build_trajectory_bundle, tasks, chunksize=1)
        try:
            for result in results:
                complete += result["complete"]
                skipped += result["skipped"]
                sensitivity_written += result["sensitivity_written"]
        finally:
            if args.workers != 1:
                executor.shutdown(cancel_futures=True)
        print(json.dumps({
            "status": "COMPLETE",
            "trajectory_partitions_written": complete,
            "trajectory_partitions_skipped": skipped,
            "sensitivity_label_partitions_written": sensitivity_written,
            "sensitivity_specs": [
                {"roundtrip_fee_dollars": fee * 2.0, "latency_seconds": latency}
                for fee, latency in sensitivity_specs
            ],
            "source_index_rows": len(index),
        }, indent=2, sort_keys=True))
        return 0
    if args.stage == "train-exit":
        preflight_roots(roots, expected_volume_name=args.volume_name)
        print(json.dumps(train_exit_campaign(
            scratch_root=roots.scratch_root,
            entry_campaign_path=args.entry_campaign,
            artifact_root=roots.artifact_root,
        ), indent=2, sort_keys=True))
        return 0
    if args.stage == "replay":
        preflight_roots(roots, expected_volume_name=args.volume_name)
        print(json.dumps(run_four_box_replay(
            scratch_root=roots.scratch_root,
            entry_campaign_path=args.entry_campaign,
            exit_campaign_path=args.exit_campaign,
            report_root=roots.data_root / "reports" / "phase1_four_box",
        ), indent=2, sort_keys=True))
        return 0
    if args.stage == "status":
        storage = preflight_roots(roots, expected_volume_name=args.volume_name)
        payload = {
            "status": "READY",
            "storage": storage,
            "corpus": str(_corpus(roots)),
            "entry_campaign": str(roots.artifact_root / "entry_v2/campaign.json"),
            "exit_campaign": str(roots.artifact_root / "exit_v1/campaign.json"),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
