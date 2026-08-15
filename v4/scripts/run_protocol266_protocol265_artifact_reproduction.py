"""AUDIT_PROTOCOL265_ARTIFACT_REPRODUCTION_V1.

Historically Protocol266. The engineer review correctly flagged that
Protocol265 was a research result, not a deployable candidate, because it did
not persist model/scaler/manifest artifacts or prove that saved artifacts can
recreate the exact strict-serial replay.

This audit loads only the saved Protocol265 artifacts and reproduces the
Protocol265 trade ledger. No model is trained here. No paid data is downloaded.
No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.model.supervised_pilot import FeatureScaler
import v4.scripts.run_protocol200_lifecycle_continuation_policy as p200
import v4.scripts.run_protocol265_source_penalty_baseline_anchored_continuation as p265
from v4.scripts.run_protocol251_premium_blend_slot_aware_lifecycle import (
    load_premium_blend_candidates,
    safe_find_normalized_path,
)


ROLE_LABEL = "AUDIT_PROTOCOL265_ARTIFACT_REPRODUCTION_V1"
HISTORICAL_ID = "Protocol266"
SOURCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_266_protocol265_artifact_reproduction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.source_dir / "summary.json"
    reference_path = args.source_dir / "source_penalty_baseline_anchored_continuation_trades.csv"
    artifact_root = args.source_dir / "model_artifacts"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    if not artifact_root.exists():
        raise FileNotFoundError(f"{artifact_root} is missing; rerun Protocol265 to persist artifacts")

    source_summary = json.loads(summary_path.read_text())
    data_used = source_summary.get("data_used", {})
    trades_path = Path(data_used.get("source_penalty_trades", p265.DEFAULT_TRADES))
    normalized_dir = Path(data_used.get("normalized_dir", p265.DEFAULT_NORMALIZED_DIR))
    forced_flat_time = str(source_summary.get("pre_registration", {}).get("forced_flat_time", "15:55"))
    forced_flat_time = str(source_summary.get("folds", [{}])[0].get("forced_flat_time", forced_flat_time))

    p200.find_normalized_path = safe_find_normalized_path
    candidates = load_premium_blend_candidates(trades_path)
    records, path_skips = p265.build_path_records(candidates, normalized_dir=normalized_dir, forced_flat_time=forced_flat_time)
    if not records:
        raise SystemExit("no executable path records were rebuilt for artifact reproduction")

    rows: list[dict[str, Any]] = []
    artifact_manifests = sorted(artifact_root.glob("*/seed_*/manifest.json"))
    if not artifact_manifests:
        raise FileNotFoundError(f"no manifest.json files found below {artifact_root}")
    for manifest_path in artifact_manifests:
        artifact = load_artifact(manifest_path)
        fold = str(artifact["manifest"]["fold"])
        model_seed = int(artifact["manifest"]["seed"])
        threshold = float(artifact["manifest"]["threshold"])
        for split in artifact["manifest"]["test_splits"]:
            split_records = [record for record in records if record.reported_split == str(split)]
            predictions = p265.predict_records(artifact["model"], artifact["scaler"], split_records)
            rows.extend(
                p265.simulate_anchor_serial(
                    split_records,
                    predictions,
                    threshold=threshold,
                    model_seed=model_seed,
                    strategy=f"{p265.CANDIDATE_LABEL}:{fold}:seed{model_seed}",
                )
            )
            if str(split) == "q1_2026":
                march_records = [record for record in split_records if record.session >= "2026-03-01"]
                march_predictions = {record.uid: predictions[record.uid] for record in march_records if record.uid in predictions}
                rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in p265.simulate_anchor_serial(
                        march_records,
                        march_predictions,
                        threshold=threshold,
                        model_seed=model_seed,
                        strategy=f"{p265.CANDIDATE_LABEL}:{fold}:seed{model_seed}:march_subset",
                    )
                )

    reproduced = pd.DataFrame(rows)
    reference = pd.read_csv(reference_path)
    comparison = compare_trade_ledgers(reproduced, reference)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "audit / saved-artifact replay reproduction for Protocol265",
        "changes_paper_default": False,
        "candidate_label": p265.CANDIDATE_LABEL,
        "baseline": p265.PAPER_DEFAULT_LABEL,
        "source_protocol": p265.HISTORICAL_ID,
        "source_dir": str(args.source_dir),
        "artifact_root": str(artifact_root),
        "artifact_count": int(len(artifact_manifests)),
        "data_used": {
            "source_penalty_trades": str(trades_path),
            "normalized_dir": str(normalized_dir),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "row_counts": {
            "candidate_entries": int(len(candidates)),
            "path_records": int(len(records)),
            "path_skips": int(len(path_skips)),
            "reference_rows": int(len(reference)),
            "reproduced_rows": int(len(reproduced)),
        },
        "reproduced_summary": p200.summarize_replay(reproduced, seed_col="combo_seed"),
        "reference_summary": p200.summarize_replay(reference, seed_col="combo_seed"),
        "comparison": comparison,
        "decision": decide(comparison),
    }
    reproduced.to_csv(args.out_dir / "reproduced_protocol265_trades.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_artifact(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    config = manifest.get("config", {})
    scaler_payload = json.loads((manifest_path.parent / "scaler.json").read_text())
    scaler = FeatureScaler(
        fill=np.asarray(scaler_payload["fill"], dtype=np.float32),
        mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
        std=np.asarray(scaler_payload["std"], dtype=np.float32),
    )
    feature_columns = manifest.get("feature_columns", p200.FEATURE_COLUMNS)
    model = p200.ContinuationMLP(input_dim=len(feature_columns), hidden_dim=int(config.get("hidden_dim", 128)))
    model.load_state_dict(torch.load(manifest_path.parent / "model.pt", map_location="cpu"))
    model.eval()
    return {"manifest": manifest, "model": model, "scaler": scaler}


def compare_trade_ledgers(reproduced: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    key_cols = [
        "reported_split",
        "fold",
        "model_seed",
        "entry_seed",
        "combo_seed",
        "session",
        "decision_time",
        "contract_id",
        "strategy",
    ]
    if reproduced.empty or reference.empty:
        return {
            "rows_match": bool(len(reproduced) == len(reference)),
            "reference_rows": int(len(reference)),
            "reproduced_rows": int(len(reproduced)),
            "missing_keys": int(len(reference)),
            "extra_keys": int(len(reproduced)),
            "max_abs_pnl_diff": None,
            "exit_time_mismatches": None,
            "exit_step_mismatches": None,
        }
    ref = reference.copy()
    rep = reproduced.copy()
    for column in key_cols:
        ref[column] = ref[column].astype(str)
        rep[column] = rep[column].astype(str)
    merged = ref.merge(rep, on=key_cols, how="outer", suffixes=("_reference", "_reproduced"), indicator=True)
    both = merged[merged["_merge"] == "both"].copy()
    pnl_diff = pd.to_numeric(both["pnl_reference"], errors="coerce") - pd.to_numeric(both["pnl_reproduced"], errors="coerce")
    exit_time_mismatches = 0
    exit_step_mismatches = 0
    if not both.empty:
        exit_time_mismatches = int((both["exit_time_reference"].astype(str) != both["exit_time_reproduced"].astype(str)).sum())
        if "exit_step_reference" in both.columns and "exit_step_reproduced" in both.columns:
            exit_step_mismatches = int(
                (
                    pd.to_numeric(both["exit_step_reference"], errors="coerce").fillna(-1)
                    != pd.to_numeric(both["exit_step_reproduced"], errors="coerce").fillna(-1)
                ).sum()
            )
    return {
        "rows_match": bool(len(reproduced) == len(reference)),
        "reference_rows": int(len(reference)),
        "reproduced_rows": int(len(reproduced)),
        "missing_keys": int((merged["_merge"] == "left_only").sum()),
        "extra_keys": int((merged["_merge"] == "right_only").sum()),
        "max_abs_pnl_diff": float(np.nanmax(np.abs(pnl_diff.to_numpy(dtype=float)))) if len(pnl_diff) else None,
        "exit_time_mismatches": exit_time_mismatches,
        "exit_step_mismatches": exit_step_mismatches,
    }


def decide(comparison: dict[str, Any]) -> str:
    if not comparison.get("rows_match"):
        return "reject_protocol265_artifact_reproduction_row_count_mismatch"
    if int(comparison.get("missing_keys") or 0) or int(comparison.get("extra_keys") or 0):
        return "reject_protocol265_artifact_reproduction_key_mismatch"
    if int(comparison.get("exit_time_mismatches") or 0) or int(comparison.get("exit_step_mismatches") or 0):
        return "reject_protocol265_artifact_reproduction_exit_mismatch"
    max_diff = comparison.get("max_abs_pnl_diff")
    if max_diff is None or float(max_diff) > 1e-6:
        return "reject_protocol265_artifact_reproduction_pnl_drift"
    return "pass_protocol265_artifact_reproduction_exact"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    comparison = payload["comparison"]
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being audited: {payload['candidate_label']}",
        f"Baseline: {payload['baseline']}",
        f"Source protocol: `{payload['source_protocol']}`",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Model training: {payload['model_training']}",
        f"Decision: `{payload['decision']}`",
        "",
        "## Reproduction Check",
        "",
        f"- Artifact manifests loaded: `{payload['artifact_count']}`",
        f"- Reference rows: `{comparison['reference_rows']}`",
        f"- Reproduced rows: `{comparison['reproduced_rows']}`",
        f"- Missing keys: `{comparison['missing_keys']}`",
        f"- Extra keys: `{comparison['extra_keys']}`",
        f"- Max absolute PnL diff: `{comparison['max_abs_pnl_diff']}`",
        f"- Exit-time mismatches: `{comparison['exit_time_mismatches']}`",
        f"- Exit-step mismatches: `{comparison['exit_step_mismatches']}`",
        "",
        "## Outputs",
        "",
        f"- Summary: `{path.parent / 'summary.json'}`",
        f"- Reproduced trades: `{path.parent / 'reproduced_protocol265_trades.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
    if marker in ledger.read_text():
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    marker,
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    f"- Candidate audited: {payload['candidate_label']}",
                    f"- Source protocol: `{payload['source_protocol']}`",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    "- Model training: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
