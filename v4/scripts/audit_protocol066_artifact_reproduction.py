from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.live.protocol066_inference import (
    OVERRIDE_THRESHOLD_EPSILON,
    load_protocol066_artifact,
    predict_protocol066_sequence,
)


DEFAULT_SELECTED = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_069_protocol066_artifact_persistence/"
    "selected_trades_sequence_exits.json"
)
DEFAULT_SEQUENCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_060_lifecycle_sequence_dataset")
DEFAULT_ARTIFACT_ROOT = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_069_protocol066_artifact_persistence/model_artifacts"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_072_protocol066_artifact_reproduction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-trades", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-artifacts", type=int, default=0)
    return parser.parse_args()


def _load_selected(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(path.read_text()))
    if frame.empty:
        raise SystemExit(f"selected trade file is empty: {path}")
    frame["trade_uid"] = frame["trade_uid"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype(int)
    frame["candidate_exit_step"] = pd.to_numeric(frame["candidate_exit_step"], errors="coerce").astype(int)
    frame["candidate_pnl"] = pd.to_numeric(frame["candidate_pnl"], errors="coerce")
    return frame


def _load_sequence(sequence_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_parquet(sequence_dir / "protocol054_lifecycle_trades.parquet")
    steps = pd.read_parquet(sequence_dir / "protocol054_lifecycle_steps.parquet")
    trades["trade_uid"] = trades["trade_uid"].astype(str)
    steps["trade_uid"] = steps["trade_uid"].astype(str)
    return trades, steps


def _expected_for_artifact(selected: pd.DataFrame, *, split: str, seed: int) -> pd.DataFrame:
    return selected[(selected["split"].eq(split)) & (selected["seed"].eq(seed))].copy()


def _simulate_one(artifact, trade: pd.Series, steps: pd.DataFrame) -> dict[str, Any]:
    ordered = steps.sort_values("step_idx").reset_index(drop=True)
    value, recovery, decay = predict_protocol066_sequence(artifact, ordered)
    protocol054_exit_step = int(trade["protocol054_exit_step"])
    exit_idx = protocol054_exit_step
    reason = "protocol054_fallback"
    for local_idx, row in ordered.iloc[: protocol054_exit_step + 1].iterrows():
        baseline_reason = str(row.get("baseline_exit_reason", ""))
        if bool(row.get("is_baseline_exit_step")) and baseline_reason in {"hard_stop", "target"}:
            exit_idx = int(local_idx)
            reason = baseline_reason
            break
        if (
            int(local_idx) < protocol054_exit_step
            and float(value[int(local_idx)]) > artifact.selected_override_threshold + OVERRIDE_THRESHOLD_EPSILON
        ):
            exit_idx = int(local_idx)
            reason = "sequence_residual_override"
            break
    exit_row = ordered.iloc[exit_idx]
    return {
        "candidate_exit_step": int(exit_idx),
        "candidate_exit_reason": reason,
        "candidate_pnl": float(exit_row["current_pnl"]),
        "predicted_continuation_value": float(value[exit_idx]),
        "predicted_recovery_probability": float(recovery[exit_idx]),
        "predicted_decay_probability": float(decay[exit_idx]),
    }


def _close_enough(left: float, right: float, *, atol: float = 1e-4) -> bool:
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return False
    return abs(float(left) - float(right)) <= atol


def _artifact_summary(
    *,
    manifest_path: Path,
    selected: pd.DataFrame,
    trades: pd.DataFrame,
    steps: pd.DataFrame,
) -> dict[str, Any]:
    artifact = load_protocol066_artifact(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    split = str(manifest["test_split"])
    expected = _expected_for_artifact(selected, split=split, seed=artifact.seed)
    trade_lookup = trades.set_index("trade_uid", drop=False)
    step_groups = {uid: frame for uid, frame in steps[steps["trade_uid"].isin(expected["trade_uid"])].groupby("trade_uid")}
    mismatches = []
    for row in expected.itertuples(index=False):
        uid = str(row.trade_uid)
        if uid not in trade_lookup.index or uid not in step_groups:
            mismatches.append({"trade_uid": uid, "kind": "missing_path"})
            continue
        actual = _simulate_one(artifact, trade_lookup.loc[uid], step_groups[uid])
        expected_reason = str(row.candidate_exit_reason)
        expected_step = int(row.candidate_exit_step)
        expected_pnl = float(row.candidate_pnl)
        if (
            actual["candidate_exit_reason"] != expected_reason
            or actual["candidate_exit_step"] != expected_step
            or not _close_enough(actual["candidate_pnl"], expected_pnl)
        ):
            mismatches.append(
                {
                    "trade_uid": uid,
                    "expected_reason": expected_reason,
                    "actual_reason": actual["candidate_exit_reason"],
                    "expected_step": expected_step,
                    "actual_step": actual["candidate_exit_step"],
                    "expected_pnl": expected_pnl,
                    "actual_pnl": actual["candidate_pnl"],
                }
            )
    reason_counts = expected["candidate_exit_reason"].value_counts().to_dict()
    fallback = expected[expected["candidate_exit_reason"].eq("protocol054_fallback")]
    return {
        "manifest": str(manifest_path),
        "fold": artifact.fold,
        "test_split": split,
        "seed": artifact.seed,
        "rows": int(len(expected)),
        "mismatches": int(len(mismatches)),
        "mismatch_examples": mismatches[:10],
        "reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
        "protocol054_fallback_rows": int(len(fallback)),
        "protocol054_fallback_fraction": float(len(fallback) / max(1, len(expected))),
    }


def _aggregate_dependency(selected: pd.DataFrame) -> dict[str, Any]:
    reason_counts = selected["candidate_exit_reason"].value_counts().to_dict()
    by_split = (
        selected.groupby(["split", "candidate_exit_reason"]).size().unstack(fill_value=0).astype(int).to_dict(orient="index")
    )
    fallback = selected[selected["candidate_exit_reason"].eq("protocol054_fallback")]
    fallback_reason_counts = fallback["protocol054_exit_reason"].value_counts().to_dict()
    pnl_by_reason = (
        selected.groupby("candidate_exit_reason")["candidate_pnl"]
        .agg(["count", "sum", "median", "mean"])
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "total_rows": int(len(selected)),
        "reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
        "reason_fractions": {str(key): float(value / max(1, len(selected))) for key, value in reason_counts.items()},
        "by_split_reason_counts": {str(split): {str(k): int(v) for k, v in row.items()} for split, row in by_split.items()},
        "protocol054_fallback_reason_counts": {str(key): int(value) for key, value in fallback_reason_counts.items()},
        "pnl_by_candidate_reason": pnl_by_reason,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    dep = payload["protocol066_dependency"]
    lines = [
        "# Protocol 066 Artifact Reproduction Audit",
        "",
        "No paid data was downloaded. This audit reloads persisted model/scaler artifacts and checks that they reproduce the frozen research selected exits.",
        "",
        "## Result",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Artifacts checked: `{payload['artifacts_checked']}`",
        f"- Rows checked: `{payload['rows_checked']}`",
        f"- Mismatches: `{payload['mismatches']}`",
        "",
        "## Dependency",
        "",
        f"- Total selected rows: `{dep['total_rows']}`",
        f"- Candidate exit reason counts: `{dep['reason_counts']}`",
        f"- Candidate exit reason fractions: `{dep['reason_fractions']}`",
        f"- Protocol 054 fallback reason counts: `{dep['protocol054_fallback_reason_counts']}`",
        "",
        "## Interpretation",
        "",
        "Persisted Protocol 066 artifacts can be treated as reproducible only if mismatches are zero. A large Protocol 054 fallback fraction means live/paper readiness still requires an explicit persisted fallback lifecycle engine or a new self-contained lifecycle model.",
    ]
    if payload["mismatch_examples"]:
        lines.extend(["", "## First Mismatches", ""])
        for mismatch in payload["mismatch_examples"][:20]:
            lines.append(f"- `{mismatch}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = _load_selected(args.selected_trades)
    trades, steps = _load_sequence(args.sequence_dir)
    manifests = sorted(args.artifact_root.glob("*/seed_*/manifest.json"))
    if args.max_artifacts > 0:
        manifests = manifests[: args.max_artifacts]
    if not manifests:
        raise SystemExit(f"no artifact manifests found under {args.artifact_root}")
    summaries = [
        _artifact_summary(manifest_path=manifest, selected=selected, trades=trades, steps=steps)
        for manifest in manifests
    ]
    mismatch_examples = [example for summary in summaries for example in summary["mismatch_examples"]]
    payload = {
        "decision": "pass" if sum(summary["mismatches"] for summary in summaries) == 0 else "fail",
        "artifacts_checked": len(summaries),
        "rows_checked": int(sum(summary["rows"] for summary in summaries)),
        "mismatches": int(sum(summary["mismatches"] for summary in summaries)),
        "mismatch_examples": mismatch_examples[:50],
        "artifact_summaries": summaries,
        "protocol066_dependency": _aggregate_dependency(selected),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(args.out_dir / "report.md")
    print(json.dumps({key: payload[key] for key in ["decision", "artifacts_checked", "rows_checked", "mismatches"]}, sort_keys=True))
    return 0 if payload["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
