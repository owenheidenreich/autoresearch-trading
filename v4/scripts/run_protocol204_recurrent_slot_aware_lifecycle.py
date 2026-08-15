"""Protocol204: recurrent slot-aware lifecycle policy.

Protocol202's slot-aware MLP fixed the major over-holding failure from
Protocol200, but still missed recent 2026 slightly. Protocol203 attributed the
remaining gap to lifecycle timing, not a new entry-side problem.

Protocol204 keeps the same frozen Protocol194 entry stream and the same
slot-aware dynamic-programming target, but replaces the per-step MLP with a
causal GRU over the holding-state path.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money, pct
from v4.scripts.run_protocol200_lifecycle_continuation_policy import (
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_REPLAY_DIRS,
    TARGET_SCALE,
    build_path_records,
    compare_summaries,
    count_by,
    fit_training_matrix,
    fold_specs,
    load_candidate_entries,
    select_threshold,
    serial_invariants,
    simulate_baseline_serial,
    simulate_serial,
    summarize_replay,
    threshold_summary,
)
from v4.scripts.run_protocol202_slot_aware_lifecycle_policy import apply_slot_aware_targets, decide_protocol202


LOOP_ID = "v4_aplus_hypothesis_204_recurrent_slot_aware_lifecycle"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")


@dataclass
class SequenceBundle:
    records: list[Any]
    features: np.ndarray
    target: np.ndarray
    mask: np.ndarray


class SlotAwareGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input(x)
        h, _ = self.gru(z)
        return self.head(h).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", action="append", type=Path, default=None)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--max-train-steps", type=int, default=650_000)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay_dirs = args.replay_dir or DEFAULT_REPLAY_DIRS
    candidates = load_candidate_entries(replay_dirs)
    records, path_skips = build_path_records(candidates, normalized_dir=args.normalized_dir, forced_flat_time=args.forced_flat_time)
    apply_slot_aware_targets(records)
    baseline_rows = []
    model_rows = []
    threshold_rows = []
    fold_payloads = []
    for spec in fold_specs():
        train_records = [r for r in records if r.reported_split in spec["train_splits"]]
        validation_records = [r for r in records if r.reported_split == spec["validation_split"]]
        if not train_records or not validation_records:
            continue
        _, _, scaler = fit_training_matrix(train_records, max_train_steps=args.max_train_steps, seed=204)
        train_bundle = build_sequence_bundle(train_records, scaler)
        validation_bundle = build_sequence_bundle(validation_records, scaler)
        for model_seed in args.seeds:
            model, history = train_sequence_model(
                train_bundle,
                seed=int(model_seed),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                hidden_dim=int(args.hidden_dim),
                learning_rate=float(args.learning_rate),
            )
            validation_predictions = predict_sequence_records(model, validation_bundle)
            threshold, sweep = select_threshold(
                validation_records,
                validation_predictions,
                split_name=str(spec["validation_split"]),
                model_seed=int(model_seed),
            )
            threshold_rows.extend({**row, "fold": spec["fold"], "model_seed": int(model_seed)} for row in sweep)
            for split in spec["test_splits"]:
                split_records = [r for r in records if r.reported_split == split]
                split_bundle = build_sequence_bundle(split_records, scaler)
                predictions = predict_sequence_records(model, split_bundle)
                model_rows.extend(
                    simulate_serial(
                        split_records,
                        predictions,
                        threshold=threshold,
                        model_seed=int(model_seed),
                        strategy=f"protocol204:{spec['fold']}:seed{model_seed}",
                    )
                )
                if split == "q1_2026":
                    march_records = [r for r in split_records if r.session >= "2026-03-01"]
                    march_bundle = build_sequence_bundle(march_records, scaler)
                    march_predictions = predict_sequence_records(model, march_bundle)
                    model_rows.extend(
                        {**row, "reported_split": "march_2026"}
                        for row in simulate_serial(
                            march_records,
                            march_predictions,
                            threshold=threshold,
                            model_seed=int(model_seed),
                            strategy=f"protocol204:{spec['fold']}:seed{model_seed}:march_subset",
                        )
                    )
            fold_payloads.append(
                {
                    "fold": spec["fold"],
                    "train_splits": list(spec["train_splits"]),
                    "validation_split": spec["validation_split"],
                    "test_splits": list(spec["test_splits"]),
                    "model_seed": int(model_seed),
                    "threshold": float(threshold),
                    "history": history,
                    "train_records": len(train_records),
                    "validation_records": len(validation_records),
                }
            )
        for split in [spec["validation_split"], *spec["test_splits"]]:
            split_records = [r for r in records if r.reported_split == split]
            baseline_rows.extend(simulate_baseline_serial(split_records, strategy="protocol194_protocol081_baseline"))
            if split == "q1_2026":
                march_records = [r for r in split_records if r.session >= "2026-03-01"]
                baseline_rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in simulate_baseline_serial(march_records, strategy="protocol194_protocol081_baseline:march_subset")
                )
    baseline_frame = pd.DataFrame(baseline_rows).drop_duplicates(
        ["reported_split", "entry_seed", "session", "decision_time", "contract_id", "strategy"],
        keep="last",
    )
    model_frame = pd.DataFrame(model_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    baseline_summary = summarize_replay(baseline_frame, seed_col="entry_seed")
    model_summary = summarize_replay(model_frame, seed_col="combo_seed")
    comparison = compare_summaries(model_summary, baseline_summary)
    invariants = serial_invariants(model_frame)
    payload = {
        "protocol": "204_recurrent_slot_aware_lifecycle",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": True,
        "source_replay_dirs": [str(path) for path in replay_dirs],
        "normalized_dir": str(args.normalized_dir),
        "row_counts": {
            "candidate_entries": int(len(candidates)),
            "path_records": int(len(records)),
            "path_skips": int(len(path_skips)),
            "baseline_trade_rows": int(len(baseline_frame)),
            "model_trade_rows": int(len(model_frame)),
        },
        "pre_registration": {
            "hypothesis": (
                "A recurrent holding-state model should improve the slot-aware lifecycle target by reading "
                "the shape of the post-entry path, not just per-minute summary features."
            ),
            "entry_stream": "frozen Protocol194 candidate entries",
            "holding_action_space": "hold or exit",
            "flat_action_space": "unchanged for this protocol",
            "threshold_selection": "validation split only",
        },
        "folds": fold_payloads,
        "baseline_summary": baseline_summary,
        "model_summary": model_summary,
        "comparison": comparison,
        "threshold_summary": threshold_summary(threshold_frame),
        "invariants": invariants,
        "path_skip_counts": count_by(path_skips, "skip_reason"),
        "decision": decide_protocol202(comparison, invariants).replace("protocol202", "protocol204"),
        "next_gate": next_gate(comparison),
    }
    baseline_frame.to_csv(args.out_dir / "protocol194_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "protocol204_model_serial_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(path_skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def build_sequence_bundle(records: Sequence[Any], scaler: FeatureScaler) -> SequenceBundle:
    if not records:
        return SequenceBundle(records=[], features=np.zeros((0, 0, 0), dtype=np.float32), target=np.zeros((0, 0), dtype=np.float32), mask=np.zeros((0, 0), dtype=np.float32))
    max_len = max(len(record.path_pnl) for record in records)
    feature_dim = records[0].features.shape[1]
    features = np.zeros((len(records), max_len, feature_dim), dtype=np.float32)
    target = np.zeros((len(records), max_len), dtype=np.float32)
    mask = np.zeros((len(records), max_len), dtype=np.float32)
    for idx, record in enumerate(records):
        length = len(record.path_pnl)
        features[idx, :length, :] = scaler.transform(record.features)
        target[idx, :length] = record.target.astype(np.float32) / TARGET_SCALE
        mask[idx, :length] = 1.0
    return SequenceBundle(records=list(records), features=features, target=target, mask=mask)


def train_sequence_model(
    bundle: SequenceBundle,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
) -> tuple[SlotAwareGRU, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SlotAwareGRU(input_dim=bundle.features.shape[-1], hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.features), torch.from_numpy(bundle.target), torch.from_numpy(bundle.mask)),
        batch_size=min(batch_size, len(bundle.records)),
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            raw_loss = F.huber_loss(pred, yb, reduction="none", delta=1.0)
            loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": float(epoch), "train_loss": epoch_loss})
    model.load_state_dict(best_state)
    return model, history


def predict_sequence_records(model: SlotAwareGRU, bundle: SequenceBundle) -> dict[str, np.ndarray]:
    if not bundle.records:
        return {}
    model.eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        pred = model(torch.from_numpy(bundle.features)).cpu().numpy() * TARGET_SCALE
    for idx, record in enumerate(bundle.records):
        length = len(record.path_pnl)
        out[record.uid] = pred[idx, :length].astype(np.float32)
    return out


def next_gate(comparison: list[dict[str, Any]]) -> str:
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in ["q1_2026", "march_2026", "recent_2026"]):
        return "Run five-seed confirmation and Protocol203-style attribution for Protocol204."
    if any(by_split.get(split, {}).get("beats_baseline") for split in ["q1_2026", "march_2026", "recent_2026"]):
        return "Attribute the mixed recurrent result before changing the objective."
    return "Reject recurrent slot-aware lifecycle at this gate; return to unified entry/lifecycle training."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol204 Recurrent Slot-Aware Lifecycle",
        "",
        "No paid data was downloaded. No broker endpoint was called. No live or paper orders were placed.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Next gate: {payload['next_gate']}",
        "",
        "## Comparison",
        "",
        "| split | model PnL | baseline PnL | delta | model PF | baseline PF | model trades | baseline trades | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | "
            f"{money(row['baseline_median_total_pnl'])} | {money(row['delta_vs_baseline'])} | "
            f"{row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} | "
            f"{row['model_median_trades']:.0f} | {row['baseline_median_trades']:.0f} | "
            f"{pct(row['model_positive_seed_fraction'])} |"
        )
    lines.extend(["", "## Thresholds", ""])
    if payload["threshold_summary"]:
        lines.extend(["| fold | seed | threshold | validation PnL | PF | trades |", "|---|---:|---:|---:|---:|---:|"])
        for row in payload["threshold_summary"]:
            lines.append(
                f"| {row['fold']} | {row['model_seed']} | {row['selected_threshold']:.0f} | "
                f"{money(row['validation_total_pnl'])} | {row['validation_profit_factor']:.3f} | {row['validation_trades']} |"
            )
    lines.extend(
        [
            "",
            "## Invariants",
            "",
            f"- Overlap violations: `{payload['invariants']['overlap_violations']}`",
            f"- Unaffordable violations: `{payload['invariants']['unaffordable_violations']}`",
            f"- NaN time rows: `{payload['invariants']['nan_time_rows']}`",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Model trades: `{path.parent / 'protocol204_model_serial_trades.csv'}`",
            f"- Baseline trades: `{path.parent / 'protocol194_baseline_serial_trades.csv'}`",
            f"- Threshold sweep: `{path.parent / 'threshold_sweep.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
