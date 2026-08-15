"""EXP_2026_05_22_UNIFIED_ENTRY_LIFECYCLE_SEQUENCE_V1.

Historically Protocol209. This is a research experiment, not a paper/live
promotion.

The repeated failure mode is lifecycle intelligence: the bot often has a good
entry stream, but separate entry and exit learners can still teach churn or
premature exits. This experiment trains one shared neural policy over the same
single-slot sequence:

* flat state: is this candidate worth occupying the only position slot?
* holding state: is this contract still worth holding, given future slot value?

It does not hardcode hold times, fixed profit targets, or percentage exits.
Labels use hindsight only for supervised training; evaluation remains strict
one-account serial replay.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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
    STARTING_CASH,
    TARGET_CLIP,
    TARGET_SCALE,
    THRESHOLD_CANDIDATES,
    build_path_records,
    count_by,
    exit_index_from_prediction,
    finite_float,
    fold_specs,
    load_candidate_entries,
    metrics_for_rows,
    serial_invariants,
    simulate_baseline_serial,
    summarize_replay,
)
from v4.scripts.run_protocol207_lifecycle_context_calibration import (
    PAPER_DEFAULT_TRADES,
    compare_against_named_baseline,
    load_paper_default_trades,
    summarize_stress,
)


ROLE_LABEL = "EXP_2026_05_22_UNIFIED_ENTRY_LIFECYCLE_SEQUENCE_V1"
HISTORICAL_ID = "Protocol209"
CANDIDATE_LABEL = "CHALLENGER_UNIFIED_ENTRY_LIFECYCLE_SEQUENCE_V1"
PAPER_DEFAULT_LABEL = "PAPER_DEFAULT_PROTOCOL101"
OTHER_BASELINE_LABEL = "CHALLENGER_FULL_ACTION_SURFACE_EDGE_V1_WITH_FROZEN_PROTOCOL081_EXITS"
ENTRY_STREAM_DESCRIPTION = "frozen full-action surface-edge entry stream from the Protocol194 challenger lineage"
LOOP_ID = "v4_aplus_hypothesis_209_unified_entry_lifecycle_sequence"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
ENTRY_THRESHOLD_CANDIDATES = (-1_000, -500, -250, 0, 250, 500, 750, 1_000, 1_500, 2_000)
HOLD_THRESHOLD_CANDIDATES = (-150, -50, 0, 100, 200, 350, 500, 750, 1_000)


class UnifiedEntryLifecycleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.04),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.04),
        )
        self.entry_head = nn.Linear(hidden_dim, 1)
        self.hold_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        return self.entry_head(z).squeeze(-1), self.hold_head(z).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", action="append", type=Path, default=None)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--paper-default-trades", type=Path, default=PAPER_DEFAULT_TRADES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-train-steps", type=int, default=650_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entry-loss-weight", type=float, default=0.7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay_dirs = args.replay_dir or DEFAULT_REPLAY_DIRS
    candidates = load_candidate_entries(replay_dirs)
    records, path_skips = build_path_records(candidates, normalized_dir=args.normalized_dir, forced_flat_time=args.forced_flat_time)
    if not records:
        raise SystemExit("no path records available for unified sequence training")
    apply_unified_slot_targets(records)

    lifecycle_baseline_rows = []
    model_rows = []
    threshold_rows = []
    fold_payloads = []
    for spec in fold_specs():
        train_records = [r for r in records if r.reported_split in spec["train_splits"]]
        validation_records = [r for r in records if r.reported_split == spec["validation_split"]]
        if not train_records or not validation_records:
            continue
        train_payload = fit_training_payload(train_records, max_train_steps=args.max_train_steps, seed=209)
        for model_seed in args.seeds:
            model, history = train_unified_model(
                train_payload,
                seed=int(model_seed),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                hidden_dim=int(args.hidden_dim),
                learning_rate=float(args.learning_rate),
                entry_loss_weight=float(args.entry_loss_weight),
            )
            validation_predictions = predict_records(model, train_payload["scaler"], validation_records)
            entry_threshold, hold_threshold, sweep = select_threshold_pair(
                validation_records,
                validation_predictions,
                split_name=str(spec["validation_split"]),
                model_seed=int(model_seed),
            )
            threshold_rows.extend({**row, "fold": spec["fold"], "model_seed": int(model_seed)} for row in sweep)
            for split in spec["test_splits"]:
                split_records = [r for r in records if r.reported_split == split]
                predictions = predict_records(model, train_payload["scaler"], split_records)
                model_rows.extend(
                    simulate_unified_serial(
                        split_records,
                        predictions,
                        entry_threshold=entry_threshold,
                        hold_threshold=hold_threshold,
                        model_seed=int(model_seed),
                        strategy=f"{CANDIDATE_LABEL}:{spec['fold']}:seed{model_seed}",
                    )
                )
                if split == "q1_2026":
                    march_records = [r for r in split_records if r.session >= "2026-03-01"]
                    march_predictions = {r.uid: predictions[r.uid] for r in march_records if r.uid in predictions}
                    model_rows.extend(
                        {**row, "reported_split": "march_2026"}
                        for row in simulate_unified_serial(
                            march_records,
                            march_predictions,
                            entry_threshold=entry_threshold,
                            hold_threshold=hold_threshold,
                            model_seed=int(model_seed),
                            strategy=f"{CANDIDATE_LABEL}:{spec['fold']}:seed{model_seed}:march_subset",
                        )
                    )
            fold_payloads.append(
                {
                    "fold": spec["fold"],
                    "train_splits": list(spec["train_splits"]),
                    "validation_split": spec["validation_split"],
                    "test_splits": list(spec["test_splits"]),
                    "model_seed": int(model_seed),
                    "entry_threshold": float(entry_threshold),
                    "hold_threshold": float(hold_threshold),
                    "history": history,
                    "train_records": len(train_records),
                    "validation_records": len(validation_records),
                    "path_train_steps_used": int(len(train_payload["hold_y"])),
                    "entry_train_records_used": int(len(train_payload["entry_y"])),
                }
            )
        for split in [spec["validation_split"], *spec["test_splits"]]:
            split_records = [r for r in records if r.reported_split == split]
            lifecycle_baseline_rows.extend(simulate_baseline_serial(split_records, strategy=OTHER_BASELINE_LABEL))
            if split == "q1_2026":
                march_records = [r for r in split_records if r.session >= "2026-03-01"]
                lifecycle_baseline_rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in simulate_baseline_serial(march_records, strategy=f"{OTHER_BASELINE_LABEL}:march_subset")
                )

    model_frame = pd.DataFrame(model_rows)
    lifecycle_baseline_frame = pd.DataFrame(lifecycle_baseline_rows).drop_duplicates(
        ["reported_split", "entry_seed", "session", "decision_time", "contract_id", "strategy"],
        keep="last",
    )
    threshold_frame = pd.DataFrame(threshold_rows)
    paper_default_frame = load_paper_default_trades(args.paper_default_trades)
    model_summary = summarize_replay(model_frame, seed_col="combo_seed")
    paper_default_summary = summarize_replay(paper_default_frame, seed_col="seed")
    lifecycle_baseline_summary = summarize_replay(lifecycle_baseline_frame, seed_col="entry_seed")
    paper_default_comparison = compare_against_named_baseline(model_summary, paper_default_summary, "paper_default")
    lifecycle_baseline_comparison = compare_against_named_baseline(model_summary, lifecycle_baseline_summary, "lifecycle_baseline")
    invariants = serial_invariants(model_frame)
    stress_010 = summarize_stress(model_frame, seed_col="combo_seed", extra_per_side=0.10)
    stress_025 = summarize_stress(model_frame, seed_col="combo_seed", extra_per_side=0.25)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "candidate_label": CANDIDATE_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "other_baseline_label": OTHER_BASELINE_LABEL,
        "what_is_this": "experiment / research challenger training run",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "source_replay_dirs": [str(path) for path in replay_dirs],
        "normalized_dir": str(args.normalized_dir),
        "paper_default_trades": str(args.paper_default_trades),
        "row_counts": {
            "candidate_entries": int(len(candidates)),
            "path_records": int(len(records)),
            "path_skips": int(len(path_skips)),
            "model_trade_rows": int(len(model_frame)),
            "lifecycle_baseline_rows": int(len(lifecycle_baseline_frame)),
            "paper_default_rows": int(len(paper_default_frame)),
        },
        "pre_registration": {
            "hypothesis": (
                "A shared entry-plus-lifecycle sequence objective should reduce churn and premature exits by "
                "training the flat entry decision and hold/exit decision against the same single-slot dynamic program."
            ),
            "candidate": CANDIDATE_LABEL,
            "paper_default_baseline": PAPER_DEFAULT_LABEL,
            "other_baseline": OTHER_BASELINE_LABEL,
            "entry_stream": ENTRY_STREAM_DESCRIPTION,
            "flat_action_space": "wait or enter candidate from the frozen candidate stream",
            "holding_action_space": "hold or exit",
            "validation_discipline": "entry and hold thresholds selected only on chronological validation splits",
            "no_hardcoded_exit_rules": True,
            "starting_cash": STARTING_CASH,
            "max_contracts": 1,
            "max_concurrent_positions": 1,
        },
        "folds": fold_payloads,
        "model_summary": model_summary,
        "paper_default_summary": paper_default_summary,
        "lifecycle_baseline_summary": lifecycle_baseline_summary,
        "paper_default_comparison": paper_default_comparison,
        "lifecycle_baseline_comparison": lifecycle_baseline_comparison,
        "stress_0_10_per_side_summary": stress_010,
        "stress_0_25_per_side_summary": stress_025,
        "threshold_summary": threshold_summary_pairs(threshold_frame),
        "invariants": invariants,
        "path_skip_counts": count_by(path_skips, "skip_reason"),
        "decision": decide(paper_default_comparison, lifecycle_baseline_comparison, invariants, stress_010),
        "next_experiment": next_experiment(paper_default_comparison, lifecycle_baseline_comparison),
    }
    model_frame.to_csv(args.out_dir / "challenger_unified_entry_lifecycle_trades.csv", index=False)
    lifecycle_baseline_frame.to_csv(args.out_dir / "lifecycle_baseline_trades.csv", index=False)
    paper_default_frame.to_csv(args.out_dir / "paper_default_protocol101_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(path_skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def apply_unified_slot_targets(records: Sequence[Any]) -> None:
    groups: dict[tuple[str, int, str], list[Any]] = {}
    for record in records:
        groups.setdefault((record.reported_split, int(record.entry_seed), record.session), []).append(record)
    for group in groups.values():
        ordered = sorted(group, key=lambda r: (r.decision_ts, r.contract_id))
        decision_ns = np.array([pd.Timestamp(record.decision_ts).value for record in ordered], dtype=np.int64)
        flat_value = np.zeros(len(ordered) + 1, dtype=np.float32)
        hold_target_by_uid: dict[str, np.ndarray] = {}
        entry_target_by_uid: dict[str, float] = {}
        for idx in range(len(ordered) - 1, -1, -1):
            record = ordered[idx]
            path_ns = np.array([pd.Timestamp(value).value for value in record.quote_times], dtype=np.int64)
            next_indices = np.searchsorted(decision_ns, path_ns, side="left")
            next_values = flat_value[np.clip(next_indices, 0, len(ordered))]
            step_total_value = record.path_pnl.astype(np.float32) + next_values
            best_from_step = np.maximum.accumulate(step_total_value[::-1])[::-1]
            hold_target_by_uid[record.uid] = np.clip(best_from_step - step_total_value, -TARGET_CLIP, TARGET_CLIP).astype(np.float32)
            enter_value = float(best_from_step[0]) if len(best_from_step) else -math.inf
            wait_value = float(flat_value[idx + 1])
            entry_target_by_uid[record.uid] = float(np.clip(enter_value - wait_value, -TARGET_CLIP, TARGET_CLIP))
            flat_value[idx] = max(wait_value, enter_value)
        for record in ordered:
            record.target = hold_target_by_uid[record.uid]
            record.entry_target = entry_target_by_uid[record.uid]


def fit_training_payload(records: Sequence[Any], *, max_train_steps: int, seed: int) -> dict[str, Any]:
    hold_x = np.vstack([record.features for record in records]).astype(np.float32)
    hold_y = np.concatenate([record.target for record in records]).astype(np.float32) / TARGET_SCALE
    if max_train_steps > 0 and len(hold_y) > max_train_steps:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(hold_y), size=max_train_steps, replace=False))
        hold_x = hold_x[idx]
        hold_y = hold_y[idx]
    entry_x = np.vstack([record.features[0] for record in records]).astype(np.float32)
    entry_y = np.array([float(getattr(record, "entry_target", 0.0)) for record in records], dtype=np.float32) / TARGET_SCALE
    scaler = FeatureScaler.fit(np.vstack([hold_x, entry_x]))
    return {
        "hold_x": scaler.transform(hold_x),
        "hold_y": hold_y,
        "entry_x": scaler.transform(entry_x),
        "entry_y": entry_y,
        "scaler": scaler,
    }


def train_unified_model(
    payload: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    entry_loss_weight: float,
) -> tuple[UnifiedEntryLifecycleMLP, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    hold_x = payload["hold_x"]
    hold_y = payload["hold_y"]
    entry_x = payload["entry_x"]
    entry_y = payload["entry_y"]
    model = UnifiedEntryLifecycleMLP(input_dim=hold_x.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    hold_loader = DataLoader(
        TensorDataset(torch.from_numpy(hold_x), torch.from_numpy(hold_y)),
        batch_size=min(batch_size, len(hold_y)),
        shuffle=True,
    )
    entry_loader = DataLoader(
        TensorDataset(torch.from_numpy(entry_x), torch.from_numpy(entry_y)),
        batch_size=min(max(256, batch_size // 8), len(entry_y)),
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        entry_iter = iter(entry_loader)
        for hold_batch_x, hold_batch_y in hold_loader:
            try:
                entry_batch_x, entry_batch_y = next(entry_iter)
            except StopIteration:
                entry_iter = iter(entry_loader)
                entry_batch_x, entry_batch_y = next(entry_iter)
            optimizer.zero_grad(set_to_none=True)
            _, hold_pred = model(hold_batch_x)
            entry_pred, _ = model(entry_batch_x)
            hold_loss = F.huber_loss(hold_pred, hold_batch_y, delta=1.0)
            entry_loss = F.huber_loss(entry_pred, entry_batch_y, delta=1.0)
            loss = hold_loss + float(entry_loss_weight) * entry_loss
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


def predict_records(model: UnifiedEntryLifecycleMLP, scaler: FeatureScaler, records: Sequence[Any]) -> dict[str, dict[str, Any]]:
    model.eval()
    out: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for record in records:
            x = scaler.transform(record.features)
            entry_pred, hold_pred = model(torch.from_numpy(x))
            out[record.uid] = {
                "entry": float(entry_pred[0].cpu().numpy() * TARGET_SCALE),
                "hold": hold_pred.cpu().numpy().astype(np.float32) * TARGET_SCALE,
            }
    return out


def select_threshold_pair(
    records: Sequence[Any],
    predictions: dict[str, dict[str, Any]],
    *,
    split_name: str,
    model_seed: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    rows = []
    best_entry = float(ENTRY_THRESHOLD_CANDIDATES[0])
    best_hold = float(HOLD_THRESHOLD_CANDIDATES[0])
    best_key = (-1e18, -1e18, 0.0)
    for entry_threshold in ENTRY_THRESHOLD_CANDIDATES:
        for hold_threshold in HOLD_THRESHOLD_CANDIDATES:
            trades = simulate_unified_serial(
                records,
                predictions,
                entry_threshold=float(entry_threshold),
                hold_threshold=float(hold_threshold),
                model_seed=model_seed,
                strategy="threshold_selection",
            )
            metrics = metrics_for_rows(pd.DataFrame(trades))
            key = (float(metrics["total_pnl"]), float(metrics["profit_factor_for_selection"]), -float(metrics["trades"]))
            rows.append(
                {
                    "validation_split": split_name,
                    "entry_threshold": float(entry_threshold),
                    "hold_threshold": float(hold_threshold),
                    **metrics,
                }
            )
            if key > best_key:
                best_key = key
                best_entry = float(entry_threshold)
                best_hold = float(hold_threshold)
    return best_entry, best_hold, rows


def simulate_unified_serial(
    records: Sequence[Any],
    predictions: dict[str, dict[str, Any]],
    *,
    entry_threshold: float,
    hold_threshold: float,
    model_seed: int,
    strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (entry_seed, session), group in group_records(records).items():
        equity = STARTING_CASH
        open_until = pd.Timestamp.min.tz_localize("UTC")
        for decision_ts, candidates_at_time in group_by_decision_time(group):
            if decision_ts < open_until:
                continue
            eligible = []
            for record in candidates_at_time:
                pred = predictions.get(record.uid)
                if pred is None:
                    continue
                entry_score = float(pred["entry"])
                if entry_score < float(entry_threshold):
                    continue
                if record.entry_premium <= 0.0 or record.entry_premium > equity:
                    continue
                eligible.append((entry_score, record))
            if not eligible:
                continue
            _, record = max(eligible, key=lambda pair: (pair[0], float(pair[1].score), -abs(float(pair[1].offset))))
            pred = predictions.get(record.uid)
            if pred is None:
                continue
            hold_prediction = np.asarray(pred["hold"], dtype=np.float32)
            if len(hold_prediction) != len(record.path_pnl):
                continue
            exit_idx = exit_index_from_prediction(hold_prediction, float(hold_threshold))
            pnl = float(record.path_pnl[exit_idx])
            exit_ts = pd.Timestamp(record.quote_times[exit_idx])
            combo_seed = int(model_seed * 100 + int(entry_seed))
            rows.append(
                {
                    "reported_split": record.reported_split,
                    "fold": record.fold,
                    "model_seed": int(model_seed),
                    "entry_seed": int(entry_seed),
                    "combo_seed": combo_seed,
                    "session": record.session,
                    "decision_time": record.decision_ts.isoformat(),
                    "exit_time": exit_ts.isoformat(),
                    "contract_id": record.contract_id,
                    "right": record.right,
                    "offset": float(record.offset),
                    "score": float(record.score),
                    "entry_threshold": float(entry_threshold),
                    "hold_threshold": float(hold_threshold),
                    "entry_score": float(pred["entry"]),
                    "entry_target": float(getattr(record, "entry_target", 0.0)),
                    "entry_ask": float(record.entry_ask),
                    "entry_premium": float(record.entry_premium),
                    "pnl": pnl,
                    "account_equity_before": float(equity),
                    "account_equity_after": float(equity + pnl),
                    "exit_step": int(exit_idx),
                    "path_points": int(len(record.path_pnl)),
                    "exit_reason": "model_exit" if exit_idx < len(record.path_pnl) - 1 else "mandatory_forced_flat",
                    "predicted_continuation_value": float(hold_prediction[exit_idx]),
                    "baseline_exit_time": record.baseline_exit_ts.isoformat(),
                    "baseline_pnl": float(record.baseline_pnl),
                    "baseline_exit_reason": record.baseline_exit_reason,
                    "strategy": strategy,
                }
            )
            equity = float(equity + pnl)
            open_until = exit_ts
    return rows


def group_records(records: Sequence[Any]) -> dict[tuple[int, str], list[Any]]:
    groups: dict[tuple[int, str], list[Any]] = {}
    for record in records:
        groups.setdefault((int(record.entry_seed), str(record.session)), []).append(record)
    return {key: sorted(value, key=lambda r: (r.decision_ts, r.contract_id)) for key, value in groups.items()}


def group_by_decision_time(records: Sequence[Any]) -> list[tuple[pd.Timestamp, list[Any]]]:
    groups: dict[pd.Timestamp, list[Any]] = {}
    for record in records:
        groups.setdefault(pd.Timestamp(record.decision_ts), []).append(record)
    return [(key, groups[key]) for key in sorted(groups)]


def threshold_summary_pairs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for (fold, model_seed), group in frame.groupby(["fold", "model_seed"], sort=True):
        best = group.sort_values(["total_pnl", "profit_factor_for_selection"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "fold": str(fold),
                "model_seed": int(model_seed),
                "entry_threshold": float(best["entry_threshold"]),
                "hold_threshold": float(best["hold_threshold"]),
                "validation_total_pnl": float(best["total_pnl"]),
                "validation_profit_factor": float(best["profit_factor"]),
                "validation_trades": int(best["trades"]),
            }
        )
    return rows


def decide(
    paper_default_comparison: list[dict[str, Any]],
    lifecycle_baseline_comparison: list[dict[str, Any]],
    invariants: dict[str, Any],
    stress_010: list[dict[str, Any]],
) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_unified_sequence_invariant_failure"
    required = ["q1_2026", "march_2026", "recent_2026"]
    paper = {row["reported_split"]: row for row in paper_default_comparison}
    lifecycle = {row["reported_split"]: row for row in lifecycle_baseline_comparison}
    stress = {row["reported_split"]: row for row in stress_010}
    beats_paper = all(paper.get(split, {}).get("beats_baseline") for split in required)
    beats_lifecycle = all(lifecycle.get(split, {}).get("beats_baseline") for split in required)
    stress_positive = all(finite_float(stress.get(split, {}).get("median_total_pnl"), 0.0) > 0.0 for split in required)
    if beats_paper and beats_lifecycle and stress_positive:
        return "keep_research_challenger_unified_sequence_common_splits_only"
    if beats_paper:
        return "mixed_unified_sequence_beats_paper_default_but_not_lifecycle_baseline"
    return "reject_unified_sequence_does_not_beat_paper_default"


def next_experiment(
    paper_default_comparison: list[dict[str, Any]],
    lifecycle_baseline_comparison: list[dict[str, Any]],
) -> str:
    paper = {row["reported_split"]: row for row in paper_default_comparison}
    lifecycle = {row["reported_split"]: row for row in lifecycle_baseline_comparison}
    if all(row.get("beats_baseline") for row in paper.values()) and all(row.get("beats_baseline") for row in lifecycle.values()):
        return (
            "Run attribution and then build no-order runtime parity. This candidate still needs Q3/Q4-capable "
            "training coverage before any paper-default decision."
        )
    if all(row.get("beats_baseline") for row in paper.values()):
        return "Attribute the lifecycle-baseline miss before changing architecture again."
    return (
        "Reject this unified formulation and return to full-action candidate-generation feature parity, because "
        "the joint objective did not clear the paper-default gate."
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        f"Does it change the paper-trading default: {'yes' if payload['changes_paper_default'] else 'no'}",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Other baseline: {payload['other_baseline_label']}",
        f"Data used: {payload['pre_registration']['entry_stream']}, existing normalized official-context SPXW rows, and existing Protocol101 paper-default trades.",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Paper Default Comparison",
        "",
        "| split | challenger | paper default | delta | challenger PF | paper PF | challenger trades | paper trades | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["paper_default_comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | "
            f"{money(row['baseline_median_total_pnl'])} | {money(row['delta_vs_baseline'])} | "
            f"{row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} | "
            f"{row['model_median_trades']:.0f} | {row['baseline_median_trades']:.0f} | "
            f"{pct(row['model_positive_seed_fraction'])} |"
        )
    lines.extend(["", "## Lifecycle Baseline Comparison", "", "| split | challenger | lifecycle baseline | delta | beats baseline |", "|---|---:|---:|---:|---|"])
    for row in payload["lifecycle_baseline_comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | "
            f"{money(row['baseline_median_total_pnl'])} | {money(row['delta_vs_baseline'])} | "
            f"{row['beats_baseline']} |"
        )
    lines.extend(["", "## Thresholds", ""])
    if payload["threshold_summary"]:
        lines.extend(
            [
                "| fold | seed | entry threshold | hold threshold | validation PnL | PF | trades |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["threshold_summary"]:
            lines.append(
                f"| {row['fold']} | {row['model_seed']} | {row['entry_threshold']:.0f} | {row['hold_threshold']:.0f} | "
                f"{money(row['validation_total_pnl'])} | {row['validation_profit_factor']:.3f} | {row['validation_trades']} |"
            )
    lines.extend(
        [
            "",
            "## Stress",
            "",
            "| split | stress $0.10/side median PnL | stress $0.25/side median PnL |",
            "|---|---:|---:|",
        ]
    )
    stress_010 = {row["reported_split"]: row for row in payload["stress_0_10_per_side_summary"]}
    stress_025 = {row["reported_split"]: row for row in payload["stress_0_25_per_side_summary"]}
    for split in sorted(stress_010):
        lines.append(
            f"| {split} | {money(stress_010[split].get('median_total_pnl'))} | "
            f"{money(stress_025.get(split, {}).get('median_total_pnl'))} |"
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
            f"- Challenger trades: `{path.parent / 'challenger_unified_entry_lifecycle_trades.csv'}`",
            f"- Lifecycle baseline trades: `{path.parent / 'lifecycle_baseline_trades.csv'}`",
            f"- Paper default trades: `{path.parent / 'paper_default_protocol101_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
