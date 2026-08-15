"""EXP_2026_05_22_ORACLE_EXIT_ACTION_CLASSIFIER_V1.

Historically Protocol230. Protocol229 showed that the position DP oracle mostly
uses hold -> exit, with reduce actions nearly absent. This experiment trains a
causal neural classifier to recognize the sparse oracle exit moment.

Entries and account-aware sizing stay frozen. The model may exit when its
predicted exit probability clears a validation-selected threshold; otherwise it
falls back to the baseline exit.

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
from v4.scripts.run_protocol225_account_aware_lifecycle_exit_policy import (
    DEFAULT_DATASET,
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_TRADES,
    REQUIRED_SPLITS,
    STARTING_CASH,
    build_path_records,
    compare_summaries,
    compute_quantity,
    count_by,
    fold_specs,
    load_entries,
    metrics_for_rows,
    row_for_trade,
    serial_invariants,
    simulate_baseline_serial,
    summarize_replay,
    threshold_summary,
)


ROLE_LABEL = "EXP_2026_05_22_ORACLE_EXIT_ACTION_CLASSIFIER_V1"
HISTORICAL_ID = "Protocol230"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_230_oracle_exit_action_classifier")
THRESHOLD_CANDIDATES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)


class ExitClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:30")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-train-steps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-risk-frac", type=float, default=0.01)
    parser.add_argument("--max-risk-frac", type=float, default=0.10)
    parser.add_argument("--hard-premium-cap-frac", type=float, default=0.12)
    parser.add_argument("--confidence-scale", type=float, default=0.75)
    parser.add_argument("--liquidity-fraction", type=float, default=0.25)
    parser.add_argument("--absolute-max-contracts", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(args.trades, dataset_path=args.dataset)
    records, skips = build_path_records(entries, normalized_dir=args.normalized_dir, forced_flat_time=str(args.forced_flat_time))
    apply_oracle_exit_targets(records)
    sizing = {
        "min_risk_frac": float(args.min_risk_frac),
        "max_risk_frac": float(args.max_risk_frac),
        "hard_premium_cap_frac": float(args.hard_premium_cap_frac),
        "confidence_scale": float(args.confidence_scale),
        "liquidity_fraction": float(args.liquidity_fraction),
        "absolute_max_contracts": int(args.absolute_max_contracts),
    }
    baseline_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    fold_payloads: list[dict[str, Any]] = []
    for spec in fold_specs():
        train_records = [r for r in records if r.reported_split in spec["train_splits"]]
        validation_records = [r for r in records if r.reported_split == spec["validation_split"]]
        if not train_records or not validation_records:
            continue
        train_x, train_y, scaler = fit_classifier_matrix(train_records, max_train_steps=int(args.max_train_steps), seed=230)
        for seed in args.seeds:
            model, history = train_classifier(
                train_x,
                train_y,
                seed=int(seed),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                hidden_dim=int(args.hidden_dim),
                learning_rate=float(args.learning_rate),
            )
            validation_probs = predict_records(model, scaler, validation_records)
            threshold, sweep = select_threshold(
                validation_records,
                validation_probs,
                sizing=sizing,
                split_name=str(spec["validation_split"]),
                model_seed=int(seed),
            )
            threshold_rows.extend({**row, "fold": spec["fold"], "model_seed": int(seed)} for row in sweep)
            for split in spec["test_splits"]:
                split_records = [r for r in records if r.reported_split == split]
                probs = predict_records(model, scaler, split_records)
                model_rows.extend(
                    simulate_classifier_serial(
                        split_records,
                        probs,
                        sizing=sizing,
                        threshold=threshold,
                        model_seed=int(seed),
                        strategy=f"protocol230:{spec['fold']}:seed{seed}",
                    )
                )
                if split == "q1_2026":
                    march_records = [r for r in split_records if r.session >= "2026-03-01"]
                    march_probs = {r.uid: probs[r.uid] for r in march_records if r.uid in probs}
                    model_rows.extend(
                        {**row, "reported_split": "march_2026"}
                        for row in simulate_classifier_serial(
                            march_records,
                            march_probs,
                            sizing=sizing,
                            threshold=threshold,
                            model_seed=int(seed),
                            strategy=f"protocol230:{spec['fold']}:seed{seed}:march_subset",
                        )
                    )
            fold_payloads.append(
                {
                    "fold": spec["fold"],
                    "train_splits": list(spec["train_splits"]),
                    "validation_split": spec["validation_split"],
                    "test_splits": list(spec["test_splits"]),
                    "model_seed": int(seed),
                    "threshold": float(threshold),
                    "history": history,
                    "train_records": int(len(train_records)),
                    "validation_records": int(len(validation_records)),
                    "train_steps_used": int(len(train_y)),
                }
            )
        for split in [spec["validation_split"], *spec["test_splits"]]:
            split_records = [r for r in records if r.reported_split == split]
            baseline_rows.extend(simulate_baseline_serial(split_records, sizing=sizing, strategy="protocol223_account_aware_baseline"))
            if split == "q1_2026":
                march_records = [r for r in split_records if r.session >= "2026-03-01"]
                baseline_rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in simulate_baseline_serial(
                        march_records,
                        sizing=sizing,
                        strategy="protocol223_account_aware_baseline:march_subset",
                    )
                )
    baseline_frame = pd.DataFrame(baseline_rows).drop_duplicates(
        ["reported_split", "session", "decision_time", "contract_id", "strategy"],
        keep="last",
    )
    model_frame = pd.DataFrame(model_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    baseline_summary = summarize_replay(baseline_frame, seed_col=None)
    model_summary = summarize_replay(model_frame, seed_col="model_seed")
    comparison = compare_summaries(model_summary, baseline_summary)
    invariants = serial_invariants(model_frame)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / causal oracle-exit action classifier",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_ORACLE_EXIT_ACTION_CLASSIFIER_V1",
        "entry_source": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1",
        "sizing_source": "EXP_2026_05_22_ACCOUNT_AWARE_CONFIDENCE_SIZING_V2",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.trades),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "pre_registration": {
            "hypothesis": "The position DP oracle is mostly hold->exit; train a class-weighted causal exit classifier instead of a value regressor.",
            "target": "one sparse positive exit label at each trade path's hindsight best executable bid",
            "fallback": "baseline exit if classifier never fires before forced-flat",
            "threshold_selection": "validation split only",
        },
        "row_counts": {
            "entries": int(len(entries)),
            "path_records": int(len(records)),
            "path_skips": int(len(skips)),
            "baseline_trade_rows": int(len(baseline_frame)),
            "model_trade_rows": int(len(model_frame)),
        },
        "folds": fold_payloads,
        "baseline_summary": baseline_summary,
        "model_summary": model_summary,
        "comparison": comparison,
        "threshold_summary": threshold_summary(threshold_frame),
        "invariants": invariants,
        "path_skip_counts": count_by(skips, "skip_reason"),
        "decision": decide(comparison, invariants),
        "next_experiment": next_experiment(comparison, invariants),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "model_trades": str(args.out_dir / "protocol230_model_serial_trades.csv"),
            "baseline_trades": str(args.out_dir / "protocol223_baseline_serial_trades.csv"),
            "threshold_sweep": str(args.out_dir / "threshold_sweep.csv"),
            "path_skips": str(args.out_dir / "path_skips.csv"),
        },
    }
    baseline_frame.to_csv(args.out_dir / "protocol223_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "protocol230_model_serial_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def apply_oracle_exit_targets(records: Sequence[Any]) -> None:
    for record in records:
        target = np.zeros(len(record.unit_pnl_path), dtype=np.float32)
        if len(target):
            target[int(np.nanargmax(record.unit_pnl_path))] = 1.0
        record.target = target


def fit_classifier_matrix(records: Sequence[Any], *, max_train_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray, FeatureScaler]:
    x = np.vstack([record.features for record in records]).astype(np.float32)
    y = np.concatenate([record.target for record in records]).astype(np.float32)
    if max_train_steps > 0 and len(y) > max_train_steps:
        rng = np.random.default_rng(seed)
        positives = np.where(y > 0.5)[0]
        negative_pool = np.where(y <= 0.5)[0]
        keep_pos = positives
        remaining = max(max_train_steps - len(keep_pos), 0)
        keep_neg = rng.choice(negative_pool, size=min(remaining, len(negative_pool)), replace=False) if remaining else np.array([], dtype=int)
        idx = np.sort(np.concatenate([keep_pos, keep_neg]))
        x = x[idx]
        y = y[idx]
    scaler = FeatureScaler.fit(x)
    return scaler.transform(x), y, scaler


def train_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
) -> tuple[ExitClassifier, list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ExitClassifier(input_dim=train_x.shape[1], hidden_dim=hidden_dim)
    positives = max(float((train_y > 0.5).sum()), 1.0)
    negatives = max(float((train_y <= 0.5).sum()), 1.0)
    pos_weight = torch.tensor(min(negatives / positives, 100.0), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=min(batch_size, len(train_y)),
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
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


def predict_records(model: ExitClassifier, scaler: FeatureScaler, records: Sequence[Any]) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for record in records:
            x = scaler.transform(record.features)
            logits = model(torch.from_numpy(x)).cpu().numpy().astype(np.float32)
            out[record.uid] = 1.0 / (1.0 + np.exp(-logits))
    return out


def select_threshold(
    records: Sequence[Any],
    probabilities: dict[str, np.ndarray],
    *,
    sizing: dict[str, Any],
    split_name: str,
    model_seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    best_threshold = float(THRESHOLD_CANDIDATES[0])
    best_key = (-1e18, -1e18, 0.0)
    for threshold in THRESHOLD_CANDIDATES:
        trades = simulate_classifier_serial(
            records,
            probabilities,
            sizing=sizing,
            threshold=float(threshold),
            model_seed=model_seed,
            strategy="threshold_selection",
        )
        metrics = metrics_for_rows(pd.DataFrame(trades))
        key = (float(metrics["total_pnl"]), float(metrics["profit_factor_for_selection"]), -float(metrics["trades"]))
        rows.append({"validation_split": split_name, "threshold": float(threshold), **metrics})
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, rows


def simulate_classifier_serial(
    records: Sequence[Any],
    probabilities: dict[str, np.ndarray],
    *,
    sizing: dict[str, Any],
    threshold: float,
    model_seed: int,
    strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equity = STARTING_CASH
    peak = equity
    open_until_by_session: dict[str, pd.Timestamp] = {}
    for record in sorted(records, key=lambda r: (r.session, r.decision_ts, r.contract_id)):
        if record.decision_ts < open_until_by_session.get(record.session, pd.Timestamp.min.tz_localize("UTC")):
            continue
        quantity, sizing_info = compute_quantity(record, equity=equity, peak=peak, sizing=sizing)
        if quantity <= 0:
            continue
        probs = probabilities.get(record.uid)
        if probs is None or len(probs) != len(record.unit_pnl_path):
            continue
        exit_idx, fired = exit_index_from_probability(record, probs, threshold)
        unit_pnl = float(record.unit_pnl_path[exit_idx]) if fired else float(record.baseline_unit_pnl)
        exit_ts = pd.Timestamp(record.quote_times[exit_idx]) if fired else pd.Timestamp(record.baseline_exit_ts)
        pnl = unit_pnl * quantity
        before = equity
        equity += pnl
        peak = max(peak, equity)
        row = row_for_trade(
            record,
            quantity,
            before,
            equity,
            pnl,
            unit_pnl,
            exit_idx,
            exit_ts,
            strategy,
            model_seed,
            sizing_info,
            probs[exit_idx],
            threshold,
        )
        row["classifier_exit_fired"] = bool(fired)
        row["exit_reason"] = "oracle_exit_classifier" if fired else "baseline_fallback_exit"
        rows.append(row)
        open_until_by_session[record.session] = exit_ts
    return rows


def exit_index_from_probability(record: Any, probs: np.ndarray, threshold: float) -> tuple[int, bool]:
    anchor = baseline_anchor_idx(record)
    search = np.asarray(probs, dtype=float)[: max(anchor + 1, 1)]
    eligible = np.where(search >= float(threshold))[0]
    if len(eligible):
        return int(eligible[0]), True
    return anchor, False


def baseline_anchor_idx(record: Any) -> int:
    times = pd.to_datetime(record.quote_times, utc=True, format="ISO8601")
    anchor = int(np.searchsorted(np.array([ts.value for ts in times], dtype=np.int64), pd.Timestamp(record.baseline_exit_ts).value, side="left"))
    return min(max(anchor, 0), len(record.unit_pnl_path) - 1)


def decide(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_oracle_exit_classifier_invariant_failure"
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "keep_oracle_exit_classifier_research_candidate"
    if any(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "mixed_oracle_exit_classifier_requires_attribution"
    return "reject_oracle_exit_classifier_no_improvement"


def next_experiment(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "Fix simulator invariants before more lifecycle work."
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "Stress and attribute classifier exits, then test a reduce/exit classifier only if reduce labels become meaningful."
    return "Exit classifier did not generalize; return to entry/sizing objective or broaden data before more lifecycle complexity."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Entry source: {payload['entry_source']}",
        f"Sizing source: {payload['sizing_source']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Comparison To Account-Aware Baseline",
        "",
        "| split | model PnL | baseline PnL | delta | model PF | baseline PF | model trades | baseline trades | model DD | baseline DD | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | {money(row['baseline_median_total_pnl'])} | "
            f"{money(row['delta_vs_baseline'])} | {row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} | "
            f"{row['model_median_trades']:.0f} | {row['baseline_median_trades']:.0f} | "
            f"{money(row['model_median_max_drawdown'])} | {money(row['baseline_median_max_drawdown'])} | "
            f"{pct(row['model_positive_seed_fraction'])} |"
        )
    lines.extend(["", "## Thresholds", ""])
    if payload["threshold_summary"]:
        lines.extend(["| fold | seed | probability threshold | validation PnL | PF | trades |", "|---|---:|---:|---:|---:|---:|"])
        for row in payload["threshold_summary"]:
            lines.append(
                f"| {row['fold']} | {row['model_seed']} | {row['selected_threshold']:.2f} | "
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
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Model trades: `{payload['outputs']['model_trades']}`",
            f"- Baseline trades: `{payload['outputs']['baseline_trades']}`",
            f"- Threshold sweep: `{payload['outputs']['threshold_sweep']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    sign = "-" if number < 0.0 else ""
    return f"{sign}${abs(number):,.0f}"


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


if __name__ == "__main__":
    raise SystemExit(main())
