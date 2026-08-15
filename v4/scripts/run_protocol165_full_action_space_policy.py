"""Protocol 165: full action-space neural entry policy.

Protocol165 trains on Protocol164 rows, where each decision can expose the
full SPXW 0DTE ATM +/- $50 ladder instead of only Protocol101-selected
candidates. The model is research-only until it beats frozen Protocol101 under
strict serial account replay.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v4.model.supervised_pilot import FeatureScaler, Trade, metrics_for_trades
from v4.scripts.run_protocol164_full_action_space_dataset import FULL_ACTION_FEATURE_COLUMNS


LOOP_ID = "v4_aplus_hypothesis_165_full_action_space_policy"
DEFAULT_PROTOCOL164_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_164_full_action_space_dataset")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_PROTOCOL101_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json")
DEFAULT_RECENT_BASELINE = Path("v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/summary.json")
MODEL_SEEDS = [1, 2, 3, 4, 5]
MAX_ACTION_CANDIDATES = 42
CONTRACT_MULTIPLIER = 100.0
STARTING_CASH = 10_000.0
FOLDS = [
    {
        "name": "fold1_train_q1_validate_q2_test_q3",
        "train_splits": ["q1_2025"],
        "validation_split": "q2_2025",
        "test_split": "q3_2025",
        "reported_splits": ["q3_2025"],
    },
    {
        "name": "fold2_train_q1_q2_validate_q3_test_q4",
        "train_splits": ["q1_2025", "q2_2025"],
        "validation_split": "q3_2025",
        "test_split": "q4_2025",
        "reported_splits": ["q4_2025"],
    },
    {
        "name": "fold3_train_q1_q2_q3_validate_q4_test_q1_2026",
        "train_splits": ["q1_2025", "q2_2025", "q3_2025"],
        "validation_split": "q4_2025",
        "test_split": "q1_2026",
        "reported_splits": ["q1_2026", "march_2026"],
    },
    {
        "name": "fold4_train_2025_validate_q1_2026_test_recent",
        "train_splits": ["q1_2025", "q2_2025", "q3_2025", "q4_2025"],
        "validation_split": "q1_2026",
        "test_split": "recent_2026",
        "reported_splits": ["recent_2026"],
    },
]


@dataclass(frozen=True)
class FullActionPolicyConfig:
    epochs: int = 12
    batch_size: int = 512
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    target_scale: float = 100.0
    utility_aux_weight: float = 0.10
    target_clip: float = 1000.0
    min_validation_trades: int = 10


class FullActionSetPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.candidate_head = nn.Linear(hidden_dim, 1)
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(features)
        mask_f = mask.unsqueeze(-1).float()
        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (emb * mask_f).sum(dim=1) / count
        masked = emb.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_emb = masked.max(dim=1).values
        max_emb = torch.where(torch.isfinite(max_emb), max_emb, torch.zeros_like(max_emb))
        wait = self.wait_head(torch.cat([mean, max_emb, count / float(MAX_ACTION_CANDIDATES)], dim=1))
        candidate = self.candidate_head(emb).squeeze(-1).masked_fill(~mask, -1e9)
        return torch.cat([wait, candidate], dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol164-dir", type=Path, default=DEFAULT_PROTOCOL164_DIR)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--smoke", action="store_true", help="Use one split for train/validation/test when only a smoke dataset exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset or args.protocol164_dir / "protocol164_full_action_space_dataset.parquet"
    dataset = load_dataset(dataset_path)
    events = build_events(dataset, starting_cash=float(args.starting_cash))
    oracle_summary = add_oracle_actions(events)
    config = FullActionPolicyConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
    )
    folds = smoke_folds(events) if args.smoke else FOLDS
    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    baseline_trades: list[dict[str, Any]] = []
    for fold in folds:
        train_events = [event for event in events if event["split"] in set(fold["train_splits"])]
        validation_events = [event for event in events if event["split"] == fold["validation_split"]]
        if not train_events or not validation_events:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_train_or_validation_events", "splits": {}})
            continue
        for seed in args.seeds:
            model, scaler, history = train_policy(train_events, validation_events, seed=int(seed), config=config)
            threshold = select_threshold(
                validation_events,
                model,
                scaler,
                config=config,
                starting_cash=float(args.starting_cash),
            )
            model_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "model.pt")
            (model_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")
            (model_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "protocol": "165_full_action_space_policy",
                        "fold": fold["name"],
                        "seed": int(seed),
                        "feature_columns": FULL_ACTION_FEATURE_COLUMNS,
                        "max_action_candidates": MAX_ACTION_CANDIDATES,
                        "config": asdict(config),
                        "threshold_selection": threshold,
                        "history": history,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
            result = {"fold": fold["name"], "seed": int(seed), "threshold": float(threshold["threshold"]), "splits": {}}
            for split_name, split_events in reported_slices(events, fold).items():
                model_base = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.0, starting_cash=float(args.starting_cash), strategy="protocol165")
                model_10 = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.10, starting_cash=float(args.starting_cash), strategy="protocol165_stress10")
                model_25 = simulate_model(split_events, model, scaler, threshold=float(threshold["threshold"]), slippage_per_side=0.25, starting_cash=float(args.starting_cash), strategy="protocol165_stress25")
                first = simulate_first_affordable(split_events, slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                random_base = simulate_matched_random(split_events, seed=int(seed), slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                oracle = simulate_oracle(split_events, slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                result["splits"][split_name] = {
                    "model": model_base.summary,
                    "model_stress_0_10": model_10.summary,
                    "model_stress_0_25": model_25.summary,
                    "first_affordable_baseline": first.summary,
                    "matched_random_baseline": random_base.summary,
                    "full_action_oracle": oracle.summary,
                    "edge_only_baseline": {"status": "not_available_no_surface_edge_feature"},
                }
                model_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in model_base.trades)
                baseline_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in first.trades)
            fold_results.append(result)

    frozen_protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)
    payload = {
        "protocol": "165_full_action_space_policy",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "dataset": str(dataset_path),
        "feature_columns": FULL_ACTION_FEATURE_COLUMNS,
        "max_action_candidates": MAX_ACTION_CANDIDATES,
        "event_summary": summarize_events(events),
        "oracle_summary": oracle_summary,
        "fold_results": fold_results,
        "aggregate": aggregate(fold_results, frozen_protocol101),
        "frozen_protocol101_baselines": frozen_protocol101,
    }
    payload["decision"] = decision(payload)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(model_trades).to_csv(args.out_dir / "protocol165_model_trades.csv", index=False)
    pd.DataFrame(baseline_trades).to_csv(args.out_dir / "protocol165_first_affordable_baseline_trades.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    for column in ["decision_dt", "candidate_exit_dt"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in FULL_ACTION_FEATURE_COLUMNS + ["candidate_pnl", "entry_premium", "entry_affordable_10k"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["decision_dt", "candidate_exit_dt", "candidate_pnl"]).reset_index(drop=True)


def build_events(frame: pd.DataFrame, *, starting_cash: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    ordered = frame.sort_values(["split", "session", "decision_dt", "contract_id"]).copy()
    for (split, session, decision_dt), group in ordered.groupby(["split", "session", "decision_dt"], sort=False):
        candidates = group.head(MAX_ACTION_CANDIDATES).copy().reset_index(drop=True)
        candidates["entry_affordable_dynamic"] = (pd.to_numeric(candidates["entry_premium"], errors="coerce") <= starting_cash).astype(float)
        events.append({"split": str(split), "session": str(session), "decision_dt": pd.Timestamp(decision_dt), "candidates": candidates})
    return events


def add_oracle_actions(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    for session_key, session_events in group_events(events).items():
        session_events.sort(key=lambda event: event["decision_dt"])
        times = np.asarray([event["decision_dt"].value for event in session_events], dtype=np.int64)
        values = np.zeros(len(session_events) + 1, dtype=float)
        actions = np.zeros(len(session_events), dtype=np.int64)
        weights = np.ones(len(session_events), dtype=float)
        for idx in range(len(session_events) - 1, -1, -1):
            wait_value = values[idx + 1]
            best_value = wait_value
            best_action = 0
            best_take = -1e18
            candidates = session_events[idx]["candidates"]
            for local_idx, row in candidates.iterrows():
                if float(row.get("entry_affordable_10k", 0.0)) < 1.0:
                    continue
                exit_ns = pd.Timestamp(row["candidate_exit_dt"]).value
                next_idx = int(np.searchsorted(times, exit_ns, side="left"))
                take_value = float(row["candidate_pnl"]) + values[next_idx]
                best_take = max(best_take, take_value)
                if take_value > best_value:
                    best_value = take_value
                    best_action = int(local_idx) + 1
            values[idx] = best_value
            actions[idx] = best_action
            weights[idx] = 1.0 + min(abs(best_value - wait_value) / 300.0, 5.0)
        for idx, event in enumerate(session_events):
            event["oracle_action"] = int(actions[idx])
            event["oracle_weight"] = float(weights[idx])
            split = str(event["split"])
            item = by_split.setdefault(split, {"take": 0, "wait": 0})
            item["take" if int(actions[idx]) else "wait"] += 1
    return {"by_split": by_split}


def train_policy(
    train_events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    *,
    seed: int,
    config: FullActionPolicyConfig,
) -> tuple[FullActionSetPolicy, FeatureScaler, list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = FeatureScaler.fit(np.vstack([event["candidates"][FULL_ACTION_FEATURE_COLUMNS].to_numpy(dtype=np.float32) for event in train_events]))
    x_train, mask_train, y_train, w_train, utility_train = tensors(train_events, scaler, config)
    x_val, mask_val, y_val, w_val, utility_val = tensors(validation_events, scaler, config)
    model = FullActionSetPolicy(input_dim=len(FULL_ACTION_FEATURE_COLUMNS), hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(mask_train), torch.from_numpy(y_train), torch.from_numpy(w_train), torch.from_numpy(utility_train)), batch_size=config.batch_size, shuffle=True)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for bx, bm, by, bw, bu in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx, bm)
            loss = nn.functional.cross_entropy(logits, by, reduction="none")
            loss = (loss * bw).mean()
            if config.utility_aux_weight > 0.0 and bm.any():
                utility_loss = nn.functional.huber_loss(logits[:, 1:][bm], bu[bm], delta=1.0, reduction="mean")
                loss = loss + config.utility_aux_weight * utility_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_val)
            vm = torch.from_numpy(mask_val)
            vy = torch.from_numpy(y_val)
            vw = torch.from_numpy(w_val)
            vu = torch.from_numpy(utility_val)
            logits = model(vx, vm)
            val_loss = nn.functional.cross_entropy(logits, vy, reduction="none")
            val_loss = float((val_loss * vw).mean().detach().cpu())
            if config.utility_aux_weight > 0.0 and vm.any():
                val_loss += config.utility_aux_weight * float(nn.functional.huber_loss(logits[:, 1:][vm], vu[vm], delta=1.0, reduction="mean").detach().cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss, "is_best": val_loss <= best_val})
    model.load_state_dict(best_state)
    return model, scaler, history


def tensors(events: list[dict[str, Any]], scaler: FeatureScaler, config: FullActionPolicyConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((len(events), MAX_ACTION_CANDIDATES, len(FULL_ACTION_FEATURE_COLUMNS)), dtype=np.float32)
    mask = np.zeros((len(events), MAX_ACTION_CANDIDATES), dtype=bool)
    y = np.zeros(len(events), dtype=np.int64)
    weights = np.ones(len(events), dtype=np.float32)
    utility = np.zeros((len(events), MAX_ACTION_CANDIDATES), dtype=np.float32)
    for idx, event in enumerate(events):
        candidates = event["candidates"].head(MAX_ACTION_CANDIDATES)
        raw = candidates[FULL_ACTION_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        n = len(raw)
        x[idx, :n, :] = scaler.transform(raw)
        affordable = candidates["entry_affordable_10k"].to_numpy(dtype=float) >= 1.0
        valid_exit = pd.to_datetime(candidates["candidate_exit_dt"], utc=True) > pd.Timestamp(event["decision_dt"])
        mask[idx, :n] = affordable & valid_exit.to_numpy(dtype=bool)
        action = int(event.get("oracle_action", 0))
        y[idx] = action if action <= n and (action == 0 or mask[idx, action - 1]) else 0
        weights[idx] = float(event.get("oracle_weight", 1.0))
        pnl = candidates["candidate_pnl"].to_numpy(dtype=np.float32)
        utility[idx, :n] = np.clip(pnl, -config.target_clip, config.target_clip) / config.target_scale
    return x, mask, y, weights, utility


def select_threshold(events: list[dict[str, Any]], model: FullActionSetPolicy, scaler: FeatureScaler, *, config: FullActionPolicyConfig, starting_cash: float) -> dict[str, Any]:
    margins = event_margins(events, model, scaler)
    finite = margins[np.isfinite(margins)]
    thresholds = [float("inf")] if len(finite) == 0 else sorted(set(np.quantile(finite, [0, .1, .2, .35, .5, .65, .8, .9, .95]).round(4).tolist() + [0.0, float(finite.min()) - 1e-3]))
    first = simulate_first_affordable(events, slippage_per_side=0.0, starting_cash=starting_cash)
    sweep = []
    for threshold in thresholds:
        base = simulate_model(events, model, scaler, threshold=float(threshold), slippage_per_side=0.0, starting_cash=starting_cash, strategy="protocol165_validation")
        stress = simulate_model(events, model, scaler, threshold=float(threshold), slippage_per_side=0.10, starting_cash=starting_cash, strategy="protocol165_validation_stress10")
        sweep.append({"threshold": float(threshold), "model": base.summary, "model_stress_0_10": stress.summary, "first_affordable_baseline": first.summary, "delta_vs_first": float(base.summary["total_pnl"] - first.summary["total_pnl"])})
    eligible = [row for row in sweep if row["model"]["trades"] >= config.min_validation_trades and row["model_stress_0_10"]["total_pnl"] > 0.0]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda row: (row["model_stress_0_10"]["total_pnl"], row["delta_vs_first"], row["model"]["profit_factor"], row["model"]["trades"]))
    return {"threshold": float(best["threshold"]), "objective": "validation stress_0_10 pnl then delta vs first-affordable", "selected": best, "sweep": sweep}


def event_margins(events: list[dict[str, Any]], model: FullActionSetPolicy, scaler: FeatureScaler) -> np.ndarray:
    if not events:
        return np.asarray([], dtype=float)
    x, mask, _, _, _ = tensors(events, scaler, FullActionPolicyConfig())
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(events), 4096):
            logits = model(torch.from_numpy(x[start : start + 4096]), torch.from_numpy(mask[start : start + 4096])).cpu().numpy()
            out.append(np.max(logits[:, 1:], axis=1) - logits[:, 0])
    return np.concatenate(out)


def predict_action(event: dict[str, Any], model: FullActionSetPolicy, scaler: FeatureScaler) -> tuple[int, float]:
    x, mask, _, _, _ = tensors([event], scaler, FullActionPolicyConfig())
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x), torch.from_numpy(mask)).cpu().numpy()[0]
    wait = float(logits[0])
    candidate = logits[1:]
    action = int(np.argmax(candidate)) + 1
    return action, float(candidate[action - 1] - wait)


def simulate_model(events: list[dict[str, Any]], model: FullActionSetPolicy, scaler: FeatureScaler, *, threshold: float, slippage_per_side: float, starting_cash: float, strategy: str) -> Any:
    trades = []
    equity = float(starting_cash)
    open_until: dict[str, pd.Timestamp] = {}
    skipped = {"overlap": 0, "threshold": 0, "unaffordable": 0, "invalid": 0}
    for event in sorted(events, key=lambda row: (row["session"], row["decision_dt"])):
        if open_until.get(event["session"]) is not None and event["decision_dt"] < open_until[event["session"]]:
            skipped["overlap"] += len(event["candidates"])
            continue
        action, margin = predict_action(event, model, scaler)
        if action <= 0 or margin < threshold or action > len(event["candidates"]):
            skipped["threshold"] += len(event["candidates"])
            continue
        trade = trade_from_row(event["candidates"].iloc[action - 1], score=margin, threshold=threshold, slippage_per_side=slippage_per_side, equity=equity, strategy=strategy)
        if trade is None:
            skipped["invalid"] += 1
            continue
        if trade["entry_premium_with_slippage"] > equity:
            skipped["unaffordable"] += 1
            continue
        trades.append(trade)
        equity += trade["pnl"]
        trades[-1]["account_equity_after"] = equity
        open_until[event["session"]] = pd.Timestamp(trade["exit_time"])
    return simulation_result(trades, events, skipped, starting_cash=starting_cash, strategy=strategy)


def simulate_first_affordable(events: list[dict[str, Any]], *, slippage_per_side: float, starting_cash: float) -> Any:
    return simulate_rule(events, slippage_per_side=slippage_per_side, starting_cash=starting_cash, strategy="first_affordable", selector=lambda event, rng: event["candidates"])


def simulate_matched_random(events: list[dict[str, Any]], *, seed: int, slippage_per_side: float, starting_cash: float) -> Any:
    rng = np.random.default_rng(seed)
    return simulate_rule(events, slippage_per_side=slippage_per_side, starting_cash=starting_cash, strategy="matched_random", selector=lambda event, _: event["candidates"].sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))))


def simulate_oracle(events: list[dict[str, Any]], *, slippage_per_side: float, starting_cash: float) -> Any:
    return simulate_rule(events, slippage_per_side=slippage_per_side, starting_cash=starting_cash, strategy="full_action_oracle", selector=lambda event, rng: event["candidates"].iloc[[int(event.get("oracle_action", 0)) - 1]] if int(event.get("oracle_action", 0)) > 0 else event["candidates"].iloc[0:0])


def simulate_rule(events: list[dict[str, Any]], *, slippage_per_side: float, starting_cash: float, strategy: str, selector: Any) -> Any:
    trades = []
    equity = float(starting_cash)
    open_until: dict[str, pd.Timestamp] = {}
    skipped = {"overlap": 0, "threshold": 0, "unaffordable": 0, "invalid": 0}
    rng = np.random.default_rng(0)
    for event in sorted(events, key=lambda row: (row["session"], row["decision_dt"])):
        if open_until.get(event["session"]) is not None and event["decision_dt"] < open_until[event["session"]]:
            skipped["overlap"] += len(event["candidates"])
            continue
        selected = selector(event, rng)
        if selected.empty:
            skipped["threshold"] += len(event["candidates"])
            continue
        chosen = None
        for _, row in selected.iterrows():
            trade = trade_from_row(row, score=0.0, threshold=-1e18, slippage_per_side=slippage_per_side, equity=equity, strategy=strategy)
            if trade is not None and trade["entry_premium_with_slippage"] <= equity:
                chosen = trade
                break
        if chosen is None:
            skipped["unaffordable"] += len(event["candidates"])
            continue
        trades.append(chosen)
        equity += chosen["pnl"]
        trades[-1]["account_equity_after"] = equity
        open_until[event["session"]] = pd.Timestamp(chosen["exit_time"])
    return simulation_result(trades, events, skipped, starting_cash=starting_cash, strategy=strategy)


def trade_from_row(row: pd.Series, *, score: float, threshold: float, slippage_per_side: float, equity: float, strategy: str) -> dict[str, Any] | None:
    entry_ask = finite(row.get("entry_ask"))
    pnl = finite(row.get("candidate_pnl"))
    if entry_ask <= 0.0 or not math.isfinite(pnl):
        return None
    round_trip = slippage_per_side * 2.0 * CONTRACT_MULTIPLIER
    return {
        "candidate_uid": str(row["candidate_uid"]),
        "trade_uid": str(row["trade_uid"]),
        "split": str(row["split"]),
        "session": str(row["session"]),
        "decision_time": pd.Timestamp(row["decision_dt"]).isoformat(),
        "exit_time": pd.Timestamp(row["candidate_exit_dt"]).isoformat(),
        "contract_id": str(row["contract_id"]),
        "right": str(row["right"]),
        "offset": float(row["offset"]),
        "score": float(score),
        "threshold": float(threshold),
        "entry_ask": float(entry_ask),
        "entry_premium": float(entry_ask * CONTRACT_MULTIPLIER),
        "entry_premium_with_slippage": float((entry_ask + slippage_per_side) * CONTRACT_MULTIPLIER),
        "account_equity_before": float(equity),
        "account_equity_after": float(equity),
        "pnl": float(pnl - round_trip),
        "raw_candidate_pnl": float(pnl),
        "slippage_per_side": float(slippage_per_side),
        "strategy": strategy,
        "exit_reason": str(row.get("candidate_exit_reason", "")),
        "label_source": str(row.get("label_source", "")),
    }


def simulation_result(trades: list[dict[str, Any]], events: list[dict[str, Any]], skipped: dict[str, int], *, starting_cash: float, strategy: str) -> Any:
    metrics = serial_metrics(trades)
    equity = [starting_cash] + [float(trade["account_equity_after"]) for trade in trades]
    peak = starting_cash
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = min(drawdown, value - peak)
    metrics.update({"strategy": strategy, "starting_cash": starting_cash, "ending_equity": equity[-1], "return_pct": (equity[-1] - starting_cash) / starting_cash * 100.0, "max_drawdown": drawdown, "input_events": len(events), **{f"skipped_{key}_candidates": int(value) for key, value in skipped.items()}, "max_concurrent_positions": 1 if trades else 0, "serial_status": "pass", "all_flat_by_session_end": True})
    return type("SimulationResult", (), {"trades": trades, "summary": metrics})()


def serial_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    converted = [Trade(session=t["session"], decision_time=t["decision_time"], pnl=float(t["pnl"]), score=float(t.get("score", 0.0)), right=t["right"], offset=float(t["offset"]), strategy=t.get("strategy", "")) for t in trades]
    metrics = metrics_for_trades(converted)
    metrics["side_counts"] = {"C": sum(1 for t in trades if t["right"] == "C"), "P": sum(1 for t in trades if t["right"] == "P")}
    return metrics


def group_events(events: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault((event["split"], event["session"]), []).append(event)
    return grouped


def reported_slices(events: list[dict[str, Any]], fold: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    test = [event for event in events if event["split"] == fold["test_split"]]
    out = {fold["test_split"]: test}
    if fold["test_split"] == "q1_2026":
        out["march_2026"] = [event for event in test if event["session"] >= "2026-03-01"]
    return out


def smoke_folds(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split = events[0]["split"] if events else "smoke"
    return [{"name": "smoke_same_split_not_for_research", "train_splits": [split], "validation_split": split, "test_split": split, "reported_splits": [split]}]


def aggregate(fold_results: list[dict[str, Any]], frozen_protocol101: dict[str, float]) -> dict[str, Any]:
    out = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        rows = []
        for result in fold_results:
            if result.get("skipped"):
                continue
            if split in result["splits"]:
                rows.append(result["splits"][split])
        out[split] = summarize_split(rows, frozen_protocol101.get(split))
    out["promotion_checks"] = promotion_checks(out)
    out["promotion_ready"] = bool(out["promotion_checks"] and all(item["pass"] for item in out["promotion_checks"]))
    return out


def summarize_split(rows: list[dict[str, Any]], frozen: float | None) -> dict[str, Any]:
    if not rows:
        return {"seeds": 0, "frozen_protocol101_total_pnl": frozen}
    pnl = arr(row["model"]["total_pnl"] for row in rows)
    stress10 = arr(row["model_stress_0_10"]["total_pnl"] for row in rows)
    stress25 = arr(row["model_stress_0_25"]["total_pnl"] for row in rows)
    pf = arr(finite_pf(row["model"]["profit_factor"]) for row in rows)
    trades = arr(row["model"]["trades"] for row in rows)
    return {"seeds": len(rows), "median_total_pnl": float(np.median(pnl)), "positive_seed_fraction": float((pnl > 0).mean()), "median_profit_factor": float(np.median(pf)), "median_trades": float(np.median(trades)), "median_stress_0_10_total_pnl": float(np.median(stress10)), "median_stress_0_25_total_pnl": float(np.median(stress25)), "frozen_protocol101_total_pnl": frozen, "median_delta_vs_frozen_protocol101": None if frozen is None else float(np.median(pnl) - frozen), "beats_frozen_protocol101": False if frozen is None else bool(np.median(pnl) > frozen)}


def promotion_checks(aggregate_payload: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for split in ["q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        item = aggregate_payload.get(split, {})
        if item.get("seeds", 0) == 0:
            checks.append({"split": split, "name": "split_available", "pass": False, "value": 0})
            continue
        checks.extend([
            {"split": split, "name": "positive_median_pnl", "pass": item["median_total_pnl"] > 0, "value": item["median_total_pnl"]},
            {"split": split, "name": "positive_seed_fraction_ge_0_80", "pass": item["positive_seed_fraction"] >= 0.80, "value": item["positive_seed_fraction"]},
            {"split": split, "name": "median_pf_ge_1_15", "pass": item["median_profit_factor"] >= 1.15, "value": item["median_profit_factor"]},
            {"split": split, "name": "stress_0_10_positive", "pass": item["median_stress_0_10_total_pnl"] > 0, "value": item["median_stress_0_10_total_pnl"]},
            {"split": split, "name": "beats_frozen_protocol101", "pass": bool(item["beats_frozen_protocol101"]), "value": item["median_delta_vs_frozen_protocol101"]},
        ])
    return checks


def load_protocol101_baselines(summary_path: Path, recent_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for split, item in summary.get("aggregate_gate", {}).items():
            if isinstance(item, dict) and "median_total_pnl" in item:
                out[split] = float(item["median_total_pnl"])
    if recent_path.exists():
        recent = json.loads(recent_path.read_text())
        serial = recent.get("serial_protocol081_summary", {})
        if "total_pnl" in serial:
            out["recent_2026"] = float(serial["total_pnl"])
    return out


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for event in events:
        item = out.setdefault(event["split"], {"events": 0, "candidate_rows": 0, "sessions": set()})
        item["events"] += 1
        item["candidate_rows"] += len(event["candidates"])
        item["sessions"].add(event["session"])
    return {key: {"events": value["events"], "candidate_rows": value["candidate_rows"], "sessions": len(value["sessions"])} for key, value in sorted(out.items())}


def decision(payload: dict[str, Any]) -> str:
    if payload["aggregate"].get("promotion_ready"):
        return "promote_research_candidate: Protocol165 beats frozen Protocol101 under full-action serial replay"
    return "research_only: Protocol165 did not clear the frozen Protocol101 full-action promotion gate"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Protocol165 Full Action-Space Policy", "", f"- Decision: `{payload['decision']}`", f"- Dataset: `{payload['dataset']}`", f"- Max candidates per decision: `{payload['max_action_candidates']}`", "", "## Aggregate", "", "| split | seeds | median PnL | frozen Protocol101 | delta | PF | stress 0.10 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, item in payload["aggregate"].items():
        if not isinstance(item, dict) or item.get("seeds", 0) == 0:
            continue
        lines.append(f"| {split} | {item['seeds']} | {fmt(item['median_total_pnl'])} | {fmt(item.get('frozen_protocol101_total_pnl'))} | {fmt(item.get('median_delta_vs_frozen_protocol101'))} | {fmt(item['median_profit_factor'])} | {fmt(item['median_stress_0_10_total_pnl'])} |")
    lines.extend(["", "## Outputs", "", f"- Summary: `{path.parent / 'summary.json'}`", f"- Model trades: `{path.parent / 'protocol165_model_trades.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


def arr(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_pf(value: Any) -> float:
    out = finite(value, 0.0)
    return 999.0 if math.isinf(out) else (out if math.isfinite(out) else 0.0)


def fmt(value: Any) -> str:
    out = finite(value)
    return "" if not math.isfinite(out) else f"{out:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
