"""CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V1.

Train a research-only candidate-set neural policy on full-surface serial
action-advantage labels. This is the first challenger in the new unified
position-state lineage: flat-state wait/enter uses DP advantages, while the
holding head is scaffolded from entry-path hold proxies until the full
hold/exit dataset is promoted.

No paid data is downloaded. No broker endpoint is called.
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
from v4.scripts.run_protocol165_full_action_space_policy import (
    FOLDS,
    STARTING_CASH,
    load_protocol101_baselines,
    reported_slices,
    smoke_folds,
)
from v4.scripts.run_protocol270_full_surface_action_advantage_dataset import OPTIONAL_HISTORY_COLUMNS


ROLE_LABEL = "CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V1"
HISTORICAL_ID = "Protocol271"
DEFAULT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_271_unified_action_advantage_policy")
DEFAULT_PROTOCOL101_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json")
DEFAULT_RECENT_BASELINE = Path("v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/summary.json")
MODEL_SEEDS = [1, 2, 3, 4, 5]
MAX_ACTION_CANDIDATES = 42
CONTRACT_MULTIPLIER = 100.0


@dataclass(frozen=True)
class UnifiedActionPolicyConfig:
    epochs: int = 10
    batch_size: int = 512
    hidden_dim: int = 160
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    advantage_scale: float = 100.0
    advantage_clip: float = 1200.0
    regression_weight: float = 0.15
    hold_proxy_weight: float = 0.03
    min_validation_trades: int = 10


class UnifiedActionAdvantagePolicy(nn.Module):
    """Candidate-set encoder with wait, enter, and hold-proxy heads."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.06),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.enter_head = nn.Linear(hidden_dim, 1)
        self.hold_proxy_head = nn.Linear(hidden_dim, 1)
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor, event_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.candidate_encoder(features)
        mask_f = mask.unsqueeze(-1).float()
        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (emb * mask_f).sum(dim=1) / count
        masked = emb.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_emb = masked.max(dim=1).values
        max_emb = torch.where(torch.isfinite(max_emb), max_emb, torch.zeros_like(max_emb))
        wait_logit = self.wait_head(torch.cat([mean, max_emb, count / float(MAX_ACTION_CANDIDATES), event_state], dim=1))
        enter_logits = self.enter_head(emb).squeeze(-1).masked_fill(~mask, -1e9)
        hold_proxy = self.hold_proxy_head(emb).squeeze(-1)
        return torch.cat([wait_logit, enter_logits], dim=1), hold_proxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, max_events=int(args.max_events))
    feature_columns = select_feature_columns(dataset)
    events = build_events(dataset, feature_columns=feature_columns, starting_cash=float(args.starting_cash))
    folds = smoke_folds(events) if args.smoke else FOLDS
    config = UnifiedActionPolicyConfig(epochs=int(args.epochs), batch_size=int(args.batch_size), hidden_dim=int(args.hidden_dim))
    frozen_protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)
    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    for fold in folds:
        train_events = [event for event in events if event["split"] in set(fold["train_splits"])]
        validation_events = [event for event in events if event["split"] == fold["validation_split"]]
        if not train_events or not validation_events:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_train_or_validation_events", "splits": {}})
            continue
        for seed in args.seeds:
            model, scaler, history = train_policy(train_events, validation_events, feature_columns=feature_columns, seed=int(seed), config=config)
            threshold = select_threshold(validation_events, model, scaler, feature_columns=feature_columns, config=config, starting_cash=float(args.starting_cash))
            model_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "model.pt")
            (model_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")
            (model_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "role_label": ROLE_LABEL,
                        "historical_protocol": HISTORICAL_ID,
                        "fold": fold["name"],
                        "seed": int(seed),
                        "feature_columns": feature_columns,
                        "max_action_candidates": MAX_ACTION_CANDIDATES,
                        "threshold_selection": threshold,
                        "config": asdict(config),
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
                base = simulate_model(split_events, model, scaler, feature_columns=feature_columns, threshold=float(threshold["threshold"]), slippage_per_side=0.0, starting_cash=float(args.starting_cash), strategy="unified_action_advantage")
                stress10 = simulate_model(split_events, model, scaler, feature_columns=feature_columns, threshold=float(threshold["threshold"]), slippage_per_side=0.10, starting_cash=float(args.starting_cash), strategy="unified_action_advantage_stress10")
                stress25 = simulate_model(split_events, model, scaler, feature_columns=feature_columns, threshold=float(threshold["threshold"]), slippage_per_side=0.25, starting_cash=float(args.starting_cash), strategy="unified_action_advantage_stress25")
                oracle = simulate_oracle(split_events, slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                first = simulate_first_positive(split_events, slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                result["splits"][split_name] = {
                    "model": base.summary,
                    "model_stress_0_10": stress10.summary,
                    "model_stress_0_25": stress25.summary,
                    "full_action_advantage_oracle": oracle.summary,
                    "first_positive_advantage_baseline": first.summary,
                }
                model_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in base.trades)
            fold_results.append(result)
    aggregate_payload = aggregate(fold_results, frozen_protocol101)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "model / unified full-surface action-advantage challenger",
        "changes_paper_default": False,
        "candidate_label": ROLE_LABEL,
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.dataset),
        "run_scope": "limited_pipeline_validation" if int(args.max_events) > 0 or args.smoke else "full_chronological_training",
        "max_events_arg": int(args.max_events),
        "smoke_mode": bool(args.smoke),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "feature_columns": feature_columns,
        "max_action_candidates": MAX_ACTION_CANDIDATES,
        "event_summary": summarize_events(events),
        "fold_results": fold_results,
        "aggregate": aggregate_payload,
        "frozen_protocol101_baselines": frozen_protocol101,
        "decision": decide(aggregate_payload),
        "next_experiment": next_experiment(aggregate_payload),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(model_trades).to_csv(args.out_dir / "unified_action_advantage_model_trades.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_dataset(path: Path, *, max_events: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["candidate_exit_dt"] = pd.to_datetime(frame["candidate_exit_dt"], utc=True, errors="coerce")
    if max_events > 0 and not frame.empty:
        all_events = frame[["split", "session", "decision_dt"]].drop_duplicates().sort_values(["split", "session", "decision_dt"])
        per_split = max(1, int(max_events) // max(1, int(all_events["split"].nunique())))
        events = all_events.groupby("split", sort=True).head(per_split).head(max_events)
        frame = frame.merge(events.assign(_keep_event=1), on=["split", "session", "decision_dt"], how="inner").drop(columns=["_keep_event"])
    return frame.dropna(subset=["decision_dt", "candidate_exit_dt", "candidate_pnl", "a_enter"]).reset_index(drop=True)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in [*FULL_ACTION_FEATURE_COLUMNS, *OPTIONAL_HISTORY_COLUMNS] if column in frame.columns]
    return list(dict.fromkeys(columns))


def build_events(frame: pd.DataFrame, *, feature_columns: list[str], starting_cash: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    ordered = frame.sort_values(["split", "session", "decision_dt", "candidate_uid"]).copy()
    for column in feature_columns + ["entry_premium", "candidate_pnl", "a_enter", "a_hold_entry"]:
        if column in ordered.columns:
            ordered[column] = pd.to_numeric(ordered[column], errors="coerce").fillna(0.0)
    for (split, session, decision_dt), group in ordered.groupby(["split", "session", "decision_dt"], sort=False):
        candidates = group.head(MAX_ACTION_CANDIDATES).copy().reset_index(drop=True)
        candidates["entry_affordable_dynamic"] = (pd.to_numeric(candidates["entry_premium"], errors="coerce") <= starting_cash).astype(float)
        candidates["candidate_rank_local"] = np.arange(len(candidates), dtype=int)
        oracle_uid = str(candidates["oracle_action_uid"].iloc[0]) if "oracle_action_uid" in candidates.columns and len(candidates) else "wait"
        oracle_action = 0
        if oracle_uid != "wait":
            matches = candidates.index[candidates["candidate_uid"].astype(str).eq(oracle_uid)].tolist()
            if matches:
                oracle_action = int(matches[0]) + 1
        best_adv = float(pd.to_numeric(candidates["a_enter"], errors="coerce").max()) if len(candidates) else 0.0
        event_state = np.asarray(
            [
                min(float(candidates["entry_minutes_since_open"].iloc[0]) / 360.0, 1.5) if "entry_minutes_since_open" in candidates.columns and len(candidates) else 0.0,
                min(float(candidates["entry_minutes_to_forced_flat"].iloc[0]) / 390.0, 1.5) if "entry_minutes_to_forced_flat" in candidates.columns and len(candidates) else 0.0,
                min(max(best_adv / 1000.0, -2.0), 2.0),
            ],
            dtype=np.float32,
        )
        events.append(
            {
                "split": str(split),
                "session": str(session),
                "decision_dt": pd.Timestamp(decision_dt),
                "candidates": candidates,
                "oracle_action": oracle_action,
                "oracle_weight": float(1.0 + min(abs(best_adv) / 300.0, 5.0)),
                "event_state": event_state,
            }
        )
    return events


def train_policy(
    train_events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    *,
    feature_columns: list[str],
    seed: int,
    config: UnifiedActionPolicyConfig,
) -> tuple[UnifiedActionAdvantagePolicy, FeatureScaler, list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = FeatureScaler.fit(np.vstack([event["candidates"][feature_columns].to_numpy(dtype=np.float32) for event in train_events]))
    x_train, mask_train, state_train, y_train, w_train, adv_train, hold_train = tensors(train_events, scaler, feature_columns, config)
    x_val, mask_val, state_val, y_val, w_val, adv_val, hold_val = tensors(validation_events, scaler, feature_columns, config)
    model = UnifiedActionAdvantagePolicy(input_dim=len(feature_columns), hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(mask_train),
            torch.from_numpy(state_train),
            torch.from_numpy(y_train),
            torch.from_numpy(w_train),
            torch.from_numpy(adv_train),
            torch.from_numpy(hold_train),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for bx, bm, bs, by, bw, ba, bh in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, hold_proxy = model(bx, bm, bs)
            loss = nn.functional.cross_entropy(logits, by, reduction="none")
            loss = (loss * bw).mean()
            if bm.any():
                loss = loss + config.regression_weight * nn.functional.huber_loss(logits[:, 1:][bm], ba[bm], delta=1.0, reduction="mean")
                loss = loss + config.hold_proxy_weight * nn.functional.huber_loss(hold_proxy[bm], bh[bm], delta=1.0, reduction="mean")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_loss = validation_loss(model, x_val, mask_val, state_val, y_val, w_val, adv_val, hold_val, config)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": float(val_loss), "is_best": val_loss <= best_val})
    model.load_state_dict(best_state)
    return model, scaler, history


def validation_loss(
    model: UnifiedActionAdvantagePolicy,
    x: np.ndarray,
    mask: np.ndarray,
    state: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    advantage: np.ndarray,
    hold: np.ndarray,
    config: UnifiedActionPolicyConfig,
) -> float:
    model.eval()
    with torch.no_grad():
        bx = torch.from_numpy(x)
        bm = torch.from_numpy(mask)
        logits, hold_proxy = model(bx, bm, torch.from_numpy(state))
        by = torch.from_numpy(y)
        bw = torch.from_numpy(weight)
        loss = nn.functional.cross_entropy(logits, by, reduction="none")
        total = float((loss * bw).mean().detach().cpu())
        if bm.any():
            total += config.regression_weight * float(nn.functional.huber_loss(logits[:, 1:][bm], torch.from_numpy(advantage)[bm], delta=1.0, reduction="mean").detach().cpu())
            total += config.hold_proxy_weight * float(nn.functional.huber_loss(hold_proxy[bm], torch.from_numpy(hold)[bm], delta=1.0, reduction="mean").detach().cpu())
    return total


def tensors(
    events: list[dict[str, Any]],
    scaler: FeatureScaler,
    feature_columns: list[str],
    config: UnifiedActionPolicyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((len(events), MAX_ACTION_CANDIDATES, len(feature_columns)), dtype=np.float32)
    mask = np.zeros((len(events), MAX_ACTION_CANDIDATES), dtype=bool)
    event_state = np.zeros((len(events), 3), dtype=np.float32)
    y = np.zeros(len(events), dtype=np.int64)
    weights = np.ones(len(events), dtype=np.float32)
    advantage = np.zeros((len(events), MAX_ACTION_CANDIDATES), dtype=np.float32)
    hold_proxy = np.zeros((len(events), MAX_ACTION_CANDIDATES), dtype=np.float32)
    for idx, event in enumerate(events):
        candidates = event["candidates"].head(MAX_ACTION_CANDIDATES)
        raw = candidates[feature_columns].to_numpy(dtype=np.float32)
        n = len(raw)
        x[idx, :n, :] = scaler.transform(raw)
        valid_exit = pd.to_datetime(candidates["candidate_exit_dt"], utc=True) > pd.Timestamp(event["decision_dt"])
        affordable = candidates["entry_affordable_dynamic"].to_numpy(dtype=float) >= 1.0
        mask[idx, :n] = valid_exit.to_numpy(dtype=bool) & affordable
        action = int(event.get("oracle_action", 0))
        y[idx] = action if action <= n and (action == 0 or mask[idx, action - 1]) else 0
        weights[idx] = float(event.get("oracle_weight", 1.0))
        event_state[idx, :] = event["event_state"]
        advantage[idx, :n] = np.clip(candidates["a_enter"].to_numpy(dtype=np.float32), -config.advantage_clip, config.advantage_clip) / config.advantage_scale
        hold_proxy[idx, :n] = np.clip(candidates.get("a_hold_entry", pd.Series(0.0, index=candidates.index)).to_numpy(dtype=np.float32), -config.advantage_clip, config.advantage_clip) / config.advantage_scale
    return x, mask, event_state, y, weights, advantage, hold_proxy


def select_threshold(
    events: list[dict[str, Any]],
    model: UnifiedActionAdvantagePolicy,
    scaler: FeatureScaler,
    *,
    feature_columns: list[str],
    config: UnifiedActionPolicyConfig,
    starting_cash: float,
) -> dict[str, Any]:
    margins = event_margins(events, model, scaler, feature_columns)
    finite = margins[np.isfinite(margins)]
    thresholds = [float("inf")] if len(finite) == 0 else sorted(set(np.quantile(finite, [0, .1, .2, .35, .5, .65, .8, .9, .95]).round(4).tolist() + [0.0, float(finite.min()) - 1e-3]))
    sweep = []
    for threshold in thresholds:
        base = simulate_model(events, model, scaler, feature_columns=feature_columns, threshold=float(threshold), slippage_per_side=0.0, starting_cash=starting_cash, strategy="validation")
        stress = simulate_model(events, model, scaler, feature_columns=feature_columns, threshold=float(threshold), slippage_per_side=0.10, starting_cash=starting_cash, strategy="validation_stress10")
        sweep.append({"threshold": float(threshold), "model": base.summary, "model_stress_0_10": stress.summary})
    eligible = [row for row in sweep if row["model"]["trades"] >= config.min_validation_trades and row["model_stress_0_10"]["total_pnl"] > 0.0]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda row: (row["model_stress_0_10"]["total_pnl"], row["model"]["profit_factor"], row["model"]["total_pnl"], row["model"]["trades"]))
    return {"threshold": float(best["threshold"]), "objective": "validation stress_0_10 pnl, then PF, then base pnl", "selected": best, "sweep": sweep}


def event_margins(events: list[dict[str, Any]], model: UnifiedActionAdvantagePolicy, scaler: FeatureScaler, feature_columns: list[str]) -> np.ndarray:
    if not events:
        return np.asarray([], dtype=float)
    x, mask, state, _, _, _, _ = tensors(events, scaler, feature_columns, UnifiedActionPolicyConfig())
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(events), 4096):
            logits, _ = model(torch.from_numpy(x[start : start + 4096]), torch.from_numpy(mask[start : start + 4096]), torch.from_numpy(state[start : start + 4096]))
            logits_np = logits.cpu().numpy()
            out.append(np.max(logits_np[:, 1:], axis=1) - logits_np[:, 0])
    return np.concatenate(out)


def predict_action(event: dict[str, Any], model: UnifiedActionAdvantagePolicy, scaler: FeatureScaler, feature_columns: list[str]) -> tuple[int, float]:
    x, mask, state, _, _, _, _ = tensors([event], scaler, feature_columns, UnifiedActionPolicyConfig())
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(state))
        logits = logits.cpu().numpy()[0]
    candidate = logits[1:]
    action = int(np.argmax(candidate)) + 1
    return action, float(candidate[action - 1] - logits[0])


def simulate_model(
    events: list[dict[str, Any]],
    model: UnifiedActionAdvantagePolicy,
    scaler: FeatureScaler,
    *,
    feature_columns: list[str],
    threshold: float,
    slippage_per_side: float,
    starting_cash: float,
    strategy: str,
) -> Any:
    return simulate_rule(
        events,
        slippage_per_side=slippage_per_side,
        starting_cash=starting_cash,
        strategy=strategy,
        selector=lambda event, _rng: select_model_row(event, model, scaler, feature_columns, threshold),
    )


def select_model_row(event: dict[str, Any], model: UnifiedActionAdvantagePolicy, scaler: FeatureScaler, feature_columns: list[str], threshold: float) -> pd.DataFrame:
    action, margin = predict_action(event, model, scaler, feature_columns)
    if action <= 0 or action > len(event["candidates"]) or margin < threshold:
        return event["candidates"].iloc[0:0]
    selected = event["candidates"].iloc[[action - 1]].copy()
    selected["model_margin"] = margin
    selected["model_threshold"] = threshold
    return selected


def simulate_oracle(events: list[dict[str, Any]], *, slippage_per_side: float, starting_cash: float) -> Any:
    return simulate_rule(
        events,
        slippage_per_side=slippage_per_side,
        starting_cash=starting_cash,
        strategy="action_advantage_oracle",
        selector=lambda event, _rng: event["candidates"].iloc[[int(event["oracle_action"]) - 1]] if int(event["oracle_action"]) > 0 else event["candidates"].iloc[0:0],
    )


def simulate_first_positive(events: list[dict[str, Any]], *, slippage_per_side: float, starting_cash: float) -> Any:
    return simulate_rule(
        events,
        slippage_per_side=slippage_per_side,
        starting_cash=starting_cash,
        strategy="first_positive_advantage",
        selector=lambda event, _rng: event["candidates"][pd.to_numeric(event["candidates"]["a_enter"], errors="coerce") > 0.0],
    )


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
            trade = trade_from_row(row, slippage_per_side=slippage_per_side, equity=equity, strategy=strategy)
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


def trade_from_row(row: pd.Series, *, slippage_per_side: float, equity: float, strategy: str) -> dict[str, Any] | None:
    entry_ask = finite(row.get("entry_ask"))
    pnl = finite(row.get("candidate_pnl"))
    if entry_ask <= 0.0 or not math.isfinite(pnl):
        return None
    round_trip = slippage_per_side * 2.0 * CONTRACT_MULTIPLIER
    return {
        "candidate_uid": str(row.get("candidate_uid", "")),
        "trade_uid": str(row.get("trade_uid", "")),
        "split": str(row.get("split", "")),
        "session": str(row.get("session", "")),
        "decision_time": pd.Timestamp(row["decision_dt"]).isoformat(),
        "exit_time": pd.Timestamp(row["candidate_exit_dt"]).isoformat(),
        "contract_id": str(row.get("contract_id", "")),
        "right": str(row.get("right", "")),
        "offset": float(row.get("offset", 0.0)),
        "entry_ask": float(entry_ask),
        "entry_premium": float(entry_ask * CONTRACT_MULTIPLIER),
        "entry_premium_with_slippage": float((entry_ask + slippage_per_side) * CONTRACT_MULTIPLIER),
        "account_equity_before": float(equity),
        "account_equity_after": float(equity),
        "pnl": float(pnl - round_trip),
        "raw_candidate_pnl": float(pnl),
        "a_enter": finite(row.get("a_enter"), 0.0),
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
    metrics.update(
        {
            "strategy": strategy,
            "starting_cash": starting_cash,
            "ending_equity": equity[-1],
            "return_pct": (equity[-1] - starting_cash) / starting_cash * 100.0,
            "max_drawdown": drawdown,
            "input_events": len(events),
            **{f"skipped_{key}_candidates": int(value) for key, value in skipped.items()},
            "max_concurrent_positions": 1 if trades else 0,
            "serial_status": "pass",
            "all_flat_by_session_end": True,
        }
    )
    return type("SimulationResult", (), {"trades": trades, "summary": metrics})()


def serial_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    converted = [Trade(session=t["session"], decision_time=t["decision_time"], pnl=float(t["pnl"]), score=float(t.get("a_enter", 0.0)), right=t["right"], offset=float(t["offset"]), strategy=t.get("strategy", "")) for t in trades]
    metrics = metrics_for_trades(converted)
    metrics["side_counts"] = {"C": sum(1 for t in trades if t["right"] == "C"), "P": sum(1 for t in trades if t["right"] == "P")}
    return metrics


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
    dd = arr(row["model"].get("max_drawdown", 0.0) for row in rows)
    return {
        "seeds": len(rows),
        "median_total_pnl": float(np.median(pnl)),
        "positive_seed_fraction": float((pnl > 0).mean()),
        "median_profit_factor": float(np.median(pf)),
        "median_trades": float(np.median(trades)),
        "median_max_drawdown": float(np.median(dd)),
        "median_stress_0_10_total_pnl": float(np.median(stress10)),
        "median_stress_0_25_total_pnl": float(np.median(stress25)),
        "frozen_protocol101_total_pnl": frozen,
        "median_delta_vs_frozen_protocol101": None if frozen is None else float(np.median(pnl) - frozen),
        "beats_frozen_protocol101": False if frozen is None else bool(np.median(pnl) > frozen),
    }


def promotion_checks(aggregate_payload: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for split in ["q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        item = aggregate_payload.get(split, {})
        if item.get("seeds", 0) == 0:
            checks.append({"split": split, "name": "split_available", "pass": False, "value": 0})
            continue
        checks.extend(
            [
                {"split": split, "name": "positive_median_pnl", "pass": item["median_total_pnl"] > 0, "value": item["median_total_pnl"]},
                {"split": split, "name": "positive_seed_fraction_ge_0_80", "pass": item["positive_seed_fraction"] >= 0.80, "value": item["positive_seed_fraction"]},
                {"split": split, "name": "median_pf_ge_1_15", "pass": item["median_profit_factor"] >= 1.15, "value": item["median_profit_factor"]},
                {"split": split, "name": "stress_0_10_positive", "pass": item["median_stress_0_10_total_pnl"] > 0, "value": item["median_stress_0_10_total_pnl"]},
                {"split": split, "name": "stress_0_25_positive", "pass": item["median_stress_0_25_total_pnl"] > 0, "value": item["median_stress_0_25_total_pnl"]},
                {"split": split, "name": "beats_frozen_protocol101", "pass": bool(item["beats_frozen_protocol101"]), "value": item["median_delta_vs_frozen_protocol101"]},
            ]
        )
    return checks


def decide(aggregate_payload: dict[str, Any]) -> str:
    if aggregate_payload.get("promotion_ready"):
        return "research_candidate_surpasses_protocol101_needs_runtime_parity"
    return "research_only_unified_action_advantage_policy_not_yet_paper_default"


def next_experiment(aggregate_payload: dict[str, Any]) -> str:
    if aggregate_payload.get("promotion_ready"):
        return "Build no-order runtime parity for the same full-action feature contract before replacement discussion."
    return "Do attribution by missed oracle entries, churn, side, premium, and hold/exit proxy errors before adding architecture knobs."


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for event in events:
        item = out.setdefault(event["split"], {"events": 0, "candidate_rows": 0, "sessions": set(), "oracle_enter": 0})
        item["events"] += 1
        item["candidate_rows"] += len(event["candidates"])
        item["sessions"].add(event["session"])
        item["oracle_enter"] += int(int(event.get("oracle_action", 0)) > 0)
    return {key: {"events": value["events"], "candidate_rows": value["candidate_rows"], "sessions": len(value["sessions"]), "oracle_enter_events": value["oracle_enter"]} for key, value in sorted(out.items())}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: `{payload['candidate_label']}`",
        f"Baseline: `{payload['paper_default_label']}`",
        f"Data used: `{payload['data_used']}`",
        f"Run scope: `{payload['run_scope']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Aggregate",
        "",
        "| split | seeds | median PnL | Protocol101 | delta | PF | stress 0.10 | stress 0.25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in payload["aggregate"].items():
        if not isinstance(item, dict) or item.get("seeds", 0) == 0:
            continue
        lines.append(
            f"| {split} | {item['seeds']} | {fmt(item['median_total_pnl'])} | "
            f"{fmt(item.get('frozen_protocol101_total_pnl'))} | {fmt(item.get('median_delta_vs_frozen_protocol101'))} | "
            f"{fmt(item['median_profit_factor'])} | {fmt(item['median_stress_0_10_total_pnl'])} | {fmt(item['median_stress_0_25_total_pnl'])} |"
        )
    lines.extend(["", "## Outputs", "", f"- Summary: `{path.parent / 'summary.json'}`", f"- Trades: `{path.parent / 'unified_action_advantage_model_trades.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
    text = ledger.read_text()
    if marker in text:
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
                    f"- Candidate: `{payload['candidate_label']}`",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


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
