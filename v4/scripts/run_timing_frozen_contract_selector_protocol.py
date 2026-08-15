"""Protocol 037: timing-frozen same-side contract selector.

Protocol 024's entry timing is the current frozen baseline. Protocol 034 showed
that hard contract-quality gates can reject March convex winners. This protocol
therefore keeps the Protocol 024 entry minute and side fixed, and trains only a
same-side contract selector:

    enter now -> choose which call/put strike expresses this setup best.

The selector cannot skip, cannot flip side, and cannot alter cooldown/trade
count. No paid data is downloaded.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.model.environment_diagnostics import time_bucket
from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    ProtocolTrial,
    SurfaceDecision,
    SurfaceVariant,
    predict_surface_actions,
    registered_aplus_surface_variants,
    registered_protocol_trials,
    selection_reward,
    stress_trades,
    summarize_random_baseline,
    token_feature_names,
    train_surface_model,
    window_seed,
)
from v4.model.supervised_pilot import FeatureScaler, PilotConfig, Trade
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import (
    _load_surface_decisions_cached,
    _paths_by_split,
    _protocol_window,
)
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_hypothesis_037_timing_frozen_contract_selector"
TARGET_SCALE = 100.0
BASELINE = {
    "selection": {"pnl": 6460.0, "trades": 16.0, "stress50": 6010.0},
    "march_2026": {"pnl": 11300.0, "trades": 20.0, "stress50": 10300.0},
    "q1_2025": {"pnl": 7170.0, "trades": 106.0, "stress50": 1870.0},
    "q2_2025": {"pnl": 8270.0, "trades": 104.0, "stress50": 2870.0},
    "q3_2025": {"pnl": 13470.0, "trades": 127.0, "stress50": 7340.0},
    "q4_2025": {"pnl": 9210.0, "trades": 111.0, "stress50": 3110.0},
}


@dataclass
class EntryEvent:
    split: str
    seed: int
    effective_seed: int
    decision: SurfaceDecision
    original_token_idx: int
    edge: float

    @property
    def side(self) -> str:
        return str(self.decision.rights[self.original_token_idx])

    @property
    def baseline_pnl(self) -> float:
        return float(self.decision.labels[self.original_token_idx])


@dataclass(frozen=True)
class SelectorConfig:
    name: str
    selector_weight: float
    candidate_band: float | None


class ContractSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.06),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived_nofee"))
    parser.add_argument("--q1-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025_nofee"))
    parser.add_argument("--q2-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025_nofee"))
    parser.add_argument("--q3-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025_nofee"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025_nofee"))
    parser.add_argument("--seed-q4-data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_037_timing_frozen_contract_selector"))
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions_nofee"))
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--variant-name", default="surface_structure_aplus_side_value_multitask")
    parser.add_argument("--trial-name", default="post_open_late_edge25_max2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--selector-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def _paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("*.pkl"))
    if not paths:
        raise SystemExit(f"no pkl files found under {data_dir}")
    return paths


def _find_variant(name: str) -> SurfaceVariant:
    for variant in registered_aplus_surface_variants():
        if variant.name == name:
            return variant
    raise SystemExit(f"unknown A+ variant: {name}")


def _find_trial(name: str) -> ProtocolTrial:
    for trial in registered_protocol_trials():
        if trial.name == name:
            return trial
    raise SystemExit(f"unknown protocol trial: {name}")


def _selector_grid() -> list[SelectorConfig]:
    configs = [SelectorConfig("baseline_original_contract", 0.0, 0.0)]
    for weight in (0.25, 0.50, 0.75, 1.00):
        for band in (25.0, 50.0, 100.0, None):
            label = "all" if band is None else int(band)
            configs.append(SelectorConfig(f"selector_w{int(weight * 100)}_band{label}", weight, band))
    return configs


def _select_entry_events(
    decisions: Sequence[SurfaceDecision],
    predictions: np.ndarray,
    *,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    split: str,
    seed: int,
    effective_seed: int,
) -> list[EntryEvent]:
    events: list[EntryEvent] = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    pnl_by_session: dict[str, float] = {}
    halted_sessions: set[str] = set()
    allowed = set(trial.allowed_buckets)
    for decision, pred in zip(decisions, predictions):
        if time_bucket(decision.decision_time) not in allowed:
            continue
        if decision.session in halted_sessions:
            continue
        if trades_by_session.get(decision.session, 0) >= trial.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        action_mask = np.concatenate([[True], decision.token_mask])
        masked = np.asarray(pred, dtype=float).copy()
        masked[~action_mask] = -np.inf
        if not np.isfinite(masked).any():
            continue
        action = int(np.nanargmax(masked))
        if action == 0:
            continue
        edge = float(masked[action] - masked[0])
        if not np.isfinite(edge) or edge < trial.min_edge_vs_no_trade:
            continue
        token_idx = action - 1
        pnl = float(decision.labels[token_idx])
        if not np.isfinite(pnl):
            continue
        events.append(
            EntryEvent(
                split=split,
                seed=int(seed),
                effective_seed=int(effective_seed),
                decision=decision,
                original_token_idx=int(token_idx),
                edge=edge,
            )
        )
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        pnl_by_session[decision.session] = pnl_by_session.get(decision.session, 0.0) + pnl
        next_time_by_session[decision.session] = decision.decision_time + timedelta(minutes=cooldown_minutes)
        if trial.daily_loss_stop is not None and pnl_by_session[decision.session] <= trial.daily_loss_stop:
            halted_sessions.add(decision.session)
    return events


def _candidate_indices(event: EntryEvent, pred: np.ndarray, config: SelectorConfig | None = None) -> np.ndarray:
    decision = event.decision
    side = event.side
    valid = decision.token_mask & np.isfinite(decision.labels) & (decision.rights == side)
    if config is not None and config.candidate_band is not None:
        scores = np.asarray(pred[1:], dtype=float)
        original_score = float(scores[event.original_token_idx])
        valid &= scores >= original_score - float(config.candidate_band)
    idxs = np.where(valid)[0]
    if len(idxs) == 0:
        return np.asarray([event.original_token_idx], dtype=int)
    return idxs.astype(int)


def _candidate_features(
    event: EntryEvent,
    pred: np.ndarray,
    *,
    token_idx: int,
) -> np.ndarray:
    decision = event.decision
    token = np.asarray(decision.token_features[token_idx], dtype=np.float32)
    token_scores = np.asarray(pred[1:], dtype=float)
    flat_score = float(pred[0])
    original_score = float(token_scores[event.original_token_idx])
    current_score = float(token_scores[token_idx])
    extra = np.asarray(
        [
            current_score / TARGET_SCALE,
            (current_score - flat_score) / TARGET_SCALE,
            (current_score - original_score) / TARGET_SCALE,
            float(token_idx == event.original_token_idx),
            (float(decision.offsets[token_idx]) - float(decision.offsets[event.original_token_idx])) / 50.0,
            abs(float(decision.offsets[token_idx]) - float(decision.offsets[event.original_token_idx])) / 50.0,
            event.edge / TARGET_SCALE,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(np.concatenate([token, extra]).astype(np.float32), nan=0.0, posinf=8.0, neginf=-8.0)


def _build_group_tensors(
    events: Sequence[EntryEvent],
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups_x = []
    groups_y = []
    max_len = 1
    raw_groups: list[tuple[list[np.ndarray], list[float]]] = []
    for event, pred in zip(events, predictions):
        idxs = _candidate_indices(event, pred, None)
        xs = [_candidate_features(event, pred, token_idx=int(idx)) for idx in idxs]
        ys = [float(event.decision.labels[int(idx)]) for idx in idxs]
        if xs:
            raw_groups.append((xs, ys))
            max_len = max(max_len, len(xs))
    if not raw_groups:
        return (
            np.empty((0, 1, 1), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0, 1), dtype=bool),
        )
    feat_dim = len(raw_groups[0][0][0])
    for xs, ys in raw_groups:
        x = np.zeros((max_len, feat_dim), dtype=np.float32)
        y = np.zeros((max_len,), dtype=np.float32)
        mask = np.zeros((max_len,), dtype=bool)
        for i, (feat, label) in enumerate(zip(xs, ys)):
            x[i] = feat
            y[i] = np.clip(label, -600.0, 600.0) / TARGET_SCALE
            mask[i] = True
        groups_x.append(x)
        groups_y.append(y)
        # Store mask in a parallel list by temporarily attaching to y? Keep clear below.
    masks = []
    for xs, _ in raw_groups:
        mask = np.zeros((max_len,), dtype=bool)
        mask[: len(xs)] = True
        masks.append(mask)
    return np.stack(groups_x), np.stack(groups_y), np.stack(masks)


def _fit_selector(
    train_events: Sequence[EntryEvent],
    train_predictions: np.ndarray,
    validation_events: Sequence[EntryEvent],
    validation_predictions: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[ContractSelector, FeatureScaler, list[dict]]:
    train_x, train_y, train_mask = _build_group_tensors(train_events, train_predictions)
    val_x, val_y, val_mask = _build_group_tensors(validation_events, validation_predictions)
    if len(train_x) == 0:
        raise ValueError("cannot train contract selector with zero timing-frozen events")
    flat_train = train_x[train_mask].astype(np.float32)
    scaler = FeatureScaler.fit(flat_train)

    def transform_group(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(-1, x.shape[-1]).astype(np.float32)
        return scaler.transform(flat).reshape(x.shape)

    train_x = transform_group(train_x)
    if len(val_x):
        val_x = transform_group(val_x)
    else:
        val_x, val_y, val_mask = train_x, train_y, train_mask
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ContractSelector(input_dim=train_x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y.astype(np.float32)),
            torch.from_numpy(train_mask),
        ),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
    )
    val_tensors = (
        torch.from_numpy(val_x),
        torch.from_numpy(val_y.astype(np.float32)),
        torch.from_numpy(val_mask),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb, mb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            huber = F.huber_loss(pred[mb], yb[mb], delta=1.0)
            masked_y = yb.masked_fill(~mb, -1_000_000.0)
            best_idx = torch.argmax(masked_y, dim=1)
            ce = F.cross_entropy(pred.masked_fill(~mb, -1_000_000.0) / 0.75, best_idx)
            loss = huber + 0.30 * ce
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensors[0])
            val_huber = F.huber_loss(val_pred[val_tensors[2]], val_tensors[1][val_tensors[2]], delta=1.0)
            val_best = torch.argmax(val_tensors[1].masked_fill(~val_tensors[2], -1_000_000.0), dim=1)
            val_ce = F.cross_entropy(val_pred.masked_fill(~val_tensors[2], -1_000_000.0) / 0.75, val_best)
            val_loss = val_huber + 0.30 * val_ce
        is_best = float(val_loss.detach().cpu()) < best_val
        if is_best:
            best_val = float(val_loss.detach().cpu())
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": float(val_loss.detach().cpu()),
                "validation_huber": float(val_huber.detach().cpu()),
                "validation_ce": float(val_ce.detach().cpu()),
                "is_best": is_best,
            }
        )
    model.load_state_dict(best_state)
    return model, scaler, history


def _predict_selector(model: ContractSelector, scaler: FeatureScaler, x: np.ndarray) -> np.ndarray:
    x_scaled = scaler.transform(x.astype(np.float32))
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x_scaled)).cpu().numpy().astype(np.float32) * TARGET_SCALE


def _choose_contract(
    event: EntryEvent,
    pred: np.ndarray,
    *,
    selector_model: ContractSelector,
    selector_scaler: FeatureScaler,
    config: SelectorConfig,
) -> tuple[int, float, bool]:
    if config.selector_weight <= 0.0:
        return event.original_token_idx, float(event.edge), False
    idxs = _candidate_indices(event, pred, config)
    features = np.vstack([_candidate_features(event, pred, token_idx=int(idx)) for idx in idxs]).astype(np.float32)
    selector_scores = _predict_selector(selector_model, selector_scaler, features)
    base_scores = np.asarray([float(pred[int(idx) + 1]) for idx in idxs], dtype=float)
    adjusted = (1.0 - config.selector_weight) * base_scores + config.selector_weight * selector_scores
    chosen_pos = int(np.argmax(adjusted))
    chosen_idx = int(idxs[chosen_pos])
    edge = float(pred[chosen_idx + 1] - pred[0])
    return chosen_idx, edge, chosen_idx != event.original_token_idx


def _trade_from_event(event: EntryEvent, token_idx: int, *, edge: float, strategy: str) -> Trade:
    return Trade(
        session=event.decision.session,
        decision_time=event.decision.decision_time.isoformat(),
        pnl=float(event.decision.labels[token_idx]),
        score=float(edge),
        right=str(event.decision.rights[token_idx]),
        offset=float(event.decision.offsets[token_idx]),
        strategy=strategy,
    )


def _simulate_selector(
    events: Sequence[EntryEvent],
    predictions: np.ndarray,
    *,
    selector_model: ContractSelector,
    selector_scaler: FeatureScaler,
    config: SelectorConfig,
) -> tuple[list[Trade], list[dict]]:
    trades = []
    trace = []
    for event, pred in zip(events, predictions):
        token_idx, edge, changed = _choose_contract(
            event,
            pred,
            selector_model=selector_model,
            selector_scaler=selector_scaler,
            config=config,
        )
        trade = _trade_from_event(event, token_idx, edge=edge, strategy=f"{LOOP_ID}:{config.name}")
        trades.append(trade)
        trace.append(
            {
                "seed": event.seed,
                "effective_seed": event.effective_seed,
                "split": event.split,
                "session": trade.session,
                "decision_time": trade.decision_time,
                "right": trade.right,
                "baseline_right": event.side,
                "offset": trade.offset,
                "baseline_offset": float(event.decision.offsets[event.original_token_idx]),
                "pnl": trade.pnl,
                "baseline_pnl": event.baseline_pnl,
                "changed_contract": bool(changed),
                "pnl_delta": float(trade.pnl - event.baseline_pnl),
            }
        )
    return trades, trace


def _metrics(trades: Sequence[Trade]) -> dict:
    return metrics_with_concentration(trades)


def _summarize(rows: Sequence[dict], split_order: Sequence[str], metric_key: str) -> list[dict]:
    out = []
    for split in split_order:
        metrics = [row[metric_key] for row in rows if row["split"] == split]
        stress50 = [row["stress50"] for row in rows if row["split"] == split]
        stress100 = [row["stress100"] for row in rows if row["split"] == split]
        changed = [row["changed_fraction"] for row in rows if row["split"] == split]
        if not metrics:
            continue
        out.append(
            {
                "split": split,
                "pnl_median": float(np.median([m["total_pnl"] for m in metrics])),
                "pf_median": float(np.median([m["profit_factor"] for m in metrics])),
                "trades_median": float(np.median([m["trades"] for m in metrics])),
                "positive_seed_fraction": float(np.mean([m["total_pnl"] > 0 for m in metrics])),
                "stress50_pnl_median": float(np.median([m["total_pnl"] for m in stress50])),
                "stress100_pnl_median": float(np.median([m["total_pnl"] for m in stress100])),
                "changed_fraction_median": float(np.median(changed)) if changed else 0.0,
            }
        )
    return out


def _write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Protocol 037: Timing-Frozen Contract Selector",
        "",
        payload["framing"],
        "",
        f"Selected config: `{payload['selected_config']['name']}`",
        "",
        "| Split | Candidate PnL | PF | Trades | Changed | +50 PnL | Protocol 024 PnL | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        base = BASELINE[row["split"]]["pnl"]
        lines.append(
            f"| {row['split']} | {row['pnl_median']:.0f} | {row['pf_median']:.3f} | "
            f"{row['trades_median']:.0f} | {row['changed_fraction_median']:.2f} | "
            f"{row['stress50_pnl_median']:.0f} | {base:.0f} | {row['pnl_median'] - base:.0f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "This is not paper/live approval.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    variant = _find_variant(args.variant_name)
    trial = _find_trial(args.trial_name)
    policy_name, cooldown = POLICY_META[args.policy_index]
    decision_cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    market_cache = MarketStructureCache()
    train_paths = _paths_by_split(args.data_dir)
    seed_q4_dir = args.seed_q4_data_dir or args.q4_data_dir
    window = _protocol_window(train_paths, _paths(seed_q4_dir))
    train_decisions = _load_surface_decisions_cached(
        train_paths["train"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="train",
        cache_dir=decision_cache_dir,
    )
    validation_decisions = _load_surface_decisions_cached(
        train_paths["validation"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="validation",
        cache_dir=decision_cache_dir,
    )
    calibration_decisions, selection_decisions = split_validation_by_session(validation_decisions)
    decision_sets = {
        "train": train_decisions,
        "calibration": calibration_decisions,
        "selection": selection_decisions,
        "march_2026": _load_surface_decisions_cached(
            train_paths["test"],
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="march",
            cache_dir=decision_cache_dir,
        ),
        "q1_2025": _load_surface_decisions_cached(
            _paths(args.q1_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q1_2025",
            cache_dir=decision_cache_dir,
        ),
        "q2_2025": _load_surface_decisions_cached(
            _paths(args.q2_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q2_2025",
            cache_dir=decision_cache_dir,
        ),
        "q3_2025": _load_surface_decisions_cached(
            _paths(args.q3_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q3_2025",
            cache_dir=decision_cache_dir,
        ),
        "q4_2025": _load_surface_decisions_cached(
            _paths(args.q4_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q4_2025",
            cache_dir=decision_cache_dir,
        ),
    }
    grid = _selector_grid()
    config_scores: dict[str, list[float]] = {config.name: [] for config in grid}
    config_by_name = {config.name: config for config in grid}
    artifacts = []
    for seed in args.seeds:
        effective_seed = window_seed(seed, window.window_id)
        print(f"{LOOP_ID} seed={seed}", flush=True)
        config = PilotConfig(
            policy_index=args.policy_index,
            policy_name=policy_name,
            cooldown_minutes=cooldown,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=128,
            seed=effective_seed,
        )
        surface_model, standardizer, surface_history = train_surface_model(
            train_decisions,
            calibration_decisions,
            config=config,
            variant=variant,
        )
        predictions = {
            split: predict_surface_actions(surface_model, standardizer, decisions, target_scale=config.target_scale)
            for split, decisions in decision_sets.items()
        }
        events = {
            split: _select_entry_events(
                decisions,
                predictions[split],
                trial=trial,
                cooldown_minutes=cooldown,
                split=split,
                seed=seed,
                effective_seed=effective_seed,
            )
            for split, decisions in decision_sets.items()
        }
        selector_model, selector_scaler, selector_history = _fit_selector(
            events["train"],
            predictions["train"],
            events["calibration"],
            predictions["calibration"],
            seed=effective_seed,
            epochs=args.selector_epochs,
            batch_size=args.batch_size,
        )
        for selector_config in grid:
            trades, _trace = _simulate_selector(
                events["selection"],
                predictions["selection"],
                selector_model=selector_model,
                selector_scaler=selector_scaler,
                config=selector_config,
            )
            config_scores[selector_config.name].append(selection_reward(_metrics(trades)))
        artifacts.append(
            {
                "seed": seed,
                "effective_seed": effective_seed,
                "surface_history": surface_history,
                "selector_history": selector_history,
                "events": events,
                "predictions": predictions,
                "selector_model": selector_model,
                "selector_scaler": selector_scaler,
            }
        )
    selected_name = max(config_scores, key=lambda name: float(np.median(config_scores[name])))
    selected_config = config_by_name[selected_name]
    print(f"{LOOP_ID} selected {selected_config.name}", flush=True)
    split_order = ["selection", "march_2026", "q1_2025", "q2_2025", "q3_2025", "q4_2025"]
    seed_rows = []
    traces = []
    for artifact in artifacts:
        for split in split_order:
            trades, trace = _simulate_selector(
                artifact["events"][split],
                artifact["predictions"][split],
                selector_model=artifact["selector_model"],
                selector_scaler=artifact["selector_scaler"],
                config=selected_config,
            )
            traces.extend(trace)
            changed_fraction = float(np.mean([row["changed_contract"] for row in trace])) if trace else 0.0
            seed_rows.append(
                {
                    "seed": artifact["seed"],
                    "effective_seed": artifact["effective_seed"],
                    "split": split,
                    "metrics": _metrics(trades),
                    "stress50": _metrics(stress_trades(trades, extra_cost_per_trade=50.0)),
                    "stress100": _metrics(stress_trades(trades, extra_cost_per_trade=100.0)),
                    "changed_fraction": changed_fraction,
                    "random_baseline": summarize_random_baseline(
                        decision_sets[split],
                        trial=trial,
                        cooldown_minutes=cooldown,
                        seed=artifact["effective_seed"],
                        target_trade_count=len(trades),
                    ),
                }
            )
    summary = _summarize(seed_rows, split_order, "metrics")
    lookup = {row["split"]: row for row in summary}
    beats_baseline = all(
        lookup[split]["pnl_median"] > BASELINE[split]["pnl"]
        for split in split_order
    )
    survives = all(
        lookup[split]["pnl_median"] > 0.0
        and lookup[split]["stress50_pnl_median"] > 0.0
        and lookup[split]["positive_seed_fraction"] >= 2 / 3
        for split in ("march_2026", "q1_2025", "q2_2025", "q3_2025", "q4_2025")
    )
    march_safe = lookup["march_2026"]["pnl_median"] >= BASELINE["march_2026"]["pnl"]
    decision = "Reject Protocol 037 as a replacement for Protocol 024."
    if beats_baseline and survives and march_safe:
        decision = "Keep Protocol 037 as the new research baseline candidate; it preserves timing and beats Protocol 024 across locked splits."
    payload = {
        "loop_id": LOOP_ID,
        "framing": (
            "Freeze Protocol 024's entry timing and side, then train a same-side contract selector. "
            "The selector cannot skip, cannot flip side, and cannot change cooldown/trade count; it only changes strike/contract expression."
        ),
        "pre_registration": [
            "No paid data.",
            "Do not alter Protocol 024 entry variant, policy, trial, seeds, labels, or split protocol.",
            "Train selector on train entry events with calibration early stopping.",
            "Select one selector blend/candidate-band config on February selection only.",
            "Score March 2026 and Q1/Q2/Q3/Q4 2025 once with the selected config.",
        ],
        "baseline": BASELINE,
        "selected_config": asdict(selected_config),
        "grid": [asdict(config) for config in grid],
        "selection_scores": {
            name: {"seed_rewards": scores, "median_reward": float(np.median(scores))}
            for name, scores in sorted(config_scores.items())
        },
        "summary": summary,
        "seed_rows": seed_rows,
        "traces": traces,
        "beats_baseline": bool(beats_baseline),
        "survives": bool(survives),
        "march_safe": bool(march_safe),
        "decision": decision,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(args.out_dir / "report.json")
    print(args.out_dir / "report.md")
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
