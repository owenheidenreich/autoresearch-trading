"""Run Protocol 004: lower-variance A+ permission variants.

Protocol 003C proved that a learned entry-permission layer can improve March
selectivity, but it remained too thin for broad-data-purchase approval because
Q4 failed +$50/trade stress. Protocol 004 screens a fixed set of permission
objectives without changing the base A+ model, the split protocol, or the audit
windows.

Promotion gate for broad-data-purchase consideration:

* positive selection, March, and frozen Q4 medians
* March and Q4 beat matched random by median
* March and Q4 survive +$50/trade stress
* March and Q4 have at least 2/3 positive seeds
* reasonable activity: at least 8 median trades in selection/March/Q4

Passing this gate would justify a narrow request to buy more historical data for
validation. It would still not approve live trading.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    bootstrap_trade_pnl,
    predict_surface_actions,
    stress_trades,
    summarize_random_baseline,
    token_feature_names,
    train_surface_model,
    window_seed,
)
from v4.model.supervised_pilot import FeatureScaler, PilotConfig
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import (
    _load_surface_decisions_cached,
    _paths_by_split,
    _protocol_window,
)
from v4.scripts.run_aplus_permission_protocol import (
    LOOP_ID as PERMISSION_LOOP_ID,
    PERMISSION_TRIAL,
    THRESHOLD_GRID,
    PermissionExample,
    PermissionMLP,
    _proposal_examples,
    _selection_reward,
    _simulate_permission_policy,
    _summarize_split,
    _summarize_stress,
    _variant,
)
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_permission_protocol_004"


@dataclass(frozen=True)
class PermissionVariant:
    name: str
    target_mode: str
    cost: float
    top_k: int = 0
    weighted: bool = False

    @property
    def variant_id(self) -> str:
        return f"{self.target_mode}_cost{int(self.cost)}_top{self.top_k}_w{int(self.weighted)}"


def registered_permission_variants() -> tuple[PermissionVariant, ...]:
    """Fixed Protocol 004 variant set."""

    return (
        PermissionVariant("bce_cost25", "positive_after_cost", 25.0),
        PermissionVariant("bce_cost50", "positive_after_cost", 50.0),
        PermissionVariant("weighted_cost25", "positive_after_cost", 25.0, weighted=True),
        PermissionVariant("weighted_cost50", "positive_after_cost", 50.0, weighted=True),
        PermissionVariant("top2_cost25", "top_k_positive", 25.0, top_k=2),
        PermissionVariant("top3_cost25", "top_k_positive", 25.0, top_k=3),
        PermissionVariant("top2_cost50", "top_k_positive", 50.0, top_k=2),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025"))
    parser.add_argument(
        "--seed-q4-data-dir",
        type=Path,
        default=None,
        help=(
            "optional holdout directory used only to derive stable window-seeded "
            "training seeds; useful when auditing a fresh holdout without changing "
            "the trained model seed stream"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_permission_protocol_004"))
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions"))
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--permission-epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--permission-batch-size", type=int, default=2048)
    parser.add_argument("--variant-names", nargs="*", default=[])
    parser.add_argument(
        "--selection-extra-cost",
        type=float,
        default=0.0,
        help="extra per-trade cost used only when selecting the permission threshold on late-February",
    )
    return parser.parse_args()


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _targets_and_weights(
    examples: Sequence[PermissionExample],
    variant: PermissionVariant,
) -> tuple[np.ndarray, np.ndarray]:
    pnl = np.asarray([example.pnl for example in examples], dtype=np.float32)
    target = pnl > float(variant.cost)
    if variant.target_mode == "top_k_positive":
        target = np.zeros(len(examples), dtype=bool)
        by_session: dict[str, list[int]] = {}
        for idx, example in enumerate(examples):
            by_session.setdefault(example.session, []).append(idx)
        for indexes in by_session.values():
            ranked = sorted(indexes, key=lambda i: float(examples[i].pnl), reverse=True)
            for idx in ranked[: variant.top_k]:
                if float(examples[idx].pnl) > float(variant.cost):
                    target[idx] = True
    elif variant.target_mode != "positive_after_cost":
        raise ValueError(f"unknown permission target mode: {variant.target_mode}")

    if variant.weighted:
        distance = np.abs(pnl - float(variant.cost))
        weights = 0.75 + np.minimum(distance / 250.0, 4.0)
    else:
        weights = np.ones(len(examples), dtype=np.float32)
    if variant.target_mode == "top_k_positive":
        weights = np.where(target, weights * 2.0, weights).astype(np.float32)
    return target.astype(np.float32), weights.astype(np.float32)


def _stack_features(examples: Sequence[PermissionExample]) -> np.ndarray:
    return np.vstack([example.features for example in examples]).astype(np.float32)


def _train_permission_model_variant(
    train_examples: Sequence[PermissionExample],
    validation_examples: Sequence[PermissionExample],
    *,
    variant: PermissionVariant,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[PermissionMLP, FeatureScaler, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train_raw = _stack_features(train_examples)
    y_train, w_train = _targets_and_weights(train_examples, variant)
    x_val_raw = _stack_features(validation_examples)
    y_val, w_val = _targets_and_weights(validation_examples, variant)
    scaler = FeatureScaler.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    model = PermissionMLP(input_dim=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1.0))], dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
            torch.from_numpy(w_train),
        ),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
    )
    val_tensors = (
        torch.from_numpy(x_val),
        torch.from_numpy(y_val),
        torch.from_numpy(w_val),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss_raw = F.binary_cross_entropy_with_logits(
                logits,
                batch_y,
                pos_weight=pos_weight,
                reduction="none",
            )
            loss = (loss_raw * batch_w).sum() / batch_w.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensors[0])
            val_raw = F.binary_cross_entropy_with_logits(
                val_logits,
                val_tensors[1],
                pos_weight=pos_weight,
                reduction="none",
            )
            val_loss = float(((val_raw * val_tensors[2]).sum() / val_tensors[2].sum().clamp(min=1.0)).detach().cpu())
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "variant": variant.name,
                "train_bce": float(np.mean(losses)),
                "validation_bce": val_loss,
                "is_best": is_best,
                "train_positive_fraction": float(y_train.mean()) if len(y_train) else 0.0,
                "validation_positive_fraction": float(y_val.mean()) if len(y_val) else 0.0,
                "train_weight_mean": float(w_train.mean()) if len(w_train) else 0.0,
            }
        )
    model.load_state_dict(best_state)
    return model, scaler, history


def _choose_threshold_for_variant(
    *,
    decisions,
    predictions,
    model,
    scaler,
    cooldown_minutes: int,
    feature_names: Sequence[str],
    variant_name: str,
    selection_extra_cost: float,
) -> tuple[float, list[dict]]:
    sweep = []
    for threshold in THRESHOLD_GRID:
        trades, _ = _simulate_permission_policy(
            decisions,
            predictions,
            permission_model=model,
            permission_scaler=scaler,
            threshold=float(threshold),
            cooldown_minutes=cooldown_minutes,
            feature_names=feature_names,
            strategy=f"{LOOP_ID}:{variant_name}:threshold_sweep",
        )
        selection_trades = (
            stress_trades(trades, extra_cost_per_trade=selection_extra_cost)
            if selection_extra_cost > 0.0
            else trades
        )
        metrics = metrics_with_concentration(selection_trades)
        sweep.append(
            {
                "threshold": float(threshold),
                "metrics": metrics,
                "selection_reward": _selection_reward(metrics),
                "selection_extra_cost": float(selection_extra_cost),
            }
        )
    ranked = sorted(
        sweep,
        key=lambda row: (
            row["selection_reward"],
            row["metrics"]["total_pnl"],
            min(float(row["metrics"]["profit_factor"]), 5.0)
            if np.isfinite(row["metrics"]["profit_factor"])
            else 5.0,
            -abs(row["threshold"] - 0.55),
        ),
        reverse=True,
    )
    return float(ranked[0]["threshold"]), sweep


def _run_variant_for_seed(
    *,
    variant: PermissionVariant,
    seed: int,
    effective_seed: int,
    predictions: dict,
    decision_sets: dict,
    feature_names: Sequence[str],
    cooldown: int,
    train_examples: Sequence[PermissionExample],
    calibration_examples: Sequence[PermissionExample],
    permission_epochs: int,
    permission_batch_size: int,
    selection_extra_cost: float,
) -> dict:
    model, scaler, history = _train_permission_model_variant(
        train_examples,
        calibration_examples,
        variant=variant,
        seed=effective_seed,
        epochs=permission_epochs,
        batch_size=permission_batch_size,
    )
    threshold, sweep = _choose_threshold_for_variant(
        decisions=decision_sets["selection"],
        predictions=predictions["selection"],
        model=model,
        scaler=scaler,
        cooldown_minutes=cooldown,
        feature_names=feature_names,
        variant_name=variant.name,
        selection_extra_cost=selection_extra_cost,
    )
    metrics_by_split = {}
    random_baseline_by_split = {}
    slippage_stress_by_split = {}
    bootstrap_by_split = {}
    selected_trades = {"selection": [], "march": [], "q4": []}
    for split in ("selection", "march", "q4"):
        trades, rows = _simulate_permission_policy(
            decision_sets[split],
            predictions[split],
            permission_model=model,
            permission_scaler=scaler,
            threshold=threshold,
            cooldown_minutes=cooldown,
            feature_names=feature_names,
            strategy=f"{LOOP_ID}:{variant.name}:threshold_{threshold:.2f}",
        )
        for row in rows:
            selected_trades[split].append({"seed": int(seed), "permission_variant": variant.name, **row})
        metrics_by_split[split] = metrics_with_concentration(trades)
        random_baseline_by_split[split] = summarize_random_baseline(
            decision_sets[split],
            trial=PERMISSION_TRIAL,
            cooldown_minutes=cooldown,
            seed=effective_seed,
            target_trade_count=len(trades),
        )
        if split in {"march", "q4"}:
            bootstrap_by_split[split] = bootstrap_trade_pnl(trades, seed=effective_seed)
            slippage_stress_by_split[split] = {
                str(extra_cost): metrics_with_concentration(
                    stress_trades(trades, extra_cost_per_trade=float(extra_cost))
                )
                for extra_cost in (25, 50, 100)
            }
    return {
        "seed": int(seed),
        "effective_seed": int(effective_seed),
        "permission_variant": asdict(variant) | {"variant_id": variant.variant_id},
        "chosen_threshold": float(threshold),
        "permission_best_epoch": next((x["epoch"] for x in history if x["is_best"]), None),
        "permission_history": history,
        "threshold_sweep": sweep,
        "metrics_by_split": metrics_by_split,
        "random_baseline_by_split": random_baseline_by_split,
        "slippage_stress_by_split": slippage_stress_by_split,
        "bootstrap_by_split": bootstrap_by_split,
        "selected_trades": selected_trades,
    }


def _combo_summary(rows: Sequence[dict], variant_name: str) -> dict:
    group = [row for row in rows if row["permission_variant"]["name"] == variant_name]
    summary = {"permission_variant": variant_name, "runs": len(group)}
    for split in ("selection", "march", "q4"):
        split_summary = _summarize_split(group, split)
        for key, value in split_summary.items():
            if isinstance(value, (int, float, bool)):
                summary[f"{split}_{key}"] = value
    stress = {split: _summarize_stress(group, split) for split in ("march", "q4")}
    summary["stress"] = stress
    summary["passes_broad_data_purchase_gate"] = bool(
        summary.get("selection_pnl_median", 0.0) > 0.0
        and summary.get("selection_trades_median", 0.0) >= 8.0
        and summary.get("march_pnl_median", 0.0) > 0.0
        and summary.get("march_profit_factor_median", 0.0) >= 1.05
        and summary.get("march_positive_seed_fraction", 0.0) >= 2 / 3
        and summary.get("march_trades_median", 0.0) >= 8.0
        and summary.get("march_edge_vs_random_median", 0.0) > 0.0
        and summary.get("q4_pnl_median", 0.0) > 0.0
        and summary.get("q4_profit_factor_median", 0.0) >= 1.05
        and summary.get("q4_positive_seed_fraction", 0.0) >= 2 / 3
        and summary.get("q4_trades_median", 0.0) >= 8.0
        and summary.get("q4_edge_vs_random_median", 0.0) > 0.0
        and stress["march"]["50"]["survives"]
        and stress["q4"]["50"]["survives"]
    )
    summary["purchase_gate_score"] = (
        summary.get("march_pnl_median", 0.0)
        + summary.get("q4_pnl_median", 0.0)
        + 1000.0 * (summary.get("march_profit_factor_median", 0.0) - 1.0)
        + 1000.0 * (summary.get("q4_profit_factor_median", 0.0) - 1.0)
        + stress["march"]["50"]["pnl_median"]
        + stress["q4"]["50"]["pnl_median"]
    )
    return summary


def _aggregate(rows: Sequence[dict], variants: Sequence[PermissionVariant]) -> dict:
    summaries = [_combo_summary(rows, variant.name) for variant in variants]
    ranked = sorted(
        summaries,
        key=lambda row: (
            row["passes_broad_data_purchase_gate"],
            row["purchase_gate_score"],
            row.get("q4_pnl_median", 0.0),
            row.get("march_pnl_median", 0.0),
        ),
        reverse=True,
    )
    return {
        "broad_data_purchase_pass_count": int(sum(row["passes_broad_data_purchase_gate"] for row in summaries)),
        "champion": ranked[0] if ranked else None,
        "ranked": ranked,
        "by_variant": {row["permission_variant"]: row for row in summaries},
    }


def _write_markdown(path: Path, payload: dict) -> None:
    aggregate = payload["aggregate"]
    champion = aggregate["champion"]
    lines = [
        "# A+ Permission Protocol 004",
        "",
        payload["framing"],
        "",
        f"Broad-data-purchase pass count: `{aggregate['broad_data_purchase_pass_count']}`",
        "",
        "## Champion",
        "",
    ]
    if champion is None:
        lines.append("No champion.")
    else:
        lines += [
            f"Best variant: `{champion['permission_variant']}`",
            f"Passes broad-data-purchase gate: `{champion['passes_broad_data_purchase_gate']}`",
            "",
            "| Split | Median PnL | Median PF | Trades | Positive Seeds | Edge vs Random |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for split in ("selection", "march", "q4"):
            lines.append(
                f"| {split} | {champion.get(split + '_pnl_median', 0.0):.0f} | "
                f"{champion.get(split + '_profit_factor_median', 0.0):.3f} | "
                f"{champion.get(split + '_trades_median', 0.0):.0f} | "
                f"{champion.get(split + '_positive_seed_fraction', 0.0):.2f} | "
                f"{champion.get(split + '_edge_vs_random_median', 0.0):.0f} |"
            )
        lines += [
            "",
            "### Champion Stress",
            "",
            "| Split | +25 PnL/PF | +50 PnL/PF | +100 PnL/PF |",
            "|---|---:|---:|---:|",
        ]
        for split in ("march", "q4"):
            stress = champion["stress"][split]
            lines.append(
                f"| {split} | {stress['25']['pnl_median']:.0f}/{stress['25']['profit_factor_median']:.3f} | "
                f"{stress['50']['pnl_median']:.0f}/{stress['50']['profit_factor_median']:.3f} | "
                f"{stress['100']['pnl_median']:.0f}/{stress['100']['profit_factor_median']:.3f} |"
            )
    lines += [
        "",
        "## Ranked Variants",
        "",
        "| Rank | Variant | Pass | Score | Sel PnL | March PnL/PF | Q4 PnL/PF | March +50 | Q4 +50 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(aggregate["ranked"], start=1):
        lines.append(
            f"| {idx} | {row['permission_variant']} | {row['passes_broad_data_purchase_gate']} | "
            f"{row['purchase_gate_score']:.0f} | {row.get('selection_pnl_median', 0.0):.0f} | "
            f"{row.get('march_pnl_median', 0.0):.0f}/{row.get('march_profit_factor_median', 0.0):.3f} | "
            f"{row.get('q4_pnl_median', 0.0):.0f}/{row.get('q4_profit_factor_median', 0.0):.3f} | "
            f"{row['stress']['march']['50']['pnl_median']:.0f} | "
            f"{row['stress']['q4']['50']['pnl_median']:.0f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    variants = list(registered_permission_variants())
    if args.variant_names:
        requested = set(args.variant_names)
        variants = [variant for variant in variants if variant.name in requested]
        missing = requested - {variant.name for variant in variants}
        if missing:
            raise SystemExit(f"unknown Protocol 004 variant(s): {sorted(missing)}")
    paths = _paths_by_split(args.data_dir)
    q4_paths = sorted(args.q4_data_dir.glob("*.pkl"))
    if not q4_paths:
        raise SystemExit(f"no Q4 pkl files found under {args.q4_data_dir}")
    window = _protocol_window(paths, q4_paths)
    seed_window = window
    if args.seed_q4_data_dir is not None:
        seed_q4_paths = sorted(args.seed_q4_data_dir.glob("*.pkl"))
        if not seed_q4_paths:
            raise SystemExit(f"no seed Q4 pkl files found under {args.seed_q4_data_dir}")
        seed_window = _protocol_window(paths, seed_q4_paths)
    surface_variant = _variant()
    policy_name, cooldown = POLICY_META[args.policy_index]
    market_cache = MarketStructureCache()
    cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    feature_names = token_feature_names(surface_variant.token_mode)

    train_decisions = _load_surface_decisions_cached(paths["train"], policy_index=args.policy_index, variant=surface_variant, market_cache=market_cache, split="train", cache_dir=cache_dir)
    validation_decisions = _load_surface_decisions_cached(paths["validation"], policy_index=args.policy_index, variant=surface_variant, market_cache=market_cache, split="validation", cache_dir=cache_dir)
    calibration_decisions, selection_decisions = split_validation_by_session(validation_decisions)
    march_decisions = _load_surface_decisions_cached(paths["test"], policy_index=args.policy_index, variant=surface_variant, market_cache=market_cache, split="march", cache_dir=cache_dir)
    q4_decisions = _load_surface_decisions_cached(q4_paths, policy_index=args.policy_index, variant=surface_variant, market_cache=market_cache, split="q4", cache_dir=cache_dir)
    decision_sets = {
        "train": train_decisions,
        "calibration": calibration_decisions,
        "selection": selection_decisions,
        "march": march_decisions,
        "q4": q4_decisions,
    }

    rows = []
    selected_trades = {variant.name: {"selection": [], "march": [], "q4": []} for variant in variants}
    for seed in args.seeds:
        effective_seed = window_seed(seed, seed_window.window_id)
        print(f"{LOOP_ID} base seed={seed}", flush=True)
        config = PilotConfig(
            policy_index=args.policy_index,
            policy_name=policy_name,
            cooldown_minutes=cooldown,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=128,
            seed=effective_seed,
        )
        base_model, standardizer, base_history = train_surface_model(
            train_decisions,
            calibration_decisions,
            config=config,
            variant=surface_variant,
        )
        predictions = {
            split: predict_surface_actions(base_model, standardizer, decisions, target_scale=config.target_scale)
            for split, decisions in decision_sets.items()
        }
        train_examples = _proposal_examples(train_decisions, predictions["train"], feature_names=feature_names)
        calibration_examples = _proposal_examples(calibration_decisions, predictions["calibration"], feature_names=feature_names)
        for variant in variants:
            print(f"{LOOP_ID} permission variant={variant.name} seed={seed}", flush=True)
            row = _run_variant_for_seed(
                variant=variant,
                seed=seed,
                effective_seed=effective_seed,
                predictions=predictions,
                decision_sets=decision_sets,
                feature_names=feature_names,
                cooldown=cooldown,
                train_examples=train_examples,
                calibration_examples=calibration_examples,
                permission_epochs=args.permission_epochs,
            permission_batch_size=args.permission_batch_size,
            selection_extra_cost=args.selection_extra_cost,
        )
            row["base_best_epoch"] = next((x["epoch"] for x in base_history if x["is_best"]), None)
            row["proposal_counts"] = {"train": len(train_examples), "calibration": len(calibration_examples)}
            rows.append(row)
            for split, trade_rows in row["selected_trades"].items():
                selected_trades[variant.name][split].extend(trade_rows)
            row.pop("selected_trades", None)
    aggregate = _aggregate(rows, variants)
    payload = {
        "loop_id": LOOP_ID,
        "framing": (
            "Fixed Protocol 004 permission-objective screen for broad-data-purchase consideration. "
            "All variants use the same A+ base model and the same March/Q4 frozen audits."
        ),
        "window": asdict(window) | {"window_id": window.window_id},
        "seed_window": asdict(seed_window) | {"window_id": seed_window.window_id},
        "surface_variant": asdict(surface_variant) | {"variant_id": surface_variant.variant_id},
        "policy_index": int(args.policy_index),
        "policy_name": policy_name,
        "permission_trial": asdict(PERMISSION_TRIAL) | {"config_id": PERMISSION_TRIAL.config_id},
        "promotion_gate": [
            "positive selection, March, and frozen Q4 medians",
            "March and Q4 beat matched random by median",
            "March and Q4 survive +$50/trade stress",
            "March and Q4 have at least 2/3 positive seeds",
            "at least 8 median trades in selection, March, and Q4",
        ],
        "args": {
            "data_dir": str(args.data_dir),
            "q4_data_dir": str(args.q4_data_dir),
            "seed_q4_data_dir": str(args.seed_q4_data_dir) if args.seed_q4_data_dir is not None else None,
            "out_dir": str(args.out_dir),
            "seeds": args.seeds,
            "epochs": args.epochs,
            "permission_epochs": args.permission_epochs,
            "batch_size": args.batch_size,
            "permission_batch_size": args.permission_batch_size,
            "variant_names": args.variant_names,
            "selection_extra_cost": args.selection_extra_cost,
        },
        "registered_permission_variants": [asdict(variant) | {"variant_id": variant.variant_id} for variant in variants],
        "aggregate": aggregate,
        "rows": rows,
        "interpretation": (
            "A passing variant is broad-data-purchase approval for validation data only, not live trading. "
            "A failing screen means keep researching on the current data."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(md_path, payload)
    for variant_name, split_map in selected_trades.items():
        variant_dir = args.out_dir / "selected_trades" / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for split, trade_rows in split_map.items():
            (variant_dir / f"selected_trades_{split}.json").write_text(
                json.dumps(trade_rows, indent=2, allow_nan=True) + "\n"
            )
    print(json_path)
    print(md_path)
    print(json.dumps(aggregate["champion"], indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
