"""Protocol 052: walk-forward lifecycle layer on frozen Protocol 051 entries.

The entry model is frozen to Protocol 051:

    surface_structure_aplus_side_value_rank / policy1 / post_open_late_edge25_max2

This script trains a separate causal lifecycle model that only acts after an
entry exists. It sees position state and option state through time, then decides
whether to keep holding or exit early. Hard stop, target, and flat-before-close
behavior remain mandatory.

No paid data is downloaded.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    ProtocolTrial,
    SurfaceDecision,
    SurfaceVariant,
    predict_surface_actions,
    registered_aplus_surface_variants,
    registered_protocol_trials,
    stress_trades,
    train_surface_model,
    window_seed,
)
from v4.model.supervised_pilot import PilotConfig, Trade, session_from_path
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import _load_surface_decisions_cached
from v4.scripts.run_sequential_risk_protocol import (
    _CONTRACT_MULTIPLIER,
    _FEATURE_INDEX,
    _POLICY,
    _RISK_TARGET_SCALE,
    RISK_FEATURE_NAMES,
    EntryProposal,
    PathPoint,
    RiskConfig,
    _causal_state_features,
    _constraint_allows_model_exit,
    _contract_path,
    _deadline,
    _fit_risk_model,
    _load_session_rows,
    _metrics,
    _option_entry_features,
    _path_samples,
    _predict_headroom,
    _risk_grid,
    _select_entry_proposals,
    _selection_reward,
    _trade_from_entry,
    _training_candidate_proposals,
)
from v4.scripts.run_soft_quality_walkforward_protocol import FoldSpec, _folds
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_hypothesis_052_protocol051_sequential_lifecycle_walkforward"
DEFAULT_VARIANT = "surface_structure_aplus_side_value_rank"
DEFAULT_TRIAL = "post_open_late_edge25_max2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q4-2024-dir", type=Path, default=None)
    parser.add_argument("--q1-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025_official_context"))
    parser.add_argument("--q2-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025_official_context"))
    parser.add_argument("--q3-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025_official_context"))
    parser.add_argument("--q4-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025_official_context"))
    parser.add_argument("--q1-2026-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2026_official_context"))
    parser.add_argument("--out-dir", type=Path, default=Path(f"v4/audit/autoresearch/{LOOP_ID}"))
    parser.add_argument("--loop-id", default=LOOP_ID)
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions_official_context"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized_official_context"))
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--variant-name", default=DEFAULT_VARIANT)
    parser.add_argument("--trial-name", default=DEFAULT_TRIAL)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--risk-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-risk-teacher-entries", type=int, default=4000)
    parser.add_argument("--exit-constraint", choices=["loss_or_giveback", "loss_only", "unconstrained"], default="loss_or_giveback")
    parser.add_argument("--market-structure-source", choices=("v2_cache", "index_bars"), default="index_bars")
    parser.add_argument("--market-spx-dir", type=Path, default=Path("data/vendor/thetadata/index/spx_1m"))
    parser.add_argument("--market-vix-dir", type=Path, default=Path("data/vendor/thetadata/index/vix_1m"))
    parser.add_argument("--es-vwap-dir", type=Path, default=None)
    parser.add_argument(
        "--save-live-artifacts",
        action="store_true",
        help="Persist entry and Protocol 054 fallback risk-model artifacts for each fold/seed.",
    )
    parser.add_argument(
        "--force-selected-configs-json",
        type=Path,
        default=None,
        help="Optional report/config JSON containing selected_configs by split. Used to persist an already-frozen protocol.",
    )
    return parser.parse_args()


def _find_variant(name: str) -> SurfaceVariant:
    for variant in registered_aplus_surface_variants():
        if variant.name == name:
            return variant
    raise SystemExit(f"unknown A+ variant: {name}")


def _find_trial(name: str) -> ProtocolTrial:
    for trial in registered_protocol_trials():
        if trial.name == name:
            return trial
    raise SystemExit(f"unknown trial: {name}")


def _forced_selected_configs(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if "selected_configs" in payload:
        payload = payload["selected_configs"]
    if not isinstance(payload, dict):
        raise SystemExit(f"forced selected config file must contain a split->config map: {path}")
    return {str(split): str(config) for split, config in payload.items()}


def _bounded_entries(entries: Sequence[EntryProposal], *, limit: int, seed: int) -> list[EntryProposal]:
    entries = list(entries)
    if limit <= 0 or len(entries) <= limit:
        return entries
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(entries), size=limit, replace=False))
    return [entries[int(i)] for i in idx]


def _normalized_session_path(normalized_dir: Path, session: str) -> Path | None:
    preferred = sorted(normalized_dir.glob(f"*{session}*official_context.parquet"))
    if preferred:
        return preferred[0]
    fallback = sorted(normalized_dir.glob(f"*{session}*.parquet"))
    return fallback[0] if fallback else None


def _normalized_option_features(row: pd.Series) -> np.ndarray:
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    mid = row.get("mid", np.nan)
    if not np.isfinite(mid):
        mid = (bid + ask) / 2.0
    spread = ask - bid if np.isfinite(bid) and np.isfinite(ask) else np.nan
    spread_frac = spread / mid if np.isfinite(spread) and np.isfinite(mid) and mid > 0 else np.nan
    return np.asarray(
        [
            bid,
            ask,
            mid,
            spread,
            spread_frac,
            float(row.get("bid_size", np.nan)),
            float(row.get("ask_size", np.nan)),
            float(row.get("option_ohlcv_volume", np.nan)),
            float(row.get("stat_open_interest", np.nan)),
            float(row.get("iv", np.nan)),
            float(row.get("delta", np.nan)),
            float(row.get("gamma", np.nan)),
            float(row.get("theta", np.nan)),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


class NormalizedPathStore:
    def __init__(self, normalized_dir: Path) -> None:
        self.normalized_dir = normalized_dir
        self._sessions: dict[str, pd.DataFrame | None] = {}

    def _load_session(self, session: str) -> pd.DataFrame | None:
        if session in self._sessions:
            return self._sessions[session]
        path = _normalized_session_path(self.normalized_dir, session)
        if path is None:
            self._sessions[session] = None
            return None
        columns = [
            "quote_time",
            "event_time",
            "contract_id",
            "bid",
            "ask",
            "mid",
            "bid_size",
            "ask_size",
            "option_ohlcv_volume",
            "stat_open_interest",
            "iv",
            "delta",
            "gamma",
            "theta",
        ]
        frame = pd.read_parquet(path, columns=columns)
        frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True)
        frame["contract_id_str"] = frame["contract_id"].astype(str)
        frame = frame.sort_values(["contract_id", "quote_time"]).reset_index(drop=True)
        self._sessions[session] = frame
        return frame

    def contract_path(self, entry: EntryProposal) -> list[PathPoint]:
        frame = self._load_session(entry.session)
        if frame is None or frame.empty:
            return []
        deadline = _deadline(entry.decision_time)
        contract_id = str(entry.contract_id)
        entry_ask = float(_option_entry_features(entry)[_FEATURE_INDEX["ask"]])
        rows = frame[
            (frame["contract_id_str"] == contract_id)
            & (frame["quote_time"] > entry.decision_time)
            & (frame["quote_time"] <= deadline)
        ].sort_values("quote_time")
        points: list[PathPoint] = []
        for _, row in rows.iterrows():
            bid = row.get("bid")
            if not np.isfinite(float(bid)):
                continue
            points.append(
                PathPoint(
                    time=row["quote_time"].to_pydatetime(),
                    features=_normalized_option_features(row),
                    pnl=(float(bid) - entry_ask) * _CONTRACT_MULTIPLIER,
                )
            )
        return points


def _split_name(fold: FoldSpec) -> str:
    return fold.name.split("_test_", 1)[-1]


def _trade_metrics(trades: Sequence[Trade]) -> dict:
    metrics = metrics_with_concentration(trades)
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.generic)):
            value = float(value)
            if np.isfinite(value):
                out[key] = value
            elif value > 0:
                out[key] = 999.0
            elif value < 0:
                out[key] = -999.0
            else:
                out[key] = 0.0
        else:
            out[key] = value
    return out


def _save_live_artifact(
    *,
    out_dir: Path,
    fold_name: str,
    split: str,
    seed: int,
    effective_seed: int,
    entry_model,
    entry_standardizer,
    entry_config: PilotConfig,
    entry_history: list[dict],
    risk_model,
    risk_scaler,
    risk_history: list[dict],
    selected_config: RiskConfig,
    variant: SurfaceVariant,
    trial: ProtocolTrial,
    args: argparse.Namespace,
    risk_train_samples: int,
    risk_validation_samples: int,
    train_entries: int,
    validation_entries: int,
    test_entries: int,
) -> dict:
    artifact_dir = out_dir / "model_artifacts" / fold_name / f"seed_{seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entry_model_path = artifact_dir / "entry_model.pt"
    entry_standardizer_path = artifact_dir / "entry_standardizer.json"
    risk_model_path = artifact_dir / "protocol054_risk_model.pt"
    risk_scaler_path = artifact_dir / "protocol054_risk_scaler.json"
    manifest_path = artifact_dir / "manifest.json"

    torch.save(
        {
            "state_dict": entry_model.state_dict(),
            "model_class": "v4.model.hypothesis_protocol.SurfaceActionModel",
            "scalar_dim": int(len(entry_standardizer.scalar.fill)),
            "token_dim": int(len(entry_standardizer.token.fill)),
            "hidden_dim": int(max(entry_config.hidden_dim, 128)),
            "target_scale": float(entry_config.target_scale),
            "policy_index": int(entry_config.policy_index),
            "variant_name": variant.name,
        },
        entry_model_path,
    )
    entry_standardizer_path.write_text(json.dumps(entry_standardizer.to_dict(), indent=2, allow_nan=False) + "\n")
    torch.save(
        {
            "state_dict": risk_model.state_dict(),
            "model_class": "v4.scripts.run_sequential_risk_protocol.RiskHeadroomModel",
            "input_dim": int(len(RISK_FEATURE_NAMES)),
            "hidden_dim": 64,
            "target_scale": float(_RISK_TARGET_SCALE),
            "risk_feature_names": list(RISK_FEATURE_NAMES),
        },
        risk_model_path,
    )
    risk_scaler_path.write_text(json.dumps(risk_scaler.to_dict(), indent=2, allow_nan=False) + "\n")
    manifest = {
        "artifact_type": "protocol054_modular_entry_and_fallback_stack",
        "fold": fold_name,
        "split": split,
        "seed": int(seed),
        "effective_seed": int(effective_seed),
        "entry_protocol": "Protocol 051",
        "fallback_protocol": "Protocol 054",
        "variant_name": variant.name,
        "trial_name": trial.name,
        "policy_index": int(entry_config.policy_index),
        "entry_config": asdict(entry_config),
        "selected_risk_config": asdict(selected_config),
        "market_structure_source": args.market_structure_source,
        "normalized_dir": str(args.normalized_dir),
        "risk_feature_names": list(RISK_FEATURE_NAMES),
        "entry_history": entry_history,
        "risk_history": risk_history,
        "risk_train_samples": int(risk_train_samples),
        "risk_validation_samples": int(risk_validation_samples),
        "train_entries": int(train_entries),
        "validation_entries": int(validation_entries),
        "test_entries": int(test_entries),
        "files": {
            "entry_model": str(entry_model_path),
            "entry_standardizer": str(entry_standardizer_path),
            "protocol054_risk_model": str(risk_model_path),
            "protocol054_risk_scaler": str(risk_scaler_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {
        "artifact_dir": str(artifact_dir),
        "entry_model_path": str(entry_model_path),
        "entry_standardizer_path": str(entry_standardizer_path),
        "protocol054_risk_model_path": str(risk_model_path),
        "protocol054_risk_scaler_path": str(risk_scaler_path),
        "manifest_path": str(manifest_path),
    }


def _simulate_lifecycle_exit(
    entry: EntryProposal,
    path: Sequence[PathPoint],
    *,
    model,
    scaler,
    config: RiskConfig,
    loop_id: str,
) -> tuple[Trade, dict]:
    if not path:
        trade = _trade_from_entry(entry, entry.baseline_pnl, f"{loop_id}:path_missing")
        return trade, {
            "exit_reason": "path_missing",
            "hold_minutes": None,
            "predicted_headroom": None,
            "blocked_model_exits": 0,
        }

    entry_features = _option_entry_features(entry)
    entry_ask = float(entry_features[_FEATURE_INDEX["ask"]])
    stop_pnl = -_POLICY.stop_loss_pct * entry_ask * _CONTRACT_MULTIPLIER
    target_pnl = _POLICY.take_profit_pct * entry_ask * _CONTRACT_MULTIPLIER
    exit_point = path[-1]
    reason = "time_flat"
    predicted_headroom = None
    blocked_model_exits = 0
    constraint_details: dict = {}

    for idx, point in enumerate(path):
        hold_minutes = max(1.0, (point.time - entry.decision_time).total_seconds() / 60.0)
        if point.pnl <= stop_pnl:
            exit_point = point
            reason = "hard_stop"
            break
        if point.pnl >= target_pnl:
            exit_point = point
            reason = "target"
            break
        if hold_minutes < config.min_hold_minutes:
            continue
        x = _causal_state_features(entry, path, idx)
        predicted_headroom = _predict_headroom(model, scaler, x)
        if predicted_headroom <= config.exit_headroom_threshold:
            allowed, constraint_reason, details = _constraint_allows_model_exit(path, idx, config)
            constraint_details = details
            if allowed:
                exit_point = point
                reason = constraint_reason
                break
            blocked_model_exits += 1

    trade = _trade_from_entry(entry, exit_point.pnl, f"{loop_id}:{config.name}:{reason}")
    return trade, {
        "exit_reason": reason,
        "exit_time": exit_point.time.isoformat(),
        "hold_minutes": float((exit_point.time - entry.decision_time).total_seconds() / 60.0),
        "predicted_headroom": predicted_headroom,
        "blocked_model_exits": int(blocked_model_exits),
        **constraint_details,
    }


def _samples_for(
    entries: Sequence[EntryProposal],
    *,
    path_for,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for entry in entries:
        x, y = _path_samples(entry, path_for(entry))
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        return (
            np.empty((0, len(RISK_FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def _summary_by_split(rows: Sequence[dict], *, splits: Sequence[str]) -> list[dict]:
    out = []
    for split in splits:
        group = [row for row in rows if row["split"] == split]
        if not group:
            continue
        metrics = [row["metrics"] for row in group]
        stress50 = [row["stress50_metrics"] for row in group]
        stress100 = [row["stress100_metrics"] for row in group]
        out.append(
            {
                "split": split,
                "pnl_median": float(np.median([m["total_pnl"] for m in metrics])),
                "pf_median": float(np.median([m["profit_factor"] for m in metrics])),
                "trades_median": float(np.median([m["trades"] for m in metrics])),
                "positive_seed_fraction": float(np.mean([m["total_pnl"] > 0.0 for m in metrics])),
                "stress50_pnl_median": float(np.median([m["total_pnl"] for m in stress50])),
                "stress100_pnl_median": float(np.median([m["total_pnl"] for m in stress100])),
                "stress50_positive_seed_fraction": float(np.mean([m["total_pnl"] > 0.0 for m in stress50])),
            }
        )
    return out


def _write_report(path: Path, payload: dict) -> None:
    baseline = {row["split"]: row for row in payload["baseline_summary"]}
    lines = [
        f"# {payload['loop_id']}",
        "",
        "No paid data was downloaded. Protocol 051 is frozen as the entry baseline; only a post-entry lifecycle layer is trained.",
        "",
        "## Frozen Entry",
        "",
        f"- Variant: `{payload['entry']['variant_name']}`",
        f"- Policy/trial: `policy{payload['entry']['policy_index']} / {payload['entry']['trial_name']}`",
        f"- Exit constraint: `{payload['exit_constraint']}`",
        f"- Seeds: `{payload['seeds']}`",
        "",
        "## Walk-Forward Summary",
        "",
        "| Split | Dynamic PnL | Baseline PnL | Delta | Dynamic PF | Trades | +50 | +100 | Selected Config |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["dynamic_summary"]:
        base = baseline.get(row["split"], {})
        base_pnl = float(base.get("pnl_median", 0.0))
        config = payload["selected_configs"].get(row["split"], "")
        lines.append(
            f"| {row['split']} | {row['pnl_median']:.0f} | {base_pnl:.0f} | "
            f"{row['pnl_median'] - base_pnl:.0f} | {row['pf_median']:.3f} | "
            f"{row['trades_median']:.0f} | {row['stress50_pnl_median']:.0f} | "
            f"{row['stress100_pnl_median']:.0f} | `{config}` |"
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


def _run_fold(
    *,
    fold: FoldSpec,
    variant: SurfaceVariant,
    trial: ProtocolTrial,
    market_cache: MarketStructureCache,
    decision_cache_dir: Path | None,
    normalized_store: NormalizedPathStore | None,
    args: argparse.Namespace,
) -> dict:
    policy_name, cooldown = POLICY_META[args.policy_index]
    split = _split_name(fold)
    session_rows = _load_session_rows(
        {
            "train": fold.train_paths,
            "validation": fold.validation_paths,
            "test": fold.test_paths,
        }
    )
    train_decisions = _load_surface_decisions_cached(
        fold.train_paths,
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split=f"{fold.name}_train",
        cache_dir=decision_cache_dir,
    )
    validation_decisions = _load_surface_decisions_cached(
        fold.validation_paths,
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split=f"{fold.name}_validation",
        cache_dir=decision_cache_dir,
    )
    test_decisions = _load_surface_decisions_cached(
        fold.test_paths,
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split=f"{fold.name}_test",
        cache_dir=decision_cache_dir,
    )
    risk_grid = _risk_grid(args.exit_constraint)
    config_scores = {config.name: [] for config in risk_grid}
    config_by_name = {config.name: config for config in risk_grid}
    seed_artifacts = []

    for seed in args.seeds:
        effective_seed = window_seed(seed, fold.window_id)
        print(f"{args.loop_id} fold={fold.name} seed={seed}", flush=True)
        entry_config = PilotConfig(
            policy_index=args.policy_index,
            policy_name=policy_name,
            cooldown_minutes=cooldown,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=128,
            seed=effective_seed,
        )
        entry_model, entry_standardizer, entry_history = train_surface_model(
            train_decisions,
            validation_decisions,
            config=entry_config,
            variant=variant,
        )

        predictions_by_split = {
            "train": predict_surface_actions(
                entry_model,
                entry_standardizer,
                train_decisions,
                target_scale=entry_config.target_scale,
            ),
            "validation": predict_surface_actions(
                entry_model,
                entry_standardizer,
                validation_decisions,
                target_scale=entry_config.target_scale,
            ),
            "test": predict_surface_actions(
                entry_model,
                entry_standardizer,
                test_decisions,
                target_scale=entry_config.target_scale,
            ),
        }
        proposals_by_split = {
            "train": _select_entry_proposals(
                train_decisions,
                predictions_by_split["train"],
                trial=trial,
                cooldown_minutes=cooldown,
                split="train",
                seed=seed,
                effective_seed=effective_seed,
            ),
            "validation": _select_entry_proposals(
                validation_decisions,
                predictions_by_split["validation"],
                trial=trial,
                cooldown_minutes=cooldown,
                split="validation",
                seed=seed,
                effective_seed=effective_seed,
            ),
            "test": _select_entry_proposals(
                test_decisions,
                predictions_by_split["test"],
                trial=trial,
                cooldown_minutes=cooldown,
                split=split,
                seed=seed,
                effective_seed=effective_seed,
            ),
        }

        path_cache: dict[tuple[str, str, object], list[PathPoint]] = {}

        def path_for(entry: EntryProposal) -> list[PathPoint]:
            key = (entry.session, entry.decision_time.isoformat(), entry.contract_id)
            if key not in path_cache:
                if normalized_store is not None:
                    path_cache[key] = normalized_store.contract_path(entry)
                else:
                    path_cache[key] = _contract_path(entry, session_rows.get(entry.session, []))
            return path_cache[key]

        train_teacher = _bounded_entries(
            _training_candidate_proposals(
                train_decisions,
                trial=trial,
                split="train_teacher_candidates",
                seed=seed,
                effective_seed=effective_seed,
            ),
            limit=args.max_risk_teacher_entries,
            seed=effective_seed,
        )
        validation_teacher = _bounded_entries(
            _training_candidate_proposals(
                validation_decisions,
                trial=trial,
                split="validation_teacher_candidates",
                seed=seed,
                effective_seed=effective_seed,
            ),
            limit=max(250, args.max_risk_teacher_entries // 4),
            seed=effective_seed + 1,
        )
        train_x, train_y = _samples_for(
            list(proposals_by_split["train"]) + train_teacher,
            path_for=path_for,
        )
        val_x, val_y = _samples_for(
            list(proposals_by_split["validation"]) + validation_teacher,
            path_for=path_for,
        )
        if len(train_x) == 0:
            raise SystemExit(f"{fold.name} seed {seed}: no lifecycle training samples")
        risk_model, risk_scaler, risk_history = _fit_risk_model(
            train_x,
            train_y,
            val_x,
            val_y,
            seed=effective_seed,
            epochs=args.risk_epochs,
            batch_size=args.batch_size,
        )

        validation_entries = proposals_by_split["validation"]
        per_config_validation = {}
        for risk_config in risk_grid:
            trades = [
                _simulate_lifecycle_exit(
                    entry,
                    path_for(entry),
                    model=risk_model,
                    scaler=risk_scaler,
                    config=risk_config,
                    loop_id=args.loop_id,
                )[0]
                for entry in validation_entries
            ]
            metrics = _metrics(trades)
            per_config_validation[risk_config.name] = metrics
            config_scores[risk_config.name].append(_selection_reward(metrics))
        seed_artifacts.append(
            {
                "seed": int(seed),
                "effective_seed": int(effective_seed),
                "entry_config": entry_config,
                "entry_model": entry_model,
                "entry_standardizer": entry_standardizer,
                "entry_history": entry_history,
                "risk_history": risk_history,
                "risk_train_samples": int(len(train_x)),
                "risk_validation_samples": int(len(val_x)),
                "train_entries": int(len(proposals_by_split["train"])),
                "validation_entries": int(len(proposals_by_split["validation"])),
                "test_entries": int(len(proposals_by_split["test"])),
                "risk_model": risk_model,
                "risk_scaler": risk_scaler,
                "path_for": path_for,
                "proposals_by_split": proposals_by_split,
                "per_config_validation": per_config_validation,
                "saved_live_artifact": None,
            }
        )

    forced_configs = _forced_selected_configs(args.force_selected_configs_json)
    forced_config_name = forced_configs.get(split)
    if forced_config_name:
        if forced_config_name not in config_by_name:
            raise SystemExit(f"forced config {forced_config_name!r} is not in the risk grid for split {split}")
        selected_config_name = forced_config_name
    else:
        selected_config_name = max(
            config_scores,
            key=lambda name: float(np.median(config_scores[name])) if config_scores[name] else -1e9,
        )
    selected_config = config_by_name[selected_config_name]
    print(
        f"{args.loop_id} fold={fold.name} selected_config={selected_config.name}"
        f"{' forced' if forced_config_name else ''}",
        flush=True,
    )

    dynamic_rows = []
    baseline_rows = []
    exit_rows = []
    selected_trades = []
    for artifact in seed_artifacts:
        seed = int(artifact["seed"])
        risk_model = artifact["risk_model"]
        risk_scaler = artifact["risk_scaler"]
        path_for = artifact["path_for"]
        test_entries = artifact["proposals_by_split"]["test"]
        if args.save_live_artifacts:
            artifact["saved_live_artifact"] = _save_live_artifact(
                out_dir=args.out_dir,
                fold_name=fold.name,
                split=split,
                seed=seed,
                effective_seed=int(artifact["effective_seed"]),
                entry_model=artifact["entry_model"],
                entry_standardizer=artifact["entry_standardizer"],
                entry_config=artifact["entry_config"],
                entry_history=artifact["entry_history"],
                risk_model=risk_model,
                risk_scaler=risk_scaler,
                risk_history=artifact["risk_history"],
                selected_config=selected_config,
                variant=variant,
                trial=trial,
                args=args,
                risk_train_samples=int(artifact["risk_train_samples"]),
                risk_validation_samples=int(artifact["risk_validation_samples"]),
                train_entries=int(artifact["train_entries"]),
                validation_entries=int(artifact["validation_entries"]),
                test_entries=int(artifact["test_entries"]),
            )
        dynamic_trades = []
        baseline_trades = []
        for entry in test_entries:
            trade, info = _simulate_lifecycle_exit(
                entry,
                path_for(entry),
                model=risk_model,
                scaler=risk_scaler,
                config=selected_config,
                loop_id=args.loop_id,
            )
            baseline = _trade_from_entry(entry, entry.baseline_pnl, f"{args.loop_id}:protocol051_entry_baseline")
            dynamic_trades.append(trade)
            baseline_trades.append(baseline)
            exit_rows.append({"fold": fold.name, "split": split, "seed": seed, **info})
            selected_trades.append(
                {
                    "fold": fold.name,
                    "split": split,
                    "seed": seed,
                    "session": entry.session,
                    "decision_time": entry.decision_time.isoformat(),
                    "contract_id": str(entry.contract_id),
                    "right": entry.right,
                    "offset": entry.offset,
                    "edge": entry.edge,
                    "baseline_pnl": entry.baseline_pnl,
                    "dynamic_pnl": trade.pnl,
                    **info,
                }
            )
        row = {
            "fold": fold.name,
            "split": split,
            "seed": seed,
            "metrics": _trade_metrics(dynamic_trades),
            "stress50_metrics": _trade_metrics(stress_trades(dynamic_trades, extra_cost_per_trade=50.0)),
            "stress100_metrics": _trade_metrics(stress_trades(dynamic_trades, extra_cost_per_trade=100.0)),
        }
        base_row = {
            "fold": fold.name,
            "split": split,
            "seed": seed,
            "metrics": _trade_metrics(baseline_trades),
            "stress50_metrics": _trade_metrics(stress_trades(baseline_trades, extra_cost_per_trade=50.0)),
            "stress100_metrics": _trade_metrics(stress_trades(baseline_trades, extra_cost_per_trade=100.0)),
        }
        dynamic_rows.append(row)
        baseline_rows.append(base_row)

        if split == "q1_2026":
            march_dynamic = [trade for trade in dynamic_trades if trade.session >= "2026-03-01"]
            march_baseline = [trade for trade in baseline_trades if trade.session >= "2026-03-01"]
            dynamic_rows.append(
                {
                    "fold": fold.name,
                    "split": "march_2026",
                    "seed": seed,
                    "metrics": _trade_metrics(march_dynamic),
                    "stress50_metrics": _trade_metrics(stress_trades(march_dynamic, extra_cost_per_trade=50.0)),
                    "stress100_metrics": _trade_metrics(stress_trades(march_dynamic, extra_cost_per_trade=100.0)),
                }
            )
            baseline_rows.append(
                {
                    "fold": fold.name,
                    "split": "march_2026",
                    "seed": seed,
                    "metrics": _trade_metrics(march_baseline),
                    "stress50_metrics": _trade_metrics(stress_trades(march_baseline, extra_cost_per_trade=50.0)),
                    "stress100_metrics": _trade_metrics(stress_trades(march_baseline, extra_cost_per_trade=100.0)),
                }
            )

    return {
        "fold": fold.summary(),
        "fold_name": fold.name,
        "split": split,
        "selected_config": asdict(selected_config),
        "selected_config_forced": bool(forced_config_name),
        "config_selection_scores": {
            name: {
                "seed_rewards": [float(x) for x in scores],
                "median_reward": float(np.median(scores)) if scores else None,
            }
            for name, scores in sorted(config_scores.items())
        },
        "seed_sample_counts": [
            {
                "seed": int(artifact["seed"]),
                "train_entries": int(artifact["train_entries"]),
                "validation_entries": int(artifact["validation_entries"]),
                "test_entries": int(artifact["test_entries"]),
                "risk_train_samples": int(artifact["risk_train_samples"]),
                "risk_validation_samples": int(artifact["risk_validation_samples"]),
                "saved_live_artifact": artifact.get("saved_live_artifact"),
            }
            for artifact in seed_artifacts
        ],
        "dynamic_seed_rows": dynamic_rows,
        "baseline_seed_rows": baseline_rows,
        "exit_rows": exit_rows,
        "selected_trades": selected_trades,
    }


def main() -> int:
    args = parse_args()
    variant = _find_variant(args.variant_name)
    trial = _find_trial(args.trial_name)
    folds = _folds(args)
    market_cache = MarketStructureCache(
        source=args.market_structure_source,
        index_spx_dir=args.market_spx_dir,
        index_vix_dir=args.market_vix_dir,
        es_vwap_dir=args.es_vwap_dir,
    )
    decision_cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    normalized_store = (
        NormalizedPathStore(args.normalized_dir)
        if args.normalized_dir is not None and args.normalized_dir.exists()
        else None
    )

    fold_payloads = []
    for fold in folds:
        fold_payloads.append(
            _run_fold(
                fold=fold,
                variant=variant,
                trial=trial,
                market_cache=market_cache,
                decision_cache_dir=decision_cache_dir,
                normalized_store=normalized_store,
                args=args,
            )
        )

    dynamic_rows = [row for payload in fold_payloads for row in payload["dynamic_seed_rows"]]
    baseline_rows = [row for payload in fold_payloads for row in payload["baseline_seed_rows"]]
    selected_trades = [row for payload in fold_payloads for row in payload["selected_trades"]]
    scored_splits = [payload["split"] for payload in fold_payloads]
    split_order = [*scored_splits]
    if "q1_2026" in scored_splits:
        split_order.append("march_2026")
    dynamic_summary = _summary_by_split(dynamic_rows, splits=split_order)
    baseline_summary = _summary_by_split(baseline_rows, splits=split_order)
    baseline_lookup = {row["split"]: row for row in baseline_summary}
    dynamic_lookup = {row["split"]: row for row in dynamic_summary}
    beat_folds = sum(
        1
        for split in scored_splits
        if dynamic_lookup.get(split, {}).get("pnl_median", -1e9)
        > baseline_lookup.get(split, {}).get("pnl_median", 1e9)
    )
    stress_positive = all(
        dynamic_lookup.get(split, {}).get("stress50_pnl_median", -1.0) > 0.0
        for split in scored_splits
    )
    required_beats = max(3, len(scored_splits) - 1)
    march_not_damaged = (
        dynamic_lookup.get("march_2026", {}).get("pnl_median", 0.0)
        >= baseline_lookup.get("march_2026", {}).get("pnl_median", 0.0)
    )
    if beat_folds >= required_beats and stress_positive and march_not_damaged:
        decision = (
            f"Keep {args.loop_id} as a lifecycle candidate. It improves the frozen Protocol 051 "
            f"entry baseline in at least {required_beats} of {len(scored_splits)} walk-forward folds, keeps +50 stress positive, "
            "and does not damage March."
        )
    else:
        decision = (
            f"Reject {args.loop_id} as a replacement lifecycle layer for now. Keep Protocol 051 "
            "as the frozen entry baseline and diagnose the fold where dynamic exits reduce edge."
        )

    selected_configs = {
        payload["split"]: payload["selected_config"]["name"]
        for payload in fold_payloads
    }
    payload = {
        "loop_id": args.loop_id,
        "pre_registration": {
            "paid_data_downloaded": False,
            "q4_2024_prehistory": args.q4_2024_dir is not None,
            "entry_variant_frozen": args.variant_name,
            "trial_frozen": args.trial_name,
            "exit_constraint": args.exit_constraint,
            "risk_config_selection": "per-fold validation-only median selection across seeds",
            "teacher_entries": "bounded train/validation-only lifecycle path augmentation; never used for test selection",
            "folds": [fold.summary() for fold in folds],
        },
        "args": {
            "q4_2024_dir": None if args.q4_2024_dir is None else str(args.q4_2024_dir),
            "q1_2025_dir": str(args.q1_2025_dir),
            "q2_2025_dir": str(args.q2_2025_dir),
            "q3_2025_dir": str(args.q3_2025_dir),
            "q4_2025_dir": str(args.q4_2025_dir),
            "q1_2026_dir": str(args.q1_2026_dir),
            "out_dir": str(args.out_dir),
            "decision_cache_dir": None if decision_cache_dir is None else str(decision_cache_dir),
            "normalized_dir": None if normalized_store is None else str(args.normalized_dir),
            "seeds": args.seeds,
            "policy_index": args.policy_index,
            "variant_name": args.variant_name,
            "trial_name": args.trial_name,
            "epochs": args.epochs,
            "risk_epochs": args.risk_epochs,
            "batch_size": args.batch_size,
            "validation_days": args.validation_days,
            "max_risk_teacher_entries": args.max_risk_teacher_entries,
            "exit_constraint": args.exit_constraint,
            "market_structure_source": args.market_structure_source,
            "market_spx_dir": None if args.market_spx_dir is None else str(args.market_spx_dir),
            "market_vix_dir": None if args.market_vix_dir is None else str(args.market_vix_dir),
            "save_live_artifacts": bool(args.save_live_artifacts),
            "force_selected_configs_json": None
            if args.force_selected_configs_json is None
            else str(args.force_selected_configs_json),
        },
        "entry": {
            "variant_name": variant.name,
            "variant": asdict(variant) | {"variant_id": variant.variant_id},
            "policy_index": args.policy_index,
            "policy_name": POLICY_META[args.policy_index][0],
            "trial_name": trial.name,
            "trial": asdict(trial) | {"config_id": trial.config_id},
        },
        "exit_constraint": args.exit_constraint,
        "seeds": args.seeds,
        "risk_feature_names": list(RISK_FEATURE_NAMES),
        "fold_results": [
            {
                key: value
                for key, value in payload.items()
                if key not in {"selected_trades"}
            }
            for payload in fold_payloads
        ],
        "dynamic_summary": dynamic_summary,
        "baseline_summary": baseline_summary,
        "selected_configs": selected_configs,
        "selected_trades_file": str(args.out_dir / "selected_trades_with_lifecycle_exits.json"),
        "decision": decision,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "selected_trades_with_lifecycle_exits.json").write_text(
        json.dumps(selected_trades, indent=2, allow_nan=False) + "\n"
    )
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps(dynamic_summary, indent=2, sort_keys=True), flush=True)
    print(args.out_dir / "report.md", flush=True)
    print(decision, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
