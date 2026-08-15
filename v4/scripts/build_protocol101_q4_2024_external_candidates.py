"""Build Q4 2024 external candidate trades from frozen Protocol 075 artifacts.

This is stage 1 of the Protocol 101 external-audit path. It does not train a
model and does not download data. It applies the frozen Protocol 051 entry
model plus Protocol 054 fallback risk model to already-collected Q4 2024
official-context sessions, producing the same selected-trade shape consumed by
``build_lifecycle_sequence_dataset.py``.

The output is not a Protocol 101 score yet. It is the candidate/outcome
foundation needed before frozen Protocol 081 and Protocol 101 can be evaluated
on Q4 2024.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    SurfaceActionModel,
    SurfaceStandardizer,
    predict_surface_actions,
    registered_aplus_surface_variants,
    registered_protocol_trials,
)
from v4.model.supervised_pilot import FeatureScaler, session_from_path
from v4.scripts.run_aplus_neural_protocol import _load_surface_decisions_cached
from v4.scripts.run_protocol052_sequential_lifecycle_walkforward import (
    NormalizedPathStore,
    _simulate_lifecycle_exit,
)
from v4.scripts.run_sequential_risk_protocol import (
    RiskConfig,
    RiskHeadroomModel,
    _select_entry_proposals,
)
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


DEFAULT_STACK_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts")
DEFAULT_Q4_DIR = Path("data/processed/spxw_0dte_neural_q4_2024_official_context")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_104_q4_2024_external_candidates")
DEFAULT_DECISION_CACHE = Path("data/cache/v4_aplus_surface_decisions_official_context")
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
DEFAULT_SPX_DIR = Path("data/vendor/thetadata/index/spx_1m")
DEFAULT_VIX_DIR = Path("data/vendor/thetadata/index/vix_1m")
ENTRY_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-dir", type=Path, default=DEFAULT_STACK_DIR)
    parser.add_argument("--q4-2024-dir", type=Path, default=DEFAULT_Q4_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--decision-cache-dir", type=Path, default=DEFAULT_DECISION_CACHE)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--market-spx-dir", type=Path, default=DEFAULT_SPX_DIR)
    parser.add_argument("--market-vix-dir", type=Path, default=DEFAULT_VIX_DIR)
    parser.add_argument("--entry-seeds", nargs="*", type=int, default=ENTRY_SEEDS)
    parser.add_argument("--max-sessions", type=int, default=0, help="Debug limit; 0 means all Q4 2024 sessions.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session_paths = sorted(args.q4_2024_dir.glob("*.pkl"))
    if args.max_sessions > 0:
        session_paths = session_paths[: args.max_sessions]
    if not session_paths:
        raise SystemExit(f"no Q4 2024 processed sessions found under {args.q4_2024_dir}")

    manifest_by_seed = _final_stack_manifests(args.stack_dir, args.entry_seeds)
    first_manifest = next(iter(manifest_by_seed.values()))
    variant = _find_variant(first_manifest["variant_name"])
    trial = _find_trial(first_manifest["trial_name"])
    policy_index = int(first_manifest["policy_index"])
    policy_name, cooldown = POLICY_META[policy_index]
    if policy_name != first_manifest["entry_config"]["policy_name"]:
        raise SystemExit("Protocol 075 manifest policy name disagrees with local POLICY_META")

    market_cache = MarketStructureCache(
        source="index_bars",
        index_spx_dir=args.market_spx_dir,
        index_vix_dir=args.market_vix_dir,
    )
    decisions = _load_surface_decisions_cached(
        session_paths,
        policy_index=policy_index,
        variant=variant,
        market_cache=market_cache,
        split="q4_2024_external_protocol101_candidates",
        cache_dir=args.decision_cache_dir,
    )
    normalized_store = NormalizedPathStore(args.normalized_dir)

    selected_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    for entry_seed in args.entry_seeds:
        manifest = manifest_by_seed[int(entry_seed)]
        entry_model, standardizer = _load_entry_artifact(manifest)
        risk_model, risk_scaler = _load_risk_artifact(manifest)
        risk_config = RiskConfig(**manifest["selected_risk_config"])
        predictions = predict_surface_actions(
            entry_model,
            standardizer,
            decisions,
            target_scale=float(manifest["entry_config"]["target_scale"]),
        )
        proposals = _select_entry_proposals(
            decisions,
            predictions,
            trial=trial,
            cooldown_minutes=int(cooldown),
            split="q4_2024_external",
            seed=int(entry_seed),
            effective_seed=int(manifest["effective_seed"]),
        )
        seed_rows = []
        path_missing = 0
        for proposal in proposals:
            path = normalized_store.contract_path(proposal)
            if not path:
                path_missing += 1
            trade, info = _simulate_lifecycle_exit(
                proposal,
                path,
                model=risk_model,
                scaler=risk_scaler,
                config=risk_config,
                loop_id="protocol104_q4_2024_external_candidates",
            )
            seed_rows.append(
                {
                    "fold": "frozen_protocol075_train_through_q4_2025_applied_to_q4_2024",
                    "split": "q4_2024_external",
                    "seed": int(entry_seed),
                    "session": proposal.session,
                    "decision_time": proposal.decision_time.isoformat(),
                    "contract_id": str(proposal.contract_id),
                    "right": proposal.right,
                    "offset": float(proposal.offset),
                    "edge": float(proposal.edge),
                    "baseline_pnl": float(proposal.baseline_pnl),
                    "dynamic_pnl": float(trade.pnl),
                    **_json_safe_info(info),
                }
            )
        selected_rows.extend(seed_rows)
        seed_summaries.append(
            {
                "entry_seed": int(entry_seed),
                "proposals": len(proposals),
                "path_missing": int(path_missing),
                "dynamic_pnl": float(sum(row["dynamic_pnl"] for row in seed_rows)),
                "baseline_pnl": float(sum(row["baseline_pnl"] for row in seed_rows)),
            }
        )

    selected_path = args.out_dir / "selected_trades_with_lifecycle_exits.json"
    selected_path.write_text(json.dumps(selected_rows, indent=2, sort_keys=True, allow_nan=False) + "\n")
    payload = {
        "protocol": "104_q4_2024_external_candidates",
        "paid_data_downloaded": False,
        "live_orders": False,
        "model_training": False,
        "source_stack_dir": str(args.stack_dir),
        "q4_2024_dir": str(args.q4_2024_dir),
        "sessions": [session_from_path(path) for path in session_paths],
        "entry_seeds": [int(seed) for seed in args.entry_seeds],
        "variant_name": variant.name,
        "trial_name": trial.name,
        "policy_index": policy_index,
        "selected_trades_file": str(selected_path),
        "selected_rows": len(selected_rows),
        "seed_summaries": seed_summaries,
        "decision": "built_q4_2024_protocol054_external_candidates",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "selected_rows": len(selected_rows)}, indent=2, sort_keys=True))
    print(args.out_dir / "report.md")
    return 0


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _scaler_from_dict(payload: dict[str, Any]) -> FeatureScaler:
    return FeatureScaler(
        fill=np.asarray(payload["fill"], dtype=np.float32),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def _standardizer_from_dict(payload: dict[str, Any]) -> SurfaceStandardizer:
    return SurfaceStandardizer(
        scalar=_scaler_from_dict(payload["scalar"]),
        token=_scaler_from_dict(payload["token"]),
    )


def _load_entry_artifact(manifest: dict[str, Any]) -> tuple[SurfaceActionModel, SurfaceStandardizer]:
    files = manifest["files"]
    checkpoint = _torch_load(Path(files["entry_model"]))
    model = SurfaceActionModel(
        scalar_dim=int(checkpoint["scalar_dim"]),
        token_dim=int(checkpoint["token_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    standardizer = _standardizer_from_dict(json.loads(Path(files["entry_standardizer"]).read_text()))
    return model, standardizer


def _load_risk_artifact(manifest: dict[str, Any]) -> tuple[RiskHeadroomModel, FeatureScaler]:
    files = manifest["files"]
    checkpoint = _torch_load(Path(files["protocol054_risk_model"]))
    model = RiskHeadroomModel(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 64)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaler = _scaler_from_dict(json.loads(Path(files["protocol054_risk_scaler"]).read_text()))
    return model, scaler


def _final_stack_manifests(stack_dir: Path, seeds: list[int]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        path = stack_dir / "model_artifacts" / "train_through_q4_2025_test_q1_2026" / f"seed_{seed}" / "manifest.json"
        if not path.exists():
            raise SystemExit(f"missing frozen Protocol 075 manifest for seed {seed}: {path}")
        out[int(seed)] = json.loads(path.read_text())
    return out


def _find_variant(name: str):
    for variant in registered_aplus_surface_variants():
        if variant.name == name:
            return variant
    raise SystemExit(f"unknown A+ variant in manifest: {name}")


def _find_trial(name: str):
    for trial in registered_protocol_trials():
        if trial.name == name:
            return trial
    raise SystemExit(f"unknown trial in manifest: {name}")


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in info.items():
        if isinstance(value, float):
            out[key] = float(value) if np.isfinite(value) else None
        else:
            out[key] = value
    return out


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 104: Q4 2024 External Candidates",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Sessions scored: `{len(payload['sessions'])}`",
        f"- Selected rows: `{payload['selected_rows']}`",
        f"- Variant/trial: `{payload['variant_name']} / {payload['trial_name']}`",
        f"- Output: `{payload['selected_trades_file']}`",
        "",
        "## Seed Summary",
        "",
        "| entry_seed | proposals | path_missing | dynamic_pnl | baseline_pnl |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seed_summaries"]:
        lines.append(
            f"| {row['entry_seed']} | {row['proposals']} | {row['path_missing']} | "
            f"{row['dynamic_pnl']:.0f} | {row['baseline_pnl']:.0f} |"
        )
    lines += [
        "",
        "## Next",
        "",
        "Run the lifecycle sequence dataset builder on this output, then apply frozen Protocol 081 sequence artifacts before scoring Protocol 101.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
