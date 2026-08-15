"""Protocol185: surface-edge enrichment for Protocol164 full-action rows.

Protocol164 fixed the live/training mismatch by exposing the full flat-state
SPXW 0DTE action space, but it intentionally did not carry Protocol101's
upstream A+ surface edge signal. That made later broad-action models compare
against Protocol101 without one of Protocol101's most important causal inputs.

This runner repairs that feature-parity gap without buying data or touching a
broker. It scores the already-collected processed rows with the frozen
Protocol051/A+ surface model, computes per-contract edge = action - flat, and
joins those causal scores onto the Protocol164 candidate dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from v4.live.protocol051_surface_edge import load_surface_edge_artifact, score_surface_decisions
from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    SurfaceDecision,
    registered_aplus_surface_variants,
    surface_decision_from_row,
)
from v4.scripts.run_protocol164_full_action_space_dataset import (
    DEFAULT_BLOCKS,
    DEFAULT_OUT_DIR as DEFAULT_PROTOCOL164_DIR,
    FULL_ACTION_FEATURE_COLUMNS,
    parse_blocks,
)


LOOP_ID = "v4_aplus_hypothesis_185_surface_edge_full_action_enrichment"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_DATASET = DEFAULT_PROTOCOL164_DIR / "protocol164_full_action_space_dataset.parquet"
DEFAULT_SURFACE_ARTIFACT_ROOT = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/model_artifacts"
)
DEFAULT_INDEX_SPX_DIR = Path("data/raw/index/spx_1m")
DEFAULT_INDEX_VIX_DIR = Path("data/raw/index/vix_1m")
SURFACE_FEATURE_COLUMNS = [
    "surface_edge",
    "surface_action_score",
    "surface_flat_score",
    "surface_valid_token_count",
    "surface_best_edge",
    "surface_mean_edge",
    "surface_best_call_edge",
    "surface_best_put_edge",
    "surface_call_minus_put_best_edge",
    "surface_above25_count",
    "surface_roll3_best_edge",
    "surface_roll3_mean_edge",
    "surface_prev_best_edge",
    "surface_prev_mean_edge",
]

SPLIT_SURFACE_FOLDS = {
    "q1_2025": "train_through_q1_2025_test_q2_2025",
    "q2_2025": "train_through_q1_2025_test_q2_2025",
    "q3_2025": "train_through_q2_2025_test_q3_2025",
    "q4_2025": "train_through_q3_2025_test_q4_2025",
    "q1_2026": "train_through_q4_2025_test_q1_2026",
    "recent_2026": "train_through_q4_2025_test_q1_2026",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--blocks", nargs="*", default=list(DEFAULT_BLOCKS), help="split=processed_dir entries")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--surface-artifact-root", type=Path, default=DEFAULT_SURFACE_ARTIFACT_ROOT)
    parser.add_argument("--surface-seed", type=int, default=11)
    parser.add_argument("--policy-index", type=int, default=1)
    parser.add_argument("--variant-name", default="surface_structure_aplus_side_value_rank")
    parser.add_argument("--index-spx-dir", type=Path, default=DEFAULT_INDEX_SPX_DIR)
    parser.add_argument("--index-vix-dir", type=Path, default=DEFAULT_INDEX_VIX_DIR)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.dataset)
    if dataset.empty:
        raise ValueError(f"empty Protocol164 dataset: {args.dataset}")
    dataset["decision_dt"] = pd.to_datetime(dataset["decision_dt"], utc=True)
    dataset["contract_id"] = dataset["contract_id"].astype(str)
    blocks = dict(parse_blocks(args.blocks))
    variant = _variant_by_name(str(args.variant_name))
    market_cache = MarketStructureCache(
        source="index_bars",
        index_spx_dir=args.index_spx_dir,
        index_vix_dir=args.index_vix_dir,
    )
    needed = _needed_index(dataset)
    edge_rows: list[dict[str, Any]] = []
    missing_sessions: list[dict[str, Any]] = []
    artifact_cache: dict[str, Any] = {}
    for split, sessions in sorted(needed.items()):
        processed_dir = blocks.get(split)
        if processed_dir is None:
            missing_sessions.extend({"split": split, "session": session, "reason": "missing_processed_block"} for session in sorted(sessions))
            continue
        artifact = artifact_cache.setdefault(
            split,
            load_surface_edge_artifact(_manifest_path(args.surface_artifact_root, split=split, seed=int(args.surface_seed))),
        )
        for session, wanted_times in sorted(sessions.items()):
            processed_path = processed_dir / f"{session}.pkl"
            if not processed_path.exists():
                missing_sessions.append({"split": split, "session": session, "reason": "missing_processed_session", "path": str(processed_path)})
                continue
            decisions = _decisions_for_session(
                split=split,
                session=session,
                processed_path=processed_path,
                wanted_times=wanted_times,
                policy_index=int(args.policy_index),
                variant=variant,
                market_cache=market_cache,
            )
            if not decisions:
                missing_sessions.append({"split": split, "session": session, "reason": "no_matching_decisions", "path": str(processed_path)})
                continue
            session_edges = _edge_rows_from_decisions(
                split=split,
                decisions=decisions,
                artifact=artifact,
                batch_size=int(args.batch_size),
            )
            edge_rows.extend(session_edges)
            print(
                json.dumps(
                    {
                        "split": split,
                        "session": session,
                        "decisions": len(decisions),
                        "edge_rows": len(session_edges),
                        "status": "scored",
                    },
                    sort_keys=True,
                )
            )

    edges = pd.DataFrame(edge_rows)
    if not edges.empty:
        edges["decision_dt"] = pd.to_datetime(edges["decision_dt"], utc=True)
        edges["contract_id"] = edges["contract_id"].astype(str)
        edges = edges.drop_duplicates(["split", "session", "decision_dt", "contract_id"], keep="last")
    enriched = dataset.merge(
        edges,
        on=["split", "session", "decision_dt", "contract_id"],
        how="left",
        validate="many_to_one",
    )
    for column in SURFACE_FEATURE_COLUMNS:
        if column not in enriched:
            enriched[column] = np.nan
    enriched = enriched.sort_values(["split", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True)
    dataset_out = args.out_dir / "protocol185_full_action_with_surface_edge.parquet"
    enriched.to_parquet(dataset_out, index=False)
    if missing_sessions:
        pd.DataFrame(missing_sessions).to_csv(args.out_dir / "missing_surface_edge_sessions.csv", index=False)

    payload = {
        "protocol": "185_surface_edge_full_action_enrichment",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "source_dataset": str(args.dataset),
        "enriched_dataset": str(dataset_out),
        "surface_feature_columns": SURFACE_FEATURE_COLUMNS,
        "training_feature_columns": [*FULL_ACTION_FEATURE_COLUMNS, *SURFACE_FEATURE_COLUMNS],
        "surface_variant": str(args.variant_name),
        "policy_index": int(args.policy_index),
        "surface_seed": int(args.surface_seed),
        "surface_artifact_root": str(args.surface_artifact_root),
        "split_surface_folds": SPLIT_SURFACE_FOLDS,
        "q1_2025_caveat": "q1_2025 uses the earliest saved Protocol075 fold as a training-split surface-score fallback; do not use q1_2025 as a headline holdout for this enrichment.",
        "rows": int(len(enriched)),
        "edge_rows": int(len(edges)),
        "matched_rows": int(enriched["surface_edge"].notna().sum()),
        "matched_fraction": _safe_ratio(float(enriched["surface_edge"].notna().sum()), float(len(enriched))),
        "matched_fraction_by_split": _matched_by_split(enriched),
        "surface_edge_summary": _numeric_summary(enriched["surface_edge"]),
        "surface_best_edge_summary": _numeric_summary(enriched["surface_best_edge"]),
        "missing_sessions": missing_sessions,
    }
    payload["decision"] = "ready_for_surface_edge_policy_test" if payload["matched_fraction"] >= 0.98 else "blocked_incomplete_surface_edge_join"
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "matched_fraction": payload["matched_fraction"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if payload["decision"] == "ready_for_surface_edge_policy_test" else 1


def _needed_index(dataset: pd.DataFrame) -> dict[str, dict[str, set[pd.Timestamp]]]:
    out: dict[str, dict[str, set[pd.Timestamp]]] = {}
    for (split, session), group in dataset.groupby(["split", "session"], sort=True):
        out.setdefault(str(split), {})[str(session)] = set(pd.to_datetime(group["decision_dt"], utc=True).dt.floor("min").tolist())
    return out


def _variant_by_name(name: str) -> Any:
    for variant in registered_aplus_surface_variants():
        if variant.name == name:
            return variant
    raise ValueError(f"unknown A+ surface variant: {name}")


def _manifest_path(root: Path, *, split: str, seed: int) -> Path:
    fold = SPLIT_SURFACE_FOLDS.get(split)
    if not fold:
        raise ValueError(f"no surface fold mapping for split={split}")
    path = root / fold / f"seed_{seed}" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing surface artifact manifest: {path}")
    return path


def _decisions_for_session(
    *,
    split: str,
    session: str,
    processed_path: Path,
    wanted_times: set[pd.Timestamp],
    policy_index: int,
    variant: Any,
    market_cache: MarketStructureCache,
) -> list[SurfaceDecision]:
    rows = pickle.loads(processed_path.read_bytes())
    decisions: list[SurfaceDecision] = []
    for row in rows:
        decision_dt = _utc_minute(row["decision_time"])
        if decision_dt not in wanted_times:
            continue
        decision = surface_decision_from_row(
            session=session,
            row=row,
            policy_index=policy_index,
            variant=variant,
            market_cache=market_cache,
        )
        decisions.append(decision)
    return decisions


def _edge_rows_from_decisions(
    *,
    split: str,
    decisions: list[SurfaceDecision],
    artifact: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
    predictions = score_surface_decisions(artifact, decisions, batch_size=batch_size)
    out: list[dict[str, Any]] = []
    previous: dict[str, float] | None = None
    rolling: list[dict[str, float]] = []
    for decision, prediction in zip(decisions, predictions):
        decision_dt = _utc_minute(decision.decision_time)
        scores = np.asarray(prediction, dtype=float)
        if len(scores) != len(decision.token_mask) + 1:
            raise ValueError("surface prediction must contain flat plus token scores")
        flat = float(scores[0])
        token_scores = scores[1:]
        token_mask = np.asarray(decision.token_mask, dtype=bool)
        edges = token_scores - flat
        eligible = token_mask & np.isfinite(edges)
        summary = _decision_edge_summary(decision, edges, eligible)
        prev = previous or _empty_surface_summary()
        roll = rolling[-3:] if rolling else [_empty_surface_summary()]
        for token_idx, edge in enumerate(edges):
            if not bool(eligible[token_idx]):
                continue
            out.append(
                {
                    "split": split,
                    "session": decision.session,
                    "decision_dt": decision_dt,
                    "contract_id": str(decision.contract_ids[token_idx]),
                    "surface_edge": float(edge),
                    "surface_action_score": float(token_scores[token_idx]),
                    "surface_flat_score": flat,
                    "surface_valid_token_count": summary["valid_token_count"],
                    "surface_best_edge": summary["best_edge"],
                    "surface_mean_edge": summary["mean_edge"],
                    "surface_best_call_edge": summary["best_call_edge"],
                    "surface_best_put_edge": summary["best_put_edge"],
                    "surface_call_minus_put_best_edge": summary["call_minus_put_best_edge"],
                    "surface_above25_count": summary["above25_count"],
                    "surface_roll3_best_edge": max(item["best_edge"] for item in roll),
                    "surface_roll3_mean_edge": float(np.mean([item["mean_edge"] for item in roll])),
                    "surface_prev_best_edge": prev["best_edge"],
                    "surface_prev_mean_edge": prev["mean_edge"],
                }
            )
        previous = summary
        rolling.append(summary)
    return out


def _decision_edge_summary(decision: SurfaceDecision, edges: np.ndarray, eligible: np.ndarray) -> dict[str, float]:
    clean = np.asarray(edges[eligible], dtype=float)
    rights = np.asarray(decision.rights, dtype=object)
    call_edges = np.asarray(edges[eligible & (rights == "C")], dtype=float)
    put_edges = np.asarray(edges[eligible & (rights == "P")], dtype=float)
    best_call = _nanmax(call_edges)
    best_put = _nanmax(put_edges)
    best = _nanmax(clean)
    mean = float(np.nanmean(clean)) if len(clean) else 0.0
    return {
        "valid_token_count": float(len(clean)),
        "best_edge": best,
        "mean_edge": mean if math.isfinite(mean) else 0.0,
        "best_call_edge": best_call,
        "best_put_edge": best_put,
        "call_minus_put_best_edge": best_call - best_put,
        "above25_count": float(np.sum(clean >= 25.0)) if len(clean) else 0.0,
    }


def _empty_surface_summary() -> dict[str, float]:
    return {
        "valid_token_count": 0.0,
        "best_edge": 0.0,
        "mean_edge": 0.0,
        "best_call_edge": 0.0,
        "best_put_edge": 0.0,
        "call_minus_put_best_edge": 0.0,
        "above25_count": 0.0,
    }


def _utc_minute(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("min")


def _nanmax(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0 or not np.isfinite(arr).any():
        return 0.0
    return float(np.nanmax(arr))


def _safe_ratio(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0


def _matched_by_split(frame: pd.DataFrame) -> dict[str, float]:
    out = {}
    for split, group in frame.groupby("split", sort=True):
        out[str(split)] = _safe_ratio(float(group["surface_edge"].notna().sum()), float(len(group)))
    return out


def _numeric_summary(series: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol185 Surface-Edge Full-Action Enrichment",
        "",
        "Repairs a feature-parity gap by joining frozen Protocol051/A+ surface scores onto the Protocol164 full-action candidate rows.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Source dataset: `{payload['source_dataset']}`",
        f"- Enriched dataset: `{payload['enriched_dataset']}`",
        f"- Rows: `{payload['rows']}`",
        f"- Matched rows: `{payload['matched_rows']}`",
        f"- Matched fraction: `{payload['matched_fraction']:.4f}`",
        f"- Matched fraction by split: `{payload['matched_fraction_by_split']}`",
        f"- Surface variant: `{payload['surface_variant']}`",
        f"- Surface fold map: `{payload['split_surface_folds']}`",
        f"- Q1 caveat: {payload['q1_2025_caveat']}",
        "",
        "## Edge Distribution",
        "",
        f"- Surface edge: `{payload['surface_edge_summary']}`",
        f"- Surface best edge: `{payload['surface_best_edge_summary']}`",
        "",
        "## Outputs",
        "",
        f"- Summary: `{path.parent / 'summary.json'}`",
        f"- Enriched dataset: `{payload['enriched_dataset']}`",
    ]
    if payload.get("missing_sessions"):
        lines.extend(["", f"- Missing sessions: `{path.parent / 'missing_surface_edge_sessions.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
