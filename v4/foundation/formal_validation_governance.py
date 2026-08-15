"""Formal validation governance for strategy-family selection risk.

This module reads existing replay summaries and builds a comparable strategy
matrix. It does not train, tune, score protected holdouts, download data, or
call broker endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROLE_LABEL = "FORMAL_VALIDATION_GOVERNANCE_V1"
PASS = "pass"
BLOCKED = "blocked"

DEFAULT_AUDIT_ROOT = Path("v4/audit/autoresearch")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/formal_validation_governance")
PAPER_DEFAULT_BASELINE = "PAPER_DEFAULT_PROTOCOL101"
RESEARCH_EXPOSED_SPLITS = ("q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026")


@dataclass(frozen=True)
class FormalValidationConfig:
    min_strategies: int = 2
    min_splits: int = 3
    max_pbo_proxy: float = 0.50


def build_formal_validation_governance(
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    *,
    config: FormalValidationConfig = FormalValidationConfig(),
) -> dict[str, Any]:
    matrix = strategy_matrix(audit_root)
    cscv = cscv_proxy(matrix, config=config)
    split_exposure = split_exposure_from_matrix(matrix)
    status = control_status(matrix, cscv, config=config)
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "formal validation controls over existing strategy-family replay summaries",
        "changes_paper_default": False,
        "paper_default_baseline": PAPER_DEFAULT_BASELINE,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "protected_holdout_scored": False,
        "control_status": status,
        "decision": "formal_validation_controls_ready" if status == PASS else "formal_validation_controls_blocked_missing_matrix",
        "pbo_cscv_status": "formal_validation_controls_ready" if status == PASS else "blocked_missing_comparable_strategy_matrix",
        "strategy_matrix_rows": int(len(matrix)),
        "comparable_strategy_count": comparable_strategy_count(matrix),
        "comparable_split_count": comparable_split_count(matrix),
        "research_exposed_splits": list(RESEARCH_EXPOSED_SPLITS),
        "split_exposure": split_exposure,
        "cscv_proxy": cscv,
        "required_before_promotion": [
            "Freeze the candidate and baseline before scoring any untouched block.",
            "Use this matrix to report family-selection risk, not to tune thresholds.",
            "Keep current repeated-research splits diagnostic only.",
            "Score the reserved untouched block only once after fill, parity, and promotion packet gates pass.",
        ],
    }


def strategy_matrix(audit_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(audit_root.glob("*/summary.json")):
        payload = read_json(path)
        if not payload:
            continue
        if payload.get("paper_default_baseline") != PAPER_DEFAULT_BASELINE:
            continue
        stress_results = payload.get("stress_results")
        if not isinstance(stress_results, list):
            continue
        for stress in stress_results:
            slippage = finite(stress.get("slippage_per_side"), 0.0)
            seed = int(finite(stress.get("seed"), 0.0))
            splits = stress.get("splits")
            if not isinstance(splits, dict):
                continue
            for split, metrics in splits.items():
                if not isinstance(metrics, dict):
                    continue
                rows.append(
                    {
                        "strategy_id": path.parent.name,
                        "summary_path": str(path),
                        "role_label": str(payload.get("role_label", "")),
                        "decision": str(payload.get("decision", "")),
                        "split": str(split),
                        "slippage_per_side": slippage,
                        "seed": seed,
                        "delta_vs_protocol101_same_scope": finite(metrics.get("delta_vs_protocol101_same_scope"), math.nan),
                        "total_pnl": finite(metrics.get("total_pnl"), math.nan),
                        "protocol101_same_scope_pnl": finite(metrics.get("protocol101_same_scope_pnl"), math.nan),
                        "trades": int(finite(metrics.get("trades"), 0.0)),
                        "challenger_entries": int(finite(metrics.get("challenger_entries"), 0.0)),
                        "max_drawdown": finite(metrics.get("max_drawdown", metrics.get("max_account_drawdown")), math.nan),
                        "profit_factor": finite(metrics.get("profit_factor"), math.nan),
                        "win_rate": finite(metrics.get("win_rate"), math.nan),
                        "all_flat_by_session_end": bool(metrics.get("all_flat_by_session_end", False)),
                        "max_concurrent_positions": int(finite(metrics.get("max_concurrent_positions"), 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def cscv_proxy(matrix: pd.DataFrame, *, config: FormalValidationConfig = FormalValidationConfig()) -> dict[str, Any]:
    if matrix.empty:
        return {"status": BLOCKED, "reason": "empty_strategy_matrix", "folds": 0, "pbo_proxy": None}
    reduced = (
        matrix.dropna(subset=["delta_vs_protocol101_same_scope"])
        .groupby(["strategy_id", "slippage_per_side", "split"], as_index=False)["delta_vs_protocol101_same_scope"]
        .mean()
    )
    strategy_keys = sorted({(row.strategy_id, row.slippage_per_side) for row in reduced.itertuples()})
    splits = sorted(set(reduced["split"].astype(str)))
    if len(strategy_keys) < config.min_strategies or len(splits) < config.min_splits:
        return {
            "status": BLOCKED,
            "reason": "insufficient_comparable_strategies_or_splits",
            "strategies": len(strategy_keys),
            "splits": len(splits),
            "folds": 0,
            "pbo_proxy": None,
        }
    values = {
        (str(row.strategy_id), float(row.slippage_per_side), str(row.split)): float(row.delta_vs_protocol101_same_scope)
        for row in reduced.itertuples()
    }
    fold_rows: list[dict[str, Any]] = []
    half = max(1, len(splits) // 2)
    combinations = list(itertools.combinations(splits, half))
    for train_splits in combinations:
        train = set(train_splits)
        test = [split for split in splits if split not in train]
        if not test:
            continue
        scored: list[tuple[tuple[str, float], float, float]] = []
        for strategy_id, slippage in strategy_keys:
            train_values = [values.get((strategy_id, slippage, split), math.nan) for split in train]
            test_values = [values.get((strategy_id, slippage, split), math.nan) for split in test]
            train_values = [value for value in train_values if math.isfinite(value)]
            test_values = [value for value in test_values if math.isfinite(value)]
            if not train_values or not test_values:
                continue
            scored.append(((strategy_id, slippage), float(np.mean(train_values)), float(np.mean(test_values))))
        if len(scored) < 2:
            continue
        selected_key, train_mean, test_mean = max(scored, key=lambda item: item[1])
        sorted_test = sorted(item[2] for item in scored)
        test_rank = sorted_test.index(test_mean) + 1
        fold_rows.append(
            {
                "train_splits": ",".join(sorted(train)),
                "test_splits": ",".join(sorted(test)),
                "selected_strategy_id": selected_key[0],
                "selected_slippage_per_side": selected_key[1],
                "train_mean_delta": train_mean,
                "test_mean_delta": test_mean,
                "test_rank_pct": test_rank / len(sorted_test),
                "test_positive": test_mean > 0.0,
                "overfit_event": test_mean <= 0.0 or test_rank <= max(1, len(sorted_test) // 2),
            }
        )
    if not fold_rows:
        return {"status": BLOCKED, "reason": "no_valid_cscv_folds", "folds": 0, "pbo_proxy": None}
    pbo_proxy = float(np.mean([row["overfit_event"] for row in fold_rows]))
    return {
        "status": PASS,
        "reason": "cscv_proxy_computed",
        "folds": len(fold_rows),
        "pbo_proxy": pbo_proxy,
        "selected_test_positive_fraction": float(np.mean([row["test_positive"] for row in fold_rows])),
        "median_selected_test_delta": float(np.median([row["test_mean_delta"] for row in fold_rows])),
        "max_allowed_pbo_proxy": float(config.max_pbo_proxy),
        "fold_rows": fold_rows,
    }


def split_exposure_from_matrix(matrix: pd.DataFrame) -> list[dict[str, Any]]:
    if matrix.empty:
        return []
    rows = []
    for split, group in matrix.groupby("split", sort=True):
        rows.append(
            {
                "split": str(split),
                "strategy_count": int(group["strategy_id"].nunique()),
                "row_count": int(len(group)),
                "classification": "research_exposed_diagnostic" if str(split) in RESEARCH_EXPOSED_SPLITS else "other",
            }
        )
    return rows


def control_status(matrix: pd.DataFrame, cscv: dict[str, Any], *, config: FormalValidationConfig = FormalValidationConfig()) -> str:
    if matrix.empty:
        return BLOCKED
    if comparable_strategy_count(matrix) < int(config.min_strategies):
        return BLOCKED
    if comparable_split_count(matrix) < int(config.min_splits):
        return BLOCKED
    if cscv.get("status") != PASS:
        return BLOCKED
    return PASS


def comparable_strategy_count(matrix: pd.DataFrame) -> int:
    if matrix.empty:
        return 0
    return int(matrix[["strategy_id", "slippage_per_side"]].drop_duplicates().shape[0])


def comparable_split_count(matrix: pd.DataFrame) -> int:
    if matrix.empty:
        return 0
    return int(matrix["split"].nunique())


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"Decision: `{payload['decision']}`",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Protected holdout scored: no",
        "",
        "## Control Summary",
        "",
        f"- Control status: `{payload['control_status']}`",
        f"- PBO/CSCV status: `{payload['pbo_cscv_status']}`",
        f"- Comparable strategy/slippage variants: `{payload['comparable_strategy_count']}`",
        f"- Comparable splits: `{payload['comparable_split_count']}`",
        f"- CSCV folds: `{payload['cscv_proxy'].get('folds', 0)}`",
        f"- PBO proxy: `{payload['cscv_proxy'].get('pbo_proxy')}`",
        "",
        "## Split Exposure",
        "",
        "| Split | Strategy count | Rows | Classification |",
        "|---|---:|---:|---|",
    ]
    for row in payload["split_exposure"]:
        lines.append(f"| {row['split']} | {row['strategy_count']} | {row['row_count']} | {row['classification']} |")
    lines.extend(["", "## Required Before Promotion", ""])
    lines.extend(f"- {item}" for item in payload["required_before_promotion"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any], matrix: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    matrix_path = out_dir / "strategy_matrix.csv"
    cscv_path = out_dir / "cscv_folds.csv"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    report_path.write_text(render_report(payload))
    matrix.to_csv(matrix_path, index=False)
    fold_rows = payload.get("cscv_proxy", {}).get("fold_rows", [])
    pd.DataFrame(fold_rows).to_csv(cscv_path, index=False)
    return summary_path, report_path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def finite(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default
