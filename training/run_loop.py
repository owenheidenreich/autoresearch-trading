#!/usr/bin/env python3
"""
Autoresearch loop: LLM-driven iterative model improvement.

Runs on the H100 GPU container. Each iteration:
  1. Reads strict contract (program.md) + current train.py + promoted history
  2. Calls Claude API to propose a modified train.py
  3. Validates syntax, runs training, captures score + reliability metrics
  4. If improved and reliable -> keep and auto-promote. Otherwise revert.
  5. Logs everything to results/run-*/experiments.v2.jsonl

Usage (on H100, after uploading data.pt):
  ANTHROPIC_API_KEY=sk-ant-xxx python -u run_loop.py --hours 8

Requires: anthropic, torch (already on container)
Install:  pip install anthropic
"""
from __future__ import annotations

import os
import sys
import ast
import gc
import json
import time
import copy
import signal
import shutil
import resource
import subprocess
import datetime
import traceback
import hashlib
from typing import Any


def _safe_score(exp: dict, default: float = -999.0) -> float:
    """Get score from experiment dict, safely handling None values."""
    v = exp.get("score")
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")
PROGRAM_MD = os.path.join(SCRIPT_DIR, "program.md")
BEST_TRAIN_PY = os.path.join(SCRIPT_DIR, "best_train.py")
BEST_MODEL_PT = os.path.join(SCRIPT_DIR, "best_model.pt")
LAB_NOTEBOOK_MD = os.path.join(SCRIPT_DIR, "lab_notebook.md")
LAB_NOTEBOOK_MAX_ENTRIES = 10  # rolling window for Best Runs table
LAB_NOTEBOOK_DEAD_ENDS_MAX = 10  # rolling window for Dead Ends
PYTHON = sys.executable


def _resolve_results_dir(script_dir: str) -> str:
    """Resolve canonical results root.

    Local repo execution:
      <repo>/training/run_loop.py -> <repo>/results
    Remote container execution:
      /root/run_loop.py -> /root/results
    """
    parent = os.path.dirname(script_dir)
    if os.path.basename(script_dir) == "training":
        if os.path.isdir(os.path.join(parent, ".git")) or os.path.exists(os.path.join(parent, "README.md")):
            return os.path.join(parent, "results")
    return os.path.join(script_dir, "results")


def _runtime_source() -> str:
    if SCRIPT_DIR == "/root":
        return "remote_container"
    if os.path.basename(SCRIPT_DIR) == "training":
        return "local_repo_training"
    return "local"


RUN_NAME = datetime.datetime.now().strftime("run-%Y-%m-%d-%H%M%S")
RESULTS_DIR = _resolve_results_dir(SCRIPT_DIR)
RUN_DIR = os.path.join(RESULTS_DIR, RUN_NAME)
ARTIFACTS_DIR = os.path.join(RUN_DIR, "artifacts")
RUN_METADATA_JSON = os.path.join(RUN_DIR, "run_metadata.json")
RUN_EXPERIMENTS_V2_LOG = os.path.join(RUN_DIR, "experiments.v2.jsonl")
RUN_RESULTS_TSV = os.path.join(RUN_DIR, "results.tsv")
STATUS_JSON = os.path.join(RUN_DIR, "status.json")
CURRENT_RUN_TXT = os.path.join(RESULTS_DIR, "current_run.txt")
RUN_BEST_TRAIN_PY = os.path.join(RUN_DIR, "best_train.py")
RUN_TRAIN_PY = os.path.join(RUN_DIR, "train.py")
RUN_BEST_MODEL_PT = os.path.join(RUN_DIR, "best_model.pt")
PROMOTED_DIR = os.path.join(RESULTS_DIR, "promoted")
PROMOTED_CURRENT_TXT = os.path.join(PROMOTED_DIR, "current.txt")
PROMOTED_HISTORY_JSONL = os.path.join(PROMOTED_DIR, "history.jsonl")

FEATURE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-trading", "features")
DATA_QUALITY_REPORT_JSON = os.path.join(FEATURE_CACHE_DIR, "data_quality_report.json")
DATA_QUALITY_FINGERPRINT_JSON = os.path.join(FEATURE_CACHE_DIR, "data_quality_fingerprint.json")

SCHEMA_VERSION = 2
FAILURE_TYPE_ENUM = {
    "api",
    "truncation",
    "syntax",
    "safety",
    "train_crash",
    "timeout",
    "parse",
    "drift_guard",
    "regression",
    "none",
}
REQUIRED_OUTPUT_METRIC_KEYS = (
    "score",
    "profit_factor",
    "trades_per_day",
    "trade_sharpe",
    "stop_loss_rate",
    "worst_chunk_pf",
)

OBSERVABILITY_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "critical_anomaly_flags": {
        "metric_inconsistent",
        "trades_per_day_extreme",
        "cost_realism_low_coverage",
        "entry_quality_too_low",
        "high_cost_entry_rate_high",
        "low_quality_entry_rate_high",
    },
    "near_tie_delta": 0.05,
    "near_tie_stability": {
        "worst_chunk_pf_min": 1.0,
        "stop_loss_rate_max": 0.35,
        "direction_collapse_pct_max": 0.85,
        "cost_realism_coverage_min": 0.30,
        "avg_entry_quality_min": 0.25,
        "high_cost_entry_rate_max": 0.50,
        "low_quality_entry_rate_max": 0.50,
    },
    "anomaly_thresholds": {
        "do_nothing_zero_max": 1e-6,
        "exit_pct_extreme_min": 0.98,
        "trades_per_day_extreme_min": 0.25,
        "trades_per_day_extreme_max": 20.0,
        "cost_realism_coverage_min": 0.20,
        "avg_entry_quality_min": 0.15,
        "high_cost_entry_rate_max": 0.65,
        "low_quality_entry_rate_max": 0.65,
        "actionable_bar_rate_min": 0.03,
    },
}
FEATURE_LOCK_COUNT = 32

# Claude model for code generation
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 32768  # train.py grows over time; 32k gives ample headroom for reasoning + full rewrite
MAX_CODEGEN_ATTEMPTS = 3
SMOKE_TIME_BUDGET = 60   # Full cold start on H100: torch import + CUDA init + 257MB data load + model init + few steps ≈ 50s
SMOKE_TIMEOUT_BUFFER = 60

# Reusable anthropic client (avoid httpx connection pool leak)
_anthropic_client = None

# Prefetch: pipeline Claude API calls during training
import concurrent.futures
_prefetch_executor: concurrent.futures.ThreadPoolExecutor | None = None
_prefetched_responses: dict[int, dict[str, Any]] = {}  # exp_id -> {"response": str, "state_hash": str}


def _get_anthropic_client():
    """Return a reusable Anthropic client (singleton)."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def _write_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)


def _write_text_atomic(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _write_json(path: str, obj: dict[str, Any] | list[Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _write_current_run_pointer() -> None:
    _ensure_dir(RESULTS_DIR)
    _write_text_atomic(CURRENT_RUN_TXT, f"{RUN_NAME}\n")


def _initialize_run_layout() -> None:
    _ensure_dir(RUN_DIR)
    _ensure_dir(ARTIFACTS_DIR)
    _ensure_dir(PROMOTED_DIR)
    _write_current_run_pointer()


def _snapshot_runtime_files_to_run_dir() -> None:
    """Keep run folder self-contained for sync/download tooling."""
    _ensure_dir(RUN_DIR)
    if os.path.exists(TRAIN_PY):
        shutil.copy2(TRAIN_PY, RUN_TRAIN_PY)
    if os.path.exists(BEST_TRAIN_PY):
        shutil.copy2(BEST_TRAIN_PY, RUN_BEST_TRAIN_PY)
    if os.path.exists(BEST_MODEL_PT):
        shutil.copy2(BEST_MODEL_PT, RUN_BEST_MODEL_PT)


def _artifact_dir(experiment_id: int) -> str:
    path = os.path.join(ARTIFACTS_DIR, f"exp-{experiment_id}")
    _ensure_dir(path)
    return path


def _save_artifact_text(artifact_dir: str, name: str, text: str) -> None:
    _write_text(os.path.join(artifact_dir, name), text)


def _save_artifact_json(artifact_dir: str, name: str, obj: dict[str, Any] | list[Any]) -> None:
    _write_json(os.path.join(artifact_dir, name), obj)


def _compute_data_quality_report(data: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    import torch

    features = data["features"]
    dates = list(data.get("dates", []))
    timestamps = list(data.get("timestamps", []))
    n_bars = int(features.shape[0])
    n_features = int(features.shape[1]) if features.ndim == 2 else 0
    nan_mask = torch.isnan(features)
    overall_nan_pct = float(nan_mask.float().mean().item() * 100.0) if n_bars and n_features else 0.0
    nan_rate_by_feature = []
    if n_features:
        per_feat = nan_mask.float().mean(dim=0).cpu().numpy()
        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            nan_rate_by_feature.append(
                {
                    "index": i,
                    "name": name,
                    "nan_rate": float(per_feat[i]),
                }
            )

    # Aggregate per-bar missingness by bar-of-day to detect time-of-day blind spots.
    by_minute_sum = [0.0] * 390
    by_minute_count = [0] * 390
    minute_idx = 0
    prev_date = None
    row_nan = nan_mask.float().mean(dim=1).cpu().numpy() if n_bars and n_features else []
    for i, d in enumerate(dates):
        if d != prev_date:
            minute_idx = 0
            prev_date = d
        else:
            minute_idx += 1
        bod = max(0, min(389, minute_idx))
        by_minute_sum[bod] += float(row_nan[i]) if i < len(row_nan) else 0.0
        by_minute_count[bod] += 1
    tod_missingness = []
    for m in range(390):
        if by_minute_count[m] == 0:
            continue
        tod_missingness.append(
            {
                "bar_of_day": m,
                "avg_nan_rate": by_minute_sum[m] / by_minute_count[m],
                "samples": by_minute_count[m],
            }
        )
    tod_missingness = sorted(tod_missingness, key=lambda x: x["avg_nan_rate"], reverse=True)[:25]

    def _coverage_for(key: str) -> float | None:
        arr = data.get(key)
        if arr is None:
            return None
        valid = (~torch.isnan(arr)).float().mean().item()
        return float(valid)

    option_coverage = {
        "atm_call_prices": _coverage_for("atm_call_prices"),
        "atm_put_prices": _coverage_for("atm_put_prices"),
        "otm5_call_prices": _coverage_for("otm5_call_prices"),
        "otm5_put_prices": _coverage_for("otm5_put_prices"),
        "otm10_call_prices": _coverage_for("otm10_call_prices"),
        "otm10_put_prices": _coverage_for("otm10_put_prices"),
        "otm15_call_prices": _coverage_for("otm15_call_prices"),
        "otm15_put_prices": _coverage_for("otm15_put_prices"),
        "otm20_call_prices": _coverage_for("otm20_call_prices"),
        "otm20_put_prices": _coverage_for("otm20_put_prices"),
    }
    sidecar_coverage = {
        "action_spread_bps": _coverage_for("action_spread_bps"),
        "action_quote_age_s": _coverage_for("action_quote_age_s"),
        "action_size": _coverage_for("action_size"),
        "action_quality_score": _coverage_for("action_quality_score"),
        "action_slippage_bps": _coverage_for("action_slippage_bps"),
        "action_cost_bps": _coverage_for("action_cost_bps"),
        "actionable_mask": _coverage_for("actionable_mask"),
        "risk_state_mask": _coverage_for("risk_state_mask"),
        "supervision_weight": _coverage_for("supervision_weight"),
    }

    date_start = min(dates) if dates else None
    date_end = max(dates) if dates else None
    num_days = len(set(dates)) if dates else 0
    payload_for_hash = {
        "n_bars": n_bars,
        "n_features": n_features,
        "date_start": date_start,
        "date_end": date_end,
        "num_days": num_days,
        "overall_nan_pct": round(overall_nan_pct, 6),
        "option_coverage": {k: (None if v is None else round(v, 6)) for k, v in option_coverage.items()},
        "sidecar_coverage": {k: (None if v is None else round(v, 6)) for k, v in sidecar_coverage.items()},
    }
    fingerprint = _sha256_text(json.dumps(payload_for_hash, sort_keys=True))

    return {
        "created_at": _now_iso(),
        "fingerprint": fingerprint,
        "summary": payload_for_hash,
        "nan_rate_by_feature": nan_rate_by_feature,
        "time_of_day_missingness_top": tod_missingness,
        "timestamp_samples": {
            "first": str(timestamps[0]) if timestamps else None,
            "last": str(timestamps[-1]) if timestamps else None,
        },
    }


def _load_saved_data_fingerprint() -> dict[str, Any] | None:
    if not os.path.exists(DATA_QUALITY_FINGERPRINT_JSON):
        return None
    try:
        with open(DATA_QUALITY_FINGERPRINT_JSON, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_data_quality(report: dict[str, Any]) -> None:
    _ensure_dir(FEATURE_CACHE_DIR)
    _write_json(DATA_QUALITY_REPORT_JSON, report)
    _write_json(
        DATA_QUALITY_FINGERPRINT_JSON,
        {
            "fingerprint": report.get("fingerprint"),
            "summary": report.get("summary", {}),
            "created_at": report.get("created_at"),
        },
    )


def _save_run_metadata(metadata: dict[str, Any]) -> None:
    _ensure_dir(RUN_DIR)
    _write_json(RUN_METADATA_JSON, metadata)


def _append_jsonl(path: str, rec: dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def _load_checkpoint_score(checkpoint_path: str) -> float | None:
    if not os.path.exists(checkpoint_path):
        return None
    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
        score = metrics.get("score")
        if score is None:
            score = metrics.get("val_sharpe")
        if score is None:
            return None
        return float(score)
    except Exception:
        return None


def _promotion_event_from_exp(exp: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": _now_iso(),
        "run_name": RUN_NAME,
        "experiment_id": exp.get("experiment_id"),
        "score": float(exp.get("score", -999.0)),
        "model_hash": exp.get("model_fingerprint_after") or _sha256_file(BEST_MODEL_PT),
        "train_hash": exp.get("train_py_after_hash") or _sha256_file(BEST_TRAIN_PY),
        "profit_factor": exp.get("profit_factor"),
        "trades_per_day": exp.get("trades_per_day"),
        "trade_sharpe": exp.get("trade_sharpe"),
        "stop_loss_rate": exp.get("stop_loss_rate"),
        "worst_chunk_pf": exp.get("worst_chunk_pf"),
        "direction_collapse_pct": exp.get("direction_collapse_pct"),
        "prompt_fingerprint": exp.get("prompt_fingerprint"),
        "program_md_fingerprint": exp.get("program_md_fingerprint"),
        "data_fingerprint": exp.get("data_fingerprint"),
        "change_summary": exp.get("change_summary"),
        "avg_entry_cost_bps": exp.get("avg_entry_cost_bps"),
        "avg_entry_quality": exp.get("avg_entry_quality"),
        "cost_realism_coverage": exp.get("cost_realism_coverage"),
        "high_cost_entry_rate": exp.get("high_cost_entry_rate"),
        "low_quality_entry_rate": exp.get("low_quality_entry_rate"),
        "actionable_bar_rate": exp.get("actionable_bar_rate"),
        "risk_off_bar_rate": exp.get("risk_off_bar_rate"),
    }


def _exp_to_prompt_record(exp: dict[str, Any]) -> dict[str, Any] | None:
    """Convert any experiment (kept or reverted) to a prompt history record."""
    try:
        score = _safe_score(exp)
    except Exception:
        return None
    if score <= -999:
        # Skip experiments that never produced a score (e.g. syntax errors)
        if exp.get("error"):
            return {
                "experiment_id": exp.get("experiment_id"),
                "kept": False,
                "score": None,
                "error": str(exp.get("error", ""))[:200],
                "change_summary": exp.get("change_summary", "unknown"),
                "failure_type": exp.get("failure_type", "error"),
            }
        return None
    return {
        "experiment_id": exp.get("experiment_id"),
        "kept": bool(exp.get("kept")),
        "score": score,
        "profit_factor": exp.get("profit_factor"),
        "trades_per_day": exp.get("trades_per_day"),
        "trade_sharpe": exp.get("trade_sharpe"),
        "change_summary": exp.get("change_summary", "unknown"),
        "error": str(exp.get("error", ""))[:200] if exp.get("error") else "",
        "failure_type": exp.get("failure_type", "none"),
        "keep_block_reason": exp.get("keep_block_reason", ""),
    }


def _promoted_to_prompt_record(event: dict[str, Any]) -> dict[str, Any] | None:
    try:
        score = float(event.get("score"))
    except Exception:
        return None
    run_name = str(event.get("run_name", "promoted"))
    exp_id = event.get("experiment_id")
    exp_label = f"{run_name}#{exp_id}" if exp_id is not None else run_name
    return {
        "experiment_id": exp_label,
        "run_name": run_name,
        "timestamp": event.get("timestamp"),
        "kept": True,
        "score": score,
        "profit_factor": event.get("profit_factor"),
        "trades_per_day": event.get("trades_per_day"),
        "trade_sharpe": event.get("trade_sharpe"),
        "stop_loss_rate": event.get("stop_loss_rate"),
        "worst_chunk_pf": event.get("worst_chunk_pf"),
        "direction_collapse_pct": event.get("direction_collapse_pct"),
        "change_summary": event.get("change_summary") or "promoted baseline",
        "error": "",
    }


def load_promoted_history() -> list[dict[str, Any]]:
    """Load prompt history strictly from promoted events."""
    if not os.path.exists(PROMOTED_HISTORY_JSONL):
        return []
    history: list[dict[str, Any]] = []
    with open(PROMOTED_HISTORY_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = _promoted_to_prompt_record(event)
            if rec is not None:
                history.append(rec)
    return history


def record_promotion_event(exp: dict[str, Any]) -> dict[str, Any]:
    """Append promotion ledger and advance promoted pointer atomically."""
    event = _promotion_event_from_exp(exp)
    _append_jsonl(PROMOTED_HISTORY_JSONL, event)
    _write_text_atomic(PROMOTED_CURRENT_TXT, f"{RUN_NAME}\n")
    return event


# --- Lab Notebook (persistent experiment context across runs) ---

_LAB_NOTEBOOK_SKELETON = """\
# Lab Notebook

## System
SPX 0DTE | 2-head gate+dir | 32 features (v2) | 5-dim position state | 8 actions | 1-min bars | learned exits (no hardcoded TP)

## Best Runs
| Run | Score | PF | TPD | Key Change |
|-----|-------|----|-----|------------|

## Dead Ends
| Change | Result | Why |
|--------|--------|-----|
"""


def load_lab_notebook() -> str:
    """Load lab notebook contents, or return empty string if not found."""
    if not os.path.exists(LAB_NOTEBOOK_MD):
        return ""
    with open(LAB_NOTEBOOK_MD, "r") as f:
        return f.read()


def _parse_notebook_sections(content: str) -> dict[str, list[str]]:
    """Parse notebook into {section_name: [lines]}."""
    sections: dict[str, list[str]] = {}
    current_section = None
    for line in content.split("\n"):
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections[current_section] = []
        elif current_section is not None:
            stripped = line.strip()
            if stripped:
                sections[current_section].append(line)
    return sections


def _rebuild_notebook(sections: dict[str, list[str]]) -> str:
    """Rebuild notebook markdown from parsed sections."""
    parts = ["# Lab Notebook\n"]
    # Core sections in fixed order
    core_order = [
        "System", "Causal Exit Labels", "Account-Aware Scoring",
        "What Works (proven across 95 experiments)", "What Fails (do NOT retry)",
        "Current Best Model (exp-52)", "Best Runs", "Dead Ends", "Next Priorities",
    ]
    seen = set()
    for header in core_order:
        if header in sections:
            parts.append(f"## {header}")
            for line in sections[header]:
                parts.append(line)
            parts.append("")
            seen.add(header)
    # Any remaining sections
    for header, lines in sections.items():
        if header not in seen:
            parts.append(f"## {header}")
            for line in lines:
                parts.append(line)
            parts.append("")
    return "\n".join(parts) + "\n"


def update_lab_notebook(exp: dict[str, Any], reasoning: str) -> None:
    """Append a kept experiment to the lab notebook's Best Runs table."""
    exp_label = f"{RUN_NAME}#{exp.get('experiment_id', '?')}"
    score = _safe_score(exp)
    pf = exp.get("profit_factor", "?")
    tpd = exp.get("trades_per_day", "?")
    change = (exp.get("change_summary") or reasoning or "unknown")[:80]

    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
    pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
    tpd_str = f"{tpd:.1f}" if isinstance(tpd, (int, float)) else str(tpd)
    new_row = f"| {exp_label} | {score_str} | {pf_str} | {tpd_str} | {change} |"

    if os.path.exists(LAB_NOTEBOOK_MD):
        with open(LAB_NOTEBOOK_MD, "r") as f:
            content = f.read()
    else:
        content = _LAB_NOTEBOOK_SKELETON

    sections = _parse_notebook_sections(content)

    best_runs = sections.get("Best Runs", [])

    # Deduplicate
    if any(exp_label in line for line in best_runs):
        return

    # Keep table header rows, add new data row
    header_rows = [r for r in best_runs if r.strip().startswith("|") and ("---" in r or "Run" in r)]
    data_rows = [r for r in best_runs if r.strip().startswith("|") and "---" not in r and "Run" not in r]
    data_rows.append(new_row)

    # Keep only most recent entries
    if len(data_rows) > LAB_NOTEBOOK_MAX_ENTRIES:
        data_rows = data_rows[-LAB_NOTEBOOK_MAX_ENTRIES:]

    sections["Best Runs"] = header_rows + data_rows

    dead_ends = sections.get("Dead Ends", [])
    de_header = [r for r in dead_ends if r.strip().startswith("|") and ("---" in r or "Change" in r)]
    de_data = [r for r in dead_ends if r.strip().startswith("|") and "---" not in r and "Change" not in r]
    if len(de_data) > LAB_NOTEBOOK_DEAD_ENDS_MAX:
        de_data = de_data[-LAB_NOTEBOOK_DEAD_ENDS_MAX:]
    sections["Dead Ends"] = de_header + de_data

    rebuilt = _rebuild_notebook(sections)
    _write_text_atomic(LAB_NOTEBOOK_MD, rebuilt)


def update_lab_notebook_dead_end(exp: dict[str, Any]) -> None:
    """Append a rejected/failed experiment to the lab notebook's Dead Ends table."""
    change = (exp.get("change_summary") or exp.get("reasoning") or "unknown")[:60]
    score = _safe_score(exp)
    failure_type = exp.get("failure_type", "unknown")
    keep_block = exp.get("keep_block_reason", "")
    error = (exp.get("error") or "")[:80]

    # Build concise result + reason
    if score <= -999:
        result_str = f"FAIL ({failure_type})"
    else:
        result_str = f"score={score:.2f}"
    why = keep_block or error or failure_type
    why = why[:80]

    new_row = f"| {change} | {result_str} | {why} |"

    if os.path.exists(LAB_NOTEBOOK_MD):
        with open(LAB_NOTEBOOK_MD, "r") as f:
            content = f.read()
    else:
        content = _LAB_NOTEBOOK_SKELETON

    sections = _parse_notebook_sections(content)

    dead_ends = sections.get("Dead Ends", [])

    # Deduplicate by change summary
    if any(change in line for line in dead_ends):
        return

    header_rows = [r for r in dead_ends if r.strip().startswith("|") and ("---" in r or "Change" in r)]
    data_rows = [r for r in dead_ends if r.strip().startswith("|") and "---" not in r and "Change" not in r]
    data_rows.append(new_row)

    # Keep only most recent entries
    if len(data_rows) > LAB_NOTEBOOK_DEAD_ENDS_MAX:
        data_rows = data_rows[-LAB_NOTEBOOK_DEAD_ENDS_MAX:]

    sections["Dead Ends"] = header_rows + data_rows
    rebuilt = _rebuild_notebook(sections)
    _write_text_atomic(LAB_NOTEBOOK_MD, rebuilt)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def prompt_contract_check(system_prompt: str, program_md: str) -> dict[str, Any]:
    text = f"{system_prompt}\n\n{program_md}"
    required_checks = [
        ("two_head_output_contract", r"two-head"),
        (
            "six_direction_semantics",
            r"CALL_ATM.*CALL_OTM5.*CALL_OTM10.*PUT_ATM.*PUT_OTM5.*PUT_OTM10",
        ),
        ("eight_action_semantics", r"8 effective actions"),
        ("feature_contract_phase_lock_32", r"32 features"),
    ]
    forbidden_checks = [
        ("stale_four_action_semantics", r"4 effective actions"),
        ("stale_two_direction_head", r"Direction head:\s*\(batch,\s*2\)"),
    ]

    violations: list[dict[str, str]] = []
    for rule_id, pattern in required_checks:
        if _re.search(pattern, text, _re.IGNORECASE | _re.DOTALL) is None:
            violations.append({"rule": rule_id, "detail": f"Missing required pattern: {pattern}"})
    for rule_id, pattern in forbidden_checks:
        if _re.search(pattern, text, _re.IGNORECASE | _re.DOTALL):
            violations.append({"rule": rule_id, "detail": f"Forbidden pattern present: {pattern}"})
    for key in REQUIRED_OUTPUT_METRIC_KEYS:
        if _re.search(rf"{_re.escape(key)}\s*:", text) is None:
            violations.append({"rule": f"missing_metric_key_{key}", "detail": f"Missing output metric key: {key}"})

    checksum_payload = {
        "required_checks": [r for r, _ in required_checks],
        "forbidden_checks": [r for r, _ in forbidden_checks],
        "required_output_metric_keys": list(REQUIRED_OUTPUT_METRIC_KEYS),
        "program_md_fingerprint": _sha256_text(program_md),
    }
    checksum = _sha256_text(json.dumps(checksum_payload, sort_keys=True))
    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "checksum": checksum,
        "checksum_short": checksum[:12],
    }


def detect_anomaly_flags(metrics: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    t = OBSERVABILITY_CONFIG["anomaly_thresholds"]
    do_nothing = float(metrics.get("do_nothing_pct", 1.0))
    exit_pct = float(metrics.get("exit_pct", 0.0))
    trades_per_day = float(metrics.get("trades_per_day", 0.0))
    num_trades = int(metrics.get("num_trades", 0))
    win_rate = float(metrics.get("win_rate", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    cost_coverage = float(metrics.get("cost_realism_coverage", 0.0))
    avg_entry_quality = float(metrics.get("avg_entry_quality", 0.0))
    high_cost_entry_rate = float(metrics.get("high_cost_entry_rate", 0.0))
    low_quality_entry_rate = float(metrics.get("low_quality_entry_rate", 0.0))
    actionable_bar_rate = float(metrics.get("actionable_bar_rate", 0.0))

    if do_nothing <= float(t["do_nothing_zero_max"]):
        flags.append("do_nothing_zero")
    if exit_pct >= float(t["exit_pct_extreme_min"]):
        flags.append("exit_pct_extreme")
    if num_trades > 0 and (trades_per_day < float(t["trades_per_day_extreme_min"]) or trades_per_day > float(t["trades_per_day_extreme_max"])):
        flags.append("trades_per_day_extreme")
    if num_trades > 0 and "cost_realism_coverage" in metrics:
        if cost_coverage < float(t["cost_realism_coverage_min"]):
            flags.append("cost_realism_low_coverage")
    if num_trades > 0 and "avg_entry_quality" in metrics:
        if avg_entry_quality < float(t["avg_entry_quality_min"]):
            flags.append("entry_quality_too_low")
    if num_trades > 0 and "high_cost_entry_rate" in metrics:
        if high_cost_entry_rate > float(t["high_cost_entry_rate_max"]):
            flags.append("high_cost_entry_rate_high")
    if num_trades > 0 and "low_quality_entry_rate" in metrics:
        if low_quality_entry_rate > float(t["low_quality_entry_rate_max"]):
            flags.append("low_quality_entry_rate_high")
    if num_trades > 0 and "actionable_bar_rate" in metrics:
        if actionable_bar_rate < float(t["actionable_bar_rate_min"]):
            flags.append("actionable_bar_rate_low")

    inconsistent = False
    if num_trades > 0 and trades_per_day <= 0:
        inconsistent = True
    if not (0.0 <= win_rate <= 1.0):
        inconsistent = True
    if profit_factor < 0:
        inconsistent = True
    if float(metrics.get("stop_loss_rate", 0.0)) > 1.0:
        inconsistent = True
    if inconsistent:
        flags.append("metric_inconsistent")

    return sorted(set(flags))


def validate_experiment_v2_schema(exp: dict[str, Any]) -> list[str]:
    required = [
        "schema_version",
        "experiment_id",
        "timestamp",
        "prompt_fingerprint",
        "program_md_fingerprint",
        "train_py_before_hash",
        "train_py_after_hash",
        "data_fingerprint",
        "model_fingerprint_before",
        "model_fingerprint_after",
        "failure_type",
        "anomaly_flags",
        "kept",
        "score",
    ]
    errs: list[str] = []
    for k in required:
        if k not in exp:
            errs.append(f"missing:{k}")
    if exp.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version:{exp.get('schema_version')}")
    failure_type = str(exp.get("failure_type"))
    if failure_type not in FAILURE_TYPE_ENUM:
        errs.append(f"failure_type:{failure_type}")
    if not isinstance(exp.get("anomaly_flags", []), list):
        errs.append("anomaly_flags:not_list")
    return errs


def _critical_anomaly_flags(flags: list[str]) -> list[str]:
    critical = OBSERVABILITY_CONFIG["critical_anomaly_flags"]
    return [f for f in flags if f in critical]


def _passes_near_tie_stability(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    cfg = OBSERVABILITY_CONFIG["near_tie_stability"]
    reasons: list[str] = []
    if float(metrics.get("worst_chunk_pf", 0.0)) < float(cfg["worst_chunk_pf_min"]):
        reasons.append("worst_chunk_pf_below_threshold")
    if float(metrics.get("stop_loss_rate", 1.0)) > float(cfg["stop_loss_rate_max"]):
        reasons.append("stop_loss_rate_above_threshold")
    if float(metrics.get("direction_collapse_pct", 1.0)) > float(cfg["direction_collapse_pct_max"]):
        reasons.append("direction_collapse_above_threshold")
    if "cost_realism_coverage" in metrics and float(metrics.get("cost_realism_coverage", 0.0)) < float(cfg["cost_realism_coverage_min"]):
        reasons.append("cost_realism_coverage_below_threshold")
    if "avg_entry_quality" in metrics and float(metrics.get("avg_entry_quality", 0.0)) < float(cfg["avg_entry_quality_min"]):
        reasons.append("avg_entry_quality_below_threshold")
    if "high_cost_entry_rate" in metrics and float(metrics.get("high_cost_entry_rate", 1.0)) > float(cfg["high_cost_entry_rate_max"]):
        reasons.append("high_cost_entry_rate_above_threshold")
    if "low_quality_entry_rate" in metrics and float(metrics.get("low_quality_entry_rate", 1.0)) > float(cfg["low_quality_entry_rate_max"]):
        reasons.append("low_quality_entry_rate_above_threshold")
    return len(reasons) == 0, reasons


def detect_strategy_collapse(history: list[dict[str, Any]], window: int = 8) -> dict[str, Any]:
    """Detect when the agent has collapsed to a fixed strategy or declining ambition.

    Checks the last `window` experiments for:
    1. Repetitive changes: >60% of summaries share the same leading keyword
    2. Zero accept rate: no experiments kept in the window
    3. Score plateau: all scored experiments within 0.01 of each other

    Returns: {"collapsed": bool, "signals": [...], "message": str}
    """
    recent = history[-window:] if len(history) >= window else history
    if len(recent) < 4:
        return {"collapsed": False, "signals": [], "message": ""}

    signals: list[str] = []

    # 1. Repetitive change summaries (first word clustering)
    summaries = [str(e.get("change_summary", "")).strip().lower() for e in recent if e.get("change_summary")]
    if len(summaries) >= 4:
        first_words = [s.split()[0] if s.split() else "" for s in summaries]
        from collections import Counter
        word_counts = Counter(first_words)
        most_common_word, most_common_count = word_counts.most_common(1)[0]
        if most_common_count / len(first_words) > 0.6:
            signals.append(f"repetitive_changes ({most_common_count}/{len(first_words)} start with '{most_common_word}')")

    # 2. Zero accept rate in window
    scored = [e for e in recent if _safe_score(e) > -999]
    kept_in_window = [e for e in recent if e.get("kept")]
    if len(scored) >= 4 and not kept_in_window:
        signals.append(f"zero_accept_rate (0/{len(scored)} kept in last {len(recent)})")

    # 3. Score plateau
    scores = [_safe_score(e) for e in scored if _safe_score(e) > -999]
    if len(scores) >= 4:
        score_range = max(scores) - min(scores)
        if score_range < 0.01:
            signals.append(f"score_plateau (range={score_range:.4f} in last {len(scores)})")

    collapsed = len(signals) >= 2
    message = ""
    if signals:
        message = (
            f"Strategy collapse warning ({len(signals)} signals): "
            + "; ".join(signals)
        )

    return {"collapsed": collapsed, "signals": signals, "message": message}


# ---------------------------------------------------------------------------
# Diagnostics — capture what kills the process and track resources
# ---------------------------------------------------------------------------

_DIAG_FILE = os.path.join(RUN_DIR, "diagnostics.log")


def _diag(msg):
    """Append a timestamped diagnostic line to diagnostics.log."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(_DIAG_FILE, 'a') as f:
            f.write(line)
    except Exception:
        pass  # best-effort


def _signal_handler(signum, frame):
    """Catch signals so we know what killed the process."""
    name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    _diag(f"SIGNAL RECEIVED: {name} (signum={signum})")
    log(f"SIGNAL: {name} — writing diagnostics and exiting")
    log_diagnostics("signal_death")
    # Re-raise for clean exit
    sys.exit(128 + signum)


def install_signal_handlers():
    """Install signal handlers to catch SIGTERM, SIGHUP, etc."""
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass  # can't catch some signals
    _diag("Signal handlers installed")


def log_diagnostics(phase: str = "check"):
    """Log RSS, cgroup memory, PID count, and disk usage."""
    parts = [f"phase={phase}"]

    # RSS of this process
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes, Linux returns KB
        if sys.platform == 'darwin':
            rss_mb = rss_kb / (1024 * 1024)
        else:
            rss_mb = rss_kb / 1024
        parts.append(f"rss_mb={rss_mb:.1f}")
    except Exception:
        pass

    # Current RSS from /proc (more accurate on Linux)
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts.append(f"vmrss_kb={line.split()[1]}")
                elif line.startswith('VmSize:'):
                    parts.append(f"vmsize_kb={line.split()[1]}")
    except FileNotFoundError:
        pass

    # cgroup v2 memory
    try:
        with open('/sys/fs/cgroup/memory.current', 'r') as f:
            cgroup_bytes = int(f.read().strip())
            parts.append(f"cgroup_mb={cgroup_bytes / (1024*1024):.0f}")
    except FileNotFoundError:
        pass

    # cgroup v1 memory (fallback)
    try:
        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
            cgroup_bytes = int(f.read().strip())
            parts.append(f"cgroup_mb={cgroup_bytes / (1024*1024):.0f}")
    except FileNotFoundError:
        pass

    # Number of processes in the container
    try:
        import glob
        pids = glob.glob('/proc/[0-9]*')
        parts.append(f"pids={len(pids)}")
    except Exception:
        pass

    # Disk usage
    try:
        st = os.statvfs('/root')
        disk_used_gb = (st.f_blocks - st.f_bavail) * st.f_frsize / (1024**3)
        disk_free_gb = st.f_bavail * st.f_frsize / (1024**3)
        parts.append(f"disk_used_gb={disk_used_gb:.1f}")
        parts.append(f"disk_free_gb={disk_free_gb:.1f}")
    except Exception:
        pass

    msg = " | ".join(parts)
    _diag(msg)
    log(f"  DIAG: {msg}")


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

class TruncatedResponseError(Exception):
    """Raised when Claude's response was cut off by the max_tokens limit."""
    pass


def call_claude(system_prompt, user_prompt: str) -> str:
    """Call Claude API with prompt caching and return the text response.

    system_prompt can be:
      - str: plain system prompt (no caching)
      - list[dict]: structured content blocks with cache_control markers
    """
    client = _get_anthropic_client()
    # Use streaming to avoid 10-minute timeout on slow models (Opus)
    collected_text = []
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31,output-128k-2025-02-19"},
    ) as stream:
        for text in stream.text_stream:
            collected_text.append(text)
    response = stream.get_final_message()
    if response.stop_reason == "max_tokens":
        raise TruncatedResponseError(
            f"Response truncated at {MAX_TOKENS} tokens. "
            f"Code was cut off — increase MAX_TOKENS or reduce train.py size."
        )
    # Always log cache performance so we can confirm caching works
    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    log(f"  Tokens: input={input_tokens}, cache_read={cache_read}, cache_create={cache_create}")
    return response.content[0].text


def _prefetch_state_hash(train_py_content: str, history_len: int) -> str:
    """Hash for detecting whether a prefetched response is still valid."""
    return _sha256_text(f"{train_py_content}:{history_len}")[:16]


def submit_prefetch(experiment_id: int, system_blocks: list[dict],
                    user_prompt: str, state_hash: str) -> None:
    """Submit a speculative Claude API call for the next experiment in background."""
    global _prefetched_responses
    if _prefetch_executor is None:
        return

    def _do_prefetch():
        try:
            response_text = call_claude(system_blocks, user_prompt)
            _prefetched_responses[experiment_id] = {
                "response": response_text,
                "state_hash": state_hash,
            }
            log(f"  [prefetch] Response ready for experiment #{experiment_id}")
        except Exception as e:
            log(f"  [prefetch] Failed for experiment #{experiment_id}: {e}")
            # Silently fail — caller will fall back to synchronous call

    # Cancel any stale prefetch for a different experiment
    _prefetched_responses.pop(experiment_id, None)
    try:
        _prefetch_executor.submit(_do_prefetch)
    except Exception:
        pass  # Thread pool shutting down, etc.


def consume_prefetch(experiment_id: int, current_state_hash: str) -> str | None:
    """Check if a valid prefetched response exists for this experiment.

    Returns the response text if valid, None if stale or missing.
    """
    entry = _prefetched_responses.pop(experiment_id, None)
    if entry is None:
        return None
    log(f"  [prefetch] Using prefetched response for #{experiment_id}")
    return entry["response"]


def build_system_prompt(program_md: str, lab_notebook: str = "") -> list[dict]:
    """Build structured system prompt with cache breakpoints.

    Returns a list of content blocks for the Anthropic API.
    The static instruction block and program.md are cached (they never change
    within a run). The lab notebook is cached separately (it changes only
    when an experiment is kept).
    """
    instructions = """\
You are an expert ML researcher running inside an autonomous experiment loop.
Your only editable target is `train.py`.

Mission:
- Improve the scalar training objective reported by the script (higher score is better).
- Propose one coherent hypothesis per iteration, not an unfocused rewrite.
- Keep changes robust under a fixed wall-clock budget.

Output contract (strict):
- First output a BRIEF hypothesis in <reasoning>...</reasoning> (max 2-3 sentences).
- Then output the COMPLETE modified `train.py` — every line, top to bottom.
- Output only reasoning tags + Python code (no markdown fences, no extra commentary).
- CRITICAL: The file is ~800 lines. You MUST output the entire file without truncation.
  Keep your reasoning SHORT to leave room for the full code.

Execution constraints:
- Do not change files other than `train.py`.
- Preserve parseable metric output keys.
- Prefer small, testable deltas if history shows instability.
- Do NOT repeat experiments from the Lab Notebook — read the Improvements Log and Dead Ends carefully.

## Evaluation Mechanics (from evaluate_trades)
- Trades enter at the model's signal bar using actual option mid-prices
- Stop-loss: -30% of premium (position closed immediately)
- Max hold: 60 bars (60 minutes)
- Cooldown: 5 bars after a stop-loss before next entry
- No trading before bar 30 (first 30 minutes of session)
- Transaction cost: deducted from P&L using per-bar sidecar cost data (avg ~200 bps round-trip)
- P&L capped at +200% to prevent lottery-ticket dependency
- Only one position at a time (no stacking)

The authoritative domain and strategy guidance is below.
If any instruction you infer conflicts with this guidance, follow this guidance."""

    blocks = [
        # Block 1: Static instructions + program.md (cached — identical every call in a run)
        {
            "type": "text",
            "text": f"{instructions}\n\n{program_md}",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    ]

    # Block 2: Lab notebook (cached separately — changes only on kept experiments)
    if lab_notebook:
        blocks.append({
            "type": "text",
            "text": (
                "\n\n## Lab Notebook (persistent learning across runs)\n"
                "Use this to understand what has been tried, what worked, and what to avoid.\n"
                "Do NOT re-try approaches listed in Dead Ends.\n"
                "Build on approaches in the Improvements Log — extend what worked.\n\n"
                + lab_notebook
            ),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        })

    return blocks


def build_user_prompt(current_train_py: str, history: list, experiment_id: int = 0) -> str:
    parts = []

    if history:
        best = max(history, key=lambda e: _safe_score(e))
        kept = [e for e in history if e.get("kept")]
        failed = [e for e in history if e.get("error")]
        scored = [e for e in history if _safe_score(e) > -999]
        accept_rate = len(kept) / len(scored) if scored else 0.0
        parts.append("## Experiment Summary")
        parts.append(
            f"Total={len(history)} | Kept={len(kept)} | Failed={len(failed)} | "
            f"Accept rate={accept_rate:.0%} ({len(kept)}/{len(scored)}) | "
            f"Best score={_safe_score(best)} (#{best['experiment_id']})\n"
        )

        # Strategy collapse detection
        collapse = detect_strategy_collapse(history)
        if collapse["signals"]:
            parts.append("## STRATEGY COLLAPSE WARNING")
            parts.append(collapse["message"])
            if collapse["collapsed"]:
                parts.append("The search has COLLAPSED. You MUST try a fundamentally different approach:")
                parts.append("  - Change the loss function structure, not just weights")
                parts.append("  - Add or remove a training technique (curriculum, augmentation)")
                parts.append("  - Rethink which feature groups matter most")
                parts.append("  - Try the OPPOSITE of what recent experiments attempted")
            parts.append("")

        # Low accept rate warning
        if len(scored) >= 6 and accept_rate < 0.15:
            parts.append("## LOW ACCEPT RATE WARNING")
            parts.append(f"Only {accept_rate:.0%} of proposals are being accepted. This means")
            parts.append("your proposals and the validation gate are misaligned.")
            parts.append("  - Make SMALLER, more conservative changes")
            parts.append("  - Focus on what the gate actually rewards (read the score formula)")
            parts.append("  - Study the gap between best score and recent scores\n")

        top_kept = sorted(
            kept,
            key=lambda e: _safe_score(e),
            reverse=True,
        )[:8]
        if top_kept:
            parts.append("## Top Kept Experiments")
            for exp in top_kept:
                score = _safe_score(exp)
                pf = exp.get("profit_factor", "?")
                tpd = exp.get("trades_per_day", "?")
                reason = exp.get("change_summary", "unknown")
                parts.append(f"  #{exp['experiment_id']}: score={score} pf={pf} tpd={tpd} — {reason}")
            parts.append("")

        recent = history[-30:]
        parts.append("## Recent Trajectory (last 30)")
        for exp in recent:
            status = "✓ KEPT" if exp.get("kept") else "✗ reverted"
            score = exp.get("score", exp.get("val_sharpe", "N/A"))
            reason = exp.get("change_summary", "unknown")
            trades = exp.get("trades_per_day", "?")
            pf = exp.get("profit_factor", "?")
            err = exp.get("error", "")
            if err and err.startswith("SAFETY:"):
                parts.append(f"  #{exp['experiment_id']}: REJECTED — {err}")
            elif err:
                parts.append(f"  #{exp['experiment_id']}: FAILED — {err[:150]}")
            else:
                parts.append(f"  #{exp['experiment_id']}: score={score} pf={pf} tpd={trades} [{status}] — {reason}")
        parts.append("")

        if history[-1].get("error"):
            parts.append(f"## LAST EXPERIMENT FAILED:\n{history[-1]['error']}\n")
            parts.append("Fix the error and try a different approach.\n")

        # Diversity nudge every 5th experiment
        if experiment_id > 0 and experiment_id % 5 == 0:
            parts.append("## DIVERSITY NUDGE")
            parts.append("This is every 5th experiment — try something FUNDAMENTALLY DIFFERENT.")
            parts.append("Don't make incremental tweaks. Instead, try a completely new approach:")
            parts.append("  - A different loss function structure")
            parts.append("  - A novel architectural component")
            parts.append("  - A different training strategy (curriculum, scheduling)")
            parts.append("  - Leveraging feature groups you haven't used yet (Greeks, OTM skew)")
            parts.append("Look at what has been tried in the history above and explore the OPPOSITE direction.\n")

        # Search-space review every 10th experiment (article: "is the search space still right?")
        if experiment_id > 0 and experiment_id % 10 == 0:
            parts.append("## SEARCH-SPACE REVIEW (every 10th experiment)")
            parts.append("Before proposing your next change, answer these questions:")
            parts.append("  1. Is the agent still exploring, or has it collapsed to a fixed strategy?")
            parts.append("  2. Are the constraints correct? Is score rewarding the right behavior?")
            parts.append("  3. What's the BIGGEST gap between current performance and the goal?")
            parts.append("  4. Which feature groups are undertested? Which loss components are undertested?")
            parts.append("  5. Are recent proposals getting more conservative/smaller? If so, go bigger.")
            parts.append("Write your analysis in <reasoning>, then propose a change that addresses the biggest gap.\n")
    else:
        # Check if the lab notebook has prior improvements (loaded into system prompt)
        if os.path.exists(LAB_NOTEBOOK_MD):
            nb = load_lab_notebook()
            if "## Improvements Log" in nb and nb.split("## Improvements Log")[1].strip().split("##")[0].strip():
                parts.append("## This is the first experiment of a NEW RUN, but you have prior context.")
                parts.append("Review the Lab Notebook in the system prompt — it contains your history of")
                parts.append("improvements and dead ends from previous runs. Do NOT run baseline as-is.")
                parts.append("Instead, propose a meaningful improvement based on what you learned.\n")
            else:
                parts.append("## This is the FIRST experiment. Run baseline as-is or make one small improvement.\n")
        else:
            parts.append("## This is the FIRST experiment. Run baseline as-is or make one small improvement.\n")

    # Reinforce current best right before the code — prevents hallucination
    # about which experiment is best (observed in exp-17 where Claude referenced
    # #12 as best despite #16 being promoted).
    if history:
        best = max(history, key=lambda e: _safe_score(e))
        parts.append(
            f"## CURRENT BEST: Experiment #{best['experiment_id']} "
            f"(score={_safe_score(best):.4f}, pf={best.get('profit_factor', '?')})\n"
            f"The code below IS experiment #{best['experiment_id']}. "
            f"Build on THIS code — it is the current best.\n"
        )

    parts.append("## Current train.py:\n```python\n" + current_train_py + "\n```\n")
    parts.append("Remember: First write <reasoning>your hypothesis</reasoning>, then output the complete modified train.py code.")

    return "\n".join(parts)


def build_repair_prompt(failed_candidate: str, failure_type: str, failure_error: str, attempt: int) -> str:
    err = (failure_error or "").strip()
    if len(err) > 1600:
        err = err[:1600] + "\n...[truncated]..."
    return f"""The previous candidate train.py failed and must be repaired.

Repair attempt: {attempt}
Failure type: {failure_type}
Failure excerpt:
{err}

Requirements:
- Return a COMPLETE corrected train.py file.
- Keep the two-head/32-feature contract intact.
- Fix the concrete failure first; make minimal additional edits.
- No markdown fences, no prose outside <reasoning>...</reasoning>.

Failed candidate train.py:
```python
{failed_candidate}
```

Output format:
<reasoning>short fix plan</reasoning>
<full python file>
"""


import re as _re


def extract_reasoning(response: str) -> str:
    """Extract reasoning from <reasoning>...</reasoning> tags."""
    m = _re.search(r'<reasoning>(.*?)</reasoning>', response, _re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def extract_code(response: str) -> str:
    """Extract Python code from LLM response, handling reasoning tags and markdown fences."""
    code = response.strip()
    # Remove <reasoning>...</reasoning> block if present
    code = _re.sub(r'<reasoning>.*?</reasoning>', '', code, flags=_re.DOTALL).strip()
    # Strip markdown code fences if present
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def extract_change_summary(response: str, old_code: str, new_code: str) -> str:
    """Generate a brief summary of what changed by diffing full files."""
    old_lines = old_code.split('\n')
    new_lines = new_code.split('\n')

    # Find changed lines (full file, not just first 60)
    old_set = set(old_lines)
    new_set = set(new_lines)

    added = new_set - old_set
    removed = old_set - new_set

    # Focus on meaningful changes (skip blank lines, comments, docstrings)
    def _meaningful(line):
        s = line.strip()
        return (s and not s.startswith('#') and not s.startswith('"""')
                and not s.startswith("'''") and len(s) > 3)

    meaningful_added = [l.strip() for l in added if _meaningful(l)]
    meaningful_removed = [l.strip() for l in removed if _meaningful(l)]

    # Detect hyperparameter changes (FOO = value lines)
    hp_changes = []
    for line in meaningful_added:
        if _re.match(r'^[A-Z_]+ = ', line):
            hp_changes.append(line)

    # Detect new class/function definitions
    new_defs = [l for l in meaningful_added if l.startswith('class ') or l.startswith('def ')]

    # Build summary
    parts = []
    if hp_changes:
        parts.extend(hp_changes[:3])
    if new_defs:
        parts.extend(new_defs[:2])
    if not parts and meaningful_added:
        parts.extend(meaningful_added[:3])

    if parts:
        return "; ".join(parts)
    if meaningful_removed and not meaningful_added:
        return "removed code"
    return "architecture change" if (len(added) + len(removed)) > 5 else "minor change"


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------

def validate_syntax(code: str) -> str | None:
    """Return None if valid, error string if not."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e}"


# Auto-strip patterns: removable without breaking the rest of the code
_COMPILE_RE = _re.compile(
    r'^([\t ]*)(?:\w+\s*=\s*)?torch\.compile\(.*\).*$', _re.MULTILINE
)


def _sanitize_code(code: str) -> tuple[str, list[str]]:
    """Auto-strip known dangerous patterns that are safely removable.
    Returns (cleaned_code, list_of_stripped_descriptions)."""
    stripped = []
    # Replace torch.compile lines with 'pass' at the same indent level
    # to avoid leaving empty if/else blocks
    if _COMPILE_RE.search(code):
        code = _COMPILE_RE.sub(lambda m: m.group(1) + "pass  # torch.compile removed", code)
        stripped.append("auto-stripped torch.compile")
    # Auto-fix common Claude typo: lookbook → lookback
    if _re.search(r'\blookbook\b', code, _re.IGNORECASE):
        code = _re.sub(r'\blookbook\b', 'lookback', code, flags=_re.IGNORECASE)
        stripped.append("auto-fixed lookbook→lookback typo")
    return code, stripped


# Patterns that WILL crash the 64Gi Akash container or break architecture
_DANGEROUS_PATTERNS = [
    # torch.compile is now auto-stripped above, not rejected
    (r'\bnn\.DataParallel\b', "DataParallel is forbidden (single GPU)"),
    (r'\bDistributedDataParallel\b', "DDP is forbidden (single GPU)"),
    (r'\btorch\.jit\.(trace|script)\b', "torch.jit is forbidden (unnecessary overhead)"),
    (r'BATCH_SIZE\s*=\s*(\d+)', None),  # checked separately below
    (r'D_MODEL\s*=\s*(\d+)', None),     # checked separately below
    (r'DEPTH\s*=\s*(\d+)', None),       # checked separately below
]

# Structural patterns: reject changes that break the two-head architecture
_STRUCTURAL_PATTERNS = [
    # Single combined head (merging gate+dir into one)
    (r'nn\.Linear\([^)]*,\s*NUM_ACTIONS\)', "Do NOT merge gate+dir into single head. Keep self.gate_head and self.dir_head separate."),
    (r'nn\.Linear\([^)]*,\s*4\)', "Do NOT merge gate+dir into single head outputting 4 logits. Keep two-head architecture."),
    # Reverting to old forward-return percentile loss
    (r'TRADE_LABEL_PERCENTILE', "Do NOT revert to forward-return percentile labels. Use option P&L from dataloader."),
]


def validate_safety(code: str) -> str | None:
    """Check Claude's output for dangerous patterns that crash the container
    or break the two-head architecture. Returns None if safe, error string if dangerous."""
    for pattern, msg in _DANGEROUS_PATTERNS:
        m = _re.search(pattern, code)
        if m and msg:
            return f"SAFETY: {msg}"

    # Structural checks: protect two-head architecture
    for pattern, msg in _STRUCTURAL_PATTERNS:
        m = _re.search(pattern, code)
        if m:
            return f"SAFETY: {msg}"

    # Check BATCH_SIZE
    m = _re.search(r'BATCH_SIZE\s*=\s*(\d+)', code)
    if m and int(m.group(1)) > 256:
        return f"SAFETY: BATCH_SIZE={m.group(1)} exceeds limit of 256"

    # Check D_MODEL
    m = _re.search(r'D_MODEL\s*=\s*(\d+)', code)
    if m and int(m.group(1)) > 128:
        return f"SAFETY: D_MODEL={m.group(1)} exceeds limit of 128"

    # Check DEPTH
    m = _re.search(r'DEPTH\s*=\s*(\d+)', code)
    if m and int(m.group(1)) > 8:
        return f"SAFETY: DEPTH={m.group(1)} exceeds limit of 8"

    # Check D_MODEL divisible by N_HEADS
    m_d = _re.search(r'D_MODEL\s*=\s*(\d+)', code)
    m_h = _re.search(r'N_HEADS\s*=\s*(\d+)', code)
    if m_d and m_h:
        d, h = int(m_d.group(1)), int(m_h.group(1))
        if h > 0 and d % h != 0:
            return f"SAFETY: D_MODEL={d} not divisible by N_HEADS={h}. Choose D_MODEL that divides evenly by N_HEADS."

    # Architecture locking — prevent changes to D_MODEL, DEPTH, N_HEADS.
    # Changing these breaks warm-start weight compatibility (tensor shape mismatch),
    # forcing the model to train from scratch in the limited experiment time budget.
    # Read locked values from best_train.py (the current best model's architecture).
    arch_err = validate_architecture_locked(code)
    if arch_err:
        return arch_err

    # Foundation lock: FEATURE_GROUPS must remain aligned to 60 features.
    feature_total = _extract_feature_groups_width(code)
    if feature_total is None:
        return (
            "SAFETY: Could not parse FEATURE_GROUPS as explicit literal ranges. "
            f"Keep FEATURE_GROUPS as a literal dict with max end index = {FEATURE_LOCK_COUNT}."
        )
    if feature_total != FEATURE_LOCK_COUNT:
        return (
            f"SAFETY: FEATURE_GROUPS covers {feature_total} features, expected {FEATURE_LOCK_COUNT}. "
            "Feature-count changes are blocked in this stabilization phase."
        )

    return None


def validate_architecture_locked(code: str) -> str | None:
    """Ensure D_MODEL, DEPTH, and N_HEADS match best_train.py.

    Changing these tensor-shape-determining hyperparameters invalidates
    warm-start weights, forcing training from scratch. With only ~5 min
    per experiment, that's not enough to converge on 32 features.
    """
    if not os.path.exists(BEST_TRAIN_PY):
        return None  # no best model yet, nothing to lock

    try:
        with open(BEST_TRAIN_PY, 'r') as f:
            best_code = f.read()
    except Exception:
        return None

    locked = {}
    for param in ('D_MODEL', 'DEPTH', 'N_HEADS'):
        m = _re.search(rf'{param}\s*=\s*(\d+)', best_code)
        if m:
            locked[param] = int(m.group(1))

    if not locked:
        return None

    for param, expected in locked.items():
        m = _re.search(rf'{param}\s*=\s*(\d+)', code)
        if m:
            actual = int(m.group(1))
            if actual != expected:
                return (f"SAFETY: {param}={actual} differs from locked value {expected} "
                        f"(best_train.py). Changing {param} breaks warm-start weight "
                        f"compatibility. Keep {param}={expected}.")

    return None


def _extract_feature_groups_width(code: str) -> int | None:
    """Return max FEATURE_GROUPS end index if parseable, otherwise None.

    This enforces the phase-locked 60-feature contract and catches 60/64 drift
    before training starts.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "FEATURE_GROUPS" for t in node.targets):
            continue
        try:
            groups = ast.literal_eval(node.value)
        except Exception:
            return None
        if not isinstance(groups, dict) or not groups:
            return None

        max_end: int | None = None
        for _, rng in groups.items():
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                return None
            try:
                start = int(rng[0])
                end = int(rng[1])
            except Exception:
                return None
            if start < 0 or end <= start:
                return None
            max_end = end if max_end is None else max(max_end, end)
        return max_end
    return None


def _reap_zombies():
    """Reap any zombie child processes to prevent PID accumulation.

    When PID 1 is not a proper init (e.g. tail -f /dev/null), zombie
    children from subprocess.run() or CUDA driver helpers can accumulate
    and eventually hit the PID namespace limit, causing the kubelet to
    kill the pod silently.
    """
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
        except ChildProcessError:
            break


def run_training(train_py_path: str, timeout: int = 420, time_budget: int = 300) -> dict:
    """Run train.py and parse the output metrics.

    Timeout: time_budget + 120s buffer for data loading/eval.
    Returns dict with metrics or {'error': 'message'}.
    Always includes 'wall_time' (seconds) for early-crash detection.
    """
    t_start = time.time()
    try:
        env = os.environ.copy()
        env["TIME_BUDGET"] = str(time_budget)
        result = subprocess.run(
            [PYTHON, "-u", train_py_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=SCRIPT_DIR,
            env=env,
        )
        wall_time = time.time() - t_start

        output = result.stdout + result.stderr

        if result.returncode != 0:
            # Get last 50 lines of output for error context (30 was too short, tracebacks got truncated)
            lines = output.strip().split('\n')
            tail = '\n'.join(lines[-50:])
            return {
                "error": f"Exit code {result.returncode}:\n{tail}",
                "error_type": "train_crash",
                "output": output,
                "wall_time": wall_time,
            }

        # Parse metrics from the --- section
        metrics = {"output": output}
        for line in output.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, _, val = line.partition(':')
                key = key.strip()
                val = val.strip()
                if key in ('score', 'val_sharpe', 'max_drawdown', 'annual_return',
                           'win_rate', 'profit_factor', 'avg_winner', 'avg_loser',
                           'trades_per_day', 'trade_sharpe', 'sortino', 'calmar',
                           'ev_per_trade', 'do_nothing_pct', 'exit_pct', 'total_return',
                           'peak_vram_mb', 'training_seconds', 'total_seconds',
                           'short_hold_pct', 'stop_loss_rate',
                           'final_capital', 'equity_sharpe',
                           'max_equity_dd', 'total_dollar_return',
                           'worst_chunk_pf', 'direction_collapse_pct',
                           'avg_entry_cost_bps', 'avg_entry_quality',
                           'cost_realism_coverage', 'high_cost_entry_rate',
                           'low_quality_entry_rate', 'actionable_bar_rate',
                           'risk_off_bar_rate',
                           'min_equity_frac', 'avg_risk_fraction',
                           'max_risk_fraction'):
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        pass
                elif key in ('num_trades', 'num_val_bars', 'num_val_days',
                             'num_steps', 'num_params', 'max_consec_loss',
                             'model_exit_count', 'cooldown_blocked',
                             'pre_10am_blocked',
                             'trades_blocked_by_balance'):
                    try:
                        metrics[key] = int(val.replace(',', ''))
                    except ValueError:
                        pass
                elif key in ('lookback', 'depth', 'd_model'):
                    metrics[key] = val
                elif key == 'hit_ruin':
                    metrics[key] = val.strip().lower() == 'true'

        if 'score' not in metrics:
            return {
                "error": f"Could not parse score from output:\n{output[-500:]}",
                "error_type": "parse",
                "output": output,
                "wall_time": wall_time,
            }

        parsed = sorted(k for k in metrics.keys() if k != "output")
        missing_required = [k for k in REQUIRED_OUTPUT_METRIC_KEYS if k not in metrics]
        metrics["parse_summary"] = {
            "parsed_metric_keys": parsed,
            "missing_required_metric_keys": missing_required,
        }
        metrics["wall_time"] = wall_time
        if missing_required:
            return {
                "error": (
                    "Missing required metric keys in train.py output: "
                    + ", ".join(missing_required)
                ),
                "error_type": "parse",
                "output": output,
                "parse_summary": metrics["parse_summary"],
                "wall_time": wall_time,
            }
        return metrics

    except subprocess.TimeoutExpired as e:
        wall_time = time.time() - t_start
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        merged = ""
        if stdout:
            merged += str(stdout)
        if stderr:
            merged += ("\n" + str(stderr)) if merged else str(stderr)
        return {
            "error": f"Training timed out after {timeout}s",
            "error_type": "timeout",
            "output": merged,
            "wall_time": wall_time,
        }
    except Exception as e:
        wall_time = time.time() - t_start
        return {"error": f"Exception: {e}", "error_type": "train_crash", "wall_time": wall_time}


def run_training_smoke(candidate_code: str, timeout: int, time_budget: int) -> dict[str, Any]:
    """Run a candidate in an isolated sandbox to catch obvious runtime crashes.

    This avoids corrupting the active workspace files while validating that
    the candidate can at least start and run briefly.
    """
    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="autoresearch-smoke-") as td:
            smoke_train = os.path.join(td, "train.py")
            smoke_prepare = os.path.join(td, "prepare.py")
            smoke_best_model = os.path.join(td, "best_model.pt")
            smoke_best_train = os.path.join(td, "best_train.py")

            with open(smoke_train, "w") as f:
                f.write(candidate_code)
            shutil.copy2(os.path.join(SCRIPT_DIR, "prepare.py"), smoke_prepare)
            if os.path.exists(BEST_MODEL_PT):
                shutil.copy2(BEST_MODEL_PT, smoke_best_model)
            if os.path.exists(BEST_TRAIN_PY):
                shutil.copy2(BEST_TRAIN_PY, smoke_best_train)

            env = os.environ.copy()
            env["TIME_BUDGET"] = str(time_budget)
            result = subprocess.run(
                [PYTHON, "-u", smoke_train],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
                env=env,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode != 0:
                lines = output.strip().split("\n")
                tail = "\n".join(lines[-50:])
                return {
                    "ok": False,
                    "error_type": "train_crash",
                    "error": f"Smoke crash (exit {result.returncode}):\n{tail}",
                    "output": output,
                }
            return {"ok": True, "output": output}
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        output = (str(stdout) if stdout else "") + (("\n" + str(stderr)) if stderr else "")
        return {
            "ok": False,
            "error_type": "timeout",
            "error": f"Smoke timed out after {timeout}s",
            "output": output,
        }
    except Exception as e:
        return {
            "ok": False,
            "error_type": "train_crash",
            "error": f"Smoke exception: {e}",
            "output": "",
        }


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

def load_history() -> list:
    """Load experiment history from THIS run folder only."""
    if not os.path.exists(RUN_EXPERIMENTS_V2_LOG):
        return []
    history = []
    with open(RUN_EXPERIMENTS_V2_LOG, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return history


def append_experiment_v2(exp: dict):
    """Append one experiment to the v2 JSONL log (strict schema)."""
    errors = validate_experiment_v2_schema(exp)
    if errors:
        raise ValueError(f"Invalid experiments.v2 record: {errors}")
    line = json.dumps(exp) + "\n"
    _ensure_dir(RUN_DIR)
    with open(RUN_EXPERIMENTS_V2_LOG, "a") as f:
        f.write(line)


def append_results_tsv(exp: dict):
    """Append one experiment to human-readable results.tsv (Karpathy-style)."""
    _ensure_dir(RUN_DIR)
    write_header = not os.path.exists(RUN_RESULTS_TSV)
    exp_id = exp.get("experiment_id", "?")
    score = _safe_score(exp)
    pf = exp.get("profit_factor", 0.0)
    tpd = exp.get("trades_per_day", 0.0)
    if exp.get("error"):
        status = "crash"
    elif exp.get("kept"):
        status = "keep"
    else:
        status = "discard"
    desc = (exp.get("change_summary") or "unknown")[:120].replace("\t", " ")
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
    pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
    tpd_str = f"{tpd:.1f}" if isinstance(tpd, (int, float)) else str(tpd)
    with open(RUN_RESULTS_TSV, "a") as f:
        if write_header:
            f.write("exp_id\tscore\tpf\ttpd\tstatus\tdescription\n")
        f.write(f"{exp_id}\t{score_str}\t{pf_str}\t{tpd_str}\t{status}\t{desc}\n")


def write_status(phase: str, experiment_id: int, best_score: float,
                 kept: int, failed: int, total: int, deadline: float,
                 last_exp: dict | None = None,
                 contract_checksum: str | None = None):
    """Write status.json for the monitor to read."""
    remaining = max(0, (deadline - time.time()) / 3600)
    # Accept rate: fraction of scored (non-failed) experiments that were kept
    scored = total - failed
    accept_rate = kept / scored if scored > 0 else 0.0
    status = {
        "phase": phase,
        "experiment_id": experiment_id,
        "best_score": round(best_score, 6) if best_score > -999 else None,
        "kept": kept,
        "failed": failed,
        "total": total,
        "accept_rate": round(accept_rate, 3),
        "time_remaining_h": round(remaining, 2),
        "updated": datetime.datetime.now().isoformat(),
    }
    if contract_checksum:
        status["contract_checksum"] = str(contract_checksum)[:12]
    if last_exp:
        status["last_change"] = last_exp.get("change_summary", "")
        status["last_score"] = last_exp.get("score", last_exp.get("val_sharpe"))
        status["last_kept"] = last_exp.get("kept", False)
        if "failure_type" in last_exp:
            status["last_failure_type"] = last_exp.get("failure_type")
        if "anomaly_flags" in last_exp:
            status["last_anomaly_flags"] = list(last_exp.get("anomaly_flags", []))
    # Atomic write to avoid partial reads
    tmp = STATUS_JSON + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, STATUS_JSON)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_one_experiment(experiment_id: int, prompt_history: list, best_score: float,
                       deadline: float, kept_count: int, failed_count: int, run_total: int,
                       time_budget: int = 300, data_fingerprint: str | None = None,
                       best_metrics: dict[str, Any] | None = None) -> dict:
    """Run a single experiment iteration. Returns experiment dict."""
    log(f"=== Experiment #{experiment_id} ===")
    total = run_total
    artifact_dir = _artifact_dir(experiment_id)

    # Read current state
    with open(PROGRAM_MD, 'r') as f:
        program_md = f.read()
    with open(TRAIN_PY, 'r') as f:
        current_code = f.read()

    program_md_fingerprint = _sha256_text(program_md)
    train_py_before_hash = _sha256_text(current_code)
    model_fingerprint_before = _sha256_file(BEST_MODEL_PT)

    exp: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "timestamp": _now_iso(),
        "program_md_fingerprint": program_md_fingerprint,
        "train_py_before_hash": train_py_before_hash,
        "train_py_after_hash": train_py_before_hash,
        "data_fingerprint": data_fingerprint,
        "model_fingerprint_before": model_fingerprint_before,
        "model_fingerprint_after": model_fingerprint_before,
        "failure_type": "none",
        "anomaly_flags": [],
        "kept": False,
        "score": -999.0,
    }

    _save_artifact_text(artifact_dir, "program.md", program_md)
    _save_artifact_text(artifact_dir, "train_before.py", current_code)

    lab_notebook = load_lab_notebook()
    system_blocks = build_system_prompt(program_md, lab_notebook=lab_notebook)
    user = build_user_prompt(current_code, prompt_history, experiment_id)
    # Flatten system blocks to text for contract checking and fingerprinting
    system_text = "\n\n".join(b["text"] for b in system_blocks)
    prompt_contract = prompt_contract_check(system_text, program_md)
    prompt_fingerprint = _sha256_text(system_text + "\n\n### USER ###\n" + user)
    exp["prompt_fingerprint"] = prompt_fingerprint
    exp["contract_checksum"] = prompt_contract["checksum_short"]
    _save_artifact_text(artifact_dir, "prompt_system.txt", system_text)
    _save_artifact_text(artifact_dir, "prompt_user.txt", user)
    _save_artifact_json(artifact_dir, "prompt_contract_check.json", prompt_contract)

    if not prompt_contract["ok"]:
        exp["failure_type"] = "drift_guard"
        exp["error"] = "Prompt contract violations detected"
        _save_artifact_json(
            artifact_dir,
            "decision.json",
            {
                "kept": False,
                "failure_type": exp["failure_type"],
                "reason": exp["error"],
                "violations": prompt_contract["violations"],
            },
        )
        _save_artifact_json(artifact_dir, "experiment.v2.json", exp)
        return exp

    write_status(
        "calling_claude",
        experiment_id,
        best_score,
        kept_count,
        failed_count,
        total,
        deadline,
        contract_checksum=prompt_contract["checksum_short"],
    )
    max_codegen_attempts = max(
        1, int(os.environ.get("AR_CODEGEN_MAX_ATTEMPTS", str(MAX_CODEGEN_ATTEMPTS)))
    )
    # Early-crash threshold: crashes within this wall time are retryable
    # (equivalent to what the smoke test used to catch — import/CUDA/data errors)
    EARLY_CRASH_THRESHOLD_S = 30

    attempt_prompt = user
    new_code = ""
    raw_response = ""
    reasoning = ""
    change_summary = ""
    stripped: list[str] = []
    api_time_total = 0.0
    attempts_used = 0
    last_failure_type = "api"
    last_error = "unknown_candidate_failure"

    # Backup current train.py and best_model.pt (once, before retry loop)
    backup_path = TRAIN_PY + ".backup"
    shutil.copy2(TRAIN_PY, backup_path)
    model_backup_path = BEST_MODEL_PT + ".backup"
    if os.path.exists(BEST_MODEL_PT):
        shutil.copy2(BEST_MODEL_PT, model_backup_path)

    for attempt in range(1, max_codegen_attempts + 1):
        write_status(
            "calling_claude",
            experiment_id,
            best_score,
            kept_count,
            failed_count,
            total,
            deadline,
            contract_checksum=prompt_contract["checksum_short"],
        )
        log(f"Calling Claude for code modification (attempt {attempt}/{max_codegen_attempts})...")
        if attempt == 1:
            code_token_est = len(current_code) // 4
            if code_token_est > MAX_TOKENS * 0.6:
                log(f"  WARNING: train.py is ~{code_token_est} tokens (max_tokens={MAX_TOKENS}) — risk of truncation")
        t0 = time.time()
        try:
            # Check for prefetched response (only on first attempt with original prompt)
            prefetched = None
            if attempt == 1:
                state_hash = _prefetch_state_hash(current_code, len(prompt_history))
                prefetched = consume_prefetch(experiment_id, state_hash)
            raw_response = prefetched if prefetched else call_claude(system_blocks, attempt_prompt)
            api_time = time.time() - t0
            api_time_total += api_time
            reasoning = extract_reasoning(raw_response)
            new_code = extract_code(raw_response)
            new_code, stripped = _sanitize_code(new_code)
            _save_artifact_text(artifact_dir, f"response_raw_attempt{attempt}.txt", raw_response)
            _save_artifact_text(artifact_dir, f"reasoning_attempt{attempt}.txt", reasoning or "")
            _save_artifact_json(artifact_dir, f"sanitize_actions_attempt{attempt}.json", {"actions": stripped})
            log(f"  Claude responded in {api_time:.1f}s")
            if reasoning:
                log(f"  Reasoning: {reasoning[:200]}")
            if stripped:
                log(f"  Auto-fixed: {'; '.join(stripped)}")
        except TruncatedResponseError as e:
            last_failure_type = "truncation"
            last_error = f"Truncation error: {e}"
            log(f"  {last_error}")
            _save_artifact_json(
                artifact_dir,
                f"candidate_attempt{attempt}_error.json",
                {"failure_type": last_failure_type, "error": last_error},
            )
            if attempt < max_codegen_attempts:
                log("  Retrying with same prompt (repair won't help truncation)...")
                continue
        except Exception as e:
            last_failure_type = "api"
            last_error = f"API error: {e}"
            log(f"  {last_error}")
            _save_artifact_json(
                artifact_dir,
                f"candidate_attempt{attempt}_error.json",
                {"failure_type": last_failure_type, "error": last_error},
            )
            if attempt < max_codegen_attempts:
                attempt_prompt = build_repair_prompt(new_code or current_code, last_failure_type, last_error, attempt + 1)
                log("  Retrying with targeted repair prompt...")
                continue
            exp["failure_type"] = last_failure_type
            exp["error"] = last_error
            _save_artifact_json(
                artifact_dir,
                "decision.json",
                {"kept": False, "failure_type": exp["failure_type"], "reason": exp["error"]},
            )
            _save_artifact_json(artifact_dir, "experiment.v2.json", exp)
            return exp

        change_summary = extract_change_summary(raw_response, current_code, new_code)
        log(f"  Candidate change: {change_summary}")

        syntax_err = validate_syntax(new_code)
        if syntax_err:
            last_failure_type = "syntax"
            last_error = syntax_err
            log(f"  Syntax error: {syntax_err}")
            _save_artifact_json(
                artifact_dir,
                f"candidate_attempt{attempt}_error.json",
                {"failure_type": last_failure_type, "error": last_error},
            )
            if attempt < max_codegen_attempts:
                attempt_prompt = build_repair_prompt(new_code, last_failure_type, last_error, attempt + 1)
                log("  Requesting syntax repair...")
                continue
            exp["failure_type"] = last_failure_type
            exp["error"] = last_error
            _save_artifact_json(
                artifact_dir,
                "decision.json",
                {"kept": False, "failure_type": exp["failure_type"], "reason": exp["error"]},
            )
            _save_artifact_json(artifact_dir, "experiment.v2.json", exp)
            return exp

        safety_err = validate_safety(new_code)
        if safety_err:
            last_failure_type = "safety"
            last_error = safety_err
            log(f"  {safety_err}")
            _save_artifact_json(
                artifact_dir,
                f"candidate_attempt{attempt}_error.json",
                {"failure_type": last_failure_type, "error": last_error},
            )
            if attempt < max_codegen_attempts:
                attempt_prompt = build_repair_prompt(new_code, last_failure_type, last_error, attempt + 1)
                log("  Requesting safety repair...")
                continue
            exp["failure_type"] = last_failure_type
            exp["error"] = last_error
            _save_artifact_json(
                artifact_dir,
                "decision.json",
                {"kept": False, "failure_type": exp["failure_type"], "reason": exp["error"]},
            )
            _save_artifact_json(artifact_dir, "experiment.v2.json", exp)
            return exp

        # Write candidate and run full training directly (no separate smoke test).
        # Early crashes (<30s wall time) are retried like the old smoke failures.
        with open(TRAIN_PY, 'w') as f:
            f.write(new_code)

        write_status(
            "training",
            experiment_id,
            best_score,
            kept_count,
            failed_count,
            total,
            deadline,
            {"change_summary": change_summary},
            contract_checksum=prompt_contract["checksum_short"],
        )
        log(f"  Training ({time_budget // 60} min budget)...")
        log_diagnostics(f"pre_train_{experiment_id}")

        # Pipeline: speculatively prefetch Claude response for the NEXT experiment
        # while this one trains. Uses current (pre-candidate) code as baseline,
        # since most experiments (~70%) are NOT kept.
        if _prefetch_executor is not None:
            next_id = experiment_id + 1
            next_user_prompt = build_user_prompt(current_code, prompt_history, next_id)
            next_state_hash = _prefetch_state_hash(current_code, len(prompt_history))
            submit_prefetch(next_id, system_blocks, next_user_prompt, next_state_hash)

        timeout = time_budget + 240  # 240s buffer: ~150s eval + ~15s data load + ~75s headroom
        metrics = run_training(TRAIN_PY, timeout=timeout, time_budget=time_budget)
        train_wall_time = metrics.get("wall_time", 999)
        exp["train_wall_time"] = round(train_wall_time, 1)
        log(f"  Done in {train_wall_time:.0f}s")
        log_diagnostics(f"post_train_{experiment_id}")
        _save_artifact_text(artifact_dir, "train_output.log", str(metrics.get("output", "")))
        if "parse_summary" in metrics:
            _save_artifact_json(artifact_dir, "parse_summary.json", metrics["parse_summary"])

        if "error" in metrics and train_wall_time < EARLY_CRASH_THRESHOLD_S:
            # Early crash — equivalent to old smoke failure. Retryable.
            last_failure_type = str(metrics.get("error_type", "train_crash"))
            last_error = str(metrics.get("error", "early crash"))
            log(f"  Early crash ({train_wall_time:.0f}s < {EARLY_CRASH_THRESHOLD_S}s): {last_error[:200]}")
            _save_artifact_json(
                artifact_dir,
                f"candidate_attempt{attempt}_error.json",
                {"failure_type": last_failure_type, "error": last_error, "early_crash": True},
            )
            # Revert train.py for next attempt
            shutil.copy2(backup_path, TRAIN_PY)
            if attempt < max_codegen_attempts:
                attempt_prompt = build_repair_prompt(new_code, last_failure_type, last_error, attempt + 1)
                log("  Requesting crash repair...")
                continue
            exp["failure_type"] = last_failure_type
            exp["error"] = last_error[:1000]
            _save_artifact_json(
                artifact_dir,
                "decision.json",
                {"kept": False, "failure_type": exp["failure_type"], "reason": exp["error"]},
            )
            _save_artifact_json(artifact_dir, "experiment.v2.json", exp)
            return exp

        # Training ran past the early-crash window — no retry, proceed to evaluation
        attempts_used = attempt
        break

    if attempts_used <= 0:
        exp["failure_type"] = last_failure_type
        exp["error"] = last_error[:1000]
        _save_artifact_json(
            artifact_dir,
            "decision.json",
            {"kept": False, "failure_type": exp["failure_type"], "reason": exp["error"]},
        )
        _save_artifact_json(artifact_dir, "experiment.v2.json", exp)
        return exp

    exp["api_time"] = round(api_time_total, 1)
    exp["codegen_attempts"] = attempts_used
    exp["repair_attempts"] = max(0, attempts_used - 1)
    exp["reasoning"] = reasoning[:300] if reasoning else ""
    exp["_full_reasoning"] = reasoning or ""
    exp["train_py_after_hash"] = _sha256_text(new_code)
    exp["change_summary"] = change_summary
    _save_artifact_text(artifact_dir, "response_raw.txt", raw_response)
    _save_artifact_text(artifact_dir, "reasoning.txt", reasoning or "")
    _save_artifact_json(artifact_dir, "sanitize_actions.json", {"actions": stripped})
    _save_artifact_text(artifact_dir, "train_after_candidate.py", new_code)
    log(f"  Using candidate after {attempts_used} attempt(s)")

    if "error" in metrics:
        log(f"  FAILED: {metrics['error'][:200]}")
        exp["error"] = metrics["error"][:1000]
        exp["failure_type"] = str(metrics.get("error_type", "train_crash"))
        # Revert train.py and model weights
        shutil.copy2(backup_path, TRAIN_PY)
        if os.path.exists(model_backup_path):
            shutil.copy2(model_backup_path, BEST_MODEL_PT)
        log("  Reverted to previous train.py + model")
    else:
        score = float(metrics["score"])
        exp["score"] = score
        exp["val_sharpe"] = metrics.get("val_sharpe", 0)
        exp["profit_factor"] = metrics.get("profit_factor", 0)
        exp["trade_sharpe"] = metrics.get("trade_sharpe", 0)
        exp["trades_per_day"] = metrics.get("trades_per_day", 0)
        exp["win_rate"] = metrics.get("win_rate", 0)
        exp["num_trades"] = metrics.get("num_trades", 0)
        exp["max_drawdown"] = metrics.get("max_drawdown", 0)
        exp["do_nothing_pct"] = metrics.get("do_nothing_pct", 0)
        exp["exit_pct"] = metrics.get("exit_pct", 0)
        exp["model_exit_count"] = metrics.get("model_exit_count", 0)
        exp["num_steps"] = metrics.get("num_steps", 0)
        exp["num_params"] = metrics.get("num_params", 0)
        exp["stop_loss_rate"] = metrics.get("stop_loss_rate", 0)
        exp["worst_chunk_pf"] = metrics.get("worst_chunk_pf", 0)
        exp["direction_collapse_pct"] = metrics.get("direction_collapse_pct", 1.0)
        exp["avg_entry_cost_bps"] = metrics.get("avg_entry_cost_bps", 0.0)
        exp["avg_entry_quality"] = metrics.get("avg_entry_quality", 0.0)
        exp["cost_realism_coverage"] = metrics.get("cost_realism_coverage", 0.0)
        exp["high_cost_entry_rate"] = metrics.get("high_cost_entry_rate", 0.0)
        exp["low_quality_entry_rate"] = metrics.get("low_quality_entry_rate", 0.0)
        exp["actionable_bar_rate"] = metrics.get("actionable_bar_rate", 0.0)
        exp["risk_off_bar_rate"] = metrics.get("risk_off_bar_rate", 0.0)
        exp["hit_ruin"] = metrics.get("hit_ruin", False)
        exp["min_equity_frac"] = metrics.get("min_equity_frac", 1.0)
        exp["avg_risk_fraction"] = metrics.get("avg_risk_fraction", 0.0)
        exp["max_risk_fraction"] = metrics.get("max_risk_fraction", 0.0)
        exp["trades_blocked_by_balance"] = metrics.get("trades_blocked_by_balance", 0)
        exp["parse_summary"] = metrics.get("parse_summary", {})

        anomaly_flags = detect_anomaly_flags(exp)
        exp["anomaly_flags"] = anomaly_flags
        critical_flags = _critical_anomaly_flags(anomaly_flags)
        improvement = score - float(best_score)
        near_tie = score > best_score and improvement < float(OBSERVABILITY_CONFIG["near_tie_delta"])
        near_tie_ok = True
        near_tie_reasons: list[str] = []
        if near_tie:
            near_tie_ok, near_tie_reasons = _passes_near_tie_stability(exp)

        # Multi-objective gate: score must improve AND secondary metrics
        # must not catastrophically regress vs the best experiment.
        # This prevents the agent from gaming score while tanking real quality.
        secondary_regression_reasons: list[str] = []
        if score > best_score and best_metrics:
            # Profit factor must not drop below 80% of best
            best_pf = float(best_metrics.get("profit_factor", 0))
            new_pf = float(exp.get("profit_factor", 0))
            if best_pf > 1.0 and new_pf < best_pf * 0.80:
                secondary_regression_reasons.append(
                    f"profit_factor_regressed ({new_pf:.2f} < {best_pf:.2f}*0.80)")
            # Trades per day must not drop below 50% of best
            best_tpd = float(best_metrics.get("trades_per_day", 0))
            new_tpd = float(exp.get("trades_per_day", 0))
            if best_tpd > 0.5 and new_tpd < best_tpd * 0.50:
                secondary_regression_reasons.append(
                    f"trades_per_day_regressed ({new_tpd:.1f} < {best_tpd:.1f}*0.50)")
            # Trade Sharpe must not go negative if it was positive
            best_ts = float(best_metrics.get("trade_sharpe", 0))
            new_ts = float(exp.get("trade_sharpe", 0))
            if best_ts > 0.5 and new_ts < 0:
                secondary_regression_reasons.append(
                    f"trade_sharpe_regressed ({new_ts:.2f} was {best_ts:.2f})")
        if secondary_regression_reasons:
            exp["secondary_regression"] = secondary_regression_reasons
            log(f"  ⚠ Secondary metric regression: {'; '.join(secondary_regression_reasons)}")

        keep_allowed = (score > best_score and not critical_flags
                        and near_tie_ok and not secondary_regression_reasons)
        if keep_allowed:
            log(f"  ✓ IMPROVED: {best_score:.4f} → {score:.4f} (pf={exp['profit_factor']:.2f} tpd={exp['trades_per_day']:.1f})")
            exp["kept"] = True
            exp["failure_type"] = "none"
            # Save as best — keep new model weights, archive old backup
            shutil.copy2(TRAIN_PY, BEST_TRAIN_PY)
            if os.path.exists(model_backup_path):
                os.remove(model_backup_path)
        else:
            exp["failure_type"] = "regression"
            reasons = []
            if score <= best_score:
                reasons.append("score_not_improved")
            if critical_flags:
                reasons.append(f"critical_anomalies:{','.join(critical_flags)}")
            if near_tie and not near_tie_ok:
                reasons.append(f"near_tie_stability_failed:{','.join(near_tie_reasons)}")
            if secondary_regression_reasons:
                reasons.append(f"secondary_regression:{','.join(secondary_regression_reasons)}")
            exp["keep_block_reason"] = ";".join(reasons) if reasons else "regression_gate"
            log(f"  ✗ No improvement gate: score={score:.4f}, reasons={exp['keep_block_reason']}")
            exp["kept"] = False
            # Always revert train.py (undo LLM code changes)
            shutil.copy2(backup_path, TRAIN_PY)
            # Selectively revert model weights: keep learned weights if the
            # experiment improved over the PREVIOUS experiment's score (even if
            # it didn't beat all-time best), unless there are critical anomalies
            # or a score regression vs. the prior experiment.
            prev_exp_score = None
            for prev in reversed(prompt_history):
                if prev.get("score") is not None:
                    prev_exp_score = float(prev["score"])
                    break
            hard_revert_model = True
            if prev_exp_score is not None and score > prev_exp_score and not critical_flags:
                hard_revert_model = False
                log(f"  Keeping model weights (score {score:.4f} > prev {prev_exp_score:.4f}, incremental improvement)")
                exp["model_weight_action"] = "kept_incremental"
            else:
                exp["model_weight_action"] = "reverted"
            if hard_revert_model and os.path.exists(model_backup_path):
                shutil.copy2(model_backup_path, BEST_MODEL_PT)

    # Clean up backups
    for bkp in [backup_path, model_backup_path]:
        if os.path.exists(bkp):
            os.remove(bkp)

    exp["model_fingerprint_after"] = _sha256_file(BEST_MODEL_PT)
    _save_artifact_json(
        artifact_dir,
        "decision.json",
        {
            "kept": exp.get("kept", False),
            "failure_type": exp.get("failure_type"),
            "reason": exp.get("error") or exp.get("keep_block_reason"),
            "anomaly_flags": exp.get("anomaly_flags", []),
        },
    )
    _save_artifact_json(
        artifact_dir,
        "experiment.v2.json",
        exp,
    )
    return exp


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autoresearch loop")
    parser.add_argument("--hours", type=float, default=8.0,
                        help="Total runtime in hours (default: 8)")
    parser.add_argument("--max-experiments", type=int, default=200,
                        help="Max experiments to run (default: 200)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup without running experiments")
    parser.add_argument("--time-budget", type=int, default=240,
                        help="Training time budget per experiment in seconds (default: 240)")
    parser.add_argument(
        "--allow-data-fingerprint-change",
        action="store_true",
        help="Allow run to proceed when data fingerprint changes from the cached baseline",
    )
    args = parser.parse_args()

    # Validate environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  Get one at https://console.anthropic.com/settings/keys")
        sys.exit(1)

    if not os.path.exists(TRAIN_PY):
        print(f"ERROR: {TRAIN_PY} not found")
        sys.exit(1)
    if not os.path.exists(PROGRAM_MD):
        print(f"ERROR: {PROGRAM_MD} not found")
        sys.exit(1)

    _initialize_run_layout()

    # -----------------------------------------------------------------------
    # Pre-flight checks — catch problems before consuming GPU time
    # -----------------------------------------------------------------------
    print("=== PRE-FLIGHT CHECKS ===")
    data_fingerprint: str | None = None

    # 1. data.pt — load and validate contents
    try:
        sys.path.insert(0, SCRIPT_DIR)
        from prepare import FEATURE_NAMES as _FEATURE_NAMES
        from prepare import load_data as _preflight_load
        import torch as _torch
        _data = _preflight_load()
        n_bars = len(_data.get('dates', []))

        required_keys = ['features', 'targets', 'valid_mask', 'dates',
                         'train_end_idx', 'val_start_idx', 'val_end_idx']
        missing = [k for k in required_keys if k not in _data]
        if missing:
            print(f"ERROR: data.pt missing keys: {missing}")
            sys.exit(1)

        n_features = _data['features'].shape[-1]
        nan_pct = _torch.isnan(_data['features']).float().mean().item() * 100
        print(f"  data.pt:   OK ({n_bars} bars, {n_features} features, {nan_pct:.1f}% NaN)")
        if n_features != FEATURE_LOCK_COUNT:
            print(
                "ERROR: Feature-contract violation. "
                f"data.pt has {n_features} features, expected {FEATURE_LOCK_COUNT}."
            )
            print(
                "  Rebuild/restore 32-feature data before running the training loop."
            )
            sys.exit(1)
        if nan_pct > 50:
            print("  WARNING: >50% NaN features — training quality will be poor")
        data_quality = _compute_data_quality_report(_data, list(_FEATURE_NAMES))
        cached_fp = _load_saved_data_fingerprint()
        if cached_fp and cached_fp.get("fingerprint") != data_quality.get("fingerprint"):
            print("ERROR: data fingerprint changed vs cached baseline.")
            print(f"  cached:  {cached_fp.get('fingerprint')}")
            print(f"  current: {data_quality.get('fingerprint')}")
            print("  Use --allow-data-fingerprint-change to acknowledge and continue.")
            if not args.allow_data_fingerprint_change:
                raise ValueError("data_fingerprint_changed")
            print("  Override enabled: accepting new fingerprint.")
        _save_data_quality(data_quality)
        _write_json(os.path.join(RUN_DIR, "data_quality_report.json"), data_quality)
        data_fingerprint = data_quality.get("fingerprint")
        print(f"  data fingerprint: {str(data_fingerprint)[:12]}")
        del _data
    except SystemExit:
        print("ERROR: data.pt not found. Upload it to the container first.")
        print("  Expected at: ~/.cache/autoresearch-trading/features/data.pt")
        print("  Or next to train.py")
        sys.exit(1)
    except Exception as e:
        if str(e) == "data_fingerprint_changed":
            sys.exit(2)
        print(f"ERROR: Could not load data.pt: {e}")
        sys.exit(1)

    # 2. GPU availability
    if _torch.cuda.is_available():
        gpu = _torch.cuda.get_device_name(0)
        props = _torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
        print(f"  GPU:       OK ({gpu}, {mem:.0f}GB)")
    else:
        print("  GPU:       WARNING — no CUDA GPU detected, training will be slow")

    # 3. Disk space
    st = os.statvfs(SCRIPT_DIR)
    free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
    print(f"  Disk:      {free_gb:.1f}GB free")
    if free_gb < 1.0:
        print("  WARNING: <1GB free — may run out during training")

    # 4. Anthropic API — verify package and key work
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    try:
        _client = anthropic.Anthropic()
        _resp = _client.messages.create(
            model=CLAUDE_MODEL, max_tokens=10,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        print(f"  Claude:    OK (model={CLAUDE_MODEL})")
        del _client, _resp
    except Exception as e:
        print(f"ERROR: Claude API test failed: {e}")
        sys.exit(1)

    if os.path.exists(LAB_NOTEBOOK_MD):
        _nb_size = os.path.getsize(LAB_NOTEBOOK_MD)
        print(f"  Notebook:  OK ({_nb_size} bytes)")
    else:
        print(f"  Notebook:  not found (will run without persistent context)")
    print("=== ALL CHECKS PASSED ===\n")
    _save_run_metadata(
        {
            "created_at": _now_iso(),
            "started_at": _now_iso(),
            "run_name": RUN_NAME,
            "run_dir": RUN_DIR,
            "results_dir": RESULTS_DIR,
            "mode": "dry_run" if args.dry_run else "loop",
            "source": _runtime_source(),
            "schema_version": SCHEMA_VERSION,
            "observability_config": _json_safe(OBSERVABILITY_CONFIG),
            "data_fingerprint": data_fingerprint,
            "allow_data_fingerprint_change": bool(args.allow_data_fingerprint_change),
            "claude_model": CLAUDE_MODEL,
            "hours": args.hours,
            "max_experiments": args.max_experiments,
            "time_budget": args.time_budget,
        }
    )

    # Histories:
    # - run_history: current run's experiments (for status and run summary)
    # - prompt_history: ALL experiment history (promoted + current run) for LLM context
    run_history = load_history()
    prompt_history = load_promoted_history()
    # Also include current run's experiments so Claude remembers prior attempts
    for prev_exp in run_history:
        rec = _exp_to_prompt_record(prev_exp)
        if rec is not None:
            prompt_history.append(rec)

    # Warm-start baseline at -5.0 (not -999): a zero-trade model returns -10.0
    # so it must actually trade profitably to beat baseline and get "kept".
    INITIAL_BASELINE = -5.0
    promoted_scores = []
    for rec in prompt_history:
        try:
            promoted_scores.append(float(rec.get("score")))
        except Exception:
            continue
    checkpoint_score = _load_checkpoint_score(BEST_MODEL_PT)
    if promoted_scores:
        best_score = max(promoted_scores)
        best_source = "promoted_history"
    elif checkpoint_score is not None:
        best_score = float(checkpoint_score)
        best_source = "checkpoint_only"
    else:
        best_score = INITIAL_BASELINE
        best_source = "baseline_default"

    run_ids: list[int] = []
    for e in run_history:
        try:
            run_ids.append(int(e.get("experiment_id", 0)))
        except Exception:
            continue
    start_id = (max(run_ids) if run_ids else 0) + 1

    log(f"Autoresearch loop starting")
    log(f"  Runtime budget: {args.hours}h ({args.hours * 60:.0f} min)")
    log(f"  Training budget per experiment: {args.time_budget}s ({args.time_budget // 60} min)")
    log(f"  Max experiments: {args.max_experiments}")
    log(f"  Current-run experiments: {len(run_history)}")
    log(f"  Prompt history (promoted + current run): {len(prompt_history)}")
    if best_source == "promoted_history":
        log(f"  Best score so far (promoted): {best_score:.4f}")
    elif best_source == "checkpoint_only":
        log(f"  Best score so far (checkpoint fallback): {best_score:.4f}")
    else:
        log("  No promoted/checkpoint score found (baseline: -5.0)")
    log(f"  Model: {CLAUDE_MODEL}")
    log(f"  Run: {RUN_NAME}")
    log(f"  Log: {RUN_EXPERIMENTS_V2_LOG}")
    log("")

    if args.dry_run:
        dry_deadline = time.time() + args.hours * 3600
        write_status(
            "dry_run",
            start_id - 1 if start_id > 0 else 0,
            best_score,
            0,
            0,
            len(run_history),
            dry_deadline,
        )
        log("Dry run complete. All pre-flight checks passed.")
        return

    # Install signal handlers to catch what kills us
    install_signal_handlers()
    _diag(f"Loop starting: hours={args.hours}, max_experiments={args.max_experiments}")
    log_diagnostics("startup")

    # Initialize prefetch executor for pipelining Claude API calls
    global _prefetch_executor
    _prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    log("  Prefetch executor initialized (pipelined API calls enabled)")

    # Save initial train.py as best if no best exists
    if not os.path.exists(BEST_TRAIN_PY):
        shutil.copy2(TRAIN_PY, BEST_TRAIN_PY)
    _snapshot_runtime_files_to_run_dir()

    deadline = time.time() + args.hours * 3600
    experiment_id = start_id
    kept_count = 0
    failed_count = 0
    consecutive_failures = 0
    consecutive_api_credit_failures = 0
    MAX_API_CREDIT_FAILURES = 3  # Auto-shutdown after 3 consecutive credit failures
    # Track best experiment's secondary metrics for multi-objective gate
    best_metrics: dict[str, Any] = {}
    # Seed from promoted history if available
    for rec in reversed(prompt_history):
        if rec.get("kept") and _safe_score(rec) > -999:
            best_metrics = {k: rec[k] for k in ("profit_factor", "trades_per_day", "trade_sharpe") if k in rec}
            break

    while time.time() < deadline and experiment_id < start_id + args.max_experiments:
        remaining_h = (deadline - time.time()) / 3600
        scored_so_far = len(run_history) - failed_count
        accept_rate = kept_count / scored_so_far if scored_so_far > 0 else 0.0
        log(f"Time remaining: {remaining_h:.1f}h | Best score: {best_score:.4f} | "
            f"Kept: {kept_count} | Failed: {failed_count} | "
            f"Accept rate: {accept_rate:.0%} ({kept_count}/{scored_so_far})")

        # Strategy collapse check
        collapse = detect_strategy_collapse(prompt_history)
        if collapse["collapsed"]:
            log(f"  ⚠ {collapse['message']}")
        elif collapse["signals"]:
            log(f"  ⚡ Collapse signal: {'; '.join(collapse['signals'])}")

        exp = run_one_experiment(experiment_id, prompt_history, best_score,
                                deadline, kept_count, failed_count,
                                run_total=len(run_history),
                                time_budget=args.time_budget,
                                data_fingerprint=data_fingerprint,
                                best_metrics=best_metrics)
        run_history.append(exp)
        append_experiment_v2(exp)
        append_results_tsv(exp)
        _snapshot_runtime_files_to_run_dir()

        # Reap zombies, GC, diagnostics after each experiment
        _reap_zombies()
        gc.collect()
        log_diagnostics(f"post_exp_{experiment_id}")

        # Add ALL experiments to prompt_history so Claude remembers prior attempts
        prompt_rec = _exp_to_prompt_record(exp)
        if prompt_rec is not None:
            prompt_history.append(prompt_rec)

        # Detect API credit exhaustion — auto-shutdown to stop wasting compute
        exp_error = exp.get("error", "")
        is_credit_failure = "credit balance" in exp_error.lower() or "billing" in exp_error.lower()
        if is_credit_failure:
            consecutive_api_credit_failures += 1
            log(f"  ⚠ API CREDIT FAILURE ({consecutive_api_credit_failures}/{MAX_API_CREDIT_FAILURES})")
            if consecutive_api_credit_failures >= MAX_API_CREDIT_FAILURES:
                log("=" * 60)
                log("AUTORESEARCH HALTED: API credits exhausted")
                log(f"  {MAX_API_CREDIT_FAILURES} consecutive credit failures detected.")
                log(f"  Best score: {best_score:.4f} | Kept: {kept_count} | Total: {len(run_history)}")
                log("  Top up credits and redeploy.")
                log("=" * 60)
                write_status("completed", experiment_id, best_score,
                             kept_count, failed_count, len(run_history), deadline,
                             contract_checksum=exp.get("contract_checksum"))
                break
        else:
            consecutive_api_credit_failures = 0

        if exp.get("kept"):
            best_score = exp["score"]
            kept_count += 1
            consecutive_failures = 0
            # Update best metrics for multi-objective gate
            best_metrics = {
                "profit_factor": exp.get("profit_factor", 0),
                "trades_per_day": exp.get("trades_per_day", 0),
                "trade_sharpe": exp.get("trade_sharpe", 0),
            }
            record_promotion_event(exp)
            full_reasoning = exp.get("_full_reasoning", exp.get("reasoning", ""))
            update_lab_notebook(exp, full_reasoning)
        elif exp.get("error"):
            failed_count += 1
            consecutive_failures += 1
            # Auto-populate Dead Ends with failed experiments that had a change summary
            if exp.get("change_summary"):
                update_lab_notebook_dead_end(exp)
        else:
            consecutive_failures = 0
            # Auto-populate Dead Ends with scored-but-rejected experiments
            if exp.get("change_summary") and _safe_score(exp) > -999:
                update_lab_notebook_dead_end(exp)

        # Update status after experiment
        write_status("between_experiments", experiment_id, best_score,
                     kept_count, failed_count, len(run_history), deadline, exp,
                     contract_checksum=exp.get("contract_checksum"))

        # Safety: if 5 consecutive failures, restore best and continue
        if consecutive_failures >= 5:
            log("WARNING: 5 consecutive failures. Restoring best_train.py")
            if os.path.exists(BEST_TRAIN_PY):
                shutil.copy2(BEST_TRAIN_PY, TRAIN_PY)
            consecutive_failures = 0

        experiment_id += 1
        log("")

    # Shut down prefetch executor
    if _prefetch_executor is not None:
        _prefetch_executor.shutdown(wait=False)

    # Final summary
    write_status("completed", experiment_id - 1, best_score,
                 kept_count, failed_count, len(run_history), deadline,
                 contract_checksum=(run_history[-1].get("contract_checksum") if run_history else None))
    log("=" * 60)
    log("AUTORESEARCH COMPLETE")
    log(f"  Total experiments: {experiment_id - start_id}")
    log(f"  Kept improvements: {kept_count}")
    log(f"  Failed: {failed_count}")
    log(f"  Best score: {best_score:.6f}")
    log("")

    if run_history:
        log("Top 5 experiments by score:")
        ranked = sorted(run_history, key=lambda e: _safe_score(e), reverse=True)
        for i, exp in enumerate(ranked[:5]):
            sc = _safe_score(exp)
            log(f"  #{exp['experiment_id']}: score={sc:.4f} pf={exp.get('profit_factor', 0):.2f} "
                f"tpd={exp.get('trades_per_day', 0):.1f} — {exp.get('change_summary', 'N/A')}")

    log(f"\nBest model saved at: {BEST_TRAIN_PY}")
    log(f"Full log at: {RUN_EXPERIMENTS_V2_LOG}")


if __name__ == "__main__":
    main()
