"""
Autoresearch utilities: validation, anomaly detection, scoring, and training helpers.

This is a utility library — NOT executable. All loop orchestration is handled by
tools/inner_loop.py (mechanical) and Claude Code Opus (strategic).

Functions kept here are imported by inner_loop.py and art2.py.
"""
from __future__ import annotations

import os
import sys
import ast
import json
import time
import hashlib
import datetime
import subprocess
import re as _re
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")
PROGRAM_MD = os.path.join(SCRIPT_DIR, "program.md")
BEST_TRAIN_PY = os.path.join(SCRIPT_DIR, "best_train.py")
BEST_MODEL_PT = os.path.join(SCRIPT_DIR, "best_model.pt")
PYTHON = sys.executable

RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results") if os.path.basename(SCRIPT_DIR) == "training" else os.path.join(SCRIPT_DIR, "results")
PROMOTED_DIR = os.path.join(RESULTS_DIR, "promoted")
PROMOTED_HISTORY_JSONL = os.path.join(PROMOTED_DIR, "history.jsonl")

SCHEMA_VERSION = 2
FAILURE_TYPE_ENUM = {
    "api", "truncation", "syntax", "safety", "train_crash",
    "timeout", "parse", "drift_guard", "regression", "none",
}
REQUIRED_OUTPUT_METRIC_KEYS = (
    "score", "profit_factor", "trades_per_day",
    "trade_sharpe", "stop_loss_rate", "worst_chunk_pf",
)
FEATURE_LOCK_COUNT = 37

OBSERVABILITY_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "critical_anomaly_flags": {
        "metric_inconsistent",
        "trades_per_day_extreme",
        "cost_realism_low_coverage",
        "entry_quality_too_low",
        "low_quality_entry_rate_high",
        "single_date_specialist",
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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_jsonl(path: str, rec: dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------
# Checkpoint introspection
# ---------------------------------------------------------------------------

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


def _load_checkpoint_dynamics(checkpoint_path: str) -> dict | None:
    """Load training_dynamics metadata from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        return ckpt.get("training_dynamics")
    return None


def _extract_current_dynamics(train_py_path: str) -> dict:
    """Parse current train.py to extract training dynamics defaults."""
    with open(train_py_path, 'r') as f:
        code = f.read()

    dynamics = {}

    m = _re.search(r'BATCH_SIZE\s*=\s*_env_int\([^,]+,\s*(\d+)', code)
    if not m:
        raise RuntimeError("FATAL: Cannot parse BATCH_SIZE from train.py")
    dynamics['batch_size'] = int(m.group(1))

    m = _re.search(r'EXIT_LOSS_WEIGHT\s*=\s*_env_float\([^,]+,\s*([\d.e-]+)', code)
    if not m:
        raise RuntimeError("FATAL: Cannot parse EXIT_LOSS_WEIGHT from train.py")
    dynamics['exit_loss_weight'] = float(m.group(1))

    dynamics['bf16'] = 'torch.amp.autocast' in code

    return dynamics


def _check_dynamics_compatibility(checkpoint_path: str, train_py_path: str) -> str | None:
    """Compare checkpoint training dynamics against current train.py.

    Returns error message if incompatible. No fallbacks.
    """
    ckpt_dynamics = _load_checkpoint_dynamics(checkpoint_path)
    current_dynamics = _extract_current_dynamics(train_py_path)

    if not ckpt_dynamics:
        return (
            "DYNAMICS MISMATCH: Checkpoint has no training_dynamics metadata. "
            "Cannot verify that the score baseline is achievable. "
            "Resave the checkpoint with current train.py or fresh-start."
        )

    mismatches = []
    for key in ('batch_size', 'exit_loss_weight', 'bf16'):
        ckpt_val = ckpt_dynamics.get(key)
        curr_val = current_dynamics.get(key)
        if ckpt_val is None:
            mismatches.append(f"  {key}: MISSING from checkpoint (current={curr_val})")
        elif curr_val is not None and ckpt_val != curr_val:
            mismatches.append(f"  {key}: checkpoint={ckpt_val} → current={curr_val}")

    if mismatches:
        return (
            "DYNAMICS MISMATCH: Training dynamics changed since checkpoint was saved.\n"
            + "\n".join(mismatches) + "\n"
            "The checkpoint's score was achieved under different training conditions.\n"
            "Score baseline is UNREACHABLE. Fresh-start required."
        )
    return None


# ---------------------------------------------------------------------------
# Promoted history
# ---------------------------------------------------------------------------

def _promotion_event_from_exp(exp: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": _now_iso(),
        "run_name": exp.get("run_name", "unknown"),
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
        "change_summary": exp.get("change_summary"),
    }


def load_promoted_history() -> list[dict[str, Any]]:
    """Load prompt history from promoted events."""
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
            history.append(event)
    return history


def record_promotion_event(exp: dict[str, Any]) -> dict[str, Any]:
    """Append promotion ledger entry."""
    event = _promotion_event_from_exp(exp)
    _append_jsonl(PROMOTED_HISTORY_JSONL, event)
    return event


# ---------------------------------------------------------------------------
# Validation: syntax, safety, architecture
# ---------------------------------------------------------------------------

def validate_syntax(code: str) -> str | None:
    """Return None if valid, error string if not."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e}"


def _sanitize_code(code: str) -> tuple[str, list[str]]:
    """Auto-strip known dangerous patterns. Returns (cleaned_code, list_of_stripped)."""
    stripped = []
    compile_re = _re.compile(r'^([\t ]*)(?:\w+\s*=\s*)?torch\.compile\(.*\).*$', _re.MULTILINE)
    if compile_re.search(code):
        code = compile_re.sub(lambda m: m.group(1) + "pass  # torch.compile removed", code)
        stripped.append("auto-stripped torch.compile")
    if _re.search(r'\blookbook\b', code, _re.IGNORECASE):
        code = _re.sub(r'\blookbook\b', 'lookback', code, flags=_re.IGNORECASE)
        stripped.append("auto-fixed lookbook→lookback typo")
    return code, stripped


# Patterns that crash the container or break architecture
_DANGEROUS_PATTERNS = [
    (r'\bnn\.DataParallel\b', "DataParallel is forbidden (single GPU)"),
    (r'\bDistributedDataParallel\b', "DDP is forbidden (single GPU)"),
    (r'\btorch\.jit\.(trace|script)\b', "torch.jit is forbidden"),
    (r'BATCH_SIZE\s*=\s*(\d+)', None),  # checked separately
    (r'D_MODEL\s*=\s*(\d+)', None),
    (r'DEPTH\s*=\s*(\d+)', None),
]

_STRUCTURAL_PATTERNS = [
    (r'nn\.Linear\([^)]*,\s*NUM_ACTIONS\)', "Do NOT merge gate+dir into single head."),
    (r'nn\.Linear\([^)]*,\s*4\)', "Do NOT merge gate+dir into single head outputting 4 logits."),
    (r'TRADE_LABEL_PERCENTILE', "Do NOT revert to forward-return percentile labels."),
]


def validate_safety(code: str) -> str | None:
    """Check for dangerous patterns. Returns None if safe, error string if dangerous."""
    for pattern, msg in _DANGEROUS_PATTERNS:
        m = _re.search(pattern, code)
        if m and msg:
            return f"SAFETY: {msg}"

    for pattern, msg in _STRUCTURAL_PATTERNS:
        m = _re.search(pattern, code)
        if m:
            return f"SAFETY: {msg}"

    # Score config lock
    _locked_defaults = {
        'win_rate_bonus': '0.0', 'rr_bonus': '0.3', 'drawdown_penalty': '0.5',
        'hold_bonus': '0.0', 'freq_center': '2.5', 'freq_width': '2.5',
        'consec_loss_threshold': '3', 'short_hold_threshold': '0.30',
        'stop_rate_threshold': '0.30', 'ruin_penalty': '1.0',
        'ruin_threshold': '0.25', 'risk_fraction_penalty': '0.5',
    }
    if '_score_config' in code:
        for key, default_val in _locked_defaults.items():
            m = _re.search(rf"'{key}':\s*([0-9.]+)", code)
            if m and m.group(1) != default_val:
                return (
                    f"SAFETY: _score_config['{key}'] changed to {m.group(1)} "
                    f"(locked at {default_val}). Modifying score_config games the "
                    f"evaluation metric without improving the model."
                )

    if _re.search(r'_env_float\(\s*["\']SCORE_', code):
        return "SAFETY: SCORE_* env vars are forbidden. The score formula is locked."

    # _env_float validation: allow up to 3 new declarations with approved prefixes
    _ALLOWED_ENV_PREFIXES = ('SCHED_', 'WEIGHT_', 'WARM_', 'REG_', 'TRAIN_')
    if os.path.exists(BEST_TRAIN_PY):
        try:
            with open(BEST_TRAIN_PY, 'r') as _f:
                _best_code = _f.read()
            best_envs = set(_re.findall(r"_env_float\(['\"](\w+)['\"]", _best_code))
            new_envs = set(_re.findall(r"_env_float\(['\"](\w+)['\"]", code))
            added_envs = new_envs - best_envs
            if len(added_envs) > 3:
                return f"SAFETY: Too many new _env_float declarations ({len(added_envs)}). Max 3 per experiment."
            for env_name in added_envs:
                if not any(env_name.startswith(p) for p in _ALLOWED_ENV_PREFIXES):
                    return (
                        f"SAFETY: New _env_float '{env_name}' must start with one of "
                        f"{_ALLOWED_ENV_PREFIXES}"
                    )
            # Verify new _env_float defaults to 0.0
            for env_name in added_envs:
                pattern = rf"_env_float\(['\"]" + _re.escape(env_name) + rf"['\"],\s*([\d.eE+-]+)"
                m_env = _re.search(pattern, code)
                if m_env:
                    try:
                        default_val = float(m_env.group(1))
                        if default_val != 0.0:
                            return f"SAFETY: New _env_float '{env_name}' must default to 0.0 (got {m_env.group(1)})"
                    except ValueError:
                        pass

            # Loss function check: allow ONE reg_* function, block all others
            _loss_fn_pattern = r'def\s+\w*(?:loss|penalty|consistency|confidence|quality)\w*\s*\('
            best_loss_fns = set(_re.findall(_loss_fn_pattern, _best_code))
            new_loss_fns = set(_re.findall(_loss_fn_pattern, code))
            added_fns = new_loss_fns - best_loss_fns
            if added_fns:
                reg_fns = {f for f in added_fns if _re.match(r'def\s+(reg_|regularize_)', f)}
                non_reg_fns = added_fns - reg_fns
                if non_reg_fns:
                    return f"SAFETY: New loss function(s) added: {non_reg_fns}. Loss function is FROZEN."
                if len(reg_fns) > 1:
                    return f"SAFETY: Only ONE regularization function allowed, found {len(reg_fns)}"
        except Exception:
            pass

    # BATCH_SIZE limit
    m = _re.search(r'BATCH_SIZE\s*=\s*(\d+)', code)
    if m and int(m.group(1)) > 1024:
        return f"SAFETY: BATCH_SIZE={m.group(1)} exceeds limit of 1024"

    # D_MODEL limit
    m = _re.search(r'D_MODEL\s*=\s*(\d+)', code)
    if m and int(m.group(1)) > 128:
        return f"SAFETY: D_MODEL={m.group(1)} exceeds limit of 128"

    # DEPTH limit
    m = _re.search(r'DEPTH\s*=\s*(\d+)', code)
    if m and int(m.group(1)) > 8:
        return f"SAFETY: DEPTH={m.group(1)} exceeds limit of 8"

    # D_MODEL divisible by N_HEADS
    m_d = _re.search(r'D_MODEL\s*=\s*(\d+)', code)
    m_h = _re.search(r'N_HEADS\s*=\s*(\d+)', code)
    if m_d and m_h:
        d, h = int(m_d.group(1)), int(m_h.group(1))
        if h > 0 and d % h != 0:
            return f"SAFETY: D_MODEL={d} not divisible by N_HEADS={h}."

    # Architecture lock
    arch_err = validate_architecture_locked(code)
    if arch_err:
        return arch_err

    # Feature count lock
    feature_total = _extract_feature_groups_width(code)
    if feature_total is None:
        return f"SAFETY: Could not parse FEATURE_GROUPS. Keep max end index = {FEATURE_LOCK_COUNT}."
    if feature_total != FEATURE_LOCK_COUNT:
        return f"SAFETY: FEATURE_GROUPS covers {feature_total} features, expected {FEATURE_LOCK_COUNT}."

    return None


def validate_architecture_locked(code: str) -> str | None:
    """Ensure D_MODEL, DEPTH, N_HEADS match best_train.py."""
    if not os.path.exists(BEST_TRAIN_PY):
        return None

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
                return (f"SAFETY: {param}={actual} differs from locked value {expected}. "
                        f"Breaks warm-start weight compatibility.")
    return None


def _extract_feature_groups_width(code: str) -> int | None:
    """Return max FEATURE_GROUPS end index if parseable."""
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
                start, end = int(rng[0]), int(rng[1])
            except Exception:
                return None
            if start < 0 or end <= start:
                return None
            max_end = end if max_end is None else max(max_end, end)
        return max_end
    return None


# ---------------------------------------------------------------------------
# Anomaly detection & scoring
# ---------------------------------------------------------------------------

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


def validate_experiment_v2_schema(exp: dict[str, Any]) -> list[str]:
    required = [
        "schema_version", "experiment_id", "timestamp",
        "train_py_before_hash", "train_py_after_hash",
        "failure_type", "anomaly_flags", "kept", "score",
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


# ---------------------------------------------------------------------------
# Training execution (local subprocess — used for smoke tests)
# ---------------------------------------------------------------------------

def run_training(train_py_path: str, timeout: int = 420, time_budget: int = 300) -> dict:
    """Run train.py locally and parse output metrics.

    Returns dict with metrics or {'error': 'message'}.
    """
    t_start = time.time()
    try:
        env = os.environ.copy()
        env["TIME_BUDGET"] = str(time_budget)
        result = subprocess.run(
            [PYTHON, "-u", train_py_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=SCRIPT_DIR, env=env,
        )
        wall_time = time.time() - t_start
        output = result.stdout + result.stderr

        if result.returncode != 0:
            lines = output.strip().split('\n')
            tail = '\n'.join(lines[-50:])
            return {"error": f"Exit code {result.returncode}:\n{tail}",
                    "error_type": "train_crash", "output": output, "wall_time": wall_time}

        metrics = _parse_training_output(output)
        metrics["wall_time"] = wall_time
        return metrics

    except subprocess.TimeoutExpired as e:
        wall_time = time.time() - t_start
        merged = (str(e.stdout or "") + "\n" + str(e.stderr or "")).strip()
        return {"error": f"Training timed out after {timeout}s",
                "error_type": "timeout", "output": merged, "wall_time": wall_time}
    except Exception as e:
        return {"error": f"Exception: {e}", "error_type": "train_crash",
                "wall_time": time.time() - t_start}


def _parse_training_output(output: str) -> dict:
    """Parse training output — prefer METRICS_JSON, fall back to line parsing."""
    metrics: dict[str, Any] = {"output": output}

    # Try METRICS_JSON first (structured, complete)
    for line in output.split('\n'):
        if line.strip().startswith("METRICS_JSON:"):
            try:
                json_str = line.strip()[len("METRICS_JSON:"):]
                parsed = json.loads(json_str)
                metrics.update(parsed)
                # Extract trade diagnostics too
                diag_start = output.find("=== TRADE DIAGNOSTICS ===")
                diag_end = output.find("=== END DIAGNOSTICS ===")
                if diag_start >= 0 and diag_end >= 0:
                    metrics["trade_diagnostics"] = output[diag_start:diag_end + len("=== END DIAGNOSTICS ===")].strip()
                return metrics
            except json.JSONDecodeError:
                pass

    # Fallback: line-by-line parsing
    float_keys = {
        'score', 'val_sharpe', 'max_drawdown', 'annual_return',
        'win_rate', 'profit_factor', 'avg_winner', 'avg_loser',
        'trades_per_day', 'trade_sharpe', 'sortino', 'calmar',
        'ev_per_trade', 'do_nothing_pct', 'exit_pct', 'total_return',
        'peak_vram_mb', 'training_seconds', 'total_seconds',
        'short_hold_pct', 'stop_loss_rate', 'final_capital',
        'equity_sharpe', 'max_equity_dd', 'total_dollar_return',
        'worst_chunk_pf', 'direction_collapse_pct',
        'avg_entry_cost_bps', 'avg_entry_quality',
        'cost_realism_coverage', 'high_cost_entry_rate',
        'low_quality_entry_rate', 'actionable_bar_rate',
        'risk_off_bar_rate', 'min_equity_frac',
        'avg_risk_fraction', 'max_risk_fraction',
    }
    int_keys = {
        'num_trades', 'num_val_bars', 'num_val_days', 'num_steps',
        'num_params', 'max_consec_loss', 'model_exit_count',
        'cooldown_blocked', 'pre_10am_blocked', 'trades_blocked_by_balance',
        'num_trade_dates',
    }

    for line in output.split('\n'):
        line = line.strip()
        if ':' in line and not line.startswith('#'):
            key, _, val = line.partition(':')
            key, val = key.strip(), val.strip()
            if key in float_keys:
                try:
                    metrics[key] = float(val)
                except ValueError:
                    pass
            elif key in int_keys:
                try:
                    metrics[key] = int(val.replace(',', ''))
                except ValueError:
                    pass
            elif key == 'hit_ruin':
                metrics[key] = val.strip().lower() == 'true'

    # Parse chunk details
    chunk_re = _re.compile(r'Chunk (\d+): (.+?) \| (\d+) trades \| PF=([\d.]+) \| WR=([\d.]+)%')
    chunk_details = []
    for line in output.split('\n'):
        m = chunk_re.search(line)
        if m:
            chunk_details.append({
                "chunk": int(m.group(1)),
                "dates": m.group(2).strip(),
                "trades": int(m.group(3)),
                "profit_factor": float(m.group(4)),
                "win_rate": float(m.group(5)) / 100.0,
            })
    if chunk_details:
        metrics["chunk_details"] = chunk_details

    # Extract trade diagnostics
    diag_start = output.find("=== TRADE DIAGNOSTICS ===")
    diag_end = output.find("=== END DIAGNOSTICS ===")
    if diag_start >= 0 and diag_end >= 0:
        metrics["trade_diagnostics"] = output[diag_start:diag_end + len("=== END DIAGNOSTICS ===")].strip()

    if 'score' not in metrics:
        return {"error": f"Could not parse score from output:\n{output[-500:]}",
                "error_type": "parse", "output": output}

    missing_required = [k for k in REQUIRED_OUTPUT_METRIC_KEYS if k not in metrics]
    if missing_required:
        return {"error": "Missing required metric keys: " + ", ".join(missing_required),
                "error_type": "parse", "output": output}

    return metrics
