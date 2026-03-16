#!/usr/bin/env python3
"""
Autoresearch loop: LLM-driven iterative model improvement.

Runs on the H100 GPU container. Each iteration:
  1. Reads program.md + current train.py + history of past experiments
  2. Calls Claude API to propose a modified train.py
  3. Validates syntax, runs training (5 min), captures val_sharpe
  4. If improved → keep. If worse → revert.
  5. Logs everything to experiments.jsonl

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")
PROGRAM_MD = os.path.join(SCRIPT_DIR, "program.md")
EXPERIMENTS_LOG = os.path.join(SCRIPT_DIR, "experiments.jsonl")
BEST_TRAIN_PY = os.path.join(SCRIPT_DIR, "best_train.py")
BEST_MODEL_PT = os.path.join(SCRIPT_DIR, "best_model.pt")
STATUS_JSON = os.path.join(SCRIPT_DIR, "status.json")
PYTHON = sys.executable

# Claude model for code generation
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 12000  # enough for full train.py rewrite

# Reusable anthropic client (avoid httpx connection pool leak)
_anthropic_client = None


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


# ---------------------------------------------------------------------------
# Diagnostics — capture what kills the process and track resources
# ---------------------------------------------------------------------------

_DIAG_FILE = os.path.join(SCRIPT_DIR, "diagnostics.log")


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

def call_claude(system_prompt: str, user_prompt: str) -> str:
    """Call Claude API and return the text response."""
    client = _get_anthropic_client()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def build_system_prompt(program_md: str) -> str:
    return f"""You are an expert ML researcher iterating on a TWO-HEAD SPX 0DTE options sniper model.
Your job: modify train.py to improve the composite score:
  if tpd <= 6: freq_mult = min(1, tpd/2)
  else:        freq_mult = max(0.1, (6/tpd)²)  # QUADRATIC decay
  score = profit_factor × trade_sharpe × freq_mult

CRITICAL: The scoring has a BELL-CURVE trade frequency:
  - trades_per_day < 0.5 → score = -10.0 (hard penalty)
  - trades_per_day 0.5–1.5 → score ramps from -5 toward raw score
  - trades_per_day 2–6 → SWEET SPOT — full score
  - trades_per_day > 6 → QUADRATIC OVER-TRADING PENALTY (tpd=10→0.36x, tpd=12→0.25x, tpd=15→0.16x)
  - Baseline is -5.0. Model MUST trade 1.5+ times/day to get a positive score.

The model is a SNIPER: 3-5 trades/day, not 20+. Over-trading is penalized.
DO NOT add TRADE bias to gate head. DO NOT add TRADE_INCENTIVE_WEIGHT.
These cause the gate to collapse to always-TRADE (0% DO_NOTHING, 22+ trades/day).
If score is negative due to ZERO trades, focus on getting the model to trade.
If score is low due to OVER-TRADING (>10 tpd), make the model more selective.

EVALUATION RULES (enforced by evaluate_trades):
  - No entries before 10:00 AM (first 30 bars blocked — model still sees them in lookback)
  - 5-bar cooldown after stop loss before re-entry is allowed
  - Individual trade P&L capped at 200% (no fat-tail lottery dependency)
  - Score penalties applied AFTER the base formula:
    * Consecutive losses >3: score *= max(0.5, 1.0 - 0.05*(consec-3))
    * 1-bar holds >30%: score *= max(0.7, 1.0 - (short_pct - 0.30))
    * Stop-loss rate >30%: score *= max(0.5, 1.0 - (sl_rate - 0.30))
  - Over-trading penalty is QUADRATIC above 6 tpd (much steeper than linear)

TUNABLE EVALUATION PARAMETERS (pass as kwargs to evaluate_trades):
  evaluate_trades(model, data, LOOKBACK, device, stop_loss_pct=0.20, max_hold_bars=45, max_trade_return=1.5)
  - stop_loss_pct: default 0.30 (30%), range 0.15–0.50. Tighter = less drawdown but more stops.
  - max_hold_bars: default 60 (60 min), range 15–90. Shorter = less theta decay risk.
  - max_trade_return: default 2.0 (200%), range 0.5–5.0. Caps individual trade P&L to prevent fat-tail dependency.
  These are strategy-level knobs you can experiment with alongside architecture/loss.

ARCHITECTURE (TWO-HEAD — DO NOT MERGE INTO SINGLE HEAD):
The model has TWO separate output heads:
  1. Gate head: (batch, 2) → [NO_TRADE, TRADE] — decides "should I be in a trade?"
  2. Direction head: (batch, 2) → [CALL, PUT] — decides "which direction?"

forward() MUST return a tuple: (gate_logits, dir_logits)
  - Gate=TRADE + Dir=CALL → BUY_CALL
  - Gate=TRADE + Dir=PUT → BUY_PUT
  - Gate=NO_TRADE while in position → EXIT (handled at inference by evaluate_trades)
  - Gate=NO_TRADE while flat → DO_NOTHING

This gives 4 effective actions: DO_NOTHING (0), BUY_CALL (1), BUY_PUT (2), EXIT (3).

LOSS (OPTION P&L — DO NOT REVERT TO FORWARD-RETURN PERCENTILES):
The loss function sniper_loss() trains on ACTUAL SPXW option P&L, not forward-return
percentiles. The dataloader yields: (x, (fwd_ret, call_pnl, put_pnl, exit_call, exit_put))
  - Gate targets: TRADE (1) when call_pnl > 0 OR put_pnl > 0 (actual profitable trade)
  - Direction targets: CALL (0) when call_pnl > put_pnl, else PUT (1)
  - Only computed on bars with option P&L data (NaN-masked)
  - Time-weighted: afternoon errors cost more (theta acceleration)

DO NOT revert to the old sniper_loss that uses forward_return percentiles.
DO NOT merge gate_head and dir_head into a single head.
The two-head architecture and option-P&L loss are load-bearing design decisions.

RULES:
- THINK FIRST: Before writing any code, write 2-3 sentences explaining your hypothesis
  and what you expect to change. Wrap this in <reasoning>...</reasoning> tags.
  Then output the complete modified train.py.
- You MUST output the COMPLETE modified train.py file, not a diff or snippet.
- Output ONLY the reasoning tags + Python code. No markdown fences, no other text.
- Do NOT modify imports from prepare.py — those are fixed.
- Do NOT change the evaluation section or save section at the bottom.
- The training budget is fixed at 300 seconds. Do not change TIME_BUDGET.
- Keep the same output format (score, profit_factor, trade_sharpe, etc.) so results parse.
- Be creative but disciplined. One major change per iteration works best.
- If the last experiment failed (syntax error, NaN loss, crash), fix it.

MEMORY / SAFETY CONSTRAINTS (VIOLATION = CONTAINER CRASH):
- NEVER use torch.compile() — it causes OOM on 64Gi Akash containers. The model is <1M params; compile overhead >> benefit.
- Keep BATCH_SIZE ≤ 256. Do NOT increase it beyond 256.
- Do NOT add nn.DataParallel or DistributedDataParallel. There is one GPU.
- Do NOT add gradient accumulation that stores extra tensors beyond what .backward() needs.
- Do NOT clone/copy the full dataset into new tensors. Use the existing data dict.
- Do NOT add data augmentation that duplicates the dataset in memory.
- Do NOT use model.half() or autocast — the model is tiny and does not need mixed precision hacks.
- Do NOT add torch.jit.trace or torch.jit.script.
- Keep model size under 2M parameters. Do NOT make D_MODEL > 128 or DEPTH > 8.
- The container has 64Gi RAM and an 80GB H100. Peak VRAM should stay under 40GB.

COMMON MISTAKES (auto-rejected or auto-fixed):
- Variable is LOOKBACK (not LOOKBOOK, lookBook, look_back, or similar).
- D_MODEL must be divisible by N_HEADS. Always verify: D_MODEL % N_HEADS == 0.
  Valid combos: 64/4, 80/4, 80/5, 96/4, 96/6, 112/4, 128/4, 128/8.
- If you change LOOKBACK, also update pos_embed shape to match.
- Loss must stay non-negative. If your loss function can produce negative values, add a lower bound.

DOMAIN KNOWLEDGE:
{program_md}"""


def build_user_prompt(current_train_py: str, history: list, experiment_id: int = 0) -> str:
    parts = []

    if history:
        parts.append("## Full Experiment History (all experiments, most recent last)\n")
        # Show ALL experiments so Claude can learn from the full trajectory
        for exp in history:
            status = "✓ KEPT" if exp.get("kept") else "✗ reverted"
            score = exp.get("score", exp.get("val_sharpe", "N/A"))
            reason = exp.get("change_summary", "unknown")
            trades = exp.get("trades_per_day", "?")
            pf = exp.get("profit_factor", "?")
            err = exp.get("error", "")
            reasoning = exp.get("reasoning", "")
            if err and err.startswith("SAFETY:"):
                parts.append(f"  #{exp['experiment_id']}: REJECTED — {err}")
            elif err:
                parts.append(f"  #{exp['experiment_id']}: FAILED — {err[:150]}")
            else:
                line = f"  #{exp['experiment_id']}: score={score} pf={pf} tpd={trades} [{status}] — {reason}"
                if reasoning:
                    line += f"\n    Reasoning: {reasoning}"
                parts.append(line)
        parts.append("")

        best = max(history, key=lambda e: e.get("score", e.get("val_sharpe", -999)))
        parts.append(f"## Current best: score={best.get('score', best.get('val_sharpe', 'N/A'))} (experiment #{best['experiment_id']})\n")

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
    else:
        parts.append("## This is the FIRST experiment. Run baseline as-is or make one small improvement.\n")

    parts.append("## Current train.py:\n```python\n" + current_train_py + "\n```\n")
    parts.append("Remember: First write <reasoning>your hypothesis</reasoning>, then output the complete modified train.py code.")

    return "\n".join(parts)


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


def run_training(train_py_path: str, timeout: int = 420) -> dict:
    """Run train.py and parse the output metrics.

    Timeout: 420s = 300s training budget + 120s buffer for data loading/eval.
    Returns dict with metrics or {'error': 'message'}.
    """
    try:
        result = subprocess.run(
            [PYTHON, "-u", train_py_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=SCRIPT_DIR,
        )

        output = result.stdout + result.stderr

        if result.returncode != 0:
            # Get last 50 lines of output for error context (30 was too short, tracebacks got truncated)
            lines = output.strip().split('\n')
            tail = '\n'.join(lines[-50:])
            return {"error": f"Exit code {result.returncode}:\n{tail}", "output": output}

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
                           'max_equity_dd', 'total_dollar_return'):
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        pass
                elif key in ('num_trades', 'num_val_bars', 'num_val_days',
                             'num_steps', 'num_params', 'max_consec_loss',
                             'model_exit_count', 'cooldown_blocked',
                             'pre_10am_blocked'):
                    try:
                        metrics[key] = int(val.replace(',', ''))
                    except ValueError:
                        pass
                elif key in ('lookback', 'depth', 'd_model'):
                    metrics[key] = val

        if 'score' not in metrics:
            return {"error": f"Could not parse score from output:\n{output[-500:]}",
                    "output": output}

        # Drop full output to free memory early (metrics dict is returned to caller)
        del metrics["output"]
        return metrics

    except subprocess.TimeoutExpired:
        return {"error": f"Training timed out after {timeout}s"}
    except Exception as e:
        return {"error": f"Exception: {e}"}


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

def load_history() -> list:
    """Load experiment history from JSONL file."""
    if not os.path.exists(EXPERIMENTS_LOG):
        return []
    history = []
    with open(EXPERIMENTS_LOG, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return history


def append_experiment(exp: dict):
    """Append one experiment to the JSONL log."""
    with open(EXPERIMENTS_LOG, 'a') as f:
        # Don't log the full output/code to keep the file manageable
        log_entry = {k: v for k, v in exp.items() if k != 'output'}
        f.write(json.dumps(log_entry) + '\n')


def write_status(phase: str, experiment_id: int, best_score: float,
                 kept: int, failed: int, total: int, deadline: float,
                 last_exp: dict | None = None):
    """Write status.json for the monitor to read."""
    remaining = max(0, (deadline - time.time()) / 3600)
    status = {
        "phase": phase,
        "experiment_id": experiment_id,
        "best_score": round(best_score, 6) if best_score > -999 else None,
        "kept": kept,
        "failed": failed,
        "total": total,
        "time_remaining_h": round(remaining, 2),
        "updated": datetime.datetime.now().isoformat(),
    }
    if last_exp:
        status["last_change"] = last_exp.get("change_summary", "")
        status["last_score"] = last_exp.get("score", last_exp.get("val_sharpe"))
        status["last_kept"] = last_exp.get("kept", False)
    # Atomic write to avoid partial reads
    tmp = STATUS_JSON + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, STATUS_JSON)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_one_experiment(experiment_id: int, history: list, best_score: float,
                       deadline: float, kept_count: int, failed_count: int) -> dict:
    """Run a single experiment iteration. Returns experiment dict."""
    log(f"=== Experiment #{experiment_id} ===")
    total = len(history)

    # Read current state
    with open(PROGRAM_MD, 'r') as f:
        program_md = f.read()
    with open(TRAIN_PY, 'r') as f:
        current_code = f.read()

    # Call Claude
    write_status("calling_claude", experiment_id, best_score,
                 kept_count, failed_count, total, deadline)
    log("Calling Claude for code modification...")
    t0 = time.time()
    try:
        system = build_system_prompt(program_md)
        user = build_user_prompt(current_code, history, experiment_id)
        raw_response = call_claude(system, user)
        reasoning = extract_reasoning(raw_response)
        new_code = extract_code(raw_response)
        new_code, stripped = _sanitize_code(new_code)
        api_time = time.time() - t0
        log(f"  Claude responded in {api_time:.1f}s")
        if reasoning:
            log(f"  Reasoning: {reasoning[:200]}")
        if stripped:
            log(f"  Auto-fixed: {'; '.join(stripped)}")
    except Exception as e:
        log(f"  Claude API error: {e}")
        return {
            "experiment_id": experiment_id,
            "error": f"API error: {e}",
            "kept": False,
            "score": -999,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # Validate syntax
    syntax_err = validate_syntax(new_code)
    if syntax_err:
        log(f"  Syntax error: {syntax_err}")
        return {
            "experiment_id": experiment_id,
            "error": syntax_err,
            "kept": False,
            "score": -999,
            "change_summary": extract_change_summary(raw_response, current_code, new_code),
            "reasoning": reasoning[:300] if reasoning else "",
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # Validate safety (torch.compile, huge batch sizes, etc.)
    safety_err = validate_safety(new_code)
    if safety_err:
        log(f"  {safety_err}")
        return {
            "experiment_id": experiment_id,
            "error": safety_err,
            "kept": False,
            "score": -999,
            "change_summary": extract_change_summary(raw_response, current_code, new_code),
            "reasoning": reasoning[:300] if reasoning else "",
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # Backup current train.py and best_model.pt
    backup_path = TRAIN_PY + ".backup"
    shutil.copy2(TRAIN_PY, backup_path)
    model_backup_path = BEST_MODEL_PT + ".backup"
    if os.path.exists(BEST_MODEL_PT):
        shutil.copy2(BEST_MODEL_PT, model_backup_path)

    # Write new train.py
    with open(TRAIN_PY, 'w') as f:
        f.write(new_code)

    change_summary = extract_change_summary(raw_response, current_code, new_code)
    log(f"  Change: {change_summary}")

    # Run training
    write_status("training", experiment_id, best_score,
                 kept_count, failed_count, total, deadline,
                 {"change_summary": change_summary})
    log("  Training (5 min budget)...")
    log_diagnostics(f"pre_train_{experiment_id}")
    t0 = time.time()
    metrics = run_training(TRAIN_PY)
    train_wall_time = time.time() - t0
    log(f"  Done in {train_wall_time:.0f}s")
    log_diagnostics(f"post_train_{experiment_id}")

    exp = {
        "experiment_id": experiment_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "change_summary": change_summary,
        "reasoning": reasoning[:300] if reasoning else "",
        "api_time": round(api_time, 1),
        "train_wall_time": round(train_wall_time, 1),
    }

    if "error" in metrics:
        log(f"  FAILED: {metrics['error'][:200]}")
        exp["error"] = metrics["error"][:500]
        exp["score"] = -999
        exp["kept"] = False
        # Revert train.py and model weights
        shutil.copy2(backup_path, TRAIN_PY)
        if os.path.exists(model_backup_path):
            shutil.copy2(model_backup_path, BEST_MODEL_PT)
        log("  Reverted to previous train.py + model")
    else:
        score = metrics["score"]
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

        if score > best_score:
            log(f"  ✓ IMPROVED: {best_score:.4f} → {score:.4f} (pf={exp['profit_factor']:.2f} tpd={exp['trades_per_day']:.1f})")
            exp["kept"] = True
            # Save as best — keep new model weights, archive old backup
            shutil.copy2(TRAIN_PY, BEST_TRAIN_PY)
            if os.path.exists(model_backup_path):
                os.remove(model_backup_path)
        else:
            log(f"  ✗ No improvement: {score:.4f} ≤ {best_score:.4f}")
            exp["kept"] = False
            # Revert train.py and model weights
            shutil.copy2(backup_path, TRAIN_PY)
            if os.path.exists(model_backup_path):
                shutil.copy2(model_backup_path, BEST_MODEL_PT)

    # Clean up backups
    for bkp in [backup_path, model_backup_path]:
        if os.path.exists(bkp):
            os.remove(bkp)

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

    # -----------------------------------------------------------------------
    # Pre-flight checks — catch problems before consuming GPU time
    # -----------------------------------------------------------------------
    print("=== PRE-FLIGHT CHECKS ===")

    # 1. data.pt — load and validate contents
    try:
        sys.path.insert(0, SCRIPT_DIR)
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
        if nan_pct > 50:
            print("  WARNING: >50% NaN features — training quality will be poor")
        del _data
    except SystemExit:
        print("ERROR: data.pt not found. Upload it to the container first.")
        print("  Expected at: ~/.cache/autoresearch-trading/features/data.pt")
        print("  Or next to train.py")
        sys.exit(1)
    except Exception as e:
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

    print("=== ALL CHECKS PASSED ===\n")

    # Load history
    history = load_history()
    # Warm-start baseline at -5.0 (not -999): a zero-trade model returns -10.0
    # so it must actually trade profitably to beat baseline and get "kept"
    INITIAL_BASELINE = -5.0
    best_score = max((e.get("score", e.get("val_sharpe", INITIAL_BASELINE)) for e in history), default=INITIAL_BASELINE)
    start_id = max((e.get("experiment_id", 0) for e in history), default=0) + 1

    log(f"Autoresearch loop starting")
    log(f"  Runtime budget: {args.hours}h ({args.hours * 60:.0f} min)")
    log(f"  Max experiments: {args.max_experiments}")
    log(f"  Previous experiments: {len(history)}")
    log(f"  Best score so far: {best_score:.4f}" if best_score > INITIAL_BASELINE else "  No previous results (baseline: -5.0)")
    log(f"  Model: {CLAUDE_MODEL}")
    log(f"  Log: {EXPERIMENTS_LOG}")
    log("")

    if args.dry_run:
        log("Dry run complete. All pre-flight checks passed.")
        return

    # Install signal handlers to catch what kills us
    install_signal_handlers()
    _diag(f"Loop starting: hours={args.hours}, max_experiments={args.max_experiments}")
    log_diagnostics("startup")

    # Save initial train.py as best if no best exists
    if not os.path.exists(BEST_TRAIN_PY):
        shutil.copy2(TRAIN_PY, BEST_TRAIN_PY)

    deadline = time.time() + args.hours * 3600
    experiment_id = start_id
    kept_count = 0
    failed_count = 0
    consecutive_failures = 0

    while time.time() < deadline and experiment_id < start_id + args.max_experiments:
        remaining_h = (deadline - time.time()) / 3600
        log(f"Time remaining: {remaining_h:.1f}h | Best score: {best_score:.4f} | "
            f"Kept: {kept_count} | Failed: {failed_count}")

        exp = run_one_experiment(experiment_id, history, best_score,
                                deadline, kept_count, failed_count)
        history.append(exp)
        append_experiment(exp)

        # Reap zombies, GC, diagnostics after each experiment
        _reap_zombies()
        gc.collect()
        log_diagnostics(f"post_exp_{experiment_id}")

        if exp.get("kept"):
            best_score = exp["score"]
            kept_count += 1
            consecutive_failures = 0
        elif exp.get("error"):
            failed_count += 1
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        # Update status after experiment
        write_status("between_experiments", experiment_id, best_score,
                     kept_count, failed_count, len(history), deadline, exp)

        # Safety: if 5 consecutive failures, restore best and continue
        if consecutive_failures >= 5:
            log("WARNING: 5 consecutive failures. Restoring best_train.py")
            if os.path.exists(BEST_TRAIN_PY):
                shutil.copy2(BEST_TRAIN_PY, TRAIN_PY)
            consecutive_failures = 0

        experiment_id += 1
        log("")

    # Final summary
    write_status("completed", experiment_id - 1, best_score,
                 kept_count, failed_count, len(history), deadline)
    log("=" * 60)
    log("AUTORESEARCH COMPLETE")
    log(f"  Total experiments: {experiment_id - start_id}")
    log(f"  Kept improvements: {kept_count}")
    log(f"  Failed: {failed_count}")
    log(f"  Best score: {best_score:.6f}")
    log("")

    if history:
        log("Top 5 experiments by score:")
        ranked = sorted(history, key=lambda e: e.get("score", e.get("val_sharpe", -999)), reverse=True)
        for i, exp in enumerate(ranked[:5]):
            sc = exp.get('score', exp.get('val_sharpe', -999))
            log(f"  #{exp['experiment_id']}: score={sc:.4f} pf={exp.get('profit_factor', 0):.2f} "
                f"tpd={exp.get('trades_per_day', 0):.1f} — {exp.get('change_summary', 'N/A')}")

    log(f"\nBest model saved at: {BEST_TRAIN_PY}")
    log(f"Full log at: {EXPERIMENTS_LOG}")


if __name__ == "__main__":
    main()
