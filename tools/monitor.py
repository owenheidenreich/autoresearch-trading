#!/usr/bin/env python3
"""
Autoresearch training dashboard — rich terminal UI with live auto-refresh.

Usage:
  # Auto-detect: reads .deploy-state for remote, falls back to local results
  python3 tools/monitor.py

  # Remote (SSH into H100):
  python3 tools/monitor.py --host provider.h100.ams.val.akash.pub --port 31116

  # Local only (watch synced results root):
  python3 tools/monitor.py --local results

  # Faster polling:
  python3 tools/monitor.py --interval 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.columns import Columns
    from rich.progress_bar import ProgressBar
    from rich import box
except ImportError:
    print("ERROR: dashboard requires `rich` (pip install rich)")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
DEPLOY_STATE = PROJECT_ROOT / ".deploy-state"
SSH_PASS = "autoresearch2026"


# -- Phase labels ----------------------------------------------------------

PHASE_LABELS = {
    "calling_claude": ("Generating code", "bold yellow"),
    "training": ("Training model", "bold green"),
    "evaluating": ("Evaluating results", "bold cyan"),
    "saving": ("Saving artifacts", "bold blue"),
    "dry_run": ("Dry run", "dim"),
    "completed": ("Completed", "bold white"),
    "error": ("Error", "bold red"),
    "startup": ("Starting up", "bold yellow"),
    "smoke_check": ("Smoke check", "bold magenta"),
    "between_experiments": ("Between experiments", "bold blue"),
}


# -- GPU health color thresholds -------------------------------------------

def gpu_util_style(pct: float) -> str:
    if pct >= 80:
        return "bold green"
    elif pct >= 40:
        return "yellow"
    elif pct > 0:
        return "bold red"
    return "dim"


def gpu_mem_style(used_mb: float, total_mb: float) -> str:
    if total_mb <= 0:
        return "dim"
    ratio = used_mb / total_mb
    if ratio >= 0.8:
        return "bold red"
    elif ratio >= 0.5:
        return "yellow"
    return "green"


def gpu_temp_style(temp_c: float) -> str:
    if temp_c >= 85:
        return "bold red"
    elif temp_c >= 70:
        return "yellow"
    return "green"


# -- Data fetching ---------------------------------------------------------

def load_deploy_state() -> dict | None:
    """Load .deploy-state if it exists."""
    if not DEPLOY_STATE.exists():
        return None
    state = {}
    for line in DEPLOY_STATE.read_text().strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            state[k.strip()] = v.strip()
    return state if state.get("SSH_HOST") else None


def fetch_remote(host: str, port: int) -> tuple[list[dict], dict | None, str | None, str | None, dict | None]:
    """Fetch experiments + status + log tail + results.tsv + GPU info from remote H100 via SSH."""
    env = {**os.environ, "SSHPASS": SSH_PASS}
    ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

    remote_script = (
        'RUN=$(cat /root/results/current_run.txt 2>/dev/null | tr -d "\\r\\n"); '
        'if [ -z "$RUN" ]; then echo "---SEP---"; echo "---SEP---"; echo "---SEP---"; echo "---SEP---"; echo "---SEP---"; exit 0; fi; '
        'cat /root/results/"$RUN"/experiments.v2.jsonl 2>/dev/null; echo "---SEP---"; '
        'cat /root/results/"$RUN"/status.json 2>/dev/null; echo "---SEP---"; '
        'tail -40 /root/loop.log 2>/dev/null; echo "---SEP---"; '
        'cat /root/results/"$RUN"/results.tsv 2>/dev/null; echo "---SEP---"; '
        'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null; echo "---SEP---"; '
        'echo "$RUN"'
    )
    try:
        r = subprocess.run(
            ["sshpass", "-e", "ssh"] + ssh_opts.split() + [
                "-p", str(port), f"root@{host}", f"bash -c {remote_script!r}",
            ],
            capture_output=True, text=True, timeout=15, env=env,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return [], None, None, None, None
        parts = r.stdout.split("---SEP---")
        experiments = _parse_jsonl(parts[0] if len(parts) > 0 else "")
        status = _parse_json(parts[1] if len(parts) > 1 else "")
        log_tail = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        results_tsv = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
        gpu_info = _parse_gpu_csv(parts[4].strip() if len(parts) > 4 else "")
        if status and len(parts) > 5 and parts[5].strip():
            status["_run_name"] = parts[5].strip()
        return experiments, status, log_tail, results_tsv, gpu_info
    except Exception:
        return [], None, None, None, None


def _parse_gpu_csv(text: str) -> dict | None:
    """Parse nvidia-smi CSV output into a dict."""
    text = text.strip()
    if not text:
        return None
    try:
        parts = [p.strip() for p in text.split(",")]
        return {
            "name": parts[0] if len(parts) > 0 else "unknown",
            "mem_used_mb": float(parts[1]) if len(parts) > 1 else 0,
            "mem_total_mb": float(parts[2]) if len(parts) > 2 else 0,
            "util_pct": float(parts[3]) if len(parts) > 3 else 0,
            "temp_c": float(parts[4]) if len(parts) > 4 else 0,
            "power_w": float(parts[5]) if len(parts) > 5 else 0,
            "power_limit_w": float(parts[6]) if len(parts) > 6 else 0,
        }
    except (ValueError, IndexError):
        return None


def fetch_local_run(run_dir: Path) -> tuple[list[dict], dict | None]:
    """Load experiments + status from a local run directory."""
    experiments = _parse_jsonl_file(run_dir / "experiments.v2.jsonl")
    status = _parse_json_file(run_dir / "status.json")
    if status:
        status["_run_name"] = run_dir.name
    return experiments, status


def fetch_all_local_runs() -> list[dict]:
    """Scan results/ for all run directories and return summary info."""
    runs = []
    if not RESULTS_ROOT.exists():
        return runs
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir() or not d.name.startswith("run-"):
            continue
        status = _parse_json_file(d / "status.json")
        experiments = _parse_jsonl_file(d / "experiments.v2.jsonl")
        meta = _parse_json_file(d / "run_metadata.json")
        runs.append({
            "name": d.name,
            "path": d,
            "status": status,
            "experiments": experiments,
            "metadata": meta,
        })
    return runs


def get_active_run_name() -> str | None:
    """Read current_run.txt pointer."""
    pointer = RESULTS_ROOT / "current_run.txt"
    if pointer.exists():
        name = pointer.read_text().strip()
        return name if name else None
    return None


# -- Parse helpers ---------------------------------------------------------

def _parse_jsonl(text: str) -> list[dict]:
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_jsonl_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _parse_jsonl(path.read_text())


def _parse_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _parse_json(path.read_text())


# -- Sparkline -------------------------------------------------------------

def sparkline(values: list[float], width: int = 30) -> str:
    if not values:
        return ""
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    return "".join(blocks[min(8, int((v - mn) / rng * 8))] for v in sampled)


def fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def fmt_score(score: float) -> Text:
    """Color-code a score value."""
    if score <= -999:
        return Text("  FAIL", style="dim red")
    elif score <= 0:
        return Text(f"{score:6.2f}", style="red")
    elif score < 2:
        return Text(f"{score:6.2f}", style="yellow")
    else:
        return Text(f"{score:6.2f}", style="green")


def phase_text(phase: str) -> Text:
    """Render phase with appropriate color."""
    label, style = PHASE_LABELS.get(phase, (phase, "white"))
    return Text(label, style=style)


# -- Dashboard panels ------------------------------------------------------

def build_gpu_panel(gpu: dict | None) -> Panel:
    """GPU health panel with color-coded metrics."""
    if not gpu:
        return Panel(
            Text("  GPU data unavailable (SSH fetch failed)", style="dim red"),
            title="[bold cyan]GPU Health[/]",
            border_style="red",
        )

    name = gpu["name"]
    mem_used = gpu["mem_used_mb"]
    mem_total = gpu["mem_total_mb"]
    util = gpu["util_pct"]
    temp = gpu["temp_c"]
    power = gpu["power_w"]
    power_limit = gpu["power_limit_w"]

    mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
    power_pct = (power / power_limit * 100) if power_limit > 0 else 0

    # Build bar visualizations
    def bar(pct: float, width: int = 20) -> str:
        filled = int(pct / 100 * width)
        return "\u2588" * filled + "\u2591" * (width - filled)

    lines = Text()
    lines.append(f"  {name}\n\n", style="bold white")

    # GPU Utilization
    lines.append("  Util:  ", style="bold")
    lines.append(f"{bar(util)} ", style=gpu_util_style(util))
    lines.append(f"{util:.0f}%\n", style=gpu_util_style(util))

    # VRAM
    lines.append("  VRAM:  ", style="bold")
    lines.append(f"{bar(mem_pct)} ", style=gpu_mem_style(mem_used, mem_total))
    lines.append(f"{mem_used:.0f}/{mem_total:.0f} MB ({mem_pct:.0f}%)\n", style=gpu_mem_style(mem_used, mem_total))

    # Temperature
    lines.append("  Temp:  ", style="bold")
    lines.append(f"{bar(min(temp, 100))} ", style=gpu_temp_style(temp))
    lines.append(f"{temp:.0f}C\n", style=gpu_temp_style(temp))

    # Power
    power_style = "bold red" if power_pct > 95 else ("yellow" if power_pct > 80 else "green")
    lines.append("  Power: ", style="bold")
    lines.append(f"{bar(power_pct)} ", style=power_style)
    lines.append(f"{power:.0f}/{power_limit:.0f}W ({power_pct:.0f}%)", style=power_style)

    border = "green"
    if util < 10 and mem_used < 100:
        border = "yellow"  # idle
    elif temp >= 85 or mem_pct >= 95:
        border = "red"  # overloaded

    return Panel(lines, title="[bold cyan]GPU Health[/]", border_style=border)


def build_session_runs_table(runs: list[dict], active_run: str | None) -> Table:
    """Table showing all runs from today's session."""
    table = Table(
        box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
        title="Session Runs", title_style="bold cyan",
    )
    table.add_column("Run", style="dim", max_width=28)
    table.add_column("Status", width=12)
    table.add_column("Exp", justify="right", width=4)
    table.add_column("Kept", justify="right", width=4)
    table.add_column("Best", justify="right", width=8)
    table.add_column("Failures", justify="right", width=8)
    table.add_column("Duration", justify="right", width=8)

    for run in runs:
        name = run["name"]
        st = run.get("status") or {}
        exps = run.get("experiments", [])

        is_active = name == active_run
        name_style = "bold white" if is_active else "dim"
        marker = "\u25b6 " if is_active else "  "

        phase = st.get("phase", "unknown")
        n_total = len(exps)
        n_kept = sum(1 for e in exps if e.get("kept"))
        n_failed = sum(1 for e in exps if e.get("error"))
        best = max((e.get("score", -999) for e in exps), default=-999)
        best_str = f"{best:.2f}" if best > -999 else "\u2014"

        duration_str = "\u2014"
        if exps:
            try:
                t0 = datetime.fromisoformat(exps[0]["timestamp"])
                t1 = datetime.fromisoformat(exps[-1]["timestamp"])
                duration_str = fmt_duration((t1 - t0).total_seconds())
            except (KeyError, ValueError):
                pass

        table.add_row(
            Text(f"{marker}{name}", style=name_style),
            phase_text(phase) if is_active else Text(phase, style="dim"),
            str(n_total),
            str(n_kept) if n_kept > 0 else Text("0", style="dim"),
            best_str,
            str(n_failed) if n_failed > 0 else Text("0", style="dim"),
            duration_str,
        )
    return table


def build_active_header(status: dict | None, experiments: list[dict], mode: str) -> Panel:
    """Build the active run header panel with progress info."""
    if not status:
        return Panel(Text("No active run detected", style="dim"), title="Active Run")

    run_name = status.get("_run_name", "unknown")
    phase = status.get("phase", "unknown")
    exp_id = status.get("experiment_id", 0)
    best_score = status.get("best_score", -999)
    kept = status.get("kept", 0)
    failed = status.get("failed", 0)
    total = status.get("total", 0)
    time_left = status.get("time_remaining_h", 0)
    updated = status.get("updated", "")
    contract = status.get("contract_checksum", "?")

    # Time since last update
    staleness = ""
    if updated:
        try:
            last = datetime.fromisoformat(updated)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            ago = (datetime.now(timezone.utc) - last).total_seconds()
            if ago > 120:
                staleness = f"  [bold red](stale: {fmt_duration(ago)} ago)[/]"
            elif ago > 30:
                staleness = f"  [yellow](updated {fmt_duration(ago)} ago)[/]"
        except ValueError:
            pass

    # Experiment rate
    rate_str = ""
    if len(experiments) >= 2:
        try:
            t0 = datetime.fromisoformat(experiments[0]["timestamp"])
            t1 = datetime.fromisoformat(experiments[-1]["timestamp"])
            elapsed_h = (t1 - t0).total_seconds() / 3600
            if elapsed_h > 0:
                rate = len(experiments) / elapsed_h
                rate_str = f"  [dim]{rate:.1f} exp/hr[/]"
        except (KeyError, ValueError):
            pass

    lines = []
    lines.append(f"[bold]{run_name}[/]  |  Mode: [cyan]{mode}[/]  |  Contract: [dim]{contract}[/]{rate_str}{staleness}")
    lines.append("")

    # Phase + progress
    phase_label, phase_style = PHASE_LABELS.get(phase, (phase, "white"))
    lines.append(f"  Phase:     [{phase_style}]* {phase_label}[/]  --  Experiment #{exp_id}")

    # Time left with progress bar
    if time_left > 0:
        total_h = 8.0  # default budget
        elapsed_h = total_h - time_left
        pct = min(100, elapsed_h / total_h * 100) if total_h > 0 else 0
        bar_w = 20
        filled = int(pct / 100 * bar_w)
        bar_str = "\u2588" * filled + "\u2591" * (bar_w - filled)
        lines.append(f"  Time:      [{bar_str}] [bold]{time_left:.1f}h left[/] ({pct:.0f}% elapsed)")
    elif phase == "completed":
        lines.append(f"  Time:      [dim]Completed[/]")

    lines.append(f"  Progress:  [green]{kept} kept[/]  |  [red]{failed} failed[/]  |  {total} total")
    lines.append(f"  Best:      [bold green]{best_score:.4f}[/]" if best_score > -999 else "  Best:      [dim]none[/]")

    # Current experiment detail from status
    last_change = status.get("last_change")
    last_failure = status.get("last_failure_type")
    last_anomalies = status.get("last_anomaly_flags", [])
    if last_change:
        lines.append(f"  Last D:    {last_change[:100]}")
    if last_failure and last_failure != "none":
        lines.append(f"  Last Fail: [red]{last_failure}[/]")
    if last_anomalies:
        flags = ", ".join(last_anomalies[:4])
        lines.append(f"  Anomalies: [yellow]{flags}[/]")

    return Panel("\n".join(lines), title="[bold cyan]Active Run[/]", border_style="cyan")


def build_experiment_table(experiments: list[dict], max_rows: int = 25) -> Table:
    """Detailed experiment history table."""
    table = Table(
        box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
        title="Experiment History", title_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Result", width=10)
    table.add_column("Score", justify="right", width=8)
    table.add_column("PF", justify="right", width=6)
    table.add_column("TPD", justify="right", width=5)
    table.add_column("Sharpe", justify="right", width=7)
    table.add_column("WR", justify="right", width=5)
    table.add_column("R:R", justify="right", width=5)
    table.add_column("SL%", justify="right", width=5)
    table.add_column("Hold", justify="right", width=5)
    table.add_column("Exit%", justify="right", width=5)
    table.add_column("API", justify="right", width=5)
    table.add_column("Train", justify="right", width=5)
    table.add_column("Change", no_wrap=False, max_width=40)

    for e in experiments[-max_rows:]:
        idx = e.get("experiment_id", "?")

        # Result column
        if e.get("kept"):
            result = Text("+ KEPT", style="bold green")
        elif e.get("error"):
            ft = e.get("failure_type", "error")
            result = Text(f"x {ft[:7]}", style="red")
        else:
            result = Text("~ revert", style="yellow")

        score = e.get("score", -999)
        pf = e.get("profit_factor", 0)
        tpd = e.get("trades_per_day", 0)
        sharpe = e.get("trade_sharpe", 0)
        wr = e.get("win_rate", 0)
        rr = e.get("rr_ratio", 0)
        sl = e.get("stop_loss_rate", 0)
        avg_hold = e.get("avg_hold_bars", 0)
        model_exit = e.get("model_exit_rate", 0)
        api_time = e.get("api_time", 0)
        train_time = e.get("train_wall_time", 0)
        change = e.get("change_summary", e.get("error", ""))[:40]

        # Color score
        if score <= -999:
            score_text = Text("  FAIL", style="dim red")
        elif score <= 0:
            score_text = Text(f"{score:6.2f}", style="red")
        else:
            score_text = Text(f"{score:6.2f}", style="green" if e.get("kept") else "yellow")

        table.add_row(
            str(idx),
            result,
            score_text,
            f"{pf:.2f}" if pf else "\u2014",
            f"{tpd:.1f}" if tpd else "\u2014",
            f"{sharpe:.2f}" if sharpe else "\u2014",
            f"{wr:.0%}" if wr else "\u2014",
            f"{rr:.2f}" if rr else "\u2014",
            f"{sl:.0%}" if sl else "\u2014",
            f"{avg_hold:.0f}" if avg_hold else "\u2014",
            f"{model_exit:.0%}" if model_exit else "\u2014",
            fmt_duration(api_time) if api_time else "\u2014",
            fmt_duration(train_time) if train_time else "\u2014",
            Text(change, style="dim"),
        )

    return table


def build_metrics_panel(experiments: list[dict]) -> Panel:
    """Sparkline charts for key metrics over time."""
    lines = []

    scored = [e for e in experiments if e.get("score", -999) > -999]
    if not scored:
        return Panel("No completed experiments yet...", title="Metrics", border_style="cyan")

    scores = [e.get("score", 0) for e in scored]
    lines.append(f"  Score  {sparkline(scores)}  [{min(scores):.1f} .. {max(scores):.1f}]")

    pfs = [e.get("profit_factor", 0) for e in scored]
    if any(pfs):
        lines.append(f"  PF     {sparkline(pfs)}  [{min(pfs):.2f} .. {max(pfs):.2f}]")

    tpds = [e.get("trades_per_day", 0) for e in scored]
    if any(tpds):
        lines.append(f"  TPD    {sparkline(tpds)}  [{min(tpds):.1f} .. {max(tpds):.1f}]")

    wrs = [e.get("win_rate", 0) for e in scored]
    if any(wrs):
        lines.append(f"  WinR   {sparkline(wrs)}  [{min(wrs):.0%} .. {max(wrs):.0%}]")

    sharpes = [e.get("trade_sharpe", 0) for e in scored]
    if any(sharpes):
        lines.append(f"  Sharpe {sparkline(sharpes)}  [{min(sharpes):.2f} .. {max(sharpes):.2f}]")

    rrs = [e.get("rr_ratio", 0) for e in scored if e.get("rr_ratio")]
    if rrs:
        lines.append(f"  R:R    {sparkline(rrs)}  [{min(rrs):.2f} .. {max(rrs):.2f}]")

    return Panel("\n".join(lines), title="[bold cyan]Metric Trends[/]", border_style="cyan")


def build_log_panel(log_tail: str | None) -> Panel:
    """Show recent log output — wider text, more lines, no truncation from borders."""
    if not log_tail:
        return Panel("[dim]No log output yet...[/]", title="Live Log", border_style="dim")

    lines = log_tail.strip().split("\n")[-20:]
    formatted = []
    PST = timezone(timedelta(hours=-7))
    for line in lines:
        # Convert UTC timestamps [HH:MM:SS] to PST
        if line and line[0] == "[" and len(line) > 9 and line[9] == "]":
            try:
                utc_time = datetime.strptime(line[1:9], "%H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
                pst_time = utc_time.astimezone(PST)
                line = f"[{pst_time.strftime('%H:%M:%S')}]{line[10:]}"
            except ValueError:
                pass
        if "ERROR" in line or "FAILED" in line:
            formatted.append(f"[red]{line}[/]")
        elif "KEPT" in line or "kept" in line or "PASSED" in line or "IMPROVED" in line:
            formatted.append(f"[green]{line}[/]")
        elif "cache_read=" in line or "Tokens:" in line:
            formatted.append(f"[cyan]{line}[/]")
        elif "WARNING" in line:
            formatted.append(f"[yellow]{line}[/]")
        elif "=== Experiment" in line:
            formatted.append(f"[bold white]{line}[/]")
        elif "score:" in line or "profit_factor:" in line:
            formatted.append(f"[bold]{line}[/]")
        else:
            formatted.append(f"[dim]{line}[/]")

    return Panel("\n".join(formatted), title="[bold cyan]Live Log (last 20 lines)[/]", border_style="dim", padding=(0, 1))


def build_latest_experiment(experiments: list[dict]) -> Panel:
    """Detailed view of the most recent experiment — full reasoning and change text."""
    if not experiments:
        return Panel("[dim]No experiments yet[/]", title="Latest Experiment")

    e = experiments[-1]
    idx = e.get("experiment_id", "?")
    lines = []

    # Status
    if e.get("kept"):
        lines.append(f"  [bold green]+ Experiment #{idx} -- KEPT[/]")
    elif e.get("error"):
        ft = e.get("failure_type", "error")
        lines.append(f"  [bold red]x Experiment #{idx} -- {ft}[/]")
    else:
        lines.append(f"  [yellow]~ Experiment #{idx} -- reverted (regression)[/]")

    # Scores — full detail with new metrics
    score = e.get("score", -999)
    if score > -999:
        pf = e.get("profit_factor", 0)
        tpd = e.get("trades_per_day", 0)
        sharpe = e.get("trade_sharpe", 0)
        wr = e.get("win_rate", 0)
        rr = e.get("rr_ratio", 0)
        avg_hold = e.get("avg_hold_bars", 0)
        model_exit = e.get("model_exit_rate", 0)
        sl = e.get("stop_loss_rate", 0)
        lines.append(
            f"  Score: {score:.4f}  |  PF: {pf:.2f}  |  TPD: {tpd:.1f}  |  "
            f"Sharpe: {sharpe:.2f}  |  WR: {wr:.0%}"
        )
        lines.append(
            f"  R:R: {rr:.2f}  |  Hold: {avg_hold:.0f} bars  |  "
            f"Exit: {model_exit:.0%}  |  SL: {sl:.0%}"
        )

    # Reasoning — show full text, wrapped naturally by Rich
    reasoning = e.get("reasoning", "")
    if reasoning:
        lines.append("")
        lines.append(f"  [bold]Reasoning:[/]")
        # Show full reasoning, let panel handle wrapping
        for r_line in reasoning.split("\n"):
            lines.append(f"    {r_line}")

    # Change — show full text
    change = e.get("change_summary", "")
    if change:
        lines.append("")
        lines.append(f"  [bold]Changes:[/]")
        for part in change.split(";"):
            part = part.strip()
            if part:
                lines.append(f"    {part}")

    # Error — show full text
    error = e.get("error", "")
    if error:
        lines.append("")
        lines.append(f"  [bold red]Error:[/]")
        for e_line in error[:500].split("\n"):
            lines.append(f"    [red]{e_line}[/]")

    # Anomalies
    anomalies = e.get("anomaly_flags", [])
    if anomalies:
        lines.append(f"  [yellow]Anomalies:[/] {', '.join(anomalies)}")

    # Timing
    api_time = e.get("api_time", 0)
    train_time = e.get("train_wall_time", 0)
    if api_time or train_time:
        lines.append(f"  [dim]API: {fmt_duration(api_time)}  |  Train: {fmt_duration(train_time)}[/]")

    return Panel("\n".join(lines), title="[bold cyan]Latest Experiment[/]", border_style="cyan", padding=(0, 1))


# -- Main dashboard --------------------------------------------------------

def build_dashboard(
    runs: list[dict],
    active_run: str | None,
    experiments: list[dict],
    status: dict | None,
    log_tail: str | None,
    gpu_info: dict | None,
    mode: str,
    poll_count: int,
    fetch_time: float = 0,
) -> Layout:
    """Assemble the full dashboard layout."""
    layout = Layout()

    # Top: timestamp + mode (PST)
    PST = timezone(timedelta(hours=-7))
    now = datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S PST")
    title = Text()
    title.append("  AUTORESEARCH DASHBOARD", style="bold cyan")
    title.append(f"  |  {now}  |  {mode}  |  poll #{poll_count}", style="dim")
    if fetch_time > 0:
        title.append(f"  |  fetch: {fetch_time:.1f}s", style="dim")

    layout.split_column(
        Layout(name="title", size=3),
        Layout(name="top_row", size=10),
        Layout(name="main"),
        Layout(name="bottom", size=18),
    )
    layout["title"].update(Panel(title, box=box.HEAVY))

    # Top row: GPU health + active header side by side
    layout["top_row"].split_row(
        Layout(name="gpu", ratio=2),
        Layout(name="active", ratio=3),
    )
    layout["top_row"]["gpu"].update(build_gpu_panel(gpu_info))
    layout["top_row"]["active"].update(build_active_header(status, experiments, mode))

    # Main area: left (metrics + runs) | right (experiment history)
    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )

    # Left: metrics + session runs
    layout["left"].split_column(
        Layout(name="metrics", size=10),
        Layout(name="runs"),
    )
    layout["left"]["metrics"].update(build_metrics_panel(experiments))
    layout["left"]["runs"].update(
        Panel(build_session_runs_table(runs, active_run), border_style="dim")
    )

    # Right: experiment table
    layout["right"].update(
        Panel(build_experiment_table(experiments, max_rows=25), border_style="cyan")
    )

    # Bottom: latest experiment (wider) + log
    layout["bottom"].split_row(
        Layout(name="latest", ratio=3),
        Layout(name="log", ratio=2),
    )
    layout["bottom"]["latest"].update(build_latest_experiment(experiments))
    layout["bottom"]["log"].update(build_log_panel(log_tail))

    return layout


# -- Entry point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autoresearch training dashboard")
    parser.add_argument("--host", type=str, help="Remote H100 hostname (overrides .deploy-state)")
    parser.add_argument("--port", type=int, help="Remote SSH port (overrides .deploy-state)")
    parser.add_argument("--local", type=str, help="Local results path (skip remote)")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval in seconds (default: 10)")
    args = parser.parse_args()

    # Resolve mode
    remote_host = args.host
    remote_port = args.port
    local_only = args.local is not None

    if not local_only and not remote_host:
        state = load_deploy_state()
        if state:
            remote_host = state.get("SSH_HOST")
            remote_port = int(state.get("SSH_PORT", 22))

    if remote_host:
        mode = f"SSH -> {remote_host}:{remote_port}"
    elif local_only:
        mode = f"Local -> {args.local}"
    else:
        mode = "Local -> results/"

    console = Console()
    console.clear()
    poll_count = 0

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        while True:
            poll_count += 1
            try:
                fetch_start = time.time()

                # Always load local runs for the session overview
                all_runs = fetch_all_local_runs()
                active_run = get_active_run_name()

                # Fetch active run data (remote preferred, local fallback)
                experiments = []
                status = None
                log_tail = None
                gpu_info = None

                if remote_host and not local_only:
                    experiments, status, log_tail, _, gpu_info = fetch_remote(remote_host, remote_port)

                # Fallback to local if remote returned nothing
                if not experiments and not status:
                    if args.local:
                        local_path = Path(args.local)
                    else:
                        local_path = RESULTS_ROOT

                    if active_run and (local_path / active_run).is_dir():
                        run_dir = local_path / active_run
                    elif local_path.is_dir() and (local_path / "current_run.txt").exists():
                        rn = (local_path / "current_run.txt").read_text().strip()
                        run_dir = local_path / rn if rn else local_path
                    else:
                        run_dir = local_path

                    if run_dir.is_dir():
                        experiments, status = fetch_local_run(run_dir)
                        for log_candidate in [run_dir / "loop.log", run_dir.parent / "loop.log"]:
                            if log_candidate.exists():
                                lines = log_candidate.read_text().split("\n")
                                log_tail = "\n".join(lines[-20:])
                                break

                fetch_time = time.time() - fetch_start

                dashboard = build_dashboard(
                    all_runs, active_run, experiments, status, log_tail, gpu_info, mode, poll_count, fetch_time,
                )
                live.update(dashboard)

            except KeyboardInterrupt:
                break
            except Exception as e:
                live.update(Panel(f"Error: {e}", border_style="red"))

            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break

    console.print("\n[dim]Dashboard stopped.[/]")


if __name__ == "__main__":
    main()
