#!/usr/bin/env python3
"""
Autoresearch training monitor — live terminal dashboard.

Usage:
  # Remote (SSH into H100):
  python3 monitor.py --host provider.h100.ams.val.akash.pub --port 31116 --password autoresearch2026

  # Local (watch canonical results root with current_run.txt):
  python3 monitor.py --local /path/to/results

  # Local (watch a specific run folder):
  python3 monitor.py --local /path/to/results/run-YYYY-MM-DD-HHMMSS
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
except ImportError:  # pragma: no cover - optional UI dependency
    Console = None
    Live = None
    Table = None
    Panel = None
    Layout = None
    Text = None
    box = None


def fetch_remote(host: str, port: int, password: str) -> tuple[list[dict], dict | None, str | None]:
    """Fetch canonical run-folder experiments.v2.jsonl + status.json via SSH."""
    env = {**os.environ, "SSHPASS": password}
    ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

    experiments = []
    status = None
    log_tail = None

    # Single SSH call to fetch active run artifacts.
    remote_cmd = (
        "RUN=$(test -f /root/results/current_run.txt && tr -d '\\r\\n' < /root/results/current_run.txt); "
        "if [ -z \"$RUN\" ]; then "
        "echo ''; echo '---SEP---'; echo ''; echo '---SEP---'; "
        "echo 'ERROR: missing /root/results/current_run.txt'; echo '---SEP---'; echo ''; exit 0; "
        "fi; "
        "RDIR=/root/results/$RUN; "
        "cat \"$RDIR/experiments.v2.jsonl\" 2>/dev/null; echo '---SEP---'; "
        "cat \"$RDIR/status.json\" 2>/dev/null; echo '---SEP---'; "
        "tail -10 /root/loop.log 2>/dev/null; echo '---SEP---'; "
        "echo \"$RUN\""
    )
    try:
        r = subprocess.run(
            f"sshpass -e ssh {ssh_opts} -p {port} root@{host} '{remote_cmd}'",
            shell=True, capture_output=True, text=True, timeout=15, env=env,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.split("---SEP---")

            # Parse experiments.v2.jsonl
            if len(parts) >= 1 and parts[0].strip():
                for line in parts[0].strip().split("\n"):
                    try:
                        experiments.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

            # Parse status.json
            if len(parts) >= 2 and parts[1].strip():
                try:
                    status = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

            # Parse log tail
            if len(parts) >= 3 and parts[2].strip():
                log_tail = parts[2].strip()
            if len(parts) >= 4 and parts[3].strip():
                if status is None:
                    status = {}
                status["_run_name"] = parts[3].strip()
            if status is None and len(parts) >= 3 and parts[2].strip().startswith("ERROR:"):
                status = {"phase": "error", "error": parts[2].strip()}
    except (subprocess.TimeoutExpired, Exception):
        pass

    return experiments, status, log_tail


def _resolve_local_run_dir(path: str) -> str:
    if os.path.isdir(path):
        pointer = os.path.join(path, "current_run.txt")
        if os.path.isfile(pointer):
            with open(pointer) as f:
                run_name = f.read().strip()
            if not run_name:
                raise RuntimeError(f"Empty run pointer: {pointer}")
            run_dir = os.path.join(path, run_name)
            if not os.path.isdir(run_dir):
                raise RuntimeError(f"Pointer target missing: {run_dir}")
            return run_dir
        if os.path.isfile(os.path.join(path, "experiments.v2.jsonl")):
            return path
    elif os.path.isfile(path):
        if os.path.basename(path) == "current_run.txt":
            with open(path) as f:
                run_name = f.read().strip()
            if not run_name:
                raise RuntimeError(f"Empty run pointer: {path}")
            root = os.path.dirname(path)
            run_dir = os.path.join(root, run_name)
            if not os.path.isdir(run_dir):
                raise RuntimeError(f"Pointer target missing: {run_dir}")
            return run_dir
        if os.path.basename(path) == "experiments.v2.jsonl":
            return os.path.dirname(path)
    raise RuntimeError(
        f"Expected results root with current_run.txt or run folder with experiments.v2.jsonl, got: {path}"
    )


def fetch_local(path: str) -> tuple[list[dict], dict | None, str | None]:
    """Read canonical experiments.v2.jsonl and status.json from local path."""
    experiments = []
    status = None
    log_tail = None

    try:
        run_dir = _resolve_local_run_dir(path)
    except Exception as e:
        return experiments, {"phase": "error", "error": str(e)}, None

    jsonl_path = os.path.join(run_dir, "experiments.v2.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    experiments.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass

    status_path = os.path.join(run_dir, "status.json")
    if os.path.exists(status_path):
        with open(status_path) as f:
            try:
                status = json.load(f)
            except json.JSONDecodeError:
                pass

    if status is None:
        status = {"phase": "error", "error": f"Missing status file: {status_path}"}
    status["_run_name"] = os.path.basename(run_dir)

    log_path = os.path.join(os.path.dirname(run_dir), "loop.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
            log_tail = "".join(lines[-5:]).strip()

    return experiments, status, log_tail


def sparkline(values: list[float], width: int = 40) -> str:
    """Render a sparkline from values."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0

    # Downsample if too many points
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    return "".join(blocks[min(8, int((v - mn) / rng * 8))] for v in sampled)


def build_dashboard(
    experiments: list[dict],
    status: dict | None,
    log_tail: str | None,
    mode: str,
    target: str,
    poll_count: int,
) -> Layout:
    """Build the rich dashboard layout."""
    if Layout is None or Table is None or Panel is None or Text is None or box is None:
        raise RuntimeError("monitor UI requires `rich` (pip install rich)")

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=7),
        Layout(name="legend", size=3),
    )
    layout["body"].split_row(
        Layout(name="chart", ratio=1),
        Layout(name="table", ratio=2),
    )

    # --- Header ---
    now = datetime.now().strftime("%H:%M:%S")
    n_exp = len(experiments)
    n_kept = sum(1 for e in experiments if e.get("kept"))
    best_score = max((e.get("score", -999) for e in experiments), default=-999)

    status_text = ""
    if status:
        phase = status.get("phase", "unknown")
        exp_num = status.get("experiment_id", "?")
        status_text = f"  |  Phase: {phase}  |  Exp: {exp_num}"
        run_name = status.get("_run_name")
        if run_name:
            status_text += f"  |  Run: {run_name}"

    header = Text()
    header.append("AUTORESEARCH MONITOR", style="bold cyan")
    header.append(f"  |  {mode}: {target}  |  {now}  |  poll #{poll_count}", style="dim")
    header.append(f"\n  Experiments: {n_exp}  |  Kept: {n_kept}  |  Best Score: {best_score:.2f}" if n_exp > 0 else "\n  Waiting for experiments...", style="white")
    header.append(status_text, style="dim yellow")
    if status and status.get("error"):
        header.append(f"\n  {status.get('error')}", style="bold red")

    layout["header"].update(Panel(header, box=box.SIMPLE))

    # --- Score Chart ---
    scores = [e.get("score", 0) for e in experiments]
    kept_flags = ["*" if e.get("kept") else " " for e in experiments]

    chart_lines = []
    if scores:
        chart_lines.append(f"Score: {sparkline(scores, width=36)}")
        chart_lines.append(f"  min={min(scores):.1f}  max={max(scores):.1f}  last={scores[-1]:.1f}")
        chart_lines.append("")

        # Trades per day sparkline
        tpd = [e.get("trades_per_day", 0) for e in experiments]
        if tpd:
            chart_lines.append(f"TPD:   {sparkline(tpd, width=36)}")
            chart_lines.append(f"  min={min(tpd):.1f}  max={max(tpd):.1f}  last={tpd[-1]:.1f}")
            chart_lines.append("")

        # Win rate sparkline
        wr = [e.get("win_rate", 0) for e in experiments]
        if wr:
            chart_lines.append(f"WinR:  {sparkline(wr, width=36)}")
            chart_lines.append(f"  min={min(wr):.1%}  max={max(wr):.1%}  last={wr[-1]:.1%}")
            chart_lines.append("")

        # Exit pct sparkline
        ep = [e.get("exit_pct", 0) for e in experiments]
        if any(v > 0 for v in ep):
            chart_lines.append(f"Exit%: {sparkline(ep, width=36)}")
            chart_lines.append(f"  min={min(ep):.1%}  max={max(ep):.1%}  last={ep[-1]:.1%}")
    else:
        chart_lines.append("No data yet...")

    layout["chart"].update(Panel("\n".join(chart_lines), title="Metrics", border_style="cyan"))

    # --- Experiment Table ---
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("Kept", width=4)
    table.add_column("Score", justify="right", width=8)
    table.add_column("PF", justify="right", width=6)
    table.add_column("TPD", justify="right", width=5)
    table.add_column("WR%", justify="right", width=5)
    table.add_column("Exit%", justify="right", width=6)
    table.add_column("DN%", justify="right", width=5)
    table.add_column("Change", width=30, no_wrap=True)

    # Show last 15 experiments
    for e in experiments[-15:]:
        idx = e.get("experiment_id", "?")
        kept = "[green]YES[/]" if e.get("kept") else "[dim]no[/]"
        score = e.get("score", 0)
        score_style = "green" if score > 0 else "red" if score < -5 else "yellow"
        pf = e.get("profit_factor", 0)
        tpd = e.get("trades_per_day", 0)
        wr = e.get("win_rate", 0)
        ep = e.get("exit_pct", 0)
        dn = e.get("do_nothing_pct", 1.0)
        change = e.get("change_summary", "")[:30]

        table.add_row(
            str(idx),
            kept,
            f"[{score_style}]{score:.1f}[/]",
            f"{pf:.1f}",
            f"{tpd:.1f}",
            f"{wr:.0%}",
            f"{ep:.0%}",
            f"{dn:.0%}",
            change,
        )

    layout["table"].update(Panel(table, title="Recent Experiments", border_style="cyan"))

    # --- Footer (log tail) ---
    footer_text = log_tail or "No log output yet..."
    layout["footer"].update(Panel(footer_text, title="Log (last 5 lines)", border_style="dim"))

    # --- Legend ---
    legend = (
        "[dim]PF[/]=Profit Factor (gross wins/gross losses)  "
        "[dim]TPD[/]=Trades Per Day  "
        "[dim]WR%[/]=Win Rate  "
        "[dim]Exit%[/]=Model-driven exits  "
        "[dim]DN%[/]=Do Nothing %  "
        "[dim]Score[/]=PF x Trade Sharpe x min(1, TPD/2)"
    )
    layout["legend"].update(Panel(legend, box=box.SIMPLE, style="dim"))

    return layout


def main():
    parser = argparse.ArgumentParser(description="Autoresearch training monitor")
    parser.add_argument("--host", type=str, help="Remote H100 hostname")
    parser.add_argument("--port", type=int, default=31116, help="Remote SSH port")
    parser.add_argument("--password", type=str, default="autoresearch2026", help="SSH password")
    parser.add_argument("--local", type=str, help="Path to local results root, run folder, or current_run.txt")
    parser.add_argument("--interval", type=int, default=15, help="Poll interval in seconds")
    parser.add_argument("--save-local", type=str, help="Directory to save periodic copies of results")
    args = parser.parse_args()

    if not args.host and not args.local:
        print("Usage: python3 monitor.py --host <H100_HOST> --port <SSH_PORT>")
        print("   or: python3 monitor.py --local /path/to/results")
        sys.exit(1)
    if Console is None or Live is None:
        print("ERROR: monitor UI requires `rich` (pip install rich)")
        sys.exit(1)

    mode = "LOCAL" if args.local else "SSH"
    target = args.local or f"{args.host}:{args.port}"

    console = Console()
    console.clear()

    poll_count = 0
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            poll_count += 1
            try:
                if args.local:
                    experiments, status, log_tail = fetch_local(args.local)
                else:
                    experiments, status, log_tail = fetch_remote(args.host, args.port, args.password)

                # Periodic local save (every 5 polls)
                if args.save_local and experiments and poll_count % 5 == 0:
                    os.makedirs(args.save_local, exist_ok=True)
                    save_path = os.path.join(args.save_local, "experiments.v2.jsonl")
                    with open(save_path, 'w') as f:
                        for exp in experiments:
                            f.write(json.dumps(exp) + '\n')
                    if status:
                        with open(os.path.join(args.save_local, "status.json"), 'w') as f:
                            json.dump(status, f, indent=2)
                    if log_tail:
                        with open(os.path.join(args.save_local, "loop_tail.log"), 'w') as f:
                            f.write(log_tail)

                dashboard = build_dashboard(experiments, status, log_tail, mode, target, poll_count)
                live.update(dashboard)
            except KeyboardInterrupt:
                break
            except Exception as e:
                live.update(Panel(f"Error: {e}", border_style="red"))

            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break

    console.print("\n[dim]Monitor stopped.[/]")


if __name__ == "__main__":
    main()
