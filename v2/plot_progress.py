"""Plot experiment progress -- Karpathy-style autoresearch chart.

Reads v2/results.tsv and plots score over experiments with kept/reverted markers
and a running-best staircase line.

Usage:
    python v2/plot_progress.py              # all experiments (default)
    python v2/plot_progress.py --from N     # from exp_N onward
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_PATH = Path("supervised/results.tsv")
OUTPUT_PATH = Path("output/progress.png")


def parse_results(path: Path, from_exp: str | None = None) -> list[dict]:
    """Parse results.tsv into a list of experiment dicts."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    # Optional: filter from a specific experiment onward
    if from_exp:
        start_idx = 0
        for i, r in enumerate(rows):
            if r["experiment"] == from_exp:
                start_idx = i
                break
        rows = rows[start_idx:]

    experiments = []
    for i, row in enumerate(rows):
        score = float(row["score"])
        desc = row.get("description", "")

        # Parse metrics from description
        trades = _extract_int(r"trades=(\d+)", desc)
        wr = _extract_float(r"wr=([\d.]+)%", desc)

        # Extract label (what changed)
        label = _extract_label(desc, row["experiment"])

        experiments.append({
            "seq": i + 1,
            "exp_id": row["experiment"],
            "score": score,
            "status": row["status"],
            "label": label,
            "trades": trades,
            "wr": wr,
            "description": desc,
        })

    return experiments


def _extract_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _extract_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _extract_label(desc: str, exp_id: str) -> str:
    """Strip metric fields from description to get the 'what changed' label."""
    if desc.startswith("GATE_FAILURE"):
        return "GATE FAIL"
    cleaned = desc
    for pat in [r"WF_BASELINE\s*", r"score=\S+", r"trades=\S+", r"wr=\S+%?",
                r"pdr=\S+%?", r"baselines=\S+", r"PF=\S+", r"NEW\s*(WF\s*)?BEST",
                r"folds=\[[\d.,-]+\]", r"days=\d+", r"std=\S+"]:
        cleaned = re.sub(pat, "", cleaned)
    cleaned = cleaned.strip(" ()")
    return cleaned if cleaned else exp_id


def plot(experiments: list[dict], output: Path):
    """Create the Karpathy-style progress chart."""
    if not experiments:
        print("No experiments to plot.")
        return

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

    ax_main = fig.add_subplot(gs[0, :])
    ax_trades = fig.add_subplot(gs[1, 0])
    ax_wr = fig.add_subplot(gs[1, 1])

    # -- Classify experiments --
    kept = [e for e in experiments if e["status"] == "keep"]
    reverted = [e for e in experiments if e["status"] == "revert" and e["score"] >= 0]
    crashed = [e for e in experiments if e["score"] < 0 or e["status"] == "crash"]

    # Compute y-axis range from non-crash scores
    valid_scores = [e["score"] for e in experiments if e["score"] >= 0]
    if valid_scores:
        y_min = min(valid_scores) - 0.5
        y_max = max(valid_scores) + 1.0
    else:
        y_min, y_max = -1, 7

    # Crash/gate-fail markers go at the bottom of the chart
    crash_y = y_min - 0.2

    # -- Main chart: score vs experiment number --

    # Reverted (gray)
    if reverted:
        ax_main.scatter(
            [e["seq"] for e in reverted],
            [e["score"] for e in reverted],
            c="#95a5a6", alpha=0.4, s=50, zorder=2, label="Discarded",
        )

    # Crashed / gate fail (red X) -- pinned to bottom of chart
    if crashed:
        ax_main.scatter(
            [e["seq"] for e in crashed],
            [crash_y] * len(crashed),
            c="#e74c3c", marker="x", s=80, zorder=3, label="Crash/Gate Fail",
        )

    # Kept (green)
    if kept:
        ax_main.scatter(
            [e["seq"] for e in kept],
            [e["score"] for e in kept],
            c="#2ecc71", edgecolors="darkgreen", s=100, zorder=4, label="Kept",
        )

    # Running best staircase
    if kept:
        best_x = [kept[0]["seq"]]
        best_y = [kept[0]["score"]]
        running_best = kept[0]["score"]
        for e in kept[1:]:
            # Extend horizontal line to this experiment
            best_x.append(e["seq"])
            best_y.append(running_best)
            # Step up if new best
            if e["score"] > running_best:
                running_best = e["score"]
            best_x.append(e["seq"])
            best_y.append(running_best)

        # Extend to end
        best_x.append(experiments[-1]["seq"])
        best_y.append(running_best)

        ax_main.plot(best_x, best_y, c="#27ae60", linewidth=2, zorder=1, label="Running best")

    # Labels on kept experiments
    for e in kept:
        if e["label"]:
            ax_main.annotate(
                e["label"],
                xy=(e["seq"], e["score"]),
                xytext=(0, 12),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                rotation=30,
                color="#27ae60",
                fontstyle="italic",
            )

    n_total = len(experiments)
    n_kept = len(kept)
    ax_main.set_title(f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements",
                       fontsize=13, fontweight="bold")
    ax_main.set_xlabel("Experiment #")
    ax_main.set_ylabel("Score")
    ax_main.set_ylim(crash_y - 0.3, y_max)
    ax_main.legend(loc="upper left", fontsize=8)
    ax_main.grid(True, alpha=0.2)

    # -- Bottom left: Trade count --
    _plot_metric_panel(ax_trades, experiments, "trades", "Trade Count", "#3498db")

    # -- Bottom right: Win rate --
    _plot_metric_panel(ax_wr, experiments, "wr", "Win Rate %", "#e67e22")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to {output}")

    # Try to show interactively
    try:
        import matplotlib
        matplotlib.use("macosx")
        plt.show()
    except Exception:
        pass


def _plot_metric_panel(ax, experiments: list[dict], key: str, title: str, color: str):
    """Small scatter panel for a secondary metric."""
    kept = [(e["seq"], e[key]) for e in experiments if e["status"] == "keep" and e[key] is not None]
    reverted = [(e["seq"], e[key]) for e in experiments if e["status"] != "keep" and e[key] is not None]

    if reverted:
        ax.scatter([x for x, _ in reverted], [y for _, y in reverted],
                   c="#95a5a6", alpha=0.3, s=20)
    if kept:
        ax.scatter([x for x, _ in kept], [y for _, y in kept],
                   c=color, edgecolors="darkgreen", s=40, zorder=3)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Experiment #", fontsize=8)
    ax.grid(True, alpha=0.2)


if __name__ == "__main__":
    from_exp = None
    if "--from" in sys.argv:
        idx = sys.argv.index("--from")
        if idx + 1 < len(sys.argv):
            n = sys.argv[idx + 1]
            from_exp = f"exp_{int(n):03d}"
    exps = parse_results(RESULTS_PATH, from_exp=from_exp)
    plot(exps, OUTPUT_PATH)
