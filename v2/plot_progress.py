"""Plot experiment progress -- Karpathy-style autoresearch chart.

Reads v2/results.tsv (post harness-integrity repair CVReport schema) and plots
stability_score over experiments with kept/reverted markers and a running-best
staircase line.

Schema note: columns are scope-named and read directly. No regex parsing.

Usage:
    python v2/plot_progress.py              # all experiments (default)
    python v2/plot_progress.py --from N     # from exp_N onward
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_PATH = Path("v2/results.tsv")
OUTPUT_PATH = Path("v2/output/progress.png")


def parse_results(path: Path, from_exp: str | None = None) -> list[dict]:
    """Parse the CVReport-shaped results.tsv into a list of experiment dicts."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    if from_exp:
        start_idx = 0
        for i, r in enumerate(rows):
            if r["experiment"] == from_exp:
                start_idx = i
                break
        rows = rows[start_idx:]

    experiments = []
    for i, row in enumerate(rows):
        score = _safe_float(row.get("stability_score"), default=0.0)
        desc = row.get("description", "")
        mode = row.get("screening_mode", "")
        pooled_pf = _safe_float(row.get("pooled_pf"), default=0.0)
        pooled_trades = _safe_int(row.get("pooled_trades"), default=0)
        any_gate = row.get("any_gate_failure", "false") == "true"

        label = _extract_label(desc, row["experiment"], mode)

        experiments.append({
            "seq": i + 1,
            "exp_id": row["experiment"],
            "score": score,
            "status": row.get("status", "unknown"),
            "screening_mode": mode,
            "label": label,
            "trades": pooled_trades,
            "pooled_pf": pooled_pf,
            "any_gate_failure": any_gate,
            "description": desc,
        })

    return experiments


def _safe_float(val: str | None, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _safe_int(val: str | None, default: int = 0) -> int:
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except ValueError:
        return default


def _extract_label(desc: str, exp_id: str, mode: str) -> str:
    if desc.startswith("GATE_FAILURE"):
        return "GATE FAIL"
    # Strip known prefixes / common noise; what remains is the hypothesis label
    cleaned = desc.replace(f"mode={mode}", "").replace("legacy:", "").strip()
    # drop the folds=[...] tail
    idx = cleaned.find("folds=[")
    if idx >= 0:
        cleaned = cleaned[:idx].strip()
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
    ax_pf = fig.add_subplot(gs[1, 1])

    kept = [e for e in experiments if e["status"] == "keep"]
    reverted = [e for e in experiments if e["status"] == "revert" and e["score"] >= 0]
    crashed = [e for e in experiments if e["score"] < 0 or e["status"] == "crash"]

    valid_scores = [e["score"] for e in experiments if e["score"] >= 0]
    if valid_scores:
        y_min = min(valid_scores) - 0.5
        y_max = max(valid_scores) + 1.0
    else:
        y_min, y_max = -1, 7

    crash_y = y_min - 0.2

    if reverted:
        ax_main.scatter(
            [e["seq"] for e in reverted],
            [e["score"] for e in reverted],
            c="#95a5a6", alpha=0.4, s=50, zorder=2, label="Discarded",
        )

    if crashed:
        ax_main.scatter(
            [e["seq"] for e in crashed],
            [crash_y] * len(crashed),
            c="#e74c3c", marker="x", s=80, zorder=3, label="Crash/Gate Fail",
        )

    if kept:
        ax_main.scatter(
            [e["seq"] for e in kept],
            [e["score"] for e in kept],
            c="#2ecc71", edgecolors="darkgreen", s=100, zorder=4, label="Kept",
        )

    if kept:
        best_x = [kept[0]["seq"]]
        best_y = [kept[0]["score"]]
        running_best = kept[0]["score"]
        for e in kept[1:]:
            best_x.append(e["seq"])
            best_y.append(running_best)
            if e["score"] > running_best:
                running_best = e["score"]
            best_x.append(e["seq"])
            best_y.append(running_best)
        best_x.append(experiments[-1]["seq"])
        best_y.append(running_best)
        ax_main.plot(best_x, best_y, c="#27ae60", linewidth=2, zorder=1, label="Running best")

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
    ax_main.set_title(
        f"Autoresearch Progress: {n_total} CV Experiments, {n_kept} Kept (stability_score)",
        fontsize=13, fontweight="bold",
    )
    ax_main.set_xlabel("Experiment #")
    ax_main.set_ylabel("Stability score (mean fold)")
    ax_main.set_ylim(crash_y - 0.3, y_max)
    ax_main.legend(loc="upper left", fontsize=8)
    ax_main.grid(True, alpha=0.2)

    _plot_metric_panel(ax_trades, experiments, "trades", "Pooled Trade Count", "#3498db")
    _plot_metric_panel(ax_pf, experiments, "pooled_pf", "Pooled Profit Factor", "#e67e22")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to {output}")


def _plot_metric_panel(ax, experiments: list[dict], key: str, title: str, color: str):
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
