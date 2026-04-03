"""ART² v2 Monitor: human oversight dashboard for the experiment loop.

Usage:
    python v2/ops/monitor.py [--watch]

Shows:
- Session status (experiments, time, limits)
- Last N experiments with scores
- Best model info
- Score trend
- Baseline deltas
- Fingerprint chain
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.ops.inner_loop import SessionState, STATE_FILE, RESULTS_TSV, format_session_status
from v2.core.metrics import score_config_fingerprint


def load_results_tsv() -> list[dict]:
    """Load results.tsv into a list of dicts."""
    if not os.path.exists(RESULTS_TSV):
        return []
    rows = []
    with open(RESULTS_TSV) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            row = {}
            for i, col in enumerate(header):
                row[col] = parts[i] if i < len(parts) else ""
            rows.append(row)
    return rows


def score_sparkline(results: list[dict], width: int = 40) -> str:
    """ASCII sparkline of scores over time."""
    scores = []
    for r in results:
        try:
            scores.append(float(r.get('score', 0)))
        except (ValueError, TypeError):
            scores.append(0.0)

    if not scores:
        return "(no data)"

    mn, mx = min(scores), max(scores)
    if mn == mx:
        return "=" * min(len(scores), width)

    blocks = " _.-=*#"
    line = []
    for s in scores[-width:]:
        idx = int((s - mn) / (mx - mn) * (len(blocks) - 1))
        idx = max(0, min(idx, len(blocks) - 1))
        line.append(blocks[idx])
    return "".join(line)


def render_dashboard():
    """Render the monitor dashboard to stdout."""
    print("\033[2J\033[H", end="")  # clear screen

    print("=" * 60)
    print("  ART² v2 Monitor")
    print("=" * 60)

    # Session state
    state = SessionState.load()
    print(f"\n--- SESSION ---")
    print(format_session_status(state))

    # Results
    results = load_results_tsv()
    print(f"\n--- EXPERIMENTS ({len(results)} total) ---")

    if results:
        # Show last 15
        shown = results[-15:]
        for r in shown:
            exp = r.get('experiment', '?')
            score = r.get('score', '?')
            status = r.get('status', '?')
            desc = r.get('description', '')[:50]

            # Color-code status
            if status == 'keep':
                marker = '+'
            elif status == 'crash':
                marker = 'X'
            else:
                marker = '-'

            print(f"  {marker} {exp:<16} score={score:<12} [{status}] {desc}")

        # Score trend
        print(f"\n--- SCORE TREND ---")
        print(f"  {score_sparkline(results)}")

        # Stats
        keeps = [r for r in results if r.get('status') == 'keep']
        reverts = [r for r in results if r.get('status') in ('revert', 'discard')]
        crashes = [r for r in results if r.get('status') == 'crash']
        print(f"\n  Keeps: {len(keeps)}  Reverts: {len(reverts)}  Crashes: {len(crashes)}")

        if keeps:
            best = max(keeps, key=lambda r: float(r.get('score', -999)))
            print(f"  Best: {best.get('experiment')} (score={best.get('score')})")
    else:
        print("  (no experiments yet)")

    # Fingerprints
    print(f"\n--- FINGERPRINTS ---")
    print(f"  Score config: {score_config_fingerprint()}")

    # Check data.pt metadata
    data_path = "v2/data.pt"
    if os.path.exists(data_path):
        import torch
        meta = torch.load(data_path, map_location="cpu", weights_only=False).get('metadata', {})
        print(f"  Dataset: {meta.get('fingerprint', 'unknown')}")
        print(f"  Labels: {meta.get('label_version', 'unknown')}")
        split = meta.get('split', {})
        if split:
            print(f"  Split: {split.get('train_days', '?')}/"
                  f"{split.get('val_days', '?')}/"
                  f"{split.get('promote_days', '?')}/"
                  f"{split.get('shadow_days', '?')} days")
    else:
        print(f"  Dataset: NOT FOUND ({data_path})")

    # Artifact info
    artifacts_dir = "v2/artifacts"
    if os.path.exists(artifacts_dir):
        artifact_count = len([d for d in os.listdir(artifacts_dir)
                             if os.path.isdir(os.path.join(artifacts_dir, d))])
        print(f"\n--- ARTIFACTS ---")
        print(f"  {artifact_count} saved artifacts in {artifacts_dir}/")
        if state.best_artifact_id:
            print(f"  Current best: {state.best_artifact_id}")

    print(f"\n{'=' * 60}")
    print(f"  Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="ART2 v2 Monitor")
    parser.add_argument("--watch", action="store_true",
                        help="Refresh every 30 seconds")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                render_dashboard()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
