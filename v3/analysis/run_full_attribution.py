"""Full-dataset Stage 1 attribution run.

For every day in the v2 cache: run the logger, apply both oracles, compute
attribution. Stream day-by-day to keep memory bounded. Write the summary
report and the per-session detail to `v3/reference/attribution_full_YYYY-MM-DD.txt`.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset, load_day_sidecar
from v3.logger.run import run_day
from v3.oracles.exit_headroom import apply_exit_headroom_oracle
from v3.oracles.opportunity import apply_opportunity_oracle
from v3.reporter.attribution import AttributionReport, build_report, format_report
from v3.reporter.feasibility import build_report as build_feasibility_report
from v3.reporter.feasibility import format_report as format_feasibility_report
from v3.reporter.selection_quality import build_report as build_selquality_report
from v3.reporter.selection_quality import format_report as format_selquality_report
from v3.teachers.failed_break import FailedBreakTeacher
from v3.teachers.orc import ORCTeacher


def main() -> int:
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    teachers = [ORCTeacher(), FailedBreakTeacher()]
    all_days = sorted(set(ds.dates))
    print(f"Loaded {len(ds.dates):,} bars across {len(all_days)} sessions")
    print(f"Rails: cap=${cfg.premium_cap_abs:.0f} daily=${cfg.daily_loss_cap_abs:.0f}")

    t0 = time.time()
    logs = []
    feasibility_logs = []  # keep same logs for feasibility pass
    for i, day in enumerate(all_days):
        log = run_day(ds, day, teachers, cfg, equity=25_000)
        if not log.bars:
            continue
        sc = load_day_sidecar(ds, day)
        if sc is None:
            continue
        day_start, _ = ds.day_bar_range(day)
        apply_opportunity_oracle(log, sc, day_start, bars_per_day=390)
        apply_exit_headroom_oracle(log, sc, bars_per_day=390)
        logs.append(log)
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(all_days)} sessions ({time.time()-t0:.0f}s)")

    print(f"Finished {len(logs)} sessions in {time.time()-t0:.0f}s")

    feas = build_feasibility_report(logs)
    attr = build_report(logs)
    selq = build_selquality_report(logs)

    stamp = datetime.now().strftime("%Y-%m-%d")
    out_dir = "v3/reference"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"attribution_full_{stamp}.txt")

    with open(out_path, "w") as f:
        f.write(f"# Full-dataset Stage 1 attribution — {stamp}\n\n")
        f.write(f"Sessions: {len(logs)} / {len(all_days)}\n")
        f.write(
            f"Rails: cap=${cfg.premium_cap_abs:.0f} ({cfg.premium_cap_pct_equity*100:.1f}%), "
            f"daily=${cfg.daily_loss_cap_abs:.0f} ({cfg.daily_loss_cap_pct_equity*100:.1f}%), "
            f"max_losers={cfg.max_full_premium_losers_per_day}\n\n"
        )
        f.write("## Feasibility\n\n")
        f.write(format_feasibility_report(feas))
        f.write("\n\n## Attribution\n\n")
        f.write(format_report(attr))
        f.write("\n\n## Selection quality (ALL teacher-entered bars)\n\n")
        f.write(format_selquality_report(selq))
        f.write("\n\n## Per-session JSON (for downstream analysis)\n\n")
        for row in attr.per_session:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {out_path}")
    print()
    print(format_feasibility_report(feas))
    print()
    print(format_report(attr))
    print()
    print("Selection quality (ALL teacher-entered bars):")
    print(format_selquality_report(selq))
    return 0


if __name__ == "__main__":
    sys.exit(main())
