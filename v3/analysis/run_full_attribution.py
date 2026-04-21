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
from v3.reporter import localization as loc
from v3.teachers.failed_break import FailedBreakTeacher
from v3.teachers.orc import ORCTeacher


def _cohorts_from_attribution(attr: AttributionReport) -> dict[str, loc.LocalizationCohort]:
    """Convert per-session attribution rows into localization cohorts. Avoids
    re-parsing the JSON we just wrote."""
    buckets: dict[str, list[tuple[str, int, str]]] = {}
    for row in attr.per_session:
        outcome = row.get("outcome")
        direction = row.get("oracle_direction")
        if outcome not in ("entered_right", "side_error", "abstention", "guardrail_suppression"):
            continue
        if direction not in ("call", "put"):
            continue
        buckets.setdefault(outcome, []).append(
            (str(row["day"]), int(row["oracle_bar"]), direction)
        )
    return {name: loc.LocalizationCohort(name=name, entries=entries) for name, entries in buckets.items()}


def _build_localization_section(ds: V2Dataset, attr: AttributionReport) -> str:
    """Compact stratified-localization addendum for the attribution file.

    Reports B∩C enrichment on the four outcome cohorts with MATCHED-direction
    controls (replacing the tainted combined_confluence.py:190 1.92× number).
    Stratified by the 3 default slices for the abstention cohort only.
    """
    cohorts = _cohorts_from_attribution(attr)
    if not cohorts:
        return "(no cohorts available; attribution had no per-session rows)\n"
    omar_cache = loc.build_omar_cache()
    b_and_c = loc.make_feature_b_and_c(omar_cache, omar_threshold=0.5)
    slices = loc.default_slices(omar_cache)

    lines: list[str] = []
    lines.append("Feature: B∩C (sigma direction + OMAR retest), matched-direction controls.")
    lines.append("(Replaces combined_confluence.py:190 — direction-hardcoded controls;")
    lines.append(" any '1.92× / 39.6%' citation should be retracted in favor of the numbers below.)")
    lines.append("")
    for name in ("entered_right", "side_error", "abstention"):
        cohort = cohorts.get(name)
        if cohort is None:
            continue
        m = loc.measure_localization(
            ds, cohort,
            feature_name="B_and_C",
            feature_fn=b_and_c,
            control_strategy="matched",
            control_multiplier=10,
            binary_threshold=0.5,
            binary_direction=">=",
        )
        lines.append(loc.format_measurement(m))
        lines.append("")
    abstention = cohorts.get("abstention")
    if abstention is not None:
        lines.append("--- Abstention stratified by default slices ---")
        for slice_name, slice_fn in slices.items():
            sub = loc.filter_cohort_by_slice(abstention, ds, slice_fn, suffix=slice_name)
            if not sub.entries:
                lines.append(f"{sub.name}: n=0")
                continue
            m = loc.measure_localization(
                ds, sub,
                feature_name="B_and_C",
                feature_fn=b_and_c,
                control_strategy="matched",
                control_multiplier=10,
                binary_threshold=0.5,
                binary_direction=">=",
            )
            lines.append(loc.format_measurement(m))
            lines.append("")
    return "\n".join(lines)


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

    print("Building stratified localization addendum...")
    localization_section = _build_localization_section(ds, attr)

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
        f.write("\n\n## Localization (matched-direction controls)\n\n")
        f.write(localization_section)
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
