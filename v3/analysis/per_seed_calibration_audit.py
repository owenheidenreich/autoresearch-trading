"""Per-seed calibration audit for the spx_combined_3seed_001 champion.

For each seed × window in the 5-seed champion stack, this:

1. Pulls validation PF + chosen `decision_margin` from each window's
   `calibration.json` — what calibration was *aiming* for.
2. Pulls OOS PF + trade count + DD from `report.json -> per_window` —
   what actually happened forward.
3. Flags windows where:
   - calibration was out-of-band (`in_band: false`) and a fallback
     margin was used.
   - val→OOS PF degradation > X% (calibration overfit candidate).
   - margin spread across the 5 seeds at the same window index is
     wide (calibration noise — 5 different "right answers").
   - OOS PF < 1.0 with non-trivial trade count (calibration shipped
     a known-bad threshold).

This is purely a diagnostic — no models are retrained, no thresholds
are changed. The output identifies whether per-seed seed PF variance
(seed 42=2.20, seed 46=1.65 on agg) is concentrated in specific
"calibration-broken" windows, and whether intervention should be
calibration-side (different objective, different fallback) or
training-side (different sb, different oracle).

Usage:

    .venv/bin/python -m v3.analysis.per_seed_calibration_audit \
        --champion-dir v3/artifacts \
        --champion-name spx_combined_3seed_001 \
        --seeds 42 43 44 45 46 \
        --out v3/artifacts/calibration_audit/spx_combined_3seed_001.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import numpy as np


def _load_window_calibration(champion_dir: str, name: str, seed: int, w: int) -> dict[str, Any]:
    path = os.path.join(
        champion_dir,
        f"layer2_unified_policy_{name}_seed{seed}",
        f"seed_{seed}",
        f"window_{w:02d}",
        "calibration.json",
    )
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_report(champion_dir: str, name: str, seed: int) -> dict[str, Any]:
    path = os.path.join(
        champion_dir,
        f"layer2_unified_policy_{name}_seed{seed}",
        f"seed_{seed}",
        "report.json",
    )
    with open(path) as f:
        return json.load(f)


def _safe_float(x: Any) -> float | None:
    try:
        f = float(x)
        if math.isfinite(f):
            return f
        # treat +inf as a "no losses" PF — return None to flag
        return None
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-dir", default="v3/artifacts")
    parser.add_argument("--champion-name", default="spx_combined_3seed_001")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--out",
        default="v3/artifacts/calibration_audit/spx_combined_3seed_001.json",
    )
    parser.add_argument(
        "--val-oos-degrade-threshold",
        type=float,
        default=0.30,
        help="Fraction val→OOS PF degradation that counts as a flag",
    )
    parser.add_argument(
        "--margin-spread-flag",
        type=float,
        default=0.20,
        help="If max - min decision_margin across seeds > this, flag",
    )
    args = parser.parse_args()

    n_windows = 13
    seeds = args.seeds

    # Per (seed, window) matrix
    records: list[dict[str, Any]] = []
    for seed in seeds:
        report = _load_report(args.champion_dir, args.champion_name, seed)
        per_w = {w["window_idx"]: w for w in report.get("per_window", [])}
        for w in range(n_windows):
            cal = _load_window_calibration(args.champion_dir, args.champion_name, seed, w)
            oos = per_w.get(w)
            rec = {
                "seed": seed,
                "window": w,
                "cal_in_band": cal.get("in_band"),
                "cal_decision_margin": cal.get("decision_margin"),
                "cal_pf": _safe_float(cal.get("objective_pf")),
                "cal_trades": cal.get("trades"),
                "cal_pf_qualified": cal.get("pf_qualified"),
                "cal_min_win_prob": cal.get("min_win_prob"),
                "cal_max_stopout_prob": cal.get("max_stopout_prob"),
                "oos_pf_raw": (oos.get("pf") if oos else None),
                "oos_pf": _safe_float(oos.get("pf")) if oos else None,
                "oos_trades": (oos.get("trades") if oos else None),
                "oos_dd_pct": (oos.get("max_dd_pct") if oos else None),
                "oos_decision_margin": (oos.get("decision_margin") if oos else None),
                "oos_beats_v1": (oos.get("beats_v1_same_bar_rate") if oos else None),
            }
            # Compute val-OOS gap if both finite
            if rec["cal_pf"] is not None and rec["oos_pf"] is not None:
                rec["val_to_oos_pf_delta"] = rec["oos_pf"] - rec["cal_pf"]
                rec["val_to_oos_pf_pct"] = (
                    (rec["oos_pf"] - rec["cal_pf"]) / max(rec["cal_pf"], 1e-9)
                )
            else:
                rec["val_to_oos_pf_delta"] = None
                rec["val_to_oos_pf_pct"] = None
            records.append(rec)

    # Per-window cross-seed margin spread
    by_window: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        by_window.setdefault(r["window"], []).append(r)
    window_summary: list[dict[str, Any]] = []
    for w, rows in sorted(by_window.items()):
        margins = [r["cal_decision_margin"] for r in rows if r["cal_decision_margin"] is not None]
        oos_pfs = [r["oos_pf"] for r in rows if r["oos_pf"] is not None]
        oos_trades = [r["oos_trades"] for r in rows if r["oos_trades"] is not None]
        n_in_band = sum(1 for r in rows if r["cal_in_band"])
        margin_spread = (max(margins) - min(margins)) if margins else 0.0
        window_summary.append({
            "window": w,
            "n_in_band": n_in_band,
            "n_seeds": len(rows),
            "margin_min": min(margins) if margins else None,
            "margin_max": max(margins) if margins else None,
            "margin_spread": margin_spread,
            "oos_pf_min": min(oos_pfs) if oos_pfs else None,
            "oos_pf_max": max(oos_pfs) if oos_pfs else None,
            "oos_pf_median": float(np.median(oos_pfs)) if oos_pfs else None,
            "oos_trades_total": sum(oos_trades) if oos_trades else 0,
            "oos_trades_min": min(oos_trades) if oos_trades else 0,
            "flagged_wide_margin_spread": margin_spread > args.margin_spread_flag,
            "flagged_some_seed_pf_below_1": (
                any(r["oos_pf"] is not None and r["oos_pf"] < 1.0 for r in rows)
                and any(r["oos_trades"] is not None and r["oos_trades"] >= 5 for r in rows)
            ),
        })

    # Per-seed window-loss attribution: how much weighted PF lift would we
    # gain if each seed dropped its single worst window?
    per_seed_drop_worst: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        seed_rows = [r for r in records if r["seed"] == seed and r["oos_pf"] is not None]
        # rank by (oos_pf, oos_trades) — worst first
        ranked = sorted(
            [r for r in seed_rows if r["oos_trades"] and r["oos_trades"] >= 5],
            key=lambda r: (r["oos_pf"], -r["oos_trades"]),
        )
        if not ranked:
            per_seed_drop_worst[seed] = {}
            continue
        worst = ranked[0]
        # Recompute aggregate PF excluding worst window
        # We don't have raw pnls here, so report median PF gain heuristic
        kept = [r for r in seed_rows if r is not worst]
        kept_pfs = [r["oos_pf"] for r in kept if r["oos_pf"] is not None]
        kept_trades = [r["oos_trades"] for r in kept]
        per_seed_drop_worst[seed] = {
            "worst_window": worst["window"],
            "worst_pf": worst["oos_pf"],
            "worst_trades": worst["oos_trades"],
            "worst_dd_pct": worst["oos_dd_pct"],
            "worst_margin": worst["cal_decision_margin"],
            "kept_pf_median": float(np.median(kept_pfs)) if kept_pfs else None,
            "kept_trades_total": sum(kept_trades),
        }

    # Flagged windows: wide margin spread, or pf<1 cluster
    flagged_windows = [
        w for w in window_summary
        if w["flagged_wide_margin_spread"] or w["flagged_some_seed_pf_below_1"]
    ]

    # Out-of-band fallback windows
    oob_records = [r for r in records if r["cal_in_band"] is False]

    # Top val→OOS degradation cases
    degraded = sorted(
        [r for r in records if r["val_to_oos_pf_pct"] is not None and r["oos_trades"] and r["oos_trades"] >= 5],
        key=lambda r: r["val_to_oos_pf_pct"],
    )[:10]

    out = {
        "champion_name": args.champion_name,
        "seeds": seeds,
        "n_windows": n_windows,
        "n_total_records": len(records),
        "records": records,
        "window_summary": window_summary,
        "flagged_windows": flagged_windows,
        "out_of_band_calibrations": oob_records,
        "top10_val_to_oos_degradation": degraded,
        "per_seed_drop_worst_window": per_seed_drop_worst,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {args.out}")

    # Console summary
    print("\n=== Per-window cross-seed summary ===")
    print(f"{'W':>3} {'in_band':>8} {'mar_lo':>8} {'mar_hi':>8} {'spread':>8} {'pf_lo':>8} {'pf_hi':>8} {'pf_med':>8} {'tr_tot':>8}")
    for w in window_summary:
        print(
            f"{w['window']:>3} {w['n_in_band']:>3}/{w['n_seeds']:<4} "
            f"{w['margin_min'] or 0:>8.3f} {w['margin_max'] or 0:>8.3f} "
            f"{w['margin_spread']:>8.3f} "
            f"{w['oos_pf_min'] if w['oos_pf_min'] is not None else 0:>8.3f} "
            f"{w['oos_pf_max'] if w['oos_pf_max'] is not None else 0:>8.3f} "
            f"{w['oos_pf_median'] if w['oos_pf_median'] is not None else 0:>8.3f} "
            f"{w['oos_trades_total']:>8}"
        )

    print(f"\n=== Out-of-band calibrations ({len(oob_records)}) ===")
    for r in oob_records:
        print(
            f"  seed {r['seed']} W{r['window']:02d}: cal_pf={r['cal_pf']}, "
            f"oos_pf={r['oos_pf']}, oos_trades={r['oos_trades']}, "
            f"margin={r['cal_decision_margin']}"
        )

    print(f"\n=== Flagged windows ({len(flagged_windows)}) ===")
    for w in flagged_windows:
        flags = []
        if w["flagged_wide_margin_spread"]:
            flags.append(f"margin-spread={w['margin_spread']:.3f}")
        if w["flagged_some_seed_pf_below_1"]:
            flags.append(f"pf<1 in some seed (min {w['oos_pf_min']:.3f})")
        print(f"  W{w['window']:02d}: {', '.join(flags)}")

    print("\n=== Per-seed: drop worst window scenario ===")
    print(f"{'seed':>5} {'worst_W':>8} {'worst_pf':>10} {'worst_n':>8} {'worst_dd':>10} {'kept_pf_med':>14}")
    for seed in seeds:
        d = per_seed_drop_worst.get(seed, {})
        if not d:
            continue
        print(
            f"{seed:>5} {d['worst_window']:>8} "
            f"{d['worst_pf']:>10.3f} {d['worst_trades']:>8} "
            f"{d['worst_dd_pct']:>10.2f} {d['kept_pf_median']:>14.3f}"
        )

    print(f"\n=== Top val→OOS PF degradation (worst {len(degraded)}) ===")
    print(f"{'seed':>5} {'W':>3} {'cal_pf':>8} {'oos_pf':>8} {'pct':>8} {'oos_trades':>10} {'margin':>8}")
    for r in degraded:
        print(
            f"{r['seed']:>5} {r['window']:>3} "
            f"{r['cal_pf'] or 0:>8.3f} {r['oos_pf']:>8.3f} "
            f"{r['val_to_oos_pf_pct']:>8.2%} "
            f"{r['oos_trades']:>10} {r['cal_decision_margin']:>8.3f}"
        )


if __name__ == "__main__":
    main()
