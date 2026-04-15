"""Trading window audit: bars 60-105 vs full day.

Answers three questions:
  Q1: Does oracle edge truly concentrate in bars 60-105?
  Q2: How much learning/opportunity budget is excluded?
  Q3: Did narrowing the window improve decision quality, or mostly reduce exposure?

Usage:
    python -m v2.analysis.window_audit
"""
from __future__ import annotations

import os
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from v2.core.policy import DecisionPolicy
from v2.core.schema import TradeIntent
from v2.core.simulator import simulate_trade

# ── constants ─────────────��─────────────────────────────────────────
POLICY = DecisionPolicy()
BARS_PER_DAY = 390
AUDIT_START = 30    # earliest bar to audit
AUDIT_END = 269     # latest bar to audit (exclusive)
BUCKET_SIZE = 15    # bars per bucket

CANDIDATE_WINDOWS = [
    ("60-105 (current)", 60, 105),
    ("30-105", 30, 105),
    ("45-120", 45, 120),
    ("60-120", 60, 120),
    ("30-150", 30, 150),
    ("30-269", 30, 269),
]

OUT_DIR = Path("v2/artifacts/window_audit")
SIDECAR_DIR = Path("v2/data_sidecars")


# ── helpers ─────────────────────────────────────────────────────────

def _bucket_for_bar(bar: int) -> int:
    return bar // BUCKET_SIZE


def _bar_range_for_bucket(bucket: int) -> str:
    lo = bucket * BUCKET_SIZE
    hi = lo + BUCKET_SIZE - 1
    return f"{lo}-{hi}"


def _compute_metrics(pnls: list[float]) -> dict:
    """Compute standard metrics from a list of per-trade net PnL percentages."""
    if not pnls:
        return {
            "n_trades": 0, "pf": 0.0, "win_rate": 0.0,
            "avg_pnl": 0.0, "median_pnl": 0.0, "sharpe": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0,
        }
    arr = np.array(pnls)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    wr = len(wins) / len(arr) if len(arr) else 0.0
    avg = float(arr.mean())
    med = float(np.median(arr))
    std = float(arr.std()) if len(arr) > 1 else 0.0
    sharpe = avg / std if std > 0 else 0.0
    return {
        "n_trades": len(arr),
        "pf": round(pf, 4),
        "win_rate": round(wr, 4),
        "avg_pnl": round(avg, 6),
        "median_pnl": round(med, 6),
        "sharpe": round(sharpe, 4),
        "gross_profit": round(gp, 6),
        "gross_loss": round(gl, 6),
    }


def _max_drawdown_pct(pnls: list[float], starting: float = 10_000.0) -> float:
    """Peak-to-trough drawdown treating each PnL as a sequential trade.

    Uses fixed $500 notional per trade ($5 mid * 100 multiplier).
    This is approximate — the real model trades one-at-a-time with
    cooldown, but this gives a directional sense of equity risk.
    """
    if not pnls:
        return 0.0
    notional = 500.0
    equity = starting
    peak = starting
    max_dd = 0.0
    for p in pnls:
        equity += p * notional
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 6)


# ── data loading ────────────────────────────────────────────────────

def load_manifest() -> dict:
    """Load data.pt and build day-to-global-index mapping."""
    print("Loading data.pt ...", flush=True)
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    X_sim = data["X_sim"].numpy()
    spot = data["spot_prices"].numpy()

    # Build day -> list[global_idx] mapping
    day_indices: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        day_indices[d].append(i)

    print(f"  {len(dates)} total bars, {len(day_indices)} days", flush=True)
    return {
        "dates": dates,
        "bar_of_day": bar_of_day,
        "X_sim": X_sim,
        "spot": spot,
        "day_indices": dict(day_indices),
    }


def load_sidecar(date: str) -> dict | None:
    path = SIDECAR_DIR / f"{date}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


# ── Analysis 1: Oracle by bar ──────────────────────────────────────

def _is_executable(sc, contract_idx: int, bar: int) -> bool:
    """Check if a contract passes executability filters at a given bar."""
    mid = float(sc["contract_mid"][contract_idx, bar])
    if not np.isfinite(mid) or mid < POLICY.min_contract_mid:
        return False
    ask = float(sc["contract_ask"][contract_idx, bar])
    bid = float(sc["contract_bid"][contract_idx, bar])
    if mid > 0:
        spread_frac = (ask - bid) / mid
        if spread_frac > POLICY.max_spread_fraction:
            return False
    quality = int(sc["contract_quality"][contract_idx, bar])
    if POLICY.require_volume_or_transactions and quality < 2:
        return False
    return True


def _simulate_oracle_bar(
    sc: dict,
    bar: int,
    feature_context: np.ndarray,
    n_bars: int,
    day: str,
    expiry: str,
) -> tuple[float | None, int, int]:
    """Simulate oracle (best contract) at a specific bar.

    Returns (best_pnl_or_None, n_executable, n_labelable).
    """
    n_contracts = sc["contract_mid"].shape[0]
    n_executable = 0
    n_labelable = 0
    best_pnl = -float("inf")
    timestamps = sc["bar_timestamps"]

    for ci in range(n_contracts):
        if not _is_executable(sc, ci, bar):
            continue
        n_executable += 1

        series = np.asarray(sc["contract_mid"][ci], dtype=np.float32)
        mid_now = float(series[bar])

        # Check forward path completeness
        fill_bar = bar + 1
        if fill_bar >= n_bars:
            continue
        max_hold = min(POLICY.max_hold_bars, n_bars - bar - 1)
        if max_hold < 2:
            continue
        end = min(n_bars, fill_bar + max_hold + 1)
        if not np.isfinite(series[fill_bar:end]).all():
            continue

        right = "P" if int(sc["contract_right"][ci]) == 1 else "C"
        strike = float(sc["contract_strike"][ci])

        intent = TradeIntent(
            trade=True,
            expiry=expiry,
            strike=strike,
            right=right,
            qty=POLICY.qty,
            entry_ref_price=mid_now,
            order_style=POLICY.order_style,
            tif=POLICY.tif,
            stop_price=mid_now * (1.0 - POLICY.stop_pct),
            take_profit_price=mid_now * (1.0 + POLICY.target_pct),
            max_hold_bars=max_hold,
            exit_policy=POLICY.exit_policy,
            confidence=0.5,
            reason_codes=("window_audit_oracle",),
            bar_index=bar,
            timestamp=str(int(timestamps[bar])) if len(timestamps) > bar else "",
            underlying_price=0.0,
        )
        trade = simulate_trade(
            intent=intent,
            option_prices=series,
            features=feature_context,
            bar_of_day=np.arange(n_bars, dtype=np.int32),
            dates=[day] * n_bars,
            global_entry_bar=bar,
            breakeven_trigger_pct=POLICY.breakeven_trigger_pct if POLICY.breakeven_trigger_pct > 0 else None,
            extra_trailing_tiers=POLICY.extra_trailing_tiers,
        )
        if trade is None:
            continue

        n_labelable += 1
        pnl = float(trade.net_pnl_pct)
        if pnl > best_pnl:
            best_pnl = pnl

    if n_labelable == 0:
        return None, n_executable, 0
    return best_pnl, n_executable, n_labelable


def oracle_by_bar(manifest: dict) -> dict[int, list[dict]]:
    """Compute oracle best-contract PnL for every bar across all days.

    Returns: {bar_of_day: [{'pnl': float, 'date': str, 'n_exec': int, 'n_label': int}, ...]}
    """
    print("\n=== Analysis 1: Oracle by bar-of-day ===", flush=True)
    results: dict[int, list[dict]] = defaultdict(list)
    sidecar_files = sorted(SIDECAR_DIR.glob("*.pt"))
    total = len(sidecar_files)

    for idx, sc_path in enumerate(sidecar_files):
        date = sc_path.stem
        sc = torch.load(sc_path, map_location="cpu", weights_only=False)
        n_bars = int(sc["n_bars"])
        expiry = sc["expiry"]

        # Get feature context for this day from data.pt
        if date not in manifest["day_indices"]:
            continue
        gidx = manifest["day_indices"][date]
        feature_context = manifest["X_sim"][gidx[0]:gidx[0] + n_bars]
        if feature_context.shape[0] != n_bars:
            continue

        # For bars in the existing window (60-105), use pre-computed labels
        # if they exist — they were computed with the same simulation logic
        for bar in range(AUDIT_START, min(AUDIT_END, n_bars)):
            if 60 <= bar < 105 and sc["bar_labelable"][bar]:
                # Use pre-computed oracle result
                pnl = float(sc["bar_best_pnl"][bar])
                # Count executable contracts from bar_ptrs
                start_row = int(sc["bar_ptrs"][bar])
                end_row = int(sc["bar_ptrs"][bar + 1])
                n_exec = end_row - start_row
                n_label = n_exec  # all rows passed executability
                results[bar].append({
                    "pnl": pnl, "date": date,
                    "n_exec": n_exec, "n_label": n_label,
                    "trade_worthy": pnl > POLICY.label_gate_min_pnl,
                })
            elif 60 <= bar < 105 and not sc["bar_labelable"][bar]:
                # In-window but not labelable
                start_row = int(sc["bar_ptrs"][bar])
                end_row = int(sc["bar_ptrs"][bar + 1])
                results[bar].append({
                    "pnl": None, "date": date,
                    "n_exec": end_row - start_row, "n_label": 0,
                    "trade_worthy": False,
                })
            else:
                # Out-of-window: simulate from raw chain data
                pnl, n_exec, n_label = _simulate_oracle_bar(
                    sc, bar, feature_context, n_bars, date, expiry,
                )
                results[bar].append({
                    "pnl": pnl, "date": date,
                    "n_exec": n_exec, "n_label": n_label,
                    "trade_worthy": pnl is not None and pnl > POLICY.label_gate_min_pnl,
                })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total} days", flush=True)

    print(f"  Done: {total} days processed", flush=True)
    return dict(results)


# ─�� Analysis 2: Baseline (ATM-always) by bar ───────────────────────

def baseline_by_bar(manifest: dict) -> dict[int, list[dict]]:
    """Compute ATM-always baseline PnL for every bar across all days."""
    print("\n=== Analysis 2: Baseline (ATM-always) by bar-of-day ===", flush=True)
    results: dict[int, list[dict]] = defaultdict(list)
    sidecar_files = sorted(SIDECAR_DIR.glob("*.pt"))
    total = len(sidecar_files)

    for idx, sc_path in enumerate(sidecar_files):
        date = sc_path.stem
        sc = torch.load(sc_path, map_location="cpu", weights_only=False)
        n_bars = int(sc["n_bars"])
        expiry = sc["expiry"]
        strikes = np.asarray(sc["contract_strike"], dtype=float)
        rights = np.asarray(sc["contract_right"], dtype=int)

        if date not in manifest["day_indices"]:
            continue
        gidx = manifest["day_indices"][date]
        feature_context = manifest["X_sim"][gidx[0]:gidx[0] + n_bars]
        spot_day = manifest["spot"][gidx[0]:gidx[0] + n_bars]
        if feature_context.shape[0] != n_bars:
            continue

        timestamps = sc["bar_timestamps"]
        n_contracts = sc["contract_mid"].shape[0]

        for bar in range(AUDIT_START, min(AUDIT_END, n_bars)):
            spot = float(spot_day[bar])
            if not np.isfinite(spot) or spot <= 0:
                results[bar].append({"pnl": None, "date": date})
                continue

            # Find nearest ATM call that passes executability
            best_ci = -1
            best_dist = float("inf")
            for ci in range(n_contracts):
                if int(rights[ci]) != 0:  # calls only for ATM baseline
                    continue
                if not _is_executable(sc, ci, bar):
                    continue
                dist = abs(strikes[ci] - spot)
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci

            if best_ci < 0:
                results[bar].append({"pnl": None, "date": date})
                continue

            series = np.asarray(sc["contract_mid"][best_ci], dtype=np.float32)
            mid_now = float(series[bar])
            max_hold = min(POLICY.max_hold_bars, n_bars - bar - 1)
            if max_hold < 2:
                results[bar].append({"pnl": None, "date": date})
                continue
            fill_bar = bar + 1
            if fill_bar >= n_bars:
                results[bar].append({"pnl": None, "date": date})
                continue
            end = min(n_bars, fill_bar + max_hold + 1)
            if not np.isfinite(series[fill_bar:end]).all():
                results[bar].append({"pnl": None, "date": date})
                continue

            intent = TradeIntent(
                trade=True,
                expiry=expiry,
                strike=float(strikes[best_ci]),
                right="C",
                qty=POLICY.qty,
                entry_ref_price=mid_now,
                order_style=POLICY.order_style,
                tif=POLICY.tif,
                stop_price=mid_now * (1.0 - POLICY.stop_pct),
                take_profit_price=mid_now * (1.0 + POLICY.target_pct),
                max_hold_bars=max_hold,
                exit_policy=POLICY.exit_policy,
                confidence=0.5,
                reason_codes=("window_audit_baseline",),
                bar_index=bar,
                timestamp=str(int(timestamps[bar])) if len(timestamps) > bar else "",
                underlying_price=spot,
            )
            trade = simulate_trade(
                intent=intent,
                option_prices=series,
                features=feature_context,
                bar_of_day=np.arange(n_bars, dtype=np.int32),
                dates=[date] * n_bars,
                global_entry_bar=bar,
                breakeven_trigger_pct=POLICY.breakeven_trigger_pct if POLICY.breakeven_trigger_pct > 0 else None,
                extra_trailing_tiers=POLICY.extra_trailing_tiers,
            )
            pnl = float(trade.net_pnl_pct) if trade is not None else None
            results[bar].append({"pnl": pnl, "date": date})

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total} days", flush=True)

    print(f"  Done: {total} days processed", flush=True)
    return dict(results)


# ── Analysis 3: Model by bar (from existing traces) ────────────────

def model_by_bar() -> dict[int, list[dict]]:
    """Extract model per-bar metrics from replay_traces.csv."""
    print("\n=== Analysis 3: Model by bar-of-day (from traces) ===", flush=True)
    trace_path = Path("v2/artifacts/replay_traces.csv")
    if not trace_path.exists():
        print("  WARNING: replay_traces.csv not found, skipping", flush=True)
        return {}

    results: dict[int, list[dict]] = defaultdict(list)
    with open(trace_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bar = int(row["bar_of_day"])
            decision = row.get("decision", "")
            model_pnl = row.get("model_pnl", "")
            oracle_pnl = row.get("oracle_pnl", "")
            results[bar].append({
                "date": row.get("date", ""),
                "decision": decision,
                "model_pnl": float(model_pnl) if model_pnl and model_pnl != "" else None,
                "oracle_pnl": float(oracle_pnl) if oracle_pnl and oracle_pnl != "" else None,
            })

    n_bars = len(results)
    n_rows = sum(len(v) for v in results.values())
    print(f"  Loaded {n_rows} trace rows across {n_bars} bars", flush=True)
    return dict(results)


# ��─ Analysis 4: Opportunity budget ──────────────────────────────────

def opportunity_table(oracle_data: dict[int, list[dict]]) -> list[dict]:
    """Compute opportunity budget for each candidate window."""
    print("\n=== Analysis 4: Opportunity budget ===", flush=True)
    rows = []
    # Collect all dates from any bar
    all_dates = set()
    for bar_entries in oracle_data.values():
        for e in bar_entries:
            all_dates.add(e["date"])
    total_days = len(all_dates)

    for name, start, end in CANDIDATE_WINDOWS:
        n_bars_window = end - start
        pct_excluded = round((1.0 - n_bars_window / BARS_PER_DAY) * 100, 1)

        # Aggregate across bars in window
        all_pnls = []
        total_exec = 0
        total_label = 0
        traded_dates = set()
        trade_worthy_dates = set()

        for bar in range(start, end):
            if bar not in oracle_data:
                continue
            for entry in oracle_data[bar]:
                total_exec += entry.get("n_exec", 0)
                total_label += entry.get("n_label", 0)
                if entry["pnl"] is not None:
                    all_pnls.append(entry["pnl"])
                    traded_dates.add(entry["date"])
                if entry.get("trade_worthy", False):
                    trade_worthy_dates.add(entry["date"])

        n_traded_days = len(traded_dates)
        n_trade_worthy_days = len(trade_worthy_dates)
        trades_per_day = len(all_pnls) / n_traded_days if n_traded_days > 0 else 0

        # Compute mean labelable contracts per day
        label_per_day = total_label / total_days if total_days > 0 else 0

        metrics = _compute_metrics(all_pnls)
        dd = _max_drawdown_pct(all_pnls)

        row = {
            "window": name,
            "bars_per_day": n_bars_window,
            "pct_day_excluded": pct_excluded,
            "labelable_contracts_per_day": round(label_per_day, 1),
            "total_oracle_trades": len(all_pnls),
            "traded_days": n_traded_days,
            "trade_worthy_days": n_trade_worthy_days,
            "trades_per_day": round(trades_per_day, 2),
            **metrics,
            "max_drawdown": dd,
        }
        rows.append(row)
        print(f"  {name:20s}: {len(all_pnls):5d} trades, PF={metrics['pf']:.3f}, "
              f"WR={metrics['win_rate']:.1%}, DD={dd:.1%}, "
              f"TPD={trades_per_day:.1f}, days={n_traded_days}", flush=True)

    return rows


# ── Analysis 5: Mechanical-filtering control ────────────────────────

def _pnls_for_bars(oracle_data: dict[int, list[dict]], bars: set[int]) -> list[float]:
    """Extract all oracle PnLs for the given set of bars."""
    pnls = []
    for bar in bars:
        if bar not in oracle_data:
            continue
        for entry in oracle_data[bar]:
            if entry["pnl"] is not None:
                pnls.append(entry["pnl"])
    return pnls


def narrowing_control(oracle_data: dict[int, list[dict]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Progressive narrowing + widening + random window controls."""
    print("\n=== Analysis 5: Mechanical-filtering controls ===", flush=True)

    # ── 5a: Progressive narrowing (remove worst buckets) ──
    print("  5a: Progressive narrowing (remove worst bucket each step)", flush=True)
    buckets = {}
    for bar in range(AUDIT_START, AUDIT_END):
        b = _bucket_for_bar(bar)
        if b not in buckets:
            buckets[b] = set()
        buckets[b].add(bar)

    # Compute per-bucket PF for ranking
    bucket_pfs = {}
    for b, bars in buckets.items():
        pnls = _pnls_for_bars(oracle_data, bars)
        m = _compute_metrics(pnls)
        bucket_pfs[b] = m["pf"]

    # Sort buckets worst-first
    sorted_buckets = sorted(bucket_pfs.keys(), key=lambda b: bucket_pfs[b])
    active = set(buckets.keys())
    narrowing_rows = []

    # Start with all buckets
    all_bars = set()
    for b in active:
        all_bars |= buckets[b]
    pnls = _pnls_for_bars(oracle_data, all_bars)
    m = _compute_metrics(pnls)
    dd = _max_drawdown_pct(pnls)
    narrowing_rows.append({
        "step": 0, "action": "all buckets",
        "n_buckets": len(active), "n_bars": len(all_bars),
        **m, "max_drawdown": dd,
    })

    for step, remove_b in enumerate(sorted_buckets, 1):
        active.discard(remove_b)
        if not active:
            break
        all_bars = set()
        for b in active:
            all_bars |= buckets[b]
        pnls = _pnls_for_bars(oracle_data, all_bars)
        m = _compute_metrics(pnls)
        dd = _max_drawdown_pct(pnls)
        narrowing_rows.append({
            "step": step,
            "action": f"removed {_bar_range_for_bucket(remove_b)} (PF={bucket_pfs[remove_b]:.3f})",
            "n_buckets": len(active), "n_bars": len(all_bars),
            **m, "max_drawdown": dd,
        })

    # ── 5b: Progressive widening (start at 60-105, add best adjacent) ──
    print("  5b: Progressive widening from 60-105", flush=True)
    current_bars = set(range(60, 105))
    remaining_buckets = set(buckets.keys()) - {_bucket_for_bar(b) for b in range(60, 105)}
    widening_rows = []

    pnls = _pnls_for_bars(oracle_data, current_bars)
    m = _compute_metrics(pnls)
    dd = _max_drawdown_pct(pnls)
    widening_rows.append({
        "step": 0, "action": "60-105 only",
        "n_bars": len(current_bars), **m, "max_drawdown": dd,
    })

    # Sort remaining by PF descending (add best first)
    remaining_sorted = sorted(remaining_buckets, key=lambda b: bucket_pfs.get(b, 0), reverse=True)
    for step, add_b in enumerate(remaining_sorted, 1):
        current_bars |= buckets[add_b]
        pnls = _pnls_for_bars(oracle_data, current_bars)
        m = _compute_metrics(pnls)
        dd = _max_drawdown_pct(pnls)
        widening_rows.append({
            "step": step,
            "action": f"added {_bar_range_for_bucket(add_b)} (PF={bucket_pfs.get(add_b, 0):.3f})",
            "n_bars": len(current_bars), **m, "max_drawdown": dd,
        })

    # ── 5c: Random window sampling ──
    print("  5c: Random window sampling", flush=True)
    random.seed(42)
    random_rows = []
    for width in [45, 60, 75, 120]:
        max_start = AUDIT_END - width
        if max_start < AUDIT_START:
            max_start = AUDIT_START
        samples = []
        for _ in range(20):
            start = random.randint(AUDIT_START, max_start)
            bars = set(range(start, start + width))
            pnls = _pnls_for_bars(oracle_data, bars)
            m = _compute_metrics(pnls)
            dd = _max_drawdown_pct(pnls)
            samples.append({
                "width": width, "start": start, "end": start + width,
                **m, "max_drawdown": dd,
            })
        random_rows.extend(samples)

        # Report summary
        pfs = [s["pf"] for s in samples]
        print(f"    width={width}: PF range [{min(pfs):.3f}, {max(pfs):.3f}], "
              f"median={np.median(pfs):.3f}", flush=True)

    # Also compute 60-105 for comparison
    c_pnls = _pnls_for_bars(oracle_data, set(range(60, 105)))
    c_m = _compute_metrics(c_pnls)
    c_dd = _max_drawdown_pct(c_pnls)
    # Count how many width-45 samples beat 60-105
    w45 = [s for s in random_rows if s["width"] == 45]
    n_beat = sum(1 for s in w45 if s["pf"] > c_m["pf"])
    print(f"    60-105 PF={c_m['pf']:.3f} | {n_beat}/20 random width-45 windows beat it", flush=True)

    return narrowing_rows, widening_rows, random_rows


# ── Analysis 6: Multi-window comparison ─────────────────────────────

def multi_window_comparison(
    oracle_data: dict[int, list[dict]],
    baseline_data: dict[int, list[dict]],
) -> list[dict]:
    """Head-to-head comparison of candidate windows."""
    print("\n=== Analysis 6: Multi-window comparison ===", flush=True)
    rows = []

    for name, start, end in CANDIDATE_WINDOWS:
        window_bars = set(range(start, end))

        # Oracle metrics
        oracle_pnls = _pnls_for_bars(oracle_data, window_bars)
        oracle_m = _compute_metrics(oracle_pnls)
        oracle_dd = _max_drawdown_pct(oracle_pnls)

        # Direction balance from oracle data
        # We don't have per-trade direction in the aggregated data,
        # so report what we have from opportunity table

        # Baseline metrics
        baseline_pnls = []
        for bar in window_bars:
            if bar not in baseline_data:
                continue
            for entry in baseline_data[bar]:
                if entry["pnl"] is not None:
                    baseline_pnls.append(entry["pnl"])
        baseline_m = _compute_metrics(baseline_pnls)
        baseline_dd = _max_drawdown_pct(baseline_pnls)

        # Drawdown contribution: what fraction of full-day DD comes from this window?
        full_pnls = _pnls_for_bars(oracle_data, set(range(AUDIT_START, AUDIT_END)))
        full_dd = _max_drawdown_pct(full_pnls)
        dd_contribution = oracle_dd / full_dd if full_dd > 0 else 0.0

        # Traded days
        traded_dates = set()
        for bar in window_bars:
            if bar not in oracle_data:
                continue
            for entry in oracle_data[bar]:
                if entry["pnl"] is not None:
                    traded_dates.add(entry["date"])

        row = {
            "window": name,
            "bars": end - start,
            "oracle_trades": oracle_m["n_trades"],
            "oracle_pf": oracle_m["pf"],
            "oracle_wr": oracle_m["win_rate"],
            "oracle_avg_pnl": oracle_m["avg_pnl"],
            "oracle_median_pnl": oracle_m["median_pnl"],
            "oracle_sharpe": oracle_m["sharpe"],
            "oracle_dd": oracle_dd,
            "oracle_dd_contribution": round(dd_contribution, 4),
            "baseline_trades": baseline_m["n_trades"],
            "baseline_pf": baseline_m["pf"],
            "baseline_wr": baseline_m["win_rate"],
            "baseline_dd": baseline_dd,
            "traded_days": len(traded_dates),
            "pct_coverage": round((end - start) / BARS_PER_DAY * 100, 1),
        }
        rows.append(row)
        print(f"  {name:20s}: oracle PF={oracle_m['pf']:.3f} DD={oracle_dd:.1%} | "
              f"baseline PF={baseline_m['pf']:.3f} DD={baseline_dd:.1%}", flush=True)

    return rows


# ── Report rendering ────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def render_bucketed_oracle(oracle_data: dict[int, list[dict]]) -> list[dict]:
    """Aggregate oracle data into 15-bar buckets."""
    bucket_rows = []
    for bucket_start in range(0, BARS_PER_DAY, BUCKET_SIZE):
        bucket_end = bucket_start + BUCKET_SIZE
        pnls = []
        total_exec = 0
        total_label = 0
        n_trade_worthy = 0
        n_entries = 0

        for bar in range(bucket_start, min(bucket_end, BARS_PER_DAY)):
            if bar not in oracle_data:
                continue
            for entry in oracle_data[bar]:
                n_entries += 1
                total_exec += entry.get("n_exec", 0)
                total_label += entry.get("n_label", 0)
                if entry["pnl"] is not None:
                    pnls.append(entry["pnl"])
                if entry.get("trade_worthy", False):
                    n_trade_worthy += 1

        m = _compute_metrics(pnls)
        dd = _max_drawdown_pct(pnls)
        in_current_window = "***" if (bucket_start >= 60 and bucket_end <= 105) else (
            "**" if (bucket_start < 105 and bucket_end > 60) else ""
        )

        bucket_rows.append({
            "bucket": f"{bucket_start}-{bucket_end - 1}",
            "in_window": in_current_window,
            "n_bar_days": n_entries,
            "n_labelable": total_label,
            "n_trade_worthy": n_trade_worthy,
            **m,
            "max_drawdown": dd,
        })

    return bucket_rows


def render_report(
    bucket_oracle: list[dict],
    bucket_baseline: list[dict],
    opp_table: list[dict],
    multi_window: list[dict],
    narrowing: list[dict],
    widening: list[dict],
    random_samples: list[dict],
) -> str:
    """Build the text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("TRADING WINDOW AUDIT REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Q1: Oracle by bucket
    lines.append("Q1: DOES ORACLE EDGE CONCENTRATE IN BARS 60-105?")
    lines.append("-" * 60)
    lines.append(f"{'Bucket':>10s} {'Window':>6s} {'Trades':>7s} {'PF':>7s} {'WR':>7s} "
                 f"{'AvgPnL':>9s} {'DD':>7s} {'Sharpe':>7s} {'Labelable':>10s}")
    for r in bucket_oracle:
        if r["n_trades"] == 0:
            continue
        lines.append(
            f"{r['bucket']:>10s} {r['in_window']:>6s} {r['n_trades']:>7d} "
            f"{r['pf']:>7.3f} {r['win_rate']:>7.1%} {r['avg_pnl']:>9.5f} "
            f"{r['max_drawdown']:>7.1%} {r['sharpe']:>7.3f} {r['n_labelable']:>10d}"
        )
    lines.append("")

    # Q2: Opportunity budget
    lines.append("Q2: OPPORTUNITY BUDGET BY WINDOW")
    lines.append("-" * 60)
    lines.append(f"{'Window':>20s} {'Bars':>5s} {'Trades':>7s} {'TPD':>5s} {'Days':>5s} "
                 f"{'PF':>7s} {'WR':>7s} {'DD':>7s} {'Excl%':>6s}")
    for r in opp_table:
        lines.append(
            f"{r['window']:>20s} {r['bars_per_day']:>5d} {r['total_oracle_trades']:>7d} "
            f"{r['trades_per_day']:>5.1f} {r['traded_days']:>5d} "
            f"{r['pf']:>7.3f} {r['win_rate']:>7.1%} {r['max_drawdown']:>7.1%} "
            f"{r['pct_day_excluded']:>5.1f}%"
        )
    lines.append("")

    # Q3: Mechanical filtering
    lines.append("Q3: IS THE IMPROVEMENT MECHANICAL OR STRUCTURAL?")
    lines.append("-" * 60)

    lines.append("")
    lines.append("Progressive narrowing (remove worst bucket each step):")
    lines.append(f"{'Step':>5s} {'Bars':>5s} {'Trades':>7s} {'PF':>7s} {'WR':>7s} "
                 f"{'DD':>7s} {'Action'}")
    for r in narrowing:
        lines.append(
            f"{r['step']:>5d} {r['n_bars']:>5d} {r['n_trades']:>7d} "
            f"{r['pf']:>7.3f} {r['win_rate']:>7.1%} {r['max_drawdown']:>7.1%} "
            f"{r['action']}"
        )

    lines.append("")
    lines.append("Progressive widening from 60-105:")
    lines.append(f"{'Step':>5s} {'Bars':>5s} {'Trades':>7s} {'PF':>7s} {'WR':>7s} "
                 f"{'DD':>7s} {'Action'}")
    for r in widening:
        lines.append(
            f"{r['step']:>5d} {r['n_bars']:>5d} {r['n_trades']:>7d} "
            f"{r['pf']:>7.3f} {r['win_rate']:>7.1%} {r['max_drawdown']:>7.1%} "
            f"{r['action']}"
        )

    lines.append("")
    lines.append("Random window sampling (PF distribution):")
    for width in [45, 60, 75, 120]:
        samples = [s for s in random_samples if s["width"] == width]
        if not samples:
            continue
        pfs = sorted(s["pf"] for s in samples)
        lines.append(
            f"  width={width}: min={pfs[0]:.3f} p25={pfs[len(pfs)//4]:.3f} "
            f"median={pfs[len(pfs)//2]:.3f} p75={pfs[3*len(pfs)//4]:.3f} "
            f"max={pfs[-1]:.3f}"
        )
    # Compare 60-105
    c_pnls = []
    for bar in range(60, 105):
        for entry in oracle_data_global.get(bar, []):
            if entry["pnl"] is not None:
                c_pnls.append(entry["pnl"])
    c_m = _compute_metrics(c_pnls)
    w45_pfs = sorted(s["pf"] for s in random_samples if s["width"] == 45)
    rank = sum(1 for p in w45_pfs if p < c_m["pf"])
    lines.append(f"  60-105 PF={c_m['pf']:.3f} → rank {rank + 1}/20 among random width-45 windows")

    lines.append("")
    lines.append("")

    # Multi-window comparison
    lines.append("MULTI-WINDOW HEAD-TO-HEAD")
    lines.append("-" * 60)
    lines.append(f"{'Window':>20s} {'O.PF':>6s} {'O.WR':>6s} {'O.DD':>6s} "
                 f"{'B.PF':>6s} {'B.WR':>6s} {'B.DD':>6s} {'Days':>5s} {'Cov%':>5s}")
    for r in multi_window:
        lines.append(
            f"{r['window']:>20s} {r['oracle_pf']:>6.3f} {r['oracle_wr']:>6.1%} "
            f"{r['oracle_dd']:>6.1%} {r['baseline_pf']:>6.3f} {r['baseline_wr']:>6.1%} "
            f"{r['baseline_dd']:>6.1%} {r['traded_days']:>5d} {r['pct_coverage']:>5.1f}"
        )

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# ── global reference for report rendering ───────────────────────────
oracle_data_global: dict[int, list[dict]] = {}


# ── main ───────────────���────────────────────────────────────────────

def main():
    global oracle_data_global

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    # Analysis 1: Oracle by bar
    oracle_data = oracle_by_bar(manifest)
    oracle_data_global = oracle_data
    bucket_oracle = render_bucketed_oracle(oracle_data)
    _write_csv(OUT_DIR / "oracle_by_bar.csv", bucket_oracle)

    # Analysis 2: Baseline by bar
    baseline_data = baseline_by_bar(manifest)
    bucket_baseline_rows = []
    for bucket_start in range(0, BARS_PER_DAY, BUCKET_SIZE):
        bucket_end = bucket_start + BUCKET_SIZE
        pnls = []
        for bar in range(bucket_start, min(bucket_end, BARS_PER_DAY)):
            if bar not in baseline_data:
                continue
            for entry in baseline_data[bar]:
                if entry["pnl"] is not None:
                    pnls.append(entry["pnl"])
        m = _compute_metrics(pnls)
        dd = _max_drawdown_pct(pnls)
        in_w = "***" if (bucket_start >= 60 and bucket_end <= 105) else (
            "**" if (bucket_start < 105 and bucket_end > 60) else ""
        )
        bucket_baseline_rows.append({
            "bucket": f"{bucket_start}-{bucket_end - 1}",
            "in_window": in_w,
            **m, "max_drawdown": dd,
        })
    _write_csv(OUT_DIR / "baseline_by_bar.csv", bucket_baseline_rows)

    # Analysis 3: Model by bar
    model_data = model_by_bar()
    if model_data:
        model_rows = []
        for bar in sorted(model_data.keys()):
            trades = [e for e in model_data[bar] if e["decision"] == "trade" and e["model_pnl"] is not None]
            pnls = [e["model_pnl"] for e in trades]
            m = _compute_metrics(pnls)
            model_rows.append({"bar_of_day": bar, **m})
        _write_csv(OUT_DIR / "model_by_bar.csv", model_rows)

    # Analysis 4: Opportunity budget
    opp_table = opportunity_table(oracle_data)
    _write_csv(OUT_DIR / "opportunity_budget.csv", opp_table)

    # Analysis 5: Mechanical-filtering controls
    narrowing, widening, random_samples = narrowing_control(oracle_data)
    _write_csv(OUT_DIR / "narrowing_control.csv", narrowing)
    _write_csv(OUT_DIR / "widening_control.csv", widening)
    _write_csv(OUT_DIR / "random_window_control.csv", random_samples)

    # Analysis 6: Multi-window comparison
    multi = multi_window_comparison(oracle_data, baseline_data)
    _write_csv(OUT_DIR / "multi_window_comparison.csv", multi)

    # Render text report
    report = render_report(
        bucket_oracle, bucket_baseline_rows, opp_table, multi,
        narrowing, widening, random_samples,
    )
    report_path = OUT_DIR / "window_audit_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n{'=' * 60}")
    print(report)
    print(f"\nReport written to {report_path}")
    print(f"CSVs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
