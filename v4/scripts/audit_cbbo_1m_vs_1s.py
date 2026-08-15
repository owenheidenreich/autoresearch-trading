"""Compare purchased CBBO-1m rows with a high-resolution CBBO-1s audit slice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--cbbo-1m-dir", type=Path, default=Path("data/raw/databento/opra_spxw_cbbo_1m"))
    p.add_argument("--cbbo-1s-dir", type=Path, default=Path("data/raw/audit/opra_spxw_cbbo_1s"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("v4/audit/cbbo_1m_vs_1s_audit_summary.json"),
    )
    p.add_argument("--max-p95-mid-diff", type=float, default=0.25)
    p.add_argument("--min-coverage", type=float, default=0.95)
    return p.parse_args()


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).reset_index()
    if "ts_recv" not in frame.columns:
        first = frame.columns[0]
        frame = frame.rename(columns={first: "ts_recv"})
    frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True)
    frame["_minute"] = frame["ts_recv"].dt.floor("min")
    frame["_minute_end"] = frame["ts_recv"].dt.ceil("min")
    frame["bid_px_00"] = pd.to_numeric(frame["bid_px_00"], errors="coerce")
    frame["ask_px_00"] = pd.to_numeric(frame["ask_px_00"], errors="coerce")
    frame = frame[
        frame["symbol"].notna()
        & frame["bid_px_00"].notna()
        & frame["ask_px_00"].notna()
        & (frame["ask_px_00"] >= frame["bid_px_00"])
        & (frame["ask_px_00"] > 0)
    ].copy()
    frame["mid"] = (frame["bid_px_00"] + frame["ask_px_00"]) / 2.0
    frame["spread"] = frame["ask_px_00"] - frame["bid_px_00"]
    return frame


def _collapse_1s(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["symbol", "_minute_end", "ts_recv"])
    last = frame.groupby(["symbol", "_minute_end"], as_index=False).tail(1)
    ranges = frame.groupby(["symbol", "_minute_end"]).agg(
        bid_min=("bid_px_00", "min"),
        bid_max=("bid_px_00", "max"),
        ask_min=("ask_px_00", "min"),
        ask_max=("ask_px_00", "max"),
        mid_min=("mid", "min"),
        mid_max=("mid", "max"),
        updates_1s=("mid", "size"),
    )
    out = last.set_index(["symbol", "_minute_end"])[["bid_px_00", "ask_px_00", "mid", "spread"]]
    out = out.rename(
        columns={
            "bid_px_00": "bid_1s_last",
            "ask_px_00": "ask_1s_last",
            "mid": "mid_1s_last",
            "spread": "spread_1s_last",
        }
    )
    return out.join(ranges).reset_index().rename(columns={"_minute_end": "_minute"})


def _collapse_1m(frame: pd.DataFrame, symbols: set[str]) -> pd.DataFrame:
    frame = frame[frame["symbol"].isin(symbols)].sort_values(["symbol", "_minute", "ts_recv"])
    last = frame.groupby(["symbol", "_minute"], as_index=False).tail(1)
    out = last[["symbol", "_minute", "bid_px_00", "ask_px_00", "mid", "spread"]].copy()
    return out.rename(
        columns={
            "bid_px_00": "bid_1m",
            "ask_px_00": "ask_1m",
            "mid": "mid_1m",
            "spread": "spread_1m",
        }
    )


def _session_summary(session: str, cbbo_1m_dir: Path, cbbo_1s_dir: Path) -> dict:
    one_s = _load(cbbo_1s_dir / f"{session}.cbbo-1s.parquet")
    one_m = _load(cbbo_1m_dir / f"{session}.cbbo-1m.parquet")
    one_s_c = _collapse_1s(one_s)
    one_m_c = _collapse_1m(one_m, set(one_s_c["symbol"]))
    merged = one_m_c.merge(one_s_c, on=["symbol", "_minute"], how="inner")
    one_m_keys = one_m_c[["symbol", "_minute"]].drop_duplicates()
    coverage = len(merged) / len(one_m_keys) if len(one_m_keys) else 0.0
    if merged.empty:
        return {
            "session": session,
            "audit_symbols": int(one_s["symbol"].nunique()),
            "matched_symbol_minutes": 0,
            "coverage": coverage,
            "acceptable": False,
        }

    merged["mid_abs_diff"] = (merged["mid_1m"] - merged["mid_1s_last"]).abs()
    merged["spread_abs_diff"] = (merged["spread_1m"] - merged["spread_1s_last"]).abs()
    merged["intraminute_mid_range"] = merged["mid_max"] - merged["mid_min"]
    merged["intraminute_bid_range"] = merged["bid_max"] - merged["bid_min"]
    result = {
        "session": session,
        "audit_symbols": int(one_s["symbol"].nunique()),
        "cbbo_1s_rows": int(len(one_s)),
        "cbbo_1m_symbol_minutes": int(len(one_m_keys)),
        "matched_symbol_minutes": int(len(merged)),
        "coverage": float(coverage),
        "median_mid_abs_diff": float(merged["mid_abs_diff"].median()),
        "p95_mid_abs_diff": float(merged["mid_abs_diff"].quantile(0.95)),
        "p99_mid_abs_diff": float(merged["mid_abs_diff"].quantile(0.99)),
        "median_spread_abs_diff": float(merged["spread_abs_diff"].median()),
        "p95_spread_abs_diff": float(merged["spread_abs_diff"].quantile(0.95)),
        "p95_intraminute_mid_range": float(merged["intraminute_mid_range"].quantile(0.95)),
        "large_intraminute_mid_range_frac": float((merged["intraminute_mid_range"] > 0.50).mean()),
    }
    return result


def main() -> int:
    args = parse_args()
    sessions = args.sessions
    if sessions is None:
        sessions = sorted(p.name.split(".")[0] for p in args.cbbo_1s_dir.glob("*.cbbo-1s.parquet"))
    days = [_session_summary(session, args.cbbo_1m_dir, args.cbbo_1s_dir) for session in sessions]
    p95_values = [x["p95_mid_abs_diff"] for x in days if "p95_mid_abs_diff" in x]
    coverage_values = [x["coverage"] for x in days]
    aggregate = {
        "sessions": len(days),
        "min_coverage": float(min(coverage_values)) if coverage_values else 0.0,
        "max_p95_mid_abs_diff": float(max(p95_values)) if p95_values else np.inf,
        "mean_p95_mid_abs_diff": float(np.mean(p95_values)) if p95_values else np.inf,
        "acceptable_for_prototype": bool(
            coverage_values
            and min(coverage_values) >= args.min_coverage
            and p95_values
            and max(p95_values) <= args.max_p95_mid_diff
        ),
        "criteria": {
            "min_coverage": args.min_coverage,
            "max_p95_mid_abs_diff": args.max_p95_mid_diff,
        },
        "days": days,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({k: v for k, v in aggregate.items() if k != "days"}, indent=2))
    print(f"WROTE {args.out}")
    return 0 if aggregate["acceptable_for_prototype"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
