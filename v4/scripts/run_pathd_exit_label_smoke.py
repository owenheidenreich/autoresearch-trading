"""Build and validate the non-circular Path-D 1-second exit label.

The target is the realized advantage of a frozen hold-to-flat policy over
exit-now: ``A_hold(t) = bid(T_flat) - bid(t)``.  Features are causal through
``t``; the future flat bid is a label only.  This remains a quarantined
pipeline-scale smoke test and does not train or assess a model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_RAW = Path("v4/raw/opra_1s_parity_probe")
DEFAULT_OUT = Path("v4/audit/autoresearch/protocol101_pathd_exit_research")
UTC = "UTC"
ENTRY_MINUTES_ET = ((10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0))
BAND_LO, BAND_HI, BAND_TGT = 3.0, 8.0, 5.0
TICK = 0.10
POSITION_KEYS = ["day", "entry_et", "symbol"]
CAUSAL_FEATURE_COLUMNS = [
    "bid", "ask", "mid", "spread", "pnl", "mfe_to_now",
    "giveback_from_peak", "mins_held", "mins_to_close",
]
LABEL_COLUMNS = ["a_hold", "oracle_adv"]
HISTOGRAM_EDGES = np.array(
    [-np.inf, -10, -5, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5, 10, np.inf],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--output-parquet", type=Path, default=DEFAULT_OUT / "pathd_exit_labels_smoke.parquet")
    p.add_argument("--validation-out", type=Path, default=DEFAULT_OUT / "label_smoke_validation.json")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def et_to_utc(day: str, hh: int, mm: int) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hh:02d}:{mm:02d}:00", tz="America/New_York").tz_convert(UTC)


def parse_symbol(sym: str) -> tuple[str, float] | None:
    match = re.match(r"SPXW\s+(\d{6})([CP])(\d{8})", sym)
    if not match:
        return None
    _, right, strike = match.groups()
    return right, int(strike) / 1000.0


def load_day(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).reset_index()
    tcol = "ts_recv" if "ts_recv" in df.columns else "ts_event" if "ts_event" in df.columns else "index"
    df = df.rename(columns={tcol: "ts"})
    cols = ["ts", "symbol", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    df = df[cols].dropna(subset=["ts", "symbol"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.floor("s")
    return df.sort_values("ts").groupby(["symbol", "ts"], as_index=False).last()


def sec_grid(day: str) -> pd.DatetimeIndex:
    return pd.date_range(et_to_utc(day, 9, 30), et_to_utc(day, 16, 0), freq="s", tz=UTC)


def _chosen_entries(df: pd.DataFrame, day: str) -> list[tuple[pd.Timestamp, str, str, float, float]]:
    entry_times = [et_to_utc(day, hh, mm) for hh, mm in ENTRY_MINUTES_ET]
    symbols = df["symbol"].drop_duplicates().astype(str).tolist()
    queries = pd.MultiIndex.from_product(
        [entry_times, symbols], names=["entry_ts", "symbol"]
    ).to_frame(index=False).sort_values("entry_ts")
    quotes = df.rename(columns={"ts": "quote_ts"}).sort_values("quote_ts")
    queries["entry_ts"] = queries["entry_ts"].astype("datetime64[ns, UTC]")
    quotes["quote_ts"] = quotes["quote_ts"].astype("datetime64[ns, UTC]")
    snapshots = pd.merge_asof(
        queries,
        quotes,
        left_on="entry_ts",
        right_on="quote_ts",
        by="symbol",
        direction="backward",
    )
    parsed = snapshots["symbol"].map(parse_symbol)
    snapshots["right"] = parsed.map(lambda x: None if x is None else x[0])
    snapshots["strike"] = parsed.map(lambda x: np.nan if x is None else x[1])
    snapshots["ask"] = pd.to_numeric(snapshots["ask_px_00"], errors="coerce")
    snapshots = snapshots[snapshots["ask"].between(BAND_LO, BAND_HI)]
    chosen: list[tuple[pd.Timestamp, str, str, float, float]] = []
    for entry_ts, group in snapshots.groupby("entry_ts"):
        for right in ("C", "P"):
            pool = group[group["right"] == right]
            if pool.empty:
                continue
            row = pool.iloc[(pool["ask"] - BAND_TGT).abs().argmin()]
            chosen.append((entry_ts, str(row["symbol"]), right, float(row["strike"]), float(row["ask"])))
    return chosen


def build_day(path: Path) -> tuple[str, pd.DataFrame]:
    day = re.search(r"(\d{4}-\d{2}-\d{2})", path.name).group(1)
    df = load_day(path)
    chosen = _chosen_entries(df, day)
    grid = sec_grid(day)
    flat_ts = et_to_utc(day, 15, 55)
    panels: dict[str, pd.DataFrame] = {}
    for symbol in sorted({row[1] for row in chosen}):
        group = df[df["symbol"] == symbol].set_index("ts").sort_index()
        panels[symbol] = group.reindex(grid).ffill()
    chunks: list[pd.DataFrame] = []
    for entry_ts, symbol, right, strike, ask in chosen:
        hold = panels[symbol].loc[(grid > entry_ts) & (grid <= flat_ts)].copy()
        hold["bid_px_00"] = pd.to_numeric(hold["bid_px_00"], errors="coerce")
        hold["ask_px_00"] = pd.to_numeric(hold["ask_px_00"], errors="coerce")
        hold = hold.dropna(subset=["bid_px_00"])
        if hold.empty:
            continue
        bid = hold["bid_px_00"].astype(float)
        ask_path = hold["ask_px_00"].astype(float)
        pnl = bid - (ask + TICK)
        out = pd.DataFrame(index=hold.index)
        out["day"] = day
        out["entry_et"] = entry_ts.tz_convert("America/New_York").strftime("%H:%M")
        out["symbol"] = symbol
        out["right"] = right
        out["strike"] = strike
        out["entry_fill"] = ask + TICK
        out["ts"] = hold.index
        seconds = (hold.index - entry_ts).total_seconds()
        out["sec_held"] = seconds.astype("int64")
        out["bid"] = bid.to_numpy()
        out["ask"] = ask_path.to_numpy()
        out["mid"] = ((bid + ask_path) / 2).fillna(bid).to_numpy()
        out["spread"] = (ask_path - bid).to_numpy()
        out["pnl"] = pnl.to_numpy()
        out["mfe_to_now"] = pnl.cummax().to_numpy()
        out["giveback_from_peak"] = (pnl.cummax() - pnl).to_numpy()
        out["mins_held"] = seconds / 60.0
        out["mins_to_close"] = (flat_ts - hold.index).total_seconds() / 60.0
        out["a_hold"] = float(bid.iloc[-1]) - bid.to_numpy()
        chunks.append(out.reset_index(drop=True))
    return day, pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _corr_state(state: dict[str, float], x: pd.Series, y: pd.Series) -> None:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if valid.empty:
        return
    xv, yv = valid["x"].to_numpy(float), valid["y"].to_numpy(float)
    state["n"] += len(valid); state["sx"] += xv.sum(); state["sy"] += yv.sum()
    state["sxx"] += np.dot(xv, xv); state["syy"] += np.dot(yv, yv); state["sxy"] += np.dot(xv, yv)


def _corr(state: dict[str, float]) -> float | None:
    n = state["n"]
    numerator = n * state["sxy"] - state["sx"] * state["sy"]
    denominator = ((n * state["sxx"] - state["sx"] ** 2) * (n * state["syy"] - state["sy"] ** 2)) ** 0.5
    return None if not denominator else float(numerator / denominator)


def main() -> int:
    args = parse_args()
    files = sorted(args.raw_dir.glob("*.parquet"))
    if args.start_date:
        files = [p for p in files if p.name[:10] >= args.start_date]
    if args.end_date:
        files = [p for p in files if p.name[:10] <= args.end_date]
    if not files:
        raise SystemExit(f"no Parquet inputs found in {args.raw_dir}")
    for path in (args.output_parquet, args.validation_out):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite to replace: {path}")
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.validation_out.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output_parquet.with_suffix(args.output_parquet.suffix + ".partial")
    temp_output.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    totals = dict(n=0, positions=0, pos=0, neg=0, below=0, equal=0, sum=0.0, sumsq=0.0, min=np.inf, max=-np.inf, oracle_sum=0.0)
    histogram = np.zeros(len(HISTOGRAM_EDGES) - 1, dtype=np.int64)
    corr_giveback = {k: 0.0 for k in ("n", "sx", "sy", "sxx", "syy", "sxy")}
    corr_pnl = {k: 0.0 for k in ("n", "sx", "sy", "sxx", "syy", "sxy")}
    per_day: dict[str, int] = {}
    try:
        for path in files:
            day, frame = build_day(path)
            if frame.empty:
                per_day[day] = 0
                print(f"[{day}] label rows=0 positions=0", flush=True)
                continue
            frame = frame.sort_values(POSITION_KEYS + ["ts"]).reset_index(drop=True)
            future_best = frame.groupby(POSITION_KEYS, sort=False)["bid"].transform(
                lambda values: values.iloc[::-1].cummax().iloc[::-1]
            )
            frame["oracle_adv"] = future_best - frame["bid"]
            values = frame["a_hold"].to_numpy(float)
            oracle = frame["oracle_adv"].to_numpy(float)
            totals["n"] += len(frame)
            totals["positions"] += frame[POSITION_KEYS].drop_duplicates().shape[0]
            totals["pos"] += int((values > 0).sum()); totals["neg"] += int((values < 0).sum())
            totals["below"] += int((values <= oracle + 1e-9).sum())
            totals["equal"] += int(np.isclose(values, oracle).sum())
            totals["sum"] += float(values.sum()); totals["sumsq"] += float(np.dot(values, values))
            totals["min"] = min(totals["min"], float(values.min())); totals["max"] = max(totals["max"], float(values.max()))
            totals["oracle_sum"] += float(oracle.sum())
            histogram += np.histogram(values, bins=HISTOGRAM_EDGES)[0]
            _corr_state(corr_giveback, frame["giveback_from_peak"], -frame["a_hold"])
            _corr_state(corr_pnl, frame["pnl"], frame["a_hold"])
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_output, table.schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
            per_day[day] = len(frame)
            print(f"[{day}] label rows={len(frame):>7} positions={frame[POSITION_KEYS].drop_duplicates().shape[0]}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if totals["n"] == 0:
        temp_output.unlink(missing_ok=True)
        raise SystemExit("no label rows were produced")
    temp_output.replace(args.output_parquet)
    n = totals["n"]
    mean = totals["sum"] / n
    variance = max(0.0, (totals["sumsq"] - n * mean * mean) / max(1, n - 1))
    histogram_rows = [
        {"lower": None if np.isneginf(lo) else float(lo), "upper": None if np.isposinf(hi) else float(hi), "count": int(count)}
        for lo, hi, count in zip(HISTOGRAM_EDGES[:-1], HISTOGRAM_EDGES[1:], histogram)
    ]
    validation = {
        "input_dir": str(args.raw_dir), "input_files": len(files), "label_rows": n,
        "positions": totals["positions"], "per_day_rows": per_day,
        "a_hold_stats": {"count": n, "mean": mean, "std": variance ** 0.5, "min": totals["min"], "max": totals["max"]},
        "a_hold_histogram": histogram_rows,
        "a_hold_frac_positive": totals["pos"] / n, "a_hold_frac_negative": totals["neg"] / n,
        "mean_oracle_adv": totals["oracle_sum"] / n, "mean_a_hold": mean,
        "a_hold_is_oracle": totals["equal"] == n,
        "a_hold_below_oracle_frac": totals["below"] / n,
        "corr_giveback_vs_neg_a_hold": _corr(corr_giveback), "corr_pnl_vs_a_hold": _corr(corr_pnl),
        "causal_feature_columns": CAUSAL_FEATURE_COLUMNS, "label_columns": LABEL_COLUMNS,
        "label_feature_overlap": sorted(set(CAUSAL_FEATURE_COLUMNS) & set(LABEL_COLUMNS)),
        "no_leakage_column_check_passed": not (set(CAUSAL_FEATURE_COLUMNS) & set(LABEL_COLUMNS)),
        "labels": "walking_skeleton|throwaway|tier-s|research",
        "note": "Pipeline-scale validation only. A_hold is the fixed hold-to-flat realized label, not the hindsight-best oracle; no model was trained.",
    }
    args.validation_out.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: validation[k] for k in ("label_rows", "positions", "mean_a_hold", "mean_oracle_adv", "a_hold_is_oracle", "a_hold_below_oracle_frac", "no_leakage_column_check_passed")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
