"""Job 2 — independent measurement review: verify the two limits from raw data.

Read-only. Loads the owned ES 1-minute bars, recomputes every number behind
STATUS §4 and the gate-chain audit §3/§10, and measures the assumptions the
brief says to attack: independence across sessions, independence within a
session, and the 1/sqrt(k) trade-frequency scaling. Writes one JSON receipt
next to this script. No model, no vendor, no holdout, no mutation of any data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(
    "/Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31"
    "/raw/databento/glbx_es_ohlcv_1m"
)
OUT = Path(__file__).parent / "analysis_receipt.json"
Z = 1.645 + 0.842  # one-sided 5%, 80% power
COST = 0.358
HORIZONS = (15, 30, 60)


def _et_close_series(path: Path) -> pd.Series:
    frame = pd.read_parquet(path, columns=["close"])
    index = frame.index.tz_convert("America/New_York")
    return pd.Series(frame["close"].to_numpy(), index=index)


def main() -> None:
    sessions: dict[str, pd.Series] = {}
    for path in sorted(DATA.glob("*.parquet")):
        series = _et_close_series(path)
        if len(series):
            sessions[path.name.split(".")[0]] = series

    # --- Limit 1: fixed 09:35 slot moves per session -------------------------
    moves: dict[int, dict[str, float]] = {h: {} for h in HORIZONS}
    for day, series in sessions.items():
        by_clock = {ts.strftime("%H:%M"): value for ts, value in series.items()}
        base = by_clock.get("09:35")
        if base is None:
            continue
        for horizon in HORIZONS:
            target = (
                pd.Timestamp(f"2000-01-01 09:35") + pd.Timedelta(minutes=horizon)
            ).strftime("%H:%M")
            if target in by_clock:
                moves[horizon][day] = float(by_clock[target] - base)

    limit1 = {}
    for horizon in HORIZONS:
        values = np.array(list(moves[horizon].values()))
        sd = float(values.std(ddof=1))
        n = len(values)
        mde_per_session = Z * sd / math.sqrt(n)
        # lag-1 autocorrelation across sessions (independence assumption)
        ordered = pd.Series(
            [moves[horizon][day] for day in sorted(moves[horizon])]
        )
        lag1 = float(ordered.autocorr(lag=1))
        effective_n = n * (1 - lag1) / (1 + lag1) if abs(lag1) < 1 else n
        limit1[f"{horizon}m"] = {
            "sessions": n,
            "session_sd_points": round(sd, 3),
            "mde_net_points_per_session_at_80pct_power": round(mde_per_session, 3),
            "lag1_autocorr_across_sessions": round(lag1, 4),
            "effective_sessions_after_lag1": round(effective_n, 1),
            "mde_with_effective_n": round(Z * sd / math.sqrt(max(effective_n, 1)), 3),
        }

    # --- the trades-per-session table and its 1/sqrt(k) assumption ----------
    sd15 = limit1["15m"]["session_sd_points"]
    n_sessions = limit1["15m"]["sessions"]
    per_trade_table = {
        str(k): round(Z * sd15 / math.sqrt(n_sessions * k), 3) for k in (1, 6, 26)
    }

    # Within-session serial correlation of non-overlapping 15-minute returns:
    # if intraday returns are dependent, k trades are worth fewer than k draws.
    intraday_lag1: list[float] = []
    for series in sessions.values():
        clock = series.groupby(series.index.strftime("%H:%M")).last()
        marks = [
            (pd.Timestamp("2000-01-01 09:35") + pd.Timedelta(minutes=15 * i)).strftime(
                "%H:%M"
            )
            for i in range(26)  # 09:35 through 15:50, 25 non-overlapping returns
        ]
        prices = [clock.get(mark) for mark in marks]
        if any(price is None for price in prices):
            continue
        returns = pd.Series(np.diff(np.array(prices, dtype=float)))
        if returns.std(ddof=1) > 0:
            intraday_lag1.append(float(returns.autocorr(lag=1)))
    mean_intraday_lag1 = float(np.mean(intraday_lag1))

    # --- sessions required for a cost-scale edge, by trades/session ---------
    def sessions_needed(edge: float, trades_per_session: int) -> int:
        return math.ceil((Z * sd15 / edge) ** 2 / trades_per_session)

    cost_scale = {
        f"trades_per_session={k}": {
            "sessions_needed_for_0.358_per_trade": sessions_needed(COST, k),
            "years_at_252": round(sessions_needed(COST, k) / 252, 1),
        }
        for k in (1, 6, 26)
    }

    # --- Limit 2: the forward-confirmation calendar (audit §10 formula) -----
    forward = {}
    for edge in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        row = {}
        for horizon in HORIZONS:
            sd = limit1[f"{horizon}m"]["session_sd_points"]
            n = math.ceil((Z * sd / edge) ** 2)
            row[f"{horizon}m"] = {"sessions": n, "years": round(n / 252, 2)}
        forward[f"edge_{edge}"] = row

    receipt = {
        "schema": "v5.measurement-review-analysis.v1",
        "data": str(DATA),
        "session_files": len(list(DATA.glob("*.parquet"))),
        "non_empty_sessions": len(sessions),
        "limit1_fixed_0935_slot": limit1,
        "detectable_per_trade_by_frequency_iid": per_trade_table,
        "mean_intraday_lag1_of_15m_returns": round(mean_intraday_lag1, 4),
        "sessions_needed_for_cost_scale_edge": cost_scale,
        "forward_confirmation_sessions": forward,
    }
    OUT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
