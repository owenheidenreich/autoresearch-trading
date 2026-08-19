"""Chain-internal state: the one information family this corpus has never fitted.

**Why this module exists at all.** Every feature family already fitted here is
measured dead — chart/clock/geometry by the 375-cell census (ledger row 338),
price-and-volume selection by rows 339/340 (zero at the mid with the spread
removed entirely), greeks by row 336. Row 338 names its own reopening conditions,
and of those, cross-asset context is barred by the SPXW/SPX-only ruling, term
structure cannot exist in a 0DTE-only corpus, and the event calendar is outside
the charter's single authorized purchase. What remains is **order-flow imbalance
and chain-wide skew**, both of which are already inside the ladder tensor and
have only ever been consumed as *geometry*: `bid_size` and `ask_size` have been
carried since the tensor was built and no fitted model has read them.

**Why it is a separate module.** `build_causal_day_dataset.py` is hash-pinned by
`PREACQUISITION_SEMANTIC_FREEZE_V1`, whose post-contact rule forbids changing a
listed source after the preflight and directs development iterations through the
alpha ledger instead. These features are therefore *derived from* the pinned
builder's output rather than added inside it.

**Causality.** Every value is computed from the contemporaneous ladder snapshot
and strictly earlier minutes. The lagged terms use the session's own minute grid,
so a missing minute produces NaN rather than silently reaching further back.

Design reference: `work/lifecycle-training/LEARNING_CONTENT_DESIGN_2026_08_16.md`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Minute-common chain state. Each must be able to change which strike the model
# prefers, via `direction(state)*is_call` or `depth(state)*moneyness`.
CHAIN_STATE_FEATURES = (
    "risk_reversal",
    "risk_reversal_change_15m",
    "atm_iv_change_5m",
    "chain_depth_imbalance",
    "put_call_depth_ratio",
    "smile_curvature",
    "implied_spot_dispersion_bps",
)

# Per-contract informational terms; these reorder directly.
CONTRACT_CHAIN_FEATURES = ("smile_residual", "contract_depth_imbalance")

RISK_REVERSAL_LAG_MINUTES = 15
ATM_IV_LAG_MINUTES = 5
ATM_BAND_POINTS = 10.0
PARITY_WINDOW_POINTS = 25.0
MIN_SMILE_POINTS = 4
MIN_PAIRED_STRIKES = 3


class ChainFeatureError(RuntimeError):
    """The ladder cannot support the declared chain-internal state."""


def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ChainFeatureError(f"ladder is missing columns: {missing}")


def _minute_index(minutes: pd.Series) -> pd.Series:
    """Minutes since midnight, so a lag is a real clock lag, not a row offset."""

    parts = minutes.astype(str).str.split(":", expand=True).astype(int)
    return parts[0] * 60 + parts[1]


def _fit_smile(moneyness: np.ndarray, iv: np.ndarray) -> tuple[np.ndarray, float]:
    """Quadratic IV-vs-moneyness fit; returns (fitted, curvature).

    The fit itself is geometry. Its *curvature* is how the chain prices tails
    against the body, and the *residual* is the market disagreeing with its own
    surface about one strike — which is information, not description.
    """

    usable = np.isfinite(moneyness) & np.isfinite(iv)
    if usable.sum() < MIN_SMILE_POINTS or np.unique(moneyness[usable]).size < 3:
        return np.full(len(iv), np.nan), float("nan")
    coefficients = np.polyfit(moneyness[usable], iv[usable], 2)
    fitted = np.polyval(coefficients, moneyness)
    return np.where(np.isfinite(moneyness), fitted, np.nan), float(coefficients[0])


def contract_chain_features(ladder: pd.DataFrame) -> pd.DataFrame:
    """Per-contract relative value and displayed lean.

    Returns a frame aligned to `ladder.index` carrying `CONTRACT_CHAIN_FEATURES`.
    """

    _require(
        ladder,
        ("session", "minute", "is_call", "moneyness_itm_points", "self_iv",
         "bid_size", "ask_size"),
    )
    out = pd.DataFrame(index=ladder.index, columns=list(CONTRACT_CHAIN_FEATURES), dtype=float)

    bid_size = pd.to_numeric(ladder["bid_size"], errors="coerce").fillna(0.0)
    ask_size = pd.to_numeric(ladder["ask_size"], errors="coerce").fillna(0.0)
    total = bid_size + ask_size
    out["contract_depth_imbalance"] = np.where(
        total > 0.0, (bid_size - ask_size) / total.where(total > 0.0, 1.0), 0.0
    )

    # The smile is fitted per side per minute: a call smile and a put smile are
    # different curves, so pooling them would manufacture residuals.
    moneyness = pd.to_numeric(ladder["moneyness_itm_points"], errors="coerce").to_numpy(float)
    iv = pd.to_numeric(ladder["self_iv"], errors="coerce").to_numpy(float)
    residual = np.full(len(ladder), np.nan)
    for _, positions in ladder.groupby(
        [ladder["session"], ladder["minute"], ladder["is_call"].astype(bool)], sort=False
    ).indices.items():
        fitted, _ = _fit_smile(moneyness[positions], iv[positions])
        residual[positions] = iv[positions] - fitted
    out["smile_residual"] = residual
    return out


def _parity_dispersion(group: pd.DataFrame) -> float:
    """Cross-strike dispersion of the parity spot, in bps of its median.

    Microstructure stress, and the quality channel for the parity underlying the
    backfill era depends on: when strikes disagree about spot, the chain is being
    repriced.
    """

    priced = group[np.isfinite(pd.to_numeric(group["mid"], errors="coerce"))]
    if priced.empty:
        return float("nan")
    wide = priced.pivot_table(
        index="strike", columns="is_call", values="mid", aggfunc="last"
    )
    if True not in wide.columns or False not in wide.columns:
        return float("nan")
    paired = wide.dropna()
    if len(paired) < MIN_PAIRED_STRIKES:
        return float("nan")
    implied = paired.index.to_numpy(float) + paired[True].to_numpy(float) - paired[
        False
    ].to_numpy(float)
    centre = float(np.median(implied))
    near = implied[np.abs(paired.index.to_numpy(float) - centre) <= PARITY_WINDOW_POINTS]
    if near.size < MIN_PAIRED_STRIKES or centre <= 0.0:
        return float("nan")
    return float(np.std(near, ddof=0) / centre * 10_000.0)


def chain_state(ladder: pd.DataFrame) -> pd.DataFrame:
    """Minute-common chain-internal state, one row per (session, minute).

    Levels are contemporaneous; the two change terms read strictly earlier
    minutes of the same session.
    """

    _require(
        ladder,
        ("session", "minute", "is_call", "strike", "mid", "self_iv",
         "moneyness_itm_points", "bid_size", "ask_size"),
    )
    frame = ladder.copy()
    frame["is_call"] = frame["is_call"].astype(bool)
    frame["self_iv"] = pd.to_numeric(frame["self_iv"], errors="coerce")
    frame["moneyness_itm_points"] = pd.to_numeric(
        frame["moneyness_itm_points"], errors="coerce"
    )
    frame["bid_size"] = pd.to_numeric(frame["bid_size"], errors="coerce").fillna(0.0)
    frame["ask_size"] = pd.to_numeric(frame["ask_size"], errors="coerce").fillna(0.0)

    rows: list[dict[str, object]] = []
    for (session, minute), group in frame.groupby(["session", "minute"], sort=True):
        calls = group[group["is_call"]]
        puts = group[~group["is_call"]]
        call_depth = float(calls["bid_size"].sum() + calls["ask_size"].sum())
        put_depth = float(puts["bid_size"].sum() + puts["ask_size"].sum())
        depth = call_depth + put_depth
        bid_total = float(group["bid_size"].sum())
        ask_total = float(group["ask_size"].sum())
        size_total = bid_total + ask_total

        atm = group[group["moneyness_itm_points"].abs() <= ATM_BAND_POINTS]
        _, curvature = _fit_smile(
            group["moneyness_itm_points"].to_numpy(float),
            group["self_iv"].to_numpy(float),
        )
        rows.append(
            {
                "session": session,
                "minute": minute,
                "risk_reversal": float(calls["self_iv"].median() - puts["self_iv"].median())
                if len(calls) and len(puts)
                else np.nan,
                "atm_iv": float(atm["self_iv"].median()) if len(atm) else np.nan,
                "chain_depth_imbalance": (bid_total - ask_total) / size_total
                if size_total > 0.0
                else 0.0,
                "put_call_depth_ratio": put_depth / depth if depth > 0.0 else 0.5,
                "smile_curvature": curvature,
                "implied_spot_dispersion_bps": _parity_dispersion(group),
            }
        )

    state = pd.DataFrame(rows)
    if state.empty:
        raise ChainFeatureError("ladder produced no minute-level chain state")

    # Clock-true lags: reindexing on the minute number means a missing minute
    # yields NaN instead of quietly reaching further into the past.
    state["_minute_index"] = _minute_index(state["minute"])
    parts = []
    for _, group in state.groupby("session", sort=True):
        group = group.sort_values("_minute_index").copy()
        indexed = group.set_index("_minute_index")
        for source, lag, name in (
            ("risk_reversal", RISK_REVERSAL_LAG_MINUTES, "risk_reversal_change_15m"),
            ("atm_iv", ATM_IV_LAG_MINUTES, "atm_iv_change_5m"),
        ):
            past = indexed[source].reindex(indexed.index - lag)
            group[name] = indexed[source].to_numpy() - past.to_numpy()
        parts.append(group)
    state = pd.concat(parts, ignore_index=True)
    return state.drop(columns=["_minute_index", "atm_iv"])[
        ["session", "minute", *CHAIN_STATE_FEATURES]
    ]
