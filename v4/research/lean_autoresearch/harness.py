"""Lean experiment harness — real HGB fit on real corpus, causal + holdout-sealed.

One trial = fit HGB on a fold's model_fit sessions, OOF-predict on outer_test, across 5 folds.
Reuses the frozen science decisions (signed-17 features, conservative HGB, $3-8 band,
session-clustered evaluation, OOF). The 30 protected-holdout sessions + burned smoke dates are
NEVER loaded here — a load attempt raises. All features are decision-time (causal); the net-PnL
label under a fixed exit is the forward outcome (target), kept separate from features.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

CORP = os.path.expanduser("~/.autoresearch-trading/pathd_2025-08-01_2026-07-31")
PROC = f"{CORP}/aligned/processed/minute_entry"
OFFICIAL_INDEX = f"{CORP}/vendor/thetadata/index"
SESSION_ASSIGNMENTS = (
    "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_v3_2_2026_08_01/"
    "session_assignments.json"
)
BAND_LO, BAND_HI = 3.0, 8.0

_SA = json.load(open(SESSION_ASSIGNMENTS))
HOLDOUT = frozenset(set(_SA["protected_holdout_30"]) | set(_SA.get("burned_smoke_dates", [])))
PRIMARY_HOLDOUT = frozenset(_SA["protected_holdout_primary_non_degraded_29"])
FOLDS = _SA["folds"]
_HOLDOUT_CAPABILITY = object()

# --- feature families (all causal + live-safe; superset built once, selected per config) ---
SIGNED17 = [
    "spx_vwap_gap_points", "spx_vwap_gap_bps", "spx_vwap_gap_over_session_range",
    "session_range_bps", "momentum_5m_bps", "momentum_15m_bps",
    "momentum_5m_over_session_range", "momentum_15m_over_session_range",
    "omar_clipped", "vwap_side_align", "omar_side_align", "momentum15_side_align",
    "straddle_mid_spot_bps", "put_call_mid_ratio", "side_smile_slope",
    "bs_delta", "bs_gamma", "is_call",
]
FEATURE_SETS = {
    "S0": SIGNED17,
    "S1": SIGNED17 + ["size_imbalance"],
    "S2": SIGNED17 + ["size_imbalance", "depth_total", "opt_spread"],
    "S3": SIGNED17 + ["size_imbalance", "depth_total", "opt_spread", "iv", "theta"],
}
MICROSTRUCTURE_GREEKS = ["size_imbalance", "depth_total", "opt_spread", "iv", "theta"]
VOLATILITY_TIME = [
    "vix_level",
    "vix_change_5m",
    "vix_change_15m",
    "vix_minus_spx_rv15_pct_points",
    "minute_of_session",
    "minutes_to_1555",
    "spx_rv_5m_annualized_pct",
    "spx_rv_15m_annualized_pct",
    "range_expansion_5m_bps",
    "range_expansion_15m_bps",
]
# S4 isolates the genuinely new regime/time family. S5 tests its union with the
# already-vetted S3 widening; neither changes the label, threshold, or replay game.
FEATURE_SETS.update({
    "S4": SIGNED17 + VOLATILITY_TIME,
    "S5": SIGNED17 + MICROSTRUCTURE_GREEKS + VOLATILITY_TIME,
})
SUPERSET = list(dict.fromkeys(SIGNED17 + MICROSTRUCTURE_GREEKS + VOLATILITY_TIME))

# the 7 precomputed exit policies live in each row's label_names; we pick by name
EXIT_POLICIES = (
    "ask_to_bid_stop50_target100_hold25m",
    "ask_to_bid_stop100_target300_hold120m",
    "ask_to_bid_stop100_target999_hold384m",
)


def refuse_if_holdout(session: str) -> None:
    if session in HOLDOUT:
        raise RuntimeError(f"HOLDOUT SEALED: refused to load {session}")


def _annualized_realized_vol_pct(close: np.ndarray, minutes: int) -> float:
    """Causal RMS log-return volatility using only the last completed minutes."""

    values = np.asarray(close, dtype=float)
    if minutes <= 0 or len(values) < minutes + 1:
        return float("nan")
    values = values[-(minutes + 1):]
    if not np.isfinite(values).all() or np.any(values <= 0):
        return float("nan")
    returns = np.diff(np.log(values))
    return float(np.sqrt(np.mean(np.square(returns)) * 252.0 * 390.0) * 100.0)


def causal_volatility_time_features(context: dict) -> dict[str, float]:
    """Derive S4/S5 context from a completed-minute official-index window.

    The last row must be available no later than the decision clock. Historical
    and live paths expose the same ordered ``spx_close``/``vix_close``
    completed-minute window, so no future labels or EOD-only fields are involved.
    """

    decision = pd.Timestamp(context["decision_time"])
    source = pd.Timestamp(context["source_context_time"])
    if decision.tzinfo is None or source.tzinfo is None:
        raise ValueError("lean context clocks must be timezone-aware")
    decision = decision.tz_convert("UTC")
    source = source.tz_convert("UTC")
    if source > decision:
        raise ValueError("future context reached lean feature builder")
    names = {
        name: index for index, name in enumerate(context["market_feature_names"])
    }
    required = {"spx_close", "vix_close", "session_range"}
    if not required.issubset(names):
        raise ValueError("lean S4/S5 market window lacks a live-twin field")
    window = np.asarray(context["market_window"], dtype=float)
    if window.ndim != 2 or len(window) < 16:
        raise ValueError("lean S4/S5 requires at least 16 completed context minutes")
    spx = window[:, names["spx_close"]]
    vix = window[:, names["vix_close"]]
    ranges = window[:, names["session_range"]]
    if not np.isfinite([spx[-1], vix[-1], ranges[-1]]).all() or spx[-1] <= 0:
        raise ValueError("lean S4/S5 current context is nonfinite")
    rv5 = _annualized_realized_vol_pct(spx, 5)
    rv15 = _annualized_realized_vol_pct(spx, 15)
    if not np.isfinite([vix[-6], vix[-16], ranges[-6], ranges[-16], rv5, rv15]).all():
        raise ValueError("lean S4/S5 lagged context is incomplete")
    local = decision.tz_convert(ZoneInfo("America/New_York"))
    minute = local.hour * 60 + local.minute - (9 * 60 + 30)
    if not 0 <= minute <= 385:
        raise ValueError("lean S4/S5 decision lies outside the 09:30-15:55 research session")
    return {
        "vix_level": float(vix[-1]),
        "vix_change_5m": float(vix[-1] - vix[-6]),
        "vix_change_15m": float(vix[-1] - vix[-16]),
        "vix_minus_spx_rv15_pct_points": float(vix[-1] - rv15),
        "minute_of_session": float(minute),
        "minutes_to_1555": float(max(0, 385 - minute)),
        "spx_rv_5m_annualized_pct": rv5,
        "spx_rv_15m_annualized_pct": rv15,
        "range_expansion_5m_bps": float((ranges[-1] - ranges[-6]) / spx[-1] * 1e4),
        "range_expansion_15m_bps": float((ranges[-1] - ranges[-16]) / spx[-1] * 1e4),
    }


@lru_cache(maxsize=600)
def _official_index_frame(session: str, symbol: str) -> pd.DataFrame:
    """Read one official ThetaData index session with strict source invariants."""

    if symbol not in {"SPX", "VIX"}:
        raise ValueError("unsupported official index symbol")
    path = f"{OFFICIAL_INDEX}/{symbol.lower()}_1m/{session}.parquet"
    frame = pd.read_parquet(path)
    required = {
        "event_time",
        "symbol",
        "close",
        "context_source",
        "is_derived",
        "is_proxy",
        "is_official_index_data",
    }
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"official {symbol} minute schema is incomplete")
    frame = frame.copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    valid = (
        frame["symbol"].eq(symbol)
        & frame["context_source"].eq("thetadata_index_history_ohlc")
        & frame["is_derived"].eq(False)
        & frame["is_proxy"].eq(False)
        & frame["is_official_index_data"].eq(True)
    )
    if not bool(valid.all()):
        raise ValueError(f"official {symbol} minute source invariants drifted")
    local_sessions = frame["event_time"].dt.tz_convert("America/New_York").dt.date.astype(str)
    if not bool(local_sessions.eq(session).all()):
        raise ValueError(f"official {symbol} minute rows crossed sessions")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if not np.isfinite(frame["close"].to_numpy(float)).all() or bool((frame["close"] <= 0).any()):
        raise ValueError(f"official {symbol} minute closes are invalid")
    frame = frame.sort_values("event_time")
    if bool(frame["event_time"].duplicated().any()):
        raise ValueError(f"official {symbol} minute clocks are duplicated")
    return frame[["event_time", "close"]].reset_index(drop=True)


def _official_volatility_time_context(entry: dict, *, session: str) -> dict:
    """Rebuild historical S4/S5 inputs under the v3.2 BAR_OPEN +60 clock law."""

    bar_open = pd.Timestamp(entry["decision_time"])
    source_bar_open = pd.Timestamp(entry["source_context_time"])
    if bar_open.tzinfo is None or source_bar_open.tzinfo is None:
        raise ValueError("processed minute clocks must be timezone-aware")
    bar_open = bar_open.tz_convert("UTC")
    source_bar_open = source_bar_open.tz_convert("UTC")
    if source_bar_open != bar_open:
        raise ValueError("processed minute source clocks do not share BAR_OPEN")
    decision = bar_open + pd.Timedelta(minutes=1)

    spx = _official_index_frame(session, "SPX")
    vix = _official_index_frame(session, "VIX")
    spx = spx[spx["event_time"] <= bar_open]
    vix = vix[vix["event_time"] <= bar_open]
    if spx.empty or vix.empty:
        raise ValueError("official SPX/VIX has no completed minute at decision")
    spx_available = pd.Timestamp(spx.iloc[-1]["event_time"]) + pd.Timedelta(minutes=1)
    vix_available = pd.Timestamp(vix.iloc[-1]["event_time"]) + pd.Timedelta(minutes=1)
    for available in (spx_available, vix_available):
        age_seconds = (decision - available).total_seconds()
        if age_seconds < 0 or age_seconds > 90:
            raise ValueError("official SPX/VIX current minute is future or stale")

    minute_index = pd.date_range(
        end=bar_open, periods=30, freq="min", tz="UTC"
    )
    spx_close = spx.set_index("event_time")["close"].reindex(minute_index, method="ffill")
    vix_close = vix.set_index("event_time")["close"].reindex(minute_index, method="ffill")
    spx_series = spx.set_index("event_time")["close"]
    session_range = (spx_series.cummax() - spx_series.cummin()).reindex(
        minute_index, method="ffill"
    )
    window = np.column_stack(
        [
            spx_close.to_numpy(float),
            vix_close.to_numpy(float),
            session_range.to_numpy(float),
        ]
    )
    return {
        "decision_time": decision,
        "source_context_time": max(spx_available, vix_available),
        "market_feature_names": ("spx_close", "vix_close", "session_range"),
        "market_window": window,
    }


@lru_cache(maxsize=500)
def _load_session_impl(session: str, capability: object | None) -> pd.DataFrame:
    """Build one row per band-filtered candidate contract at each context-ready decision minute.

    Columns: session, minute, is_call, mid, all EXIT_POLICIES net-PnL, and the SUPERSET features.
    """
    if capability is None:
        refuse_if_holdout(session)
    elif capability is not _HOLDOUT_CAPABILITY or session not in PRIMARY_HOLDOUT:
        raise RuntimeError("holdout capability cannot read a non-primary session")
    obj = pd.read_pickle(f"{PROC}/{session}.pkl")
    rows = []
    for mi, e in enumerate(obj):
        if not e.get("context_ready"):
            continue
        of = {n: i for i, n in enumerate(e["feature_names"])}
        mf = {n: i for i, n in enumerate(e["market_feature_names"])}
        ladder = np.asarray(e["option_ladder"], float)
        labels = np.asarray(e["labels_net_pnl"], float)
        mask = np.asarray(e["candidate_mask"], bool)
        mw = np.asarray(e["market_window"], float)
        offsets = list(np.asarray(e["strike_offsets"]))
        lbl_idx = {p: e["label_names"].index(p) for p in EXIT_POLICIES if p in e["label_names"]}
        if len(lbl_idx) != len(EXIT_POLICIES):
            continue
        cur = mw[-1]
        spx, vwap = cur[mf["spx_close"]], cur[mf["spx_vwap"]]
        omar, srange = cur[mf["omar"]], cur[mf["session_range"]]
        m5, m15 = cur[mf["momentum_5m"]], cur[mf["momentum_15m"]]
        if not np.isfinite([spx, vwap, srange]).all() or spx <= 0:
            continue
        gap = spx - vwap
        try:
            volatility_time = causal_volatility_time_features(
                _official_volatility_time_context(e, session=session)
            )
        except ValueError:
            continue
        atm = offsets.index(0) if 0 in offsets else int(np.argmin(np.abs(offsets)))
        atm_c, atm_p = ladder[atm, 0, of["mid"]], ladder[atm, 1, of["mid"]]
        straddle = ((atm_c + atm_p) / spx * 1e4) if np.isfinite([atm_c, atm_p]).all() else np.nan
        pc = (atm_p / atm_c) if (np.isfinite([atm_c, atm_p]).all() and atm_c > 0) else np.nan
        ip = offsets.index(5) if 5 in offsets else None
        im = offsets.index(-5) if -5 in offsets else None
        if ip is not None and im is not None:
            mp, mm = ladder[ip, 0, of["mid"]], ladder[im, 0, of["mid"]]
            smile = ((mp - mm) / spx * 1e4) if np.isfinite([mp, mm]).all() else np.nan
        else:
            smile = np.nan
        for oi in range(ladder.shape[0]):
            for ri in range(2):
                if not mask[oi, ri]:
                    continue
                mid = ladder[oi, ri, of["mid"]]
                if not np.isfinite(mid) or not (BAND_LO <= mid <= BAND_HI):
                    continue
                pnls = {p: labels[oi, ri, lbl_idx[p]] for p in EXIT_POLICIES}
                if not all(np.isfinite(v) for v in pnls.values()):
                    continue
                bsz, asz = ladder[oi, ri, of["bid_size"]], ladder[oi, ri, of["ask_size"]]
                denom = bsz + asz
                d = 1.0 if ri == 0 else -1.0
                row = dict(
                    session=session, minute=mi, is_call=float(ri == 0), mid=mid,
                    spx_vwap_gap_points=gap, spx_vwap_gap_bps=gap / spx * 1e4,
                    spx_vwap_gap_over_session_range=(gap / srange if srange else 0.0),
                    session_range_bps=srange / spx * 1e4,
                    momentum_5m_bps=m5 / spx * 1e4, momentum_15m_bps=m15 / spx * 1e4,
                    momentum_5m_over_session_range=(m5 / srange if srange else 0.0),
                    momentum_15m_over_session_range=(m15 / srange if srange else 0.0),
                    omar_clipped=float(np.clip(omar, -3, 3)),
                    vwap_side_align=1.0 if gap * d > 0 else 0.0,
                    omar_side_align=1.0 if omar * d > 0 else 0.0,
                    momentum15_side_align=1.0 if m15 * d > 0 else 0.0,
                    straddle_mid_spot_bps=straddle, put_call_mid_ratio=pc, side_smile_slope=smile,
                    bs_delta=ladder[oi, ri, of["delta"]], bs_gamma=ladder[oi, ri, of["gamma"]],
                    # microstructure / greek extras (causal, live-safe)
                    size_imbalance=((bsz - asz) / denom) if denom > 0 else 0.0,
                    depth_total=(denom if np.isfinite(denom) else 0.0),
                    opt_spread=ladder[oi, ri, of["spread"]],
                    iv=ladder[oi, ri, of["iv"]], theta=ladder[oi, ri, of["theta"]],
                    **volatility_time,
                )
                for p, v in pnls.items():
                    row[p] = v
                row["nearest_atm_rank"] = abs(offsets[oi])
                rows.append(row)
    return pd.DataFrame(rows)


def _load_session(session: str) -> pd.DataFrame:
    """Search-only loader: protected sessions always fail closed."""

    return _load_session_impl(session, None)


def load_holdout_frame(*, access_receipt: dict) -> pd.DataFrame:
    """Consume the exact one-shot receipt and decode only its sealed sessions."""

    sessions = tuple(_SA["protected_holdout_primary_non_degraded_29"])
    semantic = dict(access_receipt)
    digest = semantic.pop("receipt_sha256", None)
    if (
        access_receipt.get("schema_version")
        != "lean_autoresearch.holdout_access_receipt.v2"
        or access_receipt.get("open_count") != 1
        or access_receipt.get("sessions") != list(sessions)
        or access_receipt.get("sessions_sha256")
        != hashlib.sha256(("\n".join(sessions) + "\n").encode()).hexdigest()
        or digest
        != hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise RuntimeError("invalid lean holdout capability")
    return pd.concat(
        [_load_session_impl(session, _HOLDOUT_CAPABILITY) for session in sessions],
        ignore_index=True,
    )


def fold_frames(fold_index: int):
    """Return (train_df, test_df) for one 0-based fold; holdout excluded by construction."""
    f = FOLDS[fold_index]
    tr_s = [d for d in f["model_fit"] if d not in HOLDOUT]
    te_s = [d for d in f["outer_test"] if d not in HOLDOUT]
    tr = pd.concat([_load_session(d) for d in tr_s], ignore_index=True)
    te = pd.concat([_load_session(d) for d in te_s], ignore_index=True)
    return tr, te


def full_preholdout_frame() -> pd.DataFrame:
    """All primary, non-degraded development sessions for the sealed winner."""

    excluded = HOLDOUT | frozenset(_SA.get("opra_degraded_diagnostic_only", []))
    sessions = [
        session
        for session in _SA["pre_holdout_session_indices_1_215"]
        if session not in excluded
    ]
    return pd.concat([_load_session(session) for session in sessions], ignore_index=True)


def make_hgb(*, max_depth: int, min_samples_leaf: int, seed: int = 101):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=100, max_leaf_nodes=31,
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, l2_regularization=1.0,
        max_bins=255, early_stopping=False, random_state=seed,
    )


def _enter_mask(pred: np.ndarray, sessions: pd.Series, threshold: str) -> np.ndarray:
    if threshold == "gt0":
        return pred > 0.0
    if threshold == "gt_fee_reserve":
        return pred > 20.0  # $20 buffer beyond the already-net label
    if threshold == "gt_session_q60":
        df = pd.DataFrame({"pred": pred, "s": sessions.values})
        q = df.groupby("s")["pred"].transform(lambda x: x.quantile(0.60))
        return pred > q.values
    raise ValueError(f"unknown threshold {threshold}")


def session_clustered_econ(te: pd.DataFrame, pred: np.ndarray, target: str, threshold: str):
    """Per-session mean realized net PnL of ENTER picks minus matched-random equal-count subset."""
    df = te.assign(_pred=pred, _enter=_enter_mask(pred, te["session"], threshold))
    diffs, n_enter = [], 0
    for s, g in df.groupby("session"):
        ent = g[g._enter]
        if len(ent) == 0:
            continue
        n_enter += len(ent)
        seed = int.from_bytes(hashlib.sha256(str(s).encode()).digest()[:4], "big")
        rand = g.sample(n=min(len(ent), len(g)), random_state=seed)
        diffs.append(ent[target].mean() - rand[target].mean())
    diffs = np.asarray(diffs, float)
    return dict(econ=float(diffs.mean()) if len(diffs) else float("nan"),
                n_sessions=len(diffs), n_enter=n_enter,
                session_diffs=diffs.tolist())


def evaluate_config(cfg: dict, *, folds=range(5), include_shuffled_control: bool = True) -> dict:
    """Fit + OOF a config across folds; return per-fold OOF rho, economics, and negative controls."""
    feats = FEATURE_SETS[cfg["feature_set"]]
    target = cfg["exit_policy"]
    per_fold = []
    for fi in folds:
        tr, te = fold_frames(fi)
        Xtr = tr[feats].to_numpy(float); ytr = tr[target].to_numpy(float)
        Xte = te[feats].to_numpy(float); yte = te[target].to_numpy(float)
        model = make_hgb(max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"])
        model.fit(Xtr, ytr); pred = model.predict(Xte)
        rho = spearmanr(pred, yte).correlation
        econ = session_clustered_econ(te, pred, target, cfg["threshold"])
        # negative controls
        sr_rho = spearmanr(-pred, yte).correlation
        sr_econ = session_clustered_econ(te, -pred, target, cfg["threshold"])["econ"]
        sh_rho = sh_econ = None
        if include_shuffled_control:
            rng = np.random.default_rng(7)
            perm = rng.permutation(len(Xtr))
            sh = make_hgb(
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                seed=202,
            )
            sh.fit(Xtr[perm], ytr)
            sh_pred = sh.predict(Xte)
            sh_rho_raw = spearmanr(sh_pred, yte).correlation
            sh_rho = float(sh_rho_raw) if sh_rho_raw == sh_rho_raw else 0.0
            sh_econ = session_clustered_econ(
                te, sh_pred, target, cfg["threshold"]
            )["econ"]
        # nearest-ATM naive baseline (buy the nearest-ATM candidate per session)
        atm_pnl = te.loc[te.groupby("session")["nearest_atm_rank"].idxmin(), target].mean()
        per_fold.append(dict(
            fold=fi + 1, oof_rho=float(rho) if rho == rho else 0.0,
            econ=econ["econ"], n_enter=econ["n_enter"], n_sessions=econ["n_sessions"],
            signrev_rho=float(sr_rho) if sr_rho == sr_rho else 0.0,
            signrev_econ=sr_econ,
            shuffled_rho=sh_rho,
            shuffled_econ=sh_econ, nearest_atm_pnl=float(atm_pnl) if atm_pnl == atm_pnl else 0.0,
        ))
    return dict(config=cfg, per_fold=per_fold)
