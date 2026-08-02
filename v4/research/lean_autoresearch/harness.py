"""Lean experiment harness — real HGB fit on real corpus, causal + holdout-sealed.

One trial = fit HGB on a fold's model_fit sessions, OOF-predict on outer_test, across 5 folds.
Reuses the frozen science decisions (signed-17 features, conservative HGB, $3-8 band,
session-clustered evaluation, OOF). The 30 protected-holdout sessions + burned smoke dates are
NEVER loaded here — a load attempt raises. All features are decision-time (causal); the net-PnL
label under a fixed exit is the forward outcome (target), kept separate from features.
"""
from __future__ import annotations
import json, os
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

CORP = os.path.expanduser("~/.autoresearch-trading/pathd_2025-08-01_2026-07-31")
PROC = f"{CORP}/aligned/processed/minute_entry"
SESSION_ASSIGNMENTS = (
    "v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_v3_2_2026_08_01/"
    "session_assignments.json"
)
BAND_LO, BAND_HI = 3.0, 8.0

_SA = json.load(open(SESSION_ASSIGNMENTS))
HOLDOUT = frozenset(set(_SA["protected_holdout_30"]) | set(_SA.get("burned_smoke_dates", [])))
FOLDS = _SA["folds"]

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
SUPERSET = SIGNED17 + ["size_imbalance", "depth_total", "opt_spread", "iv", "theta"]

# the 7 precomputed exit policies live in each row's label_names; we pick by name
EXIT_POLICIES = (
    "ask_to_bid_stop50_target100_hold25m",
    "ask_to_bid_stop100_target300_hold120m",
    "ask_to_bid_stop100_target999_hold384m",
)


def refuse_if_holdout(session: str) -> None:
    if session in HOLDOUT:
        raise RuntimeError(f"HOLDOUT SEALED: refused to load {session}")


@lru_cache(maxsize=400)
def _load_session(session: str) -> pd.DataFrame:
    """Build one row per band-filtered candidate contract at each context-ready decision minute.

    Columns: session, minute, is_call, mid, all EXIT_POLICIES net-PnL, and the SUPERSET features.
    """
    refuse_if_holdout(session)
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
                )
                for p, v in pnls.items():
                    row[p] = v
                row["nearest_atm_rank"] = abs(offsets[oi])
                rows.append(row)
    return pd.DataFrame(rows)


def fold_frames(fold_index: int):
    """Return (train_df, test_df) for one 0-based fold; holdout excluded by construction."""
    f = FOLDS[fold_index]
    tr_s = [d for d in f["model_fit"] if d not in HOLDOUT]
    te_s = [d for d in f["outer_test"] if d not in HOLDOUT]
    tr = pd.concat([_load_session(d) for d in tr_s], ignore_index=True)
    te = pd.concat([_load_session(d) for d in te_s], ignore_index=True)
    return tr, te


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
        rand = g.sample(n=min(len(ent), len(g)), random_state=int(abs(hash(s)) % (2**31)))
        diffs.append(ent[target].mean() - rand[target].mean())
    diffs = np.asarray(diffs, float)
    return dict(econ=float(diffs.mean()) if len(diffs) else float("nan"),
                n_sessions=len(diffs), n_enter=n_enter)


def evaluate_config(cfg: dict, *, folds=range(5)) -> dict:
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
        rng = np.random.default_rng(7); perm = rng.permutation(len(Xtr))
        sh = make_hgb(max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"], seed=202)
        sh.fit(Xtr[perm], ytr); sh_pred = sh.predict(Xte)
        sh_rho = spearmanr(sh_pred, yte).correlation
        sh_econ = session_clustered_econ(te, sh_pred, target, cfg["threshold"])["econ"]
        # nearest-ATM naive baseline (buy the nearest-ATM candidate per session)
        atm_pnl = te.loc[te.groupby("session")["nearest_atm_rank"].idxmin(), target].mean()
        per_fold.append(dict(
            fold=fi + 1, oof_rho=float(rho) if rho == rho else 0.0,
            econ=econ["econ"], n_enter=econ["n_enter"], n_sessions=econ["n_sessions"],
            signrev_rho=float(sr_rho) if sr_rho == sr_rho else 0.0,
            shuffled_rho=float(sh_rho) if sh_rho == sh_rho else 0.0,
            shuffled_econ=sh_econ, nearest_atm_pnl=float(atm_pnl) if atm_pnl == atm_pnl else 0.0,
        ))
    return dict(config=cfg, per_fold=per_fold)
