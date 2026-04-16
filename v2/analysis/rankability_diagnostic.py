"""Rankability Diagnostic — Can We Gate on Model Competence?

Tests whether the model's ranking competence is predictable from context
features, and whether gating on competence outperforms simple trade caps.

Five analysis steps:
1. Three rankability targets per bar (binary win, relative quality, normalized rank)
2. Competence by confidence bucket with side vs strike split
3. Feature correlations + logistic regression baseline predictor
4. Counterfactual gated replays (oracle ceiling, context proxy, trivial filters)
5. Dual profiles: rankable bars vs false-confidence bars

Usage:
    python3 -m v2.analysis.rankability_diagnostic [--model path]
"""
from __future__ import annotations

import os
import warnings
from collections import defaultdict

import numpy as np
import torch

from v2.core.chain_data import (
    CONTRACT_FEATURE_FIELDS,
    QUALITY_PARTIAL,
    load_sidecar_cached,
    padded_snapshot,
)
from v2.core.policy import DecisionPolicy
from v2.replay import load_model_from_path


# ── Data loading ────────────────────────────────────────────────────

def load_bars_extended(model_path: str):
    """Load model, run inference, return per-bar records + full context matrix.

    Returns (records, context_matrix, feature_names) where context_matrix
    is (n_bars, n_features) aligned with records.
    """
    model = load_model_from_path(model_path)
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    policy = DecisionPolicy()

    features = data["X"].numpy()
    feature_names = data.get("feature_names", [])
    mask = data["promote_mask"].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    sidecar_dir = "v2/data_sidecars"
    max_contracts = data.get("metadata", {}).get("max_contracts_per_bar", 285)

    feat_idx = {name: i for i, name in enumerate(feature_names)}

    eligible = []
    for idx in range(len(mask)):
        if not mask[idx]:
            continue
        day = dates[idx]
        bod = int(bar_of_day[idx])
        if bod < policy.no_trade_before_bar or bod >= policy.no_trade_after_bar:
            continue
        eligible.append((day, idx, bod))

    # Batch inference
    lookback = 30
    gather_idx = []
    for _, g, _ in eligible:
        row = np.arange(max(0, g - lookback + 1), g + 1)
        if len(row) < lookback:
            pad = np.full(lookback - len(row), row[0])
            row = np.concatenate([pad, row])
        gather_idx.append(row)

    all_windows = features[np.stack(gather_idx)]
    snapshots = []
    for day, global_bar, local_bar in eligible:
        sc = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        snap = padded_snapshot(sc, local_bar, max_contracts)
        snapshots.append(snap)

    all_contracts = np.stack([s[0] for s in snapshots]).astype(np.float32)
    all_contract_labels = np.stack([s[1] for s in snapshots]).astype(np.float32)

    model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(all_windows).float()
        batch_c = torch.from_numpy(all_contracts).float()
        all_outputs = model(batch_x, batch_c)
        all_outputs = {k: v.cpu().numpy() for k, v in all_outputs.items()}

    # Build records + context matrix
    records = []
    context_rows = []
    for i, (day, global_bar, local_bar) in enumerate(eligible):
        opp_logit = float(all_outputs["opportunity_logit"][i])
        c_scores = all_outputs["contract_scores"][i]
        v_mask = all_outputs["valid_mask"][i].astype(bool)
        oracle_labels = all_contract_labels[i]
        contract_feats = all_contracts[i]
        n_valid = int(v_mask.sum())

        # Model's best contract
        scores = c_scores.copy()
        scores[~v_mask] = -1e9
        best_row = int(np.argmax(scores)) if v_mask.any() else -1

        # Oracle best
        ol = oracle_labels.copy()
        ol[~v_mask] = -1e9
        ol[~np.isfinite(ol)] = -1e9
        oracle_best_row = int(np.argmax(ol)) if v_mask.any() else -1

        chosen_pnl = float(oracle_labels[best_row]) if best_row >= 0 and np.isfinite(oracle_labels[best_row]) else float("nan")
        oracle_best_pnl = float(oracle_labels[oracle_best_row]) if oracle_best_row >= 0 and np.isfinite(oracle_labels[oracle_best_row]) else float("nan")

        # Rank of chosen contract among valid contracts
        if best_row >= 0 and np.isfinite(oracle_labels[best_row]):
            valid_oracle = oracle_labels[v_mask & np.isfinite(oracle_labels)]
            rank = int(np.sum(valid_oracle > chosen_pnl)) + 1
        else:
            rank = -1

        # Contract features
        if best_row >= 0:
            chosen_right_is_put = contract_feats[best_row, 2] > 0.5
            chosen_moneyness = float(contract_feats[best_row, 11])
            chosen_spread = float(contract_feats[best_row, 4])
            chosen_distance = float(contract_feats[best_row, 12])
        else:
            chosen_right_is_put = False
            chosen_moneyness = chosen_spread = chosen_distance = float("nan")

        if oracle_best_row >= 0:
            oracle_right_is_put = contract_feats[oracle_best_row, 2] > 0.5
            oracle_moneyness = float(contract_feats[oracle_best_row, 11])
        else:
            oracle_right_is_put = False
            oracle_moneyness = float("nan")

        side_correct = (chosen_right_is_put == oracle_right_is_put) if best_row >= 0 and oracle_best_row >= 0 else False

        # Rank within correct-side contracts only
        if best_row >= 0 and side_correct and np.isfinite(chosen_pnl):
            same_side_mask = v_mask.copy()
            for r in range(len(same_side_mask)):
                if same_side_mask[r]:
                    is_put = contract_feats[r, 2] > 0.5
                    if is_put != chosen_right_is_put:
                        same_side_mask[r] = False
            same_side_labels = oracle_labels[same_side_mask & np.isfinite(oracle_labels)]
            rank_within_side = int(np.sum(same_side_labels > chosen_pnl)) + 1
            n_same_side = int(same_side_mask.sum())
        else:
            rank_within_side = -1
            n_same_side = 0

        # Three rankability targets
        binary_win = 1 if np.isfinite(chosen_pnl) and chosen_pnl > 0 else 0
        relative_quality = (chosen_pnl - oracle_best_pnl) if (np.isfinite(chosen_pnl) and np.isfinite(oracle_best_pnl)) else float("nan")
        norm_rank = (1.0 - rank / n_valid) if rank > 0 and n_valid > 0 else float("nan")

        records.append({
            "day": day,
            "bar_of_day": local_bar,
            "opp_logit": opp_logit,
            "n_valid": n_valid,
            "chosen_pnl": chosen_pnl,
            "oracle_best_pnl": oracle_best_pnl,
            "rank": rank,
            "side_correct": side_correct,
            "rank_within_side": rank_within_side,
            "n_same_side": n_same_side,
            "chosen_moneyness": chosen_moneyness,
            "chosen_spread": chosen_spread,
            "chosen_distance": chosen_distance,
            "oracle_moneyness": oracle_moneyness,
            # Three rankability targets
            "binary_win": binary_win,
            "relative_quality": relative_quality,
            "norm_rank": norm_rank,
        })

        # Full context feature vector for this bar
        context_rows.append(features[global_bar])

    context_matrix = np.stack(context_rows)
    return records, context_matrix, feature_names


# ── Bucket definitions ──────────────────────────────────────────────

def make_buckets(n):
    return [
        ("Top 1%",    0, max(1, n // 100)),
        ("Top 5%",    0, max(1, n // 20)),
        ("Top 10%",   0, max(1, n // 10)),
        ("10-25%",    max(1, n // 10), max(1, n // 4)),
        ("25-50%",    max(1, n // 4), n // 2),
        ("50-75%",    n // 2, 3 * n // 4),
        ("Bottom 25%", 3 * n // 4, n),
    ]


# ── Step 1-2: Competence by bucket ─────────────────────────────────

def step1_2_competence_by_bucket(records):
    """Tables A and B: overall and side-split competence by confidence bucket."""
    records_sorted = sorted(records, key=lambda r: r["opp_logit"], reverse=True)
    n = len(records_sorted)
    buckets = make_buckets(n)

    # Table A: Overall competence
    print("\n" + "=" * 110)
    print("  STEP 1-2A: OVERALL COMPETENCE BY CONFIDENCE BUCKET")
    print("=" * 110)
    hdr = (f"{'Bucket':<14} {'Count':>6} {'OppLogit':>10} "
           f"{'BinWin%':>8} {'RelQual':>9} {'NormRank':>9} "
           f"{'ChosenPnL':>10} {'OraclePnL':>10}")
    print(hdr)
    print("-" * 110)

    for label, start, end in buckets:
        subset = records_sorted[start:end]
        avg_logit = np.mean([r["opp_logit"] for r in subset])

        wins = [r["binary_win"] for r in subset]
        bin_win_pct = np.mean(wins) * 100

        rq = [r["relative_quality"] for r in subset if np.isfinite(r["relative_quality"])]
        avg_rq = np.mean(rq) if rq else float("nan")

        nr = [r["norm_rank"] for r in subset if np.isfinite(r["norm_rank"])]
        avg_nr = np.mean(nr) if nr else float("nan")

        cp = [r["chosen_pnl"] for r in subset if np.isfinite(r["chosen_pnl"])]
        avg_cp = np.mean(cp) if cp else float("nan")

        op = [r["oracle_best_pnl"] for r in subset if np.isfinite(r["oracle_best_pnl"])]
        avg_op = np.mean(op) if op else float("nan")

        print(f"  {label:<12} {len(subset):>6} {avg_logit:>9.4f} "
              f"{bin_win_pct:>7.1f}% {avg_rq:>9.4f} {avg_nr:>9.4f} "
              f"{avg_cp:>9.4f} {avg_op:>9.4f}")

    # Table B: Side-only competence
    print("\n" + "=" * 110)
    print("  STEP 2B: SIDE COMPETENCE BY CONFIDENCE BUCKET")
    print("=" * 110)
    hdr = (f"{'Bucket':<14} {'Count':>6} {'SideAcc%':>9} "
           f"{'RankInSide':>11} {'nSameSide':>10} "
           f"{'MoneyGap':>10}")
    print(hdr)
    print("-" * 110)

    for label, start, end in buckets:
        subset = records_sorted[start:end]

        side_acc = np.mean([r["side_correct"] for r in subset]) * 100

        ris = [r["rank_within_side"] for r in subset if r["rank_within_side"] > 0]
        nss = [r["n_same_side"] for r in subset if r["rank_within_side"] > 0]
        if ris:
            avg_ris = np.mean(ris)
            avg_nss = np.mean(nss)
            # Normalized rank within side
            norm_ris = np.mean([1.0 - r["rank_within_side"] / r["n_same_side"]
                                for r in subset
                                if r["rank_within_side"] > 0 and r["n_same_side"] > 0])
        else:
            avg_ris = avg_nss = norm_ris = float("nan")

        money_gaps = [abs(r["chosen_moneyness"] - r["oracle_moneyness"])
                      for r in subset
                      if np.isfinite(r["chosen_moneyness"]) and np.isfinite(r["oracle_moneyness"])]
        avg_money_gap = np.mean(money_gaps) if money_gaps else float("nan")

        print(f"  {label:<12} {len(subset):>6} {side_acc:>8.1f}% "
              f"{avg_ris:>10.1f} {avg_nss:>10.1f} "
              f"{avg_money_gap:>9.4f}")

    # Interpretation
    print(f"\n  Side accuracy tells you: can the model get direction right?")
    print(f"  Rank-within-side tells you: given correct side, can it pick the right strike?")
    print(f"  Moneyness gap tells you: how far off is the chosen strike from oracle?")


# ── Step 3: Predictability ──────────────────────────────────────────

def step3_predictability(records, context_matrix, feature_names):
    """Feature correlations + logistic regression for rankability prediction."""

    n = len(records)
    binary_win = np.array([r["binary_win"] for r in records])
    relative_quality = np.array([r["relative_quality"] for r in records])
    norm_rank = np.array([r["norm_rank"] for r in records])
    n_valid = np.array([r["n_valid"] for r in records], dtype=float)
    bar_of_day = np.array([r["bar_of_day"] for r in records], dtype=float)

    # 3a: Raw correlations
    print("\n" + "=" * 110)
    print("  STEP 3A: FEATURE CORRELATIONS WITH RANKABILITY TARGETS")
    print("=" * 110)

    targets = {
        "binary_win": binary_win,
        "relative_quality": relative_quality,
        "norm_rank": norm_rank,
    }

    # Build extended feature matrix: context features + n_valid + bar_of_day
    ext_names = list(feature_names) + ["n_valid", "bar_of_day"]
    ext_matrix = np.column_stack([context_matrix, n_valid, bar_of_day])

    opp_logit = np.array([r["opp_logit"] for r in records])
    opp_drivers = {"n_valid", "atm_iv", "vrp", "bar_range", "log_chain_volume"}

    for tname, tvals in targets.items():
        finite_mask = np.isfinite(tvals)
        if finite_mask.sum() < 100:
            print(f"\n  {tname}: too few finite values ({finite_mask.sum()}), skipping")
            continue

        correlations = []
        for j, fname in enumerate(ext_names):
            col = ext_matrix[:, j]
            both_ok = finite_mask & np.isfinite(col)
            if both_ok.sum() < 50:
                continue
            r = np.corrcoef(tvals[both_ok], col[both_ok])[0, 1]
            correlations.append((fname, r))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n  Top 10 features for {tname}:")
        print(f"    {'Feature':<30} {'r':>8}  {'Also drives opp_logit?':>24}")
        print(f"    {'-'*30} {'-'*8}  {'-'*24}")
        for fname, r in correlations[:10]:
            is_opp = "  ← YES" if fname in opp_drivers else ""
            print(f"    {fname:<30} {r:>+8.4f}{is_opp:>24}")

        # Count overlap with opp drivers
        top10_names = {c[0] for c in correlations[:10]}
        overlap = top10_names & opp_drivers
        print(f"\n    Overlap with opportunity_logit drivers: {len(overlap)}/{len(opp_drivers)} "
              f"({', '.join(sorted(overlap)) if overlap else 'none'})")

    # 3b: Logistic regression baseline
    print("\n" + "=" * 110)
    print("  STEP 3B: LOGISTIC REGRESSION BASELINE — IS RANKABILITY PREDICTABLE?")
    print("=" * 110)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        X = ext_matrix.copy()
        y = binary_win.copy()

        # Drop rows with NaN features
        valid_rows = np.all(np.isfinite(X), axis=1)
        X = X[valid_rows]
        y = y[valid_rows]
        print(f"  Valid rows for LR: {len(y)} (dropped {n - len(y)} with NaN features)")
        print(f"  Base rate: {y.mean()*100:.1f}% positive")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        all_probs = np.zeros(len(y))
        all_true = np.zeros(len(y))

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])

            lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            lr.fit(X_tr, y[train_idx])

            probs = lr.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y[test_idx], probs)
            aucs.append(auc)
            all_probs[test_idx] = probs
            all_true[test_idx] = y[test_idx]

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"\n  5-fold CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        for i, auc in enumerate(aucs):
            print(f"    Fold {i}: {auc:.4f}")

        # Top-decile lift
        sorted_idx = np.argsort(-all_probs)
        top_decile = sorted_idx[:len(sorted_idx) // 10]
        top_decile_rate = all_true[top_decile].mean()
        base_rate = all_true.mean()
        lift = top_decile_rate / base_rate if base_rate > 0 else 0
        print(f"\n  Top-decile lift: {lift:.3f}")
        print(f"    Top-decile win rate: {top_decile_rate*100:.1f}%")
        print(f"    Base win rate: {base_rate*100:.1f}%")

        # Calibration by predicted probability quintile
        print(f"\n  Calibration by predicted probability quintile:")
        print(f"    {'Quintile':<12} {'PredProb':>10} {'ActualWR':>10} {'Count':>8}")
        quintile_edges = np.percentile(all_probs, [0, 20, 40, 60, 80, 100])
        for q in range(5):
            lo, hi = quintile_edges[q], quintile_edges[q + 1]
            in_q = (all_probs >= lo) & (all_probs < hi + 1e-9)
            if in_q.sum() == 0:
                continue
            pred_avg = all_probs[in_q].mean()
            actual_avg = all_true[in_q].mean()
            print(f"    Q{q+1:<10} {pred_avg:>9.4f} {actual_avg*100:>9.1f}% {in_q.sum():>8}")

        # Feature importance (LR coefficients from last fold)
        print(f"\n  Top 10 LR coefficient magnitudes (last fold):")
        coef = lr.coef_[0]
        coef_idx = np.argsort(np.abs(coef))[::-1]
        for j in coef_idx[:10]:
            print(f"    {ext_names[j]:<30} {coef[j]:>+8.4f}")

        # Store for Step 4
        return mean_auc, all_probs, all_true, valid_rows

    except ImportError:
        print("  sklearn not installed — skipping logistic regression baseline")
        print("  Install with: pip install scikit-learn")
        return None, None, None, None


# ── Step 4: Counterfactual replays ──────────────────────────────────

def _compute_metrics(day_pnls_dict):
    """Compute PF, WR, DD, trades/day from day→[pnl_list] dict.

    Uses oracle-label PnL directly (not simulated). Caveat: does not include
    spread costs, stop-loss exits, or trailing stops.
    """
    all_pnls = []
    trade_counts = []
    daily_totals = []

    for day in sorted(day_pnls_dict.keys()):
        pnls = day_pnls_dict[day]
        all_pnls.extend(pnls)
        trade_counts.append(len(pnls))
        daily_totals.append(sum(pnls))

    if not all_pnls:
        return {"pf": 0, "wr": 0, "dd": 0, "trades": 0, "tpd": 0,
                "net_pnl": 0, "pos_day_rate": 0, "traded_days": 0}

    wins = sum(1 for p in all_pnls if p > 0)
    gross_win = sum(p for p in all_pnls if p > 0)
    gross_loss = abs(sum(p for p in all_pnls if p <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0

    # Equity curve for DD (cumulative PnL)
    cum = np.cumsum(daily_totals)
    peak = np.maximum.accumulate(cum)
    dd_arr = peak - cum
    max_dd = float(dd_arr.max()) if len(dd_arr) > 0 else 0
    # Express as fraction of starting equity (proxy: use 1.0 as base)
    # Since PnLs are oracle label percentages, DD is in same units

    pos_days = sum(1 for d in daily_totals if d > 0)
    traded_days = len(daily_totals)

    return {
        "pf": pf,
        "wr": wins / len(all_pnls) if all_pnls else 0,
        "dd": max_dd,
        "trades": len(all_pnls),
        "tpd": np.mean(trade_counts) if trade_counts else 0,
        "net_pnl": sum(all_pnls),
        "pos_day_rate": pos_days / traded_days if traded_days > 0 else 0,
        "traded_days": traded_days,
        "avg_pnl": sum(all_pnls) / len(all_pnls) if all_pnls else 0,
    }


def _apply_filter(records, keep_mask):
    """Apply a bar-level filter and compute day-level metrics."""
    day_pnls = defaultdict(list)
    early_count = 0
    total_count = 0
    ranks = []
    oracle_opps = []

    for i, r in enumerate(records):
        if not keep_mask[i]:
            continue
        if not np.isfinite(r["chosen_pnl"]):
            continue
        day_pnls[r["day"]].append(r["chosen_pnl"])
        total_count += 1
        if r["bar_of_day"] < 60:
            early_count += 1
        if r["rank"] > 0:
            ranks.append(r["rank"])
        if np.isfinite(r["oracle_best_pnl"]):
            oracle_opps.append(r["oracle_best_pnl"])

    m = _compute_metrics(day_pnls)
    m["early_pct"] = (early_count / total_count * 100) if total_count > 0 else 0
    m["avg_rank"] = np.mean(ranks) if ranks else 0
    m["avg_oracle_opp"] = np.mean(oracle_opps) if oracle_opps else 0
    return m


def step4_counterfactual_replays(records, lr_probs, lr_true, lr_valid_rows):
    """Four-way counterfactual comparison."""
    print("\n" + "=" * 110)
    print("  STEP 4: COUNTERFACTUAL GATED REPLAYS")
    print("=" * 110)
    print("  NOTE: Uses oracle-label PnL (no spread, no stop/TP simulation).")
    print("  Compare relative performance, not absolute PnL numbers.\n")

    n = len(records)
    n_valid_arr = np.array([r["n_valid"] for r in records])
    bar_arr = np.array([r["bar_of_day"] for r in records])
    opp_arr = np.array([r["opp_logit"] for r in records])
    binary_win_arr = np.array([r["binary_win"] for r in records])

    # Compute context features for trivial filters
    atm_iv_vals = []
    for r in records:
        atm_iv_vals.append(r.get("atm_iv", float("nan")))
    # We don't have atm_iv in records — get from context matrix if needed
    # For now, use n_valid and bar_of_day which we do have

    # ── Baseline: trade every eligible bar ──
    baseline_mask = np.ones(n, dtype=bool)

    # ── Max-4 cap: first 4 per day that pass the gate ──
    max4_mask = np.zeros(n, dtype=bool)
    day_counts = defaultdict(int)
    for i, r in enumerate(records):
        if r["opp_logit"] < 0:  # gate_threshold=0.0
            continue
        if day_counts[r["day"]] < 4:
            max4_mask[i] = True
            day_counts[r["day"]] += 1

    # ── 4a: Oracle competence ceiling ──
    # NOTE: pure binary_win would give infinite PF (no losers). Instead, use
    # a softer oracle: trade bars where chosen_pnl is above median, producing
    # a mix of wins and losses that gives a meaningful PF.
    chosen_pnl_arr = np.array([r["chosen_pnl"] for r in records])
    finite_pnl = chosen_pnl_arr[np.isfinite(chosen_pnl_arr)]
    oracle_thresholds = np.percentile(finite_pnl, [50, 60, 70, 80, 90])
    # Also provide the pure-win ceiling separately
    oracle_mask = binary_win_arr.astype(bool)

    # ── 4b: Context-only proxy (if available) ──
    proxy_mask = None
    if lr_probs is not None and lr_valid_rows is not None:
        # Map probabilities back to full record array
        full_probs = np.full(n, 0.5)
        valid_indices = np.where(lr_valid_rows)[0]
        full_probs[valid_indices] = lr_probs

        # Sweep thresholds to find ~4-6 trades/day
        n_days = len(set(r["day"] for r in records))
        best_thresh = 0.5
        best_diff = float("inf")
        for thresh in np.arange(0.1, 0.95, 0.01):
            n_pass = (full_probs >= thresh).sum()
            tpd = n_pass / n_days if n_days > 0 else 0
            if abs(tpd - 5.0) < best_diff:
                best_diff = abs(tpd - 5.0)
                best_thresh = thresh

        proxy_mask = full_probs >= best_thresh
        print(f"  Context proxy threshold: {best_thresh:.2f} "
              f"(targets ~5 TPD, actual: {proxy_mask.sum()/n_days:.1f})")

    n_days = len(set(r["day"] for r in records))

    # ── 4c: Trivial regime filters ──
    p75_nvalid = np.percentile(n_valid_arr, 75)
    p50_nvalid = np.percentile(n_valid_arr, 50)
    trivial_filters = {
        "No early (bar>=60)": bar_arr >= 60,
        f"Not dense (nv<={int(p50_nvalid)})": n_valid_arr <= p50_nvalid,
        f"Not dense (nv<={int(p75_nvalid)})": n_valid_arr <= p75_nvalid,
        f"Combo (>=60 & nv<={int(p75_nvalid)})": (bar_arr >= 60) & (n_valid_arr <= p75_nvalid),
    }

    # ── Report ──
    tests = [("Baseline (all bars)", baseline_mask)]
    tests.append(("Max-4 daily cap", max4_mask))
    for name, tmask in trivial_filters.items():
        tests.append((name, tmask))
    if proxy_mask is not None:
        tests.append(("Context proxy (LR)", proxy_mask))
    tests.append(("Oracle win-only (ceiling)", oracle_mask))

    hdr = (f"{'Filter':<30} {'Trades':>7} {'TPD':>5} {'PF':>7} "
           f"{'WR%':>6} {'AvgPnL':>8} {'+Day%':>6} "
           f"{'Early%':>7} {'AvgRank':>8}")
    print(f"\n  {hdr}")
    print(f"  {'-'*100}")

    result_rows = {}
    for name, mask in tests:
        m = _apply_filter(records, mask)
        result_rows[name] = m
        pf_s = f"{m['pf']:>7.3f}" if np.isfinite(m['pf']) else "    inf"
        print(f"  {name:<30} {m['trades']:>7} {m['tpd']:>5.1f} {pf_s} "
              f"{m['wr']*100:>5.1f}% {m['avg_pnl']:>7.4f} {m['pos_day_rate']*100:>5.1f}% "
              f"{m['early_pct']:>6.1f}% {m['avg_rank']:>8.1f}")

    # Gap analysis — compare using avg PnL per trade (PF is inf for oracle ceiling)
    print(f"\n  --- Gap Analysis (avg PnL per trade) ---")
    baseline = result_rows["Baseline (all bars)"]
    max4 = result_rows["Max-4 daily cap"]
    oracle_win = result_rows["Oracle win-only (ceiling)"]

    print(f"  {'Filter':<30} {'PF':>7} {'AvgPnL':>8} {'WR%':>6} {'Trades':>7}")
    print(f"  {'-'*60}")
    for name in ["Baseline (all bars)", "Max-4 daily cap"]:
        r = result_rows[name]
        pf_s = f"{r['pf']:.3f}" if np.isfinite(r['pf']) else "inf"
        print(f"  {name:<30} {pf_s:>7} {r['avg_pnl']:>7.4f} {r['wr']*100:>5.1f}% {r['trades']:>7}")
    for name, tmask in trivial_filters.items():
        r = result_rows[name]
        print(f"  {name:<30} {r['pf']:>7.3f} {r['avg_pnl']:>7.4f} {r['wr']*100:>5.1f}% {r['trades']:>7}")
    if proxy_mask is not None:
        r = result_rows["Context proxy (LR)"]
        print(f"  {'Context proxy (LR)':<30} {r['pf']:>7.3f} {r['avg_pnl']:>7.4f} {r['wr']*100:>5.1f}% {r['trades']:>7}")
    pf_s = f"{oracle_win['pf']:.3f}" if np.isfinite(oracle_win['pf']) else "inf"
    print(f"  {'Oracle win-only (ceiling)':<30} {pf_s:>7} {oracle_win['avg_pnl']:>7.4f} {oracle_win['wr']*100:>5.1f}% {oracle_win['trades']:>7}")

    # Gap using avg PnL
    baseline_avg = baseline["avg_pnl"]
    max4_avg = max4["avg_pnl"]
    oracle_avg = oracle_win["avg_pnl"]
    oracle_gap = oracle_avg - max4_avg

    print(f"\n  Oracle ceiling avg PnL above max-4: {oracle_gap:+.4f}")

    if proxy_mask is not None:
        proxy_avg = result_rows["Context proxy (LR)"]["avg_pnl"]
        if oracle_gap > 0:
            capture = (proxy_avg - max4_avg) / oracle_gap * 100
            print(f"  Context proxy captures {capture:.1f}% of oracle-to-max4 gap (avg PnL)")
        else:
            print(f"  Oracle ceiling ≤ max-4 on avg PnL → no gap to capture")

    # Best trivial filter
    best_trivial_name = None
    best_trivial_pf = 0
    for name, mask in trivial_filters.items():
        pf = result_rows[name]["pf"]
        if pf > best_trivial_pf:
            best_trivial_pf = pf
            best_trivial_name = name
    if best_trivial_name:
        print(f"\n  Best trivial filter: '{best_trivial_name}' PF={best_trivial_pf:.3f}")
        if proxy_mask is not None:
            proxy_pf = result_rows["Context proxy (LR)"]["pf"]
            if proxy_pf > best_trivial_pf:
                print(f"  Context proxy beats best trivial by {proxy_pf - best_trivial_pf:+.3f} PF")
            else:
                print(f"  Context proxy does NOT beat best trivial filter")


# ── Step 5: Dual profiles ──────────────────────────────────────────

def step5_profiles(records, context_matrix, feature_names):
    """Profile rankable bars vs false-confidence bars."""
    records_sorted = sorted(records, key=lambda r: r["opp_logit"], reverse=True)
    n = len(records_sorted)

    # Build index from sorted order to original order
    # (for context_matrix lookup)
    orig_indices = []
    record_to_orig = {}
    for i, r in enumerate(records):
        record_to_orig[id(r)] = i

    # 5a: Rankable bars (chosen_pnl > 0 AND rank <= 5)
    rankable = [r for r in records
                if np.isfinite(r["chosen_pnl"]) and r["chosen_pnl"] > 0 and 0 < r["rank"] <= 5]
    rankable_idx = [record_to_orig[id(r)] for r in rankable]

    # 5b: False-confidence bars (top 20% confidence AND chosen_pnl < 0)
    top20_cutoff = max(1, n // 5)
    top20_records = set(id(r) for r in records_sorted[:top20_cutoff])
    false_conf = [r for r in records
                  if id(r) in top20_records
                  and np.isfinite(r["chosen_pnl"]) and r["chosen_pnl"] < 0]
    false_conf_idx = [record_to_orig[id(r)] for r in false_conf]

    def _profile(label, subset, ctx_indices):
        print(f"\n  --- {label} (n={len(subset)}) ---")
        if not subset:
            print("    (empty)")
            return

        bars = [r["bar_of_day"] for r in subset]
        nv = [r["n_valid"] for r in subset]
        side_acc = np.mean([r["side_correct"] for r in subset]) * 100
        moneyness = [r["chosen_moneyness"] for r in subset if np.isfinite(r["chosen_moneyness"])]
        oracle_money = [r["oracle_moneyness"] for r in subset if np.isfinite(r["oracle_moneyness"])]
        spreads = [r["chosen_spread"] for r in subset if np.isfinite(r["chosen_spread"])]
        distances = [r["chosen_distance"] for r in subset if np.isfinite(r["chosen_distance"])]
        opp_logits = [r["opp_logit"] for r in subset]

        print(f"    bar_of_day:       mean={np.mean(bars):.1f}  median={np.median(bars):.0f}  "
              f"std={np.std(bars):.1f}")
        print(f"    n_valid:          mean={np.mean(nv):.1f}  median={np.median(nv):.0f}")
        print(f"    opp_logit:        mean={np.mean(opp_logits):.4f}")
        print(f"    side_accuracy:    {side_acc:.1f}%")
        if moneyness:
            print(f"    chosen_moneyness: mean={np.mean(moneyness):.4f}  "
                  f"abs_mean={np.mean(np.abs(moneyness)):.4f}")
        if oracle_money:
            print(f"    oracle_moneyness: mean={np.mean(oracle_money):.4f}")
        if moneyness and oracle_money:
            gaps = [abs(m - o) for m, o in zip(moneyness, oracle_money)]
            print(f"    moneyness_gap:    mean={np.mean(gaps):.4f}")
        if spreads:
            print(f"    chosen_spread:    mean={np.mean(spreads):.4f}")
        if distances:
            print(f"    chosen_distance:  mean={np.mean(distances):.4f}")

        # Bar distribution
        early = sum(1 for b in bars if b < 60)
        mid = sum(1 for b in bars if 60 <= b < 150)
        late = sum(1 for b in bars if b >= 150)
        print(f"    time_split:       early(<60)={early} ({early/len(bars)*100:.0f}%)  "
              f"mid(60-150)={mid} ({mid/len(bars)*100:.0f}%)  "
              f"late(>150)={late} ({late/len(bars)*100:.0f}%)")

        # Top distinguishing context features (compare to full population)
        if ctx_indices and len(feature_names) > 0:
            sub_ctx = context_matrix[ctx_indices]
            full_means = np.nanmean(context_matrix, axis=0)
            full_stds = np.nanstd(context_matrix, axis=0)
            sub_means = np.nanmean(sub_ctx, axis=0)

            diffs = []
            for j, fname in enumerate(feature_names):
                if full_stds[j] > 1e-8:
                    z = (sub_means[j] - full_means[j]) / full_stds[j]
                    diffs.append((fname, z, sub_means[j], full_means[j]))

            diffs.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"\n    Top 5 distinguishing context features (z-score vs population):")
            print(f"      {'Feature':<30} {'z':>7} {'SubMean':>10} {'PopMean':>10}")
            for fname, z, sm, pm in diffs[:5]:
                print(f"      {fname:<30} {z:>+6.2f} {sm:>10.4f} {pm:>10.4f}")

    print("\n" + "=" * 110)
    print("  STEP 5: DUAL PROFILE — RANKABLE vs FALSE-CONFIDENCE BARS")
    print("=" * 110)

    _profile("RANKABLE (chosen_pnl>0 AND rank<=5)", rankable, rankable_idx)
    _profile("FALSE-CONFIDENCE (top 20% confidence AND chosen_pnl<0)", false_conf, false_conf_idx)

    # Direct comparison
    if rankable and false_conf:
        print(f"\n  --- HEAD-TO-HEAD COMPARISON ---")
        r_bar = np.mean([r["bar_of_day"] for r in rankable])
        f_bar = np.mean([r["bar_of_day"] for r in false_conf])
        r_nv = np.mean([r["n_valid"] for r in rankable])
        f_nv = np.mean([r["n_valid"] for r in false_conf])
        r_side = np.mean([r["side_correct"] for r in rankable]) * 100
        f_side = np.mean([r["side_correct"] for r in false_conf]) * 100

        print(f"    {'Metric':<25} {'Rankable':>12} {'False-Conf':>12} {'Delta':>10}")
        print(f"    {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        print(f"    {'avg bar_of_day':<25} {r_bar:>12.1f} {f_bar:>12.1f} {r_bar-f_bar:>+10.1f}")
        print(f"    {'avg n_valid':<25} {r_nv:>12.1f} {f_nv:>12.1f} {r_nv-f_nv:>+10.1f}")
        print(f"    {'side accuracy %':<25} {r_side:>11.1f}% {f_side:>11.1f}% {r_side-f_side:>+9.1f}%")


# ── Step 6: Trivial proxy sanity check ──────────────────────────────

def step6_sanity_check(records):
    """Compare trivial handcrafted gates against each other."""
    print("\n" + "=" * 110)
    print("  SANITY CHECK: TRIVIAL REGIME FILTERS vs EACH OTHER")
    print("=" * 110)
    print("  If a trivial filter captures most of the gain, retraining may be unnecessary.\n")

    n_valid_arr = np.array([r["n_valid"] for r in records])
    bar_arr = np.array([r["bar_of_day"] for r in records])
    n = len(records)

    # Generate several trivial filters
    filters = {
        "All bars": np.ones(n, dtype=bool),
        "bar >= 60": bar_arr >= 60,
        "bar >= 90": bar_arr >= 90,
        "n_valid <= 150": n_valid_arr <= 150,
        "n_valid <= 200": n_valid_arr <= 200,
        "bar>=60 & nv<=200": (bar_arr >= 60) & (n_valid_arr <= 200),
        "bar>=60 & nv<=150": (bar_arr >= 60) & (n_valid_arr <= 150),
        "bar>=90 & nv<=200": (bar_arr >= 90) & (n_valid_arr <= 200),
    }

    hdr = f"  {'Filter':<30} {'Trades':>7} {'TPD':>5} {'PF':>7} {'WR%':>6} {'DD':>8}"
    print(hdr)
    print(f"  {'-'*70}")

    for name, mask in filters.items():
        m = _apply_filter(records, mask)
        print(f"  {name:<30} {m['trades']:>7} {m['tpd']:>5.1f} {m['pf']:>7.3f} "
              f"{m['wr']*100:>5.1f}% {m['dd']:>7.4f}")


# ── Decision gate ───────────────────────────────────────────────────

def decision_gate(lr_auc, records, lr_probs, lr_valid_rows):
    """Apply the three-condition decision gate and recommend next step."""
    print("\n" + "=" * 110)
    print("  DECISION GATE")
    print("=" * 110)

    n = len(records)
    binary_win_arr = np.array([r["binary_win"] for r in records])
    n_valid_arr = np.array([r["n_valid"] for r in records])
    bar_arr = np.array([r["bar_of_day"] for r in records])

    # Oracle win-only ceiling (PF=inf but avg PnL is the meaningful metric)
    oracle_mask = binary_win_arr.astype(bool)
    oracle_m = _apply_filter(records, oracle_mask)

    # Max-4 metrics (gate-filtered: only bars with opp_logit >= 0)
    max4_mask = np.zeros(n, dtype=bool)
    day_counts = defaultdict(int)
    for i, r in enumerate(records):
        if r["opp_logit"] < 0:
            continue
        if day_counts[r["day"]] < 4:
            max4_mask[i] = True
            day_counts[r["day"]] += 1
    max4_m = _apply_filter(records, max4_mask)

    # Context proxy metrics (if available)
    proxy_pf = None
    if lr_probs is not None and lr_valid_rows is not None:
        full_probs = np.full(n, 0.5)
        valid_indices = np.where(lr_valid_rows)[0]
        full_probs[valid_indices] = lr_probs

        n_days = len(set(r["day"] for r in records))
        best_thresh = 0.5
        best_diff = float("inf")
        for thresh in np.arange(0.1, 0.95, 0.01):
            n_pass = (full_probs >= thresh).sum()
            tpd = n_pass / n_days if n_days > 0 else 0
            if abs(tpd - 5.0) < best_diff:
                best_diff = abs(tpd - 5.0)
                best_thresh = thresh
        proxy_mask = full_probs >= best_thresh
        proxy_m = _apply_filter(records, proxy_mask)
        proxy_pf = proxy_m["pf"]

    # Best trivial filter
    best_trivial_pf = 0
    for bar_min in [60, 90]:
        for nv_max in [150, 200, 9999]:
            mask = (bar_arr >= bar_min) & (n_valid_arr <= nv_max)
            m = _apply_filter(records, mask)
            if m["pf"] > best_trivial_pf:
                best_trivial_pf = m["pf"]

    # Use avg PnL per trade for gap analysis (PF is inf for oracle ceiling)
    oracle_avg = oracle_m["avg_pnl"]
    max4_avg = max4_m["avg_pnl"]
    oracle_gap = oracle_avg - max4_avg

    # Context proxy avg PnL
    proxy_avg = None
    if proxy_pf is not None:
        proxy_avg = proxy_m["avg_pnl"]

    # Conditions
    cond1 = lr_auc is not None and (lr_auc > 0.55)
    cond2 = oracle_avg > 0 and oracle_gap > 0
    if proxy_avg is not None and oracle_gap > 0:
        cond3 = (proxy_avg - max4_avg) / oracle_gap > 0.30
    else:
        cond3 = False

    print(f"\n  Condition 1 — Context proxy AUC > 0.55:")
    auc_str = f"{lr_auc:.4f}" if lr_auc is not None else "N/A"
    print(f"    AUC = {auc_str}")
    print(f"    {'✓ PASS' if cond1 else '✗ FAIL'}")

    print(f"\n  Condition 2 — Oracle ceiling avg PnL materially above max-4:")
    print(f"    Oracle avg PnL = {oracle_avg:.4f}, Max-4 avg PnL = {max4_avg:.4f}")
    print(f"    Gap = {oracle_gap:+.4f}")
    print(f"    {'✓ PASS' if cond2 else '✗ FAIL'}")

    print(f"\n  Condition 3 — Context proxy captures >30% of oracle-max4 gap:")
    if proxy_avg is not None:
        capture = (proxy_avg - max4_avg) / oracle_gap * 100 if oracle_gap > 0 else 0
        print(f"    Proxy avg PnL = {proxy_avg:.4f}, captures {capture:.1f}%")
    else:
        print(f"    (no proxy available)")
    print(f"    {'✓ PASS' if cond3 else '✗ FAIL'}")

    print(f"\n  Best trivial filter PF: {best_trivial_pf:.3f}")
    if proxy_pf is not None and proxy_pf > best_trivial_pf:
        print(f"  Context proxy beats trivial by {proxy_pf - best_trivial_pf:+.3f}")
    elif proxy_pf is not None:
        print(f"  Context proxy does NOT beat trivial filters")

    # Verdict
    print(f"\n{'='*110}")
    if cond1 and cond2 and cond3:
        if proxy_pf is not None and proxy_pf <= best_trivial_pf + 0.01:
            print("  VERDICT: SIMPLE REGIME FILTER")
            print("  Trivial filter captures most of the proxy's gain.")
            print("  → Implement rule-based overlay, skip retraining.")
        else:
            print("  VERDICT: PROCEED TO LEARNED RANKABILITY HEAD")
            print("  All three conditions met, proxy beats trivial filters.")
            print("  → Retrain with a competence-predicting gate head.")
    elif cond2:
        print("  VERDICT: ORACLE CEILING EXISTS BUT PROXY TOO WEAK")
        print("  There IS signal, but context features can't capture it well enough.")
        print("  → Consider richer features or nonlinear models before retraining.")
    else:
        print("  VERDICT: STOP — TRADE CAP IS SUFFICIENT")
        print("  Oracle ceiling barely beats max-4. Competence gating is low-leverage.")
        print("  → Keep max_daily_trades=4 overlay and focus effort elsewhere.")
    print(f"{'='*110}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="v2/models/model.pt")
    args = parser.parse_args()

    print("Loading model and data...")
    records, context_matrix, feature_names = load_bars_extended(args.model)
    print(f"  Loaded {len(records)} eligible bars, "
          f"{context_matrix.shape[1]} context features")

    # Steps 1-2
    step1_2_competence_by_bucket(records)

    # Step 3
    lr_auc, lr_probs, lr_true, lr_valid_rows = step3_predictability(
        records, context_matrix, feature_names)

    # Step 4
    step4_counterfactual_replays(records, lr_probs, lr_true, lr_valid_rows)

    # Step 5
    step5_profiles(records, context_matrix, feature_names)

    # Sanity check
    step6_sanity_check(records)

    # Decision gate
    decision_gate(lr_auc, records, lr_probs, lr_valid_rows)


if __name__ == "__main__":
    main()
