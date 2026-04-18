"""Deep analysis of exp_167b competence score.

Determines whether the learned competence score is useful as an ordering/ranking
signal even though it failed as a binary gate. Compares three scores throughout:
  1. exp_167b competence score (learned opportunity_logit)
  2. exp_165 baseline opportunity score
  3. LR proxy (logistic regression on context features)

Usage:
    python3 -m v2.analysis.competence_score_analysis \
        --model v2/models/model_candidate.pt \
        --baseline v2/models/model.pt
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
from scipy import stats as sp_stats

from v2.core.chain_data import load_sidecar_cached, padded_snapshot
from v2.core.policy import DecisionPolicy
from v2.replay import load_model_from_path


# ── Data loading ────────────────────────────────────────────────────

def load_two_model_inference(model_path: str, baseline_path: str):
    """Run inference with both models on promote_mask bars.

    Returns (records, context_matrix, feature_names).
    Each record has 'comp_score' (exp_167b) and 'opp_score' (baseline).
    """
    model = load_model_from_path(model_path)
    baseline = load_model_from_path(baseline_path)
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

    lookback = 30
    gather_idx = []
    for _, g, _ in eligible:
        row = np.arange(max(0, g - lookback + 1), g + 1)
        if len(row) < lookback:
            row = np.concatenate([np.full(lookback - len(row), row[0]), row])
        gather_idx.append(row)

    all_windows = features[np.stack(gather_idx)]
    snapshots = []
    for day, global_bar, local_bar in eligible:
        sc = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        snapshots.append(padded_snapshot(sc, local_bar, max_contracts))

    all_contracts = np.stack([s[0] for s in snapshots]).astype(np.float32)
    all_labels = np.stack([s[1] for s in snapshots]).astype(np.float32)

    batch_x = torch.from_numpy(all_windows).float()
    batch_c = torch.from_numpy(all_contracts).float()

    model.eval()
    baseline.eval()
    with torch.no_grad():
        out_comp = model(batch_x, batch_c)
        out_comp = {k: v.cpu().numpy() for k, v in out_comp.items()}
        out_base = baseline(batch_x, batch_c)
        out_base = {k: v.cpu().numpy() for k, v in out_base.items()}

    records = []
    context_rows = []
    for i, (day, global_bar, local_bar) in enumerate(eligible):
        vm = out_comp["valid_mask"][i].astype(bool)
        ol = all_labels[i]
        n_valid = int(vm.sum())
        cf = all_contracts[i]

        # Competence model's chosen contract
        cs_comp = out_comp["contract_scores"][i].copy()
        cs_comp[~vm] = -1e9
        best_comp = int(np.argmax(cs_comp)) if vm.any() else -1

        # Baseline model's chosen contract
        cs_base = out_base["contract_scores"][i].copy()
        cs_base[~vm] = -1e9
        best_base = int(np.argmax(cs_base)) if vm.any() else -1

        # Use competence model's chosen contract for PnL/rank
        best = best_comp
        if best < 0 or not np.isfinite(ol[best]):
            continue

        chosen_pnl = float(ol[best])

        # Oracle best
        ol_masked = ol.copy()
        ol_masked[~vm] = -1e9
        ol_masked[~np.isfinite(ol_masked)] = -1e9
        oracle_best = int(np.argmax(ol_masked))
        oracle_best_pnl = float(ol[oracle_best]) if np.isfinite(ol[oracle_best]) else float("nan")

        # Rank
        valid_oracle = ol[vm & np.isfinite(ol)]
        rank = int(np.sum(valid_oracle > chosen_pnl)) + 1

        # Side info
        chosen_is_put = cf[best, 2] > 0.5
        oracle_is_put = cf[oracle_best, 2] > 0.5 if oracle_best >= 0 else False
        side_correct = (chosen_is_put == oracle_is_put)

        # Context features for profiling
        ctx = features[global_bar]
        ctx_dict = {}
        for fname in ["atm_iv", "vrp", "atm_gamma", "session_range_pct",
                       "bar_range", "vix_regime", "atr_14", "realized_vol"]:
            if fname in feat_idx:
                ctx_dict[fname] = float(ctx[feat_idx[fname]])

        records.append({
            "day": day,
            "bar_of_day": local_bar,
            "n_valid": n_valid,
            "comp_score": float(out_comp["opportunity_logit"][i]),
            "opp_score": float(out_base["opportunity_logit"][i]),
            "chosen_pnl": chosen_pnl,
            "oracle_best_pnl": oracle_best_pnl,
            "rank": rank,
            "chosen_is_put": chosen_is_put,
            "oracle_is_put": oracle_is_put,
            "side_correct": side_correct,
            "binary_win": chosen_pnl > 0,
            "top5": rank <= 5,
            **ctx_dict,
        })
        context_rows.append(ctx)

    context_matrix = np.stack(context_rows)
    return records, context_matrix, feature_names


def fit_lr_proxy(records, context_matrix):
    """Fit LR proxy predicting rank<=5 from context. Returns predicted probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    nv = np.array([r["n_valid"] for r in records], dtype=float)
    bod = np.array([r["bar_of_day"] for r in records], dtype=float)
    X = np.column_stack([context_matrix, nv, bod])
    y = np.array([r["top5"] for r in records], dtype=float)
    valid = np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]

    probs = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for _, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler()
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(sc.fit_transform(X[tr]), y[tr])
        probs[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]

    # Map back to full record indices
    full_probs = np.full(len(records), 0.5)
    valid_idx = np.where(valid)[0]
    full_probs[valid_idx] = probs
    return full_probs


# ── Helpers ─────────────────────────────────────────────────────────

def _pf(pnls):
    w = sum(p for p in pnls if p > 0)
    l = abs(sum(p for p in pnls if p <= 0))
    return w / l if l > 0 else float("inf") if w > 0 else 0


def _metrics_from_mask(records, mask):
    """Compute trading metrics for bars where mask is True."""
    day_pnls = defaultdict(list)
    pnls, ranks, bars, nvs = [], [], [], []
    calls, puts, oracle_calls, oracle_puts = 0, 0, 0, 0
    side_correct_count = 0

    for i, r in enumerate(records):
        if not mask[i]:
            continue
        day_pnls[r["day"]].append(r["chosen_pnl"])
        pnls.append(r["chosen_pnl"])
        ranks.append(r["rank"])
        bars.append(r["bar_of_day"])
        nvs.append(r["n_valid"])
        if r["chosen_is_put"]:
            puts += 1
        else:
            calls += 1
        if r["oracle_is_put"]:
            oracle_puts += 1
        else:
            oracle_calls += 1
        if r["side_correct"]:
            side_correct_count += 1

    n = len(pnls)
    if n == 0:
        return {"trades": 0, "tpd": 0, "pf": 0, "wr": 0, "avg_pnl": 0,
                "avg_rank": 0, "avg_bar": 0, "avg_nv": 0, "early_pct": 0,
                "call_pct": 0, "oracle_call_pct": 0, "side_acc": 0}

    n_days = len(day_pnls)
    return {
        "trades": n,
        "tpd": n / n_days if n_days > 0 else 0,
        "pf": _pf(pnls),
        "wr": sum(1 for p in pnls if p > 0) / n,
        "avg_pnl": np.mean(pnls),
        "avg_rank": np.mean(ranks),
        "avg_bar": np.mean(bars),
        "avg_nv": np.mean(nvs),
        "early_pct": sum(1 for b in bars if b < 60) / n * 100,
        "call_pct": calls / n * 100,
        "oracle_call_pct": oracle_calls / n * 100,
        "side_acc": side_correct_count / n * 100,
    }


def _print_metrics_row(name, m, show_side=False):
    pf_s = f"{m['pf']:.3f}" if np.isfinite(m["pf"]) else "inf"
    line = (f"  {name:<32s} {m['trades']:>5} {m['tpd']:>5.1f} {pf_s:>7} "
            f"{m['wr']*100:>5.1f}% {m['avg_pnl']:>8.4f} "
            f"{m['avg_rank']:>6.1f} {m['avg_bar']:>6.1f} {m['avg_nv']:>5.1f} "
            f"{m['early_pct']:>5.1f}%")
    if show_side:
        line += f" {m['call_pct']:>5.1f}% {m['side_acc']:>5.1f}%"
    print(line)


# ── Q1: Score ordering ─────────────────────────────────────────────

def q1_score_ordering(records, lr_probs):
    print("\n" + "=" * 120)
    print("  Q1: SCORE USEFULNESS AS ORDERING")
    print("=" * 120)

    scores = {
        "Competence (167b)": np.array([r["comp_score"] for r in records]),
        "Opportunity (165)": np.array([r["opp_score"] for r in records]),
        "LR proxy": lr_probs,
    }
    chosen_pnl = np.array([r["chosen_pnl"] for r in records])
    binary_win = np.array([r["binary_win"] for r in records], dtype=float)
    base_wr = binary_win.mean()

    for sname, svals in scores.items():
        print(f"\n  --- {sname} ---")
        sorted_idx = np.argsort(-svals)
        n = len(sorted_idx)
        decile_size = n // 10

        hdr = (f"    {'Decile':<10} {'Score':>8} {'PF':>7} {'WR%':>6} "
               f"{'AvgPnL':>9} {'AvgRank':>8} {'AvgOrc':>8} {'n':>5}")
        print(hdr)
        print(f"    {'-'*10} {'-'*8} {'-'*7} {'-'*6} {'-'*9} {'-'*8} {'-'*8} {'-'*5}")

        decile_pfs = []
        for d in range(10):
            s = d * decile_size
            e = (d + 1) * decile_size if d < 9 else n
            idx = sorted_idx[s:e]
            sub = [records[i] for i in idx]
            pnls = [r["chosen_pnl"] for r in sub]
            pf = _pf(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            avg_pnl = np.mean(pnls)
            avg_rank = np.mean([r["rank"] for r in sub])
            avg_orc = np.mean([r["oracle_best_pnl"] for r in sub if np.isfinite(r["oracle_best_pnl"])])
            avg_score = np.mean(svals[idx])
            decile_pfs.append(pf)
            pf_s = f"{pf:.3f}" if np.isfinite(pf) else "inf"
            print(f"    D{d+1:<9} {avg_score:>8.4f} {pf_s:>7} {wr*100:>5.1f}% "
                  f"{avg_pnl:>8.4f} {avg_rank:>8.1f} {avg_orc:>8.4f} {len(sub):>5}")

        # Ranking quality metrics
        finite = np.isfinite(svals) & np.isfinite(chosen_pnl)
        spearman = sp_stats.spearmanr(svals[finite], chosen_pnl[finite])
        top_decile_wr = binary_win[sorted_idx[:decile_size]].mean()
        lift = top_decile_wr / base_wr if base_wr > 0 else 0
        top_q = sorted_idx[:n // 4]
        bot_q = sorted_idx[3 * n // 4:]
        pf_top = _pf([records[i]["chosen_pnl"] for i in top_q])
        pf_bot = _pf([records[i]["chosen_pnl"] for i in bot_q])

        print(f"\n    Spearman ρ: {spearman.correlation:+.4f} (p={spearman.pvalue:.2e})")
        print(f"    Top-decile lift: {lift:.3f} (top WR={top_decile_wr*100:.1f}%, base={base_wr*100:.1f}%)")
        pf_top_s = f"{pf_top:.3f}" if np.isfinite(pf_top) else "inf"
        pf_bot_s = f"{pf_bot:.3f}" if np.isfinite(pf_bot) else "inf"
        print(f"    Top-Q PF: {pf_top_s}  Bot-Q PF: {pf_bot_s}  Gap: {pf_top - pf_bot:+.3f}" if np.isfinite(pf_top) and np.isfinite(pf_bot) else f"    Top-Q PF: {pf_top_s}  Bot-Q PF: {pf_bot_s}")

        # Monotonicity check
        violations = sum(1 for i in range(len(decile_pfs) - 1) if decile_pfs[i] < decile_pfs[i + 1])
        print(f"    Monotonicity violations: {violations}/9")


# ── Q2: Threshold sweep ────────────────────────────────────────────

def q2_threshold_sweep(records, lr_probs):
    print("\n" + "=" * 120)
    print("  Q2: THRESHOLD SWEEP")
    print("=" * 120)

    scores = {
        "Competence (167b)": np.array([r["comp_score"] for r in records]),
        "Opportunity (165)": np.array([r["opp_score"] for r in records]),
        "LR proxy": lr_probs,
    }
    n = len(records)
    binary_win = np.array([r["binary_win"] for r in records])

    for sname, svals in scores.items():
        print(f"\n  --- {sname} ---")
        percentiles = np.arange(5, 100, 5)
        thresholds = np.percentile(svals, percentiles)

        hdr = (f"    {'Pctile':>6} {'Thresh':>8} {'Pass%':>6} {'Trades':>7} "
               f"{'Prec':>6} {'Recall':>7} {'PF':>7} {'WR%':>6} {'AvgPnL':>8} "
               f"{'AvgBar':>7} {'AvgNv':>6} {'AvgRnk':>7}")
        print(hdr)
        print(f"    {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*7}")

        best_pf = 0
        best_pctile = 0
        for pctile, thresh in zip(percentiles, thresholds):
            mask = svals >= thresh
            passed = [records[i] for i in range(n) if mask[i]]
            if len(passed) < 10:
                continue
            pnls = [r["chosen_pnl"] for r in passed]
            wins_passed = sum(1 for r in passed if r["binary_win"])
            total_wins = sum(binary_win)
            prec = wins_passed / len(passed) if passed else 0
            recall = wins_passed / total_wins if total_wins > 0 else 0
            pf = _pf(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            avg_pnl = np.mean(pnls)
            avg_bar = np.mean([r["bar_of_day"] for r in passed])
            avg_nv = np.mean([r["n_valid"] for r in passed])
            avg_rank = np.mean([r["rank"] for r in passed])
            pf_s = f"{pf:.3f}" if np.isfinite(pf) else "inf"

            marker = ""
            if len(passed) >= 200 and (np.isfinite(pf) and pf > best_pf):
                best_pf = pf
                best_pctile = pctile
                marker = " ← BEST"

            print(f"    p{pctile:>4} {thresh:>8.4f} {mask.sum()/n*100:>5.1f}% {len(passed):>7} "
                  f"{prec*100:>5.1f}% {recall*100:>6.1f}% {pf_s:>7} {wr*100:>5.1f}% {avg_pnl:>7.4f} "
                  f"{avg_bar:>7.1f} {avg_nv:>5.1f} {avg_rank:>7.1f}{marker}")

        print(f"\n    Best PF at >=200 trades: p{best_pctile} (PF={best_pf:.3f})")


# ── Q3: Policy comparison ──────────────────────────────────────────

def q3_policy_comparison(records, lr_probs):
    print("\n" + "=" * 120)
    print("  Q3: GATE POLICY COMPARISON (matched ~4-5 TPD where possible)")
    print("=" * 120)

    n = len(records)
    comp = np.array([r["comp_score"] for r in records])
    opp = np.array([r["opp_score"] for r in records])
    bar = np.array([r["bar_of_day"] for r in records])
    nv = np.array([r["n_valid"] for r in records])
    n_days = len(set(r["day"] for r in records))

    # Find best threshold for ~5 TPD
    best_t, best_diff = 0.0, float("inf")
    for pctile in np.arange(50, 98, 1):
        t = np.percentile(comp, pctile)
        tpd = (comp >= t).sum() / n_days
        if abs(tpd - 5.0) < best_diff:
            best_diff = abs(tpd - 5.0)
            best_t = t

    # Group by day for top-k policies
    day_indices = defaultdict(list)
    for i, r in enumerate(records):
        day_indices[r["day"]].append(i)

    policies = {}

    # 1. Threshold (best from sweep)
    policies["Threshold (best)"] = ("No", comp >= best_t)

    # 2. Top-4/day online: accept if bar would be in top-4 so far
    mask_top4_online = np.zeros(n, dtype=bool)
    for day, idxs in day_indices.items():
        seen_scores = []
        for idx in idxs:  # idxs are in chronological order
            seen_scores.append((comp[idx], idx))
            seen_scores.sort(reverse=True)
            if len(seen_scores) <= 4:
                mask_top4_online[idx] = True
            elif comp[idx] >= seen_scores[3][0]:
                # This bar is in top-4 so far
                mask_top4_online[idx] = True
                # But we may need to un-mark the previous #4
                # For simplicity: mark top-4 of what we've seen
                for _, sidx in seen_scores[:4]:
                    mask_top4_online[sidx] = True
                for _, sidx in seen_scores[4:]:
                    mask_top4_online[sidx] = False
    policies["Top-4/day (online)"] = ("No", mask_top4_online)

    # 3. Top-4/day oracle: best 4 after seeing all
    mask_top4_oracle = np.zeros(n, dtype=bool)
    for day, idxs in day_indices.items():
        scored = sorted(idxs, key=lambda i: comp[i], reverse=True)
        for idx in scored[:4]:
            mask_top4_oracle[idx] = True
    policies["Top-4/day (oracle)"] = ("Yes", mask_top4_oracle)

    # 4. Top-6/day online
    mask_top6_online = np.zeros(n, dtype=bool)
    for day, idxs in day_indices.items():
        seen = []
        for idx in idxs:
            seen.append((comp[idx], idx))
            seen.sort(reverse=True)
            for _, sidx in seen[:6]:
                mask_top6_online[sidx] = True
            for _, sidx in seen[6:]:
                mask_top6_online[sidx] = False
    policies["Top-6/day (online)"] = ("No", mask_top6_online)

    # 5. Top-25%/day oracle
    mask_top25_oracle = np.zeros(n, dtype=bool)
    for day, idxs in day_indices.items():
        scored = sorted(idxs, key=lambda i: comp[i], reverse=True)
        k = max(1, len(scored) // 4)
        for idx in scored[:k]:
            mask_top25_oracle[idx] = True
    policies["Top-25%/day (oracle)"] = ("Yes", mask_top25_oracle)

    # 6. Top-N global to match ~5 TPD
    target_n = int(5.0 * n_days)
    sorted_global = np.argsort(-comp)
    mask_topn = np.zeros(n, dtype=bool)
    mask_topn[sorted_global[:target_n]] = True
    policies["Top-N global (~5 TPD)"] = ("Yes", mask_topn)

    # 7. Competence + opportunity intersection
    policies["Comp>0 AND Opp>0"] = ("No", (comp > 0) & (opp > 0))

    # 8. Competence after bar 60
    policies["Comp>best_t AND bar>=60"] = ("No", (comp >= best_t) & (bar >= 60))

    # 9. Trivial: skip early dense
    policies["TRIVIAL: bar>=60 & nv<=55"] = ("No", (bar >= 60) & (nv <= 55))

    # 10. Trivial: skip early
    policies["TRIVIAL: bar>=60"] = ("No", bar >= 60)

    # 11. Baseline: all bars
    policies["ALL BARS (baseline)"] = ("No", np.ones(n, dtype=bool))

    hdr = (f"  {'Policy':<32s} {'LA':>3} {'Trades':>6} {'TPD':>5} {'PF':>7} "
           f"{'WR%':>6} {'AvgPnL':>8} {'AvgRnk':>7} {'AvgBar':>7} {'AvgNv':>6} {'Early%':>6} "
           f"{'Call%':>6} {'SideAc':>6}")
    print(hdr)
    print(f"  {'-'*32} {'-'*3} {'-'*6} {'-'*5} {'-'*7} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for pname, (lookahead, mask) in policies.items():
        m = _metrics_from_mask(records, mask)
        pf_s = f"{m['pf']:.3f}" if np.isfinite(m["pf"]) else "inf"
        print(f"  {pname:<32s} {lookahead:>3} {m['trades']:>6} {m['tpd']:>5.1f} {pf_s:>7} "
              f"{m['wr']*100:>5.1f}% {m['avg_pnl']:>7.4f} {m['avg_rank']:>7.1f} {m['avg_bar']:>7.1f} {m['avg_nv']:>5.1f} {m['early_pct']:>5.1f}% "
              f"{m['call_pct']:>5.1f}% {m['side_acc']:>5.1f}%")


# ── Q4: Side-bias decomposition ────────────────────────────────────

def q4_side_bias(records):
    print("\n" + "=" * 120)
    print("  Q4: SIDE-BIAS DECOMPOSITION")
    print("=" * 120)

    comp = np.array([r["comp_score"] for r in records])
    n = len(records)

    # Default gate: comp > 0
    pass_mask = comp > 0
    passed = [records[i] for i in range(n) if pass_mask[i]]

    print(f"\n  Gate: competence > 0 ({sum(pass_mask)} bars passed, {sum(pass_mask)/n*100:.1f}%)")
    if passed:
        chosen_calls = sum(1 for r in passed if not r["chosen_is_put"])
        oracle_calls = sum(1 for r in passed if not r["oracle_is_put"])
        side_acc = sum(1 for r in passed if r["side_correct"]) / len(passed) * 100
        print(f"    Model chosen side:  {chosen_calls} calls ({chosen_calls/len(passed)*100:.1f}%), "
              f"{len(passed)-chosen_calls} puts ({(len(passed)-chosen_calls)/len(passed)*100:.1f}%)")
        print(f"    Oracle side:        {oracle_calls} calls ({oracle_calls/len(passed)*100:.1f}%), "
              f"{len(passed)-oracle_calls} puts ({(len(passed)-oracle_calls)/len(passed)*100:.1f}%)")
        print(f"    Side accuracy:      {side_acc:.1f}%")

    # By competence score decile
    print(f"\n  Side distribution by competence-score decile:")
    sorted_idx = np.argsort(-comp)
    decile_size = n // 10
    print(f"    {'Decile':<10} {'Call%':>7} {'OrcCall%':>9} {'SideAcc':>8} {'n':>5}")
    print(f"    {'-'*10} {'-'*7} {'-'*9} {'-'*8} {'-'*5}")
    for d in range(10):
        s = d * decile_size
        e = (d + 1) * decile_size if d < 9 else n
        idx = sorted_idx[s:e]
        sub = [records[i] for i in idx]
        calls = sum(1 for r in sub if not r["chosen_is_put"])
        orc_calls = sum(1 for r in sub if not r["oracle_is_put"])
        sa = sum(1 for r in sub if r["side_correct"]) / len(sub) * 100
        print(f"    D{d+1:<9} {calls/len(sub)*100:>6.1f}% {orc_calls/len(sub)*100:>8.1f}% {sa:>7.1f}% {len(sub):>5}")

    # Full population comparison
    all_calls = sum(1 for r in records if not r["chosen_is_put"])
    all_orc_calls = sum(1 for r in records if not r["oracle_is_put"])
    all_sa = sum(1 for r in records if r["side_correct"]) / n * 100
    print(f"\n    Population: Call%={all_calls/n*100:.1f}% OrcCall%={all_orc_calls/n*100:.1f}% SideAcc={all_sa:.1f}%")


# ── Q5: FP/FN profiles ─────────────────────────────────────────────

def q5_fp_fn_profiles(records, context_matrix, feature_names):
    print("\n" + "=" * 120)
    print("  Q5: FALSE POSITIVE / FALSE NEGATIVE PROFILES")
    print("=" * 120)

    comp = np.array([r["comp_score"] for r in records])
    n = len(records)

    for label, top_pct, bot_pct in [("Broad (30%)", 70, 30), ("Sharp (10%)", 90, 10)]:
        top_thresh = np.percentile(comp, top_pct)
        bot_thresh = np.percentile(comp, bot_pct)

        fp_mask = (comp >= top_thresh) & np.array([not r["binary_win"] for r in records])
        fn_mask = (comp <= bot_thresh) & np.array([r["binary_win"] for r in records])

        fp_idx = np.where(fp_mask)[0]
        fn_idx = np.where(fn_mask)[0]

        print(f"\n  --- {label} ---")

        for name, idx_arr in [("FALSE POSITIVE (high score, bad trade)", fp_idx),
                               ("FALSE NEGATIVE (low score, good trade)", fn_idx)]:
            sub = [records[i] for i in idx_arr]
            if not sub:
                print(f"\n    {name}: 0 bars")
                continue

            print(f"\n    {name}: {len(sub)} bars")
            bars = [r["bar_of_day"] for r in sub]
            nvs = [r["n_valid"] for r in sub]
            ranks = [r["rank"] for r in sub]
            calls = sum(1 for r in sub if not r["chosen_is_put"])
            orc_calls = sum(1 for r in sub if not r["oracle_is_put"])
            sa = sum(1 for r in sub if r["side_correct"]) / len(sub) * 100

            print(f"      bar_of_day:  mean={np.mean(bars):.1f} median={np.median(bars):.0f}")
            print(f"      n_valid:     mean={np.mean(nvs):.1f} median={np.median(nvs):.0f}")
            print(f"      avg_rank:    {np.mean(ranks):.1f}")
            print(f"      call%:       {calls/len(sub)*100:.1f}%")
            print(f"      oracle_call%:{orc_calls/len(sub)*100:.1f}%")
            print(f"      side_acc:    {sa:.1f}%")

            # Time distribution
            early = sum(1 for b in bars if b < 60)
            mid = sum(1 for b in bars if 60 <= b < 150)
            late = sum(1 for b in bars if b >= 150)
            print(f"      time: early={early}({early/len(sub)*100:.0f}%) "
                  f"mid={mid}({mid/len(sub)*100:.0f}%) late={late}({late/len(sub)*100:.0f}%)")

            # Top-5 distinguishing features
            sub_ctx = context_matrix[idx_arr]
            full_means = np.nanmean(context_matrix, axis=0)
            full_stds = np.nanstd(context_matrix, axis=0)
            sub_means = np.nanmean(sub_ctx, axis=0)
            diffs = []
            for j, fname in enumerate(feature_names):
                if full_stds[j] > 1e-8:
                    z = (sub_means[j] - full_means[j]) / full_stds[j]
                    diffs.append((fname, z))
            diffs.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"      Top-5 distinguishing features (z-score):")
            for fname, z in diffs[:5]:
                print(f"        {fname:<28s} z={z:+.3f}")

            # Systematic vs scattered
            bar_std = np.std(bars)
            nv_std = np.std(nvs)
            pop_bar_std = np.std([r["bar_of_day"] for r in records])
            pop_nv_std = np.std([r["n_valid"] for r in records])
            concentrated = bar_std < pop_bar_std * 0.7 or nv_std < pop_nv_std * 0.7
            print(f"      Pattern: {'SYSTEMATIC (clustered)' if concentrated else 'SCATTERED (residual noise)'}")


# ── Decision table ──────────────────────────────────────────────────

def decision_table(records, lr_probs):
    print("\n" + "=" * 120)
    print("  DECISION TABLE")
    print("=" * 120)

    comp = np.array([r["comp_score"] for r in records])
    opp = np.array([r["opp_score"] for r in records])
    bar = np.array([r["bar_of_day"] for r in records])
    nv = np.array([r["n_valid"] for r in records])
    n = len(records)
    n_days = len(set(r["day"] for r in records))

    # Compute key metrics for each approach
    # 1. Best threshold for competence
    best_comp_pf, best_comp_t = 0, 0
    for pctile in range(50, 98):
        t = np.percentile(comp, pctile)
        mask = comp >= t
        if mask.sum() >= 200:
            pf = _pf([records[i]["chosen_pnl"] for i in range(n) if mask[i]])
            if np.isfinite(pf) and pf > best_comp_pf:
                best_comp_pf = pf
                best_comp_t = pctile

    # 2. Best threshold for LR proxy
    best_lr_pf, best_lr_t = 0, 0
    for pctile in range(50, 98):
        t = np.percentile(lr_probs, pctile)
        mask = lr_probs >= t
        if mask.sum() >= 200:
            pf = _pf([records[i]["chosen_pnl"] for i in range(n) if mask[i]])
            if np.isfinite(pf) and pf > best_lr_pf:
                best_lr_pf = pf
                best_lr_t = pctile

    # 3. Trivial filter
    trivial_mask = (bar >= 60) & (nv <= 55)
    trivial_pf = _pf([records[i]["chosen_pnl"] for i in range(n) if trivial_mask[i]])

    # 4. Top-4/day oracle
    day_indices = defaultdict(list)
    for i, r in enumerate(records):
        day_indices[r["day"]].append(i)
    top4_mask = np.zeros(n, dtype=bool)
    for day, idxs in day_indices.items():
        scored = sorted(idxs, key=lambda i: comp[i], reverse=True)
        for idx in scored[:4]:
            top4_mask[idx] = True
    top4_pf = _pf([records[i]["chosen_pnl"] for i in range(n) if top4_mask[i]])

    print(f"\n  Summary of best approaches:")
    print(f"    Competence threshold (p{best_comp_t}): PF={best_comp_pf:.3f}")
    print(f"    LR proxy threshold (p{best_lr_t}):     PF={best_lr_pf:.3f}")
    trivial_s = f"{trivial_pf:.3f}" if np.isfinite(trivial_pf) else "inf"
    top4_s = f"{top4_pf:.3f}" if np.isfinite(top4_pf) else "inf"
    print(f"    Trivial (bar>=60 & nv<=55):        PF={trivial_s}")
    print(f"    Top-4/day oracle (competence):      PF={top4_s}")

    # Spearman for ordering quality
    cpnl = np.array([r["chosen_pnl"] for r in records])
    rho_comp = sp_stats.spearmanr(comp, cpnl).correlation
    rho_lr = sp_stats.spearmanr(lr_probs, cpnl).correlation
    rho_opp = sp_stats.spearmanr(opp, cpnl).correlation

    print(f"\n  Ordering quality (Spearman ρ with chosen PnL):")
    print(f"    Competence: {rho_comp:+.4f}")
    print(f"    LR proxy:   {rho_lr:+.4f}")
    print(f"    Opportunity: {rho_opp:+.4f}")

    print(f"\n  VERDICT:")
    if best_comp_pf > trivial_pf + 0.02 and best_comp_pf > best_lr_pf:
        print(f"  → Competence score beats both trivial filter and LR proxy")
        if rho_comp > 0.02:
            print(f"  → Score has real ordering value (ρ={rho_comp:+.4f})")
            print(f"  → Recommended: threshold or top-k policy using competence score")
        else:
            print(f"  → But ordering is weak — may be threshold-only, not ranking")
    elif best_lr_pf > best_comp_pf and best_lr_pf > trivial_pf + 0.02:
        print(f"  → LR proxy beats competence head — head not worth complexity")
        print(f"  → Recommended: use LR proxy or retrain head with better architecture")
    elif trivial_pf >= best_comp_pf - 0.02 and trivial_pf >= best_lr_pf - 0.02:
        print(f"  → Trivial filter matches or beats learned scores")
        print(f"  → Recommended: use simple regime filter, abandon head")
    else:
        print(f"  → Mixed results — no clear winner")
        print(f"  → Consider regime-conditioned competence or different policy form")


# ── Main ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="exp_167b competence model")
    parser.add_argument("--baseline", required=True, help="exp_165 baseline model")
    args = parser.parse_args()

    print("Loading models and running inference...")
    records, ctx_matrix, feat_names = load_two_model_inference(args.model, args.baseline)
    print(f"  {len(records)} eligible bars with finite chosen PnL")

    print("Fitting LR proxy...")
    lr_probs = fit_lr_proxy(records, ctx_matrix)

    q1_score_ordering(records, lr_probs)
    q2_threshold_sweep(records, lr_probs)
    q3_policy_comparison(records, lr_probs)
    q4_side_bias(records)
    q5_fp_fn_profiles(records, ctx_matrix, feat_names)
    decision_table(records, lr_probs)


if __name__ == "__main__":
    main()
