"""Investigate why high-confidence opportunity_logit bars are toxic.

Answers 5 specific questions:
1. Is opportunity_logit a proxy for contract density / time-of-day?
2. Where does the failure happen on toxic bars (gate, side, or strike)?
3. What features distinguish toxic vs healthy bars?
4. Does constraining the candidate set fix ranking on toxic bars?
5. What is the precise toxic time window?

Usage:
    python3 -m v2.analysis.toxicity_investigation [--model path]
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch

from v2.core.chain_data import (
    CONTRACT_FEATURE_FIELDS,
    QUALITY_PARTIAL,
    describe_contract,
    load_sidecar_cached,
    padded_snapshot,
)
from v2.core.policy import DecisionPolicy
from v2.replay import load_model_from_path
from v2.train import TradingModel


def load_all_bars(model_path: str):
    """Load model, run inference on promote mask, return per-bar records."""
    model = load_model_from_path(model_path)
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    policy = DecisionPolicy()

    features = data["X"].numpy()
    feature_names = data.get("feature_names", [])
    mask = data["promote_mask"].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    spot_prices = data["spot_prices"].numpy()
    sidecar_dir = "v2/data_sidecars"
    max_contracts = data.get("metadata", {}).get("max_contracts_per_bar", 285)

    # Feature index lookup
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
    all_contract_indices = np.stack([s[2] for s in snapshots]).astype(np.int32)

    model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(all_windows).float()
        batch_c = torch.from_numpy(all_contracts).float()
        all_outputs = model(batch_x, batch_c)
        all_outputs = {k: v.cpu().numpy() for k, v in all_outputs.items()}

    records = []
    for i, (day, global_bar, local_bar) in enumerate(eligible):
        opp_logit = float(all_outputs["opportunity_logit"][i])
        side_logit = float(all_outputs["side_logit"][i])
        c_scores = all_outputs["contract_scores"][i]
        v_mask = all_outputs["valid_mask"][i].astype(bool)
        oracle_labels = all_contract_labels[i]
        contract_feats = all_contracts[i]  # (max_contracts, 22)
        contract_idxs = all_contract_indices[i]

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

        # Model chosen contract details
        chosen_pnl = float(oracle_labels[best_row]) if best_row >= 0 and np.isfinite(oracle_labels[best_row]) else float("nan")
        oracle_best_pnl = float(oracle_labels[oracle_best_row]) if oracle_best_row >= 0 and np.isfinite(oracle_labels[oracle_best_row]) else float("nan")

        # Rank of chosen contract
        if best_row >= 0 and np.isfinite(oracle_labels[best_row]):
            valid_oracle = oracle_labels[v_mask & np.isfinite(oracle_labels)]
            rank = int(np.sum(valid_oracle > chosen_pnl)) + 1
        else:
            rank = -1

        # Contract features of chosen contract
        if best_row >= 0:
            cf = contract_feats[best_row]
            chosen_right_is_put = cf[2] > 0.5
            chosen_strike = cf[1]
            chosen_moneyness = cf[11]  # moneyness_pct
            chosen_spread = cf[4]  # spread_fraction
            chosen_mid = cf[3]
        else:
            chosen_right_is_put = False
            chosen_strike = chosen_moneyness = chosen_spread = chosen_mid = float("nan")

        # Oracle contract features
        if oracle_best_row >= 0:
            ocf = contract_feats[oracle_best_row]
            oracle_right_is_put = ocf[2] > 0.5
            oracle_moneyness = ocf[11]
        else:
            oracle_right_is_put = False
            oracle_moneyness = float("nan")

        # Side accuracy: did model pick same side as oracle?
        side_correct = (chosen_right_is_put == oracle_right_is_put) if best_row >= 0 and oracle_best_row >= 0 else False

        # Context features
        ctx = features[global_bar]
        ctx_dict = {}
        for fname in ["atm_iv", "vix_regime", "option_spread_pct", "log_chain_volume",
                       "log_near_transactions", "intraday_phase", "intraday_sin",
                       "bar_range", "session_range_pct", "vrp"]:
            if fname in feat_idx:
                ctx_dict[fname] = float(ctx[feat_idx[fname]])

        # Counterfactual: near-ATM only ranking
        near_atm_mask = v_mask.copy()
        for r in range(len(near_atm_mask)):
            if near_atm_mask[r] and abs(contract_feats[r, 11]) > 2.0:  # moneyness > 2%
                near_atm_mask[r] = False
        atm_scores = c_scores.copy()
        atm_scores[~near_atm_mask] = -1e9
        atm_best_row = int(np.argmax(atm_scores)) if near_atm_mask.any() else -1
        atm_chosen_pnl = float(oracle_labels[atm_best_row]) if atm_best_row >= 0 and np.isfinite(oracle_labels[atm_best_row]) else float("nan")
        n_atm = int(near_atm_mask.sum())

        # Counterfactual: capped candidate count (top 20 by score)
        if n_valid > 20:
            top20_idx = np.argsort(c_scores)[-20:]
            cap_mask = np.zeros_like(v_mask)
            for idx in top20_idx:
                if v_mask[idx]:
                    cap_mask[idx] = True
            cap_scores = c_scores.copy()
            cap_scores[~cap_mask] = -1e9
            cap_best_row = int(np.argmax(cap_scores)) if cap_mask.any() else -1
            cap_chosen_pnl = float(oracle_labels[cap_best_row]) if cap_best_row >= 0 and np.isfinite(oracle_labels[cap_best_row]) else float("nan")
        else:
            cap_chosen_pnl = chosen_pnl

        records.append({
            "day": day,
            "bar_of_day": local_bar,
            "opp_logit": opp_logit,
            "side_logit": side_logit,
            "n_valid": n_valid,
            "chosen_pnl": chosen_pnl,
            "oracle_best_pnl": oracle_best_pnl,
            "rank": rank,
            "side_correct": side_correct,
            "chosen_moneyness": chosen_moneyness,
            "chosen_spread": chosen_spread,
            "oracle_moneyness": oracle_moneyness,
            "atm_chosen_pnl": atm_chosen_pnl,
            "n_atm": n_atm,
            "cap_chosen_pnl": cap_chosen_pnl,
            **ctx_dict,
        })

    return records


def q1_proxy_analysis(records):
    """Q1: Is opportunity_logit a proxy for contract density / time-of-day?"""
    print("\n" + "=" * 90)
    print("  Q1: IS OPPORTUNITY_LOGIT A PROXY FOR DENSITY / TIME-OF-DAY?")
    print("=" * 90)

    opp = np.array([r["opp_logit"] for r in records])
    n_valid = np.array([r["n_valid"] for r in records])
    bar = np.array([r["bar_of_day"] for r in records])

    # Correlations
    r_nvalid = np.corrcoef(opp, n_valid)[0, 1]
    r_bar = np.corrcoef(opp, bar)[0, 1]
    r_nvalid_bar = np.corrcoef(n_valid, bar)[0, 1]

    print(f"\n  Pearson correlations:")
    print(f"    opportunity_logit vs n_valid_contracts:  r = {r_nvalid:+.4f}")
    print(f"    opportunity_logit vs bar_of_day:         r = {r_bar:+.4f}")
    print(f"    n_valid_contracts vs bar_of_day:         r = {r_nvalid_bar:+.4f}")

    # Partial correlation: opp_logit vs n_valid, controlling for bar_of_day
    from numpy.linalg import lstsq
    # Residualize opp on bar
    A = np.column_stack([bar, np.ones(len(bar))])
    opp_resid = opp - A @ lstsq(A, opp, rcond=None)[0]
    nv_resid = n_valid - A @ lstsq(A, n_valid, rcond=None)[0]
    r_partial = np.corrcoef(opp_resid, nv_resid)[0, 1]
    print(f"    opportunity_logit vs n_valid (controlling for bar_of_day): r = {r_partial:+.4f}")

    # Bar residualized
    bar_resid = bar - np.column_stack([n_valid, np.ones(len(n_valid))]) @ lstsq(
        np.column_stack([n_valid, np.ones(len(n_valid))]), bar, rcond=None)[0]
    opp_resid2 = opp - np.column_stack([n_valid, np.ones(len(n_valid))]) @ lstsq(
        np.column_stack([n_valid, np.ones(len(n_valid))]), opp, rcond=None)[0]
    r_partial2 = np.corrcoef(opp_resid2, bar_resid)[0, 1]
    print(f"    opportunity_logit vs bar_of_day (controlling for n_valid): r = {r_partial2:+.4f}")

    # Interaction: bin by (bar_bucket, density_bucket) and show avg opp_logit
    print(f"\n  Interaction table: avg opportunity_logit by (bar_bucket x density_bucket)")
    bar_cuts = [(30, 60, "30-59"), (60, 120, "60-119"), (120, 180, "120-179"), (180, 270, "180-269")]
    density_cuts = [(0, 30, "<30"), (30, 60, "30-59"), (60, 100, "60-99"), (100, 300, "100+")]
    print(f"    {'':>12}", end="")
    for _, _, dl in density_cuts:
        print(f" {dl:>10}", end="")
    print()
    for blo, bhi, bl in bar_cuts:
        print(f"    {bl:>12}", end="")
        for dlo, dhi, dl in density_cuts:
            subset = [r for r in records if blo <= r["bar_of_day"] < bhi and dlo <= r["n_valid"] < dhi]
            if subset:
                avg = np.mean([r["opp_logit"] for r in subset])
                print(f" {avg:>10.4f}", end="")
            else:
                print(f" {'—':>10}", end="")
        print()


def q2_failure_localization(records):
    """Q2: Where does the failure happen on toxic bars?"""
    print("\n" + "=" * 90)
    print("  Q2: WHERE DOES THE FAILURE HAPPEN ON TOXIC BARS?")
    print("=" * 90)

    records_sorted = sorted(records, key=lambda r: r["opp_logit"], reverse=True)
    n = len(records_sorted)

    buckets = [
        ("Top 10% (toxic)", records_sorted[:n // 10]),
        ("10-25%", records_sorted[n // 10:n // 4]),
        ("25-50% (healthy)", records_sorted[n // 4:n // 2]),
    ]

    print(f"\n  {'Bucket':<22} {'OracBest':>9} {'ModelPnL':>9} {'Rank':>6} {'SideAcc':>8} {'|Moneyness|':>12} {'Spread':>8} {'nValid':>7}")
    print("  " + "-" * 85)

    for label, subset in buckets:
        oracle_pnls = [r["oracle_best_pnl"] for r in subset if np.isfinite(r["oracle_best_pnl"])]
        chosen_pnls = [r["chosen_pnl"] for r in subset if np.isfinite(r["chosen_pnl"])]
        ranks = [r["rank"] for r in subset if r["rank"] > 0]
        side_acc = [r["side_correct"] for r in subset]
        moneyness = [abs(r["chosen_moneyness"]) for r in subset if np.isfinite(r["chosen_moneyness"])]
        spreads = [r["chosen_spread"] for r in subset if np.isfinite(r["chosen_spread"])]
        n_valids = [r["n_valid"] for r in subset]

        print(f"  {label:<22} "
              f"{np.mean(oracle_pnls):>8.4f} "
              f"{np.mean(chosen_pnls):>8.4f} "
              f"{np.mean(ranks):>6.1f} "
              f"{np.mean(side_acc)*100:>7.1f}% "
              f"{np.mean(moneyness):>11.2f}% "
              f"{np.mean(spreads):>7.3f} "
              f"{np.mean(n_valids):>7.1f}")

    # Decompose the PnL gap
    print(f"\n  PnL gap decomposition (oracle_best - model_chosen):")
    for label, subset in buckets:
        gaps = [r["oracle_best_pnl"] - r["chosen_pnl"]
                for r in subset if np.isfinite(r["oracle_best_pnl"]) and np.isfinite(r["chosen_pnl"])]
        print(f"    {label:<22} avg gap = {np.mean(gaps):>8.4f}  "
              f"(model leaves {np.mean(gaps)*100:.1f}% on the table)")


def q3_feature_comparison(records):
    """Q3: What features distinguish toxic vs healthy bars?"""
    print("\n" + "=" * 90)
    print("  Q3: FEATURE COMPARISON — TOXIC (TOP 10%) vs HEALTHY (25-50%)")
    print("=" * 90)

    records_sorted = sorted(records, key=lambda r: r["opp_logit"], reverse=True)
    n = len(records_sorted)
    toxic = records_sorted[:n // 10]
    healthy = records_sorted[n // 4:n // 2]

    features_to_compare = [
        "bar_of_day", "n_valid", "option_spread_pct", "log_chain_volume",
        "log_near_transactions", "atm_iv", "vix_regime", "intraday_phase",
        "session_range_pct", "bar_range", "vrp",
    ]

    print(f"\n  {'Feature':<25} {'Toxic Mean':>12} {'Healthy Mean':>14} {'Delta':>10} {'Direction':>12}")
    print("  " + "-" * 75)

    for feat in features_to_compare:
        t_vals = [r[feat] for r in toxic if feat in r and np.isfinite(r.get(feat, float("nan")))]
        h_vals = [r[feat] for r in healthy if feat in r and np.isfinite(r.get(feat, float("nan")))]
        if not t_vals or not h_vals:
            continue
        t_mean = np.mean(t_vals)
        h_mean = np.mean(h_vals)
        delta = t_mean - h_mean
        direction = "toxic higher" if delta > 0.01 else "toxic lower" if delta < -0.01 else "~same"
        print(f"  {feat:<25} {t_mean:>12.4f} {h_mean:>14.4f} {delta:>+10.4f} {direction:>12}")


def q4_counterfactual_ranking(records):
    """Q4: Does constraining the candidate set fix ranking on toxic bars?"""
    print("\n" + "=" * 90)
    print("  Q4: COUNTERFACTUAL — DOES CONSTRAINING CANDIDATE SET FIX RANKING?")
    print("=" * 90)

    records_sorted = sorted(records, key=lambda r: r["opp_logit"], reverse=True)
    n = len(records_sorted)

    buckets = [
        ("Top 10% (toxic)", records_sorted[:n // 10]),
        ("25-50% (healthy)", records_sorted[n // 4:n // 2]),
    ]

    print(f"\n  {'Bucket':<22} {'Variant':<18} {'AvgPnL':>9} {'WR%':>7} {'PF':>7} {'nContracts':>11}")
    print("  " + "-" * 78)

    for label, subset in buckets:
        for variant_name, pnl_key, count_key in [
            ("Full set", "chosen_pnl", "n_valid"),
            ("Near-ATM only", "atm_chosen_pnl", "n_atm"),
            ("Top-20 capped", "cap_chosen_pnl", "n_valid"),
        ]:
            pnls = [r[pnl_key] for r in subset if np.isfinite(r.get(pnl_key, float("nan")))]
            if not pnls:
                continue
            avg = np.mean(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            gw = sum(p for p in pnls if p > 0)
            gl = abs(sum(p for p in pnls if p <= 0))
            pf = gw / gl if gl > 0 else 0
            n_c = np.mean([r.get(count_key, 0) for r in subset])
            print(f"  {label:<22} {variant_name:<18} {avg:>8.4f} {wr:>6.1f}% {pf:>7.3f} {n_c:>11.1f}")
        print()


def q5_precise_toxic_window(records):
    """Q5: Localize the toxic time window precisely."""
    print("\n" + "=" * 90)
    print("  Q5: PRECISE TOXIC TIME WINDOW")
    print("=" * 90)

    records_sorted = sorted(records, key=lambda r: r["opp_logit"], reverse=True)
    n = len(records_sorted)
    top10 = set(id(r) for r in records_sorted[:n // 10])

    # Bin by 10-bar windows
    bar_buckets = defaultdict(lambda: {"total": 0, "toxic": 0, "chosen_pnls": [], "toxic_pnls": []})
    for r in records:
        bucket = (r["bar_of_day"] // 10) * 10
        bar_buckets[bucket]["total"] += 1
        pnl = r["chosen_pnl"] if np.isfinite(r.get("chosen_pnl", float("nan"))) else None
        if pnl is not None:
            bar_buckets[bucket]["chosen_pnls"].append(pnl)
        if id(r) in top10:
            bar_buckets[bucket]["toxic"] += 1
            if pnl is not None:
                bar_buckets[bucket]["toxic_pnls"].append(pnl)

    print(f"\n  {'Bar Window':<12} {'Total':>6} {'#Toxic':>7} {'%Toxic':>7} "
          f"{'AllPnL':>8} {'ToxicPnL':>9} {'ToxicWR':>8} {'ToxicPF':>8}")
    print("  " + "-" * 75)

    for bucket in sorted(bar_buckets.keys()):
        d = bar_buckets[bucket]
        pct_toxic = d["toxic"] / d["total"] * 100 if d["total"] > 0 else 0
        all_avg = np.mean(d["chosen_pnls"]) if d["chosen_pnls"] else 0
        toxic_pnls = d["toxic_pnls"]
        if toxic_pnls:
            toxic_avg = np.mean(toxic_pnls)
            toxic_wr = sum(1 for p in toxic_pnls if p > 0) / len(toxic_pnls) * 100
            gw = sum(p for p in toxic_pnls if p > 0)
            gl = abs(sum(p for p in toxic_pnls if p <= 0))
            toxic_pf = gw / gl if gl > 0 else 0
        else:
            toxic_avg = toxic_wr = toxic_pf = 0

        flag = " ◀ TOXIC" if pct_toxic > 15 and d["toxic"] > 10 else ""
        print(f"  {bucket:>3}-{bucket+9:<8} {d['total']:>6} {d['toxic']:>7} {pct_toxic:>6.1f}% "
              f"{all_avg:>7.4f} {toxic_avg:>8.4f} {toxic_wr:>7.1f}% {toxic_pf:>8.3f}{flag}")

    # Concentration stat
    total_toxic = sum(d["toxic"] for d in bar_buckets.values())
    early_toxic = sum(d["toxic"] for b, d in bar_buckets.items() if b < 80)
    print(f"\n  Toxic bars in bars 30-79: {early_toxic}/{total_toxic} ({early_toxic/total_toxic*100:.1f}%)")
    mid_toxic = sum(d["toxic"] for b, d in bar_buckets.items() if 80 <= b < 160)
    print(f"  Toxic bars in bars 80-159: {mid_toxic}/{total_toxic} ({mid_toxic/total_toxic*100:.1f}%)")
    late_toxic = sum(d["toxic"] for b, d in bar_buckets.items() if b >= 160)
    print(f"  Toxic bars in bars 160+: {late_toxic}/{total_toxic} ({late_toxic/total_toxic*100:.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="v2/models/model.pt")
    args = parser.parse_args()

    print("Loading and running inference on all promote bars...")
    records = load_all_bars(args.model)
    print(f"  {len(records)} eligible bars loaded\n")

    q1_proxy_analysis(records)
    q2_failure_localization(records)
    q3_feature_comparison(records)
    q4_counterfactual_ranking(records)
    q5_precise_toxic_window(records)

    # Final synthesis
    print("\n" + "=" * 90)
    print("  SYNTHESIS")
    print("=" * 90)
    print("  Review the above to determine:")
    print("  1. Is opp_logit mainly driven by n_valid / bar_of_day? (Q1 correlations)")
    print("  2. Is the failure in gate, side, or strike selection? (Q2 rank/side/moneyness)")
    print("  3. What features does the model mistake for opportunity? (Q3 deltas)")
    print("  4. Does search-space overload cause ranking collapse? (Q4 counterfactuals)")
    print("  5. How concentrated is the toxic window? (Q5 bar breakdown)")


if __name__ == "__main__":
    main()
