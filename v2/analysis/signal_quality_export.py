"""Export comprehensive signal-quality tables for trader review.

Produces 5 CSV/tables that distinguish:
  H1: Model signals are wrong, overtrading is a symptom
  H2: Model has real confidence/order-quality signal, policy over-expresses it

Usage:
    python3 -m v2.analysis.signal_quality_export [--model path]

Outputs to v2/output/signal_quality/:
    1_per_trade.csv           — one row per trade opportunity
    2_day_decomposition.csv   — one row per trading day
    3_confidence_calibration.txt — bucketed by opportunity_logit
    4_trade_number.txt        — by position within day
    5_blocked_analysis.txt    — max-4 blocked vs taken comparison
"""
from __future__ import annotations

import csv
import dataclasses
import os
import random as _random
from collections import defaultdict

import numpy as np
import torch

from v2.core.chain_data import (
    QUALITY_PARTIAL,
    describe_contract,
    extract_contract_series,
    load_sidecar_cached,
    padded_snapshot,
)
from v2.core.metrics import compute_metrics
from v2.core.policy import DecisionPolicy
from v2.core.schema import TradeIntent
from v2.core.simulator import simulate_trade
from v2.replay import load_model_from_path
from v2.train import SIDE_W, TradingModel

OUTPUT_DIR = "v2/output/signal_quality"


def _dollar_pnl(t, policy: DecisionPolicy) -> float:
    return t.net_pnl_pct * t.entry_price * policy.contract_multiplier * policy.qty


def run_full_replay_with_diagnostics(
    model: TradingModel,
    data: dict,
    policy: DecisionPolicy,
) -> tuple[list[dict], list]:
    """Run replay capturing every decision point, not just trades.

    Returns:
        rows: list of dicts, one per eligible bar (trade or skip)
        trades: list of SimulatedTrade for taken trades
    """
    from v2.core.features import _FEAT_IDX

    features = data["X"].numpy()
    sim_features = data["X_sim"].numpy() if "X_sim" in data else features
    mask = data["promote_mask"].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    spot_prices = data["spot_prices"].numpy()
    sidecar_dir = data.get("metadata", {}).get("sidecar_dir", "v2/data_sidecars")
    if isinstance(sidecar_dir, torch.Tensor):
        sidecar_dir = "v2/data_sidecars"

    # Build eligible bars
    eligible = []
    for idx in range(len(mask)):
        if not mask[idx]:
            continue
        day = dates[idx] if isinstance(dates[idx], str) else dates[idx]
        bod = int(bar_of_day[idx])
        if bod < policy.no_trade_before_bar or bod >= policy.no_trade_after_bar:
            continue
        eligible.append((day, idx, bod))

    # Day-to-bars mapping for simulator
    day_to_bars = defaultdict(list)
    for idx in range(len(dates)):
        day_to_bars[dates[idx]].append(idx)

    # Batch inference
    lookback = 30
    n_feat = features.shape[1]
    gather_idx = np.array([
        np.arange(max(0, g - lookback + 1), g + 1) for _, g, _ in eligible
    ])
    for i, (_, g, _) in enumerate(eligible):
        row = gather_idx[i]
        if len(row) < lookback:
            pad = np.full(lookback - len(row), row[0])
            gather_idx[i] = np.concatenate([pad, row])

    all_windows = features[np.stack(gather_idx)]

    max_contracts = data.get("metadata", {}).get("max_contracts_per_bar", 285)
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
        all_outputs = {k: v.cpu() for k, v in all_outputs.items()}

    vix_idx = _FEAT_IDX.get("vix_regime", 14)

    # Replay loop with full recording
    rows = []
    trades = []
    current_day = None
    in_trade = False
    trade_exit_bar = -1
    last_stop_bar = -policy.cooldown_bars - 1
    daily_dollar_pnl = 0.0
    daily_trades = 0
    trade_number_in_day = 0

    for i, (day, global_bar, local_bar) in enumerate(eligible):
        if day != current_day:
            current_day = day
            in_trade = False
            trade_exit_bar = -1
            last_stop_bar = -policy.cooldown_bars - 1
            daily_dollar_pnl = 0.0
            daily_trades = 0
            trade_number_in_day = 0

        outputs_i = {k: v[i] for k, v in all_outputs.items()}
        c_scores_np = outputs_i["contract_scores"].numpy()
        v_mask_np = outputs_i["valid_mask"].numpy().astype(bool)
        vix_val = float(features[global_bar, vix_idx]) if global_bar < len(features) else 0.0

        # Extract model signals
        opp_logit = float(outputs_i["opportunity_logit"].item()) if "opportunity_logit" in outputs_i else 0.0
        side_logit_val = float(outputs_i["side_logit"].item()) if "side_logit" in outputs_i else 0.0
        # Contract ranking — use effective_inference_scores for consistency with replay
        from v2.replay import effective_inference_scores
        eff_scores = effective_inference_scores(
            c_scores_np, v_mask_np, all_contracts[i],
            side_logit=side_logit_val,
            side_mode=policy.side_mode,
            alpha_side=policy.alpha_side,
        )
        n_valid = int(v_mask_np.sum())
        best_model_row = int(np.argmax(eff_scores)) if v_mask_np.any() else -1
        best_model_score = float(eff_scores[best_model_row]) if best_model_row >= 0 and eff_scores[best_model_row] > -1e8 else float("nan")

        # Oracle best
        oracle_labels = all_contract_labels[i].copy()
        oracle_valid = oracle_labels.copy()
        oracle_valid[~v_mask_np] = -1e9
        oracle_valid[~np.isfinite(oracle_valid)] = -1e9
        oracle_best_row = int(np.argmax(oracle_valid)) if v_mask_np.any() else -1
        oracle_best_pnl = float(oracle_labels[oracle_best_row]) if oracle_best_row >= 0 and np.isfinite(oracle_labels[oracle_best_row]) else float("nan")

        # Oracle contract info
        sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        oracle_idx = int(all_contract_indices[i][oracle_best_row]) if oracle_best_row >= 0 else -1
        if oracle_idx >= 0:
            oracle_contract = describe_contract(sidecar, oracle_idx)
            oracle_strike = oracle_contract.strike
            oracle_right = oracle_contract.right
        else:
            oracle_strike = float("nan")
            oracle_right = ""

        # Rank of model's chosen contract among oracle rankings
        if best_model_row >= 0 and np.isfinite(oracle_labels[best_model_row]):
            chosen_oracle_pnl = float(oracle_labels[best_model_row])
            # Rank: how many valid contracts have higher oracle PnL
            valid_oracle = oracle_labels[v_mask_np & np.isfinite(oracle_labels)]
            rank = int(np.sum(valid_oracle > chosen_oracle_pnl)) + 1
        else:
            chosen_oracle_pnl = float("nan")
            rank = -1

        # Chosen contract info
        if best_model_row >= 0:
            chosen_idx = int(all_contract_indices[i][best_model_row])
            chosen_contract = describe_contract(sidecar, chosen_idx)
            chosen_strike = chosen_contract.strike
            chosen_right = chosen_contract.right
            contract_features_t = torch.from_numpy(all_contracts[i])
            is_put = float(contract_features_t[best_model_row, 2]) > 0.5
            predicted_side = "P" if is_put else "C"
        else:
            chosen_idx = -1
            chosen_strike = float("nan")
            chosen_right = ""
            predicted_side = ""

        # Determine skip reason
        skip_reason = ""
        trade_taken = False
        trade_pnl = float("nan")
        exit_reason = ""

        if in_trade and global_bar <= trade_exit_bar:
            skip_reason = "in_position"
        elif global_bar - last_stop_bar < policy.cooldown_bars:
            skip_reason = "cooldown"
        elif opp_logit <= policy.gate_threshold:
            skip_reason = "gate_reject"
        elif best_model_row < 0:
            skip_reason = "no_valid_contract"
        elif not np.isfinite(oracle_labels[best_model_row]):
            skip_reason = "unexecutable"
        else:
            # Would trade — simulate
            contract_features_t = torch.from_numpy(all_contracts[i])
            contract_indices_t = torch.from_numpy(all_contract_indices[i])
            intent_policy = policy
            from v2.train import SIDE_W as _side_w
            intent = _build_intent(
                outputs_i, all_contract_labels[i], all_contracts[i],
                all_contract_indices[i], sidecar, local_bar,
                float(spot_prices[global_bar]), intent_policy, _side_w,
            )
            if intent is None or not intent.trade:
                skip_reason = "gate_or_quality"
            else:
                # Check max_daily_trades overlay
                if daily_trades >= 4:
                    skip_reason = "max_daily_trades_4"
                else:
                    # Simulate trade
                    series = extract_contract_series(sidecar, intent.contract_index)["mid"].astype(np.float32)
                    trade = simulate_trade(
                        intent=intent,
                        option_prices=series,
                        features=sim_features[day_to_bars[day]],
                        bar_of_day=np.arange(len(day_to_bars[day]), dtype=np.int32),
                        dates=[day] * len(day_to_bars[day]),
                        global_entry_bar=local_bar,
                        breakeven_trigger_pct=policy.breakeven_trigger_pct,
                        extra_trailing_tiers=policy.extra_trailing_tiers,
                    )
                    if trade is None:
                        skip_reason = "fill_failed"
                    else:
                        trade_taken = True
                        trade.trade_date = day
                        trades.append(trade)
                        trade_pnl = trade.net_pnl_pct
                        exit_reason = trade.exit_reason
                        dollar_pnl = _dollar_pnl(trade, policy)
                        daily_dollar_pnl += dollar_pnl
                        in_trade = True
                        trade_exit_bar = global_bar + (trade.exit_bar - local_bar)
                        if trade.exit_reason == "STOP_LOSS":
                            last_stop_bar = trade_exit_bar
                        daily_trades += 1
                        trade_number_in_day += 1

        cum_before = daily_dollar_pnl - (_dollar_pnl(trades[-1], policy) if trade_taken else 0)
        cum_after = daily_dollar_pnl

        # Model's chosen contract oracle PnL (what would have happened if we took the model's pick)
        chosen_oracle_pnl = float("nan")
        if best_model_row >= 0 and np.isfinite(oracle_labels[best_model_row]):
            chosen_oracle_pnl = float(oracle_labels[best_model_row])

        row = {
            "date": day,
            "bar_of_day": local_bar,
            "trade_taken": int(trade_taken),
            "blocked_reason": skip_reason,
            "trade_number_within_day": trade_number_in_day if trade_taken else 0,
            "predicted_opportunity_logit": round(opp_logit, 6),
            "predicted_side_logit": round(side_logit_val, 6),
            "predicted_side": predicted_side,
            "chosen_contract_idx": chosen_idx,
            "chosen_strike": chosen_strike,
            "chosen_right": chosen_right,
            "chosen_contract_score": round(best_model_score, 6) if np.isfinite(best_model_score) else "",
            "chosen_trade_pnl": round(trade_pnl, 6) if np.isfinite(trade_pnl) else "",
            "chosen_oracle_pnl": round(chosen_oracle_pnl, 6) if np.isfinite(chosen_oracle_pnl) else "",
            "exit_reason": exit_reason,
            "oracle_best_idx": oracle_idx,
            "oracle_best_strike": oracle_strike,
            "oracle_best_right": oracle_right,
            "oracle_best_pnl": round(oracle_best_pnl, 6) if np.isfinite(oracle_best_pnl) else "",
            "rank_of_chosen_contract": rank,
            "n_valid_contracts": n_valid,
            "vix_regime": round(vix_val, 4),
            "cumulative_day_pnl_before_trade": round(cum_before, 2),
            "cumulative_day_pnl_after_trade": round(cum_after, 2),
        }
        rows.append(row)

    return rows, trades


def _build_intent(outputs_i, contract_labels, contracts, contract_indices,
                  sidecar, local_bar, spot_price, policy, side_w):
    """Build trade intent using model_to_intent logic."""
    from v2.replay import model_to_intent
    return model_to_intent(
        gate_logit=outputs_i.get("opportunity_logit"),
        side_logit=outputs_i.get("side_logit"),
        contract_scores=outputs_i["contract_scores"],
        contract_labels=torch.from_numpy(contract_labels),
        valid_mask=outputs_i["valid_mask"],
        contract_features=torch.from_numpy(contracts),
        contract_indices=torch.from_numpy(contract_indices),
        sidecar=sidecar,
        local_bar=local_bar,
        spot_price=spot_price,
        policy=policy,
    )


# ── Table 1: Per-trade CSV ──────────────────────────────────────────
def write_per_trade_csv(rows: list[dict], path: str):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Table 1: {path} ({len(rows)} rows)")


# ── Table 2: Day-level decomposition ────────────────────────────────
def write_day_decomposition(rows: list[dict], trades: list, policy: DecisionPolicy, path: str):
    day_trades = defaultdict(list)
    for t in trades:
        day_trades[t.trade_date].append(t)

    all_days = sorted(set(r["date"] for r in rows))
    out_rows = []

    for day in all_days:
        ts = day_trades.get(day, [])
        pnls = [_dollar_pnl(t, policy) for t in ts]
        n = len(pnls)
        cum = np.cumsum(pnls) if pnls else np.array([])

        # Max intraday DD
        if len(cum) > 0:
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            max_intraday_dd = float(dd.max())
        else:
            max_intraday_dd = 0.0

        pnl_first = pnls[0] if n >= 1 else 0.0
        pnl_first2 = sum(pnls[:2]) if n >= 2 else sum(pnls)
        pnl_first4 = sum(pnls[:4]) if n >= 4 else sum(pnls)
        pnl_later = sum(pnls[4:]) if n > 4 else 0.0
        total = sum(pnls)

        # Percent of loss after trade 4
        if total < 0 and pnl_later < 0:
            pct_loss_after_4 = abs(pnl_later) / abs(total) * 100
        else:
            pct_loss_after_4 = 0.0

        # Percent of loss after first losing trade
        first_loss_idx = next((i for i, p in enumerate(pnls) if p < 0), None)
        if first_loss_idx is not None and total < 0:
            loss_after = sum(p for p in pnls[first_loss_idx + 1:] if p < 0)
            pct_loss_after_first = abs(loss_after) / abs(total) * 100 if total < 0 else 0.0
        else:
            pct_loss_after_first = 0.0

        # Recovery check
        if n >= 2 and pnls[0] < 0:
            recovered = total >= 0
        else:
            recovered = False

        out_rows.append({
            "date": day,
            "total_trades": n,
            "day_pnl": round(total, 2),
            "max_intraday_dd": round(max_intraday_dd, 2),
            "pnl_first_trade": round(pnl_first, 2),
            "pnl_first_2": round(pnl_first2, 2),
            "pnl_first_4": round(pnl_first4, 2),
            "pnl_later_trades": round(pnl_later, 2),
            "pct_loss_after_trade_4": round(pct_loss_after_4, 1),
            "pct_loss_after_first_loser": round(pct_loss_after_first, 1),
            "recovered_after_early_losses": int(recovered),
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"  Table 2: {path} ({len(out_rows)} days)")
    return out_rows


# ── Table 3: Confidence calibration ─────────────────────────────────
def write_confidence_calibration(rows: list[dict], path: str):
    # Filter to bars where model would have traded (not in_position, not cooldown)
    tradeable = [r for r in rows if r["blocked_reason"] not in ("in_position", "cooldown", "")]
    # Include taken trades
    tradeable += [r for r in rows if r["trade_taken"]]
    # Deduplicate
    seen = set()
    unique = []
    for r in tradeable:
        key = (r["date"], r["bar_of_day"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Sort by opportunity_logit
    unique.sort(key=lambda r: r["predicted_opportunity_logit"], reverse=True)
    n = len(unique)

    buckets = [
        ("Top 1%", 0, max(1, n // 100)),
        ("Top 5%", 0, max(1, n // 20)),
        ("Top 10%", 0, max(1, n // 10)),
        ("10-25%", max(1, n // 10), max(1, n // 4)),
        ("25-50%", max(1, n // 4), n // 2),
        ("50-75%", n // 2, 3 * n // 4),
        ("Bottom 25%", 3 * n // 4, n),
    ]

    lines = []
    lines.append("=" * 100)
    lines.append("  CONFIDENCE CALIBRATION (by opportunity_logit)")
    lines.append("=" * 100)
    hdr = f"{'Bucket':<14} {'Count':>6} {'WR%':>6} {'AvgPnL':>9} {'PF':>7} {'OracBest':>9} {'AvgRank':>8} {'nValid':>7} {'AvgBar':>7}"
    lines.append(hdr)
    lines.append("-" * 100)

    for label, start, end in buckets:
        subset = unique[start:end]
        if not subset:
            continue
        taken = [r for r in subset if r["trade_taken"]]
        pnls = [r["chosen_trade_pnl"] for r in taken if r["chosen_trade_pnl"] != ""]
        wins = sum(1 for p in pnls if isinstance(p, (int, float)) and p > 0)
        losses = sum(1 for p in pnls if isinstance(p, (int, float)) and p <= 0)
        wr = wins / len(pnls) * 100 if pnls else 0
        avg_pnl = np.mean([p for p in pnls if isinstance(p, (int, float))]) if pnls else 0
        gross_win = sum(p for p in pnls if isinstance(p, (int, float)) and p > 0)
        gross_loss = abs(sum(p for p in pnls if isinstance(p, (int, float)) and p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0

        oracle_pnls = [r["oracle_best_pnl"] for r in subset if r["oracle_best_pnl"] != "" and isinstance(r["oracle_best_pnl"], (int, float))]
        avg_oracle = np.mean(oracle_pnls) if oracle_pnls else 0

        ranks = [r["rank_of_chosen_contract"] for r in subset if r["rank_of_chosen_contract"] > 0]
        avg_rank = np.mean(ranks) if ranks else 0

        n_valids = [r["n_valid_contracts"] for r in subset]
        avg_nvalid = np.mean(n_valids)

        bars = [r["bar_of_day"] for r in subset]
        avg_bar = np.mean(bars)

        lines.append(f"  {label:<12} {len(subset):>6} {wr:>5.1f}% {avg_pnl:>8.4f} {pf:>7.3f} {avg_oracle:>8.4f} {avg_rank:>8.1f} {avg_nvalid:>7.1f} {avg_bar:>7.1f}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Table 3: {path}")
    for line in lines:
        print(f"    {line}")


# ── Table 4: Trade-number-within-day ─────────────────────────────────
def write_trade_number_table(rows: list[dict], path: str):
    taken = [r for r in rows if r["trade_taken"]]
    buckets = defaultdict(list)
    for r in taken:
        n = r["trade_number_within_day"]
        key = str(n) if n <= 4 else "5+"
        buckets[key].append(r)

    lines = []
    lines.append("=" * 100)
    lines.append("  TRADE-NUMBER-WITHIN-DAY ANALYSIS")
    lines.append("=" * 100)
    hdr = f"{'Trade#':<8} {'Count':>6} {'WR%':>6} {'AvgPnL':>9} {'PF':>7} {'AvgConf':>9} {'OracBest':>9} {'AvgBar':>7}"
    lines.append(hdr)
    lines.append("-" * 100)

    for key in ["1", "2", "3", "4", "5+"]:
        subset = buckets.get(key, [])
        if not subset:
            continue
        pnls = [r["chosen_trade_pnl"] for r in subset if r["chosen_trade_pnl"] != "" and isinstance(r["chosen_trade_pnl"], (int, float))]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100 if pnls else 0
        avg_pnl = np.mean(pnls) if pnls else 0
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0

        confs = [r["predicted_opportunity_logit"] for r in subset]
        avg_conf = np.mean(confs)

        oracle_pnls = [r["oracle_best_pnl"] for r in subset if r["oracle_best_pnl"] != "" and isinstance(r["oracle_best_pnl"], (int, float))]
        avg_oracle = np.mean(oracle_pnls) if oracle_pnls else 0

        bars = [r["bar_of_day"] for r in subset]
        avg_bar = np.mean(bars)

        lines.append(f"  {key:<6} {len(subset):>6} {wr:>5.1f}% {avg_pnl:>8.4f} {pf:>7.3f} {avg_conf:>8.4f} {avg_oracle:>8.4f} {avg_bar:>7.1f}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Table 4: {path}")
    for line in lines:
        print(f"    {line}")


# ── Table 5: Blocked analysis (max_daily_trades=4) ──────────────────
def write_blocked_analysis(rows: list[dict], path: str):
    blocked = [r for r in rows if r["blocked_reason"] == "max_daily_trades_4"]
    taken = [r for r in rows if r["trade_taken"]]

    lines = []
    lines.append("=" * 100)
    lines.append("  BLOCKED BY MAX_DAILY_TRADES=4 vs TAKEN TRADES")
    lines.append("=" * 100)

    if not blocked:
        lines.append("  No trades blocked (all days had <= 4 trades)")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Table 5: {path} (no blocked trades)")
        return

    # Blocked trades: use MODEL'S chosen contract oracle PnL (not oracle best)
    # This answers: would the model's pick have been profitable?
    blocked_chosen = [r["chosen_oracle_pnl"] for r in blocked if r["chosen_oracle_pnl"] != "" and isinstance(r["chosen_oracle_pnl"], (int, float))]
    blocked_wins = sum(1 for p in blocked_chosen if p > 0)
    blocked_losses = sum(1 for p in blocked_chosen if p <= 0)
    blocked_wr = blocked_wins / len(blocked_chosen) * 100 if blocked_chosen else 0
    blocked_avg = np.mean(blocked_chosen) if blocked_chosen else 0
    blocked_confs = [r["predicted_opportunity_logit"] for r in blocked]
    blocked_avg_conf = np.mean(blocked_confs)
    # Also report oracle best for comparison
    blocked_oracle_best = [r["oracle_best_pnl"] for r in blocked if r["oracle_best_pnl"] != "" and isinstance(r["oracle_best_pnl"], (int, float))]
    blocked_oracle_best_wr = sum(1 for p in blocked_oracle_best if p > 0) / len(blocked_oracle_best) * 100 if blocked_oracle_best else 0

    # Taken trades: actual results
    taken_pnls = [r["chosen_trade_pnl"] for r in taken if r["chosen_trade_pnl"] != "" and isinstance(r["chosen_trade_pnl"], (int, float))]
    taken_wins = sum(1 for p in taken_pnls if p > 0)
    taken_wr = taken_wins / len(taken_pnls) * 100 if taken_pnls else 0
    taken_avg = np.mean(taken_pnls) if taken_pnls else 0
    taken_confs = [r["predicted_opportunity_logit"] for r in taken]
    taken_avg_conf = np.mean(taken_confs)
    taken_gw = sum(p for p in taken_pnls if p > 0)
    taken_gl = abs(sum(p for p in taken_pnls if p <= 0))
    taken_pf = taken_gw / taken_gl if taken_gl > 0 else 0

    blocked_gw = sum(p for p in blocked_chosen if p > 0)
    blocked_gl = abs(sum(p for p in blocked_chosen if p <= 0))
    blocked_pf = blocked_gw / blocked_gl if blocked_gl > 0 else 0

    hdr = f"{'Group':<20} {'Count':>6} {'WR%':>6} {'AvgPnL':>9} {'PF':>7} {'AvgConf':>9}"
    lines.append(hdr)
    lines.append("-" * 100)
    lines.append(f"  {'Taken (actual)':18} {len(taken_pnls):>6} {taken_wr:>5.1f}% {taken_avg:>8.4f} {taken_pf:>7.3f} {taken_avg_conf:>8.4f}")
    lines.append(f"  {'Blocked (model pick)':18} {len(blocked_chosen):>6} {blocked_wr:>5.1f}% {blocked_avg:>8.4f} {blocked_pf:>7.3f} {blocked_avg_conf:>8.4f}")
    lines.append(f"  (Oracle-best WR on blocked bars: {blocked_oracle_best_wr:.1f}% — opportunity existed but model pick may differ)")
    lines.append("")

    # Per-day comparison: taken trades on same day as blocked trades
    blocked_days = set(r["date"] for r in blocked)
    taken_on_blocked_days = [r for r in taken if r["date"] in blocked_days]
    taken_bd_pnls = [r["chosen_trade_pnl"] for r in taken_on_blocked_days if r["chosen_trade_pnl"] != "" and isinstance(r["chosen_trade_pnl"], (int, float))]
    if taken_bd_pnls:
        bd_wins = sum(1 for p in taken_bd_pnls if p > 0)
        bd_wr = bd_wins / len(taken_bd_pnls) * 100
        bd_avg = np.mean(taken_bd_pnls)
        bd_gw = sum(p for p in taken_bd_pnls if p > 0)
        bd_gl = abs(sum(p for p in taken_bd_pnls if p <= 0))
        bd_pf = bd_gw / bd_gl if bd_gl > 0 else 0
        bd_confs = [r["predicted_opportunity_logit"] for r in taken_on_blocked_days]
        bd_avg_conf = np.mean(bd_confs)
        lines.append(f"  {'Taken same-day':18} {len(taken_bd_pnls):>6} {bd_wr:>5.1f}% {bd_avg:>8.4f} {bd_pf:>7.3f} {bd_avg_conf:>8.4f}")

    lines.append("")
    lines.append(f"  Blocked bars: {len(blocked)}")
    lines.append(f"  Oracle WR of blocked: {blocked_wr:.1f}% (would-have-won vs would-have-lost)")
    lines.append(f"  Avg confidence of blocked: {blocked_avg_conf:.4f}")
    lines.append(f"  Avg confidence of taken: {taken_avg_conf:.4f}")
    delta = taken_avg_conf - blocked_avg_conf
    lines.append(f"  Confidence gap (taken - blocked): {delta:+.4f}")
    if delta > 0.01:
        lines.append("  → Model ranks earlier trades higher confidence — supports H2")
    elif delta < -0.01:
        lines.append("  → Blocked trades have HIGHER confidence — supports H1 (signal is wrong)")
    else:
        lines.append("  → No meaningful confidence gap — max-4 is mechanical, not signal-driven")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Table 5: {path}")
    for line in lines:
        print(f"    {line}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="v2/models/model.pt")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model and data...")
    model = load_model_from_path(args.model)
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    policy = DecisionPolicy()
    print(f"  Model: {args.model}")
    print(f"  Policy: gate={policy.gate_threshold}, cooldown={policy.cooldown_bars}")

    print("\nRunning full replay with diagnostics...")
    rows, trades = run_full_replay_with_diagnostics(model, data, policy)
    print(f"  Total bars: {len(rows)}, Trades taken: {len(trades)}")

    print("\nWriting tables...")
    write_per_trade_csv(rows, os.path.join(OUTPUT_DIR, "1_per_trade.csv"))
    write_day_decomposition(rows, trades, policy, os.path.join(OUTPUT_DIR, "2_day_decomposition.csv"))
    write_confidence_calibration(rows, os.path.join(OUTPUT_DIR, "3_confidence_calibration.txt"))
    write_trade_number_table(rows, os.path.join(OUTPUT_DIR, "4_trade_number.txt"))
    write_blocked_analysis(rows, os.path.join(OUTPUT_DIR, "5_blocked_analysis.txt"))

    print(f"\nAll tables written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
