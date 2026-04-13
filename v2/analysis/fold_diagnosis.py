"""Per-fold drawdown diagnosis for the production model.

Replays the current model against each walk-forward fold's test window
to identify what trades cause >20% drawdown in folds 1-3.

Usage:
    python3 -m v2.analysis.fold_diagnosis
"""
from __future__ import annotations

import torch

from v2.core.policy import DecisionPolicy
from v2.core.walkforward import generate_folds, dates_to_mask
from v2.core.metrics import compute_metrics, compute_score
from v2.replay import replay_validation, load_model_from_path


def _equity_curve(trades: list, starting_equity: float, contract_mult: int) -> list[tuple[str, float]]:
    """Build daily equity curve from trades, returning (date, equity) pairs."""
    daily_pnl: dict[str, float] = {}
    for t in trades:
        dollar_pnl = t.net_pnl_pct / 100.0 * t.entry_price * contract_mult
        daily_pnl[t.trade_date] = daily_pnl.get(t.trade_date, 0.0) + dollar_pnl

    equity = starting_equity
    curve = []
    for date in sorted(daily_pnl):
        equity += daily_pnl[date]
        curve.append((date, equity))
    return curve


def _find_dd_window(curve: list[tuple[str, float]]) -> tuple[str, str, float, float]:
    """Find peak-to-trough drawdown window. Returns (peak_date, trough_date, dd_pct, peak_equity)."""
    if not curve:
        return ("", "", 0.0, 0.0)

    peak_eq = curve[0][1]
    peak_date = curve[0][0]
    max_dd = 0.0
    max_dd_peak_date = peak_date
    max_dd_trough_date = curve[0][0]
    max_dd_peak_eq = peak_eq

    for date, eq in curve:
        if eq > peak_eq:
            peak_eq = eq
            peak_date = date
        dd = (peak_eq - eq) / peak_eq if peak_eq > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_peak_date = peak_date
            max_dd_trough_date = date
            max_dd_peak_eq = peak_eq

    return max_dd_peak_date, max_dd_trough_date, max_dd, max_dd_peak_eq


def main():
    print("Loading model and data...")
    model = load_model_from_path("v2/models/model.pt")
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)

    all_dates = [d for d in data["dates"]]
    unique_dates = sorted(set(all_dates))
    folds = generate_folds(unique_dates)

    policy = DecisionPolicy()
    starting_eq = policy.starting_equity
    contract_mult = policy.contract_multiplier

    print(f"Policy: stop={policy.stop_pct}, target={policy.target_pct}, "
          f"extra_tiers={policy.extra_trailing_tiers}")
    print(f"Folds: {len(folds)}, unique dates: {len(unique_dates)}")

    for fold in folds:
        print(f"\n{'='*80}")
        print(f"  FOLD {fold.fold_idx}: test {fold.test_days[0]} to {fold.test_days[-1]} "
              f"({len(fold.test_days)} days)")
        print(f"{'='*80}")

        # Build test mask and inject into data
        mask_key = f"_diag_fold_{fold.fold_idx}"
        test_set = set(fold.test_days)
        data[mask_key] = dates_to_mask(all_dates, test_set)

        metrics, trades, _ = replay_validation(
            model, data, mask_key=mask_key, policy=policy
        )

        # --- Summary ---
        gate_str = f" GATE: {metrics.gate_failure}" if metrics.gate_failure else ""
        print(f"  Score: {metrics.score:.4f}{gate_str}")
        print(f"  Trades: {metrics.total_trades}  WR: {metrics.win_rate:.1%}  "
              f"PF: {metrics.profit_factor:.3f}")
        print(f"  DD: {metrics.max_account_drawdown:.1%}  Sortino: {metrics.daily_sortino:.2f}  "
              f"+Day: {metrics.positive_day_rate:.1%}")
        print(f"  Net P&L: ${metrics.net_pnl_dollars:+,.0f}  "
              f"Avg MFE: {metrics.avg_mfe:.1f}%  Avg MAE: {metrics.avg_mae:.1f}%")

        # --- Exit reason breakdown ---
        print(f"\n  Exit reasons:")
        print(f"    SL: {metrics.stop_loss_count}  TS: {metrics.trailing_stop_count}  "
              f"TP: {metrics.take_profit_count}  EOD: {metrics.eod_count}  "
              f"MAX_HOLD: {metrics.max_hold_count}")

        # --- Direction breakdown ---
        print(f"  Direction: C={metrics.call_count} P={metrics.put_count}")

        if not trades:
            print("  No trades in this fold.")
            continue

        # --- Equity curve and DD window ---
        curve = _equity_curve(trades, starting_eq, contract_mult)
        peak_date, trough_date, dd_pct, peak_eq = _find_dd_window(curve)

        print(f"\n  DD window: {peak_date} (${peak_eq:,.0f}) -> {trough_date} "
              f"(DD {dd_pct:.1%})")

        # --- VIX regime distribution ---
        vix_values = [t.vix_regime_at_entry for t in trades]
        if vix_values:
            avg_vix = sum(vix_values) / len(vix_values)
            print(f"  VIX regime: avg={avg_vix:.2f} "
                  f"min={min(vix_values):.2f} max={max(vix_values):.2f}")

        # --- Trades within DD window ---
        if metrics.max_account_drawdown > 0.20 and peak_date and trough_date:
            dd_trades = [t for t in trades if peak_date <= t.trade_date <= trough_date]
            print(f"\n  Trades during DD window ({len(dd_trades)} trades):")
            print(f"  {'Date':>10} {'Dir':>3} {'Strike':>8} {'Entry$':>7} {'P&L%':>7} "
                  f"{'$P&L':>7} {'Exit':>14} {'MFE%':>6} {'MAE%':>6} {'Bars':>4}")
            for t in sorted(dd_trades, key=lambda x: (x.trade_date, x.entry_bar)):
                dollar_pnl = t.net_pnl_pct / 100.0 * t.entry_price * contract_mult
                print(f"  {t.trade_date:>10} {t.intent.right:>3} "
                      f"{t.intent.strike:>8.1f} {t.entry_price:>7.2f} "
                      f"{t.net_pnl_pct:>+7.1f} {dollar_pnl:>+7.0f} "
                      f"{t.exit_reason:>14} {t.mfe_pct:>6.1f} {t.mae_pct:>6.1f} "
                      f"{t.bars_held:>4}")

            # --- Daily P&L during DD window ---
            dd_daily: dict[str, list[float]] = {}
            for t in dd_trades:
                dollar_pnl = t.net_pnl_pct / 100.0 * t.entry_price * contract_mult
                dd_daily.setdefault(t.trade_date, []).append(dollar_pnl)

            print(f"\n  Daily P&L during DD window:")
            for date in sorted(dd_daily):
                pnls = dd_daily[date]
                total = sum(pnls)
                print(f"    {date}: {len(pnls)} trades, ${total:+,.0f}")

        # --- Worst 5 trades in the fold ---
        sorted_by_pnl = sorted(trades, key=lambda t: t.net_pnl_pct)[:5]
        print(f"\n  Worst 5 trades:")
        for t in sorted_by_pnl:
            dollar_pnl = t.net_pnl_pct / 100.0 * t.entry_price * contract_mult
            print(f"    {t.trade_date} {t.intent.right} strike={t.intent.strike:.0f} "
                  f"entry=${t.entry_price:.2f} P&L={t.net_pnl_pct:+.1f}% "
                  f"(${dollar_pnl:+,.0f}) exit={t.exit_reason} "
                  f"MFE={t.mfe_pct:.1f}% MAE={t.mae_pct:.1f}%")

    # --- Cross-fold comparison ---
    print(f"\n{'='*80}")
    print(f"  CROSS-FOLD COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Fold':>4} {'Test Period':>25} {'Score':>8} {'DD':>6} {'Trades':>6} "
          f"{'WR':>6} {'PF':>6} {'SL':>3} {'TS':>3} {'TP':>3} {'$P&L':>8}")


if __name__ == "__main__":
    main()
