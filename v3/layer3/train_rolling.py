"""Train and replay an honest rolling Layer 3 on top of a fixed entry policy."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from v3.layer2.common import save_json
from v3.layer3.common import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_EQUITY,
    DEFAULT_EXIT_THRESHOLDS,
    DEFAULT_FOLD0_FALLBACK,
    DEFAULT_FOLD0_FALLBACK_BARS,
    DEFAULT_MIN_TRAIN_TRADES,
    DEFAULT_OUT_DIR,
    DEFAULT_SEED,
    DEFAULT_SESSION_END_BAR,
    DEFAULT_SURFACE_DIR,
    DEFAULT_UNIFIED_CHOSEN_TRADES,
    TRADE_STATE_NAMES,
    agg_exit_metrics,
    baseline_time_stop_metrics,
    build_trade_dataset,
    load_layer25_policy,
    load_unified_policy_trades,
    replay_trade_set,
    train_models_by_window,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entry-source", default="layer25", choices=("layer25", "unified"))
    p.add_argument("--surface-dir", default=DEFAULT_SURFACE_DIR)
    p.add_argument("--chosen-trades", default=DEFAULT_UNIFIED_CHOSEN_TRADES)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--patience-threshold", type=float, default=None)
    p.add_argument("--exit-thresholds", type=float, nargs="*", default=list(DEFAULT_EXIT_THRESHOLDS))
    p.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--min-train-trades", type=int, default=DEFAULT_MIN_TRAIN_TRADES)
    p.add_argument("--session-end-bar", type=int, default=DEFAULT_SESSION_END_BAR)
    p.add_argument("--commission", type=float, default=DEFAULT_COMMISSION_PER_CONTRACT)
    p.add_argument(
        "--fold0-fallback",
        default=DEFAULT_FOLD0_FALLBACK,
        choices=("time_stop", "time_of_day_90"),
    )
    p.add_argument("--fold0-fallback-bars", type=int, default=DEFAULT_FOLD0_FALLBACK_BARS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.entry_source == "layer25":
        trades, policy_meta = load_layer25_policy(args.surface_dir, args.patience_threshold)
        source_label = f"Layer 2.5 policy patience_threshold={policy_meta['threshold']:.2f}"
    else:
        trades, policy_meta = load_unified_policy_trades(args.chosen_trades)
        source_label = (
            f"Unified policy chosen trades n={policy_meta['n_trades']} "
            f"call_share={policy_meta['call_share']:.3f}"
        )
    print(
        f"Loaded entry policy: trades={len(trades)} {source_label}",
        flush=True,
    )

    trade_data, dataset_meta = build_trade_dataset(
        trades,
        equity=args.equity,
        session_end_bar=args.session_end_bar,
        commission=args.commission,
    )
    print(
        f"Built Layer 3 trade datasets: usable={dataset_meta['n_trade_datasets']} "
        f"skipped={dataset_meta['n_skipped']}",
        flush=True,
    )

    baseline_metrics, baseline_rows = baseline_time_stop_metrics(trade_data, args.equity)
    baseline_label = "Layer 2.5 + time-stop" if args.entry_source == "layer25" else "Unified policy + time-stop"
    print(
        f"Baseline {baseline_label}: pf={baseline_metrics['pf']:.3f} "
        f"dd={baseline_metrics['max_dd_pct']:.1f}% mean=${baseline_metrics['mean_pnl']:.1f}",
        flush=True,
    )

    models, train_reports = train_models_by_window(
        trade_data,
        seed=args.seed,
        min_train_trades=args.min_train_trades,
    )
    print(flush=True)
    print(f"{'win':<5}{'train_trades':>14}{'train_rows':>12}{'test_trades':>13}{'mode':>12}", flush=True)
    for report in train_reports:
        print(
            f"{report['window_idx']:<5}{report['train_trades']:>14}{report['train_rows']:>12}"
            f"{report['test_trades']:>13}{report['mode']:>12}",
            flush=True,
        )

    threshold_results: list[dict] = []
    best_result: dict | None = None
    for threshold in args.exit_thresholds:
        all_rows = []
        per_window = []
        for window_idx in sorted({int(td['window_idx']) for td in trade_data}):
            test_data = [td for td in trade_data if int(td["window_idx"]) == window_idx]
            rows = replay_trade_set(
                test_data,
                models.get(window_idx),
                float(threshold),
                fallback_policy=args.fold0_fallback,
                fallback_bars=args.fold0_fallback_bars,
            )
            window_df = pd.DataFrame(rows)
            metrics = agg_exit_metrics(window_df, args.equity)
            per_window.append({"window_idx": int(window_idx), **metrics})
            all_rows.extend(rows)

        full_df = pd.DataFrame(all_rows)
        metrics = agg_exit_metrics(full_df, args.equity)
        result = {
            "threshold": float(threshold),
            "overall": metrics,
            "per_window": per_window,
        }
        threshold_results.append(result)
        if best_result is None or (metrics["pf"], metrics["mean_pnl"]) > (
            best_result["overall"]["pf"],
            best_result["overall"]["mean_pnl"],
        ):
            best_result = result

        threshold_tag = f"{threshold:.2f}".replace(".", "p")
        trades_path = os.path.join(args.out_dir, f"layer3_trades_thr_{threshold_tag}.csv")
        full_df.to_csv(trades_path, index=False)
        print(
            f"thr={threshold:.2f} pf={metrics['pf']:.3f} dd={metrics['max_dd_pct']:.1f}% "
            f"mean=${metrics['mean_pnl']:.1f} bars={metrics['mean_bars_held']:.1f}",
            flush=True,
        )

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_name = "layer25_time_stop_trades.csv" if args.entry_source == "layer25" else "unified_time_stop_trades.csv"
    baseline_df.to_csv(os.path.join(args.out_dir, baseline_name), index=False)

    payload = {
        "meta": {
            "entry_source": args.entry_source,
            "surface_dir": args.surface_dir,
            "chosen_trades": args.chosen_trades if args.entry_source == "unified" else None,
            "out_dir": args.out_dir,
            "entry_patience_threshold": (
                float(policy_meta["threshold"]) if args.entry_source == "layer25" else None
            ),
            "exit_thresholds": [float(x) for x in args.exit_thresholds],
            "fold0_fallback": args.fold0_fallback,
            "fold0_fallback_bars": int(args.fold0_fallback_bars),
            "session_end_bar": int(args.session_end_bar),
            "commission": float(args.commission),
            "seed": int(args.seed),
            "min_train_trades": int(args.min_train_trades),
            "trade_state_features": TRADE_STATE_NAMES,
            "note": (
                "Threshold comparison is exploratory. The best threshold in this report is "
                "not a deployment-calibrated value because it is picked on the same rolling "
                "OOS windows used for evaluation."
            ),
            "entry_policy_meta": policy_meta,
        },
        "trade_dataset": dataset_meta,
        "entry_time_stop_baseline": baseline_metrics,
        "train_reports": train_reports,
        "threshold_results": threshold_results,
        "best_threshold_by_aggregate_pf_exploratory": best_result,
    }
    out_path = os.path.join(args.out_dir, "rolling_layer3_report.json")
    save_json(out_path, payload)
    print(f"Saved: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
