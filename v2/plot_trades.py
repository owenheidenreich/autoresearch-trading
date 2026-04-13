"""Plot all trades on SPX price chart + equity curve + CSV export.

Loads the best model from the artifact system, runs replay, and generates
interactive Plotly HTML charts and a CSV trade log for analysis.

Usage:
    python -m v2.plot_trades                           # promote_mask (default)
    python -m v2.plot_trades --mask shadow             # shadow_mask
    python -m v2.plot_trades --model /path/to/model.pt # specific checkpoint
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trades(mask_key: str, model_path: str | None = None) -> tuple[list, dict, object]:
    """Load model, run replay, return (trades, data_dict, metrics).

    Default: loads the best compatible promoted artifact from the artifact system.
    If `model_path` is provided, that checkpoint is loaded directly.
    """
    from v2.replay import load_best_model, load_model_from_path, replay_validation

    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    dataset_fp = data.get("metadata", {}).get("fingerprint")

    if model_path:
        resolved = model_path
        model = load_model_from_path(resolved)
        from v2.core.policy import DEFAULT_POLICY
        policy = DEFAULT_POLICY
        print(f"Loaded model from {resolved}")
    else:
        model, policy, manifest = load_best_model(current_dataset_fingerprint=dataset_fp)
        resolved = manifest.get("experiment_id", "best_artifact")
        print(f"Loaded best compatible artifact for plotting: {resolved}")

    metrics, trades, _ = replay_validation(model, data, mask_key=mask_key, policy=policy)

    print(f"Replay: {len(trades)} trades on {mask_key}")
    print(f"  Score: {metrics.score:.3f}  WR: {metrics.win_rate:.1%}  "
          f"PF: {metrics.profit_factor:.2f}  Trades/day: {metrics.trades_per_day:.1f}")

    return trades, data, metrics


# ---------------------------------------------------------------------------
# Chart 1: SPX price with trade overlay
# ---------------------------------------------------------------------------

def plot_spx_trades(trades: list, data: dict, mask_key: str, output: Path):
    """Plot SPX price with entry/exit markers for every trade."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    spot_prices = data['spot_prices'].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    mask = data[mask_key].numpy()

    # Build per-day SPX price series for masked days only
    mask_indices = np.where(mask)[0]
    eval_dates = sorted(set(dates[i] for i in mask_indices))

    # Build sequential x-axis (no gaps between days)
    day_bars = defaultdict(list)
    for idx in mask_indices:
        day_bars[dates[idx]].append(idx)

    seq_x = []       # sequential index
    seq_spx = []     # SPX price
    seq_labels = []  # hover label
    day_boundaries = []  # (seq_x, date_str) for vertical separators
    seq_counter = 0

    bar_to_seq = {}      # global_bar_idx -> sequential x
    day_bod_to_seq = {}  # (date, bar_of_day) -> sequential x
    day_bod_to_spx = {}  # (date, bar_of_day) -> spot price

    for day in eval_dates:
        bars = sorted(day_bars[day])
        day_boundaries.append((seq_counter, day))
        for bar_idx in bars:
            bod = int(bar_of_day[bar_idx])
            bar_to_seq[bar_idx] = seq_counter
            day_bod_to_seq[(day, bod)] = seq_counter
            day_bod_to_spx[(day, bod)] = float(spot_prices[bar_idx])
            seq_x.append(seq_counter)
            seq_spx.append(float(spot_prices[bar_idx]))
            h, m = divmod(9 * 60 + 30 + bod, 60)
            seq_labels.append(f"{day} {h:02d}:{m:02d}")
            seq_counter += 1

    fig = make_subplots(rows=1, cols=1)

    # SPX price line
    fig.add_trace(go.Scatter(
        x=seq_x, y=seq_spx,
        mode='lines',
        name='SPX',
        line=dict(color='#2196F3', width=1),
        hovertemplate='%{customdata}<br>SPX: %{y:,.1f}<extra></extra>',
        customdata=seq_labels,
    ))

    # Day boundary separators
    for bx, day_str in day_boundaries:
        fig.add_vline(x=bx, line_dash="dot", line_color="rgba(128,128,128,0.2)")

    # Batch trade markers by category
    batches = {
        'win_call_entry': {'x': [], 'y': [], 'text': [], 'hover': []},
        'win_put_entry': {'x': [], 'y': [], 'text': [], 'hover': []},
        'loss_call_entry': {'x': [], 'y': [], 'text': [], 'hover': []},
        'loss_put_entry': {'x': [], 'y': [], 'text': [], 'hover': []},
        'win_exit': {'x': [], 'y': [], 'hover': []},
        'loss_exit': {'x': [], 'y': [], 'hover': []},
    }
    connector_x = []
    connector_y = []

    for i, t in enumerate(trades):
        # Trade bars are bar-of-day indices (not global), so look up by (date, bod)
        day = t.trade_date
        entry_seq = day_bod_to_seq.get((day, t.entry_fill_bar))
        exit_seq = day_bod_to_seq.get((day, t.exit_bar))
        if entry_seq is None:
            continue

        is_win = t.net_pnl_pct > 0
        is_call = t.intent.right == "C"
        prefix = 'win' if is_win else 'loss'
        direction = 'call' if is_call else 'put'

        entry_spx = day_bod_to_spx.get((day, t.entry_fill_bar), 0.0)
        exit_spx = day_bod_to_spx.get((day, t.exit_bar), entry_spx)

        entry_key = f'{prefix}_{direction}_entry'
        batches[entry_key]['x'].append(entry_seq)
        batches[entry_key]['y'].append(entry_spx)
        batches[entry_key]['text'].append(f"#{i+1}")
        batches[entry_key]['hover'].append(
            f"<b>ENTRY #{i+1}</b><br>"
            f"{t.trade_date}<br>"
            f"{'CALL' if is_call else 'PUT'} @ ${t.entry_price:.2f}<br>"
            f"SPX: {entry_spx:,.1f}<br>"
            f"Strike: {t.intent.strike}<br>"
        )

        if exit_seq is not None:
            exit_key = f'{prefix}_exit'
            batches[exit_key]['x'].append(exit_seq)
            batches[exit_key]['y'].append(exit_spx)
            batches[exit_key]['hover'].append(
                f"<b>EXIT #{i+1}</b><br>"
                f"{t.exit_reason}<br>"
                f"P&L: {t.net_pnl_pct * 100:+.1f}%<br>"
                f"MFE: {t.mfe_pct * 100:+.1f}% / MAE: {t.mae_pct * 100:+.1f}%<br>"
                f"Bars held: {t.bars_held}<br>"
            )
            connector_x.extend([entry_seq, exit_seq, None])
            connector_y.extend([entry_spx, exit_spx, None])

    # Entry markers
    entry_style = {
        'win_call_entry':  {'color': '#4CAF50', 'symbol': 'triangle-up'},
        'win_put_entry':   {'color': '#4CAF50', 'symbol': 'triangle-down'},
        'loss_call_entry': {'color': '#F44336', 'symbol': 'triangle-up'},
        'loss_put_entry':  {'color': '#F44336', 'symbol': 'triangle-down'},
    }
    for key, style in entry_style.items():
        b = batches[key]
        if not b['x']:
            continue
        label = key.replace('_entry', '').replace('_', ' ').title()
        fig.add_trace(go.Scatter(
            x=b['x'], y=b['y'],
            mode='markers+text',
            name=label,
            marker=dict(size=9, color=style['color'], symbol=style['symbol'],
                        line=dict(width=1, color='white')),
            text=b['text'], textposition='top center',
            textfont=dict(size=6, color=style['color']),
            hovertemplate=[h + '<extra></extra>' for h in b['hover']],
        ))

    # Exit markers
    exit_style = {'win_exit': '#4CAF50', 'loss_exit': '#F44336'}
    for key, color in exit_style.items():
        b = batches[key]
        if not b['x']:
            continue
        label = key.replace('_', ' ').title()
        fig.add_trace(go.Scatter(
            x=b['x'], y=b['y'],
            mode='markers',
            name=label,
            marker=dict(size=7, color=color, symbol='x',
                        line=dict(width=2, color=color)),
            hovertemplate=[h + '<extra></extra>' for h in b['hover']],
        ))

    # Entry-exit connectors
    if connector_x:
        fig.add_trace(go.Scatter(
            x=connector_x, y=connector_y,
            mode='lines',
            line=dict(color='rgba(180,180,180,0.3)', width=0.8, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))

    # X-axis: show date labels at day boundaries
    tick_vals = [bx for bx, _ in day_boundaries[::max(1, len(day_boundaries)//20)]]
    tick_text = [d for _, d in day_boundaries[::max(1, len(day_boundaries)//20)]]

    n_wins = sum(1 for t in trades if t.net_pnl_pct > 0)
    n_total = len(trades)
    wr = n_wins / n_total * 100 if n_total else 0

    fig.update_layout(
        title=dict(text=f"Trade Overlay: {n_total} trades, {wr:.0f}% win rate ({mask_key})",
                   font=dict(size=16)),
        xaxis=dict(
            tickvals=tick_vals, ticktext=tick_text,
            tickangle=-45, showgrid=False,
        ),
        yaxis=dict(title="SPX", tickformat=",.0f",
                   showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        template='plotly_dark',
        hovermode='closest',
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    fig.write_html(str(output), include_plotlyjs=True)
    print(f"Trade chart saved: {output}")


# ---------------------------------------------------------------------------
# Chart 2: Equity curve
# ---------------------------------------------------------------------------

def _day_boundary_ticks(trades: list, x_indices: list[int]) -> list[int]:
    """Return x-index of the first trade on each new date (for tick placement)."""
    ticks = []
    seen = set()
    for i, t in enumerate(trades):
        if t.trade_date not in seen:
            seen.add(t.trade_date)
            ticks.append(x_indices[i + 1])  # +1 because x_indices[0] is "Start"
    return ticks


def _day_boundary_labels(trades: list) -> list[str]:
    """Return date labels corresponding to day boundary ticks."""
    labels = []
    seen = set()
    for t in trades:
        if t.trade_date not in seen:
            seen.add(t.trade_date)
            labels.append(t.trade_date)
    return labels


def plot_equity(trades: list, output: Path, starting_equity: float = 10_000.0):
    """Plot equity curve with trade-by-trade P&L."""
    import plotly.graph_objects as go

    if not trades:
        print("No trades to plot equity curve.")
        return

    contract_multiplier = 100

    cash = starting_equity
    x_labels = [f"Start"]
    values = [cash]
    colors = ['gray']
    hover_texts = [f"Starting balance: ${cash:,.0f}"]

    for i, t in enumerate(trades):
        pnl_dollar = t.net_pnl_pct * t.entry_price * contract_multiplier
        cash += pnl_dollar
        cash = max(cash, 0.0)

        is_win = t.net_pnl_pct > 0
        x_labels.append(f"#{i+1} {t.trade_date}")
        values.append(cash)
        colors.append('#2ecc71' if is_win else '#e74c3c')
        hover_texts.append(
            f"<b>Trade #{i+1}</b><br>"
            f"{'CALL' if t.intent.right == 'C' else 'PUT'} on {t.trade_date}<br>"
            f"P&L: {t.net_pnl_pct * 100:+.1f}% (${pnl_dollar:+,.0f})<br>"
            f"Exit: {t.exit_reason}<br>"
            f"Balance: ${cash:,.0f}"
        )

    ending_cash = cash
    total_return = (ending_cash / starting_equity - 1) * 100

    # Peak / drawdown
    peak = starting_equity
    peaks = []
    max_dd = 0
    for v in values:
        peak = max(peak, v)
        peaks.append(peak)
        dd = (v - peak) / peak * 100
        max_dd = min(max_dd, dd)

    # Edge ratio
    mfes = [t.mfe_pct for t in trades if t.mfe_pct > 0]
    maes = [abs(t.mae_pct) for t in trades if t.mae_pct < 0]
    edge_ratio = (np.mean(mfes) / np.mean(maes)) if mfes and maes else 0

    n_wins = sum(1 for t in trades if t.net_pnl_pct > 0)
    wr = n_wins / len(trades) * 100

    fig = go.Figure()

    # Equity line
    x_indices = list(range(len(values)))
    fig.add_trace(go.Scatter(
        x=x_indices, y=values,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#3498db', width=2.5),
        marker=dict(size=6, color=colors, line=dict(width=1, color='white')),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        customdata=x_labels,
    ))

    # Starting cash line
    fig.add_hline(y=starting_equity, line_dash="dash", line_color="gray",
                  annotation_text=f"Start: ${starting_equity:,.0f}",
                  annotation_position="bottom left")

    # High water mark
    fig.add_trace(go.Scatter(
        x=x_indices, y=peaks,
        mode='lines',
        name='High Water Mark',
        line=dict(color='rgba(46,204,113,0.3)', width=1, dash='dot'),
        hoverinfo='skip',
    ))

    fig.update_layout(
        title=dict(
            text=(f"Equity Curve: ${starting_equity:,.0f} -> ${ending_cash:,.0f} "
                  f"({total_return:+.1f}%) | WR {wr:.0f}% | "
                  f"Max DD {max_dd:.1f}% | Edge Ratio {edge_ratio:.2f}"),
            font=dict(size=14),
        ),
        xaxis=dict(
            tickvals=_day_boundary_ticks(trades, x_indices),
            ticktext=_day_boundary_labels(trades),
            tickangle=-45,
            showgrid=True, gridcolor='rgba(128,128,128,0.2)',
        ),
        yaxis=dict(title="Portfolio Value ($)", tickformat="$,.0f",
                   showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        template='plotly_dark',
        hovermode='x unified',
        height=600,
        annotations=[
            dict(
                x=x_indices[-1], y=ending_cash,
                text=f"${ending_cash:,.0f}<br>({total_return:+.1f}%)<br>Max DD: {max_dd:.1f}%",
                showarrow=True, arrowhead=2, ax=40, ay=-40,
                font=dict(size=12, color='white'),
                bgcolor='rgba(0,0,0,0.7)', bordercolor='white',
            ),
        ],
    )

    fig.write_html(str(output), include_plotlyjs=True)
    print(f"Equity curve saved: {output}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_trades_csv(trades: list, output: Path, starting_equity: float = 10_000.0):
    """Export all trades to CSV for programmatic analysis."""
    from v2.core.features import BARS_PER_DAY

    columns = [
        "trade_num", "trade_date", "bar_of_day", "clock_time",
        "direction", "strike", "entry_price", "exit_price",
        "net_pnl_pct", "dollar_pnl", "raw_pnl_pct", "spread_cost_pct",
        "bars_held", "exit_reason",
        "mfe_pct", "mae_pct",
        "vix_at_entry", "spx_at_entry", "spx_at_exit",
        "stop_price", "tp_price", "max_hold_bars",
        "cumulative_equity",
    ]

    contract_multiplier = 100
    equity = starting_equity

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for i, t in enumerate(trades):
            pnl_dollar = t.net_pnl_pct * t.entry_price * contract_multiplier
            equity += pnl_dollar

            # Convert bar_of_day to clock time (market opens 9:30)
            bod = t.entry_bar % BARS_PER_DAY
            total_min = 9 * 60 + 30 + bod
            h, m = divmod(total_min, 60)

            writer.writerow({
                "trade_num": i + 1,
                "trade_date": t.trade_date,
                "bar_of_day": bod,
                "clock_time": f"{h:02d}:{m:02d}",
                "direction": t.intent.right,
                "strike": t.intent.strike,
                "entry_price": f"{t.entry_price:.2f}",
                "exit_price": f"{t.exit_price:.2f}",
                "net_pnl_pct": f"{t.net_pnl_pct * 100:.2f}",
                "dollar_pnl": f"{pnl_dollar:.2f}",
                "raw_pnl_pct": f"{t.raw_pnl_pct * 100:.2f}",
                "spread_cost_pct": f"{t.spread_cost_pct * 100:.2f}",
                "bars_held": t.bars_held,
                "exit_reason": t.exit_reason,
                "mfe_pct": f"{t.mfe_pct * 100:.2f}",
                "mae_pct": f"{t.mae_pct * 100:.2f}",
                "vix_at_entry": f"{t.vix_regime_at_entry:.2f}",
                "spx_at_entry": f"{t.underlying_at_entry:.2f}",
                "spx_at_exit": f"{t.underlying_at_exit:.2f}",
                "stop_price": f"{t.intent.stop_price:.2f}",
                "tp_price": f"{t.intent.take_profit_price:.2f}",
                "max_hold_bars": t.intent.max_hold_bars,
                "cumulative_equity": f"{equity:.2f}",
            })

    print(f"Trades CSV saved: {output} ({len(trades)} trades)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot trades on SPX chart + equity curve + CSV")
    parser.add_argument("--mask", default="promote_mask",
                        help="Data mask: promote_mask, shadow_mask, val_mask")
    parser.add_argument("--model", default=None,
                        help="Path to model.pt (default: load from artifact system)")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    mask_key = args.mask
    if not mask_key.endswith("_mask"):
        mask_key += "_mask"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades, data, metrics = load_trades(mask_key, model_path=args.model)

    if not trades:
        print("No trades produced. Nothing to plot.")
        sys.exit(1)

    plot_spx_trades(trades, data, mask_key, out_dir / "trades.html")
    plot_equity(trades, out_dir / "equity.html")
    export_trades_csv(trades, out_dir / "trades.csv")


if __name__ == "__main__":
    main()
