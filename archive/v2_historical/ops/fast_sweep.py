"""Fast GPU Sweep: train + evaluate inline (no subprocess overhead).

Trains each config, evaluates with batch inference, prints results.
Usage on GPU:
    /opt/conda/bin/python3 -m v2.ops.fast_sweep
"""
from __future__ import annotations

import json
import os
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from v2.core.schema import TradeIntent
from v2.core.features import (
    NUM_FEATURES, BARS_PER_DAY, _FEAT_IDX,
    NO_TRADE_BEFORE_BAR, NO_TRADE_AFTER_BAR, STOP_COOLDOWN_BARS,
)
from v2.core.simulator import simulate_trade
from v2.core.metrics import compute_metrics, compute_score
from v2.train import (
    TradingModel, TradeDataset, compute_loss,
    STRIKE_OFFSETS, LOOKBACK,
)


def train_config(
    features: torch.Tensor,
    labels: dict[str, torch.Tensor],
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 3e-4,
    d_model: int = 64,
    depth: int = 3,
    dropout: float = 0.1,
    gate_pos_weight: float = 0.3,
    lookback: int = 60,
    device: str = "cuda",
) -> tuple[TradingModel, dict]:
    """Train a single configuration and return (model, metrics)."""
    # Patch hyperparameters via module globals
    import v2.train as train_mod
    train_mod.D_MODEL = d_model
    train_mod.DEPTH = depth
    train_mod.DROPOUT = dropout
    train_mod.GATE_POS_WEIGHT = gate_pos_weight
    train_mod.LOOKBACK = lookback

    train_ds = TradeDataset(features, labels, train_mask, lookback=lookback)
    val_ds = TradeDataset(features, labels, val_mask, lookback=lookback)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    model = TradingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device)
            by = {k: v.to(device) for k, v in by.items()}
            optimizer.zero_grad()
            out = model(bx)
            loss, _ = compute_loss(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Quick val loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = {k: v.to(device) for k, v in by.items()}
                out = model(bx)
                _, ld = compute_loss(out, by)
                val_losses.append(ld['total'])

        vl = np.mean(val_losses)
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    return model, {'val_loss': float(best_val_loss), 'epochs': epochs}


def fast_replay(
    model: TradingModel,
    data: dict,
    max_days: int = 60,
    min_gate_prob: float = 0.5,
    device: str = "cuda",
):
    """Fast replay: batch inference then sequential simulation."""
    features = data['X']
    val_mask = data['val_mask'].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()
    lookback = LOOKBACK

    val_indices = np.where(val_mask)[0]
    val_dates = sorted(set(dates[i] for i in val_indices))[:max_days]

    model = model.to(device).eval()

    # Step 1: Batch inference on all valid val bars
    valid_bars = []
    for day in val_dates:
        day_bars = [i for i in range(len(dates)) if dates[i] == day]
        for b in day_bars:
            bod = int(bar_of_day[b])
            if bod < NO_TRADE_BEFORE_BAR or bod >= NO_TRADE_AFTER_BAR:
                continue
            if b < lookback:
                continue
            valid_bars.append(b)

    if not valid_bars:
        return compute_metrics([], 1)

    # Batch inference
    batch_size = 4096
    gate_probs = np.zeros(len(valid_bars))
    dir_classes = np.zeros(len(valid_bars), dtype=int)
    strike_classes = np.zeros(len(valid_bars), dtype=int)
    risk_out = np.zeros((len(valid_bars), 3))
    conf_out = np.zeros(len(valid_bars))

    with torch.no_grad():
        for start in range(0, len(valid_bars), batch_size):
            end = min(start + batch_size, len(valid_bars))
            batch_indices = valid_bars[start:end]

            windows = torch.stack([
                features[b - lookback:b] for b in batch_indices
            ]).to(device)

            outputs = model(windows)
            gate_probs[start:end] = torch.sigmoid(outputs['gate'].squeeze(-1)).cpu().numpy()
            dir_classes[start:end] = outputs['direction'].argmax(dim=-1).cpu().numpy()
            strike_classes[start:end] = outputs['strike'].argmax(dim=-1).cpu().numpy()
            risk_out[start:end] = outputs['risk'].cpu().numpy()
            conf_out[start:end] = torch.sigmoid(outputs['confidence'].squeeze(-1)).cpu().numpy()

    # Step 2: Sequential simulation with trade logic
    all_trades = []
    num_days = 0
    last_exit = -STOP_COOLDOWN_BARS - 1

    for idx, bar_idx in enumerate(valid_bars):
        # Day boundary reset
        if idx == 0 or dates[bar_idx] != dates[valid_bars[idx - 1]]:
            num_days += 1
            last_exit = -STOP_COOLDOWN_BARS - 1

        if gate_probs[idx] < min_gate_prob:
            continue

        if bar_idx - last_exit < STOP_COOLDOWN_BARS:
            continue

        bod = int(bar_of_day[bar_idx])
        spot = float(spot_prices[bar_idx])
        if np.isnan(spot) or spot <= 0:
            continue

        right = "C" if dir_classes[idx] == 0 else "P"
        strike_offset = STRIKE_OFFSETS[min(strike_classes[idx], len(STRIKE_OFFSETS) - 1)]
        atm = round(spot / 5.0) * 5.0
        strike = atm + strike_offset

        # Get option price
        offset_abs = abs(strike_offset)
        if offset_abs == 0:
            px_key = f"atm_{right.lower() == 'c' and 'call' or 'put'}_prices"
        else:
            px_key = f"otm{offset_abs}_{right.lower() == 'c' and 'call' or 'put'}_prices"

        # Simpler key lookup
        if right == "C":
            if offset_abs == 0: px_key = "atm_call_prices"
            elif offset_abs == 5: px_key = "otm5_call_prices"
            elif offset_abs == 10: px_key = "otm10_call_prices"
            elif offset_abs == 15: px_key = "otm15_call_prices"
            elif offset_abs == 20: px_key = "otm20_call_prices"
            elif offset_abs == 25: px_key = "otm25_call_prices"
            else: px_key = "otm30_call_prices"
        else:
            if offset_abs == 0: px_key = "atm_put_prices"
            elif offset_abs == 5: px_key = "otm5_put_prices"
            elif offset_abs == 10: px_key = "otm10_put_prices"
            elif offset_abs == 15: px_key = "otm15_put_prices"
            elif offset_abs == 20: px_key = "otm20_put_prices"
            elif offset_abs == 25: px_key = "otm25_put_prices"
            else: px_key = "otm30_put_prices"

        if px_key not in data:
            continue
        option_arr = data[px_key].numpy().astype(np.float32)
        px = float(option_arr[bar_idx])
        if np.isnan(px) or px <= 0:
            continue

        # Risk params
        r = risk_out[idx]
        stop_pct = 1.0 / (1.0 + np.exp(-r[0])) * 0.55 + 0.10
        target_pct = 1.0 / (1.0 + np.exp(-r[1])) * 1.5 + 0.15
        hold_frac = 1.0 / (1.0 + np.exp(-r[2]))
        max_hold = max(10, int(hold_frac * BARS_PER_DAY))

        expiry = dates[bar_idx].replace("-", "")

        intent = TradeIntent(
            trade=True, expiry=expiry, strike=strike, right=right, qty=1,
            entry_ref_price=px, order_style="MKT", tif="DAY",
            stop_price=max(0.01, px * (1.0 - stop_pct)),
            take_profit_price=px * (1.0 + target_pct),
            max_hold_bars=max_hold, exit_policy="STOP_TP_TIME",
            confidence=float(conf_out[idx]),
            reason_codes=(f"g={gate_probs[idx]:.2f}",),
            bar_index=bod, intent_id=str(uuid.uuid4()),
            underlying_price=spot,
        )

        trade = simulate_trade(
            intent, option_arr, data['X'].numpy(),
            bar_of_day, dates, bar_idx,
        )
        if trade:
            all_trades.append(trade)
            last_exit = trade.exit_bar

    return compute_metrics(all_trades, num_days=max(num_days, 1))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data...")
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    features = data['X']
    labels = {k: data[k] for k in data if k.startswith('oracle_')}
    train_mask = data['train_mask']
    val_mask = data['val_mask']

    configs = [
        # name, gate_pw, lr, d_model, depth, dropout, lookback
        ("gate01", 0.1, 3e-4, 64, 3, 0.1, 60),
        ("gate02", 0.2, 3e-4, 64, 3, 0.1, 60),
        ("gate03", 0.3, 3e-4, 64, 3, 0.1, 60),
        ("gate05", 0.5, 3e-4, 64, 3, 0.1, 60),
        ("lr1e4", 0.2, 1e-4, 64, 3, 0.1, 60),
        ("lr5e4", 0.2, 5e-4, 64, 3, 0.1, 60),
        ("lr1e3", 0.2, 1e-3, 64, 3, 0.1, 60),
        ("d128", 0.2, 3e-4, 128, 4, 0.1, 60),
        ("look120", 0.2, 3e-4, 64, 3, 0.1, 120),
        ("drop005", 0.2, 3e-4, 64, 3, 0.05, 60),
        ("drop02", 0.2, 3e-4, 64, 3, 0.2, 60),
    ]

    results = []
    for name, gpw, lr, dm, dep, dr, lb in configs:
        print(f"\n{'='*60}")
        print(f"  Training: {name} (gpw={gpw} lr={lr} d={dm} depth={dep})")
        print(f"{'='*60}")

        t0 = time.time()
        model, train_metrics = train_config(
            features, labels, train_mask, val_mask,
            epochs=50, batch_size=2048, lr=lr,
            d_model=dm, depth=dep, dropout=dr,
            gate_pos_weight=gpw, lookback=lb, device=device,
        )
        train_time = time.time() - t0
        print(f"  Trained in {train_time:.1f}s, val_loss={train_metrics['val_loss']:.4f}")

        # Save model
        torch.save(model.state_dict(), f"v2/sweep_results/{name}.pt")

        # Evaluate at multiple gate thresholds
        best_result = None
        for gate in [0.3, 0.4, 0.5, 0.6, 0.7]:
            t1 = time.time()
            m = fast_replay(model, data, max_days=60, min_gate_prob=gate, device=device)
            eval_time = time.time() - t1

            line = (f"  gate={gate}: PF={m.profit_factor:.2f} WR={m.win_rate:.1%} "
                    f"TPD={m.trades_per_day:.2f} Score={m.score:.4f} "
                    f"C={m.call_count} P={m.put_count} ({eval_time:.1f}s)")
            print(line)

            if best_result is None or m.score > best_result['score']:
                best_result = {
                    'name': name, 'gate': gate,
                    'pf': m.profit_factor, 'wr': m.win_rate,
                    'tpd': m.trades_per_day, 'score': m.score,
                    'calls': m.call_count, 'puts': m.put_count,
                    'dd': m.max_drawdown, 'trades': m.total_trades,
                    'sl': m.stop_loss_count, 'tp': m.take_profit_count,
                }

        best_result['val_loss'] = train_metrics['val_loss']
        best_result['train_time'] = round(train_time, 1)
        results.append(best_result)

    # Final report
    print(f"\n\n{'='*70}")
    print(f"  SWEEP RESULTS ({len(results)} configs)")
    print(f"{'='*70}")

    results.sort(key=lambda r: r['score'], reverse=True)

    print(f"\n{'Name':<10} {'Score':>8} {'PF':>8} {'WR':>6} {'TPD':>6} {'Gate':>5} "
          f"{'C':>4} {'P':>4} {'DD':>6} {'VL':>8} {'Time':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<10} {r['score']:>8.4f} {r['pf']:>8.3f} {r['wr']:>5.1%} "
              f"{r['tpd']:>6.2f} {r['gate']:>5.1f} "
              f"{r['calls']:>4} {r['puts']:>4} {r['dd']:>5.1%} "
              f"{r['val_loss']:>8.4f} {r['train_time']:>5.0f}s")

    with open("v2/sweep_results/fast_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    if results and results[0]['score'] > 0:
        print(f"\n*** BEST: {results[0]['name']} Score={results[0]['score']:.4f} "
              f"PF={results[0]['pf']:.3f} TPD={results[0]['tpd']:.2f}")
    else:
        print("\n*** No config achieved positive score. Need structural changes.")


if __name__ == "__main__":
    main()
