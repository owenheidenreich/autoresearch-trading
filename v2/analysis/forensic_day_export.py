"""
Forensic one-day export: dumps exactly what the model sees during training.

Usage:
    python -m v2.analysis.forensic_day_export --date 2025-09-11

Produces:
    v2/artifacts/forensic_<date>/
        day_manifest_dump.npz       – all data.pt rows for that day
        day_sidecar_dump.npz        – full sidecar tensors
        sample_dump.json            – one actual TradeDataset sample
        forward_preproc_dump.json   – forward() preprocessing trace
        day_summary.md              – human-readable summary of everything
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from v2.core.chain_data import CONTRACT_FEATURE_FIELDS, NUM_CONTRACT_FEATURES, padded_snapshot
from v2.pipeline.compute_features import ALL_FEATURE_NAMES


def export_day(date: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load data.pt ──────────────────────────────────────────────────
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    meta = data["metadata"]
    feature_names = list(data["feature_names"])
    dates = data["dates"]
    max_contracts = int(meta["max_contracts_per_bar"])

    # Find rows for this day
    day_indices = [i for i, d in enumerate(dates) if d == date]
    if not day_indices:
        print(f"ERROR: date {date} not found in data.pt")
        print(f"  Available date range: {dates[0]} .. {dates[-1]}")
        sys.exit(1)

    n_bars = len(day_indices)
    gi = np.array(day_indices)
    print(f"Date {date}: {n_bars} bars, global indices {gi[0]}..{gi[-1]}")

    # ── 1a. Manifest dump ────────────────────────────────────────────────
    manifest = {
        "global_index": gi,
        "bar_of_day": data["bar_of_day"][gi].numpy(),
        "X": data["X"][gi].numpy(),
        "X_sim": data["X_sim"][gi].numpy(),
        "spot_prices": data["spot_prices"][gi].numpy(),
        "label_trade": data["label_trade"][gi].numpy(),
        "label_trade_valid": data["label_trade_valid"][gi].numpy(),
        "best_contract_pnl": data["best_contract_pnl"][gi].numpy(),
        "best_contract_strike": data["best_contract_strike"][gi].numpy(),
        "best_contract_right": data["best_contract_right"][gi].numpy(),
    }
    np.savez_compressed(os.path.join(out_dir, "day_manifest_dump.npz"), **manifest)
    print(f"  Saved day_manifest_dump.npz ({n_bars} bars x {len(feature_names)} features)")

    # Which mask does this day belong to?
    mask_membership = []
    for mname in ("train_mask", "val_mask", "promote_mask", "shadow_mask"):
        if data[mname][gi[0]].item():
            mask_membership.append(mname)
    print(f"  Mask membership: {mask_membership}")

    # ── 2. Sidecar dump ──────────────────────────────────────────────────
    sidecar_path = os.path.join(meta["chain_sidecar_dir"], f"{date}.pt")
    if not os.path.exists(sidecar_path):
        print(f"  WARNING: sidecar not found at {sidecar_path}")
        sc = None
    else:
        sc = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        sidecar_out = {}
        for key in [
            "bar_timestamps", "contract_strike", "contract_right",
            "contract_mid", "contract_bid", "contract_ask", "contract_quality",
            "row_features", "row_labels", "row_contract_idx",
            "bar_ptrs", "bar_best_contract_idx", "bar_best_pnl",
            "bar_label_trade", "bar_labelable", "bar_quality",
        ]:
            v = sc[key]
            sidecar_out[key] = v.numpy() if hasattr(v, "numpy") else np.array(v)
        np.savez_compressed(os.path.join(out_dir, "day_sidecar_dump.npz"), **sidecar_out)
        n_contracts = len(sc["contract_strike"])
        n_rows = len(sc["row_features"])
        print(f"  Saved day_sidecar_dump.npz ({n_contracts} contracts, {n_rows} row-features)")

    # ── 3. Schema maps ───────────────────────────────────────────────────
    schema = {
        "feature_names": feature_names,
        "contract_feature_fields": list(CONTRACT_FEATURE_FIELDS),
        "metadata": {k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                     for k, v in meta.items()},
    }
    with open(os.path.join(out_dir, "schema_maps.json"), "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Saved schema_maps.json")

    # ── 4. One actual training sample ────────────────────────────────────
    # Find first trade-labeled bar on this day that's past lookback
    lookback = int(meta.get("lookback", 30))
    sample_gi = None
    sample_local_bar = None
    for i in day_indices:
        bod = int(data["bar_of_day"][i])
        if bod >= lookback and data["label_trade"][i].item() and data["label_trade_valid"][i].item():
            sample_gi = i
            sample_local_bar = bod
            break

    if sample_gi is None:
        # Fallback: first bar past lookback
        for i in day_indices:
            if int(data["bar_of_day"][i]) >= lookback:
                sample_gi = i
                sample_local_bar = int(data["bar_of_day"][i])
                break

    if sample_gi is not None and sc is not None:
        print(f"\n  Training sample: global_idx={sample_gi}, bar_of_day={sample_local_bar}")

        # Build the window exactly as TradeDataset.__getitem__ does
        window = data["X"][sample_gi - lookback: sample_gi].numpy()  # (lookback, 52)
        contracts, labels, contract_idx = padded_snapshot(sc, sample_local_bar, max_contracts)
        best_idx = int(sc["bar_best_contract_idx"][sample_local_bar])
        label_trade = bool(sc["bar_label_trade"][sample_local_bar])
        label_valid = bool(sc["bar_labelable"][sample_local_bar])

        n_valid = int((contracts[:, 0] > 0.5).sum())

        # Compute strict opportunity label (same as training default)
        from v2.train import _compute_strict_opportunity
        strict_opp = bool(_compute_strict_opportunity(sc, sample_local_bar))

        sample_dump = {
            "global_index": int(sample_gi),
            "date": date,
            "bar_of_day": sample_local_bar,
            "spot_price": float(data["spot_prices"][sample_gi]),
            "window_shape": list(window.shape),
            "window": window.tolist(),
            "contracts_shape": list(contracts.shape),
            "contracts": contracts.tolist(),
            "contract_labels": labels.tolist(),
            "contract_indices": contract_idx.tolist(),
            "n_valid_contracts": n_valid,
            "best_idx": best_idx,
            "best_contract_strike": float(sc["contract_strike"][best_idx]) if best_idx >= 0 else None,
            "best_contract_right": "P" if best_idx >= 0 and int(sc["contract_right"][best_idx]) == 1 else "C",
            "best_contract_pnl": float(sc["bar_best_pnl"][sample_local_bar]),
            "label_trade_raw": label_trade,
            "label_trade_strict": strict_opp,
            "label_trade_valid": label_valid,
        }
        with open(os.path.join(out_dir, "sample_dump.json"), "w") as f:
            json.dump(sample_dump, f, indent=2)
        print(f"    window: {window.shape}, contracts: {contracts.shape}")
        print(f"    n_valid={n_valid}, best_idx={best_idx}, label_trade={strict_opp}, pnl={sample_dump['best_contract_pnl']:+.4f}")

        # ── 5. Forward-pass preprocessing dump ───────────────────────────
        contracts_t = torch.from_numpy(contracts).unsqueeze(0)  # (1, max_c, 22)
        raw_contracts = contracts_t.clone()

        # Reproduce TradingModel.forward() preprocessing
        c = contracts_t.clone()
        valid_mask = contracts_t[:, :, 0] > 0.5
        is_put = contracts_t[:, :, 2] > 0.5

        # Step 1: Greek sign flip for puts
        put_flip = is_put.float()
        for fidx in (8, 11, 12, 16):
            c[:, :, fidx] = c[:, :, fidx] * (1.0 - 2.0 * put_flip)
        after_flip = c.clone()

        # Step 2: Per-bar z-score
        valid_f = valid_mask.float()
        count = valid_f.sum(dim=1, keepdim=True).clamp(min=1)
        for fidx in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21):
            feat = c[:, :, fidx]
            masked = feat * valid_f
            mean = masked.sum(dim=1, keepdim=True) / count
            diff = (feat - mean) * valid_f
            std = (diff.pow(2).sum(dim=1, keepdim=True) / count).sqrt().clamp(min=1e-6)
            c[:, :, fidx] = (feat - mean) / std
        after_zscore = c.clone()

        # Step 3: Zero invalid
        c = c * valid_f.unsqueeze(-1)
        after_zero = c.clone()

        # Export only the valid contracts for readability
        n_show = min(n_valid, 30)  # cap for JSON sanity
        cf = list(CONTRACT_FEATURE_FIELDS)

        def contracts_to_labeled_list(tensor, n):
            """Convert (1, max_c, 22) tensor to list of dicts with named fields."""
            rows = []
            for j in range(n):
                row = {}
                for k, name in enumerate(cf):
                    row[name] = round(float(tensor[0, j, k]), 6)
                rows.append(row)
            return rows

        preproc_dump = {
            "sample_global_index": int(sample_gi),
            "bar_of_day": sample_local_bar,
            "n_valid_contracts": n_valid,
            "n_shown": n_show,
            "valid_mask": valid_mask[0, :n_show].tolist(),
            "is_put": is_put[0, :n_show].tolist(),
            "stage_0_raw": contracts_to_labeled_list(raw_contracts, n_show),
            "stage_1_after_put_flip": contracts_to_labeled_list(after_flip, n_show),
            "stage_2_after_zscore": contracts_to_labeled_list(after_zscore, n_show),
            "stage_3_after_zero_invalid": contracts_to_labeled_list(after_zero, n_show),
            "put_flipped_fields": ["delta (8)", "moneyness_pct (11)", "distance_points (12)", "charm (16)"],
            "zscored_fields": [f"{cf[i]} ({i})" for i in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21)],
        }
        with open(os.path.join(out_dir, "forward_preproc_dump.json"), "w") as f:
            json.dump(preproc_dump, f, indent=2)
        print(f"    Saved forward_preproc_dump.json ({n_show} contracts shown)")

    # ── 6. Human-readable summary ────────────────────────────────────────
    lines = []
    lines.append(f"# Forensic Day Export: {date}\n")
    lines.append(f"## Data.pt Manifest\n")
    lines.append(f"- Bars: {n_bars} (bar_of_day {manifest['bar_of_day'][0]}..{manifest['bar_of_day'][-1]})")
    lines.append(f"- Mask: {', '.join(mask_membership)}")
    lines.append(f"- Spot range: {manifest['spot_prices'].min():.2f} .. {manifest['spot_prices'].max():.2f}")
    trade_bars = manifest["label_trade"].sum()
    valid_bars = manifest["label_trade_valid"].sum()
    lines.append(f"- label_trade=True bars: {trade_bars} / {n_bars}")
    lines.append(f"- label_trade_valid=True bars: {valid_bars} / {n_bars}")
    pos_pnl = manifest["best_contract_pnl"][manifest["best_contract_pnl"] > 0]
    lines.append(f"- best_contract_pnl > 0: {len(pos_pnl)} bars, mean={pos_pnl.mean():.4f}" if len(pos_pnl) > 0 else "- No positive pnl bars")
    lines.append("")

    if sc is not None:
        lines.append(f"## Sidecar\n")
        lines.append(f"- Schema: {sc['schema_version']}")
        lines.append(f"- Contracts: {len(sc['contract_strike'])}")
        n_calls = int((np.array(sc["contract_right"]) == 0).sum())
        n_puts = int((np.array(sc["contract_right"]) == 1).sum())
        lines.append(f"- Calls: {n_calls}, Puts: {n_puts}")
        lines.append(f"- Row features: {sc['row_features'].shape}")
        lines.append(f"- Bars labeled trade: {int(np.array(sc['bar_label_trade']).sum())}")
        lines.append(f"- Bars labelable: {int(np.array(sc['bar_labelable']).sum())}")
        lines.append("")

        # Per-bar contract count distribution
        bar_ptrs = np.array(sc["bar_ptrs"])
        contracts_per_bar = bar_ptrs[1:] - bar_ptrs[:-1]
        lines.append(f"### Contracts per bar")
        lines.append(f"- min: {contracts_per_bar.min()}, max: {contracts_per_bar.max()}, "
                      f"mean: {contracts_per_bar.mean():.1f}, median: {np.median(contracts_per_bar):.0f}")
        lines.append("")

        # Label distribution on trade bars
        trade_bar_indices = np.where(np.array(sc["bar_label_trade"]))[0]
        if len(trade_bar_indices) > 0:
            lines.append(f"### Oracle label stats (trade bars only)")
            all_trade_labels = []
            for b in trade_bar_indices:
                start, end = int(bar_ptrs[b]), int(bar_ptrs[b + 1])
                row_labels = np.array(sc["row_labels"][start:end])
                finite = row_labels[np.isfinite(row_labels)]
                all_trade_labels.extend(finite.tolist())
            all_trade_labels = np.array(all_trade_labels)
            profitable = all_trade_labels[all_trade_labels > 0]
            losing = all_trade_labels[all_trade_labels <= 0]
            lines.append(f"- Total contract-bar rows on trade bars: {len(all_trade_labels)}")
            lines.append(f"- Profitable: {len(profitable)} ({len(profitable)/max(len(all_trade_labels),1)*100:.1f}%)")
            lines.append(f"- Losing/zero: {len(losing)} ({len(losing)/max(len(all_trade_labels),1)*100:.1f}%)")
            if len(profitable) > 0:
                lines.append(f"- Profitable pnl: mean={profitable.mean():.4f}, median={np.median(profitable):.4f}, "
                              f"max={profitable.max():.4f}")
            if len(losing) > 0:
                lines.append(f"- Losing pnl: mean={losing.mean():.4f}, median={np.median(losing):.4f}, "
                              f"min={losing.min():.4f}")
            lines.append("")

    if sample_gi is not None:
        lines.append(f"## Training Sample (global_idx={sample_gi}, bar={sample_local_bar})\n")
        lines.append(f"- Spot: {sample_dump['spot_price']:.2f}")
        lines.append(f"- Window: {sample_dump['window_shape']} (lookback={lookback})")
        lines.append(f"- Valid contracts: {sample_dump['n_valid_contracts']} / {max_contracts}")
        lines.append(f"- Best idx: {sample_dump['best_idx']} "
                      f"(strike={sample_dump['best_contract_strike']}, "
                      f"right={sample_dump['best_contract_right']}, "
                      f"pnl={sample_dump['best_contract_pnl']:+.4f})")
        lines.append(f"- label_trade (raw sidecar): {sample_dump['label_trade_raw']}")
        lines.append(f"- label_trade (strict, used in training): {sample_dump['label_trade_strict']}")
        lines.append(f"- label_trade_valid: {sample_dump['label_trade_valid']}")
        lines.append("")

        lines.append(f"### Context features (last bar of window)\n")
        last_bar_x = window[-1]
        lines.append(f"| # | Feature | Value |")
        lines.append(f"|---|---------|-------|")
        for j, name in enumerate(feature_names):
            lines.append(f"| {j} | {name} | {last_bar_x[j]:.4f} |")
        lines.append("")

        if n_valid > 0:
            lines.append(f"### Contract features (raw, first {n_show} valid)\n")
            lines.append("| # | " + " | ".join(cf) + " | label |")
            lines.append("|---" + "|---" * len(cf) + "|---|")
            for j in range(n_show):
                vals = " | ".join(f"{contracts[j, k]:.4f}" for k in range(len(cf)))
                lbl = f"{labels[j]:.4f}" if np.isfinite(labels[j]) else "NaN"
                lines.append(f"| {j} | {vals} | {lbl} |")
            lines.append("")

    lines.append(f"## Files\n")
    lines.append(f"- `day_manifest_dump.npz` — {n_bars} bars, keys: {list(manifest.keys())}")
    lines.append(f"- `day_sidecar_dump.npz` — full sidecar tensors")
    lines.append(f"- `schema_maps.json` — feature names + contract fields + metadata")
    lines.append(f"- `sample_dump.json` — one training sample with window + contracts + targets")
    lines.append(f"- `forward_preproc_dump.json` — preprocessing stages (raw → flip → zscore → zero)")
    lines.append(f"- `day_summary.md` — this file")

    summary_path = os.path.join(out_dir, "day_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved day_summary.md")
    print(f"\nAll artifacts in: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forensic one-day export")
    parser.add_argument("--date", required=True, help="Date to export (YYYY-MM-DD)")
    parser.add_argument("--out", default=None, help="Output directory (default: v2/artifacts/forensic_<date>)")
    args = parser.parse_args()
    out_dir = args.out or f"v2/artifacts/forensic_{args.date}"
    export_day(args.date, out_dir)
