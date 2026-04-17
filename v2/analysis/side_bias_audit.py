"""Side Bias Audit — Where does the 82% call bias enter?

Measures call share at each stage of the scoring pipeline on exp_165 checkpoint:
1. Raw dual-head outputs (before centering)
2. After per-side centering + put_bias
3. Final chosen contract (what replay picks)
4. Oracle best contract (training signal)
5. Per-side score distributions (mean, std, peakiness)

Usage:
    python3 -m v2.analysis.side_bias_audit [--model v2/models/model.pt]
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import torch

from v2.core.chain_data import (
    QUALITY_PARTIAL,
    load_sidecar_cached,
    padded_snapshot,
)
from v2.core.policy import DecisionPolicy
from v2.replay import load_model_from_path

warnings.filterwarnings("ignore")


def load_and_infer(model_path: str):
    """Load model, run inference on promote_mask bars, return outputs + metadata."""
    model = load_model_from_path(model_path)
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    policy = DecisionPolicy()

    features = data["X"].numpy()
    mask = data["promote_mask"].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    sidecar_dir = "v2/data_sidecars"
    max_contracts = data.get("metadata", {}).get("max_contracts_per_bar", 285)

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
    all_labels = np.stack([s[1] for s in snapshots]).astype(np.float32)

    model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(all_windows).float()
        batch_c = torch.from_numpy(all_contracts).float()
        outputs = model(batch_x, batch_c)
        outputs = {k: v.cpu().numpy() for k, v in outputs.items()}

    return outputs, all_contracts, all_labels, eligible


def audit(model_path: str):
    print(f"Side Bias Audit — checkpoint: {model_path}")
    print("=" * 70)

    outputs, all_contracts, all_labels, eligible = load_and_infer(model_path)

    n_bars = len(eligible)
    contract_scores = outputs["contract_scores"]      # after centering
    call_scores_raw = outputs["call_scores_raw"]       # before centering
    put_scores_raw = outputs["put_scores_raw"]         # before centering
    valid_mask = outputs["valid_mask"].astype(bool)
    is_put = outputs["is_put"].astype(bool)
    opp_logit = outputs["opportunity_logit"]

    # Load put_bias from checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    put_bias_val = float(sd["put_bias"].item()) if "put_bias" in sd else 0.0
    print(f"Learned put_bias: {put_bias_val:.6f}")

    # ── 1. Raw-head call share (pre-bias, pre-centering) ─────────────
    # For each bar, compose raw scores: use call_scores for calls, put_scores for puts
    # Then pick argmax. This shows what the dual heads prefer BEFORE any adjustment.
    raw_call_chosen = 0
    raw_put_chosen = 0
    raw_call_max_vals = []
    raw_put_max_vals = []

    # Also track raw + put_bias stage
    raw_bias_call_chosen = 0
    raw_bias_put_chosen = 0

    for i in range(n_bars):
        vm = valid_mask[i]
        ip = is_put[i]
        if not vm.any():
            continue

        # Raw scores (no centering, no put_bias)
        raw = np.where(ip, put_scores_raw[i], call_scores_raw[i])
        raw[~vm] = -1e9
        best = int(np.argmax(raw))
        if ip[best]:
            raw_put_chosen += 1
        else:
            raw_call_chosen += 1

        # Raw + put_bias (pre-centering)
        raw_biased = np.where(ip, put_scores_raw[i] + put_bias_val, call_scores_raw[i])
        raw_biased[~vm] = -1e9
        best_biased = int(np.argmax(raw_biased))
        if ip[best_biased]:
            raw_bias_put_chosen += 1
        else:
            raw_bias_call_chosen += 1

        # Per-side max values
        call_mask = (~ip) & vm
        put_mask = ip & vm
        if call_mask.any():
            raw_call_max_vals.append(float(call_scores_raw[i][call_mask].max()))
        if put_mask.any():
            raw_put_max_vals.append(float(put_scores_raw[i][put_mask].max()))

    raw_total = raw_call_chosen + raw_put_chosen
    raw_call_pct = 100 * raw_call_chosen / raw_total if raw_total > 0 else 0
    raw_bias_total = raw_bias_call_chosen + raw_bias_put_chosen
    raw_bias_call_pct = 100 * raw_bias_call_chosen / raw_bias_total if raw_bias_total > 0 else 0

    # ── 2. Post-centering call share ──────────────────────────────────
    centered_call_chosen = 0
    centered_put_chosen = 0

    for i in range(n_bars):
        vm = valid_mask[i]
        if not vm.any():
            continue
        cs = contract_scores[i].copy()
        cs[~vm] = -1e9
        best = int(np.argmax(cs))
        if is_put[i][best]:
            centered_put_chosen += 1
        else:
            centered_call_chosen += 1

    cent_total = centered_call_chosen + centered_put_chosen
    cent_call_pct = 100 * centered_call_chosen / cent_total if cent_total > 0 else 0

    # ── 3. Final chosen (with quality filter, matching replay) ────────
    final_call_chosen = 0
    final_put_chosen = 0

    for i in range(n_bars):
        vm = valid_mask[i]
        if not vm.any():
            continue
        # Apply quality filter like replay does
        quality = all_contracts[i, :, 14]
        cs = contract_scores[i].copy()
        cs[~vm] = -1e9
        cs[quality < QUALITY_PARTIAL] = -1e9
        best = int(np.argmax(cs))
        if cs[best] <= -1e8:
            continue
        if is_put[i][best]:
            final_put_chosen += 1
        else:
            final_call_chosen += 1

    final_total = final_call_chosen + final_put_chosen
    final_call_pct = 100 * final_call_chosen / final_total if final_total > 0 else 0

    # ── 4. Oracle call/put split ──────────────────────────────────────
    oracle_call = 0
    oracle_put = 0

    for i in range(n_bars):
        vm = valid_mask[i]
        if not vm.any():
            continue
        ol = all_labels[i].copy()
        ol[~vm] = -1e9
        ol[~np.isfinite(ol)] = -1e9
        best = int(np.argmax(ol))
        if ol[best] <= -1e8:
            continue
        if is_put[i][best]:
            oracle_put += 1
        else:
            oracle_call += 1

    oracle_total = oracle_call + oracle_put
    oracle_call_pct = 100 * oracle_call / oracle_total if oracle_total > 0 else 0

    # ── 5. Per-side score distributions ───────────────────────────────
    all_call_raw = []
    all_put_raw = []
    all_call_centered = []
    all_put_centered = []

    for i in range(n_bars):
        vm = valid_mask[i]
        ip = is_put[i]
        call_m = (~ip) & vm
        put_m = ip & vm

        if call_m.any():
            all_call_raw.extend(call_scores_raw[i][call_m].tolist())
            all_call_centered.extend(contract_scores[i][call_m].tolist())
        if put_m.any():
            all_put_raw.extend(put_scores_raw[i][put_m].tolist())
            all_put_centered.extend(contract_scores[i][put_m].tolist())

    all_call_raw = np.array(all_call_raw)
    all_put_raw = np.array(all_put_raw)
    all_call_centered = np.array(all_call_centered)
    all_put_centered = np.array(all_put_centered)

    # Peakiness: std of per-bar max scores
    call_max_arr = np.array(raw_call_max_vals)
    put_max_arr = np.array(raw_put_max_vals)

    # ── 6. Per-bar: which side has higher max raw score? ──────────────
    raw_call_max_wins = 0
    raw_put_max_wins = 0

    for i in range(n_bars):
        vm = valid_mask[i]
        ip = is_put[i]
        call_m = (~ip) & vm
        put_m = ip & vm
        if not (call_m.any() and put_m.any()):
            continue
        call_max = float(call_scores_raw[i][call_m].max())
        put_max = float(put_scores_raw[i][put_m].max())
        if call_max > put_max:
            raw_call_max_wins += 1
        else:
            raw_put_max_wins += 1

    max_total = raw_call_max_wins + raw_put_max_wins
    raw_max_call_pct = 100 * raw_call_max_wins / max_total if max_total > 0 else 0

    # ── Print results ─────────────────────────────────────────────────
    print(f"\nBars analyzed: {n_bars}")
    print(f"Oracle split: {oracle_call}C / {oracle_put}P ({oracle_call_pct:.1f}% calls)")

    print(f"\n{'Stage':<40} {'Calls':>6} {'Puts':>6} {'Call%':>7}")
    print("-" * 62)
    print(f"{'Oracle best contract':<40} {oracle_call:>6} {oracle_put:>6} {oracle_call_pct:>6.1f}%")
    print(f"{'Raw dual-head (pre-bias, pre-center)':<40} {raw_call_chosen:>6} {raw_put_chosen:>6} {raw_call_pct:>6.1f}%")
    print(f"{'Raw + put_bias (pre-center)':<40} {raw_bias_call_chosen:>6} {raw_bias_put_chosen:>6} {raw_bias_call_pct:>6.1f}%")
    print(f"{'Per-bar: which side has higher max raw':<40} {raw_call_max_wins:>6} {raw_put_max_wins:>6} {raw_max_call_pct:>6.1f}%")
    print(f"{'Post-centering argmax':<40} {centered_call_chosen:>6} {centered_put_chosen:>6} {cent_call_pct:>6.1f}%")
    print(f"{'Final (+ quality filter)':<40} {final_call_chosen:>6} {final_put_chosen:>6} {final_call_pct:>6.1f}%")

    print(f"\n{'Score distributions':<25} {'Mean':>8} {'Std':>8} {'Max-mean':>10} {'Max-std':>10}")
    print("-" * 65)
    print(f"{'Call raw':<25} {all_call_raw.mean():>8.4f} {all_call_raw.std():>8.4f} {call_max_arr.mean():>10.4f} {call_max_arr.std():>10.4f}")
    print(f"{'Put raw':<25} {all_put_raw.mean():>8.4f} {all_put_raw.std():>8.4f} {put_max_arr.mean():>10.4f} {put_max_arr.std():>10.4f}")
    print(f"{'Call centered':<25} {all_call_centered.mean():>8.4f} {all_call_centered.std():>8.4f}")
    print(f"{'Put centered':<25} {all_put_centered.mean():>8.4f} {all_put_centered.std():>8.4f}")

    # ── Interpretation ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if raw_call_pct > cent_call_pct + 2:
        print(f"  Raw call share ({raw_call_pct:.1f}%) > centered ({cent_call_pct:.1f}%)")
        print("  → Centering CORRECTS some call bias. Global centering would make it worse.")
        print("  → Skip global centering ablation. Focus on soft side prior or unified scorer.")
    elif cent_call_pct > raw_call_pct + 2:
        print(f"  Centered call share ({cent_call_pct:.1f}%) > raw ({raw_call_pct:.1f}%)")
        print("  → Centering AMPLIFIES call bias. Global centering ablation is worth testing.")
    else:
        print(f"  Raw ({raw_call_pct:.1f}%) ≈ centered ({cent_call_pct:.1f}%)")
        print("  → Centering is neutral. Bias originates in the dual heads themselves.")
        print("  → Unified scorer or soft side prior are the right interventions.")

    if oracle_call_pct > 55:
        print(f"\n  Oracle is {oracle_call_pct:.1f}% calls — training signal itself is call-skewed.")
        print("  → Some model call bias is data-driven, not purely architectural.")
    elif oracle_call_pct < 45:
        print(f"\n  Oracle is {oracle_call_pct:.1f}% calls — training signal favors puts.")
    else:
        print(f"\n  Oracle is {oracle_call_pct:.1f}% calls — training signal is roughly balanced.")

    # Check if raw put scores have lower peakiness
    call_peak = call_max_arr.mean() - all_call_raw.mean()
    put_peak = put_max_arr.mean() - all_put_raw.mean()
    print(f"\n  Call peakiness (max - mean): {call_peak:.4f}")
    print(f"  Put peakiness (max - mean):  {put_peak:.4f}")
    if call_peak > put_peak * 1.2:
        print("  → Call head produces more peaked distributions → calls win argmax more often.")
    elif put_peak > call_peak * 1.2:
        print("  → Put head produces more peaked distributions → but calls still win?")
    else:
        print("  → Similar peakiness between sides.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="v2/models/model.pt")
    args = parser.parse_args()
    audit(args.model)
