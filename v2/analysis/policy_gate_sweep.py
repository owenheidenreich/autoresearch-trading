"""Sweep replay performance across policy gate thresholds.

This is a policy-level diagnostic: hold the checkpoint fixed, vary only
`DecisionPolicy.gate_threshold`, and measure full replay metrics on one or
more dataset masks.

Usage:
    python3 -m v2.analysis.policy_gate_sweep
    python3 -m v2.analysis.policy_gate_sweep --mask val --mask promote
    python3 -m v2.analysis.policy_gate_sweep --thresholds 0.0,0.24,0.30,0.32,0.34
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from v2.core.policy import DEFAULT_POLICY
from v2.replay import load_model_from_path, replay_validation


DEFAULT_THRESHOLDS = [0.0, 0.24, 0.30, 0.32, 0.34]


def _mask_key(name: str) -> str:
    if name.endswith("_mask"):
        return name
    return f"{name}_mask"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay sweep over gate thresholds.")
    parser.add_argument("--model", default="v2/models/model.pt")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument(
        "--mask",
        action="append",
        default=None,
        help="Mask name without _mask suffix. Can be passed multiple times.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
        help="Comma-separated gate thresholds to replay.",
    )
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    mask_names = args.mask if args.mask is not None else ["val", "promote"]
    masks = [_mask_key(name) for name in mask_names]

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    model = load_model_from_path(args.model)

    for mask in masks:
        print(f"\nMASK {mask}")
        print("threshold\tscore\tpf\tdd\ttrades\ttpd\twr\tcall_pct\tgate_failure")
        for threshold in thresholds:
            policy = replace(DEFAULT_POLICY, gate_threshold=threshold)
            metrics, _, _ = replay_validation(model, data, mask_key=mask, policy=policy)
            print(
                f"{threshold:.2f}\t"
                f"{metrics.score:.4f}\t"
                f"{metrics.profit_factor:.3f}\t"
                f"{metrics.max_account_drawdown:.3f}\t"
                f"{metrics.total_trades}\t"
                f"{metrics.trades_per_day:.2f}\t"
                f"{metrics.win_rate:.3f}\t"
                f"{metrics.call_pct:.3f}\t"
                f"{metrics.gate_failure}"
            )


if __name__ == "__main__":
    main()
