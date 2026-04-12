"""Local v3 experiment runner."""
from __future__ import annotations

import argparse
import os

from v3.core.artifact import utc_now_iso
from v3.core.env import RewardConfig
from v3.core.market_state import DEFAULT_MARKET_STATE_PATH
from v3.core.metrics import beats_all_baselines
from v3.train import TrainConfig, run_training


def _append_result(path: str, row: list[str]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a") as f:
        f.write("\t".join(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one v3 PPO experiment and log results")
    parser.add_argument("--id", required=True, help="Experiment id")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--market-state", default=DEFAULT_MARKET_STATE_PATH)
    parser.add_argument("--train-mask", default="train_mask")
    parser.add_argument("--eval-mask", default="promote_mask")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--rollout-days", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.20)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--eval-interval", type=int, default=2)
    parser.add_argument("--max-eval-days", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--drawdown-penalty", type=float, default=0.10)
    parser.add_argument("--terminal-drawdown-penalty", type=float, default=0.10)
    parser.add_argument("--action-penalty", type=float, default=0.0002)
    parser.add_argument("--resize-penalty", type=float, default=0.0001)
    parser.add_argument("--terminal-pnl-bonus", type=float, default=0.10)
    parser.add_argument("--trade-exploration-bonus", type=float, default=0.0)
    args = parser.parse_args()

    reward_config = RewardConfig(
        drawdown_penalty=args.drawdown_penalty,
        terminal_drawdown_penalty=args.terminal_drawdown_penalty,
        action_penalty=args.action_penalty,
        resize_penalty=args.resize_penalty,
        terminal_pnl_bonus=args.terminal_pnl_bonus,
        trade_exploration_bonus=args.trade_exploration_bonus,
    )
    artifact_dir = os.path.join("v3", "artifacts", args.id)
    checkpoint_path = os.path.join(artifact_dir, "checkpoint.pt")
    cfg = TrainConfig(
        data_path=args.data,
        market_state_path=args.market_state,
        train_mask=args.train_mask,
        eval_mask=args.eval_mask,
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
        experiment_id=args.id,
        seed=args.seed,
        updates=args.updates,
        patience=args.patience,
        rollout_days=args.rollout_days,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_interval=args.eval_interval,
        max_eval_days=args.max_eval_days,
        device=args.device,
    )
    checkpoint, metrics, baselines, _ = run_training(cfg, reward_config=reward_config)
    baseline_pass = beats_all_baselines(metrics, baselines)
    status = "pass" if (metrics.passed and baseline_pass) else "fail"
    _append_result(
        "v3/results.tsv",
        [
            args.id,
            utc_now_iso(),
            args.eval_mask,
            f"{metrics.score:.6f}",
            f"{metrics.total_return_pct:.6f}",
            f"{metrics.max_drawdown_pct:.6f}",
            f"{metrics.daily_sortino:.6f}",
            str(metrics.total_trades),
            status,
        ],
    )
    print(
        f"{args.id}: score={metrics.score:.4f} return={metrics.total_return_pct:.4f} "
        f"dd={metrics.max_drawdown_pct:.4f} trades={metrics.total_trades} "
        f"beats_baselines={baseline_pass} status={status}"
    )


if __name__ == "__main__":
    main()
