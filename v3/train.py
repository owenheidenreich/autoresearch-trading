"""Repo-native PPO training loop for v3."""
from __future__ import annotations

import argparse
import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from v3.core.artifact import create_artifact, save_checkpoint, utc_now_iso
from v3.core.data import load_data_bundle
from v3.core.env import ExactChainEnv, RewardConfig
from v3.core.execution import ExecutionConfig
from v3.core.market_state import DEFAULT_MARKET_STATE_PATH
from v3.core.metrics import EvalConfig, ReplayMetrics, beats_all_baselines
from v3.core.ppo import RolloutBuffer
from v3.core.schema import MODEL_SCHEMA_VERSION, RLArtifactManifest
from v3.model import RLPolicy, observation_to_tensors
from v3.replay import evaluate_baselines, evaluate_policy


@dataclass
class TrainConfig:
    data_path: str = "v2/data.pt"
    market_state_path: str = DEFAULT_MARKET_STATE_PATH
    train_mask: str = "train_mask"
    eval_mask: str = "val_mask"
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    updates: int = 10
    patience: int = 0
    rollout_days: int = 8
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.50
    eval_interval: int = 2
    max_eval_days: int | None = None
    checkpoint_path: str = "v3/models/policy_latest.pt"
    artifact_dir: str = ""
    experiment_id: str = "v3_rl_dev"

    def to_dict(self) -> dict:
        return asdict(self)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_days(days: list[str], count: int) -> list[str]:
    if len(days) <= count:
        return list(days)
    return random.sample(days, count)


def _collect_rollout(
    *,
    policy: RLPolicy,
    bundle,
    train_days: list[str],
    train_config: TrainConfig,
    execution_config: ExecutionConfig,
    reward_config: RewardConfig,
    device: torch.device,
) -> RolloutBuffer:
    buffer = RolloutBuffer()
    sampled_days = _sample_days(train_days, train_config.rollout_days)
    policy.train()
    for day in sampled_days:
        env = ExactChainEnv(
            bundle.episode_for_day(day),
            execution_config=execution_config,
            reward_config=reward_config,
        )
        obs = env.reset()
        while True:
            obs_t = observation_to_tensors(obs, device)
            with torch.no_grad():
                action_t, log_prob, value, aux = policy.act(obs_t, deterministic=False)
            env_action = policy.to_agent_action(obs, action_t, aux)
            next_obs, reward, done, _ = env.step(env_action)
            buffer.add(
                observation=obs_t,
                action=action_t,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=done,
            )
            obs = next_obs
            if done:
                break
    return buffer


def run_training(
    train_config: TrainConfig,
    *,
    execution_config: ExecutionConfig | None = None,
    reward_config: RewardConfig | None = None,
    eval_config: EvalConfig | None = None,
    model_config: dict | None = None,
) -> tuple[dict, ReplayMetrics, dict[str, ReplayMetrics], list]:
    execution_config = execution_config or ExecutionConfig()
    reward_config = reward_config or RewardConfig()
    eval_config = eval_config or EvalConfig()
    model_config = model_config or {}

    _set_seed(train_config.seed)
    device = torch.device(train_config.device)
    bundle = load_data_bundle(train_config.data_path, market_state_path=train_config.market_state_path)
    train_days = bundle.days_for_mask(train_config.train_mask)

    policy = RLPolicy(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    best_score = float("-inf")
    best_checkpoint: dict | None = None
    best_metrics: ReplayMetrics | None = None
    best_baselines: dict[str, ReplayMetrics] = {}
    best_traces = []
    history: list[dict] = []
    evals_without_improvement = 0

    for update in range(1, train_config.updates + 1):
        buffer = _collect_rollout(
            policy=policy,
            bundle=bundle,
            train_days=train_days,
            train_config=train_config,
            execution_config=execution_config,
            reward_config=reward_config,
            device=device,
        )
        batch = buffer.as_batch(gamma=train_config.gamma, gae_lambda=train_config.gae_lambda, device=device)
        policy.train()
        for _ in range(train_config.ppo_epochs):
            for obs_mb, act_mb, old_log_prob_mb, old_value_mb, returns_mb, adv_mb in batch.iter_minibatches(train_config.minibatch_size):
                new_log_prob, entropy, values = policy.evaluate_actions(obs_mb, act_mb)
                ratio = (new_log_prob - old_log_prob_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - train_config.clip_coef, 1.0 + train_config.clip_coef) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns_mb)
                entropy_loss = entropy.mean()
                loss = policy_loss + train_config.value_coef * value_loss - train_config.entropy_coef * entropy_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), train_config.max_grad_norm)
                optimizer.step()

        if update % train_config.eval_interval == 0 or update == train_config.updates:
            policy.eval()
            metrics, _, _, traces, _, _ = evaluate_policy(
                policy,
                bundle,
                mask_key=train_config.eval_mask,
                execution_config=execution_config,
                reward_config=reward_config,
                eval_config=eval_config,
                device=device,
                deterministic=True,
                max_days=train_config.max_eval_days,
            )
            baseline_metrics = evaluate_baselines(
                bundle,
                mask_key=train_config.eval_mask,
                execution_config=execution_config,
                reward_config=reward_config,
                eval_config=eval_config,
                max_days=train_config.max_eval_days,
            )
            history.append(
                {
                    "update": update,
                    "score": metrics.score,
                    "return": metrics.total_return_pct,
                    "drawdown": metrics.max_drawdown_pct,
                }
            )
            print(
                f"update={update} score={metrics.score:.4f} return={metrics.total_return_pct:.4f} "
                f"dd={metrics.max_drawdown_pct:.4f} trades={metrics.total_trades}",
                flush=True,
            )
            score_improved = metrics.score > best_score
            score_tied = (metrics.score == best_score) and best_metrics is not None
            tiebreak_won = score_tied and (
                metrics.max_drawdown_pct < best_metrics.max_drawdown_pct
                or (metrics.max_drawdown_pct == best_metrics.max_drawdown_pct and metrics.total_return_pct > best_metrics.total_return_pct)
            )
            if score_improved or tiebreak_won:
                best_score = metrics.score
                best_metrics = copy.deepcopy(metrics)
                best_baselines = {k: copy.deepcopy(v) for k, v in baseline_metrics.items()}
                best_traces = traces
                best_checkpoint = {
                    "model_schema_version": MODEL_SCHEMA_VERSION,
                    "model_state_dict": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
                    "model_config": model_config,
                    "training_config": train_config.to_dict(),
                    "execution_config": execution_config.to_dict(),
                    "reward_config": reward_config.to_dict(),
                    "eval_config": eval_config.to_dict(),
                    "history": history,
                    "dataset_fingerprint": bundle.dataset_fingerprint,
                    "best_metrics": best_metrics.to_dict(),
                    "baseline_metrics": {k: v.to_dict() for k, v in best_baselines.items()},
                    "beats_all_baselines": beats_all_baselines(best_metrics, best_baselines),
                }
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            if train_config.patience > 0 and evals_without_improvement >= train_config.patience:
                print(f"early stop: no improvement for {evals_without_improvement} evals", flush=True)
                break

    if best_checkpoint is None or best_metrics is None:
        raise RuntimeError("training completed without an evaluation checkpoint")

    save_checkpoint(train_config.checkpoint_path, best_checkpoint)

    if train_config.artifact_dir:
        manifest = RLArtifactManifest(
            experiment_id=train_config.experiment_id,
            created_at=utc_now_iso(),
            dataset_fingerprint=bundle.dataset_fingerprint,
            checkpoint_path=train_config.checkpoint_path,
            training_config=train_config.to_dict(),
            reward_config=reward_config.to_dict(),
            execution_config=execution_config.to_dict(),
            evaluation_metrics=best_metrics.to_dict(),
            baseline_metrics={k: v.to_dict() for k, v in best_baselines.items()},
            trace_path=f"{train_config.artifact_dir}/replay_traces.parquet",
        )
        create_artifact(
            artifact_dir=train_config.artifact_dir,
            checkpoint_payload=best_checkpoint,
            manifest=manifest,
            traces=best_traces,
        )

    return best_checkpoint, best_metrics, best_baselines, best_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a v3 PPO trading policy")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--market-state", default=DEFAULT_MARKET_STATE_PATH)
    parser.add_argument("--train-mask", default="train_mask")
    parser.add_argument("--eval-mask", default="val_mask")
    parser.add_argument("--checkpoint", default="v3/models/policy_latest.pt")
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--experiment-id", default="v3_rl_dev")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--patience", type=int, default=0, help="Early stop after N evals without improvement (0=disabled)")
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_path=args.data,
        market_state_path=args.market_state,
        train_mask=args.train_mask,
        eval_mask=args.eval_mask,
        checkpoint_path=args.checkpoint,
        artifact_dir=args.artifact_dir,
        experiment_id=args.experiment_id,
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
    _, metrics, baselines, _ = run_training(cfg)
    baseline_pass = beats_all_baselines(metrics, baselines)
    print(
        f"best score={metrics.score:.4f} return={metrics.total_return_pct:.4f} "
        f"dd={metrics.max_drawdown_pct:.4f} beats_baselines={baseline_pass} "
        f"baselines={','.join(sorted(baselines))}"
    )


if __name__ == "__main__":
    main()
