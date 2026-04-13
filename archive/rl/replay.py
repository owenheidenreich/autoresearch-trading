"""Deterministic replay and evaluation for v3 RL artifacts."""
from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from v3.core.artifact import resolve_checkpoint_path, save_traces
from v3.core.baselines import ATMFixedAgent, NoTradeAgent, SimpleRulesFixedAgent, V2BestStaticAgent
from v3.core.data import RLDataBundle, load_data_bundle
from v3.core.env import ExactChainEnv, RewardConfig
from v3.core.execution import ExecutionConfig
from v3.core.market_state import DEFAULT_MARKET_STATE_PATH
from v3.core.metrics import EvalConfig, ReplayMetrics, compute_metrics
from v3.core.schema import MODEL_SCHEMA_VERSION, EpisodeSummary, StepTrace, TradeRecord
from v3.model import RLPolicy, observation_to_tensors


def load_policy_from_checkpoint(path: str, device: str = "cpu") -> tuple[RLPolicy, dict[str, Any]]:
    checkpoint = torch.load(resolve_checkpoint_path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema {checkpoint.get('model_schema_version')} is incompatible with "
            f"current v3 schema {MODEL_SCHEMA_VERSION}; retrain under the new multiscale model"
        )
    model_config = checkpoint.get("model_config", {})
    policy = RLPolicy(**model_config).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy, checkpoint


def run_policy_episode(
    policy: RLPolicy,
    env: ExactChainEnv,
    *,
    device: torch.device,
    deterministic: bool,
) -> tuple[EpisodeSummary, list[TradeRecord], list[StepTrace], list[float], list[float]]:
    traces: list[StepTrace] = []
    was_training = policy.training
    policy.eval()
    obs = env.reset()
    while True:
        obs_t = observation_to_tensors(obs, device)
        with torch.no_grad():
            action_t, _, _, aux = policy.act(obs_t, deterministic=deterministic)
        action = policy.to_agent_action(obs, action_t, aux)
        next_obs, _, done, info = env.step(action)
        traces.append(
            env.build_step_trace(
                obs=obs,
                action=action,
                execution=info["execution"],
                pointer_scores=[float(x) for x in aux["contract_logits"][0].detach().cpu().tolist()],
                sampled_action={
                    "flat_action": int(action_t["flat_action"][0].item()),
                    "manage_action": int(action_t["manage_action"][0].item()),
                    "contract_idx": int(action_t["contract_idx"][0].item()),
                    "exit_style": int(action_t["exit_style"][0].item()),
                    "continuous": [float(x) for x in action_t["continuous"][0].detach().cpu().tolist()],
                },
            )
        )
        obs = next_obs
        if done:
            break
    if was_training:
        policy.train()
    return env.summary(), list(env.trade_records), traces, list(env.action_confidences), list(env.action_rewards)


def run_baseline_episode(agent: Any, env: ExactChainEnv) -> tuple[EpisodeSummary, list[TradeRecord], list[StepTrace], list[float], list[float]]:
    traces: list[StepTrace] = []
    agent.reset()
    obs = env.reset()
    while True:
        action = agent.act(obs)
        next_obs, _, done, info = env.step(action)
        traces.append(
            env.build_step_trace(
                obs=obs,
                action=action,
                execution=info["execution"],
                pointer_scores=[],
                sampled_action={"baseline": agent.name},
            )
        )
        obs = next_obs
        if done:
            break
    return env.summary(), list(env.trade_records), traces, list(env.action_confidences), list(env.action_rewards)


def evaluate_policy(
    policy: RLPolicy,
    bundle: RLDataBundle,
    *,
    mask_key: str,
    execution_config: ExecutionConfig,
    reward_config: RewardConfig,
    eval_config: EvalConfig | None = None,
    device: torch.device,
    deterministic: bool = True,
    max_days: int | None = None,
) -> tuple[ReplayMetrics, list[EpisodeSummary], list[TradeRecord], list[StepTrace], list[float], list[float]]:
    eval_config = eval_config or EvalConfig()
    days = bundle.days_for_mask(mask_key)
    if max_days is not None:
        days = days[:max_days]
    episodes: list[EpisodeSummary] = []
    trades: list[TradeRecord] = []
    traces: list[StepTrace] = []
    confidences: list[float] = []
    rewards: list[float] = []
    for day in days:
        env = ExactChainEnv(
            bundle.episode_for_day(day),
            execution_config=execution_config,
            reward_config=reward_config,
        )
        ep, ep_trades, ep_traces, ep_conf, ep_rewards = run_policy_episode(
            policy,
            env,
            device=device,
            deterministic=deterministic,
        )
        episodes.append(ep)
        trades.extend(ep_trades)
        traces.extend(ep_traces)
        confidences.extend(ep_conf)
        rewards.extend(ep_rewards)
    metrics = compute_metrics(episodes, trades, action_confidences=confidences, action_rewards=rewards, config=eval_config)
    return metrics, episodes, trades, traces, confidences, rewards


def evaluate_baselines(
    bundle: RLDataBundle,
    *,
    mask_key: str,
    execution_config: ExecutionConfig,
    reward_config: RewardConfig,
    eval_config: EvalConfig | None = None,
    max_days: int | None = None,
) -> dict[str, ReplayMetrics]:
    eval_config = eval_config or EvalConfig()
    days = bundle.days_for_mask(mask_key)
    if max_days is not None:
        days = days[:max_days]
    agents = [
        NoTradeAgent(),
        ATMFixedAgent(execution_config),
        SimpleRulesFixedAgent(execution_config),
        V2BestStaticAgent(execution_config),
    ]
    baseline_metrics: dict[str, ReplayMetrics] = {}
    for agent in agents:
        episodes: list[EpisodeSummary] = []
        trades: list[TradeRecord] = []
        confidences: list[float] = []
        rewards: list[float] = []
        for day in days:
            env = ExactChainEnv(
                bundle.episode_for_day(day),
                execution_config=execution_config,
                reward_config=reward_config,
            )
            ep, ep_trades, _, ep_conf, ep_rewards = run_baseline_episode(agent, env)
            episodes.append(ep)
            trades.extend(ep_trades)
            confidences.extend(ep_conf)
            rewards.extend(ep_rewards)
        baseline_metrics[agent.name] = compute_metrics(
            episodes,
            trades,
            action_confidences=confidences,
            action_rewards=rewards,
            config=eval_config,
        )
    return baseline_metrics


def _print_metrics(label: str, metrics: ReplayMetrics) -> None:
    print(
        f"{label}: score={metrics.score:.4f} return={metrics.total_return_pct:.4f} "
        f"dd={metrics.max_drawdown_pct:.4f} sortino={metrics.daily_sortino:.4f} trades={metrics.total_trades}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic replay for v3 PPO policies")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint or artifact checkpoint path")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--market-state", default=DEFAULT_MARKET_STATE_PATH)
    parser.add_argument("--mask", default="promote_mask")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--with-baselines", action="store_true")
    parser.add_argument("--save-traces", default="", help="Optional parquet path for deterministic replay traces")
    args = parser.parse_args()

    policy, checkpoint = load_policy_from_checkpoint(args.checkpoint, device=args.device)
    execution_config = ExecutionConfig(**checkpoint.get("execution_config", {}))
    reward_config = RewardConfig(**checkpoint.get("reward_config", {}))
    eval_config = EvalConfig(**checkpoint.get("eval_config", {}))
    bundle = load_data_bundle(args.data, market_state_path=args.market_state)

    metrics, _, _, traces, _, _ = evaluate_policy(
        policy,
        bundle,
        mask_key=args.mask,
        execution_config=execution_config,
        reward_config=reward_config,
        eval_config=eval_config,
        device=torch.device(args.device),
        deterministic=True,
        max_days=args.max_days,
    )
    _print_metrics("policy", metrics)
    if args.save_traces:
        save_traces(args.save_traces, traces)
        print(f"saved traces -> {args.save_traces}")
    if args.with_baselines:
        baseline_metrics = evaluate_baselines(
            bundle,
            mask_key=args.mask,
            execution_config=execution_config,
            reward_config=reward_config,
            eval_config=eval_config,
            max_days=args.max_days,
        )
        for name, base in baseline_metrics.items():
            _print_metrics(name, base)


if __name__ == "__main__":
    main()
