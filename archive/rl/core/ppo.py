"""PPO rollout utilities for v3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute returns and GAE advantages for completed episodes."""

    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    last_value = torch.zeros((), dtype=values.dtype, device=values.device)
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * last_value * non_terminal - values[t]
        last_adv = delta + gamma * gae_lambda * non_terminal * last_adv
        advantages[t] = last_adv
        last_value = values[t]
    returns = advantages + values
    return returns, advantages


@dataclass
class RolloutBatch:
    """Stacked PPO rollout tensors."""

    observations: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor]
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def iter_minibatches(self, batch_size: int) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        idx = torch.randperm(self.old_log_prob.shape[0])
        for start in range(0, len(idx), batch_size):
            sel = idx[start : start + batch_size]
            obs = {k: v[sel] for k, v in self.observations.items()}
            actions = {k: v[sel] for k, v in self.actions.items()}
            yield (
                obs,
                actions,
                self.old_log_prob[sel],
                self.old_value[sel],
                self.returns[sel],
                self.advantages[sel],
            )


class RolloutBuffer:
    """Simple CPU-side rollout storage."""

    def __init__(self) -> None:
        self.obs: list[dict[str, torch.Tensor]] = []
        self.actions: list[dict[str, torch.Tensor]] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []

    def add(
        self,
        *,
        observation: dict[str, torch.Tensor],
        action: dict[str, torch.Tensor],
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        self.obs.append({k: v.detach().cpu() for k, v in observation.items()})
        self.actions.append({k: v.detach().cpu() for k, v in action.items()})
        self.log_probs.append(log_prob.detach().cpu().reshape(()))
        self.values.append(value.detach().cpu().reshape(()))
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def as_batch(self, *, gamma: float, gae_lambda: float, device: torch.device) -> RolloutBatch:
        obs = {k: torch.cat([step[k] for step in self.obs], dim=0).to(device) for k in self.obs[0]}
        actions = {k: torch.cat([step[k] for step in self.actions], dim=0).to(device) for k in self.actions[0]}
        old_log_prob = torch.stack(self.log_probs).to(device)
        old_value = torch.stack(self.values).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        returns, advantages = compute_gae(rewards, old_value, dones, gamma=gamma, gae_lambda=gae_lambda)
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        return RolloutBatch(
            observations=obs,
            actions=actions,
            old_log_prob=old_log_prob,
            old_value=old_value,
            rewards=rewards,
            dones=dones,
            returns=returns,
            advantages=advantages,
        )

