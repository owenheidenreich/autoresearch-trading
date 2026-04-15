"""Offline trajectory dataset for AWAC training.

Loads pre-collected trajectory .pt files and serves
(window, contracts, session_state, action, return_to_go) tuples.
Returns-to-go are computed at load time so gamma can be tuned
without re-collecting data.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class OfflineTrajectoryDataset(Dataset):
    """Dataset of (state, action, return_to_go) tuples from offline rollouts."""

    def __init__(self, paths: list[str], gamma: float = 0.99):
        all_windows = []
        all_contracts = []
        all_sessions = []
        all_actions = []
        all_train_rewards = []
        all_env_rewards = []
        episode_boundaries = []  # (start_idx, length) pairs

        offset = 0
        for path in paths:
            data = torch.load(path, map_location="cpu", weights_only=False)
            n_steps = data["actions"].shape[0]
            all_windows.append(data["windows"])          # float16
            all_contracts.append(data["contracts"])       # float16
            all_sessions.append(data["session_states"])   # float32
            all_actions.append(data["actions"])           # int8
            all_train_rewards.append(data["train_rewards"])
            all_env_rewards.append(data["env_rewards"])

            # Episode boundaries: convert start indices to (start, length) pairs
            ep_starts = data["episode_boundaries"].numpy()
            for i in range(len(ep_starts)):
                start = int(ep_starts[i]) + offset
                if i + 1 < len(ep_starts):
                    length = int(ep_starts[i + 1]) - int(ep_starts[i])
                else:
                    length = n_steps - int(ep_starts[i])
                episode_boundaries.append((start, length))
            offset += n_steps

        self.windows = torch.cat(all_windows, dim=0)
        self.contracts = torch.cat(all_contracts, dim=0)
        self.session_states = torch.cat(all_sessions, dim=0)
        self.actions = torch.cat(all_actions, dim=0)
        train_rewards = torch.cat(all_train_rewards, dim=0)
        env_rewards = torch.cat(all_env_rewards, dim=0)

        # Compute returns-to-go per episode from train_rewards
        returns = torch.zeros_like(train_rewards)
        for start, length in episode_boundaries:
            G = 0.0
            for t in range(start + length - 1, start - 1, -1):
                G = float(train_rewards[t]) + gamma * G
                returns[t] = G

        self.returns = returns
        self.env_rewards = env_rewards
        self.train_rewards = train_rewards

        # Global statistics for logging
        self.return_mean = float(returns.mean())
        self.return_std = float(returns.std())
        self.n_steps = len(returns)
        self.n_episodes = len(episode_boundaries)

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx: int) -> dict:
        return {
            "window": self.windows[idx].float(),
            "contracts": self.contracts[idx].float(),
            "session_state": self.session_states[idx],
            "action": self.actions[idx].long(),
            "return_to_go": self.returns[idx],
        }
