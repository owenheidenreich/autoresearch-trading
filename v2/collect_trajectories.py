"""Collect offline trajectories from a policy checkpoint for AWAC training.

Runs deterministic rollouts through TradingEnv and stores per-step data
with both raw env rewards and shaped training rewards (entry cost + side penalty).

Walk-forward aligned: uses generate_folds() so each fold's trajectories
come only from that fold's train_days.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from v2.core.chain_data import padded_snapshot
from v2.core.env import (
    TradingEnv,
    ACT_HOLD,
    ACT_ENTER_CALL,
    ACT_ENTER_PUT,
    ACT_EXIT,
    SESSION_STATE_DIM,
)
from v2.core.policy import DEFAULT_POLICY
from v2.core.walkforward import generate_folds
from v2.seq_agent import SequentialAgent
from v2.train import TradingModel, D_MODEL, LOOKBACK

# Reward shaping config — must match the RL training that produced the checkpoint
ENTRY_COST_BASE = float(os.environ.get("RL_ENTRY_COST", 0.015))
ENTRY_COST_ESCALATION = float(os.environ.get("RL_ENTRY_ESCALATION", 0.015))
SIDE_IMBALANCE_COEFF = float(os.environ.get("RL_SIDE_IMBALANCE", 0.03))


def _load_agent(encoder_path: str, agent_path: str, device: str = "cpu"):
    """Load a SequentialAgent with auto-detected session dim."""
    encoder = TradingModel()
    if os.path.exists(encoder_path):
        ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
    encoder.eval()

    session_dim = SESSION_STATE_DIM
    seq_ckpt = torch.load(agent_path, map_location="cpu", weights_only=False)
    ckpt_dim = seq_ckpt["agent_state_dict"].get("session_proj.0.weight", torch.empty(0)).shape
    if len(ckpt_dim) == 2 and ckpt_dim[1] != SESSION_STATE_DIM:
        session_dim = ckpt_dim[1]

    agent = SequentialAgent(encoder, context_dim=D_MODEL, session_dim=session_dim, freeze_encoder=True)
    agent.load_state_dict(seq_ckpt["agent_state_dict"], strict=False)
    agent.to(device)
    agent.eval()
    return agent


def collect_fold_trajectories(
    data: dict,
    agent,
    train_days: list[str],
    device: str = "cpu",
) -> dict:
    """Collect trajectories for one fold's training days.

    Returns a dict ready to torch.save() as a .pt file.
    """
    env = TradingEnv(data, DEFAULT_POLICY, "v2/data_sidecars", agent.encoder, device, LOOKBACK)

    all_windows = []
    all_contracts = []
    all_sessions = []
    all_actions = []
    all_env_rewards = []
    all_train_rewards = []
    episode_boundaries = []
    days_collected = []

    step_offset = 0

    for day_idx, day in enumerate(train_days):
        obs = env.reset(day)
        if env._done:
            continue

        episode_boundaries.append(step_offset)
        days_collected.append(day)

        entries_so_far = 0
        day_actions = []
        day_env_rewards = []
        day_train_rewards = []

        agent_dim = agent.session_proj[0].in_features

        while not env._done:
            gi, local_bar = env._eligible_bars[env._step_idx]

            # Build tensors (same pattern as replay_sequential)
            window = env.features[gi - env.lookback: gi].numpy().copy()
            contracts_np, _, _ = padded_snapshot(env._sidecar, local_bar, env.max_contracts)
            session_state = obs.session_state.copy()
            if len(session_state) > agent_dim:
                session_state = session_state[:agent_dim]

            window_t = torch.from_numpy(window).float().to(device)
            contracts_t = torch.from_numpy(contracts_np).float().to(device)
            session_t = torch.from_numpy(session_state).float().to(device)

            with torch.no_grad():
                action, log_prob, value = agent.act(
                    window_t, contracts_t, session_t, deterministic=True
                )

            obs, env_reward, done, info = env.step(action)

            # Shaped training reward (matches REINFORCE objective exactly)
            train_reward = env_reward
            actual_action = info.action_taken
            if actual_action in (ACT_ENTER_CALL, ACT_ENTER_PUT):
                train_reward -= (ENTRY_COST_BASE + entries_so_far * ENTRY_COST_ESCALATION)
                entries_so_far += 1

            # Store step data
            all_windows.append(torch.from_numpy(window).half())
            all_contracts.append(torch.from_numpy(contracts_np).half())
            all_sessions.append(torch.from_numpy(session_state).float())
            all_actions.append(actual_action)
            all_env_rewards.append(env_reward)
            day_train_rewards.append(train_reward)
            day_actions.append(actual_action)

        # Side-imbalance penalty on last step of episode (matches train_seq.py:474-479)
        if day_train_rewards and SIDE_IMBALANCE_COEFF > 0:
            calls = sum(1 for a in day_actions if a == ACT_ENTER_CALL)
            puts = sum(1 for a in day_actions if a == ACT_ENTER_PUT)
            total_entries = calls + puts
            if total_entries >= 2:
                imbalance = abs(calls - puts) / total_entries
                day_train_rewards[-1] -= SIDE_IMBALANCE_COEFF * imbalance

        all_train_rewards.extend(day_train_rewards)
        step_offset += len(day_train_rewards)

        if (day_idx + 1) % 100 == 0:
            print(f"  Collected {day_idx + 1}/{len(train_days)} days, "
                  f"{step_offset} total steps")

    print(f"  Done: {len(days_collected)} days, {step_offset} steps, "
          f"{len(episode_boundaries)} episodes")

    return {
        "windows": torch.stack(all_windows) if all_windows else torch.empty(0),
        "contracts": torch.stack(all_contracts) if all_contracts else torch.empty(0),
        "session_states": torch.stack(all_sessions) if all_sessions else torch.empty(0),
        "actions": torch.tensor(all_actions, dtype=torch.int8),
        "env_rewards": torch.tensor(all_env_rewards, dtype=torch.float32),
        "train_rewards": torch.tensor(all_train_rewards, dtype=torch.float32),
        "episode_boundaries": torch.tensor(episode_boundaries, dtype=torch.int64),
        "days": days_collected,
        "reward_config": {
            "ENTRY_COST_BASE": ENTRY_COST_BASE,
            "ENTRY_COST_ESCALATION": ENTRY_COST_ESCALATION,
            "SIDE_IMBALANCE_COEFF": SIDE_IMBALANCE_COEFF,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Collect offline trajectories for AWAC")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--encoder", default="v2/models/model_trained_encoder.pt")
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint to roll out")
    parser.add_argument("--source-label", required=True,
                        help="Label for this source (e.g. 'bc', 'rl_ep5')")
    parser.add_argument("--output-dir", default="v2/trajectories")
    parser.add_argument("--fold", type=int, default=0, help="Which fold to collect for")
    parser.add_argument("--total-folds", type=int, default=5,
                        help="Canonical fold count — must match v2.core.walkforward.CANONICAL_N_FOLDS.")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    all_days = sorted(set(data["dates"]))

    folds = generate_folds(all_days, n_folds=args.total_folds)
    fold = folds[args.fold]
    print(f"Fold {args.fold}: {len(fold.train_days)} train days "
          f"({fold.train_days[0]} to {fold.train_days[-1]})")

    print(f"Loading agent from {args.checkpoint}...")
    agent = _load_agent(args.encoder, args.checkpoint)

    print(f"Reward config: entry_cost={ENTRY_COST_BASE}, "
          f"escalation={ENTRY_COST_ESCALATION}, "
          f"side_imbalance={SIDE_IMBALANCE_COEFF}")

    t0 = time.time()
    traj_data = collect_fold_trajectories(data, agent, fold.train_days)
    traj_data["source_policy"] = args.source_label
    elapsed = time.time() - t0

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"fold{args.fold}_{args.source_label}.pt")
    torch.save(traj_data, out_path)
    print(f"Saved {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB) in {elapsed:.1f}s")

    # Audit: show reward stream divergence
    env_r = traj_data["env_rewards"]
    train_r = traj_data["train_rewards"]
    diff = (train_r - env_r).abs()
    n_diff = (diff > 1e-6).sum().item()
    print(f"\nReward audit: {n_diff}/{len(env_r)} steps have env/train reward divergence")
    print(f"  env_reward:   mean={env_r.mean():.6f}  std={env_r.std():.6f}")
    print(f"  train_reward: mean={train_r.mean():.6f}  std={train_r.std():.6f}")


if __name__ == "__main__":
    main()
