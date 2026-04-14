"""Sequential agent training: behavioral cloning + REINFORCE.

Phase 3 of the session-agent plan. Trains the SequentialAgent to
make session-level trading decisions by imitating good oracle actions
(behavioral cloning) and then fine-tuning with policy gradient.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.core.chain_data import load_sidecar_cached, padded_snapshot
from v2.core.env import (
    TradingEnv,
    ACT_HOLD,
    ACT_ENTER_CALL,
    ACT_ENTER_PUT,
    ACT_EXIT,
    NUM_ACTIONS,
    SESSION_STATE_DIM,
)
from v2.core.policy import DEFAULT_POLICY
from v2.seq_agent import SequentialAgent
from v2.train import TradingModel, LOOKBACK, NUM_FEATURES, D_MODEL

# --- Hyperparameters ---
BC_LR = float(os.environ.get("BC_LR", 3e-4))
BC_EPOCHS = int(os.environ.get("BC_EPOCHS", 20))
RL_LR = float(os.environ.get("RL_LR", 1e-4))
RL_EPOCHS = int(os.environ.get("RL_EPOCHS", 50))
GAMMA = float(os.environ.get("GAMMA", 0.99))
ENTROPY_COEFF = float(os.environ.get("ENTROPY_COEFF", 0.01))
VALUE_COEFF = float(os.environ.get("VALUE_COEFF", 0.5))
TIME_BUDGET = int(os.environ.get("SEQ_TIME_BUDGET", 600))
SEED = int(os.environ.get("TRAIN_SEED", 123))


def _compute_oracle_actions(env: TradingEnv, day: str) -> list[dict]:
    """Generate oracle action labels for one day using path library.

    For each bar, determine the ideal action:
    - If strict opportunity exists and we're flat: ENTER (call or put based on oracle side)
    - If in position and unrealized PnL is bad: EXIT
    - Otherwise: HOLD
    """
    obs = env.reset(day)
    if env._done:
        return []

    sidecar = env._sidecar
    trajectory = []

    for step_idx in range(len(env._eligible_bars)):
        gi, local_bar = env._eligible_bars[step_idx]

        # Get path library data for this bar
        bar_ptrs = sidecar["bar_ptrs"]
        start = int(bar_ptrs[local_bar])
        end = int(bar_ptrs[local_bar + 1])

        # Check strict opportunity
        from v2.train import _compute_strict_opportunity
        strict_pos = _compute_strict_opportunity(sidecar, local_bar)

        # Determine oracle side from best contract
        oracle_side = 0  # 0 = no trade
        if strict_pos and end > start:
            best_idx = int(sidecar["bar_best_contract_idx"][local_bar])
            if best_idx >= 0:
                cidx = int(sidecar["row_contract_idx"][start + best_idx])
                is_put = int(sidecar["contract_right"][cidx]) == 1
                oracle_side = -1 if is_put else 1

        # Build the observation
        window = env.features[gi - env.lookback: gi].numpy()
        contracts_np, _, _ = padded_snapshot(sidecar, local_bar, env.max_contracts)
        session_state = obs.session_state.copy()

        # Determine oracle action
        if env._in_position:
            # In position: hold or exit based on path quality
            unrealized = env._get_unrealized(local_bar)
            if unrealized < -0.10:  # losing badly → should exit
                oracle_action = ACT_EXIT
            else:
                oracle_action = ACT_HOLD
        else:
            # Flat: enter or skip
            if strict_pos and oracle_side != 0:
                oracle_action = ACT_ENTER_CALL if oracle_side == 1 else ACT_ENTER_PUT
            else:
                oracle_action = ACT_HOLD

        trajectory.append({
            "window": window,
            "contracts": contracts_np,
            "session_state": session_state,
            "oracle_action": oracle_action,
            "bar": local_bar,
            "strict_pos": strict_pos,
        })

        # Step environment with oracle action to keep state consistent
        obs, _, done, _ = env.step(oracle_action)
        if done:
            break

    return trajectory


def train_behavioral_cloning(
    data_path: str = "v2/data.pt",
    model_path: str = "v2/models/model.pt",
    output_path: str = "v2/models/seq_agent.pt",
    train_days: list[str] | None = None,
    val_days: list[str] | None = None,
):
    """Phase 3: Train sequential agent via behavioral cloning."""
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading data and encoder...")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load frozen encoder
    encoder = TradingModel()
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"  Loaded encoder from {model_path}")
        except RuntimeError as e:
            print(f"  WARNING: Could not load {model_path} ({e}), using random encoder")
    else:
        print(f"  WARNING: No pretrained model at {model_path}, using random encoder")
    encoder = encoder.to(device)
    encoder.eval()

    # Create agent
    agent = SequentialAgent(encoder, context_dim=D_MODEL, freeze_encoder=True).to(device)
    optimizer = torch.optim.Adam(
        [p for p in agent.parameters() if p.requires_grad],
        lr=BC_LR,
    )

    # Determine train/val days
    all_days = sorted(set(data["dates"]))
    if train_days is None:
        # Use last 60 days for val, rest for train
        n_val = 60
        train_days = all_days[:-n_val]
        val_days = all_days[-n_val:]

    print(f"  Train days: {len(train_days)}, Val days: {len(val_days)}")

    # Create environment
    env = TradingEnv(data, DEFAULT_POLICY, "v2/data_sidecars", encoder, device, LOOKBACK)

    # Generate oracle trajectories
    print("Generating oracle trajectories...")
    t_gen = time.time()
    train_trajectories = []
    for day in train_days:
        traj = _compute_oracle_actions(env, day)
        if traj:
            train_trajectories.extend(traj)
    val_trajectories = []
    for day in val_days:
        traj = _compute_oracle_actions(env, day)
        if traj:
            val_trajectories.extend(traj)
    print(f"  Train samples: {len(train_trajectories)}, Val samples: {len(val_trajectories)}, {time.time()-t_gen:.1f}s")

    # Action distribution
    action_counts = defaultdict(int)
    for t in train_trajectories:
        action_counts[t["oracle_action"]] += 1
    print(f"  Action distribution: HOLD={action_counts[0]}, CALL={action_counts[1]}, PUT={action_counts[2]}, EXIT={action_counts[3]}")

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, BC_EPOCHS + 1):
        if time.time() - t_start > TIME_BUDGET:
            print(f"Time budget reached at epoch {epoch}")
            break

        agent.train()
        np.random.shuffle(train_trajectories)

        # Mini-batch training
        batch_size = 256
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_trajectories), batch_size):
            batch = train_trajectories[batch_start:batch_start + batch_size]
            if not batch:
                continue

            windows = torch.stack([torch.from_numpy(t["window"]).float() for t in batch]).to(device)
            contracts = torch.stack([torch.from_numpy(t["contracts"]).float() for t in batch]).to(device)
            sessions = torch.stack([torch.from_numpy(t["session_state"]).float() for t in batch]).to(device)
            actions = torch.tensor([t["oracle_action"] for t in batch], dtype=torch.long, device=device)

            out = agent.forward(windows, contracts, sessions)
            logits = out["action_logits"]
            # Class-weighted CE: upweight rare actions (enter/exit) vs dominant HOLD
            # Compute inverse-frequency weights
            if not hasattr(train_behavioral_cloning, '_class_weights'):
                counts = torch.bincount(torch.tensor([t["oracle_action"] for t in train_trajectories]), minlength=NUM_ACTIONS).float()
                counts = counts.clamp(min=1)
                weights = (1.0 / counts)
                weights = weights / weights.sum() * NUM_ACTIONS  # normalize to mean=1
                train_behavioral_cloning._class_weights = weights.to(device)
            loss = F.cross_entropy(logits, actions, weight=train_behavioral_cloning._class_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (logits.argmax(dim=-1) == actions).float().mean().item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_acc = epoch_acc / max(n_batches, 1)

        # Validation
        agent.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_start in range(0, len(val_trajectories), batch_size):
                batch = val_trajectories[batch_start:batch_start + batch_size]
                if not batch:
                    continue
                windows = torch.stack([torch.from_numpy(t["window"]).float() for t in batch]).to(device)
                contracts = torch.stack([torch.from_numpy(t["contracts"]).float() for t in batch]).to(device)
                sessions = torch.stack([torch.from_numpy(t["session_state"]).float() for t in batch]).to(device)
                actions = torch.tensor([t["oracle_action"] for t in batch], dtype=torch.long, device=device)

                out = agent.forward(windows, contracts, sessions)
                logits = out["action_logits"]
                loss = F.cross_entropy(logits, actions)
                val_loss += loss.item()
                val_acc += (logits.argmax(dim=-1) == actions).float().mean().item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        avg_val_acc = val_acc / max(val_batches, 1)

        print(f"Epoch {epoch:3d} | train_loss={avg_train_loss:.4f} acc={avg_train_acc:.3f} | "
              f"val_loss={avg_val_loss:.4f} acc={avg_val_acc:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "val_acc": avg_val_acc,
            }, output_path)

    print(f"\nBest epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    print(f"Training completed in {time.time() - t_start:.1f}s")
    print(f"\nMETRICS_JSON:{json.dumps({'best_epoch': best_epoch, 'val_loss': best_val_loss})}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--model", default="v2/models/model.pt")
    parser.add_argument("--output", default="v2/models/seq_agent.pt")
    parser.add_argument("--mode", default="bc", choices=["bc", "rl"])
    args = parser.parse_args()

    if args.mode == "bc":
        train_behavioral_cloning(args.data, args.model, args.output)
    else:
        print("RL training not yet implemented. Use --mode bc")
