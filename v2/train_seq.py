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


ORACLE_SIDE_LOCKOUT = int(os.environ.get("ORACLE_SIDE_LOCKOUT", 5))  # bars before side flip allowed
ORACLE_MAX_ENTRIES = int(os.environ.get("ORACLE_MAX_ENTRIES", 4))   # max entries per day


def _compute_oracle_actions(env: TradingEnv, day: str) -> list[dict]:
    """Generate oracle action labels for one day using path library.

    Thesis-persistence rules (controlled intervention, not destination):
    - Side-commitment: after entering a side, cannot flip for ORACLE_SIDE_LOCKOUT bars
    - Daily entry cap: max ORACLE_MAX_ENTRIES entries per day, then HOLD when flat
    - If in position and unrealized PnL < -10%: EXIT
    - Otherwise: HOLD
    """
    obs = env.reset(day)
    if env._done:
        return []

    sidecar = env._sidecar
    trajectory = []

    # Thesis-persistence state
    committed_side = 0       # 1=call, -1=put, 0=uncommitted
    bars_since_commit = 999  # bars since last entry (start high = unlocked)
    entries_today = 0

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

        # Determine oracle action with thesis-persistence rules
        if env._in_position:
            # In position: hold or exit based on path quality
            unrealized = env._get_unrealized(local_bar)
            if unrealized < -0.10:  # losing badly → should exit
                oracle_action = ACT_EXIT
            else:
                oracle_action = ACT_HOLD
        else:
            # Flat: enter or skip, subject to persistence rules
            oracle_action = ACT_HOLD

            if strict_pos and oracle_side != 0:
                # Check daily entry cap
                if entries_today >= ORACLE_MAX_ENTRIES:
                    oracle_action = ACT_HOLD  # exhausted budget
                # Check side-commitment lockout
                elif committed_side != 0 and oracle_side != committed_side and bars_since_commit < ORACLE_SIDE_LOCKOUT:
                    oracle_action = ACT_HOLD  # too soon to flip thesis
                else:
                    oracle_action = ACT_ENTER_CALL if oracle_side == 1 else ACT_ENTER_PUT

        # Update thesis-persistence state
        if oracle_action in (ACT_ENTER_CALL, ACT_ENTER_PUT):
            committed_side = 1 if oracle_action == ACT_ENTER_CALL else -1
            bars_since_commit = 0
            entries_today += 1
        else:
            bars_since_commit += 1

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


ENTRY_COST_BASE = float(os.environ.get("RL_ENTRY_COST", 0.01))  # base per-entry penalty
ENTRY_COST_ESCALATION = float(os.environ.get("RL_ENTRY_ESCALATION", 0.015))  # additional cost per prior entry
KL_COEFF = float(os.environ.get("RL_KL_COEFF", 0.03))        # anchor to BC policy (relaxed)
SIDE_IMBALANCE_COEFF = float(os.environ.get("RL_SIDE_IMBALANCE", 0.03))  # episode-end side-skew penalty


def train_reinforce(
    data_path: str = "v2/data.pt",
    model_path: str = "v2/models/model.pt",
    bc_checkpoint: str = "v2/models/seq_agent.pt",
    output_path: str = "v2/models/seq_agent_rl.pt",
):
    """REINFORCE fine-tuning from BC checkpoint.

    Goal: test whether the agent can learn day-level selectivity
    (not always filling its entry budget) while preserving low-flip behavior.

    Reward = env reward - entry_cost per entry - kl_coeff * KL(pi || pi_bc)
    """
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
    encoder = encoder.to(device)
    encoder.eval()

    # Load BC agent as anchor
    agent = SequentialAgent(encoder, context_dim=D_MODEL, freeze_encoder=True).to(device)
    if os.path.exists(bc_checkpoint):
        bc_ckpt = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
        agent.load_state_dict(bc_ckpt["agent_state_dict"], strict=False)
        print(f"  Loaded BC checkpoint from {bc_checkpoint}")
    else:
        print(f"  WARNING: No BC checkpoint at {bc_checkpoint}, starting from scratch")

    # Freeze a copy of BC policy for KL anchor
    bc_agent = SequentialAgent(encoder, context_dim=D_MODEL, freeze_encoder=True).to(device)
    bc_agent.load_state_dict(agent.state_dict())
    bc_agent.eval()
    for p in bc_agent.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in agent.parameters() if p.requires_grad],
        lr=RL_LR,
    )

    # Train/val split
    all_days = sorted(set(data["dates"]))
    n_val = 60
    train_days = all_days[:-n_val]
    val_days = all_days[-n_val:]
    print(f"  Train days: {len(train_days)}, Val days: {len(val_days)}")

    env = TradingEnv(data, DEFAULT_POLICY, "v2/data_sidecars", encoder, device, LOOKBACK)

    # Show entry cost schedule
    print(f"  Entry cost schedule: entry 1={ENTRY_COST_BASE:.3f}, "
          f"entry 2={ENTRY_COST_BASE+ENTRY_COST_ESCALATION:.3f}, "
          f"entry 3={ENTRY_COST_BASE+2*ENTRY_COST_ESCALATION:.3f}, "
          f"entry 4={ENTRY_COST_BASE+3*ENTRY_COST_ESCALATION:.3f}")
    print(f"  KL coeff: {KL_COEFF}")

    best_val_metric = -float("inf")
    best_epoch = 0

    for epoch in range(1, RL_EPOCHS + 1):
        if time.time() - t_start > TIME_BUDGET:
            print(f"Time budget reached at epoch {epoch}")
            break

        agent.train()
        np.random.shuffle(train_days)
        epoch_days = train_days[:100]  # sample 100 days per epoch for speed

        epoch_returns = []
        epoch_entries = []
        epoch_flips = []
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        n_episodes = 0
        train_reinforce._epoch_calls = 0
        train_reinforce._epoch_puts = 0

        for day in epoch_days:
            obs = env.reset(day)
            if env._done:
                continue

            # Collect trajectory
            log_probs = []
            values = []
            rewards = []
            entropies = []
            kl_terms = []
            actions_taken = []
            entries_so_far = 0

            while not env._done:
                gi, local_bar = env._eligible_bars[env._step_idx]
                window = torch.from_numpy(
                    env.features[gi - env.lookback: gi].numpy()
                ).float().to(device)
                contracts_np, _, _ = padded_snapshot(env._sidecar, local_bar, env.max_contracts)
                contracts_t = torch.from_numpy(contracts_np).float().to(device)
                session_t = torch.from_numpy(obs.session_state).float().to(device)

                # Forward pass (stochastic)
                out = agent.forward(
                    window.unsqueeze(0), contracts_t.unsqueeze(0), session_t.unsqueeze(0)
                )
                logits = out["action_logits"][0]
                value = out["value"][0]

                dist = torch.distributions.Categorical(logits=logits)
                action_t = dist.sample()
                action = int(action_t.item())

                log_probs.append(dist.log_prob(action_t))
                values.append(value)
                entropies.append(dist.entropy())

                # KL against BC policy
                with torch.no_grad():
                    bc_out = bc_agent.forward(
                        window.unsqueeze(0), contracts_t.unsqueeze(0), session_t.unsqueeze(0)
                    )
                    bc_logits = bc_out["action_logits"][0]
                bc_probs = F.softmax(bc_logits, dim=-1)
                pi_probs = F.softmax(logits, dim=-1)
                kl = (pi_probs * (pi_probs.log() - bc_probs.log())).sum()
                kl_terms.append(kl)

                obs, reward, done, info = env.step(action)

                # Escalating entry cost: 1st entry costs base, each subsequent costs more
                if action in (ACT_ENTER_CALL, ACT_ENTER_PUT):
                    reward -= (ENTRY_COST_BASE + entries_so_far * ENTRY_COST_ESCALATION)
                    entries_so_far += 1

                rewards.append(reward)
                actions_taken.append(action)

            if not rewards:
                continue

            # Side-imbalance penalty: discourage collapsing onto one side
            calls_today = sum(1 for a in actions_taken if a == ACT_ENTER_CALL)
            puts_today = sum(1 for a in actions_taken if a == ACT_ENTER_PUT)
            total_entries_today = calls_today + puts_today
            if total_entries_today >= 2 and SIDE_IMBALANCE_COEFF > 0:
                imbalance = abs(calls_today - puts_today) / total_entries_today
                rewards[-1] -= SIDE_IMBALANCE_COEFF * imbalance

            # Compute returns-to-go
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            # Normalize returns
            if len(returns_t) > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)
            kl_t = torch.stack(kl_terms)

            # Advantage = returns - value baseline
            advantages = returns_t - values_t.detach()

            # Losses
            policy_loss = -(log_probs_t * advantages).mean()
            value_loss = F.mse_loss(values_t, returns_t.detach())
            entropy_bonus = -entropies_t.mean()
            kl_penalty = kl_t.mean()

            loss = (policy_loss
                    + VALUE_COEFF * value_loss
                    + ENTROPY_COEFF * entropy_bonus
                    + KL_COEFF * kl_penalty)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl += kl_penalty.item()
            n_episodes += 1

            day_return = sum(rewards)
            day_calls = sum(1 for a in actions_taken if a == ACT_ENTER_CALL)
            day_puts = sum(1 for a in actions_taken if a == ACT_ENTER_PUT)
            epoch_returns.append(day_return)
            epoch_entries.append(day_calls + day_puts)
            epoch_flips.append(_count_side_flips_list(actions_taken))
            epoch_calls = getattr(train_reinforce, '_epoch_calls', 0) + day_calls
            epoch_puts = getattr(train_reinforce, '_epoch_puts', 0) + day_puts
            train_reinforce._epoch_calls = epoch_calls
            train_reinforce._epoch_puts = epoch_puts

        if n_episodes == 0:
            continue

        avg_return = np.mean(epoch_returns)
        avg_entries = np.mean(epoch_entries)
        avg_flips = np.mean(epoch_flips)
        ec = train_reinforce._epoch_calls
        ep = train_reinforce._epoch_puts
        call_pct = 100 * ec / max(ec + ep, 1)

        # Validation: deterministic replay on val days (sample 30 for speed)
        agent.eval()
        val_returns = []
        val_entries = []
        val_flips = []
        val_sample = val_days[:30]

        with torch.no_grad():
            for day in val_sample:
                obs = env.reset(day)
                if env._done:
                    continue
                day_reward = 0.0
                day_actions = []
                while not env._done:
                    gi, local_bar = env._eligible_bars[env._step_idx]
                    window = torch.from_numpy(
                        env.features[gi - env.lookback: gi].numpy()
                    ).float().to(device)
                    contracts_np, _, _ = padded_snapshot(env._sidecar, local_bar, env.max_contracts)
                    contracts_t = torch.from_numpy(contracts_np).float().to(device)
                    session_t = torch.from_numpy(obs.session_state).float().to(device)

                    action, _, _ = agent.act(window, contracts_t, session_t, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    day_reward += reward
                    day_actions.append(action)

                val_returns.append(day_reward)
                val_entries.append(sum(1 for a in day_actions if a in (ACT_ENTER_CALL, ACT_ENTER_PUT)))
                val_flips.append(_count_side_flips_list(day_actions))

        val_avg_return = np.mean(val_returns) if val_returns else 0.0
        val_avg_entries = np.mean(val_entries) if val_entries else 0.0
        val_avg_flips = np.mean(val_flips) if val_flips else 0.0
        val_zero_days = sum(1 for e in val_entries if e == 0)
        val_single_days = sum(1 for e in val_entries if e == 1)
        val_n = max(len(val_entries), 1)

        print(f"Epoch {epoch:3d} | ret={avg_return:.4f} ent={avg_entries:.1f} flip={avg_flips:.2f} C%={call_pct:.0f} "
              f"kl={total_kl/n_episodes:.4f} | "
              f"val_ret={val_avg_return:.4f} val_ent={val_avg_entries:.1f} val_flip={val_avg_flips:.2f} "
              f"0day={val_zero_days} 1day={val_single_days}")

        # Behavioral band checkpoint selection:
        # Target: entries/day in [1.0, 3.0], flips < 1.5, some selective days
        # In-band checkpoints ranked by return. Out-of-band only if nothing in-band yet.
        in_band = (1.0 <= val_avg_entries <= 3.0
                   and val_avg_flips < 1.5)
        selective_days = val_zero_days + val_single_days
        val_metric = val_avg_return  # within band, prefer higher return

        save_this = False
        if in_band:
            if not getattr(train_reinforce, '_has_in_band', False):
                save_this = True  # first in-band checkpoint
                train_reinforce._has_in_band = True
            elif val_metric > best_val_metric:
                save_this = True  # better in-band checkpoint
        elif not getattr(train_reinforce, '_has_in_band', False):
            # No in-band yet — save closest to target
            dist = abs(val_avg_entries - 2.0) + max(0, val_avg_flips - 1.0)
            if not hasattr(train_reinforce, '_best_dist') or dist < train_reinforce._best_dist:
                train_reinforce._best_dist = dist
                save_this = True

        band_label = "IN-BAND" if in_band else "out-of-band"
        print(f"         [{band_label}] sel_days={selective_days} metric={val_metric:.4f}")

        if save_this:
            best_val_metric = val_metric
            best_epoch = epoch
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "epoch": epoch,
                "val_metric": val_metric,
                "val_return": val_avg_return,
                "val_entries": val_avg_entries,
                "val_flips": val_avg_flips,
                "in_band": in_band,
                "selective_days": selective_days,
            }, output_path)

        # Always save latest for participation analysis
        latest_path = output_path.replace(".pt", "_latest.pt")
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        torch.save({
            "agent_state_dict": agent.state_dict(),
            "epoch": epoch,
            "val_metric": val_metric,
            "val_return": val_avg_return,
            "val_entries": val_avg_entries,
            "val_flips": val_avg_flips,
        }, latest_path)

    print(f"\nBest epoch: {best_epoch}, val_metric: {best_val_metric:.4f}")
    print(f"Training completed in {time.time() - t_start:.1f}s")
    print(f"\nMETRICS_JSON:{json.dumps({'best_epoch': best_epoch, 'val_metric': best_val_metric})}")


def _count_side_flips_list(actions: list[int]) -> int:
    last_side = None
    flips = 0
    for a in actions:
        if a == ACT_ENTER_CALL:
            if last_side == "put":
                flips += 1
            last_side = "call"
        elif a == ACT_ENTER_PUT:
            if last_side == "call":
                flips += 1
            last_side = "put"
    return flips


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--model", default="v2/models/model.pt")
    parser.add_argument("--output", default="v2/models/seq_agent.pt")
    parser.add_argument("--bc-checkpoint", default="v2/models/seq_agent.pt",
                        help="BC checkpoint to fine-tune from (for RL mode)")
    parser.add_argument("--mode", default="bc", choices=["bc", "rl"])
    args = parser.parse_args()

    if args.mode == "bc":
        train_behavioral_cloning(args.data, args.model, args.output)
    else:
        train_reinforce(args.data, args.model, args.bc_checkpoint, args.output)
