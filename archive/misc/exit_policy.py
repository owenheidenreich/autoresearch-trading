"""Phase D: RL Exit Policy Agent.

A lightweight MLP that decides HOLD / TIGHTEN_STOP / EXIT at each bar while
holding a position.  Trained via REINFORCE with baseline on replay trade
episodes.  Integrated into decision.py as a second opinion alongside the
supervised model's gate head.

Usage:
    # Train from replay backtest trades
    python training/exit_policy.py --trades results/backtest_trades.csv

    # Or from a data.pt replay
    python training/exit_policy.py --data training/data.pt --model training/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXIT_ACTION_HOLD = 0
EXIT_ACTION_TIGHTEN = 1
EXIT_ACTION_EXIT = 2
NUM_EXIT_ACTIONS = 3

OBS_DIM = 11  # observation vector size
STOP_TIGHTEN_FACTOR = 0.80  # multiply stop distance by this on TIGHTEN
STOP_FLOOR_PCT = 0.15       # minimum stop distance (15% of entry)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "exit_policy.pt"
)


# ---------------------------------------------------------------------------
# Exit Policy Network
# ---------------------------------------------------------------------------

class ExitPolicyNet(nn.Module):
    """Small MLP: 11-dim observation → 3-dim action logits."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden1: int = 32, hidden2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, NUM_EXIT_ACTIONS),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns action logits (batch, 3)."""
        return self.net(obs)

    def get_action(self, obs: torch.Tensor) -> tuple[int, float]:
        """Sample action from policy. Returns (action_int, log_prob)."""
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())

    def get_action_deterministic(self, obs: torch.Tensor) -> int:
        """Argmax action for inference."""
        logits = self.forward(obs)
        return int(torch.argmax(logits, dim=-1).item())

    def get_probs(self, obs: torch.Tensor) -> np.ndarray:
        """Action probabilities for logging."""
        with torch.no_grad():
            logits = self.forward(obs)
            return F.softmax(logits, dim=-1).cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Observation Builder
# ---------------------------------------------------------------------------

def build_exit_observation(
    bars_held: int,
    unrealized_pnl: float,
    account_health: float,
    loss_streak_frac: float,
    minutes_to_close: float,
    atm_iv: float,
    vix_regime: float,
    current_stop_distance: float,
    entry_confidence: float,
    best_pnl_since_entry: float,
    bars_since_last_high: int,
) -> torch.Tensor:
    """Build 11-dim observation tensor for the exit policy."""
    obs = torch.tensor([
        min(bars_held / 390.0, 1.0),           # bars_held_norm
        float(np.tanh(unrealized_pnl * 3.0)),   # unrealized_pnl_norm (saturated)
        account_health,                          # account_health [0,1]
        loss_streak_frac,                        # loss_streak_frac [0,1]
        min(minutes_to_close / 390.0, 1.0),      # minutes_to_close_norm
        min(max(atm_iv, 0.0), 1.0),             # atm_iv (already normalized in features)
        (vix_regime + 1.0) / 2.0,               # vix_regime: [-1,1] → [0,1]
        min(max(current_stop_distance, 0.0), 1.0),  # stop_distance_norm
        entry_confidence,                        # entry_confidence [0,1]
        float(np.tanh(best_pnl_since_entry * 3.0)),  # best_pnl_norm
        min(bars_since_last_high / 60.0, 1.0),  # bars_since_high_norm
    ], dtype=torch.float32)
    return obs


# ---------------------------------------------------------------------------
# Episode: one trade from entry to exit
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A single trade episode for RL training."""
    observations: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    per_step_pnl: list[float] = field(default_factory=list)  # P&L at each timestep
    reward: float = 0.0  # realized P&L at episode end


# ---------------------------------------------------------------------------
# Episode Generation from Replay
# ---------------------------------------------------------------------------

def generate_episodes_from_replay(
    model_path: str,
    data_path: str,
    device: str = "cpu",
    train_py_path: str | None = None,
) -> list[Episode]:
    """Run replay backtest and convert trades into RL episodes.

    Each bar while holding generates an observation. The terminal reward
    is the realized P&L of the trade.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from replay import load_model, run_backtest

    model, lookback, config, _ = load_model(model_path, device=device, train_py_path=train_py_path)
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    # Run replay backtest to get trade-level data
    from replay import run_replay
    from prepare import make_dataloader, BARS_PER_DAY, NUM_FEATURES, PNL_TANH_SCALE

    features = data['features']
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    val_start = int(data['val_start_idx'])
    val_end = int(data['val_end_idx'])
    day_boundaries = data['day_boundaries'].tolist()

    # Find validation days
    val_days = []
    for d in range(len(day_boundaries)):
        ds = day_boundaries[d]
        de = day_boundaries[d + 1] if d + 1 < len(day_boundaries) else len(features)
        if ds >= val_start and de <= val_end + 1:
            val_days.append((ds, de))

    episodes = []

    # Process each validation day
    for day_start, day_end in val_days:
        day_features = features[day_start:day_end]
        n_bars = day_end - day_start
        if n_bars < lookback + 10:
            continue

        # Run model inference bar by bar
        model.eval()
        in_trade = False
        current_episode: Episode | None = None
        bars_held = 0
        entry_confidence = 0.0
        best_pnl = 0.0
        bars_since_high = 0
        entry_price = 0.0
        stop_distance = 0.35  # default

        # Get option P&L arrays for this day
        call_pnl = data.get('call_pnl')
        put_pnl = data.get('put_pnl')
        call_stopped = data.get('call_stopped_pnl')
        put_stopped = data.get('put_stopped_pnl')

        if call_pnl is None or put_pnl is None:
            continue

        for bar in range(lookback, n_bars):
            global_idx = day_start + bar
            window = day_features[bar - lookback:bar]
            x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)

            pos_state = torch.zeros(1, 5, device=device)
            if in_trade:
                pos_state[0, 0] = 1.0
                pos_state[0, 1] = min(bars_held / BARS_PER_DAY, 1.0)
                cur_pnl = float(call_pnl[global_idx]) if not np.isnan(call_pnl[global_idx]) else 0.0
                pos_state[0, 2] = float(np.tanh(cur_pnl * PNL_TANH_SCALE))
            pos_state[0, 3] = 1.0  # account_health

            with torch.no_grad():
                gate_logits, dir_logits = model(x, position_state=pos_state)
                gate_probs = F.softmax(gate_logits, dim=-1)[0]
                dir_probs = F.softmax(dir_logits, dim=-1)[0]

            gate_trade = int(torch.argmax(gate_logits, dim=-1).item())
            best_dir = int(torch.argmax(dir_logits, dim=-1).item())
            confidence = float(gate_probs[1] * dir_probs[best_dir])

            if in_trade:
                # Build observation for exit policy
                cur_pnl_val = float(call_pnl[global_idx]) if not np.isnan(call_pnl[global_idx]) else 0.0
                if cur_pnl_val > best_pnl:
                    best_pnl = cur_pnl_val
                    bars_since_high = 0
                else:
                    bars_since_high += 1

                minutes_left = max(n_bars - bar, 0)
                iv_val = float(window[-1, 22]) if window.shape[1] > 22 else 0.0  # IDX_ATM_IV
                vix_val = float(window[-1, 24]) if window.shape[1] > 24 else 0.0  # vix_regime

                obs = build_exit_observation(
                    bars_held=bars_held,
                    unrealized_pnl=cur_pnl_val,
                    account_health=1.0,
                    loss_streak_frac=0.0,
                    minutes_to_close=float(minutes_left),
                    atm_iv=iv_val,
                    vix_regime=vix_val,
                    current_stop_distance=stop_distance,
                    entry_confidence=entry_confidence,
                    best_pnl_since_entry=best_pnl,
                    bars_since_last_high=bars_since_high,
                )
                current_episode.observations.append(obs)
                current_episode.actions.append(EXIT_ACTION_HOLD)
                current_episode.log_probs.append(0.0)
                current_episode.per_step_pnl.append(cur_pnl_val)

                bars_held += 1

                # Check exit conditions (matching replay)
                stopped_pnl = float(call_stopped[global_idx]) if call_stopped is not None and not np.isnan(call_stopped[global_idx]) else None
                hit_stop = stopped_pnl is not None and cur_pnl_val <= -stop_distance
                model_exit = gate_trade == 0 and bars_held > 1
                max_hold = bars_held >= 390
                eod = bar >= n_bars - 1

                if hit_stop or model_exit or max_hold or eod:
                    # Episode ends
                    current_episode.reward = cur_pnl_val
                    if len(current_episode.observations) >= 2:
                        episodes.append(current_episode)
                    in_trade = False
                    current_episode = None

            elif gate_trade == 1 and bar >= 60:  # NO_TRADE_BEFORE_BAR
                # Entry
                in_trade = True
                bars_held = 0
                entry_confidence = confidence
                best_pnl = 0.0
                bars_since_high = 0
                stop_distance = 0.35
                current_episode = Episode()

    return episodes


def generate_episodes_from_csv(csv_path: str, data_path: str) -> list[Episode]:
    """Generate episodes from backtest_trades.csv + data.pt.

    This is simpler — we don't re-run inference, just build observations
    from the trade records and per-bar data.
    """
    import csv

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    features = data['features'].numpy() if isinstance(data['features'], torch.Tensor) else data['features']
    call_pnl = data.get('call_pnl')
    if call_pnl is not None and isinstance(call_pnl, torch.Tensor):
        call_pnl = call_pnl.numpy()

    episodes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bars_held = int(row.get('bars_held', 0))
                pnl = float(row.get('pnl_pct', 0))
                entry_conf = float(row.get('entry_confidence', 0.5))
            except (ValueError, KeyError):
                continue

            if bars_held < 2:
                continue

            # Create simple episode with synthetic observations
            ep = Episode()
            for b in range(bars_held):
                # Linearly interpolate P&L from 0 to final
                frac = b / max(bars_held - 1, 1)
                cur_pnl = pnl * frac
                obs = build_exit_observation(
                    bars_held=b,
                    unrealized_pnl=cur_pnl,
                    account_health=1.0,
                    loss_streak_frac=0.0,
                    minutes_to_close=float(390 - b),
                    atm_iv=0.3,
                    vix_regime=0.0,
                    current_stop_distance=0.35,
                    entry_confidence=entry_conf,
                    best_pnl_since_entry=max(cur_pnl, 0.0),
                    bars_since_last_high=b if cur_pnl < max(pnl * (b - 1) / max(bars_held - 1, 1), 0.0) else 0,
                )
                ep.observations.append(obs)
                ep.actions.append(EXIT_ACTION_HOLD)
                ep.log_probs.append(0.0)
            ep.reward = pnl
            episodes.append(ep)

    return episodes


# ---------------------------------------------------------------------------
# REINFORCE Training
# ---------------------------------------------------------------------------

def train_exit_policy(
    episodes: list[Episode],
    n_epochs: int = 50,
    lr: float = 1e-3,
    gamma: float = 0.99,
    device: str = "cpu",
) -> tuple[ExitPolicyNet, dict]:
    """Train exit policy via REINFORCE with balanced episodes.

    Key design: oversample winning trades so the policy learns to distinguish
    winners from losers by observation features, rather than collapsing to
    always-EXIT due to negative average reward.

    Returns (trained_model, training_stats).
    """
    if not episodes:
        raise ValueError("No episodes to train on")

    policy = ExitPolicyNet().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Split into winners and losers for balanced sampling
    winners = [ep for ep in episodes if ep.reward > 0]
    losers = [ep for ep in episodes if ep.reward <= 0]
    print(f"  Winners: {len(winners)}, Losers: {len(losers)}")

    if not winners:
        print("  WARNING: No winning episodes — exit policy cannot learn meaningful behavior")
        # Fall back to unbalanced
        balanced_episodes = episodes
    else:
        # Oversample winners to 50/50 balance
        n_target = max(len(winners), len(losers))
        balanced_winners = list(np.random.choice(winners, size=n_target, replace=True))
        balanced_losers = list(np.random.choice(losers, size=n_target, replace=True)) if losers else []
        balanced_episodes = balanced_winners + balanced_losers
        print(f"  Balanced dataset: {len(balanced_episodes)} episodes (50/50 win/lose)")

    stats = {"epochs": [], "avg_return": [], "avg_loss": []}

    for epoch in range(n_epochs):
        epoch_returns = []
        epoch_losses = []

        # Shuffle balanced episodes
        indices = np.random.permutation(len(balanced_episodes))

        for idx in indices:
            ep = balanced_episodes[idx]
            if not ep.observations:
                continue

            T = len(ep.observations)
            # Per-step advantage: how much P&L remains from this point
            # Positive = holding is better than exiting now
            # Negative = should have exited already
            returns = torch.zeros(T)
            final_pnl = ep.reward
            for t in range(T):
                current_pnl = ep.per_step_pnl[t] if t < len(ep.per_step_pnl) else 0.0
                returns[t] = (final_pnl - current_pnl) * (gamma ** (T - t - 1))

            # Per-episode normalization — prevents one sign dominating
            if T > 1 and returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Forward pass through all observations
            obs_batch = torch.stack(ep.observations).to(device)
            logits = policy(obs_batch)

            # Sample actions from current policy
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            sampled_actions = dist.sample()
            log_probs_selected = dist.log_prob(sampled_actions)

            # Policy gradient: for winning episodes, early bars have positive
            # advantage (HOLD), late bars negative (EXIT to lock in gains).
            # For losing episodes, early bars positive (EXIT to cut losses),
            # late bars negative (already lost, EXIT is too late).
            policy_loss = -(log_probs_selected * returns.to(device)).mean()

            # Entropy bonus for exploration
            entropy = dist.entropy().mean()
            loss = policy_loss - 0.02 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_returns.append(ep.reward)
            epoch_losses.append(loss.item())

        avg_return = np.mean(epoch_returns) if epoch_returns else 0.0
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        stats["epochs"].append(epoch)
        stats["avg_return"].append(avg_return)
        stats["avg_loss"].append(avg_loss)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: avg_return={avg_return:.4f}, "
                  f"avg_loss={avg_loss:.4f}")

    # Analyze learned policy
    policy.eval()
    action_counts = {EXIT_ACTION_HOLD: 0, EXIT_ACTION_TIGHTEN: 0, EXIT_ACTION_EXIT: 0}
    for ep in episodes:
        for obs in ep.observations:
            action = policy.get_action_deterministic(obs.unsqueeze(0).to(device))
            action_counts[action] += 1
    total = sum(action_counts.values())
    stats["action_distribution"] = {
        "hold": action_counts[EXIT_ACTION_HOLD] / max(total, 1),
        "tighten": action_counts[EXIT_ACTION_TIGHTEN] / max(total, 1),
        "exit": action_counts[EXIT_ACTION_EXIT] / max(total, 1),
    }
    stats["num_episodes"] = len(episodes)
    stats["avg_episode_length"] = np.mean([len(ep.observations) for ep in episodes])

    return policy, stats


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_exit_policy(policy: ExitPolicyNet, path: str, stats: dict | None = None):
    """Save exit policy checkpoint."""
    torch.save({
        "model_state_dict": policy.state_dict(),
        "obs_dim": OBS_DIM,
        "num_actions": NUM_EXIT_ACTIONS,
        "stats": stats or {},
    }, path)
    print(f"Exit policy saved: {path}")


def load_exit_policy(path: str, device: str = "cpu") -> ExitPolicyNet:
    """Load exit policy from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    policy = ExitPolicyNet(obs_dim=ckpt.get("obs_dim", OBS_DIM))
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    policy.to(device)
    return policy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RL exit policy")
    parser.add_argument("--data", default="training/data.pt",
                        help="Path to data.pt")
    parser.add_argument("--model", default="training/best_model.pt",
                        help="Path to supervised model checkpoint")
    parser.add_argument("--trades", default=None,
                        help="Path to backtest_trades.csv (alternative to --model)")
    parser.add_argument("--output", default=DEFAULT_MODEL_PATH,
                        help="Output path for trained exit policy")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("=== Phase D: RL Exit Policy Training ===")

    # Generate episodes
    if args.trades and os.path.exists(args.trades):
        print(f"Generating episodes from CSV: {args.trades}")
        data_path = args.data
        if not os.path.exists(data_path):
            # Try cache
            cache_path = os.path.expanduser("~/.cache/autoresearch-trading/features/data.pt")
            if os.path.exists(cache_path):
                data_path = cache_path
        episodes = generate_episodes_from_csv(args.trades, data_path)
    else:
        print(f"Generating episodes from replay: model={args.model}, data={args.data}")
        data_path = args.data
        if not os.path.exists(data_path):
            cache_path = os.path.expanduser("~/.cache/autoresearch-trading/features/data.pt")
            if os.path.exists(cache_path):
                data_path = cache_path
        episodes = generate_episodes_from_replay(args.model, data_path, device=args.device)

    print(f"Generated {len(episodes)} episodes "
          f"(avg length: {np.mean([len(e.observations) for e in episodes]):.1f} bars)")

    if len(episodes) < 10:
        print("WARNING: Very few episodes. Results may be unreliable.")

    # Split train/val
    n_val = max(1, len(episodes) // 5)
    val_episodes = episodes[-n_val:]
    train_episodes = episodes[:-n_val]

    print(f"Train: {len(train_episodes)} episodes, Val: {len(val_episodes)} episodes")

    # Train
    policy, stats = train_exit_policy(
        train_episodes, n_epochs=args.epochs, lr=args.lr, device=args.device
    )

    # Evaluate on validation
    print("\n--- Validation Analysis ---")
    policy.eval()
    exit_pnls = []
    hold_pnls = []
    for ep in val_episodes:
        # What would the exit policy do?
        exited_early = False
        for t, obs in enumerate(ep.observations):
            action = policy.get_action_deterministic(obs.unsqueeze(0))
            if action == EXIT_ACTION_EXIT and t < len(ep.observations) - 1:
                # Would have exited early
                frac = t / max(len(ep.observations) - 1, 1)
                early_pnl = ep.reward * frac  # approximate
                exit_pnls.append(early_pnl)
                exited_early = True
                break
        if not exited_early:
            exit_pnls.append(ep.reward)
        hold_pnls.append(ep.reward)

    print(f"Hold-all P&L:     mean={np.mean(hold_pnls):.4f}")
    print(f"Exit-policy P&L:  mean={np.mean(exit_pnls):.4f}")
    print(f"Action distribution: {stats['action_distribution']}")

    # Save
    save_exit_policy(policy, args.output, stats)
    print(f"\nDone. Exit policy saved to {args.output}")
    print(json.dumps({
        "status": "exit_policy_trained",
        "num_episodes": stats["num_episodes"],
        "avg_episode_length": stats["avg_episode_length"],
        "action_distribution": stats["action_distribution"],
        "output_path": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
