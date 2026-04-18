"""AWAC (Advantage Weighted Actor-Critic) offline RL training.

Replaces REINFORCE fine-tuning with offline advantage-weighted imitation
over a fixed trajectory dataset. Same architecture, same rewards, only
the optimization changes.

Honest framing: this is AWAC-lite / filtered advantage-weighted BC,
not full-strength offline RL. It preserves interpretability and
minimizes moving parts for a diagnostic experiment.

Phase 1: Pre-train value function on offline returns-to-go.
Phase 2: Joint advantage-weighted policy + value training.
Validation: deterministic replay on FULL fold val_days (not 30-day sample).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from v2.core.trajectory_buffer import OfflineTrajectoryDataset
from v2.core.walkforward import generate_folds
from v2.replay import load_model_from_path
from v2.seq_agent import SequentialAgent
from v2.train import TradingModel, D_MODEL, LOOKBACK

# --- Hyperparameters (all configurable via env vars) ---
AWAC_LAMBDA = float(os.environ.get("AWAC_LAMBDA", 1.0))
MAX_WEIGHT = float(os.environ.get("AWAC_MAX_WEIGHT", 20.0))
BATCH_SIZE = int(os.environ.get("AWAC_BATCH_SIZE", 512))
LR = float(os.environ.get("AWAC_LR", 3e-4))
VALUE_EPOCHS = int(os.environ.get("AWAC_VALUE_EPOCHS", 10))
N_EPOCHS = int(os.environ.get("AWAC_EPOCHS", 100))
VAL_INTERVAL = int(os.environ.get("AWAC_VAL_INTERVAL", 5))
KL_COEFF = float(os.environ.get("AWAC_KL_COEFF", 0.0))  # default OFF
GAMMA = float(os.environ.get("GAMMA", 0.99))
VALUE_COEFF = float(os.environ.get("VALUE_COEFF", 0.5))
TIME_BUDGET = int(os.environ.get("AWAC_TIME_BUDGET", 1800))
SEED = int(os.environ.get("TRAIN_SEED", 123))
ENCODER_LR = float(os.environ.get("ENCODER_LR", 3e-5))             # 10x lower than AWAC_LR
ENCODER_ANCHOR_COEFF = float(os.environ.get("ENCODER_ANCHOR_COEFF", 0.0))  # 0 = off (Arm B)


def _count_side_flips(actions: list[int]) -> int:
    """Count side flips within a day's action sequence."""
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


def _validate(agent, env, val_days, device):
    """Deterministic replay on FULL val_days. Returns behavioral metrics."""
    agent.eval()
    val_returns = []
    val_entries = []
    val_flips = []

    agent_dim = agent.session_proj[0].in_features

    with torch.no_grad():
        for day in val_days:
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
                contracts_np, _, _ = padded_snapshot(
                    env._sidecar, local_bar, env.max_contracts
                )
                contracts_t = torch.from_numpy(contracts_np).float().to(device)
                session_state = obs.session_state
                if len(session_state) > agent_dim:
                    session_state = session_state[:agent_dim]
                session_t = torch.from_numpy(session_state).float().to(device)

                action, _, _ = agent.act(window, contracts_t, session_t, deterministic=True)
                obs, reward, done, info = env.step(action)
                day_reward += reward
                day_actions.append(action)

            val_returns.append(day_reward)
            val_entries.append(
                sum(1 for a in day_actions if a in (ACT_ENTER_CALL, ACT_ENTER_PUT))
            )
            val_flips.append(_count_side_flips(day_actions))

    return {
        "avg_return": np.mean(val_returns) if val_returns else 0.0,
        "avg_entries": np.mean(val_entries) if val_entries else 0.0,
        "avg_flips": np.mean(val_flips) if val_flips else 0.0,
        "zero_days": sum(1 for e in val_entries if e == 0),
        "single_days": sum(1 for e in val_entries if e == 1),
        "n_days": len(val_returns),
    }


def train_awac(
    data_path: str = "v2/data.pt",
    encoder_path: str = "v2/models/model_trained_encoder.pt",
    bc_checkpoint: str = "v2/models/seq_agent_side13.pt",
    trajectory_dir: str = "v2/trajectories",
    output_path: str = "v2/models/seq_agent_awac_fold0.pt",
    fold_idx: int = 0,
    unfreeze_last_layer: bool = False,
    encoder_anchor_coeff: float | None = None,
):
    """AWAC offline RL from pre-collected trajectories."""
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    anchor_coeff = encoder_anchor_coeff if encoder_anchor_coeff is not None else ENCODER_ANCHOR_COEFF

    print("=" * 60)
    print(f"  AWAC Training — Fold {fold_idx}")
    print(f"  lambda={AWAC_LAMBDA}  max_weight={MAX_WEIGHT}  "
          f"batch={BATCH_SIZE}  lr={LR}")
    print(f"  kl_coeff={KL_COEFF}  gamma={GAMMA}  "
          f"value_epochs={VALUE_EPOCHS}  epochs={N_EPOCHS}")
    if unfreeze_last_layer:
        print(f"  ENCODER UNFREEZE: layers.2  encoder_lr={ENCODER_LR}  "
              f"anchor_coeff={anchor_coeff}")
    print("=" * 60)

    # --- Load encoder and agent from BC checkpoint ---
    print("\nLoading encoder and BC checkpoint...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
    encoder = load_model_from_path(encoder_path, device=device).to(device)
    if not unfreeze_last_layer:
        encoder.eval()

    # Detect session dim from BC checkpoint
    session_dim = SESSION_STATE_DIM
    bc_ckpt = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
    ckpt_dim = bc_ckpt["agent_state_dict"].get("session_proj.0.weight", torch.empty(0)).shape
    if len(ckpt_dim) == 2 and ckpt_dim[1] != SESSION_STATE_DIM:
        session_dim = ckpt_dim[1]

    agent = SequentialAgent(
        encoder, context_dim=D_MODEL, session_dim=session_dim, freeze_encoder=True
    ).to(device)
    agent.load_state_dict(bc_ckpt["agent_state_dict"], strict=False)
    print(f"  Loaded BC checkpoint (session_dim={session_dim})")

    # --- Partial encoder unfreezing ---
    unfrozen_encoder_params = []
    encoder_init = {}
    if unfreeze_last_layer:
        for name, p in agent.encoder.named_parameters():
            if name.startswith("encoder.layers.2."):
                p.requires_grad = True
                unfrozen_encoder_params.append((name, p))
        encoder_init = {name: p.data.clone() for name, p in unfrozen_encoder_params}
        n_unfrozen = sum(p.numel() for _, p in unfrozen_encoder_params)
        print(f"  Unfroze encoder.layers.2: {len(unfrozen_encoder_params)} params, "
              f"{n_unfrozen} values")

    # Frozen BC copy for KL and drift monitoring
    # When unfreezing, bc_agent needs its own encoder copy so it doesn't drift
    if unfreeze_last_layer:
        bc_encoder = TradingModel()
        bc_encoder.load_state_dict(encoder.state_dict())
        bc_encoder = bc_encoder.to(device)
        bc_encoder.eval()
        for p in bc_encoder.parameters():
            p.requires_grad = False
        bc_agent = SequentialAgent(
            bc_encoder, context_dim=D_MODEL, session_dim=session_dim, freeze_encoder=True
        ).to(device)
    else:
        bc_agent = SequentialAgent(
            encoder, context_dim=D_MODEL, session_dim=session_dim, freeze_encoder=True
        ).to(device)
    bc_agent.load_state_dict(bc_ckpt["agent_state_dict"], strict=False)
    bc_agent.eval()
    for p in bc_agent.parameters():
        p.requires_grad = False

    # --- Load offline dataset ---
    print("\nLoading offline trajectories...")
    traj_files = sorted(
        f for f in os.listdir(trajectory_dir)
        if f.startswith(f"fold{fold_idx}_") and f.endswith(".pt")
    )
    if not traj_files:
        raise FileNotFoundError(
            f"No trajectory files found for fold {fold_idx} in {trajectory_dir}"
        )
    traj_paths = [os.path.join(trajectory_dir, f) for f in traj_files]
    print(f"  Files: {traj_files}")

    dataset = OfflineTrajectoryDataset(traj_paths, gamma=GAMMA)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"  {dataset.n_steps} steps, {dataset.n_episodes} episodes")
    print(f"  Return stats: mean={dataset.return_mean:.4f}  std={dataset.return_std:.4f}")

    # --- Setup optimizer with param groups ---
    value_params = list(agent.value_head.parameters())
    policy_params = (
        list(agent.session_proj.parameters()) + list(agent.policy_head.parameters())
    )
    param_groups = [
        {"params": value_params, "lr": LR},
        {"params": policy_params, "lr": LR},
    ]
    if unfrozen_encoder_params:
        param_groups.append({
            "params": [p for _, p in unfrozen_encoder_params],
            "lr": ENCODER_LR,
        })
    optimizer = torch.optim.Adam(param_groups)

    # --- Walk-forward fold for validation ---
    print("\nSetting up validation...")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    all_days = sorted(set(data["dates"]))
    folds = generate_folds(all_days)
    fold = folds[fold_idx]
    val_days = fold.val_days  # FULL val_days, not a 30-day sample
    print(f"  Fold {fold_idx}: {len(val_days)} val days "
          f"({val_days[0]} to {val_days[-1]})")

    env = TradingEnv(data, DEFAULT_POLICY, "v2/data_sidecars", encoder, device, LOOKBACK)

    # ================================================================
    # PHASE 1: Value function pre-training
    # ================================================================
    print(f"\n--- Phase 1: Value pre-training ({VALUE_EPOCHS} epochs) ---")
    for ve in range(1, VALUE_EPOCHS + 1):
        if time.time() - t_start > TIME_BUDGET:
            print("Time budget reached during value pre-training")
            break

        agent.train()
        total_vloss = 0.0
        n_batches = 0

        for batch in loader:
            windows = batch["window"].to(device)
            contracts = batch["contracts"].to(device)
            sessions = batch["session_state"].to(device)
            returns = batch["return_to_go"].to(device)

            out = agent.forward(windows, contracts, sessions)
            v_pred = out["value"]
            v_loss = F.mse_loss(v_pred, returns)

            optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(value_params, 1.0)
            optimizer.step()

            total_vloss += v_loss.item()
            n_batches += 1

        avg_vloss = total_vloss / max(n_batches, 1)
        print(f"  Value epoch {ve:3d} | loss={avg_vloss:.6f}")

    # ================================================================
    # PHASE 2: Joint AWAC training
    # ================================================================
    print(f"\n--- Phase 2: AWAC training ({N_EPOCHS} epochs) ---")

    best_val_metric = -float("inf")
    best_epoch = 0
    has_in_band = False
    best_out_of_band_dist = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        if time.time() - t_start > TIME_BUDGET:
            print(f"Time budget reached at epoch {epoch}")
            break

        agent.train()
        total_ploss = 0.0
        total_vloss = 0.0
        total_kl = 0.0
        total_avg_weight = 0.0
        total_weight_std = 0.0
        total_clipped = 0.0
        total_anchor_loss = 0.0
        total_enc_grad_norm = 0.0
        n_batches = 0

        for batch in loader:
            windows = batch["window"].to(device)
            contracts = batch["contracts"].to(device)
            sessions = batch["session_state"].to(device)
            actions = batch["action"].to(device)
            returns = batch["return_to_go"].to(device)

            out = agent.forward(windows, contracts, sessions)
            logits = out["action_logits"]
            values = out["value"]

            # Value loss
            v_loss = F.mse_loss(values, returns)

            # Advantages (per-batch normalization)
            with torch.no_grad():
                advantages = returns - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # AWAC weights
            weights = torch.exp(advantages / AWAC_LAMBDA)
            n_clipped = (weights > MAX_WEIGHT).sum().item()
            weights = weights.clamp(max=MAX_WEIGHT)

            # Policy loss: advantage-weighted log-likelihood
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            p_loss = -(weights * action_log_probs).mean()

            # Optional KL penalty (default off)
            kl_val = 0.0
            if KL_COEFF > 0:
                with torch.no_grad():
                    bc_out = bc_agent.forward(windows, contracts, sessions)
                    bc_logits = bc_out["action_logits"]
                bc_probs = F.softmax(bc_logits, dim=-1)
                pi_probs = F.softmax(logits, dim=-1)
                kl = (pi_probs * (pi_probs.log() - bc_probs.log())).sum(dim=-1).mean()
                kl_val = kl.item()
                p_loss = p_loss + KL_COEFF * kl

            # Always monitor KL drift even when not penalizing
            if KL_COEFF == 0 and epoch % VAL_INTERVAL == 0 and n_batches == 0:
                with torch.no_grad():
                    bc_out = bc_agent.forward(windows, contracts, sessions)
                    bc_logits = bc_out["action_logits"]
                    bc_probs = F.softmax(bc_logits, dim=-1)
                    pi_probs = F.softmax(logits.detach(), dim=-1)
                    kl_val = (pi_probs * (pi_probs.log() - bc_probs.log())).sum(dim=-1).mean().item()

            loss = p_loss + VALUE_COEFF * v_loss

            # Anchor loss for unfrozen encoder (Arm C)
            anchor_loss_val = 0.0
            if anchor_coeff > 0 and unfrozen_encoder_params:
                anchor_loss = sum(
                    ((p - encoder_init[name]) ** 2).sum()
                    for name, p in unfrozen_encoder_params
                )
                loss = loss + anchor_coeff * anchor_loss
                anchor_loss_val = anchor_loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            # Encoder grad diagnostics
            if unfrozen_encoder_params:
                enc_gn = torch.sqrt(sum(
                    p.grad.norm() ** 2 for _, p in unfrozen_encoder_params
                    if p.grad is not None
                )).item()
                total_enc_grad_norm += enc_gn
                total_anchor_loss += anchor_loss_val

            total_ploss += p_loss.item()
            total_vloss += v_loss.item()
            total_kl += kl_val
            total_avg_weight += weights.mean().item()
            total_weight_std += weights.std().item()
            total_clipped += n_clipped
            n_batches += 1

        n = max(n_batches, 1)
        avg_ploss = total_ploss / n
        avg_vloss = total_vloss / n
        avg_weight = total_avg_weight / n
        weight_std = total_weight_std / n
        frac_clipped = total_clipped / max(dataset.n_steps, 1)

        # --- Validation ---
        if epoch % VAL_INTERVAL == 0:
            val = _validate(agent, env, val_days, device)
            avg_kl = total_kl / n

            # Encoder drift from initial weights
            enc_drift = 0.0
            if unfrozen_encoder_params:
                enc_drift = torch.sqrt(sum(
                    ((p.data - encoder_init[name]) ** 2).sum()
                    for name, p in unfrozen_encoder_params
                )).item()

            print(f"Epoch {epoch:3d} | p_loss={avg_ploss:.4f} v_loss={avg_vloss:.4f} "
                  f"wt={avg_weight:.2f}±{weight_std:.2f} clip={frac_clipped:.3f} "
                  f"kl={avg_kl:.4f}")
            print(f"         | val_ret={val['avg_return']:.4f} "
                  f"val_ent={val['avg_entries']:.1f} "
                  f"val_flip={val['avg_flips']:.2f} "
                  f"0day={val['zero_days']} 1day={val['single_days']} "
                  f"n={val['n_days']}")
            if unfrozen_encoder_params:
                avg_enc_gn = total_enc_grad_norm / n
                avg_anchor = total_anchor_loss / n
                print(f"         | enc_grad={avg_enc_gn:.6f} "
                      f"enc_drift={enc_drift:.6f} "
                      f"anchor_loss={avg_anchor:.6f}")

            # Behavioral band checkpoint selection (same logic as train_seq.py:587-625)
            in_band = (
                1.0 <= val["avg_entries"] <= 3.0
                and val["avg_flips"] < 1.5
            )
            selective_days = val["zero_days"] + val["single_days"]
            val_metric = val["avg_return"]

            save_this = False
            if in_band:
                if not has_in_band:
                    save_this = True
                    has_in_band = True
                elif val_metric > best_val_metric:
                    save_this = True
            elif not has_in_band:
                dist = abs(val["avg_entries"] - 2.0) + max(0, val["avg_flips"] - 1.0)
                if dist < best_out_of_band_dist:
                    best_out_of_band_dist = dist
                    save_this = True

            band_label = "IN-BAND" if in_band else "out-of-band"
            print(f"         [{band_label}] sel_days={selective_days} "
                  f"metric={val_metric:.4f}")

            if save_this:
                best_val_metric = val_metric
                best_epoch = epoch
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save({
                    "agent_state_dict": agent.state_dict(),
                    "epoch": epoch,
                    "val_metric": val_metric,
                    "val_return": val["avg_return"],
                    "val_entries": val["avg_entries"],
                    "val_flips": val["avg_flips"],
                    "in_band": in_band,
                    "selective_days": selective_days,
                    "method": "awac",
                    "awac_lambda": AWAC_LAMBDA,
                    "kl_coeff": KL_COEFF,
                    "unfreeze_last_layer": unfreeze_last_layer,
                    "encoder_anchor_coeff": anchor_coeff,
                    "encoder_lr": ENCODER_LR if unfreeze_last_layer else None,
                    "encoder_drift": enc_drift if unfrozen_encoder_params else None,
                }, output_path)
                print(f"         >>> SAVED (epoch {epoch})")

            # Save latest for analysis
            latest_path = output_path.replace(".pt", "_latest.pt")
            os.makedirs(os.path.dirname(latest_path), exist_ok=True)
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "epoch": epoch,
                "val_metric": val_metric,
                "val_return": val["avg_return"],
                "val_entries": val["avg_entries"],
                "val_flips": val["avg_flips"],
                "method": "awac",
                "unfreeze_last_layer": unfreeze_last_layer,
                "encoder_drift": enc_drift if unfrozen_encoder_params else None,
            }, latest_path)
        else:
            # Non-validation epoch: just print training metrics
            print(f"Epoch {epoch:3d} | p_loss={avg_ploss:.4f} v_loss={avg_vloss:.4f} "
                  f"wt={avg_weight:.2f}±{weight_std:.2f} clip={frac_clipped:.3f}")

    elapsed = time.time() - t_start
    print(f"\nBest epoch: {best_epoch}, val_metric: {best_val_metric:.4f}")
    print(f"Training completed in {elapsed:.1f}s")
    print(f"\nMETRICS_JSON:{json.dumps({'best_epoch': best_epoch, 'val_metric': best_val_metric, 'method': 'awac', 'lambda': AWAC_LAMBDA, 'unfreeze_last_layer': unfreeze_last_layer, 'encoder_anchor_coeff': anchor_coeff})}")


def main():
    parser = argparse.ArgumentParser(description="AWAC offline RL training")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--encoder", default="v2/models/model_trained_encoder.pt")
    parser.add_argument("--bc-checkpoint", default="v2/models/seq_agent_side13.pt")
    parser.add_argument("--trajectory-dir", default="v2/trajectories")
    parser.add_argument("--output", default="v2/models/seq_agent_awac_fold0.pt")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--unfreeze-last-layer", action="store_true", default=False,
                        help="Unfreeze encoder.layers.2 during AWAC (Arm B/C)")
    parser.add_argument("--encoder-anchor-coeff", type=float, default=None,
                        help="L2 anchor coeff for unfrozen encoder (Arm C)")
    args = parser.parse_args()

    train_awac(
        data_path=args.data,
        encoder_path=args.encoder,
        bc_checkpoint=args.bc_checkpoint,
        trajectory_dir=args.trajectory_dir,
        output_path=args.output,
        fold_idx=args.fold,
        unfreeze_last_layer=args.unfreeze_last_layer,
        encoder_anchor_coeff=args.encoder_anchor_coeff,
    )


if __name__ == "__main__":
    main()
