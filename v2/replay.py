"""ART² v4 Replay: evaluate a contract-scoring model on exact chain sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from collections import defaultdict

import numpy as np
import torch

from v2.core.chain_data import describe_contract, extract_contract_series, load_sidecar_cached, padded_snapshot, QUALITY_PARTIAL
from v2.core.decision_trace import DecisionTrace, build_trace_for_bar, save_traces, print_trace_summary
from v2.core.metrics import ReplayMetrics, compute_metrics, score_config_fingerprint
from v2.core.policy import DEFAULT_POLICY, DecisionPolicy
from v2.core.schema import TradeIntent
from v2.core.simulator import simulate_trade
from v2.train import LOOKBACK, TradingModel


BATCH_SIZE = 2048
BASELINE_CACHE_PATH = "v2/state/baseline_cache.json"


def load_model_from_path(path: str, device: str = "cpu") -> TradingModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hyperparams = checkpoint.get("hyperparams", {})
    model = TradingModel(
        d_model=hyperparams.get("d_model", 96),
        depth=hyperparams.get("depth", 3),
        n_heads=hyperparams.get("n_heads", 4),
        dropout=hyperparams.get("dropout", 0.05),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_best_model(device: str = "cpu", current_dataset_fingerprint: str | None = None):
    from v2.ops.artifact import iter_artifacts_by_score, load_artifact

    artifact_errors: list[str] = []
    for artifact_dir in iter_artifacts_by_score():
        try:
            result = load_artifact(artifact_dir, current_dataset_fingerprint=current_dataset_fingerprint, device=device)
            print(f"Loaded best model from artifact {result['manifest']['experiment_id']} (score={result['manifest']['score']:.4f})")
            return result["model"], result["policy"], result["manifest"]
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            artifact_errors.append(f"{artifact_dir}: {exc}")

    for fallback_path in ("v2/models/model_best.pt", "v2/models/model.pt"):
        if os.path.exists(fallback_path):
            try:
                model = load_model_from_path(fallback_path, device=device)
            except RuntimeError as exc:
                artifact_errors.append(f"{fallback_path}: incompatible raw checkpoint ({exc})")
                continue
            print(f"WARNING: no compatible artifact bundle found; using raw checkpoint {fallback_path}")
            return model, DEFAULT_POLICY, {"experiment_id": os.path.basename(fallback_path), "score": float("nan"), "raw_checkpoint_fallback": True}

    if artifact_errors:
        raise FileNotFoundError("No compatible artifact bundle found.\n" + "\n".join(artifact_errors[:5]))
    raise FileNotFoundError("No artifacts found. Run an experiment loop first to produce a model.")


def _build_day_index(dates) -> dict[str, list[int]]:
    day_to_bars: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        day_to_bars[d].append(i)
    return day_to_bars


def model_to_intent(
    *,
    no_trade_score: torch.Tensor = None,
    gate_logit: torch.Tensor = None,
    direction_logit: torch.Tensor = None,
    contract_scores: torch.Tensor,
    contract_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    contract_features: torch.Tensor,
    contract_indices: torch.Tensor,
    sidecar: dict,
    local_bar: int,
    spot_price: float,
    policy: DecisionPolicy,
) -> TradeIntent:
    # Hierarchical decision: gate → direction → strike
    # Support: (1) opportunity_logit gate, (2) gate_logit + direction_logit, (3) legacy no_trade_score
    has_direction = direction_logit is not None and gate_logit is not None
    has_opportunity_gate = (not has_direction) and gate_logit is not None

    if has_direction:
        # Hierarchical model: explicit gate and direction heads
        gate_score = float(gate_logit.item())
        if gate_score <= policy.gate_threshold:
            return TradeIntent.no_trade(
                bar_index=local_bar,
                timestamp=str(int(sidecar["bar_timestamps"][local_bar])) if len(sidecar["bar_timestamps"]) else "",
                reason_codes=("gate_reject",),
                no_trade_score=-gate_score,
            )
        # Direction: mask contracts to predicted direction
        pred_put = float(direction_logit.item()) > 0
        is_put = contract_features[:, 2] > 0.5  # right_is_put
        dir_mask = (is_put == pred_put)
        scores = contract_scores.clone()
        scores[~valid_mask] = -float("inf")
        quality_flags = contract_features[:, 14]
        scores[quality_flags < QUALITY_PARTIAL] = -float("inf")
        scores[~dir_mask] = -float("inf")  # mask opposite direction
        best_row = int(scores.argmax().item()) if (valid_mask & dir_mask).any() else -1
        best_score = float(scores[best_row].item()) if best_row >= 0 and torch.isfinite(scores[best_row]) else -float("inf")
        abstain_score = -gate_score
    elif has_opportunity_gate:
        # Opportunity-gated model: independent gate from context, rank all contracts
        gate_score = float(gate_logit.item())
        if gate_score <= policy.gate_threshold:
            return TradeIntent.no_trade(
                bar_index=local_bar,
                timestamp=str(int(sidecar["bar_timestamps"][local_bar])) if len(sidecar["bar_timestamps"]) else "",
                reason_codes=("opportunity_reject",),
                no_trade_score=-gate_score,
            )
        scores = contract_scores.clone()
        scores[~valid_mask] = -float("inf")
        quality_flags = contract_features[:, 14]
        scores[quality_flags < QUALITY_PARTIAL] = -float("inf")
        best_row = int(scores.argmax().item()) if valid_mask.any() else -1
        best_score = float(scores[best_row].item()) if best_row >= 0 else -float("inf")
        abstain_score = -gate_score
    else:
        # Legacy model: gate from score comparison
        scores = contract_scores.clone()
        scores[~valid_mask] = -float("inf")
        quality_flags = contract_features[:, 14]
        scores[quality_flags < QUALITY_PARTIAL] = -float("inf")
        best_row = int(scores.argmax().item()) if valid_mask.any() else -1
        best_score = float(scores[best_row].item()) if best_row >= 0 else -float("inf")
        abstain_score = float(no_trade_score.item()) if no_trade_score is not None else 0.0

    if best_row < 0 or (not has_direction and not has_opportunity_gate and best_score <= max(policy.gate_threshold, abstain_score)):
        return TradeIntent.no_trade(
            bar_index=local_bar,
            timestamp=str(int(sidecar["bar_timestamps"][local_bar])) if len(sidecar["bar_timestamps"]) else "",
            reason_codes=("no_trade",),
            no_trade_score=abstain_score,
        )

    if not torch.isfinite(contract_labels[best_row]):
        return TradeIntent.no_trade(
            bar_index=local_bar,
            timestamp=str(int(sidecar["bar_timestamps"][local_bar])) if len(sidecar["bar_timestamps"]) else "",
            reason_codes=("unexecutable_history_gap",),
            no_trade_score=abstain_score,
        )

    contract_idx = int(contract_indices[best_row].item())
    contract = describe_contract(sidecar, contract_idx)
    series = extract_contract_series(sidecar, contract_idx)
    entry_mid = float(series["mid"][local_bar])
    bid_now = float(series["bid"][local_bar]) if np.isfinite(series["bid"][local_bar]) else None
    ask_now = float(series["ask"][local_bar]) if np.isfinite(series["ask"][local_bar]) else None

    return TradeIntent(
        trade=True,
        expiry=contract.expiry,
        strike=contract.strike,
        right=contract.right,
        qty=policy.qty,
        entry_ref_price=entry_mid,
        order_style=policy.order_style,
        tif=policy.tif,
        stop_price=max(0.01, entry_mid * (1.0 - policy.stop_pct)),
        take_profit_price=entry_mid * (1.0 + policy.target_pct),
        max_hold_bars=policy.max_hold_bars,
        exit_policy=policy.exit_policy,
        confidence=min(1.0, max(0.0, best_score * 3.0)),
        reason_codes=(f"score={best_score:.3f}", f"row={best_row}", f"idx={contract_idx}"),
        bar_index=local_bar,
        timestamp=str(int(sidecar["bar_timestamps"][local_bar])) if len(sidecar["bar_timestamps"]) else "",
        intent_id=str(uuid.uuid4()),
        bid_at_decision=bid_now,
        ask_at_decision=ask_now,
        underlying_price=spot_price,
        decision_day=sidecar["date"],
        snapshot_row=best_row,
        contract_index=contract_idx,
        contract_score=best_score,
        no_trade_score=abstain_score,
    )


def replay_validation(
    model: TradingModel,
    data: dict,
    mask_key: str = "promote_mask",
    max_days: int | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
    device: str = "cpu",
    collect_traces: bool = False,
    trace_path: str | None = None,
) -> tuple[ReplayMetrics, list, list[DecisionTrace] | None]:
    features = data["X"].numpy()
    sim_features = data["X_sim"].numpy() if "X_sim" in data else features
    mask = data[mask_key].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    spot_prices = data["spot_prices"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    max_contracts = int(data["metadata"]["max_contracts_per_bar"])

    mask_indices = np.where(mask)[0]
    if len(mask_indices) == 0:
        return ReplayMetrics(), [], [] if collect_traces else None

    eval_dates = sorted(set(dates[i] for i in mask_indices))
    if max_days is not None:
        eval_dates = eval_dates[:max_days]

    day_to_bars = _build_day_index(dates)
    eligible: list[tuple[str, int, int]] = []
    snapshots = []

    for day in eval_dates:
        for bar_idx in day_to_bars.get(day, []):
            if bar_idx < LOOKBACK:
                continue
            bod = int(bar_of_day[bar_idx])
            if bod < policy.no_trade_before_bar or bod >= policy.no_trade_after_bar:
                continue
            sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
            contracts, labels, contract_indices = padded_snapshot(sidecar, bod, max_contracts)
            eligible.append((day, bar_idx, bod))
            snapshots.append((contracts, labels, contract_indices))

    if not eligible:
        return ReplayMetrics(), [], [] if collect_traces else None

    window_indices = np.array([bar_idx for _, bar_idx, _ in eligible], dtype=np.int32)
    offsets = np.arange(-LOOKBACK, 0).reshape(1, -1)
    gather_idx = window_indices.reshape(-1, 1) + offsets
    all_windows = features[gather_idx]
    all_contracts = np.stack([s[0] for s in snapshots]).astype(np.float32)
    all_contract_labels = np.stack([s[1] for s in snapshots]).astype(np.float32)
    all_contract_indices = np.stack([s[2] for s in snapshots]).astype(np.int32)

    model = model.to(device)
    model.eval()

    outputs_all = []
    t_inf = time.time()
    with torch.no_grad():
        for start in range(0, len(eligible), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(eligible))
            batch_x = torch.from_numpy(all_windows[start:end]).to(device)
            batch_c = torch.from_numpy(all_contracts[start:end]).to(device)
            batch_out = model(batch_x, batch_c)
            outputs_all.append({k: v.cpu() for k, v in batch_out.items()})
    all_outputs = {k: torch.cat([o[k] for o in outputs_all], dim=0) for k in outputs_all[0]}
    print(f"  Inference: {time.time() - t_inf:.1f}s ({len(eligible)} bars)")

    from v2.core.features import _FEAT_IDX

    trades = []
    traces: list[DecisionTrace] = [] if collect_traces else []
    current_day = None
    in_trade = False
    trade_exit_bar = -1
    last_stop_bar = -policy.cooldown_bars - 1
    daily_dollar_pnl = 0.0
    daily_loss_cap_hit = False
    cumulative_equity = policy.starting_equity
    num_days = 0
    vix_idx = _FEAT_IDX.get("vix_regime", 14)

    for i, (day, global_bar, local_bar) in enumerate(eligible):
        if day != current_day:
            current_day = day
            num_days += 1
            in_trade = False
            trade_exit_bar = -1
            last_stop_bar = -policy.cooldown_bars - 1
            daily_dollar_pnl = 0.0
            daily_loss_cap_hit = False

        # --- Skip checks with trace capture ---
        outputs_i = {k: v[i] for k, v in all_outputs.items()}
        c_scores_np = outputs_i["contract_scores"].numpy()
        v_mask_np = outputs_i["valid_mask"].numpy().astype(bool)
        if "no_trade_score" in outputs_i:
            no_trade_val = float(outputs_i["no_trade_score"].item())
        elif "gate_logit" in outputs_i:
            no_trade_val = -float(outputs_i["gate_logit"].item())
        else:
            no_trade_val = 0.0
        vix_val = float(features[global_bar, vix_idx]) if global_bar < len(features) else 0.0

        def _make_trace(decision: str, skip_reason: str = "",
                        sel_strike: float = 0.0, sel_right: str = "",
                        sel_mid: float = 0.0, sel_idx: int = -1) -> DecisionTrace:
            return build_trace_for_bar(
                date=day, bar_of_day=local_bar, global_bar_idx=global_bar,
                spot_price=float(spot_prices[global_bar]),
                vix_regime=vix_val,
                no_trade_score=no_trade_val,
                contract_scores=c_scores_np, valid_mask=v_mask_np,
                contract_features=all_contracts[i],
                contract_labels=all_contract_labels[i],
                contract_indices=all_contract_indices[i],
                decision=decision, skip_reason=skip_reason,
                selected_strike=sel_strike, selected_right=sel_right,
                selected_mid=sel_mid, selected_contract_idx=sel_idx,
            )

        if cumulative_equity <= 0 or daily_loss_cap_hit:
            if collect_traces:
                reason = "equity_zero" if cumulative_equity <= 0 else "loss_cap"
                traces.append(_make_trace("no_trade", skip_reason=reason))
            continue
        if in_trade and global_bar <= trade_exit_bar:
            if collect_traces:
                traces.append(_make_trace("no_trade", skip_reason="in_position"))
            continue
        if global_bar - last_stop_bar < policy.cooldown_bars:
            if collect_traces:
                traces.append(_make_trace("no_trade", skip_reason="cooldown"))
            continue

        sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        contract_features_t = torch.from_numpy(all_contracts[i])
        contract_indices_t = torch.from_numpy(all_contract_indices[i])
        # Use opportunity_logit as primary gate when available
        gate_logit_val = outputs_i.get("opportunity_logit", outputs_i.get("gate_logit"))
        intent = model_to_intent(
            no_trade_score=outputs_i.get("no_trade_score"),
            gate_logit=gate_logit_val,
            direction_logit=outputs_i.get("direction_logit"),
            contract_scores=outputs_i["contract_scores"],
            contract_labels=torch.from_numpy(all_contract_labels[i]),
            valid_mask=outputs_i["valid_mask"],
            contract_features=contract_features_t,
            contract_indices=contract_indices_t,
            sidecar=sidecar,
            local_bar=local_bar,
            spot_price=float(spot_prices[global_bar]),
            policy=policy,
        )
        if not intent.trade:
            if collect_traces:
                traces.append(_make_trace("no_trade", skip_reason="gate"))
            continue

        series = extract_contract_series(sidecar, intent.contract_index)["mid"].astype(np.float32)
        trade = simulate_trade(
            intent=intent,
            option_prices=series,
            features=sim_features[day_to_bars[day]],
            bar_of_day=np.arange(len(day_to_bars[day]), dtype=np.int32),
            dates=[day] * len(day_to_bars[day]),
            global_entry_bar=local_bar,
            breakeven_trigger_pct=policy.breakeven_trigger_pct,
            extra_trailing_tiers=policy.extra_trailing_tiers,
        )

        if trade is None:
            if collect_traces:
                traces.append(_make_trace(
                    "no_trade", skip_reason="fill_failed",
                    sel_strike=intent.strike or 0.0,
                    sel_right=intent.right or "",
                    sel_mid=intent.entry_ref_price or 0.0,
                    sel_idx=intent.contract_index,
                ))
            continue

        trade.trade_date = day
        trades.append(trade)
        in_trade = True
        trade_exit_bar = global_bar + (trade.exit_bar - local_bar)
        dollar_pnl = trade.net_pnl_pct * trade.entry_price * policy.contract_multiplier * policy.qty
        daily_dollar_pnl += dollar_pnl
        cumulative_equity += dollar_pnl
        if daily_dollar_pnl < 0 and abs(daily_dollar_pnl) / policy.starting_equity >= policy.daily_loss_cap_pct:
            daily_loss_cap_hit = True
        if trade.exit_reason == "STOP_LOSS":
            last_stop_bar = trade_exit_bar

        # Backfill trace with realized outcome
        if collect_traces:
            t = _make_trace(
                "trade",
                sel_strike=intent.strike or 0.0,
                sel_right=intent.right or "",
                sel_mid=intent.entry_ref_price or 0.0,
                sel_idx=intent.contract_index,
            )
            t.model_pnl = trade.net_pnl_pct
            t.exit_reason = trade.exit_reason
            t.bars_held = trade.bars_held
            if np.isfinite(t.oracle_pnl):
                t.delta_pnl = t.model_pnl - t.oracle_pnl
            traces.append(t)

    metrics = compute_metrics(trades, num_days=max(num_days, 1), starting_equity=policy.starting_equity, contract_multiplier=policy.contract_multiplier)

    # Save and summarize traces
    if collect_traces:
        if trace_path:
            save_traces(traces, trace_path)
            print(f"  Traces saved: {trace_path} ({len(traces)} bars)")
        print_trace_summary(traces)

    return metrics, trades, traces if collect_traces else None


def replay_sequential(
    agent,
    data: dict,
    test_days: list[str],
    policy: DecisionPolicy | None = None,
    sidecar_dir: str = "v2/data_sidecars",
    device: str = "cpu",
    deterministic: bool = True,
) -> tuple[ReplayMetrics, list, list[dict]]:
    """Run the sequential agent through full-day episodes and collect trades.

    Returns (metrics, trades, episode_summaries) using the same compute_metrics()
    as the supervised replay, so scores are directly comparable.
    """
    from v2.core.env import TradingEnv, ACT_HOLD, ACT_ENTER_CALL, ACT_ENTER_PUT, ACT_EXIT

    policy = policy or DEFAULT_POLICY
    env = TradingEnv(data, policy, sidecar_dir, agent.encoder, device, LOOKBACK)

    all_trades = []
    episode_summaries = []
    agent.eval()

    for day in test_days:
        obs = env.reset(day)
        if env._done:
            continue

        day_actions = []
        day_bars = []
        day_rewards = []
        day_trades = []
        day_exit_reasons = []  # list of (step_idx, exit_reason) in sequence
        day_entry_sides = []  # track side of each entry for post-stop analysis
        step_counter = 0

        while not env._done:
            # Build tensors for agent
            gi, local_bar = env._eligible_bars[env._step_idx]
            window = torch.from_numpy(
                env.features[gi - env.lookback: gi].numpy()
            ).float().to(device)
            contracts_np, _, _ = padded_snapshot(
                env._sidecar, local_bar, env.max_contracts
            )
            contracts_t = torch.from_numpy(contracts_np).float().to(device)
            session_t = torch.from_numpy(obs.session_state).float().to(device)

            with torch.no_grad():
                action, log_prob, value = agent.act(
                    window, contracts_t, session_t, deterministic=deterministic
                )

            obs, reward, done, info = env.step(action)
            day_actions.append(info.action_taken)
            day_bars.append(local_bar)
            day_rewards.append(reward)

            if info.trade_opened:
                day_entry_sides.append("call" if info.action_taken == ACT_ENTER_CALL else "put")

            if info.trade_closed and info.trade_pnl != 0:
                day_exit_reasons.append((step_counter, info.exit_reason))
                # Build a SimulatedTrade-like record for metrics
                from v2.core.schema import SimulatedTrade
                trade = SimulatedTrade(
                    intent=TradeIntent(trade=True, bar_index=info.bar, decision_day=day),
                    entry_bar=info.bar,
                    entry_price=env._position_entry_price if env._position_entry_price > 0 else 1.0,
                    exit_bar=info.bar + 1,
                    exit_price=0.0,
                    exit_reason=info.exit_reason,
                    net_pnl_pct=info.trade_pnl,
                    raw_pnl_pct=info.trade_pnl + 0.02,  # approx pre-spread
                    spread_cost_pct=0.02,
                    bars_held=1,
                    trade_date=day,
                )
                day_trades.append(trade)
                all_trades.append(trade)

            step_counter += 1

        # Episode summary with extended behavioral data
        from collections import Counter
        action_counts = Counter(day_actions)
        exit_reason_counts = Counter(r for _, r in day_exit_reasons)
        episode_summaries.append({
            "day": day,
            "steps": len(day_actions),
            "trades": len(day_trades),
            "day_pnl": env._day_pnl,
            "max_dd": env._max_drawdown,
            "actions": dict(action_counts),
            "hold_rate": action_counts[ACT_HOLD] / max(len(day_actions), 1),
            "side_flips": _count_side_flips(day_actions),
            # Extended behavioral fields
            "actions_sequence": list(day_actions),
            "bars": list(day_bars),
            "exit_reasons": dict(exit_reason_counts),
            "exit_events": list(day_exit_reasons),  # [(step_idx, reason), ...]
            "entry_sides": list(day_entry_sides),
        })

    # Compute metrics through standard harness
    metrics = compute_metrics(
        all_trades,
        num_days=len(test_days),
        starting_equity=policy.starting_equity,
        contract_multiplier=policy.contract_multiplier,
    )

    # Print behavioral report if we have episodes
    if episode_summaries:
        from v2.analysis.behavioral_report import print_behavioral_report
        print_behavioral_report(episode_summaries, all_trades, metrics)

    return metrics, all_trades, episode_summaries


def _count_side_flips(actions: list[int]) -> int:
    """Count how many times the agent flips between call and put entries within a day."""
    # Action constants: 1=ENTER_CALL, 2=ENTER_PUT
    last_side = None
    flips = 0
    for a in actions:
        if a == 1:  # ENTER_CALL
            if last_side == "put":
                flips += 1
            last_side = "call"
        elif a == 2:  # ENTER_PUT
            if last_side == "call":
                flips += 1
            last_side = "put"
    return flips


def _baseline_cache_key(data: dict, mask_key: str, policy: DecisionPolicy, max_days: int | None) -> str:
    dataset_fp = data.get("metadata", {}).get("fingerprint", "unknown")
    parts = [dataset_fp, mask_key, policy.fingerprint(), score_config_fingerprint(), str(max_days)]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _load_cached_baselines(cache_key: str) -> dict | None:
    if not os.path.exists(BASELINE_CACHE_PATH):
        return None
    try:
        with open(BASELINE_CACHE_PATH) as f:
            cache = json.load(f)
        if cache.get("cache_key") == cache_key:
            return cache.get("baselines")
    except (json.JSONDecodeError, OSError):
        return None
    return None


def _save_baseline_cache(cache_key: str, baselines: dict) -> None:
    with open(BASELINE_CACHE_PATH, "w") as f:
        json.dump({"cache_key": cache_key, "baselines": baselines}, f, indent=2)


def _dict_to_replay_metrics(d: dict) -> ReplayMetrics:
    m = ReplayMetrics()
    for k, v in d.items():
        if hasattr(m, k):
            setattr(m, k, v)
    return m


def _select_snapshot_row(sidecar: dict, local_bar: int, selector: str, right: str | None = None, rng: np.random.RandomState | None = None) -> tuple[int, int] | None:
    feats, labels, contract_indices = padded_snapshot(sidecar, local_bar, int(sidecar["bar_ptrs"][local_bar + 1] - sidecar["bar_ptrs"][local_bar]))
    if len(contract_indices) == 0:
        return None
    valid = feats[:, 0] > 0.5
    valid &= np.isfinite(labels)
    if right is not None:
        valid &= ((feats[:, 2] > 0.5) if right == "P" else (feats[:, 2] < 0.5))
    rows = np.where(valid)[0]
    if len(rows) == 0:
        return None
    if selector == "random":
        assert rng is not None
        row = int(rng.choice(rows))
    else:
        dists = np.abs(feats[rows, 12])
        row = int(rows[np.argmin(dists)])
    return row, int(contract_indices[row])


def _trade_from_baseline(sidecar: dict, contract_idx: int, local_bar: int, policy: DecisionPolicy, right_reason: str, spot_price: float) -> TradeIntent:
    contract = describe_contract(sidecar, contract_idx)
    series = extract_contract_series(sidecar, contract_idx)
    entry_mid = float(series["mid"][local_bar])
    return TradeIntent(
        trade=True,
        expiry=contract.expiry,
        strike=contract.strike,
        right=contract.right,
        qty=policy.qty,
        entry_ref_price=entry_mid,
        order_style=policy.order_style,
        tif=policy.tif,
        stop_price=max(0.01, entry_mid * (1.0 - policy.stop_pct)),
        take_profit_price=entry_mid * (1.0 + policy.target_pct),
        max_hold_bars=policy.max_hold_bars,
        exit_policy=policy.exit_policy,
        confidence=0.5,
        reason_codes=(right_reason,),
        bar_index=local_bar,
        timestamp=str(int(sidecar["bar_timestamps"][local_bar])) if len(sidecar["bar_timestamps"]) else "",
        intent_id=str(uuid.uuid4()),
        underlying_price=spot_price,
        contract_index=contract_idx,
    )


def _compute_baseline_common(data: dict, mask_key: str, policy: DecisionPolicy, chooser) -> ReplayMetrics:
    dates = data["dates"]
    mask = data[mask_key].numpy()
    sim_features = data["X_sim"].numpy() if "X_sim" in data else data["X"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    day_to_bars = _build_day_index(dates)
    eval_dates = sorted(set(dates[i] for i in np.where(mask)[0]))
    all_trades = []
    num_days = 0

    for day in eval_dates:
        bars = day_to_bars[day]
        if len(bars) < 50:
            continue
        sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        num_days += 1
        result = chooser(data, sidecar, day, bars, policy)
        if result is None:
            continue
        contract_idx, local_bar, intent = result
        series = extract_contract_series(sidecar, contract_idx)["mid"].astype(np.float32)
        trade = simulate_trade(
            intent,
            series,
            sim_features[bars],
            np.arange(len(bars), dtype=np.int32),
            [day] * len(bars),
            local_bar,
            breakeven_trigger_pct=policy.breakeven_trigger_pct,
            extra_trailing_tiers=policy.extra_trailing_tiers,
        )
        if trade:
            trade.trade_date = day
            all_trades.append(trade)

    return compute_metrics(all_trades, num_days=max(num_days, 1), starting_equity=policy.starting_equity, contract_multiplier=policy.contract_multiplier)


def compute_baseline_random(data: dict, mask_key: str = "promote_mask", n_seeds: int = 5, max_days: int | None = None, policy: DecisionPolicy = DEFAULT_POLICY, day_to_bars: dict[str, list[int]] | None = None) -> ReplayMetrics:
    all_trades = []
    total_num_days = 0
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        num_days = 0

        def chooser(_data, sidecar, _day, bars, pol):
            valid_locals = [int(data["bar_of_day"][b]) for b in bars if pol.no_trade_before_bar <= int(data["bar_of_day"][b]) < pol.no_trade_after_bar]
            if not valid_locals:
                return None
            local_bar = int(rng.choice(valid_locals))
            choice = _select_snapshot_row(sidecar, local_bar, "random", rng=rng)
            if choice is None:
                return None
            _, contract_idx = choice
            spot_price = float(data["spot_prices"][bars[local_bar]])
            return contract_idx, local_bar, _trade_from_baseline(sidecar, contract_idx, local_bar, pol, "random", spot_price)

        dates = data["dates"]
        mask = data[mask_key].numpy()
        sidecar_dir = data["metadata"]["chain_sidecar_dir"]
        sim_features = data["X_sim"].numpy() if "X_sim" in data else data["X"].numpy()
        if day_to_bars is None:
            day_to_bars_local = _build_day_index(dates)
        else:
            day_to_bars_local = day_to_bars
        eval_dates = sorted(set(dates[i] for i in np.where(mask)[0]))
        if max_days is not None:
            eval_dates = eval_dates[:max_days]

        for day in eval_dates:
            bars = day_to_bars_local[day]
            if len(bars) < 50:
                continue
            sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
            num_days += 1
            result = chooser(data, sidecar, day, bars, policy)
            if result is None:
                continue
            contract_idx, local_bar, intent = result
            series = extract_contract_series(sidecar, contract_idx)["mid"].astype(np.float32)
            trade = simulate_trade(
                intent,
                series,
                sim_features[bars],
                np.arange(len(bars), dtype=np.int32),
                [day] * len(bars),
                local_bar,
                breakeven_trigger_pct=policy.breakeven_trigger_pct,
                extra_trailing_tiers=policy.extra_trailing_tiers,
            )
            if trade:
                trade.trade_date = day
                all_trades.append(trade)
        total_num_days += num_days

    avg_num_days = max(total_num_days // max(n_seeds, 1), 1)
    return compute_metrics(all_trades, num_days=avg_num_days, starting_equity=policy.starting_equity, contract_multiplier=policy.contract_multiplier)


def compute_baseline_atm_always(data: dict, mask_key: str = "promote_mask", max_days: int | None = None, policy: DecisionPolicy = DEFAULT_POLICY, day_to_bars: dict[str, list[int]] | None = None) -> ReplayMetrics:
    def chooser(_data, sidecar, _day, bars, pol):
        local_bar = 30
        choice = _select_snapshot_row(sidecar, local_bar, "atm", right="C")
        if choice is None:
            return None
        _, contract_idx = choice
        spot_price = float(data["spot_prices"][bars[local_bar]])
        intent = _trade_from_baseline(sidecar, contract_idx, local_bar, pol, "atm_always", spot_price)
        intent = dataclass_replace(intent, exit_policy="STOP_TP_TIME")
        return contract_idx, local_bar, intent

    return _compute_baseline_common(data, mask_key, policy, chooser)


def compute_baseline_simple_rules(data: dict, mask_key: str = "promote_mask", max_days: int | None = None, policy: DecisionPolicy = DEFAULT_POLICY, day_to_bars: dict[str, list[int]] | None = None) -> ReplayMetrics:
    ret_idx = 0

    def chooser(_data, sidecar, _day, bars, pol):
        for b in bars:
            local_bar = int(data["bar_of_day"][b])
            if local_bar < pol.no_trade_before_bar or local_bar >= pol.no_trade_after_bar:
                continue
            momentum = float(data["X"][b, ret_idx])
            if abs(momentum) < 0.005:
                continue
            right = "C" if momentum > 0 else "P"
            choice = _select_snapshot_row(sidecar, local_bar, "atm", right=right)
            if choice is None:
                continue
            _, contract_idx = choice
            spot_price = float(data["spot_prices"][b])
            intent = _trade_from_baseline(sidecar, contract_idx, local_bar, pol, "simple_rules", spot_price)
            intent = dataclass_replace(intent, exit_policy="STOP_TP_TIME")
            return contract_idx, local_bar, intent
        return None

    return _compute_baseline_common(data, mask_key, policy, chooser)


def compute_baseline_atm_trailing(data: dict, mask_key: str = "promote_mask", max_days: int | None = None, policy: DecisionPolicy = DEFAULT_POLICY, day_to_bars: dict[str, list[int]] | None = None) -> ReplayMetrics:
    def chooser(_data, sidecar, _day, bars, pol):
        local_bar = 30
        choice = _select_snapshot_row(sidecar, local_bar, "atm", right="C")
        if choice is None:
            return None
        _, contract_idx = choice
        spot_price = float(data["spot_prices"][bars[local_bar]])
        intent = _trade_from_baseline(sidecar, contract_idx, local_bar, pol, "atm_trailing", spot_price)
        intent = dataclass_replace(intent, exit_policy="TRAILING")
        return contract_idx, local_bar, intent

    return _compute_baseline_common(data, mask_key, policy, chooser)


def dataclass_replace(intent: TradeIntent, **kwargs) -> TradeIntent:
    d = intent.to_dict()
    d.update(kwargs)
    return TradeIntent.from_dict(d)


def print_metrics(name: str, m: ReplayMetrics):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Score={m.score:.4f}  PF={m.profit_factor:.3f}  WR={m.win_rate:.1%}")
    print(f"  Trades={m.total_trades}  TPD={m.trades_per_day:.2f}  Days={m.num_days}  Traded={m.traded_days}")
    print(f"  Sortino={m.daily_sortino:.2f}  +DayRate={m.positive_day_rate:.1%}  AcctDD={m.max_account_drawdown:.1%}")
    print(
        f"  Direction={m.call_count}C / {m.put_count}P  "
        f"Minority={m.minority_side_share:.1%}  Balance={m.direction_balance:.2f}"
    )
    if m.gate_failure:
        print(f"  GATE FAILURE: {m.gate_failure}")


def _compute_all_baselines(data, mask_key, max_days, policy, day_to_bars):
    cache_key = _baseline_cache_key(data, mask_key, policy, max_days)
    cached = _load_cached_baselines(cache_key)
    if cached is not None and "atm_trailing" in cached:
        print("  (baselines loaded from cache)")
        return (
            _dict_to_replay_metrics(cached["random"]),
            _dict_to_replay_metrics(cached["atm_always"]),
            _dict_to_replay_metrics(cached["simple_rules"]),
            _dict_to_replay_metrics(cached["atm_trailing"]),
        )

    b_random = compute_baseline_random(data, mask_key=mask_key, max_days=max_days, policy=policy, day_to_bars=day_to_bars)
    b_atm = compute_baseline_atm_always(data, mask_key=mask_key, max_days=max_days, policy=policy, day_to_bars=day_to_bars)
    b_rules = compute_baseline_simple_rules(data, mask_key=mask_key, max_days=max_days, policy=policy, day_to_bars=day_to_bars)
    b_trailing = compute_baseline_atm_trailing(data, mask_key=mask_key, max_days=max_days, policy=policy, day_to_bars=day_to_bars)
    _save_baseline_cache(
        cache_key,
        {
            "random": b_random.to_dict(),
            "atm_always": b_atm.to_dict(),
            "simple_rules": b_rules.to_dict(),
            "atm_trailing": b_trailing.to_dict(),
        },
    )
    return b_random, b_atm, b_rules, b_trailing


def main():
    parser = argparse.ArgumentParser(description="v4 exact-chain replay evaluation")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--artifact", type=str, default=None)
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--mask", type=str, default="promote", choices=["val", "promote", "shadow"])
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--traces", action="store_true", help="Collect per-bar decision traces")
    parser.add_argument("--gate", type=float, default=None)
    parser.add_argument("--sequential", action="store_true", help="Run sequential agent replay")
    parser.add_argument("--seq-model", type=str, default="v2/models/seq_agent.pt",
                        help="Path to sequential agent checkpoint")
    args = parser.parse_args()

    mask_key = f"{args.mask}_mask"
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    dataset_fp = data.get("metadata", {}).get("fingerprint")
    policy = DEFAULT_POLICY
    if args.gate is not None:
        policy = DecisionPolicy(gate_threshold=args.gate)

    day_to_bars = _build_day_index(data["dates"])

    # --- Sequential agent replay mode ---
    if args.sequential:
        from v2.seq_agent import SequentialAgent
        from v2.train import TradingModel, D_MODEL

        model_path = args.model or "v2/models/model.pt"
        encoder = TradingModel()
        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in ckpt:
                    encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
                print(f"Encoder loaded from {model_path}")
            except RuntimeError as e:
                print(f"WARNING: Could not load {model_path} ({e}), using random encoder")
        else:
            print(f"WARNING: No encoder at {model_path}, using random encoder")
        encoder.eval()

        agent = SequentialAgent(encoder, context_dim=D_MODEL, freeze_encoder=True)
        if os.path.exists(args.seq_model):
            seq_ckpt = torch.load(args.seq_model, map_location="cpu", weights_only=False)
            agent.load_state_dict(seq_ckpt["agent_state_dict"], strict=False)
            print(f"Sequential agent loaded from {args.seq_model}")
        else:
            print(f"WARNING: No seq agent at {args.seq_model}, using untrained agent")

        # Determine test days from mask
        all_days = sorted(set(data["dates"]))
        if mask_key in data:
            mask = data[mask_key]
            test_day_set = set()
            dates = data["dates"]
            for i in range(len(dates)):
                if mask[i]:
                    test_day_set.add(dates[i])
            test_days = sorted(test_day_set)
        else:
            # Fallback: last 60 days
            test_days = all_days[-60:]

        if args.days:
            test_days = test_days[:args.days]

        print(f"Sequential replay on {len(test_days)} days ({mask_key})")
        metrics, trades, episode_summaries = replay_sequential(
            agent, data, test_days, policy=policy, deterministic=True,
        )
        print_metrics("Sequential Agent", metrics)
        return

    if args.baselines:
        b_random, b_atm, b_rules, b_trailing = _compute_all_baselines(data, mask_key, args.days, policy, day_to_bars)
        print_metrics("Random", b_random)
        print_metrics("ATM-Always", b_atm)
        print_metrics("Simple-Rules", b_rules)
        print_metrics("ATM-Trailing", b_trailing)
        return

    if args.model:
        model = load_model_from_path(args.model)
        print(f"Model loaded from raw path {args.model}")
    elif args.artifact:
        from v2.ops.artifact import load_artifact

        result = load_artifact(args.artifact, current_dataset_fingerprint=dataset_fp)
        model = result["model"]
        policy = result["policy"]
        print(f"Model loaded from artifact {args.artifact} (score={result['manifest']['score']:.4f})")
    else:
        model, artifact_policy, manifest = load_best_model(current_dataset_fingerprint=dataset_fp)
        if args.gate is None:
            policy = artifact_policy

    trace_flag = getattr(args, "traces", False)
    trace_out = f"v2/artifacts/replay_traces.csv" if trace_flag else None
    metrics, trades, _ = replay_validation(
        model, data, mask_key=mask_key, max_days=args.days, policy=policy,
        collect_traces=trace_flag, trace_path=trace_out,
    )
    print_metrics("Model Replay", metrics)
    print(f"\n--- BASELINES (on {mask_key}) ---")
    b_random, b_atm, b_rules, b_trailing = _compute_all_baselines(data, mask_key, args.days, policy, day_to_bars)
    print_metrics("Random", b_random)
    print_metrics("ATM-Always", b_atm)
    print_metrics("Simple-Rules", b_rules)
    print_metrics("ATM-Trailing", b_trailing)


if __name__ == "__main__":
    main()
