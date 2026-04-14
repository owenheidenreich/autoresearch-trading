# Handoff: Wave 1 Complete — Session Agent Prototype

Read this file, then [docs/founder_intent.md](docs/founder_intent.md), then [program.md](program.md).

For domain knowledge: [docs/domain/](docs/domain/) contains 0DTE options knowledge. Read before making architecture decisions.

## What Happened in Wave 1

Wave 1 ran 2026-04-13. Goal was to establish bar-level judgment quality (opportunity, side, aggression) using supervised learning on a path-aware dataset. Five GPU experiments (exp_154–158) all scored ≤ -0.200.

**The experiments were diagnostic, not failures.** They proved:

1. **The path library works.** Strict opportunity labels (58.7% positive, down from 87.8%) changed model behavior in the intended direction — trade rate compressed during training, call bias eased.

2. **The old opportunity target was wrong.** `label_trade` (87.8% positive) answered "did the oracle find some profitable contract?" — too permissive for risk control. The strict label asks "did a robust, survivable long-premium setup exist?" using raw 10-bar returns > 12%, MAE > -8%, breakeven < 5 bars, MFE > 3%.

3. **The standalone opportunity head cannot learn.** opp_loss BCE is monotonically increasing after epoch 2. The head outputs near-constant logits. A 96-dim static context vector does not contain enough information to separate good bars from bad.

4. **Selectivity can emerge from the system** — later training epochs show trade rate dropping to 50%, but this comes from contract scoring shifting, not the opportunity head. The information is there; it can't be expressed through a single BCE head over static context.

5. **This argues FOR the session agent.** If opportunity cannot be solved from one bar's features alone, it requires session memory: what happened earlier, whether prior thesis was confirmed, how much time/decay budget remains.

## What Was Built

### Data Infrastructure (rebuilt 2026-04-13)

| Component | Spec |
|-----------|------|
| `v2/data.pt` | 382,920 bars, 986 days, 52 context features, 22 contract features |
| `v2/data_sidecars/*.pt` | Schema `v4_exact_chain_v2_paths` |
| Context features | 32 price/market + 12 option/greeks + 8 flow = 52 |
| Contract features | 19 base + 3 economic (theta_to_premium, breakeven_bars_est, gamma_dollar) = 22 |
| Intraday phase | sin/cos cyclical time + discrete phase bucket (opening/mid-morn/lunch/afternoon/power-hr/close) |
| Path library | Raw returns at [5,10,15,30,60] bars, MFE/MAE at each horizon, bars_to_breakeven, impulse_fraction |
| Multi-policy labels | Short (30-bar), Long (120-bar), EOD hold overlays per contract per bar |
| Strict opportunity | Computed at training time from path library: raw10>12%, mae10>-8%, btbe<5, mfe5>3% |

### Session Agent Architecture (new files)

| File | What |
|------|------|
| `v2/core/env.py` | `TradingEnv` — gym-like step interface. 4 actions: {hold, enter_call, enter_put, exit}. 12-dim session state. Per-step rewards. One episode = one trading day. |
| `v2/seq_agent.py` | `SequentialAgent` — frozen TradingModel encoder (417k params) + trainable policy/value heads (17k params). Conditions on 96-dim market context + 32-dim session embedding → 4-action logits + value estimate. |
| `v2/train_seq.py` | Behavioral cloning from oracle trajectories (strict opportunity + oracle side + -10% exit threshold). Class-weighted cross-entropy. REINFORCE stub for phase 4. |
| `v2/replay.py` | `replay_sequential()` — runs agent through TradingEnv, collects SimulatedTrade objects, feeds to standard compute_metrics(). Produces per-episode summaries with action counts, hold rate, side flips. |

### Model Architecture (modified files)

| File | Change |
|------|--------|
| `v2/train.py` | TradingModel exposes `context` vector in forward() return dict. Opportunity/side/aggression heads present but zeroed (SIDE_W=0, AGG_W=0). NUM_FEATURES=52, NUM_CONTRACT_FEATURES=22. Strict opportunity label computed at training time. Checkpoint on opp_loss when using strict label. |
| `v2/core/policy.py` | Added SHORT_POLICY (30-bar, 20% stop) and EOD_POLICY (hold to close, 50% stop) for multi-policy path library. |
| `v2/core/chain_data.py` | Schema bumped to `v4_exact_chain_v2_paths`. 3 new contract features (theta_to_premium, breakeven_bars_est, gamma_dollar). |
| `v2/pipeline/compute_features.py` | 3 new context features (intraday_sin, intraday_cos, intraday_phase). N_FEAT 29→32. |
| `v2/core/features.py` | Feature list updated to 52. New features excluded from z-score normalization. |
| `v2/pipeline/build_v2_dataset.py` | Path library computation: `_compute_path_metrics()`, `_simulate_under_policy()`. Multi-horizon raw returns, MFE/MAE, bars_to_breakeven, impulse_fraction. |
| `v2/ops/pre_run_gate.py` | Removed side_head/side_logit from legacy patterns. Updated doc requirement to "Opportunity head for independent gating". |
| `v2/docs/how_training_works.md` | Updated for new architecture: 52/22 features, opportunity/side/aggression heads, softer supervision. |

## Session State Vector (12-dim)

```
[0]  bar_of_session_norm      — local_bar / 390
[1]  in_position              — 0 or 1
[2]  bars_held_norm           — bars in position / max_hold
[3]  unrealized_pnl_frac      — current unrealized P&L if in position
[4]  position_mfe_frac        — running max favorable excursion
[5]  position_mae_frac        — running max adverse excursion
[6]  day_pnl_frac             — cumulative daily P&L / starting equity
[7]  num_trades_today_norm    — trades so far / 5
[8]  num_stops_today_norm     — stop-losses today / 2
[9]  bars_since_last_trade    — cooldown awareness (normalized)
[10] best_score_this_bar      — max contract score (opportunity proxy)
[11] position_side            — 0 (flat), 1 (long call), -1 (long put)
```

## Experiment History

| Exp | Hypothesis | Result | Key Finding |
|-----|-----------|--------|-------------|
| 154 | All heads stacked (opp+side+agg) + SOFT_TEMP 0.25 | -0.200, DD 52%, trade rate 84% | 6 loss terms competing in 96-dim bottleneck |
| 155 | Opportunity-only isolation (SIDE_W=0, AGG_W=0) | -0.200, DD 67%, trade rate 83% | Removing side/agg didn't help — problem is deeper |
| 156 | Strict opportunity label (58.7% positive) | -0.200, DD 60%, trade rate 77%→51% | **Trade rate compressed during training.** Strict label works but checkpoint picks epoch 2. |
| 157 | LR 1e-4 (slower learning for harder target) | -0.200, DD 66%, trade rate 76%→50% | Same pattern: later epochs selective but val loss prefers epoch 2 |
| 158 | Checkpoint on opp_loss instead of total loss | -0.200, DD 66% | opp_loss monotonically increasing — the head itself isn't learning |

**Conclusion:** Static per-bar supervised learning cannot solve the opportunity problem. The opportunity head fails because opportunity requires session context. This motivated the pivot to the session agent.

## BC Training Result (GPU, random encoder, 926 days)

```
Action distribution: HOLD=39666, CALL=825, PUT=800, EXIT=379
Best epoch: 18, val_loss: 0.4136, val_acc: 86.0%
```

Sequential replay on 60 test days:
- **139 trades**, 2.3/day (down from 3.5 in old system)
- **PF 1.007** — breakeven with a random encoder
- **80% hold rate** — agent watches more than it trades
- **52.5/47.5 call/put split** — much more balanced than old 69/31
- **86 side flips** — the main behavioral issue to fix

Some days show clean single-entry discipline (98% hold, 1 trade). Others show pathological rapid side flipping (14 flips in one day). The agent has learned sequential structure but hasn't fully stabilized thesis persistence.

## What To Trust

- `v2/data.pt` and `v2/data_sidecars/` — rebuilt 2026-04-13 with path library
- `v2/core/env.py`, `v2/seq_agent.py`, `v2/train_seq.py` — tested end-to-end
- `v2/replay.py` with `replay_sequential()` — produces comparable metrics
- This HANDOFF.md and the plan at `.claude/plans/eager-humming-beacon.md`
- Memory files at `.claude/projects/.../memory/` — project state, feedback, vision docs

## What Not To Trust

- `v2/models/model.pt` — stale from exp_146 era (47-feature encoder, 15-feature contracts). Will crash on current 52/22 data.
- `v2/models/seq_agent.pt` — trained with random encoder, useful for pipeline validation only
- Any score from exp_074–exp_146 — invalidated by scoring v3.0 reset
- Opportunity/side/aggression heads as standalone classifiers — they don't work in isolation

## Next Steps

### Immediate (exp_159–162)

1. **Train a supervised encoder** with new 52/22 feature dims to get a proper frozen backbone. Even if it doesn't pass screening, it provides learned representations for the session agent.

2. **Full BC training** with the pretrained encoder. The random-encoder BC already hit PF 1.007 — a real encoder should do better.

3. **Fix side-flip pathology.** The 86 side flips across 60 days indicate the agent hasn't learned thesis persistence. Options: add side-consistency reward, penalize flips in BC oracle trajectories, or add a "committed side" state variable.

4. **REINFORCE fine-tuning** (phase 4). Teach behaviors BC cannot: post-stop caution, day-PnL-aware aggression adjustment, time-decay-budget management.

### What success looks like

Not necessarily profitability yet. The behavioral signatures to watch:

- Fewer impulsive repeated entries
- Cleaner entry/hold/exit arcs
- Visible response to session damage (post-stop caution)
- Different behavior by time-of-day
- Less pathological overtrading in hostile regimes
- Fold 1 trajectories that look cautious rather than suicidal

The model should be able to express: "I was bullish at 10:02, the move failed by 10:11, theta is now less favorable, so I am standing down." That is a sequential judgment — what the finished system should aim for.

### Long-term (wave 3+)

After session agent proves thesis persistence:
- Unfreeze encoder for end-to-end fine-tuning
- Contract cross-attention (chain as market)
- Expand session window beyond bars 60-105
- Full trajectory optimization (RL with proper reward shaping)

## Data Limitations (permanent)

- **No open interest**: Polygon minute_aggs flat files don't include OI
- **No bid/ask**: Only OHLC + volume + transactions per contract per bar
- **Greeks are estimated**: Black-Scholes implied from OHLC close prices, not market quotes
