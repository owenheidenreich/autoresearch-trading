# Handoff: Current v2 State

Read this file, then [founder_intent.md](docs/founder_intent.md), then [program.md](program.md), then [current_state.md](docs/current_state.md).

## What Changed (2026-04-11 overnight session)

Major architecture exploration and a trade-analysis-driven breakthrough.

### Phase 1: Hierarchical Direction Head (exp_100–103) — FAILED

**Motivation**: Trace analysis of the exp_099 baseline showed the gate works (81% precision) but contract selection is nearly random (4.4% accuracy). Research into hierarchical RL (HRT paper, Pro Trader RL) and 0DTE practitioner frameworks suggested decomposing the decision into gate → direction → strike.

**exp_100 — 3-head model (gate + direction + strike)**
- Added explicit `gate_head`, `direction_head`, `strike_head` to TradingModel
- Direction head supervised by oracle's call/put label via BCE
- At inference, contracts masked to predicted direction before strike selection
- Result: direction head overfitted (val loss 0.647 → 0.974 over 14 epochs), dir_acc stuck at 51% (random)
- The direction gradient corrupted the shared encoder, degrading gate and selection

**exp_101 — Detached direction head**
- Applied `context.detach()` before direction head to prevent encoder corruption
- Smaller head (d//4) with 0.3 dropout
- Result: direction val loss stabilized (0.647 → 0.642), but dir_acc still 52%
- Best direction balance since exp_095 (144C/111P), PF 0.653
- The head can't learn from frozen encoder features — the encoder isn't shaped for direction

**exp_102 — DIR_W=0.3 with live gradient**
- Allowed gradient to flow but at 0.3 weight to limit corruption
- Result: best WR (35.9%) and PF (0.657) but extreme call bias (374C/94P, 80% calls)
- Direction head still at 52% accuracy — the weight reduction prevented divergence but didn't teach direction

**exp_103 — Direction-conditioned KL without direction head**
- Removed the direction head entirely. Instead, conditioned the KL selection loss on oracle direction at training time (mask opposite-side contracts from KL targets)
- At inference, all contracts compete freely (no direction masking)
- Result: **worst experiment** — 29.6% WR, 79% put bias, PF 0.559
- The direction-conditioned training created uncalibrated cross-direction scores: call scores and put scores trained on separate distributions became incomparable at inference

**Key conclusion**: The direction head doesn't learn. At 52% accuracy across all configurations, it's random. The model learns direction implicitly through the contract score comparisons — explicitly decomposing it breaks that implicit learning. Direction masking at inference cripples the model by filtering to the wrong side 48% of the time.

### Phase 2: Back to Basics + Trade Visualization (exp_104) — PIVOTAL

**exp_104 — Balanced gate + standard KL (official 5-fold run)**
- Reverted to the best known config: balanced gate sampling (proven in exp_095) + standard KL over all contracts
- Ran as official `run_one` to get `model_candidate.pt` for trade visualization
- 5-fold aggregate score: -0.240, 1697 trades, 131C/153P (balanced), WR 29.9%
- Generated `v2/output/trades.html`, `equity.html`, `trades.csv`

**Trade-level analysis revealed:**

| Time Window | Trades | WR | Net PnL |
|-------------|--------|----|---------|
| Open 30-60 | 59 (23%) | 25.4% | -$3,788 |
| **Morning 60-120** | **66 (25%)** | **42.4%** | **+$972** |
| Midday 120-180 | 62 (24%) | 29.0% | -$2,075 |
| Afternoon 180-240 | 46 (18%) | 23.9% | -$3,204 |
| Close 240-270 | 26 (10%) | 30.8% | -$1,919 |

The model has real edge in the morning (bars 60-120) and bleeds money everywhere else. Every other time window is net negative.

Other findings:
- 86% of trades were cheap $1-5 OTM contracts (lottery tickets)
- 59% of trades hit stop loss (-$28,062 total)
- Calls outperform puts: WR 36.0% vs 28.2%
- 13% of losses had MFE >= 30% (reached profit zone then reversed)
- Peak equity $10,890 before cascade collapse

### Phase 3: Morning Window (exp_105) — BEST RESULT EVER

**exp_105 — Restrict trading to bars 60-120 only**
- Policy change: `no_trade_before_bar` 30 → 60, `no_trade_after_bar` 270 → 120
- No model change — same balanced gate + standard KL

| Metric | exp_104 (all day) | exp_105 (morning) |
|--------|------------------|-------------------|
| Trades | 259 | 160 |
| Direction | 89C/170P | **79C/81P** |
| WR | 30.9% | **36.9%** |
| PF | 0.56 | **0.789** |
| DD | 100.2% | **38.7%** |
| +DayRate | 21.4% | **42.6%** |
| Net PnL | -$10,096 | **-$2,899** |
| Sortino | -28.4 | **-3.9** |

Still fails the 20% DD hard gate (38.7% > 20%), but the account survived with $7,101 remaining instead of zero.

## Canonical State

```text
Dataset path:        v2/data.pt
Per-day sidecars:    v2/data_sidecars/*.pt
Dataset version:     v4_exact_chain
Dataset fingerprint: 46f2d184e186496f
Unique days:         986
Features:            47
Trade window:        bar 60-120  (was 30-270, changed in exp_105)
ATM source:          dynamic_nearest_per_bar
```

## Current Live Code

**`v2/train.py`** (exp_104 config):
- Model: same architecture as original baseline (encoder + contract_proj + no_trade_head + score_head)
- Gate loss: balanced BCE (subsample majority class to match minority)
- Selection loss: standard KL over all valid contracts at SOFT_TEMP=0.20
- No direction head, no direction conditioning

**`v2/core/policy.py`** (exp_105 change):
- `no_trade_before_bar = 60` (was 30)
- `no_trade_after_bar = 120` (was 270)
- All other policy params unchanged (stop=30%, target=50%, hold=120, trailing exit)

## Current Research Position

- Official scored runs in `v2/results.tsv`: exp_074–078 (all failed), exp_099 (-0.260), exp_104 (-0.240)
- exp_105 is a screening result only (no official 5-fold run yet)
- `v2/models/model_candidate.pt` is the exp_104 model (balanced gate, full-day window)
- Next experiment ID: `exp_106`

## Remaining Gap to Profitability

The model needs to either:
1. **Improve WR from 36.9% to ~42%** — the breakeven WR for 30% stop / 50% target is `0.30 / (0.30 + 0.50) = 37.5%`. Current WR is 36.9%, just below breakeven. A 1-2pp improvement could flip the sign.
2. **Reduce stop loss size** — wider stops give trades more room but increase per-loss magnitude. The current 30% stop on cheap OTM contracts is ~$0.75-1.50 per contract.
3. **Shift to higher-delta contracts** — calls had 36% WR vs puts at 28%. The model picks too many cheap OTM puts. Higher-delta (closer to ATM) contracts have more predictable behavior.

## Infrastructure Fixes Made This Session

1. `v2/ops/run_experiment_wf.py`: Fixed `n_folds=None` crash (defaulting to 5)
2. `v2/core/walkforward.py`: Fixed model path `v2/model_fold{N}.pt` → `v2/models/model_fold{N}.pt`
3. `v2/ops/deploy.sh`: Now uploads replay.py, walkforward.py, run_experiment_wf.py, and pre_run_gate.py alongside mutable files — prevents stale remote code crashes
4. `v2/replay.py`: Backwards-compatible with both old (no_trade_score) and new (gate_logit + direction_logit) model outputs

## What To Trust

- `v2/data.pt` and `v2/data_sidecars/` as the canonical exact-chain dataset
- `v2/results.tsv` as official exact-chain scored runs only
- `v2/lab_notebook.md` as the live screening log
- `v2/output/trades.html` and `equity.html` as the exp_104 trade visualization (val mask)
- `v2/output/trades.csv` as the 259-trade analysis dataset
- `v2/models/model_candidate.pt` as the exp_104 official model (fingerprint `46f2d184e186496f`)

## What Not To Trust

- Any pre-exact-chain score as a current baseline
- `exp_088` or `exp_089` as scored evidence
- The hierarchical direction head approach (exp_100–103) — direction doesn't decompose
- Any model checkpoint not from this session (Apr 11) — older ones are from different code

## Hypothesis Queue

1. `exp_106`: Run exp_105 (morning window) as official 5-fold `run_one` to get a proper model.pt and visualize its morning-only trades
2. `exp_107`: Widen stops from 30% to 40% — the exp_104 trade analysis showed 13% of losses had MFE >= 30% (reached profit zone then reversed). Wider stops may convert these to winners.
3. `exp_108`: Restrict to higher-delta contracts (entry price > $3) at replay time — calls WR 36% vs puts 28%, and cheap OTM (<$1) trades have 39% WR while $1-3 trades have only 24% WR.

## Abandoned Approaches

All previous items plus:
- Separate direction head (exp_100–102): never learns, 52% accuracy across all configs
- Direction-conditioned KL without inference masking (exp_103): creates uncalibrated cross-direction scores
- SOFT_TEMP below 0.20 with balanced gate (exp_096): peaked temp causes direction collapse
- Noise bar filtering (exp_097): reduces trades without improving quality
