# v2 Lab Notebook — Exact-Chain Reset Era

Pre-exact-chain history lives in `archive/v2_historical/logs/lab_notebook_pre_exact_chain.md`.
The full mixed-state notebook before this reset lives in `archive/v2_historical/logs/lab_notebook_pre_reset_mixed_state_2026-04-10.md`.

## Post-Reset Diagnostic (2026-04-10)

### Why The Score Dropped from 5.4 to -0.2

Five-part diagnostic run after 6 consecutive no-improve screenings (exp_090–095).

#### Finding 1: The KL Selection Target Is Near-Uniform

At `SOFT_TEMP=0.20`, the soft KL target has median max probability **0.23** — the model is being asked to match a nearly uniform distribution over ~16 contracts. Only 10.7% of bars have a clear winner (max prob > 0.50). The selection gradient is negligible.

At `SOFT_TEMP=0.05`, median max prob jumps to **0.51** and 52% of bars have a clear winner. The current temperature is 4x too high for this number of contracts.

#### Finding 2: Contract Features Alone Have Zero Predictive Power

A logistic regression on the 15 contract features (strike, IV, delta, gamma, moneyness, etc.) achieves **zero lift** over base rate for predicting which contract the oracle picks. Per-bar exact ranking accuracy: **8.2%** (random chance: 6.2%). Direction match: **53.1%** (random: 50%).

This means the `score_head` in `train.py` — which receives `[context_embedding, contract_embedding]` — must rely almost entirely on the context embedding to differentiate contracts. But the KL target gives it almost no gradient to learn how context should influence contract selection.

#### Finding 3: Gate Imbalance Is 4:1, Not 15:1

Train split: **80% trade / 20% no-trade** (4:1 ratio). This is a learnable imbalance — the gate failure is not from impossible class ratios but from the selection loss drowning the gate gradient. Balanced gate sampling (exp_095) confirmed the gate can learn when given equal representation.

#### Finding 4: 30.6% of Bars Have No Clear "Best" Contract

The top margin (best minus 2nd-best pnl) is **< 0.01** for 30.6% of bars. These are noise bars where any selection is equally good/bad. The model is being trained on noise one-third of the time.

#### Finding 5: The Oracle Trades 190 Times Per Day

Every labelable bar has `best_pnl > 0.04` (the gate threshold). The gate threshold never filters anything in the oracle — it's a rubber stamp. The profitable filtering the old model did (1.28 trades/day at threshold 0.5) has no equivalent in the new system.

#### Root Cause Summary

The v4 exact-chain rebuild changed the task from predicting two P&L scalars (learnable via Huber regression) to ranking ~16 contracts (unlearnable from contract features alone, with near-uniform KL targets). The old model's selectivity came from domain-encoded loss asymmetry and a meaningful gate threshold. The new system has neither.

The current architecture is not wrong — it's underpowered for the task. The selection signal is buried under temperature-smoothed noise, and the gate has no meaningful threshold to optimize against.

#### Implications for Next Experiment Block

1. **Lower `SOFT_TEMP` substantially** (0.05 or lower) to create peaked targets the model can learn from
2. **Filter noisy bars** where top margin < threshold from the selection loss
3. **Decompose the task**: direction first (call/put), then strike selection — the old binary structure was the right abstraction
4. **Raise or restructure the gate threshold** so the gate has a meaningful filtering role

## Reset Reconciliation (2026-04-10)

- Restored `v2/train.py` to the `exp_080`-style baseline: gate BCE plus soft KL selection only
- Archived non-official `results.tsv` rows so live `v2/results.tsv` now holds official exact-chain scored runs only
- Moved stale live helpers and plan docs out of `v2/`
- `exp_088` and `exp_089` remain preserved in git and archive notes, but they have no authoritative scored result and are not part of the live evidence base
- Next experiment ID: `exp_092`

## Established Facts

### Diagnostic 1: Oracle Replay

| Window | Score | WR | DD | C/P | Trades |
|--------|-------|-----|-----|-----|--------|
| Test (60d) | **6.000** | 100% | 0% | 267/268 | 535 |
| Train (846d) | **6.000** | 100% | 0% | 3746/3718 | 7464 |

Verdict: the evaluation harness is achievable.

### Diagnostic 2: Direction Signal

| Features | Train acc | Test acc | Baseline |
|----------|-----------|----------|----------|
| Current bar (47 features) | 61.1% | **60.9%** | 52.6% |
| Window mean/std (lb=30, 141 features) | 62.6% | 60.2% | 52.6% |
| Window mean/std (lb=60, 141 features) | 62.4% | 60.5% | 52.6% |
| Window mean/std (lb=90, 141 features) | 62.4% | 60.9% | 52.6% |

Verdict: the signal exists in the current bar, and longer lookback has not helped in the diagnostic.

Top predictive features:

1. `put_call_txn_ratio`
2. `trend_5min`
3. `gamma`
4. `rsi_7`

## Official Exact-Chain Runs

| Exp | Score | Folds | Status | Hypothesis |
|-----|-------|-------|--------|------------|
| 074 | 0.000 | [0,0,0,0,0] | revert | exact-chain baseline attempt — zero trades all folds |
| 075 | 0.000 | [0,0,0,0,0] | revert | decouple gate from selection — still zero trades |
| 076 | -0.040 | [0,0,0,-0.2,0] | revert | gate threshold disabled — one fold fired |
| 077 | -0.300 | [-0.3,-0.3,-0.3,-0.3,-0.3] | revert | selection-only variant still negative |
| 078 | -0.740 | [-1,-0.5,-1,-1,-0.2] | revert | stronger direct PnL regression worsened edge |
| 099 | -0.260 | [-0.3,-0.3,-0.2,-0.3,-0.2] | revert | first official exact-chain baseline after reset |
| 104 | -0.240 | [-0.2,-0.3,-0.3,-0.2,-0.2] | revert | balanced gate + standard KL official baseline |
| 106 | -0.260 | [-0.2,-0.3,-0.2,-0.3,-0.3] | revert | morning-window official baseline lock |

## Screening History (2026-04-10)

| Exp | Change | Direction Balance | WR | Trades | DD | Key Finding |
|-----|--------|-------------------|----|--------|----|-------------|
| 079 | soft KL selection (`temp=0.05`) | 16C/54P | 27.1% | — | 102.9% | first exact-chain screen to beat 2/4 baselines |
| 080 | `PNL_W=0`, `SOFT_TEMP=0.20` | 179C/244P | 36.2% | 423 | 104% | best screening baseline; removing direct PnL regression fixed direction collapse |
| 081 | `NO_TRADE_W=3.0` | 62C/143P | 32.2% | 205 | 103% | gate reweighting was unstable |
| 082 | `PNL_W=0.1` | 25C/197P | 28.8% | 222 | 102% | even weak direct PnL regression reimposed direction collapse |
| 083 | auxiliary `side_head` | 568C/7P | 31.3% | 575 | 101% | side head corrupted the shared context encoder |
| 084 | `GATE_W=3.0` | 86C/277P | 33.9% | 363 | 101% | stronger gate weight did not solve overtrading |
| 085 | `gate_threshold=0.0` | 228C/132P | 32.2% | 360 | 102% | threshold tuning filtered randomly rather than by quality |
| 086 | side-masked KL targets | 316C/84P | 30.5% | 400 | 103% | masking to one side lost the useful signal |
| 087 | `SOFT_TEMP=1.0` | 250C/106P | 31.7% | 356 | 100% | overly soft KL worsened direction |
| 090 | no-change reset baseline re-screen | 179C/244P | 36.2% | 423 | 104% | post-reset baseline reproduced the old failure mode; hard gate failed on excessive drawdown with score `-0.2` |
| 091 | focal gate BCE (`alpha=0.25`, `gamma=2.0`) | 405C/6P | 32.1% | 411 | 100% | focal gate worsened behavior into severe call-side collapse; hard gate failed on direction balance with score `-0.3` |
| 092 | `tanh` on contract score head output | 549C/1P | 28.4% | 550 | 101% | tanh saturation killed selection gradients; total direction collapse into all-calls; score `-0.3` |
| 093 | `LOOKBACK=1` current-bar-only | 357C/1P | 33.0% | 358 | 102% | removing temporal context caused call collapse; lookback provides useful direction signal despite diagnostic; score `-0.3` |
| 094 | `SOFT_TEMP=0.10` | 315C/66P | 32.5% | 381 | 101% | matched baseline score `-0.2`; fewer trades, lower DD, gate starting to learn late; but direction balance worse (83% calls) |
| 095 | balanced gate sampling | 156C/205P | 34.6% | 361 | 102% | **best screening result**: gate actually learning (trd_rate 0.52-0.71), best PF 0.652, best +day_rate 31.2%, balanced direction; still score `-0.2` |
| 096 | `SOFT_TEMP=0.05` + balanced gate | 458C/22P | 33.5% | 480 | 101% | peaked temp caused call collapse even with balanced gate; direction balance requires `SOFT_TEMP=0.20`; score `-0.3` |
| 097 | balanced gate + noise filter (margin<0.01) | 65C/161P | 33.6% | 226 | 100% | fewer trades but worse quality (PF 0.540); direction balanced but all trades still losers; entire $10k account wiped; score `-0.2` |
| 099 | official baseline (5-fold) | 122C/428P | 34.7% | 1564 | 100% | first official 5-fold run; aggregate score `-0.260`; model.pt obtained for trade visualization |
| 100 | 3-head hierarchical (gate+dir+strike) | 226C/115P | 34.3% | 341 | 101% | direction head overfitted (val loss 0.647→0.974); dir_acc stuck at 51%; score `-0.2` |
| 101 | 3-head detached direction | 144C/111P | 34.9% | 255 | 100% | detach fixed divergence; dir_acc still 52%; best balance since exp_095; PF 0.653; score `-0.2` |
| 102 | 3-head DIR_W=0.3 live gradient | 374C/94P | 35.9% | 468 | 103% | best WR/PF but extreme call bias (80%); direction head still can't learn; score `-0.2` |
| 103 | balanced gate + dir-conditioned KL | 42C/157P | 29.6% | 199 | 100% | **worst**: dir-conditioned training without inference mask created uncalibrated cross-direction scores; extreme put bias; score `-0.2` |
| 104 | balanced gate + standard KL (5-fold official) | 131C/153P | 29.9% | 1697 | 101% | official baseline with balanced gate; aggregate score `-0.240`; model.pt obtained for trade analysis |
| 105 | **morning window only (bars 60-120)** | **79C/81P** | **36.9%** | 160 | **39%** | **best result ever**: PF 0.789, +DayRate 42.6%, DD 38.7%, lost only $2,899 instead of $10k; near-perfect C/P balance; still fails DD gate (38.7% > 20%) |

## Current Live Baseline

- Balanced gate BCE plus soft KL selection only
- `SOFT_TEMP=0.20`
- Morning-only policy window in `v2/core/policy.py` (`bar 60` through `120`)
- No direct PnL regression
- No auxiliary side head
- No gate reweighting
- No score regularization
- No temporal embeddings or flow dropout active; `exp_107` through `exp_109` were reverted after screening

## Abandoned Or Parked Approaches

- direct PnL regression in the live loss stack
- shared auxiliary side heads
- gate BCE reweighting
- treating `exp_088` or `exp_089` as scored evidence
- tanh-bounded score head (gradient saturation kills selection)
- LOOKBACK=1 current-bar-only (temporal context needed for direction balance)

## Hypothesis Queue

1. `exp_110`: choose the next trace-targeted modeling hypothesis after the rejected Kronos standalone screens
2. Keep the raw/sidecar audit track active, but do not rebuild `v4_exact_chain` unless the explicit trigger fires
3. Continue deferring stop/contract-filter policy work until a training-side change improves the traced failure mode

## Active Experiment Kickoff

### `exp_090` — Baseline Re-Screen (2026-04-10)

- Type: screening run
- Code change: none
- Purpose: confirm that the reset baseline, live docs, gate, and remote runner still form one trustworthy starting point
- Expected outcome: a clean 1-fold reference result for the restored `exp_080`-style baseline
- Result:
  - score: `-0.200`
  - gate failure: `excessive_drawdown (104.4% > 20%)`
  - trades: `423`
  - direction balance: `179C / 244P`
  - win rate: `36.2%`
  - baseline comparison: beat `ATM` and `ATM-trailing`, failed to beat `random` and `simple-rules`
- Decision: do not run the official `exp_090`
- Takeaway: the reset baseline is now re-established as a trustworthy post-reset reference, and it still fails for the same reason as the earlier `exp_080` screening regime: overtrading with unacceptable drawdown
- Next step: move to `exp_091` with one targeted change to the gate objective

### `exp_091` — Focal-Style Gate Loss (2026-04-10)

- Type: screening run
- Code change: replace plain mean gate BCE with focal-weighted BCE using `GATE_FOCAL_ALPHA=0.25` and `GATE_FOCAL_GAMMA=2.0`
- Purpose: penalize easy majority trade examples less and emphasize the minority no-trade mistakes that appear to drive overtrading
- Constant parts:
  - same model architecture
  - same selection KL target
  - same replay policy
  - same `SOFT_TEMP=0.20`
- Result:
  - score: `-0.300`
  - gate failure: `direction_collapse (balance=0.01 < 0.15)`
  - trades: `411`
  - direction balance: `405C / 6P`
  - win rate: `32.1%`
  - baseline comparison: failed all four baselines
- Decision: revert the `exp_091` code change and do not run the official `exp_091`
- Takeaway: focal weighting destabilized the gate and collapsed the model into an almost all-call regime instead of reducing overtrading safely
- Next step: move to `exp_092` with a bounded score-head change rather than more gate-loss weighting

### `exp_092` — Tanh-Bounded Score Head (2026-04-10)

- Type: screening run
- Code change: `torch.tanh()` applied to `score_head` output, bounding contract scores to [-1, 1]
- Purpose: prevent unbounded score magnitudes from destabilizing the gate decision
- Constant parts:
  - same model architecture
  - same selection KL target
  - same replay policy
  - same `SOFT_TEMP=0.20`
- Result:
  - score: `-0.300`
  - gate failure: `direction_collapse (balance=0.00 < 0.15)`
  - trades: `550`
  - direction balance: `549C / 1P`
  - win rate: `28.4%`
  - baseline comparison: failed all four baselines
- Decision: revert
- Takeaway: tanh saturation killed selection gradients — the model couldn't differentiate contracts within the [-1, 1] range and collapsed into all-calls. Training metrics showed gate_acc and trade_rate completely flat across all 14 epochs (0.757, 0.937), confirming the gate never learned. Bounded score approaches that saturate are not viable.
- Next step: move to `exp_093` with `LOOKBACK=1` architecture test

### `exp_093` — LOOKBACK=1 Current-Bar-Only (2026-04-10)

- Type: screening run
- Code change: `LOOKBACK` default changed from 30 to 1
- Purpose: test whether the 30-bar transformer window adds noise, since diagnostic showed current-bar features alone achieve 60.9% test direction accuracy
- Constant parts:
  - same loss function
  - same selection KL target
  - same replay policy
  - same `SOFT_TEMP=0.20`
- Result:
  - score: `-0.300`
  - gate failure: `direction_collapse (balance=0.00 < 0.15)`
  - trades: `358`
  - direction balance: `357C / 1P`
  - win rate: `33.0%`
  - profit factor: `0.625` (better than baseline's ~0.40 range)
  - baseline comparison: failed all four baselines
- Decision: revert
- Takeaway: removing temporal context caused call-side collapse despite the diagnostic showing current-bar features alone have 60.9% direction accuracy. The transformer's temporal attention provides something the logistic regression diagnostic doesn't capture — likely ordering/momentum cues needed for balanced call/put selection. The LOOKBACK=30 window is load-bearing for direction balance.
- Next step: move to `exp_094` with `SOFT_TEMP=0.10` to tighten selection targets

### `exp_094` — SOFT_TEMP=0.10 (2026-04-10)

- Type: screening run
- Code change: `SOFT_TEMP` default from 0.20 to 0.10
- Purpose: tighter KL targets to create more differentiated contract scores and strengthen gate gradient signal
- Result:
  - score: `-0.200` (matched baseline)
  - gate failure: `excessive_drawdown (100.5% > 20%)`
  - trades: `381` (down from 423 baseline)
  - direction balance: `315C / 66P` (worse than baseline's 179C/244P)
  - win rate: `32.5%`
  - profit factor: `0.637` (best screening PF so far)
  - positive day rate: `26.7%`
  - beats: ATM, ATM-trailing
  - training note: trade_rate dropped from 0.937 to 0.870 by epoch 15 — gate was finally starting to learn the minority class but ran out of time
- Decision: revert (same score as baseline, worse direction balance)
- Takeaway: SOFT_TEMP=0.10 showed the gate can start learning with tighter selection, but 93.7% trade-class imbalance remains the fundamental bottleneck. Direction balance is highly sensitive to training dynamics rather than a smooth function of temperature.
- Next step: exp_095 — address the class imbalance directly with balanced gate sampling

### `exp_095` — Balanced Gate Sampling (2026-04-10)

- Type: screening run
- Code change: subsample trade-class rows to match no-trade count in gate BCE loss computation
- Purpose: fix the 93.7% / 6.3% class imbalance that causes the gate to always predict "trade"
- Result:
  - score: `-0.200` (same as baseline)
  - gate failure: `excessive_drawdown (102.1% > 20%)`
  - trades: `361` (down from 423 baseline)
  - direction balance: `156C / 205P` (balanced, similar to baseline)
  - win rate: `34.6%`
  - profit factor: `0.652` (best screening PF)
  - positive day rate: `31.2%` (best screening +day rate)
  - sortino: `-17.48` (best screening sortino)
  - beats: ATM, ATM-trailing
  - training dynamics: gate_acc dropped from stuck-0.757 to 0.50-0.63 range; trade_rate dropped from 0.937 to 0.52-0.71; gate showed instability (epoch 3: trd_rate=0.000, recovered by epoch 4)
- Decision: revert (same score, no improvement)
- Takeaway: **Balanced gate sampling is the most impactful change tested.** The gate is learning for the first time — every metric except raw score improved. The remaining gap is that even with a learning gate, the model is still losing money (PF=0.652, WR=34.6%). The drawdown (102.1%) is the lowest seen but still above the 20% hard gate. Two potential follow-ups: (1) combine balanced sampling with tighter selection (SOFT_TEMP=0.10), (2) combine with reduced capacity.

## Session Limit: 6 No-Improve Streak

Experiments exp_090 through exp_095 have all scored -0.200 or -0.300. No score improvement in 6 consecutive screens. The program's session limit "6 no-improve streak" has fired.

**Assessment**: The balanced gate sampling breakthrough is real but not yet sufficient. The gate is learning, direction balance is preserved, and secondary metrics improved substantially. However, the score function requires all hard gates to pass, and excessive drawdown (102.1%) still fails. Continuing without new strategic insight risks burning GPU time on marginal variants.

**Recommended next session approach**: Combine balanced gate sampling with one of: (1) SOFT_TEMP=0.10 for tighter selection, (2) reduced capacity D_MODEL=48, or (3) longer training time to let the gate converge further.

## Morning-Window Baseline Lock And Kronos Block (2026-04-11)

### `exp_106` — Official Morning-Window Baseline Lock

- Type: official 5-fold run
- Code change: none beyond the kept morning-only policy from `exp_105`
- Purpose: turn the best screening result into the official baseline artifact before starting the Kronos-inspired block
- Result:
  - score: `-0.260`
  - folds: `[-0.20, -0.30, -0.20, -0.30, -0.30]`
  - gate failure: `direction_collapse (balance=0.00 < 0.15)`
  - trades: `784`
  - win rate: `38.7%`
  - profit factor: `0.890`
  - positive day rate: `50.0%`
- Mandatory promote trace:
  - gate accuracy: `21.2%`
  - selection accuracy: `6.8%`
  - average model P&L: `-0.0179`
  - average oracle P&L: `0.3279`
  - average delta gap: `-0.3458`
  - noisy bars `< 0.01`: `42.5%`
  - promote replay direction balance: `191C / 0P`
- Decision: keep the artifact as the official baseline reference, but do not treat the behavior as promotable
- Takeaway: the morning window improved economics versus `exp_104`, but the scorer still collapsed entirely to calls on the promote trace

### Revised Protocol Notes (2026-04-11)

- The active research unit is now one hypothesis, not one file change
- The default mutable surface remains `v2/train.py` plus `v2/core/policy.py`
- The approved expanded surface for the current block is:
  - `v2/train.py`
  - `v2/replay.py`
  - `v2/core/data_integrity.py`
  - `v2/ops/pre_run_gate.py`
- Promotion remains score- and baseline-gated
- A separate audit track now exists for raw-minute and full-sidecar anomaly checks; dataset migration still requires an explicit trigger

### `exp_107` — Policy-Window-Aligned Supervision

- Type: screening run
- Code change: apply gate and selection supervision only on rows inside the live morning policy window
- Result:
  - score: `-0.200`
  - gate failure: `excessive_drawdown (68.9% > 20%)`
  - trades: `160`
  - direction balance: `25C / 135P`
  - win rate: `30.6%`
  - baseline comparison: beat `ATM` and `ATM-trailing`, failed to beat `random` and `simple-rules`
- Decision: revert
- Takeaway: this removed the all-call collapse, but the economics got materially worse; supervision-window alignment alone is not enough

### `exp_108` — Learned Temporal Embeddings

- Type: screening run
- Code change: thread explicit `bar_of_day` / weekday IDs through training and replay, then add learned temporal embeddings to the context encoder
- Result:
  - score: `-0.300`
  - trades: `162`
  - direction balance: `162C / 0P`
  - win rate: `35.2%`
  - profit factor: `0.771`
  - baseline comparison: failed all four baselines
- Decision: revert
- Takeaway: explicit temporal IDs did not help as a standalone hypothesis and reintroduced full call-side collapse

### `exp_109` — Flow-Feature Dropout

- Type: screening run
- Code change: train-only 5% sample-level masking of the context flow slice `X[..., 39:47]`
- Result:
  - score: `-0.300`
  - trades: `143`
  - direction balance: `129C / 14P`
  - minority-direction share: `0.11`
  - win rate: `30.8%`
  - profit factor: `0.564`
  - baseline comparison: failed all four baselines
- Decision: revert
- Takeaway: this was the only Kronos-inspired change that partially improved direction balance, but it still missed the `0.15` hard gate and the economics worsened

### `exp_110` — Cross-Side Calibration Loss

- Type: screening run
- Code change: add a small cross-side ranking term on traded rows so the oracle side's best score must beat the opposite side without adding a separate direction head or inference mask
- Result:
  - score: `-0.200`
  - gate failure: `excessive_drawdown (82.5% > 20%)`
  - trades: `162`
  - direction balance: `35C / 127P`
  - win rate: `34.0%`
  - profit factor: `0.619`
  - baseline comparison: beat `ATM` and `ATM-trailing`, failed to beat `random` and `simple-rules`
- Decision: revert
- Takeaway: the hypothesis successfully broke the full call-collapse failure mode, but it overcorrected into a strong put bias and still lost badly on economics. This supports the idea that cross-side calibration is the right failure surface, but the current margin term is too blunt as a standalone fix.

### Separate Audit Track

- Command:
  - `.venv/bin/python3 -m v2.core.data_integrity --data v2/data.pt --raw-audit --sidecar-audit --trace-path v2/artifacts/replay_traces.csv`
- Raw audit findings:
  - SPX flagged `2024-05-30` for a 71-bar stagnant close run
  - VIX flagged `2024-08-05`, `2025-04-07`, and `2025-04-09` for structural breaks
  - SPY flagged no dates
- Sidecar audit findings:
  - 59 flagged dates total
  - 9 recurring schema-break dates on holiday-adjacent sessions:
    - `2022-11-25`
    - `2023-07-03`
    - `2023-11-24`
    - `2024-07-03`
    - `2024-11-29`
    - `2024-12-24`
    - `2025-07-03`
    - `2025-11-28`
    - `2025-12-24`
- Trace overlap:
  - raw audit overlap with worst trace days: `0`
  - sidecar audit overlap with worst trace days: `0`
- Decision: keep `v4_exact_chain` frozen; the audit found real anomalies, but it did not fire the explicit dataset-migration trigger

### `exp_111` — Soft Side-Calibration BCE Loss (SIDE_W=0.10)

- Type: screening run
- Code change: added a BCE auxiliary loss on `max_call_score - max_put_score` vs oracle side label, at `SIDE_W=0.10`, applied only on clear-label traded rows with both sides present
- Purpose: break the complete call-side collapse (191C/0P) without the overcorrection seen in exp_110's margin approach
- Result:
  - score: `-0.300`
  - gate failure: `direction_collapse (balance=0.03 < 0.15)`
  - trades: `176`
  - direction balance: `171C / 5P`
  - win rate: `32.4%`
  - profit factor: `0.582`
  - drawdown: `46.9%`
  - baseline comparison: failed all four baselines
  - training dynamics: side_loss dropped from `0.674` to `0.645` over 19 epochs; dir_acc stuck at `~0.49`; gate showed typical instability (trd_rate oscillating 0.0–0.79)
- Decision: do not promote, but continue the family
- Takeaway: the BCE side-calibration mechanism works (5 puts vs 0 in baseline) but at `SIDE_W=0.10` the gradient is ~10x weaker than gate/sel losses and gets drowned out. The formulation is fundamentally different from exp_110's hard margin — higher weight may not overcorrect the same way. Next: `exp_112` with `SIDE_W=0.50`
