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

## Current Live Baseline

- Gate BCE plus soft KL selection only
- `SOFT_TEMP=0.20`
- No direct PnL regression
- No auxiliary side head
- No gate reweighting
- No score regularization

## Abandoned Or Parked Approaches

- direct PnL regression in the live loss stack
- shared auxiliary side heads
- gate BCE reweighting
- treating `exp_088` or `exp_089` as scored evidence
- tanh-bounded score head (gradient saturation kills selection)
- LOOKBACK=1 current-bar-only (temporal context needed for direction balance)

## Hypothesis Queue

1. `exp_095`: balanced gate sampling — subsample trade-class rows to match no-trade count in gate loss
2. `exp_096`: reduced model capacity (`D_MODEL=48`) to prevent overfitting majority class
3. Defer detached two-stage side models until simpler replay-compatible changes are exhausted

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
