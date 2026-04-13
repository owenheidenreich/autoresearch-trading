# Handoff: Current v2 State

Read this file, then [founder_intent.md](docs/founder_intent.md), then [program.md](program.md), then [current_state.md](docs/current_state.md).

## What Changed (2026-04-12 exp_144)

**exp_144 PROMOTED: cooldown_bars 5→3.** Policy sweep found the 5-bar cooldown was blocking profitable re-entries. Score 0.150→0.547 (+265%), DD 12.4%→8.2%, net P&L $2,678→$4,771.

### Phase 9: Policy Sweep & Cooldown Optimization (exp_141–144)

**exp_141 — SOFT_TEMP 0.10→0.05**: screening only. Best direction balance (47.4% puts) but WR collapsed. Peaked target punishes near-miss contracts (79% of bars have 3+ near-oracle contracts). **Killed.**

**exp_142 — direction_proj (replacing put_bias)**: improved 3/5 folds but fold 4 DD 23.1% (gate failure). **Reverted.**

**exp_143 — direction_proj (residual on put_bias)**: fold 0 best-ever score (0.325). Fold 4 DD 30.3%. **Reverted.**

**Key finding from direction experiments**: the model CAN learn direction balance, and it improves weak folds. But fold 4 profits from call-heavy trading — any mechanism adding puts degrades it. The scoring system rewards fold 4's exceptional result.

**Policy sweep** (`v2/analysis/policy_sweep.py`): systematic sweep of 5 params × 3 values on promote mask. Found `cooldown_bars=3` (+148% score) and `stop_pct=0.20` (+107%) as individual winners.

**exp_144 — cooldown_bars 5→3**: **PROMOTED.** Best result in project history.

| Metric | exp_139 | exp_140 | exp_144 |
|--------|---------|---------|---------|
| Score | -0.121 | 0.150 | **0.547** |
| PF | 1.142 | 1.201 | **1.356** |
| DD | 17.5% | 12.4% | **8.2%** |
| Sortino | 1.88 | 3.87 | **7.36** |
| Net P&L | +$1,334 | +$2,678 | **+$4,771** |

### Phase 8: Exit Policy Optimization (exp_140)

**Root cause**: exp_139 trade analysis showed 21 "whipsaw" trades — stop-loss exits where MFE exceeded 15% before reversal. These trades reached the profit zone but the trailing stop didn't engage until +30%, so they reversed to the -30% hard stop. Total whipsaw losses: -$4,260.

**exp_140 — Lower breakeven trigger (0.30 → 0.15)**: Wired the dead `breakeven_trigger_pct` policy field through to `simulate_trade()` via `_build_trailing_tiers()`. Training labels unchanged (labels.py uses original `TRAILING_TIERS`). **PROMOTED.**

| Metric | exp_139 | exp_140 | Change |
|--------|---------|---------|--------|
| Score | -0.121 | **0.150** | +0.271 |
| PF | 1.142 | **1.201** | +5.2% |
| DD | 17.5% | **12.4%** | -29% |
| Sortino | 1.88 | **3.87** | +106% |
| Net P&L | +$1,334 | **+$2,678** | +101% |
| Whipsaw losses | -$4,260 | **-$163** | -96% |

### Phase 7: Contract Feature Normalization (exp_134–139) — BREAKTHROUGH

**First profitable model: exp_139.** Contract feature normalization + greek sign alignment + learned put bias.

### Phase 7: Contract Feature Normalization (exp_134–139) — BREAKTHROUGH

**Root cause discovery**: The 15 per-contract features had 340,000x scale differences (strike ~6860 vs gamma ~0.008). The `contract_proj` Linear(15, 96) was completely dominated by strike — the model was blind to greeks, IV, spread, and volume. Additionally, delta/moneyness/distance flip sign for puts, confusing the shared embedding.

**exp_134/134b — Per-side ranking KL**: Proved put head CAN rank but best-ranked puts still lose. Feature-bound.
**exp_135 — Exact-oracle CE**: Impossible task (285 classes), pulled capacity toward unlearnable objective.
**exp_136 — Multiplicative interaction**: More expressiveness → more confident bad puts.
**exp_137 — Greek sign normalization only**: First positive score (+0.184), first dir_acc movement (0.55 vs stuck at 0.49), but DD 48.7%. Centering over-equalized.
**exp_138 — Sign norm without centering**: Put head dominated without normalization. Centering needed.
**exp_139 — Full normalization + sign fix + learned put bias**: **PROMOTED. First profitable model.**

### Phase 6 (earlier same day): Side-Specific Score Heads (exp_123–133)

## What Changed (2026-04-12)

Separate call/put score heads with per-bar mean centering broke the all-call collapse.

### Phase 5: Side-Specific Score Heads (exp_123–132)

**Motivation**: The single `score_head` ranks calls and puts on the same axis. Since contract features have zero predictive power for oracle selection, the model takes a global side-bias shortcut. The v1 archive showed separate call/put heads solved this.

**exp_123 — Side-split KL only**
- Split `score_head` into `call_score_head` and `put_score_head`, KL computed separately per side
- Result: 36C/0P, dir_acc stuck at 0.489 — separate KL removes ALL cross-side gradient

**exp_124 — Separate heads + cross-side margin loss (DIR_W=0.10)**
- Added hinge margin loss to push oracle side's best score higher
- Result: **165C/0P but best single-fold economics ever** (PF 0.825, DD 25.7%, -$1,660)
- The margin works per-bar but the global call_score_head offset dominates at inference

**exp_125 — Separate heads + per-bar mean centering + unified KL** ← BEST
- Center each side's scores to zero mean per bar before merging
- No margin loss; unified KL over all contracts provides cross-side comparison
- Screening: **163C/21P** (first puts!), PF 0.816, DD 29.3%, sortino -1.74
- **Official 5-fold: score=-0.200, 164C/16P, PF 0.873, DD 28.9%**
- First official run with puts in the separate-head family

**exp_126 — Z-score normalization**: CRASH, nan from near-zero std

**exp_127–129 — Margin loss sweep (DIR_W=0.01/0.03/0.10)**
- All improved direction balance but degraded economics monotonically
- 91C/94P at DIR_W=0.10 but DD 103%; 123C/78P at DIR_W=0.03 but DD 48%
- The put selections introduced by margin loss are systematically losers

**exp_130 — NOISE_MARGIN=0.03**: 136C/55P, DD 89.7% — higher filter removed useful bars

**exp_131 — SOFT_TEMP=0.07**: 167C/40P, DD 60.2% — lower temp degrades economics

**exp_132 — SEL_W=0.5**: 90C/92P, DD 80.9% — gate emphasis also forces bad puts

**Key finding**: Direction balance vs economics is a monotonic tradeoff. Every mechanism that forces more puts degrades PF/DD proportionally. The model genuinely doesn't know how to pick winning put contracts.

### Promote Trace Follow-Up (2026-04-12 Late)

Promote-trace comparison of `exp_125` vs `exp_106` clarified that the remaining bottleneck is not "no put signal"; it is conditional side calibration plus put strike quality.

- Replayed both artifacts on the promote mask and saved traces to `v2/artifacts/analysis/exp_106_promote_traces.csv` and `v2/artifacts/analysis/exp_125_promote_traces.csv`
- `exp_125` improved promote drawdown from `34.5%` to `28.9%`, but gate accuracy stayed flat at `21.2%` and selection accuracy slipped from `6.8%` to `6.1%`
- The `16` put trades in `exp_125` lost `-$1,635`, versus `-$1,258` from `164` call trades; on the promote slice, removing puts alone brings DD from `28.9%` to `19.7%`
- `8/16` selected puts were outright wrong-side bars where the oracle was a call; `5` of those missed call winners above `+30%`
- The other `8/16` puts were on oracle-put bars, but put strike calibration was poor: exact-match rate `0%`, median strike gap `25` points, and selected puts sat much closer to spot than the oracle (`+9.9` vs `+30.6` signed points from spot)
- Real put opportunity does exist in the data: the promote trace contains `1,248` profitable oracle-put bars vs `1,578` profitable oracle-call bars, with similar average oracle P&L (`45.2%` vs `44.7%`), though put label quality is weaker (`0.045` vs `0.065`)
- Timing still matters on the call side too: on the promote slice, removing bars `105-119` alone also drops DD to `19.8%`; removing both puts and late-window trades drops DD to `11.7%` and turns P&L positive

## What Changed (2026-04-11)

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

### Phase 4: Official Baseline Lock + Kronos-Inspired Block (exp_106–109)

**exp_106 — Morning-window baseline lock (official 5-fold run)**
- Promoted the exp_105 morning-only policy into the official evidence set with no model change
- 5-fold aggregate score: `-0.260`, WR `38.7%`, PF `0.890`, +day rate `50.0%`, trades `784`
- Mandatory promote trace exposed the remaining failure mode clearly:
  - gate accuracy `21.2%`
  - selection accuracy `6.8%`
  - average model P&L `-0.0179` vs oracle `0.3279`
  - delta gap `-0.3458`
  - promote replay direction collapse: `191C / 0P`
- Key conclusion: the morning window improved economics materially, but the underlying scorer still collapses to calls on the promote trace

**Revised research rules now active**
- The default live loop is still `v2/train.py` plus `v2/core/policy.py`
- The approved expanded surface for the current block is `v2/train.py`, `v2/replay.py`, `v2/core/data_integrity.py`, and `v2/ops/pre_run_gate.py`
- The operative unit is one hypothesis, not one file change
- Promotion remains score- and baseline-gated; traces can justify continuing a hypothesis family but not promoting it
- A separate raw/sidecar anomaly audit track is now part of the live operating system, with an explicit dataset-migration trigger

**exp_107 — Policy-window-aligned supervision**
- Applied supervision only on rows inside the live morning policy window
- Result: score `-0.200`, trades `160`, direction `25C / 135P`, WR `30.6%`, DD `68.9%`
- Outcome: reverted; this fixed the all-call collapse but made the economics materially worse

**exp_108 — Learned temporal embeddings**
- Threaded explicit `bar_of_day` / weekday IDs through training and replay, then added learned temporal embeddings
- Result: score `-0.300`, trades `162`, direction `162C / 0P`, WR `35.2%`, PF `0.771`
- Outcome: reverted; explicit temporal IDs did not help as a standalone hypothesis and reintroduced full call-side collapse

**exp_109 — Train-only flow-feature dropout**
- Zeroed the context flow slice `X[..., 39:47]` for 5% of training samples
- Result: score `-0.300`, trades `143`, direction `129C / 14P`, WR `30.8%`, PF `0.564`
- Outcome: reverted; this was the only Kronos-inspired change that nudged direction balance toward sanity, but it still failed the 15% minority-direction hard gate and worsened economics

**exp_110 — Cross-side calibration loss**
- Added a small training-only margin term so the oracle side's best score had to beat the opposite side on traded rows
- Result: score `-0.200`, trades `162`, direction `35C / 127P`, WR `34.0%`, PF `0.619`, DD `82.5%`
- Outcome: reverted; this broke the full call-collapse pattern but overcorrected into puts and still failed the economics and baseline gates

**Separate audit track**
- Raw-input audit flagged 4 anomaly dates:
  - `2024-05-30` in SPX for a 71-bar stagnant close run
  - `2024-08-05`, `2025-04-07`, and `2025-04-09` in VIX for structural breaks
- Sidecar audit flagged 59 dates, including 9 recurring short-session schema-break days:
  - `2022-11-25`, `2023-07-03`, `2023-11-24`, `2024-07-03`, `2024-11-29`, `2024-12-24`, `2025-07-03`, `2025-11-28`, `2025-12-24`
- Trace overlap with the worst promote-trace days was `0` for both the raw audit and the sidecar audit
- Key conclusion: the audit found real anomalies, but it did not fire the dataset-migration trigger; `v4_exact_chain` remains the live authority

## Canonical State

```text
Dataset path:        v2/data.pt
Per-day sidecars:    v2/data_sidecars/*.pt
Dataset version:     v4_exact_chain
Dataset fingerprint: 46f2d184e186496f
Unique days:         986
Features:            47
Trade window:        bar 60-105  (was 60-120, narrowed in exp_133)
ATM source:          dynamic_nearest_per_bar
```

## Current Live Code

**`v2/train.py`** (exp_139 architecture):
- Model: encoder + contract_proj + no_trade_head + **call_score_head + put_score_head** + **learned put_bias**
- Forward preprocessing:
  1. **Greek sign normalization**: negate delta(8), moneyness_pct(11), distance_points(12) for puts
  2. **Per-bar z-score**: normalize 11 continuous features across valid contracts per bar
  3. Zero out invalid contracts after normalization
- Forward scoring: per-bar mean centering + learned `put_bias` on put centered scores
- Gate loss: balanced BCE (subsample majority class to match minority)
- Selection loss: unified KL over all valid contracts at `SOFT_TEMP=0.10`
- Noise filter: skip bars where top label margin `< 0.01`
- Inactive auxiliary losses: SIDE_SEL_W=0.0, EXACT_W=0.0

**`v2/core/policy.py`** (morning window + tighter trailing + faster re-entry):
- `no_trade_before_bar = 60`
- `no_trade_after_bar = 105`
- `breakeven_trigger_pct = 0.15` (was 0.30, exp_140)
- `cooldown_bars = 3` (was 5, exp_144)
- All other policy params unchanged (stop=30%, target=50%, hold=120, trailing exit)

**`v2/core/simulator.py`** (trailing tiers):
- `TRAILING_TIERS` constant preserved for label computation
- `_build_trailing_tiers(breakeven_trigger_pct)` builds custom tiers for evaluation
- `simulate_trade()` accepts `breakeven_trigger_pct` parameter (None = use original tiers)

**Separate audit tooling**
- `v2/core/data_integrity.py` now supports:
  - raw minute-bar anomaly audit
  - full sidecar date audit
  - side-bias audit against executable snapshot labels and model outputs
  - optional replay-trace overlap reporting
- This is audit-only infrastructure. It does not mutate `v2/data.pt`.

## Current Research Position

- Official scored runs in `v2/results.tsv`: exp_074–078, exp_099, exp_104, exp_106, exp_122, exp_125, exp_133, exp_137, exp_139, exp_140, exp_142, exp_143, **exp_144**
- **exp_144 is the current best**: score=0.547, PF 1.356, DD 8.2%, Sortino 7.36, WR 39.1%, 158C/26P, +$4,771
- `v2/models/model.pt` and `v2/models/model_best.pt` are the exp_144 promoted artifact
- `v2/artifacts/exp_144/` is the current official artifact bundle
- **Current live code: exp_144 (exp_139 architecture + breakeven_trigger_pct 0.15 + cooldown_bars 3)**
- Next experiment ID: `exp_145`

## Key Findings

1. **340,000x feature scale mismatch was the root cause** (exp_139): strike (~6860) dominated contract_proj, making the model blind to greeks, IV, spread, and volume. Per-bar z-score normalization fixes it.
2. **Exit policy was the second bottleneck** (exp_140): the 30% breakeven trailing trigger was too generous. 21 trades reached 15-25% MFE then reversed to stop-loss. Lowering to 15% eliminated 96% of whipsaw losses.
3. **Greek sign alignment matters**: negating delta/moneyness/distance for puts before normalization aligns the embedding space.
4. **Learned put bias solves the centering interaction**: centering alone over-equalizes sides; the learned bias lets the model discount puts appropriately.

## Current Performance

| Metric | exp_139 | exp_140 | exp_144 |
|--------|---------|---------|---------|
| Score | -0.121 | 0.150 | **0.547** |
| PF | 1.142 | 1.201 | **1.356** |
| DD | 17.5% | 12.4% | **8.2%** |
| WR | 45.3% | 39.2% | 39.1% |
| Net P&L | +$1,334 | +$2,678 | **+$4,771** |
| Sortino | +1.88 | +3.87 | **+7.36** |
| Selection accuracy | 7.5% | 7.4% | **8.7%** |
| Gate accuracy | 20.9% | 21.2% | 21.7% |
| Trades | 161 (136C/25P) | 176 (151C/25P) | 184 (158C/26P) |
| Baselines beaten | All 4 | All 4 | All 4 |
| Gate failure | None | None | None |
| Final equity | $11,334 | $12,678 | **$14,771** |

## Remaining Improvement Opportunities

1. **Selection accuracy is still only 7.4%** — the oracle picks 1 of ~285 contracts, the model gets it right 7.4% of the time. Improving this is the most direct lever for higher PF.
2. **Gate accuracy 21.2%** — nearly 4 in 5 trade decisions are wrong. But trace analysis showed 67.5% of losses are contract-selection errors, not timing errors.
3. **3/5 folds still hit the -0.200 floor** — fold 0 improved from -0.200 to +0.076 with the tighter trailing stop, but folds 1-3 remain stuck. Fold variance is high.
4. **Put quality** — 25 puts is healthy and puts are actually more profitable per-trade (WR 48%, avg +8.54%) than calls (WR 37.7%, avg +1.59%). But only 14.2% of trades are puts.
5. **Known data pipeline issue**: theta formula in `v2/pipeline/compute_features.py` uses call formula for all contracts. Put theta has wrong sign on the interest-rate term (~4% error for 0DTE). Not fixed to avoid confounding.

## Infrastructure Fixes Made This Session

1. `v2/ops/run_experiment_wf.py`: Fixed `n_folds=None` crash (defaulting to 5)
2. `v2/core/walkforward.py`: Fixed model path `v2/model_fold{N}.pt` → `v2/models/model_fold{N}.pt`
3. `v2/ops/deploy.sh`: Now uploads replay.py, walkforward.py, run_experiment_wf.py, and pre_run_gate.py alongside mutable files — prevents stale remote code crashes
4. `v2/replay.py`: Backwards-compatible with both old (no_trade_score) and new (gate_logit + direction_logit) model outputs

## What To Trust

- `v2/data.pt` and `v2/data_sidecars/` as the canonical exact-chain dataset
- `v2/results.tsv` as official exact-chain scored runs only
- `v2/lab_notebook.md` as the live screening log
- `v2/program.md`, `v2/docs/current_state.md`, and `v2/docs/decision_log.md` as the live protocol/state documents
- `v2/output/trades.html` and `equity.html` as the exp_144 trade visualization (promoted model)
- `v2/output/trades.csv` as the 184-trade exp_144 analysis dataset
- `v2/models/model.pt` and `v2/models/model_best.pt` as the exp_144 promoted model
- `v2/artifacts/exp_144/` is the current official artifact bundle
- `v2/artifacts/replay_traces.csv` as the exp_144 promote trace
- `v2/analysis/policy_sweep.py` as the systematic policy parameter testing tool
- `v2/core/data_integrity.py` anomaly and side-bias reports as the separate audit track

## What Not To Trust

- Any pre-exact-chain score as a current baseline
- `exp_088` or `exp_089` as scored evidence
- `exp_120` as evidence; it is a code-only state with no trustworthy local result
- The hierarchical direction head approach (exp_100–103) — direction doesn't decompose
- Policy-window-only supervision as a standalone fix (exp_107)
- Explicit temporal embeddings as a standalone fix (exp_108)
- Flow-feature dropout at 5% as a standalone fix (exp_109)
- Pairwise ranking as the live baseline; `exp_121` was rejected
- Any unlogged code state as if it were an experiment result

## Hypothesis Queue

1. Improve selection accuracy (7.4% → higher) via training-side changes — this is the most direct lever for higher PF
2. Reduce fold variance — 3/5 folds at -0.200 means the model isn't consistently profitable
3. Fix theta formula for puts in data pipeline (`v2/pipeline/compute_features.py`) — requires sidecar rebuild
4. Explore further trailing tier optimization — intermediate tiers between 15% and 50%
5. Keep the separate audit track alive; do not rebuild dataset unless trigger fires

## Abandoned Approaches

All previous items plus:
- Separate direction head (exp_100–102): never learns, 52% accuracy across all configs
- Direction-conditioned KL without inference masking (exp_103): creates uncalibrated cross-direction scores
- SOFT_TEMP=0.05 with balanced gate (exp_096): too peaked, causes call collapse at the other extreme
- Noise bar filtering (exp_097): reduces trades without improving quality
- Policy-window-only supervision as a standalone fix (exp_107): changes direction mix but worsens economics
- Explicit temporal embeddings as a standalone fix (exp_108): full call collapse
- Flow-feature dropout at 5% as a standalone fix (exp_109): slight balance improvement, still non-promotable
- Cross-side calibration with the initial margin weight (exp_110): fixes the all-call collapse but overshoots into put bias
- **Auxiliary side-calibration losses (exp_111–113)**: max-BCE and logsumexp-BCE at weights 0.10/0.30/0.50 all failed; `dir_acc` stuck at random regardless of formulation or weight
- **Hierarchical side-aware KL target (exp_114)**: restructured selection target with P(side)*P(contract|side); still 0% puts
- **SOFT_TEMP=0.15 (exp_116)**: back to 0% puts; the direction awareness cliff is between 0.10 and 0.15
- **Pairwise ranking loss (exp_121)**: turned the side collapse into an 85%-put wipeout; not a solution
- **Side-split KL without centering (exp_123)**: removes all cross-side gradient, model defaults to initial bias
- **Cross-side margin loss at any weight (exp_124/127–129)**: improves direction balance but degrades economics monotonically; put selections are systematically losers
- **Z-score normalization (exp_126)**: nan from near-zero std at initialization
- **NOISE_MARGIN=0.03 with separate heads (exp_130)**: removes useful training bars
- **SOFT_TEMP=0.07 with separate heads (exp_131)**: degrades economics
- **SEL_W=0.5 with separate heads (exp_132)**: near-perfect balance but worst PF; gate emphasis can't overcome bad put quality
