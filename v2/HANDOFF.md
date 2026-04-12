# Handoff: Current v2 State

Read this file, then [founder_intent.md](docs/founder_intent.md), then [program.md](program.md), then [current_state.md](docs/current_state.md).

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
Trade window:        bar 60-120  (was 30-270, changed in exp_105)
ATM source:          dynamic_nearest_per_bar
```

## Current Live Code

**`v2/train.py`** (restored `exp_119` working base):
- Model: same architecture as original baseline (encoder + contract_proj + no_trade_head + score_head)
- Gate loss: balanced BCE (subsample majority class to match minority)
- Selection loss: standard KL over all valid contracts at `SOFT_TEMP=0.10`
- Noise filter: skip bars where top label margin `< 0.01`
- No direction head, no direction conditioning
- No pairwise ranking loss active

**`v2/core/policy.py`** (kept clean morning window):
- `no_trade_before_bar = 60` (was 30)
- `no_trade_after_bar = 120` (was 270)
- All other policy params unchanged (stop=30%, target=50%, hold=120, trailing exit)

**Separate audit tooling**
- `v2/core/data_integrity.py` now supports:
  - raw minute-bar anomaly audit
  - full sidecar date audit
  - side-bias audit against executable snapshot labels and model outputs
  - optional replay-trace overlap reporting
- This is audit-only infrastructure. It does not mutate `v2/data.pt`.

## Current Research Position

- Official scored runs in `v2/results.tsv`: exp_074–078 (all failed), exp_099 (`-0.260`), exp_104 (`-0.240`), exp_106 (`-0.260`)
- `exp_107` through `exp_119` screened and reverted or parked; `exp_119` is the strongest surviving family
- `exp_120` exists as a code commit (`NOISE_MARGIN=0.03`) but has no authoritative local result and does not count as live evidence
- `exp_121` rejected the pairwise ranking branch after a clean-policy wipeout (`17C / 94P`, `100.2%` drawdown)
- `v2/models/model_candidate.pt` is the `exp_106` morning-window official artifact
- The revised side-collapse protocol is active (`program.md`, `decision_log.md`, `current_state.md`)
- **Current live code: restored `exp_119`**
- Next experiment ID: `exp_122`

## Key Finding: Side Collapse Survives Loss-Family Changes

Two important facts now coexist:

1. `SOFT_TEMP=0.10` plus noise filtering (`exp_119`) is the best side-aware economics the project has produced so far: `119C / 34P`, PF `0.813`, WR `37.3%`, DD `55.2%`.
2. Replacing KL with pairwise ranking (`exp_121`) did **not** solve the collapse. It produced `17C / 94P` and a total wipe, so the problem is not “KL alone creates put bias.”

The side-bias audit is now the authority for this question. It shows that the executable labels and KL target mass are roughly side-neutral to slightly call-favored, while the model still collapses to one side. The failure mode is unstable shortcut learning, not a simple raw-label majority.

## Remaining Gap to Profitability

1. **The best economics still fail the drawdown gate** — `exp_119` improved DD to `55.2%`, but that is still far from the `20%` limit.
2. **Side collapse remains unresolved** — calls collapse under some KL settings, puts collapse under ranking, and neither behavior is economically acceptable.
3. **The repo had drifted out of sync** — code history advanced to `exp_121` while the notebook/handoff still described earlier states. That mixed state is now explicitly resolved.

The next hypothesis must start from the restored `exp_119` base and target conditional side calibration directly, not broad architecture churn.

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
- `v2/output/trades.html` and `equity.html` as the earlier exp_104 trade visualization (useful for the morning-window discovery)
- `v2/output/trades.csv` as the 259-trade analysis dataset
- `v2/models/model_candidate.pt` as the `exp_106` official model (fingerprint `46f2d184e186496f`)
- `v2/artifacts/exp_106/` as the current official artifact bundle
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

1. `exp_122`: run the official 5-fold rebaseline of the restored `exp_119` family under the side-diagnostic scorer
2. Use the side-bias audit plus promote traces from that run to choose the next conditional side-calibration hypothesis
3. Keep the separate audit track alive, but do not rebuild or relabel the dataset unless the explicit migration trigger fires

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
