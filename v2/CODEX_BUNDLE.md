# Codex Context Bundle: SPX 0DTE Training Problem

## 1. Executive Summary

- **Project**: Supervised learning system for SPX 0DTE (same-day expiry) options trading. Picks one contract per bar (1-minute intervals) from a chain of up to 285 available options.
- **Model**: Transformer encoder (d=96, depth=3) processing 30-bar lookback of 52 context features. Produces: contract ranking scores (per-contract), opportunity gate logit (should I trade?), side prediction (call/put), aggression bucket (ATM/near/far OTM).
- **Training**: Trains from scratch each experiment on H100 GPU. ~24 epochs, 5-fold walk-forward validation with 60-day test windows. ~330K train bars, ~15K val, ~23K test.
- **Current best**: exp_165 — PF 0.854, DD 53.5%, 482 trades, 49.6% WR. Uses gate_threshold=0.0 + SOFT_TEMP=0.08.
- **Hard gate**: DD must be ≤25% to pass. No experiment has passed this gate. Best is 53.5%.
- **Core failure**: The model overtrades (~8-13 trades/day), picks the wrong side (call/put) on ~47% of trades, and the opportunity gate learned to fire on chain density (r=0.77 with n_valid_contracts) rather than actual opportunity quality.
- **Dead ends already tested**: 20+ screening runs trying gate weights, thresholds, opportunity labels (strict/consensus/high-threshold), side loss, quality weighting, exact oracle CE, regularization, competence head (rank<=5 target). All failed to break through the DD gate.
- **Competence head result**: A head trained to predict "will the ranker land in the top 5 contracts?" achieved comp_acc 64-67% but produced worse PF (0.631) than baseline. A simple logistic regression on context features outperforms it (AUC 0.697 vs learned head Spearman ρ ≈ 0).
- **Current best hypothesis**: The ranker has an 82% call bias across all regimes. Side accuracy is ~53% (near random). No gate can fix a ranker that picks the wrong side on half its trades. The side problem is likely the highest-leverage intervention point.
- **What Codex should focus on**: Where exactly does the call bias enter (data, architecture, loss, inference)? Is the side problem fixable within the current architecture, or does the model need to predict side first and rank within side?

## 2. Training-System Scope

### Tier 1: Must read first

| File | Purpose | Why Codex needs it |
|------|---------|-------------------|
| `v2/train.py` | Model architecture (`TradingModel`), all heads, `compute_loss()`, training loop, checkpoint logic | The entire training problem lives here. Architecture, loss weighting, label computation, competence head code. |
| `v2/core/policy.py` | `DecisionPolicy` dataclass — stops, targets, gate threshold, trade window, overlays | Controls what constitutes a valid trade. gate_threshold=0.0 is the current best setting. |
| `v2/replay.py` | Replay/evaluation loop — runs trained model on test data, simulates trades, computes metrics | How model outputs become trading decisions. The `model_to_intent()` function is the inference decision tree. |
| `v2/core/chain_data.py` | Contract feature schema (22 fields), `padded_snapshot()`, sidecar loading | Defines the 22 contract features the model sees. Greek sign flip for puts happens in train.py forward(), not here. |

### Tier 2: Important context

| File | Purpose | Why Codex needs it |
|------|---------|-------------------|
| `v2/core/simulator.py` | `simulate_trade()` — stop/TP/trailing exit logic, spread cost model | How oracle labels are generated. Affects what "profitable" means. |
| `v2/pipeline/build_v2_dataset.py` | Builds `data.pt` + sidecar `.pt` files, oracle label computation | How labels are computed: simulates every contract under fixed policy, stores net_pnl_pct as label. |
| `v2/pipeline/compute_features.py` | 52 context features + Black-Scholes greeks + charm | Feature engineering. Important for understanding what context the model sees. |
| `v2/core/metrics.py` | Score formula, hard gates (DD ≤25%, ≥30 trades, ≥15 days) | Defines what "passing" means. |
| `v2/core/config.py` | `RuntimeConfig` — shared constants (52 features, 22 contract features, 285 max contracts, etc.) | Dimensions and schema version. |

### Tier 3: Diagnostic scripts (read if investigating specific findings)

| File | Purpose | Key finding |
|------|---------|-------------|
| `v2/analysis/toxicity_investigation.py` | 5-question diagnosis of gate-ranker anti-alignment | opportunity_logit correlates r=0.77 with density; ranker worst in dense/volatile/early bars |
| `v2/analysis/rankability_diagnostic.py` | Tests whether model competence is predictable from context | LR proxy AUC 0.697 for rank<=5; top-5 rate varies 20-52% by density |
| `v2/analysis/competence_score_analysis.py` | Deep analysis of exp_167b competence score vs LR proxy vs baseline | Competence head Spearman ρ ≈ 0; LR proxy dominates; 82% call bias across all deciles |
| `v2/analysis/overlay_diagnostic.py` | Tests risk overlays (max trades, consecutive stops, gate tightening) | MaxTrades=4 achieves DD 23.7% but PF 0.929 — mechanical, not signal-driven |
| `v2/analysis/bucket_ordering_test.py` | Tests whether confidence score ordering is correct | Score is mis-ordered: top 10% chosen PnL worse than 25-50%. Focal loss cannot fix. |

## 3. Current Diagnosis

### Gate-ranker anti-alignment (established)
- `opportunity_logit` correlates r=0.77 with `n_valid_contracts` (chain density)
- Even after controlling for bar_of_day: partial r=0.76 — it's a density proxy, not opportunity
- The ranker performs WORST in dense/volatile/early bars — exactly where the gate fires hardest
- Oracle opportunity quality is flat (~0.33 avg best PnL) across all confidence buckets
- Source: `v2/analysis/toxicity_investigation.py`

### Competence head failed (established)
- Trained opportunity head on rank<=5 target (frozen teacher picks)
- Head learned (comp_acc 64-67%) but pushed opp_mean to -0.6, over-suppressing
- Competence score has Spearman ρ ≈ 0 with chosen PnL — no ordering value
- Simple LR proxy on context features strictly dominates (AUC 0.697, PF 1.586 at p95 threshold vs head's 1.071)
- The head learned "late, calm, sparse = good" which partially captures rank quality but ignores side accuracy
- Source: `v2/analysis/competence_score_analysis.py`

### Call bias is the dominant problem (established)
- Model picks calls 82% of the time across ALL competence-score deciles
- Oracle side is 52% calls — roughly balanced
- Side accuracy is 52.7% overall — essentially random
- False positives (high confidence, bad trade): 28.7% side accuracy, late/sparse bars
- False negatives (low confidence, good trade): 86.7% side accuracy, early/dense bars
- The bars the model rejects are actually where it gets side RIGHT
- Source: `v2/analysis/competence_score_analysis.py` Q4

### Architecture observations
- Call and put scoring use separate heads (`call_score_head`, `put_score_head`) with a learned `put_bias` parameter
- Greek sign flip (delta, moneyness, distance, charm) applied to puts in forward() before scoring
- Per-bar mean centering: call scores and put scores are mean-centered separately, then put_bias is added
- The side_head exists but is trained with SIDE_W=0.0 (disabled) in current best config
- When SIDE_W > 0, exp_159 showed improved WR but overall regression

### What's NOT the problem
- Contract features and labels are correctly computed (pipeline audit passed)
- The ranker (selection loss) trains normally — sel_loss decreases steadily
- The model CAN identify good contracts in sparse/late bars (rank<=5 at 52% rate in sparse chains)
- Oracle labels are policy-consistent (simulated under fixed DecisionPolicy)

## 4. Experiments Already Run

| Exp | Change | PF | DD | Trades | Status | Lesson |
|-----|--------|-----|-----|--------|--------|--------|
| 153-154 | Full-day baseline (bar 30-270) | 0.745 | 108% | 673 | Baseline | 12.9 TPD overtrading, no selectivity |
| 155 | OPP_W=1.5, GATE_W=2.0 | 0.761 | 100% | 694 | Dead end | Opp loss unlearnable (stuck 0.649) |
| 156 | SOFT_TEMP=0.08 | 0.795 | 96% | 702 | Dead end | Best temp, +0.05 PF, no selectivity |
| 159 | SIDE_W=0.5 | 0.778 | 100% | 695 | Dead end | Side head improves WR but overall regression |
| 160-162 | Opportunity label redesign (consensus, no-gate) | 0.58-0.70 | 100% | 442-573 | Dead end | All label variants regressed |
| 164c | gate_threshold=0.0 | 0.774 | 76% | 511 | **Relevant** | Best single lever: -24% trades, -30pp DD |
| **165** | gate_threshold=0.0 + SOFT_TEMP=0.08 | **0.854** | **53.5%** | 482 | **Best config** | Supervised ceiling. Still fails 25% DD gate. |
| 165b | exp_165 + OPP_W=2.0 | 0.832 | 59% | 520 | Dead end | OPP_W dominates, weakens ranking |
| 165d | exp_165 + EXACT_W=0.5 | 0.869 | 58% | 403 | Interesting | Best PF but worse DD than 165 |
| 166 | gate_threshold 0.0→0.05 | 0.814 | 56% | 439 | Dead end | Threshold fine-tuning regresses |
| 166b | cooldown 3→6 | 0.666 | 100% | 432 | Dead end | Prevents recovery trades |
| 167 | Competence head (composite label) | 0.750 | 92.7% | 546 | Dead end | Head didn't learn (comp_acc stuck ~45%) |
| **167b** | Competence head (rank<=5 label) | 0.631 | 89% | 458 | Dead end | Head learned (65%) but over-suppressed; LR proxy better |

### Diagnostic analyses (not experiments, no GPU)

| Analysis | Key finding |
|----------|-------------|
| Overlay diagnostic | MaxTrades=4: PF 0.929, DD 23.7% — passes DD gate mechanically |
| Toxicity investigation | Gate = density proxy (r=0.77). Ranker worst where gate fires hardest. |
| Bucket ordering test | Confidence score mis-ordered at extremes. Focal loss won't fix. |
| Rankability diagnostic | rank<=5 predictable (LR AUC 0.697). Signal holds across chain sizes. |
| Competence score analysis | Learned head ρ≈0; LR proxy dominates; 82% call bias is the real bottleneck |

## 5. Files to Include Verbatim

### Tier 1 — Must read first
1. **`v2/train.py`** — Full model architecture, loss computation, training loop, competence head code
2. **`v2/core/policy.py`** — DecisionPolicy with all trading parameters
3. **`v2/replay.py`** — `model_to_intent()` inference decision tree, replay loop, metric computation

### Tier 2 — Important after Tier 1
4. **`v2/core/chain_data.py`** — CONTRACT_FEATURE_FIELDS (22 features), padded_snapshot
5. **`v2/core/simulator.py`** — simulate_trade(), stop/TP/trailing exit, spread cost
6. **`v2/pipeline/build_v2_dataset.py`** — Oracle label computation, sidecar generation
7. **`v2/pipeline/compute_features.py`** — 52 context features, greeks computation

### Tier 3 — Only if investigating specific findings
8. **`v2/analysis/competence_score_analysis.py`** — Latest diagnostic (three-way score comparison)
9. **`v2/analysis/toxicity_investigation.py`** — Gate-ranker anti-alignment proof
10. **`v2/analysis/rankability_diagnostic.py`** — Rankability feasibility study
11. **`v2/core/metrics.py`** — Score formula and hard gates
12. **`v2/core/config.py`** — RuntimeConfig dimensions

## 6. Minimal Factual Context

### Data regime
- 986 trading days of SPX 0DTE options
- Trade window: bar 30-270 (out of 390 bars/day, 1 bar = 1 minute)
- 52 context features (normalized), 22 contract features per option
- Up to 285 contracts per bar (padded)
- Lookback: 30 bars

### Labels
- **Pre-computed at dataset build time**: Oracle labels = net_pnl_pct from simulating each contract under fixed DecisionPolicy (35% stop, 50% TP, trailing tiers, 120-bar max hold)
- **Pre-computed**: `label_trade` (is there a profitable contract?), `label_quality` (fraction of profitable contracts), `bar_best_contract_idx`
- **Computed during training**: Competence label (rank<=5 of teacher/model pick) — new, experimental
- **label_trade with OPP_LABEL="strict"**: requires raw_return_10bar > 12%, mae > -8%, bars_to_breakeven < 5, mfe_5bar > 3%

### How replay uses model outputs
1. `opportunity_logit > gate_threshold` (0.0) → pass/reject
2. If passed: `argmax(contract_scores)` among valid, quality-filtered contracts → chosen contract
3. Simulate trade with DecisionPolicy stops/targets
4. Compute PF, DD, WR, Sortino across all test days

### Metrics that matter most
- **PF** (profit factor) — gross wins / gross losses. Must be > 1.0 for profitability.
- **DD** (max account drawdown) — hard gate at 25%. Current best 53.5%.
- **Trades/day** — 4-8 is healthy. 12+ is overtrading.
- **Side balance** — should be ~50/50 calls/puts. Currently 82/18.
- **Chosen rank** — model's pick rank among valid contracts. Lower is better.

### Strong conclusions Codex should not re-derive
- The opportunity head is a density proxy, not an opportunity detector (r=0.77)
- Confidence score ordering is inverted at extremes (top 10% has worse PnL than 25-50%)
- rank<=5 is predictable from context (LR AUC 0.697) but the learned head failed to capture this
- Trivial regime filters (skip early + sparse) capture most of what the competence head learned
- MaxTrades=4 overlay mechanically passes the DD gate but isn't signal-driven
- The ranker's 82% call bias persists across ALL confidence/competence regimes

## 7. Current Open Questions for Codex

1. **Where exactly does the 82% call bias enter?** Is it in the data (more profitable calls in training?), the architecture (call_score_head vs put_score_head asymmetry?), the loss (selection KL treating calls and puts jointly?), or inference (Greek sign flip for puts in forward())?

2. **Is the put_bias parameter and mean-centering scheme causing the asymmetry?** The model mean-centers call scores and put scores separately, then adds a learned put_bias. Could this create a systematic call preference?

3. **Should the model predict side first, then rank within the predicted side?** The side_head exists but is currently disabled (SIDE_W=0.0). When enabled (exp_159, SIDE_W=0.5), it improved WR but regressed PF. Why? Is there an architectural coupling that makes joint training harmful?

4. **Is the per-side scoring architecture (separate call_score_head and put_score_head) introducing a learned bias?** Would a single unified scorer with side as an input feature work better?

5. **What is the interaction between the Greek sign flip (delta, moneyness for puts) and the per-side heads?** Is the normalization creating asymmetric input distributions that the heads learn differently?

6. **Is the selection loss (KL divergence on softmax of PnL / SOFT_TEMP) creating a side-blind ranking?** The KL target treats all valid contracts equally regardless of side. Could side-aware selection loss help?

7. **What is the smallest high-confidence code change worth testing next?** Given that 17+ screening runs on gate/threshold/label variants all failed, and the call bias appears to be the dominant bottleneck, what single architectural change most directly addresses the side problem?

8. **Is there a fundamental data imbalance?** Are calls genuinely more profitable in the training data, causing the model to rationally prefer them? Or is the model failing to learn put-side patterns?

## 8. What Codex Should NOT Do

- **Do not propose more gate-only experiments.** Gate tuning, competence heads, threshold sweeps, and overlay caps have been exhaustively tested. The gate is not the bottleneck.
- **Do not suggest focal loss on the opportunity head.** The score is mis-ordered, not mis-scaled. Focal loss was explicitly ruled out after Phase 0b testing.
- **Do not optimize trivial hyperparameters** (learning rate, batch size, dropout) as a first step. These were already swept.
- **Do not assume more data fixes it.** 986 days is substantial. The problem is architectural/loss, not data volume.
- **Do not propose RL immediately** unless the supervised bottleneck is clearly at the architecture level and not fixable with better loss design.
- **Do not ignore the replay/inference coupling.** The model's outputs are used in a specific decision tree (`model_to_intent`). Changes to training must be evaluated through this lens.
- **Do not focus on documentation, UI, or deployment concerns.** This is a training problem.
- **Do not re-derive known findings.** The gate-density correlation, score mis-ordering, and competence head failure are established. Build on them, don't re-prove them.

## 9. Codex Prompt

```
You are analyzing a supervised learning system for SPX 0DTE options trading.
The system picks one option contract per bar from a chain of up to 285 options,
using a Transformer encoder with multiple prediction heads.

I'm attaching the key training files and a context bundle (CODEX_BUNDLE.md).

The model's current best configuration achieves PF 0.854, DD 53.5% (needs ≤25%),
with an 82% call bias (oracle is 52% calls). The opportunity gate learned to
proxy chain density rather than actual trading opportunity. A competence head
trained on rank<=5 failed — a simple logistic regression outperforms it.

20+ experiments on gate tuning, label redesign, and competence heads have failed
to break through. The current best diagnosis is that the ranker's ~82% call bias
and ~53% side accuracy (near random) are the dominant bottleneck, not the gate.

Please:
1. Read the attached files, starting with train.py, policy.py, and replay.py
2. Trace the exact path where call/put scoring diverges in the architecture
3. Identify whether the call bias originates in data, architecture, loss, or inference
4. Assess whether the per-side scoring heads (call_score_head vs put_score_head)
   with mean-centering and put_bias are architecturally sound or creating systematic bias
5. Propose the 1-3 smallest, highest-confidence code changes to test next
6. For each proposal, explain what it would prove and what failure would mean
7. Distinguish root causes from symptoms — the gate problems are symptoms of
   the side/ranking problem, not independent issues

Do not propose:
- More gate-only experiments (exhaustively tested, not the bottleneck)
- Focal loss (score is mis-ordered, not mis-scaled)
- Trivial hyperparameter tuning
- RL as a first step
- Generic ML advice

Focus on the training architecture, loss design, and the specific call/put
scoring asymmetry. The goal is to identify why the model defaults to calls
and what the minimum intervention is to fix it.
```
