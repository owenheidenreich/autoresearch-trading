---
date: 2026-04-26
parent: phase2b_forward_walk_2026_04_26.md
status: PHASE 3 COMPLETE — 8 pathologies audited against 5 symptoms; 4 SYMPTOM-EXPLAINED, 1 REFUTED, 3 PARTIAL; ranking-loss dominance + zero side-balance by default + flat anchor compression jointly account for the calibration pathologies; L3 oracle/policy training coupling explains FW result drift
---

# Phase 3: L2 architectural review

## Method

Read `v3/layer2/{unified_policy.py, train_unified_policy.py, action_surface_dataset.py}` end-to-end. Tagged each of 8 architectural pathology hypotheses against 5 GPT-5.5 + Phase 2b documented symptoms. Cited line numbers for every finding.

## Symptoms in scope

| ID | Symptom | Source |
|---|---|---|
| A | `pred_win_prob` Spearman with hl is -0.025 (essentially flat — no ordering signal) | GPT-5.5 |
| B | `pred_clean_entry_prob` is inverted for puts (low values → PF 2.80; high values → PF 0.90) | GPT-5.5 |
| C | `decision_margin` is anti-calibrated for puts (high margin → bad PF — exploited by `avoid_put_high_decision_margin`) | GPT-5.5 + Phase 1 |
| D | Calls with `orc_triggered=True` are entered with `decision_margin ~0.60` despite adverse cell preference | GPT-5.5 |
| E | L3 oracle's prediction quality drops on alternate (row, action) pairs that L2 baseline would not have picked top-1 | Phase 2b |

## Architectural fact map (key citations)

- **One shared 160-dim bottleneck (`state_h`)** feeds all 6 heads ([unified_policy.py:165-167](../layer2/unified_policy.py#L165-L167), 170-179). Both flat (`flat_trunk(state_h)`) and contracts (`contract_action_trunk([state_h, token_h])`) share this representation.
- **Loss composition** ([unified_policy.py:842-863](../layer2/unified_policy.py#L842-L863)):
  - Ranking total weight ≈ **2.5** (`w_ranking * rank_loss + 0.5 * w_ranking * cross_rank_loss + w_ranking * flat_rank_loss` with `w_ranking=1.0`)
  - Calibration total weight = **1.45** (`w_dollar 0.25 + w_return 0.25 + w_win 0.25 + w_clean 0.35 + w_stopout 0.35`)
  - Regression weight = **0.5** (`w_regression`)
  - `w_side_contrastive = 0.0` by default ([train_unified_policy.py:67](../layer2/train_unified_policy.py#L67))
  - `side_balance_weight = 0.0` by default ([train_unified_policy.py:89](../layer2/train_unified_policy.py#L89))
- **Flat anchor pinning** ([train_unified_policy.py:200, 663](../layer2/train_unified_policy.py#L200)):
  - `utility_raw[:, 0] = 0.0` always
  - `reg_weight_train[:, 0] = 1.5` (1.5× weight on flat regression)
  - `utility_arcsinh = arcsinh(utility_raw / 100.0)` (mild compression beyond ~$300 PnL)
- **`decision_margin` formula** ([train_unified_policy.py:339-341](../layer2/train_unified_policy.py#L339-L341)):
  ```
  decision_margin = best_nonflat_score - flat_score   # both in arcsinh($/100) units
  ```
- **Action selection** ([train_unified_policy.py:295-297](../layer2/train_unified_policy.py#L295-L297)): pure `numpy.argmax` over utility logits — calibration heads don't enter selection. Tie-break is first-occurrence (lowest token index).
- **Call/put slot layout** ([action_surface_dataset.py:588-603](../layer2/action_surface_dataset.py#L588-L603)): calls in tokens 0..K-1, puts in tokens K..2K-1. Selection criteria identical per side; no greek normalization or side-asymmetric stops.
- **Targets are symmetric** between calls and puts; the asymmetry comes from data distribution (~2.83:1 call/put imbalance), not the label formulas.
- **L3 oracle training is coupled to L2 top-1 picks** ([train_unified_policy.py:1031-1062](../layer2/train_unified_policy.py#L1031-L1062)): only the trades that pass `_select_daily_trades` (decision_margin gate, win/stopout gates, daily-best argmax) become the L3 oracle's training distribution. (row, action) pairs the L2 model would not pick are absent from the oracle's training data.

## Pathology verdicts

### 1. Shared-state bottleneck

**Status:** VERIFIED → SYMPTOM-EXPLAINED for A, E. PARTIAL for B.

A single 160-dim `state_h` produced by `state_trunk(concat(scalar_h, seq_h))` feeds **both** the flat path and every contract action's representation. All 6 prediction heads (utility, dollar, return, win, clean, stopout) operate on this shared bottleneck, with only a thin per-action token-encoder branch for contracts ([unified_policy.py:119-143](../layer2/unified_policy.py#L119-L143)).

Why this matters for symptom A (flat win_prob): the shared bottleneck must serve the dominant ranking objective AND every calibration head with a single 160-dim representation. Under loss-weight imbalance (see #2), the bottleneck learns features that satisfy ranking; the win head, downstream of that bottleneck with only a thin linear projection, has limited capacity to "correct" the representation toward calibrated win probabilities.

Why this matters for symptom E: gradients from the L3 oracle's downstream training don't flow back through the shared bottleneck (L3 is trained separately, on L2's top-1 outputs). The bottleneck is shaped only by L2's loss mix and its imbalance.

### 2. Ranking-loss dominance

**Status:** VERIFIED → SYMPTOM-EXPLAINED for A, B, C, E.

Total ranking-loss weight (2.5×) is **5× the regression weight** (0.5×) and **1.7× the total calibration weight** (1.45×). The ranking loss only enforces "best contract > others by margin"; it does **not** require calibrated probabilities or accurate utility magnitudes. With BCE losses for win/clean/stopout each weighted 0.25–0.35×, the calibration heads can satisfy their loss with near-constant predictions (e.g., 0.5 prob = 0 logit) while ranking carries the gradient mass.

This directly accounts for symptom A: `pred_win_prob` flatness is the equilibrium of an under-weighted BCE head behind a bottleneck shaped by ranking. It also accounts for symptom B: the clean head, sharing the bottleneck and trained at 0.35× weight, can't learn a put-specific calibration when the bottleneck is shaped by ranking under a 2.83:1 call-skewed gradient distribution. Symptom C follows from the same dynamic: ranking learns "puts have lower utility globally" (because of imbalance), inflating their decision margins via the flat-anchor difference (#5).

### 3. No side-balance default

**Status:** VERIFIED → SYMPTOM-EXPLAINED for A, B, C.

Both side-balance levers are **off by default**:
- `w_side_contrastive = 0.0`: there is NO loss term forcing call and put predictions to align on a bar
- `side_balance_weight = 0.0`: there is NO per-bar inverse-frequency reweighting of regression / calibration gradients

The training set has 2.83:1 call:put imbalance ([train_unified_policy.py:670 comment](../layer2/train_unified_policy.py#L670)). With both levers off, ranking loss sees ~2.83× more call gradient mass than put gradient mass; the model learns a global call bias.

Note: prior work (memory: `w_side_contrastive sweep falsified`, `side-balance hypothesis partial`) tried each lever in isolation and found neither alone could fix the floor. The Phase 3 reading explains why: each lever addresses only ONE channel of the imbalance, while the symptoms are caused by the joint action of (loss imbalance × ranking dominance × shared bottleneck × flat anchor). A retraining plan has to flip multiple levers together.

### 4. Argmax tie-break

**Status:** VERIFIED → INCONCLUSIVE for D.

`numpy.argmax(token_scores, axis=1)` returns the first-occurrence index on ties ([train_unified_policy.py:297](../layer2/train_unified_policy.py#L297)). On tied utility logits the lowest-indexed contract wins. This could systematically favour low-strike calls (token slots 0..K-1) over high-strike calls and over puts (slots K..2K-1).

But ties are unlikely on a 160-dim continuous-output network — this is a theoretical concern, not a confirmed mechanism. Symptom D ("ORC-triggered calls entered with `decision_margin ~0.60`") is more plausibly explained by #2 + #3 + #5 jointly than by tie-break.

### 5. Flat anchor + arcsinh compression

**Status:** VERIFIED → SYMPTOM-EXPLAINED for C.

`utility_raw[:, 0] = 0.0` is hard-coded. `reg_weight_train[:, 0] = 1.5` actively pins the flat prediction to 0 with above-average gradient weight. `utility_arcsinh = arcsinh(utility_raw / 100.0)` compresses large dollar utilities (e.g., $500 → arcsinh(5) ≈ 2.31; $50 → arcsinh(0.5) ≈ 0.48 — a 10× ratio shrinks to 4.8×).

`decision_margin = best_nonflat_score - flat_score`. With flat pinned near 0 and arcsinh compressing positive utilities, decision_margin naturally has a soft ceiling around 1.0–1.5 in most regimes. When the model develops a call-bias, puts get systematically lower utility predictions, which means **larger decision_margin gaps for puts when puts ARE picked** (because the flat anchor stays near 0 but the put-utility gets compressed lower by the arcsinh, making the difference noisier and biased high). This produces symptom C: high decision_margin on puts is a signature of model uncertainty + arcsinh compression, not signal strength.

### 6. Target asymmetry

**Status:** REFUTED.

Calls and puts share **identical** label-construction code paths in `build_contract_action_surface` ([action_surface_dataset.py:588-708](../layer2/action_surface_dataset.py#L588-L708)). `time_stop_pnl`, `best_exit_pnl`, `horizon_pnl`, `stopout_risk`, `clean_entry`, `win_label`, and `hybrid_live_utility` are all computed from the same per-side path data with identical formulas. There is no greek-normalization step, no side-conditional stop, no asymmetric mae/mfe.

The asymmetric outcomes (call/put bias) are a function of the input data distribution (2.83:1 imbalance) and the loss-weight composition (#2, #3) — not the labels. This pathology hypothesis is wrong.

### 7. Calibration objective

**Status:** PARTIALLY VERIFIED → SYMPTOM-EXPLAINED for A, B.

All calibration heads (win, clean, stopout) use the same `_masked_bce` loss ([unified_policy.py:316-328](../layer2/unified_policy.py#L316-L328); applied at lines 849-851). `clean_loss` and `stopout_loss` do use higher weights (0.35) than `win_loss` (0.25), but all three are below the 1.45× cumulative ranking-pressure ceiling (#2).

The objective itself isn't pathological — BCE on per-action labels is correct. The pathology is in the **weighting**, which is what makes calibration heads under-trained. This is REALLY pathology #2 expressed in the calibration channel; classifying it as a separate item is a useful organisational fiction but the fix is the same: rebalance weights.

### 8. Risk-band ranking decoupling

**Status:** PARTIALLY VERIFIED → SYMPTOM-EXPLAINED for E.

`_risk_band_ranking_loss` enforces ranking only within risk-band cohorts ([unified_policy.py:364-405](../layer2/unified_policy.py#L364-L405)), while `_pairwise_ranking_loss` enforces ranking globally across all 24 contracts ([unified_policy.py:331-361](../layer2/unified_policy.py#L331-L361)). Both are added (1.0× and 0.5× the `w_ranking` knob respectively). The combined effect is "rank globally AND within band" — coherent at the loss level.

The coupling to symptom E (oracle drops on alternate pairs) is indirect: when the L3 oracle was built from L2's top-1 picks per day, the chosen risk band per (day, side) was consistent across days because of this dual-ranking pressure. Alternate (row, action) pairs that the veto's rescan picks may sit in different risk bands than the L2 baseline winners, and the L3 oracle's training distribution doesn't cover those bands well. Phase 2b's oracle-NaN behavior is an extreme version of this — alternate bars literally have no oracle predictions because the oracle's training mask excluded them.

## Cross-pathology synthesis

The five symptoms are produced by **three reinforcing mechanisms**, not eight independent pathologies:

1. **Loss-weight imbalance + class imbalance** (#2 + #3): ranking dominates calibration and regression; calls dominate puts in the gradient mass; under-weighted calibration heads can't correct.
2. **Shared bottleneck + flat anchor + arcsinh compression** (#1 + #5): the model's representation is a single 160-dim vector pinned at flat=0, with diminishing returns in the arcsinh-compressed utility scale; this turns the imbalance into specific pathologies in `decision_margin` and `pred_win_prob`.
3. **L3 oracle/policy training coupling** (#8 + the explicit Phase 2b finding): the L3 oracle is trained only on L2's top-1 picks, so any post-L2 intervention (veto, alternate-bar selection, side rebalancing at inference) lands on (row, action) pairs the oracle has not learned. The oracle's metric becomes biased AGAINST any change the policy makes after L2 training freezes.

These three together fully account for symptoms A, B, C, E. Symptom D (ORC-triggered call entry with margin ~0.60) is best explained by the same combo: a global call bias makes ORC-fired call predictions inflate their utility above the put alternative on the same bar, producing a near-1.0 raw utility that arcsinh compresses to ~0.6 decision_margin. It's not a tie-break artifact (#4); it's a calibrated outcome of an mis-calibrated objective.

## Concrete next-session retraining plan

This is the **prescription Phase 3 produces** for the next-session L2 retraining experiment. Each item targets a verified mechanism, not a speculation.

### Tier 1 — must flip (loss rebalance + side balance)

1. **Set `w_side_contrastive = 0.5–1.0`** (currently 0.0). Forces direct call-vs-put discrimination at training time; shapes the bottleneck to encode side information.
2. **Set `side_balance_weight = 0.5–1.0`** (currently 0.0). Per-bar inverse-frequency reweighting compensates for the 2.83:1 call/put imbalance in the regression / calibration gradient channels.
3. **Increase `w_clean` and `w_stopout` to 0.5–0.75**, decrease `w_ranking` to 0.5 (currently 1.0). Brings ranking : calibration ratio from 5:1 to ~2:1 — calibration heads see more gradient mass.

These three together correspond to "fix the imbalance + don't let ranking dominate." Joint flip is the hypothesis the prior memory entries (`w_side_contrastive sweep falsified`, `side-balance hypothesis partial`, `combined fix cleared floor on seed 42`) point at: each lever alone fails; combined fix worked on seed 42; needs full validation across 5 seeds.

### Tier 2 — investigate

4. **Decouple flat anchor from utility unit:** instead of `utility_raw[:, 0] = 0`, train flat as a separate logit-scale prediction not tied to the dollar arcsinh. Removes the arcsinh-driven ceiling on `decision_margin` for high-conviction trades.
5. **Rebuild L3 oracle on a wider (row, action) distribution:** instead of training the oracle only on L2's top-1 picks, train on the union of (top-K picks + random alternate bars) so the oracle's prediction quality survives downstream filtering. This directly addresses Phase 2b's finding that the veto layer looks worse on the oracle metric than it actually performs.

### Tier 3 — defer

6. Argmax tie-break randomisation (#4) — theoretical concern, no evidence it matters in practice.
7. Per-side batchnorm or dropout (would help if the shared bottleneck is the bottleneck — but Tier 1 should be tried first).

## Cost summary for retraining experiment

The Tier 1 retraining requires:
- One full 5-seed train (max_epochs=12, ~6–8 GPU-hours per 3-seed promotion run on H100, ~$200–300 Akash) — same cost envelope the user authorized in the original Phase 3 plan.
- Reuse existing dataset and L3 oracle (Tier 2 #5 is the harder rebuild, $0–100 incremental for oracle rebuild + retraining-coupled validation).

## Files

- `v3/layer2/unified_policy.py` (read-only)
- `v3/layer2/train_unified_policy.py` (read-only)
- `v3/layer2/action_surface_dataset.py` (read-only)
- `v3/reference/l2_architectural_review_2026_04_26.md` (this doc)

## What ships from Phase 3

Nothing — Phase 3 is read-only investigation. The veto layer from Phase 2b stays opt-in research-only. Tier 1 retraining is the recommended next-session work.
