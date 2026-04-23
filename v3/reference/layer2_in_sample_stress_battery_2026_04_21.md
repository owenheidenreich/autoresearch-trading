# Layer-2 In-Sample Stress Battery — 2026-04-21

## TL;DR

Three independent in-sample stress tests demolish the SOFT atm_iv
verdict. The +0.075 PF lift on the detach-side baseline was selection
bias. The full Layer-2 routing-and-gating workstream is now exhausted at
the cheap-probe level with no surviving signal beyond what the entry gate
already captures.

What survives:
- **Detach-side shared-encoder PF 1.455 / DD 36.9% baseline is intact** —
  none of the stress tests undermined the headline.
- **The shared-encoder's direction head IS doing real work** (+0.339 PF
  over random direction), unlike the tree baseline (+0.078) which is
  structurally weaker.

What's dead:
- atm_iv-bottom-2-decile suppression as a regime gate.
- Any plausible "find a single context feature that gates fallback puts"
  thesis on this artifact.

What's worse than we knew:
- The **tree baseline is structurally weak**: 3 of 5 folds negative
  (PF 0.691 / 0.620 / 0.682). The aggregate PF 1.122 is balanced
  precariously by folds 1 and 3 alone.

## Tests run

After confirming Polygon access is gone (no fresh OOS data possible),
ran four in-sample stress tests on the existing artifacts. All cheap
(seconds to ~1 minute each, CPU-only).

| # | Test | Verdict |
|---|---|---|
| 1 | Random-suppression baseline (atm_iv vs random drops) | atm_iv at p78 — modest |
| 2 | K-sensitivity on atm_iv gate (K = 1, 2, 3, 4) | K=2 single-decile peak; collapses one decile in either direction |
| 3 | Tree-baseline transfer of atm_iv gate | atm_iv suppression HURTS tree (1.122 → 1.095) |
| 4 | Reality checks (per-fold + random-direction) on tree baseline | Tree is structurally weaker — 3 negative folds, side head barely better than random |

Tests 1, 2, 3 collectively falsify the SOFT atm_iv verdict.
Test 4 is informational about the tree baseline's status.

## Test 1 — Random-suppression baseline

Question: when you drop 66 of 176 fallback bars at random (matching the
trade-count effect of atm_iv K=2 suppression), what PF distribution do
you get? Where does atm_iv sit in that distribution?

Script:
[v3/analysis/layer2_random_suppression_ablation.py](../analysis/layer2_random_suppression_ablation.py).
200 random seeds.

```
Baseline (no suppression):     PF=1.455  DD=36.9%  TPD=0.917
atm_iv K=2 suppression:        PF=1.530  DD=26.7%  TPD=0.697
Random drop 66 of 176 (n=200):
  mean=1.400  std=0.151  min=1.031  max=1.765
  p50=1.396  p70=1.494  p75=1.517  p90=1.599
```

**atm_iv reference PF (1.530) sits at the 78th percentile of the random
distribution.** 21.5% of random suppressions match or exceed the lift.

Reading: at p78, this is a weak signal but not random noise.
Independently, this would be "PLAUSIBLE." However, see Tests 2 and 3.

Artifact:
[v3/artifacts/layer2_random_suppression_ablation/random_suppression_ablation.json](../artifacts/layer2_random_suppression_ablation/random_suppression_ablation.json).

## Test 2 — K-sensitivity on atm_iv gate

Question: is K=2 (the original choice) a robust local optimum or a
data-snooped peak?

Re-ran the payoff-gating probe with K = 1, 3, 4 and compared to K=2.

| K | atm_iv kept | atm_iv suppressed | atm_iv PF (Δ vs 1.455) | atm_iv DD (Δ vs 36.9%) | Best feature at this K |
|---:|---:|---:|---:|---:|---|
| 1 | 132 | 44 | 1.429 (−0.026) | 31.5% (−5.3) | abs_sigma_pos |
| **2** | **110** | **66** | **1.530 (+0.075)** | **26.7% (−10.2)** | **atm_iv** |
| 3 | 88 | 88 | 1.447 (−0.008) | 34.6% (−2.3) | sigma_pos |
| 4 | 66 | 110 | 1.471 (+0.017) | 45.4% (+8.5) | sigma_pos |

Two damning patterns:

1. **K=2 is a single-point peak.** One decile in either direction
   (K=1, K=3) the lift evaporates: −0.026 and −0.008 respectively. A
   robust gate would degrade gradually, not collapse at one decile of
   movement. This is the textbook signature of a p-hacked threshold.
2. **The "best feature" rotates with K.** abs_sigma_pos at K=1, atm_iv
   at K=2, sigma_pos at K=3 and K=4. None of them is consistently best,
   which means none is doing genuine work — they take turns winning by
   chance.

Net: the original SOFT result was specific to (atm_iv, K=2), and that
specific cell was the lucky draw out of 5 features × 4 K values = 20
cells. P-hack territory.

## Test 3 — Tree-baseline transfer of atm_iv gate

Question: if atm_iv-bottom-2 suppression is a real regime signal, it
should at minimum not HURT the tree baseline. Does it transfer?

Re-ran the payoff-gating probe with `--baseline-run-dir
v3/artifacts/layer2_entry_side_fixedq_60_10`.

Tree baseline: PF 1.122, DD 56.2%, 287 trades (131 teacher, 156 fallback).

| Feature | Suppress side | atm_iv ↓ result | Tree result |
|---|---|---:|---:|
| sigma_pos | top | 1.324 (−0.131) | 1.071 (−0.051) |
| abs_sigma_pos | bottom | 1.295 (−0.160) | 1.208 (+0.086) |
| **atm_iv** | bottom | **1.530 (+0.075)** | **1.095 (−0.027)** |
| last10_range_over_omar | bottom | 1.426 (−0.029) | 1.154 (+0.033) |
| first15_range_pct | bottom | 1.473 (+0.019) | 1.159 (+0.038) |

**The atm_iv signal is shared-encoder-specific. On the tree baseline it
HURTS (PF goes from 1.122 to 1.095, avoided $/trade is +$144 — we throw
away winners).** A real regime feature would help both architectures or
be neutral; a model-specific overfit shows the kind of architecture
asymmetry observed here.

Tree's "best" feature is abs_sigma_pos (different from shared-encoder's
"best" of atm_iv). Neither feature is universally best — another
indicator that no single feature is identifying real payoff regimes.

Combined with Tests 1 and 2, the verdict on atm_iv as a payoff gate is
**HARD FALSIFIED**. Three independent angles converge.

## Test 4 — Reality checks on the tree baseline

Question (from the [reality-checks doc](layer2_reality_checks_2026_04_21.md)
§"What this plan does NOT change"): is the random-direction finding
shared-encoder-specific or general to Layer-2? And what does the tree
baseline's per-fold profile look like?

Per-fold:

| Fold | Tree PF | Tree DD% | Shared-Enc PF | Shared-Enc DD% |
|---:|---:|---:|---:|---:|
| 0 | **0.691** | 49.4 | 0.860 | 28.2 |
| 1 | 1.515 | 23.6 | 1.439 | 22.5 |
| 2 | **0.620** | 44.9 | 1.038 | 27.6 |
| 3 | 2.215 | 12.9 | 2.095 | 21.4 |
| 4 | **0.682** | 58.0 | 1.801 | 32.2 |

**The tree has THREE losing folds (0, 2, 4); the shared-encoder has
one (fold 0).** Aggregate PF 1.122 vs 1.455 is one symptom. The deeper
finding is that the tree is far less consistent across folds, so the
shared-encoder's incremental architecture cost is buying real fold-level
robustness.

Random-direction ablation:

| Architecture | Model PF | Random-direction PF (mean over 10 seeds) | Direction lift |
|---|---:|---:|---:|
| Tree fixedq 60/10 | 1.122 | 1.044 | +0.078 |
| Shared-encoder detach-side | 1.455 | 1.116 | +0.339 |

Both random-direction PFs are ≥ 1.0 (HARD STOP per
[plan §2b](../../.claude/plans/read-v3-reference-index-md-and-v3-handof-stateful-bengio.md))
— so the entry gate is the load-bearing piece for both. But the
shared-encoder's side head adds **4.4× the directional lift** the tree's
does (+0.339 vs +0.078). The earlier "direction head is cosmetic" framing
was correct for the tree; for the shared-encoder it understates the
contribution.

## Cross-test synthesis

The shared-encoder PF 1.455 baseline:

- **Survives intact.** None of the four stress tests undermined the
  headline replay number.
- **Side head is more substantive than tree's.** Tree gets +0.078 over
  random direction; shared-encoder gets +0.339. The detach-side variant
  earned its incremental architecture cost.
- **Has no remaining cheap-probe lift to extract.** Routing (route-aware,
  fallback-only put-vs-flat, full call/put/flat) is exhausted. Gating
  (atm_iv K=2) was selection bias. K-sensitivity, random-suppression,
  and tree-transfer all converge on falsification.

The tree baseline:

- Is **structurally weaker** than the aggregate PF 1.122 suggests. Three
  losing folds out of five. Fold 4 (PF 0.682, 1.7% calls) is especially
  exposed.
- Is not a viable fallback-architecture choice if the shared-encoder
  becomes unavailable for any reason (infrastructure, training, etc.).

## What this updates in the broader picture

- The previous SOFT verdict in
  [layer2_payoff_gating_2026_04_21.md](layer2_payoff_gating_2026_04_21.md)
  should be read alongside this stress battery. The SOFT was honest at
  the time but is now subsumed by these falsifications.
- The previous comprehensive diagnostic
  [layer2_diagnostic_full_2026_04_21.md](layer2_diagnostic_full_2026_04_21.md)
  raised four risks for paper trading. None of them is resolved by these
  stress tests; if anything, the inability to find any usable in-sample
  gating amplifies the "fold 0-like regime risk" concern.
- The four-probe routing+gating workstream on the shared-encoder
  artifact is officially closed. Any further iteration on this artifact
  needs either (a) fresh OOS data (blocked by Polygon) or (b) an
  architectural change.

## What NOT to do next

- Do **not** re-run K-sensitivity on a different feature looking for a
  K-stable peak. That is the same multi-comparison trap.
- Do **not** treat tree baseline as a fallback-architecture choice. It
  has 3 losing folds.
- Do **not** unblock paper trading on the shared-encoder. The
  fold-0-like regime risk is unchanged from the original reality-checks
  verdict.
- Do **not** spin up GPU for any retraining variant of this artifact.
  No remaining hypothesis justifies the spend.

## Defensible options going forward

The shared-encoder artifact has been thoroughly mined. Three reasonable
directions, none fully scoped:

1. **Pivot to a different research question entirely.** Treat shared-
   encoder PF 1.455 as the documented best-effort artifact under current
   data and walk away from it. Open question is what to point next at.
2. **Wait for fresh data.** If Polygon (or an alternative) reopens at a
   later point, run the OOS retest scoped earlier with no further
   iteration in the meantime. This is "paused" rather than "abandoned."
3. **Architectural changes that don't depend on new data.** Examples:
   try a sequential / state-aware model on the existing dataset; try
   per-fold ensembling; try meta-learning across folds. Each of these
   is meaningfully more expensive than a cheap probe and risks being
   another iteration of the same data-snooping problem on the same
   ~986 days.

Picking one is the user's call. The pause is honest; further iteration
on the existing artifact is not.

## Files added

- `v3/analysis/layer2_random_suppression_ablation.py` — Test 1 script
- `v3/artifacts/layer2_random_suppression_ablation/random_suppression_ablation.json`
- `v3/artifacts/layer2_payoff_gating_probe_tree/payoff_gating_probe.json` — Test 3 artifact

Reused (no changes):
- `v3/analysis/layer2_payoff_gating_probe.py` (Tests 2 and 3 via flags)
- `v3/analysis/layer2_random_direction_ablation.py` (Test 4)
- `v3/analysis/layer2_per_fold_diagnostic.py` (Test 4)

## Verification

- [x] Random-suppression: 200 seeds, atm_iv reference at p78
- [x] K-sensitivity: K=1,2,3,4 swept; only K=2 lifts
- [x] Tree-transfer: atm_iv on tree HURTS (PF 1.122 → 1.095)
- [x] Tree per-fold: 3 losing folds confirmed; tree fold 4 PF 0.682
- [x] Tree random-direction: PF 1.044 mean (vs model 1.122)
- [ ] Commit
