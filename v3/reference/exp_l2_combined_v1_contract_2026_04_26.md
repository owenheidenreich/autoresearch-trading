---
date: 2026-04-26
status: LOCKED — pre-GPU contract; no threshold fishing or mid-run changes
parent: l2_architectural_review_2026_04_26.md
experiment_id: L2_combined_side_balance_calibration_exit_exposure_v1
---

# Locked experiment contract: L2 combined retrain (side-balance + calibration + exit-exposure)

## Rationale

The veto-layer probe (Phase 1+2b) and the architectural review (Phase 3)
together established that:

> L2's selected-trade confidence is not a reliable proxy for actual
> trade quality, and L3 is distribution-coupled to old L2 picks.
> Therefore the next experiment should retrain the entry/selection
> system and exit-exposure assumptions together.

This is **not** "the rules failed, therefore retrain." This is "the
rules exposed that the model's confidence machinery has a structural
miscalibration around side choice, and the L3 oracle's training is
coupled to a frozen entry distribution."

Three mechanisms must be tested **jointly** because past memory shows
each one tested alone fails (`w_side_contrastive sweep falsified`,
`side-balance hypothesis partial`, `combined fix cleared floor on
seed 42`).

## Frozen experiment scope

### Mechanism 1 — Calibration rebalance

Current ranking : calibration : regression weight ratio is 5 : 2.9 : 1.
Rebalance to ~2 : 2 : 1 so calibration heads receive comparable gradient
mass to ranking.

| Knob | Current default | Locked target |
|---|---|---|
| `w_ranking` | 1.0 | **0.5** |
| `w_clean` | 0.35 | **0.6** |
| `w_stopout` | 0.35 | **0.6** |
| `w_win` | 0.25 | **0.4** |
| `w_regression` | 0.5 | 0.5 (unchanged) |
| `w_dollar` | 0.25 | 0.25 (unchanged) |
| `w_return` | 0.25 | 0.25 (unchanged) |

### Mechanism 2 — Side-balance + side-contrastive

Both side levers ON; train data has 2.83:1 call/put imbalance.

| Knob | Current default | Locked target |
|---|---|---|
| `w_side_contrastive` | 0.0 | **0.75** |
| `side_balance_weight` | 0.0 | **0.75** |
| `cohort_balanced` (in `_side_contrastive_loss`) | auto-on when w>0 | **on** |

### Mechanism 3 — L3 exit-exposure rebuild

Current oracle is built on L2's daily-best top-1 picks only. Rebuild on
a strictly broader (row, action) distribution so L3 sees alternates the
new L2 might pick.

**Locked oracle-rebuild rules:**

The existing `v3/layer2/build_simulated_l3_oracle.py` script already has
the required plumbing: `--l3-training-source candidate_surface` trains
the per-window L3 models on a deterministic broader sample from the
action-surface dataset (`v3/layer3/common.py:49` ⟹
`DEFAULT_CANDIDATE_TRAIN_MAX_PER_DAY = 4` per-day candidate trades vs the
~1 daily-winner under `champion` mode). This is the existing
"candidate_surface" exposure mode; we're flipping it on for the
experiment instead of writing new code.

- `--l3-training-source candidate_surface` (vs default `champion`)
- `--candidate-train-max-per-day 4` (default, locked at 4 to keep
  training cost bounded)
- Keep H3a features in L3 (`--target peak`, no special flag — H3a
  features are inside `v3/layer3/h3a_features.py` and already integrated
  into `train_models_by_window`)
- Keep the binary peak target (per `H3f regret target falsified` memory)
- Keep the same `hybrid_live_utility` scoring

## Baselines for comparison

| Baseline | Forward-walk (42-day) PF (oracle) | Offline 5-seed PF | Source |
|---|---|---|---|
| Champion (`spx_combined_3seed_001`) | 1.700 | 1.881 | `forward_walk/spx_combined_3seed_001_with_oracle.json` |
| **H3a (current committed best)** | **1.805** | **2.038** | `h3a_features_validated` memory + `forward_walk/h3a_with_oracle.json` |

The retrain must clear **H3a**, not just the original champion.

## Primary pass gates (ALL must hold)

The retrain passes the primary gate iff **all** of:

1. **5-seed FW PF (with oracle) ≥ 1.805** (the H3a benchmark)
2. **No more than 1 seed has FW Δ < -0.10 vs H3a per-seed PF**
3. **5-seed FW mean hl per calendar day ≥ H3a's** (or within -10% margin)
4. **5-seed FW total trade count ≥ 70% of H3a's**
5. **Per-cell FW PF floor does not collapse** (no cell with ≥10 trades drops to PF < 0.7)

## Secondary pass gates (informational, calibration health)

These are "the model is doing what we asked" checks. Failing all four is
itself a fail signal even if primary passes (would suggest reward-hacked
gain).

6. **`pred_win_prob` Spearman with hl improves from current ~−0.025 toward ≥ +0.10**
7. **`pred_clean_entry_prob` is no longer inverted on puts** (low/high quartile PF gap is < 1.5×, currently ~3×)
8. **`decision_margin` is no longer anti-calibrated for puts** (top-quartile-margin put PF ≥ bottom-quartile put PF)
9. **Wrong-side pick rate decreases ≥ 5 percentage points** in the 4 worst cells from Phase 1 (s0_iv2, s2_iv0, s2_iv1, s0_iv1)

## Hard-fail conditions (immediate revert; no negotiation)

- FW PF (with oracle) decreases vs H3a (1.805)
- Lift only appears offline; FW shows no improvement or degrades
- Calibration improves but actual trade quality (FW hl/day) worsens
- L3 exit performance collapses on the new L2 distribution (FW DD increases by > 50% relative)
- Any single seed crashes by > 0.5 PF vs its H3a baseline

## Anti-reward-hacking guardrails

- **No threshold fishing.** The 9 gates above are locked. If FW barely
  passes Gate 1 but fails Gates 6-9, the experiment is a fail, not a
  marginal pass.
- **No mid-run config changes.** If the run looks bad at epoch 6, it
  finishes; we don't bail out and tune.
- **Post-hoc threshold tuning is forbidden.** The decision-margin grid
  search in `_calibrate_abstention_policy` runs as before, but no
  manual overrides post-result.
- **No look-ahead in the L3 oracle exposure fix.** The L1-top-3 per-side
  picks are determined by causal entry-time features only — verified by
  the same mutate-future audit as H3a.
- **Single experiment only.** This is not a sweep over weights or
  exposure modes. If the locked config fails, the next experiment is a
  separately-locked v2, not a quick retry.

## Resource budget

| Step | Cost | Approval |
|---|---|---|
| Pre-GPU harness eval + look-ahead audit | ~30 min CPU | self |
| L3 oracle rebuild (~67 min CPU/seed; seeds 43-46 parallelized after seed 42) | ~90-110 min CPU wall-time | self |
| 5-seed L2 training on H100 (~1.5 GPU-hours/seed × 5 seeds, sequential per-seed invocations) | ~7-8 GPU-hours | **explicit user OK required** |
| FW evaluation + metrics on the 9 gates | ~10 min CPU | self |
| Total GPU spend | $200-300 Akash | **explicit user OK required** |

Note on cost-estimate revision: original 25-min CPU oracle estimate was
wrong because the smoke phase used `--max-days 50` which skipped the
dominant per-bar simulation cost. The full `candidate_surface` oracle
runs ~1M (row, action) sims per seed; this is intrinsic to broader
exposure coverage and there is no shortcut without sacrificing the
mechanism.

`DEPOSIT_ACT=1` per `feedback_screening_deposit` memory (screening), but
this is a full official run not a screen — confirm budget before launch.

Note on multi-seed invocation: `train_unified_policy.py` takes one
`--simulated-l3-oracle` path and applies it to all seeds in its loop;
to pair each seed with its own oracle, we invoke training 5 times (once
per seed) inside the run script's `train` phase. This is sequential on
single GPU and matches the per-seed oracle convention used by the
existing forward-walk infrastructure.

## Files that will change

| File | Change |
|---|---|
| `v3/layer2/build_simulated_l3_oracle.py` | invoked with `--l3-training-source candidate_surface`; no code change |
| `v3/layer2/train_unified_policy.py` | invoked with the locked config flags below; no code change |
| Run config script | NEW `scripts/exp_l2_combined_v1_run.sh` (drives oracle build → training → FW) |
| Lab notebook | append-only entry per `feedback_log_experiments` memory |
| `v3/results.tsv` | one row per seed |

## Locked invocation

```sh
# L3 oracle rebuild per seed (CPU, ~5 min/seed)
PYTHONPATH=. python3 -m v3.layer2.build_simulated_l3_oracle \
  --seed $SEED \
  --dataset v3/artifacts/layer2_action_surface_dataset.pkl \
  --l3-training-source candidate_surface \
  --candidate-train-max-per-day 4 \
  --target peak \
  --output v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed${SEED}_combined_v1.npz

# L2 5-seed training (GPU, official, ~6-8 hours total on H100)
PYTHONPATH=. python3 -m v3.layer2.train_unified_policy \
  --seed $SEED \
  --dataset v3/artifacts/layer2_action_surface_dataset.pkl \
  --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed${SEED}_combined_v1.npz \
  --utility-target hybrid_live \
  --w-ranking 0.5 \
  --w-clean 0.6 \
  --w-stopout 0.6 \
  --w-win 0.4 \
  --w-regression 0.5 \
  --w-dollar 0.25 \
  --w-return 0.25 \
  --w-side-contrastive 0.75 \
  --side-balance-weight 0.75 \
  --max-epochs 12 \
  --patience 4
```

The training script auto-resolves output directories from the dataset
name + seed. Per-seed artifact lives at
`v3/artifacts/layer2_unified_policy_<dataset_tag>_seed${SEED}/seed_${SEED}/`.

## Post-run report template

```
=== L2_combined_v1 results ===

vs H3a baseline:
  Δ FW PF (oracle):   X.XXX  (Gate 1: PASS/FAIL ≥1.805)
  Δ FW hl/day:        $XX    (Gate 3: PASS/FAIL)
  Δ FW trade count:   ±X%    (Gate 4: PASS/FAIL ≥70%)
  Per-seed Δ count <-0.10:   X (Gate 2: PASS/FAIL ≤1)
  Per-cell PF floor: X.XXX   (Gate 5: PASS/FAIL ≥0.7)

Calibration health:
  pred_win_prob Spearman:    X.XX (Gate 6: PASS/FAIL ≥0.10)
  pred_clean_entry put gap:  X.Xx (Gate 7: PASS/FAIL <1.5x)
  decision_margin put inv:   ±    (Gate 8: PASS/FAIL non-inv)
  Wrong-side pp drop:        X    (Gate 9: PASS/FAIL ≥5pp)

Hard-fail trip:               yes/no
Verdict: PASS / FAIL / HARD-FAIL

If PASS: promote spx_combined_v1 to champion; commit; update memory.
If FAIL: revert; commit revert; document what each gate told us;
         design v2 experiment with a different hypothesis.
```
