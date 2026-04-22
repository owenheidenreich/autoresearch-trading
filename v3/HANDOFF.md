# v3 Handoff — Layer-2 Mainline + Layer-2.5 Timing

**Date:** 2026-04-22  
**Project framing:** SPX 0DTE long premium, $25k account, 1 contract, Moderate-tier Layer 0 rails

## Purpose

This is the current handoff for `v3` after:
- Stage 1 research completed
- A1 ORC sigma gate shipped
- late-session teacher search falsified
- W2a soft features shipped into `v2.2`
- first `v3` Layer-2 supervised branch implemented and screened
- methodology overhaul reset the honest champion to `V0 + time-stop`
- Layer-2.5 full-surface entry patience was promoted into the architecture

The mainline is no longer “write more teachers” or “keep tuning `v2/train.py`.”
The mainline is now:
- `v3/layer2` for bar-state scoring and the new unified action-policy redesign
- `v3/layer25` for timing / entry patience on the full scored-bar surface
- `v3/layer3` for honest rolling-window exits on top of the Layer-2.5 trades

There is now also a **W1 surface branch** under `v3/layer2`:
- `export_surface_dataset.py`
- `train_surface_model.py`

This branch is the first direct response to the audit's main criticism:
the old Layer-2 flattened away sequence structure and the contract surface.
First result:
- short-budget 5-fold W1 run on the new surface branch: `287` trades, `PF 1.571`, `DD 15.3%`
- see [w1_surface_branch_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/w1_surface_branch_2026_04_22.md)

There is also now a first-pass **unified action-policy branch** under `v3/layer2`:
- [export_action_surface_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/export_action_surface_dataset.py)
- [train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py)

This is the beginning of the intended end-state:
- one policy over `flat + real contract candidates`
- teachers as features, not direction overrides
- patience / stopout supervision folded into the same model
- promotion only from the `13`-window rolling harness

Current status:
- canonical action-surface export is implemented
- rolling unified-policy smoke path is verified end-to-end
- first smoke replay was not promoted: `46` trades, `PF 0.464`, `DD 50.4%`, calibrated margin `-0.083` (incoherent — trading below the model's own flat pick)
- post-audit fix pass: calibrator floored at `0.0` margin, regression flat weight `0.5 → 1.5`, ranking hinge `0.05 → 0.20`, added bidirectional `_flat_ranking_loss`, loss-weight rebalance `w_ranking 0.75 → 1.0 / w_regression 1.0 → 0.5`, smoke budget `4/2 → 8/3` epochs/patience
- fixed smoke on the latest rolling window: `39` trades, `PF 0.909`, `DD 15.1%`, trade_share `0.65` (in-band), calibrated margin `+0.305`
- dev-tier rolling single-seed (13 windows, seed 42, CPU) aggregate: `410` trades, `PF 1.066`, `DD 48.8%`, trade_share `0.526`, fast-loser rate `0.21` (~patience-gate threshold), beats-V1 rate `0.554`
- dev-tier rolling **3-seed** (seeds 42/43/44, CPU) aggregate: `1201` trades, mean seed PF `1.112` (std `0.045`), aggregated PF `1.111`; seed 43 clears baseline at `1.172`, seeds 42/44 just short
- a weak-window calibrator fallback rule was implemented and tested: it **reduced cross-seed variance** (std `0.045 → 0.026`) but **did not lift the mean** (`1.112 → 1.107`). Reverted as net-neutral; the "weak window" hypothesis is falsified as the binding constraint
- **GPU 3-seed promotion** (plan section 1 protocol, Akash H100, `--tier promotion --device cuda`): mean PF `1.116` (std `0.025`), aggregated `1.115` across `1171` trades. Seeds `1.097 / 1.151 / 1.101`. All three clear the per-seed `PF ≥ 1.0` floor. GPU halved cross-seed std and dropped mean DD by `~13pp` (48.8→29.9% on seed 42), but **did not lift mean PF** (+`0.004` vs CPU = noise). Aggregate PF gate `1.132` remains FAIL by `1.5%`
- the unified action policy is **not yet shelved**: PF gate fails but DD is **dramatically better** than the `V0 + time-stop` PF baseline (`V0 DD 95.9%` cold-start artifact; unified policy GPU DDs `27–38%`). The honest PF champion remains `V0 + time-stop` at `PF 1.132`; the DD baseline for trade-quality gate comparison is the Layer-2.5 patience-gated run at `DD 21.4%` (different artifact, not V0)
- across all 3 GPU seeds the **call share is `93%`** (seeds: `93.5% / 91.9% / 93.5%`). This is too extreme to attribute to real regime bias without a side-contrastive ablation first. The current ranking loss (best-vs-rest + flat-vs-contract) never directly contrasts same-bar call-vs-put; the model may be collapsing into a side prior. A same-bar contrastive term is a legitimate pre-shelving experiment
- same-bar side-contrastive training is now wired into [v3/layer2/unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/unified_policy.py) and [v3/layer2/train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py) via `--w-side-contrastive`. First rolling seed-42 ablation at `0.5` **did reduce call share** (`93.5% → 90.9%`), but it also **hurt PF badly** (`1.151 → 1.002`) and worsened DD (`29.9% → 59.7%`). Conclusion: the side prior is real, but the first blunt contrastive weight is too aggressive; do not treat this as a solved fix yet
- Layer-3 can now consume **unified-policy chosen trades directly** via [v3/layer3/train_rolling.py](/Users/gduby/Documents/autoresearch-trading/v3/layer3/train_rolling.py) `--entry-source unified --chosen-trades ...`, and the first honest seed-42 replay is promising: baseline unified entries at `PF 1.151 / DD 29.9%` improved to exploratory best `PF 2.033 / DD 22.3%` at threshold `0.19`. This is still same-OOS threshold picking, so it is a research clue, not a deployable setting
- **Cross-seed Layer-3 on unified entries (seeds 42/43/44)** confirms the composition is not a seed-42 artifact: at default `thr 0.19`, per-seed PFs are `2.033 / 1.623 / 1.711` (mean `1.789`, min `1.623`); at `thr 0.15` the mean rises to `1.825`. All three seeds beat V0's `1.132` by `≥+0.49` at every threshold in the `0.15–0.30` grid; mean DD `23.7%` is close to Layer-2.5's `21.4%` and far below V0's `95.9%`. **Unified+L3 is now the leading promotion candidate.** Still same-OOS threshold picking — next cycle is honest prior-window threshold calibration. See [layer3_unified_crossseed_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer3_unified_crossseed_2026_04_22.md)
- see [unified_policy_calibration_fix_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/unified_policy_calibration_fix_2026_04_22.md) for the audit and CPU 3-seed analyses
- see [unified_policy_gpu_promotion_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/unified_policy_gpu_promotion_2026_04_22.md) for the GPU promotion result and next-step options (Layer-3 outer loop, bar-level decisions, or longer sequence context)
- see [unified_policy_side_contrastive_and_layer3_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/unified_policy_side_contrastive_and_layer3_2026_04_22.md) for the first side-contrastive ablation and the unified-entry Layer-3 follow-up

## TL;DR

What is actually true now:

- **A1 is real and shipped.** ORC with the hard `sigma_pos` direction veto reduced `side_error 402 → 176` with exact dry-run/runtime agreement. See [orc_direction_fix_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/orc_direction_fix_2026_04_20.md).
- **Late-session teacher families did not survive.** The tournament went `0/8`; that regime is Layer-2-only. See [late_session_tournament_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/late_session_tournament_2026_04_20.md).
- **W2a features shipped, but `v2` still failed.** `exp_179` confirmed the old training loop is the bottleneck, not the feature surface. See [w2a_handoff_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/w2a_handoff_2026_04_21.md).
- **The new mainline is `v3` Layer-2.** Export/train/replay tooling now exists under [v3/layer2](/Users/gduby/Documents/autoresearch-trading/v3/layer2).
- **The methodology overhaul changed the honest baseline.**
  - old flashy claim: `V1 + A3 L3 @ 0.19`
  - honest current baseline: `V0 + time-stop`, PF `1.132` across `780` OOS days
- **Layer-2.5 is now the next architectural step.**
  - later timing layers were starving on chosen-trade samples
  - the new full-surface patience layer trains on `69,863` scored bars across `780` OOS days
  - at threshold `0.40`, patience-gated policy replay gives `218` trades, `PF 1.795`, `DD 21.4%`
- **The new final-shape target is no longer the old entry/side threshold stack.**
  - the desired direction is now the unified action-policy path under `v3/layer2`
  - old `direction_mode`, `score_mode`, `side_score_weight`, and heuristic contract selection should be treated as benchmark-era controls, not champion-era ones
- **The highest-EV next move is exit composition, not more GPU.**
  - unified-entry Layer-3 replay on GPU seed `42` lifted `PF 1.151 → 2.033` and `DD 29.9% → 22.3%` on the same entries, albeit with exploratory threshold selection
  - the side-contrastive ablation proved the call-collapse concern is real, but the first strong weight was too blunt; side discrimination still needs tuning, not abandonment
- **Layer-3 now has an honest rolling prototype on top of Layer-2.5.**
  - using the same `218` patience-filtered trades, rolling Layer-3 lifts PF from `1.795` to an exploratory best `2.351` at exit threshold `0.19`
  - important caveat: the threshold sweep is exploratory, not deployment-calibrated
- **There are now two Layer-2 baselines with different roles.**
  - Stable CPU reference:
    - fixed-quantile hybrid tree policy
    - `287` trades, `PF 1.122`, `DD 56.2%`, `0.957` trades/day
  - Max-PF neural research baseline:
    - shared encoder + `--detach-side`
    - `275` trades, `PF 1.455`, `DD 36.9%`, `0.917` trades/day
    - not yet cleared for GPU or paper trading because the later diagnostics showed the edge is still dominated by entry selection plus teacher-conditioned fallback
- **The route-aware fallback follow-up failed.**
  - `q50/q60/q70/q80` all tied at `270` trades, `PF 1.131`, `DD 93.1%`, `0.900` trades/day
  - the representative run chose `270` teacher bars and `0` fallback bars
  - no GPU launch is justified from that branch
- **The fallback-only probe also failed to beat blunt put fallback.**
  - `teacher+put_or_flat_model`: `PF 1.340`, `DD 58.8%`, `0.613` trades/day
  - `teacher+fallback_model`: `PF 1.159`, `DD 96.6%`, `0.697` trades/day
  - baseline `teacher+put` remains best at `PF 1.455`, `DD 36.9%`, `0.917` trades/day
  - this points away from routing and toward fallback regime gating

References:
- [layer2_entry_side_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_entry_side_2026_04_21.md)
- [layer2_shared_encoder_diagnostics_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_shared_encoder_diagnostics_2026_04_21.md)
- [layer2_diagnostic_full_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_diagnostic_full_2026_04_21.md)
- [layer2_route_aware_fallback_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_route_aware_fallback_2026_04_21.md)
- [layer2_fallback_only_probe_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_fallback_only_probe_2026_04_21.md)
- [methodology_overhaul_summary_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/methodology_overhaul_summary_2026_04_22.md)
- [v3/layer25/README.md](/Users/gduby/Documents/autoresearch-trading/v3/layer25/README.md)
- [layer3_rolling_entry_patience_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer3_rolling_entry_patience_2026_04_22.md)

## What Changed

### Stage 1 / teacher layer

- ORC now carries the shipped `sigma_pos` hard veto.
- FailedBreak remains the only other baseline teacher.
- No new late-session teacher should be added on the current evidence.

### Feature / denominator layer

- `v2.core.market_structure` is the shared source of truth for:
  - `sigma_pos`
  - OMAR fields
  - last-10 structure
- The 8 W2a regime features are stable in `v2.2` and are now joined into `v3` Layer-2 export.

### Layer-2 tooling

New package:
- [v3/layer2/export_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/export_dataset.py)
- [v3/layer2/train_entry_side.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_entry_side.py)
- [v3/layer2/replay.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/replay.py)
- [v3/layer2/README.md](/Users/gduby/Documents/autoresearch-trading/v3/layer2/README.md)

What they do:
- export one row per eligible bar after both oracle passes
- join the W2a regime block from `v2.2`
- train simple tabular entry + side models
- calibrate thresholds on val folds only, now with explicit fixed-quantile support
- replay a one-trade-per-day policy against the post-A1 teacher baseline

New neural path:
- [v3/layer2/neural.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/neural.py)
- [v3/layer2/train_neural.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_neural.py)

What it does:
- keeps the same export bundle, replay interface, and artifact layout
- swaps the tree regressors for small torch MLP regressors
- supports `--latest-only` smoke tests and `--device cuda` for GPU training
- now also supports:
  - shared-encoder `--detach-side`
  - `route_aware_fallback` policy mode with detached call/put fallback heads
  - fallback-route diagnostics via [v3/analysis/layer2_fallback_route_diagnostic.py](/Users/gduby/Documents/autoresearch-trading/v3/analysis/layer2_fallback_route_diagnostic.py)
  - fallback-only randomization controls via [v3/analysis/layer2_random_direction_ablation.py](/Users/gduby/Documents/autoresearch-trading/v3/analysis/layer2_random_direction_ablation.py)

New unified-policy path:
- [v3/layer2/action_surface_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/action_surface_dataset.py)
- [v3/layer2/export_action_surface_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/export_action_surface_dataset.py)
- [v3/layer2/unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/unified_policy.py)
- [v3/layer2/train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py)

What it does:
- exports the canonical action-surface bundle:
  - `rows`
  - `sequence_features`
  - `sequence_mask`
  - `contract_features`
  - `contract_mask`
  - `action_labels`
- fixes the execution window at `09:45–11:30 ET`
- keeps Layer 0 as execution-only rails while exposing blocked contracts to training tokens
- trains one model over `flat + top-12 call tokens + top-12 put tokens`
- learns utility, clean-entry probability, and stopout risk jointly
- calibrates one scalar `decision_margin` on validation only inside the rolling harness

## Current Best Hypothesis

The surviving signal is still **bar-quality first, timing caution second, structural direction third**.

But the important architectural correction is:

- do not keep polishing the old decomposed Layer-2 as if it were the final shape
- use it as a benchmark and scaffolding layer
- move serious research energy into the unified action-policy path

More concretely:

- The entry model is learning something real. Top-decile and top-1/day bar selection still beat random eligible bars and teacher-triggered bars on oracle-quality metrics.
- The next bottleneck is no longer “can we score bars?” but “are we entering too greedily on bars that need too much patience?”
- The detach-side neural branch improved aggregate PF and DD, but the full diagnostic showed the directional edge is still not clean enough to promote as a standalone learned routing policy.
- The route-aware fallback follow-up did not fix the real problem. Instead of improving fallback action quality, it collapsed into teacher-only selections.
- The fallback-only probe also failed. Training only on non-teacher rows did not beat the blunt put fallback; even the narrower put-vs-flat control regressed versus always-put.
- The main criticism of the old flow is now explicit:
  - Layer-2 used a big bar-state dataset
  - later timing logic kept collapsing to tiny chosen-trade samples
  - that forced trader-like timing questions onto under-trained models

So the next useful hypothesis is not “run GPU.” It is:

- train timing / patience on the **full scored-bar surface**
- use Layer-2.5 to reject bars with poor early trade quality before final daily choice
- only then revisit Layer-3 exits on the cleaner trade set

## Commands

Dataset export:

```bash
.venv/bin/python -m v3.layer2.export_dataset
```

Canonical unified-policy dataset export:

```bash
.venv/bin/python -m v3.layer2.export_action_surface_dataset
```

Current best train run:

```bash
.venv/bin/python -m v3.layer2.train_entry_side \
  --run-dir v3/artifacts/layer2_entry_side_fixedq_60_10 \
  --entry-target entry_value_rank \
  --side-target time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_put \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 \
  --side-quantile 0.10 \
  --score-mode product
```

Replay against post-A1 teacher baseline:

```bash
.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_entry_side_fixedq_60_10
```

Train Layer-2.5 entry patience on the full scored surface:

```bash
.venv/bin/python -m v3.layer25.train_surface
```

Replay Layer-2.5 with the recommended threshold:

```bash
.venv/bin/python -m v3.layer25.replay

.venv/bin/python -m v3.layer3.train_rolling
```

Neural GPU-prep command:

```bash
.venv/bin/python -m v3.layer2.train_neural \
  --run-dir v3/artifacts/layer2_neural_gpu_fixedq \
  --entry-target entry_value_rank \
  --side-target time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_put \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 \
  --side-quantile 0.10 \
  --score-mode product \
  --device cuda
```

Unified-policy rolling smoke:

```bash
.venv/bin/python -m v3.layer2.train_unified_policy --tier smoke --device cpu
```

Unified-policy rolling promotion skeleton:

```bash
.venv/bin/python -m v3.layer2.train_unified_policy --tier promotion --device cuda
```

Route-aware fallback reproduction:

```bash
.venv/bin/python -m v3.layer2.train_neural \
  --run-dir v3/artifacts/layer2_route_fallback_q50 \
  --policy-mode route_aware_fallback \
  --entry-target entry_value_rank \
  --side-target time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_model \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 \
  --side-quantile 0.50 \
  --score-mode product \
  --device cpu

.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_route_fallback_q50

.venv/bin/python -m v3.analysis.layer2_fallback_route_diagnostic \
  --run-dir v3/artifacts/layer2_route_fallback_q50
```

## Immediate Next Work

### 1. Keep the right baselines separate

- Keep the fixed-quantile hybrid tree branch as the stable CPU reference.
- Keep the shared-encoder `--detach-side` run as the max-PF neural research baseline.
- Do not confuse “best aggregate replay artifact” with “cleared for promotion.”

### 2. Do not launch GPU from the route-aware branch

- The route-aware fallback experiment was fully implemented and failed:
  - all four fallback quantile runs tied
  - fallback threshold collapsed to `0.0`
  - the representative run selected `0` fallback trades
  - PF and DD both regressed versus the detach-side baseline
- So GPU is blocked by research, not by code readiness.

### 3. Next research target: Layer-2.5 timing / patience

- Do not keep teaching timing on tiny chosen-trade samples.
- Use the full scored-bar surface as the Layer-2.5 training universe.
- Good candidate questions:
  - which bars look directionally right but too early?
  - which bars need excessive patience before they work?
  - can Layer-2.5 improve trade quality without collapsing coverage too far?
- once Layer-2.5 is active, does Layer-3 retrained on cleaner trades generalize better?

### 3b. Next research target inside Layer-2 itself: unified action policy

- keep the action-surface export as the canonical training interface
- improve the unified model before running larger promotion budgets
- likely next tuning axes:
  - more training budget than the smoke run
  - better validation objective for `decision_margin`
  - stronger ranking loss weighting
  - GPU once the rolling path is ready for real budget
- do not reintroduce teacher direction overrides into this path

### 4. Layer-3 status: prototype, not production

- Exit modeling is no longer deferred in architecture; there is now an honest rolling Layer-3 prototype on top of Layer-2.5.
- Current caution:
  - the threshold sweep is still exploratory, not deployment-calibrated
  - early windows still fall back because there are not enough prior trades
  - slippage / execution realism still need to be re-run on this new stack
- Do not let fixed `-35/+60` stop-target logic back into Layer 0.

## Files To Read First

1. [v3/reference/orc_direction_fix_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/orc_direction_fix_2026_04_20.md)
2. [v3/reference/late_session_tournament_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/late_session_tournament_2026_04_20.md)
3. [v3/reference/w2a_handoff_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/w2a_handoff_2026_04_21.md)
4. [v3/reference/layer2_entry_side_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_entry_side_2026_04_21.md)
5. [v3/reference/layer2_diagnostic_full_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_diagnostic_full_2026_04_21.md)
6. [v3/reference/layer2_route_aware_fallback_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_route_aware_fallback_2026_04_21.md)
7. [v3/reference/layer2_fallback_only_probe_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_fallback_only_probe_2026_04_21.md)
8. [v3/layer2/README.md](/Users/gduby/Documents/autoresearch-trading/v3/layer2/README.md)

## Known Gotchas

- `v3/artifacts/layer2_dataset.pkl` is a derived artifact, not source of truth. Rebuild it if the logger/oracle/export logic changes.
- The current positive Layer-2 result is still **hybrid**. The learned side model alone has not beaten the teacher baseline.
- The detach-side neural artifact is the strongest replay result, but the route-aware follow-up showed the fallback problem is still unresolved.
- The fallback-only probe showed the unresolved piece is likely fallback regime gating, not fallback direction choice.
- The key improvement after the initial hybrid branch was **stable calibration**, not a new feature set or a more complex model.
- `replay.py --latest-only` is now available for one-fold smoke validation of neural runs.
- `v2` remains useful for sidecars and shared market-structure features, but it is no longer the mainline training loop.
