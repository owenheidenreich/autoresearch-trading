# Simulated-L3 Oracle 50/50 Mixture — Falsified — 2026-04-23

## Hypothesis

Codex's `candidate_l3_oracle_retrain_2026_04_23.md` suggested testing
"an ensemble/mixture target between chosen-trained-L3 oracle and
candidate-trained-L3 oracle rather than fully replacing the exit
target." The prior cycle (diagnostic) localized why a mixture was a
plausible rescue candidate:

- Seed 43 with candidate-L3 oracle: 107 extra trades (diluted PF)
- Seed 44 with candidate-L3 oracle: W8 **misses winners** (−$11.8k)
  + W5 **admits losers** (−$7.1k) — opposite failure modes

Chosen-L3 oracle is heavily call-biased and conservative; candidate-L3
oracle is side-diverse and admits more entries. A 50/50 blend might
split the difference and partially restore the floor while preserving
some side diversity.

## Implementation

New module [v3/layer2/build_mixture_oracle.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/build_mixture_oracle.py).
Per-cell:

```
mix_pnl[row, action] = alpha * chosenL3_pnl[row, action]
                     + (1 - alpha) * candidateL3_pnl[row, action]
```

Built three per-seed mixture oracles at `alpha = 0.5`:

```
v3/artifacts/simulated_l3_oracle_seed{42,43,44}_mix50_fp.npz
```

All three cover the same 780,844 simulated cells as the parents
(cells where both simulated, 0 only-A, 0 only-B).

## Commands

```bash
for s in 42 43 44; do
  .venv/bin/python -m v3.layer2.build_mixture_oracle \
    --oracle-a v3/artifacts/simulated_l3_oracle_seed${s}_fp.npz \
    --oracle-b v3/artifacts/simulated_l3_oracle_seed${s}_candidate_l3_mpd4_fp.npz \
    --alpha 0.5 \
    --output v3/artifacts/simulated_l3_oracle_seed${s}_mix50_fp.npz

  .venv/bin/python -m v3.layer2.train_unified_policy \
    --tier dev --device cpu --seed $s \
    --run-dir v3/artifacts/layer2_unified_policy_simL3_mix50_seed${s} \
    --utility-target simulated_l3 \
    --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_seed${s}_mix50_fp.npz

  .venv/bin/python -m v3.layer3.train_rolling \
    --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_simL3_mix50_seed${s}/seed_${s}/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_simL3_mix50_seed${s} \
    --seed $s

  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_simL3_mix50_seed${s} \
    --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90
done
```

Composition uses the **chosen-trained** Layer-3 exit (matching the
current promoted champion), not candidate-trained L3. This isolates
the oracle-blend variable.

## Results

Per-seed calibrated Layer-3 robust 0.90:

| seed | PF | DD | trades | $/trade | entry-only PF |
|---|---:|---:|---:|---:|---:|
| 42 | 1.748 | 11.4% | 277 | +$136.1 | 1.119 |
| 43 | **1.856** | 9.1% | 313 | +$141.9 | 0.964 |
| 44 | **1.393** | 13.6% | 325 | +$75.5 | 0.975 |
| mean | **1.666** | 11.4% | 305 | +$117.8 | — |
| min | 1.393 | — | — | — | — |
| max DD | — | 13.6% | — | — | — |

Aggregated across all 915 calibrated trades: PF `1.648`, mean/trade
`+$116.6`.

## Comparison

| stack | mean PF | min PF | mean DD | agg PF | trades |
|---|---:|---:|---:|---:|---:|
| prior champ (w=0.00 + chosenL3) | 1.783 | 1.711 | 22.0% | 1.782 | 1201 |
| **per-seed chosenL3 oracle (champ)** | **1.950** | **1.786** | **10.1%** | **1.976** | 809 |
| candidate-L3 oracle retrain | 2.060 | 1.635 | 12.8% | 1.960 | 982 |
| **mix50 oracle retrain** | **1.666** | **1.393** | **11.4%** | **1.648** | **915** |

**mix50 loses on every primary metric** vs the champion:
- mean PF: -0.284
- min PF: -0.393
- aggregate PF: -0.328
- DD improvement: only -1.3pp (insignificant given PF collapse)

It also loses to the candidate-L3 oracle retrain on every metric
except min PF (where it is even lower).

## Per-seed behavior

| seed | chosenL3 (parent A) | candidateL3 (parent B) | mix50 (blend) |
|---|---:|---:|---:|
| 42 | 1.786 | **2.854** | 1.748 (worst of the three) |
| 43 | 1.802 | 1.635 | **1.856** (beats both parents) |
| 44 | **2.264** | 1.691 | 1.393 (catastrophic) |

Seed 43 **is** rescued (+0.054 over chosenL3 parent, +0.221 over
candidateL3 parent) — the diagnostic prediction held for that seed.
But seeds 42 and 44 land below both parents. The W8 problem on seed
44 (champion caught 38 winners, candidateL3 missed to 12) got
**worse** under mix50 — training against an averaged signal teaches
the policy neither parent's strategy coherently.

## Falsification

A linear 50/50 blend of the two oracles is not a productive
supervision signal. The two oracles encode **incompatible entry
distributions**: chosenL3's prior is "heavily call-concentrated, few
entries, catch W8 winners"; candidateL3's prior is "side-diverse,
more entries, side-diversity". Averaging produces a signal that
trains the policy to an indecisive middle that catches fewer
winners without gaining either parent's strength.

Stop condition from prior addendum ("clearly regresses the PF
floor"): min PF drops from `1.786 → 1.393` (−0.393). Stop.

## Belief change

- **Simple scalar oracle blending (α ∈ [0,1]) is not a productive
  outer-loop knob on this stack.** The oracles do not combine
  linearly. Any future mixture work would need to be conditional —
  per-window or per-bar routing that picks one oracle's signal at
  a time.
- **Seed 43 did respond to the mixture as predicted** — floor
  recovered (+0.054). That suggests the diagnostic read of seed 43
  ("dilution by extra trades") was correct and is fixable by a
  gentler supervision signal. This seed-specific recovery does not
  justify a global alpha, but it narrows the problem: mixture
  helps on seeds where the regression is gradient-like
  (more trades) and hurts on seeds where the regression is
  categorical (miss-class of winners in W8).
- **Champion is unchanged.** The per-seed chosenL3 oracle stack
  remains provisional champion: mean PF `1.950`, min `1.786`,
  mean DD `10.1%`.

## Next-cycle candidates (not this loop)

- **Per-window oracle routing**: pick chosenL3 vs candidateL3 per
  rolling window based on prior-window composed PF. Heuristic that
  respects the champion's exit behavior when chosenL3 did well in
  priors.
- **Gated oracle** at training time: small learned gate that chooses
  per-bar which oracle's PnL to supervise against. More expensive,
  matches the diagnostic's "different windows fail differently"
  evidence.
- **Seed-44-specific feature debug**: W8 was a huge win under
  champion but a miss under both candidateL3 and mix50. If we can
  identify what feature distinguishes W8's winning days for the
  champion, we might be able to add it to the unified policy
  directly without oracle work.

None of these are in scope for this cycle; documenting as future
candidates.

## Artifacts

- `v3/layer2/build_mixture_oracle.py` (new module, ~85 lines)
- `v3/artifacts/simulated_l3_oracle_seed{42,43,44}_mix50_fp.npz`
- `v3/artifacts/layer2_unified_policy_simL3_mix50_seed{42,43,44}/`
- `v3/artifacts/layer3_unified_cpu_simL3_mix50_seed{42,43,44}/`
