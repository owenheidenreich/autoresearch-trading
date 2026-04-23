# Unified Policy Side-Contrastive Ablation + Layer-3 Follow-up — 2026-04-22

## Purpose

Follow up on the GPU promotion review with the two highest-value next
steps that do **not** muddy attribution:

1. add an explicit same-bar call-vs-put contrastive term to the unified
   policy objective
2. run rolling Layer-3 on top of the fixed unified-policy entries from
   the GPU promotion artifact

## Changes

### Unified-policy side contrastive term

Added a new `_side_contrastive_loss` to
[v3/layer2/unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/unified_policy.py)
and exposed it via
[v3/layer2/train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py)
as `--w-side-contrastive`.

The term compares the best tradeable call and best tradeable put on each
bar. When one side's true time-stop utility beats the other by at least
`$10`, the loss requires the predicted utility on the better side to
exceed the worse side by the configured hinge margin.

### Layer-3 unified-entry path

Extended [v3/layer3/common.py](/Users/gduby/Documents/autoresearch-trading/v3/layer3/common.py)
and [v3/layer3/train_rolling.py](/Users/gduby/Documents/autoresearch-trading/v3/layer3/train_rolling.py)
so Layer-3 can start from:

- `layer25` patience-gated trades, or
- `unified` policy chosen trades from a `.pkl` artifact

When the entry source is unified, Layer-3 now uses the actual chosen
strike/right from the entry artifact rather than falling back to the old
`select_contract(...)` heuristic.

## Commands Run

Side-contrastive rolling seed-42 ablation:

```bash
.venv/bin/python -m v3.layer2.train_unified_policy \
  --tier dev \
  --device cpu \
  --seed 42 \
  --run-dir v3/artifacts/layer2_unified_policy_side_contrastive_seed42 \
  --w-side-contrastive 0.5 \
  --max-epochs 8 \
  --patience 3
```

Layer-3 on top of the fixed GPU-promotion seed-42 unified entries:

```bash
.venv/bin/python -m v3.layer3.train_rolling \
  --entry-source unified \
  --chosen-trades v3/artifacts/v3_unified_promo_001/seed_42/chosen_trades.pkl \
  --out-dir v3/artifacts/layer3_unified_seed42
```

## Results

### 1. Side-contrastive ablation (`w_side_contrastive = 0.5`)

Reference unified GPU seed-42 baseline:

- trades: `352`
- PF: `1.151`
- DD: `29.9%`
- mean/trade: `+$54.3`
- call share: `93.5%`

Contrastive ablation result:

- trades: `438`
- PF: `1.002`
- DD: `59.7%`
- mean/trade: `+$0.8`
- call share: `90.9%`

Interpretation:

- The call-collapse concern was real: direct side pressure **did** pull
  call share down.
- But the first strong weight was too blunt. It increased activity,
  degraded margin quality, and nearly destroyed the edge.
- This is **not** evidence that side discrimination is a dead end. It is
  evidence that the first contrastive weight / scheduling is too harsh.

Artifact:

- [layer2_unified_policy_side_contrastive_seed42](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_unified_policy_side_contrastive_seed42)

### 2. Layer-3 on unified-policy entries

Unified seed-42 time-stop baseline:

- trades: `352`
- PF: `1.151`
- DD: `29.9%`
- mean/trade: `+$54.3`

Exploratory Layer-3 sweep on the **same entries**:

- threshold `0.15`: PF `2.011`, DD `22.3%`, mean `+$150.6`
- threshold `0.19`: PF `2.033`, DD `22.3%`, mean `+$183.4`
- threshold `0.20`: PF `1.939`, DD `22.3%`, mean `+$175.9`
- threshold `0.25`: PF `1.743`, DD `22.3%`, mean `+$162.5`
- threshold `0.30`: PF `1.632`, DD `22.3%`, mean `+$155.3`

Best exploratory threshold on the same OOS sweep:

- threshold `0.19`
- PF `2.033`
- DD `22.3%`
- mean `+$183.4`

Interpretation:

- This is the strongest evidence so far that the unified-policy branch is
  producing **clean enough entries to monetize with a learned exit**.
- The result is **exploratory**, not promotion-grade, because the exit
  threshold is still chosen on the same rolling OOS windows.
- Still, this is exactly the shape we hoped for after the GPU review:
  entry quality alone was not enough to beat V0 on PF, but entry quality
  plus a learned exit may have real headroom.

Artifact:

- [layer3_unified_seed42](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer3_unified_seed42)

## Conclusion

The repo should now carry these conclusions forward:

1. GPU promotion did **not** prove the unified policy is dead. It proved
   entry-only PF is short of the baseline while DD is materially better.
2. The side-prior issue is real, but the first naive contrastive fix is
   too aggressive.
3. The highest-EV next move remains the plan's Layer-3 composition / outer
   loop, because the first unified-entry Layer-3 replay produced the
   first meaningful PF lift on the same fixed entries.
