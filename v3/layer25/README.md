# v3 Layer-2.5

This package promotes the entry-patience idea into the architecture.

The reason it exists is simple:

- Layer-2 already trains on a large bar-state dataset.
- Later timing logic kept collapsing that down to one chosen trade per day.
- That threw away most of the scored-bar surface and left trader-style timing
  layers under-trained.

Layer-2.5 fixes that by training a timing gate on the **full scored-bar
surface** from the rolling-window Layer-2 runs, then applying that gate before
the final one-trade-per-day choice.

## Current idea

Train a classifier on all scored bars with safe features only:

- rolling Layer-2 augmented features
- Layer-2 `entry_score`
- Layer-2 `side_conf`
- chosen direction flag

Default label:

- profitable at time-stop
- AND early MAE over the first 10 minutes stays above `-20%`

This is intentionally trader-like:

- reject bars that immediately take too much heat
- prefer bars where the move starts working without excessive patience

## Commands

Train the full-surface patience layer:

```bash
.venv/bin/python -m v3.layer25.train_surface
```

Replay with the recommended threshold:

```bash
.venv/bin/python -m v3.layer25.replay
```

Replay a specific threshold:

```bash
.venv/bin/python -m v3.layer25.replay --threshold 0.40
```

That replay now saves threshold-specific artifacts too:

- `layer25_trades_thr_<threshold>.csv`
- `layer25_replay_thr_<threshold>.json`

## Architectural role

The decision stack is now:

1. Layer 0 guardrails
2. Layer 1 teachers
3. Layer 2 entry + side scoring
4. **Layer 2.5 patience gate on the full scored-bar surface**
5. final `per_day_choice`
6. contract selection
7. Layer 3 exit
8. Layer 4 sizing

This is a deliberate criticism of the earlier method:

- We should not ask a tiny chosen-trade sample to learn all timing logic.
- The timing layer should learn from the full bar universe that Layer-2
  already scores.
- If a later layer can be trained on tens of thousands of bars instead of a
  few hundred trades, that is usually the right first move.

## Downstream Layer-3

Layer-2.5 is now also the input trade set for
[v3/layer3](/Users/gduby/Documents/autoresearch-trading/v3/layer3), which
trains honest rolling-window exits on the cleaner entry set instead of trying
to reuse the older 5-fold Layer-3 work.
