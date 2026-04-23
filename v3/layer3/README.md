# v3 Layer-3

This package is the honest rolling-window exit workstream for the new
`v3` stack.

The old Layer-3 work on April 21 showed real promise, but it was tied to the
older 5-fold world and cannot be applied cleanly to the rolling-window
champion without leakage.

This package fixes that methodological gap:

- start from a fixed rolling-window entry policy
- train exits only on prior rolling windows
- replay on the held-out current window
- compare against the same entry-policy trades held to time-stop

## Current role

The stack is now:

1. Layer 0 guardrails
2. Layer 1 teachers
3. Layer 2 entry + side scoring
4. Layer 2.5 patience gate or unified-policy daily choice
5. final executable contract choice
6. Layer 3 rolling learned exit
7. Layer 4 sizing

## Important caveat

This package keeps the evaluation honest, but it does **not** yet solve
threshold calibration. The threshold sweep is exploratory. If one threshold
looks best on these same OOS windows, that is a research clue, not a deploy
setting.

## Commands

Train and replay the rolling Layer-3 sweep on top of the recommended
Layer-2.5 threshold:

```bash
.venv/bin/python -m v3.layer3.train_rolling
```

Run on a specific Layer-2.5 threshold:

```bash
.venv/bin/python -m v3.layer3.train_rolling --patience-threshold 0.40
```

Run a custom Layer-3 threshold band:

```bash
.venv/bin/python -m v3.layer3.train_rolling --exit-thresholds 0.15 0.19 0.20 0.25 0.30
```

Run the same rolling Layer-3 sweep on top of unified-policy chosen trades:

```bash
.venv/bin/python -m v3.layer3.train_rolling \
  --entry-source unified \
  --chosen-trades v3/artifacts/v3_unified_promo_001/seed_42/chosen_trades.pkl \
  --out-dir v3/artifacts/layer3_unified_seed42
```
