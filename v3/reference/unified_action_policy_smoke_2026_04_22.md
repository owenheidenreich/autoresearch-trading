# Unified Action Policy Smoke — 2026-04-22

## Purpose

Record the first end-to-end implementation checkpoint for the new
audit-driven Layer-2 redesign:

- canonical action-surface dataset
- unified policy model over `flat + contract candidates`
- rolling validation-only `decision_margin` calibration
- replay artifact with chosen bar, side, strike, and patience outputs

This note is about **architecture verification**, not champion promotion.

## New Components

- [v3/layer2/action_surface_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/action_surface_dataset.py)
- [v3/layer2/export_action_surface_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/export_action_surface_dataset.py)
- [v3/layer2/unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/unified_policy.py)
- [v3/layer2/train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py)

## Canonical Dataset

Artifact:
- [v3/artifacts/layer2_action_surface_dataset.pkl](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_action_surface_dataset.pkl)

Shape:
- `89,692` rows across `986` days
- scalar features: `42`
- sequence features: `15`
- contract token features: `14`
- action labels: `13`
- sequence window: `20` bars
- contract tokens: `12` calls + `12` puts
- action count: `25` (`flat + 24 contract candidates`)

Execution window:
- `09:45–11:30 ET`

Design details:
- contract tokens come from contract-valid rows, not only passing rows
- Layer 0 guardrails remain execution-enforced
- blocked contracts are visible in token features via per-gate booleans
- replay only allows guardrail-passing actions

## Smoke Run

Artifact:
- [v3/artifacts/layer2_unified_policy_smoke_default](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_unified_policy_smoke_default)

Command:

```bash
.venv/bin/python -m v3.layer2.train_unified_policy \
  --run-dir v3/artifacts/layer2_unified_policy_smoke_default \
  --tier smoke \
  --device cpu
```

Window:
- latest rolling window only (`window 12`)
- train days: `860`
- val days: `40`
- OOS days: `60`

Result:
- trades: `46`
- trade share: `0.767`
- PF: `0.464`
- max DD: `50.4%`
- mean/trade: `-$263`
- calibrated `decision_margin`: `-0.08295`

Chosen-trade timing profile:
- clean winners: `11`
- shakeout winners: `2`
- fast losers: `15`
- drift losers: `18`

## Interpretation

What this proves:
- the new final-shape architecture is now real code, not just a plan
- the canonical action bundle exports correctly
- rolling training, validation-only calibration, and replay all run end-to-end
- replay artifacts now include chosen action id, side, strike, margin, and patience predictions

What this does **not** prove:
- the unified policy is ready
- the smoke result is competitive
- the current loss weighting or calibration objective is correct

So this should be treated as:
- **implemented**
- **structurally verified**
- **not promoted**

## Next Work

- improve the unified policy before spending promotion-level GPU budget
- tune ranking vs regression balance
- tighten the validation objective so trade share lands inside the target band
- compare against the Layer-2.5 baseline only on full rolling runs, not smoke
