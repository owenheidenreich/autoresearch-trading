# Handoff: Current v2 State

Read this file, then [program.md](program.md), then [current_state.md](docs/current_state.md).

## What Changed

The v4 exact-chain rebuild replaced the old ATM/OTM ladder with per-bar executable contract scoring. Every experiment from `exp_074` onward uses the exact-chain harness.

- Per-day sidecars store exact contract identities and per-contract forward P&L labels
- Replay selects one exact contract row from `contract_scores`; no direction head, no coarse strike offset, no learned risk head
- Training supervises per-contract P&L regression and contract selection
- Risk is policy-driven, not learned

## Canonical State

```text
Dataset path:        v2/data.pt
Per-day sidecars:    v2/data_sidecars/*.pt
Dataset version:     v4_exact_chain
Dataset fingerprint: 46f2d184e186496f
Unique days:         986
Features:            47
Trade window:        bar 30-270
ATM source:          dynamic_nearest_per_bar
```

## Current Research Position

- All five exact-chain experiments (exp_074 through exp_078) failed: zero trades or negative edge
- The next experiment is `exp_079`: side CE from `contract_scores`
- Recovery plan: side supervision first, then soft within-side ranking, then gate calibration
- See `lab_notebook.md` for the full diagnosis and hypothesis queue

## What To Trust

- `v2/data.pt` and `v2/data_sidecars/` as the canonical dataset
- `run_experiment_wf.py` as the canonical training runner
- `results.tsv` exact-chain entries (exp_074+) as the official scored runs
- Post-exact-chain replay scores only

## What Not To Trust

- Any pre-exact-chain score as a current baseline
- Any legacy artifact trained on a different dataset fingerprint
- Any doc in `archive/` — those describe prior regimes, not current truth

## Practical Operator Notes

- `./v2/ops/deploy.sh start` uploads the canonical dataset
- `./v2/ops/deploy.sh run_one exp_NNN` is the official 5-fold run
- `./v2/ops/deploy.sh run_screen exp_NNN` is the 1-fold screening run (no artifacts, no results.tsv)
- Keep/revert: `python v2/ops/model_manage.py keep` or `revert`

## Live Status

Live and paper trading code paths are stubs. This repo is a data, training, replay, and experiment-management system.
