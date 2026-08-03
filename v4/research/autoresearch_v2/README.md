# autoresearch_v2

`autoresearch_v2` is the development-only experiment compiler that replaces
Path-D as the hypothesis-screening orchestrator. Claude/Codex may propose typed
JSON hypotheses; the engine owns causality linting, OOF fits, threshold-free
prediction caching, paired component screens, family maxT correction, strict
serial replay, power routing, and terminal status assignment.

It deliberately reuses the verified Protocol101 causal data and simulator-v5
machinery. It cannot open the protected holdout, change the paper default,
contact a broker, or promote a model.

Initialize once (the foundation file cannot be overwritten):

```bash
PYTHONPATH=. .venv/bin/python -m v4.research.autoresearch_v2.runner initialize \
  --foundation <development-foundation.json> \
  --hypotheses <hypothesis-directory>
```

Run the compiled development suite:

```bash
PYTHONPATH=. .venv/bin/python -m v4.research.autoresearch_v2.runner run \
  --foundation <development-foundation.json> \
  --hypotheses <hypothesis-directory> \
  --output <audit-directory> \
  --cache <shared-oof-cache-directory> \
  --registry <global-semantic-registry.jsonl>
```

Run the exact policy-neutral M0/M1 path-target experiment through the same
compiler, foundation verification, caches, and semantic registry:

```bash
PYTHONPATH=. .venv/bin/python -m v4.research.autoresearch_v2.path_experiment \
  --foundation <development-foundation.json> \
  --hypothesis v4/research/autoresearch_v2/hypotheses/entry_policy_neutral_m0_m1_full_epoch_v1.json \
  --output <audit-directory> \
  --risk-cache <shared-risk-set-cache-directory> \
  --prediction-cache <shared-oof-cache-directory> \
  --registry <global-semantic-registry.jsonl>
```

The global registry rejects repeated mechanics even when the hypothesis is
renamed. A deliberate reproduction after an engine fix must pass
`--allow-semantic-rerun`; the new row keeps the same semantic hash and records
the prior result path/status, so the idea cannot masquerade as new.

Every experiment terminates as one of `INVALID_EXPERIMENT`,
`MECHANICAL_FAILURE`, `UNDERPOWERED`, `NO_INCREMENTAL_EDGE`, `EXIT_ARTIFACT`,
`PROVISIONAL_EDGE`, or `CONFIRMED_EDGE`. Development runs can never emit
`CONFIRMED_EDGE`; that status requires a separately preregistered one-shot fresh
research epoch.

## Quarantined former confirmation

The 2026-08-02 corrected-v3.2 campaign originally emitted one `CONFIRMED_EDGE`
entry policy after a separately powered 29-session one-shot confirmation. The
subsequent clock audit found that all 445,063 fitted rows violated the declared
completed-minute context clock, so the experiment is now classified
`INVALID_EXPERIMENT` and the model is quarantined. The immutable evidence and
claim boundary are recorded in
`v4/docs/protocol101/training/research/AUTORESEARCH_V2_CONFIRMED_ENTRY_MODEL_2026_08_02.md`.
The confirmation set is spent and cannot be reused. The artifact is research
evidence only. Its 2026-08-03 offline runtime decision-parity gate failed: the
training context consumed the ThetaData SPX bar at its bar-open timestamp,
while the live twin can consume its close only 60 seconds later. The exact
raw-source adapter reproduced the training path at 100% under the diagnostic
training clock, isolating the clock mismatch. The frozen model must not proceed
to live shadow or paper promotion; a causal-clock replacement would be a
distinct research generation.

The model-free shared source selector for that future generation is
`live_opra_training_twin.py`. It aligns native OPRA `cbbo-1m` at interval end
`t` with the official SPX bar stamped `t-60s`, then starts entry execution and
the exit horizon from a fresh `cbbo-1s` arrival quote. Its emission lag remains
unfrozen pending a live ThetaData receipt-timing sample, so it does not
authorize fitting.

The live-first entry feature catalog is `entry_live_feature_catalog.py`. Every
new feature must bind to an exact executable contract ID, clock, missing/carry
law, and fit-readiness state. Prose claims such as "live option ladder" are
rejected. The catalog and expanded OPRA schema findings are documented in
`v4/docs/protocol101/training/research/AUTORESEARCH_V2_LIVE_FIRST_ENTRY_FEATURE_AUDIT_2026_08_03.md`.
