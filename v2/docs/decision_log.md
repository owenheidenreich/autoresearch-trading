# Decision Log

This file records stable decisions that should outlive any one experiment loop. When a question is settled, it moves here from `open_questions.md`.

## Active Decisions

### 2026-04-10 — Exact-Chain Reset Is The Live Research Baseline

- Decision: the frozen live harness is the v4 exact-chain system based on `v2/data.pt` plus `v2/data_sidecars/`.
- Why: pre-reset and pre-exact-chain evidence had too many contradictions to remain the live baseline.
- Consequence: pre-exact-chain material stays preserved, but it no longer counts as current evidence.
- Revisit when: there is an intentional dataset-version migration with explicit validation and documentation.

### 2026-04-10 — Official Evidence And Screening Notes Are Separate

- Decision: `v2/results.tsv` holds official exact-chain scored runs only, while `v2/lab_notebook.md` holds screening notes, diagnostics, and hypothesis work.
- Why: mixing official evidence with screening notes made the repo easy to misread.
- Consequence: screening rows do not belong in `results.tsv`.
- Revisit when: the experiment protocol itself changes.

### 2026-04-10 — Restored Baseline Uses Gate BCE Plus Soft KL Only

- Decision: the live baseline is the `exp_080`-style setup: gate BCE plus soft KL selection only, `SOFT_TEMP=0.20`, no direct PnL regression, no auxiliary side head.
- Why: the reset diagnostics supported simplifying the training objective before trying new architecture changes.
- Consequence: more complex heads and losses are hypotheses, not live defaults.
- Revisit when: a new official experiment beats the current baseline under the exact-chain protocol.

### 2026-04-10 — Non-Live Surfaces Stay Archived

- Decision: live/paper trading stubs, one-off repair diagnostics, legacy dataset builders, and redundant docs stay out of the live `v2/` path.
- Why: they created context drift and made the current system harder to understand.
- Consequence: historical surfaces are preserved under `archive/v2_historical/`, not deleted.
- Revisit when: a historical surface becomes part of the active exact-chain workflow again.

### 2026-04-10 — Local Runtime Files Have Dedicated Homes

- Decision: local checkpoints live under `v2/models/`, runtime state lives under `v2/state/`, and generated outputs live under `v2/output/`.
- Why: a flatter root made the project harder to scan and easier to misunderstand.
- Consequence: live code and docs should use the organized paths, not the old root-level file locations.
- Revisit when: the runtime surface needs a more formal packaging boundary.

### 2026-04-10 — Founder Judgment Has A Permanent Home

- Decision: `v2/docs/founder_intent.md`, `v2/docs/decision_log.md`, `v2/docs/open_questions.md`, and `python3 -m v2.ops.status_report` are now part of the live operating system.
- Why: the project needed a durable place for human intent, stable decisions, unresolved questions, and current health.
- Consequence: these files are live documentation, not optional notes.
- Revisit when: the project graduates into a materially different phase with different operator needs.

### 2026-04-10 — Artifact History Must Not Rewrite Official Results

- Decision: monitor and tooling may read artifact history for operator context, but they must not reconstruct or append rows into `v2/results.tsv`.
- Why: official evidence was getting silently polluted by artifact-only history, which made the live research record untrustworthy.
- Consequence: `v2/results.tsv` stays an operator-maintained log of official exact-chain runs only.
- Revisit when: the project intentionally adopts a new canonical experiment ledger with explicit migration rules.

### 2026-04-10 — `exp_090` Starts With A No-Change Baseline Re-Screen

- Decision: the first experiment after the reset is `exp_090`, a pure re-screen of the restored baseline with no `train.py` or `core/policy.py` change.
- Why: the repo needs one trustworthy post-reset reference point before new hypotheses start changing the model again.
- Consequence: `exp_090` is infrastructure confirmation as much as model screening; any result becomes the clean starting point for later experiments.
- Revisit when: `exp_090` completes and the next hypothesis is selected.
