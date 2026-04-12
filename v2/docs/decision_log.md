# Decision Log

This file records stable decisions that should outlive any one experiment loop. When a question is settled, it moves here from `open_questions.md`.

## Active Decisions

### 2026-04-11 — Direction Mix Is Diagnostic, Not A Hard Score Gate

- Decision: remove minority-direction balance from the official hard gates; keep call count, put count, minority share, and direction balance as required diagnostics on every run.
- Why: the side-collapse problem is real, but hard-gating on direction mix was hiding the deeper economic failure mode and incorrectly labeling some runs as protocol failures instead of model failures.
- Consequence: promotion still depends on economics and baseline beats; one-sided behavior must still be investigated through traces and side-bias audit outputs.
- Revisit when: a better economic model still shows pathological side concentration and a new penalty or policy is justified.

### 2026-04-11 — `exp_119` Is The Working Rebaseline Until A Better Official Run Exists

- Decision: reset `v2/train.py` to the `exp_119` family (`SOFT_TEMP=0.10`, `NOISE_MARGIN=0.01`, balanced gate BCE + soft KL) while keeping `exp_106` as the last official artifact.
- Why: `exp_119` is the strongest screening result that preserves direction balance and improves economics; `exp_121` ranking loss proved the failure mode survives a loss-family swap but was not good enough to keep.
- Consequence: the next official run starts from the restored `exp_119` base, not from the pairwise ranking branch.
- Revisit when: a new official run beats the current evidence base.

### 2026-04-11 — Side-Bias Audit Is A Standard Research-OS Diagnostic

- Decision: keep a permanent executable-snapshot side-bias audit that reports label-side structure, soft-target side mass, valid contract counts by side, model traded-side share, and side-conditioned accuracy/PnL.
- Why: the project repeatedly rediscovered side-collapse narratives without separating label structure from model dynamics.
- Consequence: future agents should use the audit before claiming the dataset itself is put-biased or that a new loss family solved the problem.
- Revisit when: the audit no longer provides distinct information beyond replay traces.

### 2026-04-11 — Kronos-Inspired Research May Cross Training, Replay, And Audit Surfaces

- Decision: the current Kronos-inspired block may modify `v2/train.py`, `v2/replay.py`, `v2/core/data_integrity.py`, and `v2/ops/pre_run_gate.py` when the hypothesis cannot be tested honestly inside the default two-file loop.
- Why: temporal-context and audit ideas can require coordinated training/inference or audit-gate wiring; forcing them into `train.py` alone created misleading half-implementations.
- Consequence: the expanded mutable surface is temporary and hypothesis-bound, not a general license to edit the frozen harness casually.
- Revisit when: the Kronos-inspired block ends or a new phase boundary is defined.

### 2026-04-11 — One Hypothesis May Span Coordinated Edits

- Decision: the operative unit is one hypothesis, not one file change.
- Why: some ideas are only falsifiable when a small set of coupled edits land together.
- Consequence: coordinated edits are allowed when inseparable, but unrelated ideas still must not be bundled together.
- Revisit when: the experiment protocol changes again.

### 2026-04-11 — Promotion Stays Score-Gated, Continuation Can Be Trace-Guided

- Decision: official promotion still requires beating the current best, beating all four baselines, and clearing hard gates; screening and follow-up decisions may continue based on trace-improved failure modes even when a run is not promotable.
- Why: representation-learning and data-quality work can produce real directional evidence before it clears the final score gate.
- Consequence: traces may justify continuing a hypothesis family, but they do not justify promoting a model artifact.
- Revisit when: the scoring contract or promotion policy changes.

### 2026-04-11 — Dataset Migration Requires An Explicit Audit Trigger

- Decision: raw-data and sidecar audits may run outside the live loop, but dataset rebuilds or version changes require an explicit trigger: multiple flagged sessions or overlap with the worst trace days.
- Why: the project needs a way to respond to real data defects without letting ad hoc rebuilds rewrite the evidence trail.
- Consequence: `v4_exact_chain` stays the default authority until a deliberate migration decision is made.
- Revisit when: an audit meets the trigger threshold.

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

- Decision: the live baseline family is gate BCE plus soft KL selection, with no direct PnL regression and no auxiliary side head.
- Why: the reset diagnostics supported simplifying the training objective before trying new architecture changes.
- Consequence: more complex heads and losses are hypotheses, not live defaults.
- Revisit when: a non-KL family produces better evidence than the restored `exp_119` baseline.

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
