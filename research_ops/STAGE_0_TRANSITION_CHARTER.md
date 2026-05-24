# Stage 0 Transition Charter

Date: 2026-05-24

## Mission

Transition the project from improvisational trading-bot research into a
controlled research lab operated by AI agents.

The transition does not begin by reorganizing `v4/`. It begins by wrapping `v4/`
in a governance layer that controls what agents are allowed to do, what evidence
counts, and which decisions are blocked.

## Starting State

```text
GitHub repo
  v4/
    current Protocol101 paper stack
    model artifacts
    replay scripts
    paper runtime
    audit reports
    logs
    docs
```

Current operational truth:

- `PAPER_DEFAULT_PROTOCOL101` is the paper default.
- Challengers are research-only unless a decision packet changes that.
- v4 contains both strong governance and stale contradictions.
- Execution/fill realism, replay/live parity, lifecycle parity, and untouched
  evaluation are not closed.

## Target State

```text
GitHub repo
  v4/
    protected trading/research code and artifacts

  research_ops/
    CEO dashboard
    assumption registry
    decision queue
    experiment RFCs
    verifier reports
    decision memos
    iteration artifacts

  .github/
    issue templates
    pull request templates
    labels/workflows for research lab operation

  docs/
    current truth, stale docs, promotion rules, evidence ladder
```

Stage 0 creates the first `research_ops/` control surface. `.github/` workflow
and issue template changes are queued, not silently added as part of this first
scaffold.

## Stage 0 Scope

Allowed:

- Create governance docs and templates.
- Register known assumptions and blockers.
- Define agent output types.
- Define decision states and evidence standards.
- Reference existing v4 docs as binding truth.

Forbidden unless explicitly authorized later:

- Trading logic changes.
- Model retraining.
- Threshold tuning.
- Protected holdout scoring.
- Paid data downloads.
- Broker connections or paper-submit runs.
- Runtime flag changes.
- Launchd changes.
- Changing the paper default.

## Stage 0 Exit Criteria

Stage 0 is complete when:

- Every agent task can be classified into one of the six artifact types.
- Current `v4/` control truth is documented in `research_ops/`.
- The highest-risk assumptions are registered with falsification paths.
- Pending governance decisions are visible in a queue.
- Future `.github/` workflow/template work is scoped as a decision, not mixed
  with trading-code work.

## Current Stage 0 Decision

Decision: `stage0_governance_scaffold_created_control_unchanged`

Does this change paper default: no

Does this authorize model work: no

Does this authorize broker/data/runtime work: no
