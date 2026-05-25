# Research Ops

This directory controls research governance around the existing `v4/` trading
system.

It does not replace trading code. It defines the operating law for AI-agent
research sessions so work becomes auditable, reversible, and evidence-driven.

## Binding Operating Law

No model, threshold, runtime, launchd, paper-order, broker, or paid-data change
is allowed unless an iteration contains:

1. Cartography report
2. Experiment RFC
3. Implementation summary
4. Verifier report
5. Decision memo
6. CEO dashboard update

This is how the project avoids "Codex changed something and now I need to
remember why."

## Current Control

The frozen control is:

```text
v4-protocol101-control-2026-05-24
```

The current operational default remains:

```text
PAPER_DEFAULT_PROTOCOL101
```

Research challengers are research-only until a decision memo and promotion gate
explicitly say otherwise.

## Prohibited By Default

The following are blocked unless the user explicitly authorizes the specific
risk in the current task:

- Modifying v4 trading logic.
- Modifying Protocol101, Protocol051, Protocol066, Protocol081, model artifacts,
  or scalers.
- Modifying runtime flags, launchd, or IBKR paper execution behavior.
- Calling broker APIs.
- Downloading paid data.
- Training models.
- Tuning thresholds.
- Promoting challengers.
- Scoring protected holdouts outside an approved validation plan.

## Required Artifact Flow

Every material research or engineering change should move through this packet:

| Step | Artifact | Purpose |
|---:|---|---|
| 1 | `templates/cartography_report.md` | Map code, data, docs, and operational truth before action. |
| 2 | `templates/experiment_rfc.md` | State the hypothesis, assumptions, falsifiers, and plan before results. |
| 3 | `templates/implementation_summary.md` | Record what changed and how behavior was protected. |
| 4 | `templates/verifier_report.md` | Independently test claims and check failure modes. |
| 5 | `templates/decision_memo.md` | Accept, reject, defer, freeze, or escalate. |
| 6 | `templates/ceo_packet.md` | Update the dashboard-level truth. |

Single-session cartography, verification, or documentation work may produce only
the relevant artifact, but it must state why the full packet is not required.

## Iteration Workflow

Create a new research cycle with:

```text
python research_ops/scripts/new_iteration.py --id ITER-001_quote_age_truth --assumption A001 --title "Quote age truth"
```

Each iteration lives under `research_ops/iterations/` and contains a request,
cartography report, RFC, implementation summary, verifier report, decision
memo, `manifest.yaml`, and `artifacts/`. Validate it with:

```text
python research_ops/scripts/validate_iteration.py ITER-001_quote_age_truth --registry
```

Update the control-tower dashboard with:

```text
python research_ops/scripts/update_dashboard.py
```

## Binding Local Truth

Read these before changing direction:

- `research_ops/CURRENT_STATE.yaml`
- `research_ops/CEO_DASHBOARD.md`
- `research_ops/ASSUMPTION_REGISTRY.csv`
- `research_ops/BRANCH_AND_PR_CONVENTIONS.md`
- `research_ops/DECISION_QUEUE.md`
- `research_ops/bootstrap/V4_BASELINE_INVENTORY.md`
- `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md`
- `docs/CURRENT_TRADING_BOT_IMPROVEMENT_QUESTIONS.md`
- `v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md`
- `v4/docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md`
- `v4/docs/NAMING_GUIDE.md`
- `v4/docs/PROJECT_SECTIONS_AND_HILL_CLIMB_GATES.md`
- `v4/docs/research_program_audit_2026_05_24.md`

## Directory Map

| Path | Role |
|---|---|
| `CURRENT_STATE.yaml` | Machine-readable governance state. |
| `CEO_DASHBOARD.md` | Human-readable operating snapshot. |
| `ASSUMPTION_REGISTRY.csv` | Falsification-oriented assumption tracker. |
| `BRANCH_AND_PR_CONVENTIONS.md` | Git branch and PR naming rules tied to iterations. |
| `DECISION_QUEUE.md` | Pending governance decisions. |
| `ROADMAP.md` | Prioritized research-ops roadmap. |
| `DO_NOT_TOUCH_WITHOUT_APPROVAL.md` | Explicit protected surfaces. |
| `EVIDENCE_LADDER.md` | Evidence standards from weak to promotion-grade. |
| `PROMOTION_GATE.md` | Minimum gates before changing operational defaults. |
| `GLOSSARY.md` | Shared terminology for agents and humans. |
| `bootstrap/` | Frozen-control cartography. |
| `templates/` | Standard artifact templates. |
| `prompts/` | Role-specific prompts for non-overlapping AI-agent sessions. |
| `iterations/` | Per-iteration packets and artifacts. |
| `schemas/` | Lightweight JSON schemas for local validation. |
| `scripts/` | Local file-management utilities only. |

## Script Boundary

Scripts under `research_ops/scripts/` may create, validate, and summarize local
iteration files. They must not import `v4`, broker clients, trading runtimes,
model code, or paid-data code.
