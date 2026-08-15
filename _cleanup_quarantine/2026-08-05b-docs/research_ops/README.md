# Research Ops

This directory is the governance layer around `v4/`.

It does not replace the trading code. It controls how AI-agent work is proposed,
verified, and accepted before any change reaches the current Protocol101 paper
stack.

## Current Stage

Stage 0: transition preparation.

Current control system:

```text
PAPER_DEFAULT_PROTOCOL101
```

Stage 0 principle:

```text
Freeze v4 as the current control. Build governance around it before changing
strategy logic, model logic, runtime behavior, or promotion rules.
```

## Allowed Stage 0 Outputs

Every agent session should produce exactly one primary artifact type:

| Output type | Purpose | Template |
|---|---|---|
| Cartography report | Map what exists and where truth lives. | `templates/cartography_report.md` |
| Experiment RFC | Propose a bounded experiment before results are known. | `templates/experiment_rfc.md` |
| Implementation patch | Change scoped non-runtime code or docs with explicit verification. | `templates/implementation_patch.md` |
| Verifier report | Independently check a claim, artifact, or patch. | `templates/verifier_report.md` |
| Decision memo | Accept, reject, freeze, block, or defer a proposal. | `templates/decision_memo.md` |
| Dashboard update | Update executive state without making research claims. | `templates/dashboard_update.md` |

If a session cannot fit one of these forms, the task is probably too vague.

## Binding Local Truth

Read these before changing direction:

- `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md`
- `docs/CURRENT_TRADING_BOT_IMPROVEMENT_QUESTIONS.md`
- `v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md`
- `v4/docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md`
- `v4/docs/NAMING_GUIDE.md`
- `v4/docs/PROJECT_SECTIONS_AND_HILL_CLIMB_GATES.md`
- `v4/docs/research_program_audit_2026_05_24.md`

## Non-Negotiables

- Do not change `PAPER_DEFAULT_PROTOCOL101` without a decision memo and the
  v4 promotion process.
- Do not train, retune thresholds, score protected holdouts, download paid data,
  call broker endpoints, mutate runtime flags, or edit launchd defaults unless
  the task explicitly authorizes that stage and risk.
- Treat replay profitability as a hypothesis until execution/fill realism,
  replay/live parity, lifecycle parity, and untouched validation are proven.
- Make stale-doc contradictions visible instead of silently resolving them in
  code.
- Prefer verifier reports over new experiments when assumptions are untested.

## Directory Map

| Path | Role |
|---|---|
| `STAGE_0_TRANSITION_CHARTER.md` | Defines the migration goal and Stage 0 boundaries. |
| `AI_AGENT_OPERATING_CONTRACT.md` | Rules for AI-agent sessions. |
| `CEO_DASHBOARD.md` | Human-readable operating snapshot. |
| `ASSUMPTION_REGISTRY.md` | Ranked assumptions and falsification status. |
| `DECISION_QUEUE.md` | Pending governance decisions. |
| `templates/` | Standard artifact templates. |

