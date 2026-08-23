# Autoresearch Trading v5

Start here. This directory is the sole active workspace for deciding whether an automated SPX 0DTE
call/put bot can be justified by evidence. Older `v4` code and artifacts remain available as a frozen
toolbox and history, but they do not define what happens next.

## Five-minute orientation

- **Goal:** build an automated day-trading bot that buys SPX 0DTE calls or puts only after directional
  skill has been demonstrated on SPX/ES itself.
- **Canonical authority:** [STATUS.md](STATUS.md) is the only current-state page and job register. This
  README is orientation, never a second status ledger.
- **Current programme:** job 47 is safe local Phase 0 for the proposed “v6” programme, which remains a
  label inside v5 governance. Its [work packet](work/v6-program/PLAN.md) is built; Phase 1 is not
  authorised.
- **Research position:** G1 is blocked by effect size and causal identification, not the obsolete
  254-session constraint. No directional family is presently evidenced as runnable. Opening-range/gap
  remain unfalsified but cannot be rerun unchanged; signed option flow is a new primitive with a current
  `STOP` verdict.
- **Next permitted action:** repository and guard infrastructure plus the owner rulings in the job-47
  packet. Do not open outcomes, fit, tune, paper trade, or inspect forward economics.
- **Economic law:** **$17.92 / 0.358 ES points is an ES round-trip quantity, not a universal SPXW entry
  bar.** No entry “trader's bar” is signed; job 47 proposes one for owner decision and keeps economic
  usefulness separate from evidence strength.
- **Evidence state:** mutable counts and corpus verdicts live in [STATUS.md](STATUS.md). The current
  two-era SPXW corpus is `NOT-USABLE` for a new pooled economic claim, and every ES/SPXW session from
  2026-08-06 onward remains confirmation-only.
- **Closed work:** exact configurations and their genuinely-new conditions are in the
  [do-not-retest ledger](research/history/DO_NOT_RETEST.md). Underpowered is not falsified, but it is also
  not permission for an unchanged retry.
- **Guards:** the Outcome Run Gate, interior book-liveness gate, and CMBP touch-semantics gate exist and
  have contract tests. Job 47 records their incomplete end-to-end integration; green unit tests do not
  authorise outcomes.
- **Safety:** no training, threshold tuning, broker/vendor contact, holdout access, paper submission,
  promotion, or unattended-job changes without current owner authorization.

The authoritative job register, gate chain, and current facts are in [STATUS.md](STATUS.md).

## Where to go

| Need | Open |
|---|---|
| Current step, blockers, and gates | [STATUS.md](STATUS.md) |
| Rules for agents and safe work | [AGENTS.md](AGENTS.md) |
| Available tools and their safety class | [TOOLBOX.md](TOOLBOX.md) |
| Evidence supporting current claims | [evidence/INDEX.md](evidence/INDEX.md) |
| What has been built and what is still missing | [workflow capability audit](research/findings/V5_WORKFLOW_CAPABILITY_AUDIT_2026_08_05.md) |
| Current v6 programme/Phase 0 packet | [work/v6-program/PLAN.md](work/v6-program/PLAN.md) |
| Historical G1 direction plan | [history/jobs/g1-direction/PLAN.md](history/jobs/g1-direction/PLAN.md) |
| What must not be repeated | [research/history/DO_NOT_RETEST.md](research/history/DO_NOT_RETEST.md) |

## End-to-end order

The eleven-rung route, with what each rung produces and what authorizes it, is
[STATUS.md §2](STATUS.md#2-the-route-to-a-trading-bot). It is the only route document; this page does not
keep a second copy.

This page authorises no research rung. Signed job-specific instruments and the current register decide
what can start. Job 47 is infrastructure work; its Phase 1 remains blocked before outcomes. A future
candidate packet must still carry authoritative trades/provenance, causal controls, exact execution,
and serial-account evidence.

## Directory rule

There is deliberately no `v5/docs/` directory. Stable rules live at the v5 root or under `governance/`,
active work lives under `work/<registered-job>/`, durable research findings live under `research/`, and
large external evidence is catalogued under `evidence/`.

## Safe first check

```bash
./.venv/bin/python v5/ops/check_project.py
```

This reads files only. It does not train, download data, contact a broker, or change runtime state.
