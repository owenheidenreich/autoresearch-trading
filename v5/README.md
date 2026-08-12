# Autoresearch Trading v5

Start here. This directory is the sole active workspace for deciding whether an automated SPX 0DTE
call/put bot can be justified by evidence. Older `v4` code and artifacts remain available as a frozen
toolbox and history, but they do not define what happens next.

## Five-minute orientation

- **Goal:** build an automated day-trading bot that buys SPX 0DTE calls or puts only after directional
  skill has been demonstrated on SPX/ES itself.
- **Current blocker:** with 254 owned ES sessions, the project cannot reliably measure an edge as small as
  the measured trading cost, and no clean historical confirmation set has been identified.
- **Current job:** obtain the independent [measurement-capacity review](history/jobs/measurement-review/BRIEF.md).
- **Next permitted action:** review the measurement limits. Do not run the G1 direction screen until that
  review is recorded in [STATUS.md](STATUS.md).
- **Cost bar:** a strategy must beat **0.358 ES points, or $17.92 per round trip**, before it has economic
  value.
- **Data state:** the protected external corpus contains 254 usable ES sessions and 251 option sessions.
  One session proved exact historical/live value identity for 914 OPRA CBBO-1m rows; multi-session
  latency and full-feature parity remain unproven.
- **Closed work:** five research campaigns found no edge; long 0DTE options at minute cadence lost about
  $13 per trade before costs. The complete list is the [do-not-retest ledger](research/history/DO_NOT_RETEST.md).
- **Tools:** v5 has local-only clock/parity, feature-admission, prior-art, project-check, and validation
  packet interfaces. The missing prerequisite for training is evidence and authorization, not software.
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
| Active measurement review | [history/jobs/measurement-review/BRIEF.md](history/jobs/measurement-review/BRIEF.md) |
| Held G1 direction plan | [history/jobs/g1-direction/PLAN.md](history/jobs/g1-direction/PLAN.md) |
| What must not be repeated | [research/history/DO_NOT_RETEST.md](research/history/DO_NOT_RETEST.md) |

## End-to-end order

The eleven-rung route, with what each rung produces and what authorizes it, is
[STATUS.md §2](STATUS.md#2-the-route-to-a-trading-bot). It is the only route document; this page does not
keep a second copy.

Only rung 1 is currently authorized. A future candidate's packet will contain an authoritative
`trades.csv` and provenance manifest plus two diagnostic views: all entries and exits on SPX, and the net
equity curve.

## Directory rule

There is deliberately no `v5/docs/` directory. Stable rules live at the v5 root or under `governance/`,
active work lives under `work/<registered-job>/`, durable research findings live under `research/`, and
large external evidence is catalogued under `evidence/`.

## Safe first check

```bash
./.venv/bin/python v5/ops/check_project.py
```

This reads files only. It does not train, download data, contact a broker, or change runtime state.
