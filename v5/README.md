# Autoresearch Trading v5

Start here. This directory is the sole active workspace for deciding whether an automated SPX 0DTE
call/put bot can be justified by evidence. Older `v4` code and artifacts remain available as a frozen
toolbox and history, but they do not define what happens next.

## Five-minute orientation

- **Goal:** build an automated day-trading bot that buys SPX 0DTE calls or puts only after directional
  skill has been demonstrated on SPX/ES itself.
- **Current blocker:** with 254 owned ES sessions, the project cannot reliably measure an edge as small as
  the measured trading cost, and no clean historical confirmation set has been identified.
- **Current job:** obtain the independent [measurement-capacity review](work/measurement-review/BRIEF.md).
- **Next permitted action:** review the measurement limits. Do not run the G1 direction screen until that
  review is recorded in [STATUS.md](STATUS.md).
- **Cost bar:** a strategy must beat **0.358 ES points, or $17.92 per round trip**, before it has economic
  value.
- **Closed work:** five research campaigns found no edge; long 0DTE options at minute cadence lost about
  $13 per trade before costs. The complete list is the [do-not-retest ledger](research/history/DO_NOT_RETEST.md).
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
| Active measurement review | [work/measurement-review/BRIEF.md](work/measurement-review/BRIEF.md) |
| Held G1 direction plan | [work/g1-direction/PLAN.md](work/g1-direction/PLAN.md) |
| What must not be repeated | [research/history/DO_NOT_RETEST.md](research/history/DO_NOT_RETEST.md) |

## Directory rule

There is deliberately no `v5/docs/` directory. Stable rules live at the v5 root or under `governance/`,
active work lives under `work/<registered-job>/`, durable research findings live under `research/`, and
large external evidence is catalogued under `evidence/`.

## Safe first check

```bash
./.venv/bin/python v5/ops/check_project.py
```

This reads files only. It does not train, download data, contact a broker, or change runtime state.
