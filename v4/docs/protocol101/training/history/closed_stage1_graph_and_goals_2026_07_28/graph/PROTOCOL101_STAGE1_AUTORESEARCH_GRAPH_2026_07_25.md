# Protocol101 Stage-1 Autoresearch Graph

Date: 2026-07-25

## Purpose

This graph replaces the manual prompt relay for the remainder of the frozen
Stage-1 entry search. It executes H1-H3 through RUN, GATE, and independent
AUDIT, then compares the accepted H0-H3 evidence.

The graph is an orchestrator, not a scientific optimizer. It cannot change
features, gates, folds, seeds, policies, thresholds, or model settings.

## Frozen Graph

Entrypoint:

```text
v4/scripts/run_protocol101_stage1_autoresearch_graph.py
```

Persistent control packet:

```text
v4/audit/autoresearch/protocol101_stage1_autoresearch_graph/
  graph_definition.json
  state.json
  events.jsonl
  logs/
  void_outputs/
```

Node order:

```text
S1-H1-RUN -> S1-H1-GATE -> S1-H1-AUDIT
          -> S1-H2-RUN -> S1-H2-GATE -> S1-H2-AUDIT
          -> S1-H3-RUN -> S1-H3-GATE -> S1-H3-AUDIT
          -> S1-ALL-SELECT
```

The frozen graph hash is recorded in `graph_definition.json`. Source changes
after initialization fail closed.

## Recovery

- Completed model units are reused only after unit identity, session
  membership, feature list, model path, and model hash are verified.
- The original preregistration is preserved during a resume.
- A completed RUN with invalid evidence is never replaced automatically.
- Partial or invalid GATE/AUDIT/SELECT packets are moved to `void_outputs/`
  before a bounded retry.
- Every node has at most three mechanical attempts.
- Scientific failures are accepted results. They are not retried.

Status:

```bash
~/.autoresearch-trading/runtime-venv/bin/python \
  -m v4.scripts.run_protocol101_stage1_autoresearch_graph --mode status
```

Resume:

```bash
~/.autoresearch-trading/runtime-venv/bin/python \
  -m v4.scripts.run_protocol101_stage1_autoresearch_graph \
  --mode run \
  --owner-approved-offline-training
```

## Stop Boundary

The graph stops after cross-hypothesis selection. It cannot run G9, open the
protected holdout, build or train learned exits, access recorder evidence,
call a broker, submit paper orders, change promotion/runtime/launchd state, or
touch real-money paths.

Under the owner-signed 2026-07-26 G8 revision, historical graph packets retain
their original G1-G8 fields, but current eligibility is recomputed from G1-G7
with G8 reported only.

If selection finds accepted entry signal but no eligible G1-G7 candidate, the
separately preregistered fixed-exit attribution must establish the cause.
Learned exits are not authorized merely because drawdown failed.
