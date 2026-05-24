# AI Agent Operating Contract

This contract governs AI-agent work on the project.

## Prime Directive

Each session must produce one primary artifact:

- cartography report,
- experiment RFC,
- implementation patch,
- verifier report,
- decision memo,
- dashboard update.

Do not produce random improvements.

## Before Acting

Every agent must identify:

```text
Primary artifact type:
System section:
Allowed mutation scope:
Forbidden actions:
Evidence sources:
Verification plan:
```

If the task involves model quality, strategy logic, replay claims, live paper
behavior, paid data, or broker interaction, the agent must explicitly state
whether it is operating under v4 Section 1, 2, 3, 4, or 5 from
`v4/docs/PROJECT_SECTIONS_AND_HILL_CLIMB_GATES.md`.

## Safety Defaults

Default forbidden actions:

- training,
- threshold tuning,
- protected holdout scoring,
- paid data downloads,
- broker endpoint calls,
- paper-submit sessions,
- runtime flag mutation,
- launchd mutation,
- model artifact replacement,
- changing `PAPER_DEFAULT_PROTOCOL101`.

An agent may perform one of those actions only when the user explicitly asks for
that action and the relevant governance gate allows it.

## Evidence Standard

An agent claim must be grounded in at least one of:

- repository file/function/artifact path,
- paper/live log path,
- replay artifact path,
- documented v4 governance rule,
- external primary source when the claim is about market structure, broker
  behavior, or research methodology.

Unsupported claims should be written as assumptions and added to
`ASSUMPTION_REGISTRY.md`.

## Handoff Standard

Every final response should include:

```text
Artifact produced:
Files changed:
Verification performed:
Open decisions:
Blocked actions:
```

For small tasks, this may be one short paragraph.

## Conflict Rule

If code and docs disagree, operational code/logs win for current behavior, but
the contradiction must be recorded. Do not silently clean up a contradiction
while doing unrelated work.

## Control Rule

`PAPER_DEFAULT_PROTOCOL101` remains the control until a formal decision memo and
v4 promotion packet say otherwise.
