# ART2

ART2 is an autoresearch-inspired system for developing and validating SPX 0DTE options models inside a frozen exact-chain historical environment.

The current mission is not live trading. The current mission is to build a trustworthy exact-chain research system.

## What This Project Is

This repo is a disciplined research loop for SPX 0DTE long-options trading ideas:

- freeze a known dataset and replay harness
- make one small training change at a time
- train from scratch
- evaluate under fixed replay rules
- keep or revert based on score
- write down the result so the repo, not the model, carries the research memory

The project is inspired by Karpathy's autoresearch pattern, but it is intentionally tighter and more gated than a generic autonomous loop.

ART2 is not "let the AI roam and improve itself."

ART2 is:

- autoresearch-inspired
- experiment-driven
- replay-constrained
- documentation-backed
- operator-auditable

## What This Project Is Not

- not a live trading bot
- not a broker execution system
- not a place to trust stale scores or historical shortcuts
- not a system that relies on model memory instead of written research memory

Live trading may come later if the research system becomes trustworthy enough. That is not the current phase.

## Design In One View

```text
clean data + exact-chain sidecars
    -> frozen dataset
    -> one-hypothesis training change
    -> train from scratch
    -> replay evaluation
    -> keep or revert by score
    -> log the outcome
    -> update durable project memory
```

## Trust Boundaries

- The live experiment surface is intentionally small.
- The evaluation harness is treated as immutable during normal research.
- Official evidence lives in `v2/results.tsv`.
- Screening notes and diagnostics live in `v2/lab_notebook.md`.
- Historical material is preserved under `archive/`, but it is not part of the live default context.

## How To Understand The Project

Start here:

1. [v2/HANDOFF.md](v2/HANDOFF.md)
2. [v2/docs/founder_intent.md](v2/docs/founder_intent.md)
3. [v2/program.md](v2/program.md)
4. [v2/docs/README.md](v2/docs/README.md)

Then run:

```bash
python3 -m v2.ops.status_report
```

That command is the fastest way to see whether the repo is healthy, what phase the project is in, which dataset is live, what the next experiment is, and what blockers still exist.

## Core Research Loop

1. Verify the repo is healthy and the live docs agree.
2. Form one hypothesis.
3. Change only the allowed training surface.
4. Run the pre-GPU integrity gate.
5. Train on the remote GPU from scratch.
6. Replay and score the model under the frozen exact-chain harness.
7. Keep or revert based on official score and baseline rules.
8. Log the result and update the research memory.

## Repo Structure

```text
v2/
  HANDOFF.md              current state and trust boundaries
  program.md              definitive operating protocol
  COMMANDS.md             supported operator commands
  train.py                live mutable training surface
  core/policy.py          live mutable policy surface
  docs/                   live documentation and founder memory
  ops/                    gates, deployment, monitoring, status reporting
  pipeline/               live data-building pipeline
  analysis/               live evaluation helpers
  models/                 local checkpoint storage
  state/                  local runtime state
  output/                 generated local outputs
  artifacts/              experiment artifacts

archive/
  historical material preserved outside the live path
```

## Professional Framing

If you need a one-paragraph description of the project, use this:

> ART2 is an autoresearch-inspired research system for SPX 0DTE options. It combines curated data, a fixed exact-chain replay harness, disciplined experiment management, and explicit research memory so model progress can be measured honestly and reproducibly. The current mission is not live trading; it is building a trustworthy exact-chain research system.

## Current Standard

If the code, docs, artifacts, and results disagree, that disagreement is a bug.

If a result cannot be defended from current data, current code, and current artifacts, it is not evidence.
