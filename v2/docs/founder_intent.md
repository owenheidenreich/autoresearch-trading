# Founder Intent

This file exists so the repo remembers what I am actually trying to build, even when experiments, cleanup passes, and agent sessions get noisy.

## Why I Am Doing This

I want to build an autoresearch system that can honestly learn how to trade SPX 0DTE long options.

I care about the real thing:

- clean data
- a truthful replay harness
- a model that actually learns something about contract selection and risk
- a research loop that can improve without lying to me

I do not want a project that wins by bookkeeping tricks, stale metrics, or a model consuming its own slop.

## What Success Means

Success is not a flashy demo.

Success is a trustworthy exact-chain research system that can:

- ingest and audit clean SPX 0DTE data
- train a model against the current exact-chain environment honestly
- evaluate that model under a replay harness I trust
- preserve decisions, evidence, and context outside the model itself

If that system becomes strong enough later, it can justify paper trading and then tighter execution work. Not before.

## What I Refuse To Fake

- I will not treat bad data as good data because a score looks exciting.
- I will not treat pre-reset or incompatible scores as live evidence.
- I will not pretend live trading exists before the research system is trustworthy.
- I will not rely on model "memory" when the information should be written down or handed to the model directly.
- I will not let documentation drift until nobody knows what is real.

## What The Project Needs Most Now

- A clear statement that the current mission is not live trading. It is a trustworthy exact-chain research system.
- A current list of open questions around data quality, training objective, and model architecture.
- A single command that tells us whether the repo is healthy.
- A durable place for my judgment, so the repo remembers my standards even when experiments get noisy.

## My Standards For The Current Phase

- The current mission is not live trading. It is a trustworthy exact-chain research system.
- Data quality, replay integrity, and experiment truthfulness come before ambition.
- The live path should stay small enough that I can understand it.
- Historical material should be preserved, but it must not silently compete with the live truth.
- If the repo and the docs disagree, the disagreement is a bug.
- If a model result cannot be defended from current data, current code, and current artifacts, it is not evidence.

## Current Phase Boundary

Right now this repo is for:

- data acquisition and auditing
- exact-chain dataset maintenance
- training experiments
- replay evaluation
- experiment logging
- operator clarity

Right now this repo is not for:

- autonomous live trading
- broker execution as a live system
- pretending risk management is solved because a head or score exists

## How To Use This File

When trade-offs are unclear, this file should win over convenience.

When the project changes phase, this file should be edited deliberately instead of letting the repo drift into a new mission by accident.
