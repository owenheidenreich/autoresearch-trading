# v2 Goal

This file describes the mission and the current phase of the project.

## Mission

Build a model that can profitably trade SPX 0DTE long options under an honest historical replay harness.

## Current Phase

The project is currently in the repaired-harness research phase.

That means:

- the historical data / replay contract has been repaired and re-frozen
- the immediate goal is to establish a strong repaired-era baseline through honest walk-forward experiments
- live and paper trading are still future work

## What Success Means Right Now

Near-term success is:

1. establish the first honest repaired-era baseline score
2. improve that score through disciplined keep/revert experimentation
3. beat all replay baselines consistently on walk-forward evaluation
4. preserve harness integrity while doing so

## What Is Not Yet In Scope

These are not current realities:

- autonomous paper trading
- a finished live execution service
- broker-connected production deployment

Those remain roadmap items after the repaired research harness proves itself.
