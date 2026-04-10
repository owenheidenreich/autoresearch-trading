# v2 Goal

This file describes the mission and the current phase of the project.

## Mission

Build a model that can profitably trade SPX 0DTE long options under an honest historical replay harness.

## Current Phase

The project is currently in the exact-chain recovery phase.

That means:

- the v4 exact-chain harness is frozen and working
- all five initial exact-chain experiments (exp_074 through exp_078) failed
- the immediate goal is to recover positive exact-chain edge and beat baselines under the exact-chain scorer
- the recovery plan is: side supervision first, then soft within-side ranking, then gate calibration

## What Success Means Right Now

Near-term success is:

1. get the model to reliably fire trades (solve gate collapse)
2. achieve positive exact-chain score across 5-fold walk-forward
3. beat all four replay baselines consistently
4. preserve harness integrity while doing so

## What Is Not Yet In Scope

These are not current realities:

- autonomous paper trading
- a finished live execution service
- broker-connected production deployment

Those remain roadmap items after the exact-chain research phase proves itself.
