# v2 Goal

This file describes the mission and the current phase of the project.

## Mission

Build a model that can profitably trade SPX 0DTE long options under an honest historical replay harness.

## Current Phase

The project is currently in the exact-chain reset and baseline re-establishment phase.

That means:

- the v4 exact-chain harness is frozen and working
- the official exact-chain scored runs (`exp_074` through `exp_078`) all failed
- screening history through `exp_087` is preserved in `v2/lab_notebook.md`
- the live training baseline has been reset to gate BCE plus soft KL selection only
- the immediate goal is to re-establish a trustworthy exact-chain baseline at `exp_090`

## What Success Means Right Now

Near-term success is:

1. keep the exact-chain docs, logs, and code in sync
2. re-screen the reset baseline cleanly
3. achieve positive exact-chain score across 5-fold walk-forward
4. beat all four replay baselines consistently without compromising harness integrity

## What Is Not Yet In Scope

These are not current realities:

- autonomous paper trading
- a finished live execution service
- broker-connected production deployment

Those remain roadmap items after the exact-chain research phase proves itself.
