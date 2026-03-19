# Ideas Backlog (Human Planning Only)

This file is intentionally out-of-band from autonomous loop prompts.
It is for human review, planning, and phased roadmap decisions.

## Purpose
Capture exploratory ideas without contaminating the strict foundation contract.

## Current Priority
Finish a reliable end-to-end foundation on the locked 60-feature two-head system:
`data -> training -> replay -> paper -> real`

## Deferred Phase: 64 + Charm Migration
Run as one coordinated migration only:
1. Rebuild data with expanded feature contract.
2. Retrain from a clean baseline with updated architecture contract.
3. Update replay + live pipelines together.
4. Re-validate mechanical consistency before promotion.

## Candidate Research Ideas (Deferred)
- Add Charm and richer IV term-structure signals.
- Explicit strategy families beyond single-leg directional trades:
  - verticals
  - butterflies
  - iron condors
- Position sizing policies tied to Greeks + confidence.
- Dynamic stop/target management and trailing logic.
- Time-of-day or regime-conditioned policy variants.
- Model architecture upgrades after foundation reliability is proven.

## Rule
Nothing in this file is an execution instruction for `run_loop.py`.
Only `training/program.md` is injected into autonomous prompt context.



## new idea


Instead of the trading bot having full authority to place trades immediately without checks, it flags setups with high confidence, but we Implement an expert style large language model layer over the top. This LLM is fine tuned / trained on all of the necessary literature about options and trading (the 0dte library, pickles personality text, pickles strategies, and possibly internet checks). All rapidly occurring in superhuman time. This LLM instantly analyzes the flagged setup and reads the “confidence data” output. then it determines if the trade is worth entering or not. This LLM could be ran on an h100 in tandem with the trained model. The length of time from ML algo flagging, and LLM observing and double checking before giving the go ahead- or stopping, should be about 1 second. (As fast as we possibly can). Then we test 50 trading days chosen at random with this expert layer over the top. 