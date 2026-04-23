# Claude Continuous Research Addendum — Post Cycle 9

Use
[/Users/gduby/Documents/autoresearch-trading/v3/reference/claude_continuous_research_prompt_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/claude_continuous_research_prompt_2026_04_22.md)
as the base prompt.

Then add these extra directions:

```text
Additional directions after cycles 7-9:

1. Treat `CPU w=0.00 + L3 robust 0.90` as the provisional composed-stack
champion.

2. Do not spend this loop on call-share reduction or generic
side-discrimination tuning. That workstream is falsified on this stack.
Every non-zero side-contrastive weight tested regressed at least one
seed.

3. Do not spend GPU on this loop. The composed stack is now CPU-sufficient.

4. The next productive workstream is NOT another oracle-peak blend.
`best_exit_pnl`-blended retraining was falsified because it rewards
late-session peaks that Layer-3 does not actually capture.

5. Your goal for this loop is to design and test the first
hold-horizon-aware composed utility target. Prefer the smallest honest
infrastructure build that aligns entry supervision with what the current
champion Layer-3 can realistically exit.

Priority order:
1. Quantify the champion Layer-3 hold horizon distribution by seed and
   window.
2. Use that evidence to propose one narrow target family for retraining.
3. Implement the smallest viable version of that target.
4. Run the minimum honest CPU evaluation against the provisional
   champion.
5. Stop if the first target family clearly fails; do not chain multiple
   target redesigns in one loop.

Good candidate target families:
- simulated-L3 exit PnL per candidate contract using the locked
  champion exit policy
- fixed-horizon utility tied to the observed Layer-3 hold distribution
  (for example, a horizon closer to what Layer-3 actually holds, not
  oracle session-peak)
- near-horizon MFE/MAE-derived utility if it is explicitly justified by
  the observed hold-time distribution

Avoid by default:
- `best_exit_pnl` / oracle-peak blends
- new side-contrastive sweeps
- bar-level multi-entry redesign
- GPU promotion
- RL / agent work

Useful references to read before starting:
- /Users/gduby/Documents/autoresearch-trading/v3/HANDOFF.md
- /Users/gduby/Documents/autoresearch-trading/v3/reference/side_contrastive_rescue_falsified_2026_04_22.md
- /Users/gduby/Documents/autoresearch-trading/v3/reference/outer_loop_blend_oracle_falsified_2026_04_22.md
- /Users/gduby/Documents/autoresearch-trading/v3/reference/layer3_robust_calibration_2026_04_22.md

Use these commands when appropriate:

# inspect current provisional champion artifacts
for s in 42 43 44; do
  ls v3/artifacts/layer3_unified_cpu_w000_seed$s
done

# baseline composed champion entry + L3 calibration artifacts
for s in 42 43 44; do
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_w000_seed$s \
    --policy prior_window_robust --robust-slack 0.90 \
    --out-suffix robust_90
done

# if a new retrained entry model is produced, compose it with the same
# honest L3 path and robust calibration
for s in 42 43 44; do
  .venv/bin/python -m v3.layer3.train_rolling \
    --entry-source unified \
    --chosen-trades <NEW_ENTRY_RUN>/seed_$s/chosen_trades.pkl \
    --out-dir <NEW_L3_RUN>/seed_$s

  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir <NEW_L3_RUN>/seed_$s \
    --policy prior_window_robust --robust-slack 0.90 \
    --out-suffix robust_90
done

Commit discipline:
- commit after every completed hypothesis cycle
- if you do a diagnostic-only cycle, commit the note + handoff update
- stage only your active write set
- never clean or revert unrelated files in this dirty worktree

Suggested commit message style:
- v3 loop cycle N: characterize champion L3 hold horizons
- v3 loop cycle N: build hold-aware composed utility target
- v3 loop cycle N: evaluate hold-aware outer-loop retrain

Stop conditions for this loop:
- the first hold-aware target clearly regresses the PF floor
- or the target build requires broad schema + trainer + evaluator
  rewrites in one pass
- or you reach a clean yes/no answer on whether the hold-aware target is
  worth continuing
```
