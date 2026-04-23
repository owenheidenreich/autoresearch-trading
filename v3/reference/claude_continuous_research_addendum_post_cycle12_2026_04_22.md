# Claude Continuous Research Addendum — Post Cycle 12

Use
[/Users/gduby/Documents/autoresearch-trading/v3/reference/claude_continuous_research_prompt_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/claude_continuous_research_prompt_2026_04_22.md)
as the base prompt.

Then apply this addendum instead of the earlier post-cycle-9 addendum:

```text
Additional directions after cycles 10-12:

1. Treat `CPU w=0.00 + L3 robust 0.90` as the locked provisional
composed-stack champion.

2. Do not spend this loop on:
- call-share reduction
- side-contrastive rescue
- fixed-horizon utility targets
- `best_exit_pnl` / oracle-peak blends
- GPU promotion

Those branches are now falsified or out of scope.

3. The addendum from post-cycle-9 is superseded. “Hold-horizon-aware”
in the simple fixed-horizon sense has now been tested and falsified.

4. The only defensible next outer-loop hypothesis is a larger
infrastructure build:
`simulated-L3 oracle per candidate contract`

That means:
- for each candidate contract on each eligible bar, estimate the PnL
  the current champion Layer-3 policy would have realized, not the
  session-end oracle and not a fixed horizon
- use that as the next composed-utility supervision target

5. If you are not prepared to build that infrastructure honestly, stop.
Do not substitute smaller proxies just to keep the loop moving.

Priority order if you continue:
1. Characterize exactly what the champion Layer-3 policy would need as
   an offline simulation oracle.
2. Design the smallest trustworthy candidate-contract simulation path.
3. Implement only the infrastructure needed to generate that label.
4. Run one narrow honest evaluation against the locked provisional
   champion.

Strong constraints:
- preserve the current champion as the comparison baseline
- CPU only
- no entry-regime redesign
- no new threshold-family sweeps unless the hypothesis is specifically
  about the simulated-L3 oracle itself
- no chaining multiple target families in one loop

Required references before starting:
- /Users/gduby/Documents/autoresearch-trading/v3/HANDOFF.md
- /Users/gduby/Documents/autoresearch-trading/v3/reference/outer_loop_blend_oracle_falsified_2026_04_22.md
- /Users/gduby/Documents/autoresearch-trading/v3/reference/hold_aware_horizon_utility_falsified_2026_04_22.md
- /Users/gduby/Documents/autoresearch-trading/v3/reference/layer3_robust_calibration_2026_04_22.md

Useful framing:
- side-contrastive failure, oracle-blend failure, and horizon-60 failure
  all showed the same seed-dependent pattern
- that pattern is evidence that loose proxies for composed utility are
  misaligned with what the champion exit can actually realize
- do not accept “better by time-stop PF” as success if composed PF gets
  worse

Commit discipline:
- if you only do feasibility / design work for the simulated-L3 oracle,
  commit the design note and any safe plumbing separately
- if the infrastructure build becomes too broad in one pass, stop and
  summarize instead of forcing partial conclusions

Suggested commit message style:
- v3 loop cycle N: design simulated L3 oracle for candidate contracts
- v3 loop cycle N: build candidate-contract L3 simulation labels
- v3 loop cycle N: evaluate simulated-L3 outer-loop retrain

Stop conditions:
- the simulated-L3 oracle requires a broad rewrite of schema, replay,
  and trainer in one pass
- or the first narrow implementation clearly regresses the PF floor
- or you reach a clean design-only stopping point without a trustworthy
  evaluation yet
```
