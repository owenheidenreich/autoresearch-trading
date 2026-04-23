# Claude Continuous Research Addendum — Post Margin-Offset Rescue

Use
[/Users/gduby/Documents/autoresearch-trading/v3/reference/claude_continuous_research_prompt_2026_04_22.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/claude_continuous_research_prompt_2026_04_22.md)
as the base prompt.

Then apply this addendum instead of the earlier simulated-L3 build
addenda:

```text
Additional directions after the iteration-2 margin-offset rescue pass:

1. Treat this as the current provisional composed-stack champion:
   simulated-L3 per-seed oracle + objective-consistent unified entry +
   Layer-3 robust 0.90

2. Treat this as the current high-mean challenger:
   iteration-2 simulated-L3 outer loop

3. The `+0.05` global decision-margin rescue is now falsified.
   Do not spend more loops on:
   - global positive decision-margin offsets
   - generic “trade less” sweeps
   - new side-contrastive rescue attempts
   - GPU reruns

4. The next useful branch must explain or reduce the mean-vs-floor
   tradeoff without relying on another scalar gate.

Priority order:
1. Diagnose why the promoted stack and the iteration-2 challenger diverge
   so sharply in window 6, especially seed 42.
2. If that diagnosis exposes one narrow mechanism, test the smallest fix.
3. If not, move to a more structural branch such as candidate-trained L3.

Strong constraints:
- keep the promoted simulated-L3 stack as the comparison baseline
- CPU only
- no bar-level multi-entry redesign
- no broad architecture rewrite unless the diagnosis clearly justifies it
- no promotion claims from mean PF alone

Required references before starting:
- /Users/gduby/Documents/autoresearch-trading/v3/HANDOFF.md
- /Users/gduby/Documents/autoresearch-trading/v3/reference/simulated_l3_perseed_oracle_promotion_2026_04_23.md
- /Users/gduby/Documents/autoresearch-trading/v3/reference/simulated_l3_iter2_tradeoff_2026_04_23.md
- /Users/gduby/Documents/autoresearch-trading/v3/reference/simulated_l3_iter2_margin_offset_falsified_2026_04_23.md

Suggested stop conditions:
- the diagnosis shows no narrow mechanism and only a broad regime change
- the proposed fix is just another global gate or scalar offset
- candidate-trained L3 would require a large uncontrolled rewrite in one pass
```
