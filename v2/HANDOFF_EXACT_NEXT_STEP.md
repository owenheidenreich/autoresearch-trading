# Exact Next Step

## Status: longer RL already tested and failed

The longer RL run (800s budget, 8 epochs) was completed on 2026-04-15. Result: **overall PF degraded from 0.963 to 0.828**. Call% collapsed from 48% to 21%. Fold 1 grind put PF dropped to 0.218. More RL epochs cause over-specialization on recent data, not generalization.

**The Fold 1 regression is structural, not undertraining.**

## Current best checkpoint
`v2/models/seq_agent_side13_rl.pt` (epoch 5, PF 0.963)

## The real next decision

Freeze Side13 epoch 5 as the current milestone. Then choose one of:

### Option A: Accept PF 0.963 and move to live monitoring
The agent demonstrates genuine stateful behavior (thesis persistence, regime-aware side selection, participation selectivity, active exits). PF 0.963 across 300 OOS days with 48% calls, 0.36 flips/day, and no catastrophic folds is a credible research outcome. The remaining -4.12 net PnL is spread thinly with no single exploitable pattern.

### Option B: Partial encoder unfreezing
The frozen encoder was trained on a different objective (static contract ranking). Unfreezing the last transformer layer while keeping the session head frozen could let the encoder learn representations more aligned with sequential decision-making. Risk: overfitting, training instability.

### Option C: Richer session memory
The current 13-dim session state is a compressed snapshot. Adding attention over recent session history (last N steps' states) could help the agent learn temporal patterns like "my last two entries failed, reduce participation." Risk: complexity, more training needed.

### Option D: Different RL method
REINFORCE is noisy and over-specializes quickly (epoch 5 sweet spot, degradation by epoch 8). Offline RL (IQL/CQL) using collected trajectories could produce more stable policies. The project has offline trajectory data from all replay runs. Risk: implementation complexity.

## Recommended path
Start with Option A: freeze, document, and assess whether PF 0.963 meets the project's practical threshold. If it does not, Option D (offline RL) is the most principled next step because it addresses the root cause (REINFORCE instability) rather than adding complexity.

## Evaluation command for current best agent
```
ENV_LATE_ENTRY_BAR=89 ENV_LATE_EXIT_BAR=95 ENV_DECAY_COEFF=0.001 \
python3 -m v2.analysis.frontier_study --data v2/data.pt
```
