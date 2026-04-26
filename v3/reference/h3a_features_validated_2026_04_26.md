---
date: 2026-04-26
parent: autoresearch_loop_summary_2026_04_25.md
status: VALIDATED — H3a (3 new trade_state features) passes 5-seed offline + forward-walk gates; first structural change to clear both since the H1/H2 falsifications
---

# H3a: New Trade-State Features — Validated

## Summary

After the 2026-04-25 autoresearch loop falsified two oracle tweaks (H1 relaxed
target, H2 regret-weighted training), the post-loop plan
(`~/.claude/plans/codex-carried-out-an-zesty-axolotl.md`) proposed a disciplined
ladder of structural changes. **H3a is the first rung that passes.**

H3a adds 3 causal scalar features to the L3 oracle's `trade_state`
(7 → 10 features):

| Feature | Definition | Look-ahead test |
|---|---|---|
| `realized_vol_10bar` | rolling std of pnls over last min(10, t+1) bars | mutate-future PASS, 8401 bars |
| `pnl_velocity_5bar` | avg per-bar pnl change over last min(5, t) bars | mutate-future PASS, 8401 bars |
| `mfe_decay_rate` | (current_pnl - mfe) / max(1, mfe_bar_age) | mutate-future PASS, 8401 bars |

Implementation in [v3/layer3/h3a_features.py](v3/layer3/h3a_features.py); audit
in [scripts/look_ahead_audit_h3.py](scripts/look_ahead_audit_h3.py); both
import the same module — single source of truth, byte-equivalent.

## Discipline anchor

Every step gated on the 2026-04-25 oracle-gate label-leakage retraction
([oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md](v3/reference/oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md)):

1. Look-ahead audit BEFORE any code change → 8401 bars × 3 features, 0 leaks.
2. Commit per hypothesis; revert via `git revert` if falsified (not manual undo).
3. Validate on full OOS (376 trades), not the misleading 5-trade FW subset.
4. Validate on truly-unseen forward-walk (42 days post-2026-02-24) before any
   promotion claim.

## Offline 5-seed evaluation (full OOS)

Comparison: `simulated_l3_oracle_*_h3a.npz` vs `*_balanced_fresh.npz` baseline,
scored through `hybrid_live_utility` on chosen trades from
`spx_combined_3seed_001/seed_*/chosen_trades.pkl`.

```
seed    n  base_pf   h3a_pf    d_pf       sm_n  sm_base   sm_h3a    d_sm
  42  376    2.195    2.308   +0.113       61  -16.555  -14.626   +1.929
  43  334    2.096    2.477   +0.381       54  -11.510  -10.647   +0.863
  44  371    1.761    1.891   +0.130       76  -12.154  -11.019   +1.135
  45  322    1.697    1.758   +0.061       59  -14.128  -12.702   +1.426
  46  357    1.653    1.755   +0.101       79  -11.355  -16.180   -4.825 (outlier)
```

Cross-seed:
- Mean base PF: **1.881**
- Mean H3a PF: **2.038**
- Mean delta PF: **+0.157** (gate: +0.10) ✓
- All 5 seeds delta_pf ≥ 0 ✓
- 4/5 seeds positive small-winner capture lift ✓

## Forward-walk validation (42 unseen days, 2026-02-25 → 2026-04-24)

Cold-start days post-W12 cutoff. L3 oracle uses `build_l3_oracle_with_trailing`
to map trailing days to W12's trained model.

```
seed    n   base_pf    h3a_pf     delta
  42    5     0.505     0.197    -0.308
  43    9     0.079     0.616    +0.538
  44   11     3.672     3.315    -0.357
  45    9     0.358     0.565    +0.207
  46   19     3.889     4.330    +0.441
```

- Cross-seed FW mean baseline: **1.700**
- Cross-seed FW mean H3a: **1.805**  →  delta **+0.104**
- Plan gate: FW PF ≥ offline mean (2.038) × 0.85 = 1.732
- H3a FW PF 1.805 ≥ 1.732 ✓ PASSES

3/5 seeds improved on FW; the FW lift (+0.104) closely matches the offline
lift (+0.157) — typical degradation of ~33%, NOT a forward-walk collapse like
V1+L3 (which went from offline +110% to FW -52%).

## H3a vs H2 (same baseline, opposite outcomes)

H2 (regret-weighted log_loss) was falsified in the autoresearch loop with the
same per-side asymmetry pattern. H3a tests the same hypothesis path differently:

| | seed-42 baseline | H2 | H3a |
|---|---|---|---|
| agg PF | 2.195 | 2.032 (-7%) | 2.308 (+5%) |
| sum | $89,757 | $60,830 (-32%) | $87,332 (-3%) |
| calls (n=146) PF | 2.04 | 1.40 (-31%) | 1.94 (-5%) |
| puts (n=230) PF | 2.30 | 2.58 (+12%) | 2.59 (+13%) |

The puts-side lift is structurally similar in magnitude (+12 vs +13%), but H2
crashed calls (-31%) while H3a barely touched them (-5%). That's structural
evidence the new features carry SIGNAL, not just regret-weighting noise.

## Anti-reward-hacking guardrails

Per the plan's checklist:
- Offline PF > 50% relative lift? +8.4%, no red flag.
- Capture-of-best approaching hindsight ceiling (PF 92)? No, h3a still
  -14.6 capture in $0-200 bucket (deeply negative; far from ceiling).
- OOS PF substantially better than train PF? FW (1.805) < offline (2.038),
  expected direction.
- All 5 seeds suspiciously similar? Range 1.755 to 2.477 — wide variance, not
  suspicious.

No red flags. Result appears genuinely real.

## Files

```
v3/layer3/h3a_features.py                    # 3 causal feature impls (single source)
v3/layer3/common.py:46-56,389-410           # TRADE_STATE_NAMES + per-bar plumbing
scripts/look_ahead_audit_h3.py              # mutate-future audit (PASS)
scripts/research_eval_h3a_oracle.py         # seed-42 offline eval
scripts/research_eval_h3a_5seed.py          # 5-seed offline eval
v3/artifacts/simulated_l3_oracle_*_h3a.npz  # 5 seed-keyed oracles (~46 min each)
v3/artifacts/research/h3a_seed42_eval.json
v3/artifacts/research/h3a_5seed_eval.json
v3/artifacts/forward_walk/h3a_with_oracle.json
v3/artifacts/forward_walk/baseline_with_oracle.json
```

## Branch state

Branch `research/h3a-features` off `research/relaxed-oracle-target`:
```
98756b8 H3a Step 6: 5-seed evaluation PASSES decision gate
8b1cc3a H3a Step 5: full-OOS eval with bootstrap CI + per-bucket capture
dfee4c2 H3a Step 2: add 3 causal trade_state features to L3 oracle
10e8902 H3a Step 1: look-ahead audit script for trade_state feature additions
```

Net code change vs autoresearch loop branch: +3 features in `trade_state`,
+ shared module `v3/layer3/h3a_features.py`, + 2 eval scripts.

## Decision tree position (per plan)

H3a Pass + FW Pass:
- ✓ keep features
- → proceed to H3b (per-side oracles) only if per-side asymmetry warrants
  - Asymmetry IS evident (puts +13%, calls -5%); H3b is conditionally triggered
- OR proceed to H3c (sequence model) for the deeper structural change

Open questions:
1. Should H3a be merged into main as the new oracle-feature baseline before
   H3b/H3c? The improvement is real but modest. Net impact: +0.157 offline
   PF, +0.104 FW PF.
2. Which of the 3 features individually contributes most? Ablation requires
   3 more 50-min builds; not run.
3. Per-side asymmetry: H3a's +13% puts PF lift suggests per-side oracles (H3b)
   could capture more of this signal. Cheap test (~2h CPU).

## Cost summary (this session)

- ~46 min CPU: seed-42 H3a oracle build (single)
- ~70 min CPU: seeds 43-46 parallel build (4× concurrent, OMP=2 each)
- ~3 min CPU: 5-seed eval + forward-walk × 2
- Total: ~120 min CPU. **$0 GPU.**
