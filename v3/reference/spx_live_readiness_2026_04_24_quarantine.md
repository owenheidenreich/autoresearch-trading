---
date: 2026-04-24
branch: codex/v3-orc-xsp-10k
status: QUARANTINED — does not clear floor-safe gate
exp_base: spx_live_hybrid_001
---

# SPX Live-Readiness Run — Quarantine Report

## Verdict

The 3-seed `spx_live_hybrid_001` stack does NOT clear the handoff floor-safe gate
at any L3 threshold. Best operating point is `thr=0.20` with composed agg PF
1.779 (need ≥1.976), mean DD 12.7% (need ≤12%), min seed PF 1.368 (need ≥1.786).
Per the handoff (`v3/reference/claude_spx_live_readiness_handoff_2026_04_24.md`),
this is an explicit "stop or quarantine" outcome and the run is NOT promoted.

## Pipeline reproducibility

- Dataset: `v2/data.pt` rebuilt with rolling IV-history prepass; fingerprint
  `b7acd6c4723a206e`; 986 sidecars; build elapsed 10470 s.
- Action surface: `v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl`,
  contract_selection_mode=`risk_band`, execution window 09:45–11:30 ET,
  89,692 rows × 25 actions across 986 days.
- Per-seed simulated-L3 oracle (`v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{42,43,44}.npz`):
  shared `dataset_fingerprint=e6005a6ba7da0a39`, 939,984 sims each, 0 skips,
  next-bar fill semantics confirmed (`exit_bar > decision_bar` invariant
  holds in spot-checked rows).
- 3-seed unified-policy promotion (`spx_live_hybrid_001`): hybrid_live target,
  Akash H100, 13 rolling windows × 3 seeds, runtime ~9 min per seed.
- Routed two-expert L3 (`layer3_unified_spx_live_hybrid_routed_seed{42,43,44}`):
  trains both `entry_policy` and `candidate_surface` per window; router selects
  per window. Routes are exercised — neither expert collapses to dominant on its
  own.

## Floor-safe gate breakdown (per L3 threshold, 3-seed mean)

```
  thr  mean_agg_pf   mean_dd%  min_seed_pf  pf_gate  dd_gate  min_gate
 0.15        1.831     12.536        1.401     FAIL     FAIL      FAIL
 0.19        1.713     14.467        1.332     FAIL     FAIL      FAIL
  0.2        1.779     12.673        1.368     FAIL     FAIL      FAIL
 0.25        1.552     16.813        1.212     FAIL     FAIL      FAIL
  0.3        1.512     19.399        1.300     FAIL     FAIL      FAIL
```

Gate floors (handoff): mean agg PF ≥1.976, mean DD ≤12%, min seed PF ≥1.786.

## Per-seed best L3 thresholds

| seed | best thr | agg PF | DD%  | mean $/trade | trades |
|------|----------|--------|------|--------------|--------|
| 42   | 0.20     | 1.734  | 16.3 | 152.6        | 387    |
| 43   | 0.15     | 2.403  |  8.8 | 231.7        | 348    |
| 44   | 0.15     | 1.401  | 12.9 |  89.5        | 225    |

Seed 43 is the only one that exceeds the agg-PF floor in isolation; seeds 42 and
44 are well under. The aggregate floor cannot be cleared by averaging because
seeds 42/44 drag the mean.

## W5/W6/W8/W12 regressions (thr=0.20)

```
 seed       W5       W6       W8      W12
   42    0.607    1.306    2.762    2.452
   43   16.415    4.615    3.277    1.320
   44    0.623    1.636    0.942    0.670
```

W5 is consistently weak across seeds 42 and 44 (PF ≈0.6 each), confirming the
W5 regression flag from the handoff. Seed 43's W5 is anomalously strong because
it took only 9 trades that window and they were single-expert (`candidate_surface`)
with nearly all winners — small sample. Seed 44 has multiple weak windows
(W5/W8/W12 all <1.0) which explains its overall failure.

## Entry-only baseline (chosen-trades + simple time-stop)

```
  seed 42: pf=0.944 dd=66.7% mean=$-22.12
  seed 43: pf=1.210 dd=16.7% mean=$77.80
  seed 44: pf=0.898 dd=55.0% mean=$-38.89
```

Without L3, the entry stack alone is unprofitable on seeds 42 and 44. L3
composition is doing real work — it lifts every seed substantially — but cannot
rescue them past the floor.

## Chosen-contract distributions (training output, hybrid_live target)

| seed | trades | call/put | mean abs_delta | risk_band 0/1/2/4 | moneyness -1/0/1 (ITM/ATM/OTM) | mean premium $ | bars 9:45–10:00 / 10:01–10:30 / 10:31–11:00 / 11:01–11:30 |
|------|--------|----------|----------------|-------------------|--------------------------------|----------------|-----------------------------------------------------------|
| 42   | 396    | 375 / 21 | 0.594          | 249 / 54 / 4 / 89 | 285 / 33 / 78                  | 1146           | 2 / 97 / 107 / 190                                        |
| 43   | 364    | 343 / 21 | 0.594          | 204 / 61 / 3 / 96 | 269 / 14 / 81                  | 1144           | 4 / 76 / 92 / 192                                         |
| 44   | 238    | 235 /  3 | 0.613          | 154 / 15 / 4 / 65 | 191 / 15 / 32                  | 1144           | 1 / 40 / 52 / 145                                         |

Observations the handoff asked to capture:

- **Side bias**: 94–99% calls across all seeds. This is a hard call-only bias,
  not a balanced selection. The hybrid_live training target did not rescue
  side balance.
- **Risk-band emergence**: band 0 ("fill") still dominates (~58–63% of chosen
  trades), but band 4 (farther-OTM / wide-delta) appears at ~22–28%. So the
  `risk_band` token is doing some work — model is occasionally choosing
  outside the nearest-to-spot — but the dominant action is still the
  fill-band slot.
- **Moneyness**: chosen trades skew ITM (`-1` bucket: 72–80%) over OTM (15–22%)
  and ATM (4–9%). Mean abs_delta ≈0.6 supports this — these are deeper-in-the-money
  picks. Premium ≈$1145 is consistent (large premium per contract). The handoff
  asked whether OTM emerges naturally; it does, modestly, but is not the
  dominant pick.
- **Time-of-day**: trades cluster in 11:01–11:30 ET (bars 76–105 of session)
  for all seeds — last 30-min of the 09:45–11:30 window. Default window is
  pulling the model to the late side of its allowed range.

## Routed L3 router behavior

train_reports per-window mode counts:

```
  seed 42: fallback=3  model=21
  seed 43: fallback=3  model=23
  seed 44: fallback=2  model=18
```

Router picks per-window expert by prior-window evidence; in `per_window` results
both `entry_policy` and `candidate_surface` are exercised across windows. So the
router is not collapsed.

## What this run argues

The handoff itself flagged that the project is moving away from "best historical
simulated-L3 backtest" toward a live-executable SPX policy. This run was the
infrastructure-level repair: dataset IV-history fix, risk-band contract surface,
hybrid_live utility, next-bar fill labels, two-expert routed L3. All wired up
cleanly, all of it converging into 3-seed evidence — but the resulting numbers
don't beat the simulated-L3-best champion on the simulated-L3-best gate.

This is consistent with the handoff's framing: distribution-and-intent mismatch
is the binding constraint, not another scalar tuning knob. The hybrid_live
target picked deep-ITM call-heavy trades, which carry a different P&L profile
from the older champion that lived ATM-nearest. They look fine under their own
hybrid_live evaluation (agg PF 1.486–1.988 raw, see seed reports), but composed
with L3 exits the seed dispersion is wide and the floor isn't met.

## Recommendations

1. **Do not promote.** Champion benchmark stands.
2. **Side balance is the leading culprit.** A 94–99% call rate across seeds
   means the policy is essentially long-vol-on-up only. The hybrid_live
   utility's bonus structure (return-multiple bounded `[-2, 4]`, late-hold
   penalty, stopout penalty) doesn't reward put picks proportionally. Worth
   testing whether a side-balanced training step (e.g., re-weight per-side
   ranking loss, or symmetric data augmentation around VWAP-flip days) lifts
   put coverage without trashing call PF.
3. **W5 weakness reproduces.** The W5 OOS window (2024-03-26..2024-06-20) was
   already a regression flag in the prior champion review. Seeds 42/44 hit
   PF≈0.6 there. Worth a window-specific diagnostic (regime characterization,
   chosen-bar timing in W5) before changing global hyperparameters.
4. **Diagnostic-only widened windows (09:45–12:30, 09:45–15:00) NOT run.**
   The handoff lists them as optional. Skipping unless the next-iteration
   plan calls for it; the 09:45–11:30 chosen-time clusters at the late end
   of the window already, suggesting the model wants more time, but
   widening for promotion needs to clear the same floor.
5. **Live-shadow scaffolding holds, no shadow yet.** Per the handoff, live
   shadow comes after offline parity passes. Offline parity here is
   gate-failure, so no shadow path is started.

## Artifacts

```
v3/artifacts/
├── layer2_action_surface_dataset_spx_live_0945_1130.pkl
├── simulated_l3_oracle_spx_live_0945_1130_seed{42,43,44}.npz
├── layer2_unified_policy_spx_live_hybrid_001_seed{42,43,44}/
│     ├── manifest.json
│     └── seed_{42,43,44}/
│           ├── report.json
│           └── chosen_trades.pkl
└── layer3_unified_spx_live_hybrid_routed_seed{42,43,44}/
      ├── rolling_layer3_report.json
      └── layer3_trades_thr_*.csv
```
