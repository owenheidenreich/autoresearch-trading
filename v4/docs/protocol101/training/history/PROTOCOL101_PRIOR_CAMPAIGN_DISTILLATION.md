# Protocol101 Prior-Campaign Distillation

**Purpose:** Stage-1 prior knowledge for the canonical v1 H0-H4 program.

**Primary evidence window:** 2026-04-18 through 2026-04-27.

**Secondary evidence:** the May protocol farm and later fair-contract audits, explicitly marked **pre-parity**.

**Prepared:** 2026-07-18.

## Executive conclusion

The durable April entry result was not “VWAP reclaim.” It was a learned opening-structure and contract-quality result: opening gap, first-15-minute settlement/acceptance, and a wide first-15 range carried the cleanest evidence. V2 produced PF `1.291`; removing `vwap_reclaim_state` raised PF to `1.320`; applying the preregistered `first15_range_pct >= 20 bps` deployment gate raised PF to `1.540`. The reclaim state was dropped because it was absent from every fold's top-five importance list and its ablation **improved** mean return by `+0.36 pp` and the Control-A gap by `+0.30 pp`.

The durable April lifecycle result was that entry selection and post-entry management are separate problems. A causal learned exit layer repeatedly lifted the modest entry baseline, and H3a's causal trade-state features improved aggregate PF, but several spectacular curves were single-window, overlapping-training, oracle-target, or forward-walk artifacts. The April 22 methodology overhaul is the epistemic break: it expanded OOS evidence from `20` to `780` days across 13 disjoint windows and reversed `always_put` from local PF `1.391` to aggregate PF `0.888`, while directional V0 posted PF `1.132`.

The later Protocol101 stack inherited real April ideas—contract-relative ranking, causal option-path state, explicit wait/take arbitration, side/time/VWAP context—but its large historical curve was **not live-equivalent**. In June, corrected historical replay produced `38` entry signals on June 5/8/9 while live produced `0` above-edge minutes. This mismatch occurred before order submission. Therefore all May protocol-farm and later fair-contract conclusions in this brief are weaker, **pre-parity evidence**, not confirmation of the April edge.

## Scope, evidence tiers, and verdict language

| Tier | Evidence | How it is used here |
|---|---|---|
| **A — primary** | Git commits and their files from 2026-04-18..2026-04-27; the off-mainline Pickles Fork-C branch is included because it falls inside the window. | Determines the Edge Ledger and do-not-retest list. |
| **B — secondary / pre-parity** | The retrospective [v4 research ledger](../ledger/RESEARCH_LEDGER.md), May protocol artifacts, and the fair-contract search directories. | Only traces April ideas into later protocols and explains why the later curve was not live-equivalent. It cannot upgrade an April verdict. |
| **C — current status** | [paper-default registry](../promotion/PAPER_TRADING_DEFAULT.json), [naming guide](NAMING_GUIDE.md), current runtime code, and later parity reports. | Decodes protocol numbers and states current roles. It does not prove profitability. |

Verdict labels are deliberately strict:

- **Had edge:** survived the test actually run, with a positive economic or predictive result. It may still be pre-deployment or pre-parity.
- **No edge:** failed its registered comparison, was retracted, or was superseded by a broader clean test that reversed the result.
- **Untested cleanly:** mechanistically plausible, but the available test used a proxy, was underpowered, leaked, overlapped training, stopped before the relevant phase, or never reached the required replay/parity gate.

PF means dollar profit factor unless the source says otherwise. A missing PF/AUC/p-value is shown as `not reported`, not inferred.

## 1. Edge Ledger

### 1.1 Had edge in the April evidence

| Strategy / hypothesis | Mechanism in plain language | Test design | Exact verdict and number | What happened next | Evidence |
|---|---|---|---|---|---|
| **V2 learned opening-structure scorer** | Let a shallow random forest choose the best admissible bar/side from opening gap, first-15 settlement/acceptance, option/Greek, spread, and context features, while freezing V1B's gates, contract selection, exits, and controls. | Five chronological folds; 300 test days; RF fixed at 200 trees, depth 5, leaf 20; trained only on fold-train outcomes; 266 selected test trades; same-day random-time Control A and random-day Control B. | **Passed:** PF `1.291`, mean `+0.944%`, versus Control A PF `0.855`, mean `-1.336%`, and Control B PF `0.642`, mean `-0.746%`; strategy-Control A gap `+2.28 pp`; 3/5 folds beat A. | Stress-tested, simplified, and promoted as V2-pruned. | `7df452a5`; [V2 plan](../../v2/docs/mechanical_baseline_v2_plan.md); [implementation](../../v2/analysis/mechanical_baseline_v2_learned_scorer.py). |
| **V2-pruned** | Same V2, but remove the VWAP-reclaim-state input. | Day-block bootstrap, 100 label-permutation refits, feature ablations, then a full five-fold rerun without the feature. | **Passed all three gates:** bootstrap gap-vs-A 95% CI `[+0.19%, +4.92%]`, `P(gap>0)=98.4%`; permutation `p=0.010` for gap, `p=0.000` for strategy mean and PF. Rerun PF `1.320`, mean `+1.234%`, gap `+2.56 pp`, all above V2. | Became the new promotable v2 baseline. `vwap_reclaim_state` moved to “falsified as core premise.” | `315dfbbd`; [stress test](../../v2/analysis/mechanical_baseline_v2_stress_test.py); [pruned runner](../../v2/analysis/mechanical_baseline_v2_pruned_opening_reversion.py); [strategy card](../../v2/docs/strategy_card_opening_reversion.md). |
| **Wide-first-15 deployment gate** | Trade the full-distribution-trained V2-pruned model only when the first 15-minute range is at least 20 bps. | Gate fixed from a population regime audit, then rerun end-to-end without retraining. | **Promoted:** `n=179`, PF `1.540`, mean `+2.98%`, gap vs A `+3.27 pp`; strategy mean CI `[+0.02%, +6.48%]`, `P(mean>0)=97.5%`; gap CI `[+0.67%, +6.40%]`, `P(gap>0)=99.1%`; PF CI `[1.04, 2.36]`. | Kept full-distribution training; rejected gated-days-only retraining. | `2372add2`, `f4e35501`; [v2 artifacts](../../v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned/summary.json). |
| **Detach-side Layer-2 entry gate** | Share an encoder for entry and side, but stop side-loss gradients from corrupting the entry trunk. | Five-fold replay of selected option entries; architectural ablations and random-direction/slippage checks. | **Best Layer-2 baseline:** PF `1.455`, DD `36.9%`; side-dropout PF `1.137`, wrong-side-alpha PF `0.832`, and detach+dropout PF `1.451`. Random direction averaged PF `1.116`; model-side incremental PF was about `+0.34`; $50 slippage left PF `1.332`. | Paper trading was paused because much of the value came from the entry gate and the directional head remained fragile. This became the entry substrate for learned exits. | `bc85b370`, `8f216476`; [Layer-2 README](../../v3/layer2/README.md); [diagnostic](../../v3/reference/layer2_shared_encoder_diagnostics_2026_04_21.md). |
| **Learned Layer-3 exits** | After entry, predict whether to exit from the causal trade path rather than use only a fixed time stop. | First measured hindsight headroom; then compared simple exits; trained a walk-forward HGB exit model; checked threshold stability, random-label controls, and slippage. | Hindsight oracle PF `133.965` versus corrected time-stop PF `1.472`. Best simple benchmark was time-of-day-90 PF `1.709`. Learned exit at threshold `0.20` reached PF `2.085`, and the reality-check rerun reached PF `2.228`, DD `26.3%`; 9/10 thresholds `[0.10,0.30]` beat `1.709`; random-label max PF `1.134`. | Cached 20-day OOS exposed entry generalization failure (entry PF `0.869`); teacher augmentation restored composed OOS PF to `1.537`. The April 22 overhaul required walk-forward retraining before any deployment claim. | `e042e641`, `3f2ccefe`, `591c3229`, `48b4addd`, `c8a5daf8`, `a6509da0`, `905c37f3`; [Layer-3 code](../../v3/layer3). |
| **Methodology-correct directional V0** | Use the learned entry/side model's direction instead of forcing puts, with a fixed time stop. | 13 disjoint rolling windows, 60 OOS days each, expanding 180-900-day train and 40-day validation; 780 OOS days total; 51 causal features. | **New champion after overhaul:** aggregate PF `1.132`; mean window PF `1.188`, median `1.033`, range `0.59-2.27`. V0 beat V1 by the meaningful margin in 10/13 windows and strictly in 12/13. | Replaced `always_put` and retired the PF `2.847` 20-day claim. Became the honest entry baseline for subsequent composed work. | `457aed6b`, `29945a67`, `01af7c9f`, corrections `90102041` and `cf125714`; [methodology summary](../../v3/reference/methodology_overhaul_summary_2026_04_22.md). |
| **Prior-window robust L3 calibration on V0** | Choose exit thresholds from the previous window and prefer a stable threshold plateau rather than the single maximum-PF point. | Three entry seeds; rolling L3; prior-window-only calibration; robust-slack policy. | Honest prior-window calibration mean PF `1.670`, min `1.423`; robust calibration later reached mean PF `1.794`, min `1.652`. The provisional CPU-weight-0 composed champion was mean PF `1.783`, min `1.711`, DD about `22%`. | Kept as composed baseline; side-contrastive and exit-target blend knobs failed to improve it consistently. | `0689d590`, `18a3a20d`, `dfcf7029`; [v3 Layer-3](../../v3/layer3). |
| **Objective-consistent simulated-L3 entry target** | Train entry choice on the payoff the current learned exit policy would actually realize, not on a mismatched fixed-time payoff. | Deterministic dataset fingerprint; objective-consistent validation calibration; three seeds; then rebuild a separate oracle from each seed's own composed stack. | Shared-oracle corrected run tied the champion: mean PF `1.775` vs `1.783`, min `1.724` vs `1.711`, DD `10.4%` vs `22.0%`. Per-seed oracle then posted mean PF `1.950`, min `1.786`, aggregate PF `1.9765`, DD `10.1%`, and beat the old champion on all three seeds. | Became the provisional offline champion; a 50/50 oracle mix later failed. It never earned live parity. | `75692194`, `98360a4e`; [objective-consistent rerun](../../v3/reference/simulated_l3_oracle_objfix_rerun_2026_04_23.md); [per-seed promotion run](../../v3/reference/simulated_l3_perseed_oracle_promotion_2026_04_23.md). |
| **Balanced oracle plus full-coverage sample weights** | Correct the learned entry model's severe call bias by balancing oracle examples and weighting both sides without deleting coverage. | Seed-42 screen, then 3-seed and 5-seed checks; later 26- and 42-day forward walk. | Seed 42 PF `2.195`, DD `12.4%`, but min fold `0.616`. Three-seed mean PF `2.017`, DD `11.64%`, min `1.761`; five-seed mean PF `1.881`, DD `12.62%`, min `1.653`. On 42 unseen days entry-only PF was only `1.14`; restoring the causal exit oracle raised it to `1.70`. | Kept as a real but weaker-than-offline effect; the `2.142` filtered deployment claim failed on unseen days and was withdrawn. | `a36738ad`, `723f2c5e`, `97feab48`, `f23cf3ff`, `1b37da44`, `963fe0e3`. |
| **H3a causal trade-state features** | Add realized-volatility-10, five-bar PnL velocity, and MFE decay so the exit model can distinguish healthy convex pullbacks from decaying trades. | Look-ahead audit; seed-42 full OOS bootstrap; five-seed replication; 42-day forward walk; regime stratification. | Seed 42 PF `2.195 -> 2.308`, delta CI `[-0.220, +0.487]`, `p(delta<=0)=0.273`. Five-seed mean PF `1.881 -> 2.038`, all 5 deltas nonnegative. Forward walk PF `1.700 -> 1.805`, 3/5 seeds improved. | Passed the aggregate and forward gates, but exposed regime dependence: bear-high PF `1.990 -> 1.488`, chop-high `2.479 -> 2.991`, bull-low `1.857 -> 2.312`. Retained as a structural clue, not a universal exit fix. | `8b1cc3aa`, `98756b84`, `1ea34474`, `579ef0be`; [H3a feature branch](../../v3/layer3). |

### 1.2 No edge / falsified or retired

| Strategy / hypothesis | Mechanism and test | Exact verdict and number | What happened next | Evidence |
|---|---|---|---|---|
| **Pickles Row-1 SPX-only VWAP-support proxy** | Approximate the journal setup with SPX above-VWAP history and reclaim behavior; compare with random bars. The original strict trigger produced zero events, so the proxy allowed a prior 10-bps-above-VWAP condition because Pickles actually watched ES. | **Falsified as an SPX proxy:** P-open `n=624`, MFE +30 bps `8.8%`, MAE -30 bps `13.5%`, median option end `-4.98%`; P-all `n=162`, MFE `6.8%`, MAE `17.9%`, option end `-15.39%`; random `n=850`, MFE `10.2%`, MAE `10.9%`, option end `-4.74%`. Best Pickles target-hit rate `9%` versus required `50%`. | Fork A stopped. The true ES/NQ/A-D version remained untested, not falsified. | `41be6a5e`; [Fork A result](../../v2/docs/fork_a1_row1_results.md). |
| **Pickles Tier-1 10:00 day-call classifier (Fork C)** | Predict whether Pickles would take a directional SPX 0DTE long day from 84 causal features at 10:00; chronological 60/20/20, top-30 threshold selected on validation, 1,000 bootstrap resamples. | **Null:** test `n=25`, 3 positives, base PR-AUC `0.120`; LR-all PR-AUC `0.128 [0.040,0.362]`, Brier `0.301`; curated LR PR-AUC `0.166 [0.040,0.500]`, Brier `0.240`; majority Brier `0.123`. HGBT skipped by preregistered trigger. | Phase 2 did not run. Row-2 Magic Time, a Row-3 target, and ES/NQ/A-D enrichment stayed open. | Off-mainline commit `33f3ca28` on `codex/fork-c-phase1`; file `v2/docs/fork_c_tier1_results.md` exists in that commit only (`git show 33f3ca28:v2/docs/fork_c_tier1_results.md`). |
| **V1A hand-coded opening-structure reversion** | Require VWAP overextension/reclaim, first-15 acceptance, and bar delta; fixed spread/Greek gates and VWAP/first-15/time exits; five folds and two random controls. | **Failed at boundary:** `n=109`, mean `-0.354%`, PF `0.918`; Control A mean `-0.513%`, PF `0.882`; Control B mean `-2.506%`, PF `0.497`; 3/5 folds positive. | Kept only as a mechanical scaffold and control framework; V1B tried IV/VRP gates. | `87b429e8`; [V1A runner](../../v2/analysis/mechanical_baseline_opening_reversion.py). |
| **V1B V1A plus IV-percentile/VRP gates** | Select one of nine IV/variance-risk-premium gate combinations on train only, then use frozen V1A mechanics. | **Failed:** `n=90`, mean `-0.002%`, PF `0.975`; Control A `-1.018%`, PF `0.736`; Control B `+0.090%`, PF `1.094`. | V2 kept the gates frozen but replaced bar/side selection with a learner. IV/VRP was not established as an independent edge. | `621ec097`; [V1B artifacts](../../v2/artifacts). |
| **V2 score threshold / coverage selection** | Trade only the highest train-score quantiles. | Train ranking was monotonic, but test transfer failed. Coverage 20/40/60/80/100% produced test gaps `+1.32/+0.21/+0.37/+0.24/+0.31 pp`; every CI crossed zero and `P(gap>0)` was only `0.53-0.61`. | Keep unthresholded V2-pruned. | `1897d85c`. |
| **Retrain only on wide first-15 days** | Remove narrow-range days from training as well as deployment. | **Failed:** `n=183`, mean `+0.24%`, PF `0.991`, gap `+0.44 pp`, versus full-distribution train/deploy-gated PF `1.540`, gap `+3.27 pp`. | Preserved narrow days as contrastive training examples. | `96810f0d`. |
| **Upsized Kelly/block-CV sizing** | Select fixed fraction from train/OOB or nested block CV. | OOB chose `0.25%` in every fold because OOB mean was `-1.10%`. Block-CV upsizing total return `1.44` and Calmar `0.46` versus 0.25%-floor return `1.34`, Calmar `3.22`, DD `2.64%` versus `0.35%`. | Accepted `0.25%` floor; no sizing-derived edge claim. Integer-contract audit found ~`$1M` needed for 176/179 trades. | `edbef1da`, `62b1bfc3`, `d65a491f`. |
| **Late-session Pickles/ORC trigger tournament** | Break-then-VWAP-reclaim reversal and second-test continuation across four regime families. | **0/8 won.** Every detector's MFE20 was `0.3-1.2 bps` below control's `+7.20 bps`; observed NR10 coincidences were `22`, with `0/22` direction matches. | Dropped as entry triggers; late-session regime features were allowed only as soft Layer-2 inputs. | `2eb4f3af`, `995922ee`. |
| **Eight late-session soft features (W2a)** | Add time/regime localization features to the neural screen. | **Gate failed:** PF `0.722`, DD `97%`, direction accuracy `0.496`. | Features remained available for diagnostics, not as evidence of edge. | `c1e7546a`, `59db6435`. |
| **Plain Layer-2 neural variants** | Separate MLP and undetached shared encoder for entry/side. | Separate MLP PF `0.899`, DD `196.3%`; shared encoder PF `1.153`, DD `75.9%`; tree fixed-quantile PF `1.122`, DD `56.2%`. | Replaced by detach-side PF `1.455`. | `2029488b`, `bc85b370`. |
| **Fallback direction routing** | Route teacher and fallback selections differently, or train fallback-only put-vs-flat / call-put-flat heads. | **Falsified:** fallback-only PF `1.340` and `1.159` versus detach-side baseline `1.455`; fallback bars sat near the 53rd percentile of same-day payoff. | Stopped routing work; diagnosed payoff-regime sufficiency instead. | `c096883f`; [route diagnostic](../../v3/reference/layer2_route_aware_fallback_2026_04_21.md). |
| **ATM-IV bottom-two-decile suppression** | Suppress fallback entries in low-atm-IV deciles. | Initial soft lift PF `1.455 -> 1.530`, but it was only the 78th percentile of 200 random suppressions (`21.5%` matched/exceeded); K=1 and K=3 deltas were `-0.026` and `-0.008`; tree transfer hurt PF `1.122 -> 1.095`. | Falsified as a regime gate. | `23c1d45b`, `d78d89ad`; [stress battery](../../v3/reference/layer2_in_sample_stress_battery_2026_04_21.md). |
| **Always-put as the default direction** | Replace the fragile learned side with puts on every selected entry. | Local Stage C1: OOS PF `1.391` vs V0 `0.869`, `n=20`, mean/trade `+$159` vs `-$68`, DD `17.0%` vs `25.9%`. Broader rolling test: **V1 PF `0.888` vs V0 `1.132` over 780 days**; V0 strictly won 12/13 windows. | The April 22 test superseded the local win. Always-put may describe a bearish/choppy regime, but is on the do-not-use-as-global-default list. | `0b2b4539`, `f833b165`, `29945a67`, `01af7c9f`; [directional variants](../../v3/reference/layer2_directional_variants_2026_04_21.md). |
| **V1 plus L3@0.19 / A3 single-window champion** | Compose always-put with learned exits; then remove `mfe_norm`. | Local PF `2.169`; bootstrap 95% PF CI `[0.520,12.645]`, only `86.6%` of resamples PF >= 1. A3 local PF `2.847`, `P(PF>=1)=92.7%`, CI `[0.680,28.770]`. | Retired after rolling-window reversal: wrong global direction and overlapping L3 training. | `17682ecd`, `5daca0a0`, `d5294668`, `01af7c9f`. |
| **Morning snapshot regime classifier** | Predict favorable regimes before the session from morning features. | **Abandoned:** in-sample AUC `0.536 < 0.55`; OOS AUC `0.196`; recall `0/3`. | No regime gate. | `7bf37581`. |
| **Conditional V0/V1 switch** | Use 51 entry-bar features to predict the rare days where puts beat learned direction. | AUC often `0.72-1.00`, but precision at 0.5 was about zero; only `21/432` trades (`4.9%`) were V1-favorable. Conditional PF `1.092` vs V0 `1.132`; oracle ceiling `1.302`. | Kept V0; no conditional switch. | `01af7c9f`; [conditional rule](../../v3/reference/conditional_directional_rule_2026_04_22.md). |
| **Unified entry/action model and fallback rule** | Replace staged entry/side selection with one calibrated action model; optionally fall back when its confidence is low. | Post-fix seed PFs `1.066/1.172/1.097`, mean `1.112`, aggregate `1.111`, below V0 `1.132`; fallback mean `1.107`. GPU mean `1.116`, aggregate `1.115`. | Shelved as signal-limited; L3 composition was tested separately. | `c0541458`, `5fb22596`, `112282f3`, `0c8e842a`. |
| **Side-contrastive loss** | Penalize insufficient call/put separation. | Weights `0.1/0.2/0.5` produced call shares `93.9/94.5/90.9%` and PF `1.074/1.091/1.002`; no weight improved every seed after L3. | Champion remained weight `0.00`. | `287c11fa`, `35988341`, `dfcf7029`. |
| **Time-of-day-90 fallback, exit-target blends, and horizon-60 utility** | Substitute time-of-day exit when L3 is uncertain; blend learned utility with best-exit oracle; or train a shorter hold-aware horizon. | Time-of-day fallback PF `1.826`, DD `33.8%` vs time-stop fallback PF `1.889`, DD `22.3%`; oracle blend alpha 0.3 mean PF `1.520`, alpha 0.1 `1.792` but min `1.565`; horizon-60 mean PF `1.462`, min `1.288`, DD `45%` vs champion `1.783/1.711/22%`. | All falsified; preserved robust composed champion. | `607d8633`, `36e1a379`, `1063524b`. |
| **Brittle heuristic exits and runtime trail prototype** | Test 14 fixed exit heuristics, then prototype a causal PnL trail after the leaky oracle gate was removed. | `bail_out_30` PF `1.833` but broke fold 0 (`0.875 -> 0.539`); defensible `time_of_day_90` PF `1.709`; trailing stops and take-profits hurt PF. The later runtime trail showed only a provisional `+6%` PF while its simulator PF `1.07` did not reconcile with reported oracle PF `1.89`. | Used `1.709` as the learned-exit bar; did not promote a trail rule. The subsequent relaxed-target experiment was tested separately and falsified. | `3f2ccefe`, `a7d45848`. |
| **Simulated-L3 outer-loop iteration 2 and global margin rescue** | Rebuild entry oracles from the newly promoted composed stack, then try a global `+0.05` decision-margin offset to recover the weaker seed floor. | Iteration 2 improved mean PF `1.950 -> 2.016` but reduced min PF `1.786 -> 1.733` and worsened DD `10.1% -> 11.8%`; not a clean promotion. Margin rescue fell to mean PF `1.705`, min `1.187`, DD `20.9%`, aggregate PF `1.689`. | Kept the first per-seed simulated-L3 stack. Global tightening was falsified; next work had to be structural. | `2b468de8`, `2329a65a`; [tradeoff](../../v3/reference/simulated_l3_iter2_tradeoff_2026_04_23.md); [margin falsification](../../v3/reference/simulated_l3_iter2_margin_offset_falsified_2026_04_23.md). |
| **50/50 chosen-L3/candidate-L3 oracle blend** | Average two entry targets to smooth their failures. | PF `1.748/1.856/1.393`, mean `1.666`, min `1.393`, below chosen-L3 provisional champion mean `1.950`, min `1.786`. | Falsified; future mixing would require conditional routing, not a scalar blend. | `393ec6e4`; [mix diagnostic](../../v3/reference/sim_l3_mix50_oracle_falsified_2026_04_23.md). |
| **W5 side-score calibration and aggressive reweighting** | Calibrate/reweight the call-heavy model so it chooses real put opportunities. | Production W5 PF `0.703`, PnL `-$1,873`; global calibration PF `0.970`; strict PF `0.774`; permissive PF `0.758`. Weight 0.5 improved overall PF only `1.486 -> 1.496` and W5 `0.703 -> 0.804`; weight 1.0 fell to `1.211` and W5 `0.677`. | Moved to combined balanced-oracle/full-coverage weighting, which had real but still pre-parity evidence. | `8e0a030a`, `80b878ee`, `95f22474`. |
| **K=2/K=3 consensus plus `cal_pf>4` guard** | Filter the offline champion to seeds/contracts with consensus and high calibration PF. | Offline PF about `2.142`; on 26 unseen days aggregate PF **`0.72`** (seed PFs `1.05/0.38/0/0.17/2.02`). On 42 days, entry PF recovered only to `1.14`. | Proof failure; do not deploy the filter. | `7a1014e0`, `f23cf3ff`, `1b37da44`. |
| **Oracle runtime gate classifier** | Detect causal states where the learned exit would truncate winners. | A hand rule initially showed PF `2.086` vs oracle `1.890`; a trained gate claimed `+24%` forward PF. Audit found `side_margin_raw` and `time_stop_margin_raw` were future labels. Clean fixed-window PF `1.736` vs baseline `1.890`; clean rolling PF `1.829`; `0%` beat baseline. | Retracted and deleted as label leakage/reward hacking. | `0f12cb3d`, `300bf8e4`, `4c331b90`. |
| **H1 relaxed exit target** | Label exit when current payoff is at least 85% of future suffix maximum. | Strict oracle upper-bound PF `92.5`; relaxed PF `84.6`; relaxed beat strict on `0/61` paths, strict on `35/61`. | Reverted and marked falsified. | `10f1822a`, `533c10e2`. |
| **H2 regret-weighted L3 training** | Upweight examples with large hindsight regret. | Full OOS PF `2.195 -> 2.032` (`-7%`); calls `2.04 -> 1.40`, puts `2.30 -> 2.58`; exits `19.5` bars earlier. | Reverted. | `14e9f029`, `c0fdcbe8`, synthesis `cafdabea`. |
| **H3e profitability-only exit target plus Greeks** | Stop teaching loser defense and focus the exit target on profitable paths, with per-bar Greeks. | PF `2.195 -> 1.591`, delta `-0.603`; spread `1.446 -> 0.838`; CI `[-1.10,-0.20]`, `p=0.997` for no improvement. | Reverted. Loser-defense labels were load-bearing; Greek plumbing alone was not credited. | `ce7bfe76`. |
| **Side-blind L3** | Remove side information to reduce regime spread. | PF `2.195 -> 1.490`, delta `-0.705`; spread narrowed to `0.710`, floor `1.311`; CI `[-1.14,-0.36]`. | Reverted. | `fe89ac31`. |
| **C-Pre continuous regime magnitude predictor** | Predict which side/regime cell will work from continuous intraday state. | Between-cell variance only `2.1%`, within-cell `97.9%`; RF `R²=0.059`. | No-go; did not train a gate. | `598a381e`. |
| **Layer-2 dollar-score gate** | Use model dollar scores to abstain in weak regime cells. | Baseline PF `1.877`, floor `1.245`; candidate thresholds left aggregate around `1.857-1.890` but worsened floor as low as `0.712-0.736`. | Falsified as anti-calibrated in the floor cell. | `066dd95e`. |
| **H3c causal TCN exit model** | Use a temporal convolution over the causal path instead of hand summaries/HGB. | Seed 42 PF `2.195 -> 2.350`, floor `1.405 -> 1.657`; five-seed mean baseline `1.877`, H3a `2.019`, TCN `1.855`; only 3/5 seeds lifted; CI `[-0.299,+0.277]`, `p=0.574`. | Not promoted; high-volatility regimes caused two crashes. | `e184114f`, `bbfd45c0`. |
| **H3f regret-regression target** | Predict future regret directly rather than peak/exit state. | PF `2.195 -> 1.230`, delta `-0.965`; spread `0.883`, floor `0.851`; CI `[-1.481,-0.554]`, `p=1.000` for no improvement. | Falsified. | `85b0bb17`. |
| **GPT-spec regime veto** | Veto candidate entries in April-identified weak L2 side/regime cells. | In-sample avoid-union veto appeared to lift PF `1.88 -> 2.26`; forward walk with oracle PF `1.700 -> 1.647` (`-0.054`). Without oracle it lifted `1.14 -> 1.226`, but key seeds fell `-0.277` and `-0.182`. | Research-only, default off; did not clear the forward gate. | `257b87fe`, `92ebaff5`. |

### 1.3 Untested cleanly / still-open threads

| Thread | Why it remains open rather than “had edge” or “no edge” | Exact available evidence | Proper next test | Evidence |
|---|---|---|---|---|
| **Pickles Row-1 as actually traded** | The falsification used SPX with unit-volume VWAP. The journals use ES/NQ VWAP and A/D confirmation, so the tested proxy omitted the stated mechanism. | SPX proxy lost to random; no ES/NQ/A-D result. | Causal cross-market replication with the exact entry clock and executable SPXW accounting. | `dbc5f441`, `41be6a5e`; [Pickles digest](../../v2/docs/pickles_digest.md). |
| **Pickles Row-2 “MAGIC TIME” and Row-3 supply-zone/confluence** | The corpus digest identified them, but Fork C stopped after the Tier-1 day-call null and did not run Phase 2. | Corpus: 12 strategy documents, 167 journals; Row-2 dominated the source vocabulary; supervision feasibility was high for day class (`95%`) and lower for setup/action (`67%` high-confidence). No PF/AUC for Row-2 or Row-3. | Preregister one event-level target at a time; use time-bucket family B/G1 features; do not recycle the failed 10:00 whole-day label. | `dbc5f441`; [supervision feasibility](../../v2/docs/journal_supervision_feasibility.md); `33f3ca28`. |
| **ORC `sigma_pos` direction gate at the open** | It reduced direction errors but was not isolated in a clean economic replay; later late-session versions failed. | Side errors `402 -> 176` (`-56%`), correct entries `117 -> 84`, abstentions `467 -> 726`; retained 68.3% of correct versus 31.5% of wrong. No isolated PF. | Test only as a preregistered interaction/uplift inside H0, never as a late-session trigger clone. | `aff9c804`; correction `2eb4f3af`. |
| **IV/VRP as a conditional interaction** | V1B's gated strategy failed and Control B beat it, so it did not prove a standalone gate. It may still interact with option surface state. | V1B PF `0.975`; best permissive configuration used IV max `0.8` and VRP p75, filtering 17%. | H2/H3/H4 ablation: compare D/E surface/Greek families with and without the feature under the same Stage-1 game. | `621ec097`. |
| **VIX 20-30 / wide-opening interaction** | V2 regime audit found a strong slice, but it was descriptive and small. | VIX 20-30 `n=48`, gap vs A `+7.44 pp`, CI `[+1.79,+14.01]`, `P(gap>0)=99.6%`; calm VIX `n=20`, gap `-3.46 pp`. | VIX is outside canonical non-VIX G1. Treat this as later optional-family evidence after H0-H4, not permission to gate Stage 1. | `2372add2`. |
| **Corrected simulated-L3 and H3a as deployable lifecycle policies** | They had honest offline/forward evidence, but no same-game live parity or fill evidence. Their mechanism belongs to lifecycle Stage 2, not the Stage-1 entry feature ladder. | Sim-L3 per-seed PF `1.950`; H3a forward PF `1.805`; both pre-parity. | Rebuild only after Stage-1 entries are frozen; use full causal paths and exact live-style replay. | `98360a4e`, `1ea34474`; current lifecycle design [Stage-2 proposal](PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md). |
| **L2 combined-v1 regime-aware retrain** | April failure analysis found large wrong-side cells and wrote the combined retrain contract, but the campaign ended during command/oracle plumbing fixes. | Examples: one cell calls PF `0.68` vs puts `1.75`; another calls `3.11` vs puts `0.94`; no completed combined-v1 verdict. | Express the entry-side part through B+G1 in H0 and C/D/E in H1-H4; lifecycle changes remain Stage 2. | `5b8f1b7c`, `eaab3042`, `98352af1`, `bbf9c340`, `81e7b3ff`. |
| **Execution with integer SPXW contracts** | The v2 backtest established an affordability fact, not fill/live edge. | At 0.25% risk, $100k produced 0 trades; $250k 2; $500k 76/179; $1M 176/179; $2M 179/179. | Stage-1 uses the current strict `$10k`, one-account serial game; do not import v2 percentage sizing as evidence. | `d65a491f`; canonical game in [Stage-1 design](PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md). |

## 2. Mapping April prior knowledge into canonical v1 Stage-1 H0-H4

The canonical design defines six entry-family blocks:

- **A:** inherited index/context canaries used to define the option ladder and ATM reference.
- **B:** day/static ladder facts—15-minute session bucket, strike, offset/absolute offset, moneyness, and call/put right.
- **C:** per-slot option-mid dynamics—quantized mid/spot, 1/5/15-step returns, 15-step path range, and realized volatility.
- **D:** near-ATM option composites—straddle/spot, put-call mid ratio, and side smile slope.
- **E:** internal Black-Scholes IV/delta/gamma computed from causal mid, SPX, and time-to-expiry.
- **G1:** live-reproducible non-VIX market context—VWAP gap, session range, 5/15-minute momentum, OMAR, and side-alignment flags.

The hypothesis ladder is H0=`A+B+G1`, H1=`H0+C`, H2=`H0+D`, H3=`H0+E`, and H4=`H0+C+D+E`. Source: [canonical Stage-1 training design](PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md) and [canonical feature definition](../audit/autoresearch/protocol101_canonical_v1_l0_l2_design_audit_attempt001/canonical_feature_definition.json).

| April had-edge or open thread | Canonical expression | Stage-1 hypothesis | Mapping judgment |
|---|---|---|---|
| V2/V2-pruned opening gap + first-15 settlement/acceptance | A/B provide day, side, strike, and time; G1 provides opening/session range, momentum, OMAR, and VWAP-relative context. | **H0** | Closest direct expression. `first15_acceptance` is not a single canonical column, so test the family, not a literal v2 feature clone. |
| Wide-first-15 gate | G1 session-range and momentum features, with B time bucket. | **H0** | Test as model-readable context first. A hard `>=20 bps` gate is prior knowledge, not a frozen Stage-1 rule. |
| Pickles true Row-1 / VWAP support | G1 VWAP-gap and side-alignment fields; B for time/right. | **H0**, incomplete | ES/NQ/A-D confirmation is not in canonical v1. Stage 1 can test the SPX-live-reproducible subset only. |
| Pickles Magic Time | B 15-minute session bucket; G1 contemporaneous market state. | **H0** | Cleanly expressible as time-conditioned state, not as a magic-clock rule. |
| Supply-zone / confluence | G1 session range, momentum, OMAR, VWAP state; B time. | **H0**, approximate | No explicit supply-zone feature. A positive result must come from canonical causal geometry, not a post-hoc zone label. |
| ORC `sigma_pos` direction clue | G1 VWAP gap/alignment plus momentum/OMAR side-alignment; B call/put right. | **H0** | Canonical v1 can test the directional interaction without recreating the failed late trigger tournament. |
| Contract-specific quality/ranking, the April bridge to Protocol051 | B ladder geometry; C own-contract dynamics; D cross-side ATM composites; E IV/Greeks. | **H1/H2/H3**, full **H4** | This is the strongest reason to preserve the ordered ablation ladder rather than jump directly to all features. |
| IV/VRP interaction | E internal IV/Greeks; D straddle and put-call composites approximate implied-move/relative richness. | **H2/H3/H4** | The old VRP gate is not canonical. Test whether D/E add out-of-sample lift; do not freeze the old threshold. |
| Directional V0 versus always-put | B option right plus G1 side-alignment; D put-call ratio and smile; E delta. | **H0**, then H2/H3/H4 | H0 should determine whether simple market-side context carries the broad directional result; richer families may repair weak cells. |
| VIX 20-30 interaction | Not in A/B/C/D/E/G1 non-VIX. | **Outside H0-H4** | Preserve as later optional prior only. Do not contaminate Stage-1 v1 with it. |
| Detach-side entry gate | B/G1 for entry and side at H0; C/D/E test whether option economics improve arbitration. | **H0-H4** | Architectural lesson: protect entry selection from a noisy auxiliary side loss. It does not mandate the old shared encoder. |
| Corrected simulated-L3 target, robust L3 exits, H3a, TCN | No honest Stage-1 family mapping: these use the post-entry causal path. C is only a superficial feature resemblance. | **Stage 2, not H0-H4** | Stage 1 should record compatible path fields, but must not train entry on an unvalidated lifecycle oracle. |
| L2 combined-v1 planned retrain | Entry portion: B/G1 then C/D/E; lifecycle portion: Stage 2. | **H0-H4 + later Stage 2** | Split the April combined plan into the current staged game; do not repeat the old entanglement. |

## 3. Consolidated VWAP findings

1. **The journal mechanism was not cleanly tested.** Pickles' Row-1 setup used ES/NQ VWAP and A/D confirmation. The SPX-only proxy had worse MFE and MAE than random and therefore falsified only the proxy (`41be6a5e`).
2. **Hand-coded SPX VWAP reclaim did not have edge.** V1A PF was `0.918`; V1B PF was `0.975`. Neither beat all controls.
3. **VWAP reclaim was not the learned V2 mechanism.** `opening_gap_pct` was the number-one feature in every fold, `first15_close_position` number two, and `first15_acceptance` always top five. `vwap_reclaim_state` was never top five.
4. **The direct ablation was adverse.** Removing `vwap_reclaim_state` improved mean return `+0.36 pp` and the Control-A gap `+0.30 pp`; the full pruned rerun improved PF `1.291 -> 1.320`. This is why the feature was dropped.
5. **VWAP-relative direction had a limited clue, not a strategy verdict.** The ORC `sigma_pos` gate cut side errors 56%, but late-session VWAP trigger variants went 0/8 and had `0/22` direction matches.
6. **Always-put did not rescue VWAP direction globally.** Its local PF `1.391` reversed to `0.888` over the 13-window methodology, below directional V0 at `1.132`.
7. **Canonical use:** retain VWAP gap and side-alignment as G1 context in H0. Do not reintroduce `vwap_reclaim_state` as a privileged premise or add a hard “VWAP pocket” rule before it clears the same Stage-1 game.
8. **Later pre-parity warning:** fair-contract searches repeatedly used `put_near_after_0940_vwap_m2_10`. On July-September evidence, a relative-rank version had validation PF `1.599` but diagnostic PF `1.099`; score-ceiling-20 had `1.861` vs `1.216`; the broader stable-selection version reached validation PF `4.454` but diagnostic PF `1.008` and failed jitter/spread checks. These are not clean confirmations of VWAP edge.

## 4. Do-not-retest list

These mechanisms should not be rerun unchanged. A new test requires a materially different causal variable, target, data source, or game—not another seed or threshold.

| Do not retest unchanged | Falsification number | What would count as genuinely new |
|---|---|---|
| SPX-only Pickles Row-1 VWAP proxy | MFE +30 bps `8.8%` vs random `10.2%`; MAE `13.5%` vs random `10.9%`. | Exact ES/NQ/A-D inputs and executable SPXW replay. |
| Pickles 10:00 whole-day binary clone on the same 125 rows | Curated PR-AUC `0.166 [0.040,0.500]`, Brier `0.240` vs majority `0.123`. | Event-level Row-2/Row-3 label or new cross-market state. |
| Hand-coded VWAP overextension/reclaim V1A | PF `0.918`; mean `-0.354%`. | Learned canonical G1 context without reclaim privilege. |
| V1B IV/VRP threshold grid | PF `0.975`; Control B PF `1.094`. | D/E family ablation under H2/H3/H4. |
| `vwap_reclaim_state` as a core feature | Drop raised mean `+0.36 pp`, gap `+0.30 pp`, PF `1.291 -> 1.320`. | A different causal VWAP representation, e.g. G1 gap/alignment. |
| Post-hoc V2 score coverage thresholds | All gap CIs crossed zero; `P(gap>0)=0.53-0.61`. | Threshold rule preregistered on a new validation hierarchy. |
| Training only on wide first-15 days | PF `0.991` vs full-train/gated-deploy `1.540`. | Keep full-distribution contrast; test feature-family uplift. |
| Late-session break/reclaim and second-test trigger tournament | `0/8` wins; `0/22` direction matches. | A learned context feature, not another trigger permutation. |
| ATM-IV bottom-two-decile suppression | 21.5% random suppressions matched/exceeded; tree PF `1.122 -> 1.095`. | D/E interaction with locked folds. |
| Always-put as a global default | Rolling PF `0.888` vs V0 `1.132`; local `1.391` was only 20 trades. | Side-aware causal model; regime-specific claim must be preregistered. |
| V1+L3@0.19 / A3 `mfe_norm` drop as PF 2.847 champion | PF CI lower bound `0.680`; overlapping L3; reversed by 780-day test. | Full walk-forward lifecycle retraining after Stage-1 freeze. |
| Morning-snapshot regime classifier | AUC `0.536`, OOS `0.196`, recall `0/3`. | New target/time horizon with adequate positives. |
| Conditional V0/V1 threshold at 0.5 | PF `1.092` vs V0 `1.132`; only `21/432` positives. | Calibrated rare-event design with new held-out regimes. |
| Unified-action fallback and side-contrastive weight sweeps | Unified aggregate PF `1.111-1.115` vs `1.132`; contrastive PF at w=.5 `1.002`. | New representation or objective, not weight nudges. |
| Simulated-L3 iteration-2 global margin rescue | Mean PF `1.705`, min `1.187`, DD `20.9%` vs champion `1.950/1.786/10.1%`. | Structural target/distribution change, not another scalar margin offset. |
| ATM-IV/dollar-score abstention gates | Dollar-score floor PF `1.245 -> 0.712-0.736`. | Explicit uncertainty calibration on new locked blocks. |
| K=2/K=3 + calibration-PF filter | Offline `2.142`, unseen PF `0.72`. | No exception; the filter is proof-failed. |
| Oracle runtime classifier using margin fields | Claimed +24%, then clean PF `1.736/1.829` vs `1.890`; future-label leakage. | Only fields available at the decision timestamp, with a look-ahead audit. |
| H1 relaxed suffix-max target | PF `84.6` vs strict `92.5`; wins `0/61`. | New economic target, not another suffix fraction. |
| H2 regret weighting | PF `2.195 -> 2.032`. | Distinct causal state or loss with preregistered side checks. |
| H3e profitable-only target | PF `2.195 -> 1.591`, CI fully negative. | Keep loser-defense supervision; test Greeks only as an ablation. |
| Side-blind L3 | PF `2.195 -> 1.490`. | Regime-conditional model retaining side. |
| C-Pre continuous magnitude gate | RF `R²=0.059`; 97.9% variance within cells. | New observables, not a new regressor on the same cells. |
| H3c TCN as implemented | Mean PF `1.855` vs H3a `2.019`, `p=0.574`. | More regimes/data and explicit high-volatility defense. |
| H3f regret-regression target | PF `2.195 -> 1.230`, `p=1.000`. | Different target semantics, not tuning this target. |
| Global scalar oracle blending | Mean PF `1.666` vs `1.950`. | Conditional routing with independent validation. |
| GPT-spec avoid-union veto | Forward PF `1.700 -> 1.647`. | New causal evidence and a fresh registered gate. |
| Path-D Wave-1 Phase-1 exit repair family: `recovery penalty`, `rebalanced exit label` via deterministic 50/50 sampling, deterministic fallback, and `risk lower bound calibration` half-LCB ablation | `NO_EDGE` over 1,031 OOF trajectories / five folds. All six trained arms lost to their best comparator; best complete repair H07 delta `-$1,350`, LCB `-$45.06`, first-step exits `91.37%`, p99/top-decile preservation `5.46%/2.01%`; maxT p `0.9672`. | A materially different causal target, representation, horizon, or trading game. Do not rerun this weighting/sampling/fallback/calibration family with new seeds or scalar weights. |
| Path-D Wave-2 60-minute fixed scores, OPRA-only members: full-ladder `surface continuation` and option `microstructure continuation` | `NO_SIGNAL` over 156,950 OOF candidates / 166 sessions / five folds. Net `-$9,821/-$9,856` versus the best comparator `+$7,053`; positive folds `2/5,1/5`; LCBs `-$227/-$260`; maxT p `0.963/0.907`; top decile never best; big-loss share `58-60%` against the charter's 2% limit; all controls failed. Both use OPRA data only, so the ES clock defect below does not touch them, and the three-member family maxT is *more* conservative than an individual test — failing it implies failing individually. | A materially different causal observable, target, position structure, or trading game. Do not retest these composites with new signs, weights, thresholds, adjacent windows, or seeds on entry-v2 OOF. |
| Path-D Wave-2 `cross-market alignment` (SPX/ES/VIX) — **STOPPED, not falsified; clock contract unverified** (superseded 2026-08-04). The automated prior-art check blocks this, which is correct: the block is a *stop pending conditions*, not a claim that the mechanism was validly tested. | Reported `-$9,904`, 2/5 folds, maxT p `0.932`, but the run applied the frozen 2,336 ms emission lag to GLBX ES. That receipt (`shared_emission_lag.json`) contains **no GLBX/ES/MDP3 measurement** — it is ThetaData p99 2,335.2 ms and Databento OPRA CBBO p99 319.5 ms only. Family verdict superseded to `BLOCKED_CLOCK_OR_DATA_CONTRACT`. Likely direction: 2,336 ms is ~7x Databento's own OPRA latency, so ES was probably fed ~2 s *staler* than reality — a handicap, which weakens rather than strengthens a negative. | **This row does not close the mechanism**, and it is not a licence to retest either. Any cross-market proposal must first argue past **Protocol 028**, which tested stitched ES futures VWAP directly and was **rejected** (March `-$80` PF `.928`; Q2 `-$2,130` PF `.552`; Q4 `-$2,790` PF `.890`). There is no positive evidence anywhere in the record that ES helps as a feature. A valid rerun needs a separately hashed GLBX ES completed-minute timing receipt — deliberately **not** purchased, because H3 lost by ~$17,000 against its comparator and ~2 s of ES freshness cannot plausibly close that gap. |

## 5. April 22 rolling-window methodology overhaul

This overhaul controls how all earlier April numbers should be read.

| Before | After | Consequence |
|---|---|---|
| One 20-day OOS slice | 13 disjoint 60-day OOS windows = `780` days (`39x`) | Single-regime “champions” stopped governing the project. |
| Fixed earlier feature basis | 42 Layer-2 features plus 9 causal additions = `51` | Added session cumulative delta, VWAP distance, 5-minute trend, force index, 6/12 returns, smoothed VWAP distance, 30-bar cumulative-delta slope, and session high/low skew. |
| V1 always-put PF `1.391`, V1+L3/A3 up to `2.847` on 20 days | V0 PF `1.132`; V1 PF `0.888` across 780 days | Always-put and PF `2.847` were retired as regime/overlap artifacts. |
| Existing L3 reused freely | L3 deferred when its training days overlapped rolling OOS | Prevented an exit model from laundering overlap into a composed curve. |
| Best-point threshold | Prior-window and robust-plateau threshold selection | Reduced same-window threshold overfit. |

The campaign's honest terminal April belief was therefore not “PF 2-3 is proven.” It was: modest broad entry edge (`1.132`), stronger but still pre-parity composed/lifecycle evidence (`~1.7-1.95` depending on target and seed), serious side/regime fragility, and a need for cleaner data and a clean-slate causal pipeline.

## 6. Why the pre-April-18 exp_1xx grind was abandoned

Everything before 2026-04-18 is excluded from the ledger. The only retained context is that the neural `exp_1xx` loop had become architecture tuning against a signal/target mismatch. The April 18 audit found logistic and tiny-MLP bar-quality AUC only `0.48-0.55` with top-decile precision at base rate; `positive_ev` and `oracle_side_call` Gate-A audits also failed. The correct Bayes behavior on one validation fold was abstention/PF 0, not a broken threshold. That evidence caused the pivot from “another scorer/gate” to explicit mechanical hypotheses, causal controls, and falsification. Evidence: `1f27ae02`, `5585a14c`; [bar-quality handoff](../../v2/docs/handoff_bar_quality_signal_audit_2026-04-18.md).

## 7. How April ideas entered the May protocol farm — pre-parity evidence only

This section is **weaker Tier B evidence**. The protocols were built after the April campaign, and their impressive historical dollars do not prove live-equivalent edge.

| April idea | Later protocol expression | What the later evidence says (pre-parity) |
|---|---|---|
| Learned contract-relative quality rather than a fixed trigger | **Protocol051** adds an intra-minute same-side contract-ranking margin to the A+ surface model. | Ten-seed medians beat Protocol039 in all registered folds: Q2 `$14,140 vs $13,330`, Q3 `$11,620 vs $10,200`, Q4 `$10,970 vs $10,525`, Q1 `$37,845 vs $32,810`. This is the clearest v4 descendant of V2's learned selection result. |
| Separate entry from lifecycle | **Protocol054**, then residual sequence **066**, then **081**. | Protocol054 added full same-contract paths; Protocol066 added causal residual/recovery logic; Protocol081 added Q4 prehistory and deterministic threshold margins. The chain directly reflects April Layer-2/Layer-3 separation and H3a-style causal path state. |
| One-account opportunity cost and wait/take choice | **Protocol092-101**. | Protocol101 added previous-event and rolling-three-event state to an explicit wait/take candidate-set policy and barely cleared the strict serial baseline on all four registered splits; Q4 margin was only `+$140`. |
| Time, side, VWAP, opening range, and momentum as context rather than a single trigger | Fair-contract filters and canonical G1. | Later searches used `put_near`, `after0940`, `vwap_m2_10`, OMAR, 15-minute momentum, near-VWAP, stable-offset selection, and side-alignment. These are descendants of April ideas, not confirmations of their edge. |
| Trade/equity visual inspection | **Protocol113**. | Exported 6,989 replay trades over 369 sessions and 145,728 SPX bars. The ledger explicitly calls them inspection artifacts, not promotion evidence. |

### Why the resulting historical curve failed live parity

The decisive later finding was behavioral, not broker-execution slippage:

| Session | Corrected historical entry signals | Live above-edge minutes | Live trades |
|---|---:|---:|---:|
| 2026-06-04 | 0 | 0 | 0 |
| 2026-06-05 | 8 | 0 | 0 |
| 2026-06-08 | 1 | 0 | 0 |
| 2026-06-09 | 29 | 0 | 0 |

Across June 5/8/9, historical produced `38` entry signals and live produced **zero** above-min-edge minutes. Live was evaluating roughly full sessions with about 41-42 option quotes and fresh quote ages, but every joined historical-entry minute was `below_min_edge` live. Median absolute max-edge deltas were `14.892`, `16.261`, and `24.175` on those dates. See [historical-vs-live comparison](HISTORICAL-VS-LIVE-IBKR-TRADING-COMPARISON.md) and [June 11 parity reference](PROTOCOL101_LIVE_HISTORICAL_PARITY_REFERENCE_2026_06_11.md).

The source-backed failure mechanisms were:

1. historical and live candidate filters were not initially identical;
2. historical quote age and live quote freshness were different semantic objects;
3. live VWAP/market windows used sparse-observation construction while history used one-minute bars;
4. historical option quote, index-bar timestamp, and repaired Greek coupling could differ from near-real-time IBKR snapshots;
5. volume/open-interest and other vendor-sensitive microstructure were not equally observable;
6. the lifecycle model required the full causal path, while an early live bridge called it on a single row; and
7. older live logs lacked full token feature/score traces, so the divergence was provable but not exactly decomposable.

These points are documented in the [May 24 research-program audit](research_program_audit_2026_05_24.md), [parity pause point](PROTOCOL101_PARITY_PAUSE_POINT_2026_06_16.md), and later synchronization handoff. Same-input replay eventually reached `312/312` on June 12 traces, which proves deterministic plumbing for those captured inputs; it does **not** retroactively make cross-vendor historical economics live-equivalent.

### What the fair-contract search says

The request referred to 44 `protocol101_fair_contract_model_search*` audit directories. The current checkout contains **43 matching directories**; the missing 44th directory is `UNKNOWN`. No substitute result is invented.

The base search's best attempt was rejected despite validation PF `1.560`, because diagnostic PF was only `1.201`; all 77 listed base attempts were rejected. The later filters repeatedly made the validation curve look excellent while diagnostic or perturbation behavior failed:

| Pre-parity fair-contract example | Validation PF | Diagnostic PF | Verdict |
|---|---:|---:|---|
| Base best teacher/near-offset attempt 062 | `1.560` | `1.201` | Rejected on diagnostic PF. |
| VWAP-pocket relative-rank attempt 101 | `1.599` | `1.099` | Rejected on diagnostic PF. |
| VWAP-pocket score-ceiling-20 attempt 103 | `1.861` | `1.216` | Rejected on diagnostic PF. |
| Stable-abs-15 near-VWAP attempt 125 | `4.454` | `1.008` | Failed diagnostic PF and spread-widening PnL drift. |
| Momentum-nonpositive attempt 133 | `2.429` | `0.859` | Rejected on diagnostic PnL/PF/trade count. |
| Side-aware momentum attempt 136 | `1.753` | `0.636` | Rejected on diagnostic drawdown/PnL/PF. |
| Aggregate decision-presence near-VWAP attempt 130 | `3.850` | `1.020` | Strict replay failed. |

Evidence: [base fair-contract report](../audit/autoresearch/protocol101_fair_contract_model_search/report.md), [stable-selection report](../audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_stable_selection_robustness/report.md), [momentum report](../audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_momentum_filtered_aggregate_objective/report.md), and [side-aware report](../audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_side_aware_aggregate_objective/report.md).

Interpretation: April's time/side/VWAP ideas did make it into the later model farm, but under a fairer live-reproducible contract their apparent curve was unstable across diagnostic periods and small option-feature perturbations. That is **pre-parity negative evidence about transfer**, not a new April falsification.

## 8. Commit-gap investigation: 2026-04-28 through 2026-05-13

### Finding

The missing protocol-creation commit history is **not recoverable from the ordinary repository evidence inspected here**.

- On `v4/phase-0`, commit `a23e78a6` (2026-05-14 13:22 -0700) has direct parent `367f7926` (2026-04-27 21:51 -0700). There is no intervening commit.
- `a23e78a6` adds only 29 paths: the retrospective [research ledger](../ledger/RESEARCH_LEDGER.md), Protocol126-130 code, and their artifacts. It does **not** add Protocol001-125 implementation/history.
- The ledger nevertheless contains dated narratives for April 29 through May 13, including Protocol051/054/066/081/101/113. Those narratives are surviving retrospective evidence, not a substitute for their missing commits.
- The current working tree contains many of the protocol scripts/artifacts, but several decoder-critical files—such as [Protocol051 scorer](../live/protocol051_surface_edge.py), `run_protocol052_sequential_lifecycle_walkforward.py`, `run_protocol061_sequence_lifecycle_model.py`, `run_protocol101_event_history_policy.py`, [Protocol113 exporter](../scripts/export_protocol101_trade_charts.py), and [Protocol160 runner](../scripts/run_protocol160_protocol101_persistent_paper_trader.py)—have no tracked Git history in this checkout. Their “born” dates below therefore come from the ledger or earliest dated audit, not from a creation commit.

### Places checked

| Source | Result |
|---|---|
| Current branch history | No commits in the gap; direct Apr-27 -> May-14 parent edge. |
| Valid local branches and remote refs | No gap commit containing the protocol farm. `origin/main` has a May-03 public-review commit, but not this history. |
| Reflogs for valid local/remote refs | No 2026-04-28..2026-05-13 protocol-creation commit recovered. |
| `archive/` and `archive_quarantine/` | 672 files inspected by name/content search; no Protocol051/054/066/081/101/113 creation history found. |
| Loose ignored duplicate objects | 455 files with names ending in ` 2`; read-only object inspection found no gap commits. |
| `git fsck --no-reflogs --unreachable` after filtering malformed duplicate filenames | No unreachable/dangling commit for the gap was reported. |

Repository caveat: a malformed ref named `refs/heads/research-ops-transition 2` and the malformed duplicate loose-object filenames make `git log --all` / ordinary `git fsck` noisy. They were not repaired or deleted because this task was read-only apart from this document and its commit. No sealed-vault path was read.

**Honest status:** the protocol narratives and artifacts survived; the granular commit lineage for Protocol001-125 appears lost from the accessible refs/reflogs/archives in this checkout. A separate external clone, backup, or hosting-provider object database would be required to investigate further.

## 9. Protocol-number decoder

| Protocol | What it is | Born when / provenance quality | Current status |
|---|---|---|---|
| **051** | Frozen surface-edge entry component: A+ side-value model plus same-side intra-minute contract-ranking margin. It turns a candidate option surface into an executable-dollar `edge`. | **2026-05-11**, retrospective ledger entry “Protocols 048-051”; creation commit missing in the gap. | Legacy upstream component used by Protocol101 via [Protocol051 scorer](../live/protocol051_surface_edge.py). The registry's surface manifest is a Protocol075 persisted 051/054 stack. Not a standalone paper default. |
| **054** | First full same-contract sequential lifecycle layer on frozen 051 entries; mandatory hard stop, target, and flat behavior remain outside the learned override. | **2026-05-11**, retrospective ledger; creation commit missing. | Legacy fallback lifecycle component. It remained necessary because Protocol066 no-fallback ablation hurt Q3/Q4/March. Persisted in Protocol075; superseded for primary exits by 081 but still part of lineage/fallback semantics. |
| **066** | Ten-seed validation of the Protocol065 residual recovery-penalty sequence model, designed to avoid exiting recoverable convex pullbacks too early. Also the generic current inference implementation name. | **2026-05-12**, retrospective ledger; creation commit missing. | Historical challenger superseded by Protocol081. [Protocol066 inference](../live/protocol066_inference.py) remains the loader/predictor API and explicitly requires the full causal path. |
| **081** | Q4-start residual sequence lifecycle candidate with deterministic `1e-4` threshold margin and persisted 40 seed/fold bundles. | **2026-05-13** (ledger section titled Protocols076-082; one heading is dated May 12 out of order); creation commit missing. | Frozen lifecycle/exit component selected by the current registry. Strong historical evidence, but lifecycle parity still requires full causal-path construction. |
| **101** | Event-history entry policy: explicit wait/take candidate-set model with previous-event and rolling-three-event state over Protocol081 outcomes. | **2026-05-13**, retrospective ledger; creation commit missing. Registered historical medians: Q3 `$59,130 vs $57,790`, Q4 `$92,460 vs $92,320`, Q1 `$95,010 vs $90,000`, March `$41,790 vs $39,100`. | `PAPER_DEFAULT_PROTOCOL101` in the [registry](../promotion/PAPER_TRADING_DEFAULT.json). This means selected paper default, not proven live-equivalent profitability or real-money approval. |
| **113** | Diagnostic exporter for `trades.html`, `equity.html`, and canonical serial-replay trade CSV. | **2026-05-13**, retrospective ledger; creation commit missing. | Current inspection tool [export_protocol101_trade_charts.py](../scripts/export_protocol101_trade_charts.py). The ledger explicitly says its curve is inspection, not promotion evidence. |
| **155** | One-contract live/paper timing and fill-evidence gate; designed to collect evidence while multi-contract sizing remained blocked. | **2026-05-14 commit `f82fff69`**; ledger entry dated 2026-05-15. | Runtime evidence harness. Initial verdict: `blocked_protocol155_no_closed_one_contract_paper_trades_yet`. Not a model or default. |
| **160** | Persistent IBKR paper-trader loop: keeps one broker connection and option ladder alive while routing 051 surface, 101 entry, and 081/066 lifecycle through paper guards. | Earliest dated audit found: **2026-05-20**. No creation commit or Protocol160 ledger entry was found; provenance is therefore weaker than 155. | Current registry entrypoint [run_protocol160_protocol101_persistent_paper_trader.py](../scripts/run_protocol160_protocol101_persistent_paper_trader.py). Paper-account only; runtime flags, broker connectivity, scheduling, and any submit session remain separately authorization-gated. |

## 10. Stage-1 priors to carry forward

1. Start with H0. Opening/session structure, time, right, and G1 non-VIX context have the strongest clean April support.
2. Preserve the family ladder. April's contract-ranking result makes C/D/E plausible, but V1B and the ATM-IV gate show that a hand threshold is not evidence of family uplift.
3. Do not privilege VWAP reclaim. Use G1 VWAP gap/alignment as ordinary causal context.
4. Keep side learning honest. `always_put` is falsified as a global default; side/regime fragility is real and should be reported by side and window.
5. Separate entry from lifecycle. Stage-1 H0-H4 must be scored with the frozen Stage-1 trade-shape game. Learned exits belong to Stage 2 after entry evidence is frozen.
6. Require rolling, serial, affordable, bid/ask-accounted evidence. A 20-day curve, overlapping candidate sum, or displayed equity curve is not sufficient.
7. Treat all Protocol101 historical results as pre-parity until the same causal contract, candidate universe, feature semantics, timing, and full lifecycle path are demonstrated live-style.
8. Do not let the missing May commit lineage create false certainty. When the ledger, current code, and Git provenance disagree, report the gap instead of guessing.

## Evidence index

Primary headline commits:

- Pickles digest `dbc5f441`; SPX proxy `41be6a5e`; Fork C `33f3ca28`.
- V1A `87b429e8`; V1B `621ec097`; V2 `7df452a5`; V2 stress/prune `315dfbbd`; regime/gate `2372add2` and `f4e35501`.
- Layer-2 detach-side `bc85b370`; learned exits `e042e641` through `905c37f3`; Stage C1 `0b2b4539`.
- Rolling overhaul `457aed6b`, `29945a67`, `01af7c9f`.
- Simulated-L3 `75692194`, `98360a4e`; forward proof failure `f23cf3ff`; leakage retraction `4c331b90`.
- H1/H2 `533c10e2`, `c0fdcbe8`; H3a `8b1cc3aa`, `98756b84`, `1ea34474`, `579ef0be`; H3e `ce7bfe76`; side-blind `fe89ac31`; C-Pre `598a381e`; L2 dollar gate `066dd95e`; H3c `bbfd45c0`; H3f `85b0bb17`; veto FW failure `92ebaff5`.
- Clean-slate v4 start `f282aa55`; data decisions `b41a522a`, `fcb75f39`; OptionsDepth deferral/edge-before-burn `367f7926`.

The April campaign ended by converting the research lesson into governance: build a clean causal substrate, buy SPXW CBBO-1m/statistics/definitions and VIX one-minute context, and defer the $250 OptionsDepth dealer-flow purchase until Block-1 microstructure first cleared PF `>1.20` and LCB `>1.05`. The recorded Phase-1 budget fell to about `$1,222`. That is the literal “edge before burn” conclusion, not a claim that the April model itself was ready to trade.
