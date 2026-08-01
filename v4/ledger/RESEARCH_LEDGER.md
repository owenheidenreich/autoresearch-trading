# Research Ledger

Append-only research decisions and experiment outcomes live here. Do not edit
past entries except to correct a typo with a dated correction note.

## Entry Template

```text
Date:
Decision / Experiment:
Reason:
Data Used:
Cost:
Result:
Next Gate:
Owner:
```

## 2026-05-22 EXP_2026_05_22_UNIFIED_ENTRY_LIFECYCLE_SEQUENCE_V1

```text
Date: 2026-05-22
Decision / Experiment: Trained CHALLENGER_UNIFIED_ENTRY_LIFECYCLE_SEQUENCE_V1, historically Protocol209, as a shared neural model for entry-slot value and hold/exit lifecycle value.
Reason: AUDIT_CONTEXT_CALIBRATED_RECENT_GAP_V1 showed the recent miss was dominated by same-entry exit timing, not extra bad entries or slot-blocked winners. The next hypothesis was that one shared sequence objective might reduce churn and premature exits better than separate entry/lifecycle pieces.
Data Used: Existing Protocol194-lineage candidate entries, existing normalized official-context SPXW rows, and existing PAPER_DEFAULT_PROTOCOL101 strict-serial trades. No paid data was downloaded and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision reject_unified_sequence_does_not_beat_paper_default. The unified model improved recent_2026 versus PAPER_DEFAULT_PROTOCOL101 ($14,070 vs $7,450), but failed Q1 2026 ($83,580 vs $95,010) and March 2026 ($17,710 vs $41,790). It also failed the stronger lifecycle baseline on every common scored split. Invariants still passed with zero overlap, unaffordable, or NaN-time rows, so this was a model/objective failure, not accounting.
Next Gate: Reject this unified formulation. Return to full-action candidate-generation feature parity and the stronger CHALLENGER_FULL_ACTION_SURFACE_EDGE_V1 lineage rather than continuing to tweak the joint MLP thresholds.
Owner: Codex
```

## 2026-05-22 AUDIT_CONTEXT_CALIBRATED_RECENT_GAP_V1

```text
Date: 2026-05-22
Decision / Experiment: Ran AUDIT_CONTEXT_CALIBRATED_RECENT_GAP_V1, historically Protocol208, to explain why CHALLENGER_LIFECYCLE_CONTEXT_CALIBRATED_V1 still missed the stronger lifecycle baseline on recent_2026.
Reason: EXP_2026_05_22_LIFECYCLE_CONTEXT_CALIBRATION_V1 beat PAPER_DEFAULT_PROTOCOL101 on common scored splits but still failed the stronger lifecycle baseline on recent_2026. Before changing the model again, the project needed to know whether the gap came from same-entry exit timing, slot-blocked baseline winners, or extra bad challenger trades.
Data Used: Existing Protocol207 challenger trades and lifecycle baseline trades only. No paid data was downloaded, no model was trained, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision recent_gap_dominated_by_same_entry_lifecycle_exit_timing. Same-entry exit changes contributed -$44,185, baseline-only missed/blocked rows contributed +$21,270, and challenger-only extra trades contributed $0, for net -$22,915. The miss was concentrated in calls (-$40,960 net) and midday/late-afternoon exit timing. Puts were positive (+$18,045).
Next Gate: Build CHALLENGER_UNIFIED_ENTRY_LIFECYCLE_SEQUENCE_V1. The repeated failure is not candidate selection alone; the model needs one sequence objective that learns entry, hold, and exit together under the same one-account serial game.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_LIFECYCLE_CONTEXT_CALIBRATION_V1

```text
Date: 2026-05-22
Decision / Experiment: Ran CHALLENGER_LIFECYCLE_CONTEXT_CALIBRATED_V1, historically Protocol207, as a validation-only side/time residual calibration experiment on top of the slot-aware lifecycle model.
Reason: AUDIT_RECENT_GAP_SLOT_AWARE_V1 showed the strongest lifecycle challenger was still giving back recent_2026 PnL in context-specific ways. This experiment tested whether the issue was prediction miscalibration by side/time rather than needing another exit rule or architecture knob.
Data Used: Existing Protocol194-lineage candidate entries, existing normalized official-context SPXW rows, and existing PAPER_DEFAULT_PROTOCOL101 strict-serial trades. No live-paper logs were used for training or promotion scoring.
Cost: $0 incremental paid data.
Result: Decision mixed_context_calibration_beats_paper_default_but_not_lifecycle_baseline. The challenger beat PAPER_DEFAULT_PROTOCOL101 on the common scored splits: Q1 2026 $264,650 vs $95,010, March 2026 $108,830 vs $41,790, and recent_2026 $21,310 vs $7,450. It did not beat the stronger lifecycle baseline on recent_2026: $21,310 vs $25,295. Stress remained positive at $0.10 and $0.25 per side, and serial invariants passed with zero overlap, unaffordable, or NaN-time rows.
Next Gate: Do not change the paper default. Treat this as research-only evidence that the Protocol194 entry stream remains much stronger than PAPER_DEFAULT_PROTOCOL101, while lifecycle calibration alone still does not solve the recent gap. Next work should attribute the recent miss against the lifecycle baseline and then move toward a unified entry-plus-lifecycle sequence objective if the miss is not a simple calibration failure.
Owner: Codex
```

## 2026-05-22 Protocol 206 Protocol202 Research Challenger Freeze

```text
Date: 2026-05-22
Decision / Experiment: Froze Protocol202 as a research challenger packet, with Protocol101 explicitly remaining the paper-trading default.
Reason: Protocol202 is the strongest lifecycle challenger so far because it learns single-slot lifecycle opportunity cost, but it still failed the protected recent_2026 baseline comparison. The project needed a clean boundary before any further architecture work so we do not accidentally promote or keep retuning a mixed result.
Data Used: Existing Protocol202, Protocol203, and Protocol205 artifacts only. No paid data was downloaded, no model was trained, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision freeze_protocol202_as_research_challenger_not_paper_default. Protocol202 beat the matching serial baseline on Q1 2026 by $67,190 median and March 2026 by $27,585 median, but missed recent_2026 by $3,080 median. Protocol205 attributed the recent gap mostly to same-entry exit changes and call-heavy late-afternoon/midday behavior, with no Protocol202-only extra-trade contribution.
Next Gate: Keep Protocol101 as paper default. Use Protocol202 only for offline comparison or no-order shadow analysis. Do not add another lifecycle architecture knob unless the same call-heavy late-afternoon/midday gap repeats by more than $5,000 median on a new locked block or live-paper replay.
Owner: Codex
```

## 2026-05-22 Protocol 205 Protocol202 Recent Gap Attribution

```text
Date: 2026-05-22
Decision / Experiment: Ran a narrow recent_2026 attribution for Protocol202 against the matching frozen Protocol194/081 serial baseline.
Reason: Protocol202 is the strongest lifecycle challenger but missed recent_2026 by a small amount. Before changing architecture or adding another knob, the project needed to know whether the gap came from same-entry exit timing, missed slot opportunities, model-only extra trades, side behavior, time buckets, or a few concentrated days.
Data Used: Protocol202 model serial trades and frozen Protocol194/081 baseline serial trades only. No paid data was downloaded, no model was trained, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision recent_gap_broad_across_combo_seeds. Median combo-seed delta versus matching baseline seed was -$2,575. Net contribution was -$5,900: same-entry exit changes -$4,770, baseline-only missed/blocked trades -$1,130, and Protocol202-only extra trades $0. The gap is mostly calls (-$23,105 net) and late afternoon/midday exit timing; puts were positive (+$17,205 net). Baseline-only rows were all blocked by Protocol202 already being in a position, but their net contribution was small.
Next Gate: Do not add an architecture knob for this small recent miss. Keep Protocol202 as the strongest lifecycle challenger, keep Protocol101 as paper default, and only test a narrow call/late-afternoon lifecycle calibration if the same recent-gap pattern repeats on additional data or live-paper logs.
Owner: Codex
```

## 2026-05-22 Protocol 204 Recurrent Slot-Aware Lifecycle

```text
Date: 2026-05-22
Decision / Experiment: Tested a recurrent GRU holding-state model using the Protocol202 slot-aware lifecycle target.
Reason: Protocol203 showed the slot-aware MLP fixed the major Q1/March lifecycle issue but still missed recent 2026 slightly, suggesting the model might need to read the shape of the post-entry path rather than per-minute summary features alone.
Data Used: Existing Protocol194 candidate entries and existing normalized official-context SPXW quote paths. No paid data was downloaded, no live data was used, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision mixed_protocol204_slot_aware_lifecycle_signal_requires_attribution. The recurrent model improved recent 2026 versus baseline by $2,785 median, but damaged Q1 2026 and March materially: Q1 median -$8,390 vs baseline $202,720, March -$9,010 vs baseline $73,440. Invariants still passed with zero overlaps, unaffordable trades, or NaN time rows.
Next Gate: Do not promote Protocol204. The recurrent architecture helped the recent block but failed broader generalization, so Protocol202 remains the better lifecycle challenger. Next work should either attribute Protocol202's recent gap or move toward a unified entry/lifecycle policy rather than switching architectures based on regime.
Owner: Codex
```

## 2026-05-22 Protocol 203 Protocol202 Mixed Result Attribution

```text
Date: 2026-05-22
Decision / Experiment: Attributed Protocol202's mixed result against the frozen Protocol194/081 serial baseline.
Reason: Protocol202 beat Q1/March but missed recent 2026 by a small amount. Before changing the model again, we needed to know whether the gap came from over-holding, missed slot opportunities, side behavior, or matched trade lifecycle changes.
Data Used: Protocol202 model serial trades and frozen Protocol194/081 baseline serial trades only. No paid data was downloaded, no model was trained, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision protocol202_improves_q1_but_recent_gap_is_missed_slot_opportunity. Protocol202 strongly improved matched trade PnL in Q1 and March and exited much earlier than baseline, but recent 2026 had a small net proxy gap of about -$5,900. The recent gap was not the original over-holding problem; it was a small lifecycle timing/slot miss after the slot-aware correction.
Next Gate: Treat Protocol202 as the strongest lifecycle research challenger, but not a promotion candidate because recent 2026 still fails the baseline comparison. Avoid paper/live replacement until this is confirmed or repaired.
Owner: Codex
```

## 2026-05-22 Protocol 202 Slot-Aware Lifecycle Policy

```text
Date: 2026-05-22
Decision / Experiment: Trained a slot-aware lifecycle MLP using a dynamic-programming target that compares holding the current contract against exiting and preserving the single account slot for future candidate entries.
Reason: Protocol200 learned current-contract continuation but ignored opportunity cost; Protocol201 showed it over-held and blocked profitable later entries. Protocol202 tests the correct lifecycle framing: whether this contract is still worth occupying the only position slot.
Data Used: Existing Protocol194 candidate entries and existing normalized official-context SPXW quote paths. No paid data was downloaded, no live data was used, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision mixed_protocol202_slot_aware_lifecycle_signal_requires_attribution. Protocol202 beat the frozen serial baseline on Q1 2026 and March 2026: Q1 median $269,910 vs $202,720, March $101,025 vs $73,440. It slightly missed recent 2026: $22,215 vs $25,295. PF was high across scored splits and invariants passed with zero overlaps, unaffordable trades, or NaN time rows.
Next Gate: Keep Protocol202 as a research challenger, not paper/live default. Attribute the recent gap before adding another model knob or using it operationally.
Owner: Codex
```

## 2026-05-22 Protocol 201 Protocol200 Failure Attribution

```text
Date: 2026-05-22
Decision / Experiment: Attributed why Protocol200 failed despite the strong lifecycle oracle signal.
Reason: Protocol200 was profitable but far below the frozen serial baseline. We needed to understand whether it was exiting too early, over-holding, or breaking serial slot behavior before changing the objective.
Data Used: Protocol200 model serial trades and frozen Protocol194/081 baseline serial trades only. No paid data was downloaded, no model was trained, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision protocol200_overheld_and_blocked_profitable_later_entries. Protocol200's matched trades lost substantial PnL versus baseline in Q1/March, and it missed many profitable later baseline trades because it held the slot too long. Q1 missed baseline PnL was about $1.45M across model-seed comparisons; March missed about $439.8k.
Next Gate: Replace current-contract-only continuation labels with slot-aware lifecycle labels that account for future candidate opportunities.
Owner: Codex
```

## 2026-05-22 Protocol 200 Lifecycle Continuation Policy

```text
Date: 2026-05-22
Decision / Experiment: Trained the first causal lifecycle continuation model for the frozen Protocol194 entry stream.
Reason: Protocol198 and Protocol199 showed same-side churn and large post-exit continuation labels. Protocol200 tested whether a causal post-entry model could learn hold/exit decisions without hardcoded target, stop, or minimum-hold rules.
Data Used: Existing Protocol194 candidate entries and existing normalized official-context SPXW quote paths. No paid data was downloaded, no live data was used, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision reject_protocol200_no_test_improvement. Protocol200 remained profitable but failed the frozen serial baseline: Q1 median $70,575 vs baseline $202,720, March $25,650 vs $73,440, recent $20,055 vs $25,295. Invariants passed, so the failure was model objective/behavior, not accounting.
Next Gate: Attribute the failure before changing the lifecycle objective.
Owner: Codex
```

## 2026-05-13 Protocol 089 Shadow-Paper Ledger

```text
Date: 2026-05-13
Decision / Experiment: Built and ran Protocol 089, a no-order shadow-paper ledger for frozen Protocol 081 router JSONL.
Reason: IBKR live market-data activation is blocked by the account minimum equity requirement, so the live shadow experiment is paused. The useful no-paid-data replacement is to prove the same router JSONL can be converted into one-contract lifecycle accounting without touching broker order endpoints.
Data Used: Existing Protocol 088 offline router shadow observations only: 1,250 rows over 50 selected March 2026 lifecycle rehearsals. No paid data was downloaded.
Cost: $0 incremental paid data.
Result: Shadow parity passed with 0 failed rows and 0 warnings. The ledger found 0 broker order fields, enforced intended_size=1, enforced SPXW PM contracts, and reconstructed bid/ask-only PnL for 50 closed trades. Closed-trade realized PnL sum was 36,150 with median 260; side mix was 44 calls and 6 puts. Two warnings remain because this input is a selected-trade rehearsal, not a real live stream: 19 trades had rows after an exit/stop/forced-flat action, and max observed concurrency was 8.
Decision: Protocol 089 is useful promotion-readiness infrastructure, not paper/live approval. It should be rerun on the eventual live JSONL with --require-all-closed --enforce-global-one-position.
Next Gate: Wait for IBKR market-data eligibility, then rerun no-order live capture and strict Protocol 089. In the meantime, keep work offline and no-paid-data.
Owner: Codex
```

## 2026-05-22 Protocol 199 Lifecycle Full-Path Oracle

```text
Date: 2026-05-22
Decision / Experiment: Added and ran a full-path lifecycle oracle over the frozen Protocol194 entries.
Reason: Protocol198 showed same-side churn can be harmful, but a churn counterfactual alone does not prove that a neural hold/exit model has enough signal. Protocol199 measures whether the same contract often had material executable continuation value after the frozen Protocol081 exit.
Data Used: Existing Protocol194 serial trades and existing normalized official-context SPXW quote paths. No paid data was downloaded, no model was trained, no live data was used, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision strong_lifecycle_training_signal. Full executable paths were available for all 12,076 trades. The hindsight oracle is not tradable, but it shows large continuation labels: material post-exit continuation appeared in 80.4% to 85.7% of rows across scored splits, and the frozen exit occurred before the path oracle in roughly 64.5% to 66.8% of historical rows. This confirms a large label surface for a causal hold/exit sequence model, while forced-flat deltas remain mixed and warn against naive hard-hold behavior.
Next Gate: Build Protocol200 as an entry-plus-lifecycle sequence training path using causal post-entry state. It must evaluate through one-account serial replay with new hold/exit decisions because changing exits changes which later entries are actually available.
Owner: Codex
```

## 2026-05-22 Protocol 198 Lifecycle Churn Hold Counterfactual

```text
Date: 2026-05-22
Decision / Experiment: Added and ran a diagnostic audit for same-side exit/re-entry churn in the frozen Protocol194 + Protocol081 serial trade stream.
Reason: The user correctly reframed timing fragility as a lifecycle intelligence problem: the model may be taking local exits and re-entering the same directional idea instead of learning whether to hold, exit, cut, or let the trade run.
Data Used: Existing Protocol194 strict serial trades and existing normalized official-context quote paths. No paid data was downloaded, no model was trained, no live data was used, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision lifecycle_churn_failure_mode_detected. With a 30-minute diagnostic re-entry horizon, 2,793 same-side chains covered 8,582 of 12,076 trades. Continuous hold counterfactuals were executable for 2,791 chains. Q3 2025 and Q4 2025 showed harmful churn: holding the first contract through the later same-side exit beat the actual exit/re-entry sequence by $15,080 and $52,545 respectively. Q1 2026, March 2026, and recent 2026 were mixed or favored the re-entry sequence in aggregate, so this is not a hard-hold rule.
Next Gate: Build a true entry-plus-lifecycle sequence training path: flat actions wait/enter call/enter put, holding actions hold/exit, using causal post-entry state such as MFE/MAE shape, PnL velocity, gamma/theta decay, spread/liquidity, and account state. The objective should learn continuation utility rather than hardcoded hold times or fixed percentage exits.
Owner: Codex
```

## 2026-05-22 Protocol 197 Protocol194 Runtime Parity And Latency Harness

```text
Date: 2026-05-22
Decision / Experiment: Built and ran a no-order Protocol194 challenger runtime parity and latency harness while keeping Protocol101 as the live-paper default.
Reason: Protocol194 is the strongest research challenger, but Protocol195/196 showed its edge is entry-timing sensitive. The next gate is therefore runtime parity and latency logging, not another entry-side model knob.
Data Used: Already-collected Protocol189 full-action surface-edge dataset, existing Protocol190/192 model artifacts, and local replay rows only. No paid data was downloaded, no model was trained, no live or paper orders were submitted, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision runtime_parity_latency_harness_ready_protocol101_default_unchanged. Protocol197 replayed 250 recent_2026 decision events through a live-safe Protocol194 path that does not use candidate_exit_dt, candidate_pnl, or future path/exit labels for runtime selection. Runtime JSONL schema validation passed with 0 errors and 0 warnings. Latency was well inside the provisional budget on this local replay: total decision p50 1.189ms, p95 2.711ms, max 4.069ms; model inference p95 0.529ms; candidate validation p95 1.591ms. The harness logged 245 waits and 5 no-order enter decisions, with live_orders_enabled false and broker_endpoint_called false on every row. Report: v4/audit/autoresearch/v4_aplus_hypothesis_197_protocol194_runtime_parity_latency/report.md
Important Caveat: This is a historical replay proxy and does not prove IBKR live-feed latency, entitlement freshness, or paper-order fill quality. It proves the local Protocol194 challenger path can be evaluated quickly and safely without future labels and without touching orders.
Next Gate: Wire this Protocol194 challenger logger into the daily live monitor as no-order shadow-only alongside Protocol101, capture real live candidate freshness and decision-to-order-intent latency, and compare it to the historical replay budget before considering any paper-runtime replacement.
Owner: Codex
```

## 2026-05-21 Protocol 196 Protocol194 Timing Attribution

```text
Date: 2026-05-21
Decision / Experiment: Attributed Protocol194's one-minute timing fragility by split, side, time bucket, premium bucket, spread bucket, score bucket, offset bucket, and exit reason.
Reason: Protocol195 showed the candidate breaks under one-minute entry+exit delay, but the useful research question is whether the problem is exit management, entry timing, a side imbalance, or a specific fragile contract region.
Data Used: Protocol194 serial trades and Protocol195 minute-delay rows only. No paid data was downloaded, no model was trained, no live orders were placed, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision entry_timing_is_primary_fragility. Across all scored blocks, entry +1m delta was sharply negative while exit +1m delta was near flat: March entry -$513,195 vs exit -$3,205; Q1 2026 entry -$1,236,325 vs exit -$32,175; Q3 2025 entry -$599,485 vs exit -$19,905; Q4 2025 entry -$921,360 vs exit +$1,080; recent 2026 entry -$165,885 vs exit -$2,710. Worst aggregate timing groups were mandatory_time_flat exits, 3k-4k premium, 1-2% spreads, post-open morning, and higher score bucket entries. Report: v4/audit/autoresearch/v4_aplus_hypothesis_196_protocol194_timing_attribution/report.md
Important Caveat: This is an execution/timing finding, not proof that the model lacks edge. It says the model's edge is in recognizing short-lived entry windows; paper replacement requires continuous evaluation, fresh quotes, and measured decision-to-order latency.
Next Gate: Prepare Protocol194 as a fast-entry research candidate: build a shared live candidate-generation contract and runtime latency logger before replacing Protocol101. If model work continues offline, avoid hiding the issue with a threshold tweak; test entry-persistence or low-latency-compatible features explicitly.
Owner: Codex
```

## 2026-05-21 Protocol 195 Protocol194 Timing Fragility

```text
Date: 2026-05-21
Decision / Experiment: Audited Protocol194's five-seed selected trades under delayed execution using already-collected normalized one-minute quote paths and existing one-second high-resolution audit files where coverage existed.
Reason: Protocol194 beat the frozen strict-serial Protocol101 baseline, but the project cannot promote a new model unless its edge can plausibly survive live execution timing. Earlier Protocol101 audits showed timing was the main fragility.
Data Used: Protocol194 serial trades, v4 normalized official-context quote paths, and existing protocol101_highres_opra cbbo-1s parquet files. No paid data was downloaded, no model was trained, no live orders were placed, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision timing_fragile: one-minute both-side delay breaks at least one split. The one-minute delay audit had near-complete coverage, but median both-side delayed PnL became negative on every scored block: March 2026 -$27,880, Q1 2026 -$47,310, Q3 2025 -$42,560, Q4 2025 -$43,490, and recent 2026 -$11,530. The asymmetry matters: exit +1m remained positive, while entry +1m broke the model. Partial one-second coverage was only ~18-21% on historical blocks and 0% on recent 2026, but covered rows stayed positive at 1s and 5s and degraded sharply by 30-60s. Report: v4/audit/autoresearch/v4_aplus_hypothesis_195_protocol194_timing_fragility/report.md
Important Caveat: One-minute both-side delay is intentionally harsh for a live bot expected to evaluate and route continuously. This does not reject Protocol194 as a research candidate; it says replacement approval requires a persistent low-latency live runtime and fresh-quote order path, plus more direct 1s/live-shadow evidence.
Next Gate: Do not add more entry knobs yet. The next research question is whether Protocol194's entry edge is specifically a fast-entry edge: build entry-timing attribution by side/time/premium/spread/score and prepare a live/training parity contract that can evaluate and route at one-second cadence.
Owner: Codex
```

## 2026-05-21 Protocol 192-194 Full-Action Five-Seed Confirmation

```text
Date: 2026-05-21
Decision / Experiment: Ran seeds 4-5 for the full-coverage surface-edge two-stage full-action policy, replayed those selected entries through frozen Protocol081 lifecycle exits, and consolidated seeds 1-5 into Protocol194.
Reason: Protocol190/191 was a strong three-seed research signal, but a model candidate should not be treated as demonstrably better without a wider seed check and strict one-account replay invariants.
Data Used: Already-collected official-context processed rows, normalized SPXW quote paths, the Protocol189 full-action surface-edge dataset, frozen Protocol081 lifecycle artifact, and Protocol190/192 selected entries. No paid data was downloaded, no live orders were placed, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Protocol194 decision keep_research_candidate: 5-seed full-action policy survives Protocol081 serial replay. Across five seeds it beat frozen strict-serial Protocol101 on every scored block: March 2026 median $73,440 vs $41,790, Q1 2026 $202,720 vs $95,010, Q3 2025 $81,870 vs $59,130, Q4 2025 $138,895 vs $92,460, and recent 2026 $25,295 vs $7,450. Median PF stayed >= 1.77, stress $0.10 and $0.25 stayed positive on every block, positive seed fraction was 1.00 everywhere, and serial invariants found zero overlap, zero unaffordable trades, zero non-positive premiums, and zero missing timestamps. Report: v4/audit/autoresearch/v4_aplus_hypothesis_194_full_action_surface_edge_5seed_confirmation/report.md
Important Caveat: Protocol190/192 trained on fast executable baseline-exit labels, then Protocol191/193/194 confirmed only the selected entries under frozen Protocol081 replay. This is a materially better research candidate, not yet approval to replace the Protocol101 live-paper default. It still needs timing/delay stress, serial equity/trade visual inspection, and a shared live candidate-generation/runtime contract before paper trading can switch to it.
Next Gate: Stop adding entry-side knobs. Run timing fragility and visual inspection for Protocol194 selected trades, then decide whether to build a direct Protocol081-label training subset or prepare the live runtime parity contract for the new full-action policy.
Owner: Codex
```

## 2026-05-21 Protocol 185-191 Full-Coverage Surface-Edge Recovery

```text
Date: 2026-05-21
Decision / Experiment: Repaired the full-action feature-parity gap and ran a full-calendar research screen for a broader neural entry policy.
Reason: The prior full-action Protocol164/183 tests were not fair: the balanced screen only had five sessions per split, so March 2026 had no events, and the full-action models were missing Protocol101's causal surface `edge` signal. The next valid question was whether a full-ladder model can learn serial entries when it sees complete calendar coverage and the same causal edge features Protocol101 used.
Data Used: Already-collected official-context processed rows, normalized SPXW quote paths, frozen Protocol051/A+ surface artifacts, and frozen Protocol081 lifecycle artifact. No paid data was downloaded and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Protocol185 enriched the 30-session Protocol164 screen with surface-edge features at 100.0% row match, but Protocol186 still failed because the screen coverage was too small. Protocol188 then built a full-calendar baseline-exit full-action screening dataset across 346 collected sessions and 3,516,386 candidate rows: q1_2025 627,540, q2_2025 634,772, q3_2025 613,174, q4_2025 634,960, q1_2026 637,481, recent_2026 368,459. Protocol189 enriched that full dataset with surface-edge features at 99.999% match. Protocol190 trained a two-stage full-action neural policy on this full-coverage baseline-exit screen with seeds 1-3 and cleared the research screen: Q3 2025 median $64,290 PF 1.85, Q4 2025 $138,765 PF 1.69, Q1 2026 $202,720 PF 2.23, March 2026 $73,440 PF 2.04, recent 2026 $15,755 PF 2.29; stress $0.10 stayed positive on every block. Protocol191 replayed the frozen Protocol190 selected entries through frozen Protocol081 exits and strict serial account rules; all 7,719 selected entries had paths, zero serial skips, and the same positive block medians survived.
Important Caveat: Protocol190's training labels came from the fast executable baseline stop/target/25m exit, not from full-candidate Protocol081 lifecycle exits. Protocol191 only proves the selected entries survive Protocol081 replay; it does not yet prove a model trained directly on full-action Protocol081 labels. This is a strong research candidate, not paper-runtime replacement approval.
Next Gate: Run a 5-seed confirmation or train the same full-action architecture on a targeted Protocol081 candidate set. Before replacing Protocol101 in paper trading, produce serial equity/trades artifacts for Protocol191, run timing/delay stress on its selected entries, and define the live candidate-generation contract for this new full-action surface-edge policy.
Owner: Codex
```

## 2026-05-21 Protocol 179-182 Full-Action Expansion And Protocol101 Anchor Overlay

```text
Date: 2026-05-21
Decision / Experiment: Expanded Protocol164 full-action coverage from 3 to 5 sessions per split, reran the full-action value learner, tested nonnegative value thresholds, then screened a frozen Protocol101-anchor plus Protocol175 additive overlay.
Reason: Protocol165/172 may have failed because the full-action screen was too thin. If full-action learning still failed, the alternate research path was to preserve Protocol101 and test whether Protocol175 could add value only when the account was flat.
Data Used: Already-collected official-context processed/normalized rows, frozen Protocol081 lifecycle labels, frozen Protocol172/175 artifacts, Protocol101 serial trades, and recent Protocol101 serial replay. No paid data was downloaded and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Protocol164 expanded to 30 sessions and 289,150 full-action candidate rows with zero skipped candidates. Protocol179 full-action value policy still failed badly versus frozen Protocol101: Q3 median -$835, Q4 $610, Q1 $3,995, March $0, recent $8,600. Protocol180 nonnegative value-threshold replay mostly abstained and failed. Protocol181 Protocol101-anchor overlay improved Q3 median delta by $560 but did not improve Q4/recent and slightly hurt Q1. Protocol182 positive-score overlay added zero trades and was rejected.
Decision: No Protocol179-182 candidate is promotable. Expanded full-action data did not fix the current value architecture; additive overlay value comes from weakly calibrated negative-score regions and is not reliable enough to pursue as-is.
Next Gate: Keep Protocol101 as operational paper default. The next serious neural step should not be another small threshold/overlay tweak; it should redesign full-action learning around richer sequence/state context or wait for broader historical coverage before training a larger unified policy.
Owner: Codex
```

## 2026-05-21 Protocol 177-178 Protocol175 Attribution And Threshold Replay

```text
Date: 2026-05-21
Decision / Experiment: Compared Protocol175 against frozen Protocol101 trade by trade on Q3/Q4, then replayed Protocol175 with validation recall-constrained thresholds.
Reason: Protocol175 nearly closed the Q4 gap but was not promotable. Attribution was needed before adding another model knob.
Data Used: Existing Protocol101 serial trade JSON, Protocol175 serial trade CSV/model artifacts, and Protocol175 serial dataset. No paid data was downloaded and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Protocol177 found Q4 underperformance was concentrated in post-open calls and especially seed 3. The largest Q4 component was flat decision loss: Protocol175 skipped Protocol101 trades while flat, producing a -$21,290 aggregate flat-decision delta across seeds; Q4 total aggregate delta was -$17,510. Protocol178 tested recall floors of 90%, 95%, and 100% without retraining. Best objective recall_100 narrowed/held the Q4 gap but still did not beat Protocol101: Q4 median $91,250 vs Protocol101 $92,460; Q1 and March stayed positive versus Protocol101, recent stayed tied.
Decision: Keep for research only. The failure is not simply threshold selectivity; Protocol175's score/ranking does not consistently recover Q4 post-open call opportunities that Protocol101 captures.
Next Gate: Do not promote Protocol175. Next useful work is either (1) broader full-action Protocol164 coverage so the neural model learns the full ladder directly, or (2) a constrained Protocol101-anchor/additive overlay screen, explicitly treated as research-only until validated chronologically.
Owner: Codex
```

## 2026-05-21 Protocol 167-176 Serial And Full-Action Model Search

```text
Date: 2026-05-21
Decision / Experiment: Ran the next autoresearch loop after Protocol164-166 realignment, stopping after five non-promotable hypotheses.
Reason: The goal was to find a demonstrably better model than frozen Protocol101 under source-of-truth serial account replay, not to improve overlapping candidate diagnostics.
Data Used: Already-collected official-context historical/recent data, Protocol092/163 serial candidate datasets, the Protocol164 full-action 3-session screen, and the already-collected Q4 2024 external serial dataset. No paid data was downloaded and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Protocol167/169/170/171 Protocol163 variants remained research-only; Protocol169 was the best regular serial-candidate candidate but still did not beat frozen Protocol101 across Q4 2025, Q1 2026, March 2026, and recent 2026. Protocol165 full-action classification failed on the 3-session screen despite a positive full-action oracle. Protocol172 value/advantage scoring and Protocol173 nonnegative value-threshold replay also failed. Protocol175 added Q4 2024 prehistory and nearly closed the Q4 gap, but still missed frozen Protocol101 on Q3 2025 (-$930) and Q4 2025 (-$1,210). Protocol176 hidden48 regularization worsened Q4 transfer.
Decision: No new model is promotable over frozen Protocol101 today. Keep Protocol101 as paper-trading default. Keep Protocol175 as useful attribution evidence, not a replacement.
Next Gate: Do Q4/Q3 attribution on Protocol175 versus Protocol101 and/or expand full-action Protocol164 coverage before adding more knobs. The current failure mode is not lack of profitability; it is failure to beat the frozen Protocol101 strict-serial baseline on every locked split.
Owner: Codex
```

## 2026-05-21 Protocol 167-171 Serial Candidate Model Search

```text
Date: 2026-05-21
Decision / Experiment: Ran a no-paid-data serial model search around Protocol163 after Protocol165 full-action infrastructure was added.
Reason: The user requested continuous experiments until the next demonstrably better model. Before spending many hours on exact full-action Protocol081 scoring, test whether the existing serial candidate neural path was capacity-, threshold-, or regularization-limited.
Data Used: Existing Protocol163 serial one-account dataset and existing repaired recent Protocol101 lifecycle paths. No paid data was downloaded and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Protocol167 wider/longer improved recent median PnL to $9,510 versus the repaired Protocol101 recent serial baseline of $7,450, but still lagged Protocol101 on Q3/Q4. Protocol168 threshold-only replay did not fix Q4. Protocol169 smaller/longer was the best regularized candidate: it beat the strict serial baseline on Q3, Q4, Q1 2026, March 2026, and recent 2026, with recent median PnL $8,490 and Q4 median PnL $90,790. Protocol170 added epochs and converged to the same result. Protocol171 hidden48 improved Q4 baseline delta but weakened Q1/recent. None of these replaces frozen Protocol101 because Q4/Q1/March still do not all beat Protocol101's locked medians.
Next Gate: Stop small MLP/threshold nudging. The next serious model improvement must come from Protocol164/165 full-action candidate-universe work or from live timing/fill evidence, not from more hidden-size tweaks on Protocol101-exposed candidates.
Owner: Codex
```

## 2026-05-13 Protocol 090 Strict Shadow Lifecycle Replay

```text
Date: 2026-05-13
Decision / Experiment: Built and ran Protocol 090, a strict serial lifecycle replay over the Protocol 088 no-order router shadow JSONL.
Reason: Protocol 089 proved the offline router stream could be accounted for, but it correctly warned that selected historical path rehearsals can overlap and can keep emitting rows after terminal decisions. A real bot must hold at most one contract and must stop the trade after exit/stop/forced-flat.
Data Used: Existing Protocol 088 offline router shadow observations only. No paid data was downloaded.
Cost: $0 incremental paid data.
Result: The transform converted 1,250 input rows / 50 selected historical path trades into 276 rows / 13 strict serial trades. It removed 137 post-terminal rows and skipped 37 overlapping candidate trades. Strict shadow parity passed with 0 failed rows. Strict shadow-paper checks all passed: no order fields, one contract, SPXW PM contracts, executable bid/ask pricing, all trades closed, terminal action final, and max concurrency 1. The strict sample had realized PnL sum 6,210, median 110, 12 calls, and 1 put.
Decision: Protocol 090 is promotion-readiness infrastructure and the correct future live no-order gate, not new model-performance evidence and not paper/live approval.
Next Gate: Once IBKR live OPRA/SPX/VIX market data is enabled, run no-order live capture and require this same strict lifecycle gate to pass before broker-connected paper trading.
Owner: Codex
```

## 2026-05-13 Protocol 091 Broader Strict Shadow Lifecycle Replay

```text
Date: 2026-05-13
Decision / Experiment: Ran a broader no-order Protocol 081 offline router rehearsal, then applied the Protocol 090 strict serial lifecycle gate.
Reason: The first strict lifecycle pass used the small 50-trade router smoke. Before waiting on IBKR market-data entitlement, test whether the strict live-like semantics still hold on a larger already-collected March 2026 sample.
Data Used: Existing Protocol 081 lifecycle sequence dataset and official-context VIX files already on disk. No paid data was downloaded.
Cost: $0 incremental paid data.
Result: The broader offline router emitted 10,425 rows over 417 path rehearsals and passed no-order parity when the offline context-age allowance was set to 60 seconds for one-minute VIX bars. Strict serial replay converted that into 2,102 rows over 87 one-contract trades, skipped 330 overlapping candidate trades, removed 323 post-terminal rows, and passed all strict checks: no order fields, one contract, SPXW PM contracts, executable bid/ask pricing, all closed, terminal action final, and max concurrency 1. The strict sample had realized PnL sum 27,130, median 180, 57 calls, and 30 puts.
Decision: The promotion-readiness lifecycle infrastructure is behaving correctly on a broader offline sample. This is not new model-performance proof because the input is still historical selected-path rehearsal, not a live candidate stream.
Next Gate: Wait for IBKR live market data, then run the real no-order live capture and require strict lifecycle replay to pass with fresh quotes and the live context-age gate before broker-connected paper trading.
Owner: Codex
```

## 2026-04-29

```text
Date: 2026-04-29
Decision / Experiment: Ran v4_autoresearch_001, a bounded autoresearch loop for the SPXW 0DTE action neural network.
Reason: Test whether the neural no-trade output can directly control one-contract long call/put entries under pre-registered time/risk trials without reward hacking.
Data Used: Existing v4 Jan-Mar 2026 Databento-derived pilot only. January train, first-half February calibration, second-half February selection, March audit-only holdout.
Cost: $0 incremental paid data.
Result: Floor-cleared validation champion was policy1_stop50_target100_hold25m with skip_first30_edge0_max4_stop1000. Selection median PnL 1152, PF 1.159, positive-day fraction 0.50. March audit median PnL 1687, PF 1.181, max drawdown -2358, positive seeds 1.00 across 3 seeds. This is a modeling lead, not a broad data-purchase or live-trading signal.
Next Gate: Improve the decision-aware no-trade loss, then rerun the same pre-registered loop. Do not add new selection knobs or buy broad data until the multi-seed gate clears.
Owner: Codex
```

## 2026-05-03 Protocols 025-027 Loop Stop: Three Failed Hypotheses

```text
Date: 2026-05-03
Decision / Experiment: Ran three no-paid-data follow-up hypotheses after Protocol 024 to test whether Q1 robustness and Q4 upside could be combined without new selection knobs.
Reason: Protocol 024 fixed Q1/March robustness but gave up too much Q4 upside. The loop objective required all audit splits positive, +$50/trade stress positive everywhere, Q1 improvement versus Protocol 018, and meaningful Q4 recovery versus Protocol 024.
Data Used: Existing processed v4 data only: January-March 2026 pilot plus Q1/Q2/Q3/Q4 2025 audit blocks. No paid data was downloaded.
Cost: $0 incremental paid data.
Protocol 025 Result: Rejected. The lighter balanced value multitask loss recovered Q4 strongly (Q4 median PnL 20562, +50 14362), but Q1 stress failed (Q1 median PnL 2100, +50 -3900) and selection was negative.
Protocol 026 Result: Rejected. Greek/timing interaction features with the Protocol 024 loss made Q1 robust even under +100 stress (Q1 median PnL 5370, +50 4370, +100 3370), but over-throttled Q4 (Q4 median PnL 7562, +50 1862, +100 -3838) and only 2/3 Q2 seeds were positive.
Protocol 027 Result: Rejected. Greek/timing interaction features with the lighter balanced loss was the closest blend: March median PnL 16030, Q1 15752, Q3 16234, Q4 12492, and Q4 +100 stayed slightly positive at 92. However Q2 broke (Q2 median PnL 678, +50 -4872, +100 -10422, positive seed fraction 0.67).
Decision: Stop the autoresearch loop per the user's guardrail after 3 consecutive failed hypotheses. Protocol 024 remains the conditional research candidate; Protocol 027 is a useful diagnostic branch but not a replacement champion.
Next Gate: Ask for direction before the next hypothesis. The evidence suggests the next decision should be methodological, not another tiny loss-weight tweak: either strengthen the generalization protocol around Q2 fragility, add a learned regime/context gate trained only on allowed periods, or move toward a richer sequential policy/exits architecture.
Owner: Codex
```

## 2026-04-30 Q3 2025 Validation Data Purchase And Fresh Audit

```text
Date: 2026-04-30
Decision / Experiment: Downloaded, built, and audited the fresh Q3 2025 validation block for the locked Protocol 007 hypothesis.
Reason: Protocol 007 cleared selection, March 2026, and frozen Q4 2025, which justified one modest unseen validation purchase. The next question was whether the locked bce_cost25 + gamma/theta veto survived a fresh block without retuning.
Data Used: Databento OPRA.PILLAR definitions, cbbo-1m, ohlcv-1m, and statistics for 2025-07-01 through 2025-09-30. Closed sessions 2025-07-04 and 2025-09-01 were skipped. Built 64 sessions, 10,069,430 normalized rows, and 22,884 neural decision rows. Q3 was not used for model training or threshold selection.
Cost: Estimated Databento spend $26.7890 under the $45 safety cap.
Consistency: Raw audit passed for Q3 2025, Q4 2025, Q1 2026, and combined 2025-07-01 through 2026-03-31: 189 expected sessions, 189 complete schema sets, 0 missing files, 0 empty/unreadable files. Processed audit passed: 189 expected pkl files, 189 present, 67,562 neural decision rows. Q3 normalized checks found 0 non-SPXW rows, 0 non-PM rows, 0 strike-step violations, and 0 bad bid/ask rows. Non-360 Q3 sessions were 2025-07-03 early close, 2025-07-30 missing four decision minutes, and 2025-09-17 missing one decision minute.
Fresh Audit Result: The locked Protocol 007 hypothesis failed Q3. Selection and March reproduced the prior locked behavior, but Q3 had median PnL 0, median trades 0, positive seed fraction 0.33, and +$50 stress median PnL 0. Only one of three seeds traded Q3 after the bce_cost25 permission layer and gamma/theta >= 0.0025 veto; that seed had raw Q3 PnL 3741 before stress but failed +$25/+50/+100 stress.
Interpretation: The data purchase was useful and clean, but the model signal did not generalize to fresh Q3. The failure mode is not bad raw data; it is unstable seed/permission coverage and over-selectivity across regimes. The previous Protocol 007 purchase approval is superseded and should not be used to justify further broad data purchases by itself.
Next Gate: Stop buying more broad data for now. Diagnose Q3 proposal coverage and permission calibration using existing Q3/Q4/Q1 data. Any next model change must be pre-registered, train only on allowed pre-holdout periods, and survive both Q3 2025 and Q4 2025 plus March 2026 without adding new post-hoc filters.
Owner: Codex
```

## 2026-04-30 A+ Teacher-Margin Economic Selection Protocol 018

```text
Date: 2026-04-30
Decision / Experiment: Ran the post-Q3 autoresearch loop one change at a time after the locked Protocol 007 failure. First audited transparent A+ value/timing rules on fresh Q3, then fixed a masked-loss bug, then tested a direct A+ teacher-margin neural objective and conservative activity caps.
Reason: Determine whether the failure was data quality, absence of edge, or neural/objective mismatch. Preserve anti-overfit discipline by selecting the final candidate from late-February selection under pre-existing domain/economic constraints: post-open/late-afternoon only, positive edge, and low daily activity.
Data Used: Existing clean v4 data for January 2026 train, early-February calibration, late-February selection, March 2026 audit, Q3 2025 fresh audit, and Q4 2025 frozen audit. Then added Q2 2025 as a locked unseen audit block only after the Protocol 018 candidate was frozen.
Cost: $35.8012 incremental Databento estimated spend for Q2 2025, under the $45 cap.
Result: Transparent A+ timing/value rules passed fresh Q3, proving the trader-like pattern/value family was not dead. Neural training had a correctness bug: invalid-token labels could make masked Huber losses NaN. After fixing that, the old A+ neural model learned to stay flat and failed. The kept model change is surface_structure_aplus_teacher_margin, which directly pushes profitable A+ train/calibration tokens above flat. The kept economic trial is policy1 / post_open_late_edge25_max2. Selection median PnL 2380 PF 1.279 over 20 trades. March median PnL 3962 PF 1.320 over 39 trades, positive seeds 1.00. Fresh Q3 median PnL 11394 PF 1.339 over 128 trades, positive seeds 1.00. Frozen Q4 median PnL 20432 PF 1.582 over 126 trades, positive seeds 1.00. Matched random was negative on Q3 and Q4. Q4 stayed positive under +$100/trade stress; Q3 stayed positive under +$50/trade stress but had median -1406 under +$100/trade stress.
Q2 Audit Addendum: Bought and built Q2 2025 as an additional locked unseen audit block after Protocol 018 cleared Q3/Q4/March. Databento estimated spend was $35.8012 under the $45 cap. Closed sessions skipped: 2025-04-18, 2025-05-26, 2025-06-19. Built 62 sessions, 10,373,194 normalized rows, and 21,991 neural rows. The same frozen Protocol 018 economic candidate passed Q2: median PnL 13492 PF 1.462 over 110 trades, positive seeds 1.00, top-day share 0.125. Q2 +$50 stress median PnL 7792; Q2 +$100 stress median PnL 2092; matched random median PnL -4327.
Q1 Partial Audit Addendum: Attempted to buy Q1 2025 next to extend the continuous historical stream backward from Q2 2025. Databento stopped the request with 402 account_insufficient_funds after $9.7690 of logged estimated spend. Complete usable data covers 2025-01-02 through 2025-02-05, with 2025-01-09 and 2025-01-20 skipped as closed sessions; 2025-02-06 is partial raw data only and was excluded from the build. Built 23 complete sessions, 4,336,740 normalized rows, and 8,280 neural rows. The frozen Protocol 018 economic candidate failed this partial Q1 audit: median PnL -7802 PF 0.593 over 46 trades, positive seeds 0.33. Q1 partial +$50 stress median PnL -10102; +$100 stress median PnL -12402; matched random median PnL -646. The runner's generalization champion moved to late_afternoon_edge25_max2, but that switch is rejected because it would tune on the new holdout.
Rejected: Fixed threshold, base gamma-veto, seed ensemble, teacher auxiliary-head, edge50, and max1 hypotheses. Edge50 collapsed March activity; max1 was robust on Q3 stress but failed the evidence floor and had weak Q4 +$100 stress.
Decision: Protocol 018 is useful but no longer a clean broad-data-purchase promotion signal. It survives March 2026 and Q2/Q3/Q4 2025, but fails the partial Q1 2025 audit. Do not promote or tune on Q1 until the missing Q1 remainder is collected and a new protocol is pre-registered.
Next Gate: Restore account credits if continuing paid downloads, complete the Q1 2025 gap from 2025-02-06 through 2025-03-31, then diagnose the Q1 failure without selecting a new rule on that holdout. Promotion toward paper/live trading still requires a 1s/tick path audit on selected trades and broker paper-trade replay.
Owner: Codex
```

## 2026-05-01 Q1 2025 Completion And Protocol 018 Full Audit

```text
Date: 2026-05-01
Decision / Experiment: Completed the Q1 2025 Databento SPXW 0DTE batch and reran the frozen Protocol 018 audit on the full Q1 block without retuning.
Reason: The earlier Q1 attempt stopped at a partial January/early-February block because Databento returned account_insufficient_funds. The partial Q1 failure could have been a sample artifact, so the correct next step was to complete the continuous Q1 stream and score the already-frozen candidate once.
Data Used: Databento OPRA.PILLAR definitions, cbbo-1m, ohlcv-1m, and statistics for 2025-01-02 through 2025-03-31. Closed sessions skipped: 2025-01-09, 2025-01-20, and 2025-02-17. Built 60 sessions, 11,400,663 normalized rows, and 21,599 neural rows. Continuous processed coverage is now present from 2025-01-02 through 2026-03-31, excluding known market closures. One minute on 2025-03-19 at 14:00 ET produced no neural decision row because the option ladder existed but zero contracts passed tradability filters due wide spreads.
Cost: Q1 total estimated Databento spend $28.0888. The incremental completion run from 2025-02-06 through 2025-03-31 cost $18.3198 under the $22 cap.
Result: The locked Protocol 018 economic candidate surface_structure_aplus_teacher_margin / policy1 / post_open_late_edge25_max2 recovered from the partial-Q1 failure but is thin on full Q1: median PnL 2190, PF 1.057, 117 trades, positive seeds 0.67, top-day share 0.134. Q1 +$50 stress median PnL -3810 and +$100 stress median PnL -9810. Matched random median PnL was -2579. The runner's Q1 generalization champion was late_afternoon_edge25_max2 with median Q1 PnL 17448 and PF 1.679, but selecting it is rejected because it would tune on Q1; it is also not uniformly better across other holdouts, with weak Q3 behavior and failed Q2/Q3 stress. A no-new-data 1s path replay of selected March trades covered 25 of 119 trades across five already-purchased CBBO-1s sessions, found 0 sign flips, 1m PnL sum 7540, 1s PnL sum 7660, median diff approximately 0, and p95 absolute difference 192. Q1 attribution shows fragility concentrated in post-open morning trades, especially puts; late-afternoon Q1 is strong but too sparse to promote as a new rule.
Decision: Protocol 018 is a conditional broad-data-purchase research signal, not live-trading or paper-trading approval. It is positive across March 2026 and Q1/Q2/Q3/Q4 2025 and beats matched random, but Q1 stress failure shows the edge is not yet economically robust.
Next Gate: Do not change the locked candidate based on Q1. Before any live-risk step, run 1s/tick path audits on selected trades and design a new pre-registered Q1-fragility diagnostic that preserves the A+ timing/value framing without adding Q1-selected knobs.
Owner: Codex
```

## 2026-04-30 A+ Permission Objective Screen 004/005

```text
Date: 2026-04-30
Decision / Experiment: Screened fixed lower-variance A+ entry-permission objectives, then reran the same screen with threshold selection stressed by an extra $50/trade on late-February selection.
Reason: Protocol 003C improved March but failed Q4 +$50/trade stress. Test whether target definition or stress-aware threshold selection could stabilize the permission layer without changing the base A+ model or adding March/Q4 knobs.
Data Used: Existing v4 Jan-Mar 2026 pilot plus frozen Q4 2025 audit block. Base model is surface_structure_aplus_huber / policy1. No new paid data.
Cost: $0 incremental paid data.
Result: Neither Protocol 004 nor the stress-selected Protocol 005 cleared the broad-data-purchase gate. Best variant remained bce_cost50. Selection median PnL 4122 PF 1.622 over 34 trades; March median PnL 1607 PF 1.197 over 42 trades; Q4 median PnL 3794 PF 1.230 over 97 trades. It beat matched random but failed March +$50 stress with median PnL -1593 PF 0.837, and Q4 +$50 was too thin at median PnL 144 PF 1.008.
Interpretation: The learned permission layer has signal, but raw threshold tuning still lets too many marginal contracts through. More selection PnL is not the right objective by itself; the next step must explicitly veto bad contract economics while preserving the one-contract simulation timeline.
Next Gate: Add deterministic, pre-registered contract-value vetoes over the selected permission stream. Champion selection must be locked on the selection split before March/Q4 scoring.
Owner: Codex
```

## 2026-04-30 A+ Value Veto Protocol 006/007

```text
Date: 2026-04-30
Decision / Experiment: Added deterministic Greek/contract-value vetoes on top of the A+ permission policy. Protocol 006 locked the veto by raw selection +$50 score; Protocol 007 locked by domain priority: gamma/theta economics first, with selection +$50 survival, positive-day stability, and top-day concentration controls.
Reason: The A+ question is not only whether the pattern is present; it is whether the current contract and context make that pattern worth paying the spread/theta/breakeven risk for. Low theta alone was too blunt, so gamma per theta was tested as the primary contract-quality veto.
Data Used: Existing v4 Jan-Mar 2026 pilot plus frozen Q4 2025 audit block. No new paid data. Vetoes only remove trades from already-simulated one-contract permission policies, so they do not introduce overlapping positions.
Cost: $0 incremental paid data.
Result: Protocol 006's raw selection-score lock chose bce_cost50 / theta <= 0.25 and failed Q4 +$50 stress: median Q4 +$50 PnL 144 PF 1.008. Protocol 007's domain-priority lock chose bce_cost25 / gamma_theta_ratio_scaled >= 0.0025 and cleared the broad-data-purchase gate. Selection median PnL 3747 PF 1.862 over 29 trades, +$50 median PnL 2297 PF 1.658. March median PnL 2816 PF 1.440 over 29 trades, +$50 median PnL 2466 PF 1.298. Frozen Q4 median PnL 1222 PF 1.771 over 95 trades, +$50 median PnL 772 PF 1.165. The locked champion beat matched random in selection, March, and Q4.
Stress / Caveats: Q4 +$100 stress failed, and under Q4 +$50 one seed still lost -4040. March +$50 also had one slightly negative seed at -65. The 1s path audit covered 16 of 91 March selected trades across four Fridays; it found zero sign flips, 1s-minus-1m PnL sum -60, median diff 0, and p95 absolute diff 62.5. This supports the label path on audited slices but does not fully de-risk all selected trades.
Decision: Narrow approval to buy a modest additional validation block for this locked Protocol 007 hypothesis. This is not live-trading approval and not approval to spend the remaining credit balance indiscriminately. The next data purchase should be targeted to fresh, unseen validation months and should be scored once with the locked bce_cost25 + gamma/theta veto protocol before any model changes.
Next Gate: Freeze Protocol 007, export a promotion packet, and price the smallest additional Databento/Cboe validation block that can test multiple regimes. Do not retrain or retune on the new block before the first audit.
Owner: Codex
```

## 2026-04-30 Generalization Protocol 002

```text
Date: 2026-04-30
Decision / Experiment: Ran v4_generalization_protocol_002 across the 30-item "things worth adding" hypothesis backlog.
Reason: Test whether richer trader-aware features, full action-surface modeling, and stricter generalization reporting improve the v4 SPXW 0DTE neural prototype without buying more data.
Data Used: Existing v4 Jan-Mar 2026 Databento-derived pilot plus frozen Q4 2025 audit block. January train, first-half February calibration, second-half February selection, March audit-only, Q4 audit-only.
Cost: $0 incremental paid data.
Result: No combination cleared the Protocol 002 gate across March and Q4. The combined screen logged 108 combos: 6 variants x 3 label policies x 6 fixed trials, each across 3 seeds. Best generalization row was surface_structure_base_huber / policy0 / late_afternoon_edge0_max2 with selection PnL 100 PF 1.215, March PnL -2385 PF 0.338, and Q4 PnL 2966 PF 1.594. This is mixed evidence, not a promotion signal.
Next Gate: Keep protocol infrastructure. Do not buy broad data. Next research should add matched random baselines, slippage stress, and regime-stratified diagnostics before further neural architecture expansion.
Owner: Codex
```

## 2026-04-29 Decision-Aware Loss V1

```text
Date: 2026-04-29
Decision / Experiment: Added decision-aware no-trade action loss v1 and reran the same pre-registered autoresearch loop without adding selection knobs.
Reason: Improve the neural no-trade objective by making false-positive trades explicitly costly during training.
Data Used: Existing v4 Jan-Mar 2026 Databento-derived pilot only. Same split protocol as v4_autoresearch_001.
Cost: $0 incremental paid data.
Result: February selection improved for the selected champion: policy1_stop50_target100_hold25m / post_open_late_edge25_max4 had selection median PnL 1656 and PF 1.429. March audit weakened materially: median PnL 160, PF 1.119, 15 trades, positive seeds 0.67. This is not a confirmed improvement over the Huber Q-regression baseline, whose March audit was median PnL 1687, PF 1.181, 42 trades, positive seeds 1.00.
Next Gate: Keep Huber Q-regression as the current baseline. Any no-trade loss v2 must be pre-registered before looking at March and cannot tune weights against the March audit.
Owner: Codex
```

## 2026-04-29 Q4 2025 Frozen Holdout

```text
Date: 2026-04-29
Decision / Experiment: Downloaded and built a targeted Q4 2025 Databento SPXW 0DTE audit block, then scored the existing Jan-Mar champions as frozen holdouts.
Reason: Test whether the promising Jan-Mar champions generalized to a separate market period before spending more credits on broad historical data.
Data Used: Databento OPRA.PILLAR definitions, cbbo-1m, ohlcv-1m, and statistics for 2025-10-01 through 2025-12-31. Built 64 sessions and 22,719 neural decision rows. Q4 was not used to select or tune trials.
Cost: Estimated Databento spend $31.4731 under the approved $80 cap.
Result: The Huber Q baseline champion failed Q4 with median PnL -13691, PF 0.597, max drawdown -14343, 188 trades, positive seeds 0.00. Decision-aware v1 also failed Q4 with median PnL -2337, PF 0.702, max drawdown -4758, 51 trades, positive seeds 0.00. This rejects the current Jan-Mar signal as a broad/generalizable edge.
Next Gate: Do not buy more broad data yet. Preserve remaining credits for narrow, pre-registered audits after model changes. Any next model must survive March 2026 and the frozen Q4 2025 audit without adding new selection knobs.
Owner: Codex
```

## 2026-04-30 Protocol 002 Controls Update

```text
Date: 2026-04-30
Decision / Experiment: Regenerated v4_generalization_protocol_002 with matched random baselines and $25/$50/$100 per-trade slippage stress controls.
Reason: Close the two main anti-overfit gaps in the first Protocol 002 report before deciding whether any hypothesis combination deserves promotion.
Data Used: Existing v4 Jan-Mar 2026 Databento-derived pilot plus frozen Q4 2025 audit block. No new paid data.
Cost: $0 incremental paid data.
Result: No combination cleared the Protocol 002 gate. The best row remained surface_structure_base_huber / policy0 / late_afternoon_edge0_max2: selection PnL 100 PF 1.215, March PnL -2385 PF 0.338, Q4 PnL 2966 PF 1.594. Matched random baselines were negative for the champion's Q4 seed rows, which is encouraging, but the champion failed March and Q4 stress fell to 291 at +$25 per trade and negative at +$50 per trade for seed 11. Structure/VWAP/OMAR features and the action surface remain worth diagnosing, but pressure/Greek feature bundles were rejected for now.
Next Gate: Do not buy broad data. Add regime-stratified diagnostics and proper IV-percentile cells, then pre-register the next model change before another March/Q4 survival run.
Owner: Codex
```

## 2026-04-30 Edge Existence Audit 001

```text
Date: 2026-04-30
Decision / Experiment: Ran a non-neural edge existence audit over transparent, trader-style SPXW 0DTE entry cells.
Reason: Test whether the current v4 data contains any executable long-call/long-put expectancy before making another neural architecture change.
Data Used: Existing v4 Jan-Mar 2026 Databento-derived pilot plus frozen Q4 2025 audit block. Candidate labels use ask-entry / bid-exit with fees. Market-structure context uses the causal v2 SPX/SPY/VIX cache.
Cost: $0 incremental paid data.
Result: The audit found 43 simulated rule pockets passing the two-trades-per-day gate across late-Feb selection, March 2026, and frozen Q4 2025. The dominant theme was call-side participation when SPX/VWAP sigma state was call-aligned, usually with ITM/ATM contracts. Champion: policy2 / calls / ITM 2+ steps / sigma_aligned. Selection PnL 9694 PF 2.292 over 18 trades; March PnL 11632 PF 1.515 over 44 trades; Q4 PnL 8844 PF 1.186 over 128 trades. Same-time random baselines were negative or near-flat, and +$25/trade stress stayed positive in March and Q4. A stricter one-trade-per-day stress did not clear the full gate, though the best rule remained positive in all three periods.
Data Caveats: CBBO-1m matches audited 1s minute-boundary quotes, but intraminute option movement is large, so stop/target path labels are coarse. Normalized contract_multiplier and min_price_increment fields are unusable sentinel/missing values and must not be model inputs. SPX/VIX context is derived/proxy, not yet an official live feed.
Next Gate: Build the next neural target around the surviving transparent theme: causal sigma/VWAP/OMAR state, call/put asymmetry, ITM/ATM contract choice, one-contract position state, and stricter entry timing. Validate with official live-equivalent SPX/VIX context and a narrow 1s audit for the champion rule before considering more data spend.
Owner: Codex
```

## 2026-04-30 Entry Timing Pattern Audit 001

```text
Date: 2026-04-30
Decision / Experiment: Ran a non-neural transfer audit for causal entry timing primitives.
Reason: The edge existence audit showed broad call-side sigma/VWAP alignment, but the next model needs transferable timing patterns rather than regime-specific filters.
Data Used: Existing v4 Jan-Mar 2026 Databento-derived pilot plus frozen Q4 2025 audit block. Patterns were discovered on January plus early February, selected on late February, then scored on March and Q4.
Cost: $0 incremental paid data.
Result: The two-trades-per-day audit found 135 pattern rules passing the transfer gate. Champion was policy0 / call / sigma_trend_continuation / ITM 1 step / good spread: selection PnL 7584 PF 4.901, March PnL 5732 PF 1.699, Q4 PnL 15948 PF 2.023, with positive bucket behavior across time, VIX, and range slices. The strict one-trade-per-day stress still found 71 passing rules. Strict champion was policy1 / call / compression_breakout / late afternoon / ITM 2+ steps: selection PnL 5004 PF 8.536, March PnL 3658 PF 1.550, Q4 PnL 5728 PF 1.668. Both champions beat same-time random baselines and stayed positive under +$25/trade stress.
Surviving Patterns: sigma_trend_continuation, last10_breakout, vwap_pullback_resume, pullback_resume, compression_breakout, omar_retest_bounce, and limited vwap_hold_continuation. These are better next-model targets than broad regime labels.
Next Gate: Build a pattern-aware neural policy that predicts entry timing primitives and abstains unless pattern quality, contract quality, and context agree. Keep March and Q4 frozen. Before any data purchase, verify the strict champion patterns against a narrow 1s stop/target audit and official live-equivalent SPX/VIX context.
Owner: Codex
```

## 2026-04-30 A+ Contract Value Audit 001

```text
Date: 2026-04-30
Decision / Experiment: Added a Greek/value-aware A+ contract audit on top of the surviving entry timing patterns.
Reason: Test the Pickles-style claim that a good 0DTE setup is not only a chart pattern; the current contract must also be worth paying the spread for, with acceptable delta exposure, convexity, theta burden, spread tax, breakeven distance, and IV-relative value.
Data Used: Existing v4 Jan-Mar 2026 Databento-derived pilot plus frozen Q4 2025 audit block. Same discovery/selection/audit split discipline as the entry timing audit. No new paid data.
Cost: $0 incremental paid data.
Result: The strict one-trade-per-day value audit found 143 transfer-gate passing rules. Champion: policy2 / call / last10_breakout / ITM 2+ steps / mid convexity. Selection PnL 5484 PF 2.760 over 8 trades; March PnL 5846 PF 1.587 over 22 trades; Q4 PnL 21900 PF 2.800 over 50 trades. Value grading now separates intuitive contract quality: A+ rows have median delta 0.887, theta burden 0.034, spread tax 0.016, breakeven 0.68 ATR; C/overpay rows have median delta 0.198, theta burden 0.577, spread tax 0.028, breakeven 10.07 ATR.
Interpretation: This is the clearest evidence so far that v4 is not just a different dataset. The useful signal is the intersection of transferable entry timing and contract value. Greeks/value should become explicit model targets and inputs, not brittle hard filters, because a strong timing pattern can sometimes justify heavier theta or wider breakeven.
Next Gate: Build the next neural policy with separate heads for pattern quality, contract value/overpay risk, and abstain/action selection. Keep March and Q4 frozen. Before promotion or more broad data spend, run a narrow 1s stop/target audit around the strict champion patterns and verify official live-equivalent SPX/VIX context.
Owner: Codex
```

## 2026-04-30 A+ Neural Protocol 003

```text
Date: 2026-04-30
Decision / Experiment: Implemented and ran a pre-registered A+ neural surface protocol with explicit timing-pattern and contract-value features.
Reason: Test whether the neural policy improves when it can evaluate both sides of the Pickles-style A+ question: whether the current market pattern is actionable and whether the selected SPXW contract is worth paying the spread/theta/breakeven risk for.
Data Used: Existing v4 Jan-Mar 2026 Databento-derived pilot plus frozen Q4 2025 audit block. January train, early-February calibration, late-February selection, March 2026 audit-only, Q4 2025 audit-only. No new paid data.
Cost: $0 incremental paid data.
Result: Protocol 003 tested two A+ variants across policy0/1/2 and seeds 11/22/33: A+ feature-only and A+ multitask. The selected champion was surface_structure_aplus_huber / policy1 / all_times_edge0_max4. Selection median PnL 726 PF 1.122 over 36 trades; March median PnL 2396 PF 1.217 over 49 trades with 2/3 positive seeds; Q4 median PnL 7291 PF 1.377 over 135 trades with 2/3 positive seeds. The same metrics appeared for the multitask variant, so the current auxiliary objective did not add measurable lift beyond the A+ features.
Controls: The champion beat matched random in selection, March, and Q4. It survived +$25/trade stress in March and Q4, but failed +$50/trade and +$100/trade stress. This is a narrow research promotion, not a live-trading or broad-data-purchase signal.
Interpretation: v4 now shows a concrete improvement over the earlier neural attempts: explicit A+ pattern/value context gives the model a surviving March/Q4 lead. The edge is still thin and friction-sensitive. The next gate is stricter entry throttling, cached A+ feature reproducibility, and a 1s path audit around selected trades.
Next Gate: Do not buy broad data yet. Build cached A+ decision tensors, add stricter trade-throttled simulation/selection, and audit the champion's selected entries against 1s CBBO path data before considering any live-risk step.
Owner: Codex
```

## 2026-04-30 A+ Strict Entry Stress 003B

```text
Date: 2026-04-30
Decision / Experiment: Added cached A+ decision tensors, exported selected contracts, and ran a fixed one-trade-per-day stress test on the Protocol 003 champion.
Reason: Test whether the A+ neural lead survives when forced to be more selective, without searching a new trial grid.
Data Used: Existing v4 Jan-Mar 2026 pilot plus frozen Q4 2025 audit block. Same trained model family as Protocol 003: surface_structure_aplus_huber / policy1. No new paid data.
Cost: $0 incremental paid data.
Result: The fixed max1/day stress failed. Selection median PnL 2432 PF 1.883 over 9 trades; March median PnL -72 PF 0.986 over 17 trades with 1/3 positive seeds; Q4 median PnL 511 PF 1.092 over 46 trades with 2/3 positive seeds. +$25/trade stress failed in both March and Q4. The original max4/day champion remains the better research lead, but it is not selective enough.
1s Audit: Replayed selected March trades on available CBBO-1s audit slices. For the original max4/day champion, 30 audited March trades had zero sign flips, median 1s-minus-1m PnL difference 0, and p95 absolute difference 41. For strict max1/day, 9 audited March trades had zero sign flips, median difference 0, and p95 absolute difference 26. This supports the current label path for audited slices; it does not explain the strict stress failure.
Diagnostic: The first qualifying trade of the day is weak. In March, champion first trades produced -5746 aggregate PnL versus -1193 for later trades. In Q4, first trades produced -2811 while later trades produced 16295. Raw neural score quartiles and simple A+ agreement were not stable enough across selection/March/Q4 to become a throttle by themselves.
Interpretation: Do not implement "first qualifying signal per day" as risk management. The next throttle must learn entry permission more directly, likely as a separate first-trade/skip-now objective that penalizes early marginal entries and rewards waiting for stronger timing/value agreement.
Next Gate: Build an explicit entry-permission head or calibrator trained only on train/calibration/selection, then rerun March/Q4 without adding new post-hoc filters. Keep 1s path audits for selected Q1 trades.
Owner: Codex
```

## 2026-04-30 A+ Permission Protocol 003C

```text
Date: 2026-04-30
Decision / Experiment: Added a learned entry-permission layer on top of the Protocol 003 A+ champion.
Reason: The fixed one-trade-per-day stress failed because first qualifying trades were weak. Test whether a pre-March permission model can choose which A+ proposals are worth entering without using March or Q4 feedback.
Data Used: Existing v4 Jan-Mar 2026 pilot plus frozen Q4 2025 audit block. Base A+ surface model trains on January and early-February calibration. Permission model trains on pre-March base proposals. Thresholds are selected on late-February only. March and Q4 remain audit-only. No new paid data.
Cost: $0 incremental paid data.
Result: The permission layer passed the fixed permission gate. Selection median PnL 2731 PF 2.003 over 32 trades with 3/3 positive seeds. March median PnL 1724 PF 1.368 over 34 trades with 3/3 positive seeds. Q4 median PnL 370 PF 1.193 over 97 trades with 2/3 positive seeds. It beat matched random in selection, March, and Q4.
Stress: March improved materially versus the original Protocol 003 champion under friction: +$25, +$50, and +$100 median stress all stayed positive. Q4 became much thinner: +$25 survived with median PnL 120 PF 1.058, while +$50 and +$100 failed.
1s Audit: Replayed selected March trades on available CBBO-1s slices. 22 audited trades had zero sign flips, median 1s-minus-1m PnL difference 0, and p95 absolute difference 50. This again supports the current label path on audited slices.
Interpretation: This is a marginal but real improvement in entry permission. It reduces trade count and improves March friction robustness, but sacrifices Q4 upside and remains fragile under Q4 +$50 stress. Treat as the next modeling lead, not a promotion to live trading or new broad data spend.
Next Gate: Stabilize the permission layer across seeds and Q4 before buying data. The next pre-registered test should compare a lower-variance permission objective or calibration method while keeping the same March/Q4 audits and 1s replay.
Owner: Codex
```

## 2026-05-01 Protocol 021 Q1 Fragility Diagnostic

```text
Date: 2026-05-01
Decision / Experiment: Pre-registered and ran the no-paid-data Q1 fragility diagnostic for the frozen Protocol 018 A+ candidate.
Reason: Do not tune directly to Q1 2025, but determine whether the full-Q1 weakness has a concrete, trader-aware failure mode: post-open morning put entries where the A+ timing pattern is not economically worth paying spread/theta/breakeven risk for.
Data Used: Existing processed v4 data only: January-March 2026 pilot, Q1/Q2/Q3/Q4 2025 audit blocks, and local cached A+ decision tensors. The frozen candidate remained surface_structure_aplus_teacher_margin / policy1 / post_open_late_edge25_max2 with seeds 11/22/33.
Cost: $0 incremental paid data.
Result: The diagnostic confirmed the pre-registered concentration. Q1 post-open morning puts had median seed PnL -2790, PF 0.880, and 171 all-seed trades, while Q1 post-open calls had median seed PnL 4124, PF 1.275, and late-afternoon trades remained positive but sparse. The same post-open put lens was positive in March 2026, Q2, Q3, and Q4. Q1 post-open put contract-quality features were directionally weaker than the same lens in other audits, especially contract value score, breakeven, and gamma/theta, but this is not a static-veto proof because Q2 post-open puts still worked with mediocre value medians.
Decision: Keep the A+ timing/value framing. Do not switch to a Q1-winning late-afternoon-only rule and do not add a hard Q1 put filter. The evidence supports a learned side-aware contract-quality calibration objective, trained before holdout scoring, that makes the model more selective about put-side overpay risk.
Next Gate: Pre-register the side-aware contract-quality calibration objective. Train it without Q1 labels, then require survival across March 2026 and Q1/Q2/Q3/Q4 2025 with matched-random and slippage-stress controls before calling it an improvement.
Owner: Codex
```

## 2026-05-03 Protocol 022 Side-Aware Contract-Quality Gate

```text
Date: 2026-05-03
Decision / Experiment: Implemented and ran a no-paid-data side-aware contract-quality neural gate on top of the frozen Protocol 018 A+ surface model.
Reason: Test the next hypothesis from Protocol 021: the setup may be good, but the current call/put contract may be too expensive for the setup. The gate learned from pre-holdout proposals only and used late-February selection for threshold selection.
Data Used: Existing processed v4 data only: January-March 2026 pilot plus Q1/Q2/Q3/Q4 2025 audit blocks. No Q1/Q2/Q3/Q4 labels were used in training or threshold selection.
Cost: $0 incremental paid data.
Result: The proposal-level gate improved Q1 and broad raw PnL but damaged Q4 stress. Median results: selection 7244 PF 5.460 over 16 trades; March 3180 PF 1.302 over 31 trades; Q1 9812 PF 1.383 over 106 trades; Q2 17946 PF 2.012 over 102 trades; Q3 22260 PF 1.952 over 126 trades; Q4 5256 PF 1.160 over 112 trades. Q1 +$50 stress improved to 4456, but Q4 +$50 stress failed at -344 and Q4 +$100 failed at -5944.
Decision: Reject Protocol 022 as the next champion. It is a useful diagnostic and a real Q1 improvement, but it sacrifices too much Q4 robustness and fails the slippage-stress gate.
Next Gate: Keep the frozen Protocol 018 candidate. Test whether the side/value lesson works better inside the neural objective rather than as a proposal-level gate.
Owner: Codex
```

## 2026-05-03 Protocol 023 In-Network Side-Quality Margin

```text
Date: 2026-05-03
Decision / Experiment: Added a side-aware put contract-quality margin directly to the A+ neural surface objective and audited it on the same fixed trial without paid data.
Reason: Protocol 022 showed the idea had Q1 signal but was too blunt as an external gate. Protocol 023 tested whether making the model itself more skeptical of marginal put-side overpay contexts would preserve the A+ timing/value edge.
Data Used: Existing processed v4 data only: January-March 2026 pilot plus Q1/Q2/Q3/Q4 2025 audit blocks. Q1/Q2/Q3/Q4 remained audit-only.
Cost: $0 incremental paid data.
Result: The in-network margin fixed the original Q1 post-open put failure but damaged overall robustness. Median results: selection 7256 PF 4.002 over 13 trades; March 2682 PF 2.500 over only 14 trades; Q1 3520 PF 1.146 over 102 trades; Q2 15136 PF 1.549 over 97 trades; Q3 15146 PF 1.529 over 127 trades; Q4 2754 PF 1.073 over 109 trades. Q1 post-open puts improved to 4052 PF 1.447, but Q4 median PnL collapsed versus the Protocol 018 baseline of 20432, and March activity became too thin.
Decision: Reject Protocol 023 as the next champion. Side-aware put quality is clearly relevant, but a direct put-quality penalty over-rotates and harms generalization.
Next Gate: Design a balanced multi-task quality head that predicts pattern quality, contract value, and side-specific overpay risk as auxiliary heads, while preserving the base action-surface ranking. Selection must stay late-February only; all 2025 blocks and March 2026 remain audit-only.
Owner: Codex
```

## 2026-05-03 Protocol 024 Side-Weighted Value Multitask Objective

```text
Date: 2026-05-03
Decision / Experiment: Added a softer side-weighted value multitask objective to the A+ neural surface model and audited it on the same fixed trial without paid data.
Reason: Protocol 022 proved side-aware contract quality had Q1 signal but was too blunt as a proposal gate. Protocol 023 proved a direct put-quality utility margin over-rotated. Protocol 024 tested whether side-weighted auxiliary value supervision could preserve the action ranking while improving robustness.
Data Used: Existing processed v4 data only: January-March 2026 pilot plus Q1/Q2/Q3/Q4 2025 audit blocks. Q1/Q2/Q3/Q4 remained audit-only.
Cost: $0 incremental paid data.
Result: Protocol 024 is the strongest new research branch from this pass. Median results: selection 6842 PF 4.471 over 15 trades; March 9854 PF 2.847 over 21 trades; Q1 8106 PF 1.311 over 106 trades; Q2 6720 PF 1.271 over 104 trades; Q3 16228 PF 1.551 over 127 trades; Q4 11060 PF 1.308 over 110 trades. Positive seed fraction was 1.00 for every split. +$50/trade stress stayed positive across all splits: selection 6392, March 8954, Q1 3506, Q2 2720, Q3 10178, Q4 5060. +$100/trade stress still failed Q1/Q2/Q4. Matched-random controls were negative in every split: selection -860, March -856, Q1 -2579, Q2 -4326, Q3 -1998, Q4 -2800. Existing March CBBO-1s path audit covered 13 of 61 selected trades across 2026-03-13, 2026-03-20, and 2026-03-27; it found 0 sign flips, 1m audited PnL 6994, 1s replay PnL 6754, median diff approximately 0, and p95 absolute diff 96. Trade-level comparison against Protocol 018 confirms Protocol 024 is not merely conservative: Q1 improved by replacing/avoiding weak 018 trades, but Q4 lost 22274 versus Protocol 018 because it missed 128742 of 018 winners while avoiding only 67360 of 018 losers.
Decision: Keep Protocol 024 as a conditional research candidate, not as live/paper approval and not as a full replacement for Protocol 018 yet. It improves Q1/March robustness and fixes the prior Q1 fragility without killing Q4, but it gives up Q2/Q4 upside versus Protocol 018 and is not robust to +$100/trade friction.
Next Gate: Design a balanced objective that keeps Protocol 024's Q1 robustness while recovering Protocol 018's Q4 post-open put upside. Contract-quality medians do not separate Q4 missed winners from Q4 avoided losers cleanly, so the next change needs market-pattern/side/time interaction context, not another blunt value penalty. Do not buy more data without explicit approval.
Owner: Codex
```

## 2026-05-10 No-Fee Objective And Official-Context Gate

```text
Date: 2026-05-10
Decision / Experiment: Removed broker commission from primary labels, added official SPX/VIX context ingestion hooks, audited current context provenance, and reran the current best candidates on no-fee labels.
Reason: Keep the objective focused on the economics that dominate one-contract SPXW 0DTE trades: paying the spread, selecting a contract with acceptable theta/gamma/value, and exiting at executable bid. Separately, determine whether the current SPX/VIX context is promotion-grade or still proxy-only.
Data Used: Existing processed v4 data only: Jan-Mar 2026 plus Q1/Q2/Q3/Q4 2025. No paid market data was downloaded. Existing labels were relabeled by removing the historical $2 round-trip commission adjustment while preserving ask-entry / bid-exit execution.
Cost: $0 incremental paid data.
Implementation: v4 primary labels now default to no broker commission, with optional fee support left available only as an explicit override. Added official-context build arguments and market-cache provenance for SPX/VIX index bars. Added a context-provenance audit that reports official versus derived/proxy context coverage.
Context Audit: Current local SPX/VIX context is not promotion-grade. SPX has 312 sessions and 122460 rows, VIX has 312 sessions and 122379 rows, but official_fraction is 0.000 for both. SPX sources are spxw_put_call_parity and a small ES futures fallback; VIX sources are spxw_0dte_atm_iv and a small VX futures fallback.
Result: Protocol 024 no-fee remains the current robustness candidate. Median results: March 2026 11300 PF 3.151 over 20 trades; Q1 2025 7170 PF 1.214 over 106 trades; Q2 2025 8270 PF 1.265 over 104 trades; Q3 2025 13470 PF 1.440 over 127 trades; Q4 2025 9210 PF 1.266 over 111 trades. Positive seed fraction was 1.00 in every split and +$50/trade stress stayed positive in every split, but +$100 stress still failed Q1/Q2/Q4.
Rejected: Protocol 027 no-fee remains rejected. It had stronger March/Q1/Q3/Q4 headline PnL, but Q2 stayed fragile: 950 median PnL, PF 1.028, 2/3 positive seeds, -4600 under +$50 stress, and -10150 under +$100 stress.
Decision: Do not train the next model on an official-context rebuild yet because official SPX/VIX bars are not present locally. Keep Protocol 024 as the no-fee proxy-context robustness branch. The next promotion-grade step is to obtain official SPX and VIX 1-minute bars for the already-collected windows, rebuild the same dataset without changing model knobs, and rerun Protocol 024/027 unchanged before introducing another hypothesis.
Next Gate: User approval is required before any paid official-context data download. Preferred zero-cost path is for the user to provide official SPX/VIX CSV or Parquet bars for 2025-01-02 through 2026-03-31. If using a paid vendor, request explicit approval with source, date range, products, symbols, estimated cost, hard cap, and why the data is necessary.
Owner: Codex
```

## 2026-05-10 Protocol 028 ES-VWAP Proxy Test

```text
Date: 2026-05-10
Decision / Experiment: Downloaded Databento ES.FUT ohlcv-1m under the approved $5 cap and tested whether a stitched ES continuous-contract VWAP improves the no-fee Protocol 024 candidate.
Reason: The user is not financially prepared for ThetaData SPX/VIX or Databento ES trades yet. ES futures volume is the cheapest plausible VWAP proxy, so test it before committing to higher-cost official context or trade-level futures data.
Data Used: Databento GLBX.MDP3, parent symbol ES.FUT, schema ohlcv-1m, 2025-01-02 through 2026-03-31. Existing SPXW no-fee processed data for Jan-Mar 2026 plus Q1/Q2/Q3/Q4 2025. No VX, VIX, SPX index, or ES trades were downloaded.
Cost: Estimated Databento spend $2.5922, under the approved $5 hard cap.
Data Audit: Wrote 311 ES continuous session files. Median RTH rows were 391; early-close sessions 2025-07-03, 2025-11-28, and 2025-12-24 had 225 RTH rows. Initial stitching briefly selected CME spread symbols on roll days; this was fixed by excluding hyphenated spread symbols and rebuilding from the same raw range. Final selected spread-symbol count is 0. Databento warned that several CME days had degraded condition, including 2025-09-17, 2025-09-24, and 2025-11-28.
Implementation: Added an ES-only downloader with hard-cap cost enforcement and added optional ES VWAP provenance to the market-structure cache. The Protocol 028 model run kept Protocol 024's no-fee variant, policy, trial, seeds, labels, and split protocol fixed; the only test variable was replacing the previous VWAP source with stitched ES OHLCV VWAP.
Result: Reject the direct ES-VWAP replacement. Median results: March 2026 -80 PF 0.928 over 1 trade; Q1 2025 9450 PF 1.996 over 55 trades; Q2 2025 -2130 PF 0.552 over 33 trades; Q3 2025 4540 PF 1.267 over 103 trades; Q4 2025 -2790 PF 0.890 over 71 trades; selection 1770 PF 3.642 over 4 trades. It improved Q1 but failed March, Q2, Q3 stress, and Q4.
Decision: Do not buy ThetaData or ES trade-level data based on this result. ES VWAP may contain conditional information, but it should not replace the current VWAP/structure source wholesale.
Next Gate: If testing ES further, add ES-derived VWAP distance/slope/agreement as separate provenance-marked features while preserving the existing structure features, or run a non-neural diagnostic first. Do not purchase official SPX/VIX or ES trades until a cheaper ES feature test shows broad survival across March and Q1/Q2/Q3/Q4.
Owner: Codex
```

## 2026-05-10 ES-VWAP Data Cleanup

```text
Date: 2026-05-10
Decision / Experiment: Removed the local ES-VWAP experiment input data after Protocol 028 rejected ES VWAP as a direct replacement.
Reason: Keep the active v4 data stack aligned with the current best no-fee Protocol 024 baseline and avoid accidentally rerunning future experiments on a rejected ES-VWAP context.
Data Removed: Full-window Databento ES.FUT ohlcv-1m range files for 2025-01-02 through 2026-03-31, generated ES continuous session files, and the ES-VWAP decision cache. The older one-day ES context-proxy files from April were left untouched.
Cost: $0 additional.
Result: Future default model runs are back to the prior data stack. The Protocol 028 report, decision file, and ledger entry remain as the research record.
Next Gate: Do not spend on ThetaData, ES trades, or more context feeds until an already-collected-data model change shows stronger broad survival. Treat ES only as a rejected direct-replacement path unless a future pre-registered test adds it as a separate diagnostic feature.
Owner: Codex
```

## 2026-05-10 Protocol 029 Sequential Risk Layer

```text
Date: 2026-05-10
Decision / Experiment: Built and ran the first sequential risk-management layer with Protocol 024 frozen as the entry baseline.
Reason: Move toward a real 0DTE bot by separating entry timing from in-trade management. The model needs to learn whether an open one-contract long option should be held, exited, or stopped out using causal position-state features rather than only entry labels.
Data Used: Existing no-fee v4 processed data only: Jan-Mar 2026 plus Q1/Q2/Q3/Q4 2025. No paid data. No ES/VX/ThetaData. The risk model trained on train/calibration one-minute option paths and selected one exit configuration on February selection only.
Cost: $0 incremental paid data.
Implementation: Added Protocol 029 sequential risk runner. The frozen entry layer is Protocol 024: surface_structure_aplus_side_value_multitask / policy1 / post_open_late_edge25_max2. The risk model predicts future headroom from current PnL, MFE/MAE, PnL velocity, realized volatility, MFE decay, current Greeks/G Greek drift, theta burden, time left, and entry pattern/value context. Hard stop and forced flat remain mandatory.
Selected Risk Config: headroom_le_150_minhold_1, selected on February selection only.
Result: Reject as a replacement for Protocol 024. Dynamic exit improved Q1 materially but sacrificed too much right-tail upside. Median dynamic results: selection 4280 PF 15.759 over 16 trades; March 5500 PF 3.632 over 20 trades; Q1 16370 PF 3.009 over 106 trades; Q2 8130 PF 1.627 over 104 trades; Q3 12150 PF 2.057 over 127 trades; Q4 3490 PF 1.163 over 111 trades. Q4 +50 stress failed at -2660, and dynamic March/Q3/Q4 PnL were below the Protocol 024 baseline.
Interpretation: Position-state features contain real signal: Q1 improved from 7170 to 16370 and PF improved broadly. But the unconstrained headroom exit is too aggressive and cuts profitable trades too early. This is useful architecture evidence, not a promotion.
Decision: Keep Protocol 024 as the frozen baseline. Reject Protocol 029 as the next champion. Continue sequential risk research with a constrained exit layer rather than a generic headroom exit.
Next Gate: No paid data. Pre-register Protocol 030 as a constrained sequential exit: model exits are only allowed in negative-PnL or meaningful MFE-giveback contexts; otherwise winners continue under the original stop/target/time behavior. Require March and Q1/Q2/Q3/Q4 survival, +50 stress, and no major March/Q4 upside collapse.
Owner: Codex
```

## 2026-05-10 Protocol 030 Constrained Sequential Risk Layer

```text
Date: 2026-05-10
Decision / Experiment: Tested a constrained sequential risk layer with Protocol 024 frozen as the entry baseline.
Reason: Protocol 029 showed causal position-state features had signal but cut too much upside. Protocol 030 limited early exits to losing trades or meaningful MFE giveback contexts so winners would mostly continue under the original stop/target/time lifecycle.
Data Used: Existing no-fee v4 processed data only: Jan-Mar 2026 plus Q1/Q2/Q3/Q4 2025. No paid data. No ES/VX/ThetaData. The risk model trained on train/calibration one-minute option paths and selected one exit configuration on February selection only.
Cost: $0 incremental paid data.
Implementation: Extended the sequential risk runner with parameterized loop/report IDs and a loss_or_giveback exit constraint. The selected configuration was headroom_le_150_minhold_5_gb50_gbfrac35, requiring at least 5 minutes held and allowing model exits only when current PnL was negative or MFE >= 100 with at least $50 and 35% giveback.
Result: Reject as a replacement for Protocol 024. Median dynamic results: selection 3450 PF 2.139 over 16 trades; March 4490 PF 2.564 over 20 trades; Q1 4280 PF 1.271 over 106 trades; Q2 -4090 PF 0.768 over 104 trades; Q3 7680 PF 1.466 over 127 trades; Q4 -10570 PF 0.623 over 111 trades. Q1/Q2/Q4 +50 stress failed, and Q2/Q4 outright failed.
Interpretation: The risk features still contain information, but broad MFE-giveback exits are not robust. The model is cutting too many trades that still need room to express their convex payoff, especially in Q2 and Q4.
Decision: Keep Protocol 024 as the frozen baseline. Reject Protocol 030 as the next champion.
Next Gate: No paid data. Pre-register Protocol 031 as loss-only damage control: the risk model may exit early only when current PnL is negative and predicted future headroom is poor. Winners continue under the original stop/target/time lifecycle.
Owner: Codex
```

## 2026-05-10 Protocol 031 Loss-Only Risk Layer

```text
Date: 2026-05-10
Decision / Experiment: Tested a loss-only sequential risk layer with Protocol 024 frozen as the entry baseline.
Reason: Protocol 030 showed broad MFE-giveback exits were not robust and still clipped too much convex payoff. Protocol 031 removed giveback exits entirely so the risk model could intervene only while the open contract was losing.
Data Used: Existing no-fee v4 processed data only: Jan-Mar 2026 plus Q1/Q2/Q3/Q4 2025. No paid data. No ES/VX/ThetaData. The risk model trained on train/calibration one-minute option paths and selected one exit configuration on February selection only.
Cost: $0 incremental paid data.
Implementation: Added a loss_only exit constraint to the sequential risk runner. The selected configuration was headroom_le_75_minhold_5_lossonly, requiring at least 5 minutes held and current negative bid-to-entry PnL before a model exit could trigger.
Result: Reject as a replacement for Protocol 024. Median dynamic results: selection 2500 PF 1.636 over 16 trades; March 4620 PF 2.200 over 20 trades; Q1 -4850 PF 0.803 over 106 trades; Q2 -6710 PF 0.697 over 104 trades; Q3 3270 PF 1.123 over 127 trades; Q4 -11520 PF 0.623 over 111 trades. Q1/Q2/Q4 outright failed and Q3 +50 stress failed.
Interpretation: Many SPXW 0DTE long-option winners appear to spend time underwater before their convex payoff emerges. A generic loss-only headroom exit is still too eager and destroys the baseline edge.
Decision: Keep Protocol 024 as the frozen baseline. Reject Protocol 031 as the next champion.
Next Gate: Stop the sequential-risk loop after three consecutive failures: Protocol 029 unconstrained headroom exit, Protocol 030 loss-or-giveback exit, and Protocol 031 loss-only exit. Choose a new direction before adding another exit knob. Candidate directions: improve entry-side contract quality, add official SPX/VIX context later, or redesign lifecycle learning as a value-of-wait/continuation objective rather than an early-exit override.
Owner: Codex
```

## 2026-05-10 Protocol 032 Ladder-Relative Contract Quality

```text
Date: 2026-05-10
Decision / Experiment: Added ladder-relative contract-quality features to the entry model and evaluated the same Protocol 024 policy/trial/seeds on existing no-fee data.
Reason: A trader compares a candidate SPXW contract against nearby same-side strikes before paying the ask. Existing features described absolute spread/theta/gamma/breakeven economics but did not explicitly tell the model whether the contract was expensive or cheap versus the local $5 ladder.
Data Used: Existing no-fee v4 processed data only: Jan-Mar 2026 plus Q1/Q2/Q3/Q4 2025. No paid data. No ES/VX/ThetaData.
Cost: $0 incremental paid data.
Implementation: Added aplus_relative_quality token mode with same-side ask/spread/theta/value ranks, side and neighbor ratios, local premium/gamma curvature, and a relative overpay score. Registered surface_structure_aplus_relative_quality_side_value_multitask and ran policy1 / post_open_late_edge25_max2 with seeds 11/22/33.
Result: Reject as a replacement for Protocol 024. Median results: selection 4210 PF 2.148 over 11 trades; March 3510 PF 4.514 over 5 trades; Q1 13700 PF 2.333 over 62 trades; Q2 5700 PF 1.692 over 39 trades; Q3 17910 PF 1.801 over 115 trades; Q4 10910 PF 1.479 over 92 trades. +50 stress stayed positive everywhere, but selection, March, and Q2 were below Protocol 024, and March traded too little.
Interpretation: Ladder-relative features are directionally useful and improved Q1/Q3/Q4, but the model became too selective. This is evidence for contract-quality signal, not a new champion.
Decision: Keep Protocol 024 as the frozen baseline. Reject Protocol 032 as the next champion.
Next Gate: Test contract quality as a proposal-level gate on top of the frozen Protocol 024 entry model. This keeps the baseline entry representation intact while asking a narrower question: when Protocol 024 wants to buy, is this specific contract worth paying the ask for?
Owner: Codex
```

## 2026-05-10 Protocols 033-036 Entry-Side Contract Quality Loop

```text
Date: 2026-05-10
Decision / Experiment: Ran four entry-side contract-quality hypotheses against the frozen Protocol 024 no-fee baseline.
Reason: The user directed the research loop to focus on making the entry model learn whether the current setup is good but the specific contract is too expensive to buy at the ask.
Data Used: Existing no-fee v4 processed data only: Jan-Mar 2026 plus Q1/Q2/Q3/Q4 2025. No paid data.
Cost: $0 incremental paid data.
Protocol 033: Learned side-aware contract-quality gate on Protocol 024 proposals. Rejected as replacement. It improved selection/Q2/Q3/Q4 but damaged March and Q1.
Protocol 034: Trade-preserving quality gate requiring at least 80% February selection trade retention. Strongest challenger so far. It beat Protocol 024 in selection, Q1, Q2, Q3, and Q4, with stronger +50 stress in Q1/Q2/Q3/Q4. It failed replacement because March dropped from 11300 to 7040 and trades dropped from 20 to 17.
Protocol 035: Full-retention quality gate requiring 100% February selection trade retention. Rejected. It improved March versus Protocol 034 but gave back Q1 and still did not beat Protocol 024 in March.
Protocol 036: Near-tie contract-quality reranker. Rejected. Selection chose baseline_no_rerank, meaning simple value/spread/theta reranking did not improve Protocol 024's top contract choice.
Decision: Keep Protocol 024 as the frozen baseline. Keep Protocol 034 as the best challenger for attribution, not as a replacement.
Next Gate: Stop adding entry-quality knobs until we inspect why Protocol 034 improves Q1/Q2/Q3/Q4 but hurts March. The next useful step is trade-set attribution between Protocol 024 and Protocol 034, then a pre-registered March-safe contract-quality objective if the attribution shows a transferable failure mode.
Owner: Codex
```

## 2026-05-10 Protocol 024 vs 034 Attribution

```text
Date: 2026-05-10
Decision / Experiment: Attributed Protocol 034's March 2026 underperformance against the frozen Protocol 024 no-fee baseline.
Reason: Protocol 034 is the strongest entry-side contract-quality challenger but it cannot replace Protocol 024 until the March damage is understood.
Data Used: Existing Protocol 024 selected-trade enriched files, Protocol 034 selected trades, and newly dumped Protocol 034 proposal attribution files. No paid data.
Cost: $0 incremental paid data.
Implementation: Added proposal attribution dumps to the side-aware quality gate runner and added v4/scripts/attribute_protocol024_vs034.py. The attribution compares exact selected-trade overlap, baseline-only trades, candidate-only trades, quality rejection reasons, side/time exposure, and feature medians.
Result: Protocol 034's March all-seed PnL was 19670 versus Protocol 024's 33000, a -13330 delta. Exact overlap was 42 trades. Protocol 034 removed 18 Protocol 024 trades worth +14140; all 18 were quality_rejected. The removed set contained 12 missed winners worth +17790 and 6 avoided losers worth -3650. Added trades after cooldown shifts contributed only +810 net.
March Finding: The damage was concentrated in puts, especially post-open morning puts. Protocol 024 post-open puts: 41 trades, +21620. Protocol 034 post-open puts: 32 trades, +10860. The top missed winners were all puts, often with quality scores just below threshold and superficially weak value/breakeven/gamma-theta features.
Interpretation: Protocol 034's failure is a false-negative contract-quality problem, not a side flip. The gate rejected contracts that looked mediocre by static economics but were actually convex winners. Hard proposal-level contract-quality vetoes are risky for 0DTE because the same "expensive" contract can be the correct one when the timing pattern is strong enough.
Decision: Keep Protocol 024 frozen. Keep Protocol 034 only as a challenger/diagnostic.
Next Gate: Do not add another hard quality gate. The next model change should preserve Protocol 024 March trade availability and use contract quality as a soft representation or confidence-shaping feature. If a rejection rule is tested, it must be March-safe by construction and pre-registered against the same locked splits.
Owner: Codex
```

## 2026-05-10 Protocol 037 Timing-Frozen Contract Selector

```text
Date: 2026-05-10
Decision / Experiment: Tested a timing-frozen same-side contract selector on top of Protocol 024.
Reason: Protocol 034 attribution showed hard quality gates rejected March put convex winners. Protocol 037 therefore preserved Protocol 024's entry minute, side, cooldown, and trade count, and only allowed the model to choose a different same-side SPXW contract.
Data Used: Existing no-fee v4 processed data only: Jan-Mar 2026 plus Q1/Q2/Q3/Q4 2025. No paid data.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_timing_frozen_contract_selector_protocol.py. The selector trained on Protocol 024 train entry events with calibration early stopping. Config selection happened on February selection only across selector blend weights and candidate-band restrictions.
Result: Reject as a replacement for Protocol 024. Selection chose baseline_original_contract, so the selected configuration made no strike changes and exactly matched Protocol 024: selection 6460, March 11300, Q1 7170, Q2 8270, Q3 13470, Q4 9210 median PnL.
Interpretation: Under the current small training window, a separate learned same-side selector did not improve Protocol 024's original contract choice. Together with Protocol 034 attribution, this suggests Protocol 024 already captures much of the useful strike-choice signal, while downstream gates/selectors introduce false negatives or fail selection.
Decision: Keep Protocol 024 as the frozen baseline. Do not promote Protocol 037.
Next Gate: Stop treating contract quality as a downstream veto or separate selector. The next entry-side model change should be a soft in-network objective that preserves Protocol 024's March trade availability while using quality features to shape confidence, or the research protocol should move to broader walk-forward training using already-collected data.
Owner: Codex
```

## 2026-05-10 Protocol 038 Soft Quality Walk-Forward

```text
Date: 2026-05-10
Decision / Experiment: Tested a combined no-paid-data Protocol 038: soft in-network contract-quality confidence plus broader expanding walk-forward training.
Reason: Protocol 034 attribution showed hard quality gates rejected March convex winners, while Protocol 037 showed a downstream same-side selector did not improve Protocol 024. The next hypothesis was to use contract quality softly inside the network while also moving beyond the tiny train window into a broader walk-forward protocol.
Data Used: Existing no-fee v4 processed data only: Q1/Q2/Q3/Q4 2025 and Q1 2026. No paid data. No official SPX/VIX, no ES VWAP, no 1s/tick expansion.
Cost: $0 incremental paid data.
Implementation: Added surface_structure_aplus_soft_quality_confidence and v4/scripts/run_soft_quality_walkforward_protocol.py. The protocol compared the Protocol 024 family, surface_structure_aplus_side_value_multitask, against the soft-quality objective using policy1 / post_open_late_edge25_max2, seeds 11/22/33, 8 epochs, 10 validation days, and expanding folds: train Q1 test Q2, train Q1-Q2 test Q3, train Q1-Q3 test Q4, train Q1-Q4 test Q1 2026 with a March slice.
Result: Reject the soft-quality candidate. Median candidate results: Q2 -480 over 3 trades, Q3 0 over 0 trades, Q4 0 over 0 trades, Q1 2026 1460 over 5 trades. Candidate beat the baseline in 0/4 folds, failed +50 stress all-fold positivity, failed matched-random all-fold superiority, and damaged March: 2050 over 5 trades versus baseline-family 13100 over 40 trades.
Baseline Observation: The broader Protocol 024-family baseline itself was strong: Q2 11330 PF 1.427 over 92 trades, Q3 12660 PF 1.378 over 126 trades, Q4 19020 PF 1.619 over 123 trades, Q1 2026 31380 PF 1.892 over 109 trades, all with positive +50 stress and negative matched-random controls. Q2 positive seed fraction was only 0.67, so this is not promotion-grade yet.
Interpretation: The soft quality loss still acted like a trade suppressor. The useful lesson is that clean additional history and a proper expanding walk-forward protocol appear more valuable than another contract-quality penalty.
Decision: Do not promote Protocol 038's soft-quality variant. Keep Protocol 024 as the frozen reference baseline. Treat the broader walk-forward Protocol 024-family result as the next research direction to validate.
Next Gate: Run no-paid-data stability and attribution on the broader baseline before adding another model knob: more seeds, side/time/contract exposure, trade-set attribution versus frozen Protocol 024, and Q2 seed fragility inspection.
Owner: Codex
```

## 2026-05-10 Protocol 039 Broader Baseline Validation

```text
Date: 2026-05-10
Decision / Experiment: Ran no-new-model validation of the broader expanding-history Protocol 024-family baseline.
Reason: Protocol 038 rejected the soft-quality objective but showed a strong broader baseline. Protocol 039 tested whether that broader baseline survives more seeds, side/time/contract exposure inspection, attribution versus frozen Protocol 024, and Q2 fragility review before any new model knob is considered.
Data Used: Existing no-fee v4 processed data only: Q1/Q2/Q3/Q4 2025 and Q1 2026. No paid data. No official SPX/VIX, no ES VWAP, no 1s/tick expansion.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_protocol039_broader_baseline_validation.py. Kept surface_structure_aplus_side_value_multitask, policy1 / post_open_late_edge25_max2, 8 epochs, 10 validation days, and expanding folds. Ran seeds 11/22/33/44/55/66/77/88/99/111. Saved enriched selected trades, side/time/moneyness exposure, common-seed attribution versus frozen Protocol 024, and Q2 seed diagnostics.
Result: Pass as a research baseline, not paper/live approval. Median results over 10 seeds: Q2 2025 9100 PF 1.341 over 97 trades, +50 stress 4275, positive seeds 0.80; Q3 13595 PF 1.414 over 126 trades, +50 stress 7320, positive seeds 1.00; Q4 18230 PF 1.577 over 126 trades, +50 stress 12005, positive seeds 1.00; Q1 2026 27095 PF 1.765 over 114 trades, +50 stress 21340, positive seeds 1.00. Every fold beat matched random in every seed.
Exposure: The strongest exposure was post-open morning puts: 446740 PnL, PF 1.916 over 1778 trades. Post-open calls were also positive. Late-afternoon calls and puts were positive but weaker.
Q2 Fragility: Q2 remains the weak fold. Two of ten seeds were negative and three of ten failed +50 stress. Losing-seed damage concentrated in late-afternoon puts: -4920 PnL, PF 0.673 over 36 trades. Q2 post-open calls and puts were positive even in losing seeds.
Attribution: Common-seed attribution versus frozen Protocol 024 is mixed: broader baseline underperformed frozen in Q2 (-4310) and Q3 (-9540), but improved Q4 (+27700) and March 2026 (+7970). Exact overlap was low, so broader training materially changes participation rather than replaying frozen Protocol 024.
Decision: Keep the broader Protocol 024-family baseline as the current research baseline. Do not add another model knob yet.
Next Gate: Promotion-readiness validation: path-level 1s/tick checks on selected broader-baseline trades, official SPX/VIX context rebuild when financially approved, and targeted Q2 late-afternoon put diagnostics before any model change.
Owner: Codex
```

## 2026-05-10 Protocol 040 Promotion-Readiness Validation

```text
Date: 2026-05-10
Decision / Experiment: Ran promotion-readiness validation for the current Protocol 039 broader baseline using only already-collected data.
Reason: Protocol 039 passed as a research baseline. The next gate was not another model change, but validation of execution-path realism, official context provenance, and Q2 fragility before any paper/live approval.
Data Used: Existing Protocol 039 selected trades, existing Databento OPRA CBBO-1s audit slices for ten Q1 2026 sessions, existing local SPX/VIX context files, and existing no-fee processed data. No paid data.
Cost: $0 incremental paid data.
Implementation: Ran v4/scripts/audit_selected_trades_1s_path.py against Protocol 039 Q1 selected trades and v4/scripts/audit_context_provenance.py. Added v4/audit/autoresearch/v4_aplus_hypothesis_040_promotion_readiness/report.md and decision.md. Added targeted Q2 late-afternoon put diagnostics from Protocol 039 selected trades.
1s Path Result: Q1 2026 selected-trade replay audited 173 of 1125 trades on available CBBO-1s sessions, 15.4% coverage. Aggregate audited Q1 1m PnL was 470 versus 1s PnL -2700, diff -3170, sign flip fraction 0.006. The March audited slice was more reassuring: 73 audited March trades, 1m PnL 27480, 1s PnL 27100, diff -380, 0 sign flips.
Context Result: Promotion grade false. SPX official fraction 0.000 and VIX official fraction 0.000 across 312 sessions. SPX is mostly spxw_put_call_parity with a small ES proxy slice; VIX is mostly spxw_0dte_atm_iv with a small VX proxy slice.
Q2 Diagnostic: Q2 fragility is concentrated in late-afternoon puts. Q2 late-afternoon puts produced only 1480 PnL, PF 1.025 over 168 trades. Losing late-afternoon put contracts had higher ask, higher theta burden, higher breakeven ATR, lower gamma/theta scaled, and worse contract value score than winners.
Decision: Protocol 039 remains the current research baseline, but it is not promotion-ready yet. Do not add another model knob.
Next Gate: Rerun Protocol 039 unchanged after official SPX/VIX 1-minute context is available and after broader 1s path audit coverage is approved/provided, especially Q2 late-afternoon put loss sessions and major March/Q4 winner sessions. Any paid data request must be separately approved with source, date range, schema/product, symbols, estimated cost, hard cap, and reason.
Owner: Codex
```

## 2026-05-10 Protocol 041 Data Request Prepared

```text
Date: 2026-05-10
Decision / Experiment: Prepared a tightly capped data request for the next promotion-readiness gate without downloading paid data.
Reason: Protocol 040 showed Protocol 039 is a valid research baseline but not promotion-ready because official SPX/VIX context is absent and 1s path coverage is too sparse.
Data Used: Protocol 039 selected trades, existing local normalized symbol maps, existing CBBO-1s audit inventory, Databento metadata cost estimates only, and public ThetaData/Cboe documentation checks. No paid data was downloaded.
Cost: $0 incremental paid data.
Request: Official SPX/VIX 1-minute bars for 2025-01-02 through 2026-03-31, preferably user-provided from ThetaData Indices Standard, hard cap $80. Targeted Databento OPRA.PILLAR cbbo-1s exact selected-symbol audit for Q2 late-afternoon put loss sessions plus major Q4/March winner sessions, estimated $0.4371, hard cap $2.00.
Implementation: Added v4/audit/autoresearch/v4_aplus_hypothesis_041_data_request_official_context_1s/DATA_REQUEST.md. Also fixed official-context build bookkeeping in v4/scripts/build_databento_neural_dataset.py so official SPX/VIX rebuilds can complete.
Decision: Waiting for explicit approval before any paid endpoint is called.
Next Gate: If approved/provided, build official-context processed blocks, rerun Protocol 039 unchanged, rerun context provenance, and rerun 1s path audit.
Owner: Codex
```

## 2026-05-10 Protocol 039 Frozen Rerun After Data Request

```text
Date: 2026-05-10
Decision / Experiment: Reran Protocol 039 unchanged after preparing the Protocol 041 official-context and targeted CBBO-1s data request.
Reason: The model must stay frozen while we separate model evidence from data-quality evidence. This rerun checks that the current Protocol 039 baseline remains reproducible before any official SPX/VIX or expanded CBBO-1s data is approved/provided.
Data Used: Existing no-fee v4 processed data only: Q1/Q2/Q3/Q4 2025 and Q1 2026. No paid data. No official SPX/VIX. No new CBBO-1s files.
Cost: $0 incremental paid data.
Implementation: Ran v4/scripts/run_protocol039_broader_baseline_validation.py with the same variant, policy, trial, seeds, epochs, validation days, processed data directories, and decision cache as Protocol 039. Only the output directory changed to v4/audit/autoresearch/v4_aplus_hypothesis_039_broader_baseline_validation_rerun_20260510 to preserve the original artifacts.
Result: Reproduced the original Protocol 039 fold summary exactly. Median results: Q2 2025 9100 PF 1.341 over 97 trades, +50 stress 4275, positive seeds 0.80; Q3 13595 PF 1.414 over 126 trades, +50 stress 7320, positive seeds 1.00; Q4 18230 PF 1.577 over 126 trades, +50 stress 12005, positive seeds 1.00; Q1 2026 27095 PF 1.765 over 114 trades, +50 stress 21340, positive seeds 1.00.
Decision: Protocol 039 remains the frozen research baseline. This is still not paper/live approval.
Next Gate: Wait for explicit approval or user-provided files for Protocol 041. After official SPX/VIX and targeted CBBO-1s coverage are available, rebuild official-context processed blocks and rerun Protocol 039 unchanged again.
Owner: Codex
```

## 2026-05-10 ThetaData Official Context Smoke Blocked

```text
Date: 2026-05-10
Decision / Experiment: Attempted the user-approved one-day ThetaData SPX/VIX smoke for 2026-01-02 using index_history_ohlc, 1-minute bars, 09:30-16:00 ET.
Reason: Verify official SPX/VIX access and file shape before downloading the full Protocol 041 official-context range.
Data Used: None. The vendor rejected the request before returning market data.
Cost: $0 incremental paid data observed.
Result: Blocked. ThetaData authenticated the client, but the index_history_ohlc request returned PERMISSION_DENIED and reported that the account attached to the supplied credentials has only a FREE subscription, not an active Index Standard/Pro subscription.
Implementation: Added clean permission-denied handling to v4/scripts/download_thetadata_index_bars.py. Verified no ThetaData index files or download audit records were written.
Decision: Do not retry until the ThetaData subscription issue is resolved.
Next Gate: User should confirm the v4/.env credentials correspond to the paid ThetaData account and that the subscription includes Index Standard/Pro access for SPX/VIX. Then rerun the same one-day smoke before any full-range download.
Owner: Codex
```

## 2026-05-10 Protocol 041 Targeted Databento CBBO-1s Complete

```text
Date: 2026-05-10
Decision / Experiment: Downloaded the approved targeted Databento CBBO-1s audit leg for Protocol 041 and reran the selected-trade 1s path replay.
Reason: ThetaData official SPX/VIX access is blocked until the subscription activates, but the other promotion-readiness data leg can be completed independently. The goal was to increase path-level realism coverage for Protocol 039 selected trades without changing the model.
Data Used: Databento OPRA.PILLAR cbbo-1s, exact Protocol 039 selected SPXW raw symbols only, on 27 pre-registered target sessions. Existing ten Q1 2026 cbbo-1s audit sessions were also reused for replay. No model changes.
Cost: Estimated Databento spend $0.685970, below the $2.00 hard cap. The final estimate was higher than the original prepared packet because the exact replay downloader included every Protocol 039 selected trade on target sessions rather than only the summary rows used to create the request table.
Implementation: Added v4/scripts/download_databento_cbbo_1s_selected.py. Downloaded 27 sessions, 198 exact session-symbol pairs, 4,603,814 rows, recorded in v4/audit/databento_cbbo_1s_selected_downloads.jsonl. Added v4/audit/autoresearch/v4_aplus_hypothesis_041_data_request_official_context_1s/DATABENTO_RESULT.md.
1s Replay Result: Reran v4/scripts/audit_selected_trades_1s_path.py on all Protocol 039 selected trade files. Audited 682 of 4,590 selected trades across 37 available cbbo-1s sessions. Aggregate audited 1m PnL was 267,570 versus 1s PnL 247,940, diff -19,630, median diff 0, p95 absolute diff 269, sign flip fraction 0.29%.
Split Result: Q1 2026 audited slice 97,400 1m vs 90,810 1s with 2 sign flips. Q4 2025 audited slice 165,350 1m vs 156,720 1s with 0 sign flips. Q2 2025 audited slice 4,820 1m vs 410 1s with 0 sign flips.
Q2 Diagnostic: Q2 late-afternoon puts remain genuinely fragile under 1s replay: 65 audited trades, 1m PnL -47,750, 1s PnL -52,590, 0 sign flips. This does not look like a 1-minute label artifact.
Decision: The targeted CBBO-1s data leg is complete and does not invalidate Protocol 039. It confirms realistic 1s path drag and reinforces the Q2 late-afternoon put fragility diagnosis.
Next Gate: Wait for ThetaData Index Standard/Pro access to activate. Then rerun the one-day official SPX/VIX smoke, rebuild official-context processed blocks, and rerun Protocol 039 unchanged.
Owner: Codex
```

## 2026-05-11 Protocol 041 ThetaData Official Context Complete

```text
Date: 2026-05-11
Decision / Experiment: Reran the approved ThetaData SPX/VIX smoke after the user switched to an Index subscription, then downloaded the full Protocol 041 official-context range.
Reason: Protocol 040 showed SPX/VIX context was proxy-derived, not official. Protocol 041 required official SPX/VIX 1-minute context before the frozen Protocol 039 promotion-readiness rerun.
Data Used: ThetaData index_history_ohlc for SPX and VIX only, 1-minute interval, 09:30-16:00 ET, 2025-01-02 through 2026-03-31. No model changes.
Cost: $0 incremental per-request cost observed under the active subscription. This used the already-approved official-context subscription path.
Smoke Result: 2026-01-02 downloaded and loaded successfully. SPX wrote 391 rows from 2026-01-02T14:30:00Z to 2026-01-02T21:00:00Z. VIX wrote 388 rows from 2026-01-02T14:31:00Z to 2026-01-02T21:00:00Z. v4 index loaders read both files with no null closes.
Implementation: Updated v4/scripts/download_thetadata_index_bars.py to chunk requests because ThetaData rejects ranges longer than 365 calendar days. Full range was downloaded in chunks: 2025-01-02 to 2026-01-01, and 2026-01-02 to 2026-03-31.
Download Result: SPX wrote 311 files and 121,059 rows. VIX wrote 320 files and 117,565 rows. Against the 311 v4 option sessions, missing SPX files 0 and missing VIX files 0. VIX has 9 extra holiday/non-option-session files that are ignored by the v4 option-session rebuild.
Build Smoke: Official-context build smoke for 2026-01-02 succeeded: 195,861 normalized option rows, 391 official SPX rows, 388 official VIX rows, and 360 neural rows.
Decision: ThetaData official SPX/VIX data leg is complete and usable.
Next Gate: Build official-context processed blocks for Q1/Q2/Q3/Q4 2025 and Q1 2026, rerun context provenance, then rerun Protocol 039 unchanged.
Owner: Codex
```

## 2026-05-11 Protocol 039 Official-Context Rerun Complete

```text
Date: 2026-05-11
Decision / Experiment: Built the full official-context processed blocks, reran context provenance, and reran frozen Protocol 039 unchanged using official ThetaData SPX/VIX context.
Reason: Protocol 039 needed to be tested against promotion-grade SPX/VIX context without changing the model, loss, selected variant, seeds, or selection knobs. This isolates whether the research baseline survives replacing proxy context with official index bars.
Data Used: Existing Databento SPXW 0DTE OPRA option data plus ThetaData index_history_ohlc SPX/VIX 1-minute bars for 2025-01-02 through 2026-03-31. No additional paid data was downloaded during this rerun.
Cost: $0 incremental paid data after the already-approved ThetaData official-context download.
Implementation: Fixed two official-context data bugs before the final build. v4/ingest/databento_opra.py now normalizes index-bar timestamps through pd.Timestamp.value so microsecond Parquet timestamps align with nanosecond quote timestamps. v4/model/hypothesis_protocol.py now regularizes official SPX/VIX session bars onto a causal 09:30-16:00 ET one-minute grid before market-structure features are indexed by minute offset. Added a regression test for microsecond official index timestamps in v4/tests/test_databento_opra.py.
Processed Blocks: Built official-context blocks for Q1 2025, Q2 2025, Q3 2025, Q4 2025, and Q1 2026 under data/processed/spxw_0dte_neural_*_official_context, with normalized rows under v4/normalized_official_context and build summaries under v4/audit/official_context.
Context Provenance: Promotion grade true. SPX official_fraction 1.000 across 311 files, 311 sessions, and 121,059 rows. VIX official_fraction 1.000 across 320 files, 320 sessions, and 117,565 rows. Output: v4/audit/autoresearch/v4_aplus_hypothesis_039_broader_baseline_validation_official_context/context_provenance.md.
Protocol 039 Result: Official-context median fold results were Q2 2025 PnL 13,330 PF 1.586 trades 84.5 positive seeds 0.80 stress50 9,030; Q3 2025 PnL 10,200 PF 1.439 trades 107.5 positive seeds 1.00 stress50 4,600; Q4 2025 PnL 10,525 PF 1.327 trades 120.0 positive seeds 1.00 stress50 4,425; Q1 2026 PnL 32,810 PF 2.217 trades 100.5 positive seeds 1.00 stress50 27,610.
Comparison: Official context improved Q2 and Q1 2026 versus the prior proxy-context rerun, but weakened Q3 and Q4. The baseline still survived all official folds on median PnL, profit factor above 1.0, and positive seed fractions at or above 0.80. Q2 remains the fragile fold because one seed is negative and beats-random seed fraction is 0.90 rather than 1.00.
Decision: Protocol 039 remains the frozen research baseline and is now supported by official SPX/VIX context. This is stronger research evidence, not paper/live approval.
Next Gate: Keep the model frozen and run path-level promotion checks against the targeted CBBO-1s coverage for the official-context selected trades. Then compare trade-set attribution versus the proxy-context Protocol 039 rerun to understand which official-context timing changes helped Q1/Q2 and hurt Q3/Q4.
Owner: Codex
```

## 2026-05-11 Protocol 042 Official-Context Promotion Checks

```text
Date: 2026-05-11
Decision / Experiment: Ran frozen-model promotion-readiness checks on Protocol 039 official-context selected trades.
Reason: The official-context rerun survived on fold medians, but promotion confidence requires checking whether the selected trades survive available 1s replay and understanding why official context helped Q1/Q2 while weakening Q3/Q4.
Data Used: Existing Protocol 039 official-context selected trades, existing targeted Databento CBBO-1s audit slices, existing official ThetaData SPX/VIX context, and existing proxy-context Protocol 039 selected trades. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Updated v4/scripts/audit_selected_trades_1s_path.py so raw-symbol maps can load official-context normalized files. Updated v4/scripts/compare_protocol_trade_sets.py so it can compare this protocol family's seed-file naming and produce a generic interpretation. Added v4/audit/autoresearch/v4_aplus_hypothesis_042_official_context_promotion_checks/DECISION.md.
1s Path Result: Replayed 417 of 4,064 official-context selected trades on available CBBO-1s slices. Coverage 10.26%. Aggregate audited 1m PnL 114,430 versus 1s PnL 108,870, diff -5,560. Median diff 0, p95 absolute diff 242, sign flip fraction 0.00, 1s worse fraction 0.415. Split replay: Q1 2026 41,990 to 40,670; Q2 2025 18,480 to 17,900; Q4 2025 53,960 to 50,300. Q3 has no 1s coverage in this audit.
Trade Attribution Result: Official context changed selection materially. Versus proxy-context Protocol 039, official context improved Q1 2026 by 62,755 and Q2 2025 by 24,520, but weakened Q3 2025 by 53,780 and Q4 2025 by 92,120 on aggregate selected-trade PnL across seeds.
Diagnostic: Official context improved Q1/Q2 mainly through stronger post-open morning calls. It hurt Q3/Q4 mainly by suppressing profitable post-open morning puts: Q3 proxy-only post-open morning puts made 136,280 versus official-only 45,350; Q4 proxy-only post-open morning puts made 133,050 versus official-only 58,200.
Decision: Protocol 039 remains the frozen official-context research baseline, and the available 1s replay does not reject it. This is still not paper/live approval because 1s coverage is limited, Q3 lacks 1s replay coverage, and official-context trade-set instability must be understood.
Next Gate: Run a no-new-model frozen-feature diagnostic focused on post-open morning puts: official-only losing puts, official-only winning puts, proxy-only missed winning puts, and proxy-only avoided losing puts. Only promote a model change if the separator is stable across Q2/Q3/Q4/Q1.
Owner: Codex
```

## 2026-05-11 Protocol 043 Post-Open Put Fragility Diagnostic

```text
Date: 2026-05-11
Decision / Experiment: Ran a no-new-model frozen-feature diagnostic on post-open morning puts after the official-context trade-set shift.
Reason: Protocol 042 showed official context improved Q1/Q2 but weakened Q3/Q4 mainly by suppressing profitable post-open morning puts. Before adding another model knob, we needed to see whether missed winners were separable from avoided losers using existing causal features.
Data Used: Existing proxy-context Protocol 039 selected trades and official-context Protocol 039 selected trades only. No paid data.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/diagnose_official_context_put_fragility.py and wrote v4/audit/autoresearch/v4_aplus_hypothesis_043_post_open_put_fragility_diagnostic/report.md.
Result: Proxy-only post-open morning puts remained meaningfully profitable in every fold: Q1 2026 126,760, Q2 2025 13,080, Q3 2025 136,280, Q4 2025 133,050. Official-only post-open morning puts were also positive but weaker: Q1 91,940, Q2 2,620, Q3 45,350, Q4 58,200.
Feature Diagnostic: The strongest stable missed-winner separators were pattern_pullback_resume, pattern_compression_ratio, liquidity_score, and pattern_count_norm, each directionally favorable in 3 of 4 splits. The signal is setup/context based, not a clean simple Greek or value-score threshold. Contract-value and breakeven effects were weak and inconsistent.
Decision: Do not add a hard post-open put filter or static Greek threshold. A model-side change is allowed only as a soft recall objective for profitable put-pattern tokens, preserving Protocol 039 selection rules.
Next Gate: Run one pre-registered model change: soft put-pattern recall loss on official-context data, same policy/trial/folds, 3-seed screen. Reject unless it beats the baseline in at least 75% of folds, stays positive under +50 stress, beats matched random, and does not damage March 2026.
Owner: Codex
```

## 2026-05-11 Protocol 044 Put-Pattern Recall Screen

```text
Date: 2026-05-11
Decision / Experiment: Tested one model change: a soft put-pattern recall loss on official-context data.
Reason: Protocol 043 suggested missed post-open put winners were often pattern-confirmed pullback/compression/liquidity setups. The hypothesis was that a recall margin for profitable put-pattern tokens could recover Q3/Q4 put upside without adding a hard filter or new selection knob.
Data Used: Official-context Q1/Q2/Q3/Q4 2025 and Q1 2026 processed blocks, existing official ThetaData SPX/VIX bars, existing Databento SPXW option data. No paid data.
Cost: $0 incremental paid data.
Implementation: Added an unregistered aplus_put_pattern_recall loss path in v4/model/hypothesis_protocol.py and a reproducible runner in v4/scripts/run_protocol044_put_pattern_recall.py. The failed candidate is not registered in registered_aplus_surface_variants, so it is not part of the default variant surface.
Pre-Registration: Same policy/trial as Protocol 039: policy1 / post_open_late_edge25_max2. Same expanding folds. Seeds 11, 22, 33. Candidate must beat baseline in at least 75% of folds, remain positive under +50 stress in every fold, beat matched random in every fold, and not damage March 2026.
Result: Candidate stayed positive, stayed +50 stress positive, beat matched random in every fold, and had slightly higher average median PnL, but it beat the baseline in only 2 of 4 folds and damaged March 2026. Fold deltas were Q2 -5,495, Q3 +10,055, Q4 +3,720, Q1 2026 -6,600. March 2026 delta was -4,020 and +50 stress delta was -3,670.
Decision: Reject Protocol 044 as a replacement or promotion candidate. Preserve Protocol 039 official-context as the frozen research baseline.
Next Gate: Treat this as failed hypothesis 1 in the current loop. The useful evidence is that put-pattern recall can help Q3/Q4 but costs Q2/March. The next hypothesis should not amplify put recall globally; it must be conditional enough to preserve March and Q2, likely by improving state/context normalization or side-specific calibration rather than adding a broader recall margin.
Owner: Codex
```

## 2026-05-11 Protocol 045 Interaction Features Screen

```text
Date: 2026-05-11
Decision / Experiment: Ran a one-change official-context screen of A+ pattern/value interaction token features.
Reason: Protocol 044 showed global put-pattern recall helped Q3/Q4 but damaged Q2/March. The next hypothesis was that interaction token features could let the model condition setup strength, gamma/theta, spread quality, breakeven, VWAP/OMAR/sigma movement, and contract value jointly without adding a hard filter or global put recall margin.
Data Used: Official-context Q1/Q2/Q3/Q4 2025 and Q1 2026 processed blocks. No paid data.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_official_context_variant_screen.py and ran surface_structure_aplus_interactions_side_value_multitask against the Protocol 039 official-context baseline using policy1 / post_open_late_edge25_max2, seeds 11/22/33, 8 epochs, and the same expanding folds.
Result: Candidate passed the three-seed screen. It beat baseline in 3 of 4 folds, stayed positive in every fold, stayed +50 stress positive in every fold, beat matched random in every fold, and did not damage March. Fold deltas were Q2 +1,700, Q3 +4,970, Q4 -3,780, Q1 2026 +7,340. March delta was +6,660.
Decision: Keep the interaction candidate and move to 10-seed validation before changing the frozen baseline.
Next Gate: Run 10-seed validation for surface_structure_aplus_interactions_side_value_multitask against Protocol 039 official-context baseline.
Owner: Codex
```

## 2026-05-11 Protocol 046 Interaction Features 10-Seed Validation

```text
Date: 2026-05-11
Decision / Experiment: Validated the Protocol 045 interaction-feature candidate with 10 seeds on official-context data.
Reason: Protocol 045 passed the small screen, but a replacement baseline must survive the same 10-seed standard as Protocol 039.
Data Used: Official-context Q1/Q2/Q3/Q4 2025 and Q1 2026 processed blocks, official ThetaData SPX/VIX bars, existing Databento SPXW option data, and existing targeted CBBO-1s audit slices. No paid data.
Cost: $0 incremental paid data.
Implementation: Extended v4/scripts/run_protocol039_broader_baseline_validation.py so it can validate a named registered variant without changing default Protocol 039 behavior. Ran surface_structure_aplus_interactions_side_value_multitask with seeds 11/22/33/44/55/66/77/88/99/111. Added v4/audit/autoresearch/v4_aplus_hypothesis_046_interactions_10seed_validation/DECISION.md.
10-Seed Result: Candidate medians were Q2 2025 PnL 11,770 PF 1.523 trades 89 stress50 7,320 positive seeds 1.00; Q3 8,745 PF 1.368 trades 99 stress50 3,170 positive seeds 1.00; Q4 10,930 PF 1.340 trades 117 stress50 5,480 positive seeds 0.90; Q1 2026 34,050 PF 2.306 trades 106 stress50 28,850 positive seeds 1.00.
Baseline Comparison: Versus Protocol 039 official medians, candidate deltas were Q2 -1,560, Q3 -1,455, Q4 +405, Q1 +1,240. It improves Q2 stability by eliminating negative seeds, but does not beat baseline median in Q2/Q3.
1s Path Result: Replayed 439 of 4,055 candidate selected trades on available CBBO-1s slices. Aggregate audited 1m PnL was 142,500 versus 1s PnL 139,650, diff -2,850, median diff 0, p95 absolute diff 270, sign flip fraction 0.00.
Trade Attribution Result: Aggregate selected-trade attribution versus Protocol 039 official baseline improved Q1 by 5,545, Q2 by 12,730, and Q4 by 10,300, but weakened Q3 by 6,610. Median validation is stricter and does not support replacement yet.
Decision: Do not replace Protocol 039 official-context as the frozen research baseline. Keep Protocol 046 as a validated challenger and useful evidence that interaction features help, especially Q2 stability, Q4, and Q1/March.
Next Gate: Test the already-registered interaction features with balanced value multitask loss: surface_structure_aplus_interactions_balanced_value_multitask. This is a one-change test from Protocol 046: same interaction tokens, balanced value objective. Reject unless it beats Protocol 039 official baseline in at least 3 of 4 fold medians, preserves March, stays +50 stress positive, and does not reintroduce Q2 negative seeds.
Owner: Codex
```

## 2026-05-11 Protocol 047 Balanced Interaction Loss Screen

```text
Date: 2026-05-11
Decision / Experiment: Tested interaction token features with the already-registered balanced value multitask loss.
Reason: Protocol 046 showed interaction features are useful but do not replace Protocol 039 because Q2/Q3 medians are weaker. The hypothesis was that the balanced value objective might preserve interaction benefits while reducing fold imbalance.
Data Used: Official-context Q1/Q2/Q3/Q4 2025 and Q1 2026 processed blocks. No paid data.
Cost: $0 incremental paid data.
Implementation: Ran v4/scripts/run_official_context_variant_screen.py with candidate surface_structure_aplus_interactions_balanced_value_multitask, same policy/trial/folds/seeds as Protocol 045.
Result: Candidate failed the three-seed screen. Fold deltas versus baseline were Q2 -8,630, Q3 +9,840, Q4 -2,120, Q1 2026 -1,440. March was not damaged (+5,060), but the candidate beat baseline in only 1 of 4 folds and failed the +50 stress-positive-all-folds rule. Q2 stress50 median was -500.
Decision: Reject Protocol 047. Do not run 10-seed validation. Keep Protocol 039 official-context frozen baseline and Protocol 046 as a validated challenger only.
Next Gate: Continue one-change screens using already-collected official-context data. The next hypothesis should focus on contract-relative economics without a hard filter: test the already-registered ladder-relative quality feature variant under the official-context walk-forward.
Owner: Codex
```

## 2026-05-11 Protocols 048-051 Official-Context Contract-Quality Loop

```text
Date: 2026-05-11
Decision / Experiment: Continued the official-context no-paid-data autoresearch loop focused on entry-side contract quality.
Reason: The model needs to learn when a call/put setup is directionally good but the specific contract is too expensive or inferior to another executable contract in the same minute.
Data Used: Already-collected official-context v4 processed data only: Q1/Q2/Q3/Q4 2025 and Q1 2026 with ThetaData SPX/VIX context plus existing targeted Databento CBBO-1s audit slices. No paid data was downloaded.
Cost: $0 incremental paid data.
Protocol 048/049 Result: The ladder-relative quality candidate passed its 3-seed screen but did not replace Protocol 039 after 10-seed validation. Ten-seed medians were Q2 11615, Q3 14558, Q4 13225, Q1 27165. It beat Protocol 039 in Q3/Q4 but lost Q2/Q1. The 1s audit covered 439 trades, found median 1s-minus-1m difference near 0, p95 absolute difference 100, and 0 sign flips.
Protocol 050 Result: Added one model change, an intra-minute same-side contract-ranking margin on top of the Protocol 039 A+ side-value objective. The 3-seed official-context screen passed: candidate beat baseline in 3/4 folds, stayed +50-stress positive in all folds, beat matched random in all folds, and improved March.
Protocol 051 Result: The 10-seed validation of `surface_structure_aplus_side_value_rank / policy1 / post_open_late_edge25_max2` beat Protocol 039 in every fold median: Q2 14140 vs 13330, Q3 11620 vs 10200, Q4 10970 vs 10525, Q1 37845 vs 32810. +50 stress stayed positive in every fold, and matched random was beaten in every fold. Trade-set attribution versus Protocol 039 improved every compared split. The targeted 1s path audit covered 429 trades, found 1m PnL 132040 vs 1s PnL 127100, median difference 0, p95 absolute difference 236, and 0 sign flips.
Decision: Promote Protocol 051 to the frozen research baseline candidate for entry-side work. This is not paper/live approval. It is the strongest current evidence that v4 is learning contract-specific entry quality beyond Protocol 039.
Caveat: +100 stress remains weak in Q4 2025 (median -1180), Q3 has one negative seed, and 1s coverage is targeted rather than complete.
Next Gate: Do not add another entry-side knob yet. Score Protocol 051 unchanged on the next explicitly approved unseen historical block, or build the first sequential lifecycle layer on top of Protocol 051 entries using already-collected data only.
Owner: Codex
```

## 2026-05-11 Protocols 052-054 Sequential Lifecycle Layer On Frozen Protocol 051

```text
Date: 2026-05-11
Decision / Experiment: Built and validated the first sequential lifecycle layer on top of frozen Protocol 051 entries.
Reason: Protocol 051 proved the strongest entry-side contract-quality baseline so far, but a real 0DTE bot also needs to decide whether to hold, exit, stop out, or flatten based on position state after entry.
Data Used: Already-collected official-context Q1/Q2/Q3/Q4 2025 and Q1 2026 processed blocks plus full normalized same-contract quote paths in v4/normalized_official_context. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_protocol052_sequential_lifecycle_walkforward.py. The runner freezes Protocol 051 entry selection and trains a separate causal lifecycle model using hold time, time left, current PnL, MFE, MAE, giveback, PnL velocity, realized volatility, MFE decay, bid/mid versus entry ask, spread, delta/gamma/theta changes, gamma/theta, theta burden, entry edge, side, and strike offset. Hard stop, target, and flat-before-close behavior remain mandatory.
Important Correction: Early lifecycle experiments that used processed decision rows as path source were superseded because selected contracts can leave the ATM +/- $50 ladder after entry. Protocol 052/054 uses full normalized official-context quote paths for the same contract. Protocol 053 is marked superseded and should not be used for decisions.
Protocol 052 Result: The 3-seed full-path lifecycle screen passed, improving Q2/Q3/Q4 and March while roughly matching Q1.
Protocol 054 Result: The 10-seed validation kept the lifecycle candidate. Median results versus frozen Protocol 051 entry baseline were Q2 2025 13460 vs 14140, Q3 15455 vs 11620, Q4 15715 vs 10970, Q1 2026 37920 vs 37845, and March 2026 19510 vs 16730. +50 and +100 stress medians were positive in every scored fold, and Q4 +100 stress improved from negative under Protocol 051 to positive under Protocol 054.
Caveat: Q2 median PnL is down 680 versus the frozen-entry baseline, and Q3 remains seed-fragile under +50 stress even though median stress PnL is positive.
Decision: Keep Protocol 054 as the validated sequential lifecycle candidate on top of frozen Protocol 051 entries. This is not paper/live approval.
Next Gate: No more entry-side knobs. Run lifecycle-only promotion checks: 1s/tick path audit for lifecycle exits where coverage exists, then attribution of Q2 degradation versus Q3/Q4/March improvement before changing the lifecycle model.
Owner: Codex
```

## 2026-05-12 Protocol 055 Lifecycle Promotion Checks

```text
Date: 2026-05-12
Decision / Experiment: Ran lifecycle-only promotion checks on Protocol 054 without changing entries or model knobs.
Reason: Protocol 054 passed 10-seed walk-forward validation as a sequential lifecycle candidate, but needed path-level verification and attribution before any further model changes.
Data Used: Existing Protocol 054 selected lifecycle trades, existing official-context normalized quote paths, and existing local Databento CBBO-1s audit slices. No paid data was downloaded. No local tick/tcbbo coverage exists for this protocol.
Cost: $0 incremental paid data.
Implementation: Extended v4/scripts/audit_selected_trades_1s_path.py with lifecycle-exit replay mode and split summaries. Added v4/scripts/attribute_protocol054_lifecycle.py for same-entry attribution versus the frozen Protocol 051 exits. Added v4/audit/autoresearch/v4_aplus_hypothesis_055_protocol054_lifecycle_promotion_checks/DECISION.md.
1s Audit Result: Covered 429 of 4120 selected lifecycle trades. Planned lifecycle exit replay matched 1m PnL almost exactly: 131740 planned 1s versus 131740 1m, median diff 0, p95 absolute diff near 0, and 0 sign flips. Conservative 1s mandatory stop/target replay reduced audited PnL to 125310, diff -6430, p95 absolute diff 250, and 0 sign flips. 76 of 429 audited trades had a 1s stop/target event before the planned lifecycle exit. Q3 had no local 1s coverage; Q4 coverage was only 57 trades.
Attribution Result: Same-entry lifecycle deltas were positive on paired attribution: Q2 +3520, Q3 +15945, Q4 +55600, Q1 2026 +6740, March +25760. The prior Q2 caveat is from independent median PnL ranking, where Protocol 054 was down 680 versus Protocol 051; paired Q2 attribution is mildly positive but fragile. Q2 weakness is concentrated in model_exit_loss (-2940), late-afternoon calls (-4080), and post-open puts (-2240). Main positive mechanism is saved/reduced losses, especially Q4 +127780, Q1 +117280, Q3 +78940, and March +51320.
Decision: Keep Protocol 054 as the current sequential lifecycle research candidate. Do not promote to paper/live trading yet.
Next Gate: No entry-side changes. Diagnose clipped winners and Q2 model_exit_loss before changing the lifecycle model. If more 1s/tick data is needed, request explicit approval for a tightly capped selected-trade audit batch first.
Owner: Codex
```

## 2026-05-12 Protocol 056 Lifecycle Failure Diagnosis

```text
Date: 2026-05-12
Decision / Experiment: Diagnosed clipped winners and Q2 model_exit_loss in Protocol 054 without changing the model.
Reason: Protocol 055 showed Protocol 054 is a valid lifecycle research candidate but not paper/live ready. The next move was to identify why the lifecycle layer clips winners and where Q2 model_exit_loss is wrong before proposing any model change.
Data Used: Existing Protocol 054 selected lifecycle trades and existing normalized official-context quote paths. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/diagnose_protocol054_lifecycle_failures.py. The script reconstructs post-exit same-contract quote paths for every selected trade and labels post-exit max/min/final PnL, recovery to frozen baseline, positive recovery, kept-falling behavior, clipped winners, and Q2 model_exit_loss failure modes.
Path Coverage: Reconstructed all 4120 selected Protocol 054 lifecycle trades from local normalized quote paths.
Clipped Winner Result: Clipped winners are the main lifecycle cost: 1401 trades, baseline PnL 1058510, lifecycle PnL 687180, delta -371330. 1066 of 1401 clipped winners recovered to the frozen Protocol 051 baseline after the lifecycle exit, accounting for -299900 of clipped-winner delta. The dominant clipped-winner exit reason is model_exit_giveback: Q1 -114730, Q2 -64700, Q3 -67690, Q4 -83880.
Q2 model_exit_loss Result: Q2 model_exit_loss has 56 trades, baseline PnL -1990, lifecycle PnL -4930, delta -2940. It is mixed: 26 rows recovered to baseline and cost -6230, while 28 no/small-recovery rows improved losses by +3920. The sharp failure subset is 22 baseline winners turned dynamic losers, costing -6970, with 86% recovering to baseline after exit.
Decision: Do not change entries. Do not change Protocol 054 yet. The next hypothesis should be lifecycle-only and should distinguish temporary pullback/recovery risk from true continuation decay.
Next Gate: Pre-register one lifecycle-only exit-confirmation or recovery-risk objective using only causal post-entry state. Reject unless it reduces clipped winners without surrendering saved-loss benefit, preserves Q4/March, improves or matches Q2, and survives the same 10-seed walk-forward standard.
Owner: Codex
```

## 2026-05-12 Protocol 057 Lifecycle Recovery Confirmation Screen

```text
Date: 2026-05-12
Decision / Experiment: Tested a lifecycle-only recovery-confirmation layer on top of frozen Protocol 054.
Reason: Protocol 056 showed that many Protocol 054 exits clip winners that later recover to the frozen-entry baseline. The hypothesis was that a separate causal recovery classifier could distinguish temporary pullback from true continuation decay before allowing model_exit_loss or model_exit_giveback.
Data Used: Existing official-context processed blocks and full normalized same-contract quote paths. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_protocol057_lifecycle_recovery_confirmation.py. Protocol 051 entries and Protocol 054 split-specific lifecycle configs stayed frozen. The recovery classifier used only post-entry state, including PnL/MFE/MAE shape, PnL velocity, time since MFE, gamma/theta change, spread change, liquidity sizes, and time left.
Result: Three-seed walk-forward medians were Q2 2025 17850 vs Protocol 054 17850, Q3 7730 vs 7730, Q4 19430 vs 19540, Q1 2026 38220 vs 38210, and March 2026 20210 vs 20280. +50 stress stayed positive in all quarterly folds, but the candidate did not beat Protocol 054 in enough folds and slightly damaged Q4/March.
Diagnosis: The classifier changed too few exits. Blocked model_exit_loss cases cost more than they saved in aggregate, while blocked giveback exits were mixed.
Decision: Reject Protocol 057. Do not tune the recovery threshold without a new diagnosis.
Next Gate: Test a different lifecycle-only recovery-risk objective that teaches the exit model the regret of exiting now versus the frozen baseline path, rather than adding a second confirmation threshold after Protocol 054 already wants out.
Owner: Codex
```

## 2026-05-12 Protocol 058 Lifecycle Baseline-Regret Objective Screen

```text
Date: 2026-05-12
Decision / Experiment: Tested a direct baseline-regret lifecycle objective on top of frozen Protocol 054.
Reason: Protocol 057's second-stage recovery classifier changed too few exits. The next hypothesis was that the lifecycle model should learn the cost of exiting now versus the frozen stop/target/time path directly, using only causal post-entry state.
Data Used: Existing official-context processed blocks and full normalized same-contract quote paths. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Extended v4/scripts/run_protocol057_lifecycle_recovery_confirmation.py with lifecycle_mode=baseline_regret_objective. Entries, hard stops, targets, time-flat, and Protocol 054 split configs stayed frozen. The candidate trained the lifecycle risk model on capped baseline-exit regret instead of future headroom.
Result: Three-seed walk-forward medians were Q2 2025 15590 vs Protocol 054 17850, Q3 13070 vs 7730, Q4 16340 vs 19540, Q1 2026 26370 vs 38210, and March 2026 11010 vs 20280. +50 and +100 stress medians stayed positive, but the candidate damaged three of four quarterly folds and March.
Diagnosis: The objective improved Q3 and raised profit factor, but it exited too aggressively and surrendered too much convex right tail in Q1/March. It turned the lifecycle layer into a conservative exit model rather than a better recovery discriminator.
Decision: Reject Protocol 058. Do not tune the target cap or thresholds without a new diagnosis.
Next Gate: Test a narrower lifecycle-only rule that confirms model_exit_giveback for one additional bar while leaving model_exit_loss unchanged.
Owner: Codex
```

## 2026-05-12 Protocol 059 Lifecycle Giveback One-Bar Confirmation Screen

```text
Date: 2026-05-12
Decision / Experiment: Tested a one-bar confirmation rule for model_exit_giveback on top of frozen Protocol 054.
Reason: Protocol 056 showed clipped winners are dominated by model_exit_giveback, while Protocol 057 and 058 showed that broad recovery/lifecycle changes either move too few trades or damage convex winners. The hypothesis was that a narrow one-minute confirmation could reduce false giveback exits without touching model_exit_loss.
Data Used: Existing official-context processed blocks and full normalized same-contract quote paths. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Extended v4/scripts/run_protocol057_lifecycle_recovery_confirmation.py with lifecycle_mode=giveback_onebar_confirmation. When Protocol 054 wanted a giveback exit, the candidate waited one same-contract minute and exited only if the exit signal persisted without PnL recovery. Entries, hard stops, targets, loss exits, time-flat, and split configs stayed frozen.
Result: Three-seed walk-forward medians were Q2 2025 17780 vs Protocol 054 17850, Q3 6500 vs 7730, Q4 16700 vs 19540, Q1 2026 37340 vs 38210, and March 2026 19220 vs 20280. Q3 +100 stress stayed negative and every fold was weaker than Protocol 054.
Diagnosis: The rule changed 220 selected trades across quarterly folds and lost PnL in every fold. Giveback exits are a clipped-winner source, but simple one-bar confirmation is not the fix.
Decision: Reject Protocol 059. This is the third consecutive lifecycle-only failure after Protocols 057 and 058.
Next Gate: Stop the tweak loop as requested. Protocol 054 remains the current lifecycle baseline. The next move should be diagnostic or architectural: either build a trade-level lifecycle attribution dataset for a sequence model, or return to data-quality/path-validation before adding another exit rule.
Owner: Codex
```

## 2026-05-12 Protocol 060 Lifecycle Sequence Dataset

```text
Date: 2026-05-12
Decision / Experiment: Built the trade-level lifecycle attribution dataset for a future sequence model.
Reason: Protocols 057, 058, and 059 showed that small exit-rule tweaks are not improving Protocol 054. The next direction is architectural: train a model that sees the post-entry sequence and learns hold/exit/decay/recovery behavior from full paths.
Data Used: Existing Protocol 054 selected trades and existing official-context normalized same-contract quote paths. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/build_lifecycle_sequence_dataset.py. The builder writes protocol054_lifecycle_trades.parquet and protocol054_lifecycle_steps.parquet under v4/audit/autoresearch/v4_aplus_hypothesis_060_lifecycle_sequence_dataset/. Step rows separate causal state features from future path label columns. Rows after Protocol 054 exit are retained and flagged so recovery/decay after the frozen exit can be learned or audited.
Coverage: Built 4120 trade rows and 103000 step rows from 2687 unique canonical entries. Path status was ok for all 4120 trades. Baseline path-vs-recorded p95 absolute difference was effectively zero; Protocol 054 path-vs-recorded p95 absolute difference was about 0.00015 PnL dollars.
Greek Fix: The normalized quote paths did not carry usable gamma/theta, so the builder computes missing IV/Greeks using Black-Scholes from mid, official-context underlying price, strike, right, and settlement time. Step-level IV/delta/gamma/theta/vega coverage is about 95%.
Decision: Keep Protocol 060 as the lifecycle-sequence data foundation. This is not model promotion and not paper/live approval.
Next Gate: Train the first actual sequence lifecycle model using Protocol 060. The model should consume causal post-entry step features and predict exit/hold value or action logits, then be evaluated against frozen Protocol 054 on the same walk-forward holdouts without adding entry-side knobs.
Owner: Codex
```

## 2026-05-12 Protocol 061 Sequence Lifecycle Model Screen

```text
Date: 2026-05-12
Decision / Experiment: Trained the first causal GRU lifecycle model from Protocol 060.
Reason: After three failed hand-coded lifecycle tweaks, the next architectural hypothesis was that a model seeing the full causal post-entry sequence could learn temporary pullback versus true continuation decay better than scalar exit rules.
Data Used: Existing Protocol 060 lifecycle sequence dataset. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_protocol061_sequence_lifecycle_model.py. The model consumes causal step features, including PnL path shape, MFE/MAE, PnL velocity, liquidity/spread, time left, and computed IV/Greeks. It predicts a fixed risk-adjusted continuation target and exits when predicted continuation value is non-positive, while preserving mandatory hard stop/target/time-flat behavior.
Result: Three-seed future-split screen beat Protocol 054 in Q3, Q4, and Q1 aggregate medians but failed the March preservation gate. Medians were Q3 130320 vs Protocol 054 120800, Q4 234130 vs 157500, Q1 366240 vs 355120, and March 166810 vs 185510. +50 stress stayed positive.
Diagnosis: The model learned an extremely early exit/scalp behavior in Q4/Q1. That behavior harvests quick gains and avoids decay but exits too early in March, giving up convex winners.
Decision: Reject Protocol 061 as a replacement for Protocol 054. Keep it as evidence that sequence modeling is useful, but the objective is too broad and too early-exit biased.
Next Gate: Test a residual sequence override: train the sequence model to predict whether exiting now beats the frozen Protocol 054 lifecycle exit, and only override Protocol 054 when the predicted advantage is positive.
Owner: Codex
```

## 2026-05-12 Protocol 062 Protocol054 Residual Sequence Screen

```text
Date: 2026-05-12
Decision / Experiment: Tested a causal GRU residual override on top of frozen Protocol 054.
Reason: Protocol 061 proved sequence modeling can add value but learned an overly broad early-exit behavior that damaged March. The next hypothesis was to make the sequence model residual: only override Protocol 054 when current executable PnL is predicted to beat the frozen Protocol 054 exit.
Data Used: Existing Protocol 060 lifecycle sequence dataset. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Extended v4/scripts/run_protocol061_sequence_lifecycle_model.py with sequence_mode=protocol054_residual_override. The target was current executable PnL minus frozen Protocol 054 PnL. If no override fired, the simulation used Protocol 054's frozen exit.
Result: Three-seed screen improved Q3 and Q4 but damaged Q1/March. Medians were Q3 169290 vs Protocol 054 120800, Q4 201270 vs 157500, Q1 333890 vs 355120, and March 159860 vs 185510.
Diagnosis: The residual model still fired too early in Q1/March. It learned a useful early-exit pattern for Q3/Q4, but that same pattern clipped convex 2026 winners.
Decision: Reject Protocol 062.
Next Gate: Test validation-calibrated residual override confidence with no-override as a valid selected outcome.
Owner: Codex
```

## 2026-05-12 Protocol 063 Calibrated Residual Sequence Screen

```text
Date: 2026-05-12
Decision / Experiment: Tested validation-calibrated residual sequence overrides on top of frozen Protocol 054.
Reason: Protocol 062's residual objective was useful but too aggressive. The next hypothesis was that validation-selected abstention thresholds could preserve the useful overrides while reducing false March/Q1 exits.
Data Used: Existing Protocol 060 lifecycle sequence dataset. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Extended v4/scripts/run_protocol061_sequence_lifecycle_model.py with --calibrate-threshold. Thresholds were selected from validation sessions only for each fold/seed, with no-override available. Test folds remained locked.
Result: This was the strongest sequence challenger so far, but still failed March preservation. Medians were Q3 169290 vs Protocol 054 120800, Q4 169070 vs 157500, Q1 395200 vs 355120, and March 180360 vs 185510. +50 and +100 stress medians stayed positive in every scored split.
Diagnosis: Calibration fixed part of the over-aggression and improved total Q1, but it still clipped enough March convexity to fail the gate by about 5150.
Decision: Reject Protocol 063 as a replacement, but keep it as a serious challenger and the best current sequence-model evidence.
Next Gate: Stop the model-tweak loop after Protocols 061, 062, and 063. Run a diagnostic comparing March false residual overrides against successful Q3/Q4/Jan-Feb overrides before changing the sequence objective again.
Owner: Codex
```

## 2026-05-12 Protocol 064 Protocol063 March Override Diagnostic

```text
Date: 2026-05-12
Decision / Experiment: Diagnosed Protocol 063 residual sequence overrides, focusing on March false overrides versus successful Q3/Q4 and Jan-Feb overrides.
Reason: Protocol 063 was the strongest sequence challenger but still failed March preservation. Before adding another model knob, we needed to know whether March degradation came from side exposure, time bucket exposure, contract quality, or clipped convex winners.
Data Used: Existing Protocol 063 selected trades and Protocol 060 lifecycle sequence dataset. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/diagnose_protocol063_residual_overrides.py. The diagnostic joins selected residual exits back to the causal lifecycle step table at the candidate exit step, then compares row-level and de-duplicated override cohorts by side, time bucket, exit step, Protocol 054 exit reason, and causal/future path features.
Result: Protocol 063's residual override mechanism is useful but fragile. Q3/Q4 positive overrides contributed +1064240 row-level delta versus Protocol 054 and Jan-Feb positive overrides contributed +627630. March negative overrides cost -399960 row-level delta. They exited at median step 0 while Protocol 054 would have exited around step 19. Median post-exit recovery headroom for March negative overrides was about 1080, versus 60 for Q3/Q4 positive overrides.
Diagnosis: The March failure is primarily false early exit on recoverable convex pullbacks, not a generic lack of sequence-model signal. The worst March side bucket was calls at -247000 row-level delta; puts were also damaged at -152960. The worst time bucket was post_open_morning at -318130. Time-flat and target-bound Protocol 054 exits were especially expensive to override too early.
Decision: Keep Protocol 054 frozen. Protocol 063 remains rejected as replacement but remains the strongest sequence-model challenger. Protocol 064 is diagnostic-only.
Next Gate: Test one lifecycle-only residual sequence objective with an asymmetric false-early-exit penalty. It must penalize early overrides when future recovery/convexity remains high, use only allowed training folds, keep entries frozen, add no paid data, and survive the same holdouts versus Protocol 054.
Owner: Codex
```

## 2026-05-12 Protocol 065 Residual Recovery-Penalty Sequence Screen

```text
Date: 2026-05-12
Decision / Experiment: Tested a recovery-aware asymmetric residual sequence objective on top of frozen Protocol 054.
Reason: Protocol 064 showed that Protocol 063's March failure came from false early residual overrides on recoverable convex pullbacks. The hypothesis was that the sequence model should still learn residual exits, but receive stronger training loss on early negative-residual states where future recovery and baseline-regret labels show that exiting now would clip convexity.
Data Used: Existing Protocol 060 lifecycle sequence dataset and existing Protocol 054 selected trades. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Extended v4/scripts/run_protocol061_sequence_lifecycle_model.py with sequence_mode=protocol054_residual_recovery_penalty. The target remains current executable PnL minus frozen Protocol 054 PnL, but value loss is up-weighted for early negative-residual states with high future recovery and baseline-regret labels. Entries, hard stop, target, time-flat, validation-calibrated thresholding, and holdouts stayed frozen.
Result: Protocol 065 passed the pre-registered screen versus Protocol 054. Median PnL was Q3 2025 133975 vs 120800, Q4 275710 vs 157500, Q1 2026 375130 vs 355120, and March 2026 189250 vs 185510. +50 stress medians stayed positive in all scored folds.
Diagnostic: The follow-up override diagnostic showed the intended behavior change. March negative override row-level delta improved from Protocol 063's -399960 to -141800, median March false override step moved from 0 to 12, and median post-exit recovery headroom fell from 1080 to 640. Q3/Q4 positive override delta improved from 1064240 to 1258350. The remaining weakness is that recoverable false exits still exist, especially in post-open morning.
Decision: Keep Protocol 065 as the current best sequence-lifecycle challenger, not paper/live approval. Protocol 054 remains the frozen baseline until Protocol 065 survives stricter validation.
Next Gate: Run no-new-model validation for Protocol 065: more seeds, trade-set attribution versus Protocol 054 and Protocol 063, side/time/exit-reason exposure, Q2/Q3/Q4/Q1/March fragility, and 1s/tick path checks where coverage exists. Do not add entry-side knobs and do not buy data.
Owner: Codex
```

## 2026-05-12 Protocol 066 Protocol065 10-Seed Validation

```text
Date: 2026-05-12
Decision / Experiment: Reran Protocol 065 unchanged with 10 model seeds and added override attribution plus available CBBO-1s lifecycle replay.
Reason: Protocol 065 passed the first 3-seed screen, but a candidate cannot move toward promotion on a small seed sample. The next gate was no-new-model validation: same objective, same data, same holdouts, more seeds, and path checks where local 1s coverage exists.
Data Used: Existing Protocol 060 lifecycle sequence dataset, existing Protocol 054 selected trades, and already-collected CBBO-1s audit slices. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Reused v4/scripts/run_protocol061_sequence_lifecycle_model.py with sequence_mode=protocol054_residual_recovery_penalty, --calibrate-threshold, and seeds 1 through 10. Reused v4/scripts/diagnose_protocol063_residual_overrides.py for override attribution and extended v4/scripts/audit_selected_trades_1s_path.py to support sequence selected rows with candidate_exit_time, candidate_pnl, and candidate_exit_reason.
Result: 10-seed medians beat Protocol 054 in all scored splits: Q3 2025 139842.5 vs 120800, Q4 270260 vs 157500, Q1 2026 366115 vs 355120, and March 2026 188740 vs 185510. +50 stress medians remained positive in all scored splits. Delta-positive seed counts were Q3 9/10, Q4 10/10, Q1 10/10, and March 8/10.
Override Attribution: Positive Q3/Q4 residual overrides contributed +4168320 row-level delta versus Protocol 054. March negative overrides still cost -301310 row-level delta, with median false override step 11 and median post-exit recovery headroom 620. The remaining fragility is concentrated in post-open morning and split roughly across calls and puts.
1s Path Audit: Available CBBO-1s coverage audited 2770 of 32490 selected rows. Overall median 1s-minus-1m diff was 0, p95 absolute diff was 60, sign-flip fraction was 0, and 1s worse fraction was 5.45%. March coverage was 1250 of 3950 rows with median diff 0, p95 absolute diff 20, and sign-flip fraction 0. Q3 had no available 1s coverage, so that remains unaudited.
Decision: Keep Protocol 066 as the current best validated sequence-lifecycle challenger. This is stronger than Protocol 065 because it survives 10 seeds and partial 1s replay, but it is still not paper/live approval and does not replace Protocol 054 as the frozen baseline until stricter promotion checks are complete.
Next Gate: No entry-side knobs. Run trade-level attribution versus Protocol 054/063/066 focused on the remaining negative delta seeds and post-open morning false exits, then decide whether the next change is a sequence objective refinement or a promotion-readiness paper-trading simulator. Do not buy data.
Owner: Codex
```

## 2026-05-12 Protocol 067 Protocol066 Seed-Fragility Diagnostic

```text
Date: 2026-05-12
Decision / Experiment: Diagnosed the remaining negative-delta seeds from Protocol 066.
Reason: Protocol 066 passed the median gate but was not seed-perfect. We needed to know whether the losing seeds exposed a structural weakness or small residual variance before changing the objective again.
Data Used: Existing Protocol 066 selected trades. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/diagnose_protocol066_seed_fragility.py. The diagnostic expands Q1 into a March view, computes candidate-versus-Protocol-054 delta by split and model seed, then attributes negative-delta seeds by side, time bucket, candidate exit reason, Protocol 054 exit reason, exit-step bucket, and selected path features.
Result: Negative-delta seeds were limited to March 2026 seed 4 at -2990, March 2026 seed 10 at -500, and Q3 2025 seed 3 at -1925. March negative seeds had low override fractions of 10.4% and 7.6%, with median candidate exit step 18. Q3 seed 3 had high override fraction at 84.1%, but the net miss was small versus the fold's median positive delta.
Diagnosis: The remaining fragility is not the catastrophic step-0 false-exit behavior from Protocol 063. March misses are small and concentrated in time-flat/target rows, mostly post-open morning. Q3 seed 3 was hurt by overrides on target and model_exit_giveback rows, while hard-stop rows were strongly improved.
Decision: Keep Protocol 066 as the best validated sequence-lifecycle challenger. Do not add another objective knob just to chase small seed misses. Do not promote yet.
Next Gate: Move toward promotion-readiness infrastructure with the model frozen: paper-trading simulator/replay harness, order-state accounting, conservative slippage stress, and clearer live-data parity checks. Keep the remaining post-open false-exit pattern on the watchlist.
Owner: Codex
```

## 2026-05-12 Protocol 068 Protocol066 Promotion-Readiness Replay

```text
Date: 2026-05-12
Decision / Experiment: Froze Protocol 066 and built the first promotion-readiness replay infrastructure.
Reason: The model-tweak loop should stop after Protocol 066/067. The next risk is execution realism: can the frozen selected decisions survive bid/ask replay, order-state accounting, conservative slippage, and explicit live-data parity checks?
Data Used: Existing Protocol 066 selected trades, existing Protocol 060 lifecycle sequence steps, and the existing Protocol 066 CBBO-1s audit summary. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/promotion/PROTOCOL_066_FREEZE.json, v4/promotion/PROTOCOL_066_PROMOTION_READINESS_PACKET.md, v4/sim/paper_replay.py, v4/scripts/run_protocol066_promotion_readiness.py, and v4/tests/test_paper_replay.py. The replay freezes artifact hashes, joins selected trades back to executable lifecycle path steps, replays entry at ask and exit at bid, applies extra adverse slippage scenarios, samples order-state records through exit_filled, and emits live-data parity checks.
Result: Freeze hash checks passed. The replay matched all 32490 selected rows to exit path steps. Candidate-vs-replayed path PnL p99 absolute difference was 0. Extra $0.25 per side slippage remained positive across all scored splits: Q3 median 87492.5, Q4 211160, Q1 315115, and March 168990, with 10/10 positive seeds in each. Order-state accounting sampled 5000 records, all one-contract and all final_state exit_filled.
Live Parity: Required bid/ask fields, bid/ask sanity, no-mid fills, one-contract behavior, flat-before-close behavior, SPXW PM contract identity, selected-vs-path PnL, decision-before-exit, and exit quote-time causality passed. Warnings remain for missing retained entry quote timestamp, missing quote-gap coverage, and partial CBBO-1s coverage. Blockers remain for live shadow-feed parity and persisted deployable model/scaler artifacts.
Decision: Keep Protocol 066 frozen as the best validated sequence-lifecycle challenger. Do not approve broker-connected paper trading yet. Protocol 068 is a promotion-readiness infrastructure milestone, not promotion.
Next Gate: Persist deployable model/scaler artifacts for the frozen Protocol 066 protocol, then build a no-order shadow-feed parity harness that compares live features and quotes against the research path before any broker order placement.
Owner: Codex
```

## 2026-05-12 Protocol 069 Protocol066 Artifact Persistence

```text
Date: 2026-05-12
Decision / Experiment: Persisted Protocol 066 model/scaler artifacts without changing the model.
Reason: Protocol 068 identified deployable model/scaler persistence as a blocker before any no-order shadow or paper-trading path. The next step was to rerun the frozen Protocol 066 lifecycle sequence protocol with artifact saving enabled, then verify the outputs.
Data Used: Existing Protocol 060 lifecycle sequence dataset only. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Extended v4/scripts/run_protocol061_sequence_lifecycle_model.py with --save-model-artifacts and per fold/seed artifact writing. Added v4/promotion/PROTOCOL_066_DEPLOYMENT_ARTIFACTS.json as a central hash/pointer manifest.
Result: Protocol 066 metrics reproduced under v4_aplus_hypothesis_069_protocol066_artifact_persistence: Q3 median PnL 139842.5, Q4 270260, Q1 366115, March 188740, all +50 stress medians positive, all positive seed fractions 1.00. Persisted 30 fold/seed bundles across 3 folds and 10 seeds per fold, with 120 total files: model.pt, scaler.json, threshold_sweep.json, and manifest.json per bundle. A sample artifact load smoke reconstructed the model/scaler and produced finite value/recovery/decay outputs.
Decision: The persisted-artifact blocker from Protocol 068 is cleared as an infrastructure item. This is still not paper/live approval.
Next Gate: Build and run no-order shadow-feed parity before broker-connected paper trading. An explicit live inference ensemble/router policy is still required before order placement.
Owner: Codex
```

## 2026-05-12 Protocol 070 Protocol066 No-Order Shadow Parity Harness

```text
Date: 2026-05-12
Decision / Experiment: Built the first Protocol 066 no-order shadow-feed parity harness.
Reason: Before paper trading, the live path must prove it can produce the same kind of decision-time evidence as the research path without placing orders: fresh option NBBO, official SPX/VIX context, complete Protocol 066 feature schema, PM-settled SPXW identity, one-contract state, model artifact reference, and hold/exit decision logging.
Data Used: No market data. The harness produced a JSONL template only because no live shadow capture exists yet. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/live/shadow_parity.py, v4/scripts/run_protocol066_shadow_parity.py, and v4/tests/test_shadow_parity.py. The harness rejects broker/order fields by design, validates quote/context freshness, enforces the Protocol 066 causal feature schema, and writes a blocked report when no live feed is supplied.
Result: Generated v4/audit/autoresearch/v4_aplus_hypothesis_070_protocol066_shadow_parity/report.md and shadow_observation_template.jsonl. Current status is intentionally blocked because no no-order live JSONL has been captured. Focused verification passed: 30 tests.
Decision: Keep Protocol 066 frozen. Promotion-readiness infrastructure is improved, but broker-connected paper trading remains blocked.
Next Gate: Capture a no-order live shadow feed and rerun Protocol 070. If it passes, the next gate is entry quote timestamp retention, stale-quote enforcement, and an approved inference ensemble/router policy.
Owner: Codex
```

## 2026-05-12 Protocol 071 Protocol066 Offline Shadow Rehearsal

```text
Date: 2026-05-12
Decision / Experiment: Built and ran an offline no-order shadow rehearsal for persisted Protocol 066 artifacts.
Reason: Before capturing a real live shadow feed, the persisted artifact path needs to prove it can load model/scaler files, consume causal lifecycle feature sequences, produce model outputs, emit no-order JSONL observations, and pass the shadow parity validator.
Data Used: Existing Protocol 060 lifecycle sequence rows, persisted Protocol 066 artifacts, and already-collected official VIX index bars. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/live/protocol066_inference.py and v4/scripts/run_protocol066_offline_shadow_rehearsal.py. The runner loads a saved Protocol 066 fold/seed artifact, replays causal feature sequences from a March slice, joins official VIX context, emits no-order shadow observations, and validates them with v4/live/shadow_parity.py.
Result: v4_aplus_hypothesis_071_protocol066_offline_shadow_rehearsal passed: 50 trades, 1250 shadow rows, 1250 passed, 0 failed, action counts hold 1178 / exit 36 / forced_flat 30 / stop 6, and VIX coverage 1.000. Feature gaps filled for offline rehearsal were quote_gap_seconds 1250, IV/Greeks 158, and option volume 70; live capture must compute these directly rather than rely on offline fills.
Decision: The persisted Protocol 066 artifact path is runnable in a no-order shadow schema. This is still offline and is not paper/live approval.
Next Gate: Capture the same schema from a real no-order live feed. Also make the full modular inference stack explicit, because Protocol 066 by itself is only a residual lifecycle override.
Owner: Codex
```

## 2026-05-12 Protocol 072 Protocol066 Artifact Reproduction Audit

```text
Date: 2026-05-12
Decision / Experiment: Reloaded all persisted Protocol 066 model/scaler artifacts and checked exact reproduction of frozen research exits.
Reason: A saved model is only useful if the deployable artifact path reproduces the research decisions, rather than merely training to similar aggregate metrics.
Data Used: Existing Protocol 060 lifecycle sequence rows and Protocol 069 persisted model artifacts. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/audit_protocol066_artifact_reproduction.py. The audit loads each fold/seed manifest, runs the saved model/scaler over the relevant test split, re-simulates Protocol 066's residual override rule, and compares exit reason, exit step, and PnL against the frozen selected_trades_sequence_exits.json file.
Result: Passed. Checked 30 artifacts and 32490 selected rows with 0 mismatches. Dependency audit showed candidate exit reasons: 22323 sequence_residual_override, 8660 protocol054_fallback, 921 target, and 586 hard_stop. Protocol 054 fallback accounts for 26.7% of selected exits.
Decision: Protocol 066 artifacts are reproducible and faithful to the research run. Live/paper readiness still requires the Protocol 054 fallback path or a replacement self-contained lifecycle model.
Next Gate: Decide whether to persist the modular fallback stack or pre-register a self-contained lifecycle model before purchasing broader historical data.
Owner: Codex
```

## 2026-05-12 Protocol 073 Protocol066 Fallback Ablation

```text
Date: 2026-05-12
Decision / Experiment: Tested whether Protocol 066 can safely omit the Protocol 054 fallback engine.
Reason: Protocol 066 is not a standalone lifecycle model; it exits early when the residual model fires, otherwise it falls back to Protocol 054. If no-fallback performance held up, the live path could be simplified. If not, the fallback must be preserved or replaced.
Data Used: Existing Protocol 069 selected trades joined to Protocol 060 lifecycle trade labels. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/audit_protocol066_fallback_ablation.py. The ablation replaces protocol054_fallback exits with the original baseline stop/target/time-flat outcome, leaving sequence overrides, hard stops, and targets unchanged.
Result: Reject no-fallback simplification. No-fallback remained positive but damaged Q3, Q4, and March: Q3 median PnL 125870 vs 139842.5, Q4 259705 vs 270260, March 170900 vs 188740. Profit factor also fell in all scored splits, including March 2.980 vs 4.131. Q1 total PnL rose slightly, but PF fell from 3.055 to 2.608.
Decision: Do not remove Protocol 054 fallback for promotion-readiness. Protocol 066 should be treated as a modular residual override until a newly validated self-contained lifecycle model beats it.
Next Gate: Persist or rebuild the Protocol 054 fallback inference stack, or pre-register a self-contained lifecycle model. Do not buy broad historical data for a simplified no-fallback Protocol 066.
Owner: Codex
```

## 2026-05-12 Protocol 075 Protocol051/054 Frozen Stack Artifact Persistence

```text
Date: 2026-05-12
Decision / Experiment: Persisted a frozen-config rerun of the Protocol 051 entry model and Protocol 054 fallback lifecycle stack.
Reason: Protocol 072 showed Protocol 066 depends on Protocol 054 fallback for about 26.7% of selected exits, and Protocol 073 rejected removing that fallback. Before any broader data spend or live shadow capture can be trusted, the upstream entry/fallback stack must exist as saved artifacts rather than only selected-trade records.
Data Used: Existing official-context processed data, existing cached surface decisions, and existing normalized option quote paths. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Extended v4/scripts/run_protocol052_sequential_lifecycle_walkforward.py with --save-live-artifacts and --force-selected-configs-json. An unforced rerun drifted in selected config for Q3, so it was stopped and replaced with a forced-config run using v4/audit/autoresearch/v4_aplus_hypothesis_054_protocol052_lifecycle_10seed_validation/report.json as the frozen config source. Added v4/promotion/PROTOCOL_054_051_STACK_ARTIFACTS.json.
Result: Protocol 075 completed with forced Protocol 054 configs for all splits. Saved 40 fold/seed bundles and 200 files: entry_model.pt, entry_standardizer.json, protocol054_risk_model.pt, protocol054_risk_scaler.json, and manifest.json per bundle. A sample artifact load smoke reconstructed the entry SurfaceActionModel and Protocol054 RiskHeadroomModel; risk feature names matched. Metrics remained a lifecycle candidate: Q2 median PnL 14420, Q3 11880, Q4 16895, Q1 38820, March 20890, with +50 stress positive in all splits. Metrics are not byte-identical to the original Protocol 054 report, so this is a saved-stack candidate rerun rather than exact historical reproduction.
Decision: Keep Protocol 075 as the persisted modular stack candidate needed under Protocol 066. This is infrastructure progress, not paper/live approval.
Next Gate: Build the live inference router that combines Protocol 051 entry, Protocol 054 fallback, and Protocol 066 residual override in no-order mode, then run shadow parity. Do not buy broader data until the router/parity result is clean or until a pre-registered self-contained model plan explicitly supersedes this modular stack.
Owner: Codex
```

## 2026-05-13 Protocols 076-082 Q4-Start Sequence Lifecycle Candidate

```text
Date: 2026-05-13
Decision / Experiment: Added the newly downloaded Q4 2024 official-context block to the causal walk-forward path and trained a new residual sequence-lifecycle candidate.
Reason: The model needed to use the new Q4 2024 data without tuning directly to it. The pre-registered direction was to treat Q4 2024 as earlier history, then test forward on Q1 2025, Q2 2025, Q3 2025, Q4 2025, Q1 2026, and March 2026.
Data Used: Existing local Q4 2024 official-context block plus already-collected 2025/Q1 2026 official-context blocks. No paid data was downloaded during training.
Cost: $0 incremental paid data.
Implementation: Extended the walk-forward runners to accept a Q4 2024 prehistory block and infer chronological sequence folds. Protocol 076 retrained the Protocol 051/054 modular baseline from Q4 2024 forward. Protocol 077 built a Q4-start lifecycle sequence dataset from Protocol 076 selected trades. Protocol 078 trained the residual recovery sequence model. Protocol 079 persisted artifacts. Protocol 080 found one floating-point threshold-boundary reproduction mismatch, so Protocol 081 added a deterministic 1e-4 threshold margin and reran artifact persistence. Protocol 082 reproduced all saved artifacts.
Result: Protocol 076 was positive but rejected as a replacement baseline because dynamic exits lost median edge in Q2 2025 and Q1 2026. Protocol 081 passed the sequence-lifecycle gate against Protocol 054: Q2 median PnL 201585 vs 200635, Q3 218765 vs 156410, Q4 346880 vs 158410, Q1 2026 397815 vs 319560, and March 206065 vs 165330. All scored splits kept +50 stress positive and positive seed fraction 1.00. Persisted 40 model/scaler/threshold/manifest bundles. Protocol 082 checked 40 artifacts and 42170 selected rows with 0 mismatches.
Decision: Keep Protocol 081 as the new best sequence-lifecycle challenger. It is stronger than Protocol 066 on the overlapping Q3/Q4/Q1/March evidence and adds a Q2 2025 scored fold. This is not paper/live approval.
Next Gate: Promotion-readiness validation for Protocol 081: replay/order-state accounting, conservative slippage stress, no-order shadow parity, fallback/router integration, and targeted 1s path checks before any paper trading.
Owner: Codex
```

## 2026-05-13 Protocols 083-087 Protocol081 Promotion-Readiness Validation

```text
Date: 2026-05-13
Decision / Experiment: Moved frozen Protocol 081 into promotion-readiness infrastructure without changing the model.
Reason: Protocol 081 passed the research gate, but paper/live readiness requires replay accounting, artifact freeze hashes, high-resolution path checks where coverage exists, no-order shadow parity, and a fallback/router decision before any broker-connected paper trading.
Data Used: Existing Protocol 081 selected trades/artifacts, Protocol 077 lifecycle sequence data, existing normalized official-context option data, existing local CBBO-1s audit slices, and existing official VIX bars. No paid data was downloaded.
Cost: $0 incremental paid data.
Implementation: Generalized the Protocol 066 replay/shadow harness to accept a protocol id/label while preserving Protocol 066 defaults. Aligned the live inference helper with the research simulator so ordinary Protocol 054 fallback exit steps emit exit/protocol054_fallback rather than hold. Added v4/scripts/build_protocol081_promotion_artifacts.py to generate v4/promotion/PROTOCOL_081_FREEZE.json and v4/promotion/PROTOCOL_081_DEPLOYMENT_ARTIFACTS.json. Added v4/promotion/PROTOCOL_081_PROMOTION_READINESS_PACKET.md.
Result: Protocol 083 one-second path audit covered 4160 trades with 0 sign flips, p95 absolute diff 10, planned lifecycle exit diff 0, and 276 mandatory 1s stop/target events before lifecycle exit. Protocol 084 promotion replay passed freeze hashes, replayed 42170 rows with 1.0 matched exit-step fraction, kept +0.25 each-side slippage positive in Q2/Q3/Q4/Q1/March, and sampled 5000 order records with all exit_filled and all one contract. Protocol 085 no-order shadow template remains intentionally blocked until a real live JSONL capture exists. Protocol 086 offline shadow rehearsal passed 1250 rows with 0 failures and actions hold/exit/stop/forced_flat represented. Protocol 087 rejected no-fallback simplification because no-fallback damaged Q3/Q4 and lowered PF despite improving Q2/March PnL.
Decision: Keep Protocol 081 frozen as the best validated sequence-lifecycle challenger, but do not approve broker-connected paper trading. Promotion readiness is blocked by live shadow-feed parity, live router integration, entry quote timestamp retention, stale-quote enforcement, and missing Q3 CBBO-1s coverage.
Next Gate: Build/run the full no-order live shadow router for Protocol 051 entry + Protocol 054 fallback + Protocol 081 residual override, then rerun shadow parity on captured live JSONL before any order placement.
Owner: Codex
```

## 2026-05-12 Protocol 088 Protocol081 No-Order Shadow Router

```text
Date: 2026-05-12
Decision / Experiment: Built and ran the no-order shadow router for the Protocol 051 entry stack, Protocol 054 fallback stack, and Protocol 081 residual sequence override.
Reason: Before paper trading, the saved model pieces need to behave as one routed system that can emit live-compatible shadow JSONL without touching a broker order endpoint.
Data Used: Existing Protocol 075 stack artifacts, existing Protocol 081 artifacts, existing Protocol 077 lifecycle sequence data, existing official VIX bars, and an after-hours IBKR/TWS live probe. No paid data was downloaded and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_protocol081_live_shadow_router.py. Offline-smoke mode loads the Protocol 051/054 stack manifest and Protocol 081 artifact, routes causal lifecycle rows, emits shadow JSONL, and validates parity. ibkr-live-probe mode verifies whether a live market-hours capture can run and writes a blocked report instead of fake rows when the market/feed is unavailable.
Result: Offline router smoke passed: 1250 shadow rows, 1250 passed, 0 failed, actions hold 1200 / exit 12 / forced_flat 31 / stop 7. Explicit shadow parity rerun passed on the emitted JSONL. The live probe ran at 2026-05-12 20:08 Pacific, outside regular market hours, and correctly blocked with captured_rows 0 and outside_regular_market_hours. Live shadow parity on the empty after-hours log remained blocked.
Decision: The no-order router path is built and safe for a real regular-hours capture, but paper trading is still not approved. A true market-hours IBKR/TWS capture with fresh SPXW quotes is still required.
Next Gate: During regular market hours, run Protocol 088 ibkr-live-probe/capture with TWS paper market data active, capture non-empty live shadow JSONL, and rerun parity. Only after that should broker-connected paper execution be considered.
Owner: Codex
```

## 2026-05-13 Protocol 088 Market-Hours IBKR No-Order Capture Attempt

```text
Date: 2026-05-13
Decision / Experiment: Ran the Protocol 081 no-order live router against IB Gateway during regular market hours.
Reason: The next promotion gate required a real market-hours live shadow JSONL and parity rerun before any broker-connected paper order placement.
Data Used: IBKR market-data subscriptions only. No paid historical data was downloaded and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Extended v4/scripts/run_protocol081_live_shadow_router.py with ibkr-live-capture, automatic IB Gateway/TWS port fallback, JSON-safe blocked reports, live/delayed market-data classification, small SPXW 0DTE chain discovery, Black-Scholes live Greek computation from executable quotes, and no-order shadow JSONL emission. The harness still never creates or submits broker orders.
Result: IB Gateway was reachable on 127.0.0.1:4002. Live probe and live capture both blocked because IBKR returned error 354 for SPX and VIX: requested market data is not subscribed and delayed market data is available. The live capture JSONL had 0 rows, and the explicit live parity rerun remained blocked with 0 rows. A separate delayed-data plumbing check captured 24 SPXW 0DTE rows across ATM +/- $5, passed schema/freshness/no-order parity internally, and was correctly blocked as delayed_market_data_not_promotion_grade.
Decision: Do not approve paper order placement. The router and JSONL/parity plumbing are ready, but live market-data entitlements are not. Delayed IBKR data is useful only for plumbing checks, not for promotion-grade live confidence.
Next Gate: Enable live IBKR market data for Cboe SPX/VIX index quotes and OPRA/SPX options, then rerun ibkr-live-capture without --allow-delayed-market-data. Require non-empty fresh SPXW rows and live parity pass before any paper order test.
Owner: Codex
```

## 2026-05-13 Protocol 092 Serial Opportunity-Cost Neural Policy

```text
Date: 2026-05-13
Decision / Experiment: Built and ran the serial opportunity-cost entry policy on top of frozen Protocol 081 lifecycle exits.
Reason: Protocol 081 was strong historically but its selected paths can overlap, while the real bot can hold only one contract. The next neural question was whether an entry-time scorer can learn which candidate is worth occupying the single position slot, without changing exits.
Data Used: Existing Protocol 081 selected lifecycle exits and Protocol 077 lifecycle trade table. Q1 2025 training rows used Protocol 077/054 frozen lifecycle outcomes as a documented train-only fallback because Protocol 081 selected exits begin in Q2 2025. No paid data was downloaded, no live IBKR data was used, and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Added v4/model/serial_opportunity.py, v4/scripts/run_protocol092_serial_opportunity_policy.py, and v4/tests/test_serial_opportunity.py. The runner builds a 52,310-row candidate dataset, trains a Huber-plus-same-decision-ranking MLP across chronological folds, selects thresholds only on validation splits, and evaluates strict one-contract serial replay with $0.10 and $0.25 extra each-side stress.
Result: Reject for promotion. Protocol 092 stayed strongly positive across scored splits but did not beat the strict first-available serial baseline in Q3 2025 or Q4 2025. Median results: Q3 PnL 57,640 PF 4.007 vs baseline 57,790; Q4 PnL 90,140 PF 5.705 vs baseline 92,320; Q1 2026 PnL 94,940 vs baseline 90,000; March 2026 PnL 41,220 vs baseline 39,100. Positive seed fraction was 1.00 and $0.25 stress stayed positive on every reported split.
Decision: Keep the Protocol 092 infrastructure and dataset, but do not promote this scorer. The result says the entry-arbitration network preserves the edge and helps Q1/March, but it has not yet proven opportunity-cost improvement over the frozen strict serial baseline in earlier regimes.
Next Gate: Do attribution on Q3/Q4 baseline misses before adding another knob: identify whether the model skipped early serial winners, over-traded calls/puts, changed time buckets, or lost same-minute arbitration versus the first-available baseline. Use that to pre-register the next entry-side architecture or training objective.
Owner: Codex
```

## 2026-05-13 Protocol 093 Protocol092 vs Strict Serial Baseline Attribution

```text
Date: 2026-05-13
Decision / Experiment: Compared Protocol 092 against the strict first-available serial baseline trade-by-trade for Q3 2025 and Q4 2025.
Reason: Protocol 092 stayed strongly positive but failed the registered beat-baseline gate in Q3/Q4. Before adding another model knob, we needed to know whether the failure came from wrong side, put filtering, time buckets, same-minute contract ranking, or slot occupancy.
Data Used: Existing Protocol 092 serial policy trade ledger, strict serial baseline ledger, and Protocol 092 candidate dataset. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/attribute_protocol092_vs_serial_baseline.py and wrote v4/audit/autoresearch/v4_aplus_hypothesis_093_protocol092_serial_attribution/report.md with model-only, baseline-only, and same-minute swap ledgers.
Result: Q3 model total PnL was 286,760 vs 295,730 baseline, delta -8,970. Same-minute swaps were net +1,740, so ranking at the same decision minute was not the main failure. The damage came from flat threshold/no-entry (-7,660) and slot occupancy (-3,050), concentrated in post-open morning. Q4 aggregate model PnL was 456,680 vs 456,550 baseline, delta +130, but the registered median-vs-median gate still failed; seed 4 had -6,590, mostly post-open put-side selection. No broad wrong-side failure was found: same-minute swaps were overwhelmingly same-side. No broad put over-filter was found: put counts were close in Q3 and higher in Q4.
Decision: Do not add a crude side filter or put throttle. Keep Protocol 092 rejected for promotion, but use the attribution to shape the next hypothesis: threshold calibration plus slot-occupancy opportunity cost, with a secondary objective to preserve high-convexity same-minute winners.
Next Gate: Pre-register a Protocol 094 entry-only change that adjusts abstention/threshold and slot-occupancy learning without changing Protocol 081 exits or adding new data.
Owner: Codex
```

## 2026-05-13 Protocols 094-096 Serial Opportunity Follow-Up Loop

```text
Date: 2026-05-13
Decision / Experiment: Ran three pre-registered follow-ups after Protocol 093 showed Protocol 092's Q3/Q4 issue was threshold/slot occupancy rather than a broad side failure.
Reason: The next hypotheses needed to target opportunity cost without changing Protocol 081 exits, buying data, or adding crude call/put filters.
Data Used: Existing Protocol 092 candidate dataset and artifacts, plus existing Protocol 095 artifacts once produced. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Protocol 094 added v4/scripts/run_protocol094_opportunity_threshold.py and changed only validation threshold selection to maximize validation delta versus the strict serial baseline. Protocol 095 added v4/scripts/run_protocol095_slot_occupancy_target.py and changed only the training target to candidate_pnl minus the best positive candidate that would be blocked before exit. Protocol 096 added v4/scripts/run_protocol096_profit_slot_blend.py and selected a fixed-grid blend of the Protocol 092 profit scorer and Protocol 095 slot scorer on validation only.
Result: Protocol 094 was an exact no-op versus Protocol 092: Q3 57,640, Q4 90,140, Q1 94,940, March 41,220. Protocol 095 fixed Q3 and beat baseline there: Q3 60,570 vs 57,790 baseline, but hurt Q4 to 89,520 and Q1 2026 to 86,780, failing promotion. Protocol 096 preserved the useful Q3 improvement while reverting later splits toward Protocol 092: Q3 60,570 and beat baseline, Q4 90,140 but still below 92,320 baseline, Q1 94,940, March 41,220. It still failed promotion because Q4 did not beat the strict serial baseline.
Decision: Stop this hypothesis loop after three non-promoting follow-ups. Keep Protocol 096 as a research clue, not a replacement: slot awareness helps Q3 but does not solve Q4. Do not add more threshold/blend knobs now.
Next Gate: Move beyond independent candidate scoring. The next serious hypothesis should be a sequential candidate-set model that sees the ordered opportunity stream and learns "wait vs take" directly, or perform a Q4-specific attribution to determine why the first-available baseline is so hard to beat without tuning to Q4.
Owner: Codex
```

## 2026-05-13 Protocols 097-099 Sequential Event Policy

```text
Date: 2026-05-13
Decision / Experiment: Moved beyond independent candidate scoring into a sequential candidate-set policy with an explicit wait action.
Reason: Protocols 092-096 showed that the edge survives strict one-contract replay, but independent candidate scoring could not reliably beat the Q4 strict serial baseline. The next model needed to learn "wait vs take" directly from the ordered opportunity stream.
Data Used: Existing Protocol 092 candidate dataset and frozen Protocol 081 candidate exits. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_protocol097_sequential_event_policy.py. Protocol 097 builds decision-minute events with up to 10 candidates, labels wait/take actions with a train-period dynamic-programming oracle, trains a candidate-set neural policy, and selects a validation-only margin threshold. Protocol 098 added v4/scripts/run_protocol098_event_policy_utility_aux.py with one fixed candidate-PnL auxiliary loss. Protocol 099 added v4/scripts/run_protocol099_event_recall_threshold.py to test whether Q4 weakness was caused by overly timid wait thresholds.
Result: Protocol 097 is the best new research direction but not a promotion candidate. It improved every scored split versus Protocol 092: Q3 59,210 (+1,570), Q4 91,150 (+1,010), Q1 2026 95,290 (+350), March 42,290 (+1,070). It beat strict serial baseline in Q3, Q1, and March, but still missed Q4 baseline by 1,170. Protocol 098 improved Q3/Q4 versus Protocol 092 but damaged Q1/March and did not beat Q4. Protocol 099 rejected the recall-threshold hypothesis; it worsened Q3/Q4.
Decision: Keep Protocol 097 as the current best research-only architecture direction. Do not promote it, do not paper trade it, and do not add more threshold tweaks. The remaining Q4 gap is small but real and appears concentrated in seed-level opportunity sequencing, especially Q4 seed 3 under-trading/missed winners.
Next Gate: Attribute Protocol 097's Q4 seed-level failures against Protocol 092 and strict baseline, then either build a richer sequential model that includes short recent event history/state memory, or pause model work until broader historical data can test whether the Q4 gap is structural or sample-specific.
Owner: Codex
```

## 2026-05-13 Protocols 100-101 Q4 Attribution and Short-History Event Policy

```text
Date: 2026-05-13
Decision / Experiment: Attributed Protocol 097's Q4 seed-level failure, then tested causal short-history/state memory in the sequential event policy.
Reason: Protocol 097 was the first architecture to improve Q3/Q4/Q1/March versus Protocol 092, but it still missed Q4 strict baseline. The user requested Q4 seed-level attribution first, then short event-history/state memory if attribution supported it.
Data Used: Existing Protocol 092 candidate dataset, Protocol 097 trade ledgers/artifacts, and frozen Protocol 081 candidate exits. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/attribute_protocol097_q4_seed_failures.py. Protocol 100 found the Q4 gap was seed-concentrated: seed 3 was -18,610 versus strict baseline and under-traded by 59 trades, with 62,830 of positive baseline-only missed winners. Added v4/scripts/run_protocol101_event_history_policy.py. Protocol 101 appends causal previous-event and rolling-3 event summary features to each candidate before training the same wait/take event policy. Added a unit test to assert the history features use past events only.
Result: Protocol 101 cleared the registered strict serial gate across Q3, Q4, Q1 2026, and March 2026. Median PnL: Q3 59,130 vs 57,790 baseline; Q4 92,460 vs 92,320 baseline; Q1 2026 95,010 vs 90,000 baseline; March 41,790 vs 39,100 baseline. $0.10 and $0.25 each-side stresses remained positive on every scored split. Compared with Protocol 097, Protocol 101 improved Q4 by 1,310 but gave back 80 in Q3, 280 in Q1, and 500 in March.
Decision: Keep Protocol 101 as a research promotion candidate, not paper/live approval. It is the first v4 model to beat the strict serial baseline gate across all registered historical splits, but the Q4 margin is only +140 median PnL, so it needs independent historical validation before any broader confidence claim.
Next Gate: Pause model tweaks and run broader historical validation when approved/available. Specifically score Protocol 101 unchanged on additional locked periods, then follow with 1s/tick path audits and live shadow parity before any paper trading.
Owner: Codex
```

## 2026-05-13 Protocol 102 Protocol101 Readiness Diagnostics

```text
Date: 2026-05-13
Decision / Experiment: Froze Protocol 101 as the current research promotion candidate and ran promotion-readiness diagnostics without changing the model.
Reason: Protocol 101 was the first v4 policy to clear the registered strict serial baseline gate across Q3 2025, Q4 2025, Q1 2026, and March 2026, but its Q4 margin was thin and paper/live readiness still needed a formal blocker list.
Data Used: Existing Protocol 101 event-history policy outputs, existing Protocol 092 strict serial baseline artifacts, and existing v4 audit outputs. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_protocol102_protocol101_readiness.py. The script writes v4/audit/autoresearch/v4_aplus_hypothesis_102_protocol101_readiness/report.md, v4/audit/autoresearch/v4_aplus_hypothesis_102_protocol101_readiness/summary.json, v4/promotion/PROTOCOL_101_FREEZE.json, and v4/promotion/PROTOCOL_101_PROMOTION_READINESS_PACKET.md.
Result: Decision is research_promotion_candidate_needs_broader_validation. Protocol 101 beat the strict serial baseline on the registered median gate: Q3 +1340, Q4 +140, Q1 2026 +5010, and March +2690. Stress remained positive at +0.25 each side. Readiness diagnostics also found seed-margin fragility versus baseline: positive seed-margin fractions were Q3 0.60, Q4 0.60, Q1 0.80, and March 0.60. Month diagnostics showed July 2025 -2020 and December 2025 -7500 versus baseline, while August, September, October, November, January, February, and March were positive versus baseline. Concentration was not extreme: top day profit share 3.1%, top five day profit share 12.1%, with 162 positive days and 26 negative days.
Decision: Keep Protocol 101 frozen as the current research promotion candidate, not as paper/live approval. Stop entry/exit knob tweaking until broader locked validation or path/live-readiness checks justify the next move.
Next Gate: Broader locked historical validation of Protocol 101 unchanged, with explicit user approval before any paid data download. After broader validation, run Protocol 101 selected-trade 1s/tick path audit and no-order live shadow parity before any broker-connected paper trading.
Owner: Codex
```

## 2026-05-13 Protocol 103 Protocol101 External-Audit Readiness

```text
Date: 2026-05-13
Decision / Experiment: Checked whether the already-collected Q4 2024 official-context block can be scored as a frozen Protocol 101 external audit without buying data or changing the model.
Reason: Protocol 101 cleared the registered strict serial gate but has a thin Q4 2025 margin. Before asking for more data, the project should use already-collected Q4 2024 if it can be scored honestly.
Data Used: Existing Q4 2024 official-context integrity/build summaries, Protocol 101 artifacts, Protocol 092 candidate dataset, Protocol 081 selected exits, Protocol 077 lifecycle trade table, and Protocol 076 prehistory output. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_protocol103_protocol101_external_audit_readiness.py and wrote v4/audit/autoresearch/v4_aplus_hypothesis_103_protocol101_external_audit_readiness/report.md plus summary.json.
Result: Q4 2024 official-context data is clean: integrity pass, 64 sessions built. Protocol 101 artifacts exist, but Q4 2024 cannot be scored as Protocol 101 unchanged yet because no q4_2024 Protocol 092-style candidate dataset exists, no q4_2024 frozen Protocol 081 candidate exits exist, and Protocol 076 used Q4 2024 as prehistory rather than producing Q4 2024 test rows.
Decision: Treat the Q4 2024 external audit as blocked by missing candidate outcomes, not by data quality. Do not buy more data until the local candidate-outcome builder exists or the user explicitly approves a paid batch with source/date/schema/symbol/cap details.
Next Gate: Build a no-training Q4 2024 candidate builder that applies frozen Protocol 051 entries, builds same-contract lifecycle paths from normalized official-context data, applies frozen Protocol 081 exits, creates a Protocol 092-compatible q4_2024_external candidate table, and then scores it with frozen Protocol 101 artifacts as a temporal-regime stress audit.
Owner: Codex
```

## 2026-05-13 Protocols 104-107 Q4 2024 External Temporal Stress

```text
Date: 2026-05-13
Decision / Experiment: Built and scored a no-training Q4 2024 external stress path for frozen Protocol 101 using already-collected official-context data.
Reason: Protocol 103 showed Q4 2024 data was clean but missing Protocol 101-compatible candidate outcomes. The next no-cost step was to construct those outcomes locally rather than buy more data.
Data Used: Existing Q4 2024 official-context processed sessions, existing normalized Q4 2024 quote paths, frozen Protocol 075 Protocol 051/054 stack artifacts, frozen Protocol 081 sequence artifacts, and frozen Protocol 101 event-history artifacts. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/build_protocol101_q4_2024_external_candidates.py, v4/scripts/apply_protocol081_to_q4_2024_external_sequence.py, and v4/scripts/score_protocol101_q4_2024_external.py. Protocol 104 applied frozen Protocol 075 entry/risk artifacts to all 64 Q4 2024 sessions and wrote 1098 Protocol 054 candidate rows with zero path misses. Protocol 105 reused the lifecycle sequence builder and wrote 1098 trade rows and 27450 step rows, all path_status ok. Protocol 106 applied five frozen Protocol 081 sequence seeds and wrote 5490 candidate-exit rows. Protocol 107 built a Protocol 092-compatible q4_2024_external candidate table and scored frozen Protocol 101 fold-3 artifacts.
Result: Protocol 107 passed as a research-only temporal stress audit, not a chronological promotion gate. Aggregate median PnL was 58280 versus strict serial baseline median 54590, median margin +1290, median PF 3.460, positive seed fraction 1.00, +0.25 each-side stress median PnL 47730. Seed margins versus strict baseline were +3690, +5380, -2140, -16770, and +1290, so positive seed-margin fraction was 0.60.
Decision: This supports Protocol 101 as the current research candidate because it did not collapse on the already-collected pre-2025 block. It is still not paper/live approval, and it does not replace chronological broader validation because the Q4 2024 stress uses 2025-trained frozen artifacts applied backward.
Next Gate: Prepare an explicit paid-data request for Q3 2024 or another adjacent locked quarter, but only after user approval. The request should include Databento OPRA.PILLAR definition/cbbo-1m/ohlcv-1m/statistics for SPXW 0DTE, ThetaData SPX/VIX 1-minute bars, estimated Databento cost $25-$36, and hard cap $45.
Owner: Codex
```

## 2026-05-13 Protocol 108 Protocol107 Q4 2024 External Attribution

```text
Date: 2026-05-13
Decision / Experiment: Attributed frozen Protocol 101's Q4 2024 external stress against the strict serial baseline before changing the model or requesting more data.
Reason: Protocol 107 passed the median external stress but only 3 of 5 seeds beat strict serial baseline. The project needed to know whether the weakness came from wrong-side selection, worse time buckets, same-minute contract swaps, over-filtering, or missed convex winners.
Data Used: Existing Protocol 107 model and baseline trade ledgers plus the q4_2024_external Protocol 092-compatible candidate table. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/attribute_protocol107_q4_2024_external.py. The attribution compares model and strict-baseline trade sets by seed, session, decision_time, side, offset, and PnL; separates same-minute swaps, slot occupancy, and flat-threshold/no-entry misses; and writes v4/audit/autoresearch/v4_aplus_hypothesis_108_protocol107_q4_2024_external_attribution/report.md plus JSON ledgers.
Result: Protocol 101 remains supportive but seed-fragile on Q4 2024. Aggregate model PnL was 261080 versus strict serial baseline 269630, delta -8550, while the median seed result from Protocol 107 remained positive. The largest overall drag was flat threshold/no-entry at -25110, partly offset by same-minute contract swaps at +14540 and slot occupancy at +2020. Worst seed 4 was -16770 versus baseline, driven by flat threshold/no-entry -12700 and slot occupancy -7140. The weakness is mostly post-open calls: overall side deltas were calls -10350 and puts +1800; seed 4 side deltas were calls -15100 and puts -1670. Baseline-only convex winners >= 1000 totaled 83780 across 54 trades.
Decision: Do not add a hand-coded side/time filter. The attribution suggests the model sometimes abstains from or is already holding through post-open convex call winners, but same-minute contract ranking is net helpful. Keep Protocol 101 frozen as the research candidate and treat Q4 2024 as supportive but not sufficient for paper/live approval.
Next Gate: Broader chronological validation, preferably Q3 2024, requires explicit user approval before any paid data download. If the same seed fragility persists there, the next architecture should use richer event-state memory or seed ensembling rather than another entry-side threshold knob.
Owner: Codex
```

## 2026-05-13 Protocol 109 Frozen Protocol101 Seed Ensemble

```text
Date: 2026-05-13
Decision / Experiment: Tested whether averaging the five frozen Protocol 101 seed logits reduces Q4 2024 external seed fragility without retraining.
Reason: Protocol 108 showed that Protocol 101's Q4 2024 external weakness was not a same-minute ranking failure; the largest drag was flat threshold/no-entry behavior and the worst case was seed-specific. A seed ensemble was a no-paid way to test whether this is mostly initialization noise.
Data Used: Existing Protocol 092 candidate dataset, existing Protocol 101 frozen model artifacts, and existing Protocol 107 q4_2024_external candidate table. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_protocol109_frozen_seed_ensemble.py. It performs no training, averages logits from frozen Protocol 101 seed artifacts, selects thresholds only on each fold's registered validation split, and applies Q4 2025-selected fold-3 thresholds to the Q4 2024 external stress.
Result: Rejected. The ensemble improved Q4 2024 seed 4 from Protocol 107's -16770 margin to -1920, but it damaged the registered gate. Registered median PnL versus Protocol 101 changed by Q3 -910, Q4 -900, Q1 2026 -3580, and March -2710. It failed the strict serial baseline check in Q4 2025 by -760 and March by -20. Q4 2024 external remained positive on median PnL at 56160 versus baseline 54590, but positive seed-margin fraction stayed 0.60.
Decision: Do not replace Protocol 101 with the frozen seed ensemble. Protocol 101 remains the frozen research candidate. The seed-ensemble idea helped one external seed but over-smoothed useful registered-split behavior.
Next Gate: Pause model tweaks unless there is a new non-knob architecture hypothesis. The cleaner next evidence step is broader chronological validation on another locked quarter, with explicit user approval before any paid data download.
Owner: Codex
```

## 2026-05-13 Protocol 110 Q3 2024 Data Batch Preflight

```text
Date: 2026-05-13
Decision / Experiment: Prepared a no-paid preflight and approval manifest for the next chronological validation block, Q3 2024.
Reason: Protocol 101 remains the frozen research candidate, Protocol 108 identified external abstention fragility, and Protocol 109 rejected seed ensembling. The next useful evidence is a broader locked quarter, but the project guardrail requires explicit approval before paid/licensed data requests.
Data Used: Local filesystem only. Scanned existing Databento raw folders, ThetaData SPX/VIX folders, normalized official-context files, and promotion request documents. No Databento, ThetaData, Cboe, IBKR, or other market-data endpoint was called.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/run_protocol110_q3_2024_data_batch_preflight.py. It verifies existing coverage continuity, enumerates the 64 expected Q3 2024 trading sessions, writes v4/audit/autoresearch/v4_aplus_hypothesis_110_q3_2024_data_batch_preflight/report.md, and writes v4/promotion/PROTOCOL_101_Q3_2024_DOWNLOAD_MANIFEST.json.
Result: Ready for explicit user approval. Existing local coverage is continuous for 375 expected sessions from 2024-10-01 through 2026-03-31 across Databento definitions, cbbo-1m, ohlcv-1m, statistics, ThetaData SPX/VIX, and normalized official-context outputs. Q3 2024 has 64 expected trading sessions from 2024-07-01 through 2024-09-30 and none are present locally. ThetaData VIX has 10 extra holiday files in the existing block, but no missing expected sessions.
Decision: Stop before any paid/licensed data request. The exact approval text and blocked commands are recorded in the manifest.
Next Gate: Only after explicit approval, download Q3 2024 Databento OPRA.PILLAR definition/cbbo-1m/ohlcv-1m/statistics for filtered SPXW 0DTE raw symbols under a $45 Databento hard cap, plus ThetaData SPX/VIX 1-minute bars through the existing subscription.
Owner: Codex
```

## 2026-05-13 Protocol 111 Paid Data Guardrail Hardening

```text
Date: 2026-05-13
Decision / Experiment: Converted the paid-data approval rule from documentation-only discipline into executable checks for the Q3 2024 download path.
Reason: The user asked to continue, but had not provided the exact approval text. The safest no-cost work was to make accidental paid downloads impossible from the relevant scripts.
Data Used: Local code and manifest files only. No Databento, ThetaData, Cboe, IBKR, or other market-data endpoint was called for data. A ThetaData dry-run printed only a planned request; a no-approval download attempt exited at the guard before client initialization.
Cost: $0 incremental paid data.
Implementation: Added v4/checks/paid_data_guard.py, added v4/tests/test_paid_data_guard.py, wired approval args into v4/scripts/download_thetadata_index_bars.py and v4/scripts/download_databento_pilot.py, regenerated v4/promotion/PROTOCOL_101_Q3_2024_DOWNLOAD_MANIFEST.json through Protocol 110, and wrote v4/audit/autoresearch/v4_aplus_hypothesis_111_paid_data_guardrail/report.md.
Result: Implemented. Dry-runs and auth checks remain allowed without approval; Databento cost estimation remains allowed without approval; ThetaData index_history_ohlc and Databento timeseries.get_range downloads require the exact manifest approval text via --approval-text or V4_PAID_DATA_APPROVAL_TEXT.
Decision: Keep Q3 2024 blocked until explicit approval. The code now enforces that boundary for the proposed Q3 path.
Next Gate: If the user provides the exact manifest approval text, run the approval-gated Q3 2024 download commands. Otherwise continue only with no-paid local research/infrastructure.
Owner: Codex
```

## 2026-05-13 Protocol 113 Protocol101 Trade Charts

```text
Date: 2026-05-13
Decision / Experiment: Recreated the v2/v3-style visual inspection artifacts for v4 Protocol 101: trades.html, equity.html, and trades.csv.
Reason: The user wanted to visually inspect selected trades on an SPX historical chart and inspect the equity curve without downloading more data.
Data Used: Existing local ThetaData SPX 1-minute bars, Protocol 101 selected serial trades, Protocol 107 Q4 2024 external-stress trades, and Protocol 112 train/validation replay rows for Q1/Q2 2025 continuity. No paid data was downloaded, no market-data endpoint was called, and no broker endpoint was used.
Cost: $0 incremental paid data.
Implementation: Added v4/scripts/export_protocol101_trade_charts.py. The script writes standalone dependency-free HTML using embedded data and canvas rendering, joins SPX entry/exit prices onto trades, avoids duplicate March rows, labels train/validation/test/external_stress segments, and emits a CSV used by both charts.
Result: Wrote v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.html, equity.html, trades.csv, report.md, and summary.json. The export contains 6989 replay trades across seeds 1-5, 369 traded sessions, and 145728 SPX bars from 2024-10-01 through 2026-03-31. Every trade has an SPX entry and exit price attached. Training/validation rows are included for visual continuity only and are explicitly labeled as such.
Decision: Keep these as inspection artifacts, not promotion evidence. The equity curve starts from the configurable displayed starting balance, default $10000, but Protocol 101 itself still uses replay PnL and does not enforce account equity or compounding.
Next Gate: Use the charts to inspect clustered winners/losers, post-open missed convexity behavior, and whether the model is visually entering after actual SPX structure. No paid data is needed for this inspection.
Owner: Codex
```


## 2026-05-13 Protocol 114 Skeptical Falsification

```text
Date: 2026-05-13
Decision / Experiment: Ran a no-training skeptical falsification audit for frozen Protocol 101 before any further neural architecture work.
Reason: The equity curve is exciting enough to be dangerous. The project needed to try to disprove the Protocol 101 edge with baselines, matched random, concentration, paper-account realism, slippage, and one-minute quote-delay stress.
Data Used: Existing Protocol 101, Protocol 107, Protocol 092, Protocol 097, Protocol 113, and normalized official-context artifacts only. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision fragile_needs_more_data. Headline trades 4952; model trades audited 5493; stressed ending and detailed split results are in v4/audit/autoresearch/v4_aplus_hypothesis_114_protocol101_skeptical_falsification/report.md.
Next Gate: prioritize 1s/tick/live-shadow validation because edge is timing-sensitive
Owner: Codex
```


## 2026-05-13 Protocol 115 Protocol101 Existing CBBO-1s Validation

```text
Date: 2026-05-13
Decision / Experiment: Replayed frozen Protocol 101 selected trades on already-collected Databento CBBO-1s slices.
Reason: Protocol 114 showed the Protocol 101 equity curve is timing-sensitive under one-minute delayed entry/exit stress. Before buying broad history or designing a bigger neural network, the project needed to check whether existing one-second evidence contradicts the one-minute executable replay.
Data Used: Existing Protocol 101 and Protocol 107 selected trades, existing normalized official-context symbol maps, and existing local CBBO-1s audit files only. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision partial_support_needs_targeted_1s_or_live_shadow. Audited 524 of 5493 Protocol 101 rows; detailed split coverage and repricing are in v4/audit/autoresearch/v4_aplus_hypothesis_115_protocol101_existing_1s_path_audit/report.md.
Next Gate: do not buy broad history yet; either run no-order live shadow when IBKR is ready, or request a tightly capped targeted CBBO-1s batch for missing Protocol 101 sessions
Owner: Codex
```


## 2026-05-13 Protocol 116 Protocol101 Targeted High-Resolution Request

```text
Date: 2026-05-13
Decision / Experiment: Prepared a no-download targeted CMBP-1/CBBO-1s validation request for frozen Protocol 101.
Reason: Protocol 114 showed timing sensitivity and Protocol 115 found supportive but incomplete one-second evidence. The next paid step, if approved, should be selected-contract one-second validation rather than broad historical backfill.
Data Used: Existing Protocol 115 replay rows and local Databento CBBO-1s coverage metadata only. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data. Estimated requested Databento cost is $4.7930 with proposed hard cap $10.00.
Result: Decision ready_for_explicit_user_approval. Request and exact approval text are in v4/audit/autoresearch/v4_aplus_hypothesis_116_protocol101_targeted_1s_request/request.md.
Next Gate: Only run a paid high-resolution Databento download if the user provides the exact approval text from the Protocol 116 request.
Owner: Codex
```


## 2026-05-14 Protocol 117 Protocol101 Targeted High-Resolution Validation

```text
Date: 2026-05-14
Decision / Experiment: Downloaded the explicitly approved Protocol 116 targeted Databento high-resolution batch and replayed frozen Protocol 101 selected trades against it.
Reason: Protocol 114 showed one-minute timing sensitivity and Protocol 115 had supportive but incomplete one-second evidence. The project needed selected-contract high-resolution validation before trusting the Protocol 101 equity curve further.
Data Used: Databento OPRA.PILLAR CMBP-1 for 2024 sessions and CBBO-1s for 2025-02-20 onward, limited to Protocol 101 selected SPXW raw symbols from the manifest. No live broker data was used and no orders were placed.
Cost: Estimated approved Databento cost $4.7930, hard cap $10.00. Download artifacts occupy the local Protocol 101 overlay directory.
Result: Decision covered_1s_replay_supports_protocol101_timing_assumption. Audited 5493 of 5493 Protocol 101 rows with 100.0% coverage; detailed split repricing is in v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_validation/report.md.
Next Gate: run no-order live shadow parity for frozen Protocol 101 before broker-connected paper trading
Owner: Codex
```

## 2026-05-14 Protocol 118 Protocol101 No-Order Shadow Rehearsal

```text
Date: 2026-05-14
Decision / Experiment: Built a historical no-order shadow-router rehearsal for frozen Protocol 101 using the Protocol 117 targeted high-resolution replay.
Reason: Before broker-connected paper trading, the project needed to prove the frozen selected-trade behavior can be emitted as a strict no-order JSONL stream with one contract, no overlap, ask-entry/bid-exit accounting, official historical context, and end-flat behavior.
Data Used: Existing Protocol 117 high-resolution replay, existing Protocol 092/107 candidate datasets, and existing local official SPX/VIX context files only. No paid data was downloaded, no live broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_historical_no_order_shadow_rehearsal_live_capture_next. Rehearsed 1028 seed-1 trades with max concurrent positions 1 and 0 unaffordable trades from $10,000.00 starting cash.
Next Gate: Run a real no-order live shadow capture for Protocol 101 when IBKR market data/cash status is ready; keep order placement disabled until that live parity gate passes.
Owner: Codex
```

## 2026-05-14 Protocol 119 Protocol101 Live Readiness

```text
Date: 2026-05-14
Decision / Experiment: Checked whether frozen Protocol 101 is ready for no-order live shadow capture.
Reason: Protocol 118 passed historical no-order rehearsal, so the next gate is live-readiness rather than another historical model tweak.
Data Used: Existing Protocol 101 artifacts, Protocol 118 summary, and the no-order IBKR entitlement check output only. No paid data was downloaded, this script did not call IBKR, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision blocked_missing_live_market_data_entitlements. Report: v4/audit/autoresearch/v4_aplus_hypothesis_119_protocol101_live_readiness/report.md
Next Gate: Resolve the listed blockers before Protocol 101 no-order live capture: IB Gateway is reachable, but live SPX/VIX and OPRA/SPXW market data are not available to this API session. Check market-data subscriptions and API market-data acknowledgement., Rerun the no-order entitlement check and require live SPX, VIX, and OPRA/SPXW NBBO market data before Protocol 101 no-order live capture.
Owner: Codex
```

## 2026-05-14 Protocol 120 Surface Edge Portability

```text
Date: 2026-05-14
Decision / Experiment: Loaded the frozen Protocol 051/A+ surface scorer as a reusable edge generator and scored cached official-context SurfaceDecision rows.
Reason: Protocol 119 identified the upstream `edge` feature family as a blocker for Protocol 101 live deployment. The project needed to prove the frozen scorer can be loaded independently before wiring it into the live entry router.
Data Used: Existing Protocol 075 surface artifact and local cached official-context surface decisions only. No paid data was downloaded, no live broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_surface_edge_generator_loads_offline_live_router_next. Scored 1000 decisions with finite_edge_fraction=1.000. Report: v4/audit/autoresearch/v4_aplus_hypothesis_120_surface_edge_portability/report.md
Next Gate: Wire the surface edge scorer into the Protocol 101 no-order live entry router, then rerun Protocol 119.
Owner: Codex
```

## 2026-05-14 Protocol 121 Protocol101 Entry Router Smoke

```text
Date: 2026-05-14
Decision / Experiment: Wired the frozen Protocol 051/A+ surface edge scorer into the frozen Protocol 101 entry-policy inference path and smoke-tested it on cached official-context SurfaceDecision rows.
Reason: Protocol 119 showed that Protocol 101 live deployment cannot zero-fill `edge`; this test proves the model path can produce edge and edge-history features causally before broker live capture.
Data Used: Existing Protocol 075 surface artifact, Protocol 101 artifact, and local cached official-context surface decisions only. No paid data was downloaded, no live broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_protocol101_entry_router_edge_wired. Events with candidates=266; candidate_feature_rows=1360; nonfinite_feature_rows=0. Report: v4/audit/autoresearch/v4_aplus_hypothesis_121_protocol101_entry_router_smoke/report.md
Next Gate: Rerun Protocol 119; if IBKR live data is available, run no-order Protocol 101 live capture.
Owner: Codex
```

## 2026-05-14 Protocol 122 Protocol101 Capital Realism

```text
Date: 2026-05-14
Decision / Experiment: Ran a no-training capital realism audit for frozen Protocol 101 with $10,000 as the intended paper/live trading baseline and $500 treated only as the IBKR access reserve.
Reason: Live paper execution is blocked by cash settlement and live market-data subscriptions. The project needed to prevent confusing the $500 account-access cash with the $10,000 paper/live trading bankroll.
Data Used: Existing Protocol 101/107 trade artifacts, local official SPX bars, and normalized local option quote files only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_10000_paper_capital_baseline. Report: v4/audit/autoresearch/v4_aplus_hypothesis_122_protocol101_capital_realism/report.md
Next Gate: Use $10,000 as the paper-account and eventual real-money capital baseline. The $500 IBKR cash is only an access reserve. Next research should focus on live-data parity and order-state rehearsal, not shrinking the strategy to fit a $500 trading account.
Owner: Codex
```

## 2026-05-14 Protocol 123 Protocol101 Order-State Rehearsal

```text
Date: 2026-05-14
Decision / Experiment: Rehearsed frozen Protocol101 paper trades through the v4 order-state machine using $10,000 starting paper cash and treating $500 only as the IBKR access reserve.
Reason: Cash settlement blocks paper orders, so the next no-cost promotion-readiness work is order-state accounting around the intended $10,000 bankroll.
Data Used: Existing Protocol101/107 trade artifacts and local normalized quote files only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_10000_order_state_rehearsal_live_data_pending. Rehearsed 1028 trades; skipped 0; all_exit_filled=True; max_concurrent_positions=1. Report: v4/audit/autoresearch/v4_aplus_hypothesis_123_protocol101_order_state_rehearsal/report.md
Next Gate: Run no-order Protocol101 live shadow capture once live SPX/VIX and OPRA/SPXW subscriptions are active. Do not place paper orders until live-data parity passes.
Owner: Codex
```

## 2026-05-14 Protocol 124 Protocol101 Live-Data Parity Checkpoint

```text
Date: 2026-05-14
Decision / Experiment: Consolidated Protocol101 live-data parity state after delayed IBKR plumbing succeeded but live subscriptions remained unavailable.
Reason: The current work queue is live-data parity, no-order Protocol101 shadow capture, and order-state rehearsal around a $10,000 paper account. This checkpoint keeps delayed plumbing evidence separate from promotion-grade live-data evidence.
Data Used: Existing IBKR entitlement summary, Protocol119 readiness summary, Protocol121 entry-router smoke summary, and latest no-order delayed capture summary. No paid data was downloaded, no broker order endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision blocked_live_subscriptions_delayed_plumbing_passed. Report: v4/audit/autoresearch/v4_aplus_hypothesis_124_protocol101_live_data_parity_checkpoint/report.md
Next Gate: Enable live Cboe index market data for SPX and VIX in the IBKR API session.; Enable live OPRA top-of-book data so SPXW option NBBO is available to the API session.
Owner: Codex
```

## 2026-05-13 Protocol 114 Skeptical Falsification

```text
Date: 2026-05-13
Decision / Experiment: Ran a no-training skeptical falsification audit for frozen Protocol 101 before any further neural architecture work.
Reason: The equity curve is exciting enough to be dangerous. The project needed to try to disprove the Protocol 101 edge with baselines, matched random, concentration, paper-account realism, slippage, and one-minute quote-delay stress.
Data Used: Existing Protocol 101, Protocol 107, Protocol 092, Protocol 097, Protocol 113, and normalized official-context artifacts only. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision fragile_needs_more_data. Headline trades 4952; model trades audited 5493; stressed ending and detailed split results are in v4/audit/autoresearch/v4_aplus_hypothesis_114_protocol101_skeptical_falsification/report.md.
Next Gate: prioritize 1s/tick/live-shadow validation because edge is timing-sensitive
Owner: Codex
```

## 2026-05-14 Protocol 125 Protocol101 Pre-Tuesday Readiness Pack

```text
Date: 2026-05-14
Decision / Experiment: Packaged frozen Protocol101 replay realism, execution-skepticism status, no-order shadow event contract, visual inspection targets, and the Tuesday live-shadow checklist.
Reason: IBKR cash settlement blocks live paper trading, but the project still needed the Tuesday no-order shadow run to be precise instead of improvised.
Data Used: Existing Protocol101/107/113/114/118/123/124 artifacts only. No paid data was downloaded, no live broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision ready_for_tuesday_no_order_live_shadow_only. Shadow event verification pass with 2056 events and max_open_positions=1. Report: v4/audit/autoresearch/v4_aplus_hypothesis_125_protocol101_pre_tuesday_readiness/report.md
Next Gate: On Tuesday, run Protocol101 no-order live shadow capture first. Compare emitted JSONL to protocol101_shadow_event_schema.json and keep paper orders disabled until live feature/quote parity passes.
Owner: Codex
```

## 2026-05-14 Protocol 126 Protocol101 Timing Fragility Hardening

```text
Date: 2026-05-14
Decision / Experiment: Repriced frozen Protocol101 selected trades under sub-minute entry/exit delay stress using already-collected high-resolution quote data.
Reason: Protocol 114 found one-minute timing fragility; the project needed a tighter timing budget and live-style expiry rule before paper trading.
Data Used: Existing Protocol 117 high-resolution replay and local high-resolution quote files only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision blocked_incomplete_timing_coverage. Delay rows 32958; report v4/audit/autoresearch/v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening/report.md.
Next Gate: Use this expiry rule in no-order live shadow and paper-order rehearsal. Protocol 101 is not paper-order ready until live rows prove quote/context freshness inside budget.
Owner: Codex
```

## 2026-05-14 Protocol 127 Protocol101 Live Shadow Schema Hardening

```text
Date: 2026-05-14
Decision / Experiment: Defined and validated the expanded Protocol101 no-order live shadow schema.
Reason: Tuesday needs to capture all decisions, risk blocks, market snapshots, and account state before any paper-order rehearsal.
Data Used: Existing Protocol113 trade replay only for examples. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_schema_hardening_ready_for_live_capture. Schema v4/audit/autoresearch/v4_aplus_hypothesis_127_protocol101_live_shadow_schema_hardening/protocol101_shadow_v2_schema.json; examples v4/audit/autoresearch/v4_aplus_hypothesis_127_protocol101_live_shadow_schema_hardening/protocol101_shadow_v2_examples.jsonl.
Next Gate: Tuesday live shadow capture must emit this schema and pass validation before order-state rehearsal or any request for paper-order approval.
Owner: Codex
```

## 2026-05-14 Protocol 129 Protocol101 Offline Position Sizing

```text
Date: 2026-05-14
Decision / Experiment: Tested adaptive 1-to-3 contract sizing offline while keeping Protocol101 entries and exits frozen.
Reason: Position scaling should not be introduced into Tuesday paper trading until it proves it improves return without materially worsening drawdown or loss clustering.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision reject_scaling_loss_clustering_worse. Report v4/audit/autoresearch/v4_aplus_hypothesis_129_protocol101_offline_position_sizing/report.md.
Next Gate: Reject scaling for initial paper trading. Keep one contract as the live-paper constraint and revisit sizing only after the one-contract path survives live shadow and paper replay.
Owner: Codex
```

## 2026-05-14 Protocol 128 Protocol101 Paper Risk Gate

```text
Date: 2026-05-14
Decision / Experiment: Added a deterministic $10,000 paper-account risk gate around frozen Protocol101.
Reason: Live paper trading needs explicit stale-data, premium, overlap, daily-loss, and settlement blocks before any broker endpoint is enabled.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_paper_risk_gate_overlay_ready_for_live_shadow. Report v4/audit/autoresearch/v4_aplus_hypothesis_128_protocol101_paper_risk_gate/report.md.
Next Gate: Use this same risk gate in Tuesday's no-order shadow stream. Do not enable any paper-order endpoint until the live schema, quote freshness, and order-state rehearsal all pass.
Owner: Codex
```

## 2026-05-14 Protocol 130 Protocol101 Tuesday No-Order Live Shadow Runbook

```text
Date: 2026-05-14
Decision / Experiment: Prepared the Tuesday no-order live-shadow runbook and preflight contract for frozen Protocol101.
Reason: The next live session must test data parity, schema, quote freshness, and order-state accounting before any paper order is considered.
Data Used: Existing hardening summaries only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision ready_for_tuesday_no_order_live_shadow_only. Runbook v4/audit/autoresearch/v4_aplus_hypothesis_130_protocol101_tuesday_no_order_live_shadow_runbook/tuesday_no_order_live_shadow_runbook.md.
Next Gate: Tuesday is a no-order live data parity experiment. Paper-order rehearsal happens only after schema, freshness, risk-gate, and order-state checks pass on captured live rows.
Owner: Codex
```

## 2026-05-14 Protocol 131 Protocol101 Multi-Contract P&L Tracking

```text
Date: 2026-05-14
Decision / Experiment: Tested offline multi-contract P&L tracking policies around frozen Protocol101.
Reason: User asked whether the model could scale lots only after profits; this requires account-state tracking before any leverage touches paper trading.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision reject_multi_contract_risk_not_improved_enough. Best candidate conservative_profit_ladder. Report v4/audit/autoresearch/v4_aplus_hypothesis_131_protocol101_multi_contract_pnl_tracking/report.md.
Next Gate: Keep Tuesday and initial paper trading at one contract. Multi-contract sizing remains rejected until a safer policy improves PnL without worsening drawdown or loss clustering.
Owner: Codex
```

## 2026-05-14 Protocol 132 Protocol101 Confidence Sizing Autoresearch

```text
Date: 2026-05-14
Decision / Experiment: Ran a confidence/account-aware sizing autoresearch loop around frozen Protocol101.
Reason: Long-term bot behavior may scale contracts as a $10,000 account grows, but only if account state and model confidence reduce the loss-clustering problem.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_confidence_sizing_research_candidate_not_live. Accepted 3 sizing hypotheses. Report v4/audit/autoresearch/v4_aplus_hypothesis_132_protocol101_confidence_sizing_autoresearch/report.md.
Next Gate: Treat the accepted policy as offline research only. It needs split-by-split attribution and live one-contract parity before any multi-contract paper test.
Owner: Codex
```

## 2026-05-14 Protocol 133 Protocol101 Sizing Split Stress

```text
Date: 2026-05-14
Decision / Experiment: Validated accepted Protocol132 sizing policies by split, month, day, and slippage stress.
Reason: Aggregate multi-contract PnL is not enough; sizing must not simply move loss clustering into a later month or disappear under execution stress.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_sizing_candidate_survives_split_stress_not_live. Best candidate lower_two_contract_threshold. Report v4/audit/autoresearch/v4_aplus_hypothesis_133_protocol101_sizing_split_stress/report.md.
Next Gate: Run trade-set attribution for the surviving sizing policy and keep it offline until one-contract live paper parity is proven.
Owner: Codex
```

## 2026-05-14 Protocol 134 Protocol101 Sizing Attribution

```text
Date: 2026-05-14
Decision / Experiment: Attributed the surviving Protocol133 sizing policy trade-by-trade against the one-contract baseline.
Reason: Multi-contract sizing must prove it is not just a few large scaled winners masking concentrated risk.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_sizing_attribution_research_candidate_not_live. Incremental PnL 34950.0. Report v4/audit/autoresearch/v4_aplus_hypothesis_134_protocol101_sizing_attribution/report.md.
Next Gate: Keep the sizing policy as an offline research candidate. Next test should add split-by-split equity curves and paper-account visualization, not live multi-contract trading.
Owner: Codex
```

## 2026-05-14 Protocol 135 Protocol101 Starting-Cash Sensitivity

```text
Date: 2026-05-14
Decision / Experiment: Tested one always-on account-aware sizing policy across multiple starting paper account sizes and slippage stresses.
Reason: The production goal is a turn-it-on bot that scales risk from account state, not manual model selection.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_always_on_sizer_starting_cash_sensitivity_not_live. Report v4/audit/autoresearch/v4_aplus_hypothesis_135_protocol101_starting_cash_sensitivity/report.md.
Next Gate: Promote the account-aware sizer to an offline research candidate artifact, then test equity visualizations and live-shadow-compatible account-state serialization.
Owner: Codex
```

## 2026-05-14 Protocol 136 Protocol101 Account-Scaled Daily Stop

```text
Date: 2026-05-14
Decision / Experiment: Tested account-scaled daily loss stops for the always-on Protocol101 sizing policy.
Reason: A fixed daily stop can be too small after account growth; the real bot needs risk controls that scale with equity without amplifying drawdown.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_account_scaled_daily_stop_candidate_not_live. Accepted 2 daily-stop hypotheses. Report v4/audit/autoresearch/v4_aplus_hypothesis_136_protocol101_account_scaled_daily_stop/report.md.
Next Gate: Fold the accepted daily-stop rule into the single offline sizing candidate and rerun attribution/visualization.
Owner: Codex
```

## 2026-05-14 Protocol 137 Protocol101 Account-Aware Sizer V1

```text
Date: 2026-05-14
Decision / Experiment: Consolidated the best current account-aware multi-contract sizing rule into one offline candidate artifact.
Reason: The product goal is a turn-it-on policy that scales risk from account state, confidence, and drawdown without manual model selection.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_account_aware_sizer_v1_research_candidate_not_live. Config v4/audit/autoresearch/v4_aplus_hypothesis_137_protocol101_account_aware_sizer_candidate/account_aware_sizer_v1.json. Report v4/audit/autoresearch/v4_aplus_hypothesis_137_protocol101_account_aware_sizer_candidate/report.md.
Next Gate: Use this config for offline visual/account-state reports. It is still not paper/live multi-contract approval.
Owner: Codex
```

## 2026-05-14 Protocol 138 Protocol101 Base Contract Protection

```text
Date: 2026-05-14
Decision / Experiment: Tested whether premium exposure caps should protect scaling without blocking the base one-contract trade.
Reason: The prior account-aware sizer underperformed the Q4 2024 external block partly because a scaling cap could skip ordinary one-contract trades.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision reject_base_contract_protection_still_external_negative. Report v4/audit/autoresearch/v4_aplus_hypothesis_138_protocol101_base_contract_protection/report.md.
Next Gate: Do not keep this change; inspect the segment summary for why base-contract protection failed.
Owner: Codex
```

## 2026-05-14 Protocol 139 Protocol101 Daily Stop Retune

```text
Date: 2026-05-14
Decision / Experiment: Retuned the always-on account-aware sizer daily stop after base-contract protection.
Reason: The prior fixed stop skipped recovery winners in the external block; the sizer needs account-aware risk control without blocking ordinary one-contract recovery trades too aggressively.
Data Used: Existing Protocol113 replay trades only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_daily_stop_retune_candidate_not_live. Best stop -1500.0 plus 0.500% equity. Report v4/audit/autoresearch/v4_aplus_hypothesis_139_protocol101_daily_stop_retune/report.md.
Next Gate: Update account_aware_sizer_v1 to use the retuned daily stop and rerun consolidated validation.
Owner: Codex
```

## 2026-05-14 Protocol 140 IBKR Paper Autostart Prep

```text
Date: 2026-05-14
Decision / Experiment: Prepared IB Gateway paper-mode morning autostart and Protocol101 API preflight assets.
Reason: User wants the system to start IB Gateway paper mode automatically before the market opens, so the project needs OS-level startup and a broker API readiness check before paper trading.
Data Used: Local IB Gateway app path and redacted JTS config only. No paid data was downloaded, no broker order endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision ready_to_install_ib_gateway_paper_autostart. Report: v4/audit/autoresearch/v4_aplus_hypothesis_140_ibkr_autostart_prep/report.md
Next Gate: Install LaunchAgents when desired, then run paper-order guard and live shadow parity before enabling paper orders.
Owner: Codex
```

## 2026-05-14 Protocol 141 IBKR Paper-Order Guard

```text
Date: 2026-05-14
Decision / Experiment: Added and ran an IBKR paper-order permission guard around future Protocol101 broker-connected paper trading.
Reason: User approved paper trades, but the project needs an explicit paper-only permission layer so no real-money order path can be reached by accident.
Data Used: Local guard configuration only unless account-probe mode was requested. No paid data was downloaded and no broker order endpoint was called.
Cost: $0 incremental paid data.
Result: Decision pass_account_probe_connected_orders_still_disabled. Report: v4/audit/autoresearch/v4_aplus_hypothesis_141_ibkr_paper_order_guard/report.md
Next Gate: Run account-probe with IB Gateway open, then run no-order live shadow parity before enabling paper-order submission.
Owner: Codex
```

## 2026-05-14 Protocol 142 IBKR Paper Executor Smoke

```text
Date: 2026-05-14
Decision / Experiment: Added a guarded IBKR paper-order executor and smoke-tested the non-submitting path.
Reason: User approved eventual paper trades, so the project needs an explicit executor that can submit only validated paper-order intents after live shadow parity passes.
Data Used: Local IBKR paper connection in dry-run mode. No paid data was downloaded and no paper order was submitted unless paper-submit mode is explicitly used.
Cost: $0 incremental paid data.
Result: Decision pass_paper_executor_validates_without_order_submission. Report: v4/audit/autoresearch/v4_aplus_hypothesis_142_ibkr_paper_executor_smoke/report.md
Next Gate: Feed this executor only from a live Protocol101 order-intent stream that has passed schema/freshness/risk checks.
Owner: Codex
```

## 2026-05-14 Protocol 143 IBKR LaunchAgent Activation Check

```text
Date: 2026-05-14
Decision / Experiment: Installed and verified the IB Gateway paper-mode morning LaunchAgents.
Reason: User wants IB Gateway paper mode to start before the market so the bot can run without a 6:30 AM manual login.
Data Used: Local launchd state only. No paid data was downloaded, no paper order was submitted, and no broker order endpoint was called.
Cost: $0 incremental paid data.
Result: Decision pass_ibkr_paper_launchagents_installed. Report: v4/audit/autoresearch/v4_aplus_hypothesis_143_ibkr_launchd_activation_check/report.md
Next Gate: During the next market session, use the preflight/live-shadow logs to confirm live paper API parity before enabling paper order submission.
Owner: Codex
```

## 2026-05-14 Protocol 144 Paper Trade Logging

```text
Date: 2026-05-14
Decision / Experiment: Added and validated the append-only Protocol101 paper trade log contract.
Reason: User requested live/paper trades be logged for later analysis, so every future paper session needs a durable JSONL source of truth plus CSV export.
Data Used: Synthetic local sample events only. No paid data was downloaded, no broker order endpoint was called, and no paper order was submitted.
Cost: $0 incremental paid data.
Result: Decision pass_paper_trade_logging_ready_for_live_paper_analysis. Rows=3; event_counts={'model_decision': 1, 'risk_gate': 1, 'paper_order_dry_run': 1}. Report: v4/audit/autoresearch/v4_aplus_hypothesis_144_paper_trade_logging/report.md
Next Gate: Wire live Protocol101 decisions/orders/fills/exits/account state into this trade journal during paper sessions.
Owner: Codex
```

## 2026-05-14 Protocol 145 Tuesday Cold-Start Rehearsal

```text
Date: 2026-05-14
Decision / Experiment: Rehearsed Tuesday's cold-start path with IB Gateway closed, using the same startup script as the morning LaunchAgent.
Reason: User wanted to know whether the unattended startup path works and expected failure at the missing username/password login stage.
Data Used: Local app/launchd/API-port checks only. No paid data was downloaded, no market-data endpoint was called, and no order endpoint was called.
Cost: $0 incremental paid data.
Result: Decision pass_cold_start_api_ready. Ports before=[]; ports after=[4002]. Report: v4/audit/autoresearch/v4_aplus_hypothesis_145_tuesday_cold_start_rehearsal/report.md
Next Gate: Run Protocol141 account probe, then Protocol119/124 live-data parity, then paper executor dry-run.
Owner: Codex
```

## 2026-05-14 Protocol 146 IBC Credential Readiness

```text
Date: 2026-05-14
Decision / Experiment: Added IBC credential-backed login readiness for IB Gateway paper mode.
Reason: IB Gateway logs out daily, so the morning startup path must enter username/password from local secure storage instead of requiring manual wake-up.
Data Used: Local IBC installation and macOS Keychain presence checks only. No credentials were committed or printed, no paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision pass_ibc_credentials_ready_for_cold_start_rehearsal. IBC installed=True; username_present=True; password_present=True. Report: v4/audit/autoresearch/v4_aplus_hypothesis_146_ibc_credential_readiness/report.md
Next Gate: Run Protocol145 cold-start rehearsal again; IBC should enter credentials and expose the paper API port after any required 2FA approval.
Owner: Codex
```

## 2026-05-15 Protocol 154 Multi-Contract Promotion Decision

```text
Date: 2026-05-15
Decision / Experiment: Aggregated Protocol151/152/153 into a single multi-contract promotion verdict.
Reason: User asked for all verifications necessary to decide whether the account-aware multi-contract challenger is promotable.
Data Used: Existing Protocol101 replay artifacts, existing high-resolution timing rows, and existing live-stack compatibility rows only.
Cost: $0 incremental paid data.
Result: Decision blocked_multi_contract_promotion_missing_timing_evidence. Report: v4/audit/autoresearch/v4_aplus_hypothesis_154_protocol101_multi_contract_promotion_decision/report.md
Next Gate: Do not enable multi-contract paper trading yet. Unblock by proving the same frozen candidate under >=95% critical high-resolution timing coverage or by replaying sufficiently fresh live-shadow/paper logs.
Owner: Codex
```

## 2026-05-15 Protocol 155 One-Contract Live Paper Timing Evidence

```text
Date: 2026-05-15
Decision / Experiment: Added the one-contract live paper timing evidence protocol for Protocol101.
Reason: Multi-contract sizing is economically promising but blocked by timing evidence; live paper should collect timing and fill data while executing only one contract.
Data Used: Existing paper/live trade log rows only. No paid data was downloaded.
Cost: $0 incremental paid data.
Result: Decision blocked_protocol155_no_closed_one_contract_paper_trades_yet. Report: v4/audit/autoresearch/v4_aplus_hypothesis_155_protocol101_live_timing_evidence/2026-05-15/protocol155_readiness_dry_run/report.md
Next Gate: Run the next market session in one-contract paper mode and capture closed entry/exit fill rows.
Owner: Codex
```

## 2026-05-18 Protocol 156 IBKR Autostart Observability

```text
Date: 2026-05-18
Decision / Experiment: Added a single IBKR morning autostart observability report covering launchd state, log tails, API port probes, and known failure signals.
Reason: The user needs to know whether the automatic IBKR login/session path is working and where it fails before Tuesday's paper-trading run.
Data Used: Local launchd state and local logs only. No paid data was downloaded, no broker order endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision blocked_launchagents_not_loaded. Report: v4/audit/autoresearch/protocol101_june22_autostart_prep/report.md
Next Gate: Use the report after the scheduled 6:28/6:29/6:30 Pacific automation to diagnose startup, preflight, session, and market-data blockers.
Owner: Codex
```

## 2026-05-21 Protocol 161 May 2026 Historical Replay

```text
Date: 2026-05-21
Decision / Experiment: Downloaded the approved May 19-20 2026 historical SPXW 0DTE replay slice and ran frozen Protocol101 entry inference across both sessions.
Reason: Live paper/shadow had taken no trades for three days, so the project needed to test whether the frozen model would have seen trades on the same sessions with historical OPRA + official SPX/VIX context.
Data Used: Databento OPRA.PILLAR definition, cbbo-1m, ohlcv-1m, and statistics for 2026-05-19 through 2026-05-20; ThetaData SPX/VIX 1m index bars for the same sessions; existing frozen Protocol051/Protocol101 artifacts. No model was retrained and no broker endpoint was called.
Cost: Approved hard Databento cap $5.00; actual estimated Databento spend $1.1209. ThetaData used the existing subscription.
Result: Decision historical_replay_produced_protocol101_entries. Historical replay produced 0 entries on 2026-05-19 and 12 entry signals on 2026-05-20, all in post-open morning. Live logs did not capture candidate sets until about 11:01 ET on 2026-05-20, after the historical entry window.
Next Gate: Treat the no-trade live conclusion as invalid for May 20. Fix/verify early-session live coverage and live-vs-historical feature parity before changing the model or lowering the edge gate.
Owner: Codex
```

## 2026-05-21 Protocol 162 May 2026 Serial Lifecycle Replay

```text
Date: 2026-05-21
Decision / Experiment: Converted Protocol161 May 20 entry signals into a one-position serial replay with frozen Protocol081 lifecycle exits.
Reason: Protocol161 counted independent entry opportunities, but the live paper bot can hold only one contract at a time; this audit checks whether the apparent May 20 edge survives real slot occupancy.
Data Used: Already-downloaded May 19-20 2026 normalized official-context SPXW rows and existing frozen Protocol081 lifecycle artifact. No paid data was downloaded, no model was retrained, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision reject_independent_entry_pnl_as_evidence: serial replay is not profitable. Independent entries showed 12 trades, +$6,810 PnL, 92% win rate; serial one-position replay showed 3 trades, -$1,090 PnL, 67% win rate. The first 10:12 ET put lost -$1,930 and occupied the slot while later call winners appeared.
Next Gate: Future historical/live equity artifacts must be serial/account-aware. The next model hypothesis should focus on same-window opportunity cost and side arbitration so the bot can reject a weak put when better call opportunities are emerging, rather than treating every entry independently.
Owner: Codex
```

## 2026-05-21 Protocol 163 Recent Data Catch-Up Request

```text
Date: 2026-05-21
Decision / Experiment: Prepared the Protocol163 recent-data catch-up request manifest before retraining the serial account policy.
Reason: The official-context training stream ends at 2026-03-31, while recent April-May 2026 market action should be available before retraining the account-level opportunity-cost model.
Data Used: Local file coverage and existing Databento audit logs only. Databento metadata was attempted but stalled, so the final manifest uses local calibrated cost estimates. No Databento timeseries.get_range call, no ThetaData historical download, no broker endpoint, and no model retraining occurred.
Cost: $0 incremental paid data. Estimated Databento spend for 33 missing sessions from 2026-04-01 through 2026-05-18 is $17.6462; conservative estimate $21.1491; proposed hard cap $30.00. Existing local May 19-20 files are reused.
Result: Decision approval_required_before_paid_download. Request: v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_data_catchup_request/request.md. Manifest: v4/promotion/PROTOCOL_163_RECENT_DATA_CATCHUP_REQUEST.json.
Next Gate: Ask for exact user approval text from the manifest before any Databento/ThetaData paid download endpoint is called.
Owner: Codex
```

## 2026-05-21 Protocol 163 Recent Data Catch-Up Build And Integrity

```text
Date: 2026-05-21
Decision / Experiment: Downloaded the approved recent-data catch-up block, rebuilt official-context neural rows, and added a Protocol163-specific integrity gate.
Reason: The next training path must use clean, official-context rows and must not train on candidates with missing contract-quality Greeks.
Data Used: Databento OPRA.PILLAR definition, cbbo-1m, ohlcv-1m, and statistics for missing sessions from 2026-04-01 through 2026-05-18; reused local May 19-20 files; ThetaData SPX/VIX 1m official index bars for 2026-04-01 through 2026-05-20. No broker endpoint was called and no model was trained.
Cost: Approved Databento hard cap $30.00; estimated cumulative Databento spend $18.8406. ThetaData used the existing subscription.
Result: Decision ready_for_protocol163_training. Built 35 sessions, 12,599 neural decision rows, and 366,276 tradable candidates. Added a conservative Greek-validity candidate filter so quote-valid but Greek-invalid contracts are excluded before training. Report: v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_data_integrity/report.md
Next Gate: Use the cleaned block to train the serial one-account policy, while keeping May 19-20 marked as prior live-parity diagnostic data.
Owner: Codex
```

## 2026-05-21 Protocol 163 Recent Frozen Protocol101 Serial Baseline

```text
Date: 2026-05-21
Decision / Experiment: Ran the frozen Protocol101 entry stack and frozen Protocol081 lifecycle exits across the new April-May 2026 block before retraining.
Reason: Establish an account-real baseline on the new data so retraining cannot hide whether the old model was only benefiting from overlapping independent entries.
Data Used: Cleaned Protocol163 official-context processed rows, normalized SPXW rows, and existing frozen Protocol051/Protocol101/Protocol081 artifacts. No paid data was downloaded by the runner, no model was retrained, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Frozen Protocol101 emitted 627 independent entry signals; strict one-position serial replay took 90 trades, total PnL $7,330, PF 1.274, win rate 48.9%, calls +$4,160 and puts +$3,170. Independent replay showed $53,030 and overstated tradable opportunity due to 537 overlapping skipped signals. Report: v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/report.md
Next Gate: Protocol163 training should optimize account-level slot occupancy directly: enter only when the current contract is worth occupying the single position slot, and penalize weak early trades that block better later opportunities.
Owner: Codex
```

## 2026-05-21 Protocol 163 Greek Repair Rebuild

```text
Date: 2026-05-21
Decision / Experiment: Rebuilt the Protocol163 recent official-context neural block after replacing blunt missing/invalid Greek skipping with a repair-first path.
Reason: The user correctly rejected silently skipping repairable candidates. Contracts should only be excluded when Greeks cannot be calculated from observable bid/mid/ask prices and contract metadata.
Data Used: Already-downloaded Databento OPRA + ThetaData SPX/VIX files for 2026-04-01 through 2026-05-20. No paid data was downloaded, no model was trained, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision ready_for_protocol163_training. Built 35 sessions and 12,599 neural decision rows; tradable candidate count increased from 366,276 to 368,459 after repair. The repair path uses mid first, then executable ask, then bid, and skips only if all observable prices fail. Report: v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_data_integrity/report.md
Next Gate: Re-run the frozen Protocol101 serial baseline and Protocol163 training on the repaired block.
Owner: Codex
```

## 2026-05-21 Protocol 163 Repaired Frozen Protocol101 Baseline

```text
Date: 2026-05-21
Decision / Experiment: Re-ran frozen Protocol101 entry inference and frozen Protocol081 serial lifecycle replay on the repaired-Greek recent block.
Reason: Protocol163 must compare against a baseline built from the same cleaned candidate universe.
Data Used: Repaired Protocol163 processed rows, normalized official-context SPXW rows, and existing frozen Protocol051/Protocol101/Protocol081 artifacts. No paid data was downloaded, no model was retrained, and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Frozen Protocol101 emitted 631 independent entry signals. Strict one-position serial replay took 90 trades, total PnL $7,450, PF 1.278, win rate 50.0%, calls +$4,700 and puts +$2,750. Report: v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/report.md
Next Gate: Protocol163 must beat this repaired strict one-account baseline, not the old overlapping or pre-repair artifacts.
Owner: Codex
```

## 2026-05-21 Protocol 163 Serial One-Account Training

```text
Date: 2026-05-21
Decision / Experiment: Implemented and ran Protocol163, a serial one-account neural entry policy with $10,000 starting cash, one contract max, affordability checks, one open position max, ask-entry/bid-exit labels, and frozen Protocol081 exits.
Reason: The training/evaluation path must match the future paper bot more closely: when the bot takes a trade, that trade occupies the only position slot and blocks other opportunities until exit.
Data Used: Existing Protocol092 historical candidate dataset plus the repaired Protocol163 recent candidate lifecycle paths. No paid data was downloaded by the runner and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Decision keep_for_attribution_only: Protocol163 was profitable but did not clear the one-account improvement gate. Recent 2026 median model PnL was $6,310 across five seeds versus the repaired frozen Protocol101 serial baseline of $7,450; PF 1.241, 88 median trades, stress $0.10 still +$4,550 and stress $0.25 +$1,910. It improved Q3 2025, Q1 2026, and March 2026 medians, but underperformed Q4 2025 and the new recent block. Report: v4/audit/autoresearch/v4_aplus_hypothesis_163_serial_one_account_training/report.md
Next Gate: Do not promote Protocol163 over Protocol101. Use Protocol163 for attribution to identify which recent trades it skipped or filtered, especially seed 5 versus the median seeds, before adding another model knob.
Owner: Codex
```

## 2026-05-21 Protocol 164-166 Full Action-Space Realignment

```text
Date: 2026-05-21
Decision / Experiment: Implemented the next full-action-space training path: Protocol164 dataset builder, Protocol165 serial neural policy runner, and Protocol166 live/training parity contract.
Reason: Protocol163 was still learning from Protocol101-exposed candidates. The next research path must let the neural policy see the full SPXW 0DTE ATM +/- $50 ladder while flat, decide wait versus one specific long call/put, and score under the same one-account serial constraints intended for paper trading.
Data Used: Already-collected official-context processed and normalized rows only. Smoke verification used one existing recent session. No paid data was downloaded and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: Protocol164 smoke built 42 full-ladder candidates with no Protocol101 min-edge, preferred-time-bucket, or selected-candidate gate. Protocol165 smoke trained/evaluated on that smoke dataset and remained research-only, as expected for a one-event smoke. Protocol166 wrote a parity contract declaring SPXW PM-only, $5-aligned ATM +/- $50 candidates, ask-entry/bid-exit accounting, one contract, one open position max, and disallowed Protocol101 pre-entry gates.
Next Gate: Build the full Protocol164 dataset across collected blocks, train Protocol165 on chronological folds, and compare against the frozen strict-serial Protocol101 baseline before considering any paper-runtime replacement.
Owner: Codex
```

## 2026-05-21 Protocol 179-184 Full-Action And Prehistory Search Batch

```text
Date: 2026-05-21
Decision / Experiment: Ran five follow-up hypotheses after Protocol164-166: Protocol179 full-action value policy on the expanded 30-session full-ladder screen, Protocol180 nonnegative value-threshold discipline, Protocol181/182 Protocol101-anchor overlays, Protocol183 two-stage full-action event/ranking policy, and Protocol184 Q4-2024-prehistory smaller/longer serial policy.
Reason: The goal was to keep searching one hypothesis at a time for a model that could beat frozen Protocol101 under strict serial account replay, without buying data or using live broker endpoints.
Data Used: Already-collected official-context historical blocks, Protocol164 full-action screen, existing Protocol101/Protocol163/Protocol175 artifacts, and existing Q4 2024 prehistory data. No paid data was downloaded and no broker endpoint was called.
Cost: $0 incremental paid data.
Result: No candidate cleared the frozen Protocol101 gate. Protocol183 separated trade/no-trade from contract ranking but still underperformed Protocol101 materially: Q3 2025 median $0 vs $59,130, Q4 2025 $4,805 vs $92,460, Q1 2026 $1,625 vs $95,010, March $0 vs $41,790, recent 2026 $7,270 vs $7,450. Protocol184 stayed profitable but also failed: Q3 $58,620 vs Protocol101 $59,130, Q4 $79,210 vs $92,460, Q1 $90,310 vs $95,010, March $40,170 vs $41,790, recent $7,450 vs $7,450.
Key Finding: The full-action path is not yet a fair replacement for Protocol101 because Protocol164 omitted the causal surface `edge` feature that Protocol101 depends on. The next research move should be feature-parity repair, not another threshold or architecture knob: build/score a full-action dataset with causal surface-edge features and history summaries, then rerun the full-action serial policy.
Next Gate: Freeze Protocol101 as operational paper default. Stop model-tweak searching until Protocol164 full-action feature parity is repaired and verified.
Owner: Codex
```

## 2026-05-22 AUDIT_FULL_ACTION_FEATURE_PARITY_V1

```text
Date: 2026-05-22
Historical ID: Protocol210
What is this: Diagnostic / audit.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_189_full_coverage_surface_edge_enrichment/protocol185_full_action_with_surface_edge.parquet.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision repair_missing_causal_history_features. The full-action surface-edge dataset had all surface-edge fields but was missing 14 Protocol101-style causal short-history fields, including previous/rolling candidate count, gamma, theta burden, spread, call/put counts, and call-minus-put edge.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_210_full_action_feature_parity_audit/report.md
Next Experiment: Repair the full-action dataset by adding the missing causal short-history features before rerunning the two-stage full-action policy.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_FULL_ACTION_HISTORY_FEATURE_REPAIR_V1

```text
Date: 2026-05-22
Historical ID: Protocol211
What is this: Dataset repair experiment.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_189_full_coverage_surface_edge_enrichment/protocol185_full_action_with_surface_edge.parquet.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision ready_for_full_action_history_policy_test. Built v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet with 3,516,386 rows. All repaired history feature NaN counts were zero. The history features are causal: each decision row only uses earlier decision events from the same session.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/report.md
Next Experiment: Run the two-stage full-action policy using surface-edge plus repaired causal-history features.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_FULL_ACTION_SURFACE_EDGE_HISTORY_POLICY_SMOKE_V1

```text
Date: 2026-05-22
Historical ID: Protocol212 smoke, using the Protocol183 two-stage policy runner.
What is this: Smoke experiment, not a promotion score.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: One complete q1_2025 session carved from the Protocol211 repaired dataset: 2025-02-21, 360 decision events, 11,535 full-action candidate rows.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: The repaired feature set trained and evaluated successfully through the two-stage full-action policy path with one seed and four epochs. Decision remained research_only as expected because the smoke fold contains no protected scoring blocks.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_212_full_action_surface_edge_history_policy_smoke/report.md
Next Experiment: Run the repaired full-action surface-edge-history policy on the all-session dataset as a controlled long research run, then compare against PAPER_DEFAULT_PROTOCOL101 under strict one-account serial replay.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_FULL_ACTION_SURFACE_EDGE_HISTORY_POLICY_SCREEN_V1

```text
Date: 2026-05-22
Historical ID: Protocol213, using the Protocol183 two-stage policy runner.
What is this: One-seed all-session research screen.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Other Baselines: First-affordable, matched-random, and full-action oracle from the reusable full-action runner.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: One-seed screen passed the reusable runner gate but is not a promotion decision. It beat PAPER_DEFAULT_PROTOCOL101 on every scored block: q3_2025 +$35,345, q4_2025 +$25,550, q1_2026 +$109,120, March 2026 +$18,910, and recent_2026 +$22,540. It also passed serial/account invariants with 0 unaffordable trades, 0 overlap violations, and valid exit ordering.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_213_full_action_surface_edge_history_policy_screen/report.md
Next Experiment: Run the same repaired full-action surface-edge/history policy across additional seeds before any promotion decision. If the multi-seed confirmation survives, do attribution and runtime parity before any paper-default replacement.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_FULL_ACTION_SURFACE_EDGE_HISTORY_ADDITIONAL_SEEDS_V1

```text
Date: 2026-05-22
Historical ID: Protocol214, using the Protocol183 two-stage policy runner.
What is this: Additional seed confirmation run.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Seeds 2-5 independently beat PAPER_DEFAULT_PROTOCOL101 on every scored block under strict serial replay. Median deltas versus Protocol101 across the four seeds were q3_2025 +$48,497.50, q4_2025 +$36,520.00, q1_2026 +$102,002.50, March 2026 +$20,062.50, and recent_2026 +$15,522.50.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_214_full_action_surface_edge_history_policy_additional_seeds/report.md
Next Experiment: Combine seed 1 and seeds 2-5 into one five-seed confirmation artifact.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_FULL_ACTION_SURFACE_EDGE_HISTORY_5SEED_CONFIRMATION_V1

```text
Date: 2026-05-22
Historical ID: Protocol215, combined from Protocol213 seed 1 and Protocol214 seeds 2-5.
What is this: Five-seed research confirmation.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision confirmed_research_challenger_not_paper_default. The candidate beat PAPER_DEFAULT_PROTOCOL101 on every protected block with positive seed fraction 1.00: q3_2025 median $104,840 vs $59,130, q4_2025 $128,755 vs $92,460, q1_2026 $204,130 vs $95,010, March 2026 $60,700 vs $41,790, and recent_2026 $25,150 vs $7,450. $0.10 and $0.25 per-side stress remained positive on every scored block. Serial invariants passed with 0 unaffordable trades, 0 overlap violations, and 0 bad exit ordering.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_215_full_action_surface_edge_history_5seed_confirmation/report.md
Next Experiment: Do attribution and live/training parity checks for CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1 before any paper-default replacement. Specifically inspect whether it reduces churn, whether fills/timing remain plausible, and whether the live runtime can generate the same full-action feature set.
Owner: Codex
```

## 2026-05-22 AUDIT_CHALLENGER_FULL_ACTION_HISTORY_VS_PROTOCOL101_V1

```text
Date: 2026-05-22
Historical ID: Protocol216.
What is this: Diagnostic / attribution audit.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol215 confirmed challenger trades, Protocol101 historical serial trades, Protocol101 recent serial replay, and local normalized official-context quote paths.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision attribution_supports_challenger_next_runtime_parity. Exact trade overlap was tiny, so the challenger is not just replaying Protocol101. It won by taking more full-action entries, especially larger directional-move captures. Median duration was generally 25 minutes for the challenger versus 8-14 minutes for Protocol101 on historical blocks. Directional move capture improved materially: for example q3_2025 >=10 SPX-point directional captures were 274 challenger trades for $449,230 versus 28 Protocol101 trades for $30,700. The challenger still has same-side re-entry/churn, but same-contract hold counterfactuals did not show a broad simple-hold fix for the challenger.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_216_full_action_history_vs_protocol101_attribution/report.md
Next Experiment: Build the no-order runtime parity path for CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1 before any paper-default replacement.
Owner: Codex
```

## 2026-05-22 RUNTIME_FULL_ACTION_HISTORY_NO_ORDER_PARITY_V1

```text
Date: 2026-05-22
Historical ID: Protocol217.
What is this: Runtime / no-order parity harness.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol211 repaired full-action surface-edge/history dataset, fold4 seed1 challenger model artifact, and Protocol215 historical selected trades for parity comparison.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision runtime_parity_ready_no_order_protocol101_default_unchanged. Replayed 750 recent_2026 decision events with no orders. Runtime schema validation passed, feature audit passed with no future-only columns and all repaired history features present, latency budget pass fraction was 1.0, p95 total decision latency was 1.793 ms, and the runtime selected exactly the same 18 trades as the frozen historical artifact inside the replayed event window.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_217_full_action_history_runtime_parity/report.md
Next Experiment: Inspect selected-trade charts and produce a replacement-readiness packet only after confirming the runtime can build the same full-action history features from live data, not just replayed parquet rows.
Owner: Codex
```

## 2026-05-22 DIAGNOSTIC_CHALLENGER_FULL_ACTION_HISTORY_TRADE_CHARTS_V1

```text
Date: 2026-05-22
Historical ID: Protocol218.
What is this: Diagnostic / visual inspection artifact.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_215_full_action_surface_edge_history_5seed_confirmation/model_trades_5seed.csv plus local SPX and normalized option quote files for quote-path backfill.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision visual_inspection_artifacts_ready_paper_default_unchanged. Built serial one-account charts for seed 1 with $10,000 starting equity, one affordable contract at a time, ask-entry/bid-exit replay, and no duplicated March slice. The paper replay contained 1,950 trades, 0 skipped trades, 100% premium/path quote coverage, $446,605 total PnL, max drawdown -$9,550, and $417,605 stressed ending equity at $0.10 per side.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_218_challenger_full_action_history_trade_charts/report.md
Next Experiment: Inspect biggest winners, biggest losers, worst days, and churn chains before any paper-default replacement decision.
Owner: Codex
```

## 2026-05-22 RUNTIME_FULL_ACTION_HISTORY_FEATURE_BUILDER_PARITY_V1

```text
Date: 2026-05-22
Historical ID: Protocol219.
What is this: Runtime / parity harness.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Baseline Compared: Historical Protocol211 parquet feature build.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision live_style_history_feature_builder_parity_passed. A live-style one-event-at-a-time builder reproduced the repaired causal history features over all collected blocks: 123,782 decision events, 3,516,386 candidate rows, 0 mismatch events, max absolute diff 6.82e-13, and every split passed. This confirms the challenger's short-history features are compatible with a streaming/no-order runtime path rather than requiring future parquet context.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_219_live_style_history_feature_builder_parity/report.md
Next Experiment: Wire the stream-safe feature builder into a no-order challenger shadow path while keeping PAPER_DEFAULT_PROTOCOL101 unchanged until that live-safe path passes.
Owner: Codex
```

## 2026-05-22 AUDIT_CHALLENGER_MONEYNESS_AND_PROMOTION_BLOCKERS_V1

```text
Date: 2026-05-22
Historical ID: Protocol220.
What is this: Diagnostic / audit.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol218 seed-1 serial replay trades, Protocol215 five-seed selected trades, and local ThetaData SPX bars for entry moneyness reconstruction.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision moneyness_audit_complete_paper_default_unchanged. The candidate is overwhelmingly ITM/high-premium. In the source-of-truth seed-1 serial replay, trader-ladder moneyness was 1,747 ITM trades for $419,295 PnL, 100 ATM-band trades for $22,125 PnL, and 103 OTM trades for $5,185 PnL. Across all five research seeds with the duplicate March slice removed, trader-ladder moneyness was 9,850 ITM trades for $2,165,160 PnL, 540 ATM-band trades for $88,720 PnL, and 642 OTM trades for $30,450 PnL. The $3k-$4k premium bucket dominated all five seeds: 8,092 trades for $1,845,270 PnL. 872 all-seed rows used offset fallback for moneyness because the local SPX entry bar was missing.
Interpretation: The challenger appears to capture larger SPX directional moves partly by buying higher-delta/intrinsic contracts rather than cheaper OTM convexity. This is not an automatic rejection, but it is an operational promotion issue because live paper must prove the same premium/moneyness selection is affordable, fresh, and executable.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_220_challenger_moneyness_promotion_audit/report.md
Next Experiment: Run a no-order live-shadow challenger using the stream-safe full-action/history feature path, then compare live candidate surface, premium distribution, and selected strikes against this audit before any paper-default replacement.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_RETURN_ON_PREMIUM_FULL_ACTION_POLICY_V1

```text
Date: 2026-05-22
Historical ID: Protocol221.
What is this: Experiment / model change.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Baseline Challenger: CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision research_only_return_on_premium_objective_did_not_clear_frozen_protocol101_gate. The pure return-on-premium objective worked mechanically but overcorrected. Median premiums collapsed from the existing challenger's roughly $2k-$3.2k range to $100-$175 on scored splits, and the trade profile became mostly OTM: 1,774 OTM trades, 110 ATM-band trades, and 33 ITM trades. Capital efficiency and profit factor improved, but dollar PnL fell below PAPER_DEFAULT_PROTOCOL101 on Q3 2025, Q4 2025, Q1 2026, and March 2026. It only beat the paper default on recent_2026: $9,555 vs $7,450.
Interpretation: Premium-normalized training is useful, but pure return-on-premium is too aggressive. The next model hypothesis should be a blended dollar-plus-premium utility that preserves the existing challenger's directional-move capture while penalizing overpaying for high-premium ITM contracts.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_221_return_on_premium_full_action_policy/report.md
Next Experiment: Test a single pre-registered blended utility, not a grid: keep dollar-PnL opportunity cost as the main target but add premium-normalized regret/efficiency as an auxiliary objective.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_CONFIDENCE_SCALED_PREMIUM_SIZING_V1

```text
Date: 2026-05-22
Historical ID: Protocol222.
What is this: Experiment / offline sizing simulator.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1_WITH_CONFIDENCE_SIZING.
Entry/Exit Source: CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_221_return_on_premium_full_action_policy/model_trades.csv plus entry ask-size joins from v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision confidence_sizing_improves_return_on_premium_candidate_offline. The sizing overlay materially increased dollar PnL versus the one-contract return-on-premium candidate on every scored split: q3_2025 $313,715 vs $20,715, q4_2025 $479,270 vs $43,640, q1_2026 $783,185 vs $48,415, March 2026 $286,740 vs $19,010, and recent_2026 $140,650 vs $9,555. However, this is not promotion-ready: the 20-contract research cap bound on 96.5% of trades, median size was 20 contracts, max drawdown increased sharply, and $0.25 per-side stress turned q3_2025 and q4_2025 negative.
Interpretation: The user's multi-contract hypothesis is worth pursuing, but not as a blunt leverage overlay. A max-one diagnostic showed the same return-on-premium trade set stayed positive under $0.25 per-side stress on every split, so the failure is aggressive risk allocation under stress rather than the cheap-OTM candidate set itself. The next model needs liquidity/slippage-aware confidence sizing, drawdown-aware risk throttling, or a blended dollar-plus-premium objective that chooses better contracts before increasing quantity.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_222_confidence_scaled_premium_sizing/report.md
Next Experiment: Do not promote multi-contract sizing. Test a single blended dollar-plus-premium utility and/or a liquidity-aware sizing policy that reduces size when edge is spread-sensitive, instead of saturating at max contracts.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_ACCOUNT_AWARE_CONFIDENCE_SIZING_V2

```text
Date: 2026-05-22
Historical ID: Protocol223.
What is this: Experiment / offline account-aware sizing simulator.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1_WITH_ACCOUNT_AWARE_SIZING.
Entry/Exit Source: CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_221_return_on_premium_full_action_policy/model_trades.csv plus entry ask-size joins already present in the Protocol223 input stream.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision account_aware_sizing_candidate_clears_initial_offline_gate. At $10,000 starting cash with no added slippage, the account-aware sizing rule beat PAPER_DEFAULT_PROTOCOL101 on every scored split: q3_2025 $70,760 vs $59,130, q4_2025 $141,955 vs $92,460, q1_2026 $403,720 vs $95,010, March 2026 $109,510 vs $41,790, and recent_2026 $65,515 vs $7,450. The rule stayed positive under $0.25 per-side stress on every scored split, with max drawdown percentage 20.6% at $10,000. It failed under $0.50 per-side stress, which is a useful blocker against overconfidence.
Risk Controls Tested: hard premium cap 12% of equity, max risk fraction 10%, drawdown throttle, 25% of displayed ask-size liquidity cap, absolute 100-contract cap, and 50% account halt guard. Across account scales from $10,000 to $1,000,000, the rule scaled quantity without full-porting; p90 premium exposure was 2.3% of equity and max realized premium exposure was 11.7%.
Interpretation: This is the first multi-contract result that looks directionally plausible rather than absurd compounding. It is still research-only because sizing has not been trained as a model action, $0.50 stress breaks, and scale-in/scale-out behavior is not yet learned.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_223_account_aware_confidence_sizing/report.md
Next Experiment: Build the scale-in/scale-out path audit before adding actions. Specifically measure whether adding tends to press winners or average down into losers, and whether exits leave continuation value that a lifecycle model could learn.
Owner: Codex
```

## 2026-05-22 AUDIT_2026_05_22_SCALE_IN_OUT_PATH_OPPORTUNITY_V1

```text
Date: 2026-05-22
Historical ID: Protocol224.
What is this: Diagnostic / scale-in scale-out path opportunity audit.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1_WITH_ACCOUNT_AWARE_SIZING.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_223_account_aware_confidence_sizing/account_aware_sized_trades.csv and local normalized official-context option quote paths.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision scale_in_requires_caution_many_best_adds_are_average_down. The audit found large lifecycle opportunity: at $10,000/no-stress account-aware sizing, best full-path PnL was far above baseline selected exits on every scored split, and material post-exit continuation appeared in 35%-49% of trades depending on split. However, the best hindsight add opportunity often occurred while the existing position was red: 68.8% of positive March add opportunities, 65.6% of Q1, 56.2% of Q3, and 67.3% of Q4 were average-down cases. Recent_2026 was the exception: only 46.8% average-down and 52.3% adding-to-winner.
Greek Repair: Entry and best-add Greeks were repaired from quote path fields with Black-Scholes where normalized path Greeks were missing. Entry delta coverage was 99.95%; best-add delta coverage was 99.69%.
Interpretation: Scale-out/hold improvement is clearly worth pursuing because current exits leave continuation value. Scale-in must be treated carefully: naive adding would learn to average down into losers too often. The next model should first learn position management and scale-out/hold/exit; add actions should be gated by learned recovery quality, not a blind lower-average-cost objective.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_224_scale_in_out_path_opportunity/report.md
Next Experiment: Build a supervised position-management dataset with actions hold/reduce/exit first, and add a separate candidate label for add-one-contract only when the path evidence shows it is not merely averaging down.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_ACCOUNT_AWARE_LIFECYCLE_EXIT_POLICY_V1

```text
Date: 2026-05-22
Historical ID: Protocol225.
What is this: Experiment / causal lifecycle exit model.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_ACCOUNT_AWARE_LIFECYCLE_EXIT_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol223 account-aware sized trade stream and local normalized quote paths.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision reject_account_aware_lifecycle_exit_no_improvement. The model cut trades too early and destroyed move capture: March $34,500 vs Protocol223 baseline $109,510, Q1 $42,235 vs $403,720, and recent $3,955 vs $65,515.
Interpretation: A naive causal exit model learns risk reduction, not profitable lifecycle management. Do not promote.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_225_account_aware_lifecycle_exit_policy/report.md
Next Experiment: Test lifecycle changes that do not allow the model to clip winners before the frozen baseline exit.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_BASELINE_ANCHORED_CONTINUATION_V1

```text
Date: 2026-05-22
Historical ID: Protocol226.
What is this: Experiment / continuation-only lifecycle model.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_BASELINE_ANCHORED_CONTINUATION_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol223 trade stream and local normalized quote paths.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision reject_baseline_anchored_continuation_no_improvement. After fixing the exact-baseline accounting bug, the result matched baseline PnL on March, Q1, and recent; true extensions were rare and not useful.
Interpretation: Current causal continuation signal does not extract the large hindsight post-exit opportunity.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_226_baseline_anchored_continuation/report.md
Next Experiment: Audit scale-out and early-exit behavior rather than adding another continuation knob.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_SCALE_OUT_RUNNER_SCREEN_V1

```text
Date: 2026-05-22
Historical ID: Protocol227.
What is this: Experiment / scale-out runner screen.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_SCALE_OUT_RUNNER_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol223 account-aware sized trade stream and local normalized quote paths.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision reject_scale_out_runner_no_improvement. Validation selected no runner; forced-flat runners reduced PnL across validation paths.
Interpretation: Simple scale-out by holding a runner to close is not a good proxy for learned lifecycle intelligence.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_227_scale_out_runner_screen/report.md
Next Experiment: Test learned early-exit/hold targets instead of hardcoded runner fractions.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_BASELINE_RELATIVE_EARLY_EXIT_V1

```text
Date: 2026-05-22
Historical ID: Protocol228.
What is this: Experiment / early-exit model.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_BASELINE_RELATIVE_EARLY_EXIT_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol223 trade stream and local normalized quote paths.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision reject_baseline_relative_early_exit_no_improvement. Median results matched the frozen baseline; the model did not find a robust generalizable early-exit improvement.
Interpretation: Path-local baseline-relative labels were too weak; the lifecycle target needs a clearer action-state formulation.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_228_baseline_relative_early_exit/report.md
Next Experiment: Build a position-action oracle to understand the reachable hold/reduce/exit label surface.
Owner: Codex
```

## 2026-05-22 AUDIT_2026_05_22_POSITION_ACTION_DP_ORACLE_V1

```text
Date: 2026-05-22
Historical ID: Protocol229.
What is this: Diagnostic / hindsight position-action oracle.
Does it change the paper-trading default: No.
Candidate Being Tested: Protocol223 account-aware trade stream.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol223 trades and local normalized quote paths.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision position_action_oracle_supports_exit_training_only. The oracle shows huge theoretical lifecycle upside: Q1 baseline $403,720 vs oracle $3,099,895, Q4 baseline $141,955 vs oracle $864,960, and recent baseline $65,515 vs oracle $268,330. But reduce/scale-out labels were rare; the oracle is mostly hold then exit-all.
Interpretation: Do not force scale-out. The immediate learnable target is better hold/exit timing.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_229_position_action_dp_oracle/report.md
Next Experiment: Train a sparse oracle-exit classifier and require executable-dollar improvement, not just cleaner risk.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_ORACLE_EXIT_ACTION_CLASSIFIER_V1

```text
Date: 2026-05-22
Historical ID: Protocol230.
What is this: Experiment / sparse oracle exit classifier.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_ORACLE_EXIT_ACTION_CLASSIFIER_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol229 oracle labels and local normalized quote paths.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision reject_oracle_exit_classifier_no_improvement. The classifier improved PF and drawdown but gave up median PnL: March $108,255 vs $109,510, Q1 $388,630 vs $403,720, and recent $63,545 vs $65,515.
Interpretation: The risk-adjusted behavior is interesting, but the project objective still requires beating executable PnL before promotion.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_230_oracle_exit_action_classifier/report.md
Next Experiment: Return to contract-quality and premium-efficiency objectives before more lifecycle knobs.
Owner: Codex
```

## 2026-05-22 AUDIT_2026_05_22_ACCOUNT_AWARE_SIZING_RISK_SCREENS_V1

```text
Date: 2026-05-22
Historical ID: Protocol231.
What is this: Diagnostic / sizing risk screen.
Does it change the paper-trading default: No.
Candidate Being Tested: Protocol223 account-aware sizing variants.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol221 trade stream.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Conservative and balanced variants did not improve the original Protocol223 overlay. Conservative risk reduced PnL and failed Q4 under $0.25 stress; balanced risk lowered PnL without solving the key stress fragility.
Interpretation: The better path is not simply lowering risk fractions. Improve contract selection/objective first.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_231_conservative_account_aware_sizing_screen/report.md and v4/audit/autoresearch/v4_aplus_hypothesis_231_balanced_account_aware_sizing_screen/report.md
Next Experiment: Train a blended dollar-plus-premium objective.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_BLENDED_DOLLAR_PREMIUM_POLICY_V1

```text
Date: 2026-05-22
Historical ID: Protocol232.
What is this: Experiment / model objective change.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_BLENDED_DOLLAR_PREMIUM_FULL_ACTION_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Baseline Challenger: CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1.
Data Used: v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision research_candidate_survives_frozen_protocol101_gate_with_blended_dollar_premium_training. The 65/35 dollar/premium utility beat Protocol101 on every required split but still leaned heavily ITM with median premium about $3,030.
Interpretation: Blending dollar and premium utility works, but a more premium-leaning version may better match capital-efficient option trading without losing the edge.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_232_blended_dollar_premium_policy/report.md
Next Experiment: Screen a premium-leaning blend.
Owner: Codex
```

## 2026-05-22 AUDIT_2026_05_22_BLENDED_POLICY_ACCOUNT_AWARE_SIZING_SCREEN_V1

```text
Date: 2026-05-22
Historical ID: Protocol233.
What is this: Diagnostic / sizing overlay screen.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_BLENDED_DOLLAR_PREMIUM_FULL_ACTION_V1 with account-aware sizing.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol232 model trades.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision research_only_account_aware_sizing_improves_capital_efficiency_but_not_protocol101. The overlay helped Q1 and recent but failed to beat Protocol101 on March, Q3, and Q4 at $10,000/no-stress and became fragile under stress.
Interpretation: The sizing overlay is not the promoted path for this candidate.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_233_blended_policy_account_aware_sizing_screen/report.md
Next Experiment: Test a premium-leaning one-contract objective before revisiting sizing.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_PREMIUM_LEANING_BLEND_SCREEN_V1

```text
Date: 2026-05-22
Historical ID: Protocol234.
What is this: Experiment / model objective change.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol211 repaired full-action history dataset.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision research_candidate_survives_frozen_protocol101_gate_with_blended_dollar_premium_training. The 45/55 dollar/premium utility beat Protocol101 on every required split in the one-seed screen while lowering median premium to about $1,660 and increasing OTM participation.
Interpretation: This is the best capital-efficiency tradeoff found so far.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_234_premium_leaning_blend_screen/report.md
Next Experiment: Screen sizing and then run seed stability if sizing remains fragile.
Owner: Codex
```

## 2026-05-22 AUDIT_2026_05_22_PREMIUM_LEANING_ACCOUNT_AWARE_SIZING_SCREEN_V1

```text
Date: 2026-05-22
Historical ID: Protocol235.
What is this: Diagnostic / sizing overlay screen.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1 with account-aware sizing.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol234 model trades.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision research_only_account_aware_sizing_fails_slippage_stress. The overlay amplified Q4 and March stress losses and was not better than Protocol101 across the strict $10,000 gate.
Interpretation: Keep multi-contract sizing research-only. Promote only one-contract evidence for this challenger.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_235_premium_leaning_account_aware_sizing_screen/report.md
Next Experiment: Test whether an even stronger premium blend improves contract efficiency or just buys too much cheap convexity.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_STRONG_PREMIUM_BLEND_SCREEN_V1

```text
Date: 2026-05-22
Historical ID: Protocol236.
What is this: Experiment / model objective change.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_STRONG_PREMIUM_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol211 repaired full-action history dataset.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision research_candidate_survives_frozen_protocol101_gate_with_blended_dollar_premium_training. The 25/75 dollar/premium utility became much more OTM and dropped median premium to about $380, but total PnL fell below the 45/55 blend on most splits.
Interpretation: The capital-efficiency signal is real, but pushing too hard toward cheap contracts gives up too much dollar edge.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_236_strong_premium_blend_screen/report.md
Next Experiment: Do not promote this over the 45/55 blend unless sizing proves uniquely better.
Owner: Codex
```

## 2026-05-22 AUDIT_2026_05_22_STRONG_PREMIUM_ACCOUNT_AWARE_SIZING_SCREEN_V1

```text
Date: 2026-05-22
Historical ID: Protocol237.
What is this: Diagnostic / sizing overlay screen.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_STRONG_PREMIUM_BLENDED_UTILITY_V1 with account-aware sizing.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol236 model trades.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision research_only_account_aware_sizing_fails_slippage_stress. The overlay produced strong no-stress March/Q1/Q3 results but failed Q4 and became sharply stress-fragile, including negative Q3/Q4 under $0.25 per-side stress.
Interpretation: Cheaper contracts do not automatically make multi-contract sizing safe.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_237_strong_premium_account_aware_sizing_screen/report.md
Next Experiment: Run seed stability on the 45/55 premium-leaning one-contract challenger.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_PREMIUM_LEANING_BLEND_SEED_STABILITY_V1

```text
Date: 2026-05-22
Historical ID: Protocol238.
What is this: Experiment / seed stability run.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol211 repaired full-action history dataset.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision research_candidate_survives_frozen_protocol101_gate_with_blended_dollar_premium_training. Seeds 1-3 survived: median q3_2025 $94,430 vs Protocol101 $59,130, q4_2025 $143,855 vs $92,460, q1_2026 $181,640 vs $95,010, March $64,665 vs $41,790, and recent $54,805 vs $7,450.
Interpretation: The challenger is not a one-seed accident.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_238_premium_leaning_blend_seed_stability/report.md
Next Experiment: Run seeds 4-5 as a holdout stability check.
Owner: Codex
```

## 2026-05-22 EXP_2026_05_22_PREMIUM_LEANING_BLEND_SEED_HOLDOUT_4_5_V1

```text
Date: 2026-05-22
Historical ID: Protocol239.
What is this: Experiment / seed holdout run.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol211 repaired full-action history dataset.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision research_candidate_survives_frozen_protocol101_gate_with_blended_dollar_premium_training. Seeds 4-5 also survived, beating Protocol101 on every required split in median.
Interpretation: Combine with Protocol238 into a five-seed decision packet.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_239_premium_leaning_blend_seed_holdout_4_5/report.md
Next Experiment: Freeze the five-seed challenger as research-only and build runtime parity before any paper-default replacement.
Owner: Codex
```

## 2026-05-22 DECISION_2026_05_22_PREMIUM_LEANING_BLEND_FIVE_SEED_V1

```text
Date: 2026-05-22
Historical ID: Protocol240.
What is this: Freeze / promotion decision packet.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol238 and Protocol239 summaries/trades, both from the Protocol211 repaired full-action history dataset.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision freeze_research_challenger_premium_leaning_blend_five_seed_survives. The candidate survived five seeds and beat Protocol101 under strict one-account serial replay: q3_2025 median $87,055 vs $59,130, q4_2025 $143,855 vs $92,460, q1_2026 $165,125 vs $95,010, March $64,665 vs $41,790, and recent $62,575 vs $7,450. Positive seed fraction was 1.00 on every split; beat-Protocol101 seed fraction was 0.80 on q3_2025 and 1.00 on all other required splits. $0.10 and $0.25 per-side stress stayed positive on every required split.
Trade Profile: Combined five-seed replay had 12,633 trades, $2,616,745 total research PnL, median entry premium $1,790, PnL per entry premium 0.1136, 5,929 calls for $1,048,795, and 6,704 puts for $1,567,950.
Interpretation: This is now the strongest research challenger, but it is not the paper default. The next work is no-order runtime parity and trade-level attribution against Protocol101 before any replacement decision.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_240_premium_leaning_blend_five_seed_decision/report.md
Next Experiment: Build no-order runtime parity for CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1 and compare live-style features/candidate selection against historical replay.
Owner: Codex
```

## 2026-05-22 RUNTIME_PREMIUM_LEANING_BLEND_NO_ORDER_PARITY_V1

```text
Date: 2026-05-22
Historical ID: Protocol241.
What is this: Runtime / no-order parity harness.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol211 repaired full-action history dataset, Protocol238 fold4 seed1 model artifact, and Protocol240 five-seed trade file for historical match.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision runtime_parity_ready_no_order_protocol101_default_unchanged. Replayed 750 recent_2026 decision events with no orders. Runtime validation passed, feature audit passed, budget pass fraction was 1.0, p95 total decision latency was 2.907 ms, and runtime selected exactly the same 21 trades as the frozen historical artifact inside the replayed event window.
Interpretation: The frozen research challenger can be evaluated through a live-style no-order runtime path without changing the paper default. This is historical replay parity, not live-market parity.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_241_premium_blend_runtime_parity/report.md
Next Experiment: Run attribution versus Protocol101 and prepare a replacement-readiness checklist. Do not replace PAPER_DEFAULT_PROTOCOL101 until live/no-order feature parity is demonstrated.
Owner: Codex
```

## 2026-05-22 AUDIT_PREMIUM_LEANING_BLEND_VS_PROTOCOL101_V1

```text
Date: 2026-05-22
Historical ID: Protocol242.
What is this: Diagnostic / attribution audit.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol240 five-seed challenger trades, Protocol101 serial policy trades, recent Protocol101 serial lifecycle replay, and local normalized official-context quote paths.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision attribution_supports_challenger_paper_default_unchanged. The challenger outperformed Protocol101 primarily through more full-action trades and better directional-move capture, not exact trade overlap. Exact overlap was tiny: q3_2025 had only 5 common exact trades, q4_2025 28, March 41, Q1 67, and recent 10. Directional move capture improved materially: q4_2025 >=10 SPX-point directional captures were 523 challenger trades for $910,415 versus 74 Protocol101 trades for $114,240; q3_2025 was 184 challenger trades for $308,240 versus 28 Protocol101 trades for $30,700. The challenger traded longer, with median duration 25 minutes across scored splits, and used both calls and puts, with puts contributing more total PnL in the combined attribution.
Caveat: The challenger also has more same-side re-entry/churn chains because it trades much more often. This is not an immediate rejection because those chains are profitable, but it remains a lifecycle behavior to monitor before replacement.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_242_premium_blend_vs_protocol101_attribution/report.md
Next Experiment: Draft a replacement-readiness checklist and run live/no-order feature parity for the premium-leaning challenger before any paper-default replacement.
Owner: Codex
```

## 2026-05-23 RUNTIME_PREMIUM_BLEND_OFFHOURS_STACK_BRIDGE_V1

```text
Date: 2026-05-23
Historical ID: Protocol243.
What is this: Runtime / off-hours stack bridge and promotion-workaround audit.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol211 repaired full-action history dataset, Protocol238 fold4 seed1 challenger artifact manifest, and recorded Protocol101/081 live-shadow JSONL logs from May 19-20.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision conditional_bridge_ready_but_recorded_live_logs_too_narrow_for_promotion. Added a live-style full-action challenger adapter and verified all 64 model features have causal live-style sources. The adapter rebuilt 3,789 historical candidate rows across 120 May 19-20 decision events with max feature difference 4.55e-13 versus the training parquet. Historical full-action May 19-20 events had median 32 candidates/event, while the recorded Protocol101 live-shadow logs had median 10 contracts/event, so those recorded logs are too narrow to prove full challenger live-surface breadth off-hours.
Interpretation: The model-construction gap is mostly solved off-hours: the challenger can build the same feature surface it was trained on from point-in-time quote/context objects. The remaining replacement blocker is live quote breadth/freshness for the broader ATM +/- $50 full-action ladder, not the neural model artifact itself.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_243_premium_blend_offhours_stack_bridge/report.md
Next Experiment: Wire the challenger adapter into a no-order/paper-dry-run variant of the persistent trader, then require an automatic premarket live-surface breadth/freshness gate before any paper-default switch.
Owner: Codex
```

## 2026-05-23 RUNTIME_PREMIUM_BLEND_PAPER_RUNTIME_SHELL_V1

```text
Date: 2026-05-23
Historical ID: Protocol244.
What is this: Runtime / paper-style no-broker shell.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol211 repaired full-action history dataset and Protocol238 fold4 seed1 challenger artifact manifest.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Decision paper_runtime_shell_ready_no_broker_protocol101_default_unchanged. The shell replayed May 20, 2026 through the live-style full-action adapter and emitted monitorable paper-log events using the shared paper-trading JSONL/CSV contract. It replayed 166 decision events, emitted 5 enter intents, all 5 passed the paper order-intent risk validation, and trade-log validation passed across 831 rows with zero broker endpoint rows.
Interpretation: The challenger can now produce the same style of operational logs that the paper monitor/analyzer expects, without touching IBKR. This moves the blocker from "model construction/log format" to "broker-connected live candidate breadth and freshness."
Report: v4/audit/autoresearch/v4_aplus_hypothesis_244_premium_blend_paper_runtime_shell/report.md
Next Experiment: Use this same shell shape in a broker-connected challenger no-order runner. Only switch paper default after live surface breadth/freshness and log validation pass automatically.
Owner: Codex
```

## 2026-05-23 RUNTIME_PREMIUM_BLEND_LIVE_SURFACE_AUTOTEST_V1

```text
Date: 2026-05-23
Historical ID: Protocol245.
What is this: Runtime / Tuesday broker-connected no-order live surface breadth and challenger scoring check.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Configuration/artifact validation only today; Tuesday run will use IBKR live market data. No paid historical data is downloaded.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Added a no-order live-surface autotest runner plus LaunchAgent wiring for Tuesday May 26, 2026 at 06:33 PT. The config-only smoke passed and confirmed the challenger artifact loads, the surface-edge artifact loads, and the expected SPXW ATM +/- $50 0DTE ladder is 42 raw contracts before NBBO/live-safe filters.
Interpretation: This is the off-hours workaround for the recorded-log breadth blocker. Tuesday's automated check must prove IBKR is feeding the broader live candidate surface before any paper-default replacement can be considered.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_245_premium_blend_live_surface_autotest/2026-05-23/protocol245_config_check_2026-05-23/report.md
Next Experiment: Let the scheduled no-order live surface check run on Tuesday during market hours, then compare the challenger decisions with PAPER_DEFAULT_PROTOCOL101 for the same live session.
Owner: Codex
```

## 2026-05-23 AUDIT_PREMIUM_BLEND_METRIC_RECONCILIATION_V1

```text
Date: 2026-05-23
Historical ID: Protocol247.
What is this: Diagnostic / metric reconciliation.
Does it change the paper-trading default: No.
Candidate Being Tested: CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: Protocol246 single-seed challenger chart trades, Protocol113 Protocol101 chart trades, Protocol163 recent Protocol101 serial replay, and Protocol242 attribution trades.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Confirmed the previous comparison mixed three scopes: five-seed promotion medians, five-seed directional-move attribution, and a one-seed paper-account equity chart. The $910,415 Q4 figure was directional-capture attribution across all five seeds, not Q4 net account PnL. In the single-seed visual chart, the challenger made $465,340 total and $158,830 in Q4; Protocol101 has materially higher win rate and average PnL per trade.
Interpretation: The challenger remains a stronger total-PnL historical research candidate, but the lower win rate, higher trade count, and confusing attribution scope mean it should not be promoted from the old narrative alone. Future comparison language must state metric scope explicitly.
Report: v4/audit/autoresearch/v4_aplus_hypothesis_247_premium_blend_metric_reconciliation/report.md
Next Experiment: Use the corrected apples-to-apples framing when reviewing challenger trade charts and wait for live/no-order surface parity before any paper-default replacement.
Owner: Codex
```

## 2026-05-23 GUIDE_MODEL_IMPROVEMENT_DISCIPLINE_V1

```text
Date: 2026-05-23
Historical ID: None; governing documentation.
What is this: Guideline / model-improvement rulebook.
Does it change the paper-trading default: No.
Candidate Being Tested: None.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: No market data; documentation only.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Added v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md and linked it from README, OPERATING_MEMORY, and NAMING_GUIDE. The guide defines the strict game definition, pre-registration workflow, metric-scope rules, evaluation views, replacement bar, chart rules, failure handling, and paid-data guardrails for future model-improvement work.
Interpretation: The premium-blend metric mismatch showed that profitable research artifacts are not enough. Future challengers must be compared against PAPER_DEFAULT_PROTOCOL101 under explicit metric scope before they can be called better, and lower win rate / higher churn must be treated as a tradeoff requiring attribution rather than ignored.
Report: v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md
Next Experiment: Use this guideline as the required preflight for any new challenger experiment or replacement discussion.
Owner: Codex
```

## 2026-05-23 GUIDE_HYPOTHESIS_TO_PROMOTION_PROCESS_V1

```text
Date: 2026-05-23
Historical ID: None; governing documentation.
What is this: Guideline / stage-gate research and promotion process.
Does it change the paper-trading default: No.
Candidate Being Tested: None.
Paper Default Baseline: PAPER_DEFAULT_PROTOCOL101.
Data Used: No market data; documentation only.
Paid Data Downloaded: No.
Broker Endpoint Called: No.
Result: Added v4/docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md and linked it from README, MODEL_IMPROVEMENT_GUIDELINES, OPERATING_MEMORY, NAMING_GUIDE, and PROMOTION_SEQUENCE. The document defines the required path from observation to hypothesis, experiment, validation, research freeze, runtime parity, paper promotion, paper operation, and real-money review.
Interpretation: This is the process guardrail that should prevent a research-useful challenger from skipping directly into promotion. A model can now only move forward by satisfying explicit stage transitions with clear artifacts, metric scope, and blocker decisions.
Report: v4/docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md
Next Experiment: Use this process for the next model-improvement hypothesis; do not discuss replacing PAPER_DEFAULT_PROTOCOL101 unless the challenger has passed validation and runtime parity.
Owner: Codex
```

## Protocol249 - EXP_ENTRY_QUALITY_CALIBRATOR_V1

- What is this: experiment / model change
- Changes paper default: no
- Candidate: CHALLENGER_ENTRY_QUALITY_CALIBRATED_PREMIUM_BLEND_V1
- Baseline: PAPER_DEFAULT_PROTOCOL101
- Data used: `v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet`
- Paid data downloaded: false
- Broker endpoint called: false
- Decision: `rejected_entry_quality_calibrator_no_clear_improvement`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_249_entry_quality_calibrator/report.md`

## Protocol250 - EXP_CONTRACT_VALUE_SELECTION_HEAD_V1

- What is this: experiment / model change
- Changes paper default: no
- Candidate: CHALLENGER_CONTRACT_VALUE_SELECTION_HEAD_V1
- Baseline: PAPER_DEFAULT_PROTOCOL101
- Data used: `v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet`
- Paid data downloaded: false
- Broker endpoint called: false
- Decision: `rejected_contract_value_head_did_not_clear_protocol101_gate`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_250_contract_value_selection_head/report.md`

## Protocol251 - EXP_PREMIUM_BLEND_SLOT_AWARE_LIFECYCLE_V1

- What is this: experiment / lifecycle screen
- Changes paper default: no
- Candidate: CHALLENGER_PREMIUM_BLEND_SLOT_AWARE_LIFECYCLE_V1
- Baseline: PAPER_DEFAULT_PROTOCOL101
- Data used: `v4/audit/autoresearch/v4_aplus_hypothesis_240_premium_leaning_blend_five_seed_decision/five_seed_model_trades.csv` plus existing normalized quote paths
- Paid data downloaded: false
- Broker endpoint called: false
- Decision: `rejected_premium_blend_lifecycle_does_not_beat_paper_default`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_251_premium_blend_slot_aware_lifecycle_screen/report.md`

## Protocol252 - EXP_BASELINE_PRESERVING_LIFECYCLE_EXTENSION_V1

- What is this: experiment / conservative lifecycle extension screen
- Changes paper default: no
- Candidate: CHALLENGER_BASELINE_PRESERVING_LIFECYCLE_EXTENSION_V1
- Baseline: PAPER_DEFAULT_PROTOCOL101
- Data used: `v4/audit/autoresearch/v4_aplus_hypothesis_240_premium_leaning_blend_five_seed_decision/five_seed_model_trades.csv` plus existing normalized quote paths
- Paid data downloaded: false
- Broker endpoint called: false
- Decision: `research_only_extension_beats_paper_but_not_base`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_252_baseline_preserving_lifecycle_extension_screen/report.md`

## Protocol253 - EXP_RISK_ADJUSTED_UTILITY_GATE_V1

- What: risk-adjusted utility gate over `CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1`.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `rejected_risk_adjusted_gate_no_clear_improvement`.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_253_risk_adjusted_utility_gate/report.md`.

## Protocol254 - EXP_POLICY_ROUTER_UNION_SLOT_AWARE_V1

- What: policy-router event model over Protocol101 and premium-blend proposal streams from the existing attribution ledger.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `rejected_policy_router_does_not_clear_protocol101_gate`.
- Interpretation: Partial-screen signal only. The router beat both streams on Q1/March but lost to premium-blend on recent 2026, and the input artifact lacked q1/q2 proposal rows needed for q3/q4 protected validation.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_254_policy_router_union_slot_aware/report.md`.

## Protocol255 - EXP_POLICY_ROUTER_HISTORY_FEATURES_V1

- What: policy-router event model with causal short-history features over Protocol101 and premium-blend proposal streams.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `rejected_policy_router_does_not_clear_protocol101_gate`.
- Interpretation: Strongest router screen so far. It improved combined Q1/March/recent PnL and win rate versus Protocol254, but still underperformed premium-blend on recent 2026 and lacks complete q3/q4 validation coverage.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_255_policy_router_history_features/report.md`.

## Protocol256 - EXP_POLICY_ROUTER_RELATIVE_STREAM_FEATURES_V1

- What: policy-router event model with same-decision comparative stream features added to Protocol255.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `rejected_policy_router_does_not_clear_protocol101_gate`.
- Interpretation: Comparative stream features did not beat the history router. The failure mode remains recent-2026 Protocol101 substitution weakness plus incomplete early-fold proposal coverage.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_256_policy_router_relative_stream_features/report.md`.

## Protocol254 - EXP_POLICY_ROUTER_UNION_SLOT_AWARE_V1

- What: policy-router event model over Protocol101 and premium-blend proposal streams.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `rejected_policy_router_does_not_clear_protocol101_gate`.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_254_policy_router_union_slot_aware/report.md`.

## Protocol255 - EXP_POLICY_ROUTER_HISTORY_FEATURES_V1

- What: policy-router event model over Protocol101 and premium-blend proposal streams.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `rejected_policy_router_does_not_clear_protocol101_gate`.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_255_policy_router_history_features/report.md`.

## Protocol256 - EXP_POLICY_ROUTER_RELATIVE_STREAM_FEATURES_V1

- What: policy-router event model over Protocol101 and premium-blend proposal streams.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `rejected_policy_router_does_not_clear_protocol101_gate`.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_256_policy_router_relative_stream_features/report.md`.

## Protocol258 - EXP_POLICY_ROUTER_COMPLETE_STREAM_HISTORY_V1

- What: policy-router event model over Protocol101 and premium-blend proposal streams.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `research_only_policy_router_beats_protocol101_but_not_best_challenger_stream`.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_258_policy_router_complete_stream_history/report.md`.

## Protocol260 - EXP_POLICY_ROUTER_RELIABILITY_PRIORS_V1

- What: policy-router event model over Protocol101 and premium-blend proposal streams.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `research_only_policy_router_beats_protocol101_but_not_best_challenger_stream`.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_260_policy_router_reliability_priors/report.md`.

## Protocol261 - EXP_ROUTER_SOURCE_PENALTY_CALIBRATION_V1

- What: policy-router event model over Protocol101 and premium-blend proposal streams.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `research_only_policy_router_beats_protocol101_but_not_best_challenger_stream`.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_261_router_source_penalty_calibration/report.md`.

## Protocol262 - EXP_SOURCE_PENALTY_ROUTER_SLOT_LIFECYCLE_V1

- What: position-state lifecycle hold/exit model over the frozen Protocol261 entry stream.
- Paper default changed: no.
- Paid data downloaded: no.
- Broker endpoint called: no.
- Decision: `research_only_premium_blend_lifecycle_beats_paper_but_not_base`.
- Interpretation: This is the first lifecycle-style model in this loop that beats PAPER_DEFAULT_PROTOCOL101 on Q1 2026, March 2026, and recent 2026 common lifecycle-harness splits. It does not yet replace the paper default because it still trails the Protocol261 base stream on recent 2026 and needs attribution/runtime parity.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_262_source_penalty_slot_lifecycle/report.md`.

## Protocol263 - AUDIT_SOURCE_PENALTY_LIFECYCLE_ATTRIBUTION_V1

- What is this: diagnostic / lifecycle attribution audit
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `lifecycle_beats_paper_but_recent_gap_is_base_stream_opportunity_loss`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_263_source_penalty_lifecycle_attribution/report.md`

## Protocol264 - EXP_SOURCE_PENALTY_ROUTER_UNIFIED_ENTRY_LIFECYCLE_V1

- What is this: experiment / research challenger training run
- Changes paper default: no
- Candidate: `CHALLENGER_SOURCE_PENALTY_UNIFIED_ENTRY_LIFECYCLE_V1`
- Baseline: `PAPER_DEFAULT_PROTOCOL101`; secondary baseline `CHALLENGER_ROUTER_SOURCE_PENALTY_CALIBRATED_V1`
- Data used: frozen Protocol261 source-penalty router entry stream plus normalized official-context SPXW rows
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `mixed_unified_sequence_beats_paper_default_but_not_lifecycle_baseline`
- Interpretation: Unified entry-plus-hold training still beats PAPER_DEFAULT_PROTOCOL101 on March 2026, Q1 2026, and recent 2026, but it remains research-only because it trails the Protocol261 base stream on recent 2026.
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_264_source_penalty_unified_entry_lifecycle/report.md`

## Protocol265 - EXP_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1

- What is this: experiment / baseline-anchored lifecycle continuation model
- Changes paper default: no
- Candidate: CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1
- Baseline: PAPER_DEFAULT_PROTOCOL101; secondary baseline CHALLENGER_ROUTER_SOURCE_PENALTY_CALIBRATED_V1
- Data used: `v4/audit/autoresearch/v4_aplus_hypothesis_261_router_source_penalty_calibration/model_trades.csv` plus normalized quote paths
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `research_only_source_penalty_baseline_anchor_beats_paper_but_not_base`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation/report.md`

## Protocol266 - AUDIT_PROTOCOL265_ARTIFACT_REPRODUCTION_V1

- What is this: audit / saved-artifact replay reproduction for Protocol265
- Changes paper default: no
- Candidate audited: CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1
- Source protocol: `Protocol265`
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `pass_protocol265_artifact_reproduction_exact`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_266_protocol265_artifact_reproduction/report.md`

## Protocol267 - DECISION_FREEZE_PROTOCOL265_RESEARCH_ONLY_V1

- What is this: freeze / research-only decision packet
- Changes paper default: no
- Candidate: `CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1`
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `freeze_protocol265_research_only_protocol101_unchanged`
- Report: `v4/promotion/PROTOCOL_265_RESEARCH_FREEZE.md`

## Protocol268 - AUDIT_PROTOCOL265_EXTENSION_REGIME_ATTRIBUTION_V1

- What is this: diagnostic / extension-regime attribution audit
- Changes paper default: no
- Candidate: `CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1`
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `extension_value_is_regime_dependent_and_learnable`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_268_protocol265_extension_regime_attribution/report.md`

## Protocol269 - RUNTIME_PROTOCOL265_NO_ORDER_PARITY_V1

- What is this: runtime / no-order historical replay proxy for Protocol265
- Changes paper default: no
- Candidate: `CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1`
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `runtime_protocol265_no_order_parity_passed_historical_proxy`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_269_protocol265_no_order_runtime_parity/report.md`

## Protocol270 - DATASET_FULL_SURFACE_ACTION_ADVANTAGE_V1

- What is this: dataset / full-surface serial action-advantage labels
- Changes paper default: no
- Candidate: `CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V1_INPUT`
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `action_advantage_dataset_ready_for_unified_policy_training`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/report.md`

## Protocol271 - CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V1

- What is this: model / unified full-surface action-advantage challenger
- Changes paper default: no
- Candidate: `CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V1`
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `research_only_unified_action_advantage_policy_not_yet_paper_default`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_271_unified_action_advantage_policy/report.md`

## Protocol272 - AUDIT_FILL_MODEL_READINESS_V1

- What is this: diagnostic / fill-model calibration readiness audit
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `blocked_insufficient_fill_observations_keep_stress_replay`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness/report.md`

## Protocol273 - AUDIT_MODEL_SELECTION_OVERFIT_RISK_V1

- What is this: diagnostic / model-selection overfit and false-discovery risk audit
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `model_selection_overfit_risk_confirmed_reserve_new_untouched_block`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/report.md`

## Protocol274 - DATASET_POSITION_STATE_ACTION_ADVANTAGE_V1

- What is this: dataset / holding-state hold-vs-exit action-advantage labels
- Changes paper default: no
- Candidate: `CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V2_INPUT`
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `position_state_action_advantage_dataset_ready_for_lifecycle_training`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/report.md`

## Protocol275 - CHALLENGER_POSITION_STATE_LIFECYCLE_POLICY_V1

- What is this: model / holding-state lifecycle advantage policy
- Changes paper default: no
- Candidate: `CHALLENGER_POSITION_STATE_LIFECYCLE_POLICY_V1`
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `position_lifecycle_policy_has_learned_signal_needs_serial_integration`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_275_position_state_lifecycle_policy/report.md`

## Protocol276 - CHALLENGER_INTEGRATED_ENTRY_LIFECYCLE_SERIAL_REPLAY_V1

- What is this: model integration / strict one-account serial replay of full-action entry plus lifecycle hold-exit policies
- Changes paper default: no
- Candidate: `CHALLENGER_INTEGRATED_ENTRY_LIFECYCLE_SERIAL_REPLAY_V1`
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `research_only_integrated_entry_lifecycle_does_not_surpass_protocol101`
- Report: `v4/audit/autoresearch/v4_aplus_hypothesis_276_integrated_entry_lifecycle_serial_replay/report.md`

## FOUNDATION_HARDENING_READINESS_PACKET_V1

- What is this: implementation audit / foundation hardening packet
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Decision: `foundation_hardening_required_before_model_experiments`
- Report: `v4/audit/autoresearch/foundation_hardening_review/report.md`
- Result: Open-ended model experiments remain paused until the blocked foundation gates close.

## AUDIT_PROTOCOL276_INTEGRATED_LIFECYCLE_FAILURE_ATTRIBUTION_V1

- What is this: foundation audit / Protocol276 failure attribution
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `protocol276_failure_attribution_complete_foundation_work_required`
- Report: `v4/audit/autoresearch/protocol276_integrated_lifecycle_failure_attribution/report.md`
- Result: Protocol276 failure remains foundation work; no new model search is authorized by this audit.

## FOUNDATION_UNIFIED_CONSERVATIVE_OFFLINE_POLICY_V1

- What is this: foundation spec / unified conservative offline policy direction
- Changes paper default: no
- Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
- Abandoned candidate: `CHALLENGER_INTEGRATED_ENTRY_LIFECYCLE_SERIAL_REPLAY_V1 / Protocol276`
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `unified_conservative_offline_policy_foundation_frozen_no_model_training`
- Report: `v4/audit/autoresearch/unified_conservative_offline_policy_foundation/report.md`

## DATASET_UNIFIED_POLICY_TRAJECTORY_FOUNDATION_V1

- What is this: dataset foundation / unified conservative policy trajectory manifest
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `unified_trajectory_foundation_ready_training_blocked_by_foundation_gates`
- Report: `v4/audit/autoresearch/unified_policy_trajectory_foundation/report.md`

## FOUNDATION_UNTOUCHED_EVAL_BLOCK_RESERVATION_V1

- What is this: foundation gate / untouched evaluation block reservation
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `untouched_holdout_reserved_pending_new_data_collection`
- Report: `v4/audit/autoresearch/unified_untouched_holdout_reservation/report.md`
- Result: Existing Q3/Q4/Q1/March/recent blocks are diagnostics; a future unseen block is reserved for final claims.

## FOUNDATION_UNIFIED_NEURAL_TRAINING_READINESS_V1

- What is this: foundation gate / unified conservative policy neural-training readiness
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Training decision: `neural_training_not_ready_foundation_gates_blocked`
- Protocol101 challenge decision: `protocol101_challenge_not_ready_foundation_gates_blocked`
- Report: `v4/audit/autoresearch/unified_neural_training_readiness/report.md`
- Training blockers: `['Full serial wait/enter/hold/exit DP oracle', 'Protocol101 baseline action attachment']`

## DATASET_PROTOCOL101_BASELINE_ACTION_ATTACHMENT_V1

- What is this: dataset foundation / Protocol101 baseline action attachment
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `protocol101_baseline_attachment_partial_missing_trajectory_split_baselines`
- Report: `v4/audit/autoresearch/unified_protocol101_baseline_attachment/report.md`

## DATASET_UNIFIED_SERIAL_DP_ORACLE_V1

- What is this: dataset foundation / unified serial DP oracle training-scope manifest
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `unified_serial_dp_oracle_ready_for_baseline_aligned_training_scope`
- Report: `v4/audit/autoresearch/unified_serial_dp_oracle/report.md`

## CHALLENGER_UNIFIED_CONSERVATIVE_NEURAL_POLICY_V1

- What is this: preregistered model training / conservative unified neural policy
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: yes
- Decision: `conservative_neural_policy_trained_replay_and_challenge_still_blocked`
- Report: `v4/audit/autoresearch/unified_conservative_neural_policy_v1/report.md`
- Result: Trained only; Protocol101 challenge remains blocked.

## REPLAY_UNIFIED_CONSERVATIVE_NEURAL_POLICY_STRICT_SERIAL_V1

- What is this: strict one-account serial replay of the trained conservative neural policy with Protocol101 defer fallback
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `strict_replay_complete_model_deferred_to_protocol101_flat_gate_too_conservative`
- Report: `v4/audit/autoresearch/unified_conservative_neural_policy_strict_replay_v1/report.md`
- Result: Strict replay complete; Protocol101 challenge remains blocked.

## DIAGNOSTIC_UNIFIED_CONSERVATIVE_FLAT_GATE_V1

- What is this: diagnostic / flat-entry conservative gate abstention analysis
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `flat_entry_gate_overconservative_zero_model_overrides`
- Report: `v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic/report.md`

## CHALLENGER_UNIFIED_CONSERVATIVE_NEURAL_POLICY_FLAT_CALIBRATED_V1

- What is this: preregistered flat-entry calibration repair for conservative unified neural policy
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: yes
- Decision: `conservative_neural_policy_trained_replay_and_challenge_still_blocked`
- Report: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_v1/report.md`
- Result: Calibration produced nonzero candidate overrides but remained research-only pending strict replay.

## REPLAY_UNIFIED_CONSERVATIVE_NEURAL_POLICY_FLAT_CALIBRATED_STRICT_SERIAL_V1

- What is this: strict one-account serial replay of flat-calibrated conservative neural policy
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `strict_replay_complete_protocol101_challenge_still_blocked`
- Report: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_strict_replay_v1/report.md`
- Result: Positive same-scope total PnL but mixed split deltas; Protocol101 challenge remains blocked.

## DIAGNOSTIC_UNIFIED_CONSERVATIVE_FLAT_GATE_FLAT_CALIBRATED_V1

- What is this: diagnostic / flat-calibrated entry gate override analysis
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `flat_entry_gate_produces_model_overrides_needs_replay_attribution`
- Report: `v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic_flat_calibrated_v1/report.md`

## ATTRIBUTION_UNIFIED_CONSERVATIVE_NEURAL_OVERRIDES_V1

- What is this: attribution / calibrated conservative neural challenger overrides
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `override_attribution_mixed_split_research_only`
- Report: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_override_attribution_v1/report.md`

## ATTRIBUTION_UNIFIED_CONSERVATIVE_Q1_Q3_UNDERPERFORMANCE_V1

- What is this: attribution / flat-calibrated Q1-Q3 underperformance versus same-scope Protocol101
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost`
- Report: `v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1/report.md`

## FOUNDATION_UNIFIED_SLOT_OPPORTUNITY_DEFER_OVERLAY_V1

- What is this: foundation / slot opportunity-cost defer overlay contract and oracle diagnostic
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator`
- Report: `v4/audit/autoresearch/unified_slot_opportunity_defer_overlay_foundation/report.md`

## DATASET_UNIFIED_SLOT_OPPORTUNITY_COST_LABELS_V1

- What is this: dataset / candidate-level blocked-Protocol101 opportunity-cost labels
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `slot_opportunity_cost_labels_ready_for_causal_estimator`
- Report: `v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/report.md`

## FOUNDATION_UNIFIED_SLOT_OPPORTUNITY_COST_ESTIMATOR_V1

- What is this: foundation estimator / causal Protocol101 slot opportunity-cost model
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: yes
- Decision: `slot_opportunity_cost_estimator_ready_for_defer_overlay_replay`
- Report: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/report.md`

## REPLAY_UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_V1

- What is this: strict replay / learned slot opportunity-cost defer overlay on the flat-calibrated conservative policy
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `learned_slot_opportunity_defer_overlay_replay_safe_but_deferred_all_overconservative`
- Report: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay/report.md`

## PREREGISTERED_UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_V1

- What is this: preregistration / exactly one learned-defer neural policy run
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `preregistered_single_run_before_training`
- Report: `v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_PREREGISTERED_SPEC_V1.md`

## CHALLENGER_UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_PREREGISTERED_V1

- What is this: preregistered model training / conservative unified neural policy with learned-defer replay frozen in advance
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: yes
- Decision: `conservative_neural_policy_trained_replay_and_challenge_still_blocked`
- Report: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/report.md`

## REPLAY_UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_PREREGISTERED_V1

- What is this: strict replay / preregistered learned-defer neural policy with frozen slot opportunity-cost overlay
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Decision: `learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training`
- Result: Diagnostic same-scope deltas versus Protocol101 were positive at `$0.00`, `$0.10`, and `$0.25` slippage stress, but Protocol101 challenge remains blocked.
- Report: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1/report.md`

## VALIDATION_LEARNED_DEFER_CHALLENGER_RESEARCH_PACKET_V1

- What is this: research-only validation packet for the frozen learned-defer challenger
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `learned_defer_packet_complete_holdout_blocked_by_calibration`
- Result: Frozen replay reproduction passed exactly; flat-entry anomaly was resolved as policy-weighting mismatch; concentration passed with Q4/top-trade warnings; holdout scoring remains blocked by slot-cost calibration instability plus fill, live parity, untouched data, and formal validation gates.
- Report: `v4/audit/autoresearch/learned_defer_challenger_research_packet_v1/report.md`

## STRATEGY_AUDIT_TRADING_BOT_ENGINEER_HANDOFF_2026_05_24

- What is this: strategy-forensics handoff / Protocol101-centered audit for outside trading bot engineer guidance
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `strategy_audit_complete_model_experiments_paused_until_protocol101_trade_questions_answered`
- Result: Reframed the next phase away from generic challenger training and toward Protocol101 trade anatomy, weak-point attribution, missed-winner/runners/slot-cost diagnostics, execution realism, and validation integrity. Protocol101 remains paper default.
- Report folder: `v4/docs/trading_bot_engineer_strategy_audit_2026_05_24/`

## AUDIT_PROTOCOL101_STRATEGY_FORENSICS_PACKET_V1

- What is this: research-only Protocol101 strategy-forensics packet / claim-by-claim analysis of the outside response
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_strategy_forensics_packet_complete_model_training_still_blocked`
- Result: The response is directionally supported, but promotion/model work remains blocked. Current evidence answers the trade-atlas, hard-stop, losing-day, score-proxy, and timing-fragility questions from existing artifacts; runner, missed-winner, internal slot-cost, fill realism, and formal validation questions require new counterfactual or live evidence artifacts.
- Report: `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_STRATEGY_FORENSICS_PACKET_V1.md`

## FOUNDATION_TRUTH_GROUNDED_REPLACEMENT_PROGRAM_V1

- What is this: foundation / truth-grounded Protocol101 replacement research program
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `truth_grounded_replacement_program_created_training_and_replacement_blocked`
- Result: Created the three-track replacement program: Protocol101 forensics, named alpha playbooks, and a playbook-aware replacement ML stack. Seeded the strategy hypothesis registry, Protocol101 weakness matrix, research data layer, stage-gate rulebook, and replacement candidate protocol spec. Training and replacement claims remain blocked by execution realism, formal strategy-matrix validation, and untouched-holdout gates.
- Report: `v4/audit/autoresearch/truth_grounded_replacement_program_v1/report.md`
- Doc: `v4/docs/TRUTH_GROUNDED_REPLACEMENT_PROGRAM_V1.md`

## AUDIT_PROTOCOL101_TRACK_A_FORENSICS_V1

- What is this: Track A Protocol101 forensics / hard-stop, losing-day, runner, internal slot-cost, side, and score diagnostics
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_track_a_forensics_partial_complete_training_still_blocked`
- Result: Created executable Track A diagnostics from current artifacts. Seed-1 Protocol101 has `1,028` trades, `$319,050` PnL, `0.761` win rate, and `4.421` PF. Hard stops remain severe but sparse: `11` rows for `-$14,470`; current causal proxy rules classify `2` rows as path-management candidates, `1` as fast-adverse, and `8` as unclassified/unavoidable pending manual chart/path review. Exact selected-trade runner paths now cover `709 / 1,028` selected trades and show `517` runner/giveback-guard rows, but this remains hindsight diagnostic evidence until a causal in-trade runner/giveback rule is defined. Track A now also references `198,261` Protocol101 hold/exit action-advantage rows; the high future-best hold signal is explicitly blocked from training until slot opportunity, switching cost, fill/latency, distributional risk, and validation terms are added. The prior internal slot-cost blocker has been replaced by a counterfactual-flat diagnostic: `3,723` model-approved Protocol101 entries were blocked while actual Protocol101 was already holding, with `$399,510` best-blocked-minus-open total, but this is still an upper-bound/non-additive diagnostic and not a training label.
- Report: `v4/audit/autoresearch/protocol101_track_a_forensics_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_TRACK_A_FORENSICS_V1.md`

## AUDIT_PROTOCOL101_EXACT_SELECTED_TRADE_RUNNER_PATHS_V1

- What is this: exact selected-trade Protocol101 post-exit runner/giveback path audit
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_exact_selected_runner_paths_complete_training_still_blocked`
- Result: Attached normalized quote paths to `709 / 1,028` Protocol101 selected trades. Hindsight post-exit best delta is `$1,147,620`, but naive forced-flat delta is `-$123,260`, so the runner question is real but cannot be answered with a blanket hold-longer rule. The exact-path state summary shows `314` runner-extension candidates and `203` giveback-guard-required rows. Runner research must now define causal in-trade confirmation and giveback guards before any model training.
- Report: `v4/audit/autoresearch/protocol101_exact_selected_trade_runner_paths_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_EXACT_SELECTED_TRADE_RUNNER_PATHS_V1.md`

## FOUNDATION_PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_V1

- What is this: Protocol101 hold/exit action-advantage foundation, aligned to `engineer-response.md` lifecycle guidance
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_hold_exit_action_advantage_foundation_complete_training_blocked`
- Result: Built `198,261` holding-state rows over `709 / 1,028` Protocol101 selected trades. The audit labels `A_hold = Q(hold) - Q(exit now at bid)` and keeps future/path labels out of the causal feature contract. At Protocol101 exits, oracle-best hold fraction is `0.917`, but one-step hold fraction is only `0.258`, reinforcing that this is not a blanket hold-longer recommendation. Training remains blocked pending counterfactual flat-slot opportunity cost, switching cost, calibrated fill/latency realism, distributional risk targets, train/live parity, untouched validation, and formal overfit controls.
- Report: `v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_FOUNDATION_V1.md`

## AUDIT_PROTOCOL101_INTERNAL_SLOT_COST_COUNTERFACTUAL_V1

- What is this: Track A counterfactual-flat Protocol101 internal slot-cost diagnostic
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_internal_slot_cost_counterfactual_complete_training_still_blocked`
- Result: Scored the frozen Protocol101 event policy at every eligible event as if the account were flat. The audit found `8,179` hypothetical flat Protocol101 entries and `3,723` entries blocked by actual open Protocol101 positions. `2,291` open trades had at least one blocked entry; `1,109` had a best blocked entry whose frozen candidate PnL exceeded the actual open-trade PnL. Aggregate best-blocked-minus-open was `$399,510`, concentrated mostly in Q1/March, but blocked events inside one open interval are mutually exclusive and live fill/latency realism is not calibrated.
- Report: `v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_INTERNAL_SLOT_COST_COUNTERFACTUAL_V1.md`

## AUDIT_PROTOCOL101_SLOT_COST_ARCHETYPE_DECOMPOSITION_V1

- What is this: Track A internal slot-cost archetype decomposition
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_slot_cost_archetype_decomposition_complete_model_design_ready_training_blocked`
- Result: Decomposed the `3,723` blocked Protocol101 internal slot-cost events into open-trade and blocked-signal archetypes with `0` actual-trade join failures. Top hypotheses are losing open trade blocks positive later signal (`485` rows, `$635,240`), opposite-side later signal blocked (`428`, `$572,430`), sequence residual slot cost (`741`, `$472,930`), long-duration slot cost (`396`, `$409,130`), and same-side later signal blocked (`681`, `$287,560`). This reaches the Track A stopping point: manual trading review must choose exactly one named hypothesis before model design.
- Report: `v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_SLOT_COST_ARCHETYPE_DECOMPOSITION_V1.md`

## SYNTHESIS_PROTOCOL101_TRACK_A_FOUNDATIONAL_TRUTH_V1

- What is this: Track A stopping-point synthesis for the 0DTE trading bot / ML problem
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `track_a_foundational_truth_complete_manual_strategy_selection_required`
- Result: Consolidated Track A findings into the foundational truth: Protocol101 remains best because it encodes a real deployable trader playbook, while the most promising improvement paths are narrow playbook hypotheses around loss/reversal exits, opposite-side regime flips, same-side switch/runner behavior, and long-duration/fallback slot cost. Generic neural model work remains blocked until one hypothesis is selected, reviewed manually, and converted into causal labels plus mutually exclusive replay with fill/latency realism.
- Doc: `v4/docs/PROTOCOL101_TRACK_A_FOUNDATIONAL_TRUTH_V1.md`

## AUDIT_PROTOCOL101_STRATEGY_SELECTION_PACKET_V1

- What is this: Track A strategy-selection packet / single next hypothesis recommendation
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_strategy_selection_complete_first_diagnostic_recommended_training_blocked`
- Result: Ranked the immediate Protocol101 improvement hypotheses from the slot-cost archetype diagnostics. The selected first diagnostic is `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1`: `485` evidence rows, `$635,240` positive slot-cost total, and a strong overlap with the regime-flip subcase (`325` rows, `$517,980`, `0.815` of loss-reversal positive slot cost). This authorizes only manual review and a no-training mutually exclusive diagnostic replay; neural training and Protocol101 replacement remain blocked.
- Report: `v4/audit/autoresearch/protocol101_strategy_selection_packet_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_STRATEGY_SELECTION_PACKET_V1.md`

## AUDIT_PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_DIAGNOSTIC_V1

- What is this: no-training diagnostic for the selected `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1` hypothesis
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_loss_reversal_exit_gate_diagnostic_partial_causal_state_coverage_training_blocked`
- Result: Confirmed the key label risk before model work. The slot-cost artifact has `485` final-loss candidates and `$635,240` positive slot-cost total, but causal holding state at the later blocked-signal time matched only `65 / 485` rows. Among matched rows, `46` were actually losing at the blocked signal, while `19` were not. The first live-test bucket is current-loss plus opposite-side plus one-step-hold-negative (`20` rows, `$32,790`). Final-loser status is not a valid live ML label; the next step is full causal state attachment plus mutually exclusive keep-hold vs exit/switch replay.
- Report: `v4/audit/autoresearch/protocol101_loss_reversal_exit_gate_diagnostic_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_DIAGNOSTIC_V1.md`

## AUDIT_PROTOCOL101_LOSS_REVERSAL_CAUSAL_STATE_ATTACHMENT_V1

- What is this: direct quote-path causal-state attachment for the selected loss-reversal hypothesis
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_loss_reversal_causal_state_attachment_partial_quote_coverage_replay_scaffold_ready_training_blocked`
- Result: Recomputed live state directly from normalized quote paths instead of relying on the seed-1 hold/exit foundation. Coverage improved to `441 / 485` rows (`0.909`). Among matched rows, `293` were currently losing at the blocked signal, `148` were not, and `145` fell into the strongest current-loss/opposite-side/negative-next/exit-deteriorates bucket. Neural training remains blocked; the next step is action pricing and then full serial replay.
- Report: `v4/audit/autoresearch/protocol101_loss_reversal_causal_state_attachment_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_LOSS_REVERSAL_CAUSAL_STATE_ATTACHMENT_V1.md`

## AUDIT_PROTOCOL101_LOSS_REVERSAL_LOCAL_ACTION_PRICING_V1

- What is this: local action pricing for the selected loss-reversal hypothesis
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_loss_reversal_local_action_pricing_complete_full_serial_replay_required_training_blocked`
- Result: Priced `441` matched rows as mutually exclusive local actions: keep holding, exit now, or exit and switch into the later Protocol101-approved signal. Switch-minus-keep totals `$511,500`, `$502,680` under `$0.10` two-side stress, and `$489,450` under `$0.25` two-side stress. The priority-1 bucket has `145` rows and remains `$223,120` positive under `$0.25` stress. This supports the playbook as a candidate for full serial replay, not for neural training or paper-default change.
- Report: `v4/audit/autoresearch/protocol101_loss_reversal_local_action_pricing_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_LOSS_REVERSAL_LOCAL_ACTION_PRICING_V1.md`

## AUDIT_PROTOCOL101_LOSS_REVERSAL_FULL_SERIAL_REPLAY_V1

- What is this: full serial flat-stream replay for the selected priority-1 loss-reversal overlay
- Changes paper default: no
- Paid data downloaded: no
- Broker endpoint called: no
- Model training: no
- Untouched holdout scored: no
- Decision: `protocol101_loss_reversal_full_serial_replay_complete_training_still_blocked`
- Result: Replayed the frozen hypothetical-flat Protocol101 proposal stream under one-position serial constraints and applied only the priority-1 loss-reversal release rule. The baseline reproduces `4,456` trades at zero stress in the diagnostic fold/seed/split scope. The overlay fires `145` exits and improves same-scope baseline by `$201,350` at zero stress, `$198,990` at `$0.10` per-side stress, and `$195,450` at `$0.25` per-side stress; every diagnostic split is positive under all three stress settings. This is the first Track A result that looks like a concrete Protocol101 improvement playbook, but neural training and paper-default change remain blocked by exposed splits, fill/latency/quote-freshness realism, missing quote-state rows, and live no-order parity.
- Report: `v4/audit/autoresearch/protocol101_loss_reversal_full_serial_replay_v1/report.md`
- Doc: `v4/docs/PROTOCOL101_LOSS_REVERSAL_FULL_SERIAL_REPLAY_V1.md`

## PROTOCOL101_WALKING_SKELETON_FORCED_BUY_STAGE2_3_V1

- What is this: quarantined forced-BUY downstream plumbing harness, 1-second lifecycle baseline, serial replay, and D60 visualization for walking-skeleton Stages 2–3
- Quarantine labels: `walking_skeleton / throwaway / paper-only / forced_buy_plumbing_harness`
- Changes paper default: no
- Paid data downloaded: no; used only the exact 13-session firewall-safe official `cbbo-1s` slice pinned by Stage 0
- Broker endpoint called: no
- Model training: yes, one throwaway HGB HOLD/EXIT baseline on the frozen 8-session fit slice with threshold derived only from the 2-session calibration slice
- Untouched/protected/outer/holdout data scored: no
- Decision: `walking_skeleton_stages2_3_independently_verified_stop_before_stage4_owner_authorizations`
- Result: The real frozen Stage-1 composer remains `ABSTAINS`; its owned replay output contained exactly `552` real A6-passing proposals. A downstream-only wrapper selected six deterministic fill-safe, D48-safe proposals per each of 13 sessions (`78` intents) and built `771,018` canonical official 1-second lifecycle rows. Simulator v5 accepted `9` one-account replay trades across the three owned validation sessions (`6 / 2 / 1`), with `4` learned exits, `4` floor triggers, `1` forced flat, `31,454` logged HOLD states, zero simulator skips, and all accepted entries passing D48/D49. Descriptive estimated P&L was `-$552` after fees; the four buckets were `0` big wins, `4` scratch/small wins, `2` small losses, and `3` big losses.
- Learning: The 1-second HGB baseline produced descriptive calibration ROC AUC `0.8372` but average precision only `0.0664`; disposition: retain strictly as plumbing evidence, make no quality/alpha claim, and do not progress to Stage 4. The D58 report-only minute-vs-1-second early-drawdown comparison had Spearman `0.9857` and worst-quartile overlap `1.0`; disposition: record but do not use for entry admission or promotion. The negative replay and missing big-win column are shaped by explicit branch canaries and are not an optimizable profile; disposition: no threshold/floor tuning from this replay. Independent verification corrected an exact-15:30 admission boundary, preserved the raw exchange-event clock and nonzero quote age alongside simulator-v5's frozen occupancy-clock adapter, and fixed global-to-dense bar coordinates in the multi-session trade chart. Local browser QA then confirmed the three-session span and marker-bearing views. Disposition: Stages 2–3 plumbing independently verified; stop before Stage 4 pending its separate owner authorizations.
- Evidence: `v4/audit/autoresearch/protocol101_walking_skeleton_stage2_3/receipt.json`
- Delta review: `v4/audit/autoresearch/protocol101_walking_skeleton_stage2_3/delta_scoped_review.json` (`PASS`, `38 / 38` checks, including D52 floor-off comparison, exact entry cutoff, dual-clock provenance, dense three-session chart coordinates, and separately attested local-browser visual QA)
- Report: `v4/audit/autoresearch/protocol101_walking_skeleton_stage2_3/report.md`
