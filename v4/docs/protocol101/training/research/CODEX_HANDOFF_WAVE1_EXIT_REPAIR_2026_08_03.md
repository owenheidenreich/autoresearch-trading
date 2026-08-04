# Codex Handoff — Wave 1: Repair the Exit Model (2026-08-03)

Self-contained. Read §1–§3, implement §4, run §5, report §6. End with `STOP_FOR_CLAUDE_VERIFICATION`.

---

## 1. What is newly available (Claude built this; use it, do not rebuild it)

Two new modules, 39 tests, all green (`bf681824`, `36c96b38`, `d097fadd`).

### `v4/research/pathd_model_gate.py`

Rejection tests — each returns `TestResult(name, passed, detail, metrics)` and each exists because it
would have caught a real failure:

| Function | Catches | Limit |
|---|---|---|
| `exit_first_step_rate(exit_indices)` | Protocols 029/030/031 always-exit | >50% fails |
| `tail_preservation(policy, baseline)` | amputating the convex payoff | p99 or top-decile <80% of baseline fails |
| `decile_monotonicity(scores, outcomes, folds)` | April's closed coverage thresholds | top decile must be best and beat bottom |
| `label_balance(labels)` | the 98%-label disaster and its mirror | outside 25–90% positive fails |

Charter-mandated diagnostics (`PROTOCOL101_TRADER_CHARTER.md`, SIGNED, lists these under "New (to
implement)"): `four_bucket_distribution`, `harvest_ratio`, `underwater_duration`, `pnl_concentration`.
**Every packet must report all four.** Only the big-loss share gates (≤2%). Win rate and concentration are
report-only by charter mandate — do not turn either into an objective.

`acceptance_tier(...)` returns `TIER_A / TIER_B / NO_EDGE / INVALID`. `INVALID` takes precedence: an
accepted negative control discards the run.

**Validation already performed:** all four rejection tests were run against the current artifacts and
**all four FAIL**, reproducing 95.1% first-step exits, p99 $86 vs $3,422, best-decile 5, ~16% positive
labels. That is the module working, not a defect.

### `v4/research/pathd_research_loop.py`

`run_wave(spec, runner, registry_path=...)` gives bounded, deduplicated, prior-art-checked iteration.

- `Hypothesis(hypothesis_id, mechanism, params, rationale)` — `mechanism` drives both the prior-art search
  and the semantic hash, so renaming does not evade dedup.
- `WaveSpec(wave_id, objective, hypotheses, budget)` — **the declared hypothesis count IS the maxT family
  size.** `validate()` refuses to declare more than the budget.
- `ExperimentResult(...)` — what your runner returns.
- `prior_art_check(mechanism)` / `blocked_by_prior_art(hits)` — searches both canonical history documents.

**Read this carefully:** the check blocks on a do-not-retest row **or a rejecting verdict anywhere in the
protocol decoder**. A §4-only check waves through `"loss-only damage-control exit"`, because that is
recorded as "Reject / exit-loop stop" in the lineage and appears nowhere in §4 — which is exactly how
Path-D re-ran Protocols 029/030/031. It correctly does **not** block `"recovery penalty"`.

---

## 2. What is wrong with the current exit model

Not broken — **degenerate**, and in a specific, previously-diagnosed way.

- `learned_exit_index == 0` in **980/1031 (95.1%)**; identical to `exit_immediate` in 982/1031.
- Sealed models reproduce stored predictions with **zero error** — this is not a serialization defect.
- Per-fold `mean_lcb90` offsets of **−$813 to −$895** swamp nearly every point prediction.
- Only **17.46%** of first-state `a_ref_dollars` labels are positive.
- Result: p99 **$86** and top-decile contribution **+$2,331**, versus hold-to-close **$3,422** and
  **+$212,841**. The convex tail — the entire reason to hold a long option — is amputated.

The leverage is real: the exit cuts control-entry loss **−$251,499 → −$48,804**. It is mis-calibrated,
not useless.

---

## 3. The fix April already validated

- **P065 recovery-aware penalty** — up-weight *early negative-residual states with high future recovery /
  baseline regret*. Ten-seed validated: delta-positive seeds 9/10, 10/10, 10/10, 8/10. **This is the exact
  mechanism Path-D omitted.**
- **P073 / P087 — a deterministic fallback is MANDATORY.** Both rejected removing it; 26.7% of exits
  depended on it. The learned model chooses only among non-catastrophic states; the catastrophic floor
  stays deterministic.
- **H3e — loser-defense labels are load-bearing** (removing them: PF 2.195 → 1.591, p=0.997). Our 17.46%
  positive rate is the *mirror* failure: nearly all defense, no upside capture. Rebalance toward the
  `label_balance` band and **report the rate before training**.
- **Stage-2 contract** requires the objective reward **BOTH** loss-truncation (the scratch engine) and
  tail capture (the big-win engine).

**Forbidden — all closed, the loop will block them:** H1 relaxed suffix-max target · H2 regret weighting ·
H3e profitable-only target · H3f regret regression · side-blind L3 · generic/giveback/loss-only early exits.

---

## 4. Implement: a `Runner` for the Phase-1 exit

Write `v4/research/pathd_exit_runner.py` exposing a callable `Hypothesis -> ExperimentResult` that:

1. **Pre-flight:** compute `label_balance` on the training target and record it **before** fitting. If it
   fails the band, that is a finding — report it and continue to the next hypothesis rather than training
   into a known-degenerate target.
2. **Train** the exit variant described by `hypothesis.params` on the SSD
   (`/Volumes/AR_TRADING_DATA`), reusing the existing Phase-1 machinery. **Do not modify** `FILL_LAW`, the
   causal t−60s clock, the label law, or the OOF firewall.
3. **Replay** through the existing four-box to get pooled PnL, per-fold deltas, bootstrap LCB, and the
   negative controls.
4. **Score:** run all four rejection tests plus the four charter diagnostics, and return them in
   `ExperimentResult`.

**Environment:** `AR_TRADING_DATA_ROOT=/Volumes/AR_TRADING_DATA`,
`AR_TRADING_SCRATCH_ROOT=/Volumes/AR_TRADING_DATA`,
`AR_TRADING_ARTIFACT_ROOT=/Volumes/AR_TRADING_DATA/artifacts`. Prefix long runs with `caffeinate -dimsu`
and run `storage-preflight` before and after each long stage — the 150 GB cap is checked at stage start
only.

---

## 5. Run Wave 1

Pre-register the wave (freeze the document, then run). Suggested shape — **budget 8, and the declared
count is the maxT family**:

| # | Mechanism | Rationale |
|---|---|---|
| 1 | recovery-aware exit penalty | the P065 repair, ten-seed validated |
| 2 | recovery penalty + deterministic fallback | P073/P087 made the fallback mandatory |
| 3 | rebalanced exit label toward the band | 17.46% positive is the mirror of the 98% disaster |
| 4 | reduced risk-lower-bound weight | the −$813…−$895 offsets swamp the signal |
| 5–8 | your proposals | each must clear `prior_art_check` |

Every hypothesis gets all three negative controls (sign-reversed, session-shuffled, constant). **If any
control clears the gate, the wave is `INVALID` and results are discarded** — that happened on 2026-08-03
and the rule was applied correctly; do it again.

Stop on the first `TIER_A`, on `INVALID`, or at budget exhaustion (which is `NO_EDGE`, a successful
outcome). **Do not extend the budget to keep searching.**

---

## 6. Deliverable

A results document with: the prior-art check and its citations; the frozen pre-registration; per-hypothesis
tier with every gate component; all negative controls; the four rejection tests; the four charter
diagnostics; and a bottom line — `TIER_A` / `TIER_B` / `NO_EDGE` / `INVALID`.

Append any new falsification to the do-not-retest ledger so it is never run a third time. Update the
roadmap status board. Then `STOP_FOR_CLAUDE_VERIFICATION`.

---

## 7. Hard stops and two honest notes

- **Protected 36-session firewall is SPENT** — never reopen; `holdout_open_count` stays 0. `TIER_A` means
  "worth a forward live-paper test", never "deployable".
- **Causal clock is t−60s.** Verify feature-availability-clock parity, not just future-outcome guards —
  the mutate-future lint did not catch the signed18 leak; the runtime-parity gate did.
- No paid data, no broker/order/live/paper-submit, no promotion or paper-default change, no
  launchd/plist/runtime-flag edits, without explicit owner authorization.

**Note 1 — the fill law overcharges fees.** `ENTRY_FEE_PER_SIDE_DOLLARS = 1.50`; the owner's actual IBKR
cost is **$0.65/side**. The frozen law is therefore conservative by ~$1.70 round trip. **Do not edit
`FILL_LAW`.** Report economics under the frozen law, and if you show a corrected-fee figure, label it
explicitly as a separate counterfactual.

**Note 2 — latency is already frozen for Phase-1.** `ENTRY_EMISSION_LAG_MS = 2336` matches the FROZEN
receipt (`shared_emission_lag.json`, `theta_p99_ns` 2,335,230,000); `ENTRY_ORDER_LATENCY_MS = 1000` is a
frozen conservative assumption without an independent receipt. The `live_opra_training_twin` foundation
lists both as unfrozen with 10 pre-fit blockers — **that applies to the twin, not to Phase-1.** Wave 1 runs
on Phase-1's law and is not blocked by it. Do not treat the twin's blockers as satisfied, and do not fit
against the twin foundation.

*Prepared by Claude Opus 5 — 2026-08-03.*
