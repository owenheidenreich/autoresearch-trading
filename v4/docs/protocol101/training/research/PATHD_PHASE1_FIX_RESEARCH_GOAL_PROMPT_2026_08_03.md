# Path-D Phase-1 — Fix-Research Goal Prompt (2026-08-03)

**What this is.** A ready-to-paste seed prompt for a fresh Claude (or Codex) session that researches
whether the Path-D Phase-1 loss is fixable and returns a recommendation. Paste the fenced block below
into a cold session started at the repo root.

**Why it is scoped the way it is.** GATE 1 returned `UNDERPOWERED` → STOP on 2026-08-03
(Claude-verified: replay SHA reproduces, 2 of 7 conditions passed, firewall closed). A follow-up
failure-mode diagnostic showed the entry model is **not** the defect — the mean trade loses **−$39.48**
across 156,950 causal OOF candidates, and the loss is **flat across all five chronological folds**, i.e.
a constant structural cost rather than a decaying edge. The cost decomposition is the only reason a fix
is conceivable:

| Component | Per trade | Share |
|---|---|---|
| Entry execution (half-spread $6.74 + one-tick-through $10.00) | −$16.74 | 42% |
| Structural option decay/direction (mid → exit bid) | −$19.74 | 50% |
| Fees | −$3.00 | 8% |

42% of the loss comes from `FILL_LAW` at `v4/research/pathd_phase1_entry.py:46`, which fills every entry
at **ask + one valid tick**. That is a deliberately conservative *modeling choice*, not a market fact —
which makes it the one lever that is both large and honestly measurable.

**Three failure modes this prompt is built to block:**

1. A clock leak invalidated the prior "confirmed edge" (signed18, +$540/session) — the prompt states the
   t−60s rule explicitly.
2. The protected holdout is **SPENT**. An open-ended "find an edge" loop over the same 215 sessions will
   manufacture a false positive by chance with nothing left to catch it — so the prompt forbids an
   unbudgeted entry-signal search and rank-orders specific measurable questions instead.
3. Passive-fill analysis is adversely selected. Measuring the unfilled counterfactual is a hard
   requirement, not a suggestion.

---

## The prompt

````markdown
# Goal: Path-D Phase-1 — is the 0DTE long-premium loss fixable, or is the class dead?

## Your role
Research and propose. Do NOT train a model, place an order, contact a broker, or buy data.
End with a written recommendation and STOP for owner review.

## Cold-start reading (in order)
1. `CLAUDE.md` (loads automatically) — safety rules, document precedence
2. `v4/docs/protocol101/training/research/FRESH_CONVERSATION_HANDOFF_2026_08_03.md`
3. `v4/docs/protocol101/training/research/PATHD_PHASE1_TO_LIVE_ROADMAP_AND_STATUS_2026_08_03.md`
   — especially the GATE 1 section and its two correction notes
4. `v4/docs/protocol101/training/execution/PROTOCOL101_WALKING_SKELETON_LEARNINGS_LEDGER_2026_07_30.md` (tail)

## Verified state — do not re-derive, but verify anything you doubt
- **GATE 1 = `UNDERPOWERED` → STOP** (2026-08-03). Report:
  `/Volumes/AR_TRADING_DATA/reports/phase1_four_box/replay.json`, semantic SHA `b07784a0…`
  (recompute `stable_hash(payload minus replay_sha256)` — it matches).
- Across **156,950** causal OOF candidates: mean **−$39.48/trade**, win rate ~33%.
- Decomposition: entry execution **−$16.74 (42%)**, structural decay mid→exit_bid **−$19.74 (50%)**,
  fees **−$3.00 (8%)**. **Even at perfect mid fills with zero fees this loses ~$19.74/trade.**
- **No subpopulation is positive.** Every hour (−$27.93…−$43.76), both sides (call −$38.82 /
  put −$40.12), every moneyness bucket (−$37.44/−$38.80/−$44.06), every fold
  (−$45.91/−$40.81/−$38.02/−$35.54/−$37.03). The per-fold loss is FLAT → constant structural cost,
  not a decaying edge.
- Ranking skill ≈ 0 everywhere (Spearman +0.072/+0.032/+0.040/NaN/−0.009). **Fold 3 is degenerate**:
  `calibrated_prediction` mean=median=p99=max=−37.703 (intercept-only collapse) — but that constant
  accurately estimates fold 3's realized −$35.54, i.e. the model is well calibrated and correctly
  refuses to trade. Entries per fold: 57 → 86 → 10 → 0 → 0.
- **The learned EXIT model has real skill** — it cuts control-entry loss −$251,499 → −$48,804 and
  learned-entry −$41,379 → −$7,024, with 4/5 folds positive target skill. Preserve this asset.

## The frozen fill law — your primary research target
`v4/research/pathd_phase1_entry.py:46`
```
limit = "completed_cbbo_1m_ask_plus_one_valid_tick"
fill  = "arrival_ask_at_or_below_limit_fills_at_limit"
```
Every entry pays **ask + one tick**. That is a conservative MODELING CHOICE, not a market fact, and it
accounts for $16.74 of the $39.48 loss.

## Primary question
Is any part of the −$39.48 recoverable, or is this strategy class structurally dead?

Investigate in this rank order and stop when the evidence is decisive:

**A. EXECUTION — 42% of the loss, and measurable.**
If entries were posted passively (at bid / mid / ask-without-the-tick) instead of crossing, how much of
the $16.74 is recovered **net of fill-rate loss and adverse selection**? You have full 1-second CBBO
paths, so this is measurable rather than speculative: for a limit at price P posted at time t, did the
market actually trade through P inside the decision window?
**THE TRAP — do not fall in it:** passive fills are adversely selected. You get filled preferentially
when the market is moving against you, so the fills you win are worse than average. You MUST measure
what happened to the candidates that would NOT have filled and report the selection-corrected number.
An analysis that assumes mid fills is worthless and will be rejected.

**B. HOLDING PERIOD / STRUCTURE — 50% of the loss.**
Long premium bleeds mid→exit_bid. Does a materially shorter hold, or a defined-risk structure (a spread
rather than a naked long), change the sign? Use the existing 1-second trajectories.

**C. DIRECTION — different strategy class, not a tweak.**
The mirror position (selling premium) would collect the decay but pays its own spread and carries a
fundamentally different tail. If you evaluate it, evaluate the tails honestly and label it as requiring
its own governance packet.

## Hard constraints
- **The protected 36-session firewall is SPENT.** Never reopen it. There is no clean historical
  out-of-sample data left; forward confirmation = fresh live paper data only.
- **Causal clock is t−60s.** Never source SPX context from the bar stamped `t` — that look-ahead is
  exactly what invalidated the prior "confirmed edge" (signed18, +$540/session). Verify
  feature-availability-clock parity, not just future-outcome guards; the mutate-future lint did NOT
  catch that leak.
- **Do NOT run an unbudgeted entry-signal search** over the 215 development sessions. With the holdout
  spent, enough hypotheses manufacture a false positive by chance and nothing remains to catch it. If
  you propose search, it must be pre-registered with a hard budget under family maxT correction
  (`v4/research/autoresearch_v2/` already has this machinery, plus a semantic registry that rejects
  repeated mechanics even when renamed).
- **Do not modify** the frozen causal clock, fill/label law, or entry→exit OOF firewall to improve a
  number. If you believe the fill law should change, propose it as an explicitly-flagged counterfactual
  — never silently.
- **No reward-hacking.** A large improvement is grounds for MORE skepticism, not celebration.
  `NO_FIX` is a first-class, respectable answer and may well be the correct one.
- **CLAUDE.md safety:** no broker/order/live/paper-submit/market-data scripts, no paid Databento or
  Polygon downloads, no training / threshold-tuning / promotion, no launchd/plist/runtime-flag edits,
  without explicit owner authorization.

## Data available (read-only)
- `/Volumes/AR_TRADING_DATA/artifacts/entry_v2/oof_scores.parquet` — 156,950 scored candidates:
  `calibrated_prediction`, `entry_feature_bid/ask/mid`, `fill_price`, `exit_bid`, `net_pnl_dollars`,
  `outer_fold`, `strike_offset`, `is_call`, `entry_arrival_ns`, `filled`, `exit_reason`
- `/Volumes/AR_TRADING_DATA/exit_features/session=*/*.parquet` — 1,167 trajectories, full 1-second paths
- `/Volumes/AR_TRADING_DATA/exit_labels/`, `/Volumes/AR_TRADING_DATA/exit_sensitivity_labels/`
- `/Volumes/AR_TRADING_DATA/reports/phase1_four_box/` — `replay.json`, `trajectory_outcomes.parquet`
- Corpus: `/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31` (internal source preserved)

Operational: the SSD must stay mounted. Prefix long jobs with `caffeinate -dimsu`, and run
`storage-preflight` before and after any long stage (the 150 GB cap is checked at stage start only).

## Deliverable
A written recommendation — not code merged to a branch:
1. Which lever (if any) is worth pursuing, with the measured number that justifies it.
2. The counterfactual you measured and the **adverse-selection control** you applied.
3. An explicit verdict: **`FIX_CANDIDATE`** or **`NO_FIX`**.
4. If `FIX_CANDIDATE`: a pre-registered experiment design — hypothesis, hard budget, gates, negative
   controls. A design, not a result.
5. What evidence would falsify your recommendation.

Show your measurements. Reproduce every headline number a second, independent way before reporting it.
Verify — do not rubber-stamp.
````

---

## Accepting or rejecting what comes back

The returned recommendation is acceptable only if all of these hold:

- **Lever A is selection-corrected.** If it claims recovered execution edge, it must report what happened
  to the candidates that would NOT have filled. A number that assumes mid fills is invalid — reject it.
- **Headline numbers reproduce.** Spot-check its claims against
  `/Volumes/AR_TRADING_DATA/artifacts/entry_v2/oof_scores.parquet` independently.
- **Nothing frozen was mutated.** `git diff` must show no change to `FILL_LAW`, the t−60s clock, or the
  OOF firewall in `pathd_phase1_entry.py` / `phase1_exit_model.py` / `pathd_phase1_replay.py`.
- **Firewall untouched** — `holdout_open_count` still 0, no firewall session decoded.
- **A `NO_FIX` verdict is accepted as success**, not sent back for another attempt. That outcome is
  consistent with every measurement taken so far, and pushing past it is how false edges get born.

*Signed: Claude Opus 5 — 2026-08-03.*
