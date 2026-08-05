# Path-D Fable-Review Implementation Plan — 2026-08-04

**Status: OWNER-APPROVED EXECUTION PLAN — hand to the executing agent (Opus 4.8), budget ~15 minutes.**

Prepared by Claude Fable 5 from the 2026-08-04 outside review
([`FABLE_REVIEW_BRIEF_2026_08_04.md`](../research/FABLE_REVIEW_BRIEF_2026_08_04.md)).
Owner approved this plan and its two embedded decisions on 2026-08-04. Execute the five deliverables
in order. Every file path, reuse target, and frozen parameter below was verified against the repo and
the mounted SSD on 2026-08-04 — do not re-derive them, and do not expand scope.

---

## Context

The Fable review reached a **hard-narrowing** verdict, not STOP:

- The 0DTE-long-premium class is settled dead (−$13.00/trade gross, 5/5 folds negative; the variance
  risk premium was rediscovered empirically). Re-derivations must stop — enforce closure in the
  do-not-retest ledger.
- The 65-feature certification queue services a training campaign that is barred by the Phase-1
  closeout. Freeze the queue; keep the ledger machinery and enforcement.
- The one free decisive experiment has never run: **do the 9 SPX context features predict SPX forward
  returns at all** (15/30/60 min), measured directly on the index with no option noise? If no → no
  directional strategy on any instrument has evidence; if yes → the feasibility study already says
  where it clears friction (ES 15–60 min).
- The owner then needs a one-page decision memo keyed on the screen's outcome.

**Owner decisions already made (2026-08-04):** (1) stand down the 08-05..07 capture now — this plan
includes the owner-run uninstall block; (2) the SPX screen is authorized to be pre-registered AND run
in the same session (read-only local analysis on owned data; no training, no broker, no paid data).

**Hard boundaries for the executing agent:** no model training (Spearman rank stats only — no
estimator, no `fit`), no broker/IBKR contact, no paid downloads, no launchd mutation (owner-only), no
ledger regeneration, no holdout access (`protected_holdout_opened` stays false everywhere), no
deletion or rewrite of existing receipts/docs (amendment banners only).

---

## Deliverable 1 — Pre-registration doc (write + commit BEFORE running anything)

**Path:** `v4/docs/protocol101/training/research/PATHD_SPX_DIRECTIONAL_SKILL_PREREGISTRATION_2026_08_04.md`

House style: dated `#` title; bold status line (`**Status: FROZEN — pre-registered before
execution.**`); verdict tokens in backticks; evidence/reproduction section; italic signature footer.
Model it on `PATHD_WAVE3_DELTA_UNIVERSE_PREREGISTRATION_2026_08_04.md` (same directory).

Freeze exactly this spec:

- **Question.** Does any of the 9 side-free SPX context features rank-predict SPX forward returns at
  15/30/60 min under the causal t−60s clock, family-wise corrected?
- **Family.** 9 features × 3 horizons = **27 members**. The 9 = `CONTEXT_FEATURES` minus the 3
  option-side alignment flags (they are functions of option right, i.e. sign indicators of features
  already in the family): `spx_vwap_gap_points`, `spx_vwap_gap_bps`,
  `spx_vwap_gap_over_session_range`, `session_range_bps`, `momentum_5m_bps`, `momentum_15m_bps`,
  `momentum_5m_over_session_range`, `momentum_15m_over_session_range`, `omar_clipped_neg3_pos3`.
- **Data.** The 215 dev sessions only (`development_sessions()` prefix rule). Protected holdout:
  never read.
- **Clock.** Feature at boundary `t` uses SPX bars with `available_at = event_time + 60s ≤ t`, 90 s
  staleness cap — i.e. the exact reused pipeline code, no reimplementation.
- **Target.** `y_h(t)` = (SPX close available at `t + h·60s`) − (SPX close available at `t`), same
  availability rule — the return the feature could not see. Boundaries: every minute 10:00–15:00 ET.
- **Statistic.** Per-session Spearman IC(feature, y_h) over that session's boundaries; sessions with
  <30 valid boundaries dropped. Aggregation: `paired_summary` per member + `session_blocked_max_t`
  over the 27-member family, `permutations=20_000`, `seed=2033`.
- **Fold stability.** Fold map from `entry_expanding_folds(development_sessions(...))`; report
  per-fold mean IC per member.
- **Negative controls.** (a) session-shuffled target with deterministic seed
  `int(sha256(session)[:16], 16)` (idiom: `pathd_phase1_replay.py:373-379`) — must produce null maxT;
  (b) sign-reversed feature — IC must equal −real.
- **Decision rule (frozen).** A member is a `DIRECTIONAL_SKILL_CANDIDATE` iff
  `maxT_p_one_sided < 0.05` AND mean IC has the same sign in ≥4/5 folds AND both controls behave.
  Otherwise the screen verdict is `NO_DIRECTIONAL_SKILL`.
- **What a positive does NOT license.** No training, no wave, no capture consumption. Holdout is
  SPENT → any candidate's confirmation path is fresh live-paper observation only, per closeout.

---

## Deliverable 2 — Screen script + run

**Path:** `v4/research/pathd_spx_directional_skill_screen.py` (~250 lines, `main()`, no CLI args)

Shape: the `feasibility_gross_expectancy.py` skeleton (question-stating docstring, frozen constants
up top, one loader, one accumulator, one render block) **plus** the `pathd_phase1_replay.py:334-354`
receipt idiom. Fail closed — no `except Exception: continue`.

**Reuse (do not reimplement):**

| Piece | Source |
|---|---|
| Corpus root + fallback | `/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31`, fallback `~/.autoresearch-trading/pathd_2025-08-01_2026-07-31` |
| Session list (215 dev) | `development_sessions()` — `v4/research/pathd_phase1_entry.py:67-81` |
| SPX bar loader | `_official_spx(path, session)` — `pathd_phase1_entry.py:204-238` (handles provenance asserts + `context_source` normalization) |
| Feature computation | `official_spx_market_window_from_rows` — `v4/research/pathd_entry_features.py:101` → `feature_matrix` with the **degenerate ladder** (`strike_offsets=[0.0]`, `rights=("C","P")`, ladder all-NaN, `feature_names=("mid",)`) — `v4/model/protocol101_canonical_stage1_contract.py:215-297`; keep only the 9 side-free scalars |
| Fold map | `entry_expanding_folds` — `pathd_phase1_entry.py:596-615` |
| Stats | `from v4.research.autoresearch_v2.statistics import paired_summary, session_blocked_max_t` (import the submodule directly; zero project deps) |
| Rank corr | `scipy.stats.spearmanr` |
| Hash/receipt | `stable_hash` — `v4/research/phase1_exit_model.py:94-98`; self-seal `payload["receipt_sha256"] = stable_hash(payload)`; scrub NaN before hashing (`allow_nan=False`) |

**Typing gotchas (verified — will bite otherwise):** `session` must be exactly `str` and
`decision_time_ns` exactly `int` (not `np.int64`) into `official_spx_market_window_from_rows`;
`_official_spx` already normalizes `context_source` to the literal
`"thetadata_index_history_ohlc"`.

**Guards in code:** assert session list length == 215 and equals the `development_sessions` prefix;
refuse to run if output dir exists (`mkdir(parents=True, exist_ok=False)`); receipt carries
`protected_holdout_opened: false`, `paper_order_submitted: false`, `source_hashes` (sha256 of this
script + the two imported feature modules), `schema_version: "pathd.spx-directional-skill-screen.v1"`,
and the prereg doc's path + sha256.

**Output:** `v4/audit/autoresearch/pathd_spx_directional_skill_screen_2026_08_04/` → `receipt.json`
(self-sealed), `results.csv` (27 rows × {mean IC, se, ci, p, maxT_p, per-fold means, control
values}), `results.md` (fixed-width table + verdict line).

**Results doc:** short
`v4/docs/protocol101/training/research/PATHD_SPX_DIRECTIONAL_SKILL_RESULTS_2026_08_04.md` — verdict
token in the status line, headline numbers, evidence paths. Do NOT edit the frozen prereg doc.

## Deliverable 3 — Do-not-retest ledger row

**Path:** `v4/docs/protocol101/training/history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md`, append
one long-form row to the §4 table (match the Path-D rows at lines 177–179):

- **Col 1:** `Path-D 0DTE long-premium class: buying SPXW 0DTE premium at minute cadence — ANY
  features, model, exit policy, or execution style` (**CLOSED, structural**).
- **Col 2:** `NO_EDGE` — 156,950 OOF candidates / 215 sessions / 5 folds. Gross directional
  expectancy **−$13.00/trade with all friction removed**, negative 5/5 folds
  (−18.59/−14.00/−11.63/−9.26/−11.33); instant round-trip friction −$26.48 = 4.68% of $565 avg
  premium; deciles flat & non-monotonic (middle best); no subpopulation positive net (every
  hour/side/moneyness/fold); best window 10:30–10:59 ET +$5.99 gross → −$20.08 net; passive-fill
  recovery +$22.76 UPPER BOUND → still 0/5 folds profitable.
- **Col 3:** genuinely new = a different **position structure** (defined-risk short premium, longer
  tenor — new data, new charter), a different **instrument** (ES 15–60 min, hurdle 52–54%), or
  **demonstrated directional skill on the underlying** (Deliverable 2). Explicitly not new: features,
  seeds, thresholds, exit policies, or execution tweaks on this class. The 10:30–10:59 effect is
  recorded as **loss-shaping only** (1-of-~13 windows, 4× short of friction) — never as an edge.

## Deliverable 4 — Build Order amendment (freeze the queue)

**Path:** `v4/docs/protocol101/training/contracts/PATHD_BUILD_ORDER_AMENDMENT_2026_08_04.md` — model
on `PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md` (Status /
Effective date / Authorization source / supersession paragraph). **Status: PROPOSED — REQUIRES OWNER
SIGN-OFF** (the executing agent must not self-authorize).

Content:

1. Phase-0 certification queue (the 65 `missing_required_receipts`/`parent_family_not_admitted`
   features) is **FROZEN**: no certification work, no ledger regeneration, no wave re-freezing, no
   capture consumption, until an owner-authorized strategy class with measured clearable friction
   exists. The ledger file and `admitted_feature_matrix` enforcement stay exactly as-is (the ledger
   vocabulary is binary; the freeze lives here, not in the ledger).
2. The 08-05..07 capture is **stood down by owner decision (2026-08-04)**. Include the owner-run
   uninstall block verbatim:

   ```bash
   launchctl bootout gui/$(id -u)/com.autoresearch.tracka.capture
   launchctl bootout gui/$(id -u)/com.autoresearch.tracka.capture.midday
   rm ~/Library/LaunchAgents/com.autoresearch.tracka.capture.plist
   rm ~/Library/LaunchAgents/com.autoresearch.tracka.capture.midday.plist
   ```

   The executing agent does NOT run this — it goes in the doc for the owner, flagged **run before
   09:28 ET 2026-08-05**.
3. Add a `⚠` addendum banner (not a rewrite) to
   `PATHD_PHASE1_TO_LIVE_ROADMAP_AND_STATUS_2026_08_03.md:88` noting the stale 5-session capture
   declaration was superseded by v5 (3 sessions) and is now stood down entirely.

## Deliverable 5 — Owner decision memo

**Path:** `v4/docs/protocol101/training/contracts/PATHD_NEXT_CLASS_OWNER_DECISION_2026_08_04.md`
(binding decisions live in `contracts/`). One page. Fill in the screen verdict from Deliverable 2
before writing.

Structure: the screen result up top, then three options with cost/prerequisite/what-it-buys,
**presented, not decided**:

- **A — Confirm ES friction** from GLBX quote data (Codex probe: bbo-1s ≈ $0.073/session — paid
  data, owner authorization required; check existing entitlements FIRST). Relevant only if the screen
  found skill (ES is the only measured instrument whose hurdle, 52–54% @15–60 min, is clearable).
- **B — Longer-tenor options data purchase** (HARD STOP; not recommended until A or the screen
  resolves; the variance risk premium means long premium at any tenor still needs directional or
  vol-timing alpha).
- **C — Stand down research**; keep the execution plane (proven, strategy-agnostic) warm at zero
  cost.
- Decision key: `NO_DIRECTIONAL_SKILL` → directional strategies on ANY instrument have no evidence →
  C, or a non-directional class as a new program. Skill candidate(s) → A before any purchase or
  training question.

## Execution order (15 min) + commits

1. Deliverable 1 → **commit** (`research(path_d): pre-register SPX directional-skill screen`).
   Prereg must be committed before the run.
2. Deliverable 2: write script → run once → verify receipt self-hash recomputes → write results doc.
3. Deliverables 3, 4, 5 (memo consumes the verdict).
4. **Commit** the rest (`research(path_d): SPX skill screen result + queue freeze proposal + class
   closure row`). Follow repo commit-message style; end with the Claude Co-Authored-By line.

## Verification

- Receipt: pop `receipt_sha256`, recompute `stable_hash`, must match. Re-running the script must
  refuse (`exist_ok=False`).
- Controls: session-shuffled maxT p must be non-significant; sign-reversed IC must equal −real IC
  exactly.
- Isolation: grep the new script — no `fit(`, no `sklearn` estimator use, no `ibkr`, no `databento`
  client import, no holdout path; session count printed = 215.
- Repo hygiene: only the 6 new/edited files in the commits; no existing receipt or frozen doc bytes
  changed (amendment banners are appends); `git status` shows the pre-existing dirty files untouched.

*Prepared by Claude Fable 5 — 2026-08-04. Owner-approved same day.*
