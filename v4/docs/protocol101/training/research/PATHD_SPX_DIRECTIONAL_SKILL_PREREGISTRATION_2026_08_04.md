# Path-D SPX Directional-Skill Screen — PRE-REGISTRATION (2026-08-04)

**Status: FROZEN — pre-registered before execution.**

Authorized by the owner on 2026-08-04 via
[`PATHD_FABLE_REVIEW_IMPLEMENTATION_PLAN_2026_08_04.md`](../execution/PATHD_FABLE_REVIEW_IMPLEMENTATION_PLAN_2026_08_04.md)
(Deliverable 1). Read-only local analysis on owned data: no training, no estimator, no broker, no paid
download, no holdout access.

Nothing below may be edited after the run starts. Amendments are dated appendices that leave the original
intact.

---

## 1. The question

**Do any of the nine side-free SPX context features rank-predict SPX forward returns at all?**

Every Path-D campaign to date has measured a *strategy* — features, an option, an entry law, an exit law,
and friction, all at once. Each returned `NO_EDGE`. That design cannot distinguish two very different
worlds:

- **World A.** The features carry directional information, and the 0DTE long-premium *wrapper* destroys it
  (friction −$26.48 = 4.68% of a $565 premium; the variance risk premium; a convex payoff amputated by a
  degenerate exit).
- **World B.** The features carry no directional information at all, in which case no strategy on any
  instrument built from them can work, and the instrument question is moot.

This screen separates them by deleting the option entirely and measuring the features **directly against
the index**. It is free: owned data, no fit, no purchase.

**It is a screen, not a strategy.** A positive result licenses one thing — asking the instrument question
seriously. A negative result closes directional strategies on this feature set for *every* instrument.

## 2. Family

**Nine features.** The `CONTEXT_FEATURES` tuple from
[`v4/model/protocol101_canonical_stage1_contract.py:31`](../../../../model/protocol101_canonical_stage1_contract.py#L31)
minus the three option-side alignment flags (`vwap_side_alignment_flag`, `omar_side_alignment_flag`,
`momentum15_side_alignment_flag`). Those three are excluded because they are sign indicators of features
already in the family, evaluated against the option's `right` — they carry no information the parent
feature does not, and they are undefined once the option is removed.

| # | Feature |
|---|---|
| 1 | `spx_vwap_gap_points` |
| 2 | `spx_vwap_gap_bps` |
| 3 | `spx_vwap_gap_over_session_range` |
| 4 | `session_range_bps` |
| 5 | `momentum_5m_bps` |
| 6 | `momentum_15m_bps` |
| 7 | `momentum_5m_over_session_range` |
| 8 | `momentum_15m_over_session_range` |
| 9 | `omar_clipped_neg3_pos3` |

**Three horizons:** 15, 30, 60 minutes. **27 feature × horizon members.**

### 2.1 Declared deviation from the plan outline — the family is 54, not 27

The plan outline specified a 27-member family under `session_blocked_max_t`. That statistic is
**one-sided**: it compares each member's observed *signed* `t` against the permutation maximum
(`v4/research/autoresearch_v2/statistics.py:59-65`). A feature whose IC is reliably **negative** —
mean reversion is exactly this, and is an entirely ordinary form of directional skill — would score
`maxT_p ≈ 1.0` and be recorded as no-skill. A screen whose whole purpose is to decide World A vs World B
must not be blind to half the alternative hypothesis.

**Frozen resolution.** The family is declared as **54 members**: each of the 27, plus its sign-mirrored
twin. Under sign-flip permutation the null maximum over a ±-paired family is the null max of `|t|`, so
this is the standard Westfall–Young two-sided family-wise correction, obtained with the existing
statistics module **unmodified**.

This is **strictly more conservative** than the 27-member one-sided test — the family is larger, so every
p-value is weakly larger — while additionally being able to detect negative-sign skill. It cannot loosen
the gate. A member is reported at its better-scoring sign, and the sign is reported alongside.

## 3. Data

- **Corpus:** `/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31`, fallback
  `~/.autoresearch-trading/pathd_2025-08-01_2026-07-31`.
- **Sessions:** the **215** development sessions returned by `development_sessions()`
  ([`pathd_phase1_entry.py:67`](../../../../research/pathd_phase1_entry.py#L67)) — the pre-firewall prefix
  of the frozen 251-session inventory.
- **Protected holdout: never read.** `protected_holdout_opened` stays `false`. The 36-session firewall is
  SPENT and this screen does not touch it.
- **Bars:** official ThetaData SPX 1-minute, loaded by `_official_spx()`
  ([`pathd_phase1_entry.py:204`](../../../../research/pathd_phase1_entry.py#L204)), which enforces the
  provenance asserts (`symbol == SPX`, official index data, not derived, not proxy).

## 4. Clock

**Unchanged from Phase-1, and reused rather than reimplemented.** ThetaData `event_time` is the bar-*open*
stamp, so a bar becomes available at `event_time + 60s`. A feature evaluated at boundary `t` may use only
bars with `available_at ≤ t`, subject to a **90-second staleness cap**
(`_OFFICIAL_SPX_MAX_AGE_NS`, [`pathd_entry_features.py:32`](../../../../research/pathd_entry_features.py#L32)).

Features are produced by the frozen kernel `official_spx_market_window_from_rows` →
`feature_matrix`, invoked through a **degenerate ladder** (`strike_offsets=[0.0]`, `rights=("C","P")`,
option ladder all-NaN, `feature_names=("mid",)`). With an all-NaN ladder the option-dependent branches
(`D.*`, `E.bs.*`) yield NaN and the nine context scalars are unchanged — they are pure functions of the
market window and are identical across every `(strike, right)` cell. This guarantees the screen measures
**exactly** the features the entry model saw, bit for bit, rather than a reimplementation of them.

**Boundaries:** every minute from 10:00 to 15:00 ET inclusive.

## 5. Target

`y_h(t)` = `close_available_at(t + h·60s)` − `close_available_at(t)`, where `close_available_at(·)` applies
the **same** availability rule and 90-second staleness cap as the feature clock. This is the forward move
in the observable price series — the return the feature could not see and a trader could actually have
captured.

`close_available_at` is computed from the same `(available_at_ns, close)` arrays the frozen kernel builds,
and the script **asserts** at every boundary that it reproduces the kernel's `spx_close` exactly. A
boundary whose target endpoint has no bar within the staleness cap is **dropped for that horizon**; drop
counts are recorded in the receipt. Boundary sets therefore differ by horizon (the 60-minute horizon loses
the late-session boundaries), which is expected and is not a defect.

Units are SPX points. Spearman is scale-invariant, so units do not affect any reported statistic.

## 6. Statistic

- **Per session:** Spearman IC between the feature and `y_h` over that session's valid boundaries
  (`scipy.stats.spearmanr`). Sessions with **fewer than 30** valid boundaries are dropped.
- **Per member:** `paired_summary` over the per-session ICs — mean, se, 95% CI, one-sided p.
- **Family:** `session_blocked_max_t` over the 54 declared members, `permutations=20_000`, `seed=2033`.

## 7. Fold stability

Fold map from `entry_expanding_folds(development_sessions(...))`
([`pathd_phase1_entry.py:596`](../../../../research/pathd_phase1_entry.py#L596)) — five expanding folds
with a one-session embargo. Per-fold mean IC is reported per member over each fold's **test** sessions.
The first 44 train-only sessions appear in no test fold; this is the frozen fold map and is not adjusted.

## 8. Negative controls

**(a) Session-shuffled target — the null control.** Within each session the target vector is permuted with
deterministic seed `int(sha256(session).hexdigest()[:16], 16)` (the idiom at
[`pathd_phase1_replay.py:182-184`](../../../../research/pathd_phase1_replay.py#L182)). This destroys the
feature→future link while preserving both marginal distributions and the session block structure. **It must
produce a null family maxT.** If any shuffled member clears the decision rule, the screen is `INVALID` and
its results are discarded.

**(b) Sign-reversed feature — an identity check, not a null control.** IC(−x, y) must equal −IC(x, y)
exactly. This is an implementation-correctness assertion. It is explicitly **not** a falsification control,
because under §2.1 the sign-mirrored member is a declared family member.

## 9. Decision rule — frozen

A member is a **`DIRECTIONAL_SKILL_CANDIDATE`** iff **all** of:

1. `maxT_p_one_sided < 0.05` at its reported sign, over the declared 54-member family;
2. mean IC carries the **same sign in ≥ 4 of 5 folds**;
3. both controls behave as specified in §8.

Otherwise the screen verdict is **`NO_DIRECTIONAL_SKILL`**.

If any session-shuffled member clears rule 1, the verdict is **`INVALID`** and nothing is reported as a
finding.

## 10. What a positive result does NOT license

- **No training.** The Phase-1 closeout bars training on the 0DTE long-premium class, and a candidate here
  does not reopen it.
- **No wave, no capture consumption, no ledger regeneration.** The Phase-0 certification queue stays frozen
  per the Build Order amendment.
- **No confirmation claim.** The protected holdout is SPENT. The only remaining confirmation path for
  anything found here is **fresh live-paper observation**, which is a separate owner-authorized step.

A candidate licenses exactly one thing: taking the instrument question seriously, beginning with the
measured-friction question in the owner decision memo.

**Claude's prior, recorded in advance.** `NO_DIRECTIONAL_SKILL` is more likely than not. These are nine
public, unconditional trend/mean-reversion primitives on the most heavily arbitraged index in the world,
and eight of the nine are near-duplicates of two underlying quantities (VWAP gap and momentum). A
per-session Spearman IC that survives a 54-member family correction would be a genuine surprise and must be
argued for, not celebrated.

## 11. Evidence and reproduction

- **Script:** `v4/research/pathd_spx_directional_skill_screen.py` (no CLI arguments; `main()`).
- **Output:** `v4/audit/autoresearch/pathd_spx_directional_skill_screen_2026_08_04/` — `receipt.json`
  (self-sealed via `stable_hash`), `results.csv` (54 rows), `results.md`.
- **Results doc:** `PATHD_SPX_DIRECTIONAL_SKILL_RESULTS_2026_08_04.md` (this document is not edited).
- The receipt carries `schema_version`, `protected_holdout_opened: false`, `paper_order_submitted: false`,
  the sha256 of the script and both imported feature modules, and the path and sha256 of **this**
  pre-registration.
- The output directory is created with `exist_ok=False`; a second run refuses.

*Signed: Claude Opus 5 — 2026-08-04 — FROZEN before execution.*
