# Protocol101 Round 15: Fable → Codex Handoff

Generated: 2026-07-06 (evening)

Direction: this is a Fable-to-Codex handoff. Round 14 review is complete, the
Oct 2024 → Jun 2025 acceptance campaign has been executed, and this document
specifies the next implementation slice. Everything below was verified against
the working tree as of 2026-07-06 ~17:15 local.

## One-Sentence State

The fair-data foundation is verified and enforced end to end for 169 of 186
target sessions; the only remaining data defect is a single builder-level
issue (quote-driven decision grid) that silently omits untradable minutes, and
fixing it is the last acceptance-phase code slice before fold work.

## What Is Now DONE (do not redo)

### Campaign result (calibrated verifier v3.1, verifier_version 31)

Full chronological acceptance, Oct 2024 → Jun 2025, uniform default
thresholds, monthly registries with recomputable `registry_hash`:

| Month | Pass | Report-only | Fail | Artifact dir (under `v4/audit/autoresearch/`) |
|---|---|---|---|---|
| 2024-10 | 23 | 0 | 0 | `protocol101_owned_raw_acceptance_2024_10_v31_full` |
| 2024-11 | 19 | 1 (11-29 early close) | 0 | `protocol101_owned_raw_acceptance_2024_11_v31_full` |
| 2024-12 | 20 | 1 (12-24 early close) | 0 | `protocol101_owned_raw_acceptance_2024_12_v31_full` |
| 2025-01 | 20 | 0 | 0 | `protocol101_owned_raw_acceptance_2025_01_v31_full` |
| 2025-02 | 19 | 0 | 0 | `protocol101_owned_raw_acceptance_batch_2025_02_v31_full` |
| 2025-03 | 19 | 0 | 2 | `protocol101_owned_raw_acceptance_batch_2025_03_v31_full` |
| 2025-04 | 10 | 0 | 11 | `protocol101_owned_raw_acceptance_batch_2025_04_v31_full` |
| 2025-05 | 19 | 0 | 2 | `protocol101_owned_raw_acceptance_batch_2025_05_v31_full` |
| 2025-06 | 20 | 0 | 0 | `protocol101_owned_raw_acceptance_batch_2025_06_v31_full` |

Totals: 169 pass, 2 report_only, 15 fail, 186 sessions. Feb–Jun artifacts are
batch-runner summaries; the acceptance registry sits in their `acceptance/`
subdirectory. Oct–Jan are direct verifier outputs.

Directory conventions (keep these):

```text
processed:   data/processed/spxw_0dte_neural_protocol101_owned_raw_acceptance_{YYYY_MM}_live_v1/
normalized:  v4/normalized_protocol101_owned_raw_acceptance_{YYYY_MM}_live_v1/
```

### Governance/enforcement changes Fable made after Round 14

1. Protected-holdout owner override narrowed
   (`v4/model/protocol101_governed_loader.py`): the override token now permits
   holdout sessions for role `test` ONLY. Any other role (including `train`)
   gets blocker `protected_holdout_override_only_permits_test_role`. Every
   session validation payload records `protected_holdout_override_used`.
   Tests updated in `v4/tests/test_protocol101_governed_loader.py`.
2. Loader perf: `validate_session_for_role` accepts an optional
   `governance_blockers` argument; `resolve_governed_split_paths` computes the
   registry-hash recomputation once instead of per session.
3. `min_label_finite_share` recalibrated 0.45 → 0.40 (dataclass default + CLI
   default in the verifier, with a calibration comment). Basis: 2024-11-11
   (0.440), 2024-12-10 (0.444), 2024-12-16 (0.442) failed ONLY this check with
   every sweep/spot/context check clean — thin-liquidity holidays populate a
   legitimate tail (corpus range 0.434–0.755) far above the ≤0.05 placeholder
   pathology this floor exists to catch.

Round 14 items verified good as delivered: registry-hash payload complete;
synthetic `_raw_path_label` branch tests cover all five branches plus the
policy-1 15:30 ET boundary; pending-holdout artifact correctly fail-closes all
governed loading.

### Operational facts

- The iCloud dataless-file problem is FIXED at the data level:
  `after_dataless_count = 0` across all six acceptance roots (1,065 files,
  ~658 MB materialized). Tool: `v4/scripts/materialize_protocol101_data_tree.py`
  (`--execute`, per-file subprocess timeout; timeouts still make progress
  because macOS continues the download after the reader is killed — loop until
  `after_dataless_count == 0`).
- Your Round 14 pytest "import collection stall" was NOT a code issue: the
  repo-local `.venv` contained 32,309 cloud-evicted files (232 native `.so`).
  Always use `~/.autoresearch-trading/runtime-venv/bin/python`. The repo
  `.venv` should be deleted or excluded from iCloud sync.
- Post-materialization the whole campaign ran with ZERO build timeouts
  (~30-60s per session build).
- Targeted suite status at handoff: 78 passed —

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python -m pytest -q \
  v4/tests/test_protocol101_owned_raw_acceptance_verifier.py \
  v4/tests/test_protocol101_owned_raw_acceptance_batch.py \
  v4/tests/test_protocol101_governed_loader.py \
  v4/tests/test_protocol101_fair_contract_training_runner.py \
  v4/tests/test_protocol101_era_role_policy.py \
  v4/tests/test_protocol101_session_era_manifest.py \
  v4/tests/test_protocol101_serial_simulator.py \
  v4/tests/test_protocol101_fair_contract_selected_candidate_replay_gate.py
```

## The 15 Failed Sessions — One Root Cause

`v4/dataset/spxw_0dte_neural.py:616` derives the decision grid from data:

```python
decision_times = sorted(options["quote_time"].drop_duplicates())
```

and skips a minute entirely when `_latest_quotes_at` returns empty
(`spxw_0dte_neural.py:657-659`). When every 0DTE quote in a minute fails the
tradability filters (spread > 0.50 abs / 0.25 frac, missing bid, size floors),
that minute is OMITTED from the processed session instead of being emitted as
a no-candidate decision row. The verifier's `expected_decision_row_count` /
`decision_timestamps_match_calendar` checks then fail the session (correct,
fail-closed).

Verified mechanism example (2025-03-19 18:00 UTC, FOMC statement minute): raw
CBBO has 508 records at that minute, 29% with no bid, the rest blown-out wide
→ zero candidates survive → row missing. The raw data is complete; the grid is
not.

The failures:

```text
Single missing macro minute (10:00 ET ISM / 14:00 ET FOMC / similar):
  2025-03-03 (359/360), 2025-03-19 (359/360), 2025-04-01 (359/360),
  2025-04-03 (359/360), 2025-04-04 (359/360), 2025-04-14 (359/360),
  2025-04-29 (359/360), 2025-05-01 (359/360), 2025-05-07 (359/360)

Multi-minute / crisis-week (April 2025 tariff shock, VIX ~50+):
  2025-04-07 (201/360, also label_finite_share + near_atm fails)
  2025-04-08 (354/360, also near_atm fail)
  2025-04-09 (257/360, also label_finite + mean_tradable fails)
  2025-04-10 (333/360, also label_finite + near_atm fails)
  2025-04-11 (342/360, also near_atm + no_leading_backfill fails)
  2025-04-23 (356/360)
```

Why this is load-bearing, not cosmetic: a live system runs every minute and
decides "no tradable candidate" — it never skips. Historical replay must
represent those minutes the same way or replay and live are different games at
exactly the moments that matter. Worse, excluding these sessions builds a
survivorship-shaped corpus: the model would never train or validate on the
crisis days where abstention is the correct behavior.

## The Slice To Implement: calendar-grid decision rows (builder) + mask-aware metrics (verifier v3.2)

### Part A — builder

In `build_neural_dataset` (`v4/dataset/spxw_0dte_neural.py`):

1. Derive `decision_times` from the session calendar, not from quotes:
   09:31 ET through `no_new_entries_after` (15:30 ET) inclusive, one-minute
   steps, early-close aware (use the same early-close source as the verifier's
   `EARLY_CLOSE_TIMES_ET`; for early-close days last decision = close − 1min).
   Timezone handling must match the existing rows (UTC storage, NY calendar).
2. When `_latest_quotes_at` returns empty for a minute, EMIT the row instead
   of `continue`: all-False `candidate_mask`, all-NaN `option_ladder`,
   all-None `contract_ids`, all-NaN `labels_net_pnl`/`labels_mid_pnl`
   (never zeros — zeros are the placeholder pathology the verifier rejects),
   empty `contract_quote_metadata`, with `market_window`, `atm_strike`,
   `source_context_time`, `context_*` fields computed exactly as for normal
   rows (context exists on these minutes; only the ladder is empty).
3. Keep every existing filter and feature computation byte-identical for
   minutes that do have surviving quotes. This change may only ADD rows.
4. Record a builder/grid version marker in each row or the build summary
   (e.g. `decision_grid: "calendar_v2"`), so registries can distinguish
   grid semantics.

Control test (mandatory): rebuild 2-3 already-accepted sessions (e.g.
2024-10-01, 2025-01-15) into a scratch dir and assert row-for-row equality of
decision_times, candidate_mask, option_ladder, labels, and market_window
against the accepted pickles. The fix must be provably additive.

Unit tests: a synthetic day where one mid-session minute has only
filter-failing quotes must produce a full grid with an empty-mask row at that
minute; an early-close day must produce the shortened grid.

### Part B — verifier (bump to v3.2 / verifier_version 32)

The new empty rows change metric denominators; make the label metrics
mask-aware so honest no-candidate minutes are not penalized:

1. `label_finite_share` (and nonzero/pos/neg shares): compute over label cells
   where `candidate_mask` is True, not over all cells. Re-derive the floors
   from the Oct–Jun corpus after the change (expect the masked-only finite
   share to sit much higher and tighter; set the floor accordingly and update
   the calibration comment).
2. Keep `expected_decision_row_count` exact (tolerance 0) — with the calendar
   grid there is no excuse for a missing row.
3. `tradable_minute_share` (0.50) and `near_atm_tradable_share` (0.50) become
   honest liquidity descriptors for crisis sessions. Decision for the owner
   (present it, do not bury it): a session that passes every integrity check
   but sits below a liquidity floor should probably become
   `report_only` with reason `low_tradable_liquidity` (mirroring early-close
   handling) rather than `fail`, so "broken data" and "honest thin market"
   stay distinguishable in the registry. 2025-04-07 (~0.56 tradable-minute
   share after the fix) is the test case.
4. `MIN_FOLD_PLACEMENT_VERIFIER_VERSION` → 32 so the loader rejects v3.1
   registries once v3.2 exists, forcing the re-acceptance below.

### Part C — rebuild and re-accept

1. Rebuild ONLY the 15 failed sessions with the fixed builder (their monthly
   processed dirs; `--skip-existing` will skip the good ones since their
   pickles already exist — pass the failed session dates explicitly as
   single-day builds via the batch runner to be precise).
2. Rerun acceptance for all nine months under v3.2 so the whole corpus shares
   one verifier version and one thresholds vector (cheap: ~1-2 min per month;
   use the same out-dir naming with `_v32_full`).
3. Expected end state: 186 sessions = pass or report_only(reason), zero fails.
   If any session still fails, report it as a finding, do not tune it away.

## After This Slice (not yours to start without owner sign-off)

1. Owner declares protected-holdout sessions
   (`v4/scripts/build_protocol101_protected_holdout_artifact.py --session ...`);
   until then the governed loader blocks all training by design.
2. Commit everything — verifier, loader, batch runner, materializer, holdout
   builder, tests, monthly registries are ALL still untracked. This needs
   explicit owner authorization per repo rules, but flag it every round until
   it happens: the hash chain has no version history.
3. Move the data tree (and delete/relocate the repo `.venv`) out of iCloud
   sync, or eviction will eventually recur.
4. Then fold-scaffold phase: first diagnostics folds and null/permutation
   canaries through the governed loader (`resolve_governed_split_paths`),
   implementing the promotion guard (`regime_bound_requires_owner_review`)
   as code in the first fold harness.

## Hard Guardrails (unchanged)

- No model training, threshold tuning beyond the specified recalibration,
  promotion, paper-submit, broker contact, or paid downloads.
- No weakening of acceptance checks to make sessions pass; every threshold
  change needs corpus evidence recorded next to the default.
- Use `~/.autoresearch-trading/runtime-venv/bin/python` for everything.
- Run the 78-test targeted suite plus your new tests before handing back.

## What To Return To Fable

1. The builder diff + control-test evidence (row-for-row equality on accepted
   sessions).
2. v3.2 verifier diff with re-derived mask-aware floors and their calibration
   basis.
3. The nine `_v32_full` month registries with the final pass/report_only
   table.
4. Any session that still fails v3.2, with its failing checks — unexplained
   failures are findings, not annoyances.
