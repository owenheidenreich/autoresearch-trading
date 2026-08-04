# Path-D Data Acquisition + Processing Plan

**STATUS: PLANNING — owner-approved. Contains the Codex download goal.** Prepared
2026-08-01. Scope: acquire + process + validate the 12-month Path-D corpus to reach the
starting line of the entry+exit model plan. No model training, no live/broker, no
governance contracts.

## Why

Path D is committed (train Databento+ThetaData / decide Databento-live-OPRA / execute IBKR).
The OPRA Standard subscription is purchased and verified ($0 for trailing-12-months). We now
need the full 12-month corpus — OPRA options + official SPX/VIX context + ES/VX futures
context — acquired cheaply with **zero accidental pay-as-you-go**, processed into the aligned
corpus, and analyzed/tested so it demonstrably feeds the model pipeline. That is the starting
line for planning the entry and exit models.

## Owner directives (binding)
1. **Check the subscription first**; download only free-under-sub by default.
2. **No pay-as-you-go by default** — EXCEPT auto-approve the small ES/VX futures context
   (~$9). Any other paid item (older-than-12mo L1, `cmbp-1`, etc.) STOPS for authorization.
3. **Acquire all ThetaData + associated market data the project uses.**
4. **Analyze + test** the data after download/processing.
5. **End at the starting line** of the entry+exit model plan.

## Facts (reuse existing tooling; do not reinvent)
- OPRA Standard: L0 (definition/ohlcv-1m/statistics) free all-history; L1 (cbbo-1s/cbbo-1m)
  free trailing-12mo; older-L1 + ES/VX = pay-as-you-go. `cmbp-1` ≈ 300 GB → DEFERRED (Phase 2).
  `tcbbo` = unused → drop.
- ThetaData downloader exists: `v4/scripts/download_thetadata_index_bars.py` (`--auth-check`,
  `--dry-run`, paid-data guard, out `data/vendor/thetadata/index/{spx,vix}_1m`). **429 SPX +
  439 VIX days already on disk from 2024-10-01** → gap-fill only.
- Databento: `download_databento_cbbo_1s_audit.py`, `run_protocol101_walking_skeleton_clean_1s_acquisition.py`
  (0DTE symbol resolution), `download_databento_context_proxies.py` (ES/VX futures).
- Process: `v4/scripts/build_databento_neural_dataset.py --context-mode official` +
  `v4/ingest/{databento_opra,index_bars,derived_context}.py` → `v4/normalized*`.
- Safety/validate: `v4/checks/paid_data_guard.py`, `v4/tests/test_paid_data_guard.py`,
  and `v4/scripts/run_pathd_exit_label_smoke.py` (proves the corpus feeds the model pipeline).

---

## CODEX GOAL (copy-paste)

```text
Goal: PATH-D 12-MONTH DATA ACQUISITION + PROCESSING + VALIDATION. Reuse existing tooling;
do NOT reinvent. Ends STOP_FOR_CLAUDE_VERIFICATION. Constraints: NO cmbp-1 (Phase 2), NO
tcbbo, NO model training beyond the smoke-pipeline scale check, NO live/broker, NO governance
contracts, NO runtime/registry/launchd change, do NOT modify Protocol160/158 or the parent
authority, never print/commit the v4/.env secret. Quarantine outputs under
v4/audit/autoresearch/protocol101_pathd_data_acquisition/.

SPEND POLICY (owner-set): free-under-subscription only by DEFAULT; AUTO-APPROVE ES+VX futures
context (~$9 total); for ANY other paid item (older-than-12mo L1, cmbp-1, ...) cost-estimate
via get_cost and STOP for owner authorization. Route paid calls through v4/checks/paid_data_guard.py.

PHASE A - Subscription + auth safety:
- Databento: get_cost a recent cbbo-1s AND cbbo-1m day -> assert $0 (sub active); abort if not.
- ThetaData: python -m v4.scripts.download_thetadata_index_bars --auth-check (confirm creds/sub).

PHASE B - Acquire (window = trailing 12 months, ~2025-08-01 -> present; each schema a single
contiguous range; parquet only, drop DBN.zst; land OUTSIDE iCloud - confirm data/ & v4/raw are
iCloud-excluded or land under ~/.autoresearch-trading/):
- Databento OPRA SPXW, FREE under sub: definition, ohlcv-1m, statistics (L0, all-history OK),
  cbbo-1m, cbbo-1s (L1, trailing-12mo). Resolve per-day SPXW PM 0DTE raw symbols from the
  definition schema. NOT cmbp-1, NOT tcbbo. (~4.2 GB.)
- Databento ES (GLBX.MDP3) + VX (XCBF.PITCH) ohlcv-1m futures context (~$9, auto-approved) via
  download_databento_context_proxies.py (is_proxy=True).
- ThetaData SPX+VIX official 1m: download_thetadata_index_bars.py for the window, SKIPPING days
  already on disk (fill gaps to present only).
- Record vendor-availability boundaries as EXPECTED (cbbo-1s>=2025-02-20, VX>=2026-04,
  ThetaData>=2024-10) - gaps before these are not errors.

PHASE C - Process:
- build_databento_neural_dataset.py --context-mode official
  --official-spx-dir data/vendor/thetadata/index/spx_1m --official-vix-dir .../vix_1m
  over the window -> the aligned canonical corpus (v4/normalized*), producing BOTH the minute
  entry substrate AND the 1-second exit substrate (cbbo-1s). Use derived_context as a
  cross-check where official is missing.

PHASE D - Analyze + test:
- Data-quality report: per-schema row counts, contiguity, option<->SPX/VIX pairing coverage,
  gap analysis (only known vendor boundaries), integrity hashes, acquisition manifest, and a
  SPEND LEDGER (free vs paid; total paid <= authorized ~$9).
- Run v4/tests/test_paid_data_guard.py (no accidental-spend path).
- Re-run v4/scripts/run_pathd_exit_label_smoke.py on the FULL 12-month corpus -> confirm it
  scales, the non-oracle (a_hold <= oracle) + no-leakage checks still pass, and emit the
  12-month a_hold distribution (proves the corpus feeds the model pipeline end-to-end).

PHASE E - Starting line:
- Handoff summary: available data + exact date windows per substrate; the spend ledger; the
  aligned-corpus locations; and the open inputs for the entry+exit model plan. STOP for Claude
  verification + owner review.

Highest allowed claim: "the 12-month Path-D corpus is acquired, processed, and validated; it
feeds the exit-label pipeline; total paid spend within authorization; ready for entry+exit
model planning." No model quality/alpha/promotion claim.
```

## Verification (Claude, after Codex returns)
1. Subscription free-check ($0 trailing-12mo) + ThetaData `--auth-check` pass.
2. Spend ledger: total paid ≤ authorized (~$9 ES/VX); no `cmbp-1`/`tcbbo`/older-L1; `test_paid_data_guard` green.
3. Data-quality: contiguous per-schema coverage; option↔SPX/VIX pairing present; only known vendor-boundary gaps.
4. `run_pathd_exit_label_smoke.py` at 12-month scale: non-oracle holds, no leakage, sane `a_hold`.
5. Reproduce a sample day's free-cost; spot-check contiguity + pairing; confirm no secret/data committed + outside-iCloud landing → owner review → proceed to the entry+exit model plan.
