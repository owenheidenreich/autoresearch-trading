# FT2 Delta-Scoped Review Attempt 002 — Seat 1 Trading Realism

- Role: fresh independent trading-realism reviewer
- Scope authority: `protocol101_ft2_20_delta_scoped_review_fixes_attempt001/diff_manifest.json`
- Diff-manifest SHA-256: `2aa1ff2d307d9eccdd483484cde882dee230cc2703869b05dc8561eb41f18ee7`
- Changed scope: 41 files, limited to the listed files/JSON pointers and their direct consequences
- Verdict: **`DELTA_REVIEW_CLEAN`**
- Next: **`STOP_FOR_OWNER_DECISION`**

## Conclusion

No in-scope blocking defect remains in the bounded repair.

`FT2-DELTA-B1` is genuinely fixed. The non-P5 oracle and quality selectors now
construct the causal time-`t` universe, rank and commit one exact identity, and
only then recheck that selected row at `t+1`. A failed recheck records a
zero-charge event, opens no position, permits no same-minute substitute, and
prevents another decision before `t+2`.

The planted two-contract case independently reproduced the required result:

```text
selected identity          future-best-rejects
trade_count                0
pooled_pnl                 0
rejected_fill_count        1
position_opened            false
premium_charged_cents      0
fee_charged_cents          0
substitute                 none
```

Across the regenerated serial evidence there are 622 rejection events and
31,329 trades. Every rejection resolves exactly at `t+1`, has no position,
premium, fee, or realized-PnL charge, has zero overlap with a trade for the same
policy minute, and has no subsequent trade before `t+2`. All nine oracle
variants now expose positive rejection counts; their total is 375. The
regenerated oracle total is 13,669 trades and `$5,673,928` pooled PnL. P5
remains 17,660 trades and `$1,329,305` pooled PnL, with zero P5 trade-count or
PnL delta in the regenerated v3/v4 impact comparison.

## Blocking-finding dispositions

### FT2-DELTA-B1 — fixed

Static inspection and the targeted synthetic replay agree. Future `t+1`
recheck results are no longer used to prefilter all candidates before action
selection. Only the committed identity is rechecked. Rejection evidence,
serial occupancy, and retry timing reconcile with the canonical no-substitute
law.

### FT2-DELTA-B2 — fixed

The active builder pins authority
`d115b953d8959fe777923ca5c1e375246754a181847ae77b57d37d24f0a279ca`
and passes its authority verifier. The current semantic `{path, SHA-256}` scan
has zero mismatches. The FT2-04/05/08/10/11 receipts reproduce 157 of 157
declared deliverables and all pin the current authority. FT2-08/10/11 pin the
actual rerun002 receipt bytes at
`17cc9983c38ef4c75b8fece0cd67d4662e559f84a27de8eba2a84a34f85d96ac`.

The FT2-05 repair points to the preserved V4 receipt at commit
`a7602fdcce541589440b4aa2bc0bd0e2be6d1bbd`; its raw SHA-256 reproduces as
`ffbef1058afb26392de0f1d3efcfba911fca619cf9bd6055fae756d20b64397d`.
All 90 prior receipt-declared checkpoint label and summary artifacts still
match that seal. This independently supports the checkpoint-only regeneration
lineage without reopening source data.

### FT2-DELTA-B3 — fixed

`forecast_heads.json` and `calibration_spec.json` import the same exact RLAC
target at `/targets/WAIT_head`, pin the current RLAC SHA-256, use
`y_wait(t)=1 if WAIT_justified(t) else 0`, forbid a scalar `U_label` threshold,
and consistently exclude and count zero-intent-eligible and label-incomplete
minutes. The synthetic one-way RLAC test remains 5/5.

### FT2-DELTA-D1 — fixed

The friction table now distinguishes all relevant populations. Its
premium-band totals reconcile exactly:

| Population | Rows |
|---|---:|
| Time-`t` intent | 128,758 |
| Successful `t+1` fill recheck | 121,553 |
| Rejected `t+1` fill recheck | 7,205 |
| Finite friction | 121,553 |
| Finite decision-time spread | 128,758 |

Every table row satisfies intent = fill-pass + rejected,
fill-pass = friction-valid, and spread-valid = intent. The governed rejection
rate remains `7,205 / 460,937 = 1.563120%`; the conditional recheck-population
rate remains `7,205 / 128,758 = 5.595769%`.

### FT2-DELTA-D2 — documentation-only, unchanged

The duplicated historical authority amendment identifier `A4` remains the
previously disclosed documentation-only observation. The authority was a
pinned immutable input to this bounded mechanical repair. This does not reopen
or block the repaired delta.

## Reproduced gates

| Gate | Independent result |
|---|---|
| All diff-manifest after-hashes | 41/41 exact |
| Bounded-fix checker | PASS, 20/20 |
| Targeted FT2-05 pytest | PASS, 12/12 |
| T1 synthetic RLAC | PASS, 5/5 |
| T3/T4 census reconciliation | PASS, 11/11 |
| T6 private-copy regression | PASS, 4/4; FT2-08 22/22 |
| Graph topology | PASS, 47 nodes / 104 edges / 47 reachable |
| Current semantic path/hash scan | PASS, 0 mismatches |
| Packet deliverables | PASS, 157/157 |
| Preserved FT2-05 checkpoints | PASS, 90/90 |

T6 used a private FT2-08 copy because its validator writes `validation.json`.
The reviewed validation artifact remained unchanged at
`d0fd65b55ed38d88d82541e7a2fe1ecec3903b0bb61c6bc41e3c24368cd5fd6e`.

## Limits and routing

This verdict accepts only the manifest-bounded repair. It is not evidence of
profitability, model selection, protected-data readiness, live readiness,
paper authorization, or approval to advance to FT2-21.

All forbidden side effects are false. No broker, recorder, paid-data, live,
protected, outer, or holdout path was accessed; no real model was trained or
fit; simulator v5, runtime, paper default, promotion state, launchd, and orders
were not modified. Durable writes are confined to this Seat 1 directory.

**Route: `DELTA_REVIEW_CLEAN` → `STOP_FOR_OWNER_DECISION`. Do not approve or
route FT2-21.**
