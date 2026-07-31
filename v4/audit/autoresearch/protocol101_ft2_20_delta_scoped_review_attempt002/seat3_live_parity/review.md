# FT2 Delta-Scoped Fresh-Seat Review — Seat 3 Live Parity

- Goal: `FT2-DELTA-SCOPED-FRESH-SEAT-REVIEW`
- Attempt: `002`
- Seat: `seat3_live_parity`
- Baseline commit: `a7602fdcce541589440b4aa2bc0bd0e2be6d1bbd`
- Scope authority: `v4/audit/autoresearch/protocol101_ft2_20_delta_scoped_review_fixes_attempt001/diff_manifest.json`
- Diff-manifest SHA-256: `2aa1ff2d307d9eccdd483484cde882dee230cc2703869b05dc8561eb41f18ee7`
- Scope: exactly 41 manifest-listed files/JSON pointers and their direct authority, receipt, semantic-lineage, graph, test, and live-game consequences
- Other-seat isolation: no other attempt-002 seat directory was read
- Seat verdict: **`DELTA_REVIEW_CLEAN`**
- Required next state: **`STOP_FOR_OWNER_DECISION`**

## Executive verdict

No in-scope blocking defect remains for Seat 3.

The bounded repair closes the two prior live-parity blockers:

1. The historical oracle and quality selectors now commit one exact time-`t`
   identity before the selected-only `t+1` fill recheck. A rejected action
   opens no position, charges no premium or fee, creates an observable
   rejection event, permits no same-minute substitute, and cannot retry before
   `t+2`.
2. The authority and semantic lineage is re-sealed end to end. The active
   builder, FT2-04/05/08/10/11 receipts, all 157 receipt-declared
   deliverables, current nested path/hash consumers, source-transfer contract,
   action-calibration contract, graph, and raw rerun002 receipt reproduce the
   bytes they claim.

This is a delta-review acceptance only. It does not establish historical/IBKR
parity, authorize live observation, approve FT2-21, authorize training, or
authorize paper orders.

## FT2-DELTA-B2 authority and lineage

### Authority and executable builder

The consolidated authority hashes to
`d115b953d8959fe777923ca5c1e375246754a181847ae77b57d37d24f0a279ca`.
The active FT2-05 builder now pins that exact hash; the superseded
`3c7a0aaf...` value is absent from the active builder. Its
`verify_authority_and_inputs()` check passes against the current authority and
FT2-04 receipt.

The graph hashes to
`35859a40747ebbd75cbb222345b24f45c23beff80a740fb45170b894b01581e9`.
Both labeled graph references in the authority resolve to that exact value.

### Receipt and deliverable chain

| Packet | Receipt SHA-256 | Deliverables | Mismatches |
|---|---|---:|---:|
| FT2-04 | `b4b29dcf9b96b0172adb614bf0aa975ab1c86ef9c92ba2d91423a14c4b52cffe` | 7 | 0 |
| FT2-05 | `ecaa16c923d4834c314816bca471597e09b9668ec5f1fe76d956be9c9d97d168` | 117 | 0 |
| FT2-08 | `c4c9c729c8d19ae76f827ed280f66e7ee1143f7f51fd5b753615d6d968adb726` | 14 | 0 |
| FT2-10 | `b0504ea7d78c041ed066b238229fd5bdda5584da2d1a0854adeefba91b63a3a2` | 9 | 0 |
| FT2-11 | `3cb4693185bc2d1c77c5ecea38817766822b684b33679ba021c0446fade1534c` | 10 | 0 |

All five receipts pin the current authority. FT2-08/10/11 also pin the current
FT2-05 receipt and the exact immutable graph, FT2-04 receipt, canonical
intent/fill law, and rerun002 receipt.

All 41 manifest `after_sha256` values reproduce with zero mismatches. The
producer receipt hashes to
`f7c01162f8637b70e5d0492495ce870a6172d5f91713478b345e47d70d1bf532`.
The producer aggregate receipt raw file hashes to
`4532e1829b8cb99320de474fe96b204b3c99ab85bd519bb49c397a49a266d4db`;
its canonical self-hash independently reproduces as
`f7ae2a4aa768fa441220fad2945e9a7d712a018978e1fe6bd041a477046e8c02`.

### Current semantic path/hash consumers

The formerly stale current references now resolve exactly:

| Source target | Current SHA-256 | Direct current consumers |
|---|---|---|
| RLAC specification | `adcbec44bc254fe9a968562df77ca6a5736aa5d39d9f8e69dcf477a378df4f09` | `forecast_heads.json`, `calibration_spec.json` |
| Calibration specification | `22837145aa2744760ff2ad34c78a4db2998e9946e875118aafa3467f0c6f2e57` | `evidence_standard.json`, `shadow_sufficiency_spec.json` |
| FT2-05 census results | `d30382a868bd1939aec7cc3edc01bfc0e8a5b1a3920a96b6bad9e7288c277715` | `composer_spec.json`, `mde_spec.json` |
| FT2-05 guardrail rates | `6fdd5e75e94e4eb97f60584efabb5263cd79d32648ebbd6418be0f5a1c2b3a3b` | `composer_spec.json`, `calibration_spec.json`, `mde_spec.json` |
| FT2-05 v3/v4 impact | `cd5ec600ea7a5e4870cb4b46a23a391f77f3d2eb6404d65dbc0d284b7242fe2d` | `composer_spec.json`, `mde_spec.json` |
| FT2-05 receipt | `ecaa16c923d4834c314816bca471597e09b9668ec5f1fe76d956be9c9d97d168` | FT2-08/10/11 receipts, `composer_spec.json`, `mde_spec.json` |

The generic current non-superseded path/hash scan reports zero mismatches.

### Actual rerun002 raw receipt

The raw file
`v4/audit/autoresearch/protocol101_ft2_20_parallel_design_review_rerun002/receipt.json`
hashes to
`17cc9983c38ef4c75b8fece0cd67d4662e559f84a27de8eba2a84a34f85d96ac`.
That exact raw hash appears in each current FT2-08, FT2-10, and FT2-11 receipt.
The prior stale `17cc72ea...` pin is absent from the manifest-scoped current
packet and its direct current consumers.

## Source-transfer, calibration, and Phase-F topology

The live-parity lineage is fail-closed and unambiguous:

- `shadow_sufficiency_spec.json:/historical_to_IBKR_source_transfer` points to
  `calibration_spec.json:/source_transfer_gate` and pins the current full-file
  calibration SHA-256 `22837145...`.
- `evidence_standard.json` pins the same current calibration bytes for the
  mandatory action-conditioned calibration gate.
- Forecast and calibration import the same exact RLAC WAIT target at
  `/targets/WAIT_head`; both prohibit a scalar `U_label` substitute and use
  the same zero-intent-eligible population rule.
- Candidate-specific historical/IBKR transfer is not treated as already
  satisfied. FT2-92 must execute it after FT2-91.
- The source-transfer outcome vocabulary is exactly `pass`, `fail`, and
  `insufficient_evidence`.

Graph V2 has 47 nodes, 104 edges, and all 47 nodes are reachable from
`FT2-00-GRAPH-RESET`. The relevant Phase-F edges are exactly:

```text
FT2-92 pass                  -> FT2-93
FT2-92 fail                  -> STOP-CANDIDATE-REJECTED
FT2-92 insufficient_evidence -> STOP-OWNER-DECISION

FT2-93 producer_complete -> FT2-94 independent live-shadow audit
FT2-93 fail              -> STOP-CANDIDATE-REJECTED
```

Thus incomplete source-transfer evidence cannot enter no-order live shadow,
and producer completion at FT2-93 cannot bypass independent audit or owner
paper authorization.

## Oracle repair and live-game implications

The two-contract planted regression now produces the canonical result:

```text
trade_count=0
rejected_fill_count=1
rejected_contract=future-best-rejects
position_opened=false
premium_charged_cents=0
fee_charged_cents=0
substitute=false
```

The rebuilt evidence contains 622 rejected-fill events: 375 from oracle
variants and 247 from P5 variants. Every event has `position_opened=false`,
zero premium, zero fee, and zero realized-PnL change. All nine oracle variants
now report a positive rejection count, replacing the prior impossible-looking
zero-rejection output.

Historical implication: oracle ceilings, trade counts, occupancy, PnL,
variance/MDE inputs, and subsequent opportunity availability now reflect a
commit-then-recheck game rather than a future-fill-prefiltered game.

Future live implication: this repairs the historical reference mechanics, not
a live policy. The oracle still uses hindsight only for diagnostic ranking.
Any later candidate must choose from causal time-`t` inputs, persist the exact
source-neutral identity, recheck only that identity at `t+1`, and prove
historical/IBKR tensor/action transfer at FT2-92. No live runtime file was
changed and no parity pass is claimed here.

## Independent mechanical reproduction

All tests were inspected before execution. T6 was run against a private
byte-identical temporary FT2-08 copy because its validator writes
`validation.json`; the reviewed source validation artifact remained unchanged
at SHA-256
`d0fd65b55ed38d88d82541e7a2fe1ecec3903b0bb61c6bc41e3c24368cd5fd6e`.
The temporary copy and temporary outputs were moved to Trash after use.

| Gate | Result |
|---|---|
| Bounded-fix checker | `PASS`, 20/20, zero failed |
| Targeted census pytest | `PASS`, 12/12 |
| T1 RLAC synthetic one-way pipeline | `PASS`, 5/5 |
| T3/T4 census reconciliation | `PASS`, 11/11 |
| T6 private-copy regression | `PASS`, 4/4; FT2-08 22/22 and three terminal fixtures |
| Graph structure/topology | `PASS`, 47 nodes / 104 edges / 47 reachable |
| Manifest after-hash reproduction | `PASS`, 41/41 |
| Current packet deliverables | `PASS`, 157/157 |

## Documentation-only observation

The consolidated authority still reuses amendment label `A4`. This was already
classified as documentation-only, the authority is an immutable input to this
bounded mechanical repair, and the duplicate label does not create an
ambiguous graph hash or route. It is not an in-scope blocker.

## Side effects and route

All forbidden side effects are false: no broker, recorder, paid-data,
protected/outer/holdout, live, training, simulator mutation, runtime,
paper-default, promotion, launchd, or order path was used. No reviewed artifact
was modified. Durable workspace writes are limited to `review.md` and
`receipt.json` in this private seat directory.

**Verdict: `DELTA_REVIEW_CLEAN`.**

Next remains **`STOP_FOR_OWNER_DECISION`**. This seat does not approve or route
FT2-21.
