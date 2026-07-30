# Protocol101 FT2 Scoped Final Round — Five-Finding Producer Report

Status: `PRODUCER_COMPLETE_PENDING_FABLE_VERIFICATION`

This packet implements exactly the five owner-authorized rerun002 repairs.
It does not approve FT2-21, run an independent review, train a real model,
access outer/protected evidence, contact a broker or paid-data endpoint, alter
simulator v5, or change the paper default.

## Frozen authority and chains

- Amended authority SHA-256:
  `3c7a0aaf2334ae7f04090fb3e67eb2db16591c40545bcf3c09e78c04f0640033`.
- Amended Graph V2 SHA-256:
  `35859a40747ebbd75cbb222345b24f45c23beff80a740fb45170b894b01581e9`.
- Canonical intent/fill law remains byte-identical:
  `5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0`.
- Simulator v5 remains byte-identical:
  `7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548`.
- Chained v4 receipt hashes:
  FT2-04 `fc9c27e7ed1025aeeab4461d4176504bb8789340296cd1e9e0ad4cec67a127ca`;
  FT2-05 `ddc6167abdf763070416f5e729225ece97d594a9b21b64893b8d2bd1f25d6928`;
  FT2-08 `74d0e3be59dff05be12144046326cd8e1d91832b899e7f22304273e4b17a0813`;
  FT2-10 `0af1ce13d7839ba5b64706afac493d2d95ff25fc08abb0d31b2b42fe172de1e9`;
  FT2-11 `b1ebff7c3647cf6da6b0a30f61c217f7e2b1542fb558ecf8d269de21fe4400e4`.

## Five repairs

1. **Calibration circularity:** the model-free, one-pass RLAC now constructs
   anchor-specific WAIT targets on every included minute and per-contract
   regret targets on every label-complete intent-eligible row. It uses no
   model or runtime-composer output. Heads fit, calibrate on disjoint blocks,
   freeze, and only then feed the runtime composer. Marginal regret coverage
   is the hard claim; composer-selected-subset coverage is report-only and
   routes collapse to owner review.
2. **Oracle/intent contradiction:** FT2-04 now references the unchanged
   canonical two-stage law. Time-`t` intent eligibility defines the D48
   reference/distribution universe; selected-only `t+1` rejection is logged
   but cannot redefine that universe.
3. **Phase-F outcomes:** Graph V2 now maps FT2-92
   `insufficient_evidence` to `STOP-OWNER-DECISION`. FT2-92 emits exactly
   `pass`, `fail`, or `insufficient_evidence`; FT2-93 emits only
   `producer_complete` or `fail`.
4. **D48 transition evidence:** the row audit covers all 460,937 unique
   contract-minute identities across v2, v3, and v4, with premium, moneyness,
   phase, economics, and six label-family summaries.
5. **Denominators:** the v4 report and results disclose both
   `7,205 / 460,937 = 1.563120%` of governed rows and
   `7,205 / 128,758 = 5.595769%` conditional on reaching recheck.

## Census v4 and row-audit result

The 45-session rebuild is `feasible` under the census-only claim. It contains
460,937 governed rows, 128,758 time-`t` fee-3 intent rows, 7,205 selected-only
`t+1` rejections, and 128,758 v4 D48-reference rows.

The row audit clarifies a limitation in the earlier aggregate explanation.
The identity

```text
159,312 - 121,553 = 30,554 + 7,205
```

is correct as a **net aggregate** decomposition, but the two right-hand terms
are not disjoint v2-to-v3 row buckets. The actual three-version memberships
are:

| v2 | v3 | v4 | Rows |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 297,317 |
| 0 | 0 | 1 | 4,308 |
| 1 | 0 | 0 | 34,862 |
| 1 | 0 | 1 | 2,897 |
| 1 | 1 | 1 | 121,553 |

Thus v2→v3 removed 37,759 rows, while v3→v4 restored 7,205 rows. Every row is
accounted for. The changed cohorts differ descriptively in premium, moneyness,
phase, missingness, and path-label distributions, as expected from the rules
that select them. This is not a causal or profitability claim.

## Mechanical tests

- **T1:** PASS — five synthetic no-circularity assertions, including target
  byte identity under composer swap and rejection of a deliberate circular
  negative control.
- **T2:** PASS — extended checker v3, 29/29 checks. The original 26 checks
  remain green; new law-identity, graph-outcome, and target-provenance checks
  are green.
- **T3/T4:** PASS — 11/11 census reconciliation, dual-denominator, row
  identity, bucket, overlap, and distribution assertions.
- **T5:** PASS — 47/47 graph nodes reachable, 104 edges, no structural errors,
  and the new edge present.
- **T6:** PASS — 22/22 FT2-08 checks, three terminal examples, and both
  observed/replicate SE fixtures independently recomputed.
- **Targeted census regressions:** PASS — 10/10 current-v4 and preserved-v2
  tests. The preserved-v2 tests now explicitly import the preserved v2
  builder, so they cannot accidentally assert retired v2 behavior on v4.

## Required next action

Stop here for Fable verification. Fable should verify the hash chains, rerun
T1/T2/T5 independently, inspect census v4 plus the D48 audit, and judge the
RLAC design against the owner plan. Only after that should the owner launch
the separately authorized three-seat delta-scoped review. This packet itself
does not start either review and does not route to FT2-21.
