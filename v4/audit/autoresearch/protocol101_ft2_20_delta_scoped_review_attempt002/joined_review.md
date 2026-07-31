# FT2 Delta-Scoped Fresh-Seat Review — Joined Review

- Goal: `FT2-DELTA-SCOPED-FRESH-SEAT-REVIEW`
- Attempt: `002`
- Baseline commit: `a7602fdcce541589440b4aa2bc0bd0e2be6d1bbd`
- Scope authority: `v4/audit/autoresearch/protocol101_ft2_20_delta_scoped_review_fixes_attempt001/diff_manifest.json`
- Diff-manifest SHA-256: `2aa1ff2d307d9eccdd483484cde882dee230cc2703869b05dc8561eb41f18ee7`
- Consolidated authority SHA-256: `d115b953d8959fe777923ca5c1e375246754a181847ae77b57d37d24f0a279ca`
- Verdict: **`DELTA_REVIEW_CLEAN`**
- Next: **`STOP_FOR_OWNER_DECISION`**

## Controller conclusion

All three fresh seats completed independently before the controller read any
private seat output. Every seat returned `DELTA_REVIEW_CLEAN`, and the
controller reproduced the three review hashes, the three raw receipt hashes,
and all nine receipt-declared deliverable hashes with zero mismatch.

The manifest-bounded repair closes all three prior blocking findings and the
friction-population documentation finding:

1. Non-P5 oracle selection is now causal at decision time: it commits one exact
   identity before the selected-only `t+1` recheck, and a rejection opens no
   position, charges no premium or fee, permits no substitute, and cannot retry
   before `t+2`.
2. The executable builder and current FT2-04/05/08/10/11 dependency chain are
   re-sealed to the current authority and semantic bytes.
3. Forecasting and calibration import one exact RLAC WAIT target and population;
   a scalar `U_label` threshold substitute is explicitly forbidden.
4. The friction table discloses and reconciles the time-`t` intent,
   successful-recheck, rejected-recheck, friction-valid, and spread-valid
   populations.

The duplicated historical authority amendment label `A4` remains a disclosed
documentation-only observation. The immutable authority's graph hashes and
routes are unambiguous, so this does not block the repaired delta.

This review does not authorize or route FT2-21.

## Independent seat results

| Seat | Review SHA-256 | Receipt SHA-256 | Verdict |
|---|---|---|---|
| `seat1_trading_realism` | `60ab47b33eed80a72ed5d5c7547680c828d7a2aa4b1c996593967ce03cce6203` | `d310b5577a13d65017e813c7c8922f9fd1fc16bcf23c7b05b1242583584386bd` | `DELTA_REVIEW_CLEAN` |
| `seat2_ml_statistics` | `add6d52c2e4d78bad3a809638b71922ece284b163616a7d08ee659206ec385af` | `ff38f4c7ecd0cbcf09589004f919415a802cd9dbee2c2dcc8d531b89a0372c2c` | `DELTA_REVIEW_CLEAN` |
| `seat3_live_parity` | `9ef638202252fa59eca3326cb8048793a6a587721b9d52cde758dc15cedee0ce` | `6436cf617746d1dc905d8248141889c3e88801e5363517efd3d75f875122e247` | `DELTA_REVIEW_CLEAN` |

## Finding dispositions

| Finding | Final disposition | Joined evidence |
|---|---|---|
| `FT2-DELTA-B1` | **FIXED** | Source inspection, the planted two-contract regression, and regenerated rejection evidence all enforce commit-then-recheck with no substitute or charge. |
| `FT2-DELTA-B2` | **FIXED** | Builder verification passes; manifest hashes are 41/41; current semantic path/hash mismatches are 0; packet deliverables are 157/157. |
| `FT2-DELTA-B3` | **FIXED** | Forecast and calibration carry the same structured RLAC target authority, formula, exclusions, and denominator; T1 remains 5/5. |
| `FT2-DELTA-D1` | **FIXED** | Intent rows equal fill-pass plus rejected rows in every friction band; fill-pass equals friction-valid and spread-valid equals intent. |
| `FT2-DELTA-D2` | **DOCUMENTATION_ONLY_UNCHANGED** | Duplicate historical `A4` label remains disclosed; no graph hash or route is ambiguous. |

## Joined mechanical evidence

| Gate | Result |
|---|---|
| Repair diff-manifest after-hashes | `PASS`, 41/41 |
| Bounded-fix checker | `PASS`, 20/20 |
| Targeted FT2-05 pytest | `PASS`, 12/12 |
| T1 one-way RLAC fixture | `PASS`, 5/5 |
| T3/T4 census reconciliation | `PASS`, 11/11 |
| T6 private-copy regression | `PASS`, 4/4; FT2-08 22/22 |
| Graph topology | `PASS`, 47 nodes / 104 edges / 47 reachable |
| Current semantic path/hash scan | `PASS`, 0 mismatches |
| Current packet receipt deliverables | `PASS`, 157/157 |
| Preserved FT2-05 checkpoint artifacts | `PASS`, 90/90 |
| Seat receipt-declared deliverables | `PASS`, 9/9 |

The planted two-contract oracle case produces zero trades, one rejected fill,
zero premium, zero fees, no opened position, and no lower-ranked substitute.
The rebuilt evidence contains 622 zero-charge rejection events. All nine oracle
variants have positive rejection counts.

The friction populations reconcile to 128,758 time-`t` intent rows, 121,553
successful `t+1` rechecks, and 7,205 rejected rechecks. The rates are
`7,205 / 460,937 = 1.563120%` for the governed universe and
`7,205 / 128,758 = 5.595769%` conditional on the intent population.

## Reproduced seals

| Artifact | SHA-256 |
|---|---|
| Consolidated authority | `d115b953d8959fe777923ca5c1e375246754a181847ae77b57d37d24f0a279ca` |
| Graph V2 | `35859a40747ebbd75cbb222345b24f45c23beff80a740fb45170b894b01581e9` |
| Intent/fill recheck law | `5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0` |
| Simulator v5 | `7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548` |
| Matched-random generator | `8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754` |
| RLAC specification | `adcbec44bc254fe9a968562df77ca6a5736aa5d39d9f8e69dcf477a378df4f09` |
| Producer diff manifest | `2aa1ff2d307d9eccdd483484cde882dee230cc2703869b05dc8561eb41f18ee7` |
| Producer receipt | `f7c01162f8637b70e5d0492495ce870a6172d5f91713478b345e47d70d1bf532` |
| Producer aggregate canonical self-hash | `f7ae2a4aa768fa441220fad2945e9a7d712a018978e1fe6bd041a477046e8c02` |
| Rerun002 raw receipt | `17cc9983c38ef4c75b8fece0cd67d4662e559f84a27de8eba2a84a34f85d96ac` |

## Limits, side effects, and route

The verdict accepts only the 41-file manifest-bounded repair and its direct
consequences. It is not evidence of profitability, protected-data readiness,
historical/IBKR parity, live readiness, paper authorization, or promotion.

All forbidden side effects are false. No broker, recorder, paid-data, live,
protected, outer, or holdout path was used; no real model was trained or fit;
simulator v5, runtime flags, paper default, promotion state, launchd, and orders
were not modified.

**Final verdict: `DELTA_REVIEW_CLEAN`.**

**Required next state: `STOP_FOR_OWNER_DECISION`. Do not approve or route
FT2-21.**
