# Path-D Phase-1 Storage Footprint Estimate

Date: 2026-08-03  
Scope: pre-purchase dry-run only; no corpus decode, model fit, holdout access, drive operation, broker action, or registry change.

## Verdict

`PASS_WITH_HEADROOM`

The conservative Phase-1 planning envelope is **89.69 GB decimal** (about
83.53 GiB), including the existing 20.02 GB vendor corpus. The frozen
`phase1_storage.validate_phase1_allocation` cap is 150.00 GB decimal (about
139.70 GiB), leaving **60.31 GB decimal** (about 56.17 GiB) of headroom, or
40.2% of the cap.

A 1 TB SSD technically suffices for Phase 1 by a wide margin. A 2 TB SSD is a
rational purchase because it leaves materially more active workspace for the
deferred approximately 300 GB `cmbp-1` event corpus and its working expansion.
A 4 TB SSD is not required for Phase 1.

This estimate does not decode or count rows inside the final 36-session
firewall. The firewall remains closed.

## Authoritative storage law

`v4/research/phase1_storage.py` freezes both independent storage gates:

- `PHASE1_MAX_ALLOCATED_BYTES = 150_000_000_000`.
- `MINIMUM_FREE_FRACTION = 0.25` after the proposed incoming allocation.
- `validate_phase1_allocation(data_root, incoming_bytes=...)` recursively sums
  every regular file below the Phase-1 data root, rejects symlinks, adds the
  proposed incoming bytes, and fails closed when the result is greater than
  150,000,000,000 bytes.

The SSD runbook places `vendor`, canonical data, scratch trajectories, and
artifacts under `/Volumes/AR_TRADING_DATA`. Therefore the estimate and the
runtime check both count the relocated vendor corpus rather than treating it as
free or external to the cap.

## Frozen campaign geometry

The estimate uses the implemented fold and trajectory laws, not just the
approximately 1,500-row shorthand:

| Quantity | Derivation | Conservative count |
|---|---:|---:|
| Development sessions | Frozen development scope | 215 |
| Outer-test sessions | `215 - 44 initial train - 5 embargo` | 166 |
| Initial-history exit-training sessions | prequential indices 21 through 43 | 23 |
| Sessions able to emit trajectories | `166 + 23` | 189 |
| Maximum policy trajectories per session | 6 time blocks x learned/control | 12 |
| Maximum trajectory partitions | `189 x 12` | 2,268 |
| Rows under the 1,500-row shorthand | `2,268 x 1,500` | 3,402,000 |

The implementation does **not** truncate an exit trajectory at 25 minutes.
`build_trajectory_tables` creates a completed-second grid from the entry fill,
rounded up to the next second, through 15:55 ET inclusive. With the entry order
arriving 3.336 seconds after a minute boundary, an earliest possible fill in
each fixed block produces:

| Block start, minutes after 09:30 | Maximum rows through 15:55 |
|---:|---:|
| 31 | 21,237 |
| 90 | 17,697 |
| 150 | 14,097 |
| 210 | 10,497 |
| 270 | 6,897 |
| 330 | 3,297 |
| **One policy role per session** | **73,722** |
| **Learned plus control per session** | **147,444** |

The code-shaped worst case is therefore **27,866,916 feature rows**, 8.19 times
the 1,500-row shorthand. This is intentionally conservative: an actual learned
policy may not emit a positive decision in every block, a fill can fail, and
serial occupancy can suppress later receipts. Storage planning does not take
credit for those reductions.

## Byte calibration and envelope

A disposable 100,000-row, schema-shaped Zstandard Parquet dry-run was used to
calibrate row sizes without reading vendor data or the firewall. Random
64-character hashes were included so string compression was not overstated.
It measured:

- exit features: 541.02 bytes per row for 11 identity/hash columns plus the 52
  float64 features;
- exit labels: 107.81 bytes per row for the 15-column `ExitLabelRowV1` table;
- OOF exit predictions: 63.46 bytes per evaluation row.

The builder stores one baseline label table and all eight combinations of two
fee laws ($3/$4 round trip) and four latency laws (0/1/2/5 seconds). Because the
headline $3/1-second combination also appears in the sensitivity tree, the
on-drive worst case is nine label copies, not eight.

| Component | Conservative method | GB decimal |
|---|---|---:|
| Existing immutable vendor corpus | measured inventory rounded up | 20.02 |
| Exit feature partitions | 27,866,916 x 541.02 bytes | 15.08 |
| Baseline plus 8 sensitivity label partitions | 27,866,916 x 107.81 bytes x 9 | 27.04 |
| OOF exit-prediction tables | 24,475,704 evaluation rows x 63.46 bytes | 1.55 |
| Small-file/Parquet/partition overhead | explicit reserve | 2.00 |
| Entry canonical tables, labels, receipts, and OOF caches | explicit reserve | 1.00 |
| 29 entry artifacts plus 6 exit ensembles/72 HGB heads | deliberately large reserve | 5.00 |
| Disposable caches, replay outputs, reports, and resumability reserve | explicit reserve | 18.00 |
| **Conservative Phase-1 envelope** | | **89.69** |
| **Frozen allocation cap** | | **150.00** |
| **Headroom** | `150.00 - 89.69` | **60.31** |

The entry reserve covers roughly 247,250 candidate rows if the observed 1,150
rows per session holds across all 215 development sessions. The model reserve
is intentionally much larger than expected serialized HGB artifacts.

## Drive-size decision

### 1 TB

A nominal 1 TB drive has approximately 1,000 GB decimal capacity. Preserving
25% free permits at most about 750 GB of total allocation. The 89.69 GB envelope
uses about 12.0% of that permitted allocation, and the separate 150 GB Phase-1
cap remains the binding campaign limit. **One terabyte is technically more than
sufficient for Phase 1.**

### 2 TB

A nominal 2 TB drive permits about 1.50 TB of allocation while keeping 25%
free. After the 89.69 GB Phase-1 envelope, approximately 1.41 TB remains inside
that reserve. If Phase 1 earns Tier T, a 300 GB raw `cmbp-1` copy would leave
about 1.11 TB for event indexes, certified one-second derivatives, temporary
sort/shuffle data, models, and reports. That working margin is what the 2 TB
purchase buys; it is not evidence that Tier T has been earned and it is not a
substitute for the separately required cold-copy/redownload plan.

On a 1 TB drive, Phase 1 plus a future 300 GB raw event copy would leave only
about 360 GB inside the 25%-free rule for all Phase-2 derivatives and scratch.
That is possible but operationally tight.

## Cap and estimate risks

1. **The 1,500-row assumption is not the implementation.** Full paths to 15:55
   are the largest multiplier. The corrected estimate already uses the larger
   27.87-million-row bound.
2. **Nine label copies are written.** The baseline is duplicated inside the
   eight sensitivity combinations. The estimate includes all nine.
3. **Per-stage preflight is not a per-partition quota.** The CLI calls the
   allocation check before each stage with zero incoming bytes; trajectory
   generation does not re-check the cap after every partition. The staged
   runbook therefore requires a preflight before and after every long stage.
   The 60.31 GB planning margin makes this an enforcement gap to watch, not a
   current capacity warning.
4. **Actual receipt count is unknown before the lawful OOF fit.** The estimate
   uses the maximum six learned and six control receipts per eligible session.
5. **Unexpected compression loss is bounded but observable.** Abort Phase 1 if
   measured allocation exceeds the envelope materially, if projected total
   exceeds 150 GB, or if the volume would fall below 25% free. Do not delete
   source data or open the firewall to make space.
6. **Phase 2 can expand beyond raw size.** A 2 TB active drive is reasonable for
   a 300 GB raw event corpus, but pathological 4x-plus working expansion can
   approach its 1.50 TB allocation ceiling. Re-estimate before any `cmbp-1`
   purchase.

## Pre-purchase conclusion

**Phase-1 is purchase-ready on storage math.** The corrected full-trajectory
estimate fits below the 150 GB cap with **60.31 GB of headroom**. A 1 TB SSD is
technically sufficient; 2 TB is justified by potential Phase-2 active-workspace
needs, not by Phase-1 necessity.

