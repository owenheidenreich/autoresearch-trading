# Protocol101 FT2-08 Data/Tensor/Label Contract

Status: frozen design  
Node: `FT2-08-DATA-TENSOR-LABEL-CONTRACT`  
Outcome: `design_ready`  
Product contract SHA-256: `893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832`

## 1. Scope

This contract freezes how Protocol101 will represent the synchronized market,
the full 42-contract ladder, causal 90-minute histories, and the already-frozen
FT2-04 path labels. It is a design contract only. This packet builds no real
tensor, computes no new market statistic, trains no model, and does not begin
FT2-10.

The runtime product remains:

```text
flat: WAIT or BUY one exact eligible call/put/strike
open: HOLD or EXIT
```

Masks enforce safety and physical availability. They do not decide timing,
direction, strike, or ordinary exits and are not alpha features.

## 2. Authority And Inputs

The following files were verified before this contract was written:

| Input | SHA-256 |
|---|---|
| Consolidated signed authority | `893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832` |
| Graph V2 JSON | `b06a26be59307c130da84f2dc5b6f3224c272e6c4093e83abd5bc0b280ca6d09` |
| FT2-04 `label_spec.json` | `d724195999acf579a43768756b79f30ad122ec446933347b93cbbd8a7bbe9781` |
| FT2-04 `oracle_rules.json` | `23a4cab2715c925a3c4261a508e69ac2c0f70157b93083ff129f4848ae03eb81` |
| FT2-04 `census_sessions.json` | `f77e94e58d3214e490a8d87f888aa92d8b757b028299e533a63b83442df247cc` |
| FT2-05 `census_results.json` | `8a8805b22ef04a290edf421d5229d75eebc2301bc5246bd3bc595ee616cae583` |
| Governed `runner_plan.json` | `3b0c01b216d1d5928c256060ce38c62ef2c4f912aef9b6ef98f40bc4ee7fcc73` |

The Graph V2 route is exactly:

```text
FT2-05 feasible
  -> FT2-08
  -> design_ready
  -> FT2-10
```

No other FT2-08 success route exists.

## 3. Corpus And Fold Roles

The five expanding-window folds are adopted by immutable reference from
`runner_plan.json:/expanding_folds`. They are not regenerated here. The source
uses five 45-session validation blocks, expanding training blocks of 45, 90,
135, 180, and 225 sessions, and one governed session of embargo immediately
before each validation block.

Rules:

1. Training-role sessions may fit parameters, normalization, and
   training-calibrated thresholds.
2. The one-session embargo is used for no fitting, calibration, selection, or
   evaluation.
3. Validation-role sessions are outer evidence only. Their labels and outcomes
   cannot alter a candidate, transform, composer, threshold, or stopping rule.
4. The 46 FT2-05 census sessions are design-diagnostic only. They were proved to
   occur in no outer-test slice and can inform definitions explicitly carried
   into this contract. They are never model-quality evidence.
5. The protected holdout remains untouchable. No row, aggregate, schema sample,
   or diagnostic may be read from it before the later graph gate explicitly
   authorizes the one-shot use.
6. Any session identity appearing in incompatible roles for the same operation
   fails closed before materialization.

The exact source pointers, fold hashes, dates, and firewall rules are in
`fold_roles.json`.

## 4. Canonical Identities

### 4.1 Contract-path key

The storage key is:

```text
(session_date_yyyymmdd, expiry_yyyymmdd, contract_id_utf8,
 strike_milli_points, right_code)
```

`right_code` is `0=call`, `1=put`. The redundant exact fields are deliberate.
The `contract_id` must decode to the same session, expiry, strike, and right.
Any disagreement or duplicate key fails closed.

### 4.2 Decision key

The decision key is:

```text
(session_date_yyyymmdd, decision_time_ns_utc, position_state)
```

Flat-state contract labels and candidate rows join on:

```text
(session_date_yyyymmdd, decision_time_ns_utc, expiry_yyyymmdd,
 contract_id_utf8, strike_milli_points, right_code)
```

Canonical ladder indices are presentation coordinates, not identities.
`strike_idx` and `right_idx` may locate a contract in the current snapshot but
may never key its history or labels.

### 4.3 Recentring

At each decision minute, the current 21-strike ladder is rebuilt around the
canonical ATM strike. A contract's 90-minute history follows its exact identity.
If the ATM moves, the contract may move to a different current-relative slot;
its history must not move to another contract and a slot must not inherit the
history of its former occupant.

Historical and live builders must produce the same sorted identity tuple and
the same current-slot-to-identity map for identical causal input. FT2-30 must
test exact map equality, exact valid-mask equality, and value equality within
the field's precision contract. A mismatch invalidates machinery acceptance.

## 5. Storage Contract

The source store is partitioned by session and exact contract identity. It
contains one copy of each causal contract-minute path and one copy of each
market-context minute. Decision tensors are assembled per batch by joining
current ladder identities to these stores. Adjacent decision rows therefore
reuse path segments instead of materializing repeated 90 x 42 windows.

Required logical partitions:

```text
manifest/
identity_dictionary/
market_context/session=YYYYMMDD/
contract_paths/session=YYYYMMDD/expiry=YYYYMMDD/right=C|P/
flat_decisions/session=YYYYMMDD/
open_state_scaffolds/session=YYYYMMDD/
labels/session=YYYYMMDD/
checkpoints/
```

Each completed session partition is written to a temporary name, validated,
hashed, and atomically committed. Resume skips only a partition whose input
hash, builder hash, schema hash, row count, logical-content hash, and completion
marker all match. Partial or mismatched partitions are rebuilt; they are never
appended in place.

`storage_spec.json` is the machine-readable authority for layout, deduplication,
resume, sorting, and precision.

## 6. Tensor Contract

### 6.1 Decision cadence

The history axis has 90 completed one-minute steps, oldest to newest. At
decision minute `t`:

- option state may use the completed interval at `t`;
- synchronized index/context fields use information through `t-1 minute`;
- no future value may enter any model tensor;
- missing history near the open remains masked and is never backfilled from a
  prior session.

### 6.2 Market history

`market_history_fp32` has shape `[90, 19]`:

1. the signed 17 parity-core features, in the exact order recorded by
   `tensor_schema.json`;
2. `minutes_since_open`;
3. `minutes_to_close`.

`market_phase_id` has shape `[90]` and uses the frozen seven-category mapping.
`market_history_available_mask` has shape `[90]`.

The 17 core features remain the only active alpha channels in this node.
Guard/fill/audit fields never enter this block.

### 6.3 Per-contract causal paths

The core path tensor is `contract_path_core_fp32[90,42,4]`, ordered:

```text
offset_points
moneyness_bps
internal_delta
internal_gamma
```

All four values are recomputed or aligned for the exact contract identity at
each historical minute. Raw vendor Greeks are prohibited.

The exact/audit path arrays include identity dictionary indices, timestamps,
strike, expiry, right, and bid/ask/mid fixed-point prices. These arrays support
identity proof, masks, fills, and label rebuilds; they are not model alpha
inputs.

### 6.4 Reserved admitted-price extension

The following arrays exist from schema version 1:

```text
contract_path_extension_fp32[90,42,3]
ladder_snapshot_extension_fp32[42,3]
extension_channel_active_mask[3]
extension_value_available_mask[90,42,3]
```

They reserve exactly the three admission attempts allowed by D41. Before
FT2-25/26 admission, all channel-active bits are false, all value-available bits
are false, and values are canonical zero placeholders ignored by the model.

An admitted channel may be activated only by a signed admission manifest that
fixes its name, formula, units, normalization, source hashes, and channel index.
Activation changes no existing field's shape, order, dtype, or meaning. A
rejected transform leaves its channel inactive; channels are never recycled
under the same schema version.

### 6.5 Current ladder snapshot

The snapshot is an explicit current-minute view, not a second source of truth:

```text
ladder_snapshot_core_fp32[42,4]
ladder_snapshot_extension_fp32[42,3]
ladder_contract_dictionary_index[42]
ladder_strike_milli_points[42]
ladder_right_code[42]
```

The core and extension values equal the newest history step wherever the exact
contract identity matches. Any mismatch fails closed.

### 6.6 Masks

The tensor carries all of these masks separately:

- `market_history_available_mask[90]`;
- `contract_path_available_mask[90,42]`;
- `missing_minute_mask[90,42]`;
- `pre_listing_mask[90,42]`;
- `current_quote_available_mask[42]`;
- `tradability_eligibility_mask[42]`;
- `d48_premium_cap_mask[42]`;
- `d49_budget_mask[42]`;
- `action_mask[43]`, with index 0 equal to WAIT.

`contract_path_available_mask` is true only when the exact contract exists at
that minute and the causal path values required by the active model lane are
available. `missing_minute_mask` and `pre_listing_mask` explain why it is false.
No mask is replaced by a numeric sentinel.

WAIT is always physically available while flat. A contract action is available
only when the current quote, tradability, D48, D49, identity, and all required
active-channel masks pass. Masks may gate attention or actions but may not be
fed as alpha values unless a later signed admission says so.

The complete field list, shapes, dtypes, sentinel rules, and model-facing flags
are frozen in `tensor_schema.json`.

## 7. D40 Open-State Row Contract

FT2-28 must build open-state source-row scaffolds for every completed minute
from 15:31 through 15:55 ET, inclusive, at most 25 rows per full session.

Each row uses the same exact identities, 90-minute causal histories, masks,
precision, and one-minute index/context lag as an earlier row. Differences:

- `position_state=open`;
- no entry action is legal;
- the open contract identity and entry state are exact required fields;
- 15:31 through 15:54 expose `HOLD` and `EXIT` to the later lifecycle
  machinery;
- the 15:55 row is a forced-flat trace row whose executable action mask allows
  `EXIT` only.

The scaffold is built independently of any candidate entry policy. A later
trajectory join determines whether a position is actually open and therefore
whether the row is consumed. No entry is permitted after 15:30.

## 8. Precision Contract

Field-by-field law:

- session and expiry dates: `int32` `YYYYMMDD`;
- timestamps: `int64` UTC nanoseconds;
- dictionary indices: signed integer with `-1` missing;
- strike: `int32` milli-index-points;
- option prices: `int32` cents, with a separate exact tick-regime code when
  needed;
- right and categorical IDs: unsigned integers;
- masks: Boolean bitmaps;
- SPX levels, normalized features, internal Greeks, returns, and path
  differences: `float32`;
- label dollar amounts: integer cents where exact, with model views converted
  to `float32`;
- no `float16` anywhere in stored or assembled model tensors in this contract.

FT2-30 may propose an fp16 model-input view only after an fp32 reference test
shows identical actions and score drift below a separately preregistered bound.
Until then, fp16 is forbidden.

## 9. Frozen Label Join

The six FT2-04 families and seven horizons attach one-to-one to the exact
contract-minute key. Their definitions, entry ask, future executable bid,
$3 fee overlay, censoring, missing-minute behavior, and tie-breaking are
inherited byte-for-byte and are not redefined here.

Labels live in a physically separate partition from runtime features. A feature
loader cannot request label columns. A training loader must perform an explicit
keyed join after the feature tensor is complete and after duplicate checks pass.

For every horizon, the label row stores:

- available and nominal minutes;
- horizon-censored and target-valid masks;
- all applicable family metrics in frozen dollars/return units;
- time-to-event censoring and boundary;
- exact event timestamps and extrema timestamps where defined.

Near-close censoring is valid evidence with `censored=true`; it is not silently
converted into a shorter nominal horizon. A window with no usable future mark
has `target_valid=false`. Missing/no-bid future minutes are excluded and break
run continuity exactly as FT2-04 requires.

Rebuild reproducibility requires raw input hashes, builder-code hash,
`label_spec.json` hash, schema hash, fixed sort order, fixed column order,
fixed-point conversion law, and per-session logical-content hashes. The full
rules are in `label_join_spec.json`.

## 10. Census-Informed Descriptive Bands

These fields are context and reporting definitions. Census values may inform
later training-side calibration, but nothing in FT2-05 is evidence that a model
works.

### Premium band

Based on executable entry ask:

```text
cheap_le_1:        ask <= 1.00
small_1_3:         1.00 < ask <= 3.00
medium_3_8:        3.00 < ask <= 8.00
large_8_20:        8.00 < ask <= 20.00
very_large_20p:    ask > 20.00
```

Provenance: FT2-05 `friction_by_premium_band.csv`, SHA-256
`6a28e19be5f6b61bfecb5406ce69016289ba8f5c85334545baf22ad9bedabe39`,
and the producing FT2-05 runner definition. Only the first three bands appeared
in the D48-reference census table; the full deterministic definition remains
necessary for accounts where the 5% cap scales above the reference.

### Moneyness band

Based on absolute current ladder offset in SPX points:

```text
atm:   abs(offset) <= 10
near:  10 < abs(offset) <= 25
wing:  abs(offset) > 25
```

Provenance: FT2-05 `family_distributions.csv`, SHA-256
`5bd59171cf6be963fb9bc72ea19fadc0c7bb5ac956900f73e8c7aa7dd63b314e`,
using the project-standard `moneyness_band` definition.

### Market phase

Half-open New York intervals:

```text
opening_discovery:         [09:30, 09:45)
primary_morning:           [09:45, 11:15)
europe_close_transition:   [11:15, 11:45)
lunch:                     [11:45, 13:30)
afternoon:                 [13:30, 15:00)
power_hour_entry:          [15:00, 15:31)
manage_exit_only:          [15:31, 15:56)
```

The six entry phases are present in FT2-05 `family_distributions.csv`. The
seventh phase has no flat-entry census rows and is frozen from D40 for the
open-state extension; no census statistic is imputed for it.

## 11. Synthetic Worked Examples

### Identity continuity under recentering

At 10:00 the ATM strike is 6000 and contract `SPXW-X-6000-C` occupies offset
0. At 10:01 the ATM strike becomes 6005. The same 6000 call now occupies offset
-5. Its history remains keyed to `SPXW-X-6000-C`; the new offset-0 6005 call
does not inherit the 6000 call's 10:00 value.

### Available-history and pre-listing masks

At 09:33, only four same-session minute positions exist in the 90-minute axis.
Those four positions may be available; the preceding 86 are masked. If a
contract first appears at 09:32, its 09:30 and 09:31 positions additionally
carry `pre_listing_mask=true`. No prior-day value fills either gap.

### D48/D49 action mask

With $10,000 session-start equity, D48 permits premium plus fee up to $500.
A contract with $480 premium plus $3 fee passes D48. If realized session loss
is already $40, D49 rejects it because `$40 + $483 > $500`. WAIT remains
available.

### Label join

For a synthetic call bought at ask $2.00 with a $3 round-trip fee, future bids
at minutes 1, 2, and 3 are $1.90, $2.10, and $2.40. Fee-adjusted PnL is -$13,
+$7, and +$37. The 3-minute labels therefore include early drawdown -$13,
time-to-first-real-profit 2 minutes, pre-profit adverse excursion -$13, one
underwater minute, profitable-minute fraction 2/3, and MFE +$37. These values
are labels only and never appear in the causal feature tensor.

### D40 row

At 15:40, an open-state row uses option information completed at 15:40 and
index/context information through 15:39. It cannot enter a new contract. It
offers HOLD/EXIT for the exact open contract, subject to later lifecycle
machinery. At 15:55 only forced EXIT is executable.

## 12. Fail-Closed Conditions

Materialization or loading must stop on:

- authority, graph, fold, schema, or label hash mismatch;
- duplicate or contradictory decision/contract identities;
- contract-id decomposition disagreement;
- slot history used without exact identity continuity;
- a future timestamp in any feature tensor;
- prior-session backfill;
- label columns visible to runtime/model features;
- a model action outside `action_mask`;
- an active extension channel without the required admission receipts;
- fp16 use before FT2-30 equivalence acceptance;
- protected-holdout access;
- any fold-role or embargo violation.

## 13. Freeze And Next Route

The four machine-readable specifications and this document are immutable inputs
to later design and machinery nodes. A change requires a new schema version,
a replacement receipt, and the graph's bounded design-repair route.

Highest allowed claim:

> The data/tensor/label contract is frozen; FT2-10 may be drafted against it.
