# FT2-08 Path-D 1-Second Exit Tensor + Label Contract — PROPOSAL

**STATUS: PROPOSAL (Claude writes → Codex reviews → owner signs).** NOT yet authority.
An **additive** Path-D extension: it adds a 1-second exit tensor + exit labels on top of
the **frozen** legacy FT2-08 minute entry tensor/labels (entry side byte-unchanged). Scope:
Phase-1 Tier-S feasibility. Constituent of the Path-D Phase-1 authority overlay.

The legacy FT2-08 contract (`v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/`)
is NOT mutated; this is a new Path-D-namespace contract that references it.

## 1. Scope and additivity

- **Entry side: unchanged.** The legacy minute entry tensor, the six FT2-04 entry label
  families, and their fill law remain exactly as pinned.
- **Exit side: new.** This contract defines the 1-second exit tensor and exit labels for
  the learned lifecycle model (FT2-60 v2), on the Tier-S `cbbo-1s`→1s substrate.

## 2. Substrate and cadence

- Sub-minute substrate = **Tier-S `cbbo-1s`** (A7), consumed at 1-second cadence.
- Exit decisions fire on **completed seconds**; entry remains completed-minute. Occupancy
  spans from the entry fill (completed-minute `t+1`) until exit/forced-flat.
- Clock: the governed spec from the model-plane contract v2 (event/source-receipt/
  local-receipt; interval closure; cross-feed watermark; latency bound).

## 3. Exit tensor (per open-position, per second)

Feature columns come EXCLUSIVELY from the model-plane v2 exit allowlist + lineage manifest
(each row carries a valid lineage record or is rejected fail-closed). Grouped:
- **Market alpha (Databento/ThetaData):** current bid/ask/mid, spread, sizes, self-computed
  IV/delta/gamma, SPX path/context — as enumerated in the exit allowlist.
- **Permitted position state (broker-confirmed):** entry fill price/time/qty/identity;
  running PnL vs the actual entry fill; MFE/MAE; giveback-from-peak; velocities; time-in-
  trade; minutes-to-close; occupancy; D48/D49 budget state.
- **Masks/clock/fees.**
No IBKR quote-derived alpha, broker outcome labels, or raw vendor greeks (model-plane §1C).

## 4. Exit labels (targets for FT2-60 v2)

Label-only columns, excluded from the runtime feature contract (mutate-future audit
enforced). Defined per FT2-60 v2:
- `Q_exit`, `Q_hold`, `A_hold` — **deployable** values under the finite-horizon
  continuation policy + the frozen provisional fill law + the flat-slot law (NO oracle /
  `future_best_*`; the legacy foundation's hindsight `q_hold` is explicitly NOT used).
- switching/re-entry components (residual only, no spread/fee double-count);
- the distributional-head targets (defined quantiles, horizon H, censoring at forced-flat/
  no-bid, loss = pinball).
- The exit valuation uses executable bid via the **provisional fill law contract**.

## 5. Serial occupancy-release semantics

- One account, one open position. The slot is occupied from entry fill to exit fill; the
  flat-slot opportunity value (FT2-60 §3a) begins at the **actual simulated
  occupancy-release time** produced by the provisional fill law (not at the decision
  instant).
- No-bid / locked / 15:55 forced-flat handled per the provisional fill law; censored labels
  marked accordingly.

## 6. OOF / provenance

- Exit trajectories are generated from **fold-specific frozen entry** models/calibrators/
  composers/thresholds (never the final full-fit model) — see FT2-60 §5.
- Session-clustered; sessions indivisible.
- Every tensor/label column registered in the fail-closed feature-lineage manifest.

## 7. Implementation + verification path (for Codex)

1. Review additivity (legacy FT2-08 byte-unchanged?), the exit tensor/label definitions,
   occupancy-release semantics, and consistency with model-plane v2 + FT2-60 v2 + the
   provisional fill law. If a blocking defect: STOP and report.
2. If sound: implement as a Path-D-namespace contract; regenerate only the downstream
   hashes it introduces; register its hash in the Path-D Phase-1 overlay. Do NOT touch the
   legacy FT2-08 contract or its receipts.
3. Claude verifies (additivity; no oracle in labels; lineage-manifest coverage; legacy
   FT2-08 byte-unchanged).
4. Owner signs (as part of the overlay).

No training, download, broker/recorder contact, or runtime change is authorized here.
