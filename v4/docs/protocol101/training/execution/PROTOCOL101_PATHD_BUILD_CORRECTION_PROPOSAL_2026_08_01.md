# Path-D Build Correction Proposal (evidence vs frozen build)

**STATUS: PROPOSAL (Claude-written → Codex reviews/implements → owner approves).** Apply at
the **pre-fit pause** — AFTER Codex finishes its current decoder/integration task, BEFORE any
corpus fit or evidence opening. Research-grade; no live/broker; do not change the entry=signed-17
baseline (decision #1) or open the holdout.

## Why (what changed since the prereg was frozen)
Two research findings landed AFTER Codex froze the preregistration, and one build event needs
reconciliation. Each correction below cites the **evidence**, the **build as-is**, and the **fix**.

Research inputs:
- `v4/audit/autoresearch/protocol101_pathd_feature_research/FINDING_databento_live_field_parity_2026_08_01.md`
- `v4/audit/autoresearch/protocol101_pathd_feature_research/FINDING_entry_microstructure_signal_2026_08_01.md`
Build artifacts reviewed:
- `.../protocol101_pathd_entry_exit_model_research_2026_08_01/{preregistration.json,feature_lineage.json}`
- `.../pathd_test_contamination_quarantine_2026_08_01/outer_primary_burned_receipt.frozen-gate-transaction.json`

---

## Correction 1 (P1, BLOCKING) — frozen-foundation drift + burned outer-fold-1

**Evidence/observation:** the burn receipt shows outer fold 1 primary evidence
`BURNED_NO_REOPEN`, `reason: FROZEN_FOUNDATION_DRIFT_BEFORE_DECODE`, `access_count: 0`,
`reopen_permitted: false`. The "frozen" foundation drifted after the freeze; Codex correctly
burned fold-1 pre-access to protect the firewall.

**Build as-is:** acceptance requires "≥4/5 outer folds," but fold-1-primary is now permanently
gone → the gate effectively rests on 4 survivors (needs 4/4). And a frozen foundation that
drifts signals possible instability → more folds could burn during the fit.

**Fix:**
1. **Root-cause the drift** and report it: what changed between freeze and decode — nondeterministic
   decode, a hash-input change, a touched corpus file, or a code edit? Classify BENIGN (semantics
   unchanged, e.g., a deterministic-decode fix) vs REAL (data actually changed).
2. If **BENIGN**: RE-FREEZE a stable foundation and **restore all 5 outer folds** — fold-1 had
   `access_count: 0` (never observed), so a clean re-freeze can legitimately reinstate it with a new
   freeze receipt. Preserve the burn record as history.
3. If **REAL**: keep fold-1 burned and explicitly record that acceptance is now ≥4/4 survivors (or
   route `insufficient_evidence` if that violates min-power) — do not silently treat 4 as 5.
4. **Add a foundation-stability gate:** before decoding ANY fold, assert the frozen-foundation hash
   reproduces byte-identically; on mismatch, FAIL LOUDLY and stop (don't silently burn folds one by
   one). This catches instability before spending the run.

## Correction 2 (P2) — `last_causal_open_interest` is EOD-only; enforce live-twin ground truth

**Evidence:** field-parity finding — `open_interest`/`stat_open_interest` live in the **statistics
schema = daily/EOD**, NOT real-time. Live OPRA = consolidated `cbbo-1s`/`cmbp-1` (carry bid/ask +
sizes, confirmed in our own files; MBP-1/TBBO retired for OPRA May 2025). So OI has **no real-time
live twin**.

**Build as-is:** the frozen exit feature contract (49) includes `last_causal_open_interest` with
`minute_volume_open_interest_carry_bound_seconds: 90`. A 90-second carry implies intraday updates
that do not exist for OI. (`feature_lineage.json` already carries a `feature_without_live_twin`
fail-closed fixture — but the classification predates this finding.)

**Fix:**
1. **`last_causal_open_interest`: drop it, OR redefine as prior-day-EOD static** (one value per
   session sourced from the prior session's EOD statistics; NOT a 90s-carry field). Recommended:
   **drop** — the probe showed OI is only a weak signal (+0.089 incremental) and it's live-awkward.
2. **`last_causal_minute_volume`:** confirm it is sourced from live-derivable intraday volume
   (ohlcv-1m / trades aggregation), which IS live-available, and that its carry semantics match the
   live feed. If not cleanly live, drop it too (also weak in the probe).
3. **Adopt the field-parity inventory as the live-twin ground truth** and re-verify EVERY entry (17)
   and exit (49) feature against it: bid/ask/`bid_size`/`ask_size`/`size_imbalance`/spread/
   self-computed-greeks/official-SPX-context are live ✓; anything without a real-time live twin is
   barred from intraday use. The `feature_without_live_twin` fixture must fail on OI-as-intraday.

## Correction 3 (P3, forward note — not a change to this run) — size_imbalance is a validated, live-safe enrichment lead

**Evidence:** the entry probe found **`size_imbalance` is the strongest incremental candidate-
ranking signal (+0.129 over moneyness, t≈2.6)** — stronger than the moneyness geometry the signed-17
relies on — and field-parity confirms it is **live-usable** (sizes are in the consolidated CBBO).

**Build as-is:** the exit already uses `option_size_imbalance` ✓ (good, live-safe). Entry excludes
it per decision #1 (entry=signed-17).

**Fix:** NO change to this run (keep entry=signed-17 — do not change two things at once). Just
**record** `size_imbalance` (and optionally `bid_size`/`ask_size`) as the pre-vetted, live-safe
**widen-entry** candidate to test as a follow-up IF the signed-17 baseline underwhelms — tested
fresh under the same fair firewall + guards, never bolted onto this run.

---

## Verification (Codex reports; Claude re-verifies)
1. Drift root-cause documented + classified; foundation re-frozen with 5 folds restored (if benign)
   OR fold-1 kept burned with the gate explicitly re-stated (if real); a byte-identical
   foundation-stability gate added before fold decode.
2. `last_causal_open_interest` dropped (or prior-day-EOD-static); every feature re-verified against
   the field-parity inventory; the `feature_without_live_twin` fixture fails on OI-as-intraday.
3. size_imbalance recorded as a widen-entry follow-up lead; entry unchanged (signed-17).

## Scope guard
Apply at the pre-fit pause only. Do NOT open the holdout, do NOT fit the corpus, do NOT change the
entry=signed-17 baseline, no live/broker, no governance-contract freezing. These are corrections to
the frozen research foundation so the eventual fit is honest and live-deployable.
