# D59 Staged Sub-Minute Representation Amendment A7 — Codex Review

Status: **ENDORSED WITH REQUIRED IMPLEMENTATION CLARIFICATIONS**  
Review role: Codex reviews Claude's proposal before implementation  
Pre-amendment authority SHA-256:
`82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2`  
Graph V2 SHA-256, required unchanged:
`9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08`

## Verdict

The risk-laddered Tier-S/Tier-T cut is sound if four clarifications are folded
into the implementation. None changes the owner's economic decision; each is
needed to make its safety guarantee technically true and mechanically
enforceable.

1. Databento `cbbo-1s` is a time-space **last consolidated BBO on interval**,
   not an event-complete record or necessarily one printed record for every
   instrument-second. Databento documents that no record is printed when no
   trade or CBBO update occurs within the interval. It can therefore omit an
   intra-second path even though it retains the interval's last state.
2. Databento `cmbp-1` provides every consolidated top-of-book update event, not
   literally every quote at every depth. That event completeness is sufficient
   to observe a consolidated-bid floor cross and subsequent recovery.
3. Retaining raw `cmbp-1` is not enough by itself. If floor/stop labels are
   computed only after last-state-per-second downsampling, the intra-second
   cross is still lost. Tier-T trusted crossing labels must inspect the raw
   consolidated top-of-book event path; the derived 1-second tensor remains the
   model input. This preserves the original “raw ticks are not model inputs”
   guarantee.
4. A prose-only checker cannot enforce “every corpus.” A7 therefore needs a
   machine-readable policy plus a canonical corpus registry. Pre-A7 immutable
   corpora receive their tier through a registry sidecar; every new corpus must
   embed a tier tag and register before use. The checker validates policy,
   current registry coverage, tier/source/fidelity compatibility, and the
   fail-closed trusted-consumer rule.

With those clarifications, implementation may proceed.

## A. Intra-second floor-crossing rationale

**Correct, with precise vendor terminology.** `cbbo-1s` records the last
consolidated BBO in time space for an interval. A bid can cross a floor and
recover before the retained state, making that path invisible in `cbbo-1s`.
`cmbp-1` records each event updating consolidated top of book, so the crossing
path is observable. Official schema references:

- https://databento.com/docs/schemas-and-data-formats/cbbo
- https://databento.com/docs/schemas-and-data-formats/mbp-1

The relevant distinction is path observability, not whether both products show
a valid BBO at a one-second decision boundary.

## B. Tier-S/Tier-T cut

**Correct for staged evidence.** Tier S is appropriate for inexpensive parity
probes, quarantined skeleton/prototype iteration, and a one-second observation
plane. Its floor/stop labels must remain explicitly `1-second-approximate` and
non-promotable. Tier T is appropriate for trusted floor/stop labels and must be
required before promotion, paper readiness, paper operation, or real-money use.

The 2025-02-20 lower bound for a Tier-S combined evaluation is coherent with
the documented `cbbo-1s` coverage and prevents entry-only history from being
mistaken for combined entry/lifecycle evidence.

## C. Existing assumptions and conflicts

- The signed FT2-21 approval bundle's original D59 says to acquire `cmbp-1`
  only. A7 is a later scoped amendment that supersedes that acquisition rule
  for Tier S while retaining it verbatim for Tier T. The signed bundle remains
  historical provenance and is not rewritten.
- The current FT2-08 tensor schema and authority sections 2.4/5.3 use completed-
  minute decision semantics. A7 makes `cbbo-1s` an eligible representation for
  a **separately governed** future one-second lifecycle cadence; it does not
  silently activate that cadence, train a model, or mutate runtime semantics.
  The later lifecycle/tensor design must reconcile cadence explicitly.
- D58 remains coherent when split by evidence tier: Tier-S calibration is
  provisional/report-only; trusted intra-minute floor protection is Tier T.
- No active FT2-08/10/11 machine spec currently consumes a promotion-grade
  sub-minute corpus. The new registry is therefore fail-closed with no current
  Tier-T corpus rather than fabricating one.

## D. Checker enforceability

The named rule is enforceable only through the following concrete contract:

1. `sub_minute_tier_policy.json` freezes allowed source schemas, tags,
   fidelity labels, consumers, and Tier-T raw-event/downsampler requirements.
2. `sub_minute_corpus_registry.json` is the canonical sidecar for pre-A7
   immutable corpora and the registration point for future corpora.
3. The checker rejects an unregistered discovered current sub-minute raw
   corpus; a Tier-S source with a trusted tag/consumer; a Tier-S floor label
   not marked `1-second-approximate`; or a trusted consumer whose corpus is not
   `cmbp-1`-derived, raw-event-retaining, raw-event-label-built, and
   downsampler-certified.
4. Absence of a current Tier-T corpus passes only because no trusted consumer
   is active. A trusted consumer without Tier T fails.

## Scope and highest review claim

This review endorses the data-representation staging only. It authorizes no
download, fitting, protected-resource access, broker/recorder contact,
one-second runtime activation, graph change, promotion, or order.

Highest review claim: “A7's Tier-S/Tier-T representation cut is sound with raw-
event trusted-label construction, prospective cadence wording, and a
machine-readable fail-closed registry/checker contract.”

