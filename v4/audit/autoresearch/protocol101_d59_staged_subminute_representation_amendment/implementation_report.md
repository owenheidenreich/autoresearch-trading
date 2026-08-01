# D59 Staged Sub-Minute Representation A7 — Implementation Report

Status: **IMPLEMENTED — PENDING CLAUDE VERIFICATION AND OWNER SIGNATURE**

## Result

- Pre-A7 authority SHA-256:
  `82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2`
- Post-A7 authority SHA-256:
  `1d215845cf7b853550c5cf27af5bafca66db2355e0f12493e2c5a8922278d4bc`
- Graph V2 SHA-256 before and after:
  `9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08`
- Graph changed: **no**
- Cross-contract checker: **33/33 pass**
- New rule: `sub_minute_corpus_tier_tag_required` — **pass**
- Terminal: `STOP_FOR_CLAUDE_VERIFICATION`

## Implemented policy

- Tier S accepts official `cbbo-1s` for quarantined parity probes, Walking
  Skeletons, prototypes, feasibility, and as an eligible observation
  representation for a separately governed one-second exit cadence.
- Tier-S corpora carry `skeleton`, `prototype`, or `probe`; floor/stop labels
  are `1-second-approximate`; trusted consumers are forbidden.
- Tier T remains raw-`cmbp-1`-derived, retains the raw event stream, requires
  the in-house downsampler certification against the owned official
  `cbbo-1s` pilot, and uses the raw consolidated top-of-book event path to
  construct trusted floor/stop crossing labels. Raw ticks are not model inputs.
- Promotion, paper-readiness, paper, and real-money floor/stop-label paths
  require Tier T.
- Tier-S combined evaluation is restricted to the resolution overlap beginning
  2025-02-20.
- A Tier-S acquisition still requires a free cost estimate, hard cap, explicit
  owner green-light, and quarantined output. A7 performed no acquisition,
  authorizes no `cmbp-1` purchase, and leaves D57 deferred.

## Review clarifications

The proposal was endorsed after correcting vendor terminology and making the
trusted-label mechanism explicit. `cbbo-1s` is a last-on-interval time-space
record and can hide an intra-second crossing path. `cmbp-1` provides every
consolidated top-of-book update event. Trusted labels must inspect those raw
events before downsampling; otherwise Tier T would inherit Tier S's missed-
cross defect.

A7 does not silently activate one-second training or runtime. The current
completed-minute FT2-08 tensor contract remains until a separate governed
lifecycle/tensor authorization changes it.

## Enforcement

The new rule validates the authority text, machine-readable tier policy,
canonical corpus registry, current sub-minute-manifest coverage, and trusted-
consumer bindings. Its negative fixtures prove rejection of:

1. a Tier-S corpus marked promotion-grade;
2. a trusted consumer bound to Tier S; and
3. an unregistered current sub-minute corpus.

The registry covers 41 currently discovered sub-minute raw manifests through
three registered corpus/derivative roots. No Tier-T corpus or trusted consumer
is currently registered; the first trusted consumer will fail closed until a
fully compliant Tier-T corpus exists.

## Side-effect boundary

No model training or fitting, data download or purchase, protected-resource
access, broker/recorder contact, runtime/default change, promotion, order, or
graph edit occurred.

Highest allowed claim: “D59 staged into Tier-S/Tier-T via A7; trusted
floor/stop guarantee preserved; graph unchanged; awaiting Claude verification +
owner signature.”

