# Pipeline Parallelism Map (Fake-Edge Test) + Autoresearch Diamond

**STATUS: METHODOLOGY NOTE, NOT AUTHORITY.** A lens applied to the *existing*
Graph V2 contracts — it invents no new science and changes no frozen design.
It answers two questions: which upcoming steps must stay serial vs. which can
fan out, and what shape the autoresearch loop takes when we reach Phase C.
Prepared 2026-07-30 (Fable), applying the graph-engineering fake-edge test and
diamond pattern.

Key framing (from the comparison): our **governance spine** is a control/trust
state machine and must stay serial — real dependencies, owner gates, "approve
every step" — which the fake-edge test *confirms*. The parallelism lives one
layer down, in the **diagnostics** and the **autoresearch trials**.

---

## Part 1 — Fake-edge test over the actual pipeline

The test, per step: *does this step actually need the output of the one
before it?* Yes → real edge, keep serial. No → fake edge, fan out.

### SERIAL — real edges (must stay a controlled sequence)

| Step / edge | Why the edge is REAL |
|---|---|
| FT2-21 approval → Phase B machinery | Machinery is built against the approved+amended authority |
| Walking Skeleton 0→1→2→3→4→5→6 | Each stage *consumes* the prior: spec→entry→combine w/ exit→replay→paper→record→3-way parity. A true pipeline; the fake-edge test finds NO fake edges here |
| Entry train → entry freeze → lifecycle train | Lifecycle specializes on *frozen-entry* OOF trajectories (D-series) |
| Inner autoresearch → shortlist freeze → outer eval | Outer evaluation must see a *frozen* shortlist (no peeking) |
| Any producer → its independent audit | The audit consumes the producer's frozen output |
| Census/label freeze → training | Training consumes frozen labels |
| Parity gate → real training | Hard gate; training may not precede it |
| Data acquisition → build → train | Obvious real edges |

Governing rule: the spine is deliberately serial with owner gates. Do **not**
parallelize it for speed — latency was never our bottleneck; trust is.

### FAN-OUT — fake edges (independent; can run in parallel, capped)

| Independent work group | Members (no edges between them) |
|---|---|
| **Diagnostics layer** | band-profitability report · MDE refinement · census excluded-winners / friction-by-band / regime stratification · raw-DBN quote-timestamp revival check (D55 pre-step) · tick→1s downsampler certification (D59) |
| **Feature-admission candidates (D56)** | calendar/event flags · prior-day levels — two independent groups, each independently parity-tested + diagnosed, then reduced to keep/reject |
| **Machinery sub-builds** | minute-tensor build ∥ 1-second-tensor build (independent until the harness pilot joins them) |
| **Lifecycle broad pretraining** | can run parallel to entry training; only lifecycle *specialization* waits for the frozen entry (partial edge) |
| **Autoresearch trials** | the N candidate configs inside the inner loop — the big fan-out (Part 2) |

Real edges that DO constrain the fan-out layer (where one feeds a gate):
MDE → GPU-tranche gate; band-profitability → D48 cap decision; each admission
candidate → the parity gate before it can become alpha.

**Net:** the fake-edge test finds essentially no fake edges in the *spine*
(correctly serial), and a real parallelism dividend in the *diagnostics* and
*trials*. Fan those out under hard caps; keep the gates serial.

### Acquisition lesson — bulk batch, not serial streaming (empirical, 2026-07-30)

From the walking-skeleton Stage-0 clean-1s run: acquiring **13 `cbbo-1s`
sessions took ~66 min wall-clock but only ~37 CPU-seconds** — the process was
~99% idle, waiting on the vendor. The time sink was **per-request latency ×
retries × batch-job queue** (4 streaming attempts failed and fell back to
Databento batch jobs), **not** data volume (~1.8 GB is trivial throughput).

Implications for the real backfill (do NOT inherit the serial-streaming
pattern):

- **Independent sessions are a fake-edge fan-out** (already flagged in the
  *machinery sub-builds* row). Acquire by **bulk batch submission** — submit
  every session up front, let the vendor prepare them server-side in parallel,
  download when ready — under the same cost caps, rather than a session-by-
  session streaming loop with retries.
- **1-second wall-clock is a pessimistic upper bound, not representative.**
  `cbbo-1s` is the heaviest schema we pull; the backfill *bulk* is `cbbo-1m`
  minute data (~10-30× lighter per session). Do not extrapolate 1s-session
  timing to the whole backfill.
- **The serial gate still holds:** acquisition → build → train stays a real
  edge; only the *within-acquisition* per-session work parallelizes. Cost caps
  and cost-estimate-before-download remain mandatory regardless of pattern.

---

## Part 2 — The autoresearch loop as a diamond (ready for Phase C)

This re-expresses the **existing** inner-autoresearch contract (the bounded
loop, 12-trial plateau, ≤3 shortlist, independent audit) in the diamond
vocabulary: **fan-out → reduce → fresh-audit → synthesize.** Same for entry
(Phase C) and exit/lifecycle (Phase D).

```
                 ┌──────── FAN-OUT ────────┐
   frozen  ──►   │ N candidate configs,    │
   contract      │ trained in parallel on  │
   (anchor)      │ inner-dev folds only    │
                 └───────────┬─────────────┘
                             ▼
                 ┌──────── REDUCE (plain code, NO model) ────────┐
                 │ dedup configs · drop mechanically-invalid ·   │
                 │ count SERIOUS trials toward the 12-plateau ·  │
                 │ rank by the FROZEN inner objective            │
                 └───────────┬───────────────────────────────────┘
                             ▼
                 ┌──────── FRESH AUDIT (independent, empty context) ─┐
                 │ reproduce shortlist metrics/controls; NEVER the   │
                 │ producer's context; leakage/identity/parity checks│
                 └───────────┬───────────────────────────────────────┘
                             ▼
                 ┌──────── SYNTHESIZE / SHORTLIST ────────┐
                 │ freeze ≤3 meaningfully-different        │
                 │ candidates for one-shot OUTER eval      │
                 └────────────────────────────────────────┘
```

### Node contract (every trial)
- **JOB:** one config change (architecture / loss-weight-within-family /
  optimizer / regularization / calibration-impl), nothing else.
- **IN:** frozen config hash + inner-dev fold manifest.
- **OUT (schema-enforced):** `{ inner_objective, per-fold metrics, model_hash,
  seeds, receipt }`. Free text is rejected.
- **WHY:** a fixed output shape is what lets REDUCE consume it without a human.

### Anchors — the nodes that cannot be argued with (§9 of the article)
These are frozen precisely because an optimizer would bend them to win:
- the frozen **inner objective** and the **frozen target family**;
- **fold boundaries** and inner/outer separation (outer folds untouchable);
- **fees, the serial simulator, the gates** (G1-G9, G4 survival);
- the **plateau counter** — computed by the controller *from the trial
  ledger*, never self-reported by the loop;
- the loop may change weights/architecture; it may **never** change rules,
  budgets, evidence, or the stopping criterion.

### Caps — cost + runaway control (§12 of the article)
- **GPU tranche** ($20) gates each block of trials; **MDE-before-spend** gates
  the first tranche; exhausting a tranche pauses for owner decision.
- **12 serious non-improving trials → plateau stop.** Duplicates, failed jobs,
  cosmetic changes, seed-only reruns do NOT count.
- **≤3 candidates** freeze to the outer evaluation; baselines don't consume
  shortlist slots.

### Fan-in guard — catch the silently-dead node (§7 of the article)
REDUCE counts returned trials against the ledger's dispatched count; a trial
that died is **flagged**, never silently dropped, and the loop never
synthesizes on a partial set.

### Isolation — no shared context, no shared resource (§6-7 of the article)
- The audit is a **fresh, independent** context — the exact producer/reviewer
  separation we already enforce in the 3-seat reviews.
- Parallel trials get **isolated workspaces/seeds**; any two writing the same
  artifact need an edge, not parallelism (the false-independence trap we
  already caught and fixed once).

### Mapping to the graph
This diamond IS the existing entry inner-autoresearch node → shortlist-freeze
node → outer-evaluation node → independent-entry-audit node, viewed as one
shape. The lifecycle campaign (Phase D) is the same diamond on frozen-entry
OOF trajectories. Nothing here amends the frozen contracts; it is the
execution pattern for when those nodes run.

**Use:** pull this up at Phase C kickoff so the inner loop is built as a capped
diamond (parallel trials → deterministic reduce → fresh audit → ≤3 shortlist)
rather than a serial trial-by-trial crawl — faster and cheaper, same rigor.
