# Fork C Phase 1 — label rules for `is_spx_long_0dte_day`

**Purpose.** This document is the spec for the binary label that drives Fork C
Phase 1. Every row in `v2/fork_c/tier1_labels.csv` is curated against these
rules. Every judgment call on ambiguous days is resolved by the rules here,
not in the parser code.

**Lock discipline.** This file is committed **before** any parsing begins.
Any rule change after curation starts forces a full re-review of all prior
rows (the rules, not the excerpts, are the source of truth; changing them
invalidates cached labels).

**Provenance.** The SHA-256 of this file at run time is hashed into every
Fork-C run's provenance block (see plan §Governance). Two runs are only
comparable if their `label_rules_sha256` values match.

**Source corpus.** 167 journal files at
`/Users/gduby/Documents/picklesGPT/pickles/journal/*.txt`, dated
2023-10-22 through ~2024-06-13. File format is raw Discord export.

**Timestamp handling — read carefully.**

- **Exported Discord timestamps are Pacific-localized display timestamps**
  (e.g. `12/14/2023 7:08 AM`). These appear on the line immediately after
  a name line. They are the display time in Pickles' (or the exporter's)
  Pacific locale and must be converted `+3 hours` to ET when populating
  `first_qualifying_time_et`. Pickles' "7:08 AM" Discord timestamp is
  10:08 ET.

- **Explicit in-message market-time references are ET and must NOT be
  reconverted.** When Pickles writes phrases like "1000 MAGIC TIME",
  "1030 IB", "0830 EST", "1200 EST", "1130 EST", he is naming ET clock
  times inside the message body. These are already ET — do not add 3
  hours to them. The ET times Pickles writes inline map directly to bars
  in `v2/data.pt` (1000 = bar 30, 1030 = bar 60, etc.).

  The same file can contain a Pacific display timestamp and an inline ET
  reference on adjacent lines — e.g. his "7:08 AM" stamp (PT display =
  10:08 ET) introducing a message that mentions "1030 IB" (already ET).
  Treat them separately.

- **The bar-30 cutoff (10:00 ET) in the feature-extraction step is an ET
  reference** — aligns with Pickles' inline ET usage, not his PT display
  timestamps.

---

## 1. Target

**`is_spx_long_0dte_day ∈ {0, 1}` per session.**

Positive iff on that session Pickles **executed** or **explicitly claimed
to hold** a directional SPX 0DTE long call or put (i.e., a debit, opening
trade, not a credit spread, not a delta-neutral hedge).

"Execute" and "explicitly claim to hold" are narrower than "mention" — see
§4 (evidence_type).

## 2. Positive rules (label = 1)

A day is positive if **any** of the following is observed in the journal:

**P1 — `evidence_type = executed`.** An unambiguous announcement of entry
into an SPX 0DTE long call or put, with directional intent.

Examples of acceptable phrasings:
- "LONG SPX CALLS on ES VWAP SUPPORT..." (2023-12-14, 7:08 AM PT)
- "LONG SPX 0DTE 4510" (2023-11-17, 7:17 AM PT)
- "went LONG SPX CALLS on ES WEEKLY VWAP BOUNCE" (2023-11-09, 8:00 AM PT)
- "snuck in a few SPX LONG 0DTE CALLS before the 15 minute cool-down window" (2024-05-14, 6:13 AM PT)
- "going into LONG PUTS here" (2024-01-04, 7:03 AM PT)
- "SPX LONG PUTS" (2024-06-13, 7:44 AM PT)

**P2 — `evidence_type = explicitly_claimed`.** He did not announce the entry
in real time, but later refers to holding the position as an established
fact — "out of SPX CALLS", "my SPX longs", "those calls I took", etc.

**Phrase-level examples** (for pattern recognition; these appear on days
that already had earlier `executed` events, so the per-§4 resolution rule
makes those days `executed` at the row level):
- "out of SPX CALLS at HOD RE-TEST" (2023-11-20, 7:47 AM PT) — follows
  the earlier 7:28 AM `executed` entry "LONG 0DTE 4525 SPX CALLS" on the
  same day.
- "averaged down into these SPX LONG PUTS and finally closed out for
  profit" (2024-01-04, 9:32 AM PT) — follows earlier 7:03 AM `executed`
  entry.
- "out of SPX CALLS" on 2024-02-16 at 8:17 AM PT — follows earlier 7:03
  AM "LONG SPX for 1000 MAGIC TIME" `executed` entry.

**Day-row example** (the canonical case where `explicitly_claimed` *is*
the day's evidence_type): see Ex. 6 (2024-03-22) — *"only trade i made
today, was during 1000 MAGIC TIME"* is the only evidence the day has;
there is no real-time entry announcement.

**Directional intent is assumed from the side and timing.** If he says
"LONG SPX CALLS" during RTH while a directional thesis is active in the
narrative, that is directional intent. Bearish days with "LONG SPX PUTS"
also qualify — the target is directional-long (debit-long), not long-call
specifically.

## 3. Negative rules (label = 0)

A day is negative if **any** of the following holds (and no positive rule
applies):

**N1 — `evidence_type = narrative_only`.** He discusses the trade as a
plan, possibility, or retrospective without clear execution and without
later references-as-held.

Examples:
- "if PS CLOSE doesn't hold in here or in RTH I'll be looking at 5200 SPX
  0DTE LONGS" (2024-05-09, 5:38 AM PT) — this statement *alone* would be
  narrative_only. On the actual 2024-05-09, Pickles followed up at 7:21
  AM PT with an executed SPX LONG CALLS entry, so the **day** is label=1
  via the 7:21 entry, not the 5:38 note. The 5:38 note is a pure illustration
  of what narrative_only looks like in isolation.
- "looking at 4555 SPX CALLS should PRICE find a FLOOR" without follow-up.
- "was thinking about playing the 1000 MAGIC TIME reversal earlier but
  didn't" — explicit miss.

**N2 — No-trade day.** Personal, holiday, or declared sit-out.

Examples:
- "have to take care of some personal matters. Won't be trading today.
  Managing WEEKLY IC depending on PA" (2023-10-24) — N2, also no SPX
  0DTE long mentioned at all.
- "Holiday break yea" (2023-12-18) — long thread of Q&A, no trading.
- "Sitting this week out entirely from trading wouldn't be a bad choice"
  (2023-11-30 pre-market commentary) — **however**, Pickles then went on
  to trade SPX LONG CALLS on 11-30, so that day IS label=1 despite the
  morning sit-out sentiment.

**N3 — Sit-out / chop / manage existing only.** No new SPX 0DTE long
opened; existing positions managed.

Example:
- 2024-02-09: only pre-market narration and a "great job weekly rangers"
  at 2:14 PM. No entry during RTH. The weekly IC is managed but no
  directional debit long touched. Label=0.

**N4 — Only credit spreads / theta-gang / iron condors / flies.** If
0DTE activity on the day is only credit-spread based, label=0.

Example:
- 2023-10-31 7:53 AM PT: "0DTE SPX PCS 4160 / 4145. 3,540 - 1,500 = $2,040
  CREDIT RECEIVED" + 9:55 closed for profit. That's a credit spread, not
  a directional debit long. If no other SPX 0DTE long mentioned that day,
  label=0. (In fact 2023-10-31 has a 7:24 AM "beautiful 1005 EST reversal
  trade" reference that was in trader-chat; if the journal alone is our
  corpus, and there's no `executed` or `explicitly_claimed` SPX 0DTE long
  *in the journal*, label=0 with confidence=Medium-Low.)

**N5 — Hedge-only 0DTE long (`hedge_only = True`).**

`hedge_only=True` **only when the SPX long is explicitly described as
offsetting or restructuring an existing SPX short option / spread
exposure.** Repairing PnL from futures losses alone does NOT make it
hedge-only (see Ex. 7 below). If both motives are present and you cannot
cleanly separate them, send the row to shadow (Low confidence).

The discriminator is structural, not motivational: the SPX long must be
paired with an existing SPX short leg in a spread or credit-defense
structure for `hedge_only=True`.

Example:
- 2024-02-22, 11:13 AM PT: *"LONG SPX 4085 to effectively convert into a
  back ratio'd (~2:3) 10-wide CDS with the 4095 SHORT CALLS from the
  weekly range."* The 4085 long is explicitly paired with the 4095
  SHORT CALLS — an existing SPX short-option exposure. `hedge_only=True`,
  `evidence_type=executed`, **label=0** by this rule.

**N6 — Futures-only scalping.** Day contains ES/NQ/RTY futures entries
but no SPX 0DTE long.

## 4. `evidence_type` values

| Value | Meaning | Counts as positive? |
|---|---|---|
| `executed` | Announces entry in real time with clear side/instrument | Yes |
| `explicitly_claimed` | Later refers to the position as held (no real-time entry) | Yes |
| `narrative_only` | Described as plan, possibility, or missed trade | No (→ label=0) |
| null | No mention of SPX 0DTE long at all (e.g. no-trade day, spread-only day) | No (→ label=0) |

**Resolution rule.** If a day has multiple 0DTE-long mentions, pick the
**first** qualifying one. If the first is `narrative_only` but a later one
is `executed`, use `executed` and record the `first_qualifying_time_et` as
the later executed one (because narrative_only does not qualify a day as
positive). If all mentions are `narrative_only`, label=0.

## 5. Confidence values

**High.** The journal text is explicit enough that two independent reviewers
would reach the same label without discussion. Both side and timing of the
qualifying trade are clear, and the `evidence_type` classification is
unambiguous. Required: non-empty `excerpt`.

**Medium.** The label is defensible but some field is fuzzy:
- `first_qualifying_time_et` is approximate (window, not minute)
- Side inferrable from context but not stated on the entry line
- Rapid flip between narrative and execution
- He refers to "those calls" without clear prior entry

**Low.** The label depends on interpretive reconstruction across multiple
entries, or ambiguity cannot be cleanly resolved (e.g. "1 scalp then done"
with no instrument named). **Low rows go to `tier1_labels_shadow.csv` and
are excluded from primary training.** They are used only for the sensitivity
re-run (plan §Sensitivity).

## 6. Curation template

Every row in `v2/fork_c/tier1_labels.csv` has these columns in this order:

| Column | Type | Rule |
|---|---|---|
| `date` | YYYY-MM-DD | Must parallel `v2/data.pt` dates. |
| `label` | 0 or 1 | Per rules §2/§3. |
| `confidence` | `High` / `Medium` / `Low` | Per §5. Low goes to shadow file. |
| `evidence_type` | `executed` / `explicitly_claimed` / `narrative_only` / empty | Per §4. Empty if no 0DTE-long mention. |
| `first_qualifying_time_et` | `HH:MM` or empty | ET clock time (not PT). Empty iff label=0. |
| `side` | `call` / `put` / `both` / empty | First qualifying side. **Usually non-empty iff label=1**, but may be empty on a positive row when the side is genuinely unrecoverable from the journal (see Ex. 6: *"only trade i made today was during 1000 MAGIC TIME"* confirms label=1 but names no side). The preflight audit counts null-side positives separately so the downstream consumer can decide whether to drop them or treat them as "unknown-side positives". |
| `hedge_only` | `True` / `False` | True forces label=0 per N5. |
| `excerpt` | string | Verbatim journal quote (with journal's own timestamp). Required for High+Medium. |
| `journal_ref` | `{filename}:{line_start}-{line_end}` | Locator inside the source file. |
| `reviewer` | string | Curator identifier. |
| `notes` | string | Any judgment calls or non-obvious reasoning. |

## 7. Worked examples

Split into two subsections. **§7.a is real journal entries** (with PT
timestamps as they appear in the source; ET equivalents shown for reference).
**§7.b is synthetic representative patterns** — patterns a curator will
likely encounter but for which the real corpus in my context did not
supply a clean isolated illustration. Synthetic examples are labeled
explicitly and carry no `journal_ref`.

### 7.a — Real journal examples

### Ex. 1 — Clean executed positive (High confidence)

**Journal (2023-12-14, Pickles):**
> 12/14/2023 7:08 AM — "LONG SPX CALLS on ES VWAP SUPPORT, quick in & out,
> not expecting a re-visit to AM HIGHS, TP at +1 VWAP / OPENING CANDLE HIGH"

**Label row:**
- `date=2023-12-14, label=1, confidence=High`
- `evidence_type=executed, first_qualifying_time_et=10:08, side=call`
- `hedge_only=False`
- `excerpt="LONG SPX CALLS on ES VWAP SUPPORT..."`
- `notes="Canonical Row-1 template. Later SL'd (10:41 ET)."`

### Ex. 2 — Executed put positive (High)

**Journal (2024-01-04, Pickles):**
> 1/4/2024 7:03 AM — "going into LONG PUTS here. Green TICKS are weak on
> this run-up."

**Label row:**
- `date=2024-01-04, label=1, confidence=High, evidence_type=executed`
- `first_qualifying_time_et=10:03, side=put, hedge_only=False`

### Ex. 3 — Hedge-only negative (High) — same-day 0DTE long but forced to 0

**Journal (2024-02-22, Pickles, 11:13 AM PT):**
> "LONG SPX 4085 to effectively convert into a back ratio'd ( ~2:3 )
> 10-wide CDS with the 4095 SHORT CALLS from the weekly range. The DELTA
> from the 4085's will out-pace the DELTA from the 4095's, the defense
> will result in profit from a range break"

**Label row:**
- `date=2024-02-22, label=0, confidence=High, evidence_type=executed`
- `hedge_only=True` — forces label=0 per N5
- `first_qualifying_time_et=empty, side=empty` (per curation template:
  empty iff label=0; the hedge trade is recorded in `notes` + `excerpt`
  for audit, not in the qualifying-time columns)
- `excerpt="LONG SPX 4085 to effectively convert into a back ratio'd (~2:3)
  10-wide CDS with the 4095 SHORT CALLS..."`
- `journal_ref="2024-02-22.txt:43"`
- `notes="N5 applies: the 4085 long is paired explicitly with an existing
  4095 SHORT CALLS spread leg — structural SPX-long-offsetting-SPX-short.
  hedge_only=True."`

Commentary: this is the exact edge case the rules must handle unambiguously.
A naive keyword parser would flag the 4085 LONG as a positive; the
`hedge_only` column plus N5's structural test (SPX long paired with existing
SPX short leg) are how we prevent that. Contrast with Ex. 7 (repair-mode
futures-offset) which is label=1.

**Note on 2023-12-29** — an earlier draft used 2023-12-29's "17k 4515c
0dte calls as COUNTER DELTA" as the hedge-only example. That text is
Pickles' commentary on JPM's (someone else's) roll trade, NOT his own
execution, and therefore does not belong in this corpus's label at all.
Any date containing only third-party-trade commentary gets `label=0,
evidence_type=empty` with a `notes` flag.

### Ex. 4 — Explicit no-trade negative (High)

**Journal (2023-10-24, Pickles):**
> "have to take care of some personal matters. Won't be trading today.
> Managing WEEKLY IC depending on PA"

**Label row:**
- `date=2023-10-24, label=0, confidence=High, evidence_type=empty`
- `hedge_only=False`
- `notes="N2 (no-trade personal). Weekly IC management is not a new SPX
  0DTE long."`

### Ex. 5 — Sit-out / spread-only negative (Medium)

**Journal (2024-02-09, Pickles):**
> 5:21 AM — pre-market narration, ES levels, weekly rangers.
> 2:14 PM — "great job weekly rangers!! see you next week!"
> No RTH entries narrated.

**Label row:**
- `date=2024-02-09, label=0, confidence=Medium, evidence_type=empty`
- `hedge_only=False`
- `notes="N3 (sit-out). Confidence Medium because absence-of-entry is
  inferred from absence-of-narration, not an explicit no-trade statement.
  Parser should verify no intra-day SPX 0DTE long line exists."`

### Ex. 6 — One-trade-only positive (High, rare-execution day)

**Journal (2024-03-22, Pickles):**
> 11:41 AM — "only trade i made today, was during 1000 MAGIC TIME. apologies
> for late post"

**Label row:**
- `date=2024-03-22, label=1, confidence=High, evidence_type=explicitly_claimed`
- `first_qualifying_time_et=10:00` (MAGIC TIME window; precise minute unknown)
- `side=unknown` → leave empty if side not recoverable, note the gap
- `hedge_only=False`
- `notes="Timing is the MAGIC TIME window (09:55-10:10 ET). Exact minute and
  side not stated in journal. Side=empty; the parser must be tolerant of
  missing side when confidence=High and evidence_type=explicitly_claimed."`

Commentary: this is a case where we keep High confidence on the label
(he explicitly states he traded) but accept null side. The preflight
audit will count null-side rows separately.

### Ex. 7 — Executed positive with repair-mode context (High)

**Journal (2024-02-16, Pickles):**
> 7:03 AM — "LONG SPX for 1000 MAGIC TIME"
> 8:05 AM — "I'm in a 'repair a lost trade' mode rather than go for the
> win mode. risky to throw more capital at a losing trade to come out at
> break-even for the day"
> 8:17 AM — "out of SPX CALLS"

**Label row:**
- `date=2024-02-16, label=1, confidence=High, evidence_type=executed`
- `first_qualifying_time_et=10:03, side=call, hedge_only=False`
- `notes="Repair-mode context suggests the directional call was partly
  offsetting NQ losses. It is still a directional SPX 0DTE long entry, not
  a spread hedge. Label=1. hedge_only=False because the SPX long is not
  paired with an offsetting SPX short leg."`

Commentary: repair-mode is a judgment call. The rule: `hedge_only` is
reserved for trades that pair an SPX long with an offsetting SPX short
leg in a spread structure. A directional SPX long used to offset **futures**
losses is still directional — positive.

*(Ex. 8 moved to §7.b — synthetic.)*

### Ex. 9 — Complex positive with multiple instruments (High)

**Journal (2024-05-14, Pickles):**
> 5:41 AM — "LONG ES off PPI candle being over 50% bottom wick"
> 6:13 AM — "snuck in a few SPX LONG 0DTE CALLS before the 15 minute
> cool-down window, will TP likely within OPENING DRIVE"
> 6:32 AM — "out of SPX LONG CALLS"
> 7:39 AM — "0DTE PCS" (a put credit spread)

**Label row:**
- `date=2024-05-14, label=1, confidence=High, evidence_type=executed`
- `first_qualifying_time_et=09:13, side=call, hedge_only=False`
- `notes="First qualifying SPX 0DTE long is the 6:13 AM PT entry. The
  7:39 AM PCS is a credit spread (doesn't count as SPX 0DTE long per rules).
  The ES long is futures, not SPX options."`

### Ex. 10 — Averaged-down/recovery-mode positive (Medium)

**Journal (2023-11-30, Pickles):**
> 6:24 AM — "going to set-up a lightning quick SPX LONG CALLS at market
> open. literally in & out less than 120 seconds"
> 6:31 AM — [entry image/note]
> 11:29 AM — "spent all day averaging down into my SPX LONGS from this
> morning to force a win off this PM bull-run..."

**Label row:**
- `date=2023-11-30, label=1, confidence=Medium, evidence_type=executed`
- `first_qualifying_time_et=09:31, side=call, hedge_only=False`
- `notes="Executed at market open (09:30-09:31 ET). Day had extensive
  add-downs; we count the first qualifying entry, not the last. Medium
  confidence because the initial entry timestamp is inferred from the
  6:24 AM plan + 6:31 AM note rather than an explicit entry-line ET time."`

### Ex. 11 — *(moved to §7.b — synthetic)*

### Ex. 12 — Weekend / non-trading-day prep post (not curated)

**Journal (2023-10-22, Sunday):**
> Sunday commentary, BB signals, "current weekend positions."

**No row in `tier1_labels.csv`.** Weekend/market-closed days are filtered
out before curation. Only RTH trading days parallel to `v2/data.pt` sessions
are curated.

### 7.b — Synthetic representative patterns

These are patterns a curator will likely encounter but that my context
did not supply a clean isolated real example for. They are NOT drawn from
an actual journal. Any synthetic row used in production curation would be
flagged and removed — these exist only to document the target behavior.

### Ex. S-1 (was Ex. 8) — Narrative-only negative

**Journal (SYNTHETIC, formatted like Pickles' style):**
> 6:45 AM — "Looking at 4555 SPX CALLS should PRICE find a FLOOR today"
> (No follow-up. No later entry, no "out of", no reference to the calls
> as held.)

**Label row:**
- `date=YYYY-MM-DD, label=0, confidence=High, evidence_type=narrative_only`
- `first_qualifying_time_et=empty, side=empty, hedge_only=False`
- `journal_ref=N/A (synthetic)`
- `excerpt="Looking at 4555 SPX CALLS should PRICE find a FLOOR today"`
- `notes="SYNTHETIC illustration of N1 narrative_only. No entry narrated,
  no follow-up. Day counts as 0 despite the SPX-calls mention. Pure
  narrative_only days may be rare in the real corpus (most 'looking at'
  phrases lead to eventual execution — see 2024-05-09 for a day where
  a narrative-at-5:38 was followed by execution-at-7:21)."`

### Ex. S-2 (was Ex. 11) — Side ambiguous → Low → shadow

**Journal (SYNTHETIC, real wording pattern):**
> 10:14 AM ET — "scalped SPX 0DTE, done for the day"
> (No side stated. No strike. No excerpt revealing call vs put. No follow-up.)

**Label row (goes to shadow file):**
- `date=YYYY-MM-DD, label=1, confidence=Low, evidence_type=executed`
- `first_qualifying_time_et=10:14, side=empty, hedge_only=False`
- `journal_ref=N/A (synthetic)`
- `notes="SYNTHETIC illustration. Executed SPX 0DTE scalp claimed, but
  side and strike absent. Low confidence per §5. Shadow file; excluded
  from primary training."`

## 8. Process for ambiguous cases

1. **First pass:** apply rules §2/§3 mechanically based on journal text.
2. **Ambiguous?** Use §4 `evidence_type` decision: if entry is real-time
   and explicit → `executed`. If post-facto reference-as-held → `explicitly_claimed`.
   If plan/possibility/missed → `narrative_only`.
3. **Still ambiguous?** Mark `confidence=Low`, populate `notes` with the
   specific ambiguity, send to shadow file.
4. **If two curators disagree:** the row goes to shadow file regardless of
   either's individual confidence — disagreement itself is the Low signal.

## 9. Change log

| Date | Change | Rationale |
|---|---|---|
| 2026-04-19 | Initial version. | Lock-before-parse per plan §Step-1. |
| 2026-04-19 | Revision 1: (a) Ex. 3 replaced — 2023-12-29 is JPM commentary, not Pickles' own trade; swapped for 2024-02-22 11:13 AM clean hedge-only case. (b) `explicitly_claimed` §2 examples relabeled as phrase-level with a pointer to Ex. 6 as the canonical day-level case. (c) §7 split into 7.a (real) and 7.b (synthetic); Ex. 8 and Ex. 11 moved to 7.b as S-1 and S-2. (d) N5 tightened: hedge_only=True only when SPX long is paired with an existing SPX short leg; futures-repair stays label=1. | Codex critique round 3. No curated labels affected (file pre-curation). |
| 2026-04-19 | Revision 2: Timestamp handling section rewritten. Exported Discord timestamps are Pacific-localized display (must +3h → ET); explicit inline ET references (e.g. "1000 MAGIC TIME", "1030 IB") are already ET and must NOT be reconverted. Avoids the curation-mistake class Codex flagged. | Codex critique round 4. File pre-curation. |
| 2026-04-19 | Revision 3: §6 curation template — `side` field rule updated to explicitly allow empty on positive rows when unrecoverable, per Ex. 6 precedent. Aligns template with the worked-example it already accepts. | User review of first-pass proposed labels flagged the inconsistency between §6 ("Empty iff label=0") and Ex. 6 (empty side on label=1). File during curation — no existing labeled rows invalidated (the existing Medium null-side rows now validate under the corrected rule). |

Any future edit to this file invalidates all previously curated labels —
re-review required.
