# Addendum — the stop level, and distinguishing "no edge" from "could not see one"

**Status: SIGNED 2026-08-16 by the repository owner.** Instruction given in conversation: *"Sign …
as drafted. It binds a −40%-or-wider floor on any declared stop in job-46 fit declarations, and reads
§9.3 of the ticket/breaker amendment as requiring a powered negative — an underpowered evaluation
does not revert the cap and may not be reported as evidence against the strategy. §9.1 and §9.2 are
untouched."* Signed as drafted and unmodified, after independent verification of its evidence (§6).

This is an addendum to
[`CHARTER_AMENDMENT_TICKET_AND_BREAKER_2026_08_16.md`](CHARTER_AMENDMENT_TICKET_AND_BREAKER_2026_08_16.md),
signed 2026-08-16. It does not reopen the cap, the breaker, the account law, or §4's verification
requirement — all of which stand. It closes two gaps the signed text leaves open, both of which
would otherwise be settled silently by whoever writes the first fit declaration.

## 1. Why this exists

The signed amendment makes the $2,000 cap defensible by requiring a **declared stop** that actually
fires (§4). But it does not say what the stop level may be, and it makes the cap revert if the first
evaluation "shows no edge" (§9.3) without saying how an absent edge is told apart from a test that
could not have seen one. Both are load-bearing. A stop chosen carelessly destroys the strategy it is
meant to protect, and a reversion triggered by an underpowered test would withdraw the cap for a
reason the evidence does not support.

## 2. The stop level is bounded by measurement, not chosen freely

Measured on 32,976 quote-priced 60-minute paths over 251 owned sessions, mid-based, entry at the
ask:

**Normal behaviour is violent.** Median path drawdown is **−44.2%**, with quartiles at −66% and
−22%. A long same-day option routinely halves before doing anything at all.

**Catastrophe is rare.** A trade loses ≥90% at some point in **4.32%** of cases, ≥95% in **2.27%**,
and ≥99% in **0.34%**.

**A tight stop does not cut losers, it cuts winners.** Of the trades that eventually reach a given
gain, this share first dipped past the stop — every one of them a winner the stop would have ended
before it developed:

| Stop | Winners reaching +30% | Winners reaching +50% | Winners reaching +100% |
|---|---:|---:|---:|
| −20% | **58.7%** | **49.0%** | 35.8% |
| −30% | 42.4% | 32.3% | 21.6% |
| −35% | 35.6% | 26.2% | 16.3% |
| −40% | 29.2% | 20.7% | 12.6% |
| −50% | 19.1% | 12.8% | 7.7% |

A −20% stop destroys **half of every trade that would have doubled-digit won**. This is not a new
finding — it is the mechanism that made the closed −20%/−30% stop family move the break-even bar
*up* by 15 points (do-not-retest ledger, 2026-08-13). This addendum prevents its rediscovery.

**Therefore:** a declared stop in any job-46 fit declaration must sit at **−40% or wider**
(more negative). A stop tighter than −40% is refused at review. Within that bound the level is a
declaration choice, and the measured trade-off above must be restated in the declaration rather
than inherited from this document.

**Two limits on the table itself, stated rather than hidden.** It measures whether a path *touched*
a level at any point in the 60-minute frame, which is a conservative upper bound on winners harmed —
a trade that dips to −41% at minute 3 and finishes +80% is counted as harmed by a −40% stop, and it
would be. And it is measured on random entries; a skilled entry selects a different population, so
these shares are a bound on the harm, not a forecast of it.

## 3. §9.3 reversion requires a powered negative, not merely a negative

§9.3 returns the cap to $1,000 if the first full evaluation "shows no edge." That is correct for a
result that measured an absent edge. It is wrong for a result that could not have detected an edge
of any plausible size — the two are different findings, and this project has confused them before at
real cost (G1 closed `UNDERPOWERED` twice, and job 45 closed at its own preflight for exactly this
reason).

**Therefore, §9.3 is read as follows:**

1. The evaluation must carry the **sensitivity statement** produced by the phase-3 known-answer
   preflight — the smallest effect the full gate can recover at the achieved corpus geometry.
2. If the measured result is negative **and** the preflight certifies the gate could have recovered
   an effect of the size the strategy needs, the finding is `NEGATIVE` and §9.3 fires: the cap
   returns to $1,000.
3. If the preflight cannot certify recovery at that size, the finding is `UNDERPOWERED`. **§9.3 does
   not fire.** The cap is unchanged, the result may not be reported as evidence against the
   strategy, and the deficiency is recorded as a measurement limit.
4. `UNDERPOWERED` is not a licence to continue indefinitely. Two consecutive underpowered
   evaluations require a fresh owner decision before further work on that member.

This narrows nothing the owner intended to keep: §9.1 (equity) and §9.2 (a breached maximum loss)
are untouched and fire on their own terms. §9.2 in particular remains unconditional — a realised
loss beyond the declared maximum is a failed risk check regardless of power.

## 4. What is unchanged

The $2,000 cap, the deep-ITM ceiling and `moneyness_band` bar, the 20% daily breaker, §4's
requirement that the declared stop fire in the simulator and that realised worst loss be checked
every packet, the serial compounding account, the deferred alpha charge, and every hard rail of
`DEVELOPMENT_CHARTER_2026_08.md`. This addendum adds two constraints and relaxes none.

## 5. Signature

Signing binds the −40% stop floor and the powered-negative reading of §9.3 on every job-46 fit
declaration.

- **Signed: repository owner, 2026-08-16**, as drafted and unmodified.
- Countersigned by Claude Opus 5 as to verification only: this addendum was drafted by a different
  session, so every quantitative claim was recomputed from source before the signature was recorded.
  See §6.

## 6. Independent verification before signature

This addendum was **not drafted by the session that signed it**, so its claims were recomputed from
`quoted_exit_paths.parquet` (32,976 trades, 251 sessions) rather than accepted. **Every figure
reproduces exactly.**

- Median path drawdown **−44.2%**; quartiles **−65.5% / −21.6%** — as stated.
- Catastrophe rates **4.32% / 2.27% / 0.34%** at ≥90% / ≥95% / ≥99% — as stated.
- All **fifteen cells** of the §2 stop-versus-winner table reproduce to the decimal.

The §2 footnote's self-limitation was tested rather than taken on trust. Under the stricter reading —
the stop firing **before** the gain is reached — the same population gives **29.3 / 16.7 / 12.0 /
8.4 / 3.8%** of +30% winners harmed at stops of −20 / −30 / −35 / −40 / −50%, against the **58.7 /
42.4 / 35.6 / 29.2 / 19.1%** in the table. The table is therefore a genuine upper bound, roughly
2–3.5x conservative, exactly as it claims. **The −40% floor survives either reading:** a −20% stop
harms 2.0x as many winners as −40% by the table's measure and 3.5x by the stricter one.

## Evidence

- Verification recomputation: 2026-08-16, recorded in the job-46 work log.

- Path statistics: `/Volumes/AR_TRADING_DATA/derived/quoted_exit_paths.parquet`, 32,976 trades,
  251 sessions, recomputed 2026-08-16.
- Closed tight-stop family: [`DO_NOT_RETEST.md`](../research/history/DO_NOT_RETEST.md), 2026-08-13
  scalp-exit row — take-20/stop-20 moved break-even from 54.35% to 69.62%.
- Underpowered precedents: [`DRAWDOWN_PREFLIGHT_2026_08_15.md`](../research/findings/DRAWDOWN_PREFLIGHT_2026_08_15.md)
  and the two G1 campaign closures.
