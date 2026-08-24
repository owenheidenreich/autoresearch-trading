# The morning near-the-money asymmetry — v6's first real result

> **SUPERSEDED THE SAME DAY, AND THE ERROR IS THIS DOCUMENT'S.** The overshoot join
> ([`MORNING_ASYMMETRY_EXPECTANCY_NULL_2026_08_23.md`](MORNING_ASYMMETRY_EXPECTANCY_NULL_2026_08_23.md))
> answered the question §7 left open, and the answer is that **the asymmetry pays nothing**. Across 36
> cells over 1,011 sessions **no cell's 95% interval clears zero and 31 of 36 lose outright**; the best
> is 09:35 ATM at **+$2.01 on CI [−$34.61, +$36.98], median −$61.54**.
>
> **The asymmetry itself reproduces stronger, not weaker** — R=1.14 against J=10.94, won **79.8%**
> rather than 72%. **It is worth nothing because winning that race means crossing zero, not profiting.**
>
> **The analytical error, stated plainly: this document read a DISTANCE ratio as if it were a PAYOFF
> ratio.** A 12:1 gap between the stop and the break-even point is not 12:1 odds — it is two
> thresholds whose crossing probabilities the market has already priced. Wins are small and frequent,
> losses large and rare, and that is what a fairly-priced option looks like. Verified independently:
> held to the 20-minute horizon, arriving at +R returns **exactly $0.00** whenever it arrives.
>
> Everything below stands as measurement. Its framing as "the strongest result" does not.


**2026-08-23. Read this after `STATUS.md`. It is the current state of job 47 (v6) and the strongest
measurement this programme has produced. Nothing here is adopted or tradeable.**

---

## 1. What changed, in one paragraph

For the first time this project has a **mechanism-grounded reason to believe a specific trade shape
might work**, rather than a fitted number that dissolved on inspection. Buying a near-the-money SPX
0DTE option in the first hour requires SPX to move **1.2 points** in your favour to break even, while
a **−40% stop sits 14.5 points away**. You need to travel a **twelfth as far** in your favour as
against, and that race is won **72%** of the time. By 15:00 the same asymmetry is gone — 2.5 against
2.8, a coin flip at 49/47.

**This is not an edge yet.** Winning the race to break-even means *breaking even*. What is missing is
the overshoot beyond it, and until that is measured nothing here justifies a trade.

## 2. Why v6 exists

v5 produced hundreds of negative results and no bot. The diagnosis is not "no edge exists" — it is
four measured structural errors, all now addressed or understood:

1. **The option layer was fitted before direction was established.** The charter says option-layer
   training is prohibited until G1 direction passes. G1 never passed; it stopped **UNDERPOWERED, not
   falsified**, with a detection floor 22–88× the cost bar. Job 46 then spent a summer on chain
   internals anyway. A long call is a directional bet with negative carry.
2. **Occupancy made the research unfalsifiable.** Cost-scale detection needs ~8,778 sessions at
   once-daily and **338 at 26 trades/session**. The two-ticket risk law forced the unmeasurable regime.
3. **The bot never learned to wait** — 209 entries a session across the whole ladder.
4. **The exit destroyed value and the entry was trained under it.** Always-hold was −$1.74/trade
   against the bracket's −$24.18.

And a fifth, methodological: **the certification bar guaranteed a null.** 45–50% survival with signed
power analysis is academic; a working bot would have failed v5's framework.

## 3. The foundation v6 is built on

Three modules, all reading only what a live feed holds at decision time, so the same function runs on
both sides of the train/live boundary.

- **[`greeks.py`](../greeks.py)** — first-order greeks from price. **Hash-pinned** by the semantic
  freeze; not editable.
- **[`greeks_higher_order.py`](../greeks_higher_order.py)** — vanna, charm, vomma, speed, color, built
  *additively* because of that pin. Verified against numerical differentiation of the pinned pricer.
- **[`contract_economics.py`](../contract_economics.py)** — what SPX move a contract needs to break
  even, by repricing rather than Taylor expansion, at measured friction ($3.08 fees plus the spread
  crossed once).

**The finding that fell out of building them:** charm accelerates into expiry **only near the money**.
A 10pt-OTM call goes −2.0e−4 to −1.2e−2 per minute between 240 and 10 minutes, sixtyfold. A 50pt-OTM
call *peaks* at 120 minutes and collapses to −1.6e−5 by 20 — **not because it is safe but because its
delta is already gone**. A model reading only the level sees a small charm on the cheap strike and
calls it stable. That is the mechanism behind `RIGHT IDEA, WRONG UNITS` and job 46's **$579 average
ticket**.

## 4. The result

Measured IV clock, 20-minute hold, pooled over 1,011 clean sessions, session-clustered intervals.
`R` is the break-even move; `J` the adverse move that trips the −40% stop.

| start | IV | strike | ask | R | J | win | lose | flat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **09:35** | 10.0% | ATM | $1,748 | **1.2** | **14.5** | **72%** | 10% | 18% |
| 09:35 | 10.0% | 10pt | $1,292 | 1.6 | 12.5 | **72%** | 10% | 18% |
| 11:30 | 8.9% | ATM | $1,314 | 1.2 | 10.4 | 67% | 9% | 24% |
| 13:30 | 8.9% | 10pt | $549 | 2.4 | 4.6 | 59% | 25% | 16% |
| 15:00 | 9.6% | ATM | $657 | 2.5 | 2.8 | 49% | 47% | 5% |
| 15:00 | 9.6% | 25pt | $54 | 8.4 | **0.0** | 8% | 59% | 32% |

**Three things worth reading twice.**

- **The cheap late contract is dead on arrival.** At 15:00, 25 points out, `J = 0.0` — spread and
  immediate decay have already breached a −40% stop **before SPX moves at all**. That $54 ticket
  cannot win.
- **Stops and winners coexist here.** Of 282 paths that eventually reached +10 from 09:31, **240 got
  there without touching −5 first (85.1%, CI 80.5–88.8)**. A −5 stop costs 15% of eventual winners.
- **It inverts what every v5 model did.** They bought **cheap, far, late and often**. The terrain
  rewards **expensive, near, early and rare**.

## 5. Corrections recorded, both from the executing session

- **The `$17.92` round trip is ES futures friction**, not options. Option friction is **$3.08 + the
  spread crossed once** = $13.08 ATM / $23.08 OTM. The code was right; the prose was not.
- **`330 minutes` is 10:30, not 09:31.** The open has 389 minutes; first decision 385.
- **The `$2,000` cap did not exclude the best cell — it excluded it on ~41% of sessions.** That claim
  used a flat 13% IV sitting near the 70th percentile of measured. At the **median 9.95%** the ATM ask
  is **$1,748, inside the old cap**. The 2026-08-23 raise to $2,500 cuts the binding share to ~24% and
  remains defensible, but the rationale is narrower than first written.

## 6. What is NOT established

- **No expectancy.** Reaching `R` is break-even. P&L needs the overshoot distribution beyond it.
- **Everything is unconditional.** This is the terrain a rule would operate on, **not a rule**. No
  signal has been tested and no alpha has been spent on one; the ledger stands at 6 with a 0.66059463
  bar.
- **The `flat` state is a full loss** to theta and runs 5–32%.
- **Era stability is unresolved** — source and date remain perfectly confounded.
- **The 1,014-session corpus is `NOT-USABLE`** for certification and there is no more 0DTE history to
  buy: the corpus already holds **93.3%** of every session that exists, because SPX had no daily 0DTE
  before 2022. **Calendar, not money, is the binding constraint.**

## 7. The next step

**Join the overshoot distribution to the race**, turning a break-even probability into an expected
value. The inputs exist: `overshoot_distribution.csv` and `conditional_quantiles.csv` from the
2026-08-23 race work, and `contract_economics.py` on this side.

After that, and only after: an admissible-set filter, then a stated rule, then forward paper on live
data. **Live OPRA renewal is paused**, so a live decision feed must be restored before anything runs
forward — and the parity work is against Databento live, not IBKR, which serves execution only.
