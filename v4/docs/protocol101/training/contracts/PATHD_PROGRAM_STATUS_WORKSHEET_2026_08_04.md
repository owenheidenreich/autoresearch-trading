# Path-D Program Status Worksheet — where we actually are (2026-08-04)

**One line:** the machine is nearly built and nothing is wrong with it. What is missing is a **reason to
trade** — no edge has been found, and no market feature is yet certified to exist live at decision time.

Read with [`PATHD_BUILD_ORDER_2026_08_04.md`](PATHD_BUILD_ORDER_2026_08_04.md), which fixes the order these
must be done in.

---

## 1. The five capabilities, scored

| # | Capability | Status | Evidence |
|---|---|---|---|
| 1 | **Historical replay validation** | ✅ **WORKS** | four-box replay, maxT, negative controls, bootstrap LCB all operational and reproducible |
| 2 | **IB Gateway live execution** | ✅ **PROVEN** | guarded round trip 2026-08-04 on `DU***40`; contract-identity bridge confirmed |
| 3 | **Databento OPRA live decisions** | 🟡 **PARTIAL** | 60 s smoke capture passed; 3-session arrival-clock capture armed for 08-05..07 |
| 4 | **Certified feature set** | 🔴 **8 / 73** | all 8 are calendar/geometry; **zero market observables admitted** |
| 5 | **Discovered, approved edge** | 🔴 **NONE** | four independent negatives (§4) |

**Entry model:** not built. Cannot be — no market feature is admitted, and the enforcement layer refuses to
train on a non-admitted feature.
**Exit model:** exists but is degenerate (exits at the first step in 95.1% of trajectories). Its
owner-authorized repair wave returned `NO_EDGE` on all six trained arms.

---

## 2. What is genuinely proven

These are settled and do not need redoing.

**Execution plane (IBKR).** Paper port 4002 open, 4001 (live) correctly closed. Account `DU***40` confirmed
paper, `real_money_trading: false`. SPX spot read live. A real 0DTE contract qualified as
`SPXW  260804C07775000` — **that `localSymbol` matches the OSI format in the training corpus exactly**, so
the Databento-training → IBKR-execution contract bridge is confirmed live. That is precisely what June's
parity crisis was about, and it is now closed.

**Guard layer.** Refuses a live (`U`-prefix) account outright, refuses without the env flag, refuses
without each acknowledgement. Verified by direct exercise, not just unit tests.

**Fee truth-up.** Measured **$1.54/side = $3.08 round trip**. The 2026-07-19 estimate of $2.80–3.00 was
correct; the frozen `FILL_LAW` $3.00 overlay slightly *under*charges. (The `$0.65` figure is the IBKR
commission line item only — one of three components. Correcting that misread is done.)

**Databento live path.** Definitions + market capture executed end to end. Real arrival clock observed:
**cbbo-1m p99 584.6 ms**, cbbo-1s 504.8 ms, ohlcv-1m 194.7 ms.

**Research machinery.** Executable gate (four rejection tests + four charter diagnostics), bounded
autoresearch loop with maxT and semantic dedup, and a prior-art check that blocks closed mechanisms. All
tested against known-bad artifacts where they must fail — and they do.

---

## 3. What is not proven

**No market feature is certified.** 8 of 73 admitted, all calendar and contract geometry:
`day_of_week`, `minute_of_session`, `strike`, `is_call`, and four clock fields. **No price, no spread, no
volatility, no greeks.** A model on this set can learn a date rule, not a strategy.

**65 features are blocked behind one missing receipt** — `cbbo1m_native`'s multi-session arrival
distribution. That capture is armed for 08-05..07.

**Account state (6 features) has passing parity receipts but no arrival clock.** Deliberately not admitted:
admitting on an unmeasured clock is the exact defect corrected in Track B, where `implied_spot` carried a
4.898 ms local *compute* p99 while its inputs arrive at ~585 ms — a 119× error.

**Open interest (and 9 others) can never be used.** `BARRED` permanently: no live source with matching
decision-time semantics.

---

## 4. The edge question — four independent negatives

None of these is fixed by any of the others.

| # | Finding | Number |
|---|---|---|
| 1 | 0DTE long premium is negative-EV **before any cost** | −$13.00/trade gross, negative 5/5 folds |
| 2 | The 18-feature contract has **zero ranking power** | deciles: top −$15.37, **middle −$10.79**, bottom −$16.32 |
| 3 | The validated exit repair **failed** | 6/6 trained arms lost; maxT p 0.967–0.994 |
| 4 | Outcome shape is **incompatible** with the Charter's risk profile | 23–69% big-loss share vs a 2% limit; scratch 3–15% vs 73% target |

**The one thing that survived every attack:** horizon dominates instrument choice. The same option held
60 minutes rather than 1 clears a far lower bar on all three estimators. That is why Wave 3 exists.

**The one live lead:** 10:30–10:59 ET shows **+$5.99 mean gross**, positive in **all five delta bands** — so
it is a time effect, not a strike artifact. Net is still −$20.08. It came from an 11-way scan, so it is
pre-registered as a single confirmatory window, and it is a quasi-replication, not a confirmation.

---

## 5. The dependency chain — what actually blocks what

```
  [Databento 3-session capture]  08-05..07, armed
            │
            ▼
  cbbo1m_native ADMITTED ──► cross_section (11)
            └──────────────► implied_spot (8) ──► IV (6) ──► greeks (7)
            │
            ▼
  ~65 / 73 features certified
            │
            ▼
  Wave 3 re-frozen on ADMITTED features only
            │
            ▼
  Entry + wait model  ·  Exit + hold model      ◄── needs an EDGE, which we do not have
            │
            ▼
  Composed trader ──► live-shadow ──► guarded paper ──► real money (separate owner packet)
```

**Everything sits behind one capture.** Separately, account state needs one instrumented round trip to
measure fill-notification latency — small, and it can ride along with any future paper session.

---

## 6. Honest bottom line

**We have a trading system with no trade to make.**

The plumbing is real: data in, decisions out, orders filled, all guarded and auditable. The research
machinery is honest enough to have produced four negatives and to have caught its own errors repeatedly —
including a 60-second look-ahead, a 119× clock error, a re-run of closed experiments, and a fee figure
wrong by 2.4×.

What has never been demonstrated is an edge. Every campaign to date has returned `NO_EDGE`, `UNDERPOWERED`,
`NO_SIGNAL`, or `INVALID`. That is not a failure of the machine; it is the machine reporting honestly on a
strategy class that has so far not shown one.

**The realistic near-term outcome is a fifth negative.** Wave 3's own pre-registration records the prior
that `NO_SIGNAL` is more likely than not. The value of continuing is that we would now learn it *honestly
and cheaply*, on certified features, with a live path already proven — rather than after another campaign
built on assumptions.

**What would change the picture:** a feature family with actual ranking power. Nothing in the current
18-feature contract has it, and adding certified greeks and microstructure is the first genuinely new
information the model will have had.

---

## 7. Immediate queue

| When | What | Needs |
|---|---|---|
| 08-05..07, automatic | Databento 3-session arrival-clock capture | nothing — armed, gate verified |
| after capture | Certify cbbo1m_native → compose Track-B clocks → regenerate ledger | Codex (back 08-08) or me |
| after ledger | Re-freeze Wave 3 against ADMITTED features only | Wave 3 as written specifies uncertified greeks |
| any paper session | Instrument fill-notification latency → close Track C | owner authorization |
| after 08-07 | **Uninstall the launchd jobs** — temporary, not a standing service | owner |

*Signed: Claude Opus 5 — 2026-08-04.*
