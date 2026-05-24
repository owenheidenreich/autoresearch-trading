# Assumption Registry

This registry tracks assumptions that can invalidate research conclusions.

Status values:

- `open`
- `partially_tested`
- `falsified`
- `supported`
- `accepted_risk`
- `blocked`

## High-Risk Assumptions

| ID | Assumption | Status | Importance | Fragility | Current evidence | Falsification path | Next artifact |
|---|---|---|---|---|---|---|---|
| A001 | Ask-entry/bid-exit replay approximates executable fills. | open | critical | high | Replay uses ask/bid; recent inspected paper logs show zero submitted orders/fills. | Paper/no-order fill table by side, premium, spread, quote age, latency, time bucket. | Verifier report |
| A002 | Live quote age semantics match historical quote age semantics. | open | critical | high | Historical and live quote-age construction differ across normalized/dataset/runtime paths. | Raw quote timestamp, received timestamp, decision timestamp, and guard-age diff audit. | Cartography report |
| A003 | Live candidate filtering matches historical training candidate filtering. | open | critical | high | Historical filters include spread/mid/size; live `_valid_quote` is looser. | Feature/candidate parity harness for matched historical/live-style rows. | Verifier report |
| A004 | Protocol066 lifecycle live input matches sequence-model training semantics. | open | critical | high | Inference doc requires full sequence; bridge builds one current row. | Replay known trade path through live row builder and compare actions at each step. | Verifier report |
| A005 | Protocol051 edge is real opportunity quality, not quote/liquidity artifact. | open | high | medium-high | Protocol101 depends on Protocol051 surface edge and narrow candidate filters. | Matched controls by spread, premium, moneyness, side, time, quote age, and fillability. | Experiment RFC |
| A006 | Protocol101 margin ranks expected realized utility. | partially_tested | high | medium | Selected-trade score calibration is weak/negative in forensics. | Same-event candidate calibration including rejected candidates and cost/fill assumptions. | Verifier report |
| A007 | Repeated exposed splits still predict future behavior. | open | critical | high | Overfit-risk audit marks major splits as repeated-research diagnostics. | Freeze candidate, score one future untouched block once after gates pass. | Decision memo |
| A008 | `$10k` account and reserve semantics match replay. | open | high | medium | Guard has `$500` reserve but affordability check does not subtract it. | Guard-identical account replay with explicit reserve treatment. | Verifier report |
| A009 | IBKR paper fills approximate live execution. | blocked | high | high | Paper-submit evidence is insufficient; real-money trading not authorized. | Compare paper fill/cancel data to approved live micro-observations only if later authorized. | Decision memo |
| A010 | Fees/commissions are negligible for current trade distribution. | open | medium | medium | Main labels use zero fees. | Fee sensitivity by side/premium/time/archetype. | Verifier report |

## Registry Rules

- New experiments must reference the assumptions they reduce or depend on.
- If an assumption is critical and open, no agent may use it as proven.
- If an assumption is falsified, affected claims must be queued in
  `DECISION_QUEUE.md`.
