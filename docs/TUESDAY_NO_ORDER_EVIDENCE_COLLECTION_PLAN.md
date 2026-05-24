# Tuesday Paper Fill Observation Collection Plan

Target session: 2026-05-26, America/Los_Angeles launchd times.

This plan is designed to answer the current truth-gate blockers without training,
threshold tuning, protected-holdout scoring, live-money trading, or production
default changes. It opens IB Gateway, collects broker-connected market data,
submits bounded IBKR paper-only fill probes for Protocol101 selected or
near-selected candidates, then regenerates readiness summaries after the session.

## Schedule

| Time PT | LaunchAgent | Action | Order risk |
|---:|---|---|---|
| 06:05 | `com.autoresearch.tuesday.ibgateway.paper` | Start IB Gateway paper through IBC and hold API availability. | No order script. |
| 06:20 | `com.autoresearch.tuesday.protocol101.paper-preflight` | Wait for IBKR API and entitlement probes. | No order script. |
| 06:31 | `com.autoresearch.tuesday.paper-fill-observation` | Run premium-blend no-order surface check, then bounded Protocol101 paper fill probes until caps or market close. | Paper-only approval flags are required; max 8 probes and max 3 filled round trips by default. |
| 13:15 | `com.autoresearch.tuesday.evidence-postsession` | Regenerate fill, no-order parity, holdout, section, and neural training readiness packets. | No broker/order work. |
| 13:35 | `com.autoresearch.tuesday.ibgateway.paper-shutdown` | Stop the paper Gateway stack. | No order script. |

## Blocker mapping

| Blocker | Tuesday evidence path | What can pass Tuesday | What cannot pass without human approval |
|---|---|---|---|
| Calibrated stochastic fill model: `0 / 30` observations | Protocol101 selected or near-selected candidates are submitted as one-contract IBKR paper probes at intended ask entry, with cancel/timeout and immediate paper exit attempts. | A bounded execution-truth packet: fill/cancel/timeout outcomes, quote-age truth, entry/exit prices, latency, and round-trip PnL where filled. | A statistically calibrated stochastic fill model still requires more observations across regimes; Tuesday is meant to answer whether the strategy is executable enough to keep researching, not to fit a parametric fill model. |
| Untouched holdout availability | Post-session audit reruns `run_untouched_holdout_availability`. | Confirms whether reserved future unseen block exists/frozen after any already-collected artifacts. | It cannot invent future data or score protected holdout for exploratory selection. |
| Live no-order full-action parity | Protocol245 still runs a broker-connected no-order surface check before any fill probes; the paper-fill runner also reconstructs Protocol101 candidates from live quotes. | Full-action/no-order scaffold can move from "market closed" to observed live-session evidence if Gateway/data are available. | Protocol101/challenger promotion remains gated on post-session reconstruction and governance. |

## Installed artifacts

| Path | Purpose |
|---|---|
| `v4/ops/ibkr/run_tuesday_paper_fill_observation_collection.sh` | Date-guarded Tuesday market-data collection plus bounded paper fill probes. |
| `v4/scripts/run_tuesday_protocol101_paper_fill_observation.py` | Broker-connected IBKR paper-only Protocol101 selected/near-selected fill probe runner. |
| `v4/ops/ibkr/run_tuesday_post_session_truth_audit.sh` | Post-session readiness regeneration plus consolidated Tuesday evidence packet generation. |
| `v4/scripts/run_tuesday_no_order_evidence_packet.py` | Reads Protocol245, paper-fill observation outputs, and JSONL logs to decide whether Tuesday produced usable execution evidence. |
| `v4/ops/launchd/install_tuesday_no_order_evidence.sh` | Installs only the approved Tuesday LaunchAgents and disables overlapping order-enabled/duplicate collectors. |
| `v4/ops/launchd/com.autoresearch.tuesday.*.plist` | One-date launchd schedule for Gateway, preflight, paper-fill observation, post-session audit, and shutdown. |

## Human-confirmation-only items

- Approving IBKR two-factor authentication if Gateway requires it.
- Any paper-submit work beyond the bounded Tuesday observation caps.
- Any new model training, threshold tuning, challenger promotion, or protected-holdout scoring.
