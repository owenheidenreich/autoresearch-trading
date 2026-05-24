# Current Trading Bot Improvement Questions

Audit date: 2026-05-24
Source reviewed: `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md`
Scope: questions to answer before changing the bot, changing defaults, tuning thresholds, or training/retraining any model.

This document deliberately does **not** propose model training. The diagnostics below are evidence-gathering, replay/parity checks, log audits, fill studies, and forensics using frozen artifacts/current code unless explicitly marked as requiring human-approved live/paper observation.

## Prioritization Principle

The highest-value questions are the ones that determine whether Protocol101 is a real, executable trading playbook or a replay artifact. The order below prioritizes execution realism, strategy identity, lifecycle behavior, missed winners/slot cost, train/replay/live parity, and validation credibility.

## 1. Is Protocol101's high-premium ITM quick-capture edge executable after real quote age, latency, and fill probability?

| Field | Detail |
|---|---|
| Why it matters | This is the core falsification test for the current default. If the edge depends on idealized ask-entry/bid-exit timing, architecture changes will not help. |
| Current evidence | Protocol101 appears to monetize high-premium, often ITM SPXW 0DTE contracts in short post-open/late-day windows. Forensics show timing fragility, and latest inspected paper logs had no submitted orders or fills. |
| Missing evidence | Real paper fill/cancel observations by archetype, true quote age, latency from quote to intent, and whether the selected ask was actually reachable. |
| Diagnostic that would answer it | Build an execution-realism packet from current logs plus human-approved paper observations: stratify candidate decisions by time bucket, side, premium, spread, moneyness, quote age, intended ask, actual submitted limit, fill/cancel outcome, and delay. No threshold changes. |
| Code/data/artifacts required | `v4/logs/paper_trading/**/*.jsonl`, `v4/live/ibkr_paper_executor.py`, `v4/live/ibkr_paper_guard.py`, Protocol101 manifest/scaler/model, broker paper fill/cancel logs if human-approved. |
| Confirms hypothesis if | The same archetypes that dominate replay show high fill probability with bounded slippage/latency and post-fill PnL remains positive after realistic cancel/miss treatment. |
| Falsifies hypothesis if | Selected asks are rarely fillable, quote age/latency erases edge, filled trades underperform the replay archetypes, or the strategy mostly trades only when historical quotes were non-executable. |
| Decision enabled | Keep Protocol101 paper observation as the default candidate for validation, or demote to no-order shadow/research until execution realism is proven. |

## 2. Are live quote ages real, or are freshness guards passing because runtime supplies placeholder ages?

| Field | Detail |
|---|---|
| Why it matters | The paper guard requires quote freshness, but the source-of-truth audit found runtime paths that can pass `quote_age_ms=0`. That can make stale quotes appear fresh. |
| Current evidence | `validate_order_intent` checks quote age, while the current live bridge may construct/persist option quote age as zero. Shadow schemas are stricter than the paper-submit path. |
| Missing evidence | Raw quote timestamp, received timestamp, decision timestamp, and computed quote age for every candidate and selected contract. |
| Diagnostic that would answer it | Run a read-only log completeness audit first. If missing, define a required observability field list for future human-approved no-order/paper sessions and compare computed age distributions against guard values. |
| Code/data/artifacts required | `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`, `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`, `v4/live/protocol101_live_entry.py`, paper JSONL logs, IBKR quote timestamp fields if available. |
| Confirms hypothesis if | Logged raw timestamps produce quote ages within guard limits and match persisted `quote_age_ms` for selected and rejected candidates. |
| Falsifies hypothesis if | Persisted age is zero or near-zero while raw quote timestamps are stale, missing, delayed, or unavailable. |
| Decision enabled | Trust or block paper-submit decisions that depend on quote freshness; prioritize observability before strategy changes. |

## 3. How much historical PnL survives a fill/cancel/slippage model grounded in paper observations?

| Field | Detail |
|---|---|
| Why it matters | Current replay uses deterministic ask-entry/bid-exit and stress slippage variants, not an observed fill model. Profitability may be a quote-crossing artifact. |
| Current evidence | `v4/sim/simulator.py` contains `NullSimulator`; `v4/sim/paper_replay.py` applies deterministic ask/bid and adverse slippage. Latest inspected logs show zero broker order endpoint rows. |
| Missing evidence | Empirical fill probability by spread, premium, side, time, quote age, and limit placement; cancel/timeout rates; partial/non-fill treatment. |
| Diagnostic that would answer it | Fit no model; instead produce a non-parametric fill table from paper observations and replay current selected trades under conservative bucketed fill assumptions. |
| Code/data/artifacts required | Paper order/fill/cancel logs, `v4/sim/paper_replay.py`, Protocol101 selected trade artifacts, historical quote paths, current guard config. |
| Confirms hypothesis if | Conservative bucketed fill assumptions leave Protocol101 positive in the same robust archetypes without relying on unfilled trades. |
| Falsifies hypothesis if | PnL concentrates in buckets with low fill probability, wide spreads, high quote age, or poor cancel outcomes. |
| Decision enabled | Decide whether replay profitability is worth further validation or whether simulator/fill evidence must precede all strategy work. |

## 4. What exact trade archetype is Protocol101 monetizing?

| Field | Detail |
|---|---|
| Why it matters | Improvements should target the real playbook, not an abstract "options bot." The current system may be a narrow high-premium timing strategy. |
| Current evidence | Source-of-truth notes high-premium ITM, post-open quick captures, median duration around 10 minutes, and specific allowed buckets `post_open_morning` and `late_afternoon`. |
| Missing evidence | Stable archetype decomposition across folds/recent periods using side, strike moneyness, premium, time bucket, spread, VIX/SPX context, exit reason, and fillability. |
| Diagnostic that would answer it | Build an archetype attribution table from frozen Protocol101 selected trades and current no-order/paper candidates; report PnL, hit rate, drawdown, fillability proxies, and concentration per archetype. |
| Code/data/artifacts required | Protocol101 summary/report CSVs, selected trade artifacts, `v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1/*`, paper logs, normalized option paths. |
| Confirms hypothesis if | A small number of archetypes repeatedly explain positive PnL across validation/test/recent and are observable live with feasible spreads. |
| Falsifies hypothesis if | PnL is diffuse, period-specific, dominated by one date/regime, or concentrated in archetypes that cannot be observed/fill-tested live. |
| Decision enabled | Preserve, restrict, or decompose the current playbook before considering any model architecture change. |

## 5. Are calls and puts separate strategies that should be evaluated separately?

| Field | Detail |
|---|---|
| Why it matters | A shared policy may hide that calls and puts have different timing, loss, liquidity, and exit behavior. |
| Current evidence | Forensics mention side-specific profiles; hard-stop examples skewed toward puts in inspected rows. Protocol101 uses shared candidate scoring and shared entry threshold. |
| Missing evidence | Side-stratified PnL, drawdown, fillability, score calibration, exit reason, quote age, and missed-winner behavior across the same splits. |
| Diagnostic that would answer it | Side-specific frozen-artifact audit: no new training, no new thresholds, just split existing selected/rejected candidates and logs by call/put. |
| Code/data/artifacts required | Protocol101 selected trades, rejected candidates if available, paper logs, `v4/live/protocol101_entry.py`, lifecycle reports, side strategy audit files. |
| Confirms hypothesis if | Calls and puts show materially different edge sources, failure modes, fill profiles, or lifecycle needs. |
| Falsifies hypothesis if | Side-stratified behavior is statistically and operationally similar after controlling for premium/time/moneyness. |
| Decision enabled | Decide whether future research should treat sides as distinct playbooks or keep one shared policy. |

## 6. Is post-open morning the actual edge, and is late-afternoon the same game?

| Field | Detail |
|---|---|
| Why it matters | Protocol101 hard-codes allowed buckets. If late-day and morning behavior differ, combined metrics can hide fragility. |
| Current evidence | Candidate generation allows `post_open_morning` and `late_afternoon`; reports point to post-open dominance and timing sensitivity. |
| Missing evidence | Separate profitability, fillability, quote coverage, spread, exit reason, and drawdown by allowed bucket and side. |
| Diagnostic that would answer it | Bucket-specific forensics with frozen entries: morning vs late-day, with identical cost assumptions and no threshold changes. Include no-order/paper live coverage by bucket. |
| Code/data/artifacts required | `protocol101_candidate_frame_from_surface`, `environment_diagnostics.time_bucket`, Protocol101 selected trades, paper logs, normalized context. |
| Confirms hypothesis if | One or both buckets are independently robust and live-observable with feasible spreads/fills. |
| Falsifies hypothesis if | Late-day or morning PnL is weak, concentrated, unfillable, or only survives when pooled with the other bucket. |
| Decision enabled | Decide whether bucket restrictions are valid current identity or need human review before further model work. |

## 7. Are high-premium ITM contracts a real edge or an affordability/fill trap?

| Field | Detail |
|---|---|
| Why it matters | Protocol101 seems to favor expensive contracts while paper account assumptions are around `$10k` and one contract. High premium can create affordability skips and low fill probability. |
| Current evidence | Forensics cite median Protocol101 premium around `$2,750`; Protocol276 had many unaffordable skips; guard blocks `qty * limit_price * 100 > cash`. |
| Missing evidence | PnL/fillability by premium and moneyness bucket under the exact current account guard, including reserve treatment. |
| Diagnostic that would answer it | Account-aware premium/moneyness replay using frozen selections and current guard logic, plus live candidate coverage/fill proxies per premium bucket. |
| Code/data/artifacts required | `v4/live/ibkr_paper_guard.py`, `v4/sim/protocol101_position_sizing.py`, Protocol101 selected/candidate artifacts, paper logs, option quotes. |
| Confirms hypothesis if | High-premium ITM buckets retain positive expectancy after affordability, spread, and fill constraints. |
| Falsifies hypothesis if | Apparent winners are unaffordable, rarely fillable, or fragile to small spread/latency stresses. |
| Decision enabled | Decide whether current playbook is account-compatible or whether account/risk assumptions must be resolved before strategy research. |

## 8. Does Protocol101 enter too early and block better later opportunities?

| Field | Detail |
|---|---|
| Why it matters | The one-slot constraint may be a core bottleneck. Improving entry may be less valuable than understanding opportunity cost from being in a mediocre open position. |
| Current evidence | Internal slot-cost diagnostics found 3,723 blocked Protocol101 entries and a large best-blocked-minus-open counterfactual. Replay skips while a position is open. |
| Missing evidence | Causal state at the time of blocked signals, fill-adjusted switch/hold alternatives, and whether blocked winners were knowable without future leakage. |
| Diagnostic that would answer it | Build a causal slot-cost packet: for every blocked candidate, log current open trade state at that timestamp, candidate features, executable quote state, and later realized outcomes under existing costs. |
| Code/data/artifacts required | Protocol101 internal slot reports, selected trade path data, rejected/blocked candidate streams, lifecycle step features, quote paths. |
| Confirms hypothesis if | Blocked winners occur when the open position has observable weak state, high giveback, or poor continuation value before the later signal. |
| Falsifies hypothesis if | Blocked winners are hindsight-only, unfillable, or not distinguishable from losers at blocked-signal time. |
| Decision enabled | Decide whether to investigate defer/switch/early-exit diagnostics or keep strict one-position holding behavior. |

## 9. Are hard-stop losses avoidable before entry, or are they unavoidable cost of the strategy?

| Field | Detail |
|---|---|
| Why it matters | If hard stops are avoidable regimes, the bot needs better guards; if unavoidable, tuning exits may just overfit. |
| Current evidence | Hard-stop rows lost materially in forensics; some had early MFE, and a loss autopsy classified many as unresolved. |
| Missing evidence | Pre-entry features, quote state, side/time/premium context, market regime, and path shape for hard-stop trades versus matched winners. |
| Diagnostic that would answer it | Matched-control hard-stop audit: compare each hard-stop entry to same-day/same-bucket candidates and nearby winners using only entry-time causal fields. |
| Code/data/artifacts required | Hard-stop autopsy CSVs/reports, Protocol101 features, selected/rejected candidates, normalized paths, context fields. |
| Confirms hypothesis if | Hard-stop trades share causal entry signatures absent in matched winners, such as spread, quote age, regime shock, side/moneyness, or adverse pre-entry movement. |
| Falsifies hypothesis if | Hard stops are not distinguishable before entry and only reveal themselves after adverse path movement. |
| Decision enabled | Decide whether to add a future pre-entry exclusion research question or focus on accepting/capping these losses. |

## 10. Are exits rational, inherited, or accidental?

| Field | Detail |
|---|---|
| Why it matters | Protocol101 entry value is inseparable from the lifecycle policy. The lifecycle stack is inherited from Protocol054/066/081, not purely Protocol101-owned. |
| Current evidence | Lifecycle is a hybrid of mandatory hard stop/target/time-flat and a learned residual sequence model. Live may pass only a single current row into a GRU-like sequence model. |
| Missing evidence | Exit reason attribution showing whether each exit improved or harmed PnL relative to causal hold alternatives after costs. |
| Diagnostic that would answer it | Exit counterfactual packet using frozen selected trades: for each exit reason, compare realized exit to causal next-step hold, target/stop/fallback, and forced-flat alternatives without retuning. |
| Code/data/artifacts required | `v4/live/protocol066_inference.py`, lifecycle artifact manifest, `build_lifecycle_sequence_dataset.py`, selected trade paths, paper/replay logs. |
| Confirms hypothesis if | Exit reasons consistently improve cost-adjusted outcomes in their intended states and live lifecycle inputs match replay semantics. |
| Falsifies hypothesis if | Exit reasons destroy value, trigger at inconsistent states, or depend on sequence state not available live. |
| Decision enabled | Decide whether lifecycle must be validated/repaired before any entry-model changes. |

## 11. Is the live lifecycle input equivalent to the replay/training sequence state?

| Field | Detail |
|---|---|
| Why it matters | A GRU trained on sequences can behave incorrectly if live runtime supplies only a single current row or mismatched state representation. |
| Current evidence | Source-of-truth marks lifecycle features as weak parity: training/replay use full sequences, live uses current state/single row. No direct parity test found. |
| Missing evidence | Feature-by-feature and action-by-action parity between replay lifecycle sequence state and live lifecycle row for the same simulated timestamp. |
| Diagnostic that would answer it | Build a read-only lifecycle parity harness: replay a known trade path, construct the exact live lifecycle row at each timestamp, and compare features/actions to replay sequence inference. |
| Code/data/artifacts required | `v4/live/protocol066_inference.py`, `v4/scripts/build_lifecycle_sequence_dataset.py`, lifecycle manifest/scaler, selected trade paths. |
| Confirms hypothesis if | Live row construction produces the same scaled features and lifecycle actions as replay for each causal timestamp. |
| Falsifies hypothesis if | Feature values, sequence lengths, masks, or resulting hold/exit actions diverge materially. |
| Decision enabled | Decide whether current lifecycle can be trusted in paper runtime or must be treated as unvalidated infrastructure. |

## 12. Are runner/giveback exits leaving a large, causal continuation edge on the table?

| Field | Detail |
|---|---|
| Why it matters | Track A forensics found runner-extension candidates, but post-exit best-path gains can be hindsight traps. This determines whether lifecycle improvement is worth studying. |
| Current evidence | Exact selected paths had many runner-extension candidates with large post-exit best deltas; forced-flat extension could also destroy value. |
| Missing evidence | Whether the continuation opportunity was identifiable at exit time using causal MFE/MAE/giveback/velocity/context fields. |
| Diagnostic that would answer it | Causal runner/giveback audit: at each actual exit, compute only state available at that timestamp and compare next-step hold distributions, not best future highs alone. |
| Code/data/artifacts required | Lifecycle step rows, selected trade paths, MFE/MAE/giveback formulas, exit reason logs/reports. |
| Confirms hypothesis if | Specific causal states predict positive continuation after costs without increasing tail loss excessively. |
| Falsifies hypothesis if | Apparent runner gains are only visible using post-exit hindsight or add unacceptable drawdown. |
| Decision enabled | Decide whether runner/giveback research is justified after parity and fill gates. |

## 13. Does the current stop/target/fallback policy match the actual distribution of Protocol101 trades?

| Field | Detail |
|---|---|
| Why it matters | Stop/target parameters inherited from earlier labels may not match Protocol101's selected trade archetypes. |
| Current evidence | Hard stop is `-0.50 * entry_ask * 100`, target is `+1.00 * entry_ask * 100`, fallback deadline is `min(entry+25m, 15:55)`. These are embedded in lifecycle dataset/logic. |
| Missing evidence | Distribution of MAE/MFE/time-to-best/time-to-loss for Protocol101 selected trades by side/time/premium. |
| Diagnostic that would answer it | Stop/target/fallback descriptive audit only: report where actual path extrema occur relative to inherited stop/target/deadline, without optimizing thresholds. |
| Code/data/artifacts required | `build_lifecycle_sequence_dataset.py`, Protocol101 selected trade paths, lifecycle reports. |
| Confirms hypothesis if | The inherited stop/target/deadline align with natural path distributions and robust archetypes. |
| Falsifies hypothesis if | Stops/targets/deadlines are systematically misaligned, e.g. frequent early MFE then stop or target rarely reachable before forced flat. |
| Decision enabled | Decide whether lifecycle assumptions are compatible with the current entry playbook. |

## 14. Are unaffordable candidates distorting research conclusions for a `$10k` one-contract account?

| Field | Detail |
|---|---|
| Why it matters | If profitable historical candidates exceed buying power, research metrics overstate what the current bot can trade. |
| Current evidence | Paper guard checks cash affordability; Protocol276 reported many unaffordable skips; the `$500` reserve is documented but may not be subtracted by guard affordability. |
| Missing evidence | Official Protocol101 replay metrics under the exact current paper guard, including starting cash, max quantity, max concurrency, and reserve treatment. |
| Diagnostic that would answer it | Guard-identical account replay for frozen Protocol101 decisions: apply current `validate_order_intent` affordability and concurrency semantics to historical selected candidates. |
| Code/data/artifacts required | `v4/live/ibkr_paper_guard.py`, paper enablement config, Protocol101 selected trades, replay account/sizing code. |
| Confirms hypothesis if | Most profitable selected trades are affordable under current rules and reserve semantics are explicitly accounted for. |
| Falsifies hypothesis if | PnL comes from trades the live/paper guard would block or from cash the runbook says should remain reserved. |
| Decision enabled | Decide whether current account assumptions invalidate replay claims. |

## 15. Does Protocol101's score margin actually rank expected value?

| Field | Detail |
|---|---|
| Why it matters | The entry model uses a wait-vs-enter logit margin and threshold. If margin is not monotonic with outcome, confidence-based changes are unsafe. |
| Current evidence | Source-of-truth cites weak selected-trade score calibration, including a negative/near-zero Spearman relationship in diagnostics. |
| Missing evidence | Same-event candidate-level calibration including rejected candidates, not only selected trades, under cost/fill assumptions. |
| Diagnostic that would answer it | Calibration audit with frozen logits: bin margins by side/time/premium/archetype and compare realized cost-adjusted PnL for selected and rejected candidates. Do not change threshold. |
| Code/data/artifacts required | Protocol101 model outputs/logits if logged or reconstructable, selected/rejected candidate frames, feature scaler/manifest, path outcomes. |
| Confirms hypothesis if | Higher margin reliably maps to better realized utility within comparable candidate groups. |
| Falsifies hypothesis if | Margins are non-monotonic, side/bucket-specific, or negatively related to realized utility. |
| Decision enabled | Decide whether margin can be trusted for monitoring, abstention analysis, or future challenger comparison. |

## 16. Are candidates excluded before Protocol101 sees them hiding most of the missed winners?

| Field | Detail |
|---|---|
| Why it matters | Protocol101 does not inspect the full option surface. It only sees Protocol051 candidates that pass edge/time filters. Missed winners may be upstream, not in the entry model. |
| Current evidence | Candidate generation filters on surface edge and time bucket, sorts by score, and keeps top 10. Challengers with broader action spaces sometimes improved results. |
| Missing evidence | Outcome distribution of candidates dropped by Protocol051 edge filter, time filter, and top-10 truncation, with execution and affordability constraints. |
| Diagnostic that would answer it | Pre-model exclusion audit: label each surface token as passed/failed by filter reason and compare post-cost outcomes by fail reason. |
| Code/data/artifacts required | `v4/live/protocol051_surface_edge.py`, `v4/live/protocol101_entry.py`, historical surface decisions, normalized quote paths, candidate reports. |
| Confirms hypothesis if | Dropped candidates contain stable, executable winners identifiable by causal features. |
| Falsifies hypothesis if | Dropped winners are rare, unfillable, unaffordable, or hindsight-only after costs. |
| Decision enabled | Decide whether improvement should target upstream candidate generation or leave it unchanged. |

## 17. Are the best research challengers finding real alpha or exploiting a broader, less realistic action space?

| Field | Detail |
|---|---|
| Why it matters | Protocol194/240/265 show interesting results, but promotion without runtime/fill/parity evidence would blur research with operational truth. |
| Current evidence | Several challengers beat Protocol101 in frozen reports, while Protocol276 underperformed and all are marked research-only/no default change. |
| Missing evidence | Apples-to-apples comparison under current runtime guard, fill realism, no-order parity, feature parity, and same validation governance. |
| Diagnostic that would answer it | Challenger attribution packet: for each challenger improvement, classify whether gain comes from new candidates, different lifecycle, account treatment, or unrealistic execution assumptions. |
| Code/data/artifacts required | Challenger reports/artifacts, Protocol101 baseline artifacts, paper guard, replay code, candidate/action traces. |
| Confirms hypothesis if | Challenger gains persist after identical execution/account/parity constraints and have clear causal attribution. |
| Falsifies hypothesis if | Gains disappear under current guard/fill assumptions or depend on candidates/runtime fields unavailable to Protocol101 paper mode. |
| Decision enabled | Decide which challenger ideas deserve later validation work and which should remain archived. |

## 18. Can every live/paper decision be reconstructed from logs alone?

| Field | Detail |
|---|---|
| Why it matters | Without full reconstruction, failures cannot be debugged and parity cannot be proven. |
| Current evidence | Source-of-truth says current logs include decisions and some market/account fields but not full feature tensors, all raw logits, all rejected candidate quotes, true timestamps, or latency breakdowns. |
| Missing evidence | A per-decision log completeness matrix proving enough fields exist to rerun candidate generation, scaling, inference, guard evaluation, and order intent. |
| Diagnostic that would answer it | Decision reconstruction audit: choose representative log rows and attempt to reconstruct model input, logits, selected contract, guard result, and order intent exactly from persisted data. |
| Code/data/artifacts required | Paper JSONL logs, Protocol101 manifest/scaler/model, surface/lifecycle artifacts, candidate logs, runtime source code. |
| Confirms hypothesis if | Reconstruction succeeds byte-for-byte or within deterministic tolerances for every inspected decision. |
| Falsifies hypothesis if | Missing feature, quote, timestamp, candidate, or artifact references prevent exact reconstruction. |
| Decision enabled | Decide whether observability must be improved before paper/live conclusions are trusted. |

## 19. Does live Protocol051-to-Protocol101 feature construction match replay for the same market state?

| Field | Detail |
|---|---|
| Why it matters | Entry parity can fail even if the same model artifact is loaded. Small feature mismatches can change logits and decisions. |
| Current evidence | Source-of-truth marks entry features high confidence but surface/lifecycle parity medium. Protocol101 has 55 features and a frozen scaler. |
| Missing evidence | Hash or tolerance comparison between replay-built and live-built feature rows for identical timestamp/contract/candidate sets. |
| Diagnostic that would answer it | Feature parity harness: feed frozen historical snapshots through the live construction path and compare features/scaled tensors/logits against replay artifacts. |
| Code/data/artifacts required | `v4/live/protocol101_entry.py`, `v4/live/protocol051_surface_edge.py`, Protocol101 manifest/scaler/model, historical surface rows, replay candidate frames. |
| Confirms hypothesis if | Candidate sets, feature values, masks, scaled tensors, logits, and actions match for the same inputs. |
| Falsifies hypothesis if | Any material mismatch appears in candidate ordering, edge values, history features, scaling, masks, or logits. |
| Decision enabled | Decide whether runtime/replay parity is strong enough to use replay metrics for paper decisions. |

## 20. Are live market data fields equivalent to historical fields used in replay?

| Field | Detail |
|---|---|
| Why it matters | Historical data may include fields or cleaned/repaired values not available live; live may have fields not present historically. This can break causality and parity. |
| Current evidence | Historical normalized parquet data and live IBKR SPX/VIX/SPXW NBBO probes exist. Missing/stale quote handling and Greeks repair are noted as caveats. |
| Missing evidence | Field-by-field replay/live availability table with null rates, timestamp semantics, bid/ask/size/OI/Greeks definitions, and contract identifiers. |
| Diagnostic that would answer it | Market reconstruction parity audit: compare historical and live snapshot schemas and missingness for the fields consumed by Protocol051/101/066. |
| Code/data/artifacts required | `v4/dataset/spxw_0dte_neural.py`, `v4/ingest/databento_opra.py`, live logs, normalized official context, feature manifests. |
| Confirms hypothesis if | Runtime-required fields are available live with matching semantics and acceptable missingness/staleness. |
| Falsifies hypothesis if | Key fields are historical-only, repaired post hoc, unavailable live, or semantically different. |
| Decision enabled | Decide whether current replay game is comparable to the live/paper data game. |

## 21. Are missing contract quotes systematically removing the best or safest opportunities?

| Field | Detail |
|---|---|
| Why it matters | Live ladder coverage gaps can bias the candidate set and invalidate no-entry conclusions. |
| Current evidence | Logs/monitoring mention `no_valid_spxw_nbbo_quotes`; Protocol276 had thousands of missing-contract-quote skips. |
| Missing evidence | Quote coverage by strike/moneyness/side/time, especially around Protocol101 candidate windows and selected/rejected archetypes. |
| Diagnostic that would answer it | Coverage heatmap from no-order/paper logs: expected contracts versus valid NBBO quotes by minute, side, strike distance, premium, and time bucket. |
| Code/data/artifacts required | Live option quote logs, contract ladder construction code, paper logs, historical candidate universe for expected coverage. |
| Confirms hypothesis if | Coverage is high and missing quotes are not concentrated in profitable/safe archetypes. |
| Falsifies hypothesis if | Missing quotes cluster exactly where replay finds edge or where risk controls need exits. |
| Decision enabled | Decide whether candidate absence means true abstention or data coverage failure. |

## 22. Are no-entry/wait decisions correct abstentions or symptoms of candidate/data failure?

| Field | Detail |
|---|---|
| Why it matters | Latest inspected session was mostly wait/no-entry. That could be good selectivity, missing data, stale quotes, weak model confidence, or runtime failure. |
| Current evidence | 2026-05-21 paper log had many wait decisions and connection/quote issues, with no order endpoint calls. |
| Missing evidence | For each wait: candidate count, rejected filter reasons, best margin, quote coverage, context freshness, IBKR connection state, and whether a historical/replay equivalent would have entered. |
| Diagnostic that would answer it | Wait attribution report from paper logs: classify waits as no candidates, model abstention, guard block, stale/missing data, connectivity failure, or affordability. |
| Code/data/artifacts required | Paper JSONL logs, Protocol101 model decision fields, guard logs, candidate set logs, runtime heartbeat/error logs. |
| Confirms hypothesis if | Most waits occur in low-opportunity states with healthy data and low model margin. |
| Falsifies hypothesis if | Waits are dominated by data gaps, stale quotes, bridge failures, or missing candidate logging. |
| Decision enabled | Decide whether no-entry behavior is model selectivity or operational failure. |

## 23. Does forced-flat behavior work under disconnection, stale quotes, and market-close pressure?

| Field | Detail |
|---|---|
| Why it matters | For 0DTE options, failure to exit near close is operationally dangerous even in paper. |
| Current evidence | Forced-flat and time-flat concepts exist in lifecycle/replay, but latest inspected logs showed repeated IBKR connection failures and no filled exits. |
| Missing evidence | End-to-end evidence that an open paper position is detected, guarded, exited, logged, and reconciled under close-time and connection-failure scenarios. |
| Diagnostic that would answer it | Synthetic/fake-IB runtime test review plus human-approved paper scenario checklist; do not run broker paths without confirmation. Verify logs for every forced-flat branch. |
| Code/data/artifacts required | `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`, `v4/live/ibkr_paper_executor.py`, lifecycle code, fake IB tests, paper logs. |
| Confirms hypothesis if | Forced-flat paths fail closed, emit clear logs, and submit/suppress exits exactly as configured under stale/disconnected states. |
| Falsifies hypothesis if | Open positions can remain unresolved, exits cannot price due to missing quotes, or logs cannot prove final flat state. |
| Decision enabled | Decide whether runtime safety is adequate for continued paper observation. |

## 24. Is the current account/risk layer enforcing the documented assumptions?

| Field | Detail |
|---|---|
| Why it matters | A strategy may look safe because docs mention constraints that code does not enforce exactly. |
| Current evidence | Guard enforces paper-only, account prefix, max qty, max concurrent position, quote/context freshness, and affordability; reserve exists in config but may not be subtracted in affordability. |
| Missing evidence | A complete requirement-to-code-to-log matrix for every account/risk guard, including failure rows in real logs. |
| Diagnostic that would answer it | Guard audit: enumerate each documented guard, its exact code condition, unit tests, and observed log fields for pass/block cases. |
| Code/data/artifacts required | `v4/live/ibkr_paper_guard.py`, `v4/tests/test_protocol142_paper_executor.py`, `v4/tests/test_paper_trade_log.py`, paper logs, runbooks. |
| Confirms hypothesis if | Every documented assumption is enforced in code and visible in logs with tested failure behavior. |
| Falsifies hypothesis if | Docs mention constraints not enforced by code, or code blocks conditions not visible to operators. |
| Decision enabled | Decide whether risk governance is reliable enough for any further paper activity. |

## 25. Are stale or conflicting docs causing humans to reason from the wrong default?

| Field | Detail |
|---|---|
| Why it matters | Architecture decisions become unsafe if teams use stale promotion packets or READMEs instead of runtime truth. |
| Current evidence | Source-of-truth found conflicts: older Protocol101 readiness says not paper-approved, while launchd/runtime flag show paper-submit scheduling; v4 README claims no v2 imports while code imports v2. |
| Missing evidence | Human-owned governance resolution marking which docs are binding, superseded, or historical. |
| Diagnostic that would answer it | Documentation conflict register: for each conflicting claim, cite code/artifact/log truth and assign status `current`, `superseded`, or `historical`. |
| Code/data/artifacts required | `README.md`, `v4/README.md`, `v4/docs/*`, `v4/promotion/*`, ops plists/scripts/runtime flags/logs. |
| Confirms hypothesis if | Every operational claim has one current source and stale docs are clearly labeled. |
| Falsifies hypothesis if | Multiple documents continue to make incompatible claims about default, readiness, or safety. |
| Decision enabled | Decide what humans should trust before changing configs, thresholds, or runtime modes. |

## 26. What would prove Protocol101's edge is not just validation overfit or family-selection bias?

| Field | Detail |
|---|---|
| Why it matters | Many protocols/challengers exist. Even frozen validation metrics can be biased by repeated experimentation and exposed diagnostics. |
| Current evidence | Protocol101 threshold was selected on validation; many later diagnostics and challengers exist; protected holdout use is explicitly constrained. |
| Missing evidence | A formal validation ledger: which periods were used for training, threshold selection, diagnostics, challenger choice, and untouched evaluation. |
| Diagnostic that would answer it | Validation provenance audit, not new scoring: map every artifact/report to data periods and decision use; identify untouched or contaminated windows. |
| Code/data/artifacts required | Protocol101/194/240/265/276 manifests, summary JSONs, reports, freeze files, script arguments, generated output timestamps. |
| Confirms hypothesis if | There remains a clearly untouched evaluation path or credible paper/live validation standard independent of model selection. |
| Falsifies hypothesis if | All candidate evidence comes from windows repeatedly used for thresholding, diagnostics, or challenger selection. |
| Decision enabled | Decide whether existing historical evidence is enough for governance or whether only future no-order/paper observation can validate. |

## 27. Are research reports using the same trading game as the operational paper bot?

| Field | Detail |
|---|---|
| Why it matters | A challenger or diagnostic can look better by changing candidate universe, affordability, lifecycle, slippage, or overlap assumptions. |
| Current evidence | Source-of-truth separates operational default, replay/backtest, research challengers, and diagnostics; parity varies by component. |
| Missing evidence | Per-report game-definition diff against current Protocol101 paper runtime. |
| Diagnostic that would answer it | Game-diff matrix: for each important report, compare instrument universe, candidate generation, entry price, exit price, lifecycle, slippage, account cash, max positions, and guards. |
| Code/data/artifacts required | Protocol reports, replay scripts, `v4/live` runtime code, paper guard, manifests/configs. |
| Confirms hypothesis if | Reported conclusions are based on the same or explicitly adjusted game as the paper bot. |
| Falsifies hypothesis if | Improvements depend on game changes not present in current runtime. |
| Decision enabled | Decide which reports can inform current bot changes and which are only historical research notes. |

## 28. Which current losses are avoidable by data-quality guards rather than strategy logic?

| Field | Detail |
|---|---|
| Why it matters | Some losses may come from stale/missing quotes, wide spreads, bad context, or coverage failures. A model change would be the wrong response. |
| Current evidence | Failure modes include stale quotes, missing contract quotes, quote age weakness, connection failures, and hard-stop losses. |
| Missing evidence | Loss attribution by data-quality state at entry and during lifecycle, including quote age, spread, size, missing Greeks, and connection health. |
| Diagnostic that would answer it | Data-quality loss autopsy: join selected losing trades and paper/no-order candidates to quote/context quality fields at entry and exit checkpoints. |
| Code/data/artifacts required | Paper logs, historical selected trades, quote path data, feature manifests, IBKR connection/error logs. |
| Confirms hypothesis if | A meaningful share of losses occur in identifiable low-quality data states that can be blocked or treated as non-actionable. |
| Falsifies hypothesis if | Losses occur under clean data conditions with normal spreads and fresh quotes. |
| Decision enabled | Decide whether to focus on data/ops guards before strategy modifications. |

## 29. Is the current bot undertrading because the model is too narrow, or correctly abstaining from bad opportunities?

| Field | Detail |
|---|---|
| Why it matters | Widening the universe could add edge or introduce churn. Latest logs with mostly waits cannot distinguish selectivity from missing opportunity. |
| Current evidence | Protocol101 sees top candidates from Protocol051 only; challengers found possible broader opportunities; current paper logs mostly wait/no-entry. |
| Missing evidence | Outcome distribution of rejected candidates, pre-filtered candidates, and no-entry days under identical execution/account assumptions. |
| Diagnostic that would answer it | Missed-winner audit: classify rejected/dropped candidates by reason and compare realized outcomes with fill/affordability filters. No new entry rule selection. |
| Code/data/artifacts required | Candidate frames, Protocol051 scores, Protocol101 logits/margins, selected/rejected paths, paper/no-order logs. |
| Confirms hypothesis if | Rejected/dropped candidates contain stable, causal, executable winners not captured by current filters. |
| Falsifies hypothesis if | Rejected winners are sparse, concentrated, unfillable, unaffordable, or indistinguishable from losers. |
| Decision enabled | Decide whether candidate coverage is an actual bottleneck or a tempting but unsupported expansion. |

## 30. What evidence would justify stopping paper-submit and returning Protocol101 to no-order shadow only?

| Field | Detail |
|---|---|
| Why it matters | The current state is guarded paper runtime, but evidence may show paper-submit is premature if runtime/fill/parity gaps dominate. |
| Current evidence | Paper-submit is scheduled/enabled, but latest inspected session had no broker endpoint rows, repeated connection failures, and no fills. Quote age and lifecycle parity gaps remain high severity. |
| Missing evidence | Explicit kill/demotion criteria tied to quote freshness, connectivity, fillability, reconstruction, lifecycle parity, and guard completeness. |
| Diagnostic that would answer it | Readiness scorecard using only current evidence: mark each required trust condition pass/fail/unknown and define human-confirmed demotion thresholds. |
| Code/data/artifacts required | Source-of-truth doc, paper logs, monitor summaries, ops launchd/runtime flags, guard tests, parity diagnostics. |
| Confirms hypothesis if | Multiple critical trust conditions remain failed or unknown after inspection, especially quote freshness, lifecycle parity, and reconstructability. |
| Falsifies hypothesis if | Runtime logs and diagnostics prove fresh data, exact parity, safe guards, and sufficient paper execution evidence. |
| Decision enabled | Decide whether continued paper-submit is justified or should pause until no-order evidence improves. |

## Cross-Question Dependency Map

| Dependency | Questions blocked by it | Reason |
|---|---|---|
| True quote timestamp/age logging | 1, 2, 3, 18, 20, 21, 28 | Execution and data-quality conclusions require actual freshness evidence. |
| Paper fill/cancel observations | 1, 3, 7, 17 | Replay cannot prove executability without observed fills or misses. |
| Feature/logit reconstruction | 15, 18, 19, 22, 29 | Abstention, calibration, and parity require exact model inputs/outputs. |
| Lifecycle sequence parity | 10, 11, 12, 13, 23 | Exit questions are unsafe if live lifecycle inputs differ from replay/training. |
| Account guard replay | 7, 14, 24, 27 | Strategy evidence must match `$10k`, one-contract, current guard behavior. |
| Validation provenance | 17, 26, 27 | Challenger claims need governance before they can influence the default. |

## Immediate Non-Training Diagnostic Backlog

| Priority | Diagnostic | Answers questions | Safety note |
|---:|---|---|---|
| 1 | Current log reconstruction audit | 2, 18, 22 | Read-only over JSONL/artifacts. |
| 2 | Quote-age field completeness and timestamp audit | 1, 2, 20, 28 | Read-only unless future logging change is later approved. |
| 3 | Feature parity harness design for Protocol051 -> Protocol101 | 18, 19 | Read-only comparison against frozen artifacts. |
| 4 | Lifecycle sequence-vs-live parity harness design | 10, 11, 12, 13 | Read-only replay of known paths. |
| 5 | Account/guard-identical replay audit | 7, 14, 24, 27 | Uses existing artifacts and guard logic; no threshold changes. |
| 6 | Archetype attribution packet | 4, 5, 6, 7 | Descriptive stratification only. |
| 7 | Slot-cost causal packet | 8, 29 | Must avoid future-leakage labels when interpreting blocked signals. |
| 8 | Validation provenance ledger | 17, 26, 27 | Governance/read-only artifact audit. |
| 9 | Human-approved paper fill observation plan | 1, 3, 7 | Do not run broker/order paths without explicit human approval. |
| 10 | Readiness/demotion scorecard | 23, 24, 25, 30 | Governance decision aid, not a bot change. |

## Actions Explicitly Out Of Scope

- Do not train or retrain any entry, lifecycle, fill, defer, or challenger model.
- Do not tune Protocol101 thresholds, lifecycle thresholds, guard thresholds, or bucket filters.
- Do not score protected or untouched holdout data for exploratory selection.
- Do not promote any challenger based on these questions alone.
- Do not call broker endpoints or submit paper/live/shadow orders as part of this document.
- Do not mutate `v4/runtime/protocol101_paper_order_enablement.json`, launchd files, model artifacts, or production configs.
- Do not treat diagnostic counterfactuals as new policy rules without a separate validation and safety process.
