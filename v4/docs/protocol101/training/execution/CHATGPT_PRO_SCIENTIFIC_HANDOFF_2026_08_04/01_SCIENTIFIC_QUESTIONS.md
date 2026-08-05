# Scientific Questions for ChatGPT Pro Thinking

Date: 2026-08-04  
Project: SPXW 0DTE long-call / long-put trading system  
Review mode: attack the scientific plan before any implementation

## Your assignment

Read all 20 numbered files in this package. Then answer the questions below as an adversarial scientific reviewer. Do not write code, train a model, tune thresholds, or recommend promotion. The immediate job is to determine whether the proposed options-chain directional screen is identifiable, causal at decision time, economically meaningful, and small enough to falsify honestly.

For every material claim in your review:

- Cite the relevant package file and line or section.
- Label it `EVIDENCE`, `INFERENCE`, `ASSUMPTION`, or `UNKNOWN`.
- Identify contradictions between documents rather than silently choosing one.
- Treat a rejecting result as useful. `NO_EDGE` is a successful scientific outcome.

Stop after the written review. The project owner will decide whether any experiment is authorized.

## Non-negotiable objective and constraints

The end product is a bot that buys SPXW 0DTE calls or puts. Do not replace that objective with ES, short premium, spreads, longer-dated options, or a generic execution strategy. It is valid to conclude that the long-option objective is not supported by the present evidence.

The current closure must be taken seriously:

- The broad long-premium programme lost about **$13 per trade gross with all friction removed**, negative in **5 of 5 folds**.
- The option does not create directional edge; it expresses and leverages one. The class reopens only after demonstrated directional skill on the underlying over the intended holding horizon.
- The 36-session holdout has been spent. There is no untouched confirmation firewall. Historical results can screen and falsify, but future confirmation must be forward paper evidence.
- A prior apparent result materially above the honest historical range came from a 60-second look-ahead leak. Any unusually strong result is presumed defective until its clock law is proved.
- Honest historical calibration is approximately entry PF **1.132** and composed PF **1.7–1.95**, both before full parity.
- Measured ES friction is **0.358 points per trade**, but this is not automatically the correct survival bar for a one-account, serially occupied strategy.
- Prior entry and exit artifacts were not a compatible model merely because they ran in sequence. The exit chose index zero in **95.1%** of trajectories and amputated the convex tail: p99 about **$86** versus roughly **$3,422** for hold-to-close.
- A compatible entry-plus-exit system needs one objective that rewards both loss truncation and tail capture, a mandatory deterministic catastrophic floor, validation on the entry policy's own out-of-fold distribution, one-account serial occupancy, and a fresh composed-system gate.
- Prior-art checks are mandatory before every hypothesis. A candidate is blocked by a do-not-retest row or by a rejecting verdict anywhere in the protocol decoder, not only by one convenient section.
- Raw economics comes first. No null, learned model, or governance expansion is justified until a prespecified raw candidate clears its economic gate.
- Declared hypothesis count is the multiplicity family. Do not widen the family after results are known.

## Data facts that require verification, not trust

The owned historical corpus appears to contain 251 SPXW sessions. A direct census reported approximately 392–1,288 distinct same-day contracts per session, median 478 and mean 532. The available chain data includes CBBO snapshots with displayed bid/ask sizes and interval last-sale fields. It does **not** appear to contain participant-labelled positions or a historical trade-by-trade signed-flow substrate. Verify these facts from the package and mark missing proof as `UNKNOWN`.

Two important corrections must remain visible:

1. The options chain has not been used as a full-chain directional signal, but it is false to say options information has never entered a model. The canonical feature contract includes near-ATM straddle, put/call, smile-slope, delta, and gamma fields.
2. Public chain data does not identify dealer positioning. Any construction from gamma and displayed quote sizes must be named as a proxy, not dealer GEX, unless its sign and participant identity are actually observed.

## Candidate mechanism family to attack

The proposed smallest family is **K = 2** at one primary horizon of **5 minutes**.

### H1 — five-minute risk-reversal change

Construct a symmetric near-ATM put-versus-call skew measure using causal quotes and per-contract implied volatilities. Define:

```text
RR_t = put_skew_t - call_skew_t
score_t = -(RR_t - RR_(t-5m))
```

Positive score maps to `CALL`, negative score maps to `PUT`, and exactly zero or unavailable input maps to `WAIT`. The sign convention, strike pairing, IV solver, quote-quality rules, and treatment of missing sides are not yet frozen. Determine whether the economic interpretation and sign are defensible.

### H2 — gamma-weighted displayed quote pressure

For each eligible option quote:

```text
imbalance_i,t = (bid_size_i,t - ask_size_i,t)
                / (bid_size_i,t + ask_size_i,t)

score_t = [sum(call_gamma_i,t * imbalance_i,t)
           - sum(put_gamma_i,t * imbalance_i,t)]
          / sum(all_eligible_gamma_i,t)
```

Positive score maps to `CALL`, negative score maps to `PUT`, and exactly zero or unavailable input maps to `WAIT`. This is a gamma-weighted displayed-quote-pressure proxy, **not dealer positioning or dealer GEX**. Determine whether the sign, normalization, contract universe, and causal interpretation survive scrutiny.

The 5-minute horizon is a proposed falsification horizon, not an established optimum. Existing horizon evidence appears poor against the old charter: about 23.1% big-loss rate at 5 minutes versus a 2% limit, worsening at longer horizons. Determine what that does and does not imply for this new directional screen.

## Scientific questions

### 1. What is the estimand?

Is the project trying to estimate future SPX return sign, signed SPX points, conditional move magnitude, or the conditional net payoff of a specified long option? Write the smallest hierarchy of estimands that can legitimately reopen the 0DTE long-call/put wrapper. In particular, is `E[long-option net payoff | causal chain state] > 0` the ultimate target, with underlying direction only a necessary intermediate condition, or is that framing incomplete?

### 2. Can the chain variables predict rather than merely describe?

Which available chain quantities could plausibly contain incremental information about future SPX direction over five minutes? Separate prediction from reaction, nowcasting, common response to SPX, volatility forecasting, and liquidity-state detection. State what lead/lag evidence would falsify each story.

### 3. Are H1 and H2 the right smallest mechanisms?

Audit their formulas, sign conventions, causal clocking, strike selection, normalization, and expected failure modes. If either is not scientifically coherent, reject it rather than repairing it after looking at outcomes. If one minimal correction is indispensable, specify it before results and explain whether it changes K.

### 4. What claims about dealer hedging are identifiable?

Given quotes, displayed sizes, aggregate interval last-sale fields, greeks, and no participant-labelled inventory, what can and cannot be inferred about dealer gamma, dealer inventory, trade sign, and hedge demand? Name the strongest defensible mechanism and the evidence that would distinguish it from a storytelling proxy.

### 5. Is five minutes the right single primary horizon?

Evaluate five minutes against theta, spread, quote cadence, observed loss incidence, decision latency, and one-account occupancy. If five minutes is defensible only as the least-bad falsification horizon, say so. Propose one primary horizon and at most one clearly labelled sensitivity that does not expand the confirmatory maxT family.

### 6. What is the correct raw survival statistic?

A per-trade average can overstate harvestable edge when signals fire many times while one account can hold only one position. Should the primary screen report signed SPX points **per session under strict serial occupancy**, points per executed interval, both, or a prespecified option payoff? Define the clock, cooldown, tie-breaking, forced-flat rule, and treatment of skipped overlapping signals.

### 7. What proves incrementality over already-closed price signals?

Design the smallest deterministic comparison against an already-closed SPX-price baseline, such as fixed-horizon SPX momentum, without introducing a learned model or a post hoc null. Must the chain signal beat the comparator in all five folds, and what statistic should be compared?

### 8. What exact clock law prevents leakage?

Specify event time, receive/availability time, interval boundaries, feature cutoffs, label start, label end, quote staleness, and as-of joins. Explain how to prove that the score cannot see any quote, trade summary, SPX bar, IV input, or session aggregate that becomes available after the decision.

### 9. How should score become CALL, PUT, or WAIT without threshold fishing?

Is sign-only mapping adequate? If a deadband or liquidity mask is necessary, how can it be fixed from market structure or data-quality requirements rather than optimized on returns? Explain how missingness and zero denominators behave.

### 10. What exact gate reopens the option wrapper?

Audit this proposed directional gate:

- strict one-account serial non-overlap;
- positive signed SPX points per session and per executed interval in 5 of 5 folds;
- beats a frozen SPX-momentum comparator in 5 of 5 folds;
- no pooled-result rescue.

Then define the next, separate wrapper gate using a predetermined nearest-ATM call/put, a fixed five-minute hold, zero-friction mid-to-mid gross first, followed by actual ask-to-bid accounting, fees, a deterministic catastrophic floor, and full serial replay. State whether direction alone is sufficient to reopen the class.

### 11. How do we distinguish directional mean from volatility or risk-premium information?

Skew, gamma, and quote pressure may forecast absolute moves, downside variance, liquidity, or crash risk without forecasting signed returns. Specify the minimal signed-return, absolute-return, placebo, and sign-reversal checks that distinguish these cases without turning the exercise into a large hypothesis family.

### 12. Is displayed size scientifically usable here?

Assess cancellation, flicker, stale quotes, exchange aggregation, locked/crossed states, sparse strikes, varying contract universes, zero sizes, and strike-distance concentration. Define the smallest quality rules that can be fixed before outcomes. Explain whether 1-minute CBBO is too coarse and whether using 1-second data would silently change the family.

### 13. If raw direction survives, how can entry and exit be trained compatibly?

Give one mathematically explicit joint objective that values loss truncation and convex-tail capture on the entry policy's own out-of-fold distribution. Explain how it avoids both the always-exit solution and the mirror failure created by 98%-dominant labels. Preserve the deterministic catastrophic floor and require the composed system to clear the full gate again.

### 14. What can be claimed with a spent holdout?

Define the strongest honest historical claim, the role of cross-fold consistency, and the forward paper protocol needed before promotion. Explain what cannot be called confirmation, regardless of p-values or apparent fold stability.

### 15. What is the smallest sufficient falsification matrix?

Freeze the exact rows, metrics, folds, comparator, occupancy law, data-quality exclusions, and pass/fail logic. The declared family should remain K = 2 unless you explicitly conclude that one hypothesis is invalid before outcomes. No adaptive expansion, threshold sweep, horizon sweep, or pooled rescue is allowed.

### 16. Where does the plan most likely fail?

Rank the fatal risks: non-identifiability, leakage, simultaneous response to SPX, insufficient timestamp resolution, unstable displayed size, skew forecasting variance rather than direction, economic magnitude below theta/spread, multiple testing, nonstationarity, occupancy dilution, and wrapper failure after an underlying pass. Add any missing failure that could produce a convincing false positive.

## Required response format

Return one written review with these sections:

1. **Bottom line** — `GO`, `REVISE`, or `NO-GO` for the raw K=2 screen; this is not authorization to code.
2. **Corrections to the factual premise** — with file citations and evidence labels.
3. **Frozen scientific specification** — estimand, K, horizon, formulas, mapping, clock law, occupancy law, and exclusions.
4. **Raw-screen ledger and gates** — exact fold-level outputs and pass/fail rules.
5. **Conditional option-wrapper and compatible-model design** — only if the raw gate survives.
6. **Adversarial failure analysis** — ranked by probability and damage.
7. **Decision** — what should happen next and the exact `NO_EDGE` stop condition.

Do not provide code. Do not widen the search. Do not treat historical screening as confirmation. Do not propose changing the live paper default.

## Package map and source provenance

All copies are snapshots of the working tree on 2026-08-04. The numeric prefix is the reading order.

| File | Original source | Why included |
|---|---|---|
| `01_SCIENTIFIC_QUESTIONS.md` | New review brief | Defines the questions and response contract. |
| `02_PROJECT_OVERVIEW_AGENTS.md` | `AGENTS.md` | End-to-end project, gates, safety, and current default. |
| `03_CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md` | `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md` | Current operating and historical authority. |
| `04_PROTOCOL101_TRAINING_README.md` | `v4/docs/protocol101/training/README.md` | Current training authority and permitted next gate. |
| `05_DATA_CONTRACT.md` | `v4/docs/DATA_CONTRACT.md` | Causality, schema, and provenance contract. |
| `06_PRIOR_CAMPAIGN_DISTILLATION.md` | `v4/docs/protocol101/training/history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md` | Prior-art and do-not-retest evidence. |
| `07_OPTIONS_CHAIN_DIRECTION_HANDOFF.md` | `v4/docs/protocol101/training/execution/CODEX_HANDOFF_OPTIONS_CHAIN_DIRECTION_2026_08_04.md` | Immediate proposed direction and open questions. |
| `08_PHASE1_CLOSEOUT.md` | `v4/docs/protocol101/training/research/PATHD_PHASE1_CLOSEOUT_2026_08_03.md` | Latest directional-programme closure. |
| `09_HORIZON_VS_CHARTER_SWEEP.md` | `v4/docs/protocol101/training/research/PATHD_HORIZON_VS_CHARTER_SWEEP_2026_08_04.md` | Horizon evidence and charter conflict. |
| `10_FEATURE_CERTIFICATION_REPORT.md` | `v4/docs/protocol101/training/research/PATHD_PHASE0_FEATURE_CERTIFICATION_REPORT_2026_08_04.md` | Feature availability and certification state. |
| `11_PAPER_TRADING_DEFAULT.json` | `v4/promotion/PAPER_TRADING_DEFAULT.json` | Machine-readable active paper default. |
| `12_PATHD_PHASE1_ENTRY.py` | `v4/research/pathd_phase1_entry.py` | Representative entry research implementation. |
| `13_PATHD_OPRA_PARITY_FEATURES.py` | `v4/research/pathd_opra_parity_features.py` | OPRA parity and options-feature construction. |
| `14_CANONICAL_STAGE1_CONTRACT.py` | `v4/model/protocol101_canonical_stage1_contract.py` | Canonical feature contract, including existing chain-derived fields. |
| `15_SPX_DIRECTIONAL_SKILL_SCREEN.py` | `v4/research/pathd_spx_directional_skill_screen.py` | Existing raw SPX directional-screen machinery. |
| `16_PATHD_RESEARCH_LOOP.py` | `v4/research/pathd_research_loop.py` | Research loop and prior-art enforcement. |
| `17_PATHD_MODEL_GATE.py` | `v4/research/pathd_model_gate.py` | Model-admission and firewall behavior. |
| `18_ENTRY_LIVE_FEATURE_CATALOG.py` | `v4/research/autoresearch_v2/entry_live_feature_catalog.py` | Live feature availability and contracts. |
| `19_PROTOCOL101_ENTRY.py` | `v4/live/protocol101_entry.py` | Current entry decision path. |
| `20_PROTOCOL160_PERSISTENT_PAPER_TRADER.py` | `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py` | End-to-end persistent paper runtime. |

