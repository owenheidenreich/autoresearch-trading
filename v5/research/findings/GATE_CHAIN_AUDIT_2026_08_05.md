# Path-D Gate-Chain Research — the Route That Actually Exists

> **Dated research finding.** Its measurements and gate conclusions remain evidence; its proposed
> execution sequence is superseded by [STATUS.md](../../STATUS.md).

**Decision finding, 2026-08-05:** there is no credible route through all nine gates yet. G1 is cheap to
test, but the 254 owned sessions can detect only a large 15–60 minute edge—not an edge merely large enough
to cover the measured 0.358-point cost. A neural network is unjustified on this sample, two capture days
cannot establish a worst-case latency, and a fresh paper confirmation could take roughly one year to
multiple decades unless the edge is several ES points per session.

This is an adversarial research handoff, not authorization to train, capture, contact a broker or vendor,
open the spent holdout, change runtime state, or trade. It follows [v5 STATUS](../../STATUS.md),
the [restart record](../../governance/PROGRAM_RESTART_RECORD_2026_08_05.md), the
[stand-down record](../../../v4/docs/protocol101/training/contracts/PATHD_PROGRAMME_STAND_DOWN_RECORD_2026_08_04.md), and rows 177–183 of the
[do-not-retest ledger](../history/DO_NOT_RETEST.md).

## 1. Ranked leverage — the headline

“Leverage” here means the probability of reaching and passing a gate from today's state, multiplied by
what it unlocks, divided by the cheapest honest cost of finding out. The probability bands are planning
judgments, not measured facts. Downstream bands include the chance that earlier gates fail; this prevents
a cheap downstream unit test from appearing more important than the economic gate that blocks it.

| Rank | Gate | Estimated chance from today | What it unlocks | Cheapest honest decision cost | Leverage judgment |
|---:|---|---:|---|---|---|
| 1 | **G1 direction** | **5–15%** | Every economic gate | 1–2 analyst days, $0, no waiting | Highest. It can kill or reopen the whole chain cheaply. Run no model before this. |
| 2 | **G3 feature certification** | 70–90% for a useful subset; 0% for a literal worst-case claim from two days | Causal option features for G4; **not G1** | 1 analysis day after the already-authorized 08-06/08-07 capture; $0 marginal; 2 trading sessions | Worth finishing in parallel because the marginal cost is already sunk, but it does not answer direction. |
| 3 | **G2 option wrapper** | 2–8% from today; about 30–50% conditional on a real 60-minute G1 pass | The owner's desired SPX 0DTE product | 1 local replay day, $0, no waiting | Very high conditional leverage. Do it immediately after, and only after, a G1 pass. |
| 4 | **G5 validation replay** | 1–4% for a future candidate; near 100% that the gate itself can be repaired | A trustworthy economic verdict | 1–2 engineering/analysis days, $0 | Repair before any future replay. The current gate contains more defects than the known fee cancellation. |
| 5 | **G6 runtime parity** | 1–4% from today; 80–95% conditional on a sound frozen candidate | Permission to build shadow evidence | 1 local day, $0 | The existing standard is mostly right. There is no acceptable same-input mismatch to bargain away. |
| 6 | **G4 train** | 1–6% for a shallow model; **below 1% for a defensible neural model on current data** | A candidate for G5 | 2–5 local compute/analysis days after separate owner authorization; $0 marginal | Lower than it looks: training cannot create ranking information absent from G1. |
| 7 | **G7 live shadow** | Below 3% from today; 80–95% conditional on G6 | Evidence that the sealed policy behaves live without orders | About 3 engineering days plus 20 trading sessions; $0 marginal | Necessary engineering proof, not an economic confirmation. |
| 8 | **G8 guarded paper** | Below 2% from today; 30–60% conditional on G7 | The only strong forward execution/economics test before money | At least 45 sessions for an unusually large 5-point 15-minute edge; up to thousands for small edges; broker authorization required | Essential but very low leverage until the measured edge and variance make the calendar affordable. |
| 9 | **G9 real money** | **UNKNOWN**; owner decision | Real-money operation | UNKNOWN time and capital; real loss is possible | Not a research gate. No agent should optimize for it before G1–G8 are evidenced. |

The execution order is still G1 → G2/G3 → G4 → G5 → G6 → G7 → G8 → G9. G3 is the only safe parallel
branch. A low-ranked gate is not optional; it is simply expensive or too remote to work on now.

## 2. Load-bearing refutations

These findings change the route. They should be resolved before treating the handoff's premises as a
specification.

1. **The proposed historical firewall is not unused.** The suggested 2024-08 through 2025-07 period
   appears in earlier research: `v2/lab_notebook.md` names 2024-08-05 and 2025-07-03, and
   `v4/audit/protocol163_recent_context_provenance.md` inventories SPX/VIX data throughout 2025-01 through
   2025-07. More decisively, `v4/audit/databento_es_vwap_downloads.jsonl` records owned continuous ES
   OHLCV from 2025-01-02 through 2026-03-31. Calling that year “never seen by any campaign” is false.

2. **The five 608–2,335 ms latency samples are not evidence for the new OPRA capture.** They are corrected
   no-order **ThetaData** samples. The source says so explicitly in
   [legacy one-second exit implementation](../../../v4/docs/protocol101/training/research/PHASE1_ONE_SECOND_EXIT_MODEL_IMPLEMENTATION_2026_08_03.md).
   The same file separates ThetaData p99 2,335.230 ms from Databento p99 319.521 ms; STATUS records a later
   OPRA CBBO-1m p99 of 584.6 ms. The Track-A declaration also explicitly says `theta_2336ms_used: false`
   in `v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/capture_declaration_v6.json`.

3. **“65 unblocked features” is backwards.** The current ledger contains 83 features: 8 `ADMITTED` and
   75 `BARRED`. In the 73-feature Phase-0 scope, 65 are barred and 8 are admitted. Evidence:
   `v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/feature_admission_ledger.json`.

4. **G3 is not on the G1 critical path.** M1 and M3 use ES OHLCV, prior closes, and session-relative
   volume. The 65 barred Phase-0 features are option-surface, option-quote, implied-spot, Greeks, and
   account-state fields. They may serve G2/G4 later, but none is needed to answer whether an ES directional
   rule clears 0.358 point.

5. **The G5 fee defect is real, and the other conditions are not all sound.** In
   `v4/research/pathd_phase1_replay.py`, sensitivity `delta` is integrated policy minus comparator after
   both receive the same fee. The fee cancels. The code also accepts a target-skill check when only one
   fold is positive, calls a run powered merely because it has 30 sessions and five fold labels, and
   builds `time_shifted` with `shift(-1)`, which imports the next row. Details are in §7.

6. **The G6 float tolerance is already defined.** The runtime parity code freezes feature comparison at
   absolute tolerance `1e-12` and relative tolerance `0`, requires bit-identical model scores, and demands
   exact decisions and block outcomes. The premise that a tolerance still needs to be chosen is false.
   Evidence: `FEATURE_ATOL`, `FEATURE_RTOL`, and `certified` in
   `v4/research/autoresearch_v2/runtime_decision_parity.py`.

7. **“No model can fix the option trade” is too broad if read literally.** The minute-cadence long-premium
   class is correctly closed by ledger row 181, but the row itself permits a genuinely different temporal
   structure or demonstrated underlying skill. A local 60-minute payoff query in §4 shows that correct
   direction is net positive on average and incorrect direction is net negative; theta does not erase all
   payoff regardless of direction. This does **not** prove a tradable wrapper. It refutes only the universal
   wording.

8. **A max-of-two-days cannot support “worst-case latency.”** If each day's stressed p99 is one draw from
   the distribution of stressed-day p99 values, the larger of two days has only a 9.75% chance of exceeding
   the population 95th percentile and a 1.99% chance of exceeding its 99th percentile. Fifty-nine days are
   needed for 95% confidence of exceeding the daily 95th percentile; 299 are needed for the daily 99th.
   Two days can support an observed-envelope engineering rule, not an extreme-value claim.

The capture-governance mismatch suspected early in this review is now resolved: declaration v6 exists,
supersedes v5, narrows the evidence sessions to 08-06 and 08-07 before the excluded 08-05 midday data was
observed, and records the 08-05 run as infrastructure verification only. Evidence is the v6 receipt cited
above. This is not a refutation.

## 3. G1 — direction

### What blocks it

There is no costed directional mechanism yet. The prior direction screen is uninterpretable because its
price-level features and forward-return target share the same current price; matched no-predictability
surrogates reproduced 78.5% of the headline (ledger row 183). OMAR is closed economically at −0.324
point/trade for the raw fade and +0.324 for momentum, both below 0.358 cost (row 182). The broad
full-ladder/microstructure and exit-rescue searches are also closed (rows 177–181).

The owned sample imposes a second blocker: power. A **minimum detectable effect (MDE)** is the smallest
true mean edge an experiment detects with a chosen probability. Using fixed 09:35 ES moves from the 254
owned sessions, 8,000 whole-session resamples, a one-sided 95% lower confidence bound, and the extra rule
that at least four of five chronological fold means be positive, the optimistic 80%-power MDE is:

| Horizon | Session standard deviation | Detectable **net** points/session | Gross mean needed for one round trip/day |
|---|---:|---:|---:|
| 15 minutes | 13.486 | **2.236** | **2.594** after adding 0.358 cost |
| 30 minutes | 18.099 | **2.938** | **3.296** after adding 0.358 cost |
| 60 minutes | 24.807 | **4.041** | **4.399** after adding 0.358 cost |

Source data: `/Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31/raw/databento/glbx_es_ohlcv_1m`.
The query used the 254 non-empty sessions and the actual chronological five-fold split. These are optimistic
floors for a roughly daily fixed-time policy: independent resampling does not penalize serial dependence,
and an abstaining policy has fewer effective trades, although unusually selective low-variance outcomes
could change the exact number. Under the observed fixed-slot variance, the headline question “can an edge
merely clear 0.358?” is therefore **unanswerable on the owned year**. The exact MDE must be recomputed for
the frozen M1/M3 occupancy before its outcome is read. The current data can answer whether M1/M3 contain a
much larger edge.

### Candidate mechanisms and causal warm-up

Keep exactly the two proposed mechanisms. Add no third mechanism merely to fill a quota.

- **M1, first-five-minute acceptance:** overnight inventory that is accepted with unusual opening volume
  may continue; inventory rejected in the first five minutes may revert. This subsumes the still-open
  opening-range/ORC clue in the ledger without cloning the failed late-session trigger tournament.
- **M3, overnight revaluation:** a large close-to-open move may continue or revert depending on whether
  the first five minutes confirm it. It is distinct from OMAR's intraday construction.

The interaction `gap × first-five-minute acceptance` is one predeclared M1/M3 joint member, not a new
mechanism. Do not add VIX (already used in the lean harness), MOC drop (no signed causal feed), calendar
events (no frozen signed rule), or futures/index basis (required clock unavailable). Those would expand
the search before the two cheap candidates are costed.

At `09:35:00 ET + the declared ES arrival guard`, the 09:30–09:34 bars are complete. The 09:35 bar is not.
The following fields are causal if rebuilt inside each training fold:

| Field group | Exact contents | Shared-term risk |
|---|---|---|
| Prior-session only | Previous close, range, realized volatility, last-30-minute return, and prior same-clock volume history | No direct shared current-price term. |
| Overnight | Today's 09:30 open minus prior 15:59 close; signed and scaled by prior range/volatility | Contains a current price level and must face matched surrogates. |
| First five minutes | 09:30–09:34 return, high-low range, volume sum, close position within that five-minute range | Return and close position share current price with the target; surrogate control is mandatory. |
| Session-relative baseline | Opening volume divided by an expanding median of earlier sessions' first-five-minute volume | Escapes the algebraic price-sharing trap, provided the median uses earlier sessions only. |
| Confirmation | Gap sign × first-five-minute return sign and gap magnitude × volume surprise | Inherits the shared-term risk of its price inputs. |

The historical stitched series contains four instrument changes. A local instrument-id audit found
cross-session gaps of +72.00 points on 2025-09-22, +145.50 on 2025-12-22, +49.75 on 2026-03-23, and
+135.75 on 2026-06-19. Exclude those four boundary sessions by instrument id before looking at outcomes.
Do not “adjust” them without the expiring/new-contract spread. This leaves 249 valid gap sessions from
253 possible gaps. Protocol 028's objection to stitched ES VWAP is related but not identical: it rejected
a contaminated continuous-contract feature throughout a session; here the contamination is confined to
four known cross-contract boundaries and exact pre-outcome exclusion removes it.

### Exact pass criterion

The primary outcome is **net ES points per calendar session** under one serial account. “Calendar session”
means every eligible trading day contributes one number, including zero on a no-trade day. The replay must
use causal selection, one position, no overlap, frozen tie-breaks, and forced flat by the close:

`session net = signed ES point movement − 0.358 × completed round trips`.

The handoff's proposed primary is right but incomplete. Absolute profitability alone can pass on ordinary
market drift. G1 passes only if the entire frozen M1/M3 family satisfies all of these:

1. One-sided 95% whole-session block-bootstrap lower confidence bound for mean net points/session is
   above `0.000`.
2. The paired lower bound versus the best frozen same-entry-time constant-side control is above `0.000`.
   Controls are always-long, always-short, and no-trade; comparator selection happens inside training
   folds or is frozen before the evaluation rows.
3. Net points/session and paired control delta are positive in at least `4/5` chronological folds.
4. Gross points per executed trade exceed `0.358` at the point estimate, and the absolute net criterion
   above still passes after every no-trade day is included.
5. The complete search over two mechanisms, their single joint member, directions, and 15/30/60-minute
   horizons is familywise-controlled by the surrogate campaign below. No unregistered variant may be
   substituted after outcomes are seen.
6. The full eligible index is evaluated: 254 sessions for M1 and 249 predeclared non-roll gap sessions for
   M3/joint. “Underpowered” is determined from the pre-run MDE, not by counting fold labels afterward.

This gate answers “large, harvestable edge,” not the narrower cost-scale question. Failing it does not
mathematically prove no +0.358-point edge exists; it proves the owned sample cannot justify building on one.

### Cheapest experiment, self-deception control, and downstream cost

The cheapest experiment is one raw, no-model replay of M1, M3, and the joint member on owned ES OHLCV.
Time: 1–2 analyst days. Money: $0. Calendar wait: 0. Run raw economics before correlations or a model.

The **matched surrogate** preserves each session's timestamps, volume, gap magnitude, one-minute absolute
returns, and high-low ranges, but independently flips the sign of the overnight gap and each bar's price
change, then rebuilds the complete price path, every feature, every selection, and every target. It must
not reuse real-data signals. Before the real verdict is allowed, run 1,000 complete surrogate campaigns:

- at most 5.0% may pass the full family gate, and the upper one-sided 95% Wilson bound must be at most 7.5%;
- a bounded-path fixture designed to produce the row-183 shared-term correlation must still pass at most
  5.0%; and
- an injected causal effect equal to the relevant MDE must be recovered with the correct sign and a full
  gate pass in at least 80.0% of campaigns.

Freeze these fixtures and seeds before inspecting the real verdict. This catches shared terms, search over
the family, and a null that is so destructive it cannot recover a known signal. Roll exclusion and the
paired constant-side controls catch the other likely self-deceptions.

If G1 is falsely passed, every economic conclusion in G2 and G4–G9 is invalid. If it is falsely failed,
the cost is an abandoned research route, not lost capital; that asymmetry is why the gate should remain
strict.

## 4. G2 — option wrapper reopens

### What blocks it

Ledger row 181 closes the already-tested minute-cadence long-premium class: gross was about −$13/trade
before its full friction burden. It permits re-entry only for a different structure/instrument/game or
demonstrated underlying skill. “Demonstrated” needs the following exact meaning:

> G2 research reopens only when one locked **60-minute** G1 policy passes every G1 criterion on the full
> eligible development index, including an absolute net lower bound above zero after 0.358 ES point,
> a paired control lower bound above zero, and positive net results in at least four of five chronological
> folds.

This opens one option replay; it does not approve an option model or paper trading.

### Does 60 minutes escape theta?

It can, in principle. I joined the 1,031 evaluation-eligible trajectories in
`/Volumes/AR_TRADING_DATA/reports/phase1_four_box/trajectory_outcomes.parquet` to their causal paths in
`/Volumes/AR_TRADING_DATA/exit_features/session=*/*.parquet`. At the first row at or after each horizon,
“correct” means call with positive official-SPX movement or put with negative movement. The existing
`current_net_pnl_dollars` field is net of the Phase-1 default `$3.00` round-trip fee
(`v4/research/phase1_exit_model.py`). I subtracted another `$0.08` from every path below so the table uses
the binding measured `$3.08` round trip.

| Hold | Eligible paths | Mean net when direction correct | Mean net when wrong | Accuracy that makes the two conditional means break even | Actual unconditional mean |
|---|---:|---:|---:|---:|---:|
| 15 minutes | 1,031 | +$162.54 | −$245.12 | 60.13% | −$54.14 |
| 30 minutes | 1,031 | +$232.62 | −$321.21 | 58.00% | −$60.14 |
| 60 minutes | 852 | +$308.17 | −$425.33 | **57.99%** | −$62.89 |

The break-even calculation is
`−mean(wrong) / (mean(correct) − mean(wrong))`. It assumes accuracy is independent of move size, which a
real policy will violate, so 57.99% is a screen—not a promotion threshold. The decisive fact is narrower:
the mean correct-direction 60-minute option is positive after fees. Theta does not make the structure
negative regardless of direction. The owner's product remains reachable in principle, but only with
unusually strong and payoff-weighted directional selection.

### Exact option pass, cheapest experiment, and error control

After a G1 60-minute pass, replay the exact locked timestamps, sides, abstentions, and 60-minute exits on
owned SPXW paths under the frozen fill law. G2 passes only if:

1. mean option net dollars per **all eligible calendar sessions**, including no-trade days, has a one-sided
   95% whole-session bootstrap lower bound above `$0.00`;
2. the point estimate is positive before fees as well as after the measured `$3.08` round trip;
3. net dollars/session are positive in at least `4/5` chronological folds; and
4. the conclusion survives every already-declared aggressive/touch/one-tick/two-tick fill scenario; no
   scenario is dropped because it is inconvenient.

Time: 1 local replay day. Money: $0. Calendar wait: 0. The likely self-deception is substituting scalar
directional accuracy for joint payoff: a model can be correct on small moves and wrong on large ones.
The control is the exact option-dollar replay, paired by session, with the G1 policy unchanged. A false G2
pass invalidates the owner's desired product and all option-based G3–G9 work. A true G1 pass plus G2 fail
would leave ES futures as the honest alternative—a different product requiring an owner decision.

## 5. G3 — feature certification

### What blocks it

For G1, nothing in the current 65-feature barred set blocks the test. For a later option model, most
features are barred because live/historical identity, causal arrival clocks, or parent families are not
certified. The eight currently admitted features are contract/static clocks such as option right, strike,
minute/seconds from open, seconds to close/expiry, day of week, and early-close flag. Evidence:
`v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/feature_admission_ledger.json`.

The two-session capture has a deliberate limitation. Let `D` be a stressed day's true p99 latency and let
`D_max` be the larger of two observed days. For any distribution of `D`,
`P(D_max exceeds its q-th percentile) = 1 − q^2`. That gives only 9.75% coverage of the daily 95th
percentile and 1.99% coverage of the daily 99th. No finite multiplier turns two days into a defensible
population worst-case without an assumed tail model, and the project has no justified tail model.

### Exact pass criterion and safety margin

G3 passes **for a proposed model**, not for every feature in the repository, when all model inputs are
`ADMITTED` in a signed ledger and all of the following are true:

1. all four declared 08-06/08-07 windows complete with the exact 510-symbol universe, local receipt
   sidecars, hashes, and no post-observation substitution under capture declaration v6;
2. each source/schema reports per-window p99, maximum, missing/late messages, and the worst observed result
   is used—never the quiet-window mean;
3. historical availability uses the conservative engineering clock
   `L = max(10,000 ms, 4 × worst observed live p99)` and live code uses actual receipt timestamps, waiting
   rather than deciding whenever a required input is not available by its decision watermark;
4. historical/live feature identity passes the feature's frozen tolerance on 100% of registered golden
   rows; and
5. the certification wording is limited to “causal under this fixed guard and the observed two-day
   envelope.” It may not say “worst-case latency.”

The 10-second/4× rule is an explicit conservative engineering choice, not an extreme-value estimate. It
covers the largest currently documented 2.335-second example by more than four times while scaling upward
if the new OPRA result is worse. Runtime receipt clocks, fail-closed waiting, and later shadow tests are the
controls for the tail that two days cannot estimate. If the intended claim is instead “our measured maximum
exceeds the daily 95th percentile with 95% confidence,” collect 59 stressed sessions; for the daily 99th,
collect 299. The present authorized capture cannot pass those claims.

The six causal account-state features have a separate blocker. Their parity receipts pass, but the order
submit → execution-report arrival clock was not instrumented. The receipt is explicit:
`v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/phase0b_receipts/causal_account_state_arrival_clock_OUTSTANDING.json`.
The cheapest fix is to add monotonic timestamps at order submit, IBKR execution callback receipt, and
account-reducer application, then piggyback on a future **owner-authorized** guarded-paper run. Do not place
extra orders merely to measure latency. Keep these six features barred until at least 100 execution reports
across at least 20 sessions exist; publish p50/p95/p99/max and use the same conservative guard. This adds no
marginal commission or calendar beyond G8, but it cannot be completed now.

### Cost of the experiment and of being wrong

The ordinary Track-A decision costs one analysis day after the already-authorized two capture sessions,
$0 marginal under the recorded active subscription, and two trading sessions ending 08-07. The likely
self-deception is calling the maximum of a small, mixed-source sample a worst case. Source separation,
fixed guard, actual receipt clocks, and restricted language catch it. A false admission creates look-ahead
and invalidates any G4–G8 model using the feature. It does not invalidate the ES-only G1 screen.

## 6. G4 — train

### What blocks it

Training is blocked scientifically until G1 shows raw economic ordering and procedurally until the owner
authorizes a training run in the current conversation. The earlier 18-feature contract had no ordered
ranking—top, middle, and bottom deciles were all negative and unordered—so adding model complexity would
fit noise. The April record's honest benchmark was a shallow random forest with 200 trees, depth 5, and
minimum leaf 20, not a neural result (ledger April sections and `lean_autoresearch/harness.py`).

### Model class and the numeric neural threshold

If G1 passes, start with a shallow ranker: fixed shallow random forest or histogram gradient boosting,
chronological five-fold **out-of-fold** predictions, where each prediction is made by a model that did not
train on that session. Do not start with a neural network.

There is no universal theorem saying when a neural network wins, so the true crossover is `UNKNOWN`. The
project should nevertheless use a numerical guard: do not even compare a neural candidate until there are
at least **20 independent sessions per trainable parameter** and at least **1,140 independent sessions**.
A tiny `5 inputs → 8 hidden units → 1 output` network has 57 parameters, hence `57 × 20 = 1,140`, roughly
4.5 trading years. Multiple option rows within a day do not make the market regime independent; uncertainty
must still be clustered by session. The current 254 sessions permit at most 12 parameters under this guard,
less than the tiny network.

The signal-to-noise guard is also numeric: before any neural comparison, the frozen raw mechanism score
must have a top-minus-bottom bin mean divided by its session-level standard deviation of at least `0.20`,
a one-sided 95% paired lower bound above zero, and positive spread in `4/5` chronological folds. Even then,
the neural model is accepted only if its paired out-of-fold net dollars/session versus the shallow ranker
has a lower bound above zero and wins in `4/5` folds. Owner preference is not evidence that it generalizes.

### Diagnostic, pass criterion, cost, and self-deception

Before training, place the predeclared M1/M3 score into five equal-count bins inside each chronological
fold. G1 must show:

- ordered mean net economics from low to high in at least four folds;
- top-minus-bottom paired net spread lower bound above zero; and
- the selected top bin's absolute net points/session lower bound above zero after 0.358 cost.

If the bins are flat or unordered, stop. This predicts whether a ranker has anything to rank.

G4 passes when a single frozen shallow specification produces out-of-fold predictions for 100% of its
eligible index, every input is G3-admitted or G1-causal, no fold trains on its evaluation sessions, and the
ranking diagnostic above persists in its out-of-fold scores. Final profitability remains G5's job.

The cheapest future experiment is one frozen shallow run and one fixed baseline, 2–5 local compute/analysis
days, $0 marginal, and no external waiting—but it requires fresh owner authorization because training is a
hard stop. The likely self-deception is counting thousands of correlated candidate rows as thousands of
independent examples. Session-clustered folds, one-account economics, the parameter guard, and the shallow
baseline catch it. A false G4 pass invalidates G5–G9.

## 7. G5 — validation replay

### What blocks it and what the code really tests

The known fee cancellation is confirmed. In `v4/research/pathd_phase1_replay.py`, each of the eight cells
recomputes integrated and comparator outcomes with the same `$3` or `$4` round-trip fee, then tests only
whether their difference is positive. The four `$3` cells and four `$4` cells are therefore four latency
tests duplicated. The payload does record absolute integrated net P&L, but the gate does not require it to
remain positive in each fee cell.

The other conditions need correction too:

| Current claim | Audit | Required correction |
|---|---|---|
| “Not underpowered” | **Unsound.** It checks only `sessions >= 30` and five distinct fold labels. | Freeze effect size, session variance, MDE, and required full-index session count before replay. |
| Learned pooled P&L beats best comparator | Necessary, not sufficient; the comparator is selected on the same development rows. | Freeze comparator selection inside training folds and require a paired session-level lower bound above zero. |
| Learned absolute P&L lower bound > 0 | **Sound and necessary.** | Keep it, computed per all calendar sessions rather than only learned-entry rows. |
| Positive delta in 4/5 folds | Directionally useful but ignores uncertainty and absolute loss. | Require both absolute net and paired delta positive in 4/5, plus pooled lower bounds. |
| Positive exit-target skill | **Unsound.** `positive_target_skill_folds > 0` means one fold suffices; the correlation is pooled by row. | Require 4/5 folds and session-clustered known-answer skill. |
| All fee/latency deltas positive | **Fee half is false; latency half is too weak.** Raw positive delta can be arbitrarily small. | Test absolute integrated net at each fee/latency and paired delta lower bounds at each latency. |
| No negative control beats comparator | Incomplete. Negative controls are not run through the full gate, and `time_shifted = shift(-1)` uses the next row. | Use a causal lag `shift(+1)` or causal feature transform and require every negative to fail the same corrected gate. |

The “best of 15 comparators” also creates selection uncertainty. Selecting the strongest comparator makes
the economic hurdle conservative, but paired inference still has to treat its identity as frozen; it may
not be reselected on the final confirmation set.

### Corrected exact pass criterion

G5 passes only if all conditions below hold on the full preregistered session index:

1. a pre-run power receipt says the index has at least 80% power for the declared practical effect;
2. absolute integrated net dollars per calendar session has a one-sided 95% session-block-bootstrap lower
   bound above `$0` at every declared fee/latency cell;
3. paired integrated-minus-frozen-comparator dollars/session has a lower bound above `$0` at latency
   0/1/2/5 seconds; fee levels may be reported for absolute economics but are acknowledged as algebraically
   identical for the paired delta;
4. both absolute net and paired delta are positive in `4/5` chronological folds;
5. target skill passes in `4/5` session-clustered folds and its null passes a known-answer gate before use;
6. the comparator identity and all thresholds are frozen before the evaluation rows; and
7. constant, sign-reversed, causally lagged, and session-matched shuffled controls each fail the **same full
   gate**. Any negative-control pass makes the result invalid, not merely weak.

The cheapest experiment is first a one-day code/test repair, then a one-day replay of an already-frozen
candidate. Money: $0. Calendar wait: 0. No training is authorized by this document. The likely
self-deception is reporting a relative uplift while the candidate still loses money absolutely; absolute
stress cells catch it. A false G5 pass invalidates G6–G9.

## 8. G6 — runtime parity

### What blocks it

No candidate exists. The parity implementation itself already encodes the right same-input standard. It
requires identical candidate sets, 100% feature cells within absolute tolerance `1e-12` and relative
tolerance `0`, bit-identical scores, exact actions/sides/contracts, and exact fixed-block outcomes. Evidence:
`v4/research/autoresearch_v2/runtime_decision_parity.py`, especially `FEATURE_ATOL`, `FEATURE_RTOL`, and
the `certified` predicate.

### Exact pass and acceptable differences

For every registered development trace and every future shadow trace, G6 passes only with:

- 100% identical candidate identities;
- 100% feature cells within the already-frozen tolerance;
- 100% bit-identical model scores;
- 100% action, signal time, side, selected contract, and block-policy matches; and
- zero training rows whose declared source was unavailable at decision time.

For the same sealed input, **no mismatch is acceptable**. Float nondeterminism means the build/runtime is
not reproducible; freeze library/runtime versions or choose deterministic inference. Different tie-breaking
is a bug. A quote exactly on the age boundary must follow one frozen `<=` rule. Reconnects, corrections,
duplicates, and sequence gaps can legitimately change what input has arrived, but they do not waive parity:
the live system must fail closed to `WAIT`, preserve the sealed decision, and log the input divergence.
Widening the tolerance would hide a mechanism rather than repair it.

The cheapest future experiment is one local same-input parity run plus boundary fixtures for quote age,
ties, missing messages, corrections, and reconnects: about one engineering day, $0, no waiting. The likely
self-deception is labeling different inputs “float noise.” Candidate hashes and per-input divergence logs
catch it. A false G6 pass invalidates all G7/G8 evidence and makes G9 unsafe.

## 9. G7 — live shadow

### What blocks it and what passes

G7 is blocked by the absence of a G6-certified candidate and a new no-order shadow service. Shadow means
the live system records what it would decide but has no order submission path.

G7 passes after **20 consecutive eligible trading sessions** in which:

1. every scheduled decision produces a complete event/receipt/adapter/decision-clock trace;
2. offline replay of those sealed inputs reproduces 100% of candidate sets, features, bit scores, actions,
   sides, contracts, and block outcomes under the G6 standard;
3. every stale/missing/sequence-gap/reconnect case emits `WAIT` and no sealed decision changes later;
4. early closes and exchange holidays follow the frozen calendar; and
5. the service has no callable broker-order path and submits zero orders.

Twenty sessions is an engineering coverage bar, not statistical proof of profit. Do not inspect economic
outcomes during G7 if the first forward economic block is meant to remain clean for G8 design.

Cheapest cost: about three engineering days, $0 marginal, and 20 trading sessions—roughly 28 calendar days.
The likely self-deception is checking only happy-path parity while reconnects silently alter candidates.
The registered fault fixtures and replay of every sealed live input catch it. A false pass invalidates G8
and G9 because the paper system may not be the historical policy.

## 10. G8 — guarded paper and the calendar cost

### Power curve

For a one-sided 5% test with 80% power, independent session outcomes, session standard deviation `sigma`,
and true net edge `X`, the normal lower-bound session count is:

`N = ceil(((1.645 + 0.842) × sigma / X)^2)`.

Using the owned 09:35 fixed-horizon standard deviations from §3 gives the following **optimistic lower
bounds**. Each cell is `trading sessions / approximate years` at 252 trading sessions per year. The final
pre-registration must simulate the locked policy's actual session outcomes with serial blocks and the
4-of-5 sign rule; that can only increase N.

| True net edge X | 15m, σ=13.486 | 30m, σ=18.099 | 60m, σ=24.807 |
|---:|---:|---:|---:|
| 0.5 point/session | 4,498 / 17.85y | 8,101 / 32.15y | 15,219 / 60.39y |
| 1.0 | 1,125 / 4.46y | 2,026 / 8.04y | 3,805 / 15.10y |
| 2.0 | 282 / 1.12y | 507 / 2.01y | 952 / 3.78y |
| 3.0 | 125 / 0.50y | 226 / 0.90y | 423 / 1.68y |
| 4.0 | 71 / 0.28y | 127 / 0.50y | 238 / 0.94y |
| 5.0 | 45 / 0.18y | 82 / 0.33y | 153 / 0.61y |

If G1 only barely passes at its development MDE, testing zero against the **same full effect** takes about
225–235 sessions before the fold-sign penalty—roughly one calendar year. A more honest test asks whether
the forward lower bound exceeds half the development mean, allowing at most 50% decay. If the true forward
mean equals the full development mean, the distance between that alternative and the half-mean null is
half the §3 MDE; the normal requirement is about 900–939 sessions, or 3.6–3.7 trading years. This is the
power result most likely to sink the plan.

For an SPX option paper policy, recompute the table in dollars using the G2 session-level standard deviation.
Do not use ES variance to certify option fills.

### Exact pre-registration and pass

Before the first paper session, freeze:

- one candidate hash, feature ledger, runtime hash, fill law, trade windows, invalid-session rule, and
  chronological session list;
- the expected alternative `A`, equal to the locked development net mean, and practical bar
  `B_ES = max(0.50 point, 50% of A)` for an ES test, or
  `B_option = max($25/session, 50% of A)` for the desired option product;
- `N = max(60, ceil((2.487 × locked development session SD / (A − B))^2))`, then increase N if a
  session-block power simulation with the 4-of-5 rule has less than 80% pass probability. If `A <= B`,
  the proposed policy is not practically testable and paper must not start; and
- one-sided 5% significance, 80% target power, and no early stop for success.

G8 passes only after N valid sessions when mean net per all declared sessions has a one-sided 95% lower
bound above **B**, at least four of five contiguous forward blocks are positive, and all safety/fill/parity
conditions remain satisfied. Stop early only for safety, invalid data, or the preregistered futility rule
at N/2 when conditional power is below 10%; never stop early because a temporary curve looks good. Invalid
sessions are replaced only under rules frozen before their outcomes.

Money for paper orders is nominally $0 real capital, but broker contact/order submission and any runtime
change are Tier-1 owner decisions. Calendar cost is the table above. The likely self-deception is repeated
peeking and stopping at a high-water mark; a sealed N, no-success-stop rule, and append-only receipts catch
it. A false G8 pass is the last research error before real losses at G9.

## 11. G9 — real money

G9 is out of scope and is not passed by a statistic alone. The eventual owner packet must include:

- signed receipts for every G1–G8 criterion and hashes tying the live artifact to them;
- the protected-holdout access record showing it was not reopened after its one spent access;
- full forward-paper distribution by session, fold, side, horizon, fill, fee, drawdown, and outage;
- parity and reconnect/fail-closed evidence with every incident and unresolved mismatch;
- proposed capital, maximum daily/total loss, contract limit, kill switch, rollback, monitoring ownership,
  broker permissions, and an explicit statement of what paper trading did not test; and
- an owner-signed decision that states the dollars at risk and the stopping conditions.

Pass criterion: the packet is complete, no earlier gate is stale or waived, and the owner explicitly
authorizes a named artifact, capital limit, and start/end window. Cost, time, and pass probability are
`UNKNOWN`. The likely self-deception is treating paper fills as real fills; conservative execution stress
and a tiny, owner-capped launch are controls, not guarantees. Getting G9 wrong can lose real money.

## 12. Rebuilding the confirmation firewall

### Historical range

No demonstrably unused range has yet been identified. The proposed 2024-08 through 2025-07 scope is
contaminated by prior observation as documented in §2. A provenance audit must first produce a date range
absent from campaign inputs, reports, notebooks, charts, and analyst decisions. Only then should an exact
Databento metadata quote be requested for `GLBX.MDP3`, `ohlcv-1m`, `ES.FUT`, with exact UTC start/end and
the same continuous-contract construction declared.

The exact price is **UNKNOWN**. Obtaining it requires contacting the vendor, which the handoff's hard stop
forbids and the repository's Tier-1 rules reserve to the owner. I did not contact Databento. The best local
price evidence is `v4/audit/databento_es_vwap_downloads.jsonl`: one 2025-01-02 through 2026-03-31 request,
710,045 raw rows and 311 session records, records total estimated cost **$2.592221274972**. A one-year
request is plausibly only a few dollars, but that is not an exact quote and must not be presented as one.

Once a genuinely unused range is found, the protocol is: freeze candidate and analysis code; record range
and hashes before access; open once; reject or accept without tuning. If it fails and the model is changed,
that range becomes development data and a new confirmation set is required.

### Forward sessions and whether both can be used

Use both, for different questions, without reusing either:

1. one untouched older range is a one-shot regime-robustness test after G5/G6; and
2. later forward G8 sessions test non-staleness, live data timing, and paper execution.

The forward block is the stronger final firewall because it tests the current regime and operational path.
The older block is cheaper and immediate after purchase, but can miss structural change. Do not tune after
the older block and then call the same block confirmation; do not count G7 engineering sessions again as
G8 economic confirmation if their economics were inspected.

## 13. Impassability verdict

The whole chain is **not proven impossible**, but it has no currently evidenced viable route.

- **The cost-scale G1 question is impassable on the owned 254 sessions under the observed fixed-time
  variance and roughly daily occupancy.** A real +0.358-point-scale edge can exist and remain undetectable.
  An unusually low-variance selective rule is `UNKNOWN` until its occupancy is frozen and its MDE is
  recomputed. What is currently passable is a test for a much larger roughly 2.24–4.04 net point/session
  edge. A predeclared M1/M3 result at that size would change this verdict.
- **A neural G4 is impassable on current effective sample size.** At least 1,140 independent sessions and
  a raw standardized ranking spread of 0.20 are needed before even comparing the tiny network specified
  here. More data or a clear, repeated neural-over-shallow paired forward win would change this verdict.
- **A population worst-case G3 latency claim is impassable from two capture sessions.** Fifty-nine stressed
  sessions change the daily-q95 coverage claim; 299 change daily-q99. A conservative observed-envelope
  certification remains possible now.
- **A one-week path through G8 is impossible.** Even an unusually large four-point 60-minute net edge needs
  about 238 paper sessions before extra fold/serial penalties. A materially lower session variance or a
  much larger locked edge would change the calendar.
- **G2 is not structurally impossible.** The 60-minute conditional payoff can overcome theta, but no policy
  has shown the needed payoff-weighted direction. A locked G1 pass followed by positive option-dollar
  lower bounds under all fill scenarios would change the current low-probability judgment.

The honest current bet is that G1 fails or is too small to confirm, not that a neural network rescues it.
That is useful: it tells us to spend days on mechanism economics, not months on model and runtime layers.

## 14. The only useful one-week sequence

There is no one-week route through G1–G9. There is a one-week route to the decision that determines whether
the rest should exist:

| Date | Work | Stop rule | Cost/authority |
|---|---|---|---|
| Wed 08-05 | Freeze the M1/M3/joint feature law, roll exclusions, complete search family, MDE receipt, and surrogate known-answer gate. Separately specify the G5 repairs. | No outcome query until fixtures and family are frozen. | Local analysis/code only, $0. |
| Thu 08-06 | Let the already-authorized Track-A capture run independently. Run only the G1 surrogate known-answer campaign on owned ES. | If false-pass >5%, Wilson upper bound >7.5%, or injected-effect recovery <80%, repair the null once and refreeze; do not inspect real G1 economics. | Capture authority already recorded; this research task changes nothing. |
| Fri 08-07 | Complete the second authorized capture day. If the null passed, run one raw M1/M3/joint economic screen. | If no member clears the full G1 gate, stop the economic chain: no training and no option replay. | $0, local analysis plus already-authorized capture. |
| Sat–Sun 08-08/09 | Compile Track-A receipts and update the existing admission ledger only. Implement/test the corrected G5 gate without a model run. | Do not call two sessions “worst case”; keep failed families barred. | Local analysis/code, $0. |
| Mon 08-10 | If and only if G1 passed at 60 minutes, run the exact no-training G2 60-minute option-dollar replay. Otherwise write the direct stop finding into the do-not-retest ledger. | G2 fail means the desired option product is closed for this mechanism. | Local owned data, $0. |
| Tue 08-11 | If G1 and G2 passed, freeze the shallow G4 specification and request separate owner authorization for one training run. Begin provenance audit for a genuinely unused historical range; do not quote or download. | No training/vendor call without owner authorization. | Local research, $0. |
| Wed 08-12 | Owner checkpoint: G1/G2 verdict, G3 admitted subset, corrected G5 contract, power-based forward calendar, and explicit stop/go recommendation. | G4–G9 remain unpassed. | No external action. |

This sequence serves the project's single open question directly. It either finds a large, cost-clearing
directional mechanism with a valid null or stops the option/model programme before it consumes more time.

## Evidence and calculation notes

- Binding state and costs: [v5 STATUS](../../STATUS.md).
- Programme scope and hard stops: [restart record](../../governance/PROGRAM_RESTART_RECORD_2026_08_05.md)
  and [stand-down record](../../../v4/docs/protocol101/training/contracts/PATHD_PROGRAMME_STAND_DOWN_RECORD_2026_08_04.md).
- Closed experiments and shared-term evidence: rows 177–183 of the
  [do-not-retest ledger](../history/DO_NOT_RETEST.md).
- ES measurement source: `/Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31/raw/databento/glbx_es_ohlcv_1m`.
- Option payoff source: `/Volumes/AR_TRADING_DATA/reports/phase1_four_box/trajectory_outcomes.parquet`
  and `/Volumes/AR_TRADING_DATA/exit_features/session=*/*.parquet`.
- Feature state: `v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/feature_admission_ledger.json`.
- Capture scope: `v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04/capture_declaration_v6.json`.
- G5 implementation: `v4/research/pathd_phase1_replay.py` and `v4/research/phase1_exit_model.py`.
- G6 implementation: `v4/research/autoresearch_v2/runtime_decision_parity.py`.
- Comparable ES vendor receipt: `v4/audit/databento_es_vwap_downloads.jsonl`.

All computations in this document were read-only on owned local artifacts. No model was fitted, no capture
was started or changed, no broker/vendor was contacted, no money was spent, and the protected holdout was
not opened.
