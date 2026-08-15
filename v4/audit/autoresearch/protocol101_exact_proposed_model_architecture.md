# Protocol101 Exact Proposed Model Architecture

## 1. Current status

### Learned entry models already trained

Protocol101 trained **420 fresh HGB entry models**:

- Four feature hypotheses: H0, H1, H2, and H3.
- Seven fixed exit policies: P0 through P6.
- Three random seeds.
- Five chronological folds.

Each model scored eligible option contracts using one of the four approved feature sets. A threshold then determined whether the system entered a trade.

The 420 models completed training and independent reproduction, but **none passed final entry selection**. Five rows showed adjusted signal: H0/P5, H0/P6, H1/P5, H1/P6, and H2/P5. They remain quarantined because the D1 negative-control investigation showed that much of the apparent profit came from profitable P5 exposure, while the model's incremental benefit over a correctly matched random baseline remained inconclusive. The 420 models are benchmark evidence, not accepted trading models.

### Learned contract-selection models already trained

Two exploratory HGB contract selectors were trained in the non-promotable Stage-0 pilot:

- **M0:** internal delta and gamma only.
- **M1:** all 17 synchronized features.

At minutes when P5 already permitted an entry, each model selected one contract from the governed option risk set.

Both models beat exact-random contract selection, but both lost to P5's deterministic nearest-eligible-ATM selection. The pilot therefore ended with `stop_no_preliminary_signal`. M0 and M1 failed to justify a full contract-selector campaign and were not promoted.

### Learned HOLD/EXIT models already trained

One exploratory HGB HOLD/EXIT model was trained, along with two shuffled controls. It evaluated an open P5 position once per completed minute and predicted whether holding for one more minute was better than selling immediately.

This model did not pass. On the strict serial account replay, fixed P5 earned approximately `$1,185`, while the learned HOLD/EXIT policy lost approximately `$3,005`. The attribution found two problems:

- It exited the five trades shared with P5 at worse outcomes, costing approximately `$2,280`.
- Its early exits freed the single trading slot for ten additional losing entries, costing approximately `$1,910`.

The one-minute HOLD/EXIT model is not validated and is not a candidate for paper trading.

### Overall status

- Validated learned entry model: **none**.
- Validated learned contract selector: **none**.
- Validated learned HOLD/EXIT model: **none**.
- New model currently being trained: **no**.
- Full Trader redesign: **proposal only**. Its exact scientific contract has not been frozen and its training has not begun.

## 2. Exact proposed trading flow

The current proposal is:

`synchronized market and option-ladder data -> P5 opportunity, direction, and contract rule -> learned WAIT/ENTER decision -> buy one contract -> learned HOLD/EXIT decisions -> sell or force-flat`

### Stage 1: Synchronized market and option-ladder data

- **Type:** Hand-coded data and feature machinery.
- **Responsibility:** Build the causal market view that historical training and IBKR can both reproduce.
- **Receives:** SPX context, approved option-ladder state, contract geometry, approved option composites, and approved internal Greeks.
- **Outputs:** The governed market state and eligible option risk set.

### Stage 2: P5 opportunity, direction, and contract rule

- **Type:** Hand-coded.
- **Responsibility:** Decide whether a usable P5 opportunity exists, choose call or put, and choose the nearest eligible strike to ATM on that side.
- **Receives:** SPX relative to VWAP and the governed eligible option contracts.
- **Outputs:** Either no P5 candidate or one specific call or put contract.

### Stage 3: Flat-state WAIT/ENTER decision

- **Type:** Proposed learned model.
- **Responsibility:** Decide whether the current P5 candidate is worth buying now or whether the trader should remain flat and wait.
- **Receives:** The causal flat-state feature row and the P5-selected contract. The exact frozen input list is **not yet specified**.
- **Outputs:** `WAIT` or `ENTER`.

If the output is `WAIT`, the trader remains flat and evaluates the next completed minute. If it is `ENTER`, the trader buys one P5-selected contract using the executable entry-price rules.

### Stage 4: Open-position state

- **Type:** Hand-coded state tracking.
- **Responsibility:** Track the one open contract and compute only information available up to the current minute.
- **Receives:** Entry price and time, current executable quote, elapsed time, unrealized result, MFE/MAE to date, synchronized context, geometry, and approved Greeks.
- **Outputs:** A causal open-position feature row.

### Stage 5: HOLD/EXIT decision

- **Type:** Proposed learned model.
- **Responsibility:** Decide whether the expected value of keeping the position, including the cost of occupying the only trading slot, is better than selling now.
- **Receives:** The causal open-position feature row. The exact opportunity-cost target and final input list are **not yet specified**.
- **Outputs:** `HOLD` or `EXIT-NOW`.

If the output is `HOLD`, the trader evaluates the position again after the next completed minute. If it is `EXIT-NOW`, the trader sells using the executable exit-price rules. Hard safety rules still force an exit by the session deadline.

## 3. Exact role of P5

P5's current contract rule works as follows:

1. At each decision minute, inspect `spx_vwap_gap_points`.
2. If SPX is at or above VWAP, choose the **call** side.
3. If SPX is below VWAP, choose the **put** side.
4. Restrict the option ladder to currently eligible contracts on that side.
5. Choose the contract with the smallest absolute strike offset from ATM.
6. If contracts are equally distant, prefer the lower numeric offset, followed by the fixed contract ordering.
7. If no eligible contract exists on the selected side, emit no P5 opportunity.

Direct answers:

- **Does P5 decide when an opportunity exists?** Yes, in a broad mechanical sense. It emits an opportunity whenever at least one eligible contract exists on the VWAP-selected side.
- **Does P5 decide call versus put?** Yes. At or above VWAP means call; below VWAP means put.
- **Does P5 choose nearest ATM?** Yes. It chooses the eligible contract nearest to ATM on the selected side.
- **Can P5 select contracts away from ATM?** Yes, but only when the exact ATM or closer strikes are unavailable or ineligible. It still chooses the nearest eligible strike.
- **Does P5 generate opportunities almost continuously?** Yes. In the Stage-0 sample, it permitted 6,414 of 7,176 decision minutes, about 89.4%.
- **Is P5 a complete trading strategy, an entry heuristic, a contract selector, or some combination?** Its entry portion is a broad opportunity heuristic, a direction rule, and a contract selector. Historical P5 evaluations also paired that entry behavior with a fixed exit policy, making the no-model P5 baseline a complete mechanical strategy. It is not a learned model.
- **Which parts remain permanently hard-coded in the proposed redesign?** The current proposal keeps P5's VWAP-based call/put direction and nearest-eligible-ATM contract selection hard-coded. The learned flat-state model would decide whether to act on that P5 candidate.

## 4. Proposed flat-state model

The proposed flat-state model is a selective `WAIT/ENTER` model placed after P5.

- **Model family:** **not yet specified**. HGB is an available baseline, but no final model-family contract has been approved.
- **One training row represents:** **not yet specified**. The proposal implies one causal, flat-account decision minute with a valid P5-selected candidate, but this row contract is not frozen.
- **Inputs:** **not yet specified**. The signed 17-feature set is available, but the redesign has not frozen which fields the flat-state model will receive.
- **Target or label:** **not yet specified**. It must measure whether entering the P5 candidate now adds value relative to waiting, without merely rewarding the general profitability of P5 exposure.
- **Output:** `WAIT` or `ENTER`.
- **Does it choose timing?** Yes. This is its main job.
- **Does it choose direction?** No under the current proposal. P5 chooses call or put.
- **Does it choose the contract?** No under the current proposal. P5 chooses the nearest eligible strike to ATM.
- **How does it differ from H0-H3?** H0-H3 scored every eligible contract for a fixed exit policy and then used thresholds for entry. The proposed model would evaluate the single P5-selected opportunity and answer only whether to trade now or wait.
- **How does it avoid relearning P5's historical profitability?** The proposal requires judging the model's incremental benefit over matched P5 and exact-random exposure rather than treating total P5 profit as learned edge. The exact target, exposure matching, and acceptance test are **not yet specified**.
- **Is the design frozen?** No. It remains undecided and has not begun training.

## 5. Proposed open-state model

The proposed open-state model would control an already-open one-contract position.

- **Model family:** **not yet specified**. The unsigned Stage-2 draft proposes a transparent HGB baseline and a neural or sequence challenger, but neither has been approved as the final architecture.
- **Inputs:** Proposed causal inputs include current executable quote state, elapsed time, unrealized return, MFE/MAE and giveback to date, recent price velocity, synchronized market context, contract geometry, and approved internal Greeks. The final list is **not yet specified**.
- **Target or label:** **not yet specified**. It must compare selling now with continuing while accounting for the economic cost of occupying the single trading slot.
- **Output:** `HOLD` or `EXIT-NOW`.
- **Decision frequency:** Once per completed minute while a position is open.
- **Does it model the value of freeing the trading slot?** The proposed redesign requires this, but the exact method is **not yet specified**.
- **How does it account for future opportunities without hindsight?** **not yet specified**. Any final method must use only causal information available at the decision time; future opportunities cannot be supplied directly as model inputs.
- **How does it differ from the failed one-minute HGB pilot?** The failed pilot predicted only the next minute's executable-bid advantage. It did not value the newly available trading opportunities created by exiting. The redesign must model that serial opportunity cost rather than optimizing each open trade in isolation.
- **Is the design frozen?** No. It remains undecided and has not begun training.

## 6. What has been abandoned

- **Learned call/put selection:** Deferred from the current redesign. M0/M1 were allowed to choose calls or puts but did not beat P5. This does not prove learned direction can never work; it means it is not part of the current proposed first architecture.
- **Learned strike selection:** Deferred from the current redesign. P5's nearest-eligible-ATM rule beat M0/M1 in the pilot.
- **Learned contract selection:** The M0/M1 full campaign has been abandoned because the Stage-0 feasibility gate failed. A materially different future hypothesis could still be preregistered.
- **Original H0-H3 threshold entry system:** Quarantined and replaced in the proposed route. The 420 models remain benchmark evidence, but none established trustworthy incremental entry timing or earned selection.
- **M0/M1:** Abandoned as current contract-selector candidates because both lost to deterministic P5 selection.
- **Previous one-minute HOLD/EXIT objective:** Abandoned for the redesign. It optimized the next minute of the current trade but ignored the value and risk of freeing the single trading slot, and it lost under serial replay.

## 7. Direct conclusion

> Protocol101 currently has the hand-coded P5 rule that emits an opportunity whenever an eligible contract exists on the VWAP-selected side as its entry-opportunity rule, the hand-coded P5 rule that chooses calls at or above VWAP, puts below VWAP, and then selects the nearest eligible strike to ATM as its direction and contract-selection rule, and no validated learned entry-timing or HOLD/EXIT model. The proposed redesign would train a flat-state WAIT/ENTER model and an open-state HOLD/EXIT model with serial opportunity cost. It would hard-code nearest-ATM contract selection. This redesign has not begun training. The exact unresolved design decisions are the flat-state model family, row definition, input set, target and matched-exposure gate; the open-state model family, input set and opportunity-aware target; the causal method for valuing future opportunities; and the final acceptance gates.
