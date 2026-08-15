# Scoped reopening: the causal day trader may be fitted

**Signed by the owner 2026-08-14, in session. Supersedes nothing; scopes three blockers.**

This releases the fit gate at [`research/causal_day_policy_gate.py`](../research/causal_day_policy_gate.py)
for **one named experiment**. It is not a general reopening of option-model fitting, of the long side, or
of the neural sample rules. Anything outside the scope below stays refused, and the gate still fails
closed.

## 1. What is released, and on what authority

| Blocker | Release | Authority |
|---|---|---|
| **G1 `UNDERPOWERED` prohibits option-model fitting** | Released for this experiment only | Owner authorization, [`work/entry-exit-attribution/GOAL.md`](../work/entry-exit-attribution/GOAL.md) §9: activation "authorizes Tier-3 local computation and model fitting on owned pre-cutoff data." Precedented — jobs 29, 32, 36, 37 and 41 each ran on the same per-job authorization |
| **[`DO_NOT_RETEST.md`](../research/history/DO_NOT_RETEST.md) closes selective long-side entry models on this corpus** | Released **only** for the ITM-depth magnitude label at 60/90/120-minute horizons with origin-owned exits | Owner ruling on the §16 conflict, on the evidence in §2 below |
| **20 sessions per neural parameter; 1,140-session floor** | The **ratio of 20 is kept**; its **unit** changes from sessions to causal decision states for this comparison. Floor waived. See §4 | Pre-registered amendment under the knob's own unfreeze condition |

## 2. The evidence for the narrow reopening, stated with its counter-evidence

**This is the weakest of the three releases and must not be described as well-supported.** An honest
statement of both sides:

**For reopening:**

- Row 41 (2026-08-14) found that every prior exit study measured under a **random-entry** regime, in which
  the conditional value of holding cancels exactly. Measured conditionally, the 36.7% of trades that
  become big movers return **+$675** held 60 minutes. That is a genuinely new mechanism observation, and
  the ledger rows predate it.
- Row 332's stated reopening condition is **"a magnitude or volatility label."** The ITM-depth target
  (10/20/30 SPX points) is a magnitude label, not the direction or percentage-excursion labels that failed.
- Job 42 (2026-08-14) established that the owned 243 sessions **can** resolve this target if the effect is
  the claimed size — so this is not a knowingly underpowered run.

**Against reopening, and not resolved by any of the above:**

- Row 340's decisive measurement is that the long-side edge is **absent with the spread removed entirely**
  — mid-to-mid −$3.5/trade averaged over an out-of-time run. This is not a cost problem that a better
  label repairs.
- Row 335's conditional drift census found **0 of 375 declared causal cells** clearing zero, and 77
  significantly negative. A rule is a function of state; a census with no positive state admits no rule.
- Row 338 closed the **side and the instrument**, not a rule set.
- Job 42 also measured the bar this must clear: the 30-point target needs a **precision lift of 11.1x**
  over its 1.57% base rate. Nothing this project has measured has produced better than about 1.24x.

**The owner is proceeding with the odds understood.** The purpose of §3 is to make sure that if this
fails, it fails visibly and for a legible reason, and that it cannot succeed spuriously.

## 3. Pre-committed kill conditions

Declared **before** the fit, and binding on the result. Each encodes a specific failure this project has
already suffered. A run that violates any of these is a **negative result** and must be recorded as one;
no re-scoring, re-labelling or operating-point search may follow within this reopening.

1. **Mid-to-mid gross must be positive.** Score every trade with the spread removed entirely. If gross
   mid-to-mid P&L per trade is ≤ 0, the experiment is over. *(Row 340: the edge was zero before costs, so
   no execution improvement could rescue it.)*
2. **It must beat a composition-matched control** matched on side, delta and premium — not a random
   control. *(Rows 336 and 337: the matched control reached the same hit rate; the apparent edge was
   instrument choice.)*
3. **It must beat its own shuffled-label null** out of sample. *(Row 29's model lost to its null.)*
4. **A per-feature timestamp audit must pass**, column by column, before economics are read. *(The
   2026-08-13 leak: an entry feature read the minute after the decision and the shuffled null structurally
   could not catch it.)*
5. **No slot may be filtered on anything measured after the entry minute.** *(The 18.7% look-ahead filter
   worth 3.27 accuracy points.)*
6. **Chronological out-of-sample only**, with the declared family size passed to the corrected bootstrap.
7. **No session on or after 2026-08-06** may enter any fitted or scored population. The forward
   confirmation reservation is untouched by this ruling.

## 4. The amended unit for the sessions-per-parameter rule

The frozen rule required 20 independent **sessions** per trainable parameter and an absolute floor of
1,140 sessions. The argument below is pre-registered here rather than derived from any result.

**The ratio of 20 is not what is wrong; the unit is.** Twenty-per-parameter is a reasonable guard against
an overfitted fit, and it is kept unchanged. This project applied it with *sessions* as the unit because
the session is the independent unit for **economic inference** — the session-block bootstrap resamples
sessions, and that remains correct and untouched. But the unit relevant to **fitting** is the decision the
model is asked to make. The corpus holds **93,798 causal minute states** across those 243 sessions, and
the policy makes a decision at each one.

Charging the fit at the resolution of the economic inference conflates two different questions. It is the
reason a 676-parameter shallow model — which this project is content to fit elsewhere without objection —
would nominally require 13,520 sessions.

**What the rule protects against is retained, and strengthened.** Its purpose is to stop an overfitted
model being presented as skill. That is delivered here by chronological out-of-sample evaluation and by
kill conditions 2, 3 and 6, which bind on held-out sessions. Those catch overfitting **empirically**,
which a parameter count can only approximate.

**Scope.** For this comparison only: 20 **causal decision states** per trainable parameter, on the 93,798
states in the owned corpus, with the absolute session floor waived. The frozen 20-sessions-per-parameter
knob and the 1,140-session floor are **unchanged in the registry** and remain in force for every other
purpose, including any promotion decision. This amendment authorizes a **comparison**, never a promotion.

At 93,798 states this admits up to **4,689 trainable parameters**, which covers four of the five declared
architectures:

| Architecture | Parameters | Admitted |
|---|---:|---|
| `shallow_joint` | 676 | yes |
| `shallow_four_head` | 720 | yes |
| `neural_joint` | 1,252 | yes |
| `neural_four_head` | 1,296 | yes |
| `four_independent` | 4,920 | **no** — and separately conditional on prior shared-head evidence |

The comparison must report the shallow control alongside every sequence model, and **neural complexity
earns credit only if it beats that control out of sample**, as `GOAL.md` §1 already requires.

## 5. What remains barred

- Any long-side selective model on this corpus **outside** the ITM-depth magnitude label and the declared
  horizons.
- The short side and defined-risk spreads, which remain barred by the long-premium-only charter clause and
  are a separate owner decision (STATUS §17).
- Promotion of any resulting model, real or paper orders, vendor contact, data purchase, and any use of
  reserved sessions.
- Re-running this experiment with a changed label, horizon or operating point after seeing a result. This
  reopening is spent on one declared comparison.
