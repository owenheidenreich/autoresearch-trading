# Job 24 result — Phases 0 to 2 executed, Phase 1 held at the owner, Phase 3 not started

**2026-08-13.** The plan is at [`PLAN.md`](PLAN.md). This records what executing it produced and why it
stops where it does.

## The short version

Phase 0 was run first, as the plan required, and it **cleared the plan's stated kill condition** — the
disputed break-even settled at the optimistic value, and by a wider margin than either candidate. But
carrying that correction into Phase 2 destroyed the plan's reason for existing: **occupancy is worth 0.69
accuracy points, not 3**, and the charter's own risk limits rule out the occupancy design at the account
size it was written for.

Phases 0 and 2 are complete and evidenced. Phase 1 is a vendor download and needs an owner signature.
Phase 3 is not started, because the cell it would freeze no longer exists.

## Phase 0 — settled, and it settled against the plan's reasoning

[Finding](../../research/findings/EXIT_PRICE_CONVENTION_2026_08_13.md) ·
[receipt](../../../v4/audit/autoresearch/exit_price_convention_2026_08_13/receipt.json)

The pooled near-ATM 60-minute break-even is **50.94%, not 54.24%**. The last-print convention is right and
dropping is wrong, on two independent grounds: dropping is a look-ahead filter that no bot could apply, and
the contracts it drops are **91.9% winners** whose last print stands +55% above entry. The plan's stated
reason for suspecting the pessimistic number — "dropping them removes trades that likely went to zero" —
is backwards. Illiquidity at this horizon is a symptom of winning.

Settled against real quotes on the 230 overlapping sessions: valuing the vanished contracts at the price
that actually existed instead of their last print moves the break-even by **0.0002 accuracy points**, and
**not one of the 161 was worthless**.

## Phase 1 — prepared, held

[Draft manifest](../../../v4/audit/autoresearch/protocol101_pathd_data_acquisition/paid_data_approval_manifest_spxw_history_pre2022_2026_08_13.json)

Extending the corpus to 2013 is a vendor request, which is a Tier-1 decision. The manifest is written to
the same shape as the one signed on 2026-08-13, its `owner_authorization` block is empty, and **no request
has been issued**. The existing signed manifest forbids any span outside 2022-06-01..2026-07-31, so this
needs its own signature rather than an extension of that one.

It is worth roughly half an accuracy point and costs $0.00. It is no longer urgent: the binding problem
found in Phase 2 is not sample size.

## Phase 2 — run, and it rules out every cell

[Finding](../../research/findings/OCCUPANCY_AND_CHARTER_RISK_2026_08_13.md) ·
[occupancy receipt](../../../v4/audit/autoresearch/hold_occupancy_2026_08_13/receipt.json) ·
[risk receipt](../../../v4/audit/autoresearch/occupancy_risk_2026_08_13/receipt.json)

| Hold | Trades/session | Break-even | Provable at | Plan said provable at |
|---:|---:|---:|---:|---:|
| 5 min | 36.1 | 51.88% | 52.52% | 53.57% |
| **10 min** | **22.0** | **51.01%** | **51.83%** | — |
| 15 min | 16.0 | 51.01% | 51.99% | 54.12% |
| 30 min | 8.8 | 50.69% | 52.05% | 54.95% |
| 60 min | 4.9 | 50.43% | **52.28%** | 57.11% |

Every bar fell by 1 to 5 points, and the spread across the whole range collapsed to 0.69 points. The
60-minute design the project already had needs **52.28%**, not 57.11%.

The charter risk check then rules the occupancy design out at a $10,000 account, and rules out the
60-minute design too. A near-ATM contract costs about $1,050, which is 10.5% of the account, so one losing
trade costs 6.1% of it and every cell keeps only **15-17% of its nominal occupancy** — the 60-minute cell
does not reliably get even one trade a session. The smallest account passing the three declared criteria is
**$250,000**. These risk figures were corrected on 2026-08-13 after a resampling defect was found that had
biased them pessimistic; the direction of the finding did not change and its severity fell.

## Phase 3 — not started, and should not be

Phase 3 freezes a hypothesis and spends the programme's last pre-committed attempt on it. It cannot be run
as written: its Phase 2 input was "15 minutes unless the risk check rules it out", and the risk check ruled
out 15 minutes along with everything else on the ladder.

Freezing something else instead would be choosing a cell after seeing the results, which is the failure
mode the plan's own ordering exists to prevent.

## Follow-up 2026-08-13: can a charter change give us a better chance?

Asked by the owner after the above. Three tests, then two draft amendments.
[Finding](../../research/findings/TICKET_SIZE_AND_THE_BINDING_SCREEN_2026_08_13.md).

**No, on risk limits.** Every ticket size from $50 to $3,200 and every daily breaker from 5% to 25% was
measured, 420 declared cells. Widening the breaker buys occupancy and leaves the survival floor unchanged
to within noise. Shrinking the ticket raises the accuracy bar steeply, because the round trip is 1.6% of a
$1,600 contract and 26.6% of a $50 one. There is no cheap ticket that is also winnable.

**Yes, on two specific clauses.** The 13% ceiling breaches the survival floor in 84% of simulated years at
a $10,000 account; 4% is the largest size that leaves room for a second trade inside the daily breaker. And
the charter has never had a minimum-account rule, which is what actually protects the floor.

**And yes, on the rule that blocks model building.** Training is gated on the ES screen, which is now
harder than a direct option screen at every horizon — 52.50–55.12% against 51.83–52.52%. The gate chain was
written before 909 option sessions existed.

**The number that decides everything:** the lowest bar this instrument offers is ~52.4%, and collecting it
needs about a $100,000 account. At $10,000 the survivable tickets ask for 55.1–59.5%.

## The build path, if the drafts are signed

1. **Charge the alpha ledger** for both amendments, since each is a constraint setting.
2. **Compute and record which corpus is binding** — today the option corpus at 10 minutes, 51.83%.
3. **Freeze one hypothesis**: entry family, hold, ticket size, moneyness band, exit rule. Content-hashed
   before any replay. The generator in [`research/autoresearch/`](../../research/autoresearch/) already
   enumerates 6 entry × 4 exit candidates and knows its own multiplicity.
4. **Known-answer campaign first** — both nulls under 5%, recovery at or above 80%. This is what stopped
   the last two attempts, and if it stops this one the economics are not read.
5. **Only then**: entry ranker out-of-fold, frozen; exit fitted on the frozen entry stream; validation
   packet; parity.

Steps 3 onward need G4, which needs the training precondition amended and separate owner authorization.
Steps 1 and 2 need nothing but a signature.

## What is now on the table, for an owner decision

1. **The bar is 4.8 points lower than the project believed.** 52.28% at 60 minutes rather than 57.11%.
   Nothing about that requires occupancy, a new corpus, or a charter change — it is the existing design,
   correctly measured.
2. **The account, not the statistics, is the binding constraint.** At $10,000 the survivable tickets ask
   for 55.1-59.5% accuracy; the 52.4% bar needs roughly $100,000. That is an owner question about capital,
   not a research one.
3. **Two clauses are worth amending and one is not.** Drafts written and unsigned:
   [ticket and account](../../governance/CHARTER_AMENDMENT_TICKET_AND_ACCOUNT_2026_08_13.md),
   [training precondition](../../governance/TRAINING_PRECONDITION_AMENDMENT_2026_08_13.md). The daily
   breaker is measurably not worth touching.
4. **Phase 1 remains free** and is the only part of the original plan still worth executing as written.

## What was not done, and why

- No hypothesis was frozen and no known-answer campaign was run.
- No vendor request was issued.
- No economics of any policy were computed; every number here is a property of the instrument.
- The reserved forward sessions from 2026-08-06 were not touched. The corpus ends 2026-07-31.
