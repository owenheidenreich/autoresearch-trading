# Amendment: what must be proven before an entry or exit model may be fitted

**Status: DRAFT. NOT SIGNED. Binds nothing.**
Proposed 2026-08-13. Amends the training precondition in [`AGENTS.md`](../AGENTS.md) §1 and the G4 row of
the gate chain in [`STATUS.md`](../STATUS.md). Follows the amendment procedure established by the signed
[position-sizing amendment](CHARTER_AMENDMENT_POSITION_SIZING_2026_08_13.md).

## One sentence for the owner

The rule "no option-model training until the ES direction screen passes" was right when ES was the larger
corpus, but 909 free option sessions reversed that, and the ES screen is now **harder at every horizon** —
so the rule is blocking work behind the tougher of two tests rather than the one that matters.

## What the rule says today

> Model training on the option layer is prohibited until G1 direction passes.

Its purpose is sound and this amendment does not touch it: **a model fitted on top of a direction signal
that does not exist cannot create information.** Nothing below weakens that. What changes is which
measurement is allowed to demonstrate the direction signal.

## Why the calibration moved

The bar the rule is enforced at — **65.73% directional accuracy** — was derived on 2026-08-12 from:

- **251 owned option sessions**, one trade each;
- a payoff pair charged the **fee-only $3.08**, not the $25 round trip that includes crossing the spread.

Both inputs have since changed, and neither by choice:

- the option corpus is **909 sessions**, acquired for $0.00 on 2026-08-13, and a serial one-position clock
  puts 4.9 to 36.1 trades in each of them;
- the payoff pair is measured on the exit-price convention settled the same day, at full measured cost.

## The measurement

What each candidate screen can prove, at 80% power and a one-sided 95% standard for a single
pre-registered hypothesis
([receipt](../../v4/audit/autoresearch/screen_requirement_2026_08_13/receipt.json)):

| Horizon | Option screen, 909 sessions | ES screen, 2,435 sessions | Harder |
|---:|---:|---:|---|
| 5 min | **52.52%** | 55.12% | ES |
| **10 min** | **51.83%** | 53.86% | ES |
| 15 min | **51.99%** | 53.36% | ES |
| 30 min | **52.05%** | 52.80% | ES |
| 60 min | **52.28%** | 52.50% | ES |

**The ES screen is harder at every horizon**, by 0.2 to 2.6 accuracy points. It was the easier one when the
option corpus held 251 sessions; buying 909 reversed it, and nothing in the gate chain noticed because the
chain was written before the purchase was possible.

## What is proposed

**Replace** "until G1 direction passes" with:

> Option-model training is prohibited until a **pre-registered direction screen passes on the corpus that
> is binding**, where binding means the corpus on which the required accuracy is *higher*. Which corpus
> that is must be computed and recorded before the screen is frozen, not chosen after.

Three things this deliberately does **not** do:

1. **It does not lower any statistical standard.** Same power, same one-sided 95%, same known-answer
   campaign, same requirement that both nulls pass before any economics are read. Only the instrument the
   screen runs on changes.
2. **It does not retire G1.** The ES screen remains available and remains the right choice whenever it is
   the binding one. Ledger rows 181 and 187 stand.
3. **It does not authorize training.** G4 continues to require separate owner authorization, and the
   feature-admission ledger, the knob registry and the divergence register all continue to bind.

## Why "binding" rather than "either"

Left as a choice, this clause would be a free multiplicity: run both screens, report the one that passes.
Requiring the *harder* one removes that. The computation is mechanical, it happens before the freeze, and
its result is recorded in the pre-registration.

## What this unblocks, concretely

A direct option screen at the 10-minute horizon, needing **51.83%**, against a break-even of 51.01%. That
is a margin of 0.82 accuracy points — small, and honestly so, but it is a bar this project can state,
freeze, and test rather than one it has already failed twice.

## What it does not fix

**The account.** The companion [ticket-and-account draft](CHARTER_AMENDMENT_TICKET_AND_ACCOUNT_2026_08_13.md)
shows a $10,000 account cannot honour the 50% survival floor at any tradeable ticket size. A screen passing
would establish that a signal exists; it would not make the signal fundable at the current account. These
are separate decisions and should not be bundled.

## Review and expiry conditions

1. **Void if the option corpus stops being the binding one** — for instance if a materially larger ES
   corpus or a higher-occupancy ES design changes the comparison. The clause recomputes rather than
   assumes.
2. **Void if the option payoff pair is re-measured and moves the bar by more than 1.0 accuracy point.**
   Every figure rests on a $25 round trip carried from the owned quote corpus into 2022–2025 sessions
   whose spreads were never observed.
3. **Expires 2027-02-13** if no screen has been frozen under it, so a released precondition does not sit
   open indefinitely.

## The honest case against this amendment

Three arguments, stated because an amendment that only argues its own side is not evidence.

- **It relaxes a rule immediately after two failures.** The programme stopped twice at the recovery
  criterion, and the natural response to a stop is not to re-derive the bar. The defence is that the bar
  moved for a measured reason — a corpus that did not exist when the bar was set — and that the new bar is
  computed by the same machinery, not chosen.
- **The corrected bars partly reflect a regime.** On the most recent 250 sessions the 60-minute break-even
  is 49.36% against 54.00% in 2022: long premium has become cheaper relative to realised movement across
  this corpus. A screen frozen on the pooled number is being judged against an average of five materially
  different years, and a model fitted on them inherits that.
- **The margin is thin.** 0.82 accuracy points at ten minutes is a real but small gap, and it assumes the
  intra-session independence measured here (design effect 1.00 to 1.12) holds for a policy rather than
  just for direction.

## Alpha

Re-deriving a precondition is a constraint setting, and the autoresearch ledger prices constraint settings
as it prices experiments. This draft must be charged before any screen is frozen under it.
