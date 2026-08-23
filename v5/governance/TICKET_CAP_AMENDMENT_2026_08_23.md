# Ticket cap raised to $2,500

> **STATUS: SIGNED AND IN FORCE. Owner ruling 2026-08-23.**
> Amends the $2,000 absolute ticket cap set by
> [`CHARTER_AMENDMENT_TICKET_AND_BREAKER_2026_08_16.md`](CHARTER_AMENDMENT_TICKET_AND_BREAKER_2026_08_16.md).
> Everything else in that amendment — the 20% daily breaker, the −40%-or-wider declared stop, the
> compounding account, dollars-not-percentages — is **preserved unchanged**.

## The measurement that prompted it

The 2026-08-23 terrain join measured, for each strike and clock, the SPX move a contract needs to
break even and how often the tape delivers it. The **highest-probability cell in the entire grid is
09:35 at-the-money: it needs 1.4 points and gets them on 73% of sessions.**

**The $2,000 cap excluded that cell.** At the true open the at-the-money ask is **$2,278**, and at
09:35 it is **$2,266**. The cap was written for capital safety and was silently doing something else:
**pushing contract selection away from the lowest-requirement strikes and toward ones needing
multi-point moves.** The measured consequence across job 46 was a **$579 average traded ticket** in
exactly the far-OTM corner where required moves are largest.

## What changes

**`MAX_ENTRY_TICKET_USD` and every sibling constant move from `$2,000` to `$2,500`.** The figure is
chosen to admit the at-the-money contract at the open with headroom, not as a round number.

## The risk this buys, stated plainly rather than buried

On the signed $10,000 starting account:

| | $2,000 cap | $2,500 cap |
|---|---:|---:|
| Share of equity per ticket | 20% | **25%** |
| Deployed at two tickets/day | 40% | **50%** |
| A −40% stop on one ticket | −$800 (8% of equity) | **−$1,000 (10% of equity)** |

The owner has seen these figures and ruled. The 20% daily breaker and the declared stop are unchanged
and remain the binding protections; this raises the size of a single loss, not the number of them.

## What this does NOT do

It does not authorise more tickets — **two per day stands**. It does not make the at-the-money
contract mandatory or preferred; it makes it *admissible*, and whether it should be bought is a
question for the entry rule. And it does not license buying the most expensive available contract:
the terrain join says near-the-money is favourable **early**, and by 15:00 the same strike needs
3.3 points and gets them 46% of the time.
