# G1 Direction Research

The ES-only opening-range and overnight-gap direction screen.

**State: CLOSED 2026-08-09 with the verdict `UNDERPOWERED`.** The known-answer campaign passed both
null criteria and failed the recovery criterion: the gate's real detection floor is 8-16 net points per
session against a 0.358-point cost bar. **No member's profit and loss was ever computed.** Verdict:
[finding](../findings/G1_KNOWN_ANSWER_CAMPAIGN_2026_08_09.md).

The declaration below stands as the record of what was frozen and built. [`family.py`](family.py) is the frozen declaration of the 18-member
family, the causal clock, the economics, the fold and comparator rules, and the surrogate known-answer
gates. It computes nothing about returns and reads no bars; it exists so the search space cannot widen
after a result is seen.

Freeze hash: `157fe437998d968d41e7312a914dd65fa110b82c25600df0523c62a01699985a` (247 M1-eligible, 243
gap-eligible). Reissued twice, both times before any outcome existed: from `5ec8b5a4…` on 2026-08-06 for
a renamed home directory, and from `f43b92c2…` on 2026-08-09 when the owner excluded the seven sessions
on which SPXW does not trade. Both records:
[re-freeze](../../history/jobs/g1-direction/REFREEZE_2026_08_06.md).

Code added here must use only information available at the decision time, clear the 0.358-point cost bar,
use one-position-at-a-time accounting with every no-trade day contributing zero, and pass the
matched-surrogate known-answer test **before** real economics are inspected. Session-shuffle nulls are
forbidden — see ledger row 183.

Closed packet: [`v5/history/jobs/g1-direction/`](../../history/jobs/g1-direction/).
