# G1 Direction Research

The ES-only opening-range and overnight-gap direction screen.

**State: RELEASED and FROZEN.** [`family.py`](family.py) is the frozen declaration of the 18-member
family, the causal clock, the economics, the fold and comparator rules, and the surrogate known-answer
gates. It computes nothing about returns and reads no bars; it exists so the search space cannot widen
after a result is seen.

Freeze hash: `f43b92c2ed33a84e53732983999fdf889d31453cfe8663e3e09af1f182d8db49`, reissued 2026-08-06
from `5ec8b5a4…` for a renamed home directory — one field of 213, the corpus path, with no outcome
inspected. See the [re-freeze record](../../work/g1-direction/REFREEZE_2026_08_06.md).

Code added here must use only information available at the decision time, clear the 0.358-point cost bar,
use one-position-at-a-time accounting with every no-trade day contributing zero, and pass the
matched-surrogate known-answer test **before** real economics are inspected. Session-shuffle nulls are
forbidden — see ledger row 183.

Plan and current step: [`v5/work/g1-direction/PLAN.md`](../../work/g1-direction/PLAN.md).
