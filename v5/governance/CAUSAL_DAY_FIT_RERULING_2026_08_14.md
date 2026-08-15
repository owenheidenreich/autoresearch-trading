# Re-ruling of §4: the budget is measured, the width is set by it, the fit may proceed

**Signed by the owner 2026-08-14. Lifts the suspension of the same date. Supersedes §4 of the
[reopening](CAUSAL_DAY_FIT_REOPENING_2026_08_14.md); every other section of that document stands
unchanged.**

## 1. A withdrawal first

The [suspension](CAUSAL_DAY_FIT_SUSPENSION_2026_08_14.md) was issued on the finding that all five of §4's
parameter counts were wrong by 2.4–3.0x. **That finding is withdrawn. §4's counts were correct.**

The counts were recomputed at `ArchitectureDimensions.hidden_size`'s *dataclass default of 16*. Every
recorded artifact in this project was built at **width 8**, set explicitly at
[`ops/record_causal_day_architecture_interfaces.py:23`](../ops/record_causal_day_architecture_interfaces.py).
At width 8 the counts are 676 / 720 / 1,252 / 1,296 / 4,920 — exactly what §4 stated.

| Width | `shallow_joint` | `shallow_four_head` | `neural_joint` | `neural_four_head` | `four_independent` |
|---:|---:|---:|---:|---:|---:|
| 8 — what §4 meant, and was right about | 676 | 720 | 1,252 | 1,296 | 4,920 |
| 16 — the unused dataclass default | 1,604 | 1,688 | 3,780 | 3,864 | 14,952 |

The review verified the arithmetic and not the premise. The arithmetic was right; the premise was wrong.
**The trap is now removed at source: `hidden_size` has no default, so no caller can silently inherit a
width nobody chose.**

## 2. What actually binds, and it is not the counts

The suspension's *stated* reason is withdrawn. Its *effect* was correct, because a genuine defect was found
in the same review and it is larger.

§4 charged the 20-per-parameter rule against **93,798 raw decision states**, arguing the unit relevant to
fitting is the decision the model makes. That was never measured. Measured
([finding](../research/findings/EFFECTIVE_SAMPLE_SIZE_2026_08_14.md)), those states are worth:

| Route | Effective observations | Budget at 20:1 |
|---|---:|---:|
| Integrated within-session autocorrelation | 2,879 – **7,557** | 143 – **377** |
| Clustering design effect | 590 – 1,016 | 29 – 50 |

Autocorrelation times run 10.2–26.8 minutes, which is what a 60-minute forward label must show.
**§4's budget of 4,689 parameters was too large by roughly 12x on the generous route and 100x on the
conservative one.** At width 8 the smallest architecture, 676 parameters, exceeds even the generous budget.

## 3. The ruling

1. **The 20:1 ratio is kept.** It is charged against **measured effective observations**, not sessions and
   not raw states. This is the faithful reading: the rule always counted independent units, and the
   effective count is the measured number of them.
2. **The budget is 377 trainable parameters** — the generous route, on the sparsest scoped label
   (`reached_30_itm_60m`, 7,557 effective observations). The generous figure is chosen deliberately so the
   comparison cannot be accused of being sized by the harshest available method.
3. **`causal_day_hidden_size` is frozen at 3**, and is registered in
   [`knobs.py`](../research/knobs.py). Three is the largest width whose four comparison architectures fit
   the budget. **The width is set by the evidence, not chosen for capacity.**

| Architecture | Parameters at width 3 | Budget 377 |
|---|---:|---|
| `shallow_joint` | 226 | fits |
| `shallow_four_head` | 245 | fits |
| `neural_joint` | 322 | fits |
| `neural_four_head` | 341 | fits |
| `four_independent` | 1,250 | **refused** — and separately conditional |

4. **The budget binds every architecture, shallow included.** §4's version applied the ratio only to
   sequence models. Effective sample size is a property of the label and corpus, so a shallow parameter
   costs the same evidence as a sequence parameter. Enforced in `fit_blockers`.
5. **The seven kill conditions of §3 are unchanged and still binding**, including the decisive one:
   mid-to-mid gross must be positive.

## 4. The limitation this ruling accepts, stated plainly

**The conservative route says nothing fits.** At a design-effect budget of 29–50 parameters, no
architecture in the family is admissible — width 3's smallest is 226. This ruling proceeds on the generous
route, and that is a judgement, not a measurement.

What that means for any result: a comparison run under this ruling is **sized at roughly 4–8x the
conservative evidence budget**. A positive result must therefore be treated as provisional against that
uncertainty, and the design-effect figure must be reported alongside it rather than omitted. It is not a
reason to withhold the comparison — the alternative is no comparison at all — but it is a reason no
outcome here promotes anything.

## 5. Suspension lifted

The suspension is lifted for the scope above and no wider. `SUSPENSION_LIFTED_BY` in
[`causal_day_policy_gate.py`](../research/causal_day_policy_gate.py) names this document. Everything §5 of
the original reopening barred remains barred, including the short side, spreads, promotion, and re-running
the comparison with a changed label or horizon after seeing a result.
