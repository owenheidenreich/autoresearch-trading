# Autoresearch v2 replacement and first development results

Date: 2026-08-02

Status: implemented and exercised; development-only, non-promotable

## Decision

Adopt `v4/research/autoresearch_v2` as the Protocol101 hypothesis-screening
orchestrator. Retire governed Path-D as a screening loop. Preserve Path-D's
causal corpus, feature-parity work, two-clock labels, simulator-v5, and protected
holdout machinery. Heavy governance begins only after a typed experiment emits
`PROVISIONAL_EDGE` under paired development evidence and strict serial replay.

No experiment in this packet emitted `PROVISIONAL_EDGE`. The protected holdout
therefore remains sealed.

## What the replacement enforces

The engine compiles typed `hypothesis.json` files and fails closed before fit
when the experiment uses a future-session threshold, a missing live twin,
intraday open interest, an unavailable target, a non-development role, an
incomplete exposure match, an excessive fit budget, or a noncanonical terminal
status set.

Executable runs additionally enforce:

- byte-for-byte foundation verification before pickle decode;
- five expanding, chronological OOF folds;
- mutate-future invariance on every fitted or cached fold model;
- OOF cache keys over `(features, target, model, fold, seed, foundation)` and
  not over thresholds;
- session-blocked family maxT correction;
- exact paired component deltas before serial economics;
- simulator-v5 realized-exit occupancy, affordability, daily-loss, fee, and
  account-continuity semantics;
- a global semantic registry, with explicit history links for deliberate
  reproductions, so a renamed Goal cannot become a new hypothesis;
- terminal results limited to `INVALID_EXPERIMENT`, `MECHANICAL_FAILURE`,
  `UNDERPOWERED`, `NO_INCREMENTAL_EDGE`, `EXIT_ARTIFACT`, `PROVISIONAL_EDGE`,
  and `CONFIRMED_EDGE`.

The frozen development foundation contains 164 verified two-clock sessions
from 2025-08-01 through 2026-03-31. It has foundation hash
`9916e8aaa7753f7cc671ff15a98c44c34dfee8cd0ffac24e81803f20ba05b349`.
Holdout access count is zero.

The legacy `signed17` family name currently resolves to an exact 18-column
tuple in both the prior lean harness and this compiler. Autoresearch v2 freezes
the actual tuple and reports 18 columns; it does not silently alter the runtime
entry model. The name/count mismatch should be corrected as nomenclature in a
separate compatibility-preserving change.

## First compiled family

The main dollar-PnL family uses the same fixed threshold
(`OOF prediction > $20`) and the same frozen 120-minute reference exit unless
the exit itself is the tested component. Its family correction spans short-25
exit, direction, contract ranking, wait-5, wait-15, and the multi-policy target.
The exact policy-neutral M0/M1 row below is a separate typed target experiment:
it selects one contract per opportunity from decision-local path utility and
uses its own five-contrast family correction.

| Experiment | Paired development result | Terminal status |
|---|---:|---|
| 25-minute exit vs frozen 120-minute exit | Strict matched replay: +$12.82/session; 95% CI -$158.72 to +$184.35; p=0.441; MDE80=$242.54; 242 matched trades/119 sessions | `UNDERPOWERED` |
| Wait 5 minutes vs enter now | Cheap screen: +$18.03/session, family maxT p=0.026. Strict matched replay: +$12.31/session; 95% CI -$46.23 to +$70.85; p=0.339; MDE80=$82.77; 237 matched trades/119 sessions | `UNDERPOWERED` |
| Wait 15 minutes vs enter now | Cheap screen: +$30.70/session; family maxT p=0.0428; MDE80=$36.62. Not the preregistered primary timing contrast for serial routing | part of timing `UNDERPOWERED` |
| Call-vs-put direction | +$96.62/session; 95% CI -$77.74 to +$270.97; family maxT p=0.564; MDE80=$246.46 | `UNDERPOWERED` |
| Full-epoch within-minute contract ranking vs nearest ATM | -$7.37/session; 95% CI -$17.98 to +$3.24; family maxT p=1.0; 22,272 pairs/119 sessions | `NO_INCREMENTAL_EDGE` |
| Multi-policy path-value target vs single reference target | -$6.55/session; 95% CI -$16.70 to +$3.60; family maxT p=1.0; 8,681 pairs/116 sessions | `NO_INCREMENTAL_EDGE` |
| MFE/MAE/time-to-breakeven heads | Required target fields are absent from the two-clock entry corpus; compiler refused fit | `INVALID_EXPERIMENT` |
| Exact existing policy-neutral M0/M1 target expanded beyond five sessions | M1 vs M0: +0.000634 utility; 95% CI -0.002453 to +0.003721; family maxT p=0.7566; MDE80=0.00437; 120 OOF sessions | `NO_INCREMENTAL_EDGE` |

The short-25 absolute replay made $7,552 across 826 trades while the 120-minute
absolute replay made $2,783 across 244 trades. Those totals are not the causal
comparison because the exit changes occupancy and therefore trade count. The
matched lockstep result above is the relevant estimate: +$1,525 total over the
same 242 entries, with variance too large to establish a practical edge.

At the observed strict-serial variance, reaching MDE80=$25/session would require
roughly 11,200 sessions for the short-25 contrast and 1,305 sessions for wait-5.
Adding a few months cannot resolve either question under the present estimand.

## Interpretation

1. The prior short-25 observation reproduced in sign and approximate magnitude,
   but not at actionable power. It is a real lead only in the colloquial sense;
   its research status is `UNDERPOWERED`, not edge.
2. Wait-5 and wait-15 are the only component screens that survived family maxT.
   Wait-5 did not survive the real trading game at usable precision. This is the
   best development lead, but it does not justify confirmation or holdout use.
3. Expanded M0/M1-style contract ranking is now a powered negative at the
   component-screen level. The upper 95% bound is below the preregistered
   +$10/session practical bar.
4. The tested multi-policy target did not improve selection. More mixtures of
   existing policy PnLs are not the recommended next move.
5. Direction remains unresolved because matched side outcomes are extremely
   noisy. It should not consume a fresh epoch in its current form.
6. The exact preexisting path target was also expanded: executable returns at
   1/2/5/10/15 minutes, MFE, MAE, time to breakeven, and barrier ordering. M0
   and M1 both beat exact random by about 0.039 utility (family maxT p=0.0002),
   proving that the machinery detects structure. Neither beat deterministic
   nearest ATM at the +0.01 practical bar, and M1 added only 0.000634 over M0.
   The old five-session M0/M1 clue is therefore closed as no incremental edge.

## Recommended next gates

1. Do not open the protected holdout. No candidate earned it.
2. Materialize exact two-clock exit fields for the remaining April-July
   development sessions so the compiler can use all 214 already-open development
   sessions. Treat this as completeness work, not fresh confirmation; it will not
   by itself solve the observed power deficit.
3. The exact existing MFE/MAE/breakeven path utility is now materialized and
   tested in a separate target-only cache. Do not resubmit M1 feature variants
   against it under new names. A future target experiment must change the
   economic estimand, not merely reweight the same utility components post hoc.
4. If timing research continues, preregister a deployable wait policy including
   what happens when the same contract is no longer tradable after 5/15 minutes.
   Use a lower-variance opportunity definition such as one frozen signal per
   session or one per fixed time block, while retaining raw strict-serial dollars
   as the final economic gate. This is a new semantic hypothesis, not threshold
   tuning of the current result.
5. Reserve fresh rolling research epochs for one independently selected survivor.
   Once opened, each epoch becomes development data. Keep the existing protected
   holdout for a genuinely new candidate that passes power before access.

## Evidence locations

- Engine: `v4/research/autoresearch_v2/`
- Typed hypotheses: `v4/research/autoresearch_v2/hypotheses/`
- Frozen foundation: `v4/research/autoresearch_v2/foundations/development_2025-08-01_2026-03-31_two_clock.json`
- Final development packet: `v4/audit/autoresearch/autoresearch_v2_development_epoch_2026_08_02_attempt004/`
- Exact full-epoch path-target packet: `v4/audit/autoresearch/autoresearch_v2_policy_neutral_path_full_epoch_2026_08_02_attempt001/`
- Global semantic registry: `v4/audit/autoresearch/autoresearch_v2_semantic_registry.jsonl`

No broker, live market data, paper/default registry, promotion state, protected
holdout, or real-money path was accessed or changed.
