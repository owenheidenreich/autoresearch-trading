# Lean S4/S5 entry autoresearch result

## Verdict

`NULL_NO_NEW_ENTRY_EDGE`

No S4/S5 candidate both survived the five guards and demonstrated a
forward-stable incremental effect over its paired old-family control. The
protected holdout remains sealed, no entry model may be built, and the
pre-registered pivot is exit/lifecycle research.

## Frozen experiment

- New candidates: 72 (S4=36, S5=36).
- Paired old-family controls: 72 (S4→S0, S5→S3).
- Fixed multiplicity budget: 144 evaluated configurations.
- Development data: 214 allowed sessions, 442,963 candidate rows.
- Foundation stability: exact code, session-assignment, and 642 source-file
  hashes reproduced before fold decoding and again after the run.
- Holdout opens: 0.
- Entry-model artifacts: 0.

S4 adds the causal volatility/regime/time family to signed-17. S5 adds the same
family to signed-17 plus the previously tested microstructure/Greek widening.
The family contains completed-minute VIX level and 5m/15m changes, VIX minus
annualized 15m SPX realized volatility, minute-of-session/time-to-15:55, 5m/15m
SPX realized volatility, and 5m/15m cumulative session-range expansion. These
are model inputs, not gates.

Historical SPX/VIX rows were rebuilt from official ThetaData per-session files
under the corrected-v3.2 clock law: `event_time` is bar-open and becomes
available at `event_time + 60 seconds`. All 214 development sessions were
checked for finite S4/S5 values before preregistration, with no protected-session
intersection.

## Guard attribution

| Guard | Passed |
|---|---:|
| G1 controls + paired feature attribution | 0/72 |
| G2 forward stability | 24/72 |
| G3 minimum power | 72/72 |
| G4 economic bar | 39/72 |
| G5 leakage tripwire | 72/72 |

G1's ordinary negative-control portion passed for 24 candidates, but its paired
incremental-attribution portion passed for only 2; no candidate passed both.
All 24 forward-stable candidates used the short 25-minute exit. No mid or long
exit candidate passed G2. This reproduces the prior finding that the stable
effect is an exit-policy effect rather than an entry-feature effect.

The two candidates that passed paired attribution were both long-exit variants.
Neither beat the negative controls, neither was forward-stable, and neither
passed the economic-fold bar. Their mean candidate OOF rank correlations were
negative, so they cannot support an entry-edge claim.

The best raw mean economics were +$49.04/session for S4 with the short exit, but
that candidate added only +0.0055 mean OOF rank correlation and lost
-$6.16/session versus its signed-17 paired control. The best worst-fold economics
were +$35.94/session for S5 with the short exit, but its mean rank correlation
fell -0.0269 versus S3. These are not incremental entry effects.

## Immutable evidence chain

- Preregistration SHA-256:
  `e0a9a6a434efb395380c7e54325c54ec12a30c5530aee0ff21c5232b4df50347`
- Search-results SHA-256:
  `ee0d9eceb1eb4d02293e62f708ba3cd7c11a04f0c37b5bee9ee2a610bbc61ab4`
- Verdict SHA-256:
  `fa0087c15f80cf73c29ef8396895909b6fe395530cf7f99dfd7cf696d76b57f6`

## Next scope

Close this entry hypothesis without holdout access. The evidence-backed next
direction is exit/lifecycle research, with the prior caveat that it likely needs
a larger corpus before a worthwhile fit campaign.
