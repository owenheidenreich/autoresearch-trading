# Scope activation defect V1 — signed V2 cannot be activated without invalidating itself

**Status: owner scope is signed; fit remains correctly refused. No model or economics ran.**

## Defect

The signed re-ruling authorizes the exact V2 fit and evaluation declaration self-hashes. Those
declarations, in turn, hash `v5/research/causal_day_policy_gate.py` as an implementation dependency.

The hashed gate still recognizes only the earlier `itm_depth_magnitude` reopening. To recognize the new
signed `serial_action_advantage_120m` scope, the gate must change. Any such change changes the gate hash,
so the V2 declaration verifier refuses before reaching the newly authorized gate. Regenerating a V3
declaration would repair implementation integrity, but V3 would not have either exact self-hash named in
the signed re-ruling.

Therefore there is no honest executable path through the current signed tuple:

1. unchanged gate + signed V2 → scope refusal;
2. changed gate + signed V2 → implementation-hash refusal; and
3. changed gate + regenerated V3 → outside the exact declaration hashes the owner signed.

Monkeypatching, suppressing an implementation-hash check, hand-constructing a `Reopening`, or editing the
signed document after signature would bypass rather than repair the gate and is prohibited.

## Root cause

The authorization document and declaration were ordered cyclically. A fit declaration may hash the final
gate implementation, or an owner ruling may bind the exact declaration hash, but both cannot change in
response to each other. The correct order is:

1. owner signs a stable semantic scope that does not name a not-yet-final declaration hash;
2. the gate pins that signed scope;
3. declarations are generated against the final gate and bind the signed scope hash; and
4. the fit runs only if the declaration's research-law projection is identical to V2.

## Smallest owner-controlled correction

Authorize one **mechanical reseal only**: preserve every V2 research choice, control, stopping law, family
count, risk limit and forbidden action byte-for-byte in meaning; pin the already signed scope in the gate;
then generate V3 declarations whose only permitted differences are authority/declaration identifiers,
implementation hashes, self-hashes and unused output/evidence path versioning.

Before fitting, an executable semantic-diff check must prove zero research-law differences between V2 and
V3. Any non-mechanical difference refuses the reseal. This amendment would not authorize another model,
label, horizon, selector, operating point, seed, loss, fold, control or retry.

That check is now implemented in `causal_day_declaration_reseal.py`. The complete V2 research-law
projection is pinned by test at
`3a7e26e78ad8c6f1c972ba779fd7a6ac1b640e95979dda49ce74891248499627`. Tests prove that schema names,
authority hashes and unused output-path versions may move while changes to the seed, label horizon,
selector floor, trade cap, family correction or risk limit are refused.

Until that narrow correction is explicitly authorized, the signed document is preserved unchanged and
the fit remains blocked.
