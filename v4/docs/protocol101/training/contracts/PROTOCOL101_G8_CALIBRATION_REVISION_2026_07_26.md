# Protocol101 Stage-1 G8 Calibration Revision - SIGNED FINAL

Status: **OWNER-SIGNED AND BINDING**

Effective date: 2026-07-26

This revision supersedes the original hard-pass G8 language in
`PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md` for the frozen H0-H3
Stage-1 campaign.

It changes only the eligibility role of G8. It does not change any fitted
model, feature, label policy, seed, fold, threshold, trade, fill, fee, PnL,
or G1-G7/G9 definition.

## Signed Replacement Language

> **G8 Calibration Diagnostic (v2, owner-signed 2026-07-26):**
> Out-of-fold expected calibration error remains required and must be reported
> for every seed using the frozen 10-bin definition and the payoff-score to
> realized-win confidence readout. ECE `<= 0.10` remains the diagnostic
> benchmark, not an eligibility threshold. While calibrated confidence does not
> abstention, contract selection, position sizing, routing, exits, or any other
> trading behavior, G8 is report-only. It cannot pass or fail Stage-1
> eligibility, G9 eligibility, or protected-holdout eligibility. The HGB
> training target remains fee-adjusted payoff / return-on-premium, never win
> probability.
>
> Before calibrated confidence may control any trading behavior, a separate
> action-conditioned calibration gate must be preregistered, smoke-tested,
> independently audited, and owner-signed. That future gate must evaluate the
> behavior actually controlled by confidence.

## Scope

This revision applies globally and retroactively to every frozen H0-H3 result.
It may not be applied only to H2 policy 5.

The required sequence is:

1. Reaggregate all frozen H0-H3 results without fitting or changing a model.
2. Apply G1-G7 as hard eligibility gates and report G8 for every row.
3. Independently audit the reaggregation and select at most one candidate
   using the already signed selection rule.
4. Only if H2 policy 5 remains the independently selected candidate, spend
   never-used seed 45 once for G9.

G9 remains exactly as signed: the fresh seed must independently satisfy G1,
G2, and G4. G8 is reported for seed 45 but is not part of G9 pass/fail.

## Evidence And Rationale

The independent gate-validity and integrity audit found:

- all 420 H0-H3 model units and model hashes verified;
- zero protected-holdout overlap and zero split-role overlap;
- no direct feature/label/future leakage defect;
- 7 G1-profitable hypothesis-policy rows, none passing the original hard G8;
- 21 non-G1 rows, 20 passing the original hard G8;
- the calibrated confidence readout was computed after action and slot
  selection and controlled no trading behavior;
- ECE and PnL were positively associated in this campaign
  (`Spearman rho = 0.6108`, `p = 0.00056`).

Evidence:

- `v4/audit/autoresearch/protocol101_stage1_gate_validity_and_integrity_audit_attempt001/summary.json`
- `v4/audit/autoresearch/protocol101_stage1_gate_validity_and_integrity_audit_attempt001/report.md`
- `v4/docs/PROTOCOL101_TRAINING_REGIMEN_AND_INTEGRITY_CONTROL_2026_07_26.md`

This amendment does not declare H2 or any other result eligible. Eligibility
must be recomputed globally from frozen evidence and independently selected
before G9.

## Owner Sign-Off

I approve the G8 replacement language above and its global H0-H3 application.
I authorize model-free reaggregation, independent selection, and, only if
earned by that process, the one-time seed-45 G9 run.

Signature: OWEN HEIDENREICH

Date: 07-26-2026
