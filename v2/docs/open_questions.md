# Open Questions

This file contains the active unresolved questions for the current phase. Resolved items move to `decision_log.md`.

Format:

- `ID | priority | area | question | next: concrete next step`

## Active Questions

- `OQ-001 | high | data quality | Do we need a dedicated audited raw-data track before architecture work continues, or is the current frozen exact-chain dataset sufficient for the next experiment block? | next: define the minimum data-audit acceptance checklist for the frozen dataset`
- `OQ-003 | high | artifact compatibility | What is the correct promotion policy when raw local checkpoints are incompatible with the restored architecture and no compatible promoted artifact bundle exists? | next: decide whether a compatible official artifact is a hard prerequisite before any local promoted checkpoint is trusted`
- `OQ-006 | medium | data quality | Which exact data-quality checks belong in the permanent pre-GPU gate versus a separate audited data track? | next: split mandatory checks from deeper audits`
- `OQ-007 | medium | research memory | Which recurring facts should graduate out of lab_notebook.md into decision_log.md or program.md after each experiment block? | next: define the promotion rule for notebook content`
- `OQ-008 | high | training objective | Given that the KL target is near-uniform at SOFT_TEMP=0.20 (median max prob 0.23) and contract features have zero predictive power alone, should the task be decomposed into direction-first (call/put) then strike selection? | next: design the direction-first decomposition for exp_098`
- `OQ-009 | high | training objective | Should bars with top margin < 0.01 (30.6% of data) be excluded from the selection loss? These are noise bars where any contract is equally good. | next: implement margin-filtered selection loss for exp_097`

## Resolved Questions

- `OQ-004 | resolved | The tanh-bounded score head (exp_092) collapsed due to gradient saturation. Bounded approaches that saturate are not viable for the contract scoring task.`
- `OQ-005 | resolved | LOOKBACK=1 (exp_093) caused call-side collapse. The temporal context is load-bearing for direction balance, despite the logistic regression diagnostic suggesting otherwise.`

## Current Blockers

- `OQ-003`
- `OQ-008`
