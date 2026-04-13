# Open Questions

Resolved items move to `decision_log.md`.

## Active Questions

- `OQ-001 | medium | data quality | Do the anomaly-audit flagged dates justify any dataset action? The frozen v4_exact_chain dataset has been stable through exp_074-144. | next: only revisit if new experiments show trace overlap with flagged dates`
- `OQ-007 | low | research memory | Which recurring facts should graduate from lab_notebook.md into decision_log.md? | next: review after next major phase`

## Resolved Questions

- `OQ-003 | resolved | Artifact compatibility is handled by the walk-forward pipeline. Fold 4 model always becomes production.`
- `OQ-006 | resolved | Current pre-GPU gate is sufficient. Anomaly audit stays separate.`
- `OQ-008 | resolved | Direction head decomposition failed (exp_100-103). 52% accuracy = random.`
- `OQ-009 | resolved | NOISE_MARGIN=0.01 is the live baseline filter.`
- `OQ-010 | resolved | Kronos standalone queue retired.`
- `OQ-011 | resolved | Temporal IDs deferred. Existing minutes_to_close feature is sufficient.`
- `OQ-012 | resolved | Holiday-adjacent schema breaks are harmless calendar edge cases, not a dataset blocker.`
- `OQ-013 | resolved | Side collapse was solved by contract feature normalization (exp_139). Model now trades both sides. Remaining call bias (86%) is appropriate given oracle is 57% calls in the trading window.`
- `OQ-014 | resolved | NOISE_MARGIN=0.01 is working. SOFT_TEMP=0.05 tested in exp_141 — peaked targets hurt because 79% of bars have 3+ near-oracle contracts.`
