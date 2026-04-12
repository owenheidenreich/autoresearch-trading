# Open Questions

This file contains the active unresolved questions for the current phase. Resolved items move to `decision_log.md`.

Format:

- `ID | priority | area | question | next: concrete next step`

## Active Questions

- `OQ-001 | high | data quality | The separate raw/sidecar anomaly track now exists and has been run; do its flagged dates justify any dataset action, or is the frozen dataset still the correct authority? | next: compare the flagged sessions against future worst-trace days before opening any dataset-version decision`
- `OQ-003 | high | artifact compatibility | What is the correct promotion policy when raw local checkpoints are incompatible with the restored architecture and no compatible promoted artifact bundle exists? | next: decide whether a compatible official artifact is a hard prerequisite before any local promoted checkpoint is trusted`
- `OQ-006 | medium | data quality | Is the current split between pre-GPU integrity checks and the separate anomaly audit sufficient, or should any anomaly checks graduate into the permanent gate? | next: decide whether repeated sidecar schema-break dates belong in the hard gate`
- `OQ-007 | medium | research memory | Which recurring facts should graduate out of lab_notebook.md into decision_log.md or program.md after each experiment block? | next: define the promotion rule for notebook content`
- `OQ-011 | medium | inference interface | Are explicit temporal IDs worth reintroducing after exp_108 failed, or should any future time-aware work derive from existing features unless a stronger hypothesis appears? | next: treat replay-threaded temporal IDs as deferred unless a new hypothesis clearly requires them`
- `OQ-012 | medium | sidecar integrity | The anomaly audit surfaced recurring short-session sidecar schema breaks on holiday-adjacent dates. Are these harmless calendar edge cases or a dataset-version blocker? | next: classify the 9 schema-break dates against exchange calendar expectations before escalating`
- `OQ-013 | high | side collapse | If the executable labels and soft targets are roughly side-neutral, what specific shortcut is causing the model to collapse to one side under both KL and ranking objectives? | next: compare side-bias audit outputs and promote traces on the restored exp_119 baseline before proposing exp_122+ follow-ups`
- `OQ-014 | medium | label noise | Does the `NOISE_MARGIN=0.01` filter capture the right ambiguity boundary, or should a stricter threshold like the unlogged exp_120 code probe be revisited with proper evidence? | next: only consider a new threshold sweep after the official exp_119 rebaseline is complete`

## Resolved Questions

- `OQ-008 | resolved | Explicit direction-first decomposition was tested in exp_100-exp_103 and failed. Revisit only if the decomposition changes the context representation rather than adding a separate direction head.`
- `OQ-009 | resolved | Margin-filtered KL selection is now part of the restored exp_119 working baseline (`NOISE_MARGIN=0.01`).`
- `OQ-010 | resolved | The Kronos standalone queue is retired. Keep the anomaly-audit ideas, but stop treating Kronos-style architecture changes as the main improvement path.`
- `OQ-004 | resolved | The tanh-bounded score head (exp_092) collapsed due to gradient saturation. Bounded approaches that saturate are not viable for the contract scoring task.`
- `OQ-005 | resolved | LOOKBACK=1 (exp_093) caused call-side collapse. The temporal context is load-bearing for direction balance, despite the logistic regression diagnostic suggesting otherwise.`

## Current Blockers

- `OQ-003`
- `OQ-013`
- `OQ-012`
