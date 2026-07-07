# Protocol101 Protected Holdout Artifact

- status: `pass`
- session_count: `30`
- protected_holdout_hash: `8eb62bb1195bc809ad523c09e6c96bf18581f995c428c0358b8bd810ea8f6281`
- paper_submit_allowed: `False`
- model_training_executed: `False`
- threshold_tuning_executed: `False`

## Sessions

- `2025-05-16`
- `2025-05-19`
- `2025-05-20`
- `2025-05-21`
- `2025-05-22`
- `2025-05-23`
- `2025-05-27`
- `2025-05-28`
- `2025-05-29`
- `2025-05-30`
- `2025-06-02`
- `2025-06-03`
- `2025-06-04`
- `2025-06-05`
- `2025-06-06`
- `2025-06-09`
- `2025-06-10`
- `2025-06-11`
- `2025-06-12`
- `2025-06-13`
- `2025-06-16`
- `2025-06-17`
- `2025-06-18`
- `2025-06-20`
- `2025-06-23`
- `2025-06-24`
- `2025-06-25`
- `2025-06-26`
- `2025-06-27`
- `2025-06-30`

## Loader Meaning

- `pass` means the governed loader can train/evaluate non-holdout splits.
- `pending_owner_declaration` blocks the governed loader by default.
- Protected sessions are rejected unless an explicit owner one-shot evaluation override is supplied, and even then only for the `test` role — never train/tune.
