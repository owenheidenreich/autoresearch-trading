# Protocol101 Protected Holdout Artifact

- status: `pending_owner_declaration`
- session_count: `0`
- protected_holdout_hash: `89237ef5bc44a707967ea1afd693dceae00a631e98a1a68f76f7a5ea2dabaf07`
- paper_submit_allowed: `False`
- model_training_executed: `False`
- threshold_tuning_executed: `False`

## Sessions

- No protected holdout sessions have been owner-declared yet.

## Loader Meaning

- `pass` means the governed loader can train/evaluate non-holdout splits.
- `pending_owner_declaration` blocks the governed loader by default.
- Protected sessions are rejected unless an explicit owner one-shot evaluation override is supplied.
