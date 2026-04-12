# v2 Models

This directory holds the local checkpoint files used by the exact-chain workflow.

- `model.pt` — current promoted local checkpoint
- `model_best.pt` — best promoted checkpoint kept on disk
- `model_candidate.pt` — latest downloaded official-run checkpoint awaiting keep/revert

These files are local runtime artifacts. The GPU node trains from scratch each run.
