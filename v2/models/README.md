# v2 Models

This directory holds the current operational checkpoint. The GPU node trains from scratch each run.

## Convention

`model.pt` is the **current operational default** — the checkpoint that replay, plotting, and health checks use. It is not necessarily the milestone-best model. It may be:

- **validated**: passed all promotion gates under the current evaluator
- **provisional**: promoted under a prior evaluator, not yet re-evaluated
- **stale**: trained on an incompatible dataset or config

Run `python -m v2.ops.health model` to check which state it's in.

Run `python -m v2.ops.health model` to check status.

## Other checkpoints

After a GPU training run, `model_candidate.pt` appears here pending keep/revert. Once promoted, it becomes `model.pt`.
