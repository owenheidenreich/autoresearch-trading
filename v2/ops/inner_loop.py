"""Autoresearch experiment orchestrator.

Manages the experiment cycle: mutate v2/train.py -> run pre-GPU checks
(baselines.md) -> train on GPU -> replay on held-out days -> compute
promotion score -> keep or revert.

v2 changes from v1:
- Scores using replay P&L (profit factor primary), not prediction accuracy
- Pre-GPU gates enforce baseline comparison before any GPU spend
- Evaluator version fingerprinting prevents score drift

v1 origin: tools/inner_loop.py (cmd_experiment, state management) +
training/run_loop.py (validate_safety, validate_syntax, run_training,
detect_anomaly_flags, record_promotion_event)
"""
# TODO: ExperimentRunner class
# TODO: Pre-GPU checklist enforcement (baselines.md)
# TODO: Score evaluation via core.metrics
# TODO: Keep/revert logic with evaluator version checking
# TODO: State management (.inner_loop_state.json)
# TODO: Safety validation (syntax, architecture, anomaly flags)
