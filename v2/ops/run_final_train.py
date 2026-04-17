"""Final-train: the ONLY path that produces a promotable model artifact.

After walk-forward CV (v2.ops.run_experiment_wf) selects a config, this module
trains one model on the full pre-shadow span using an explicit internal
validation slice. The resulting artifact is tagged ArtifactKind.FINAL_TRAIN —
the only kind `v2.ops.model_manage.keep` accepts.

Usage:
    python -m v2.ops.run_final_train \
        --config-from exp_NNN \
        --internal-val-slice tail_20d

See /Users/gduby/.claude/plans/delightful-yawning-tiger.md Appendix F.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.core.artifact_kind import ArtifactKind
from v2.core.cv_report import CVReport
from v2.core.metrics import score_config_fingerprint
from v2.core.policy import DEFAULT_POLICY, DecisionPolicy
from v2.core.walkforward import dates_to_mask
from v2.ops.artifact import ARTIFACTS_DIR, save_artifact

# Internal val slice choices. Each returns (train_days, val_days, label) from
# a full pre-shadow span.
def _slice_tail_20d(pre_shadow_days: list[str]) -> tuple[list[str], list[str], str]:
    val = pre_shadow_days[-20:]
    train = pre_shadow_days[:-20]
    return train, val, "tail_20d"


def _slice_tail_40d(pre_shadow_days: list[str]) -> tuple[list[str], list[str], str]:
    val = pre_shadow_days[-40:]
    train = pre_shadow_days[:-40]
    return train, val, "tail_40d"


INTERNAL_VAL_SLICES = {
    "tail_20d": _slice_tail_20d,
    "tail_40d": _slice_tail_40d,
}


def _load_cv_report(source_exp_id: str) -> CVReport:
    artifact_dir = Path(ARTIFACTS_DIR) / source_exp_id
    cv_path = artifact_dir / "cv_report.json"
    if not cv_path.exists():
        raise SystemExit(
            f"ERROR: No cv_report.json at {cv_path}. "
            f"Did {source_exp_id} come from v2.ops.run_experiment_wf?"
        )
    return CVReport.from_json(str(cv_path))


def _load_source_policy(source_exp_id: str) -> DecisionPolicy:
    """Rehydrate the DecisionPolicy the CV actually used.

    The CV_EVAL artifact stores a policy.json snapshot. Loading it ensures the
    final-train + replay + keep path uses the exact same side_mode, alpha_side,
    stops, targets, and trailing params the CV scored under.
    """
    artifact_dir = Path(ARTIFACTS_DIR) / source_exp_id
    policy_path = artifact_dir / "policy.json"
    if not policy_path.exists():
        raise SystemExit(
            f"ERROR: No policy.json at {policy_path}. "
            f"CV_EVAL artifact is incomplete — refusing to guess with DEFAULT_POLICY."
        )
    loaded = DecisionPolicy.from_json(policy_path.read_text())
    # Sanity: fingerprint must match what the CV recorded.
    cv = _load_cv_report(source_exp_id)
    if loaded.fingerprint() != cv.policy_fingerprint:
        raise SystemExit(
            f"ERROR: policy.json fingerprint {loaded.fingerprint()!r} does not "
            f"match CVReport.policy_fingerprint {cv.policy_fingerprint!r}. "
            f"Artifact is corrupt — refusing to final-train."
        )
    return loaded


def _verify_config_unchanged(cv: CVReport) -> None:
    """Hard-fail if either the evaluator OR the training config drifted.

    Evaluator drift invalidates score comparability (the CV said score=X under
    one formula; we're about to produce a model under a different formula).
    Training-config drift invalidates model reproducibility (the CV said this
    architecture; we're about to produce a different one).
    """
    current_eval = score_config_fingerprint()
    if cv.evaluator_fingerprint and current_eval != cv.evaluator_fingerprint:
        raise SystemExit(
            f"ERROR: Evaluator config changed since CV was run.\n"
            f"  CV evaluator_fingerprint:          {cv.evaluator_fingerprint}\n"
            f"  current score_config_fingerprint:  {current_eval}\n"
            f"Refusing to final-train against a different evaluator."
        )

    from v2.train import _get_config_fingerprint
    current_cfg = _get_config_fingerprint()
    if cv.training_config_fingerprint and current_cfg != cv.training_config_fingerprint:
        raise SystemExit(
            f"ERROR: Training config changed since CV was run.\n"
            f"  CV training_config_fingerprint:  {cv.training_config_fingerprint}\n"
            f"  current RuntimeConfig fingerprint: {current_cfg}\n"
            f"Refusing to final-train under a different architecture/schema."
        )


def _apply_cv_env_overrides(overrides: dict[str, str]) -> list[str]:
    """Stamp the CV's training env onto os.environ. Returns clobbered key list."""
    clobbered = []
    for k, v in overrides.items():
        if k in os.environ and os.environ[k] != v:
            clobbered.append(k)
        os.environ[k] = str(v)
    if clobbered:
        print(f"  NOTE: overrode existing env values for: {clobbered}")
    return list(overrides.keys())


def run_final_train(
    *,
    source_exp_id: str,
    data_path: str = "v2/data.pt",
    final_exp_id: str | None = None,
    internal_val_slice: str = "tail_20d",
    shadow_days: int = 20,
    base_seed: int = 9001,
) -> str:
    """Train once on the full pre-shadow span. Output: FINAL_TRAIN artifact."""
    if internal_val_slice not in INTERNAL_VAL_SLICES:
        raise SystemExit(
            f"ERROR: Unknown internal_val_slice={internal_val_slice!r}. "
            f"Expected one of: {sorted(INTERNAL_VAL_SLICES)}"
        )
    if final_exp_id is None:
        final_exp_id = f"{source_exp_id}_final"

    cv = _load_cv_report(source_exp_id)
    _verify_config_unchanged(cv)
    source_policy = _load_source_policy(source_exp_id)

    if cv.screening_mode != "full":
        print(f"WARNING: Source CV was mode={cv.screening_mode}, not 'full'.")
        print(f"         Final-train against a non-full CV is unusual.")

    if cv.stability.any_fold_gate_failure:
        raise SystemExit(
            f"ERROR: Source CV {source_exp_id} has gate-failing folds. "
            f"Refusing to final-train a config that did not pass all CV gates."
        )

    print(f"\n{'='*60}")
    print(f"  FINAL TRAIN: {final_exp_id}  (from CV {source_exp_id})")
    print(f"  internal_val_slice: {internal_val_slice}")
    print(f"  base_seed: {base_seed}")
    print(f"  policy_fingerprint: {source_policy.fingerprint()}")
    print(f"  training_config_fingerprint: {cv.training_config_fingerprint}")
    print(f"{'='*60}")

    # Stamp the CV's training env onto os.environ *before* importing v2.train.
    # v2.train reads module-level constants from env at import, so downstream
    # control is only effective if this runs first.
    applied_env_keys = _apply_cv_env_overrides(cv.training_env_overrides)
    if applied_env_keys:
        print(f"Applied CV env overrides: {sorted(applied_env_keys)}")

    print(f"Loading dataset from {data_path}...")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    all_dates = data["dates"]
    unique_dates = sorted(set(all_dates))

    # Pre-shadow span = everything before the reserved shadow tail
    pre_shadow = unique_dates[:-shadow_days]
    slicer = INTERNAL_VAL_SLICES[internal_val_slice]
    train_days, val_days, slice_label = slicer(pre_shadow)

    print(f"  Pre-shadow days: {len(pre_shadow)}")
    print(f"  Train days:      {len(train_days)}  ({train_days[0]} -> {train_days[-1]})")
    print(f"  Internal val:    {len(val_days)}    ({val_days[0]} -> {val_days[-1]})")
    print(f"  Shadow reserved: {shadow_days} days (untouched)")

    train_set = set(train_days)
    val_set = set(val_days)
    train_mask = dates_to_mask(all_dates, train_set - val_set)
    val_mask = dates_to_mask(all_dates, val_set)

    artifact_dir = Path(ARTIFACTS_DIR) / final_exp_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(artifact_dir / "model.pt.tmp")

    os.environ["TRAIN_SEED"] = str(base_seed)

    from v2.train import train as train_model
    print(f"\n--- TRAINING ---")
    t_train = time.time()
    train_model(
        data_path=data_path,
        model_path=model_path,
        train_mask_override=train_mask,
        val_mask_override=val_mask,
    )
    train_seconds = time.time() - t_train
    print(f"Training completed in {train_seconds:.1f}s")

    dataset_fp = data.get("metadata", {}).get("fingerprint", "unknown")

    # Save as FINAL_TRAIN artifact using the rehydrated source-CV policy.
    # NEVER stamp DEFAULT_POLICY — the CV evaluated under a specific policy and
    # the promoted artifact must replay under that same policy.
    saved = save_artifact(
        experiment_id=final_exp_id,
        model_path=model_path,
        score=cv.stability.mean_fold_score,
        policy=source_policy,
        dataset_fingerprint=dataset_fp,
        extra_metadata={
            "source_experiment": source_exp_id,
            "source_cv_stability_score": cv.stability.mean_fold_score,
            "source_cv_pooled_pf": cv.pooled.profit_factor,
            "source_cv_mode": cv.screening_mode,
            "internal_val_slice": slice_label,
            "trained_on_span": f"{train_days[0]}..{train_days[-1]}",
            "shadow_days_reserved": shadow_days,
            "training_config_fingerprint": cv.training_config_fingerprint,
            "applied_env_overrides": cv.training_env_overrides,
            "training_seconds": train_seconds,
        },
        kind=ArtifactKind.FINAL_TRAIN,
    )
    print(f"FINAL_TRAIN artifact saved: {saved}")

    # Sync the artifact's model into the staging slot for model_manage keep
    candidate_path = Path("v2/models/model_candidate.pt")
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.path.join(saved, "model.pt"), candidate_path)
    print(f"Candidate staged: {candidate_path}")
    print("")
    print("Next step: python3 -m v2.ops.model_manage keep")

    # Remove the tmp copy
    if os.path.exists(model_path):
        os.remove(model_path)

    return saved


def main():
    p = argparse.ArgumentParser(description="Train the single deployable model after CV.")
    p.add_argument("--config-from", dest="source_exp_id", required=True,
                   help="CV experiment id whose config this final-train reproduces.")
    p.add_argument("--data", default="v2/data.pt")
    p.add_argument("--id", dest="final_exp_id", default=None)
    p.add_argument("--internal-val-slice", default="tail_20d",
                   choices=sorted(INTERNAL_VAL_SLICES))
    p.add_argument("--shadow-days", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=9001)
    args = p.parse_args()

    saved = run_final_train(
        source_exp_id=args.source_exp_id,
        data_path=args.data,
        final_exp_id=args.final_exp_id,
        internal_val_slice=args.internal_val_slice,
        shadow_days=args.shadow_days,
        base_seed=args.base_seed,
    )
    print(f"\nRESULTS_JSON:{json.dumps({'artifact_dir': saved, 'artifact_kind': ArtifactKind.FINAL_TRAIN.value})}")


if __name__ == "__main__":
    main()
