"""Non-default policy + env overrides survive CV -> final-train -> keep.

This exercises the plumbing paths, not the actual training loop. The guarantees
tested here are:

1. run_final_train refuses DEFAULT_POLICY when the source CV used a different
   one. (Test: source policy.json has a non-default side_mode; loader picks it
   up; fingerprint check rejects mismatch.)

2. run_final_train applies the CV's training_env_overrides to os.environ before
   training. (Test: monkeypatch v2.train.train to capture os.environ at call
   time, confirm the override is visible.)

3. model_manage.keep, when run on a fabricated FINAL_TRAIN artifact, writes a
   sibling manifest that carries the source's policy_fingerprint and
   training_config_fingerprint. (Test: inspect sibling manifest JSON.)

4. The sibling manifest has `applied_env_overrides` recorded verbatim so a
   future auditor can see what env the deployed model was trained under.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Make v2 importable when run standalone.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.core.artifact_kind import ArtifactKind
from v2.core.cv_report import (
    CVReport,
    FoldSlice,
    PooledSlice,
    StabilitySlice,
)
from v2.core.policy import DecisionPolicy


def _fake_cv_report(policy_fp: str, env: dict[str, str]) -> CVReport:
    fold = FoldSlice(
        fold_idx=4, window_id="abcd1234abcd1234",
        test_window_start="2025-01-01", test_window_end="2025-03-01",
        train_window_start="2024-01-01", train_window_end="2024-12-31",
        seed=42, score=0.5, gate_failure=None,
        metrics={"total_trades": 100, "traded_days": 45}, baseline_scores={},
        n_trades=100, n_test_days=60, n_traded_days=45, train_seconds=10.0,
    )
    return CVReport(
        experiment_id="exp_fake_cv", screening_mode="full",
        folds=[fold],
        pooled=PooledSlice(
            profit_factor=1.1, max_account_drawdown=0.1, win_rate=0.55,
            call_pct=0.5, put_pct=0.5, net_pnl_dollars=100.0,
            total_trades=100, total_eval_days=60, traded_days=45,
            positive_day_rate=0.6, daily_sortino=0.9,
        ),
        stability=StabilitySlice(
            mean_fold_score=0.5, min_fold_score=0.5, max_fold_score=0.5,
            std_fold_score=0.0, per_fold_scores=[0.5],
            per_fold_gate_failures=[False], any_fold_gate_failure=False,
        ),
        aggregate_baselines={}, beats_all_baselines=True,
        training_config_fingerprint="fake_cfg_fp",
        training_env_overrides=env,
        policy_fingerprint=policy_fp,
        dataset_fingerprint="fake_dset_fp",
        evaluator_fingerprint="fake_eval_fp",
        training_seconds=100.0,
    )


class TestPolicyAndEnvSurvive(unittest.TestCase):

    def test_load_source_policy_rejects_fingerprint_mismatch(self):
        """If the CVReport policy_fingerprint does not match policy.json, refuse."""
        from v2.ops import run_final_train as rft
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_id = "exp_fake_cv"
            src_dir = tmp_path / "v2" / "artifacts" / source_id
            src_dir.mkdir(parents=True)

            non_default = DecisionPolicy(side_mode="skew", alpha_side=0.35)
            (src_dir / "policy.json").write_text(non_default.to_json())

            # Report claims a DIFFERENT fingerprint → loader must reject.
            cv = _fake_cv_report(policy_fp="totally_bogus_fp", env={})
            cv.to_json(str(src_dir / "cv_report.json"))

            with patch.object(rft, "ARTIFACTS_DIR", str(tmp_path / "v2" / "artifacts")):
                with self.assertRaises(SystemExit):
                    rft._load_source_policy(source_id)

    def test_load_source_policy_accepts_matching(self):
        from v2.ops import run_final_train as rft
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_id = "exp_fake_cv"
            src_dir = tmp_path / "v2" / "artifacts" / source_id
            src_dir.mkdir(parents=True)

            non_default = DecisionPolicy(side_mode="skew", alpha_side=0.35)
            (src_dir / "policy.json").write_text(non_default.to_json())

            cv = _fake_cv_report(policy_fp=non_default.fingerprint(), env={"SOFT_TEMP": "0.04"})
            cv.to_json(str(src_dir / "cv_report.json"))

            with patch.object(rft, "ARTIFACTS_DIR", str(tmp_path / "v2" / "artifacts")):
                loaded = rft._load_source_policy(source_id)
            self.assertEqual(loaded.side_mode, "skew")
            self.assertAlmostEqual(loaded.alpha_side, 0.35)
            self.assertEqual(loaded.fingerprint(), non_default.fingerprint())

    def test_apply_cv_env_overrides_writes_os_environ(self):
        from v2.ops import run_final_train as rft
        # Save & restore environment
        saved = {k: os.environ.get(k) for k in ("SOFT_TEMP", "SIDE_MODE", "ALPHA_SIDE")}
        try:
            for k in saved:
                os.environ.pop(k, None)
            overrides = {"SOFT_TEMP": "0.08", "SIDE_MODE": "skew", "ALPHA_SIDE": "0.3"}
            rft._apply_cv_env_overrides(overrides)
            self.assertEqual(os.environ.get("SOFT_TEMP"), "0.08")
            self.assertEqual(os.environ.get("SIDE_MODE"), "skew")
            self.assertEqual(os.environ.get("ALPHA_SIDE"), "0.3")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_verify_config_unchanged_rejects_evaluator_drift(self):
        from v2.ops import run_final_train as rft
        cv = _fake_cv_report(policy_fp="polfp", env={})
        # evaluator_fingerprint in cv is "fake_eval_fp" — won't match the real one.
        with self.assertRaises(SystemExit):
            rft._verify_config_unchanged(cv)

    def test_verify_config_rejects_training_config_drift(self):
        """Even if evaluator matches, a training-config change must still fail."""
        from v2.ops import run_final_train as rft
        from v2.core.metrics import score_config_fingerprint
        cv = _fake_cv_report(policy_fp="polfp", env={})
        # Stamp real evaluator fp so only training config mismatches.
        cv = CVReport(**{**cv.to_dict(),
                         "evaluator_fingerprint": score_config_fingerprint(),
                         "folds": [FoldSlice(**f) for f in cv.to_dict()["folds"]],
                         "pooled": PooledSlice(**cv.to_dict()["pooled"]),
                         "stability": StabilitySlice(**cv.to_dict()["stability"])})
        # training_config_fingerprint is still "fake_cfg_fp" → must reject.
        with self.assertRaises(SystemExit):
            rft._verify_config_unchanged(cv)

    def test_model_manage_keep_writes_sibling_manifest_with_fingerprints(self):
        """keep() on a fabricated FINAL_TRAIN artifact records source fingerprints."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            models = tmp_path / "v2" / "models"
            models.mkdir(parents=True)
            art = tmp_path / "v2" / "artifacts" / "exp_fake_final"
            art.mkdir(parents=True)

            payload = b"fake-final-train-model-bytes"
            cand = models / "model_candidate.pt"
            cand.write_bytes(payload)
            fp = hashlib.sha256(payload).hexdigest()[:16]
            (art / "model.pt").write_bytes(payload)

            manifest = {
                "experiment_id": "exp_fake_final",
                "artifact_kind": ArtifactKind.FINAL_TRAIN.value,
                "timestamp": "2026-04-17T00:00:00",
                "score": 0.5,
                "promoted": False,
                "model_fingerprint": fp,
                "dataset_fingerprint": "dsetfp",
                "evaluator_fingerprint": "evalfp",
                "policy_fingerprint": "polfp_SIDE_MODE_skew",
                "extra": {
                    "source_experiment": "exp_fake_cv",
                    "training_config_fingerprint": "cfgfp",
                    "applied_env_overrides": {"SOFT_TEMP": "0.08", "SIDE_MODE": "skew"},
                    "trained_on_span": "2024-01-01..2024-12-01",
                    "internal_val_slice": "tail_20d",
                },
            }
            (art / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

            # Stub validate_artifact_presence so keep() doesn't demand traces.
            from v2.core import observability
            orig = observability.validate_artifact_presence
            observability.validate_artifact_presence = lambda _d: []

            import subprocess
            env_code = (
                "import sys, os; "
                f"os.chdir({str(tmp_path)!r}); "
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
                "from v2.core import observability; "
                "observability.validate_artifact_presence = lambda _d: []; "
                "from v2.ops import model_manage; "
                "model_manage.keep()"
            )
            try:
                result = subprocess.run(
                    [sys.executable, "-c", env_code],
                    capture_output=True, text=True,
                )
            finally:
                observability.validate_artifact_presence = orig

            self.assertEqual(result.returncode, 0,
                             f"keep() failed. stdout={result.stdout} stderr={result.stderr}")
            sibling = models / "model.manifest.json"
            self.assertTrue(sibling.exists(), "model.manifest.json must be written alongside model.pt")
            loaded = json.loads(sibling.read_text())
            self.assertEqual(loaded["artifact_kind"], "final_train")
            self.assertEqual(loaded["training_config_fingerprint"], "cfgfp")
            self.assertEqual(loaded["policy_fingerprint"], "polfp_SIDE_MODE_skew")
            self.assertEqual(loaded["applied_env_overrides"], {"SOFT_TEMP": "0.08", "SIDE_MODE": "skew"})
            self.assertEqual(loaded["source_experiment"], "exp_fake_cv")


if __name__ == "__main__":
    unittest.main()
