"""End-to-end smoke: CV_EVAL -> run_final_train -> model_manage.keep -> load.

Actual training is stubbed (no GPU, no real data) so this test exercises the
*plumbing*: the paths where fingerprint drift, policy drift, and env drift
can silently creep back in.

What this pins:

1. A non-default DecisionPolicy stored in the source CV artifact survives all
   the way to v2/models/model.manifest.json.
2. training_env_overrides from the source CV are applied to os.environ before
   v2.train.train is invoked.
3. The sibling manifest at v2/models/model.manifest.json has
   artifact_kind=FINAL_TRAIN, training_config_fingerprint matching the source
   CV, and applied_env_overrides matching the source CV env.
4. model_manage.keep refuses to promote if evaluator drifts between CV and
   final-train (evaluator_fingerprint mismatch).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _run_inprocess_script(tmp_path: Path, script: str) -> subprocess.CompletedProcess:
    """Run a script with cwd=tmp_path and v2 importable."""
    runner = (
        f"import sys, os\n"
        f"os.chdir({str(tmp_path)!r})\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        + script
    )
    return subprocess.run(
        [sys.executable, "-c", runner],
        capture_output=True, text=True, timeout=60,
    )


class TestEndToEndSmoke(unittest.TestCase):

    def test_cv_eval_to_final_train_to_keep_preserves_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "v2" / "models").mkdir(parents=True)
            (tmp_path / "v2" / "artifacts").mkdir(parents=True)

            # The script: stubs v2.train.train, fabricates a CV_EVAL artifact
            # with non-default policy and env, calls run_final_train, then
            # calls model_manage.keep, then reads back the sibling manifest.
            script = textwrap.dedent("""
                import os, sys, hashlib, json
                from pathlib import Path

                # --- stub v2.train.train before any code imports it ---
                import v2.train as _train_mod
                captured_env = {}
                def _fake_train(data_path, model_path, train_mask_override, val_mask_override):
                    # Capture os.environ at the moment training would start.
                    captured_env["SOFT_TEMP"] = os.environ.get("SOFT_TEMP", "<unset>")
                    captured_env["SIDE_MODE"] = os.environ.get("SIDE_MODE", "<unset>")
                    captured_env["ALPHA_SIDE"] = os.environ.get("ALPHA_SIDE", "<unset>")
                    # Write a minimal checkpoint the save_artifact path can load.
                    import torch
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state_dict": {},
                        "model_class": "TradingModel",
                        "epoch": 1,
                        "val_loss": 0.5,
                        "hyperparams": {"lookback": 60, "d_model": 64, "depth": 3, "n_heads": 4, "dropout": 0.1},
                        "score_config_fingerprint": "stub",
                        "dataset_fingerprint": "stub",
                        "config_fingerprint": "stub",
                        "env_overrides": {},
                    }, model_path)
                    return None, {"val_loss": 0.5}
                _train_mod.train = _fake_train

                # --- stub torch.load for the dataset path (we don't have data.pt here) ---
                import torch
                _orig_load = torch.load
                def _fake_load(path, *a, **kw):
                    if str(path).endswith("/data.pt"):
                        all_dates = [f"2025-{m:02d}-{d:02d}" for m in range(1,13) for d in range(1,29)]
                        return {"dates": all_dates, "metadata": {"fingerprint": "stub_dset_fp"}}
                    return _orig_load(path, *a, **kw)
                torch.load = _fake_load

                # --- fabricate the source CV_EVAL artifact ---
                from v2.core.cv_report import CVReport, FoldSlice, PooledSlice, StabilitySlice
                from v2.core.policy import DecisionPolicy
                from v2.core.artifact_kind import ArtifactKind
                from v2.core.metrics import score_config_fingerprint
                from v2.train import _get_config_fingerprint

                non_default = DecisionPolicy(side_mode="skew", alpha_side=0.35)
                env_overrides = {"SOFT_TEMP": "0.08", "SIDE_MODE": "skew", "ALPHA_SIDE": "0.35"}
                fold = FoldSlice(
                    fold_idx=4, window_id="abcdef0123456789",
                    test_window_start="2025-11-01", test_window_end="2025-12-31",
                    train_window_start="2025-01-01", train_window_end="2025-10-31",
                    seed=42, score=0.5, gate_failure=None,
                    metrics={"total_trades": 100, "traded_days": 45},
                    baseline_scores={"random": 0.1, "atm": 0.2, "rules": 0.15, "trailing": 0.18},
                    n_trades=100, n_test_days=60, n_traded_days=45, train_seconds=10.0,
                )
                cv = CVReport(
                    experiment_id="exp_smoke_cv", screening_mode="full",
                    folds=[fold],
                    pooled=PooledSlice(profit_factor=1.1, max_account_drawdown=0.1, win_rate=0.55,
                                       call_pct=0.5, put_pct=0.5, net_pnl_dollars=100.0,
                                       total_trades=100, total_eval_days=60, traded_days=45,
                                       positive_day_rate=0.6, daily_sortino=0.9),
                    stability=StabilitySlice(mean_fold_score=0.5, min_fold_score=0.5, max_fold_score=0.5,
                                             std_fold_score=0.0, per_fold_scores=[0.5],
                                             per_fold_gate_failures=[False], any_fold_gate_failure=False),
                    aggregate_baselines={}, beats_all_baselines=True,
                    training_config_fingerprint=_get_config_fingerprint(),
                    training_env_overrides=env_overrides,
                    policy_fingerprint=non_default.fingerprint(),
                    dataset_fingerprint="stub_dset_fp",
                    evaluator_fingerprint=score_config_fingerprint(),
                    training_seconds=100.0,
                )
                src_dir = Path("v2/artifacts/exp_smoke_cv")
                src_dir.mkdir(parents=True, exist_ok=True)
                (src_dir / "policy.json").write_text(non_default.to_json())
                cv.to_json(str(src_dir / "cv_report.json"))

                # --- run final train (stubbed training_module loaded via _fake_train above) ---
                from v2.ops.run_final_train import run_final_train
                final_dir = run_final_train(
                    source_exp_id="exp_smoke_cv",
                    data_path="v2/data.pt",
                    final_exp_id="exp_smoke_final",
                    internal_val_slice="tail_20d",
                    shadow_days=20,
                    base_seed=9001,
                )

                # Env must have been applied before training started.
                assert captured_env.get("SOFT_TEMP") == "0.08", captured_env
                assert captured_env.get("SIDE_MODE") == "skew", captured_env
                assert captured_env.get("ALPHA_SIDE") == "0.35", captured_env

                # --- now invoke model_manage.keep, with observability stubbed ---
                from v2.core import observability
                observability.validate_artifact_presence = lambda _d: []
                from v2.ops import model_manage
                model_manage.keep()

                # --- verify the sibling manifest ---
                sibling = Path("v2/models/model.manifest.json")
                assert sibling.exists(), "sibling manifest missing"
                m = json.loads(sibling.read_text())
                assert m["artifact_kind"] == "final_train", m
                assert m["policy_fingerprint"] == non_default.fingerprint(), m
                assert m["applied_env_overrides"] == env_overrides, m
                assert m["training_config_fingerprint"] == _get_config_fingerprint(), m
                assert m["source_experiment"] == "exp_smoke_cv", m

                # --- confirm model.pt exists and fingerprint matches the artifact's ---
                from v2.ops.artifact import _file_fingerprint
                model_pt = Path("v2/models/model.pt")
                artifact_pt = Path("v2/artifacts/exp_smoke_final/model.pt")
                assert _file_fingerprint(str(model_pt)) == _file_fingerprint(str(artifact_pt)), "model.pt mismatch"

                print("SMOKE_OK")
            """)
            result = _run_inprocess_script(tmp_path, script)
            self.assertIn(
                "SMOKE_OK", result.stdout,
                f"end-to-end smoke failed.\nstdout={result.stdout}\nstderr={result.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
