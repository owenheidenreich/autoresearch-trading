"""Full-pipeline smoke: run_final_train -> model_manage.keep -> status_report.

This is the end-to-end smoke the second-pass review asked for. It stitches the
full operator path together (with v2.train.train stubbed, since we don't have
a GPU in test) and asserts that the evidence ledger (`results.tsv`), the
deployed model manifest (`v2/models/model.manifest.json`), and the status
report all end up coherent:

  - run_final_train applied the CV's policy + env to the artifact
  - model_manage.keep() flipped `results.tsv` status for the source CV row
  - the sibling model.pt manifest carries the right fingerprints
  - status_report sees a single full-mode keep row and no schema errors
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestFullPipelineSmoke(unittest.TestCase):

    def test_cv_to_final_train_to_keep_to_status_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "v2" / "models").mkdir(parents=True)
            (tmp_path / "v2" / "artifacts").mkdir(parents=True)

            script = textwrap.dedent(r"""
                import os, sys, json
                from pathlib import Path

                # --- stub training so we can exercise plumbing without GPU ---
                import v2.train as _train_mod
                seen_env = {}
                def _fake_train(data_path, model_path, train_mask_override, val_mask_override):
                    seen_env["SOFT_TEMP"] = os.environ.get("SOFT_TEMP", "<unset>")
                    seen_env["SIDE_MODE"] = os.environ.get("SIDE_MODE", "<unset>")
                    import torch
                    model = _train_mod.TradingModel(d_model=64, depth=3, n_heads=4, dropout=0.1)
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_class": "TradingModel",
                        "epoch": 1, "val_loss": 0.3,
                        "hyperparams": {"lookback": 60, "d_model": 64, "depth": 3, "n_heads": 4, "dropout": 0.1},
                        "score_config_fingerprint": "stub",
                        "dataset_fingerprint": "stub",
                        "config_fingerprint": "stub",
                        "env_overrides": {},
                    }, model_path)
                    return None, {"val_loss": 0.3}
                _train_mod.train = _fake_train

                # --- stub data.pt load ---
                import torch
                _orig_load = torch.load
                def _fake_load(path, *a, **kw):
                    if str(path).endswith("/data.pt"):
                        all_dates = [f"2025-{m:02d}-{d:02d}" for m in range(1,13) for d in range(1,29)]
                        return {"dates": all_dates, "metadata": {"fingerprint": "stub_dset_fp"}}
                    return _orig_load(path, *a, **kw)
                torch.load = _fake_load

                # --- fabricate source CV_EVAL artifact + a "full"-mode results.tsv row ---
                from v2.core.cv_report import CVReport, FoldSlice, PooledSlice, StabilitySlice, header_line
                from v2.core.policy import DecisionPolicy
                from v2.core.artifact_kind import ArtifactKind
                from v2.core.metrics import score_config_fingerprint
                from v2.train import _get_config_fingerprint

                non_default = DecisionPolicy(side_mode="skew", alpha_side=0.35)
                env_overrides = {"SOFT_TEMP": "0.08", "SIDE_MODE": "skew"}
                fold = FoldSlice(
                    fold_idx=4, window_id="abcdef0123456789",
                    test_window_start="2025-11-01", test_window_end="2025-12-31",
                    train_window_start="2025-01-01", train_window_end="2025-10-31",
                    seed=42, score=0.6, gate_failure=None,
                    metrics={"total_trades": 120, "traded_days": 50}, baseline_scores={},
                    n_trades=120, n_test_days=60, n_traded_days=50, train_seconds=10.0,
                )
                cv = CVReport(
                    experiment_id="exp_smoke_cv", screening_mode="full",
                    folds=[fold],
                    pooled=PooledSlice(profit_factor=1.3, max_account_drawdown=0.1, win_rate=0.55,
                                       call_pct=0.5, put_pct=0.5, net_pnl_dollars=250.0,
                                       total_trades=120, total_eval_days=60, traded_days=50,
                                       positive_day_rate=0.6, daily_sortino=0.9),
                    stability=StabilitySlice(mean_fold_score=0.6, min_fold_score=0.6, max_fold_score=0.6,
                                             std_fold_score=0.0, per_fold_scores=[0.6],
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

                # Results.tsv: start with a revert row matching the source CV.
                tsv_path = Path("v2/results.tsv")
                tsv_path.write_text(
                    header_line() + "\n" +
                    "\t".join(["exp_smoke_cv", "full", "revert", "0.600000",
                               "1.3000", "0.1000", "120", "50", "false",
                               "[0.60]", "smoke"]) + "\n"
                )

                # --- run final_train (stubbed training) ---
                from v2.ops.run_final_train import run_final_train
                run_final_train(
                    source_exp_id="exp_smoke_cv",
                    data_path="v2/data.pt",
                    final_exp_id="exp_smoke_final",
                    internal_val_slice="tail_20d",
                    shadow_days=20,
                    base_seed=9001,
                )
                assert seen_env["SOFT_TEMP"] == "0.08", seen_env
                assert seen_env["SIDE_MODE"] == "skew", seen_env

                # --- run keep, stubbing observability ---
                from v2.core import observability
                observability.validate_artifact_presence = lambda _d: []
                from v2.ops import model_manage
                model_manage.keep()

                # --- verify sibling manifest ---
                sib = Path("v2/models/model.manifest.json")
                assert sib.exists(), "sibling model.manifest.json missing"
                m = json.loads(sib.read_text())
                assert m["artifact_kind"] == "final_train"
                assert m["policy_fingerprint"] == non_default.fingerprint()
                assert m["applied_env_overrides"] == env_overrides
                assert m["training_config_fingerprint"] == _get_config_fingerprint()

                # --- verify results.tsv flipped to keep ---
                body = tsv_path.read_text().splitlines()
                row = body[1].split("\t")
                header = body[0].split("\t")
                status_col = header.index("status")
                assert row[status_col] == "keep", f"expected keep, got {row[status_col]}"

                # --- verify status_report sees the keep row as latest official ---
                from v2.ops import status_report as sr
                sr.RESULTS_PATH = tsv_path
                res = sr.load_results_summary()
                assert res["schema_error"] is None, res["schema_error"]
                assert len(res["official_rows"]) == 1
                assert res["official_rows"][0]["status"] == "keep"
                assert res["latest_official"]["status"] == "keep"

                print("FULL_SMOKE_OK")
            """)
            runner = (
                f"import sys, os\n"
                f"os.chdir({str(tmp_path)!r})\n"
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
                + script
            )
            result = subprocess.run(
                [sys.executable, "-c", runner],
                capture_output=True, text=True, timeout=60,
            )
            self.assertIn(
                "FULL_SMOKE_OK", result.stdout,
                f"full-pipeline smoke failed.\nstdout={result.stdout}\nstderr={result.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
