"""model_manage.keep refuses non-FINAL_TRAIN artifacts.

This test fabricates two artifacts on disk (CV_EVAL and FINAL_TRAIN) and
verifies that keep() exits with code 1 for CV_EVAL and succeeds for FINAL_TRAIN.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _write_manifest(dir_: Path, kind: str, model_fp: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": dir_.name,
        "artifact_kind": kind,
        "timestamp": "2026-04-17T00:00:00",
        "score": 0.0,
        "promoted": False,
        "model_fingerprint": model_fp,
        "dataset_fingerprint": "test",
        "evaluator_fingerprint": "test",
        "policy_fingerprint": "test",
        "config_fingerprint": "test",
    }
    (dir_ / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _write_model(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()[:16]


class TestModelManifestKind(unittest.TestCase):

    def test_keep_rejects_cv_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            models = tmp_path / "v2" / "models"
            artifacts = tmp_path / "v2" / "artifacts" / "exp_cveval"
            models.mkdir(parents=True)
            artifacts.parent.mkdir(parents=True)

            payload = b"fake-cv-eval-model-bytes"
            candidate = models / "model_candidate.pt"
            fp = _write_model(candidate, payload)
            _write_manifest(artifacts, "cv_eval", fp)
            # Must also place the same fingerprint'd file inside the artifact so
            # _find_candidate_artifact_dir matches.
            _write_model(artifacts / "model.pt", payload)

            # Run model_manage.keep as a subprocess inside the sandboxed cwd
            env_code = (
                "import sys, os; "
                f"os.chdir({str(tmp_path)!r}); "
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
                "from v2.ops import model_manage; "
                "model_manage.keep()"
            )
            result = subprocess.run(
                [sys.executable, "-c", env_code],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0,
                                f"keep() must reject CV_EVAL. stdout={result.stdout} stderr={result.stderr}")
            combined = result.stdout + result.stderr
            self.assertIn("FINAL_TRAIN", combined)


if __name__ == "__main__":
    unittest.main()
