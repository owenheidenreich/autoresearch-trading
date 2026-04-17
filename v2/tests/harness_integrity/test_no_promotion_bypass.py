"""walkforward.py must never write to v2/models/model.pt.

This test inspects the source file for the deleted bypass and the function
signature for the removed `model_path` parameter. If either regrows, CI fails.
"""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import v2.core.walkforward as wf

WF_SRC_PATH = Path(wf.__file__)


class TestNoPromotionBypass(unittest.TestCase):

    def test_source_does_not_write_to_models_dir(self):
        src = WF_SRC_PATH.read_text()
        self.assertNotIn(
            "v2/models/model.pt", src,
            "walkforward.py references v2/models/model.pt — promotion bypass has returned.",
        )
        self.assertNotIn(
            "v2/models/model_fold", src,
            "walkforward.py writes fold checkpoints under v2/models/ — must move to v2/artifacts/.",
        )

    def test_source_does_not_call_shutil_copy2(self):
        src = WF_SRC_PATH.read_text()
        self.assertNotIn(
            "shutil.copy2", src,
            "walkforward.py calls shutil.copy2 — fold checkpoints must not be promoted.",
        )

    def test_run_walkforward_signature_has_no_model_path(self):
        sig = inspect.signature(wf.run_walkforward)
        self.assertNotIn(
            "model_path", sig.parameters,
            "run_walkforward still accepts model_path — the promotion surface exists.",
        )
        self.assertIn(
            "screening_mode", sig.parameters,
            "run_walkforward must expose screening_mode as the replacement API.",
        )
        self.assertIn(
            "experiment_id", sig.parameters,
            "run_walkforward must require experiment_id to anchor fold checkpoint paths.",
        )


if __name__ == "__main__":
    unittest.main()
