"""Regression tests for the operator-facing paths the harness repair touched.

The first-pass repair left these surfaces half in the old world:
  - status_report.py crashed on legacy rows by calling float("legacy")
  - cmd_download clobbered model_candidate.pt with the deployed model.pt
  - model_manage.keep/revert never flipped the results.tsv `status` column
  - monitor.py read the missing `score` column and silently scored everything
    as -999
  - the status dashboard printed "(no experiments yet)" even on success

Each of those is pinned here so regression is loud.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _seed_v2_tree(tmp_path: Path) -> Path:
    (tmp_path / "v2" / "models").mkdir(parents=True)
    (tmp_path / "v2" / "artifacts").mkdir(parents=True)
    return tmp_path


def _write_migrated_tsv(path: Path) -> None:
    """Write a tsv that mixes a legacy row with a full-mode keep and revert."""
    from v2.core.cv_report import header_line
    body = [
        header_line(),
        "\t".join(["exp_151", "legacy", "revert", "-0.200000", "0.7137",
                   "0.0000", "0", "0", "true", "[-0.200]", "legacy row"]),
        "\t".join(["exp_200", "full", "revert", "0.625000", "1.2345",
                   "0.1200", "500", "240", "false", "[0.62,0.63,0.65,0.60,0.63]",
                   "full mode folds"]),
        "\t".join(["exp_201", "full", "keep", "0.700000", "1.4000",
                   "0.1000", "550", "260", "false", "[0.70,0.69,0.71,0.70,0.70]",
                   "full mode folds"]),
        "",
    ]
    path.write_text("\n".join(body))


class TestStatusReport(unittest.TestCase):
    """status_report must not crash on the migrated TSV and must bucket rows correctly."""

    def test_load_results_summary_handles_legacy_row_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _seed_v2_tree(tmp_path)
            tsv = tmp_path / "v2" / "results.tsv"
            _write_migrated_tsv(tsv)

            # Run status_report.load_results_summary as a subprocess with cwd set
            # so it picks up the fake results.tsv without cross-polluting the
            # real project ledger.
            script = (
                "import sys; "
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
                "from v2.ops import status_report as s; "
                f"s.RESULTS_PATH = __import__('pathlib').Path({str(tsv)!r}); "
                "import json; print('JSON=' + json.dumps(s.load_results_summary(), default=str))"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, cwd=tmp_path,
            )
            self.assertEqual(result.returncode, 0,
                             f"status_report crashed. stdout={result.stdout}\nstderr={result.stderr}")
            payload = [ln for ln in result.stdout.splitlines() if ln.startswith("JSON=")][0][5:]
            data = json.loads(payload)
            self.assertIsNone(data.get("schema_error"))
            # legacy is NOT counted as official — only full-mode rows are.
            self.assertEqual(len(data["official_rows"]), 2)
            self.assertEqual(len(data["non_official_rows"]), 1)
            self.assertEqual(data["non_official_rows"][0]["experiment"], "exp_151")
            # best_official should be the higher stability_score
            self.assertEqual(data["best_official"]["experiment"], "exp_201")
            self.assertAlmostEqual(data["best_official"]["stability_score"], 0.7)

    def test_load_results_summary_reports_schema_error_on_legacy_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _seed_v2_tree(tmp_path)
            tsv = tmp_path / "v2" / "results.tsv"
            tsv.write_text(
                "experiment\tscore\tstatus\tdescription\n"
                "exp_old\t0.5\tkeep\tpre-harness-repair row\n"
            )
            script = (
                "import sys; "
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
                "from v2.ops import status_report as s; "
                f"s.RESULTS_PATH = __import__('pathlib').Path({str(tsv)!r}); "
                "import json; print('JSON=' + json.dumps(s.load_results_summary(), default=str))"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, cwd=tmp_path,
            )
            self.assertEqual(result.returncode, 0)
            payload = [ln for ln in result.stdout.splitlines() if ln.startswith("JSON=")][0][5:]
            data = json.loads(payload)
            self.assertIsNotNone(data.get("schema_error"))
            self.assertIn("CVReport schema", data["schema_error"])


class TestStatusTsvRenderWithLegacyRows(unittest.TestCase):
    """status_tsv.render must not crash on the migrated TSV that mixes legacy + full."""

    def test_render_handles_mixed_legacy_and_full(self):
        from v2.ops.status_tsv import render
        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "results.tsv"
            _write_migrated_tsv(tsv)
            out = render(tsv)
            self.assertIn("exp_151", out)
            self.assertIn("legacy", out)
            self.assertIn("exp_200", out)
            self.assertIn("exp_201", out)
            self.assertIn("keep", out)
            self.assertIn("0.700", out)


class TestModelManageFlipsLedger(unittest.TestCase):
    """model_manage.keep/revert must flip results.tsv status by source_experiment."""

    def test_keep_flips_status_to_keep(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _seed_v2_tree(tmp_path)
            tsv = tmp_path / "v2" / "results.tsv"
            _write_migrated_tsv(tsv)

            # Fabricate a FINAL_TRAIN artifact whose source_experiment = exp_200.
            art = tmp_path / "v2" / "artifacts" / "exp_200_final"
            art.mkdir(parents=True)
            payload = b"final_train_bytes"
            cand = tmp_path / "v2" / "models" / "model_candidate.pt"
            cand.write_bytes(payload)
            fp = hashlib.sha256(payload).hexdigest()[:16]
            (art / "model.pt").write_bytes(payload)

            manifest = {
                "experiment_id": "exp_200_final",
                "artifact_kind": "final_train",
                "timestamp": "2026-04-17T00:00:00",
                "score": 0.625,
                "promoted": False,
                "model_fingerprint": fp,
                "dataset_fingerprint": "d", "evaluator_fingerprint": "e",
                "policy_fingerprint": "p",
                "extra": {
                    "source_experiment": "exp_200",
                    "training_config_fingerprint": "cfg",
                    "applied_env_overrides": {},
                    "trained_on_span": "2025-01-01..2025-12-01",
                    "internal_val_slice": "tail_20d",
                },
            }
            (art / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

            # Run model_manage.keep in a subprocess with cwd=tmp_path so it
            # operates on the fake ledger/artifact.
            env_code = (
                "import sys, os; "
                f"os.chdir({str(tmp_path)!r}); "
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
                "from v2.core import observability; "
                "observability.validate_artifact_presence = lambda _d: []; "
                "from v2.ops import model_manage; "
                "model_manage.keep()"
            )
            result = subprocess.run(
                [sys.executable, "-c", env_code],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0,
                             f"keep() failed. stdout={result.stdout}\nstderr={result.stderr}")
            # Ledger row for exp_200 must now be 'keep'. exp_201 already was;
            # exp_151 stays 'revert' (legacy is unrelated).
            updated = tsv.read_text().splitlines()
            header = updated[0].split("\t")
            status_col = header.index("status")
            exp_col = header.index("experiment")
            rows = {r.split("\t")[exp_col]: r.split("\t")[status_col] for r in updated[1:] if r}
            self.assertEqual(rows["exp_200"], "keep")
            self.assertEqual(rows["exp_201"], "keep")
            self.assertEqual(rows["exp_151"], "revert")

    def test_revert_flips_status_to_revert(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _seed_v2_tree(tmp_path)
            tsv = tmp_path / "v2" / "results.tsv"
            _write_migrated_tsv(tsv)

            # exp_201 is currently 'keep'; fabricate an artifact pointing at
            # it, then call revert and confirm the status flips back.
            art = tmp_path / "v2" / "artifacts" / "exp_201_final"
            art.mkdir(parents=True)
            payload = b"whatever"
            cand = tmp_path / "v2" / "models" / "model_candidate.pt"
            cand.write_bytes(payload)
            fp = hashlib.sha256(payload).hexdigest()[:16]
            (art / "model.pt").write_bytes(payload)
            manifest = {
                "experiment_id": "exp_201_final",
                "artifact_kind": "final_train",
                "timestamp": "2026-04-17T00:00:00",
                "score": 0.7, "promoted": True,
                "model_fingerprint": fp,
                "extra": {"source_experiment": "exp_201"},
            }
            (art / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

            env_code = (
                "import sys, os; "
                f"os.chdir({str(tmp_path)!r}); "
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
                "from v2.ops import model_manage; "
                "model_manage.revert()"
            )
            result = subprocess.run(
                [sys.executable, "-c", env_code],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            updated = tsv.read_text().splitlines()
            header = updated[0].split("\t")
            status_col = header.index("status")
            exp_col = header.index("experiment")
            rows = {r.split("\t")[exp_col]: r.split("\t")[status_col] for r in updated[1:] if r}
            self.assertEqual(rows["exp_201"], "revert")


class TestCmdDownloadCandidatePath(unittest.TestCase):
    """cmd_download must not stage /root/v2/models/model.pt as the candidate.

    The command uses SSH to reach the remote, so we can only assert on the
    shape of the script itself — specifically that it references
    `/root/v2/models/model_candidate.pt`, never substitutes the deployed
    `model.pt`, and includes the defensive guard.
    """

    def test_deploy_sh_uses_model_candidate_path(self):
        src = (PROJECT_ROOT / "v2" / "ops" / "deploy.sh").read_text()
        # Must fetch the candidate explicitly.
        self.assertIn("/root/v2/models/model_candidate.pt", src)
        # In cmd_download, the candidate download must be guarded by a
        # remote-side existence test. The specific substring we look for lives
        # in the new logic.
        self.assertIn(
            "test -f /root/v2/models/model_candidate.pt",
            src,
            "cmd_download must guard on remote /root/v2/models/model_candidate.pt existence",
        )
        # The legacy anti-pattern — substituting the deployed model.pt into
        # the local candidate slot — must be gone from cmd_download.
        anti_pattern = 'root@$SSH_HOST:/root/v2/models/model.pt" "$PROJECT_ROOT/v2/models/model_candidate.pt"'
        self.assertNotIn(
            anti_pattern, src,
            "cmd_download must not substitute the deployed model.pt as the candidate",
        )


class TestStopKillsBothRunTypes(unittest.TestCase):
    """cmd_stop must kill run_experiment_wf AND run_final_train."""

    def test_deploy_sh_pkill_covers_both_run_types(self):
        src = (PROJECT_ROOT / "v2" / "ops" / "deploy.sh").read_text()
        self.assertIn("run_experiment_wf|run_final_train", src,
                      "cmd_stop must kill both run types")


class TestStatusHeredocHasNoElseAfterTry(unittest.TestCase):
    """The `else:` that printed '(no experiments yet)' even on success is gone."""

    def test_no_stale_else_clause_after_render(self):
        src = (PROJECT_ROOT / "v2" / "ops" / "deploy.sh").read_text()
        # Extract the heredoc region.
        begin = src.find("PYEOF")
        end = src.find("PYEOF", begin + 1)
        self.assertTrue(begin > 0 and end > begin, "could not locate status heredoc")
        heredoc = src[begin:end]
        # The phrase "(no experiments yet)" must be gone from the status heredoc;
        # status_tsv.render emits its own "(no experiments yet)" message when
        # appropriate, so there's no reason for the heredoc to print it too.
        self.assertNotIn(
            "(no experiments yet)", heredoc,
            "cmd_status heredoc must not print '(no experiments yet)' — delegate to status_tsv.render",
        )


class TestMonitorReadsStabilityScore(unittest.TestCase):
    """monitor.py must read stability_score (post-repair) not blindly `score`."""

    def test_monitor_reads_stability_score(self):
        src = (PROJECT_ROOT / "v2" / "ops" / "monitor.py").read_text()
        self.assertIn(
            'row.get("stability_score")',
            src,
            "monitor.py must read the CVReport `stability_score` column",
        )


if __name__ == "__main__":
    unittest.main()
