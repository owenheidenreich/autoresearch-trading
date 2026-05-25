import ast
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_OPS = REPO_ROOT / "research_ops"
SCRIPTS = RESEARCH_OPS / "scripts"


def test_research_ops_scripts_do_not_import_trading_or_external_systems():
    forbidden_prefixes = {
        "v4",
        "ib_insync",
        "polygon",
        "databento",
        "theta",
        "torch",
        "pandas",
        "numpy",
    }
    for path in sorted(SCRIPTS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        bad = sorted(name for name in imports if name.split(".")[0] in forbidden_prefixes)
        assert bad == [], f"{path} imports protected modules: {bad}"


def test_assumption_registry_has_expected_header():
    with (RESEARCH_OPS / "ASSUMPTION_REGISTRY.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == [
        "assumption_id",
        "title",
        "status",
        "importance",
        "fragility",
        "falsification_risk",
        "category",
        "current_evidence",
        "falsification_test",
        "confidence_increases_if",
        "confidence_destroyed_if",
        "next_artifact",
        "owner",
        "last_updated",
    ]


def test_new_iteration_validate_and_summarize_roundtrip(tmp_path):
    sandbox = tmp_path / "repo"
    shutil.copytree(RESEARCH_OPS, sandbox / "research_ops")

    create = subprocess.run(
        [
            sys.executable,
            str(sandbox / "research_ops" / "scripts" / "new_iteration.py"),
            "execution-realism",
            "--date",
            "2026-05-24",
            "--root",
            str(sandbox),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    iteration_dir = Path(create.stdout.strip())
    assert iteration_dir.name == "20260524_execution-realism"

    manifest = json.loads((iteration_dir / "iteration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["control_ref"] == "v4-protocol101-control-2026-05-24"

    subprocess.run(
        [
            sys.executable,
            str(sandbox / "research_ops" / "scripts" / "validate_iteration.py"),
            str(iteration_dir),
            "--root",
            str(sandbox),
            "--registry",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(sandbox / "research_ops" / "scripts" / "summarize_iteration.py"),
            str(iteration_dir),
            "--root",
            str(sandbox),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert (iteration_dir / "reports" / "iteration_summary.md").exists()
