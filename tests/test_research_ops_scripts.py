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
        "id",
        "priority",
        "layer",
        "assumption",
        "current_evidence",
        "risk_if_false",
        "falsification_test",
        "confidence_increases_if",
        "confidence_collapses_if",
        "required_artifacts",
        "blocked_actions",
        "status",
        "next_diagnostic",
    ]


def test_assumption_registry_seeds_required_stage3_assumptions():
    with (RESEARCH_OPS / "ASSUMPTION_REGISTRY.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["id"]: row for row in rows}
    assert {f"A{idx:03d}" for idx in range(1, 13)} <= set(by_id)
    assert by_id["A001"]["assumption"] == "quote age truth"
    assert by_id["A002"]["assumption"] == "ask-entry/bid-exit replay is executable"
    assert by_id["A011"]["priority"] == "P0"
    assert by_id["A014"]["status"] == "blocked"
    assert set(row["priority"] for row in rows) <= {"P0", "P1", "P2"}


def test_current_state_declares_audit_control_fields():
    text = (RESEARCH_OPS / "CURRENT_STATE.yaml").read_text(encoding="utf-8")
    for required in [
        "current_default:",
        "name: \"PAPER_DEFAULT_PROTOCOL101\"",
        "scope: \"guarded IBKR paper runtime only\"",
        "real_money: false",
        "frozen_control:",
        "protocol: \"Protocol101\"",
        "surface_model: \"Protocol051\"",
        "lifecycle_model: \"Protocol066_081\"",
        "blocked_actions:",
        "p0_risks:",
        "source_documents:",
        "next_phase: \"execution-and-parity falsification\"",
        "not_next_phase: \"model capacity expansion\"",
    ]:
        assert required in text


def test_do_not_touch_boundaries_cover_stage4_required_surfaces():
    text = (RESEARCH_OPS / "DO_NOT_TOUCH_WITHOUT_APPROVAL.md").read_text(encoding="utf-8")
    required_terms = [
        "v4/runtime/protocol101_paper_order_enablement.json",
        "v4/ops/launchd/*",
        "v4/ops/ibkr/run_protocol101_paper_session.sh",
        "v4/live/ibkr_paper_executor.py",
        "v4/live/ibkr_paper_guard.py",
        "Protocol101 model, scaler, manifest",
        "Protocol051/054 surface model, scaler, manifest",
        "Protocol066/081 lifecycle model, scaler, manifest",
        "Paid data download scripts",
        "Protected holdout scoring scripts",
        "Threshold selection logic",
        "paper-submit default behavior",
        "placeOrder",
        "safe read-only work",
        "Requires CEO Decision Memo",
        "Requires A Separate Branch",
        "Requires Human Approval",
    ]
    for required in required_terms:
        assert required in text


def test_pull_request_template_ties_prs_to_research_ops_controls():
    template = (REPO_ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    for required in [
        "## Purpose",
        "What assumption or RFC does this PR address?",
        "Iteration ID:",
        "Assumption ID:",
        "No trading logic changed unless explicitly approved",
        "No runtime flags changed",
        "No launchd files changed",
        "No broker calls added",
        "No paid data downloads added",
        "No training added",
        "No threshold tuning",
        "No model promotion",
        "No protected holdout scored for exploration",
        "## CEO Decision Required",
    ]:
        assert required in template


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
