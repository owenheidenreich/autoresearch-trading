from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import plistlib

import pytest

from v4.ops.tracka.build_python_tracka_plists import (
    AUTHORIZED_PYTHON,
    INSTALLED_LAUNCHER,
    build_plist,
)
from v4.ops.tracka.python_tracka_launcher import (
    REPO_ROOT_ENV,
    TrackALaunchError,
    _declaration_hash,
    build_capture_plan,
    main,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = (
    REPO_ROOT
    / "v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
)
DECLARATION = CAPTURE_ROOT / "capture_declaration_v6.json"
AUTHORIZATION = CAPTURE_ROOT / "authorization.json"
DEFINITION_PATH = (
    CAPTURE_ROOT
    / "2026-08-05/midday/definitions/current_session_definitions.parquet"
)


def _copies(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    declaration = json.loads(DECLARATION.read_text())
    authorization = json.loads(AUTHORIZATION.read_text())
    declaration_path = tmp_path / "capture_declaration.json"
    authorization_path = tmp_path / "authorization.json"
    declaration_path.write_text(json.dumps(declaration))
    authorization_path.write_text(json.dumps(authorization))
    return declaration_path, authorization_path, declaration, authorization


def _write_declaration(path: Path, payload: dict) -> None:
    payload["declaration_sha256"] = _declaration_hash(payload)
    path.write_text(json.dumps(payload))


def test_real_declaration_builds_a_fail_closed_plan(tmp_path: Path) -> None:
    declaration_path, authorization_path, _, _ = _copies(tmp_path)
    plan = build_capture_plan(
        repo_root=REPO_ROOT,
        declaration_path=declaration_path,
        authorization_path=authorization_path,
        window="open",
        session=date(2026, 8, 6),
    )
    assert plan["expected_symbol_count"] == 510
    assert plan["market_schemas"] == ["cbbo-1s", "cbbo-1m", "ohlcv-1m", "trades"]
    assert len(plan["verified_implementation"]) == 3


@pytest.mark.parametrize("window", ["close", "", "OPEN"])
def test_unknown_window_is_refused(tmp_path: Path, window: str) -> None:
    declaration_path, authorization_path, _, _ = _copies(tmp_path)
    with pytest.raises(TrackALaunchError, match="unknown window"):
        build_capture_plan(
            repo_root=REPO_ROOT,
            declaration_path=declaration_path,
            authorization_path=authorization_path,
            window=window,
            session=date(2026, 8, 6),
        )


def test_undeclared_session_is_refused(tmp_path: Path) -> None:
    declaration_path, authorization_path, _, _ = _copies(tmp_path)
    with pytest.raises(TrackALaunchError, match="session not declared"):
        build_capture_plan(
            repo_root=REPO_ROOT,
            declaration_path=declaration_path,
            authorization_path=authorization_path,
            window="open",
            session=date(2026, 8, 8),
        )


def test_authorization_and_symbol_mismatches_are_refused(tmp_path: Path) -> None:
    declaration_path, authorization_path, _, authorization = _copies(tmp_path)
    authorization["authorized_scope"]["sessions"] = ["2026-08-07"]
    authorization_path.write_text(json.dumps(authorization))
    with pytest.raises(TrackALaunchError, match="session not authorized"):
        build_capture_plan(
            repo_root=REPO_ROOT,
            declaration_path=declaration_path,
            authorization_path=authorization_path,
            window="open",
            session=date(2026, 8, 6),
        )

    authorization = json.loads(AUTHORIZATION.read_text())
    authorization["authorized_scope"]["expected_symbol_count"] = 509
    authorization_path.write_text(json.dumps(authorization))
    with pytest.raises(TrackALaunchError, match="symbol-count authorization mismatch"):
        build_capture_plan(
            repo_root=REPO_ROOT,
            declaration_path=declaration_path,
            authorization_path=authorization_path,
            window="open",
            session=date(2026, 8, 6),
        )


def test_tampered_frozen_hash_and_nonempty_output_are_refused(tmp_path: Path) -> None:
    declaration_path, authorization_path, declaration, _ = _copies(tmp_path)
    tampered = deepcopy(declaration)
    tampered["measurements"]["frozen_implementation"][0]["sha256"] = "0" * 64
    _write_declaration(declaration_path, tampered)
    with pytest.raises(TrackALaunchError, match="frozen implementation hash mismatch"):
        build_capture_plan(
            repo_root=REPO_ROOT,
            declaration_path=declaration_path,
            authorization_path=authorization_path,
            window="open",
            session=date(2026, 8, 6),
        )

    _write_declaration(declaration_path, declaration)
    output = tmp_path / "2026-08-06/open"
    output.mkdir(parents=True)
    (output / "unexpected.txt").write_text("occupied")
    with pytest.raises(TrackALaunchError, match="output directory is not empty"):
        build_capture_plan(
            repo_root=REPO_ROOT,
            declaration_path=declaration_path,
            authorization_path=authorization_path,
            window="open",
            session=date(2026, 8, 6),
        )


def test_python_launcher_executes_both_dry_runs_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not DEFINITION_PATH.is_file():
        pytest.skip("owned 2026-08-05 definition capture is unavailable")
    declaration_path, authorization_path, _, _ = _copies(tmp_path)
    monkeypatch.setenv(REPO_ROOT_ENV, str(REPO_ROOT))
    assert main(
        [
            "--declaration",
            str(declaration_path),
            "--authorization",
            str(authorization_path),
            "--window",
            "open",
            "--dry-run",
            "--dry-run-definition-path",
            str(DEFINITION_PATH),
            "--dry-run-definition-session-date",
            "2026-08-05",
            "--session-date",
            "2026-08-06",
        ]
    ) == 0


def test_generated_plists_are_python_direct_and_exact_date() -> None:
    declaration = json.loads(DECLARATION.read_text())
    for window in ("open", "midday"):
        payload = build_plist(
            declaration=declaration,
            declaration_path=DECLARATION,
            authorization_path=AUTHORIZATION,
            repo_root=REPO_ROOT,
            window=window,
        )
        assert payload["ProgramArguments"][:2] == [AUTHORIZED_PYTHON, INSTALLED_LAUNCHER]
        assert payload["EnvironmentVariables"] == {REPO_ROOT_ENV: str(REPO_ROOT)}
        assert str(REPO_ROOT) not in "\n".join(payload["ProgramArguments"])
        assert "WorkingDirectory" not in payload
        rows = payload["StartCalendarInterval"]
        assert [(row["Month"], row["Day"]) for row in rows] == [(8, 6), (8, 7)]
        assert all("Weekday" not in row for row in rows)


def test_staged_plists_are_valid_python_direct_templates() -> None:
    for path in (
        REPO_ROOT / "v4/ops/tracka/com.autoresearch.tracka.capture.plist.staged",
        REPO_ROOT / "v4/ops/tracka/com.autoresearch.tracka.capture.midday.plist.staged",
    ):
        payload = plistlib.loads(path.read_bytes())
        assert payload["ProgramArguments"][0] == AUTHORIZED_PYTHON
        assert payload["ProgramArguments"][1] == INSTALLED_LAUNCHER
        assert payload["EnvironmentVariables"] == {REPO_ROOT_ENV: str(REPO_ROOT)}
        assert str(REPO_ROOT) not in "\n".join(payload["ProgramArguments"])
        assert "StartCalendarInterval" not in payload
        assert any(
            Path(argument).name == "REQUIRES_NEW_CAPTURE_DECLARATION.json"
            for argument in payload["ProgramArguments"]
        )
        assert any(
            Path(argument).name == "REQUIRES_NEW_CAPTURE_AUTHORIZATION.json"
            for argument in payload["ProgramArguments"]
        )


def test_launcher_contains_no_shell_or_subprocess_handoff() -> None:
    source = (
        REPO_ROOT / "v4/ops/tracka/python_tracka_launcher.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("subprocess", "os.system", "/bin/zsh", "/usr/bin/python3", "brctl"):
        assert forbidden not in source


def test_shell_runners_carry_the_correction_not_the_false_cancellation() -> None:
    """A 2026-08-05 stanza claimed the owner canceled declaration v6. The owner
    made no such cancellation (owner statement, 2026-08-05 evening), so the
    runners must stay free of that claim while keeping the real fail-closed
    gates: undeclared sessions refused, ungranted shells refused."""

    for relative in (
        "v4/ops/tracka/run_tracka_window.sh",
        "v4/ops/tracka/run_tracka_attended.sh",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "CANCELED_BY_OWNER" not in source
        assert "Correction of record 2026-08-05" in source

    window = (REPO_ROOT / "v4/ops/tracka/run_tracka_window.sh").read_text(encoding="utf-8")
    assert "is not a declared session" in window
    assert "exit 78" in window

    attended = (REPO_ROOT / "v4/ops/tracka/run_tracka_attended.sh").read_text(encoding="utf-8")
    assert "does not hold Documents access" in attended
