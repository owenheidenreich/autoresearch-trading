"""Build exact-date Python-direct Track-A LaunchAgent plists."""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import plistlib
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from v4.ops.tracka.python_tracka_launcher import REPO_ROOT_ENV


LABELS = {
    "open": "com.autoresearch.tracka.capture",
    "midday": "com.autoresearch.tracka.capture.midday",
}
AUTHORIZED_PYTHON = "/Users/gduby/.autoresearch-trading/runtime-venv/bin/python"
INSTALLED_LAUNCHER = "/Users/gduby/.autoresearch-trading/tracka/python_tracka_launcher.py"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("declaration must be a JSON object")
    return payload


def _calendar_rows(
    declaration: Mapping[str, Any], window: str
) -> list[dict[str, int]]:
    sessions = declaration.get("capture_window", {}).get("sessions", [])
    windows = declaration.get("capture_window", {}).get("windows", [])
    matching = [row for row in windows if row.get("name") == window]
    if len(matching) != 1:
        raise ValueError(f"window missing or duplicated:{window}")
    time_text, timezone_name = str(matching[0]["start_local"]).split(" ", 1)
    hour, minute, _second = (int(value) for value in time_text.split(":"))
    rows: list[dict[str, int]] = []
    for value in sessions:
        session = date.fromisoformat(str(value))
        local = datetime(
            session.year,
            session.month,
            session.day,
            hour,
            minute,
            tzinfo=ZoneInfo(timezone_name),
        )
        rows.append(
            {
                "Month": local.month,
                "Day": local.day,
                "Hour": local.hour,
                "Minute": local.minute,
            }
        )
    if not rows:
        raise ValueError("declaration has no sessions")
    return rows


def build_plist(
    *,
    declaration: Mapping[str, Any],
    declaration_path: Path,
    authorization_path: Path,
    repo_root: Path,
    window: str,
) -> dict[str, Any]:
    if window not in LABELS:
        raise ValueError(f"unknown window:{window}")
    repo_root = repo_root.resolve()
    try:
        declaration_relative = declaration_path.resolve().relative_to(repo_root)
        authorization_relative = authorization_path.resolve().relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("declaration and authorization must be inside repo root") from exc
    log_root = "/Users/gduby/.autoresearch-trading/tracka/logs"
    return {
        "Label": LABELS[window],
        "ProgramArguments": [
            AUTHORIZED_PYTHON,
            INSTALLED_LAUNCHER,
            "--declaration",
            str(declaration_relative),
            "--authorization",
            str(authorization_relative),
            "--window",
            window,
        ],
        "EnvironmentVariables": {REPO_ROOT_ENV: str(repo_root)},
        "RunAtLoad": False,
        "StandardErrorPath": f"{log_root}/python-direct.{window}.err",
        "StandardOutPath": f"{log_root}/python-direct.{window}.out",
        "StartCalendarInterval": _calendar_rows(declaration, window),
    }


def write_plists(
    *,
    declaration_path: Path,
    authorization_path: Path,
    repo_root: Path,
    output_dir: Path,
) -> list[Path]:
    declaration = _load(declaration_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for window, label in LABELS.items():
        path = output_dir / f"{label}.plist"
        payload = build_plist(
            declaration=declaration,
            declaration_path=declaration_path,
            authorization_path=authorization_path,
            repo_root=repo_root,
            window=window,
        )
        path.write_bytes(plistlib.dumps(payload, sort_keys=True))
        written.append(path)
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path in write_plists(
        declaration_path=args.declaration,
        authorization_path=args.authorization,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
