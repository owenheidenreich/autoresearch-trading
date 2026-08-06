"""Python-direct Track-A launcher for use by macOS launchd.

The launchd program must be the uv Python interpreter that holds the macOS
Documents-folder grant.  This module performs orchestration in that same
process; it never hands the capture to a shell or a second interpreter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo


WINDOWS = {"open", "midday"}
REPO_ROOT_ENV = "AUTORESEARCH_REPO_ROOT"
CAPTURE_MODULES = (
    "v4/scripts/capture_databento_live_opra_definitions.py",
    "v4/scripts/capture_databento_live_opra_training_twin.py",
)


class TrackALaunchError(RuntimeError):
    """The launcher refused an unsafe, stale, or inconsistent invocation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declaration_hash(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material.pop("declaration_sha256", None)
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackALaunchError(f"{name} unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise TrackALaunchError(f"{name} must be a JSON object:{path}")
    return payload


def _resolve_input(repo_root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    return candidate.resolve(strict=True)


def _repo_file(repo_root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise TrackALaunchError(f"frozen implementation escapes repo:{value}") from exc
    if not resolved.is_file():
        raise TrackALaunchError(f"frozen implementation missing:{resolved}")
    return resolved


def _validate_frozen_implementation(
    repo_root: Path, declaration: Mapping[str, Any]
) -> list[dict[str, str]]:
    rows = declaration.get("measurements", {}).get("frozen_implementation", [])
    if not isinstance(rows, list) or not rows:
        raise TrackALaunchError("declaration has no frozen implementation hashes")
    verified: list[dict[str, str]] = []
    required = set(CAPTURE_MODULES)
    observed: set[str] = set()
    for row in rows:
        value = str(row.get("path", ""))
        expected = str(row.get("sha256", ""))
        path = _repo_file(repo_root, value)
        actual = _sha256(path)
        if len(expected) != 64 or actual != expected:
            raise TrackALaunchError(f"frozen implementation hash mismatch:{value}")
        observed.add(value)
        verified.append({"path": value, "sha256": actual})
    missing = sorted(required - observed)
    if missing:
        raise TrackALaunchError(f"capture modules absent from frozen hashes:{missing}")
    return verified


def build_capture_plan(
    *,
    repo_root: Path,
    declaration_path: Path,
    authorization_path: Path,
    window: str,
    session: date,
    allow_existing_output: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    if window not in WINDOWS:
        raise TrackALaunchError(f"unknown window:{window}")
    declaration_path = _resolve_input(repo_root, declaration_path)
    authorization_path = _resolve_input(repo_root, authorization_path)
    declaration = _load_json(declaration_path, "declaration")
    authorization = _load_json(authorization_path, "authorization")

    expected_declaration_hash = declaration.get("declaration_sha256")
    if expected_declaration_hash:
        actual_declaration_hash = _declaration_hash(declaration)
        if actual_declaration_hash != expected_declaration_hash:
            raise TrackALaunchError("declaration_sha256 mismatch")
    else:
        raise TrackALaunchError("declaration is not hash sealed")

    gate = declaration.get("authorization_gate") or {}
    if gate.get("connection_allowed") is not True:
        raise TrackALaunchError("declaration does not allow a connection")
    declared_sessions = declaration.get("capture_window", {}).get("sessions", [])
    authorized_sessions = authorization.get("authorized_scope", {}).get("sessions", [])
    session_text = session.isoformat()
    if session_text not in declared_sessions:
        raise TrackALaunchError(f"session not declared:{session_text}")
    if session_text not in authorized_sessions:
        raise TrackALaunchError(f"session not authorized:{session_text}")

    windows = declaration.get("capture_window", {}).get("windows", [])
    matching = [row for row in windows if row.get("name") == window]
    if len(matching) != 1:
        raise TrackALaunchError(f"window missing or duplicated:{window}")
    window_row = matching[0]
    duration = float(window_row.get("duration_seconds", 0))
    if not 1 <= duration <= 300:
        raise TrackALaunchError(f"invalid window duration:{duration}")

    expected_symbols = declaration.get("subscription", {}).get("expected_symbol_count")
    authorized_symbols = authorization.get("authorized_scope", {}).get(
        "expected_symbol_count"
    )
    if not isinstance(expected_symbols, int) or expected_symbols <= 0:
        raise TrackALaunchError("invalid declaration symbol count")
    if expected_symbols != authorized_symbols:
        raise TrackALaunchError(
            f"symbol-count authorization mismatch:{expected_symbols}!={authorized_symbols}"
        )
    schemas = list(declaration.get("subscription", {}).get("schemas", []))
    market_schemas = [value for value in schemas if value != "definition"]
    if not market_schemas:
        raise TrackALaunchError("declaration has no market schemas")

    approval_text = authorization.get("approval_required", {}).get("exact_approval_text")
    if not isinstance(approval_text, str) or not approval_text.strip():
        raise TrackALaunchError("authorization has no exact approval text")

    verified_hashes = _validate_frozen_implementation(repo_root, declaration)
    output_root = declaration_path.parent / session_text / window
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing_output:
        raise TrackALaunchError(f"output directory is not empty:{output_root}")

    definition_duration = float(
        declaration.get("capture_window", {}).get("definition_duration_seconds", 0)
    )
    if not 1 <= definition_duration <= 60:
        raise TrackALaunchError(f"invalid definition duration:{definition_duration}")

    return {
        "repo_root": repo_root,
        "declaration_path": declaration_path,
        "authorization_path": authorization_path,
        "authorization_text": approval_text,
        "session": session,
        "window": window,
        "duration_seconds": duration,
        "definition_duration_seconds": definition_duration,
        "expected_symbol_count": expected_symbols,
        "market_schemas": market_schemas,
        "output_root": output_root,
        "verified_implementation": verified_hashes,
        "start_local": str(window_row.get("start_local", "")),
    }


@contextmanager
def _argv(values: Sequence[str]) -> Iterator[None]:
    original = sys.argv[:]
    sys.argv = list(values)
    try:
        yield
    finally:
        sys.argv = original


def _call_main(main: Callable[[], int], argv: Sequence[str]) -> None:
    with _argv(argv):
        try:
            rc = main()
        except SystemExit as exc:
            rc = int(exc.code or 0)
    if rc != 0:
        raise TrackALaunchError(f"capture entry point returned {rc}:{argv[0]}")


def execute_capture_plan(
    plan: Mapping[str, Any],
    *,
    dry_run: bool,
    dry_run_definition_path: Path | None,
    dry_run_definition_session: date | None,
) -> None:
    repo_root = Path(plan["repo_root"])
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    from v4.scripts import capture_databento_live_opra_definitions as definitions
    from v4.scripts import capture_databento_live_opra_training_twin as market

    output_root = Path(plan["output_root"])
    authorization_path = Path(plan["authorization_path"])
    session = plan["session"].isoformat()
    os.environ["V4_PAID_DATA_APPROVAL_TEXT"] = str(plan["authorization_text"])
    os.chdir(repo_root)

    definition_dir = output_root / "definitions"
    definition_argv = [
        str(definitions.__file__),
        "--session-date",
        session,
        "--duration-seconds",
        str(plan["definition_duration_seconds"]),
        "--output-dir",
        str(definition_dir),
        "--env-file",
        str(repo_root / "v4/.env"),
        "--approval-manifest",
        str(authorization_path),
    ]
    if dry_run:
        definition_argv.append("--dry-run")
    _call_main(definitions.main, definition_argv)

    if dry_run:
        if dry_run_definition_path is None:
            raise TrackALaunchError("dry-run requires --dry-run-definition-path")
        definition_path = dry_run_definition_path.resolve(strict=True)
    else:
        definition_path = definition_dir / "opra_live_definitions.dbn.zst"
        if not definition_path.is_file():
            raise TrackALaunchError(f"definition capture missing:{definition_path}")

    market_session = (
        dry_run_definition_session.isoformat()
        if dry_run and dry_run_definition_session is not None
        else session
    )
    market_argv = [
        str(market.__file__),
        "--session-date",
        market_session,
        "--definition-path",
        str(definition_path),
        "--duration-seconds",
        str(plan["duration_seconds"]),
        "--expected-symbol-count",
        str(plan["expected_symbol_count"]),
        "--schemas",
        *plan["market_schemas"],
        "--output-dir",
        str(output_root / "market"),
        "--env-file",
        str(repo_root / "v4/.env"),
        "--approval-manifest",
        str(authorization_path),
    ]
    if dry_run:
        market_argv.append("--dry-run")
    _call_main(market.main, market_argv)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=os.environ.get(REPO_ROOT_ENV),
        required=REPO_ROOT_ENV not in os.environ,
        help=f"repository root; defaults to ${REPO_ROOT_ENV}",
    )
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--window", choices=sorted(WINDOWS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-definition-path", type=Path)
    parser.add_argument(
        "--dry-run-definition-session-date",
        type=date.fromisoformat,
        help="dry-run fixture date; never accepted for a live invocation",
    )
    parser.add_argument(
        "--session-date",
        type=date.fromisoformat,
        help="dry-run only; live invocations always use today's PT date",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.session_date is not None and not args.dry_run:
        raise TrackALaunchError("--session-date override is dry-run only")
    if args.dry_run_definition_session_date is not None and not args.dry_run:
        raise TrackALaunchError(
            "--dry-run-definition-session-date is dry-run only"
        )
    session = args.session_date or datetime.now(
        ZoneInfo("America/Los_Angeles")
    ).date()
    plan = build_capture_plan(
        repo_root=args.repo_root,
        declaration_path=args.declaration,
        authorization_path=args.authorization,
        window=args.window,
        session=session,
        allow_existing_output=args.dry_run,
    )
    print(
        json.dumps(
            {
                "status": "DRY_RUN" if args.dry_run else "CAPTURE_START",
                "session": session.isoformat(),
                "window": args.window,
                "interpreter": os.path.realpath(sys.executable),
                "repo_root": str(plan["repo_root"]),
                "output_root": str(plan["output_root"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    execute_capture_plan(
        plan,
        dry_run=args.dry_run,
        dry_run_definition_path=args.dry_run_definition_path,
        dry_run_definition_session=args.dry_run_definition_session_date,
    )
    print(
        json.dumps(
            {
                "status": "DRY_RUN_COMPLETE" if args.dry_run else "CAPTURE_COMPLETE",
                "session": session.isoformat(),
                "window": args.window,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {"status": "REFUSED_OR_FAILED", "error": f"{type(exc).__name__}:{exc}"},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
