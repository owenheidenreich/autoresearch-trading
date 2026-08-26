#!/usr/bin/env python3
"""Prepare and validate the Job-49 CMBP catalogue entirely offline.

The executable surface is intentionally limited to local JSON and Parquet
files.  It contains no service client and has no command that performs a
network request, time-series request, download, purchase, or outcome read.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

from v5.research import cmbp_catalogue_preflight as catalogue


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = REPO_ROOT / "v5/research/cmbp_catalogue_preflight.py"
WRAPPER_PATH = Path(__file__).resolve()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def implementation_code_hashes() -> dict[str, str]:
    return {
        "v5/ops/prepare_cmbp_catalogue_preflight.py": catalogue.file_sha256(WRAPPER_PATH),
        "v5/research/cmbp_catalogue_preflight.py": catalogue.file_sha256(CORE_PATH),
    }


def dependency_manifest() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("pyarrow",):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": packages,
        "network_client_packages_imported": [],
    }


def _early_close_sessions(path: Path | None) -> Sequence[str] | None:
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("early_close_sessions")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("early-close JSON must be a list of ISO sessions")
    return value


def _prepare_manifest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    declaration = catalogue.prepare_catalogue_declaration(
        manifest,
        code_hashes=implementation_code_hashes(),
        dependency_manifest=dependency_manifest(),
        source_manifest_file_sha256=catalogue.file_sha256(manifest_path),
    )
    write_json(Path(args.output), declaration)
    print(
        f"{declaration['status']}: {declaration['source_session_count']} source sessions, "
        f"{declaration['request_session_count']} event-era request sessions"
    )
    return 0


def _prepare_ladder(args: argparse.Namespace) -> int:
    manifest = catalogue.manifest_from_ladder_directory(
        Path(args.ladder_root),
        coverage_start=args.coverage_start,
        expected_source_session_count=args.expected_source_sessions,
        early_close_sessions=_early_close_sessions(
            Path(args.early_close_sessions) if args.early_close_sessions else None
        ),
    )
    source_manifest_file_sha256: str | None = None
    if args.manifest_output:
        manifest_path = Path(args.manifest_output)
        write_json(manifest_path, manifest)
        source_manifest_file_sha256 = catalogue.file_sha256(manifest_path)
    declaration = catalogue.prepare_catalogue_declaration(
        manifest,
        code_hashes=implementation_code_hashes(),
        dependency_manifest=dependency_manifest(),
        source_manifest_file_sha256=source_manifest_file_sha256,
    )
    write_json(Path(args.output), declaration)
    print(
        f"{declaration['status']}: {declaration['source_session_count']} source sessions, "
        f"{declaration['excluded_pre_event_era_session_count']} structurally excluded, "
        f"{declaration['request_session_count']} event-era request sessions"
    )
    return 0


def _template(args: argparse.Namespace) -> int:
    declaration = load_json(Path(args.declaration))
    catalogue.validate_catalogue_declaration(
        declaration,
        expected_code_hashes=implementation_code_hashes(),
        expected_dependency_manifest=dependency_manifest(),
    )
    value = catalogue.response_template(declaration, source=args.source)
    write_json(Path(args.output), value)
    print(f"OFFLINE_RESPONSE_TEMPLATE: {len(value['sessions'])} sessions")
    return 0


def _build_receipt(args: argparse.Namespace, *, write: bool) -> int:
    declaration = load_json(Path(args.declaration))
    responses = load_json(Path(args.responses))
    receipt = catalogue.build_preflight_receipt(
        declaration,
        responses,
        hard_cap_usd=args.hard_cap_usd,
        expected_code_hashes=implementation_code_hashes(),
        expected_dependency_manifest=dependency_manifest(),
    )
    catalogue.validate_preflight_receipt(receipt, declaration=declaration)
    if write:
        write_json(Path(args.output), receipt)
    print(
        f"{receipt['status']}: {receipt.get('response_session_count', 0)} response sessions, "
        f"exact cost ${receipt.get('exact_total_cost_usd', 'NOT_PASSED')}"
    )
    return 0 if receipt["status"] == catalogue.STATUS_PREFLIGHT_PASS_ONLY else 3


def _validate_declaration(args: argparse.Namespace) -> int:
    declaration = load_json(Path(args.declaration))
    catalogue.validate_catalogue_declaration(
        declaration,
        expected_code_hashes=implementation_code_hashes(),
        expected_dependency_manifest=dependency_manifest(),
    )
    print(
        f"{declaration['status']}: self-hash and all scope/request/implementation hashes valid"
    )
    return 0


def _validate_receipt(args: argparse.Namespace) -> int:
    receipt = load_json(Path(args.receipt))
    declaration = load_json(Path(args.declaration)) if args.declaration else None
    catalogue.validate_preflight_receipt(receipt, declaration=declaration)
    print(f"{receipt['status']}: receipt self-hash and safety fields valid")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)

    prepare_manifest = sub.add_parser(
        "prepare-manifest", help="freeze an already supplied offline session/symbol manifest"
    )
    prepare_manifest.add_argument("--manifest", required=True, type=Path)
    prepare_manifest.add_argument("--output", required=True, type=Path)
    prepare_manifest.set_defaults(handler=_prepare_manifest)

    prepare_ladder = sub.add_parser(
        "prepare-ladder",
        help="read only ladder filenames and raw_symbol columns, then freeze the scope",
    )
    prepare_ladder.add_argument("--ladder-root", required=True, type=Path)
    prepare_ladder.add_argument(
        "--coverage-start",
        required=True,
        help="must explicitly declare the cmbp-1 event-era start (2023-03-28)",
    )
    prepare_ladder.add_argument("--expected-source-sessions", type=int, default=1_014)
    prepare_ladder.add_argument(
        "--early-close-sessions",
        type=Path,
        help="optional JSON list replacing the pinned 2022-2026 XNYS early-close set",
    )
    prepare_ladder.add_argument("--manifest-output", type=Path)
    prepare_ladder.add_argument("--output", required=True, type=Path)
    prepare_ladder.set_defaults(handler=_prepare_ladder)

    template = sub.add_parser(
        "response-template", help="write a fill-only offline response skeleton"
    )
    template.add_argument("--declaration", required=True, type=Path)
    template.add_argument("--source", choices=("synthetic", "externally_supplied"), default="synthetic")
    template.add_argument("--output", required=True, type=Path)
    template.set_defaults(handler=_template)

    receipt = sub.add_parser(
        "receipt", help="validate supplied per-session metadata JSON and write a pass/STOP receipt"
    )
    receipt.add_argument("--declaration", required=True, type=Path)
    receipt.add_argument("--responses", required=True, type=Path)
    receipt.add_argument("--hard-cap-usd", required=True)
    receipt.add_argument("--output", required=True, type=Path)
    receipt.set_defaults(handler=lambda args: _build_receipt(args, write=True))

    validate_responses = sub.add_parser(
        "validate-responses", help="validate metadata JSON and print the disposition without writing"
    )
    validate_responses.add_argument("--declaration", required=True, type=Path)
    validate_responses.add_argument("--responses", required=True, type=Path)
    validate_responses.add_argument("--hard-cap-usd", required=True)
    validate_responses.set_defaults(handler=lambda args: _build_receipt(args, write=False))

    validate_declaration = sub.add_parser(
        "validate-declaration", help="verify a declaration and current implementation hashes"
    )
    validate_declaration.add_argument("--declaration", required=True, type=Path)
    validate_declaration.set_defaults(handler=_validate_declaration)

    validate_receipt = sub.add_parser(
        "validate-receipt", help="verify an offline pass/STOP receipt"
    )
    validate_receipt.add_argument("--receipt", required=True, type=Path)
    validate_receipt.add_argument("--declaration", type=Path)
    validate_receipt.set_defaults(handler=_validate_receipt)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except catalogue.CataloguePreflightError as exc:
        print(f"{exc.status}: {exc}", file=sys.stderr)
        return 2
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{catalogue.STOP_SCOPE_OR_CODE_DRIFT}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
