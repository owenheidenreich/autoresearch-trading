"""Guardrails for paid/licensed market-data downloads.

Cost-estimation and local dry-runs are allowed without approval. Any path that
calls a market-data download endpoint should call ``require_paid_data_approval``
immediately before the endpoint call.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_APPROVAL_ENV_VAR = "V4_PAID_DATA_APPROVAL_TEXT"
DEFAULT_Q3_2024_MANIFEST = Path("v4/promotion/PROTOCOL_101_Q3_2024_DOWNLOAD_MANIFEST.json")


def add_paid_data_approval_args(
    parser: argparse.ArgumentParser,
    *,
    default_manifest: Path = DEFAULT_Q3_2024_MANIFEST,
) -> None:
    parser.add_argument(
        "--approval-manifest",
        type=Path,
        default=default_manifest,
        help="JSON manifest containing approval_required.exact_approval_text.",
    )
    parser.add_argument(
        "--approval-text",
        default=None,
        help="Exact user approval text. Prefer using V4_PAID_DATA_APPROVAL_TEXT to avoid shell history.",
    )
    parser.add_argument(
        "--approval-env-var",
        default=DEFAULT_APPROVAL_ENV_VAR,
        help="Environment variable containing exact user approval text.",
    )


def exact_approval_text(manifest_path: Path) -> str:
    if not manifest_path.exists():
        raise SystemExit(f"paid-data approval manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    try:
        text = manifest["approval_required"]["exact_approval_text"]
    except KeyError as exc:
        raise SystemExit(
            f"paid-data approval manifest is missing approval_required.exact_approval_text: {manifest_path}"
        ) from exc
    if not isinstance(text, str) or not text.strip():
        raise SystemExit(f"paid-data approval manifest has empty approval text: {manifest_path}")
    return text


def provided_approval_text(*, approval_text: str | None, approval_env_var: str) -> str:
    if approval_text is not None:
        return approval_text
    return os.environ.get(approval_env_var, "")


def require_paid_data_approval(
    *,
    manifest_path: Path,
    approval_text: str | None,
    approval_env_var: str = DEFAULT_APPROVAL_ENV_VAR,
    operation: str,
) -> None:
    required = exact_approval_text(manifest_path)
    provided = provided_approval_text(
        approval_text=approval_text,
        approval_env_var=approval_env_var,
    )
    if provided == required:
        return
    raise SystemExit(
        "paid-data approval required before "
        f"{operation}. Provide the exact text from {manifest_path} via "
        f"--approval-text or the {approval_env_var} environment variable."
    )
