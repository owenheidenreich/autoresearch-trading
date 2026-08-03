"""Append-only semantic hypothesis registry."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .compiler import CompiledHypothesis


class DuplicateSemanticHypothesis(RuntimeError):
    pass


def read_registry(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text().splitlines() if line.strip()]


def register(
    path: str | Path,
    compiled: CompiledHypothesis,
    *,
    status: str,
    result_path: str,
    engine_source_hash: str = "",
    allow_reexecution: bool = False,
) -> dict[str, Any]:
    target = Path(path)
    prior = read_registry(target)
    duplicate = next(
        (item for item in prior if item["semantic_hash"] == compiled.semantic_hash),
        None,
    )
    if duplicate is not None and not allow_reexecution:
        raise DuplicateSemanticHypothesis(
            f"semantic hypothesis already registered as {duplicate['hypothesis_id']} "
            f"with status {duplicate['status']}"
        )
    row = {
        "schema_version": "autoresearch_v2.semantic_registry.v1",
        "semantic_hash": compiled.semantic_hash,
        "hypothesis_id": compiled.spec.hypothesis_id,
        "component": compiled.spec.component,
        "epoch_id": compiled.spec.epoch_id,
        "status": status,
        "result_path": result_path,
        "engine_source_hash": engine_source_hash,
        "reexecution_of": duplicate.get("result_path") if duplicate is not None else None,
        "previous_status": duplicate.get("status") if duplicate is not None else None,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return row


def find_semantic(path: str | Path, compiled: CompiledHypothesis) -> Mapping[str, Any] | None:
    return next(
        (item for item in read_registry(path) if item["semantic_hash"] == compiled.semantic_hash),
        None,
    )
