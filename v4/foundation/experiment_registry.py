"""Append-only experiment registry for hill-climb governance."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ExperimentRegistryEntryV1"
DEFAULT_REGISTRY_PATH = Path("v4/audit/autoresearch/experiment_registry/registry.jsonl")
TERMINAL_DECISIONS = {"accepted", "rejected", "abandoned", "blocked"}


@dataclass(frozen=True)
class ExperimentRegistryEntry:
    experiment_id: str
    hypothesis: str
    code_version: str
    dataset_version: str
    feature_set: str
    label_definition: str
    train_windows: tuple[str, ...]
    validation_windows: tuple[str, ...]
    protected_test_windows: tuple[str, ...]
    primary_metric: str
    baseline_id: str
    candidate_id: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    score: dict[str, Any] = field(default_factory=dict)
    decision: str = "blocked"
    reason: str = ""
    negative_result_stored: bool = True
    protected_holdout_scored: bool = False
    protected_holdout_used_for_selection: bool = False
    paid_data_used: bool = False
    model_training: bool = False
    threshold_tuning: bool = False
    promotion_change: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["train_windows"] = list(self.train_windows)
        out["validation_windows"] = list(self.validation_windows)
        out["protected_test_windows"] = list(self.protected_test_windows)
        return out


def make_experiment_id(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:12]}"


def validate_experiment_registry_entry(entry: ExperimentRegistryEntry | dict[str, Any]) -> dict[str, Any]:
    row = entry.to_dict() if isinstance(entry, ExperimentRegistryEntry) else dict(entry)
    errors: list[str] = []
    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append("invalid_schema_version")
    for field_name in (
        "experiment_id",
        "hypothesis",
        "code_version",
        "dataset_version",
        "feature_set",
        "label_definition",
        "primary_metric",
        "baseline_id",
        "candidate_id",
        "decision",
        "reason",
    ):
        if not row.get(field_name):
            errors.append(f"missing_{field_name}")
    if row.get("decision") not in TERMINAL_DECISIONS:
        errors.append("decision_must_be_terminal")
    if not row.get("train_windows"):
        errors.append("missing_train_windows")
    if not row.get("validation_windows"):
        errors.append("missing_validation_windows")
    if not row.get("protected_test_windows"):
        errors.append("missing_protected_test_windows")
    if row.get("protected_holdout_used_for_selection"):
        errors.append("protected_holdout_used_for_selection")
    if row.get("promotion_change"):
        errors.append("registry_entry_must_not_change_paper_default")
    if row.get("decision") in {"rejected", "abandoned"} and not row.get("negative_result_stored"):
        errors.append("negative_result_not_stored")
    return {"status": "pass" if not errors else "fail", "errors": errors}


def append_experiment_registry_entry(
    entry: ExperimentRegistryEntry | dict[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    row = entry.to_dict() if isinstance(entry, ExperimentRegistryEntry) else dict(entry)
    validation = validate_experiment_registry_entry(row)
    if validation["status"] != "pass":
        raise ValueError(f"invalid experiment registry entry: {validation['errors']}")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return registry_path


def load_experiment_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in registry_path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows
