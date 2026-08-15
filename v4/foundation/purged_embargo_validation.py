"""Purged and embargoed chronological validation plan checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any


SCHEMA_VERSION = "PurgedEmbargoValidationPlanV1"
ALLOWED_ROLES = {"train", "validation", "test", "protected_test"}


@dataclass(frozen=True)
class SplitWindow:
    name: str
    role: str
    start: str
    end: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PurgedEmbargoValidationPlan:
    windows: tuple[SplitWindow, ...]
    purge_days: int = 1
    embargo_days: int = 1
    split_unit: str = "session_day"
    protected_holdout_names: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["windows"] = [window.to_dict() for window in self.windows]
        out["protected_holdout_names"] = list(self.protected_holdout_names)
        return out


def validate_purged_embargo_plan(plan: PurgedEmbargoValidationPlan | dict[str, Any]) -> dict[str, Any]:
    plan_obj = _coerce_plan(plan)
    errors: list[str] = []
    warnings: list[str] = []
    if plan_obj.schema_version != SCHEMA_VERSION:
        errors.append("invalid_schema_version")
    if plan_obj.split_unit != "session_day":
        errors.append("split_unit_must_be_session_day")
    if plan_obj.purge_days < 0 or plan_obj.embargo_days < 0:
        errors.append("negative_purge_or_embargo_days")
    if not plan_obj.windows:
        errors.append("missing_windows")
    parsed = []
    names = set()
    for window in plan_obj.windows:
        if window.name in names:
            errors.append(f"duplicate_window_name:{window.name}")
        names.add(window.name)
        if window.role not in ALLOWED_ROLES:
            errors.append(f"invalid_role:{window.name}:{window.role}")
        start = _parse_date(window.start)
        end = _parse_date(window.end)
        if start > end:
            errors.append(f"window_start_after_end:{window.name}")
        parsed.append((window, start, end))
    parsed.sort(key=lambda item: (item[1], item[2], item[0].name))
    for left, right in zip(parsed, parsed[1:]):
        left_window, _left_start, left_end = left
        right_window, right_start, _right_end = right
        min_gap = plan_obj.purge_days + plan_obj.embargo_days
        actual_gap = (right_start - left_end).days - 1
        if actual_gap < 0:
            errors.append(f"overlapping_windows:{left_window.name}:{right_window.name}")
        elif _requires_buffer(left_window.role, right_window.role) and actual_gap < min_gap:
            errors.append(
                f"insufficient_purge_embargo_gap:{left_window.name}:{right_window.name}:required={min_gap}:actual={actual_gap}"
            )
    for holdout_name in plan_obj.protected_holdout_names:
        matched = [window for window, _start, _end in parsed if window.name == holdout_name]
        if not matched:
            errors.append(f"missing_protected_holdout_window:{holdout_name}")
        elif matched[0].role != "protected_test":
            errors.append(f"protected_holdout_must_have_protected_test_role:{holdout_name}")
    if not any(window.role == "validation" for window, _start, _end in parsed):
        warnings.append("no_validation_window")
    if not any(window.role == "protected_test" for window, _start, _end in parsed):
        warnings.append("no_protected_test_window")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "purge_days": plan_obj.purge_days,
        "embargo_days": plan_obj.embargo_days,
        "split_unit": plan_obj.split_unit,
        "window_count": len(plan_obj.windows),
    }


def _requires_buffer(left_role: str, right_role: str) -> bool:
    return left_role != right_role or "protected_test" in {left_role, right_role}


def _coerce_plan(plan: PurgedEmbargoValidationPlan | dict[str, Any]) -> PurgedEmbargoValidationPlan:
    if isinstance(plan, PurgedEmbargoValidationPlan):
        return plan
    windows = tuple(SplitWindow(**window) for window in plan.get("windows", []))
    return PurgedEmbargoValidationPlan(
        windows=windows,
        purge_days=int(plan.get("purge_days", 1)),
        embargo_days=int(plan.get("embargo_days", 1)),
        split_unit=str(plan.get("split_unit", "session_day")),
        protected_holdout_names=tuple(str(item) for item in plan.get("protected_holdout_names", ())),
        schema_version=str(plan.get("schema_version", SCHEMA_VERSION)),
    )


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()
