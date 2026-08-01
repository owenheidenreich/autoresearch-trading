"""Standard-library-only helpers for strict Path-D wire contracts."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def semantic_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def require_exact_keys(
    payload: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    name: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(payload))
    extras = sorted(set(payload) - allowed)
    if missing or extras:
        raise ValueError(f"{name} keys invalid: missing={missing}, extras={extras}")


def require_choice(value: str, choices: Iterable[str], name: str) -> str:
    normalized = str(value)
    allowed = tuple(choices)
    if normalized not in allowed:
        raise ValueError(f"{name} must be one of {allowed}; got {normalized!r}")
    return normalized


def require_nonempty(value: str, name: str) -> str:
    normalized = str(value)
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def require_utc_timestamp(value: str, name: str) -> str:
    normalized = str(value)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must be UTC")
    return normalized


def require_sha256(value: str, name: str) -> str:
    normalized = str(value)
    if not normalized.startswith("sha256:") or len(normalized) != 71:
        raise ValueError(f"{name} must be sha256:<64 lowercase hex chars>")
    digest = normalized[7:]
    if any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must use lowercase hexadecimal")
    return normalized


def require_int(value: int, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


STRICT_OBJECT = {"additionalProperties": False}

