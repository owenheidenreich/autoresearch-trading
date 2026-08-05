"""Source-neutral validation artifacts for future v5 candidates."""
from __future__ import annotations

from typing import Any


__all__ = ["CandidatePacketError", "export_candidate_packet"]


def __getattr__(name: str) -> Any:
    """Keep package exports without preloading the command-line module."""

    if name in __all__:
        from . import candidate_packet

        return getattr(candidate_packet, name)
    raise AttributeError(name)
