"""Live-readiness utilities for v4 promotion checks.

These modules are deliberately broker-disconnected until promotion gates clear.
"""
from .shadow_lifecycle import StrictShadowLifecycleResult, strict_serial_shadow_rows

__all__ = ["StrictShadowLifecycleResult", "strict_serial_shadow_rows"]
