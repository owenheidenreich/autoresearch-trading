"""Layer 1: teacher playbooks.

Small library of simple, auditable policies that generate directional signals
from bar-level context. Teachers do not pick contracts — contract selection
is a separate step that applies Layer 0 guardrails to the signal. Teachers
are pure functions over `BarContext`.
"""
