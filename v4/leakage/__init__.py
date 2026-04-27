"""v4.leakage — designed-to-fail leak-detection harness.

See detector.py. Two probes are run on every promotion:
- shuffled_label_test: confirms features have no hidden time-leak
- planted_leak_test: confirms the detector itself works (catches a planted leak)
"""
from .detector import (
    LeakProbeResult,
    baseline_probe,
    planted_leak_test,
    probe_auc,
    shuffled_label_test,
)

__all__ = [
    "LeakProbeResult",
    "baseline_probe",
    "planted_leak_test",
    "probe_auc",
    "shuffled_label_test",
]
