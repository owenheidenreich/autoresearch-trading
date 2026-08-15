from __future__ import annotations

import numpy as np

from v5.ops.audit_causal_day_atlas_profit import corrected_interval


def test_corrected_interval_uses_every_session_value() -> None:
    values = np.array([100.0, 0.0, -100.0])
    got = corrected_interval(values)
    assert got["n_sessions"] == 3
    assert got["mean_usd_per_session"] == 0.0
    assert got["ci_low"] <= 0.0 <= got["ci_high"]
