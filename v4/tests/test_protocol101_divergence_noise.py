from __future__ import annotations

import numpy as np
import pandas as pd

from v4.model.protocol101_divergence_noise import DivergenceNoiseModel, moneyness_band


def test_moneyness_band_boundaries() -> None:
    assert moneyness_band(0) == "atm"
    assert moneyness_band(10) == "atm"
    assert moneyness_band(15) == "near"
    assert moneyness_band(25) == "near"
    assert moneyness_band(30) == "wing"


def test_loader_derives_bands_and_keeps_zero_divergence_untouched(tmp_path) -> None:
    rows = [
        {
            "pair_key": "a",
            "feature": "B.ladder.abs_offset",
            "family": "B",
            "historical_value": 5.0,
            "ibkr_value": 5.0,
        },
        {
            "pair_key": "b",
            "feature": "B.ladder.abs_offset",
            "family": "B",
            "historical_value": 30.0,
            "ibkr_value": 30.0,
        },
        {
            "pair_key": "a",
            "feature": "C.mid.mid_tick_q",
            "family": "C",
            "historical_value": 1.0,
            "ibkr_value": 1.05,
        },
        {
            "pair_key": "b",
            "feature": "C.mid.mid_tick_q",
            "family": "C",
            "historical_value": 2.0,
            "ibkr_value": 1.90,
        },
        {
            "pair_key": "a",
            "feature": "A.context.spx_for_ladder",
            "family": "A",
            "historical_value": 6000.0,
            "ibkr_value": 6000.0,
        },
    ]
    path = tmp_path / "divergence.parquet"
    pd.DataFrame(rows).to_parquet(path)
    model = DivergenceNoiseModel.from_parquet(path)
    assert model.feature_has_nonzero_noise("C.mid.mid_tick_q")
    assert not model.feature_has_nonzero_noise("A.context.spx_for_ladder")
    assert model.samples[("C.mid.mid_tick_q", "atm")].tolist() == [0.050000000000000044]
    assert model.samples[("C.mid.mid_tick_q", "wing")].tolist() == [-0.10000000000000009]


def test_injection_is_seed_deterministic_and_preserves_family_a(tmp_path) -> None:
    rows = []
    for i in range(6):
        rows.append(
            {
                "pair_key": f"atm{i}",
                "feature": "B.ladder.abs_offset",
                "family": "B",
                "historical_value": 5.0,
                "ibkr_value": 5.0,
            }
        )
        rows.append(
            {
                "pair_key": f"atm{i}",
                "feature": "C.mid.mid_tick_q",
                "family": "C",
                "historical_value": 1.0,
                "ibkr_value": 1.0 + (i + 1) * 0.01,
            }
        )
        rows.append(
            {
                "pair_key": f"atm{i}",
                "feature": "A.context.spx_for_ladder",
                "family": "A",
                "historical_value": 6000.0,
                "ibkr_value": 6000.0,
            }
        )
    path = tmp_path / "divergence.parquet"
    pd.DataFrame(rows).to_parquet(path)
    model = DivergenceNoiseModel.from_parquet(path)
    frame = pd.DataFrame(
        {
            "B.ladder.abs_offset": [5.0] * 20,
            "C.mid.mid_tick_q": [1.0] * 20,
            "A.context.spx_for_ladder": [6000.0] * 20,
        }
    )
    a = model.inject_dataframe(frame, seed=123)
    b = model.inject_dataframe(frame, seed=123)
    c = model.inject_dataframe(frame, seed=124)
    assert np.allclose(a["C.mid.mid_tick_q"], b["C.mid.mid_tick_q"])
    assert not np.allclose(a["C.mid.mid_tick_q"], c["C.mid.mid_tick_q"])
    assert np.allclose(a["A.context.spx_for_ladder"], frame["A.context.spx_for_ladder"])
