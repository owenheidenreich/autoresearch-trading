from __future__ import annotations

import numpy as np

from training.prepare import FEATURE_NAMES
from training.live.contracts import (
    FeatureContractVersion,
    LiveContextBundle,
    load_bundle,
    save_bundle,
)


def test_feature_contract_shape_and_names() -> None:
    contract = FeatureContractVersion()
    arr = np.zeros((12, len(FEATURE_NAMES)), dtype=np.float32)
    assert contract.validate_feature_shape(arr)
    assert contract.validate_feature_names(FEATURE_NAMES)
    assert not contract.validate_feature_shape(np.zeros((12, len(FEATURE_NAMES) + 1)))


def test_bundle_round_trip(tmp_path) -> None:
    bundle = LiveContextBundle(
        as_of_date="2026-03-17",
        context_start_date="2026-02-17",
        context_end_date="2026-03-17",
        feature_contract_version="ibkr_live_v1",
        raw_features=np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32),
        normalized_features=np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32),
        valid_mask=np.ones((4,), dtype=bool),
        dates=["2026-03-17"] * 4,
        timestamps=["2026-03-17 09:30"] * 4,
        norm_raw_buffer=np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32),
        norm_valid_buffer=np.ones((4,), dtype=bool),
        market_rows=[],
        vix_rows={},
        options_data={},
        chain_data={},
        prior_levels={},
        multi_timeframe_stats={},
        source_meta={},
    )
    path = tmp_path / "bundle.pt"
    save_bundle(str(path), bundle)
    loaded = load_bundle(str(path))
    assert loaded.as_of_date == "2026-03-17"
    assert loaded.raw_features.shape == bundle.raw_features.shape

