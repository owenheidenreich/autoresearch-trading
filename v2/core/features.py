"""Feature engineering: 39-feature vector from market data.

Extracted from v1 training/prepare.py compute_features() function.
See docs/v2/feature_schema.md for the complete feature specification.

Computes features from raw OHLCV + VIX + options data. Used by both
v2/pipeline/build_dataset.py (offline) and v2/live/market.py (real-time).

v1 origin: training/prepare.py (FEATURE_NAMES, compute_features,
normalize_features, _FEAT_IDX, _NO_NORMALIZE)
"""
# TODO: FEATURE_NAMES list (39 features, same as v1)
# TODO: NUM_FEATURES = 39
# TODO: _FEAT_IDX dict (name -> index lookup)
# TODO: _NO_NORMALIZE set (11 features excluded from z-score)
# TODO: compute_features(df) -> np.ndarray
# TODO: normalize_features(features, valid, dates) -> np.ndarray
# TODO: FeatureContractVersion class (from v1 contracts.py)
