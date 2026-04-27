"""v4.feature_eng — feature row builder + forward-fill with age tracking.

Note: directory `v4/feature/` is the *data* directory (Parquet output,
gitignored content); this engineering code lives in `v4/feature_eng/`.
"""
from .builder import FeatureRow, ForwardFiller, feature_rows_to_table

__all__ = ["FeatureRow", "ForwardFiller", "feature_rows_to_table"]
