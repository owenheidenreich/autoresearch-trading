"""Layer 2 supervised tooling for v3.

High-confidence bar-state models live here:
- export a per-eligible-bar supervised dataset
- train simple entry / side tabular models
- replay a one-trade-per-day policy against the post-A1 teacher baseline
"""

from .common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RUN_DIR,
    W2A_FEATURE_NAMES,
)

__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_RUN_DIR",
    "W2A_FEATURE_NAMES",
]
