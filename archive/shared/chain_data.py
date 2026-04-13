"""Chain data primitives: contract representation, sidecar loading, snapshots.

Source: supervised.core.chain_data (aliased as v2.core.chain_data)
"""
from supervised.core.chain_data import (
    CONTRACT_FEATURE_FIELDS,
    NUM_CONTRACT_FEATURES,
    QUALITY_VALID,
    build_contract_row,
    describe_contract,
    extract_contract_series,
    load_sidecar_cached,
    padded_snapshot,
    to_wide_bar,
)

__all__ = [
    "CONTRACT_FEATURE_FIELDS",
    "NUM_CONTRACT_FEATURES",
    "QUALITY_VALID",
    "build_contract_row",
    "describe_contract",
    "extract_contract_series",
    "load_sidecar_cached",
    "padded_snapshot",
    "to_wide_bar",
]
