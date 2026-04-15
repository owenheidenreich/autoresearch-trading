"""RuntimeConfig: single source of truth for shared constants.

Every file that needs BARS_PER_DAY, NUM_FEATURES, LOOKBACK, or other shared
constants should import from here instead of defining its own copy.

When any field changes, the fingerprint changes, which flags all downstream
artifacts as stale.

Created 2026-04-15 as part of the stage-decoupled architecture fix.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable configuration shared across all pipeline stages."""

    # --- Dimensions ---
    bars_per_day: int = 390
    num_features: int = 52
    num_contract_features: int = 22
    lookback: int = 30
    max_contracts_per_bar: int = 100

    # --- Trading constants ---
    strike_grid: float = 5.0
    contract_multiplier: int = 100
    starting_equity: float = 10_000.0

    # --- Schema versioning ---
    chain_schema_version: str = "v4_exact_chain_v2_paths"

    def fingerprint(self) -> str:
        """SHA-256[:16] of all config values. Changes invalidate downstream artifacts."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        payload = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, path: str) -> None:
        import pathlib
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: str) -> RuntimeConfig:
        import pathlib
        d = json.loads(pathlib.Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})

    def validate_dataset_metadata(self, meta: dict) -> list[str]:
        """Check dataset metadata against this config. Returns list of errors."""
        errors = []
        if meta.get("n_features", self.num_features) != self.num_features:
            errors.append(
                f"dataset n_features={meta.get('n_features')} != config {self.num_features}"
            )
        stored_names = meta.get("feature_names")
        if stored_names is not None:
            from v2.core.features import FEATURE_NAMES
            if list(stored_names) != list(FEATURE_NAMES):
                errors.append("dataset feature_names order does not match current FEATURE_NAMES")
        if meta.get("chain_schema_version", self.chain_schema_version) != self.chain_schema_version:
            errors.append(
                f"dataset chain_schema={meta.get('chain_schema_version')} != config {self.chain_schema_version}"
            )
        return errors

    def validate_checkpoint(self, checkpoint: dict, dataset_fingerprint: str | None = None) -> list[str]:
        """Check model checkpoint against this config. Returns list of errors."""
        errors = []
        hp = checkpoint.get("hyperparams", {})
        if hp.get("lookback", self.lookback) != self.lookback:
            errors.append(f"checkpoint lookback={hp.get('lookback')} != config {self.lookback}")
        if hp.get("contract_features", self.num_contract_features) != self.num_contract_features:
            errors.append(
                f"checkpoint contract_features={hp.get('contract_features')} != config {self.num_contract_features}"
            )
        if dataset_fingerprint is not None:
            ckpt_fp = checkpoint.get("dataset_fingerprint", "unknown")
            if ckpt_fp != "unknown" and ckpt_fp != dataset_fingerprint:
                errors.append(
                    f"checkpoint trained on dataset {ckpt_fp}, current dataset is {dataset_fingerprint}"
                )
        return errors


# Module-level singleton — import this everywhere
RUNTIME_CONFIG = RuntimeConfig()
