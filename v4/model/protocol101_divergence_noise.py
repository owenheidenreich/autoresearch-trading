"""Measured cross-feed divergence noise for Protocol101 canonical features.

The canonical v1 training design requires models to survive the feature drift
that was actually measured between Databento/ThetaData-style history and IBKR
recorder replays. This module is intentionally small and data-driven: it loads
the signed residuals from the L0/L2 design audit and can perturb candidate
feature frames by feature and moneyness band.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DIVERGENCE_DISTRIBUTIONS = Path(
    "v4/audit/autoresearch/protocol101_canonical_v1_l0_l2_design_audit_attempt001/"
    "divergence_distributions.parquet"
)
ZERO_DIVERGENCE_FAMILIES = frozenset({"A", "B"})
MONEYNESS_BANDS = ("atm", "near", "wing")


def moneyness_band(abs_offset: float | int | None) -> str:
    """Return the project-standard broad moneyness band for a ladder offset."""
    try:
        value = abs(float(abs_offset))
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(value):
        return "unknown"
    if value <= 10.0:
        return "atm"
    if value <= 25.0:
        return "near"
    return "wing"


def _signed_raw_drift(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["ibkr_value"], errors="coerce") - pd.to_numeric(
        frame["historical_value"], errors="coerce"
    )


def _band_map_from_distribution(frame: pd.DataFrame) -> dict[str, str]:
    offset_rows = frame[frame["feature"] == "B.ladder.abs_offset"]
    out: dict[str, str] = {}
    for row in offset_rows.itertuples(index=False):
        out[str(row.pair_key)] = moneyness_band(row.historical_value)
    return out


@dataclass(frozen=True)
class DivergenceNoiseModel:
    """Signed residual sampler keyed by canonical feature and moneyness band."""

    samples: dict[tuple[str, str], np.ndarray]
    feature_family: dict[str, str]
    source_path: str

    @classmethod
    def from_parquet(cls, path: Path | str = DEFAULT_DIVERGENCE_DISTRIBUTIONS) -> "DivergenceNoiseModel":
        source = Path(path)
        frame = pd.read_parquet(source)
        required = {"feature", "family", "historical_value", "ibkr_value", "pair_key"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing divergence columns: {sorted(missing)}")
        work = frame.copy()
        if "moneyness_band" not in work.columns:
            band_by_pair = _band_map_from_distribution(work)
            work["moneyness_band"] = work["pair_key"].map(band_by_pair).fillna("unknown")
        work["signed_raw_drift"] = _signed_raw_drift(work)
        work = work[np.isfinite(work["signed_raw_drift"])]
        feature_family = {
            str(row.feature): str(row.family)
            for row in work[["feature", "family"]].drop_duplicates().itertuples(index=False)
        }
        samples: dict[tuple[str, str], np.ndarray] = {}
        for (feature, band), group in work.groupby(["feature", "moneyness_band"], sort=True):
            arr = group["signed_raw_drift"].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            samples[(str(feature), str(band))] = arr
        return cls(samples=samples, feature_family=feature_family, source_path=str(source))

    def feature_names(self) -> list[str]:
        return sorted(self.feature_family)

    def feature_has_nonzero_noise(self, feature: str, band: str | None = None) -> bool:
        family = self.feature_family.get(str(feature), "")
        if family in ZERO_DIVERGENCE_FAMILIES:
            return False
        bands: Iterable[str]
        if band is None:
            bands = MONEYNESS_BANDS
        else:
            bands = (str(band),)
        for item in bands:
            values = self.samples.get((str(feature), item))
            if values is not None and len(values) and np.nanmax(np.abs(values)) > 0.0:
                return True
        return False

    def sample_noise(
        self,
        *,
        feature: str,
        bands: Iterable[str],
        rng: np.random.Generator,
        scale: float = 1.0,
    ) -> np.ndarray:
        values: list[float] = []
        for band in bands:
            samples = self.samples.get((str(feature), str(band)))
            if samples is None or len(samples) == 0:
                samples = self.samples.get((str(feature), "unknown"))
            if samples is None or len(samples) == 0:
                values.append(0.0)
            else:
                values.append(float(rng.choice(samples)) * float(scale))
        return np.asarray(values, dtype=float)

    def inject_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: Iterable[str] | None = None,
        seed: int = 0,
        scale: float = 1.0,
        band_column: str = "moneyness_band",
        abs_offset_column: str = "B.ladder.abs_offset",
    ) -> pd.DataFrame:
        """Return a copy with measured residual noise added to feature columns."""
        out = frame.copy()
        if band_column in out.columns:
            bands = out[band_column].astype(str).tolist()
        elif abs_offset_column in out.columns:
            bands = [moneyness_band(value) for value in out[abs_offset_column]]
            out[band_column] = bands
        elif "abs_offset" in out.columns:
            bands = [moneyness_band(value) for value in out["abs_offset"]]
            out[band_column] = bands
        else:
            bands = ["unknown"] * len(out)
            out[band_column] = bands

        columns = list(feature_columns) if feature_columns is not None else [
            name for name in self.feature_names() if name in out.columns
        ]
        rng = np.random.default_rng(seed)
        for feature in columns:
            if feature not in out.columns:
                continue
            if not self.feature_has_nonzero_noise(feature):
                continue
            numeric = pd.to_numeric(out[feature], errors="coerce").to_numpy(dtype=float)
            noise = self.sample_noise(feature=feature, bands=bands, rng=rng, scale=scale)
            out[feature] = numeric + noise
        return out

    def summarize(self) -> pd.DataFrame:
        rows: list[dict[str, float | str | int | bool]] = []
        for (feature, band), values in sorted(self.samples.items()):
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "feature": feature,
                    "family": self.feature_family.get(feature, ""),
                    "moneyness_band": band,
                    "sample_count": int(len(finite)),
                    "mean": float(np.mean(finite)) if len(finite) else 0.0,
                    "p50_abs": float(np.percentile(np.abs(finite), 50)) if len(finite) else 0.0,
                    "p95_abs": float(np.percentile(np.abs(finite), 95)) if len(finite) else 0.0,
                    "p99_abs": float(np.percentile(np.abs(finite), 99)) if len(finite) else 0.0,
                    "nonzero_noise_enabled": self.feature_has_nonzero_noise(feature, band),
                }
            )
        return pd.DataFrame.from_records(rows)
