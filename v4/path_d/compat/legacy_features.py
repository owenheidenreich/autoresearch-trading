"""Adapt legacy DataFrames to the source-neutral builder without moving code."""
from __future__ import annotations

import pandas as pd

from v4.path_d.features.source_neutral import IndexObservation, SourceNeutralFeatureBuilder, feature_hash


class LegacyHistoricalFeatureShim:
    """Compatibility adapter used to prove and monitor legacy feature parity."""

    def __init__(self, builder: SourceNeutralFeatureBuilder | None = None) -> None:
        self.builder = builder or SourceNeutralFeatureBuilder()

    def market_features(
        self,
        spx_bars: pd.DataFrame,
        vix_bars: pd.DataFrame,
        decision_time: pd.Timestamp,
    ):
        return self.builder.market_features(
            _observations(spx_bars),
            _observations(vix_bars),
            _utc_iso(decision_time),
        )

    def market_window(
        self,
        spx_bars: pd.DataFrame,
        vix_bars: pd.DataFrame,
        decision_time: pd.Timestamp,
        market_window_minutes: int,
    ):
        start = decision_time - pd.Timedelta(minutes=market_window_minutes - 1)
        rows = [self.market_features(spx_bars, vix_bars, ts) for ts in pd.date_range(start=start, end=decision_time, freq="min", tz="UTC")]
        import numpy as np
        return np.vstack(rows)

    def feature_hash(self, values) -> str:
        return feature_hash(values)

    def option_features(self, row: pd.Series, *, atm_strike: int):
        return self.builder.legacy_option_features(
            row.to_dict(),
            underlying_price=float(row["underlying_price"]),
            strike=float(row["strike"]),
            right=str(row["right"]),
            atm_strike=int(atm_strike),
        )


def _observations(frame: pd.DataFrame) -> tuple[IndexObservation, ...]:
    if frame is None or frame.empty:
        return ()
    return tuple(
        IndexObservation(
            received_timestamp_utc=_utc_iso(pd.Timestamp(row.event_time)),
            close=float(row.close),
            volume=float(getattr(row, "volume", 0.0)),
        )
        for row in frame.itertuples(index=False)
    )


def _utc_iso(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")
