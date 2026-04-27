"""FeatureRow builder + forward-fill with age tracking.

Per the Data Contract Section 2.6, every Feature row must carry causality
metadata: is_live_reproducible, is_label, feature_age_seconds, feature_source,
feature_refresh_interval, feature_is_estimated, feature_is_revised. Building
a Feature row without those fields silently is exactly what produced v3's
2026-04-25 label-leakage incident.

The builder enforces those fields at construction time. The ForwardFiller
tracks feature_age_seconds when a feature value is carried across decision_time
boundaries (e.g., a 10-minute OptionsDepth snapshot used in a 1-minute
decision loop).

Note: directory naming — `feature/` is the *data* directory (parquet output);
this engineering code lives in `feature_eng/` so the data layer can be
gitignored cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import pyarrow as pa

from v4.schema.feature import FEATURE_SCHEMA, validate_feature_table
from v4.schema.types import SCHEMA_VERSION


@dataclass(frozen=True)
class FeatureRow:
    """One Feature-layer row. All causality metadata required at construction."""

    decision_time: datetime
    contract_id: str
    feature_name: str
    feature_value: float | None
    is_live_reproducible: bool
    is_label: bool
    feature_age_seconds: float
    feature_source: str
    feature_refresh_interval: float | None
    feature_is_estimated: bool
    feature_is_revised: bool
    ingest_run_id: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.is_live_reproducible:
            raise ValueError(
                "FeatureRow requires is_live_reproducible=True. "
                "Non-reproducible rows belong in the audit or label layer."
            )
        if self.is_label:
            raise ValueError(
                "FeatureRow rejects is_label=True. Label rows belong in the label layer."
            )
        if self.feature_age_seconds < 0:
            raise ValueError(
                f"feature_age_seconds must be non-negative; got {self.feature_age_seconds}"
            )


def feature_rows_to_table(rows: Iterable[FeatureRow]) -> pa.Table:
    """Build a Feature pyarrow Table from FeatureRow instances.

    Raises if any row violates the contract — same checks as
    validate_feature_table, but at the unit level for clearer errors.
    """
    rows = list(rows)
    if not rows:
        return pa.Table.from_pylist([], schema=FEATURE_SCHEMA)

    records = [
        {
            "decision_time": r.decision_time,
            "contract_id": r.contract_id,
            "feature_name": r.feature_name,
            "feature_value": r.feature_value,
            "is_live_reproducible": r.is_live_reproducible,
            "is_label": r.is_label,
            "feature_age_seconds": r.feature_age_seconds,
            "feature_source": r.feature_source,
            "feature_refresh_interval": r.feature_refresh_interval,
            "feature_is_estimated": r.feature_is_estimated,
            "feature_is_revised": r.feature_is_revised,
            "ingest_run_id": r.ingest_run_id,
            "schema_version": r.schema_version,
        }
        for r in rows
    ]
    table = pa.Table.from_pylist(records, schema=FEATURE_SCHEMA)
    validate_feature_table(table)
    return table


@dataclass
class ForwardFiller:
    """Carry a feature value forward with age tracking.

    Use case: OptionsDepth dealer-flow snapshots refresh every 10 minutes,
    but the decision loop runs every 1 minute. Without explicit age tracking
    the model would silently consume a 9-minute-stale snapshot as if it were
    fresh — exactly what the Data Contract Section 3 forbids.

    Usage:
        ff = ForwardFiller(refresh_interval=600.0, source="optionsdepth_pro_max")
        ff.observe(timestamp=t1, value=v1)  # vendor refresh
        row_at_t2 = ff.row_at(t2, contract_id="...", feature_name="net_gex", ...)
        # row_at_t2.feature_age_seconds = (t2 - t1).total_seconds()

    The filler does NOT extrapolate; it carries the last observed value with
    an explicit age. The model decides whether the age is acceptable.
    """

    refresh_interval: float
    source: str
    is_estimated: bool = False
    last_value: float | None = None
    last_observed_at: datetime | None = None

    def observe(self, *, timestamp: datetime, value: float | None) -> None:
        self.last_value = value
        self.last_observed_at = timestamp

    def value_at(self, t: datetime) -> tuple[float | None, float]:
        """Return (carried_value, age_seconds). Raises if observe() not called."""
        if self.last_observed_at is None:
            raise RuntimeError(
                "ForwardFiller has no observation; call observe() before value_at()"
            )
        age = (t - self.last_observed_at).total_seconds()
        if age < 0:
            raise ValueError(
                f"value_at({t}) is before last_observed_at ({self.last_observed_at})"
            )
        return self.last_value, age

    def row_at(
        self,
        t: datetime,
        *,
        contract_id: str,
        feature_name: str,
        ingest_run_id: str,
        is_live_reproducible: bool = True,
    ) -> FeatureRow:
        value, age = self.value_at(t)
        return FeatureRow(
            decision_time=t,
            contract_id=contract_id,
            feature_name=feature_name,
            feature_value=value,
            is_live_reproducible=is_live_reproducible,
            is_label=False,
            feature_age_seconds=age,
            feature_source=self.source,
            feature_refresh_interval=self.refresh_interval,
            feature_is_estimated=self.is_estimated,
            feature_is_revised=False,
            ingest_run_id=ingest_run_id,
        )
