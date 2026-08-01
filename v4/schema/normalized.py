"""Normalized layer schema.

Unified contract symbology, timestamps, NBBO, trades, definitions, and
statistics/OI. The Normalized layer is a deterministic function of Raw.
Re-running ingest produces byte-identical Normalized output.

Greeks/IV columns are nullable here: vendors that supply IV/Greeks (e.g.,
OptionsDX) populate them on ingest; vendors that don't (e.g., Databento)
leave them null and the v4 greeks module fills them later. `iv_source` /
`greek_source` discriminate.
"""
from __future__ import annotations

import pyarrow as pa

from .types import SCHEMA_VERSION, GreekSource, OptionRight, VendorSource


NORMALIZED_SCHEMA = pa.schema(
    [
        # --- time fields ---
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("receive_time", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("timestamp_source", pa.string(), nullable=False),
        # --- contract identity ---
        pa.field("contract_id", pa.string(), nullable=False),
        pa.field("raw_symbol", pa.string(), nullable=True),
        pa.field("instrument_id", pa.int64(), nullable=True),
        pa.field("root", pa.string(), nullable=False),
        pa.field("expiry", pa.date32(), nullable=False),
        pa.field("strike", pa.decimal128(18, 6), nullable=False),
        pa.field("right", pa.string(), nullable=False),
        pa.field("settlement_style", pa.string(), nullable=True),
        pa.field("settlement_time_utc", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("min_price_increment", pa.float64(), nullable=True),
        pa.field("contract_multiplier", pa.int64(), nullable=True),
        # --- market data ---
        pa.field("bid", pa.float64(), nullable=True),
        pa.field("ask", pa.float64(), nullable=True),
        pa.field("bid_size", pa.int64(), nullable=True),
        pa.field("ask_size", pa.int64(), nullable=True),
        pa.field("mid", pa.float64(), nullable=True),
        pa.field("last_trade", pa.float64(), nullable=True),
        pa.field("last_trade_size", pa.int64(), nullable=True),
        pa.field("quote_time", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("last_trade_time", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("quote_age_ms", pa.int64(), nullable=True),
        pa.field("quote_gap_seconds", pa.float64(), nullable=True),
        pa.field("open_interest", pa.int64(), nullable=True),
        pa.field("open_interest_asof_date", pa.date32(), nullable=True),
        pa.field("stat_open_interest", pa.int64(), nullable=True),
        pa.field("volume", pa.int64(), nullable=True),
        pa.field("volume_asof_time", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("option_ohlcv_volume", pa.int64(), nullable=True),
        # --- underlying ---
        pa.field("underlying_price", pa.float64(), nullable=True),
        # --- Greeks / IV (nullable; iv_source/greek_source discriminate) ---
        pa.field("iv", pa.float64(), nullable=True),
        pa.field("iv_source", pa.string(), nullable=True),
        pa.field("delta", pa.float64(), nullable=True),
        pa.field("gamma", pa.float64(), nullable=True),
        pa.field("theta", pa.float64(), nullable=True),
        pa.field("vega", pa.float64(), nullable=True),
        pa.field("rho", pa.float64(), nullable=True),
        pa.field("charm", pa.float64(), nullable=True),
        pa.field("vanna", pa.float64(), nullable=True),
        pa.field("vomma", pa.float64(), nullable=True),
        pa.field("greek_source", pa.string(), nullable=True),
        pa.field("greek_computation_ts", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("risk_free_rate_used", pa.float64(), nullable=True),
        pa.field("dividend_yield_used", pa.float64(), nullable=True),
        # --- provenance ---
        pa.field("vendor_source", pa.string(), nullable=False),
        pa.field("ingest_run_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)


REQUIRED_FIELDS_NORMALIZED = [f.name for f in NORMALIZED_SCHEMA if not f.nullable]


def validate_normalized_table(table: pa.Table) -> None:
    """Verify a pyarrow Table conforms to NORMALIZED_SCHEMA."""
    if not table.schema.equals(NORMALIZED_SCHEMA, check_metadata=False):
        raise ValueError(
            "Normalized schema mismatch. See docs/DATA_CONTRACT.md.\n"
            f"Expected:\n{NORMALIZED_SCHEMA}\nGot:\n{table.schema}"
        )

    valid_rights = {r.value for r in OptionRight}
    rights_in_data = set(table["right"].to_pylist())
    bad_rights = rights_in_data - valid_rights
    if bad_rights:
        raise ValueError(f"Invalid right(s) in normalized data: {bad_rights}")

    valid_settlement_styles = {"AM", "PM"}
    settlement_styles = {
        s for s in table["settlement_style"].to_pylist() if s is not None
    }
    bad_settlement_styles = settlement_styles - valid_settlement_styles
    if bad_settlement_styles:
        raise ValueError(
            f"Invalid settlement_style(s) in normalized data: {bad_settlement_styles}"
        )

    for i, (root, settlement_style) in enumerate(
        zip(table["root"].to_pylist(), table["settlement_style"].to_pylist(), strict=True)
    ):
        if root == "SPXW" and settlement_style not in (None, "PM"):
            raise ValueError(
                f"SPXW rows must be PM-settled; row {i} has {settlement_style!r}"
            )
        if root == "SPX" and settlement_style not in (None, "AM"):
            raise ValueError(
                f"SPX rows must be AM-settled; row {i} has {settlement_style!r}"
            )

    valid_vendors = set(VendorSource.values())
    bad_vendors = set(table["vendor_source"].to_pylist()) - valid_vendors
    if bad_vendors:
        raise ValueError(f"Unknown vendor_source(s): {bad_vendors}")

    iv_sources_seen = {s for s in table["iv_source"].to_pylist() if s is not None}
    valid_iv_sources = set(GreekSource.values())
    bad_iv = iv_sources_seen - valid_iv_sources
    if bad_iv:
        raise ValueError(f"Unknown iv_source(s): {bad_iv}")
