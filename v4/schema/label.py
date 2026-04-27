"""Label layer schema.

Future outcomes; never joined into features except as explicit training
targets. Per Section 4.3 of the protocol, every candidate contract per bar
is labeled (single-best-contract labeling is the v3 oracle-optimism trap).

Labels are stored as a sparse map (`label_name`, `label_value`) so adding
a new oracle definition does not require a schema-version bump. Required
provenance fields are fixed.

The protocol stores per-λ utilities under a sensitivity grid:
  utility_lambda_0.0, utility_lambda_0.25, utility_lambda_0.50, utility_lambda_1.0
plus separate components: net_pnl_market_entry, net_pnl_passive_entry_if_filled,
return_on_premium, max_drawdown_pct, mfe_pct, mae_pct, time_to_mfe, time_to_mae,
fill_probability, expected_slippage, tail_loss_probability.
"""
from __future__ import annotations

import pyarrow as pa


LABEL_SCHEMA = pa.schema(
    [
        # --- identity ---
        pa.field("decision_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("contract_id", pa.string(), nullable=False),
        # --- label payload ---
        pa.field("label_name", pa.string(), nullable=False),
        pa.field("label_value", pa.float64(), nullable=True),
        pa.field("label_definition_version", pa.string(), nullable=False),
        # --- the oracle / reference exit policy this label was computed under ---
        pa.field("reference_exit_policy", pa.string(), nullable=False),
        pa.field("simulator_version", pa.string(), nullable=False),
        # --- provenance ---
        pa.field("ingest_run_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)


KNOWN_LABEL_NAMES = {
    # raw outcome components (Section 4.3)
    "net_pnl_market_entry",
    "net_pnl_passive_entry_if_filled",
    "return_on_premium",
    "max_drawdown_pct",
    "mfe_pct",
    "mae_pct",
    "time_to_mfe_seconds",
    "time_to_mae_seconds",
    "fill_probability",
    "expected_slippage",
    "tail_loss_probability",
    # utility under λ-sensitivity grid
    "utility_lambda_0.0",
    "utility_lambda_0.25",
    "utility_lambda_0.50",
    "utility_lambda_1.0",
    # opportunity-aware entries
    "best_call_utility",
    "best_put_utility",
    "best_available_utility",
    "abstain_utility",
}


def validate_label_table(table: pa.Table) -> None:
    """Verify a pyarrow Table conforms to LABEL_SCHEMA."""
    if not table.schema.equals(LABEL_SCHEMA, check_metadata=False):
        raise ValueError(
            "Label schema mismatch. See docs/DATA_CONTRACT.md.\n"
            f"Expected:\n{LABEL_SCHEMA}\nGot:\n{table.schema}"
        )

    names_in_data = set(table["label_name"].to_pylist())
    unknown = names_in_data - KNOWN_LABEL_NAMES
    if unknown:
        raise ValueError(
            f"Unknown label_name(s): {unknown}. "
            f"Add to KNOWN_LABEL_NAMES in v4/schema/label.py if intentional."
        )
