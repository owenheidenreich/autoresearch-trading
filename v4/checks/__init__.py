"""v4.checks — sanity + integrity checks on the Normalized layer.

CheckResult is the unit of output. The PIPELINE_INTEGRITY_REPORT.md
generator (audit/) consumes a list of these.
"""
from .integrity import (
    check_no_duplicate_keys,
    check_required_non_null,
    check_timestamp_monotonic_per_contract,
    run_all_integrity_checks,
)
from .paid_data_guard import (
    add_paid_data_approval_args,
    exact_approval_text,
    require_paid_data_approval,
)
from .sanity import (
    CheckResult,
    check_ask_positive,
    check_bid_ask_ordering,
    check_bid_nonnegative,
    run_all_sanity_checks,
)

__all__ = [
    "CheckResult",
    "check_ask_positive",
    "check_bid_ask_ordering",
    "check_bid_nonnegative",
    "check_no_duplicate_keys",
    "check_required_non_null",
    "check_timestamp_monotonic_per_contract",
    "add_paid_data_approval_args",
    "exact_approval_text",
    "require_paid_data_approval",
    "run_all_integrity_checks",
    "run_all_sanity_checks",
]
