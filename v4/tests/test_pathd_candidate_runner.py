from __future__ import annotations

import inspect

from v4.scripts import run_pathd_candidate as runner


def test_candidate_runner_has_only_no_submit_modes() -> None:
    assert runner.MODES == (
        "databento-no-order",
        "ibkr-paper-dry-run",
        "joint-live-no-order",
    )
    forbidden = "place" + "Order"
    assert forbidden not in inspect.getsource(runner)


def test_joint_mode_connects_to_ibkr_readonly_only() -> None:
    """The joint mode adds a broker leg, so the readonly guarantee must hold.

    It reuses _ibkr_dry_run, whose only connect() call passes readonly=True.
    """

    source = inspect.getsource(runner)
    assert source.count("ib.connect(") == 1
    assert "readonly=True" in source


def test_joint_latency_ledger_sums_opra_arrival_and_broker_leg() -> None:
    """End-to-end is the worst OPRA family p99 plus the measured broker leg.

    The worst family is used rather than a mean because a decision cannot be
    emitted until its slowest required input has arrived.
    """

    summary = {
        "receipt_minus_interval_end_ns": {
            "cbbo-1m": {"p99": 319_521_000},
            "cbbo-1s": {"p99": 120_000_000},
        }
    }
    ledger = runner.joint_latency_ledger(summary, ibkr_ready_ns=400_000_000)
    assert ledger["opra_worst_p99_ns"] == 319_521_000
    assert ledger["end_to_end_ns"] == 719_521_000
    assert ledger["within_emission_lag_budget"] is True
    assert ledger["verdict"] == "END_TO_END_WITHIN_TRAINED_EMISSION_LAG"


def test_joint_latency_ledger_flags_a_path_slower_than_the_trained_lag() -> None:
    """A live path slower than L means the model decided earlier than possible.

    That is the signed18 look-ahead class. The ledger must report it rather
    than silently pass, because the repair is to raise L, never to discard the
    measurement.
    """

    summary = {"receipt_minus_interval_end_ns": {"cbbo-1m": {"p99": 2_000_000_000}}}
    ledger = runner.joint_latency_ledger(summary, ibkr_ready_ns=900_000_000)
    assert ledger["end_to_end_ms"] == 2900.0
    assert ledger["within_emission_lag_budget"] is False
    assert ledger["verdict"] == "END_TO_END_EXCEEDS_TRAINED_EMISSION_LAG"


def test_joint_latency_ledger_tolerates_a_capture_with_no_interval_clock() -> None:
    """Only CBBO-1s/1m and OHLCV-1m carry an interval-end clock.

    Other schemas intentionally return None, so an empty map must not crash the
    joint run -- it should report a zero OPRA leg and let the broker leg stand.
    """

    ledger = runner.joint_latency_ledger({}, ibkr_ready_ns=250_000_000)
    assert ledger["opra_worst_p99_ns"] == 0
    assert ledger["end_to_end_ns"] == 250_000_000


def test_frozen_emission_lag_matches_the_signed_receipt() -> None:
    assert runner.FROZEN_EMISSION_LAG_MS == 2336
