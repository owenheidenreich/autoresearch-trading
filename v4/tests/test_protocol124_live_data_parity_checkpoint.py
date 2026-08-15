from __future__ import annotations

from v4.scripts.run_protocol124_protocol101_live_data_parity_checkpoint import build_payload, decide


def test_protocol124_ready_requires_live_spx_vix_and_spxw_nbbo() -> None:
    checks = {
        "protocol101_entry_router_wired": True,
        "spx_live_price": True,
        "vix_live_price": True,
        "spxw_live_nbbo_rows": 4,
        "ibkr_connected": True,
        "delayed_plumbing_rows": 0,
        "delayed_plumbing_parity_pass": False,
    }

    assert (
        decide(
            checks=checks,
            protocol119={"decision": "ready_for_protocol101_no_order_live_capture"},
            delayed_capture={},
        )
        == "ready_for_protocol101_no_order_live_capture"
    )


def test_protocol124_delayed_plumbing_does_not_clear_live_gate() -> None:
    payload = build_payload(
        ibkr={
            "ibkr_connected": True,
            "blocked_reason": "missing_live_market_data_entitlements",
            "feed_status": {
                "spx": {"live_price_available": False},
                "vix": {"live_price_available": False},
                "spxw_options": {"live_nbbo_rows": 0},
            },
        },
        protocol119={"decision": "blocked_missing_live_market_data_entitlements", "feature_dependency_audit": {"status": "pass"}},
        protocol121={"decision": "pass_protocol101_entry_router_edge_wired"},
        delayed_capture={
            "decision": "blocked",
            "blocked_reason": "delayed_market_data_not_promotion_grade",
            "captured_rows": 24,
            "shadow_parity": {"status": "pass"},
            "feed_probe": {"spx_market_data_type": "delayed", "observed_option_market_data_types": {"delayed": 24}},
        },
    )

    assert payload["decision"] == "blocked_live_subscriptions_delayed_plumbing_passed"
    assert payload["checks"]["delayed_plumbing_parity_pass"] is True
