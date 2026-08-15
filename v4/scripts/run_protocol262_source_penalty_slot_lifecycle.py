"""EXP_SOURCE_PENALTY_ROUTER_SLOT_LIFECYCLE_V1.

Historically Protocol262. Protocol261 is a strong research challenger versus
PAPER_DEFAULT_PROTOCOL101, but it still relies on fixed candidate exits. This
experiment applies the slot-aware lifecycle hold/exit learner to the Protocol261
entry stream.

Scope:
* entry stream stays frozen to CHALLENGER_ROUTER_SOURCE_PENALTY_CALIBRATED_V1
* the lifecycle model only changes post-entry hold/exit timing
* no fixed hold time, percentage target, or percentage stop is introduced
* evaluation remains strict one-account serial replay

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import v4.scripts.run_protocol251_premium_blend_slot_aware_lifecycle as p251


def main() -> int:
    p251.ROLE_LABEL = "EXP_SOURCE_PENALTY_ROUTER_SLOT_LIFECYCLE_V1"
    p251.HISTORICAL_ID = "Protocol262"
    p251.CANDIDATE_LABEL = "CHALLENGER_SOURCE_PENALTY_ROUTER_SLOT_LIFECYCLE_V1"
    p251.BASE_CHALLENGER_LABEL = "CHALLENGER_ROUTER_SOURCE_PENALTY_CALIBRATED_V1"
    p251.DEFAULT_TRADES = p251.Path(
        "v4/audit/autoresearch/v4_aplus_hypothesis_261_router_source_penalty_calibration/model_trades.csv"
    )
    p251.DEFAULT_OUT_DIR = p251.Path("v4/audit/autoresearch/v4_aplus_hypothesis_262_source_penalty_slot_lifecycle")
    return p251.main()


if __name__ == "__main__":
    raise SystemExit(main())
