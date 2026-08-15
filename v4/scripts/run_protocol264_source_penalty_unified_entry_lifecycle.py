"""EXP_SOURCE_PENALTY_ROUTER_UNIFIED_ENTRY_LIFECYCLE_V1.

Historically Protocol264. This reuses the unified entry-plus-lifecycle sequence
trainer from Protocol209, but points it at the current Protocol261
source-penalty router entry stream.

What is this: experiment / model change.
Does it change the paper-trading default: no.
Candidate: CHALLENGER_SOURCE_PENALTY_UNIFIED_ENTRY_LIFECYCLE_V1.
Baseline: PAPER_DEFAULT_PROTOCOL101 and the frozen Protocol261 source-penalty
router stream with its original exits.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

import v4.scripts.run_protocol200_lifecycle_continuation_policy as p200
import v4.scripts.run_protocol209_unified_entry_lifecycle_sequence as p209
from v4.scripts.run_protocol251_premium_blend_slot_aware_lifecycle import (
    load_premium_blend_candidates,
    safe_find_normalized_path,
)


HISTORICAL_ID = "Protocol264"
ROLE_LABEL = "EXP_SOURCE_PENALTY_ROUTER_UNIFIED_ENTRY_LIFECYCLE_V1"
CANDIDATE_LABEL = "CHALLENGER_SOURCE_PENALTY_UNIFIED_ENTRY_LIFECYCLE_V1"
BASELINE_LABEL = "CHALLENGER_ROUTER_SOURCE_PENALTY_CALIBRATED_V1"
DEFAULT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_261_router_source_penalty_calibration/model_trades.csv")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_264_source_penalty_unified_entry_lifecycle")


def load_source_penalty_candidates(_: Sequence[Path]) -> pd.DataFrame:
    return load_premium_blend_candidates(DEFAULT_TRADES)


def main() -> int:
    p209.ROLE_LABEL = ROLE_LABEL
    p209.HISTORICAL_ID = HISTORICAL_ID
    p209.CANDIDATE_LABEL = CANDIDATE_LABEL
    p209.OTHER_BASELINE_LABEL = BASELINE_LABEL
    p209.ENTRY_STREAM_DESCRIPTION = "frozen Protocol261 source-penalty router entry stream"
    p209.LOOP_ID = "v4_aplus_hypothesis_264_source_penalty_unified_entry_lifecycle"
    p209.DEFAULT_OUT_DIR = DEFAULT_OUT_DIR
    p209.load_candidate_entries = load_source_penalty_candidates
    p200.find_normalized_path = safe_find_normalized_path
    return p209.main()


if __name__ == "__main__":
    raise SystemExit(main())
