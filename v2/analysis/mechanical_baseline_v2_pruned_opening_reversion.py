"""V2-pruned — V2 learned scorer with `vwap_reclaim_state` dropped.

One disciplined simplification pass (user-requested, 2026-04-19) after the
V2 stress test revealed that ablating `vwap_reclaim_state` improves the
learned model by +0.30pp gap vs Control A. This script reruns V2 with the
feature actually removed from the input set (not just zeroed), producing a
clean 12-feature + side-indicator model artifact for comparison.

Everything else is identical to V2:
- V1B gates (same 3x3 grid selection per fold)
- Candidate universe unchanged
- Contract selection unchanged
- Exit engine unchanged
- Controls unchanged (A same-day random-bar, B same-bar random-day)
- RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=20)

Implementation note: this module patches V2's module-level constants
(`CORE_FEATURE_NAMES`, `N_MODEL_FEATURES`, `SIDE_INDICATOR_IDX`,
`EXPERIMENT_ID`, `OUT_DIR_DEFAULT`) before dispatching to V2's `main()`.
Python looks up those names at call time, so V2's internals use the
pruned set transparently. No changes to V2 required.
"""
from __future__ import annotations

import sys

from v2.analysis import mechanical_baseline_v2_learned_scorer as v2


DROPPED_FEATURES = ("vwap_reclaim_state",)
PRUNED_NAMES = tuple(n for n in v2.CORE_FEATURE_NAMES if n not in DROPPED_FEATURES)
EXPERIMENT_ID = "mechbase_opening_reversion_v2_pruned"
OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned"


def main() -> int:
    v2.CORE_FEATURE_NAMES = PRUNED_NAMES
    v2.N_MODEL_FEATURES = len(PRUNED_NAMES) + 1
    v2.SIDE_INDICATOR_IDX = len(PRUNED_NAMES)
    v2.EXPERIMENT_ID = EXPERIMENT_ID
    v2.OUT_DIR_DEFAULT = OUT_DIR_DEFAULT
    print(f"[V2-pruned] Dropping features: {list(DROPPED_FEATURES)}")
    print(f"[V2-pruned] Active CORE_FEATURE_NAMES: {list(PRUNED_NAMES)}")
    print(f"[V2-pruned] Model feature dimension: {v2.N_MODEL_FEATURES}")
    return v2.main()


if __name__ == "__main__":
    sys.exit(main())
