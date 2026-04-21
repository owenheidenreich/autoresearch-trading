from __future__ import annotations

import argparse
import time

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    build_export_rows_for_day,
    build_folds_for_dataset,
    build_labeled_day,
    export_metadata,
    finalize_export_dataframe,
    fold_test_day_map,
    save_pickle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the v3 Layer-2 supervised dataset.")
    p.add_argument("--output", default=DEFAULT_DATASET_PATH, help="Output pickle path.")
    p.add_argument("--equity", type=float, default=25_000.0, help="Research account equity.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    folds = build_folds_for_dataset(ds)
    test_fold_map = fold_test_day_map(folds)
    all_days = sorted(set(ds.dates))

    rows = []
    kept_days = 0
    for i, day in enumerate(all_days):
        log, _sidecar = build_labeled_day(ds, day, cfg, equity=args.equity)
        if log is None:
            continue
        kept_days += 1
        rows.extend(build_export_rows_for_day(ds, day, log, test_fold_map.get(day, -1), sidecar=_sidecar))
        if (i + 1) % 100 == 0:
            print(f"  exported {i+1}/{len(all_days)} days ({len(rows):,} rows)")

    df = finalize_export_dataframe(rows)
    meta = export_metadata(ds, df, folds)
    save_pickle(args.output, {"rows": df, "meta": meta})

    print(f"Wrote {args.output}")
    print(f"Rows: {len(df):,} across {kept_days} days")
    print(f"Model features: {len(meta['feature_names'])}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
