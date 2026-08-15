"""Build derived SPX/volatility context bars from normalized SPXW quotes."""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from v4.ingest.derived_context import write_derived_context


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument(
        "--normalized-in",
        type=Path,
        default=None,
        help="Defaults to v4/normalized/databento_spxw_0dte_<session>.parquet",
    )
    p.add_argument("--spx-out", type=Path, default=None)
    p.add_argument("--vol-out", type=Path, default=None)
    p.add_argument("--normalized-out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    session = args.session
    normalized_in = args.normalized_in or Path(
        f"v4/normalized/databento_spxw_0dte_{session}.parquet"
    )
    spx_out = args.spx_out or Path(f"data/raw/index/spx_1m/{session}.derived_spxw_parity.parquet")
    vol_out = args.vol_out or Path(f"data/raw/index/vix_1m/{session}.derived_spxw_atm_iv.parquet")
    normalized_out = args.normalized_out or Path(
        f"v4/normalized/databento_spxw_0dte_{session}_derived_context.parquet"
    )
    normalized = pq.read_table(normalized_in)
    result = write_derived_context(
        normalized,
        session=session,
        spx_path=spx_out,
        vol_path=vol_out,
        normalized_out=normalized_out,
    )
    print(f"spx_rows={len(result.spx_bars)} path={spx_out}")
    print(f"vol_rows={len(result.vol_bars)} path={vol_out}")
    print(f"normalized_rows={result.normalized.num_rows} path={normalized_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
