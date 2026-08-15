"""Create commission-free processed decision rows from existing v4 pickles.

Older v4 datasets subtracted ``2 * fee_per_contract`` from every one-contract
round-trip label. The current primary research labels exclude broker
commission, so the conversion is a deterministic addition to finite label
arrays rather than a full OPRA rebuild.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--removed-roundtrip-fee", type=float, default=2.0)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def _adjust_array(value: object, amount: float) -> object:
    arr = np.asarray(value, dtype=float).copy()
    mask = np.isfinite(arr)
    arr[mask] += amount
    return arr


def _convert_file(path: Path, out_path: Path, amount: float) -> dict:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    finite_net = 0
    finite_mid = 0
    for row in rows:
        if "labels_net_pnl" in row:
            row["labels_net_pnl"] = _adjust_array(row["labels_net_pnl"], amount)
            finite_net += int(np.isfinite(row["labels_net_pnl"]).sum())
        if "labels_mid_pnl" in row:
            row["labels_mid_pnl"] = _adjust_array(row["labels_mid_pnl"], amount)
            finite_mid += int(np.isfinite(row["labels_mid_pnl"]).sum())
        row["label_commission_excluded"] = True
        row["removed_roundtrip_fee"] = float(amount)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "session": path.stem,
        "rows": len(rows),
        "finite_net_labels": finite_net,
        "finite_mid_labels": finite_mid,
        "input_path": str(path),
        "output_path": str(out_path),
    }


def main() -> int:
    args = parse_args()
    if args.clear_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    records = []
    for path in sorted(args.input_dir.glob("*.pkl")):
        out_path = args.output_dir / path.name
        record = _convert_file(path, out_path, args.removed_roundtrip_fee)
        records.append(record)
        print(json.dumps({"session": record["session"], "rows": record["rows"]}), flush=True)
    payload = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "removed_roundtrip_fee": args.removed_roundtrip_fee,
        "sessions": len(records),
        "total_rows": sum(record["rows"] for record in records),
        "total_finite_net_labels": sum(record["finite_net_labels"] for record in records),
        "total_finite_mid_labels": sum(record["finite_mid_labels"] for record in records),
        "records": records,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.summary_out)
    print(json.dumps({k: payload[k] for k in ("sessions", "total_rows", "total_finite_net_labels")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
