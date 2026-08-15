"""Inventory owned vendor fields for Protocol101 data-plane jitter work.

This is Step 0 of the DB-vs-TD / live-vs-historical field-delta work. It never
contacts a vendor and never reads labels, PnL, or strategy outputs. The goal is
only to state which vendor/product/session files and sample fields already
exist locally, so later jitter fitting can branch correctly.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "Protocol101VendorOverlapInventoryV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_vendor_overlap_inventory")
SESSION_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class ProductRoot:
    vendor: str
    product: str
    asset_class: str
    root: Path
    pattern: str


DEFAULT_PRODUCT_ROOTS = (
    ProductRoot("databento", "opra_spxw_definition", "option", Path("data/raw/databento/opra_spxw_definition"), "*"),
    ProductRoot("databento", "opra_spxw_cbbo_1m", "option", Path("data/raw/databento/opra_spxw_cbbo_1m"), "*"),
    ProductRoot("databento", "opra_spxw_ohlcv_1m", "option", Path("data/raw/databento/opra_spxw_ohlcv_1m"), "*"),
    ProductRoot("databento", "opra_spxw_statistics", "option", Path("data/raw/databento/opra_spxw_statistics"), "*"),
    ProductRoot("thetadata", "spx_1m", "index", Path("data/vendor/thetadata/index/spx_1m"), "*"),
    ProductRoot("thetadata", "vix_1m", "index", Path("data/vendor/thetadata/index/vix_1m"), "*"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-sample-files", type=int, default=3)
    return parser.parse_args()


def session_from_path(path: Path) -> str | None:
    match = SESSION_RE.search(str(path))
    return match.group(0) if match else None


def file_format(path: Path) -> str:
    name = path.name
    if name.endswith(".dbn.zst"):
        return "dbn.zst"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".jsonl"):
        return "jsonl"
    return path.suffix.lstrip(".") or "unknown"


def sample_columns(path: Path) -> list[str]:
    fmt = file_format(path)
    try:
        if fmt == "parquet":
            import pandas as pd

            frame = pd.read_parquet(path, columns=None)
            return [str(column) for column in frame.columns]
        if fmt == "csv":
            with path.open(newline="") as handle:
                reader = csv.reader(handle)
                return [str(item) for item in next(reader, [])]
        if fmt == "json":
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return [str(key) for key in payload.keys()]
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                return [str(key) for key in payload[0].keys()]
    except Exception as exc:  # noqa: BLE001 - inventory should report unreadable samples, not fail.
        return [f"sample_read_error:{type(exc).__name__}"]
    return []


def inventory_product(root: ProductRoot, *, max_sample_files: int) -> dict[str, Any]:
    files = [
        path
        for path in sorted(root.root.glob(root.pattern))
        if path.is_file() and session_from_path(path)
    ]
    sessions = sorted({session_from_path(path) for path in files if session_from_path(path)})
    formats: dict[str, int] = {}
    for path in files:
        fmt = file_format(path)
        formats[fmt] = formats.get(fmt, 0) + 1
    samples = []
    for path in files[: max(0, int(max_sample_files))]:
        samples.append(
            {
                "path": str(path),
                "session": session_from_path(path),
                "format": file_format(path),
                "columns": sample_columns(path),
            }
        )
    return {
        "vendor": root.vendor,
        "product": root.product,
        "asset_class": root.asset_class,
        "root": str(root.root),
        "file_count": len(files),
        "session_count": len(sessions),
        "sessions": sessions,
        "formats": dict(sorted(formats.items())),
        "sample_files": samples,
    }


def build_inventory(*, max_sample_files: int) -> dict[str, Any]:
    products = [
        inventory_product(root, max_sample_files=max_sample_files)
        for root in DEFAULT_PRODUCT_ROOTS
    ]
    option_vendors = sorted(
        {item["vendor"] for item in products if item["asset_class"] == "option" and item["session_count"] > 0}
    )
    index_vendors = sorted(
        {item["vendor"] for item in products if item["asset_class"] == "index" and item["session_count"] > 0}
    )
    theta_option_products = [
        item for item in products if item["vendor"] == "thetadata" and item["asset_class"] == "option" and item["session_count"] > 0
    ]
    conclusion = {
        "option_side_pairing_available_from_owned_databento": "databento" in option_vendors,
        "option_side_pairing_available_from_owned_thetadata": bool(theta_option_products),
        "index_side_pairing_available_from_owned_thetadata": "thetadata" in index_vendors,
        "requires_thetadata_option_quote_sample_for_td_option_diff": not bool(theta_option_products),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "evidence_grade": "data_plane_only",
        "labels_used": False,
        "pnl_used": False,
        "strategy_metrics_used": False,
        "recorder_days_used_for_selection": False,
        "products": products,
        "option_vendors": option_vendors,
        "index_vendors": index_vendors,
        "conclusion": conclusion,
    }


def write_product_csv(path: Path, products: list[dict[str, Any]]) -> None:
    fields = ("vendor", "product", "asset_class", "root", "file_count", "session_count", "formats")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in products:
            writer.writerow(
                {
                    "vendor": item["vendor"],
                    "product": item["product"],
                    "asset_class": item["asset_class"],
                    "root": item["root"],
                    "file_count": item["file_count"],
                    "session_count": item["session_count"],
                    "formats": json.dumps(item["formats"], sort_keys=True),
                }
            )


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Vendor Overlap Inventory",
        "",
        f"- Status: `{payload['status']}`",
        f"- Evidence grade: `{payload['evidence_grade']}`",
        f"- Labels used: `{str(payload['labels_used']).lower()}`",
        f"- PnL used: `{str(payload['pnl_used']).lower()}`",
        f"- Strategy metrics used: `{str(payload['strategy_metrics_used']).lower()}`",
        "",
        "## Product Coverage",
        "",
    ]
    for item in payload["products"]:
        lines.append(
            f"- `{item['vendor']}/{item['product']}` ({item['asset_class']}): "
            f"files=`{item['file_count']}`, sessions=`{item['session_count']}`, formats=`{item['formats']}`"
        )
    lines.extend(["", "## Conclusion", ""])
    for key, value in payload["conclusion"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an inventory only. It does not fit deltas, labels, PnL, or strategy performance.",
            "- If ThetaData option quote files are absent, TD option-side diffing requires a separately approved small option-quote sample.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    payload = build_inventory(max_sample_files=int(args.max_sample_files))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    csv_path = args.out_dir / "product_inventory.csv"
    report_path = args.out_dir / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_product_csv(csv_path, payload["products"])
    report_path.write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "report": str(report_path),
                "requires_thetadata_option_quote_sample_for_td_option_diff": payload["conclusion"][
                    "requires_thetadata_option_quote_sample_for_td_option_diff"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
