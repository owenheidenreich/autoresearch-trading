"""Audit SPX/VIX context provenance for v4 processed-data promotion gates.

This is intentionally read-only. It distinguishes derived/proxy context from
official or live-equivalent index bars before a model report is treated as
promotion-grade evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spx-dir", type=Path, default=Path("data/raw/index/spx_1m"))
    parser.add_argument("--vix-dir", type=Path, default=Path("data/raw/index/vix_1m"))
    parser.add_argument("--out", type=Path, default=Path("v4/audit/context_provenance/report.json"))
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.DataFrame()


def _summarize(directory: Path) -> dict:
    files = sorted(
        path
        for path in directory.glob("*")
        if path.is_file() and path.suffix.lower() in {".parquet", ".csv", ".txt", ".jsonl"}
    )
    source_counts: dict[str, int] = {}
    rows = 0
    official_rows = 0
    derived_rows = 0
    proxy_rows = 0
    sessions: set[str] = set()
    for path in files:
        try:
            frame = _read(path)
        except Exception:
            source_counts["unreadable"] = source_counts.get("unreadable", 0) + 1
            continue
        if frame.empty:
            source_counts["empty"] = source_counts.get("empty", 0) + 1
            continue
        rows += len(frame)
        if "event_time" in frame:
            dates = pd.to_datetime(frame["event_time"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
            sessions.update(dates.dropna().unique().tolist())
        else:
            sessions.add(path.name[:10])
        for column in ("context_source", "proxy_source", "source"):
            if column in frame:
                for value, count in frame[column].dropna().astype(str).value_counts().items():
                    source_counts[value] = source_counts.get(value, 0) + int(count)
        if "is_official_index_data" in frame:
            official_rows += int(pd.Series(frame["is_official_index_data"]).fillna(False).astype(bool).sum())
        if "is_derived" in frame:
            derived_rows += int(pd.Series(frame["is_derived"]).fillna(False).astype(bool).sum())
        if "is_proxy" in frame:
            proxy_rows += int(pd.Series(frame["is_proxy"]).fillna(False).astype(bool).sum())
        if not any(col in frame for col in ("context_source", "proxy_source", "source", "is_official_index_data", "is_derived", "is_proxy")):
            source_counts["unknown_or_unstamped"] = source_counts.get("unknown_or_unstamped", 0) + len(frame)
    return {
        "directory": str(directory),
        "files": len(files),
        "rows": int(rows),
        "sessions": len(sessions),
        "first_session": min(sessions) if sessions else None,
        "last_session": max(sessions) if sessions else None,
        "official_rows": int(official_rows),
        "derived_rows": int(derived_rows),
        "proxy_rows": int(proxy_rows),
        "official_fraction": float(official_rows / rows) if rows else 0.0,
        "source_counts": dict(sorted(source_counts.items())),
    }


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Context Provenance Audit",
        "",
        f"Promotion grade: `{payload['promotion_grade']}`",
        "",
        "| Context | Files | Sessions | Rows | Official Fraction | First | Last |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name in ("spx", "vix"):
        row = payload[name]
        lines.append(
            f"| {name.upper()} | {row['files']} | {row['sessions']} | {row['rows']} | "
            f"{row['official_fraction']:.3f} | {row['first_session']} | {row['last_session']} |"
        )
    lines += ["", "## Sources", ""]
    for name in ("spx", "vix"):
        lines.append(f"### {name.upper()}")
        for source, count in payload[name]["source_counts"].items():
            lines.append(f"- `{source}`: {count}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    payload = {
        "spx": _summarize(args.spx_dir),
        "vix": _summarize(args.vix_dir),
    }
    payload["promotion_grade"] = (
        payload["spx"]["rows"] > 0
        and payload["vix"]["rows"] > 0
        and payload["spx"]["official_fraction"] >= 0.99
        and payload["vix"]["official_fraction"] >= 0.99
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    md_path = args.out.with_suffix(".md")
    _write_markdown(md_path, payload)
    print(args.out)
    print(md_path)
    print(json.dumps({"promotion_grade": payload["promotion_grade"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
