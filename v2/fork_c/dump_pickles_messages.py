"""Fork C Phase 1 — dump all Pickles-authored messages per day as a single
markdown file the model can read end-to-end for curation.

Reuses the parser's Pickles-only detection (two-phase author + quote-strip)
so the dump reflects exactly what the shortlist was built from.

Output: one markdown file with per-day sections, PT timestamp + ET
conversion, full Pickles content verbatim, line numbers for journal_ref.
Zero-hit days still appear in the output — the curator's worklist is all
167 dates, not just the keyword-hit subset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

from v2.fork_c.parse_tier1_labels import (
    parse_journal,
    pt_to_et,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal-dir", required=True)
    ap.add_argument("--out", default="v2/fork_c/pickles_messages_by_day.md")
    args = ap.parse_args()

    jdir = Path(args.journal_dir)
    files = sorted(jdir.glob("*.txt"))
    out_lines: list[str] = []
    out_lines.append(f"# Pickles messages per trading day ({len(files)} days)\n")
    out_lines.append(
        "Format: each day section shows Pickles-authored messages only "
        "(reply-quote context stripped). `PT` timestamps are from the "
        "Discord export; `ET` is +3h conversion for the `first_qualifying_time_et` "
        "column. Inline ET references in message text are NOT reconverted.\n\n"
    )

    total_msgs = 0
    for p in files:
        date_str, messages = parse_journal(p)
        out_lines.append(f"## {date_str}  ({p.name})\n")
        if not messages:
            out_lines.append("_No Pickles-authored messages in this file._\n\n")
            continue
        for m in messages:
            ts_et = pt_to_et(m["ts_pt"], date_str)
            line_range = (
                f"{m['line_start']}" if m["line_start"] == m["line_end"]
                else f"{m['line_start']}-{m['line_end']}"
            )
            out_lines.append(
                f"- **PT `{m['ts_pt']}` / ET `{ts_et}`** "
                f"(`{p.name}:{line_range}`)\n"
            )
            # Indent content with > for readability
            for cl in m["content"].split("\n"):
                if cl.strip():
                    out_lines.append(f"  > {cl}\n")
            out_lines.append("\n")
            total_msgs += 1
        out_lines.append("\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    print(f"Wrote {args.out} — {total_msgs} messages across {len(files)} days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
