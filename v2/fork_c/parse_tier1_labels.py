"""Fork C Phase 1 — keyword-shortlist parser for SPX 0DTE long mentions.

**Scope and discipline.** This tool is a *candidate emitter*, not a labeler.
It walks the 167 raw-Discord-export journal files, identifies Pickles-authored
messages, runs a conservative regex shortlist for SPX 0DTE long / short
patterns, and emits two CSVs for the human curator to work from. It does
NOT assign labels. The human applies ``v2/fork_c/label_rules.md`` to the
full day's content to produce ``tier1_labels.csv`` and
``tier1_labels_shadow.csv``.

Guardrails (explicit):
1. Only emit hits from messages whose author is "Pickles" per the Discord
   export's name-line convention. Third-party commentary (e.g. JPM roll
   notes, replies by other users) is excluded.
2. Deduplicate repeated message blocks by content hash. Discord exports
   repeat earlier messages as quote context when someone replies; those
   repeats should not double-count.
3. No ``label`` / ``confidence`` columns in the output. The parser emits
   ``patterns_matched`` as a hint only. Final evidence-type and label
   decisions are the curator's, per §4 of the rules.

Timestamp convention (per ``v2/fork_c/label_rules.md`` timestamp section):
exported Discord timestamps are Pacific-localized display; converted to ET
by +3 hours for the ``ts_et`` column. Inline ET references inside message
text ("1000 MAGIC TIME", "1030 IB") are NOT reconverted; this parser does
not touch them.

Usage::

    python3 -m v2.fork_c.parse_tier1_labels \\
        --journal-dir /Users/gduby/Documents/picklesGPT/pickles/journal \\
        --out-hits v2/fork_c/parser_hits.csv \\
        --out-days v2/fork_c/parser_candidate_days.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


# Discord exports use U+202F (NARROW NO-BREAK SPACE) between the clock time
# and AM/PM indicator. Accept either regular space or U+202F. The space
# between date and time is also sometimes U+202F; tolerate both there too.
_SP = r"[\s\u202f]"
FULL_TS_RE = re.compile(rf"^(\d{{1,2}})/(\d{{1,2}})/(\d{{4}}){_SP}(\d{{1,2}}):(\d{{2}}){_SP}(AM|PM)$")
SHORT_TS_RE = re.compile(rf"^(\d{{1,2}}):(\d{{2}}){_SP}(AM|PM)$")

# Lines that look like names but aren't authors.
NON_AUTHOR_TOKENS = {"(edited)", "Click to see attachment"}


KEYWORD_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Likely directional-long-execution phrasing. Broadened to catch
    # "LONG SPX" without strict suffix (e.g. "LONG SPX for 1000 MAGIC
    # TIME" — 2024-02-16 template) plus "SPX LONG(S)" with optional
    # plural possessive.
    (
        "executed_long",
        re.compile(
            r"(\bLONG\s+SPX\b"
            r"|\bSPX\s+LONG(S)?\b"
            r"|\bwent\s+LONG\s+SPX\b"
            r"|\bgoing\s+LONG\s+SPX\b"
            r"|\binto\s+(SPX\s+LONG|LONG\s+(PUTS|CALLS|SPX))\b"
            r"|\bgoing\s+into\s+LONG\s+(PUTS|CALLS|SPX)\b"
            r"|\badd(ed|ing)\s+to\s+(these\s+|my\s+)?(SPX\s+)?LONG"
            r")",
            re.IGNORECASE,
        ),
    ),
    # Post-facto reference-as-held — he talks about a position as
    # already open without a real-time entry announcement on the same
    # message. Curator still reads full-day context to distinguish
    # explicitly_claimed from narrative_only.
    (
        "explicitly_claimed",
        re.compile(
            r"(\bout\s+of\s+SPX\s+(CALLS?|PUTS?|LONGS?)\b"
            r"|\bthose\s+SPX\s+(CALLS?|PUTS?|LONGS?)\b"
            r"|\bmy\s+SPX\s+(CALLS?|PUTS?|LONGS?)\b"
            r"|\bscaling\s+out\s+of\s+SPX\b"
            r"|\bclosed\s+(out\s+)?of\s+SPX\s+(CALLS?|PUTS?|LONGS?)\b"
            r"|\bthese\s+SPX\s+LONG\s+(CALLS?|PUTS?)\b"
            r"|\bSPX\s+LONG\s+(CALLS?|PUTS?)\s+SL'd\b)",
            re.IGNORECASE,
        ),
    ),
    # "SPX 0DTE" co-mention (with any order).
    (
        "spx_0dte_reference",
        re.compile(
            r"(\bSPX\b[^\n]{0,40}\b0DTE\b"
            r"|\b0DTE\b[^\n]{0,40}\bSPX\b)",
            re.IGNORECASE,
        ),
    ),
    # Narrative / planning phrasing.
    (
        "narrative_plan",
        re.compile(
            r"(\blooking at\b[^\n]{0,60}\bSPX\b[^\n]{0,40}\b(CALLS?|PUTS?|0DTE)\b"
            r"|\bI['\u2019]ll be looking at\b[^\n]{0,60}\bSPX\b"
            r"|\bthinking\b[^\n]{0,60}\bSPX\b[^\n]{0,40}\b(CALLS?|PUTS?)\b)",
            re.IGNORECASE,
        ),
    ),
    # Hedge / delta-repair signal. Not a label hint; curator checks structure.
    (
        "hedge_signal",
        re.compile(
            r"(\bCOUNTER\s+DELTA\b"
            r"|\bcounter\s+DELTA\s+the\b"
            r"|\bconvert\s+(into|the)\b[^\n]{0,40}\b(CDS|CCS|PCS|PDS|IC|spread)\b"
            r"|\bto\s+(counter|offset)\b[^\n]{0,40}\bSHORT\s+(CALLS|PUTS)\b"
            r"|\bback\s+ratio['\u2019]?d\b)",
            re.IGNORECASE,
        ),
    ),
]


def _looks_like_name_token(line: str) -> bool:
    """Primitive filter — necessary but not sufficient. Final name detection
    requires the two-phase check in ``find_author_lines`` (followed-by-TS)."""
    if not line or line[:1].isspace():
        return False
    if len(line) > 40:
        return False
    if FULL_TS_RE.match(line) or SHORT_TS_RE.match(line):
        return False
    if line in NON_AUTHOR_TOKENS:
        return False
    # Names in this corpus don't carry sentence punctuation.
    if any(c in line for c in ",/:;!?"):
        return False
    # Reject lines with 3+ consecutive digits (dates, years, strikes).
    if re.search(r"\d{3,}", line):
        return False
    if not any(c.isalpha() for c in line):
        return False
    if sum(1 for c in line if c.isalpha()) < 2:
        return False
    return True


def find_author_lines(lines: list[str]) -> dict[int, str]:
    """Return ``{line_idx: author_name}`` for lines that are true message-start names.

    Two-phase forward test:
      1. Line passes ``_looks_like_name_token``.
      2. Within the next 4 lines (skipping blank/space-only), the first
         non-blank line is a FULL_TS. This matches the Discord export
         convention: author line, blank, timestamp, content.

    Section headers in content (e.g. "ES LEVELS", "THURS, 14 DEC 2023")
    fail phase 1 (commas, digits) or phase 2 (no TS follows).
    """
    out: dict[int, str] = {}
    n = len(lines)
    for i in range(n):
        ln = lines[i].rstrip("\n")
        if not _looks_like_name_token(ln):
            continue
        # Phase 2: look forward up to 4 lines for a FULL_TS, skipping blanks.
        j = i + 1
        while j < n and j < i + 5:
            nxt = lines[j].rstrip("\n")
            if nxt.strip() == "":
                j += 1
                continue
            if FULL_TS_RE.match(nxt):
                out[i] = ln
            break
    return out


def find_trailing_author_for_ts(
    lines: list[str],
    ts_idx: int,
    author_lines: dict[int, str],
    assigned_ts: set[int],
    max_scan: int = 60,
) -> str | None:
    """Discord exports sometimes use a *trailing-name* format: a FULL_TS
    appears at the top of a block, content follows, and the author name
    lives at the END of the block (below its own content) instead of above
    it. Common on short single-message files like no-trade days
    (2023-10-22 Sunday prep, 2023-10-24 personal-matters).

    For a FULL_TS at ``ts_idx`` not already paired with a forward-pass
    author (``ts_idx not in assigned_ts``), scan forward up to ``max_scan``
    lines — stopping at any subsequent FULL_TS or already-confirmed
    forward-author line — for a plausible trailing author. Prefer an exact
    match to an already-confirmed author name elsewhere in the file
    (strong signal); otherwise accept a name-token line with no later TS
    in its own 4-line window (weaker, common at file tail).
    """
    n = len(lines)
    known_authors = set(author_lines.values())
    end = min(n, ts_idx + 1 + max_scan)
    for k in range(ts_idx + 1, end):
        if k in author_lines:
            end = k
            break
        ln = lines[k].rstrip("\n")
        if FULL_TS_RE.match(ln):
            end = k
            break

    # Primary: exact match to a forward-pass author name in this file.
    for k in range(ts_idx + 1, end):
        ln = lines[k].rstrip("\n")
        if ln in known_authors:
            return ln
    # Fallback: this corpus's primary journal author is "Pickles". Accept
    # a trailing "Pickles" name-token line directly. Any other trailing
    # name-like candidate is rejected — too many content lines pass the
    # loose name-token test (e.g. "current weekend positions") and would
    # falsely claim to be authors.
    for k in range(ts_idx + 1, end):
        ln = lines[k].rstrip("\n")
        if ln == "Pickles":
            return "Pickles"
    return None


def pt_to_et(pt_time: str, date_str: str) -> str:
    """Convert a Pacific-localized Discord timestamp to ET clock (HH:MM).

    Handles both ``M/D/YYYY H:MM AM/PM`` and short ``H:MM AM/PM`` (short
    is a continuation within the same day — uses ``date_str`` for the date
    context).
    """
    def _parts_to_dt(year: int, month: int, day: int, hour: int, minute: int, ampm: str) -> datetime:
        h = hour
        if ampm == "PM" and h != 12:
            h += 12
        elif ampm == "AM" and h == 12:
            h = 0
        return datetime(year, month, day, h, minute, tzinfo=ZoneInfo("America/Los_Angeles"))

    m = FULL_TS_RE.match(pt_time)
    if m:
        month, day, year, hour, minute, ampm = m.groups()
        dt = _parts_to_dt(int(year), int(month), int(day), int(hour), int(minute), ampm)
        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M")

    m = SHORT_TS_RE.match(pt_time)
    if m and date_str:
        try:
            base = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return ""
        hour, minute, ampm = m.groups()
        dt = _parts_to_dt(base.year, base.month, base.day, int(hour), int(minute), ampm)
        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M")

    return ""


def parse_journal(path: Path) -> tuple[str, list[dict]]:
    """Parse one journal file, return (date_str, list of Pickles-authored messages).

    Each message record:
        {ts_pt, content, line_start (1-indexed), line_end (1-indexed, inclusive)}

    Algorithm:
      1. First pass: find true author lines via ``find_author_lines``
         (name token + followed-by-FULL_TS).
      2. Walk the file using the author-line index. When author=="Pickles",
         capture the message block from the author line to the next author
         line (or EOF). Within the block, split subsections on FULL_TS and
         SHORT_TS markers.
      3. Ignore any lines that appear BEFORE the first TS inside a Pickles
         block — those are quote-context lines from replies, not Pickles'
         own content.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # Extract date from file header if present.
    date_str = ""
    for ln in lines[:10]:
        if ln.startswith("# DATE:"):
            date_str = ln.split(":", 1)[1].strip()
            break
    if not date_str:
        date_str = path.stem

    n = len(lines)
    author_lines = find_author_lines(lines)
    author_idx_sorted = sorted(author_lines.keys())
    # The set of confirmed author-name strings anywhere in the file.
    # Used to distinguish reply-quote markers (real names re-appearing
    # without their own TS) from content lines that coincidentally look
    # name-like (e.g. "SPX LONG PUTS", "ES LEVELS").
    confirmed_author_names: set[str] = set(author_lines.values())

    # ts_to_author_forward maps a FULL_TS line index to its forward-pass
    # author name (if an author line precedes it within 4 lines).
    ts_to_author_forward: dict[int, str] = {}
    for a_idx in author_idx_sorted:
        k = a_idx + 1
        while k < n and k < a_idx + 5:
            peek = lines[k].rstrip("\n")
            if peek.strip() == "":
                k += 1
                continue
            if FULL_TS_RE.match(peek):
                ts_to_author_forward[k] = author_lines[a_idx]
            break

    # Discover orphan FULL_TSes (no forward-author) and try trailing-name
    # resolution. Populates ts_to_author_trailing for those cases.
    ts_to_author_trailing: dict[int, str] = {}
    for i in range(n):
        if i in ts_to_author_forward:
            continue
        ln = lines[i].rstrip("\n")
        if not FULL_TS_RE.match(ln):
            continue
        assigned = set(ts_to_author_forward.keys()) | set(ts_to_author_trailing.keys())
        trailing = find_trailing_author_for_ts(lines, i, author_lines, assigned)
        if trailing is not None:
            ts_to_author_trailing[i] = trailing

    # Build a combined TS -> author map. Forward-pass wins when both exist.
    ts_to_author: dict[int, str] = dict(ts_to_author_trailing)
    ts_to_author.update(ts_to_author_forward)

    # Collect per-TS block extents. A block runs from TS+1 until the next
    # FULL_TS (any author), next forward-author line, or EOF — whichever
    # comes first. SHORT_TS markers inside a block create subsections
    # under the same author. confirmed_author_names inside content is a
    # reply-quote marker and flushes the current subsection.
    messages: list[dict] = []
    ts_sorted = sorted(ts_to_author.keys())
    forward_author_idx_set = set(author_idx_sorted)

    for ti, ts_idx in enumerate(ts_sorted):
        author = ts_to_author[ts_idx]
        if author != "Pickles":
            continue
        # Block end: next FULL_TS, next forward-author line, or EOF.
        block_end = n
        if ti + 1 < len(ts_sorted):
            block_end = min(block_end, ts_sorted[ti + 1])
        for fa in author_idx_sorted:
            if fa > ts_idx:
                block_end = min(block_end, fa)
                break

        # Within [ts_idx+1, block_end), split on SHORT_TS; the FULL_TS at
        # ts_idx itself is the first subsection's stamp.
        cur_ts: str = lines[ts_idx].rstrip("\n")
        cur_start: int = ts_idx + 1  # 0-indexed line after the TS
        cur_content: list[str] = []

        def _flush(end_exclusive: int) -> None:
            if cur_content:
                text = "".join(cur_content).strip()
                if text:
                    messages.append({
                        "ts_pt": cur_ts,
                        "content": text,
                        "line_start": cur_start + 1,
                        "line_end": end_exclusive,
                    })

        for j in range(ts_idx + 1, block_end):
            ln = lines[j].rstrip("\n")
            if SHORT_TS_RE.match(ln):
                _flush(j)
                cur_ts = ln
                cur_start = j + 1
                cur_content = []
                continue
            # Reply-quote marker: line whose exact text is a confirmed
            # author name elsewhere in this file. Flush and stop.
            if ln in confirmed_author_names:
                _flush(j)
                cur_ts = ""
                cur_start = j
                cur_content = []
                # Mark remainder of block as "don't capture"
                break
            cur_content.append(lines[j])
        else:
            _flush(block_end)
            continue
        # Broke out due to reply-quote: nothing more to capture in block.

    return date_str, messages


def shortlist_hits(messages: list[dict], date_str: str) -> list[dict]:
    """Apply keyword patterns to Pickles messages. Deduplicate by content hash."""
    seen_hashes: set[str] = set()
    hits: list[dict] = []
    for m in messages:
        content = m["content"]
        # Dedup: hash on a normalized prefix (Discord sometimes re-quotes
        # with minor whitespace differences).
        norm = re.sub(r"\s+", " ", content[:400]).strip().lower()
        h = hashlib.md5(norm.encode()).hexdigest()[:16]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        matched = []
        for name, pat in KEYWORD_PATTERNS:
            if pat.search(content):
                matched.append(name)
        if not matched:
            continue
        hits.append({
            **m,
            "date": date_str,
            "patterns": matched,
            "ts_et": pt_to_et(m["ts_pt"], date_str),
        })
    return hits


def aggregate_days(
    all_files: list[Path],
    all_hits: list[dict],
    msg_counts_by_day: dict[str, int],
) -> list[dict]:
    """One row per journal day. Includes days with zero hits for curator
    awareness. ``n_keyword_hits = 0`` rows are the likely label=0 days."""
    by_day: dict[str, dict] = {}
    for hit in all_hits:
        d = hit["date"]
        row = by_day.setdefault(d, {
            "n_hits": 0, "patterns": set(), "hits": [], "first_hit": None,
        })
        row["n_hits"] += 1
        row["patterns"].update(hit["patterns"])
        row["hits"].append(hit)
        if row["first_hit"] is None or (
            hit.get("ts_et", "") and row["first_hit"].get("ts_et", "") > hit["ts_et"]
        ):
            row["first_hit"] = hit

    out: list[dict] = []
    for p in all_files:
        d = p.stem
        row = by_day.get(d, {"n_hits": 0, "patterns": set(), "hits": [], "first_hit": None})
        fh = row["first_hit"]
        out.append({
            "date": d,
            "n_pickles_messages": msg_counts_by_day.get(d, 0),
            "n_keyword_hits": row["n_hits"],
            "patterns_matched": "|".join(sorted(row["patterns"])) if row["patterns"] else "",
            "first_hit_ts_pt": fh["ts_pt"] if fh else "",
            "first_hit_ts_et": fh["ts_et"] if fh else "",
            "first_hit_line": fh["line_start"] if fh else "",
            "first_hit_excerpt": (
                (fh["content"][:300].replace("\n", " / ")) if fh else ""
            ),
        })
    return sorted(out, key=lambda r: r["date"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--journal-dir", required=True)
    ap.add_argument("--out-hits", default="v2/fork_c/parser_hits.csv")
    ap.add_argument("--out-days", default="v2/fork_c/parser_candidate_days.csv")
    args = ap.parse_args()

    jdir = Path(args.journal_dir)
    files = sorted(jdir.glob("*.txt"))
    print(f"Parsing {len(files)} journal files from {jdir}")

    all_hits: list[dict] = []
    msg_counts: dict[str, int] = {}
    for p in files:
        date_str, messages = parse_journal(p)
        msg_counts[p.stem] = len(messages)
        hits = shortlist_hits(messages, date_str)
        for h in hits:
            h["file"] = p.name
            all_hits.append(h)

    n_days_with_hits = len({h["date"] for h in all_hits})
    print(f"Total Pickles messages across corpus: {sum(msg_counts.values())}")
    print(f"Total keyword hits (post-dedup): {len(all_hits)}")
    print(f"Days with at least one hit: {n_days_with_hits} / {len(files)}")

    # Per-hit CSV.
    Path(args.out_hits).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_hits, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "file", "line_start", "line_end",
            "ts_pt", "ts_et", "patterns", "excerpt",
        ])
        w.writeheader()
        for h in all_hits:
            w.writerow({
                "date": h["date"], "file": h["file"],
                "line_start": h["line_start"], "line_end": h["line_end"],
                "ts_pt": h["ts_pt"], "ts_et": h["ts_et"],
                "patterns": "|".join(h["patterns"]),
                "excerpt": h["content"][:300].replace("\n", " / "),
            })

    # Per-day aggregate (includes no-hit days).
    day_rows = aggregate_days(files, all_hits, msg_counts)
    with open(args.out_days, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "n_pickles_messages", "n_keyword_hits",
            "patterns_matched", "first_hit_ts_pt", "first_hit_ts_et",
            "first_hit_line", "first_hit_excerpt",
        ])
        w.writeheader()
        for row in day_rows:
            w.writerow(row)

    print(f"Wrote {args.out_hits} and {args.out_days}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
