"""Fork C Phase 1 — model-drafted first-pass curation proposer.

**Not a labeler. A proposer.** Produces ``tier1_labels_proposed.csv`` with
one row per trading day in the 167-file journal corpus. Every row carries
a ``proposed_reasoning`` column explaining the decision path so the human
reviewer can audit, accept, edit, or move to shadow.

Discipline (see plan §Curation pass):
- Rules come from ``v2/fork_c/label_rules.md``. This script embodies those
  rules as Python logic so the draft is reproducible and reviewable.
- Low-confidence / ambiguous rows are flagged ``confidence=Low`` (destined
  for ``tier1_labels_shadow.csv`` on human review).
- When in doubt, the script proposes the more-conservative label (label=0
  unless evidence clearly supports 1) and tags confidence accordingly.
- The human is authoritative; this script is the first draft.

Decision tree (mirrors label_rules.md §2-§3):
  1. No Pickles messages in file → label=0, confidence=High, evidence=empty,
     notes="No Pickles-authored content".
  2. Any message contains an `executed_long` match with hedge_signal AND
     structural hedge context ("to counter DELTA ... SHORT", "with the
     ... SHORT CALLS", "convert ... into ... CDS/PCS/CCS") → label=0,
     hedge_only=True, evidence=executed, confidence=High.
  3. Any message contains a bare `executed_long` match (no hedge context)
     or `explicitly_claimed` match for SPX longs → label=1,
     evidence=executed or explicitly_claimed, confidence=High or Medium
     depending on clarity.
  4. Only narrative_only / spx_0dte_reference matches without entry →
     label=0, evidence=narrative_only, confidence=High.
  5. Only `spx_0dte_reference` on spread lines (PCS/CCS/IC) →
     label=0, evidence=empty (spread-only), confidence=High.
  6. Any case the script cannot confidently resolve → confidence=Low,
     notes include "AMBIGUOUS: <reason>".

Side detection: "LONG SPX CALLS" / "SPX LONG CALLS" → call.
"LONG SPX PUTS" / "SPX LONG PUTS" / "LONG PUTS" (in SPX context) → put.
Ambiguous → side empty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from v2.fork_c.parse_tier1_labels import (
    KEYWORD_PATTERNS,
    parse_journal,
    pt_to_et,
)


# Structural hedge indicators — body-of-message text that marks the SPX
# long as paired with an existing SPX short leg / spread structure.
HEDGE_STRUCTURE_RE = re.compile(
    r"(\bto\s+(counter|offset)\s+DELTA\b"
    r"|\bcounter\s+DELTA\s+the\b"
    r"|\bwith\s+the\s+\d{3,5}\s+SHORT\s+(CALLS|PUTS)\b"
    r"|\bconvert\s+(into|the)\b[^\n]{0,40}\b(CDS|CCS|PCS|PDS|IC|spread)\b"
    r"|\bSHORT\s+CALLS?\s+from\s+the\s+weekly\s+range\b"
    r"|\bback[-\s]?ratio['\u2019]?d\b.*\bSHORT\s+CALLS?\b"
    r"|\bdefense\b[^\n]{0,40}\bSHORT\s+(CALLS|PUTS)\b)",
    re.IGNORECASE,
)

# Third-party commentary indicators — text that discusses SOMEONE ELSE's
# trade, not Pickles' own. Flags days like 2023-12-29 (JPM roll commentary).
THIRD_PARTY_RE = re.compile(
    r"(\bJPM\b[^\n]{0,60}\b(roll|fund|collar)\b"
    r"|\btheir\s+(expiring|position|long|short|calls|puts)\b"
    r"|\bcame\s+to\s+defend\b)",
    re.IGNORECASE,
)

# Narrative-only markers: "looking at", "thinking", conditional phrasing.
NARRATIVE_MARKER_RE = re.compile(
    r"(\blooking at\b|\bI['\u2019]ll be looking at\b|\bthinking\b|\bif\b[^\n]{0,40}\bI['\u2019]ll\b)",
    re.IGNORECASE,
)

# Side detection. Broadened to capture bare "0DTE SPX CALLS" context without
# a preceding "LONG" word.
SIDE_CALL_RE = re.compile(
    r"\b("
    r"LONG\s+SPX\s+CALLS?"
    r"|SPX\s+LONG\s+(0DTE\s+)?CALLS?"
    r"|went\s+LONG\s+SPX\s+CALLS?"
    r"|going\s+(into\s+)?LONG\s+SPX\s+CALLS?"
    r"|going\s+LONG\s+(SPX\s+)?CALLS?"
    r"|0DTE\s+SPX\s+CALLS?"
    r"|SPX\s+0DTE\s+CALLS?"
    r"|into\s+LONG\s+CALLS?\b[^\n]{0,40}\bSPX\b"
    r"|into\s+SPX\s+LONG\s+CALLS?"
    r")\b",
    re.IGNORECASE,
)
SIDE_PUT_RE = re.compile(
    r"\b("
    r"LONG\s+SPX\s+PUTS?"
    r"|SPX\s+LONG\s+(0DTE\s+)?PUTS?"
    r"|going\s+(into\s+)?LONG\s+PUTS?"
    r"|into\s+LONG\s+PUTS?"
    r"|0DTE\s+SPX\s+PUTS?"
    r"|SPX\s+0DTE\s+PUTS?"
    r"|averaged\s+down\s+into\s+these\s+SPX\s+LONG\s+PUTS?"
    r")\b",
    re.IGNORECASE,
)

# Spread-only patterns (any of these in a hit means it's NOT a directional long).
SPREAD_ONLY_RE = re.compile(
    r"\b(PCS|CCS|IC|PDS|CDS|iron\s+condor|iron\s+fly|CREDIT\s+SPREAD|DEBIT\s+SPREAD|CALL\s+LADDER|PUT\s+LADDER|ALBATROSS|GUT\s+SPREAD|CALL\s+SPREAD|PUT\s+SPREAD)\b",
    re.IGNORECASE,
)

# Exit-reference patterns: these are explicitly_claimed markers, not a fresh
# directional entry. Important for days where Pickles opens a hedge and
# later says "closed out of SPX LONG CALLS" — that close refers to the
# hedge, not a separate directional bet.
EXIT_REFERENCE_RE = re.compile(
    r"\b("
    r"closed\s+out\s+of\s+SPX"
    r"|out\s+of\s+SPX\s+(CALLS?|PUTS?|LONGS?|LONG\s+CALLS?|LONG\s+PUTS?)"
    r"|scaling\s+out\s+of\s+SPX"
    r"|TP['\u2019]?d\s+on\s+SPX"
    r"|SPX\s+LONG\s+(CALLS?|PUTS?)\s+SL['\u2019]?d"
    r"|those\s+SPX\s+(CALLS?|PUTS?|LONGS?)"
    r"|these\s+SPX\s+LONG"
    r"|averaged\s+down\s+into\s+these\s+SPX"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class ProposedRow:
    date: str
    label: int
    confidence: str
    evidence_type: str
    first_qualifying_time_et: str
    side: str
    hedge_only: bool
    excerpt: str
    journal_ref: str
    reviewer: str = "model-draft"
    notes: str = ""
    proposed_reasoning: str = ""


def _classify_message(content: str) -> dict:
    """Return a classification of one Pickles message for SPX-0DTE-long intent.

    Categories (first matching wins):
        executed_long_directional: executed SPX long, no hedge context.
        executed_long_hedge: executed SPX long paired with SPX short leg.
        explicitly_claimed: post-facto reference to a held SPX long.
        narrative_only: plan/possibility only, no execution.
        spread_only: credit/debit spread entry (not a directional long).
        third_party: talking about someone else's trade (e.g. JPM).
        irrelevant: no SPX 0DTE long mention.
    """
    # Check keyword patterns from the parser.
    matched: list[str] = []
    for name, pat in KEYWORD_PATTERNS:
        if pat.search(content):
            matched.append(name)

    if not matched:
        return {"category": "irrelevant", "matched": matched}

    is_third_party = bool(THIRD_PARTY_RE.search(content))
    is_hedge_structural = bool(HEDGE_STRUCTURE_RE.search(content))
    is_exit_ref = bool(EXIT_REFERENCE_RE.search(content))
    is_spread_only_context = bool(SPREAD_ONLY_RE.search(content))
    is_narrative = bool(NARRATIVE_MARKER_RE.search(content))

    # Third-party commentary takes precedence over everything else.
    if is_third_party:
        return {"category": "third_party", "matched": matched}

    if "executed_long" in matched:
        # Exit references ("closed out of SPX CALLS") are explicitly_claimed
        # markers, NOT new directional entries. Check before the directional
        # branch so a day's second message that just closes the hedge doesn't
        # look like a second directional trade.
        if is_exit_ref and not re.search(
            r"\b(went\s+LONG\s+SPX|going\s+(into\s+)?LONG\s+SPX|entered\s+LONG\s+SPX|into\s+SPX\s+LONG)\b",
            content, re.IGNORECASE,
        ):
            return {"category": "explicitly_claimed", "matched": matched}
        if is_hedge_structural:
            return {"category": "executed_long_hedge", "matched": matched}
        # If spread_only context is strong AND no clear "LONG SPX CALLS/PUTS"
        # entry phrasing, treat as spread context rather than directional.
        # This catches "0DTE PUT LADDER" days where keyword hit fires on
        # "LONG" within LADDER structure.
        if is_spread_only_context and not re.search(
            r"\b(LONG\s+SPX\s+(CALLS?|PUTS?|\d{3,5})|went\s+LONG\s+SPX|going\s+(into\s+)?LONG\s+SPX|into\s+SPX\s+LONG|LONG\s+SPX\s+for\b)\b",
            content, re.IGNORECASE,
        ):
            return {"category": "spread_only", "matched": matched}
        # Directional: could be a real execution. If narrative markers ALSO
        # present ("will be looking at SPX LONG CALLS if ..."), treat as
        # narrative only — conditional statement, not fired.
        if is_narrative and not re.search(
            r"\b(went|entered|into|going into|LONG\s+SPX\s+(for|CALLS?|PUTS?|\d{3,5}))\b",
            content, re.IGNORECASE,
        ):
            return {"category": "narrative_only", "matched": matched}
        return {"category": "executed_long_directional", "matched": matched}

    if "explicitly_claimed" in matched:
        return {"category": "explicitly_claimed", "matched": matched}

    if "narrative_plan" in matched and not ("executed_long" in matched):
        return {"category": "narrative_only", "matched": matched}

    if "spx_0dte_reference" in matched and is_spread_only_context:
        return {"category": "spread_only", "matched": matched}

    if "hedge_signal" in matched and "executed_long" not in matched and "explicitly_claimed" not in matched:
        # hedge_signal without a corresponding long is usually narration about
        # managing a credit spread — not directional.
        return {"category": "spread_only", "matched": matched}

    # Fallback — keyword fired but none of the above shapes fit cleanly.
    return {"category": "ambiguous", "matched": matched}


def _detect_side(content: str) -> str:
    if SIDE_CALL_RE.search(content) and SIDE_PUT_RE.search(content):
        return "both"
    if SIDE_CALL_RE.search(content):
        return "call"
    if SIDE_PUT_RE.search(content):
        return "put"
    return ""


def _excerpt(content: str, max_len: int = 220) -> str:
    text = re.sub(r"\s+", " ", content).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def propose_for_day(
    date: str, file_name: str, messages: list[dict]
) -> ProposedRow:
    """Apply the decision tree. Returns a single ProposedRow."""

    if not messages:
        return ProposedRow(
            date=date, label=0, confidence="High", evidence_type="",
            first_qualifying_time_et="", side="", hedge_only=False,
            excerpt="", journal_ref=f"{file_name}:0",
            notes="No Pickles-authored content in file.",
            proposed_reasoning="no_pickles_messages -> label=0 High",
        )

    # Classify each message in chronological order and find the FIRST
    # qualifying (positive) message, tracking hedge / third-party context
    # across all messages to resolve day-level label.
    classifications = []
    for m in messages:
        cls = _classify_message(m["content"])
        cls["message"] = m
        classifications.append(cls)

    # Any third-party-only day? (Flags like 2023-12-29 JPM commentary.)
    only_third_party_or_irrelevant = all(
        c["category"] in ("third_party", "irrelevant") for c in classifications
    )

    if only_third_party_or_irrelevant:
        # Pick first third-party message for excerpt, else first message.
        tp = next(
            (c for c in classifications if c["category"] == "third_party"),
            classifications[0],
        )
        m = tp["message"]
        return ProposedRow(
            date=date, label=0, confidence="High", evidence_type="",
            first_qualifying_time_et="", side="", hedge_only=False,
            excerpt=_excerpt(m["content"]),
            journal_ref=f"{file_name}:{m['line_start']}-{m['line_end']}",
            notes=(
                "Keyword hits only reference third-party trades "
                "(e.g. JPM roll commentary). Not Pickles' own execution."
                if tp["category"] == "third_party"
                else "No SPX 0DTE long mention in any Pickles message."
            ),
            proposed_reasoning=f"categories={[c['category'] for c in classifications]} -> label=0 High",
        )

    # Day-level hedge-vs-directional resolution.
    # If the day contains ANY executed_long_hedge AND the only other SPX-long
    # evidence is explicitly_claimed (no new executed_long_directional entry),
    # the entire day is a single-position hedge lifecycle (open + close of
    # the same hedge contracts). hedge_only=True, label=0 wins.
    has_hedge = any(c["category"] == "executed_long_hedge" for c in classifications)
    has_directional = any(c["category"] == "executed_long_directional" for c in classifications)
    if has_hedge and not has_directional:
        first_hedge = next(c for c in classifications if c["category"] == "executed_long_hedge")
        m = first_hedge["message"]
        return ProposedRow(
            date=date, label=0, confidence="High",
            evidence_type="executed",
            first_qualifying_time_et="", side="", hedge_only=True,
            excerpt=_excerpt(m["content"]),
            journal_ref=f"{file_name}:{m['line_start']}-{m['line_end']}",
            notes=(
                "SPX long executed but structurally paired with existing SPX "
                "short leg (N5 hedge-only). Subsequent exit references on "
                "the same day close the same hedge position — not a "
                "separate directional bet. label=0."
            ),
            proposed_reasoning="has_hedge and no_separate_directional -> label=0 High hedge_only=True",
        )

    # Find first executed_long_directional or explicitly_claimed (positive candidate).
    first_positive = next(
        (c for c in classifications
         if c["category"] in ("executed_long_directional", "explicitly_claimed")),
        None,
    )

    if first_positive is not None:
        m = first_positive["message"]
        side = _detect_side(m["content"])
        evidence = "executed" if first_positive["category"] == "executed_long_directional" else "explicitly_claimed"
        # Confidence: High if side is unambiguous + no conflict with hedge/narrative.
        day_has_hedge = any(c["category"] == "executed_long_hedge" for c in classifications)
        day_has_narrative = any(c["category"] == "narrative_only" for c in classifications)
        if side == "" or side == "both":
            confidence = "Medium"
            notes_side = " Side ambiguous from keywords; requires human review."
        elif day_has_hedge:
            confidence = "Medium"
            notes_side = " Day also contains hedge-only SPX long; directional hit(s) make day positive but review structure."
        elif day_has_narrative:
            confidence = "High"
            notes_side = ""
        else:
            confidence = "High"
            notes_side = ""
        return ProposedRow(
            date=date, label=1, confidence=confidence,
            evidence_type=evidence,
            first_qualifying_time_et=pt_to_et(m["ts_pt"], date),
            side=side, hedge_only=False,
            excerpt=_excerpt(m["content"]),
            journal_ref=f"{file_name}:{m['line_start']}-{m['line_end']}",
            notes=notes_side.strip(),
            proposed_reasoning=f"first_positive={first_positive['category']} side={side or 'none'} -> label=1 {confidence}",
        )

    # No directional positive. Check for hedge_only OR narrative_only OR spread_only.
    first_hedge = next(
        (c for c in classifications if c["category"] == "executed_long_hedge"),
        None,
    )
    if first_hedge is not None:
        m = first_hedge["message"]
        return ProposedRow(
            date=date, label=0, confidence="High",
            evidence_type="executed",
            first_qualifying_time_et="", side="", hedge_only=True,
            excerpt=_excerpt(m["content"]),
            journal_ref=f"{file_name}:{m['line_start']}-{m['line_end']}",
            notes=(
                "SPX long executed but structurally paired with existing SPX "
                "short leg (N5 hedge-only applies). label=0."
            ),
            proposed_reasoning="first_qualifying=executed_long_hedge -> label=0 High hedge_only=True",
        )

    first_narrative = next(
        (c for c in classifications if c["category"] == "narrative_only"),
        None,
    )
    if first_narrative is not None:
        m = first_narrative["message"]
        return ProposedRow(
            date=date, label=0, confidence="High",
            evidence_type="narrative_only",
            first_qualifying_time_et="", side="", hedge_only=False,
            excerpt=_excerpt(m["content"]),
            journal_ref=f"{file_name}:{m['line_start']}-{m['line_end']}",
            notes="SPX-long language appears only as plan/possibility (N1 narrative_only).",
            proposed_reasoning="first_qualifying=narrative_only -> label=0 High",
        )

    # Only spread_only / irrelevant → label=0, evidence_type empty.
    first_spread = next(
        (c for c in classifications if c["category"] == "spread_only"),
        None,
    )
    if first_spread is not None:
        m = first_spread["message"]
        return ProposedRow(
            date=date, label=0, confidence="High",
            evidence_type="",
            first_qualifying_time_et="", side="", hedge_only=False,
            excerpt=_excerpt(m["content"]),
            journal_ref=f"{file_name}:{m['line_start']}-{m['line_end']}",
            notes="SPX 0DTE activity is spread-only (credit/debit spreads). No directional long (N4).",
            proposed_reasoning="first_qualifying=spread_only -> label=0 High",
        )

    # Ambiguous fallback.
    first_ambig = next(
        (c for c in classifications if c["category"] == "ambiguous"),
        classifications[0],
    )
    m = first_ambig["message"]
    return ProposedRow(
        date=date, label=0, confidence="Low",
        evidence_type="",
        first_qualifying_time_et="", side="", hedge_only=False,
        excerpt=_excerpt(m["content"]),
        journal_ref=f"{file_name}:{m['line_start']}-{m['line_end']}",
        notes=(
            "AMBIGUOUS: keyword hit but the decision tree could not cleanly "
            f"resolve the category (saw: {[c['category'] for c in classifications]}). "
            "Human review required; shadow file."
        ),
        proposed_reasoning=f"categories={[c['category'] for c in classifications]} -> AMBIGUOUS Low",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal-dir", required=True)
    ap.add_argument("--out", default="v2/fork_c/tier1_labels_proposed.csv")
    args = ap.parse_args()

    jdir = Path(args.journal_dir)
    files = sorted(jdir.glob("*.txt"))
    rows: list[ProposedRow] = []
    for p in files:
        date_str, messages = parse_journal(p)
        row = propose_for_day(date_str, p.name, messages)
        rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "label", "confidence", "evidence_type",
            "first_qualifying_time_et", "side", "hedge_only",
            "excerpt", "journal_ref", "reviewer", "notes",
            "proposed_reasoning",
        ])
        w.writeheader()
        for r in rows:
            w.writerow({
                "date": r.date, "label": r.label, "confidence": r.confidence,
                "evidence_type": r.evidence_type,
                "first_qualifying_time_et": r.first_qualifying_time_et,
                "side": r.side, "hedge_only": str(r.hedge_only),
                "excerpt": r.excerpt, "journal_ref": r.journal_ref,
                "reviewer": r.reviewer, "notes": r.notes,
                "proposed_reasoning": r.proposed_reasoning,
            })

    # Summary statistics
    from collections import Counter
    label_dist = Counter(r.label for r in rows)
    conf_dist = Counter(r.confidence for r in rows)
    evi_dist = Counter(r.evidence_type or "empty" for r in rows)
    print(f"Proposed {len(rows)} rows → {args.out}")
    print(f"  label: {dict(label_dist)}")
    print(f"  confidence: {dict(conf_dist)}")
    print(f"  evidence_type: {dict(evi_dist)}")
    print(f"  hedge_only=True: {sum(1 for r in rows if r.hedge_only)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
