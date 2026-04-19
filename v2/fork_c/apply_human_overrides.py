"""Fork C Phase 1 — apply the human reviewer's overrides to the model-drafted
proposed labels and split into canonical + shadow CSVs.

Overrides are encoded in code (not in a separate data file) so the
curation decisions are version-controlled and auditable. Each override
names the row by date and lists which fields change. Unspecified fields
retain their values from ``tier1_labels_proposed.csv``.

Discipline:
- This script is rerun-safe: run it again with more overrides and you get
  the updated canonical CSVs without touching the proposed-labels input.
- The proposed-labels CSV itself is NEVER modified — it's the immutable
  model draft. All human intervention lands here.
- Shadow CSV captures rows that should be excluded from primary training
  per label_rules.md §5 (Low confidence by default, or user-explicitly
  tagged ``in_shadow=True`` below).

Usage::
    python3 -m v2.fork_c.apply_human_overrides \\
        --in v2/fork_c/tier1_labels_proposed.csv \\
        --out-canonical v2/fork_c/tier1_labels.csv \\
        --out-shadow v2/fork_c/tier1_labels_shadow.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Override:
    """A human override applied on top of a proposed-labels row.

    Any field left as ``None`` inherits from the proposed row. String fields
    set to the empty string (``""``) actively clear the field. ``label`` and
    ``hedge_only`` take their typed values."""
    label: int | None = None
    confidence: str | None = None
    evidence_type: str | None = None
    first_qualifying_time_et: str | None = None
    side: str | None = None
    hedge_only: bool | None = None
    journal_ref: str | None = None
    notes: str | None = None
    in_shadow: bool = False   # route to tier1_labels_shadow.csv
    reviewer: str = "gduby"


# ------------------------------------------------------------------------
# Human override list — gduby review round 1 (2026-04-19)
#
# 2026-04-19 addendum: 2023-10-30 added after preflight-driven audit.
# The proposer classified it label=1 side=call because of the "1:50 PM"
# timestamp on a 0DTE-LONG line, but the full journal (lines 50-58) shows
# the 0DTE longs were explicitly defending a breached/rolled CCS short
# strike — hedge_only=True per §N5. Catching this late is consistent with
# the plan's "curator authoritative" design; the override ledger is the
# one place where this kind of catch is recorded and versioned.
# ------------------------------------------------------------------------

OVERRIDES: dict[str, Override] = {
    # Definite overrides ---------------------------------------------------
    "2023-10-30": Override(
        label=0, confidence="High", evidence_type="",
        first_qualifying_time_et="", side="", hedge_only=True,
        journal_ref="2023-10-30.txt:50-58",
        notes=(
            "Hedge-only per §N5. Journal is explicit: 'The 0DTE LONGs came "
            "to defend the BREACHED & ROLLED CCS SHORT STRIKE and offset "
            "the loss on the the trade.' The LONG SPX CALL added alongside "
            "is a spread-repair entry, not a directional 0DTE long. Proposer "
            "mis-labeled label=1 side=call with first_qualifying_time_et=16:50 "
            "(a PT→ET conversion of the 1:50 PM post-hoc recap timestamp, "
            "which is past the 16:00 ET close in any case)."
        ),
    ),
    "2024-03-22": Override(
        label=1, confidence="High", evidence_type="explicitly_claimed",
        first_qualifying_time_et="10:00", side="",
        journal_ref="2024-03-22.txt:54",
        notes=(
            "Explicitly_claimed — 'only trade i made today, was during "
            "1000 MAGIC TIME.' Day-level case per Ex. 6. Side unrecoverable "
            "from journal; side=empty allowed on positive rows per §6 "
            "(Revision 3)."
        ),
    ),
    "2024-04-15": Override(
        label=1, confidence="Medium", evidence_type="explicitly_claimed",
        first_qualifying_time_et="11:50", side="call",
        journal_ref="2024-04-15.txt:123",
        notes=(
            "Explicitly_claimed — 'out of these calls at a loss. 1 for 3 "
            "on trades today.' The earlier 6:58 AM PT 'looking at LONGS "
            "here heading into 1000 MAGIC TIME' is narrative_only and "
            "does not qualify the day; the first qualifying evidence is "
            "the 8:50 AM PT exit reference (11:50 ET)."
        ),
    ),
    "2024-04-30": Override(
        label=0, confidence="High", evidence_type="",
        first_qualifying_time_et="", side="", hedge_only=False,
        journal_ref="2024-04-30.txt:72",
        notes=(
            "Hedge / defense education commentary, not Pickles execution. "
            "Content is hypothetical worked examples answering another "
            "trader's defense question. No directional SPX 0DTE long was "
            "taken on this day."
        ),
    ),
    "2024-06-21": Override(
        label=1, confidence="High", evidence_type="executed",
        first_qualifying_time_et="09:51", side="put", hedge_only=False,
        journal_ref="2024-06-21.txt:52",
        notes=(
            "Executed — 'LONG 0DTE SPX PUTS' at 6:51 AM PT (9:51 ET). "
            "Ambiguous-category Low in the proposed draft was a proposer "
            "false positive; the journal text is an unambiguous "
            "executed_long entry on the put side."
        ),
    ),
    # Confidence / side upgrades ------------------------------------------
    "2023-11-16": Override(
        confidence="High", evidence_type="executed",
        first_qualifying_time_et="10:23", side="call",
        journal_ref="2023-11-16.txt:109",
        notes=(
            "'switching from FUTURES to SPX 0DTE on higher confidence in "
            "this trade: 0DTE SPX CALLS on ES & NQ back above VWAP' — "
            "executed long calls. Side and timing both unambiguous; "
            "upgrade to High."
        ),
    ),
    "2024-02-16": Override(
        confidence="High", first_qualifying_time_et="10:03", side="call",
        journal_ref="2024-02-16.txt:83",
        notes=(
            "'LONG SPX for 1000 MAGIC TIME' at 7:03 AM PT (10:03 ET). "
            "Later same-position references at 8:02 AM PT ('it was ATM "
            "when i first entered... i'd go OTM but i want that DELTA') "
            "and 8:17 AM PT ('out of SPX CALLS') make the CALL side "
            "unambiguous. Upgrade to High."
        ),
    ),
    "2023-12-14": Override(
        confidence="High",
        notes=(
            "10:08 ET directional 'LONG SPX CALLS on ES VWAP SUPPORT' is "
            "the canonical Row-1 template (Ex. 1). Earlier 9:04 ET "
            "hedge-only 4750 CALLS is structurally separate. Directional "
            "trade is explicit; upgrade to High."
        ),
    ),
    # Side resolutions (keep Medium, add side from context) ----------------
    "2023-11-13": Override(
        side="call",
        notes=(
            "'LONG SPX off ES & NQ bounce off opening 15m candle' — "
            "bullish 15m bounce context implies long calls. Side resolved "
            "from narrative direction (no explicit CALLS/PUTS word)."
        ),
    ),
    "2023-11-17": Override(
        side="call",
        notes=(
            "'went LONG SPX 0DTE 4510' and later 'enter LONG SPX 4510 0DTE, "
            "COUNTER-TREND this dip, expecting a rip' — both are long "
            "calls. Side resolved from strike + bullish intent."
        ),
    ),
    "2023-12-04": Override(
        evidence_type="explicitly_claimed",
        first_qualifying_time_et="09:30", side="call",
        notes=(
            "Reclassified from executed to explicitly_claimed per human "
            "review: 'SMALL SPX LONG when the bell rings' is intent "
            "(narrative) at open; 'out of SPX LONGS' is the first "
            "qualifying held-state evidence. Side=call from bullish bias. "
            "First qualifying time aligned with implied open-bell entry."
        ),
    ),
    "2024-03-28": Override(
        side="call",
        notes=(
            "Explicitly_claimed SPX call position per human review. "
            "Side=call resolved from context."
        ),
    ),
    # Keep-as-proposed (documented for audit trail) -----------------------
    # "2023-12-29": already label=0 High third_party — no override needed.
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_path", default="v2/fork_c/tier1_labels_proposed.csv"
    )
    ap.add_argument("--out-canonical", default="v2/fork_c/tier1_labels.csv")
    ap.add_argument("--out-shadow", default="v2/fork_c/tier1_labels_shadow.csv")
    args = ap.parse_args()

    with open(args.in_path, encoding="utf-8") as f:
        proposed_rows = list(csv.DictReader(f))

    canonical_rows: list[dict] = []
    shadow_rows: list[dict] = []
    override_hits = 0

    for r in proposed_rows:
        d = r["date"]
        ov = OVERRIDES.get(d)
        out = dict(r)
        if ov is not None:
            override_hits += 1
            if ov.label is not None:
                out["label"] = str(ov.label)
            if ov.confidence is not None:
                out["confidence"] = ov.confidence
            if ov.evidence_type is not None:
                out["evidence_type"] = ov.evidence_type
            if ov.first_qualifying_time_et is not None:
                out["first_qualifying_time_et"] = ov.first_qualifying_time_et
            if ov.side is not None:
                out["side"] = ov.side
            if ov.hedge_only is not None:
                out["hedge_only"] = str(ov.hedge_only)
            if ov.journal_ref is not None:
                out["journal_ref"] = ov.journal_ref
            if ov.notes is not None:
                out["notes"] = ov.notes
            out["reviewer"] = ov.reviewer
            existing_reasoning = out.get("proposed_reasoning", "")
            out["proposed_reasoning"] = (
                f"[HUMAN OVERRIDE] {existing_reasoning}"
            )

        # Route: Low → shadow, unless the human override explicitly rescued it.
        if (ov is not None and ov.in_shadow) or (
            ov is None and out["confidence"] == "Low"
        ):
            shadow_rows.append(out)
        else:
            canonical_rows.append(out)

    fieldnames = [
        "date", "label", "confidence", "evidence_type",
        "first_qualifying_time_et", "side", "hedge_only",
        "excerpt", "journal_ref", "reviewer", "notes",
        "proposed_reasoning",
    ]

    Path(args.out_canonical).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_canonical, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in canonical_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    with open(args.out_shadow, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in shadow_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    # Summary
    from collections import Counter
    label_dist = Counter(r["label"] for r in canonical_rows)
    conf_dist = Counter(r["confidence"] for r in canonical_rows)
    evi_dist = Counter((r.get("evidence_type") or "empty") for r in canonical_rows)
    side_dist = Counter((r.get("side") or "empty") for r in canonical_rows)

    print(f"Applied {override_hits} human overrides on {len(proposed_rows)} proposed rows.")
    print(f"→ {args.out_canonical}: {len(canonical_rows)} rows")
    print(f"   label:      {dict(label_dist)}")
    print(f"   confidence: {dict(conf_dist)}")
    print(f"   evidence:   {dict(evi_dist)}")
    print(f"   side:       {dict(side_dist)}")
    print(f"   hedge_only=True: {sum(1 for r in canonical_rows if r.get('hedge_only') == 'True')}")
    print(f"→ {args.out_shadow}: {len(shadow_rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
