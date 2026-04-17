"""Render the experiment-history block of the status dashboard.

Kept as a standalone module so deploy.sh and tests both consume the same
parser. Hard-fails if results.tsv does not have the current CVReport schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

from v2.core.cv_report import RESULTS_TSV_HEADER


def render(results_path: str | Path) -> str:
    """Return the history block as a single string. Safe for subprocess stdout."""
    path = Path(results_path)
    lines_out: list[str] = []
    if not path.exists():
        return "  (no results.tsv yet)\n"

    with path.open() as f:
        raw = [line.rstrip("\n") for line in f if line.strip()]
    if len(raw) <= 1:
        return "  (no experiments yet)\n"

    header = raw[0].split("\t")
    if header != RESULTS_TSV_HEADER:
        return (
            "  ERROR: results.tsv header does not match CVReport schema.\n"
            f"  expected: {RESULTS_TSV_HEADER}\n"
            f"  got:      {header}\n"
        )

    lines_out.append(f"  {'exp':>10}  {'mode':>8}  {'stab':>8}  "
                     f"{'poolPF':>7}  {'Kept':>6}  Summary")
    lines_out.append(f"  {'---':>10}  {'----':>8}  {'----':>8}  "
                     f"{'------':>7}  {'----':>6}  -------")

    for line in raw[1:]:
        cols = line.split("\t")
        if len(cols) < len(header):
            continue
        row = dict(zip(header, cols))
        eid = row.get("experiment", "?")
        mode = row.get("screening_mode", "?")
        stab = row.get("stability_score", "?")
        pool_pf = row.get("pooled_pf", "?")
        kept = row.get("status", "?")
        summary = row.get("description", "")[:40]
        try:
            stab_f = float(stab); pool_f = float(pool_pf)
            lines_out.append(
                f"  {eid:>10}  {mode:>8}  {stab_f:>8.3f}  "
                f"{pool_f:>7.2f}  {kept:>6}  {summary}"
            )
        except (TypeError, ValueError):
            lines_out.append(
                f"  {eid:>10}  {mode:>8}  {stab:>8}  "
                f"{pool_pf:>7}  {kept:>6}  {summary}"
            )
    return "\n".join(lines_out) + "\n"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "v2/results.tsv"
    sys.stdout.write(render(path))


if __name__ == "__main__":
    main()
