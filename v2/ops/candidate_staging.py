"""Guard the v2/models/model_candidate.pt slot.

Sync paths (deploy.sh cmd_download, _run_sync) call `clear_stale_candidate`
when the remote has no staged candidate, so that a previous run's leftover
cannot be picked up by `model_manage.keep`. Keeping this in Python (not a
shell `rm -f` sprinkled around) means the cleanup is unit-testable and the
behavior cannot silently drift per call site.

Invariant: after any sync cycle, v2/models/model_candidate.pt reflects the
*current* remote state. If the remote has no candidate, the local slot must
be empty too.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_CANDIDATE = Path("v2/models/model_candidate.pt")


def clear_stale_candidate(
    candidate_path: Path | str = DEFAULT_CANDIDATE,
    reason: str = "remote has no candidate",
) -> bool:
    """Delete the local candidate slot if it exists. Return True iff a file was removed."""
    path = Path(candidate_path)
    if not path.exists():
        print(f"  candidate slot empty, nothing to clean ({path})")
        return False
    try:
        path.unlink()
        print(f"  removed stale local candidate: {path} ({reason})")
        return True
    except OSError as exc:
        print(f"  WARNING: could not remove stale candidate {path}: {exc}")
        return False


def main() -> int:
    """CLI used by deploy.sh on the no-remote-candidate branch."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=str(DEFAULT_CANDIDATE),
                   help="Local candidate path (default: v2/models/model_candidate.pt)")
    p.add_argument("--reason", default="remote has no candidate",
                   help="Reason string logged when a stale file is removed")
    args = p.parse_args()
    clear_stale_candidate(args.path, args.reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
