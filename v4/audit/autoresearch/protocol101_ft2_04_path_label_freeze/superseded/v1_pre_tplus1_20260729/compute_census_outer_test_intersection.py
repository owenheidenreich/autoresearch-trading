"""FT2-04 read-only intersection verifier (protected-holdout repair, 2026-07-29).

Proves the frozen census session manifest is disjoint from EVERY outer-test
slice of EVERY governed fold AND from the protected-holdout sessions (D42 as
amended 2026-07-29), and that it is exactly the governed corpus minus the
outer-test union minus the protected holdout (completeness). It computes ONLY
set membership on session-date strings drawn from already-frozen manifests. It
reads no market data, computes no census statistics, and trains nothing.

Authoritative sources (defaults):
  - fold manifest + governance: runner_plan.json `expanding_folds`
    (fold_role_map maps the outer-test role; here validation -> test) and
    `governance.protected_holdout_sessions`;
  - governed corpus: canonical_processed_session_manifest.json `included_sessions`;
  - census: the FT2-04 census_sessions.json produced in this packet.

Usage:
  python compute_census_outer_test_intersection.py            # verify defaults
  python compute_census_outer_test_intersection.py --out X.json
The pure functions (outer_test_union, protected_holdout, corpus_sessions, verify)
are import-safe for the focused test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path("/Users/gduby/Documents/autoresearch-trading")
DEFAULT_RUNNER_PLAN = (
    ROOT
    / "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_runner/runner_plan.json"
)
DEFAULT_CORPUS_MANIFEST = (
    ROOT
    / "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_preflight/canonical_processed_session_manifest.json"
)
DEFAULT_CENSUS = (
    ROOT
    / "v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/census_sessions.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sessions(role_entry: Any) -> set[str]:
    out: set[str] = set()
    for item in role_entry:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, Mapping):
            value = item.get("session") or item.get("date")
            if value:
                out.add(str(value))
    return out


def outer_test_union(runner_plan: Mapping[str, Any]) -> tuple[set[str], dict]:
    """Return (union of all outer-test sessions, per-fold detail)."""
    ef = runner_plan["expanding_folds"]
    role_map = ef["fold_governance"]["fold_role_map"]
    test_roles = [role for role, mapped in role_map.items() if mapped == "test"]
    if not test_roles:
        raise ValueError("no fold role maps to 'test'; fold manifest is ambiguous")
    union: set[str] = set()
    per_fold: dict[str, list[str]] = {}
    for fold, roles in ef["fold_sessions"].items():
        fold_test: set[str] = set()
        for role in test_roles:
            if role not in roles:
                raise ValueError(f"fold {fold} missing declared test role {role!r}")
            fold_test |= _sessions(roles[role])
        per_fold[fold] = sorted(fold_test)
        union |= fold_test
    return union, {"test_roles": test_roles, "per_fold": per_fold}


def protected_holdout(runner_plan: Mapping[str, Any]) -> set[str]:
    """Protected-holdout sessions from the governance manifest (D42 exclusion)."""
    gov = runner_plan.get("governance", {})
    if "protected_holdout_sessions" not in gov:
        raise ValueError("governance.protected_holdout_sessions missing; cannot prove holdout firewall")
    return _sessions(gov["protected_holdout_sessions"])


def corpus_sessions(corpus_manifest: Mapping[str, Any]) -> set[str]:
    return _sessions(corpus_manifest["included_sessions"])


def verify(
    runner_plan: Mapping[str, Any],
    corpus_manifest: Mapping[str, Any],
    census: Mapping[str, Any],
) -> dict:
    """Pure verification over already-frozen manifests (D42, amended 2026-07-29)."""
    census_sessions = set(census["census_sessions"])
    corpus = corpus_sessions(corpus_manifest)
    test_union, detail = outer_test_union(runner_plan)
    holdout = protected_holdout(runner_plan)

    test_intersection = sorted(census_sessions & test_union)
    holdout_intersection = sorted(census_sessions & holdout)
    # Completeness reference: corpus minus BOTH exclusion sets (D42 amended).
    expected_census = corpus - test_union - holdout
    missing_from_census = sorted(expected_census - census_sessions)
    extra_in_census = sorted(census_sessions - expected_census)
    census_outside_corpus = sorted(census_sessions - corpus)

    disjoint_test = not test_intersection
    disjoint_holdout = not holdout_intersection
    complete = not missing_from_census and not extra_in_census
    within_corpus = not census_outside_corpus
    test_within_corpus = test_union.issubset(corpus)
    holdout_within_corpus = holdout.issubset(corpus)

    return {
        "disjoint_census_and_outer_test": disjoint_test,
        "disjoint_census_and_protected_holdout": disjoint_holdout,
        "census_equals_corpus_minus_outer_test_minus_holdout": complete,
        "census_within_corpus": within_corpus,
        "outer_test_within_corpus": test_within_corpus,
        "protected_holdout_within_corpus": holdout_within_corpus,
        "passed": bool(
            disjoint_test
            and disjoint_holdout
            and complete
            and within_corpus
            and test_within_corpus
            and holdout_within_corpus
        ),
        "counts": {
            "governed_corpus": len(corpus),
            "outer_test_union": len(test_union),
            "protected_holdout": len(holdout),
            "census": len(census_sessions),
            "outer_test_intersection": len(test_intersection),
            "holdout_intersection": len(holdout_intersection),
        },
        "outer_test_intersection_sessions": test_intersection,
        "holdout_intersection_sessions": holdout_intersection,
        "missing_from_census": missing_from_census,
        "extra_in_census": extra_in_census,
        "census_outside_corpus": census_outside_corpus,
        "test_roles": detail["test_roles"],
        "outer_test_sessions_by_fold": detail["per_fold"],
        "protected_holdout_sessions": sorted(holdout),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-plan", type=Path, default=DEFAULT_RUNNER_PLAN)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    runner_plan = json.loads(args.runner_plan.read_text())
    corpus_manifest = json.loads(args.corpus_manifest.read_text())
    census = json.loads(args.census.read_text())

    result = verify(runner_plan, corpus_manifest, census)
    report = {
        "schema_version": "Protocol101FT204IntersectionProofV2",
        "node": "FT2-04-PATH-LABEL-FREEZE",
        "claim": "census sessions intersect NO outer-test slice of any fold AND NO protected-holdout session (D42, amended 2026-07-29)",
        "sources": {
            "runner_plan": str(args.runner_plan),
            "runner_plan_sha256": sha256_file(args.runner_plan),
            "corpus_manifest": str(args.corpus_manifest),
            "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
            "census": str(args.census),
            "census_sha256": sha256_file(args.census),
        },
        "result": result,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
