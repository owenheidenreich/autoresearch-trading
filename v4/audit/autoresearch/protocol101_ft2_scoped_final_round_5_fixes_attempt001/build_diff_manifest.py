#!/usr/bin/env python3
"""Build the rerun002-to-scoped-round delta manifest for fresh-seat review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PACKET_ROOT = REPO / "v4/audit/autoresearch"
AUTHORITY = REPO / (
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
GRAPH = REPO / (
    "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2.json"
)
BUILDER = REPO / "v4/scripts/run_protocol101_ft2_05_opportunity_census.py"
DIRECT_TEST_UPDATES = {
    REPO / "v4/tests/test_protocol101_ft2_05_opportunity_census.py": (
        "fbca2dd9d77dbb9396b3bed7063f8d73ddae74a45dbc3e7a902fea5184dcaaa6"
    ),
    REPO / "v4/tests/test_protocol101_ft2_05_opportunity_census_v2.py": (
        "8236aa2b14a5eba3b582a2cf313c203e7ad9b24b32970cdcb9e75fa433798fba"
    ),
}
PACKETS = {
    "FT2-04": PACKET_ROOT / "protocol101_ft2_04_path_label_freeze",
    "FT2-05": PACKET_ROOT / "protocol101_ft2_05_opportunity_census",
    "FT2-08": PACKET_ROOT / "protocol101_ft2_08_data_tensor_label_contract",
    "FT2-10": PACKET_ROOT / "protocol101_ft2_10_entry_science_contract",
    "FT2-11": PACKET_ROOT / "protocol101_ft2_11_evidence_statistics_contract",
}
PARENT_RECEIPTS = {
    "FT2-04": "c763dee85293d73a4f367627e8f483a209565d3f486f82fb5b1c17f3ff611775",
    "FT2-05": "a252feb2007b2aece77f377461bc6fa5a882e929bd2f7d3f23a47c07401cfdce",
    "FT2-08": "731cc6fb4c44bd0650d658e71fffbdac7c3e0f5d68701b2e05b8576b014b078c",
    "FT2-10": "c04a591fd3034a2caba756e8f9dc1d7f5c24c397170894b0b6b0eca7a083fa6f",
    "FT2-11": "bdb52c4f9178da1cab75ec4a80bd13c1db8c0074c1c398f380729d1df80c5ef9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def changed_pointers(before: Any, after: Any, pointer: str = "") -> list[str]:
    if type(before) is not type(after):
        return [pointer or "/"]
    if isinstance(before, dict):
        out: list[str] = []
        for key in sorted(set(before) | set(after)):
            child = f"{pointer}/{json_pointer_escape(str(key))}"
            if key not in before or key not in after:
                out.append(child)
            else:
                out.extend(changed_pointers(before[key], after[key], child))
        return out
    if isinstance(before, list):
        return [] if before == after else [pointer or "/"]
    return [] if before == after else [pointer or "/"]


def inventory(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "superseded" not in path.relative_to(root).parts
        and "__pycache__" not in path.relative_to(root).parts
    }


def compare_file(
    *,
    repo_path: Path,
    before: Path | None,
    after: Path | None,
    category: str,
) -> dict[str, Any] | None:
    before_hash = sha256(before) if before is not None and before.is_file() else None
    after_hash = sha256(after) if after is not None and after.is_file() else None
    if before_hash == after_hash:
        return None
    item: dict[str, Any] = {
        "path": str(repo_path.relative_to(REPO)),
        "category": category,
        "before_sha256": before_hash,
        "after_sha256": after_hash,
        "change": (
            "added"
            if before_hash is None
            else "removed"
            if after_hash is None
            else "modified"
        ),
    }
    if (
        before is not None
        and after is not None
        and before.suffix == ".json"
        and after.suffix == ".json"
        and before.is_file()
        and after.is_file()
    ):
        try:
            item["changed_json_pointers"] = changed_pointers(
                json.loads(before.read_text(encoding="utf-8")),
                json.loads(after.read_text(encoding="utf-8")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            item["changed_json_pointers"] = ["<binary_or_noncanonical_json>"]
    return item


def main() -> int:
    changes: list[dict[str, Any]] = []
    authority_before = HERE / "baseline" / "authority_pre_round.md"
    graph_before = HERE / "baseline" / "graph_v2_pre_round.json"
    for repo_path, before, after, category in (
        (
            AUTHORITY,
            authority_before,
            AUTHORITY,
            "owner_authority_amendment",
        ),
        (GRAPH, graph_before, GRAPH, "graph_outcome_edge"),
    ):
        item = compare_file(
            repo_path=repo_path,
            before=before,
            after=after,
            category=category,
        )
        if item:
            changes.append(item)

    for goal, packet in PACKETS.items():
        baseline = (
            packet
            / "superseded"
            / "v3_pre_scoped_final_round_20260730"
        )
        before_files = inventory(baseline)
        after_files = inventory(packet)
        for relative in sorted(set(before_files) | set(after_files)):
            item = compare_file(
                repo_path=packet / relative,
                before=before_files.get(relative),
                after=after_files.get(relative),
                category=f"{goal}_packet",
            )
            if item:
                changes.append(item)

    changes.append(
        {
            "path": str(BUILDER.relative_to(REPO)),
            "category": "FT2-05_builder",
            "change": "modified",
            "before_sha256": (
                "1871312e476c64d73db8873e8ed842a70efb9598ba858cc13d2d373e1294ace6"
            ),
            "after_sha256": sha256(BUILDER),
            "changed_regions": [
                "authority and schema constants",
                "D48 reference membership uses time-t intent eligibility",
                "v3-to-v4 impact writer",
                "dual rejection-denominator disclosure",
                "row-level v2-to-v3-to-v4 transition audit",
                "v4 receipt chain",
            ],
        }
    )
    for path, before_hash in DIRECT_TEST_UPDATES.items():
        changes.append(
            {
                "path": str(path.relative_to(REPO)),
                "category": "direct_census_test_consequence",
                "change": "modified",
                "before_sha256": before_hash,
                "after_sha256": sha256(path),
                "changed_regions": [
                    "v4 fixture now carries canonical time-t intent and selected-only t+1 recheck fields"
                    if path.name.endswith("census.py")
                    else "preserved-v2 regression imports the preserved v2 builder rather than the current v4 builder"
                ],
            }
        )

    round_added = [
        path
        for path in sorted(HERE.iterdir())
        if path.is_file()
        and path.name not in {
            "aggregate_receipt.json",
            "diff_manifest.json",
        }
    ]
    for path in round_added:
        changes.append(
            {
                "path": str(path.relative_to(REPO)),
                "category": "scoped_round_evidence",
                "change": "added",
                "before_sha256": None,
                "after_sha256": sha256(path),
            }
        )

    payload = {
        "schema_version": "Protocol101FT2ScopedFinalRoundDiffManifestV1",
        "baseline": {
            "rerun002_receipt_sha256": (
                "17cc72ea85c629ae9a2d23059ae13020782464705ee6049f577c9936dd2579ac"
            ),
            "authority_sha256": (
                "2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a"
            ),
            "graph_sha256": (
                "b06a26be59307c130da84f2dc5b6f3224c272e6c4093e83abd5bc0b280ca6d09"
            ),
            "packet_receipt_sha256": PARENT_RECEIPTS,
        },
        "current": {
            "authority_sha256": sha256(AUTHORITY),
            "graph_sha256": sha256(GRAPH),
        },
        "scope_law": "Only the five owner-authorized findings and their direct hash/test/receipt consequences changed; the remaining reviewed science is frozen.",
        "changed_file_count": len(changes),
        "changed_files": sorted(changes, key=lambda item: item["path"]),
        "post_manifest_seal_excluded": {
            "path": "v4/audit/autoresearch/protocol101_ft2_scoped_final_round_5_fixes_attempt001/aggregate_receipt.json",
            "reason": "The aggregate receipt is written after this manifest and pins the manifest hash; excluding its self-dependent hash avoids a circular seal."
        },
        "delta_review_instruction": "Review the listed changed JSON pointers and direct consequences only. Generated census/checkpoint files are evidence outputs; use the transition audit for their semantic delta.",
    }
    (HERE / "diff_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"changed_file_count": len(changes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
