"""Write the Stage 1 pre-commit review packet."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


SECRET_PATTERN = r"API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE KEY|BEGIN RSA|BEGIN OPENSSH|DU[0-9]{3,}|account_id"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--test-output", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = read_manifest(args.full_manifest)
    staged = git(args.snapshot, "diff", "--cached", "--name-only").splitlines()
    diff_stat = git(args.snapshot, "diff", "--cached", "--stat", "--summary")
    branch = git(args.snapshot, "symbolic-ref", "--short", "HEAD").strip()
    secret_files = secret_scan(args.snapshot)
    staged_large = large_staged_files(args.snapshot, staged)
    category_counts, category_bytes, action_counts = summarize(rows)
    tests = args.test_output.read_text(encoding="utf-8", errors="replace") if args.test_output and args.test_output.exists() else "No test output file provided."
    recommendation = "do not commit" if secret_files else "commit"

    lines = [
        "# Stage 1 Pre-Commit Review",
        "",
        f"- Branch name: `{branch}`",
        f"- Source workspace: `{args.source}`",
        f"- Clean worktree path: `{args.snapshot}`",
        f"- Files staged: `{len(staged)}`",
        f"- Evidence bundle path: `{args.bundle}`",
        f"- Evidence bundle sha256: `{sha256_file(args.bundle) if args.bundle.exists() else 'MISSING'}`",
        f"- Explicit recommendation: `{recommendation}`",
        "",
        "## Staged File List",
        "",
    ]
    lines.extend(f"- `{path}`" for path in staged)
    lines.extend(["", "## Staged Diff Stat", "", "```text", diff_stat.strip() or "(no staged diff)", "```", ""])
    lines.extend(["## Files Excluded From Git By Category", "", "| category | file_count | total_bytes |", "|---|---:|---:|"])
    for category in sorted(category_counts):
        lines.append(f"| `{category}` | {category_counts[category]} | {category_bytes[category]} |")
    lines.extend(
        [
            "",
            "## Exclusion Action Counts",
            "",
            f"- Bundle-only count: `{action_counts.get('bundle_only', 0)}`",
            f"- Excluded-entirely count: `{action_counts.get('excluded_entirely', 0)}`",
            f"- Sensitive-excluded count: `{action_counts.get('sensitive_excluded', 0)}`",
            "",
            "## Secret Scan Filename-Only Results",
            "",
        ]
    )
    if secret_files:
        lines.extend(f"- `{path}`" for path in secret_files)
    else:
        lines.append("- No candidate files reported by filename-only scan.")
    lines.extend(["", "## Large-File Summary", ""])
    if staged_large:
        lines.append("Staged files larger than 1 MB:")
        lines.extend(f"- `{path}`: {size} bytes" for path, size in staged_large)
    else:
        lines.append("- No staged files larger than 1 MB.")
    lines.extend(
        [
            "",
            "Excluded large files are summarized by category in the table above. Exact sensitive/runtime/paid-data paths remain in the local-only full manifest.",
            "",
            "## Runtime And Account-Risk Files Excluded",
            "",
            "- `v4/runtime/**` excluded from Git.",
            "- `v4/logs/paper_trading/**` excluded from Git.",
            "- `.env` and `v4/.env` excluded from Git and evidence bundle.",
            "- Runtime/account-risk exact paths and hashes are local-only.",
            "",
            "## Paid-Data Files Excluded",
            "",
            "- `data/**` excluded from Git and evidence bundle.",
            "- `v4/raw/**` excluded from Git and evidence bundle.",
            "- `v4/normalized*/**` excluded from Git and evidence bundle.",
            "- Paid-data exact paths and hashes are local-only.",
            "",
            "## Tests Run",
            "",
            "```text",
            trim_tests(tests),
            "```",
            "",
            "## Failures",
            "",
        ]
    )
    lines.append(f"- Secret scan candidates: `{len(secret_files)}`.")
    if "failed" in tests.lower() or "error" in tests.lower():
        lines.append("- Test output contains failure/error text; review before commit.")
    else:
        lines.append("- No test failure text detected in captured output.")
    lines.extend(["", "## Recommendation", "", f"`{recommendation}`"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True, stderr=subprocess.STDOUT)


def secret_scan(cwd: Path) -> list[str]:
    cmd = [
        "rg",
        "-i",
        "-l",
        SECRET_PATTERN,
        "--glob",
        "!research_ops/bootstrap/*FULL*",
        "--glob",
        "!v4/logs/**",
        "--glob",
        "!v4/runtime/**",
        "--glob",
        "!.env",
        "--glob",
        "!v4/.env",
        ".",
    ]
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode not in {0, 1}:
        return [f"secret_scan_error_exit_{proc.returncode}"]
    return sorted(line.strip().lstrip("./") for line in proc.stdout.splitlines() if line.strip())


def large_staged_files(snapshot: Path, staged: list[str]) -> list[tuple[str, int]]:
    out = []
    for rel in staged:
        path = snapshot / rel
        if path.is_file():
            size = path.stat().st_size
            if size > 1_000_000:
                out.append((rel, size))
    return sorted(out, key=lambda item: item[1], reverse=True)


def read_manifest(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, object]]) -> tuple[Counter[str], dict[str, int], Counter[str]]:
    category_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    category_bytes: dict[str, int] = defaultdict(int)
    for row in rows:
        category = str(row["category"])
        action = str(row["bundle_action"])
        category_counts[category] += 1
        action_counts[action] += 1
        category_bytes[category] += int(row["size_bytes"])
    return category_counts, category_bytes, action_counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trim_tests(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= 80:
        return text.strip()
    return "\n".join(lines[:40] + ["..."] + lines[-40:])


if __name__ == "__main__":
    raise SystemExit(main())
