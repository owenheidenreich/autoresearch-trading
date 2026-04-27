"""Pipeline integrity report generator.

Runs every Phase-0 check end-to-end and writes a markdown report to
v4/audit/PIPELINE_INTEGRITY_REPORT.md. Generates a parallel JSONL audit
record per protocol Section 9.5.

Phase 0 is complete when this report comes out GREEN (all checks pass)
on a meaningful sample of data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from v4.checks import CheckResult, run_all_integrity_checks, run_all_sanity_checks
from v4.greeks.reconcile import reconcile_optionsdx_table
from v4.ingest.fingerprint import build_id, file_sha256, new_run_id, table_sha256
from v4.ingest.optionsdx import ingest_optionsdx_file
from v4.leakage import (
    baseline_probe,
    planted_leak_test,
    shuffled_label_test,
)


@dataclass
class IntegrityReport:
    """Aggregate results of every Phase-0 integrity check."""

    started_at: str
    finished_at: str
    build_id: str
    ingest_run_id: str
    input_file_path: str
    input_file_sha256: str
    normalized_rows: int
    normalized_fingerprint: str
    deterministic_rebuild_match: bool
    sanity_checks: list[CheckResult] = field(default_factory=list)
    integrity_checks: list[CheckResult] = field(default_factory=list)
    greeks_reconciliation: CheckResult | None = None
    leak_detector_works: bool = False
    leak_detector_baseline_auc: float = 0.0
    leak_detector_shuffled_auc: float = 0.0
    leak_detector_planted_auc: float = 0.0

    @property
    def all_passed(self) -> bool:
        all_checks: list[CheckResult] = (
            list(self.sanity_checks) + list(self.integrity_checks)
        )
        if self.greeks_reconciliation is not None:
            all_checks.append(self.greeks_reconciliation)
        return (
            self.deterministic_rebuild_match
            and self.leak_detector_works
            and all(c.passed for c in all_checks)
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["all_passed"] = self.all_passed
        return d


def _run_leak_detector_self_check(*, seed: int = 0) -> tuple[bool, float, float, float]:
    """Verify the leak detector works on synthetic data.

    Returns (works, baseline_auc, shuffled_auc, planted_auc).

    The detector "works" when:
    - On a signal dataset: baseline AUC > 0.7
    - Shuffling labels crushes AUC to chance (within 0.6)
    - Planted leak pushes AUC near 1.0 (> 0.9)
    """
    rng = np.random.default_rng(seed)
    n = 400
    X = rng.normal(size=(n, 5))
    logits = 1.5 * X[:, 0] + 0.5 * X[:, 1]
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)

    baseline = baseline_probe(X, y, seed=seed)
    shuffled = shuffled_label_test(X, y, seed=seed)
    planted = planted_leak_test(X, y, leak_strength=0.95, seed=seed)

    works = (
        baseline.auc > 0.7
        and shuffled.auc < 0.6
        and planted.auc > 0.9
    )
    return works, baseline.auc, shuffled.auc, planted.auc


def run_pipeline_integrity_report(
    *,
    optionsdx_input: Path,
    output_dir: Path,
) -> IntegrityReport:
    """Run every Phase-0 check end-to-end and write the markdown report.

    Args:
        optionsdx_input: Path to an OptionsDX-format CSV (the fixture for
            Phase 0; real data once it lands)
        output_dir: Directory to write PIPELINE_INTEGRITY_REPORT.md and
            audit JSONL into. Created if missing.

    Returns the IntegrityReport (also serialized to disk).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    # 1. Ingest twice with the same run_id to verify determinism.
    fixed_run_id = new_run_id()
    result_a = ingest_optionsdx_file(optionsdx_input, ingest_run_id=fixed_run_id)
    result_b = ingest_optionsdx_file(optionsdx_input, ingest_run_id=fixed_run_id)
    sort_keys = ["event_time", "contract_id", "right"]
    fp_a = table_sha256(result_a.normalized, sort_keys=sort_keys)
    fp_b = table_sha256(result_b.normalized, sort_keys=sort_keys)
    deterministic = fp_a == fp_b

    # 2. Sanity checks
    sanity = run_all_sanity_checks(result_a.normalized)

    # 3. Integrity checks (require non-null core columns from contract Section 2.6)
    integrity = run_all_integrity_checks(
        result_a.normalized,
        required_non_null_cols=[
            "event_time", "contract_id", "root", "expiry", "strike", "right",
            "vendor_source", "ingest_run_id", "schema_version",
        ],
    )

    # 4. Greeks reconciliation. Phase 0 uses a synthetic OptionsDX fixture
    # whose Greeks were authored to be plausible but not BS-consistent; loose
    # tolerances here. Real OptionsDX data in Phase 1+ will use the tighter
    # tolerances from greeks/reconcile.py:DEFAULT_TOLERANCES.
    looser_tols = {
        "delta": 0.20,
        "gamma": 0.10,
        "vega_per_1pct": 5.0,
        "theta_per_day": 100.0,
        "rho_per_1pct": 1.0,
    }
    greeks_check, _ = reconcile_optionsdx_table(
        result_a.normalized,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        tolerances=looser_tols,
    )

    # 5. Leak-detector self-check
    works, baseline_auc, shuffled_auc, planted_auc = _run_leak_detector_self_check()

    finished_at = datetime.now(timezone.utc).isoformat()
    report = IntegrityReport(
        started_at=started_at,
        finished_at=finished_at,
        build_id=build_id(),
        ingest_run_id=fixed_run_id,
        input_file_path=str(optionsdx_input),
        input_file_sha256=file_sha256(optionsdx_input),
        normalized_rows=result_a.normalized.num_rows,
        normalized_fingerprint=fp_a,
        deterministic_rebuild_match=deterministic,
        sanity_checks=sanity,
        integrity_checks=integrity,
        greeks_reconciliation=greeks_check,
        leak_detector_works=works,
        leak_detector_baseline_auc=baseline_auc,
        leak_detector_shuffled_auc=shuffled_auc,
        leak_detector_planted_auc=planted_auc,
    )

    # 6. Write markdown report
    md_path = output_dir / "PIPELINE_INTEGRITY_REPORT.md"
    md_path.write_text(_render_markdown(report))

    # 7. Append JSONL audit record
    jsonl_path = output_dir / "integrity_runs.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(report.to_dict(), default=str) + "\n")

    return report


def _render_markdown(report: IntegrityReport) -> str:
    """Render an IntegrityReport as a human-readable markdown report."""
    lines: list[str] = []
    status = "🟢 GREEN" if report.all_passed else "🔴 RED"
    lines.append(f"# Pipeline Integrity Report — {status}")
    lines.append("")
    lines.append(f"- **Started**: {report.started_at}")
    lines.append(f"- **Finished**: {report.finished_at}")
    lines.append(f"- **Build ID**: `{report.build_id}`")
    lines.append(f"- **Ingest run ID**: `{report.ingest_run_id}`")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Path: `{report.input_file_path}`")
    lines.append(f"- SHA-256: `{report.input_file_sha256}`")
    lines.append(f"- Normalized rows emitted: {report.normalized_rows}")
    lines.append(f"- Fingerprint: `{report.normalized_fingerprint}`")
    lines.append("")
    lines.append("## Determinism")
    lines.append("")
    if report.deterministic_rebuild_match:
        lines.append("✅ Re-running ingest with the same run_id produced byte-identical output.")
    else:
        lines.append("❌ Deterministic-rebuild check FAILED. Two ingest runs produced different fingerprints.")
    lines.append("")
    lines.append("## Sanity checks")
    lines.append("")
    for c in report.sanity_checks:
        ok = "✅" if c.passed else "❌"
        lines.append(f"- {ok} **{c.name}** — {c.details or ('passed' if c.passed else 'failed')}")
    lines.append("")
    lines.append("## Integrity checks")
    lines.append("")
    for c in report.integrity_checks:
        ok = "✅" if c.passed else "❌"
        lines.append(f"- {ok} **{c.name}** — {c.details or ('passed' if c.passed else 'failed')}")
    lines.append("")
    lines.append("## Greeks reconciliation (vs OptionsDX vendor Greeks)")
    lines.append("")
    if report.greeks_reconciliation is not None:
        c = report.greeks_reconciliation
        ok = "✅" if c.passed else "❌"
        lines.append(f"- {ok} **{c.name}** — {c.details}")
    lines.append("")
    lines.append("## Leak-detector self-check")
    lines.append("")
    ok = "✅" if report.leak_detector_works else "❌"
    lines.append(f"- {ok} On synthetic signal data:")
    lines.append(f"  - baseline AUC = {report.leak_detector_baseline_auc:.3f} (expect > 0.7)")
    lines.append(f"  - shuffled-label AUC = {report.leak_detector_shuffled_auc:.3f} (expect < 0.6)")
    lines.append(f"  - planted-leak AUC = {report.leak_detector_planted_auc:.3f} (expect > 0.9)")
    lines.append("")
    lines.append("## Phase 0 exit criteria (per protocol Section 9.5)")
    lines.append("")
    if report.all_passed:
        lines.append(
            "🟢 **Pipeline-integrity report is GREEN.** All Phase-0 substrate checks pass. "
            "The substrate is ready for Phase 0.5 vendor verification — but Phase 0.5 must "
            "still gate every paid data purchase on written vendor terms."
        )
    else:
        lines.append("🔴 **Report is RED.** Investigate the failed checks above before proceeding.")
    lines.append("")
    lines.append("## What this report does NOT validate")
    lines.append("")
    lines.append(
        "- Edge or PF on real data (Phase 2A onward; Phase 0 deliberately makes no PF claim)."
    )
    lines.append(
        "- Vendor terms for OptionsDepth, Databento, IBKR (Phase 0.5)."
    )
    lines.append(
        "- Fill-model calibration vs IBKR (Phase 4 / 4.5)."
    )
    lines.append(
        "- Modern-regime data quality (Phase 1 once Databento 2024+ is pulled)."
    )
    lines.append("")
    return "\n".join(lines) + "\n"
