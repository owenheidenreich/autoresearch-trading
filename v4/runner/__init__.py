"""v4.runner — orchestration for pipeline-integrity checks.

run_pipeline_integrity_report() runs every Phase-0 check end-to-end and
writes the markdown report to v4/audit/PIPELINE_INTEGRITY_REPORT.md.
"""
from .integrity import run_pipeline_integrity_report

__all__ = ["run_pipeline_integrity_report"]
