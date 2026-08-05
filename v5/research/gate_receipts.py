"""Content-addressed proof that a gate actually passed.

Until now, gate ordering was convention: ``knobs.assert_search_space`` took the
set of passed gates as a caller argument, so any run could claim ``{'G1','G4'}``
and be permitted.  This module replaces the claim with a receipt.  A gate pass
is recorded once, with the evidence files behind it, and a search is released
only by verifying those receipts — never by trusting the caller.

The module is model-free and network-free.  It cannot pass a gate; it can only
record and verify that one was passed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from v5.research import knobs


GATE_PASS_SCHEMA_VERSION = "v5.gate-pass-receipt.v1"
KNOWN_GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9")
REPO_ROOT = Path(__file__).resolve().parents[2]


class GateReceiptError(RuntimeError):
    """A gate-pass receipt failed verification and releases nothing."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _hash_payload(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


@dataclass(frozen=True)
class GatePassReceipt:
    """One gate, passed on one date, backed by named evidence files."""

    schema_version: str
    gate: str
    passed_on: str
    evidence_paths: tuple[str, ...]
    summary: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_valid(self, *, repo_root: Path = REPO_ROOT) -> "GatePassReceipt":
        """Refuse a receipt whose bytes, gate, date, or evidence do not hold up."""

        if self.schema_version != GATE_PASS_SCHEMA_VERSION:
            raise GateReceiptError(f"unknown gate receipt schema: {self.schema_version}")
        if _hash_payload(self.to_dict()) != self.receipt_sha256:
            raise GateReceiptError(
                f"gate receipt self-hash mismatch for {self.gate}; the bytes were edited"
            )
        if self.gate not in KNOWN_GATES:
            raise GateReceiptError(f"unknown gate: {self.gate}")
        try:
            datetime.strptime(self.passed_on, "%Y-%m-%d")
        except ValueError as exc:
            raise GateReceiptError(
                f"gate receipt for {self.gate} has an invalid pass date: {self.passed_on}"
            ) from exc
        if not self.summary.strip():
            raise GateReceiptError(f"gate receipt for {self.gate} has no summary")
        if not self.evidence_paths:
            raise GateReceiptError(f"gate receipt for {self.gate} names no evidence")
        for raw in self.evidence_paths:
            path = Path(raw)
            resolved = path if path.is_absolute() else repo_root / path
            if not resolved.is_file():
                raise GateReceiptError(
                    f"gate receipt for {self.gate} cites missing evidence: {raw}"
                )
        return self


def make_gate_pass_receipt(
    *,
    gate: str,
    passed_on: str,
    evidence_paths: Iterable[str],
    summary: str,
) -> GatePassReceipt:
    unsigned = {
        "schema_version": GATE_PASS_SCHEMA_VERSION,
        "gate": str(gate),
        "passed_on": str(passed_on),
        "evidence_paths": tuple(str(path) for path in evidence_paths),
        "summary": str(summary),
    }
    return GatePassReceipt(**unsigned, receipt_sha256=_hash_payload(unsigned))


def write_gate_pass_receipt(
    receipt: GatePassReceipt, path: Path, *, repo_root: Path = REPO_ROOT
) -> None:
    """Persist a verified receipt; an existing file is never overwritten."""

    receipt.assert_valid(repo_root=repo_root)
    if path.exists():
        raise GateReceiptError(f"refusing to overwrite gate receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_gate_pass_receipt(path: Path, *, repo_root: Path = REPO_ROOT) -> GatePassReceipt:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateReceiptError(f"gate receipt unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise GateReceiptError(f"gate receipt is not an object: {path}")
    if isinstance(payload.get("evidence_paths"), list):
        payload["evidence_paths"] = tuple(payload["evidence_paths"])
    try:
        receipt = GatePassReceipt(**payload)
    except TypeError as exc:
        raise GateReceiptError(f"gate receipt has wrong fields: {path}") from exc
    return receipt.assert_valid(repo_root=repo_root)


def released_gates(
    receipts: Iterable[GatePassReceipt], *, repo_root: Path = REPO_ROOT
) -> frozenset[str]:
    """The set of gates these receipts prove, after verifying every one."""

    return frozenset(
        receipt.assert_valid(repo_root=repo_root).gate for receipt in receipts
    )


def assert_search_space_released(
    params: Mapping[str, Any],
    *,
    gate_receipts: Iterable[GatePassReceipt],
    repo_root: Path = REPO_ROOT,
) -> None:
    """The receipt-consuming form of ``knobs.assert_search_space``.

    A search is permitted only when every knob it varies is SEARCHABLE, within
    its declared space, and released by a gate for which a *verified* receipt
    exists.  Passing an empty receipt set releases nothing.
    """

    knobs.assert_search_space(
        params, released_gates=released_gates(gate_receipts, repo_root=repo_root)
    )
