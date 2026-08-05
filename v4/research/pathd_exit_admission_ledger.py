"""Signed, fail-closed Path-D EXIT feature-admission law.

The entry side has had a signed admission ledger since Phase 0. The exit side
had a live-twin inventory but no certification path at all, so
``admitted_feature_matrix`` raised ``feature_not_in_admission_ledger`` on every
one of the 49 exit features: the exit model could not be lawfully fitted even
in principle.

This module is the exit peer of ``pathd_feature_admission_ledger``. It reuses
that module's signing, receipt-hashing and root-blocker primitives rather than
restating them, so the two ledgers cannot drift in their notion of what
"blocked by a parent" means.

Design note on shared substrate. The exit quote block is the SAME Databento
OPRA substrate the entry families read; it is not an independent source. So
``exit.opra_cbbo_quote.v1`` declares the entry 1-second rolling family as its
parent, and ``exit.causal_account_state.v1`` declares the entry account-state
family as its. One consequence worth stating plainly: the Track-A multi-session
arrival capture unblocks the exit OPRA families at the same time it unblocks
the entry ones, and Track C's execution-report arrival clock is likewise shared.
Neither needs to be measured twice.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from v4.research.pathd_feature_admission_ledger import (
    ADMITTED,
    BARRED,
    REPO_ROOT,
    SCHEMA_VERSION,
    AdmissionLedgerError,
    ledger_sha256,
    root_blocking_parents,
    sha256_file,
)
from v4.research.pathd_feature_live_twin import EXIT49_FEATURE_NAMES

# The fitted exit contract is EXIT47_CORRECTED + these five, so the ledger must
# cover them or enforcement raises NOT_IN_EXIT_LEDGER -- an error that reads
# like a wiring bug rather than "this feature was never certified".
#
# They are time derivatives of current_net_pnl_dollars, so they inherit the
# account-state arrival clock exactly and belong to that family. Certifying
# them separately would imply an independent source that does not exist.
PNL_VELOCITY_FEATURE_NAMES: tuple[str, ...] = (
    "pnl_velocity_1s_dollars",
    "pnl_velocity_5s_dollars",
    "pnl_velocity_15s_dollars",
    "pnl_velocity_30s_dollars",
    "pnl_velocity_60s_dollars",
)
EXIT_LEDGER_FEATURE_NAMES: tuple[str, ...] = tuple(EXIT49_FEATURE_NAMES) + PNL_VELOCITY_FEATURE_NAMES

EXIT_SCHEMA_VERSION = "pathd.exit-feature-admission-ledger.v1"
DEFAULT_EXIT_LEDGER_PATH = (
    REPO_ROOT
    / "v4/audit/autoresearch/pathd_exit_feature_certification_2026_08_04/exit_feature_admission_ledger.json"
)

# Family decomposition mirrors the slicing already frozen in
# pathd_feature_live_twin, so the two cannot disagree about which feature
# belongs to which source.
_QUOTE = EXIT49_FEATURE_NAMES[:22]
_MINUTE_VOLUME = (EXIT49_FEATURE_NAMES[22],)
_OPEN_INTEREST = (EXIT49_FEATURE_NAMES[23],)
_OFFICIAL_SPX = EXIT49_FEATURE_NAMES[24:29]
_SELF_GREEKS = EXIT49_FEATURE_NAMES[29:36]
_CAUSAL_STATE = EXIT49_FEATURE_NAMES[36:]

EXIT_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "exit.opra_cbbo_quote.v1": _QUOTE,
    "exit.official_spx_context.v1": _OFFICIAL_SPX,
    "exit.self_computed_greeks.v1": _SELF_GREEKS,
    "exit.causal_account_state.v1": _CAUSAL_STATE + PNL_VELOCITY_FEATURE_NAMES,
}
EXIT_PERMANENT_BARRED_CONTRACT_ID = "exit.barred_no_live_twin.v1"
EXIT_PERMANENT_BARRED_FEATURES = _OPEN_INTEREST + _MINUTE_VOLUME

# Cross-namespace parents. The exit families do not own independent substrate;
# they read the same feeds the entry families read, so they inherit the entry
# families' unmet arrival receipts rather than inventing their own.
EXIT_PARENTS: Mapping[str, tuple[str, ...]] = {
    "exit.opra_cbbo_quote.v1": ("entry.opra_cbbo1s_rolling.v1",),
    "exit.self_computed_greeks.v1": ("exit.opra_cbbo_quote.v1",),
    "exit.causal_account_state.v1": ("entry.causal_account_state.v1",),
}

EXIT_REQUIRED_RECEIPTS: Mapping[str, tuple[str, ...]] = {
    "exit.opra_cbbo_quote.v1": (
        "multi-session local receipt-latency distribution",
        "one-second rolling-window shared implementation hash",
        "no-update and reconnect mutation tests",
    ),
    "exit.official_spx_context.v1": (
        "live ThetaData completed-minute adapter observation",
        "historical/live official SPX value identity",
    ),
    "exit.self_computed_greeks.v1": (
        "historical/live golden-vector identity",
        "shared solver constants hash",
    ),
    "exit.causal_account_state.v1": (
        "execution-report arrival clock",
        "historical/live ledger transition identity",
        "serial replay parity",
        "mutate-future invariance",
    ),
}


def _canonical_features() -> tuple[str, ...]:
    ordered: list[str] = []
    for names in EXIT_FAMILIES.values():
        ordered.extend(names)
    ordered.extend(EXIT_PERMANENT_BARRED_FEATURES)
    return tuple(ordered)


def _validate_partition() -> None:
    """Every EXIT49 feature must land in exactly one family.

    A feature that falls through the slicing would be silently absent from the
    ledger, and an absent feature is indistinguishable from an unknown one at
    the enforcement boundary -- it would raise, but for the wrong reason, and
    the fix would look like "add it to the model" rather than "certify it".
    """

    covered = _canonical_features()
    if len(covered) != len(set(covered)):
        raise AdmissionLedgerError("exit feature families overlap")
    if set(covered) != set(EXIT_LEDGER_FEATURE_NAMES):
        missing = sorted(set(EXIT_LEDGER_FEATURE_NAMES) - set(covered))
        extra = sorted(set(covered) - set(EXIT_LEDGER_FEATURE_NAMES))
        raise AdmissionLedgerError(
            f"exit family partition drift: missing={missing} extra={extra}"
        )


def generate_exit_ledger(
    *,
    ledger_path: Path = DEFAULT_EXIT_LEDGER_PATH,
    entry_admitted_families: Iterable[str] = (),
    admit: Mapping[str, Mapping[str, Any]] | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Build the exit ledger. Nothing is ADMITTED without its receipts.

    ``entry_admitted_families`` carries the entry ledger's admitted set so
    cross-namespace parents resolve. Until Track A and Track C land, that set
    contains only the contract-clock family, so every exit family is barred --
    which is the correct and expected initial state, not a failure.
    """

    _validate_partition()
    if not allow_overwrite and ledger_path.exists() and ledger_path.stat().st_size > 0:
        try:
            verify_exit_ledger(ledger_path)
        except AdmissionLedgerError:
            pass
        else:
            raise AdmissionLedgerError(
                f"refusing to overwrite a verifying signed exit ledger: {ledger_path}"
            )
    admitted = set(entry_admitted_families)
    certifications = dict(admit or {})
    unknown = sorted(set(certifications) - set(EXIT_FAMILIES))
    if unknown:
        raise AdmissionLedgerError(f"unknown exit family in admit: {unknown}")
    features: list[dict[str, Any]] = []
    for contract_id, names in EXIT_FAMILIES.items():
        blockers = root_blocking_parents_across(contract_id, admitted)
        certification = certifications.get(contract_id)
        # A family may be admitted only when its own receipts are supplied AND
        # nothing upstream is still blocked. Admitting past an unadmitted parent
        # is precisely the defect that put implied_spot, IV and the greeks into
        # the signed18 look-ahead class on the entry side.
        if certification is not None and not blockers:
            receipts = list(certification["receipts"])
            clock_ms = certification["availability_clock_ms"]
            if not receipts or clock_ms is None:
                raise AdmissionLedgerError(
                    f"exit family {contract_id} cannot be admitted without receipts and a clock"
                )
            for name in names:
                features.append(
                    {
                        "name": name,
                        "family": contract_id,
                        "contract_id": contract_id,
                        "status": ADMITTED,
                        "receipts": receipts,
                        "availability_clock_ms": clock_ms,
                        "tolerance": float(certification.get("tolerance", 0.0)),
                        "barred_reason": None,
                    }
                )
            admitted.add(contract_id)
            continue
        if blockers:
            reason = "parent_family_not_admitted:" + ",".join(blockers)
        elif certification is not None:
            reason = "exit_required_receipt_failed"
        else:
            reason = "missing_required_receipts:" + "|".join(
                EXIT_REQUIRED_RECEIPTS[contract_id]
            )
        for name in names:
            features.append(
                {
                    "name": name,
                    "family": contract_id,
                    "contract_id": contract_id,
                    "status": BARRED,
                    "receipts": [],
                    "availability_clock_ms": None,
                    "tolerance": 0.0,
                    "barred_reason": reason,
                }
            )
    for name in EXIT_PERMANENT_BARRED_FEATURES:
        features.append(
            {
                "name": name,
                "family": EXIT_PERMANENT_BARRED_CONTRACT_ID,
                "contract_id": EXIT_PERMANENT_BARRED_CONTRACT_ID,
                "status": BARRED,
                "receipts": [],
                "availability_clock_ms": None,
                "tolerance": 0.0,
                "barred_reason": "permanently_barred_no_matching_live_twin",
            }
        )
    unsigned = {
        "schema_version": EXIT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_admitted_families": sorted(admitted),
        "features": sorted(features, key=lambda row: row["name"]),
    }
    payload = dict(unsigned)
    payload["ledger_sha256"] = ledger_sha256(unsigned)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_exit_ledger(ledger_path)
    return payload


def root_blocking_parents_across(contract_id: str, admitted_families: Iterable[str]) -> list[str]:
    """Root blockers for an exit family, walking the cross-namespace graph.

    Delegates the traversal semantics to the entry module so "root blocker"
    means exactly one thing across both ledgers.
    """

    from v4.research import pathd_feature_admission_ledger as entry_ledger

    merged = dict(entry_ledger.PARENTS)
    merged.update(EXIT_PARENTS)
    original = entry_ledger.PARENTS
    try:
        entry_ledger.PARENTS = merged
        return root_blocking_parents(contract_id, set(admitted_families))
    finally:
        entry_ledger.PARENTS = original


def verify_exit_ledger(path: Path = DEFAULT_EXIT_LEDGER_PATH) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise AdmissionLedgerError(f"exit admission ledger missing_or_empty:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AdmissionLedgerError(f"exit admission ledger unreadable:{path}") from exc
    if payload.get("schema_version") != EXIT_SCHEMA_VERSION:
        raise AdmissionLedgerError("exit admission ledger schema mismatch")
    if payload.get("ledger_sha256") != ledger_sha256(payload):
        raise AdmissionLedgerError("exit admission ledger_sha256 mismatch")
    rows = payload.get("features")
    if not isinstance(rows, list) or not rows:
        raise AdmissionLedgerError("exit admission ledger has no features")
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("name", ""))
        if not name or name in seen:
            raise AdmissionLedgerError(f"duplicate_or_empty_exit_feature:{name}")
        seen.add(name)
        if row.get("status") not in {ADMITTED, BARRED}:
            raise AdmissionLedgerError(f"invalid_exit_status:{name}:{row.get('status')}")
        for receipt in row.get("receipts") or []:
            receipt_path = Path(str(receipt.get("path", "")))
            if not receipt_path.is_absolute():
                receipt_path = REPO_ROOT / receipt_path
            if not receipt_path.is_file() or sha256_file(receipt_path) != receipt.get("sha256"):
                raise AdmissionLedgerError(f"exit_receipt_hash_mismatch:{name}:{receipt_path}")
        if row.get("status") == ADMITTED:
            if not row.get("receipts") or row.get("availability_clock_ms") is None:
                raise AdmissionLedgerError(f"admitted_exit_feature_missing_receipt_or_clock:{name}")
        elif not row.get("barred_reason"):
            raise AdmissionLedgerError(f"barred_exit_feature_missing_reason:{name}")
    if seen != set(EXIT_LEDGER_FEATURE_NAMES):
        raise AdmissionLedgerError("exit ledger does not cover the frozen exit feature set")
    return payload


def assert_exit_feature_matrix_admitted(
    feature_names: Iterable[str], *, ledger_path: Path = DEFAULT_EXIT_LEDGER_PATH
) -> tuple[str, ...]:
    payload = verify_exit_ledger(ledger_path)
    index = {str(row["name"]): row for row in payload["features"]}
    names = tuple(str(name) for name in feature_names)
    if not names:
        raise AdmissionLedgerError("exit feature matrix cannot be empty")
    problems: list[str] = []
    for name in names:
        row = index.get(name)
        if row is None:
            problems.append(f"{name}:NOT_IN_EXIT_LEDGER")
        elif row["status"] != ADMITTED:
            problems.append(f"{name}:{row['status']}")
    if problems:
        raise AdmissionLedgerError(
            "exit feature matrix contains non-admitted features: " + ",".join(sorted(problems))
        )
    return names


def admitted_exit_feature_matrix(
    frame: pd.DataFrame,
    feature_names: Iterable[str],
    *,
    ledger_path: Path = DEFAULT_EXIT_LEDGER_PATH,
) -> pd.DataFrame:
    names = assert_exit_feature_matrix_admitted(feature_names, ledger_path=ledger_path)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise AdmissionLedgerError(f"exit feature matrix missing columns: {sorted(missing)}")
    return frame.loc[:, list(names)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_EXIT_LEDGER_PATH)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    if args.verify:
        payload = verify_exit_ledger(args.ledger)
        counts: dict[str, int] = {}
        for row in payload["features"]:
            counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
        print(json.dumps({"ledger": str(args.ledger), "counts": counts, "status": "EXIT_LEDGER_VERIFIED"}, sort_keys=True))
        return
    from v4.research.pathd_feature_admission_ledger import verify_ledger

    entry = verify_ledger()
    by_family: dict[str, set[str]] = {}
    for row in entry["features"]:
        by_family.setdefault(str(row["contract_id"]), set()).add(str(row["status"]))
    admitted = {family for family, statuses in by_family.items() if statuses == {ADMITTED}}
    payload = generate_exit_ledger(
        ledger_path=args.ledger,
        entry_admitted_families=admitted,
        allow_overwrite=args.allow_overwrite,
    )
    print(json.dumps({"ledger": str(args.ledger), "ledger_sha256": payload["ledger_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
