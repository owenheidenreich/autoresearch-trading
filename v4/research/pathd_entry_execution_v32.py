"""Deterministic corrected-v3.2 entry/control replay producers.

The producers in this module consume typed, self-hashed causal opportunities and
derive every economic value from their executable ask/bid path.  Callers cannot
provide PnL, terminal-journal hashes, pass flags, or selected exits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Sequence

from v4.research import pathd_entry_exit as prereg


CONTRACT_MULTIPLIER = 100
STARTING_EQUITY_MICROS = 10_000_000_000
D48_FRACTION = 0.05


def _hash(value: Any) -> str:
    return prereg.stable_hash(value)


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class CausalEntryOpportunityV1:
    schema_version: str
    outer_fold: int
    session: str
    decision_time_ns: int
    source_neutral_contract_id: str
    right_code: str
    moneyness_class: str
    premium_band: str
    entry_ask_micros: int
    causal_bid_path: tuple[tuple[int, int], ...]
    hgb_enter: bool
    neural_enter: bool
    p5_rank: int | None
    nearest_atm_rank: int | None
    signed_score_micros: int
    opportunity_sha256: str


@dataclass(frozen=True)
class ReplayTradeV1:
    opportunity_sha256: str
    session: str
    policy_id: str
    exit_policy_id: str
    entry_time_ns: int
    exit_elapsed_seconds: int
    entry_ask_micros: int
    exit_bid_micros: int
    fee_micros: int
    net_pnl_micros: int
    terminal_reason: str
    trade_sha256: str


@dataclass(frozen=True)
class ReplayJournalV1:
    schema_version: str
    outer_fold: int
    policy_id: str
    exit_policy_id: str
    fee_micros: int
    sessions: tuple[str, ...]
    trades: tuple[dict[str, Any], ...]
    session_pnl_micros: tuple[int, ...]
    session_ending_equity_micros: tuple[int, ...]
    skipped_overlap_count: int
    skipped_unaffordable_count: int
    flat_close_every_session: bool
    journal_sha256: str


def validate_opportunity(value: CausalEntryOpportunityV1) -> CausalEntryOpportunityV1:
    if type(value) is not CausalEntryOpportunityV1:
        raise TypeError("entry opportunity must be CausalEntryOpportunityV1")
    semantic = asdict(value)
    digest = semantic.pop("opportunity_sha256")
    path = tuple(value.causal_bid_path)
    if (
        value.schema_version != "pathd.causal_entry_opportunity.v1"
        or value.outer_fold not in range(1, 6)
        or not value.session
        or type(value.decision_time_ns) is not int
        or value.decision_time_ns < 0
        or not value.source_neutral_contract_id
        or value.right_code not in {"C", "P"}
        or value.moneyness_class not in {"ATM", "NEAR", "WING"}
        or value.premium_band not in {
            "tiny_lt_1", "small_1_3", "medium_3_8", "large_8_20", "very_large_ge_20"
        }
        or type(value.entry_ask_micros) is not int
        or value.entry_ask_micros <= 0
        or not path
        or any(
            type(row) not in (tuple, list)
            or len(row) != 2
            or type(row[0]) is not int
            or type(row[1]) is not int
            or row[0] < 0
            or row[1] < 0
            for row in path
        )
        or tuple(row[0] for row in path) != tuple(sorted({row[0] for row in path}))
        or type(value.hgb_enter) is not bool
        or type(value.neural_enter) is not bool
        or (value.p5_rank is not None and (type(value.p5_rank) is not int or value.p5_rank < 0))
        or (
            value.nearest_atm_rank is not None
            and (type(value.nearest_atm_rank) is not int or value.nearest_atm_rank < 0)
        )
        or type(value.signed_score_micros) is not int
        or digest != _hash(semantic)
    ):
        raise ValueError("causal entry opportunity drift")
    return value


def seal_opportunity(**values: Any) -> CausalEntryOpportunityV1:
    semantic = {"schema_version": "pathd.causal_entry_opportunity.v1", **values}
    return validate_opportunity(
        CausalEntryOpportunityV1(
            **semantic, opportunity_sha256=_hash(semantic)
        )
    )


def _exit_quote(
    opportunity: CausalEntryOpportunityV1, exit_policy_id: str
) -> tuple[int, int, str]:
    path = tuple(opportunity.causal_bid_path)
    if exit_policy_id == "EXIT_IMMEDIATE":
        elapsed, bid = path[0]
        return elapsed, bid, "EXIT_IMMEDIATE"
    if exit_policy_id == "HOLD_TO_FLAT":
        elapsed, bid = path[-1]
        return elapsed, bid, "FORCED_FLAT"
    if exit_policy_id.startswith("TIME_"):
        seconds = int(exit_policy_id.split("_", 1)[1])
        for elapsed, bid in path:
            if elapsed >= seconds:
                return elapsed, bid, "TIME_EXIT"
        elapsed, bid = path[-1]
        return elapsed, bid, "FORCED_FLAT"
    if exit_policy_id == "LEGACY_P5_LIFECYCLE":
        for elapsed, bid in path:
            return_micros = (bid - opportunity.entry_ask_micros) * CONTRACT_MULTIPLIER
            if return_micros <= -25 * opportunity.entry_ask_micros:
                return elapsed, bid, "LEGACY_STOP"
            if return_micros >= 50 * opportunity.entry_ask_micros:
                return elapsed, bid, "LEGACY_TARGET"
        elapsed, bid = path[-1]
        return elapsed, bid, "FORCED_FLAT"
    if exit_policy_id.startswith("FIXED_STOP_"):
        parts = exit_policy_id.split("_")
        stop_pct = int(parts[2][1:])
        target_pct = int(parts[4][1:])
        time_seconds = int(parts[6])
        for elapsed, bid in path:
            return_micros = (bid - opportunity.entry_ask_micros) * CONTRACT_MULTIPLIER
            if return_micros <= -(stop_pct * opportunity.entry_ask_micros):
                return elapsed, bid, "FIXED_STOP"
            if return_micros >= target_pct * opportunity.entry_ask_micros:
                return elapsed, bid, "FIXED_TARGET"
            if elapsed >= time_seconds:
                return elapsed, bid, "TIME_EXIT"
        elapsed, bid = path[-1]
        return elapsed, bid, "FORCED_FLAT"
    raise ValueError(f"unregistered exit policy: {exit_policy_id}")


def _decision_map(
    opportunities: tuple[CausalEntryOpportunityV1, ...], policy_id: str, seed: int | None
) -> dict[str, bool]:
    if policy_id == "HGB":
        return {row.opportunity_sha256: row.hgb_enter for row in opportunities}
    if policy_id == "NEURAL":
        return {row.opportunity_sha256: row.neural_enter for row in opportunities}
    if policy_id == "P5":
        by_frame: dict[tuple[str, int], list[CausalEntryOpportunityV1]] = {}
        for row in opportunities:
            by_frame.setdefault((row.session, row.decision_time_ns), []).append(row)
        selected = {
            min(
                (row for row in frame if row.p5_rank is not None),
                key=lambda row: (row.p5_rank, row.opportunity_sha256),
            ).opportunity_sha256
            for frame in by_frame.values()
            if any(row.p5_rank is not None for row in frame)
        }
        return {row.opportunity_sha256: row.opportunity_sha256 in selected for row in opportunities}
    if policy_id == "NEAREST_ATM":
        selected: set[str] = set()
        frames: dict[tuple[str, int], list[CausalEntryOpportunityV1]] = {}
        for row in opportunities:
            frames.setdefault((row.session, row.decision_time_ns), []).append(row)
        for frame in frames.values():
            eligible = [row for row in frame if row.nearest_atm_rank is not None]
            if eligible:
                selected.add(min(eligible, key=lambda row: (row.nearest_atm_rank, row.opportunity_sha256)).opportunity_sha256)
        return {row.opportunity_sha256: row.opportunity_sha256 in selected for row in opportunities}
    if "::" in policy_id:
        family, control = policy_id.split("::", 1)
        base = _decision_map(opportunities, family, None)
        ordered = tuple(opportunities)
        if control == "CONSTANT":
            rate = sum(base.values()) / len(base)
            count = int(math.floor(rate * len(ordered)))
            ranked = sorted(ordered, key=lambda row: row.opportunity_sha256)
            selected = {row.opportunity_sha256 for row in ranked[:count]}
        elif control == "SIGN_REVERSED":
            selected = {key for key, enter in base.items() if not enter}
        elif control == "TIME_SHIFTED_FEATURES":
            selected = set()
            last_by_session: dict[str, bool] = {}
            for row in ordered:
                if last_by_session.get(row.session, False):
                    selected.add(row.opportunity_sha256)
                last_by_session[row.session] = base[row.opportunity_sha256]
        elif control.startswith("SHUFFLED_TARGET_"):
            index = int(control.rsplit("_", 1)[1]) - 1
            shuffle_seed = prereg.MATCHED_RANDOM_SEEDS[index]
            flags = [base[row.opportunity_sha256] for row in ordered]
            ranked = sorted(
                ordered,
                key=lambda row: hashlib.sha256(
                    f"{shuffle_seed}:{row.opportunity_sha256}".encode("utf-8")
                ).hexdigest(),
            )
            selected = {
                row.opportunity_sha256
                for row, flag in zip(ranked, flags, strict=True)
                if flag
            }
        else:
            raise ValueError(f"unregistered negative control: {policy_id}")
        return {row.opportunity_sha256: row.opportunity_sha256 in selected for row in ordered}
    if policy_id == "MATCHED_RANDOM":
        if seed not in prereg.MATCHED_RANDOM_SEEDS:
            raise ValueError("matched-random seed drift")
        hgb = _decision_map(opportunities, "HGB", None)
        quotas: dict[tuple[str, str, str, str], int] = {}
        for row in opportunities:
            if hgb[row.opportunity_sha256]:
                cell = (row.session, row.right_code, row.moneyness_class, row.premium_band)
                quotas[cell] = quotas.get(cell, 0) + 1
        selected: set[str] = set()
        for cell, count in sorted(quotas.items()):
            candidates = [
                row for row in opportunities
                if (row.session, row.right_code, row.moneyness_class, row.premium_band) == cell
            ]
            ranked = sorted(
                candidates,
                key=lambda row: hashlib.sha256(
                    f"{seed}:{row.opportunity_sha256}".encode("utf-8")
                ).hexdigest(),
            )
            selected.update(row.opportunity_sha256 for row in ranked[:count])
        return {row.opportunity_sha256: row.opportunity_sha256 in selected for row in opportunities}
    raise ValueError(f"unregistered entry policy: {policy_id}")


def replay_policy(
    *, outer_fold: int, sessions: Sequence[str], opportunities: Sequence[CausalEntryOpportunityV1],
    policy_id: str, exit_policy_id: str, fee_dollars: int = 3,
    matched_random_seed: int | None = None,
) -> ReplayJournalV1:
    """Replay one policy on a serial $10k ledger with mandatory daily flat close."""

    ordered_sessions = tuple(sessions)
    rows = tuple(validate_opportunity(row) for row in opportunities)
    if not ordered_sessions or len(set(ordered_sessions)) != len(ordered_sessions):
        raise ValueError("replay sessions must be nonempty and unique")
    if any(row.outer_fold != outer_fold or row.session not in ordered_sessions for row in rows):
        raise ValueError("replay opportunity scope drift")
    if tuple(rows) != tuple(sorted(rows, key=lambda row: (ordered_sessions.index(row.session), row.decision_time_ns, row.opportunity_sha256))):
        raise ValueError("replay opportunities are not canonical")
    decisions = _decision_map(rows, policy_id, matched_random_seed)
    journal_policy_id = (
        f"MATCHED_RANDOM::SEED_{matched_random_seed}"
        if policy_id == "MATCHED_RANDOM"
        else policy_id
    )
    fee_micros = fee_dollars * 1_000_000
    equity = STARTING_EQUITY_MICROS
    trades: list[dict[str, Any]] = []
    pnl_rows: list[int] = []
    equity_rows: list[int] = []
    overlap = 0
    unaffordable = 0
    for session in ordered_sessions:
        session_start = equity
        session_rows = [row for row in rows if row.session == session]
        occupied_until = -1
        for row in session_rows:
            if not decisions[row.opportunity_sha256]:
                continue
            if row.decision_time_ns < occupied_until:
                overlap += 1
                continue
            maximum_loss = row.entry_ask_micros * CONTRACT_MULTIPLIER + fee_micros
            if maximum_loss > int(session_start * D48_FRACTION) or maximum_loss > equity:
                unaffordable += 1
                continue
            elapsed, exit_bid, reason = _exit_quote(row, exit_policy_id)
            net = (exit_bid - row.entry_ask_micros) * CONTRACT_MULTIPLIER - fee_micros
            trade_semantic = {
                "opportunity_sha256": row.opportunity_sha256,
                "session": session,
                "policy_id": journal_policy_id,
                "exit_policy_id": exit_policy_id,
                "entry_time_ns": row.decision_time_ns,
                "exit_elapsed_seconds": elapsed,
                "entry_ask_micros": row.entry_ask_micros,
                "exit_bid_micros": exit_bid,
                "fee_micros": fee_micros,
                "net_pnl_micros": net,
                "terminal_reason": reason,
            }
            trade = ReplayTradeV1(
                **trade_semantic, trade_sha256=_hash(trade_semantic)
            )
            trades.append(asdict(trade))
            equity += net
            occupied_until = row.decision_time_ns + elapsed * 1_000_000_000
        pnl_rows.append(equity - session_start)
        equity_rows.append(equity)
    semantic = {
        "schema_version": "pathd.entry_replay_journal.v1",
        "outer_fold": outer_fold,
        "policy_id": journal_policy_id,
        "exit_policy_id": exit_policy_id,
        "fee_micros": fee_micros,
        "sessions": ordered_sessions,
        "trades": tuple(trades),
        "session_pnl_micros": tuple(pnl_rows),
        "session_ending_equity_micros": tuple(equity_rows),
        "skipped_overlap_count": overlap,
        "skipped_unaffordable_count": unaffordable,
        "flat_close_every_session": True,
    }
    result = ReplayJournalV1(**semantic, journal_sha256=_hash(semantic))
    validate_replay_journal(result, opportunities=rows)
    return result


def validate_replay_journal(
    journal: ReplayJournalV1, *, opportunities: Sequence[CausalEntryOpportunityV1]
) -> ReplayJournalV1:
    if type(journal) is not ReplayJournalV1:
        raise TypeError("replay journal type drift")
    semantic = asdict(journal)
    digest = semantic.pop("journal_sha256")
    opportunity_hashes = {validate_opportunity(row).opportunity_sha256 for row in opportunities}
    if (
        journal.schema_version != "pathd.entry_replay_journal.v1"
        or not journal.sessions
        or len(journal.session_pnl_micros) != len(journal.sessions)
        or len(journal.session_ending_equity_micros) != len(journal.sessions)
        or journal.flat_close_every_session is not True
        or any(not _hex64(row.get("trade_sha256")) for row in journal.trades)
        or any(row.get("opportunity_sha256") not in opportunity_hashes for row in journal.trades)
        or any(
            row.get("trade_sha256")
            != _hash({key: value for key, value in row.items() if key != "trade_sha256"})
            for row in journal.trades
        )
        or digest != _hash(semantic)
    ):
        raise ValueError("replay terminal journal drift")
    reconstructed = {
        session: sum(
            row["net_pnl_micros"] for row in journal.trades if row["session"] == session
        )
        for session in journal.sessions
    }
    if tuple(reconstructed[session] for session in journal.sessions) != tuple(journal.session_pnl_micros):
        raise ValueError("replay journal PnL reconstruction drift")
    return journal


def select_shared_control_exit(
    *, outer_fold: int, model_fit_sessions: Sequence[str],
    opportunities: Sequence[CausalEntryOpportunityV1], legacy_causal_inputs_complete: bool,
) -> dict[str, Any]:
    """Evaluate the honest exit panel on P5 and select from reconstructed PnL."""

    if len(tuple(model_fit_sessions)) < 25:
        raise RuntimeError("shared control-exit selection is underpowered (<25 sessions)")

    fixed = tuple(
        f"FIXED_STOP_{stop}_TARGET_{target}_TIME_{seconds}"
        for stop in ("M25", "M50")
        for target in ("P25", "P50", "P100")
        for seconds in (60, 300, 900)
    )
    panel = ("EXIT_IMMEDIATE", "HOLD_TO_FLAT", "TIME_60", "TIME_300", "TIME_900", *fixed)
    p5_hold = replay_policy(
        outer_fold=outer_fold, sessions=model_fit_sessions,
        opportunities=opportunities, policy_id="P5", exit_policy_id="HOLD_TO_FLAT",
    )
    legacy_eligible = legacy_causal_inputs_complete and len(p5_hold.trades) >= 50
    if legacy_eligible:
        panel = (*panel, "LEGACY_P5_LIFECYCLE")
    journals = tuple(
        replay_policy(
            outer_fold=outer_fold, sessions=model_fit_sessions,
            opportunities=opportunities, policy_id="P5", exit_policy_id=exit_id,
        )
        for exit_id in panel
    )
    totals = tuple(sum(journal.session_pnl_micros) for journal in journals)
    best = max(totals)
    selected = min(exit_id for exit_id, total in zip(panel, totals, strict=True) if total == best)
    semantic = {
        "schema_version": "pathd.shared_control_exit_selection.v2",
        "outer_fold": outer_fold,
        "model_fit_sessions_sha256_newline": prereg.canonical_session_hash(model_fit_sessions),
        "candidate_policy_ids": panel,
        "terminal_journal_sha256s": tuple(journal.journal_sha256 for journal in journals),
        "total_net_pnl_micros": totals,
        "legacy_causal_inputs_complete": legacy_causal_inputs_complete,
        "legacy_completed_trade_count": len(p5_hold.trades),
        "legacy_eligible": legacy_eligible,
        "selected_policy_id": selected,
        "selection_rule": "MAX_TOTAL_FEE3_NET_PNL_THEN_LEXICOGRAPHIC_CANONICAL_COMPARATOR_ID",
    }
    return {**semantic, "selection_sha256": _hash(semantic)}


def produce_outer_replay_inventory(
    *, outer_fold: int, model_fit_sessions: Sequence[str], outer_sessions: Sequence[str],
    model_fit_opportunities: Sequence[CausalEntryOpportunityV1],
    outer_opportunities: Sequence[CausalEntryOpportunityV1],
    legacy_causal_inputs_complete: bool,
) -> dict[str, Any]:
    """Build the complete P5/HGB/NEURAL/control/random replay inventory."""

    selection = select_shared_control_exit(
        outer_fold=outer_fold, model_fit_sessions=model_fit_sessions,
        opportunities=model_fit_opportunities,
        legacy_causal_inputs_complete=legacy_causal_inputs_complete,
    )
    exit_id = selection["selected_policy_id"]
    policy_ids = ("HGB", "NEURAL", "P5", "NEAREST_ATM")
    positive = tuple(
        replay_policy(
            outer_fold=outer_fold, sessions=outer_sessions,
            opportunities=outer_opportunities, policy_id=policy_id,
            exit_policy_id=exit_id,
        )
        for policy_id in policy_ids
    )
    controls = tuple(
        replay_policy(
            outer_fold=outer_fold, sessions=outer_sessions,
            opportunities=outer_opportunities, policy_id=f"{family}::{control}",
            exit_policy_id=exit_id,
        )
        for family in ("HGB", "NEURAL")
        for control in prereg.ENTRY_NEGATIVE_CONTROL_IDS
    )
    randoms = tuple(
        replay_policy(
            outer_fold=outer_fold, sessions=outer_sessions,
            opportunities=outer_opportunities, policy_id="MATCHED_RANDOM",
            exit_policy_id=exit_id, matched_random_seed=seed,
        )
        for seed in prereg.MATCHED_RANDOM_SEEDS
    )
    semantic = {
        "schema_version": "pathd.corrected_v32.outer_replay_inventory.v1",
        "outer_fold": outer_fold,
        "shared_control_exit": selection,
        "positive_policy_ids": policy_ids,
        "positive_journal_sha256s": tuple(row.journal_sha256 for row in positive),
        "negative_control_policy_ids": tuple(
            f"{family}::{control}"
            for family in ("HGB", "NEURAL")
            for control in prereg.ENTRY_NEGATIVE_CONTROL_IDS
        ),
        "negative_control_journal_sha256s": tuple(row.journal_sha256 for row in controls),
        "matched_random_seeds": tuple(prereg.MATCHED_RANDOM_SEEDS),
        "matched_random_journal_sha256s": tuple(row.journal_sha256 for row in randoms),
        "all_replays_share_exit_policy_id": exit_id,
        "journal_count": len(positive) + len(controls) + len(randoms),
        "hardcoded_pass_flags": False,
    }
    return {**semantic, "inventory_sha256": _hash(semantic)}


def produce_outer_replay_inventory_from_sealed_input(
    payload: dict[str, Any], /
) -> dict[str, Any]:
    """Validate a frozen upstream producer payload and execute the whole replay."""

    if type(payload) is not dict:
        raise TypeError("outer replay input must be an exact object")
    expected = {
        "schema_version", "outer_fold", "model_fit_sessions", "outer_sessions",
        "model_fit_opportunities", "outer_opportunities",
        "legacy_causal_inputs_complete", "source_dataset_sha256s", "input_sha256",
    }
    semantic = dict(payload)
    digest = semantic.pop("input_sha256", None)
    if (
        set(payload) != expected
        or payload.get("schema_version") != "pathd.corrected_v32.outer_replay_input.v1"
        or not _hex64(digest)
        or digest != _hash(semantic)
        or type(payload.get("legacy_causal_inputs_complete")) is not bool
        or type(payload.get("source_dataset_sha256s")) not in (tuple, list)
        or not payload["source_dataset_sha256s"]
        or any(not _hex64(value) for value in payload["source_dataset_sha256s"])
    ):
        raise ValueError("sealed outer replay input drift")
    fit_rows = tuple(CausalEntryOpportunityV1(**row) for row in payload["model_fit_opportunities"])
    outer_rows = tuple(CausalEntryOpportunityV1(**row) for row in payload["outer_opportunities"])
    return produce_outer_replay_inventory(
        outer_fold=payload["outer_fold"],
        model_fit_sessions=tuple(payload["model_fit_sessions"]),
        outer_sessions=tuple(payload["outer_sessions"]),
        model_fit_opportunities=fit_rows,
        outer_opportunities=outer_rows,
        legacy_causal_inputs_complete=payload["legacy_causal_inputs_complete"],
    )


def build_spx_reference_transaction(
    *, anchors: Sequence[tuple[str, int]], rows: Sequence[dict[str, Any]],
    source_receipt_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Build causal SPX_REFERENCE current/lag-15 selections and receipt root."""

    ordered_anchors = tuple(anchors)
    ordered_receipts = tuple(source_receipt_sha256s)
    if not ordered_anchors or len(set(ordered_anchors)) != len(ordered_anchors):
        raise ValueError("SPX anchor identity drift")
    if (
        not ordered_receipts
        or len(set(ordered_receipts)) != len(ordered_receipts)
        or any(not _hex64(value) for value in ordered_receipts)
    ):
        raise ValueError("SPX source receipt identity drift")
    canonical_rows: list[dict[str, Any]] = []
    required = {
        "event_time_ns", "available_at_ns", "close_float64_hex", "publisher_id",
        "instrument_id", "row_group", "row_index", "canonical_row_sha256",
    }
    for row in rows:
        if type(row) is not dict or set(row) != required or not _hex64(row["canonical_row_sha256"]):
            raise ValueError("SPX source row schema drift")
        if row["available_at_ns"] < row["event_time_ns"]:
            raise ValueError("SPX row availability precedes event")
        float.fromhex(row["close_float64_hex"])
        canonical_rows.append(dict(row))
    selections: list[dict[str, Any]] = []
    for session, anchor_ns in ordered_anchors:
        selected: list[dict[str, Any] | None] = []
        for clock in (anchor_ns, anchor_ns - 900_000_000_000):
            eligible = [
                row for row in canonical_rows
                if row["available_at_ns"] <= clock
                and clock - row["available_at_ns"] <= 90_000_000_000
            ]
            selected.append(
                max(
                    eligible,
                    key=lambda row: (
                        row["available_at_ns"], row["event_time_ns"],
                        row["row_group"], row["row_index"], row["canonical_row_sha256"],
                    ),
                ) if eligible else None
            )
        current, lag = selected
        selections.append(
            {
                "session": session,
                "anchor_time_ns": anchor_ns,
                "status": "FRESH" if current is not None and lag is not None else "NO_CAUSAL_BAR",
                "current_locator": current,
                "lag15_locator": lag,
            }
        )
    semantic = {
        "schema_version": "pathd.spx_reference_context_transaction.v1",
        "anchor_count": len(ordered_anchors),
        "source_row_count": len(canonical_rows),
        "source_rows_sha256": _hash(canonical_rows),
        "source_receipt_sha256s": ordered_receipts,
        "source_receipts_root_sha256": _hash(list(ordered_receipts)),
        "selections": selections,
        "fresh_count": sum(row["status"] == "FRESH" for row in selections),
        "degraded_count": sum(row["status"] != "FRESH" for row in selections),
    }
    return {**semantic, "transaction_sha256": _hash(semantic)}
