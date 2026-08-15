"""Export a deterministic validation packet from an immutable trade ledger.

The CSV and content-addressed manifest are authoritative.  The two HTML files
are diagnostic views: one places every entry and exit on SPX, and the other
shows cumulative net equity.  This module does not replay, fit, tune, contact a
vendor, or contact a broker.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCHEMA_VERSION = "v5.candidate-validation-packet.v1"
MEASURED_COMMISSION_PER_SIDE_USD = 1.54
# Fees plus spread crossing on an aggressive option round trip, measured over
# 156,950 candidates (do-not-retest ledger row 181).  Recorded here only as the
# reference a fee-only export is compared against; nothing charges it.
MEASURED_AGGRESSIVE_OPTION_ROUND_TRIP_USD = 26.48
REVIEWED_ANTECEDENT = {
    "path": "v4/scripts/export_protocol101_trade_charts.py",
    "sha256": "02bb26cdaa45b71bbf7e10a3c5557e888d87eea323a0faf7b6de214418dea7fb",
}
ALLOWED_EVALUATION_ROLES = frozenset(
    {"out_of_fold", "confirmation", "external_stress"}
)
REQUIRED_TRADE_COLUMNS = (
    "trade_id",
    "session",
    "fold",
    "evaluation_role",
    "feature_time",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "quantity",
    "entry_price",
    "exit_price",
    "gross_pnl",
)
REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "candidate_sha256",
    "feature_contract_sha256",
    "entry_stream_sha256",
    "fill_law",
    "label_law",
    "fold_policy",
    "fold_boundaries",
)
OUTPUT_NAMES = (
    "trades.csv",
    "trades.html",
    "equity.html",
    "summary.json",
    "manifest.json",
)


class CandidatePacketError(RuntimeError):
    """The candidate ledger or requested export failed closed."""


@dataclass(frozen=True)
class CandidatePacketResult:
    output_dir: Path
    trades_csv: Path
    trades_html: Path
    equity_html: Path
    summary_json: Path
    manifest_json: Path
    trade_count: int
    net_pnl: float


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _frame_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.10g")


def _validate_candidate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_CANDIDATE_FIELDS if not manifest.get(name)]
    if missing:
        raise CandidatePacketError(
            "candidate_manifest_missing_fields:" + ",".join(missing)
        )
    return {str(key): value for key, value in sorted(manifest.items())}


def _validate_fold_boundaries(
    trades: pd.DataFrame, fold_boundaries: Mapping[str, Any]
) -> None:
    if not isinstance(fold_boundaries, Mapping) or not fold_boundaries:
        raise CandidatePacketError("candidate_manifest_invalid_fold_boundaries")
    parsed: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for fold, raw in fold_boundaries.items():
        if not isinstance(raw, Mapping):
            raise CandidatePacketError(f"invalid_fold_boundary:{fold}")
        start = pd.to_datetime(raw.get("start"), utc=True, errors="coerce")
        end = pd.to_datetime(raw.get("end"), utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise CandidatePacketError(f"invalid_fold_boundary:{fold}")
        parsed[str(fold)] = (start, end)
    ordered = sorted(parsed.items(), key=lambda item: item[1][0])
    for (_, (_, previous_end)), (fold, (next_start, _)) in zip(ordered, ordered[1:]):
        if next_start < previous_end:
            raise CandidatePacketError(f"overlapping_fold_boundaries:{fold}")
    used = set(trades["fold"].astype(str))
    missing = used - set(parsed)
    if missing:
        raise CandidatePacketError(
            "trade_fold_missing_declared_boundary:" + ",".join(sorted(missing))
        )
    for fold in used:
        start, end = parsed[fold]
        decisions = trades.loc[trades["fold"].astype(str).eq(fold), "decision_time"]
        if not (decisions.ge(start) & decisions.lt(end)).all():
            raise CandidatePacketError(f"trade_outside_declared_fold_boundary:{fold}")


def _normalize_spx(spx: pd.DataFrame) -> pd.DataFrame:
    required = {"event_time", "close"}
    missing = required - set(spx.columns)
    if missing:
        raise CandidatePacketError(f"spx_series_missing_columns:{sorted(missing)}")
    result = spx.loc[:, ["event_time", "close"]].copy()
    result["event_time"] = pd.to_datetime(result["event_time"], utc=True, errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    if result["event_time"].isna().any() or result["close"].isna().any():
        raise CandidatePacketError("spx_series_contains_invalid_values")
    if (~result["close"].map(math.isfinite)).any() or (result["close"] <= 0.0).any():
        raise CandidatePacketError("spx_series_contains_nonpositive_or_nonfinite_prices")
    if result["event_time"].duplicated().any():
        raise CandidatePacketError("spx_series_contains_duplicate_timestamps")
    result = result.sort_values("event_time", kind="stable").reset_index(drop=True)
    result["event_time"] = result["event_time"].map(lambda value: value.isoformat())
    return result


def _normalize_trades(
    trades: pd.DataFrame,
    *,
    starting_equity: float,
    commission_per_side_usd: float,
    slippage_per_side_points: float,
    contract_multiplier: float,
    fold_boundaries: Mapping[str, Any],
) -> pd.DataFrame:
    missing = set(REQUIRED_TRADE_COLUMNS) - set(trades.columns)
    if missing:
        raise CandidatePacketError(f"trade_ledger_missing_columns:{sorted(missing)}")
    if not math.isfinite(starting_equity) or starting_equity <= 0.0:
        raise CandidatePacketError("starting_equity_must_be_positive")
    for name, value in (
        ("commission_per_side_usd", commission_per_side_usd),
        ("slippage_per_side_points", slippage_per_side_points),
        ("contract_multiplier", contract_multiplier),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise CandidatePacketError(f"{name}_must_be_finite_and_nonnegative")
    if contract_multiplier == 0.0:
        raise CandidatePacketError("contract_multiplier_must_be_positive")

    result = trades.loc[:, list(REQUIRED_TRADE_COLUMNS)].copy()
    time_columns = ("feature_time", "decision_time", "entry_time", "exit_time")
    for column in time_columns:
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
        if result[column].isna().any():
            raise CandidatePacketError(f"trade_ledger_invalid_timestamp:{column}")
    for column in ("quantity", "entry_price", "exit_price", "gross_pnl"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or (~result[column].map(math.isfinite)).any():
            raise CandidatePacketError(f"trade_ledger_invalid_numeric:{column}")

    if result["trade_id"].astype(str).eq("").any() or result["trade_id"].duplicated().any():
        raise CandidatePacketError("trade_ids_must_be_nonempty_and_unique")
    if result["fold"].astype(str).eq("").any():
        raise CandidatePacketError("trade_fold_must_be_nonempty")
    invalid_roles = set(result["evaluation_role"].astype(str)) - ALLOWED_EVALUATION_ROLES
    if invalid_roles:
        raise CandidatePacketError(
            "candidate_packet_rejects_training_rows:" + ",".join(sorted(invalid_roles))
        )
    sides = result["side"].astype(str).str.upper()
    if not sides.isin({"CALL", "PUT", "LONG", "SHORT"}).all():
        raise CandidatePacketError("trade_side_must_be_call_put_long_or_short")
    result["side"] = sides
    if (result["quantity"] <= 0).any() or (result["quantity"] % 1 != 0).any():
        raise CandidatePacketError("trade_quantity_must_be_a_positive_integer")
    result["quantity"] = result["quantity"].astype(int)
    if (result[["entry_price", "exit_price"]] < 0.0).any().any():
        raise CandidatePacketError("trade_prices_must_be_nonnegative")

    direction = result["side"].map(
        {"CALL": 1.0, "PUT": 1.0, "LONG": 1.0, "SHORT": -1.0}
    )
    expected_gross = (
        (result["exit_price"] - result["entry_price"])
        * direction
        * contract_multiplier
        * result["quantity"]
    )
    if not (result["gross_pnl"] - expected_gross).abs().le(1e-6).all():
        raise CandidatePacketError("trade_gross_pnl_breaks_price_accounting_identity")

    ordered_clocks = (
        result["feature_time"].le(result["decision_time"])
        & result["decision_time"].le(result["entry_time"])
        & result["entry_time"].lt(result["exit_time"])
    )
    if not ordered_clocks.all():
        raise CandidatePacketError("trade_clocks_are_not_causal_and_monotone")
    _validate_fold_boundaries(result, fold_boundaries)
    entry_sessions = result["entry_time"].dt.tz_convert("America/New_York").dt.date.astype(str)
    exit_sessions = result["exit_time"].dt.tz_convert("America/New_York").dt.date.astype(str)
    declared_sessions = result["session"].astype(str)
    if not (entry_sessions.eq(exit_sessions) & entry_sessions.eq(declared_sessions)).all():
        raise CandidatePacketError("spx_0dte_trade_crosses_or_mismatches_session")

    result = result.sort_values(["entry_time", "exit_time", "trade_id"], kind="stable").reset_index(drop=True)
    if len(result) > 1:
        overlap = result["entry_time"].iloc[1:].reset_index(drop=True).lt(
            result["exit_time"].iloc[:-1].reset_index(drop=True)
        )
        if overlap.any():
            raise CandidatePacketError("serial_trade_ledger_has_overlapping_positions")

    round_trip_commission = 2.0 * commission_per_side_usd * result["quantity"]
    round_trip_slippage = (
        2.0 * slippage_per_side_points * contract_multiplier * result["quantity"]
    )
    result["commission_usd"] = round_trip_commission
    result["slippage_usd"] = round_trip_slippage
    result["total_cost_usd"] = round_trip_commission + round_trip_slippage
    result["net_pnl"] = result["gross_pnl"] - result["total_cost_usd"]
    result["equity"] = starting_equity + result["net_pnl"].cumsum()
    for column in time_columns:
        result[column] = result[column].map(lambda value: value.isoformat())
    return result


def _nearest_spx_price(spx: pd.DataFrame, timestamp: str) -> float | None:
    if spx.empty:
        return None
    times = pd.to_datetime(spx["event_time"], utc=True)
    target = pd.Timestamp(timestamp)
    eligible = times.le(target)
    if eligible.any():
        return float(spx.loc[eligible[eligible].index[-1], "close"])
    return float(spx.iloc[0]["close"])


def _scale(values: list[float], low: float, high: float) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [(low + high) / 2.0 for _ in values]
    return [low + (value - minimum) * (high - low) / (maximum - minimum) for value in values]


def _html_shell(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{escape(title)}</title><style>
body{{font:14px system-ui,sans-serif;margin:24px;color:#17202a;background:#fff}}
svg{{border:1px solid #ccd2d8;background:#fbfcfd;max-width:100%;height:auto}}
.axis{{stroke:#7f8c8d;stroke-width:1}} .series{{fill:none;stroke:#234e70;stroke-width:2}}
.entry-marker{{fill:#16865c}} .exit-marker{{fill:#c84630}} .equity{{fill:none;stroke:#6b46c1;stroke-width:2}}
table{{border-collapse:collapse;margin-top:18px}} th,td{{border:1px solid #d7dce1;padding:5px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
</style></head><body><h1>{escape(title)}</h1>{body}</body></html>
"""


def _trade_chart_html(trades: pd.DataFrame, spx: pd.DataFrame) -> str:
    width, height, pad = 1100.0, 520.0, 45.0
    if spx.empty:
        chart = '<p id="no-spx-bars">No SPX bars supplied.</p>'
    else:
        times = [float(value) for value in pd.to_datetime(spx["event_time"], utc=True).astype("int64")]
        prices = [float(value) for value in spx["close"]]
        xs = _scale(times, pad, width - pad)
        ys = _scale(prices, height - pad, pad)
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
        minimum_time, maximum_time = min(times), max(times)
        time_span = max(maximum_time - minimum_time, 1.0)
        minimum_price, maximum_price = min(prices), max(prices)
        price_span = max(maximum_price - minimum_price, 1e-12)
        markers: list[str] = []
        for row in trades.itertuples(index=False):
            for kind, timestamp in (("entry", row.entry_time), ("exit", row.exit_time)):
                numeric_time = float(pd.Timestamp(timestamp).value)
                price = _nearest_spx_price(spx, timestamp)
                if price is None:
                    continue
                x = pad + (numeric_time - minimum_time) * (width - 2 * pad) / time_span
                y = height - pad - (price - minimum_price) * (height - 2 * pad) / price_span
                markers.append(
                    f'<circle class="{kind}-marker" data-trade-id="{escape(str(row.trade_id))}" '
                    f'cx="{x:.2f}" cy="{y:.2f}" r="4"><title>{escape(str(row.trade_id))} '
                    f'{kind} {escape(timestamp)}</title></circle>'
                )
        chart = (
            f'<svg role="img" aria-label="SPX with trade entries and exits" viewBox="0 0 {width:.0f} {height:.0f}">'
            f'<line class="axis" x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}"/>'
            f'<polyline class="series" points="{points}"/>{"".join(markers)}</svg>'
        )
    return _html_shell(
        title="Candidate trades on SPX",
        body=(
            '<p>Green markers are entries; red markers are exits. This is a diagnostic view of trades.csv.</p>'
            + chart
            + f'<p id="trade-count">Trades: {len(trades)}</p>'
        ),
    )


def _equity_chart_html(trades: pd.DataFrame, *, starting_equity: float) -> str:
    width, height, pad = 1100.0, 520.0, 45.0
    values = [starting_equity] + [float(value) for value in trades["equity"]]
    xs = _scale([float(index) for index in range(len(values))], pad, width - pad)
    ys = _scale(values, height - pad, pad)
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    rows = "".join(
        f"<tr><td>{escape(str(row.trade_id))}</td><td>{float(row.net_pnl):.2f}</td><td>{float(row.equity):.2f}</td></tr>"
        for row in trades.itertuples(index=False)
    )
    table = (
        "<table><thead><tr><th>Trade</th><th>Net P&amp;L</th><th>Equity</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    return _html_shell(
        title="Candidate net equity",
        body=(
            "<p>Net of the declared commission and slippage contract. Diagnostic view only.</p>"
            f'<svg role="img" aria-label="Net equity curve" viewBox="0 0 {width:.0f} {height:.0f}">'
            f'<line class="axis" x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}"/>'
            f'<polyline class="equity" points="{points}"/></svg>{table}'
        ),
    )


def _describe_cost_model(
    *,
    commission_per_side_usd: float,
    slippage_per_side_points: float,
    contract_multiplier: float,
) -> dict[str, Any]:
    """Say in the output exactly which cost components were charged.

    Settled 2026-08-12 by the friction decomposition
    (``v5/research/findings/FRICTION_DECOMPOSITION_2026_08_12.md``): **no frozen
    cost constant in this project contains slippage**, so this packet's
    ``slippage_per_side_points`` is additive rather than a double count.  Its
    real job is spread crossing, which nothing else here charges.

    That matters because the default is zero.  A packet exported at the default
    charges the measured $3.08 round-trip *fee* and no spread at all, while the
    measured aggressive option round trip is **$26.48** — 8.6x larger.  A
    candidate judged on the default would be credited a cost it would not pay.
    Refusing the default outright would break every caller that legitimately
    wants a fee-only diagnostic, so the packet states its model instead of
    guessing the caller's intent.
    """

    charged = ["commission"]
    omitted: list[str] = []
    if slippage_per_side_points > 0.0:
        charged.append("spread_or_slippage")
    else:
        omitted.append("spread_crossing")
    return {
        "components_charged": charged,
        "components_omitted": omitted,
        "charged_round_trip_usd": round(
            2.0 * commission_per_side_usd
            + 2.0 * slippage_per_side_points * contract_multiplier,
            6,
        ),
        "excludes_spread_crossing": not slippage_per_side_points > 0.0,
        "reference_measured_aggressive_round_trip_usd": (
            MEASURED_AGGRESSIVE_OPTION_ROUND_TRIP_USD
        ),
        "evidence": "v5/research/findings/FRICTION_DECOMPOSITION_2026_08_12.md",
    }


def export_candidate_packet(
    *,
    trades: pd.DataFrame,
    spx: pd.DataFrame,
    candidate_manifest: Mapping[str, Any],
    output_dir: Path,
    starting_equity: float = 10_000.0,
    commission_per_side_usd: float = MEASURED_COMMISSION_PER_SIDE_USD,
    slippage_per_side_points: float = 0.0,
    contract_multiplier: float = 100.0,
    allow_overwrite: bool = False,
) -> CandidatePacketResult:
    """Create the five-file packet without mutating either input frame."""

    manifest_input = _validate_candidate_manifest(candidate_manifest)
    normalized_spx = _normalize_spx(spx)
    normalized_trades = _normalize_trades(
        trades,
        starting_equity=starting_equity,
        commission_per_side_usd=commission_per_side_usd,
        slippage_per_side_points=slippage_per_side_points,
        contract_multiplier=contract_multiplier,
        fold_boundaries=manifest_input["fold_boundaries"],
    )
    if len(normalized_trades):
        if normalized_spx.empty:
            raise CandidatePacketError("nonempty_trade_ledger_requires_spx_bars")
        spx_times = pd.to_datetime(normalized_spx["event_time"], utc=True)
        entry_times = pd.to_datetime(normalized_trades["entry_time"], utc=True)
        exit_times = pd.to_datetime(normalized_trades["exit_time"], utc=True)
        if entry_times.min() < spx_times.min() or exit_times.max() > spx_times.max():
            raise CandidatePacketError("spx_series_does_not_cover_all_trade_markers")
    output_dir = Path(output_dir)
    output_paths = {name: output_dir / name for name in OUTPUT_NAMES}
    if not allow_overwrite:
        existing = [name for name, path in output_paths.items() if path.exists()]
        if existing:
            raise CandidatePacketError(
                "refusing_to_overwrite_candidate_packet:" + ",".join(existing)
            )

    trades_text = _frame_csv(normalized_trades)
    spx_text = _frame_csv(normalized_spx)
    candidate_text = _canonical_json(manifest_input)
    gross_pnl = float(normalized_trades["gross_pnl"].sum()) if len(normalized_trades) else 0.0
    total_cost = float(normalized_trades["total_cost_usd"].sum()) if len(normalized_trades) else 0.0
    net_pnl = float(normalized_trades["net_pnl"].sum()) if len(normalized_trades) else 0.0
    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": manifest_input["candidate_id"],
        "trade_count": len(normalized_trades),
        "sessions": int(normalized_trades["session"].nunique()) if len(normalized_trades) else 0,
        "folds": sorted(set(normalized_trades["fold"].astype(str))),
        "evaluation_roles": sorted(set(normalized_trades["evaluation_role"].astype(str))),
        "starting_equity": float(starting_equity),
        "ending_equity": float(starting_equity + net_pnl),
        "gross_pnl": gross_pnl,
        "total_cost_usd": total_cost,
        "net_pnl": net_pnl,
        "commission_per_side_usd": float(commission_per_side_usd),
        "slippage_per_side_points": float(slippage_per_side_points),
        "contract_multiplier": float(contract_multiplier),
        "cost_model": _describe_cost_model(
            commission_per_side_usd=commission_per_side_usd,
            slippage_per_side_points=slippage_per_side_points,
            contract_multiplier=contract_multiplier,
        ),
        "first_entry": normalized_trades.iloc[0]["entry_time"] if len(normalized_trades) else None,
        "last_exit": normalized_trades.iloc[-1]["exit_time"] if len(normalized_trades) else None,
        "authoritative_outputs": ["trades.csv", "manifest.json"],
        "diagnostic_outputs": ["trades.html", "equity.html"],
    }
    contents = {
        "trades.csv": trades_text,
        "trades.html": _trade_chart_html(normalized_trades, normalized_spx),
        "equity.html": _equity_chart_html(
            normalized_trades, starting_equity=starting_equity
        ),
        "summary.json": _json_text(summary),
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "reviewed_antecedent": REVIEWED_ANTECEDENT,
        "candidate": manifest_input,
        "inputs": {
            "candidate_manifest_sha256": _sha256_bytes(candidate_text.encode("utf-8")),
            "normalized_spx_sha256": _sha256_bytes(spx_text.encode("utf-8")),
            "normalized_trade_ledger_sha256": _sha256_bytes(trades_text.encode("utf-8")),
        },
        "outputs": {
            name: _sha256_bytes(content.encode("utf-8"))
            for name, content in sorted(contents.items())
        },
    }
    provenance["receipt_sha256"] = _sha256_bytes(
        _canonical_json(provenance).encode("utf-8")
    )
    contents["manifest.json"] = _json_text(provenance)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        output_paths[name].write_text(contents[name], encoding="utf-8")
    return CandidatePacketResult(
        output_dir=output_dir,
        trades_csv=output_paths["trades.csv"],
        trades_html=output_paths["trades.html"],
        equity_html=output_paths["equity.html"],
        summary_json=output_paths["summary.json"],
        manifest_json=output_paths["manifest.json"],
        trade_count=len(normalized_trades),
        net_pnl=net_pnl,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--spx", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    parser.add_argument(
        "--commission-per-side-usd",
        type=float,
        default=MEASURED_COMMISSION_PER_SIDE_USD,
    )
    parser.add_argument("--slippage-per-side-points", type=float, default=0.0)
    parser.add_argument("--contract-multiplier", type=float, default=100.0)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = export_candidate_packet(
        trades=pd.read_csv(args.trades),
        spx=pd.read_csv(args.spx),
        candidate_manifest=json.loads(args.candidate_manifest.read_text(encoding="utf-8")),
        output_dir=args.output_dir,
        starting_equity=args.starting_equity,
        commission_per_side_usd=args.commission_per_side_usd,
        slippage_per_side_points=args.slippage_per_side_points,
        contract_multiplier=args.contract_multiplier,
        allow_overwrite=args.allow_overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "trade_count": result.trade_count,
                "net_pnl": result.net_pnl,
                "status": "CANDIDATE_PACKET_WRITTEN",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
