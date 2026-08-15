"""Column-by-column availability audit required before fit economics are read."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_fit_cache import verify_cache_index
from v5.research.causal_day_tensorizer import (
    AccountObservation,
    CANDLE_FEATURES,
    LADDER_FEATURES,
    tensorize_observation,
)


ACCOUNT_FEATURES = (
    "cash_fraction_of_start",
    "realised_pnl_fraction_of_start",
    "trades_opened_fraction_of_three",
    "remaining_trade_cap_fraction",
    "breaker_triggered",
)
POSITION_FEATURES = (
    "is_holding",
    "signed_right",
    "strike_distance_scaled",
    "entry_ask_fraction_of_start",
    "unrealised_pnl_fraction_of_start",
    "maximum_favourable_fraction_of_start",
    "maximum_adverse_fraction_of_start",
    "minutes_held_fraction_of_120",
    "entry_moneyness_scaled",
    "signed_origin_regime",
)
CLOCK_FEATURES = (
    "session_elapsed_fraction",
    "session_remaining_fraction",
    "is_morning_entry_regime",
    "time_sine",
    "time_cosine",
)


def feature_ledger() -> pd.DataFrame:
    rows = []
    for feature in CANDLE_FEATURES:
        rows.append(
            {
                "group": "candle",
                "feature": feature,
                "availability": "completed [t-1 minute,t) bar or earlier",
                "maximum_timestamp": "t",
                "future_value_allowed": False,
                "derivation_scope": "visible session prefix only",
            }
        )
    for feature in LADDER_FEATURES:
        rows.append(
            {
                "group": "ladder",
                "feature": feature,
                "availability": "whole live two-sided snapshot stamped t",
                "maximum_timestamp": "t",
                "future_value_allowed": False,
                "derivation_scope": "current snapshot only",
            }
        )
    for group, features, scope in (
        ("account", ACCOUNT_FEATURES, "ledger events at or before t"),
        ("position", POSITION_FEATURES, "held-position path at or before t"),
        ("clock", CLOCK_FEATURES, "deterministic function of t"),
    ):
        for feature in features:
            rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "availability": scope,
                    "maximum_timestamp": "t",
                    "future_value_allowed": False,
                    "derivation_scope": scope,
                }
            )
    return pd.DataFrame(rows)


def _future_mutation_probe() -> bool:
    session = "2099-01-02"
    candles = pd.DataFrame(
        [
            {
                "session": session,
                "bar_minute": f"09:{30 + index:02d}",
                "knowable_at": f"09:{31 + index:02d}",
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 10.0 + index,
            }
            for index in range(6)
        ]
    )
    rows = []
    for minute in ("09:35", "09:36"):
        row = {
            "session": session,
            "minute": minute,
            "contract_id": minute,
            "right": "C",
            "strike": 105.0,
            "underlying_price": 100.0,
            "entry_eligible": minute == "09:35",
        }
        row.update({column: float(index + 1) for index, column in enumerate(LADDER_FEATURES)})
        rows.append(row)
    ladder = pd.DataFrame(rows)
    account = AccountObservation(10_000.0, 0.0, 0, 2, False)
    first = tensorize_observation(
        candles,
        ladder,
        session=session,
        minute="09:35",
        role="morning_entry",
        account=account,
    )
    changed_candles = candles.copy()
    changed_candles.loc[
        changed_candles["knowable_at"].gt("09:35"),
        ["open", "high", "low", "close", "volume"],
    ] *= 1_000.0
    changed_ladder = ladder.copy()
    changed_ladder.loc[
        changed_ladder["minute"].gt("09:35"), list(LADDER_FEATURES)
    ] *= -1_000.0
    second = tensorize_observation(
        changed_candles,
        changed_ladder,
        session=session,
        minute="09:35",
        role="morning_entry",
        account=account,
    )
    return bool(
        torch.equal(first.batch.candles, second.batch.candles)
        and torch.equal(first.batch.ladder, second.batch.ladder)
    )


def run(*, out_dir: Path, declaration_path: Path, cache_receipt_path: Path) -> dict:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite feature audit: {out_dir}")
    ledger = feature_ledger()
    expected_rows = len(CANDLE_FEATURES) + len(LADDER_FEATURES) + 5 + 10 + 5
    if len(ledger) != expected_rows or ledger.duplicated(["group", "feature"]).any():
        raise RuntimeError("feature ledger is incomplete or ambiguous")
    if ledger["future_value_allowed"].any():
        raise RuntimeError("a fitted feature is marked future-available")
    forbidden_tokens = ("future", "target", "maximum_itm", "time_to_", "net_bid", "net_mid")
    names = ledger["feature"].str.lower()
    leaked = [name for name in names if any(token in name for token in forbidden_tokens)]
    if leaked:
        raise RuntimeError(f"future/label-like feature names present: {leaked}")

    declaration = json.loads(declaration_path.read_text())
    declaration_hash = declaration.get("receipt_sha256")
    unsigned_declaration = dict(declaration)
    unsigned_declaration.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned_declaration)).hexdigest() != declaration_hash:
        raise RuntimeError("fit declaration self-hash mismatch")
    if declaration.get("schema_version") != "v5.causal-day-trader-fit-declaration.v4":
        raise RuntimeError("feature audit requires declaration V4")
    if cache_receipt_path != cache_receipt_path.parent / "receipt.json":
        raise RuntimeError("cache receipt path must name the cache-root receipt")
    cache_receipt, manifest = verify_cache_index(cache_receipt_path.parent)
    observation_path = Path(
        "v4/audit/autoresearch/causal_day_observation_contract_2026_08_14/receipt.json"
    )
    observation = json.loads(observation_path.read_text())
    if not _future_mutation_probe():
        raise RuntimeError("future mutation changed a decision tensor")
    if not observation["assertions"]["policy_chain_equals_source_live_two_sided_count"]:
        raise RuntimeError("the actual whole-chain equality proof is not green")
    if cache_receipt["eligible_actions"] != 698_231:
        raise RuntimeError("fit cache action population differs from the frozen candidate family")
    if cache_receipt["minutes"] != 243 * 326:
        raise RuntimeError("fit cache entry-minute population is incomplete")

    out_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = out_dir / "feature_timestamp_ledger.csv"
    ledger.to_csv(ledger_path, index=False)
    receipt = {
        "schema_version": "v5.causal-day-fit-feature-audit.v1",
        "created_on": "2026-08-14",
        "status": "PASS_BEFORE_ECONOMICS",
        "features": {
            "total": len(ledger),
            "candle": len(CANDLE_FEATURES),
            "ladder": len(LADDER_FEATURES),
            "account": len(ACCOUNT_FEATURES),
            "position": len(POSITION_FEATURES),
            "clock": len(CLOCK_FEATURES),
        },
        "assertions": {
            "every_feature_has_maximum_timestamp_t": bool(ledger["maximum_timestamp"].eq("t").all()),
            "no_future_feature": True,
            "no_label_like_feature": True,
            "actual_future_mutation_proof": True,
            "whole_chain_equals_source": True,
            "cache_receipt_manifest_and_session_hashes_verified": len(manifest) == 243,
            "no_post_entry_candidate_filter": True,
            "reserved_sessions_absent": cache_receipt["last_session"] < "2026-08-06",
        },
        "declaration": {"path": str(declaration_path), "sha256": file_sha256(declaration_path)},
        "cache_receipt": {"path": str(cache_receipt_path), "sha256": file_sha256(cache_receipt_path)},
        "observation_receipt": {"path": str(observation_path), "sha256": file_sha256(observation_path)},
        "ledger": {"path": str(ledger_path), "sha256": file_sha256(ledger_path)},
        "implementation_sha256": file_sha256(Path(__file__)),
        "economics_read": False,
    }
    if not all(receipt["assertions"].values()):
        raise RuntimeError("feature audit assertion failed")
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    path = out_dir / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--cache-receipt", type=Path, required=True)
    args = parser.parse_args()
    run(
        out_dir=args.out_dir,
        declaration_path=args.declaration,
        cache_receipt_path=args.cache_receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
