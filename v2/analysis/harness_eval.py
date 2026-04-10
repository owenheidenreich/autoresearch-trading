"""Harness regression and holdout suite for the v4 exact-chain environment."""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from v2.core.chain_data import (
    HarnessEvalCase,
    QUALITY_VALID,
    describe_contract,
    dump_suite,
    load_sidecar_cached,
    load_suite,
    padded_snapshot,
    sidecar_path,
)
from v2.core.features import _FEAT_IDX
from v2.core.policy import DEFAULT_POLICY
from v2.core.schema import TradeIntent
from v2.core.simulator import simulate_trade


SUITE_PATH = os.path.join("v2", "harness_eval", "suite.json")
VALID_EXIT_REASONS = {"STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP", "MAX_HOLD", "EOD"}


def _case_to_dict(case: HarnessEvalCase) -> dict:
    return {
        "case_id": case.case_id,
        "date": case.date,
        "bar_of_day": case.bar_of_day,
        "tags": list(case.tags),
        "notes": case.notes,
    }


def _time_tags(local_bar: int) -> list[str]:
    tags = ["time_of_day"]
    if local_bar < 60:
        tags.append("open")
    elif local_bar >= 240:
        tags.append("late_day")
    else:
        tags.append("midday")
    return tags


def _signed_moneyness_points(spot: np.ndarray, strike: float, right: str) -> np.ndarray:
    if right == "C":
        return spot - strike
    return strike - spot


def _regime_tags(sim_row: np.ndarray, contract_row: np.ndarray, local_bar: int) -> list[str]:
    tags = ["market_regime"]
    ret_12 = float(sim_row[_FEAT_IDX["ret_12"]])
    ret_6 = float(sim_row[_FEAT_IDX["ret_6"]])
    vix_roc = float(sim_row[_FEAT_IDX["vix_roc"]])
    atm_gamma = float(sim_row[_FEAT_IDX["atm_gamma"]])
    spread_frac = float(contract_row[4])

    if local_bar < 60 and float(sim_row[_FEAT_IDX["vix_regime"]]) >= 1.0:
        tags.append("high_vol_open")
    if abs(ret_12) >= 0.004 and np.sign(ret_12) == np.sign(ret_6):
        tags.append("trend")
    elif abs(ret_12) >= 0.003 and np.sign(ret_12) != np.sign(ret_6) and abs(ret_6) >= 0.001:
        tags.append("reversal")
    else:
        tags.append("chop")
    if vix_roc <= -0.03:
        tags.append("vol_crush")
    if local_bar >= 240 and abs(atm_gamma) > 0.01:
        tags.append("late_day_gamma")
    if spread_frac >= 0.18:
        tags.append("spread_blowout")
    return tags


def _path_tags(sidecar: dict, local_bar: int, contract_idx: int, spot_day: np.ndarray) -> list[str]:
    tags: list[str] = []
    contract = describe_contract(sidecar, contract_idx)
    fill_bar = local_bar + 1
    if fill_bar >= len(spot_day):
        return tags

    signed = _signed_moneyness_points(spot_day[fill_bar:], contract.strike, contract.right)
    entry_signed = float(_signed_moneyness_points(np.asarray([spot_day[local_bar]], dtype=np.float32), contract.strike, contract.right)[0])
    if len(signed):
        if -10.0 <= entry_signed < 0.0 and float(np.nanmax(signed)) > 0.0:
            tags.append("otm_to_itm")
        if entry_signed > 0.0 and float(np.nanmin(signed)) <= -20.0:
            tags.append("itm_to_far_otm")
        if len(spot_day) > fill_bar + 15 and float(np.nanmax(np.abs(spot_day[fill_bar:fill_bar + 16] - spot_day[local_bar]))) >= 15.0:
            tags.append("fast_multi_strike_move")

    series = sidecar["contract_mid"][contract_idx]
    gap_end = min(len(series), fill_bar + DEFAULT_POLICY.max_hold_bars + 1)
    if gap_end > fill_bar and not np.isfinite(series[fill_bar:gap_end]).all():
        tags.append("forward_gap")
    return tags


def _snapshot_count(sidecar: dict, local_bar: int) -> int:
    return int(sidecar["bar_ptrs"][local_bar + 1] - sidecar["bar_ptrs"][local_bar])


def _wide_vs_tight_tag(contract_rows: np.ndarray, best_local: int) -> bool:
    if len(contract_rows) < 2 or best_local < 0 or best_local >= len(contract_rows):
        return False
    row = contract_rows[best_local]
    right_is_put = row[2] > 0.5
    strike = row[1]
    peer_mask = np.abs(contract_rows[:, 1] - strike) <= 10.0
    peer_mask &= (contract_rows[:, 2] > 0.5) if right_is_put else (contract_rows[:, 2] < 0.5)
    peers = contract_rows[peer_mask]
    if len(peers) < 2:
        return False
    spread = float(row[4])
    tighter = float(np.nanmin(peers[:, 4]))
    return spread >= tighter * 1.25 and spread - tighter >= 0.02


def _pick_core_cases(all_cases: list[dict]) -> list[dict]:
    required_tags = [
        "contract_identity",
        "spot_alignment",
        "execution_fidelity",
        "risk_management",
        "call",
        "put",
        "open",
        "midday",
        "late_day",
        "otm_to_itm",
        "itm_to_far_otm",
        "fast_multi_strike_move",
        "eod_exit",
        "wide_vs_tight",
        "forward_gap",
        "partially_observed",
    ]
    selected_ids: set[str] = set()
    core: list[dict] = []
    for tag in required_tags:
        picked = next((case for case in all_cases if tag in case["tags"] and case["case_id"] not in selected_ids), None)
        if picked is not None:
            core.append(picked)
            selected_ids.add(picked["case_id"])

    for case in all_cases:
        if case["case_id"] in selected_ids:
            continue
        core.append(case)
        selected_ids.add(case["case_id"])
        if len(core) >= 40:
            break
    return core


MAX_OPT_CASES = 500
MAX_HOLDOUT_CASES = 500


def _stratified_sample(cases: list[dict], max_cases: int, rng: np.random.Generator) -> list[dict]:
    """Stratified sample: keep at least one case per tag, then fill randomly."""
    if len(cases) <= max_cases:
        return cases
    selected_ids: set[str] = set()
    selected: list[dict] = []
    # First pass: one case per tag for coverage
    tag_pool: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        for tag in case["tags"]:
            tag_pool[tag].append(case)
    for tag in sorted(tag_pool):
        if len(selected) >= max_cases:
            break
        candidates = [c for c in tag_pool[tag] if c["case_id"] not in selected_ids]
        if candidates:
            pick = candidates[int(rng.integers(len(candidates)))]
            selected.append(pick)
            selected_ids.add(pick["case_id"])
    # Second pass: fill remaining budget with random draws
    remaining = [c for c in cases if c["case_id"] not in selected_ids]
    if remaining and len(selected) < max_cases:
        n_fill = min(max_cases - len(selected), len(remaining))
        indices = rng.choice(len(remaining), size=n_fill, replace=False)
        for i in sorted(indices):
            selected.append(remaining[i])
    return selected


def _split_remaining_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    if not cases:
        return [], []

    date_to_cases: dict[str, list[dict]] = defaultdict(list)
    date_to_tags: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        date_to_cases[case["date"]].append(case)
        date_to_tags[case["date"]].update(case["tags"])

    dates = sorted(date_to_cases)
    if len(dates) < 2:
        return cases[::2], cases[1::2]

    target_holdout_days = max(1, int(round(len(dates) * 0.3)))
    tag_targets = {}
    for tag in sorted({tag for tags in date_to_tags.values() for tag in tags}):
        support = sum(1 for day in dates if tag in date_to_tags[day])
        tag_targets[tag] = max(1, int(round(support * 0.3))) if support >= 2 else 0

    holdout_days: set[str] = set()
    holdout_tag_counts: Counter[str] = Counter()
    sorted_days = sorted(dates, key=lambda d: (-len(date_to_tags[d]), d))
    for day in sorted_days:
        if len(holdout_days) >= target_holdout_days:
            break
        day_tags = date_to_tags[day]
        gain = sum(1 for tag in day_tags if holdout_tag_counts[tag] < tag_targets.get(tag, 0))
        if gain <= 0 and len(holdout_days) > 0:
            continue
        holdout_days.add(day)
        holdout_tag_counts.update(day_tags)

    for day in sorted_days:
        if len(holdout_days) >= target_holdout_days:
            break
        if day not in holdout_days:
            holdout_days.add(day)
            holdout_tag_counts.update(date_to_tags[day])

    optimization_full = [case for case in cases if case["date"] not in holdout_days]
    holdout_full = [case for case in cases if case["date"] in holdout_days]
    if not optimization_full and len(holdout_full) > 1:
        optimization_full = holdout_full[::2]
        holdout_full = holdout_full[1::2]

    rng = np.random.default_rng(42)
    optimization = _stratified_sample(optimization_full, MAX_OPT_CASES, rng)
    holdout = _stratified_sample(holdout_full, MAX_HOLDOUT_CASES, rng)
    return optimization, holdout


def build_suite(data_path: str = "v2/data.pt", suite_path: str = SUITE_PATH) -> dict:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    sim_features = data["X_sim"].numpy() if "X_sim" in data else data["X"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    dates = data["dates"]
    spot = data["spot_prices"].numpy()
    day_to_bars: dict[str, list[int]] = defaultdict(list)
    for i, day in enumerate(dates):
        day_to_bars[day].append(i)

    all_cases: list[dict] = []
    for day in sorted(day_to_bars):
        bars = day_to_bars[day]
        sidecar = load_sidecar_cached(sidecar_path(sidecar_dir, day))
        spot_day = spot[bars]
        for local_bar in range(sidecar["n_bars"]):
            global_idx = bars[local_bar]
            count = _snapshot_count(sidecar, local_bar)
            time_tags = _time_tags(local_bar)
            if count == 0 or not bool(sidecar["bar_labelable"][local_bar]):
                tags = ["chain_completeness", "partially_observed", *time_tags]
                notes = "no executable contract rows"
                if count > 0:
                    tags.append("forward_gap")
                    notes = "visible contracts exist but forward replay is incomplete"
                all_cases.append(_case_to_dict(HarnessEvalCase(
                    case_id=f"{day}-b{local_bar:03d}-partial",
                    date=day,
                    bar_of_day=local_bar,
                    tags=tuple(dict.fromkeys(tags)),
                    notes=notes,
                )))
                continue

            best_local = int(sidecar["bar_best_contract_idx"][local_bar])
            if best_local < 0:
                continue
            rows, _, contract_indices = padded_snapshot(sidecar, local_bar, count)
            contract_idx = int(contract_indices[best_local])
            contract = describe_contract(sidecar, contract_idx)
            tags = [
                "contract_identity",
                "spot_alignment",
                "execution_fidelity",
                "risk_management",
                "direction",
                *time_tags,
                *(["call"] if contract.right == "C" else ["put"]),
            ]
            tags.extend(_regime_tags(sim_features[global_idx], rows[best_local], local_bar))
            tags.extend(_path_tags(sidecar, local_bar, contract_idx, spot_day))
            if _wide_vs_tight_tag(rows, best_local):
                tags.append("wide_vs_tight")

            series = sidecar["contract_mid"][contract_idx].astype(np.float32)
            intent = TradeIntent(
                trade=True,
                expiry=contract.expiry,
                strike=contract.strike,
                right=contract.right,
                qty=DEFAULT_POLICY.qty,
                entry_ref_price=float(series[local_bar]),
                order_style=DEFAULT_POLICY.order_style,
                tif=DEFAULT_POLICY.tif,
                stop_price=float(series[local_bar]) * (1.0 - DEFAULT_POLICY.stop_pct),
                take_profit_price=float(series[local_bar]) * (1.0 + DEFAULT_POLICY.target_pct),
                max_hold_bars=DEFAULT_POLICY.max_hold_bars,
                exit_policy=DEFAULT_POLICY.exit_policy,
                confidence=0.5,
                reason_codes=("harness_eval",),
                bar_index=local_bar,
                underlying_price=float(spot_day[local_bar]),
            )
            trade = simulate_trade(
                intent,
                series,
                sim_features[bars],
                np.arange(len(bars), dtype=np.int32),
                [day] * len(bars),
                local_bar,
            )
            if trade is not None and trade.exit_reason == "EOD":
                tags.append("eod_exit")

            all_cases.append(_case_to_dict(HarnessEvalCase(
                case_id=f"{day}-b{local_bar:03d}-trade",
                date=day,
                bar_of_day=local_bar,
                tags=tuple(dict.fromkeys(tags)),
                notes=f"{contract.right} {contract.strike:.0f}",
            )))

    all_cases.sort(key=lambda c: (c["date"], c["bar_of_day"], c["case_id"]))
    core_regression = _pick_core_cases(all_cases)
    core_ids = {case["case_id"] for case in core_regression}
    remaining = [case for case in all_cases if case["case_id"] not in core_ids]
    optimization, holdout = _split_remaining_cases(remaining)

    suite = {
        "schema_version": "v1",
        "metadata": {
            "data_path": data_path,
            "n_cases": len(all_cases),
            "n_days": len(set(case["date"] for case in all_cases)),
        },
        "core_regression": core_regression,
        "optimization": optimization,
        "holdout": holdout,
    }
    dump_suite(suite_path, suite)
    print(f"Built harness suite at {suite_path}")
    print(f"  core_regression={len(core_regression)} optimization={len(optimization)} holdout={len(holdout)}")
    return suite


def _evaluate_case_with_context(case: dict, sidecar: dict, day_indices: np.ndarray,
                                spot_day: np.ndarray, sim_features_day: np.ndarray,
                                day: str) -> dict:
    """Evaluate a single case using pre-loaded day context."""
    local_bar = int(case["bar_of_day"])
    actual_count = _snapshot_count(sidecar, local_bar)
    result = {"case_id": case["case_id"], "tags": case["tags"], "pass": True, "checks": {}}

    if "partially_observed" in case["tags"]:
        result["checks"]["snapshot_flagged_partial"] = actual_count == 0 or not bool(sidecar["bar_labelable"][local_bar])
        if "forward_gap" in case["tags"]:
            result["checks"]["forward_gap_unlabeled"] = not bool(sidecar["bar_labelable"][local_bar])
        result["pass"] = all(bool(v) for v in result["checks"].values())
        return result

    if actual_count <= 0:
        result["pass"] = False
        result["checks"]["best_contract_present"] = False
        return result

    feats, labels, contract_indices = padded_snapshot(sidecar, local_bar, actual_count)
    best_local = int(sidecar["bar_best_contract_idx"][local_bar])
    if best_local < 0 or best_local >= len(contract_indices):
        result["pass"] = False
        result["checks"]["best_contract_present"] = False
        return result

    contract_idx = int(contract_indices[best_local])
    contract = describe_contract(sidecar, contract_idx)
    strike = float(contract.strike)
    right = contract.right
    spot_now = float(spot_day[local_bar])
    row = feats[best_local]
    expected_moneyness = (strike - spot_now) / spot_now * 100.0 if spot_now > 0 else 0.0

    result["checks"]["contract_identity"] = contract_idx >= 0 and contract.expiry == day.replace("-", "") and right in {"C", "P"}
    result["checks"]["spot_alignment"] = abs(float(row[11]) - expected_moneyness) < 1e-3

    series = sidecar["contract_mid"][contract_idx].astype(np.float32)
    intent = TradeIntent(
        trade=True,
        expiry=contract.expiry,
        strike=contract.strike,
        right=contract.right,
        qty=DEFAULT_POLICY.qty,
        entry_ref_price=float(series[local_bar]),
        order_style=DEFAULT_POLICY.order_style,
        tif=DEFAULT_POLICY.tif,
        stop_price=float(series[local_bar]) * (1.0 - DEFAULT_POLICY.stop_pct),
        take_profit_price=float(series[local_bar]) * (1.0 + DEFAULT_POLICY.target_pct),
        max_hold_bars=DEFAULT_POLICY.max_hold_bars,
        exit_policy=DEFAULT_POLICY.exit_policy,
        confidence=0.5,
        reason_codes=("harness_eval",),
        bar_index=local_bar,
        underlying_price=spot_now,
    )
    n_day_bars = len(day_indices)
    trade = simulate_trade(
        intent,
        series,
        sim_features_day,
        np.arange(n_day_bars, dtype=np.int32),
        [day] * n_day_bars,
        local_bar,
    )

    expected = float(labels[best_local])
    result["checks"]["execution_fidelity"] = trade is not None and np.isfinite(expected) and abs(float(trade.net_pnl_pct) - expected) < 1e-4
    result["checks"]["risk_management"] = trade is not None and trade.bars_held <= DEFAULT_POLICY.max_hold_bars and trade.exit_reason in VALID_EXIT_REASONS
    if "eod_exit" in case["tags"]:
        result["checks"]["eod_exit"] = trade is not None and trade.exit_reason == "EOD"

    result["pass"] = all(bool(v) for v in result["checks"].values())
    return result


def run_suite(data_path: str = "v2/data.pt", suite_path: str = SUITE_PATH) -> dict:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    if not os.path.exists(suite_path):
        build_suite(data_path, suite_path)
    suite = load_suite(suite_path)
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    sim_features = data["X_sim"].numpy() if "X_sim" in data else data["X"].numpy()
    spot = data["spot_prices"].numpy()
    dates_arr = np.asarray(data["dates"])

    # Pre-compute day indices once
    day_index_cache: dict[str, np.ndarray] = {}
    for day in sorted(set(data["dates"])):
        day_index_cache[day] = np.where(dates_arr == day)[0]

    out = {}
    for split in ("core_regression", "optimization", "holdout"):
        cases = suite.get(split, [])
        # Group cases by day to load each sidecar once
        day_cases: dict[str, list[dict]] = defaultdict(list)
        for case in cases:
            day_cases[case["date"]].append(case)

        results: list[dict] = []
        for day in sorted(day_cases):
            day_indices = day_index_cache[day]
            spot_day = spot[day_indices]
            sim_day = sim_features[day_indices]
            sc = load_sidecar_cached(sidecar_path(sidecar_dir, day))
            for case in day_cases[day]:
                results.append(_evaluate_case_with_context(
                    case, sc, day_indices, spot_day, sim_day, day))

        passed = sum(1 for r in results if r["pass"])
        out[split] = {"passed": passed, "total": len(results), "results": results}
        print(f"{split}: {passed}/{len(results)} passed")
    return out


def main():
    parser = argparse.ArgumentParser(description="Build/run the v4 harness eval suite")
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--suite", type=str, default=SUITE_PATH)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.build:
        build_suite(args.data, args.suite)
    else:
        run_suite(args.data, args.suite)


if __name__ == "__main__":
    main()
