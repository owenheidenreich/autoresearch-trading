"""G4 feasibility supplement: FREQUENCY-FORCED oracles.

The v2 feasibility oracle had perfect foresight AND the freedom to skip any
minute without a profitable option — hence $0 drawdown and no calibration
value. The G7 gate requires 0.3-6.0 trades/day: the trader class we are
building must show up. This measurement forces a hindsight-perfect selector
to trade exactly k times per session (the k best-PnL minutes, even when
every option loses), for k in {1, 2, 4}. Its drawdown distribution is the
honest lower bound of achievable drawdown at mandated frequency, and the
anchor for a signable G4.

Also runs the selective-capped diagnostic (positive-only, max 4/day) to
show caps alone do not create drawdown; only forced cadence does.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from v4.scripts.protocol101_training_scope import load_training_scope
from v4.scripts.run_protocol101_stage1_experiment import (
    FEE_PER_TRADE,
    extract_table,
    replay,
    subset,
)

OUT_DIR = Path("v4/audit/autoresearch/protocol101_stage1_g4_forced_oracle_feasibility")
FORCED_K = (1, 2, 4)


def best_per_minute(test: Any) -> dict[str, dict[str, int]]:
    """session -> minute -> index of best-PnL candidate that minute."""
    out: dict[str, dict[str, int]] = {}
    for i, (session, minute) in enumerate(zip(test.session_name, test.decision_time)):
        s, m = str(session), str(minute)
        cur = out.setdefault(s, {})
        if m not in cur or float(test.pnl[i]) > float(test.pnl[cur[m]]):
            cur[m] = i
    return out


def forced_k_selection(test: Any, k: int) -> np.ndarray:
    picks: list[int] = []
    for _, minutes in best_per_minute(test).items():
        ranked = sorted(minutes.values(), key=lambda i: float(test.pnl[i]), reverse=True)
        picks.extend(ranked[:k])
    return np.asarray(sorted(picks), dtype=int)


def capped_selective_selection(test: Any, k: int) -> np.ndarray:
    picks: list[int] = []
    for _, minutes in best_per_minute(test).items():
        ranked = [i for i in sorted(minutes.values(), key=lambda i: float(test.pnl[i]), reverse=True)
                  if float(test.pnl[i]) > FEE_PER_TRADE]
        picks.extend(ranked[:k])
    return np.asarray(sorted(picks), dtype=int)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scope = load_training_scope()
    table = extract_table(scope.sessions)
    session_arr = np.asarray(table.session_name)
    rows: list[dict[str, Any]] = []
    for fold in scope.folds:
        test = subset(table, np.isin(session_arr, fold["test_sessions"]))
        n_sessions = len(fold["test_sessions"])
        results = {}
        for k in FORCED_K:
            sel = forced_k_selection(test, k)
            r = replay(test, sel, FEE_PER_TRADE, f"forced_oracle_k{k}")
            results[f"forced_oracle_k{k}"] = {
                "max_drawdown": float(r["max_drawdown"]),
                "net_pnl": float(r["net_pnl"]),
                "trades": int(len(sel)),
                "trades_per_day": float(len(sel)) / max(n_sessions, 1),
                "calmar": (float(r["net_pnl"]) / float(r["max_drawdown"]))
                          if float(r["max_drawdown"]) > 0 else None,
            }
        sel = capped_selective_selection(test, 4)
        r = replay(test, sel, FEE_PER_TRADE, "selective_capped_k4")
        results["selective_capped_k4"] = {
            "max_drawdown": float(r["max_drawdown"]),
            "net_pnl": float(r["net_pnl"]),
            "trades": int(len(sel)),
            "trades_per_day": float(len(sel)) / max(n_sessions, 1),
        }
        rows.append({"fold": fold["fold"], "test_sessions": n_sessions, "results": results})

    aggregates: dict[str, Any] = {}
    for name in [f"forced_oracle_k{k}" for k in FORCED_K] + ["selective_capped_k4"]:
        dd = [row["results"][name]["max_drawdown"] for row in rows]
        pnl = [row["results"][name]["net_pnl"] for row in rows]
        aggregates[name] = {
            "drawdown_mean": float(np.mean(dd)), "drawdown_max": float(np.max(dd)),
            "net_pnl_mean": float(np.mean(pnl)), "net_pnl_min": float(np.min(pnl)),
            "calmar_worst": (min(p / d for p, d in zip(pnl, dd) if d > 0)
                             if any(d > 0 for d in dd) else None),
        }
    summary = {
        "schema_version": "Protocol101Stage1G4ForcedOracleFeasibilityV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sessions": len(scope.sessions),
        "fee_per_trade": FEE_PER_TRADE,
        "folds": rows,
        "aggregates": aggregates,
        "side_effects": {"model_training_executed": False, "threshold_selection_executed": False,
                         "broker_endpoint_called": False, "paid_data_download": False},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# G4 Forced-Oracle Feasibility", ""]
    for name, agg in aggregates.items():
        lines.append(f"- `{name}`: DD mean ${agg['drawdown_mean']:,.0f} / max ${agg['drawdown_max']:,.0f}; "
                     f"PnL mean ${agg['net_pnl_mean']:,.0f} / min ${agg['net_pnl_min']:,.0f}; "
                     f"worst Calmar {agg['calmar_worst'] if agg['calmar_worst'] is None else round(agg['calmar_worst'], 2)}")
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(aggregates, indent=1))


if __name__ == "__main__":
    main()
