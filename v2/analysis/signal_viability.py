"""Signal viability from per-fold replay traces.

Answers: does the model's score rank bars usefully? Would any quantile/top-K
gate produce positive expectancy and DD < 25%? Is there an oracle ceiling
achievable even under perfect ranking?

Usage:
    python3 -m v2.analysis.signal_viability \
        --traces v2/artifacts/exp_171/folds/09802c942e02b9b6/replay_traces.csv \
                 v2/artifacts/exp_171/folds/cf38c16e2f55ddd9/replay_traces.csv \
                 v2/artifacts/exp_171/folds/e56a4d66770d7097/replay_traces.csv \
                 v2/artifacts/exp_171/folds/9b92333bccfcbbb7/replay_traces.csv \
                 v2/artifacts/exp_171/folds/3b2f7c5202c9ee35/replay_traces.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(len(ra)))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    if da == 0 or db == 0:
        return float("nan")
    return num / (da * db)


def profit_factor(pnls: list[float]) -> float:
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def equity_dd(daily_pnls: list[float], start: float = 1.0) -> float:
    curve = [start]
    for p in daily_pnls:
        curve.append(curve[-1] * (1.0 + p))
    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
        else:
            dd = 1.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _float(x: str) -> float:
    try:
        if x == "" or x.lower() == "nan":
            return float("nan")
        return float(x)
    except ValueError:
        return float("nan")


def load_trace(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def trade_subset(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r["decision"] != "trade":
            continue
        pnl = _float(r["model_pnl"])
        if math.isnan(pnl):
            continue
        out.append(r)
    return out


def eligible_bars(rows: list[dict]) -> list[dict]:
    """Bars where a trade decision could have been made (not in_position / cooldown)."""
    out = []
    for r in rows:
        skip = r.get("skip_reason", "")
        if r["decision"] == "trade":
            out.append(r)
            continue
        if skip == "" or skip in ("gate", "loss_cap", "equity_zero"):
            out.append(r)
    return out


def pnls_to_daily(rows: list[dict], pnl_field: str) -> list[float]:
    by_day = defaultdict(float)
    dates = sorted({r["date"] for r in rows})
    for r in rows:
        p = _float(r[pnl_field])
        if not math.isnan(p):
            by_day[r["date"]] += p
    return [by_day.get(d, 0.0) for d in dates]


def pct_to_equity_daily(per_trade_pct: list[float], risk_per_trade: float = 0.01) -> list[float]:
    """Convert per-trade pct PnL (model_pnl is pct of entry cost) to fractional equity change.

    We approximate: each trade risks `risk_per_trade` of equity; PnL fraction = pct * risk_per_trade.
    This lets DD/equity math be comparable across slices (not a dollar calc — relative only).
    """
    return [p * risk_per_trade for p in per_trade_pct]


def slice_stats(rows: list[dict], pnl_field: str, label: str, risk_per_trade: float = 0.01) -> dict:
    pcts = [_float(r[pnl_field]) for r in rows]
    pcts = [p for p in pcts if not math.isnan(p)]
    if not pcts:
        return {
            "label": label,
            "n_trades": 0,
            "avg_pnl": float("nan"),
            "win_rate": float("nan"),
            "pf": float("nan"),
            "dd": float("nan"),
        }
    wins = sum(1 for p in pcts if p > 0)
    pf = profit_factor(pcts)

    by_day = defaultdict(float)
    for r in rows:
        p = _float(r[pnl_field])
        if not math.isnan(p):
            by_day[r["date"]] += p * risk_per_trade
    dates = sorted(by_day.keys())
    daily_equity_returns = [by_day[d] for d in dates]
    dd = equity_dd(daily_equity_returns)

    return {
        "label": label,
        "n_trades": len(pcts),
        "avg_pnl": sum(pcts) / len(pcts),
        "win_rate": wins / len(pcts),
        "pf": pf,
        "dd": dd,
        "sum_pct": sum(pcts),
    }


def decile_table(trades: list[dict]) -> list[dict]:
    sorted_by_score = sorted(trades, key=lambda r: _float(r["best_contract_score"]))
    n = len(sorted_by_score)
    out = []
    for d in range(10):
        lo = d * n // 10
        hi = (d + 1) * n // 10 if d < 9 else n
        slab = sorted_by_score[lo:hi]
        pcts = [_float(r["model_pnl"]) for r in slab if not math.isnan(_float(r["model_pnl"]))]
        if not pcts:
            continue
        out.append({
            "decile": d + 1,
            "n": len(pcts),
            "score_lo": _float(slab[0]["best_contract_score"]) if slab else float("nan"),
            "score_hi": _float(slab[-1]["best_contract_score"]) if slab else float("nan"),
            "avg_pnl": sum(pcts) / len(pcts),
            "win_rate": sum(1 for p in pcts if p > 0) / len(pcts),
            "pf": profit_factor(pcts),
        })
    return out


def quantile_gate(trades: list[dict], qs: list[float]) -> list[dict]:
    out = []
    sorted_trades = sorted(trades, key=lambda r: _float(r["best_contract_score"]))
    n = len(sorted_trades)
    for q in qs:
        cutoff_idx = int(q * n)
        kept = sorted_trades[cutoff_idx:]
        stats = slice_stats(kept, "model_pnl", f"q>={q:.1f}")
        stats["threshold"] = _float(sorted_trades[cutoff_idx]["best_contract_score"]) if cutoff_idx < n else float("nan")
        out.append(stats)
    return out


def oracle_top_k(trades: list[dict], ks: list[int]) -> list[dict]:
    """Pick top-K bars per day by oracle_pnl (hindsight). Sets physics ceiling."""
    out = []
    by_day = defaultdict(list)
    for r in trades:
        by_day[r["date"]].append(r)
    for k in ks:
        kept = []
        for _, day_rows in by_day.items():
            ranked = sorted(day_rows, key=lambda r: -_float(r["oracle_pnl"]))
            kept.extend(ranked[:k])
        stats = slice_stats(kept, "model_pnl", f"oracle-top-{k}")
        out.append(stats)
    return out


def model_top_k_per_day(trades: list[dict], ks: list[int]) -> list[dict]:
    out = []
    by_day = defaultdict(list)
    for r in trades:
        by_day[r["date"]].append(r)
    for k in ks:
        kept = []
        for _, day_rows in by_day.items():
            ranked = sorted(day_rows, key=lambda r: -_float(r["best_contract_score"]))
            kept.extend(ranked[:k])
        stats = slice_stats(kept, "model_pnl", f"model-top-{k}")
        out.append(stats)
    return out


def side_by_regime(trades: list[dict]) -> list[dict]:
    """Split trades into 4 quartiles of vix_regime; call% + PF per quartile."""
    regimes = [_float(r["vix_regime"]) for r in trades]
    if not regimes:
        return []
    rs = sorted(regimes)
    qs = [rs[len(rs) // 4], rs[len(rs) // 2], rs[3 * len(rs) // 4]]
    buckets = [[] for _ in range(4)]
    for r in trades:
        v = _float(r["vix_regime"])
        if v <= qs[0]:
            buckets[0].append(r)
        elif v <= qs[1]:
            buckets[1].append(r)
        elif v <= qs[2]:
            buckets[2].append(r)
        else:
            buckets[3].append(r)
    out = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        ncall = sum(1 for r in b if r["selected_right"] == "C")
        total = len(b)
        pnls = [_float(r["model_pnl"]) for r in b]
        pnls = [p for p in pnls if not math.isnan(p)]
        out.append({
            "quartile": i + 1,
            "vix_range": f"≤{qs[i-1]:.2f}→≤{qs[i]:.2f}" if i < 3 else f">{qs[2]:.2f}",
            "n": total,
            "call_pct": ncall / total if total else 0.0,
            "avg_pnl": sum(pnls) / len(pnls) if pnls else float("nan"),
            "pf": profit_factor(pnls),
        })
    return out


def analyze_fold(path: str) -> dict:
    rows = load_trace(path)
    trades = trade_subset(rows)
    fold_id = path.split("/")[-2][:8]
    date = rows[0]["date"] if rows else "?"
    last_date = rows[-1]["date"] if rows else "?"

    scores = [_float(r["best_contract_score"]) for r in trades]
    pnls = [_float(r["model_pnl"]) for r in trades]
    rho = spearman(scores, pnls)

    gate_logits = [_float(r["gate_logit"]) for r in trades]
    rho_gate = spearman(gate_logits, pnls)

    return {
        "fold_id": fold_id,
        "date_range": f"{date}→{last_date}",
        "n_eligible_bars": len(rows),
        "n_trades": len(trades),
        "tpd": len(trades) / max(1, len({r["date"] for r in trades})),
        "rho_contract_score_pnl": rho,
        "rho_gate_score_pnl": rho_gate,
        "deciles": decile_table(trades),
        "quantile_gate": quantile_gate(trades, [0.0, 0.3, 0.5, 0.7, 0.9]),
        "oracle_top_k": oracle_top_k(trades, [1, 2, 3, 5]),
        "model_top_k": model_top_k_per_day(trades, [1, 2, 3, 5]),
        "side_by_regime": side_by_regime(trades),
        "full_trade_stats": slice_stats(trades, "model_pnl", "all-trades"),
    }


def render(analyses: list[dict]) -> str:
    out = ["# Signal viability — exp_171 per-fold traces", ""]

    out.append("## Rank correlations (Spearman ρ with model_pnl)")
    out.append("")
    out.append("| fold | window | n_trades | TPD | ρ(best_contract_score, pnl) | ρ(gate_logit, pnl) |")
    out.append("|---|---|---|---|---|---|")
    for a in analyses:
        out.append(
            f"| {a['fold_id']} | {a['date_range']} | {a['n_trades']} | {a['tpd']:.2f} | "
            f"{a['rho_contract_score_pnl']:+.3f} | {a['rho_gate_score_pnl']:+.3f} |"
        )
    out.append("")

    out.append("## Decile table per fold (best_contract_score → avg_pnl, PF)")
    out.append("")
    for a in analyses:
        out.append(f"### Fold {a['fold_id']}")
        out.append("")
        out.append("| decile | n | score_lo | score_hi | avg_pnl | WR | PF |")
        out.append("|---|---|---|---|---|---|---|")
        for d in a["deciles"]:
            pf_s = f"{d['pf']:.3f}" if not math.isinf(d["pf"]) and not math.isnan(d["pf"]) else "—"
            out.append(
                f"| {d['decile']} | {d['n']} | {d['score_lo']:+.3f} | {d['score_hi']:+.3f} | "
                f"{d['avg_pnl']:+.4f} | {d['win_rate']:.1%} | {pf_s} |"
            )
        out.append("")

    out.append("## Quantile gating — filter top-(1-q) of model trades by best_contract_score")
    out.append("")
    out.append("| fold | q-cutoff | threshold | n | avg_pnl | WR | PF | DD |")
    out.append("|---|---|---|---|---|---|---|---|")
    for a in analyses:
        for s in a["quantile_gate"]:
            pf_s = f"{s['pf']:.3f}" if not math.isinf(s["pf"]) and not math.isnan(s["pf"]) else "—"
            out.append(
                f"| {a['fold_id']} | {s['label']} | {s.get('threshold', float('nan')):+.3f} | "
                f"{s['n_trades']} | {s['avg_pnl']:+.4f} | {s['win_rate']:.1%} | {pf_s} | {s['dd']:.1%} |"
            )
    out.append("")

    out.append("## Model top-K per day (model's own ranking of its trade bars)")
    out.append("")
    out.append("| fold | slice | n | avg_pnl | WR | PF | DD |")
    out.append("|---|---|---|---|---|---|---|")
    for a in analyses:
        for s in a["model_top_k"]:
            pf_s = f"{s['pf']:.3f}" if not math.isinf(s["pf"]) and not math.isnan(s["pf"]) else "—"
            out.append(
                f"| {a['fold_id']} | {s['label']} | {s['n_trades']} | {s['avg_pnl']:+.4f} | "
                f"{s['win_rate']:.1%} | {pf_s} | {s['dd']:.1%} |"
            )
    out.append("")

    out.append("## Oracle top-K per day (PHYSICS CEILING — hindsight ranking by oracle_pnl)")
    out.append("")
    out.append("| fold | slice | n | avg_pnl | WR | PF | DD |")
    out.append("|---|---|---|---|---|---|---|")
    for a in analyses:
        for s in a["oracle_top_k"]:
            pf_s = f"{s['pf']:.3f}" if not math.isinf(s["pf"]) and not math.isnan(s["pf"]) else "—"
            out.append(
                f"| {a['fold_id']} | {s['label']} | {s['n_trades']} | {s['avg_pnl']:+.4f} | "
                f"{s['win_rate']:.1%} | {pf_s} | {s['dd']:.1%} |"
            )
    out.append("")

    out.append("## Side bias × vix regime quartile")
    out.append("")
    out.append("| fold | Q | vix_range | n | call_pct | avg_pnl | PF |")
    out.append("|---|---|---|---|---|---|---|")
    for a in analyses:
        for s in a["side_by_regime"]:
            pf_s = f"{s['pf']:.3f}" if not math.isinf(s["pf"]) and not math.isnan(s["pf"]) else "—"
            out.append(
                f"| {a['fold_id']} | {s['quartile']} | {s['vix_range']} | {s['n']} | "
                f"{s['call_pct']:.1%} | {s['avg_pnl']:+.4f} | {pf_s} |"
            )
    out.append("")

    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", nargs="+", required=True, help="Per-fold replay_traces.csv paths")
    p.add_argument("--out", default=None, help="Write markdown here; omit for stdout")
    args = p.parse_args()

    analyses = [analyze_fold(t) for t in args.traces]
    md = render(analyses)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
