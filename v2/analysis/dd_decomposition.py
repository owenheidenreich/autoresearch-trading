"""Per-fold drawdown decomposition from cv_report.json.

Answers: *where* does the DD come from? Tail days, uniform bleed, symmetric
expectancy, or exit-reason asymmetry?

Usage:
    python3 -m v2.analysis.dd_decomposition \
        --cv-report v2/artifacts/exp_171/cv_report.json
"""
from __future__ import annotations

import argparse
import json
import math
from statistics import mean


def equity_curve(daily_returns: list[float], start: float = 1.0) -> list[float]:
    curve = [start]
    for r in daily_returns:
        curve.append(curve[-1] * (1.0 + r))
    return curve[1:]


def max_drawdown(curve: list[float]) -> tuple[float, int, int, int]:
    peak = curve[0]
    peak_idx = 0
    max_dd = 0.0
    trough_idx = 0
    dd_peak_idx = 0
    for i, v in enumerate(curve):
        if v > peak:
            peak = v
            peak_idx = i
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
            trough_idx = i
            dd_peak_idx = peak_idx
    recovery_idx = -1
    if trough_idx < len(curve) - 1:
        target = curve[dd_peak_idx]
        for j in range(trough_idx + 1, len(curve)):
            if curve[j] >= target:
                recovery_idx = j
                break
    return max_dd, dd_peak_idx, trough_idx, recovery_idx


def longest_losing_streak(daily_returns: list[float]) -> int:
    best = cur = 0
    for r in daily_returns:
        if r < 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def kurtosis(xs: list[float]) -> float:
    if len(xs) < 4:
        return float("nan")
    m = mean(xs)
    n = len(xs)
    var = sum((x - m) ** 2 for x in xs) / n
    if var <= 0:
        return float("nan")
    m4 = sum((x - m) ** 4 for x in xs) / n
    return m4 / (var ** 2) - 3.0


def skew(xs: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    m = mean(xs)
    n = len(xs)
    var = sum((x - m) ** 2 for x in xs) / n
    if var <= 0:
        return float("nan")
    m3 = sum((x - m) ** 3 for x in xs) / n
    return m3 / (var ** 1.5)


def decompose_fold(fold: dict) -> dict:
    m = fold["metrics"]
    returns = m["daily_returns"]
    curve = equity_curve(returns)
    dd, peak_idx, trough_idx, recov_idx = max_drawdown(curve)
    streak = longest_losing_streak(returns)

    losing = [r for r in returns if r < 0]
    winning = [r for r in returns if r > 0]
    total_neg = sum(losing) if losing else 0.0
    worst5 = sorted(losing)[:5]
    worst5_contrib = (sum(worst5) / total_neg) if total_neg < 0 and worst5 else 0.0
    worst1pct = percentile(returns, 0.01)
    worst5pct = percentile(returns, 0.05)

    expectancy_per_trade = (
        m["win_rate"] * m["avg_win"] + (1.0 - m["win_rate"]) * m["avg_loss"]
    )

    tp_count = m.get("take_profit_count", 0)
    sl_count = m.get("stop_loss_count", 0)
    tr_count = m.get("trailing_stop_count", 0)
    eod_count = m.get("eod_count", 0)
    other = m.get("max_hold_count", 0)

    return {
        "fold_idx": fold["fold_idx"],
        "window": f"{fold['test_window_start']}→{fold['test_window_end']}",
        "num_days": m["num_days"],
        "traded_days": m["traded_days"],
        "net_pnl": m["net_pnl"],
        "account_dd_reported": m["max_account_drawdown"],
        "account_dd_recomputed": dd,
        "dd_peak_day": peak_idx,
        "dd_trough_day": trough_idx,
        "dd_recovered_day": recov_idx,
        "longest_losing_streak": streak,
        "pos_days": sum(1 for r in returns if r > 0),
        "neg_days": sum(1 for r in returns if r < 0),
        "flat_days": sum(1 for r in returns if r == 0),
        "worst1pct": worst1pct,
        "worst5pct": worst5pct,
        "worst5_contrib_pct_of_all_loss": worst5_contrib,
        "return_mean": mean(returns),
        "return_std": (sum((r - mean(returns)) ** 2 for r in returns) / len(returns)) ** 0.5,
        "return_skew": skew(returns),
        "return_kurt": kurtosis(returns),
        "win_rate": m["win_rate"],
        "avg_win": m["avg_win"],
        "avg_loss": m["avg_loss"],
        "expectancy_per_trade": expectancy_per_trade,
        "trades_per_day": m["trades_per_day"],
        "call_pct": m.get("call_pct", 0.0),
        "put_pct": m.get("put_pct", 0.0),
        "exits": {
            "stop_loss": sl_count,
            "trailing": tr_count,
            "take_profit": tp_count,
            "eod": eod_count,
            "max_hold": other,
        },
    }


def attribute(summary: dict) -> str:
    """Return one-word attribution of DD cause per fold."""
    exp = summary["expectancy_per_trade"]
    tpd = summary["trades_per_day"]
    worst5 = summary["worst5_contrib_pct_of_all_loss"]
    kurt = summary["return_kurt"]
    streak = summary["longest_losing_streak"]
    num_days = summary["num_days"]

    causes = []
    if exp < -0.01 and tpd > 5:
        causes.append("neg-expectancy×frequency")
    if worst5 > 0.5 and kurt > 2.0:
        causes.append("tail-days")
    if streak >= max(5, num_days // 8):
        causes.append("long-losing-streak")
    if summary["avg_win"] + summary["avg_loss"] < -0.02:
        causes.append("loss-size-asymmetry")
    if not causes:
        causes.append("uniform-bleed")
    return ", ".join(causes)


def render_markdown(folds_out: list[dict], pooled: dict) -> str:
    lines = ["# Drawdown decomposition — exp_171", ""]
    lines.append("## Per-fold summary")
    lines.append("")
    lines.append(
        "| fold | window | days | WR | avg_win | avg_loss | expectancy | TPD | DD(reported) | DD(recomp) | longest_loss_streak | worst-5 contrib | skew | kurt | attribution |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for s in folds_out:
        attr = attribute(s)
        lines.append(
            "| {fold_idx} | {window} | {days} | {wr:.3f} | {aw:+.3f} | {al:+.3f} | {exp:+.4f} | {tpd:.2f} | {ddr:.2%} | {ddc:.2%} | {streak} | {worst5:.2%} | {skew:+.2f} | {kurt:+.2f} | {attr} |".format(
                fold_idx=s["fold_idx"],
                window=s["window"],
                days=s["num_days"],
                wr=s["win_rate"],
                aw=s["avg_win"],
                al=s["avg_loss"],
                exp=s["expectancy_per_trade"],
                tpd=s["trades_per_day"],
                ddr=s["account_dd_reported"],
                ddc=s["account_dd_recomputed"],
                streak=s["longest_losing_streak"],
                worst5=s["worst5_contrib_pct_of_all_loss"],
                skew=s["return_skew"],
                kurt=s["return_kurt"],
                attr=attr,
            )
        )
    lines.append("")

    lines.append("## Exit-reason breakdown")
    lines.append("")
    lines.append("| fold | stop_loss | trailing | take_profit | eod | max_hold | total |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in folds_out:
        e = s["exits"]
        tot = sum(e.values())
        lines.append(
            f"| {s['fold_idx']} | {e['stop_loss']} | {e['trailing']} | {e['take_profit']} | {e['eod']} | {e['max_hold']} | {tot} |"
        )
    lines.append("")

    lines.append("## Side bias")
    lines.append("")
    lines.append("| fold | call% | put% |")
    lines.append("|---|---|---|")
    for s in folds_out:
        lines.append(f"| {s['fold_idx']} | {s['call_pct']:.1%} | {s['put_pct']:.1%} |")
    lines.append("")

    lines.append("## Pooled reported")
    lines.append("")
    for k, v in pooled.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Attribution counts")
    lines.append("")
    counts: dict[str, int] = {}
    for s in folds_out:
        for tok in attribute(s).split(", "):
            counts[tok] = counts.get(tok, 0) + 1
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}/{len(folds_out)} folds")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cv-report", default="v2/artifacts/exp_171/cv_report.json")
    p.add_argument("--out", default=None, help="Write markdown here; omit to stdout")
    args = p.parse_args()

    with open(args.cv_report) as f:
        report = json.load(f)

    folds_out = [decompose_fold(f) for f in report["folds"]]
    pooled = report.get("pooled", {})
    md = render_markdown(folds_out, pooled)

    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
