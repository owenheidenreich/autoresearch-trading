"""Frozen Phase 0 spec for the L2 post-selection veto layer.

This is the immutable rule contract for `phase_c_gpt55_v0_2026_04_26`. Once
Phase 1 validation runs against these definitions, any threshold change must
become a new versioned spec (`phase_c_gpt55_v1_*`) with its own validation
cycle. Threshold retuning after observing Phase 1 / forward-walk results is
explicitly prohibited.

Origin: GPT-5.5 independent analysis of v3/artifacts/research/phase_c_trade_review.csv
(1664-trade Phase C sample, 5 seeds, OOS bars 30-119).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd


FILTER_SET_ID = "phase_c_gpt55_v0_2026_04_26"


# Frozen sigma_pos / iv_percentile tertile cutoffs.
# Derived once from the 1664-trade Phase C CSV via pd.qcut(..., 3) at the
# 1/3 and 2/3 quantiles. Forward-walk and live deployment MUST use these
# verbatim — recomputing tertiles on a different sample lets future
# distribution information leak into rule definitions.
SIGMA_POS_TERTILES = (-0.5039522052, 1.2239481211)
IV_PERCENTILE_TERTILES = (0.1632336179, 0.5469231009)


def assign_bucket(value: float, tertiles: tuple[float, float]) -> Optional[int]:
    """Return 0/1/2 bucket index for a scalar value using frozen tertiles.

    Matches pandas.qcut(..., 3, duplicates='drop') boundary semantics on the
    Phase C sample exactly (verified at 1.0000 agreement).
    """
    if pd.isna(value):
        return None
    if value <= tertiles[0]:
        return 0
    if value <= tertiles[1]:
        return 1
    return 2


def attach_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add sigma_b / iv_b / cell columns using frozen tertile cutoffs.

    Does not mutate the input frame. Use this in every validation /
    forward-walk path so cell membership is computed identically everywhere.
    """
    out = df.copy()
    out["sigma_b"] = out["sigma_pos"].map(lambda v: assign_bucket(v, SIGMA_POS_TERTILES))
    out["iv_b"] = out["iv_percentile"].map(lambda v: assign_bucket(v, IV_PERCENTILE_TERTILES))

    def _cell(row):
        sb, ib = row["sigma_b"], row["iv_b"]
        if sb is None or ib is None:
            return None
        return f"s{int(sb)}_iv{int(ib)}"

    out["cell"] = out.apply(_cell, axis=1)
    return out


@dataclass(frozen=True)
class FilterRule:
    rule_id: str
    direction: str  # "avoid" or "favor"
    side: str  # "call", "put", or "any"
    description: str
    causal_rationale: str
    sample_size: int
    in_sample_pf: float
    predicate: Callable[[pd.Series], bool]
    tags: tuple[str, ...] = field(default_factory=tuple)


def _is_orc(row: pd.Series) -> bool:
    val = row.get("orc_triggered", False)
    if isinstance(val, bool):
        return val
    return bool(val) and not pd.isna(val)


# --------------------------------------------------------------------------
# Avoid rules (eligible for runtime veto if Phase 1 + FW pass)
# --------------------------------------------------------------------------

AVOID_RULES: List[FilterRule] = [
    FilterRule(
        rule_id="avoid_call_orc_triggered",
        direction="avoid",
        side="call",
        description="Call entries on bars where ORC trigger fired.",
        causal_rationale=(
            "Phase C side-error analysis: ORC's directional fire cluster on "
            "the wrong side of VWAP — ORC is a sigma-extreme momentum trigger "
            "that historically failed when bought as calls in chop, since the "
            "trigger frequently captures a fade rather than a continuation."
        ),
        sample_size=43,
        in_sample_pf=0.39,
        predicate=lambda r: r["side"] == "call" and _is_orc(r),
        tags=("phase_c_side_error", "orc_calls"),
    ),
    FilterRule(
        rule_id="avoid_call_cell_s0_iv2",
        direction="avoid",
        side="call",
        description="Call entries in cell s0_iv2 (low sigma, high IV).",
        causal_rationale=(
            "Phase C per-cell PF: s0_iv2 calls average PF 0.68 over 69 trades. "
            "Cell-conditional side preference: s0_iv2 favours puts (PF 1.75); "
            "calls in this cell are the opposite side from the regime-coherent "
            "preference."
        ),
        sample_size=69,
        in_sample_pf=0.68,
        predicate=lambda r: r["side"] == "call" and r.get("cell") == "s0_iv2",
        tags=("phase_c_wrong_side", "cell_conditional"),
    ),
    FilterRule(
        rule_id="avoid_put_high_decision_margin",
        direction="avoid",
        side="put",
        description="Put entries with decision_margin >= 0.694.",
        causal_rationale=(
            "L2 anti-calibration on puts: high decision_margin on puts is "
            "associated with WORSE forward PF (Phase 1 floor-cell analysis "
            "showed pred_dollar_score is most negative when realized PnL is "
            "most positive; this rule captures the corresponding inversion "
            "in the side-margin scalar). Sample-size note: this spec yields "
            "n=108 vs GPT-5.5's reported 107 — no row sits exactly at the "
            "0.694 boundary in either direction; the 1-row delta is a rounding "
            "artifact in GPT-5.5's stratification, not a threshold question."
        ),
        sample_size=108,
        in_sample_pf=0.53,
        predicate=lambda r: (
            r["side"] == "put"
            and r.get("decision_margin") is not None
            and not pd.isna(r.get("decision_margin"))
            and r["decision_margin"] >= 0.694
        ),
        tags=("anti_calibration", "self_leakage_risk"),
    ),
    FilterRule(
        rule_id="avoid_put_late_session_high_sigma",
        direction="avoid",
        side="put",
        description=(
            "Put entries in sigma_b=2 cells with iv_b in {0,1} after bar_index >= 106."
        ),
        causal_rationale=(
            "Late-session puts at sigma_pos extremes with non-extreme IV: cell "
            "preference flips. s2 cells favour calls (s2_iv0 PF 3.11, s2_iv1 "
            "PF 2.78); late-session puts in these cells are wrong-side picks."
        ),
        sample_size=100,
        in_sample_pf=0.53,
        predicate=lambda r: (
            r["side"] == "put"
            and r.get("sigma_b") == 2
            and r.get("iv_b") in (0, 1)
            and r.get("bar_index") is not None
            and not pd.isna(r.get("bar_index"))
            and r["bar_index"] >= 106
        ),
        tags=("phase_c_wrong_side", "late_session"),
    ),
    FilterRule(
        rule_id="avoid_put_low_iv_percentile",
        direction="avoid",
        side="put",
        description="Put entries with iv_percentile <= 0.031.",
        causal_rationale=(
            "Sub-3rd-percentile IV puts: theta carry is unfavourable for "
            "premium-paying directional bets, and floor-cell s0_iv0 puts had "
            "cell preference but the extreme-low IV slice within is dominated "
            "by post-vol-crush bars where realized vol no longer supports "
            "downside extension."
        ),
        sample_size=79,
        in_sample_pf=0.64,
        predicate=lambda r: (
            r["side"] == "put"
            and r.get("iv_percentile") is not None
            and not pd.isna(r.get("iv_percentile"))
            and r["iv_percentile"] <= 0.031
        ),
        tags=("iv_extreme_low", "vol_crush"),
    ),
]


# --------------------------------------------------------------------------
# Favor rules (DIAGNOSTIC ONLY in Phase 2; never used to boost / size / prioritise)
# --------------------------------------------------------------------------

FAVOR_RULES: List[FilterRule] = [
    FilterRule(
        rule_id="favor_put_cell_s1_iv2",
        direction="favor",
        side="put",
        description="Put entries in cell s1_iv2 (mid sigma, high IV).",
        causal_rationale=(
            "Strongest cell-conditional preference: s1_iv2 puts average PF "
            "4.12 over 128 trades. Mean-reversion-from-above dynamic in mid-"
            "sigma high-IV environments."
        ),
        sample_size=128,
        in_sample_pf=4.12,
        predicate=lambda r: r["side"] == "put" and r.get("cell") == "s1_iv2",
    ),
    FilterRule(
        rule_id="favor_call_high_sigma_pos",
        direction="favor",
        side="call",
        description="Call entries with sigma_pos >= 1.228.",
        causal_rationale=(
            "Calls in price-extreme-high-vs-VWAP regime: 219 trades, PF 3.26. "
            "Momentum-continuation thesis at upper sigma extremes."
        ),
        sample_size=219,
        in_sample_pf=3.26,
        predicate=lambda r: (
            r["side"] == "call"
            and r.get("sigma_pos") is not None
            and not pd.isna(r.get("sigma_pos"))
            and r["sigma_pos"] >= 1.228
        ),
    ),
    FilterRule(
        rule_id="favor_put_high_iv_percentile",
        direction="favor",
        side="put",
        description="Put entries with iv_percentile >= 0.742.",
        causal_rationale=(
            "Puts in elevated IV environments: 236 trades, PF 2.68. Premium-"
            "rich downside paths."
        ),
        sample_size=236,
        in_sample_pf=2.68,
        predicate=lambda r: (
            r["side"] == "put"
            and r.get("iv_percentile") is not None
            and not pd.isna(r.get("iv_percentile"))
            and r["iv_percentile"] >= 0.742
        ),
    ),
    FilterRule(
        rule_id="favor_put_high_volume_ratio",
        direction="favor",
        side="put",
        description="Put entries with volume_ratio >= 1.472.",
        causal_rationale=(
            "Puts on elevated-volume bars: 243 trades, PF 3.11. Active "
            "session participation supports follow-through."
        ),
        sample_size=243,
        in_sample_pf=3.11,
        predicate=lambda r: (
            r["side"] == "put"
            and r.get("volume_ratio") is not None
            and not pd.isna(r.get("volume_ratio"))
            and r["volume_ratio"] >= 1.472
        ),
    ),
    FilterRule(
        rule_id="favor_call_low_omar_range_pct_MONITOR_ONLY",
        direction="favor",
        side="call",
        description=(
            "Call entries with omar_range_pct <= 0.000505. MONITOR-ONLY: "
            "n=65 is too small for robust deployment even if Phase 1 looks "
            "favourable."
        ),
        causal_rationale=(
            "Compressed opening-range calls: 65 trades, PF 8.34. Suggests a "
            "coiled-spring breakout pattern. Sample is too small to deploy "
            "as anything other than a monitoring signal."
        ),
        sample_size=65,
        in_sample_pf=8.34,
        predicate=lambda r: (
            r["side"] == "call"
            and r.get("omar_range_pct") is not None
            and not pd.isna(r.get("omar_range_pct"))
            and r["omar_range_pct"] <= 0.000505
        ),
        tags=("monitor_only", "small_sample"),
    ),
]


ALL_RULES: List[FilterRule] = AVOID_RULES + FAVOR_RULES


def rule_index() -> Dict[str, FilterRule]:
    return {r.rule_id: r for r in ALL_RULES}


def apply_rule_mask(df: pd.DataFrame, rule: FilterRule) -> pd.Series:
    """Boolean mask of rows matching this rule. Operates on the prepared
    frame (call attach_buckets first if rule uses cell/sigma_b/iv_b)."""
    return df.apply(rule.predicate, axis=1)


# Veto semantics declaration.
# `drop_only` matches Phase C offline computation (filters are evaluated on
# already-chosen trades, so vetoed candidates simply disappear). Phase 2a
# reads the live pipeline to determine whether existing gates use the same
# semantics; the live veto layer must match them.
VETO_SEMANTICS_DEFAULT = "drop_only"
