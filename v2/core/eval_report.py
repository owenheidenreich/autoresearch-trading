"""EvalReport: durable evaluation artifact with stored trades.

An EvalReport captures the full output of a replay evaluation:
- All trades as serialized dicts (enables re-scoring without re-replaying)
- Dollar and percentage metrics
- Daily equity curve
- Per-fold breakdowns
- Provenance fingerprints

Created 2026-04-15. See v2/docs/incidents/2026-04-15-pf-metric-bug.md for why
durable eval artifacts with stored trades are necessary.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from v2.core.metrics import ReplayMetrics, score_config_fingerprint


@dataclass
class EvalReport:
    """Durable evaluation result with full trade provenance."""

    # Identity
    report_id: str = ""
    report_timestamp: str = ""
    experiment_id: str = ""

    # Provenance fingerprints
    model_fingerprint: str = ""
    dataset_fingerprint: str = ""
    config_fingerprint: str = ""
    evaluator_fingerprint: str = ""
    policy_fingerprint: str = ""

    # Raw trades (the key: enables re-scoring without re-replaying)
    trades: list[dict] = field(default_factory=list)

    # Daily equity curve (enables plot regeneration from report)
    daily_equity_curve: list[float] = field(default_factory=list)
    daily_dates: list[str] = field(default_factory=list)

    # Dollar metrics (PRIMARY — the economic truth)
    dollar_pf: float = 0.0
    dollar_gross_profit: float = 0.0
    dollar_gross_loss: float = 0.0
    dollar_net_pnl: float = 0.0

    # Percentage metrics (diagnostic only)
    pct_pf: float = 0.0

    # Full metrics snapshot
    metrics: dict = field(default_factory=dict)

    # Per-fold breakdown (for walk-forward)
    per_fold: list[dict] = field(default_factory=list)

    # Baselines
    baselines: dict = field(default_factory=dict)
    beats_all_baselines: bool = False

    # Promotion
    score: float = 0.0
    gate_failure: str | None = None

    def to_json(self, path: str) -> None:
        """Save report to JSON file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def from_json(cls, path: str) -> EvalReport:
        """Load report from JSON file."""
        d = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in __import__('dataclasses').fields(cls)}})

    @classmethod
    def from_replay(
        cls,
        metrics: ReplayMetrics,
        trades: list,
        experiment_id: str = "",
        model_fingerprint: str = "",
        dataset_fingerprint: str = "",
        config_fingerprint: str = "",
        policy_fingerprint: str = "",
    ) -> EvalReport:
        """Build an EvalReport from replay results."""
        # Serialize trades
        trade_dicts = []
        for t in trades:
            if hasattr(t, 'to_dict'):
                trade_dicts.append(t.to_dict())
            elif isinstance(t, dict):
                trade_dicts.append(t)

        # Build equity curve from metrics
        equity_curve = []
        equity = metrics.starting_equity
        equity_curve.append(equity)
        for dr in metrics.daily_returns:
            equity += dr * metrics.starting_equity
            equity_curve.append(equity)

        return cls(
            report_id=str(uuid.uuid4())[:8],
            report_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            experiment_id=experiment_id,
            model_fingerprint=model_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=config_fingerprint,
            evaluator_fingerprint=score_config_fingerprint(),
            policy_fingerprint=policy_fingerprint,
            trades=trade_dicts,
            daily_equity_curve=equity_curve,
            dollar_pf=metrics.profit_factor,
            dollar_gross_profit=metrics.gross_profit,
            dollar_gross_loss=metrics.gross_loss,
            dollar_net_pnl=metrics.net_pnl,
            pct_pf=metrics.pct_profit_factor,
            metrics=metrics.to_dict(),
            score=metrics.score,
            gate_failure=metrics.gate_failure,
        )
