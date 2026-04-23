"""CVReport: scope-separated result schema for walk-forward cross-validation.

Replaces the old blended dict from run_experiment_wf.py that mixed pooled trade
totals with last-fold PF/Sortino/DD. The dataclass shape makes scope-mixing a
type error rather than a convention.

Three independent scopes:
    - folds:     per-fold metrics (FoldSlice), indexed by window_id
    - pooled:    economics across all evaluated folds (PooledSlice)
    - stability: per-fold score distribution (StabilitySlice)

Each answers a different question:
    - Pooled:   "how would a periodically retrained strategy have done overall?"
    - Stability: "how consistent is this config across regimes?"
    - Folds:    "what happened in each specific window?"

Config selection must consider stability, not just pooled economics. A pooled
PF of 1.3 driven by one great window and four losing ones is not the same as
a pooled PF of 1.3 spread evenly across all windows.

See /Users/gduby/.claude/plans/delightful-yawning-tiger.md Appendix D.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "cv_report_v2"


@dataclass(frozen=True)
class FoldSlice:
    """Metrics from one fold of walk-forward CV.

    `window_id` is the SHA-1 hash of the fold's six boundary dates. Same
    calendar window -> same window_id -> same training seed, regardless of
    whether the fold was run alone (screen_latest) or inside a 5-fold job.
    """
    fold_idx: int
    window_id: str
    test_window_start: str
    test_window_end: str
    train_window_start: str
    train_window_end: str
    seed: int
    score: float
    gate_failure: str | None
    metrics: dict  # full ReplayMetrics.to_dict() snapshot
    baseline_scores: dict[str, float] = field(default_factory=dict)
    n_trades: int = 0
    n_test_days: int = 0
    n_traded_days: int = 0
    train_seconds: float = 0.0


@dataclass(frozen=True)
class PooledSlice:
    """Economics across all evaluated folds as if one continuous strategy.

    NOTE: positive_day_rate and daily_sortino in the underlying ReplayMetrics
    are computed over traded days only, not full evaluation horizon. See
    v2/core/metrics.py for the caveat. Pooled metrics here inherit that scope.
    """
    profit_factor: float
    max_account_drawdown: float
    win_rate: float
    call_pct: float
    put_pct: float
    net_pnl_dollars: float
    total_trades: int
    total_eval_days: int
    traded_days: int
    positive_day_rate: float
    daily_sortino: float


@dataclass(frozen=True)
class StabilitySlice:
    """How much does fold score vary across regimes?"""
    mean_fold_score: float
    min_fold_score: float
    max_fold_score: float
    std_fold_score: float
    per_fold_scores: list[float]
    per_fold_gate_failures: list[bool]
    any_fold_gate_failure: bool


@dataclass(frozen=True)
class CVReport:
    """Walk-forward CV result. NOT a deployable model artifact.

    Fingerprint discipline (v2):
      - `training_config_fingerprint` — RuntimeConfig (model size, schema,
        feature count). Same config -> same architecture + data shape.
      - `evaluator_fingerprint`       — score formula (gates, weights). Same
        evaluator -> same score interpretation.
      - `policy_fingerprint`          — DecisionPolicy (side_mode, alpha_side,
        stops, targets). Same policy -> same simulator behavior.
      - `training_env_overrides`      — snapshot of env vars that shaped
        training (SOFT_TEMP, SIDE_SEL_W, etc.). run_final_train replays these
        via os.environ so the deployed model is trained under the same knobs.
    These are deliberately disjoint: a score-formula tweak and a config change
    must invalidate different things.
    """
    experiment_id: str
    screening_mode: str  # "latest" | "mini" | "full"
    folds: list[FoldSlice]
    pooled: PooledSlice
    stability: StabilitySlice
    aggregate_baselines: dict[str, float]
    beats_all_baselines: bool
    training_config_fingerprint: str
    training_env_overrides: dict[str, str]
    policy_fingerprint: str
    dataset_fingerprint: str
    evaluator_fingerprint: str
    training_seconds: float
    schema_version: str = SCHEMA_VERSION

    # ---- serialization ----
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def from_json(cls, path: str) -> CVReport:
        from pathlib import Path
        d = json.loads(Path(path).read_text())
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> CVReport:
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"CVReport schema mismatch: got {d.get('schema_version')}, "
                f"expected {SCHEMA_VERSION}. Rebuild the artifact."
            )
        folds = [FoldSlice(**f) for f in d["folds"]]
        pooled = PooledSlice(**d["pooled"])
        stability = StabilitySlice(**d["stability"])
        return cls(
            experiment_id=d["experiment_id"],
            screening_mode=d["screening_mode"],
            folds=folds,
            pooled=pooled,
            stability=stability,
            aggregate_baselines=d.get("aggregate_baselines", {}),
            beats_all_baselines=d.get("beats_all_baselines", False),
            training_config_fingerprint=d["training_config_fingerprint"],
            training_env_overrides=d.get("training_env_overrides", {}),
            policy_fingerprint=d["policy_fingerprint"],
            dataset_fingerprint=d["dataset_fingerprint"],
            evaluator_fingerprint=d["evaluator_fingerprint"],
            training_seconds=d.get("training_seconds", 0.0),
            schema_version=d["schema_version"],
        )

    # ---- canonical summary ----
    def to_tsv_row(self) -> str:
        """Canonical TSV row: tab-separated, matches results.tsv header.

        Columns (in order):
            experiment, screening_mode, status, stability_score,
            pooled_pf, pooled_dd, pooled_trades, pooled_traded_days,
            any_gate_failure, per_fold_scores, description
        """
        status = "revert"  # only model_manage.keep() flips to "keep"
        pfs = ",".join(f"{s:.3f}" for s in self.stability.per_fold_scores)
        desc = f"mode={self.screening_mode} folds=[{pfs}]"
        if self.stability.any_fold_gate_failure:
            desc = "GATE_FAILURE " + desc
        cols = [
            self.experiment_id,
            self.screening_mode,
            status,
            f"{self.stability.mean_fold_score:.6f}",
            f"{self.pooled.profit_factor:.4f}",
            f"{self.pooled.max_account_drawdown:.4f}",
            str(self.pooled.total_trades),
            str(self.pooled.traded_days),
            "true" if self.stability.any_fold_gate_failure else "false",
            f"[{pfs}]",
            desc,
        ]
        return "\t".join(cols)


RESULTS_TSV_HEADER = [
    "experiment",
    "screening_mode",
    "status",
    "stability_score",
    "pooled_pf",
    "pooled_dd",
    "pooled_trades",
    "pooled_traded_days",
    "any_gate_failure",
    "per_fold_scores",
    "description",
]


def header_line() -> str:
    return "\t".join(RESULTS_TSV_HEADER)
