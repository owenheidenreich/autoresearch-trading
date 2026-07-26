"""Run constrained offline model search for the Protocol101 live-v1 contract.

This script orchestrates owner-authorized offline training attempts only. It
does not contact brokers, download data, change paper defaults, promote models,
or use June/July recorder days for training. Each attempt is written to its own
artifact directory and registered in an experiment JSONL file.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION,
)
from v4.scripts.run_protocol101_fair_contract_selected_candidate_replay_gate import (
    STRICT_REPLAY_IMPLEMENTATION_VERSION,
)
from v4.scripts.run_protocol101_fair_contract_feature_jitter_gate import (
    IMPLEMENTATION_VERSION as FEATURE_JITTER_GATE_IMPLEMENTATION_VERSION,
)
from v4.model.supervised_pilot import (
    FEATURE_NOISE_AUGMENTATION_NONE,
    FEATURE_NOISE_AUGMENTATION_VENDOR_MICROSTRUCTURE_JITTER_V1,
    FEATURE_TRANSFORM_BUCKET_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
    FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
    FEATURE_TRANSFORM_NONE,
    SELECTION_MODE_STABLE_ABS_OFFSET_10,
    SELECTION_MODE_STABLE_ABS_OFFSET_15,
    SELECTION_MODE_STABLE_ABS_OFFSET_20,
    SELECTION_MODE_TOP_SCORE,
)


DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_model_search"
)
DEFAULT_DESIGN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_labels_design/summary.json"
)
EXPERIMENT_REGISTRY_ENTRY_VERSION = "Protocol101FairContractExperimentRegistryEntryV4"
REUSABLE_REGISTRY_ENTRY_VERSIONS = {
    EXPERIMENT_REGISTRY_ENTRY_VERSION,
    "Protocol101FairContractExperimentRegistryEntryV3",
    "Protocol101FairContractExperimentRegistryEntryV2",
}
MODEL_SEARCH_SUMMARY_VERSION = "Protocol101FairContractModelSearchSummaryV4"


def current_implementation_versions() -> dict[str, str]:
    return {
        "selected_candidate_export": SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION,
        "strict_replay_gate": STRICT_REPLAY_IMPLEMENTATION_VERSION,
        "feature_jitter_gate": FEATURE_JITTER_GATE_IMPLEMENTATION_VERSION,
    }


@dataclass(frozen=True)
class AttemptConfig:
    attempt_id: str
    hypothesis: str
    policy_index: int = 1
    threshold_rule: str = "risk_adjusted_stressed"
    fit_mode: str = "full_train"
    hidden_dim: int = 96
    epochs: int = 8
    max_train_examples: int = 350_000
    batch_size: int = 8192
    seed: int = 42
    threshold_stress_per_trade: float = 20.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    model_family: str = "mlp"
    target_mode: str = "regression"
    target_clip: float = 600.0
    positive_label_threshold: float = 20.0
    relative_target_weight: float = 0.5
    teacher_min_seed_count: int = 1
    ensemble_seeds: str = ""
    entry_filter: str = "none"
    min_score_margin: float = 0.0
    max_score_ceiling: float = 0.0
    max_trades_per_session: int = 0
    max_daily_loss: float = 0.0
    sample_weight_mode: str = "none"
    feature_transform: str = FEATURE_TRANSFORM_NONE
    feature_noise_augmentation: str = FEATURE_NOISE_AUGMENTATION_NONE
    selection_mode: str = SELECTION_MODE_TOP_SCORE
    run_feature_jitter_gate: bool = False


def default_attempts() -> list[AttemptConfig]:
    return [
        AttemptConfig(
            attempt_id="attempt_001_policy1_stressed_pnl_h96_s42",
            hypothesis="Use the 25-minute label but choose the threshold on validation PnL after two-sided slippage stress.",
            threshold_rule="max_validation_stressed_pnl",
        ),
        AttemptConfig(
            attempt_id="attempt_002_policy1_risk_adjusted_h96_s42",
            hypothesis="Use the 25-minute label and choose a validation threshold that balances stressed PnL, PF, drawdown, and trade sufficiency.",
            threshold_rule="risk_adjusted_stressed",
        ),
        AttemptConfig(
            attempt_id="attempt_003_policy0_risk_adjusted_h96_s42",
            hypothesis="Test whether shorter 10-minute labels reduce path fragility under the fair live-v1 contract.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
        ),
        AttemptConfig(
            attempt_id="attempt_004_policy2_risk_adjusted_h96_s42",
            hypothesis="Test whether longer 45-minute labels capture enough runners to offset fair-contract feature degradation.",
            policy_index=2,
            threshold_rule="risk_adjusted_stressed",
        ),
        AttemptConfig(
            attempt_id="attempt_005_policy0_stressed_pnl_h96_s11",
            hypothesis="Keep the stronger 10-minute label but choose directly on validation PnL after slippage stress.",
            policy_index=0,
            threshold_rule="max_validation_stressed_pnl",
            hidden_dim=96,
            seed=11,
        ),
        AttemptConfig(
            attempt_id="attempt_006_policy0_risk_adjusted_h64_s7",
            hypothesis="Keep the stronger 10-minute label, reduce capacity, and change seed to test validation overfit.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=64,
            seed=7,
        ),
        AttemptConfig(
            attempt_id="attempt_007_policy0_risk_adjusted_clip300_h96_s42",
            hypothesis="Keep the stronger 10-minute label but reduce target clipping to limit outlier-PnL chasing.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            target_clip=300.0,
        ),
        AttemptConfig(
            attempt_id="attempt_008_policy0_risk_adjusted_clip200_h96_s42",
            hypothesis="Further reduce target clipping on the 10-minute label to test whether robust labels improve diagnostic stability.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            target_clip=200.0,
        ),
        AttemptConfig(
            attempt_id="attempt_009_policy0_profit_classifier_h96_s42",
            hypothesis="Replace noisy dollar-PnL regression with a classifier for positive stressed 10-minute outcomes.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_010_policy0_profit_classifier_h64_s7",
            hypothesis="Use the profit classifier with lower capacity and a different seed to test classifier stability.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            target_mode="profit_classifier",
            hidden_dim=64,
            seed=7,
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_011_policy0_frequency_sufficient_h96_s42",
            hypothesis="Keep the best 10-minute regression family but require validation trade sufficiency during threshold selection.",
            policy_index=0,
            threshold_rule="frequency_sufficient_stressed",
        ),
        AttemptConfig(
            attempt_id="attempt_012_policy0_classifier_frequency_sufficient_h96_s42",
            hypothesis="Use the profit classifier but require validation trade sufficiency during threshold selection.",
            policy_index=0,
            threshold_rule="frequency_sufficient_stressed",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_013_policy0_daily_stability_h96_s42",
            hypothesis="Keep the strongest 10-minute regression family but choose threshold by validation day-stability under slippage stress.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
        ),
        AttemptConfig(
            attempt_id="attempt_014_policy0_classifier_daily_stability_h96_s42",
            hypothesis="Use the profit classifier and choose threshold by validation day-stability under slippage stress.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_015_policy0_ensemble3_risk_adjusted_h96",
            hypothesis="Average three 10-minute regression models to reduce seed variance before risk-adjusted stressed threshold selection.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=17,
            ensemble_seeds="17,23,42",
        ),
        AttemptConfig(
            attempt_id="attempt_016_policy0_ensemble3_daily_stability_h96",
            hypothesis="Average three 10-minute regression models and choose threshold by validation day-stability to test diagnostic robustness.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=17,
            ensemble_seeds="17,23,42",
        ),
        AttemptConfig(
            attempt_id="attempt_017_policy0_relative_risk_adjusted_h96_s42",
            hypothesis="Train the 10-minute model on decision-relative candidate advantage to improve within-minute ranking under the fair contract.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_relative_regression",
        ),
        AttemptConfig(
            attempt_id="attempt_018_policy0_relative_daily_stability_h96_s42",
            hypothesis="Use decision-relative candidate advantage with day-stability thresholding to test whether ranking quality generalizes across March.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_relative_regression",
        ),
        AttemptConfig(
            attempt_id="attempt_019_policy0_blend35_risk_adjusted_h96_s42",
            hypothesis="Blend absolute 10-minute PnL with 35% decision-relative advantage to balance validation profitability and diagnostic ranking.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
        ),
        AttemptConfig(
            attempt_id="attempt_020_policy0_blend50_risk_adjusted_h96_s42",
            hypothesis="Blend absolute 10-minute PnL with 50% decision-relative advantage to test whether stronger ranking signal improves diagnostic stability.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="blended_relative_regression",
            relative_target_weight=0.50,
        ),
        AttemptConfig(
            attempt_id="attempt_021_policy0_blend10_risk_adjusted_h96_s42",
            hypothesis="Apply a light 10% decision-relative ranking nudge to preserve absolute-PnL validation behavior while testing diagnostic improvement.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="blended_relative_regression",
            relative_target_weight=0.10,
        ),
        AttemptConfig(
            attempt_id="attempt_022_policy0_blend20_risk_adjusted_h96_s42",
            hypothesis="Apply a moderate 20% decision-relative ranking nudge to test the transition between absolute-PnL and relative-ranking behavior.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="blended_relative_regression",
            relative_target_weight=0.20,
        ),
        AttemptConfig(
            attempt_id="attempt_023_policy0_vwap_aligned_risk_adjusted_h96_s42",
            hypothesis="Test a live-causal two-stage filter that only lets the 10-minute model enter calls above VWAP or puts below VWAP before risk-adjusted thresholding.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            entry_filter="vwap_aligned",
        ),
        AttemptConfig(
            attempt_id="attempt_024_policy0_vwap_aligned_daily_stability_h96_s42",
            hypothesis="Use the same live-causal VWAP-aligned first-stage filter but choose threshold by day-stability to test diagnostic robustness.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            entry_filter="vwap_aligned",
        ),
        AttemptConfig(
            attempt_id="attempt_025_policy0_premium_floor3_risk_adjusted_h96_s42",
            hypothesis="Test a live-causal contract-quality filter requiring ask premium at least $3 before 10-minute risk-adjusted thresholding.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            entry_filter="premium_floor_3",
        ),
        AttemptConfig(
            attempt_id="attempt_026_policy0_premium_floor3_daily_stability_h96_s42",
            hypothesis="Use the same ask-premium floor but choose threshold by validation day-stability to test whether low-premium noise was driving diagnostic failure.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            entry_filter="premium_floor_3",
        ),
        AttemptConfig(
            attempt_id="attempt_027_policy0_margin5_risk_adjusted_h96_s42",
            hypothesis="Test calibrated abstention: only enter when the top candidate score beats the runner-up by at least 5 points before risk-adjusted thresholding.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            min_score_margin=5.0,
        ),
        AttemptConfig(
            attempt_id="attempt_028_policy0_margin10_risk_adjusted_h96_s42",
            hypothesis="Test stricter calibrated abstention with a 10-point top-vs-runner-up score margin to reduce score-distribution drift.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            min_score_margin=10.0,
        ),
        AttemptConfig(
            attempt_id="attempt_029_policy0_top_profit_classifier_h96_s42",
            hypothesis="Train a within-decision classifier that marks only the best profitable candidate as positive to directly improve candidate ranking.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_030_policy0_top_profit_classifier_margin10_h96_s42",
            hypothesis="Combine the within-decision top-profitable-candidate classifier with probability-margin abstention to reduce low-confidence ranking drift.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            min_score_margin=0.10,
        ),
        AttemptConfig(
            attempt_id="attempt_031_policy0_top_profit_classifier_frequency_h96_s42",
            hypothesis="Use the within-decision top-profitable-candidate classifier but require validation trade-frequency sufficiency during threshold selection to test diagnostic coverage.",
            policy_index=0,
            threshold_rule="frequency_sufficient_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_032_policy0_top_profit_classifier_daily_stability_h96_s42",
            hypothesis="Use the within-decision top-profitable-candidate classifier with day-stability thresholding to test whether validation concentration is driving diagnostic collapse.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_033_policy0_top_profit_classifier_positive_context_h96_s42",
            hypothesis="Gate the top-profitable-candidate classifier to post-open above-VWAP, positive-OMAR regimes identified by validation/diagnostic failure attribution.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="above_vwap_omar_pos_after_open",
        ),
        AttemptConfig(
            attempt_id="attempt_034_policy0_top_profit_classifier_positive_context_margin10_h96_s42",
            hypothesis="Combine favorable-context gating with top-profitable-candidate classification and probability-margin abstention to reduce regime and score drift.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="above_vwap_omar_pos_after_open",
            min_score_margin=0.10,
        ),
        AttemptConfig(
            attempt_id="attempt_035_policy0_top_profit_regression_h96_s42",
            hypothesis="Train a dollar-scale top-profit regression target: only the best profitable candidate in each minute receives positive value, others are zero.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_top_profit_regression",
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_036_policy0_top_profit_regression_daily_stability_h96_s42",
            hypothesis="Use the top-profit regression target with day-stability thresholding to test whether dollar-scale ranking improves generalization without probability calibration collapse.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_top_profit_regression",
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_037_policy0_listwise_top_profit_risk_adjusted_h96_s42",
            hypothesis="Train a decision-level listwise ranker so each minute competes candidates directly instead of learning isolated candidate labels.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_top_profit_listwise",
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_038_policy0_listwise_top_profit_daily_stability_h96_s42",
            hypothesis="Use the decision-level listwise ranker with validation day-stability thresholding to test whether ranking improves diagnostic participation without hand-picked context gates.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            target_mode="decision_top_profit_listwise",
            positive_label_threshold=20.0,
        ),
        AttemptConfig(
            attempt_id="attempt_039_policy0_hgb_regression_risk_adjusted_s42",
            hypothesis="Use a non-neural histogram-gradient-boosted tabular regressor to capture live-v1 threshold interactions that the MLP score surface missed.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_040_policy0_hgb_top_classifier_risk_adjusted_s42",
            hypothesis="Use a non-neural histogram-gradient-boosted classifier on the within-decision top-profitable-candidate target to test tabular ranking under fair live-v1 features.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_041_policy0_hgb_top_classifier_daily_stability_s42",
            hypothesis="Use the boosted top-candidate classifier with day-stability thresholding to test whether a tabular family improves month-boundary robustness without diagnostic tuning.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_042_policy0_hgb_top_classifier_near_offset_risk_adjusted_s42",
            hypothesis="Test a live-causal candidate-geometry filter that limits the boosted top-candidate classifier to near 10-20 point offsets, based on validation bucket attribution showing mid-offset over-selection.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_043_policy0_hgb_top_classifier_near_offset_daily_stability_s42",
            hypothesis="Use the boosted top-candidate classifier with the near 10-20 point offset filter and validation day-stability thresholding to test whether geometry control reduces drawdown without removing participation.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_044_policy0_hgb_by_right_top_classifier_risk_adjusted_s42",
            hypothesis="Train separate live-v1 boosted top-candidate classifiers for calls and puts to test whether side mixing is causing validation-to-diagnostic ranking drift.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting_by_right",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_045_policy0_hgb_by_right_top_classifier_daily_stability_s42",
            hypothesis="Use separate call/put boosted top-candidate classifiers with validation day-stability thresholding to test side-specialized robustness under the fair contract.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting_by_right",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_046_policy0_hgb_top_classifier_drawdown_guarded_s42",
            hypothesis="Use the boosted top-candidate classifier with validation-only drawdown-guarded thresholding to reduce overtrading and strict-replay drawdown.",
            policy_index=0,
            threshold_rule="drawdown_guarded_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_047_policy0_hgb_near_offset_drawdown_guarded_s42",
            hypothesis="Use the boosted top-candidate classifier with near-offset geometry and validation-only drawdown-guarded thresholding to test whether risk control can preserve the strongest fair-contract validation pocket.",
            policy_index=0,
            threshold_rule="drawdown_guarded_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_048_policy0_hgb_by_right_drawdown_guarded_s42",
            hypothesis="Use separate call/put boosted top-candidate classifiers with validation-only drawdown-guarded thresholding to test whether side specialization needs stricter risk calibration.",
            policy_index=0,
            threshold_rule="drawdown_guarded_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting_by_right",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_049_policy0_hgb_top_classifier_janfit_febcal_daily_stability_s42",
            hypothesis="Fit the boosted top-candidate classifier on January only and calibrate threshold on February to test whether internal walk-forward selection reduces March validation/diagnostic overfit.",
            policy_index=0,
            fit_mode="jan_fit_feb_calibration",
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_050_policy0_hgb_near_offset_janfit_febcal_daily_stability_s42",
            hypothesis="Fit the near-offset boosted top-candidate classifier on January and calibrate on February to test whether the strongest geometry pocket survives a cleaner walk-forward protocol.",
            policy_index=0,
            fit_mode="jan_fit_feb_calibration",
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_051_policy0_hgb_top_classifier_morning_daily_stability_s42",
            hypothesis="Use a live-causal morning-only time specialist with the boosted top-candidate classifier to test whether fair-contract edge is concentrated in the first post-open window.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="morning_1000_1129",
            learning_rate=0.05,
            weight_decay=0.01,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_052_policy0_hgb_near_offset_morning_daily_stability_s42",
            hypothesis="Combine the live-causal morning window with near-offset geometry to test a narrower time/contract specialist under the fair contract.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="morning_near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_053_policy0_hgb_near_offset_morning_drawdown_guarded_s42",
            hypothesis="Use the morning plus near-offset specialist with validation-only drawdown-guarded thresholding to test whether the first positive diagnostic shape can pass validation risk gates.",
            policy_index=0,
            threshold_rule="drawdown_guarded_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="morning_near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_054_policy0_hgb_near_offset_cap2_drawdown_guarded_s42",
            hypothesis="Retest the broader near-offset boosted top-candidate classifier with a live-reproducible two-trade session cap to address clustered repeat-loss mornings without lowering gates.",
            policy_index=0,
            threshold_rule="drawdown_guarded_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_055_policy0_hgb_near_offset_cap2_daily_loss500_s42",
            hypothesis="Add a realized $500 daily loss stop to the two-trade near-offset boosted classifier to test whether loss-cluster containment can preserve diagnostic participation.",
            policy_index=0,
            threshold_rule="drawdown_guarded_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            max_daily_loss=500.0,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_056_policy0_hgb_profit_classifier_near_offset_cap2_s42",
            hypothesis="Switch from top-candidate classification to any-positive-outcome classification under the near-offset geometry, with a two-trade cap, to test whether a less winner-take-all label improves diagnostic stability.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_057_policy0_hgb_profit_classifier_morning_near_cap2_s42",
            hypothesis="Apply the any-positive-outcome boosted classifier to the narrower morning near-offset pocket with a two-trade session cap to test whether the positive diagnostic pocket can gain enough validation breadth.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="morning_near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_058_policy0_hgb_profit_classifier_near_offset_cap2_drawdown_guarded_s42",
            hypothesis="Keep the closest broad profit-classifier branch but choose its threshold with validation drawdown pressure to test whether diagnostic PF improves without losing trade sufficiency.",
            policy_index=0,
            threshold_rule="drawdown_guarded_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_059_policy0_hgb_profit_classifier_by_right_near_offset_cap2_s42",
            hypothesis="Use side-specialized boosted profit classifiers under the near-offset cap-two branch because attempt 056 had opposite call/put behavior in validation versus diagnostic.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting_by_right",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_060_policy0_hgb_profit_classifier_near_offset_cap2_margin05_s42",
            hypothesis="Raise the top-vs-runner-up score-margin requirement on the closest profit-classifier branch to test calibrated abstention against low-PF diagnostic trades.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.05,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_061_policy0_hgb_profit100_near_offset_cap2_s42",
            hypothesis="Require larger positive training outcomes for the broad near-offset profit classifier to test whether a higher-quality label improves diagnostic profit factor under the fair contract.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="profit_classifier",
            positive_label_threshold=100.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_062_policy0_hgb_teacher_near_offset_cap2_s42",
            hypothesis="Use Protocol101 teacher enter contracts as supervised labels under the fair live-v1 features, while keeping the same near-offset cap-two risk game, to test whether the old behavior is learnable from causal inputs.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_063_policy0_hgb_profitable_teacher_near_offset_cap2_s42",
            hypothesis="Use only Protocol101 teacher enter contracts that are also profitable under the fair 10-minute label, testing whether teacher guidance can improve causal ranking without copying losing legacy actions.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_profitable_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_064_policy0_hgb_profitable_teacher_near_offset_cap2_frequency_s42",
            hypothesis="Keep the profitable-teacher target but choose a validation frequency-sufficient threshold to test whether the high-quality sparse branch can reach the required trade count without losing diagnostic PF.",
            policy_index=0,
            threshold_rule="frequency_sufficient_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_profitable_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_065_policy0_hgb_profitable_teacher_cap2_frequency_s42",
            hypothesis="Remove the near-offset hand filter from the profitable-teacher target and use the same two-trade cap plus frequency-sufficient thresholding to test whether the filter was suppressing causal teacher-like opportunities.",
            policy_index=0,
            threshold_rule="frequency_sufficient_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_profitable_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="none",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_066_policy0_hgb_teacher_seed2_near_offset_cap2_s42",
            hypothesis="Retest the closest teacher branch with a two-seed teacher-consensus target to reduce noisy one-seed legacy actions while keeping the fair live-v1 runtime feature set unchanged.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=2,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_067_policy0_hgb_teacher_seed3_near_offset_cap2_s42",
            hypothesis="Retest the closest teacher branch with a three-seed teacher-consensus target to see whether stronger Protocol101 agreement improves diagnostic PF without sacrificing the 20-trade sufficiency gate.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=3,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_068_policy0_hgb_teacher_near_offset_cap2_drawdown_guarded_s42",
            hypothesis="Keep the closest teacher-label branch but choose its threshold with validation-only drawdown guarding to test whether bad diagnostic PF is caused by low-quality threshold selection rather than target design.",
            policy_index=0,
            threshold_rule="drawdown_guarded_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_069_policy0_hgb_teacher_near_offset_cap2_risk_adjusted_s42",
            hypothesis="Keep the closest teacher-label branch but choose its threshold with the validation-only risk-adjusted stressed rule to test whether PF improves without diagnostic-aware tuning.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_070_policy0_hgb_profitable_teacher_near_offset_cap2_nomargin_s42",
            hypothesis="Relax only the top-vs-runner-up score-margin gate on the profitable-teacher branch to test whether its high-PF but sparse behavior can reach trade sufficiency without changing fair live-v1 features or using diagnostic-aware tuning.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_profitable_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_071_policy0_hgb_profitable_teacher_near_offset_cap2_nomargin_risk_s42",
            hypothesis="Use the same no-margin profitable-teacher branch with validation-only risk-adjusted threshold selection to test whether the trade-sufficiency repair survives a different predeclared threshold rule.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_profitable_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_072_policy0_hgb_by_right_teacher_near_offset_cap2_s42",
            hypothesis="Train separate call/put boosted teacher classifiers under the same near-offset cap-two fair-contract game to test whether side-specific causal structure improves diagnostic PF without a hand-coded side block.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting_by_right",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_073_policy0_hgb_by_right_profitable_teacher_near_offset_cap2_s42",
            hypothesis="Train separate call/put boosted profitable-teacher classifiers under the same fair-contract near-offset cap-two game, testing whether side specialization can keep the cleaner label's PF while restoring trade sufficiency.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting_by_right",
            target_mode="protocol101_teacher_profitable_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_074_policy0_hgb_teacher_breakeven_near_offset_cap2_s42",
            hypothesis="Relax the profitable-teacher label from strictly +$20 to break-even teacher actions, testing whether it keeps enough Protocol101-like breadth while removing the worst teacher losers under the fair contract.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_profitable_classifier",
            positive_label_threshold=0.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_075_policy0_hgb_teacher_loss_tolerant_near_offset_cap2_s42",
            hypothesis="Relax the profitable-teacher label to allow only small fair-label losses above -$100, testing whether a loss-tolerant teacher target restores trade sufficiency without copying severe legacy losers.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_profitable_classifier",
            positive_label_threshold=-100.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.02,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_076_policy0_hgb_teacher_edge20_near_offset_cap2_s42",
            hypothesis="Train a teacher-edge regressor that copies Protocol101 actions only to the extent their fair-contract label clears a $20 stress cushion, testing whether the near-miss teacher branch can keep trade sufficiency while rejecting low-cushion legacy actions.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_edge_regression",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=20.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_077_policy0_hgb_teacher_edge60_near_offset_cap2_s42",
            hypothesis="Require a larger $60 fair-label cushion in the teacher-edge regressor to test whether additional adverse-fill buffer improves diagnostic PF without collapsing trade count.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_edge_regression",
            positive_label_threshold=60.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=20.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_078_policy0_hgb_teacher_near_offset_cap2_nomargin_s42",
            hypothesis="Remove the top-vs-runner-up score-margin gate from the closest teacher branch to test whether expanded fair-contract training can restore validation/diagnostic trade sufficiency without using diagnostic-aware tuning.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_079_policy0_hgb_teacher_near_offset_cap2_nomargin_frequency_s42",
            hypothesis="Use the no-margin teacher branch with validation-only frequency-sufficient thresholding, testing whether the expanded data can choose enough trades while preserving fair-contract edge.",
            policy_index=0,
            threshold_rule="frequency_sufficient_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_080_policy0_hgb_teacher_near_offset_cap2_nomargin_daily_loss500_s42",
            hypothesis="Add a live-reproducible $500 realized daily loss stop to the no-margin teacher branch because expanded-data counterfactuals improved PF/drawdown but lost trade count when the rule was applied after training.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            max_daily_loss=500.0,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_081_policy0_hgb_teacher_near_offset_cap2_nomargin_risk_s42",
            hypothesis="Keep the strongest no-margin teacher branch but switch from daily-stability thresholding to validation-only risk-adjusted stressed thresholding to test whether the near-pass diagnostic PF miss was threshold-rule-specific.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_082_policy0_hgb_teacher_near_offset_cap2_nomargin_drawdown_s42",
            hypothesis="Keep the strongest no-margin teacher branch but choose its threshold with validation-only drawdown guarding to test whether the near-pass branch can preserve PF while reducing tail-loss drag.",
            policy_index=0,
            threshold_rule="drawdown_guarded_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_083_policy0_hgb_teacher_balanced_near_offset_cap2_nomargin_s42",
            hypothesis="Repair the strongest teacher branch's extreme rare-positive class imbalance with explicit balanced classifier sample weights, preserving the same live-v1 features, near-offset filter, cap-two risk control, and validation-only daily-stability thresholding.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            sample_weight_mode="balanced_classifier",
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_084_policy0_hgb_teacher_balanced_near_offset_cap2_risk_s42",
            hypothesis="Use the balanced teacher classifier with validation-only risk-adjusted thresholding to test whether the class-imbalance repair generalizes across threshold rules without diagnostic-aware tuning.",
            policy_index=0,
            threshold_rule="risk_adjusted_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            sample_weight_mode="balanced_classifier",
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_085_policy0_hgb_profitable_teacher_balanced_near_offset_cap2_s42",
            hypothesis="Apply the same class-imbalance repair to the profitable-teacher label to test whether a cleaner teacher target can reach trade sufficiency without copying low-quality legacy actions.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_profitable_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            sample_weight_mode="balanced_classifier",
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_086_policy0_hgb_profit_classifier_tail20_near_offset_cap2_s42",
            hypothesis="Move threshold selection off March by fitting on earlier train sessions and calibrating on the last 20 train sessions for the near-offset profit classifier.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            fit_mode="train_tail20_calibration",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_087_policy0_hgb_teacher_tail20_near_offset_cap2_nomargin_s42",
            hypothesis="Move threshold selection off March for the most stable teacher branch to test whether train-tail calibration improves forward validation/diagnostic balance.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            fit_mode="train_tail20_calibration",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_088_policy0_hgb_top_classifier_tail20_near_offset_cap2_s42",
            hypothesis="Use train-tail calibration for the decision-top-profit classifier to test whether candidate ranking can generalize without tuning threshold on March.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            fit_mode="train_tail20_calibration",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_089_policy0_hgb_profit_classifier_decision_balanced_near_offset_cap2_s42",
            hypothesis="Use decision-balanced classifier weighting so dense candidate minutes cannot dominate rare positive profit labels under the fair live contract.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            sample_weight_mode="decision_balanced_classifier",
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_090_policy0_hgb_teacher_decision_balanced_near_offset_cap2_nomargin_s42",
            hypothesis="Use decision-balanced classifier weighting on the stable teacher branch to test whether equal-minute weighting improves forward PF without changing live features.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            sample_weight_mode="decision_balanced_classifier",
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_091_policy0_hgb_top_classifier_decision_balanced_near_offset_cap2_s42",
            hypothesis="Use decision-balanced classifier weighting for the decision-top-profit target to improve within-minute ranking stability across adjacent forward windows.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            sample_weight_mode="decision_balanced_classifier",
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_092_policy0_hgb_teacher_put_near_cap2_s42",
            hypothesis="Apply the diagnostic-supported causal put-near geometry gate to the most stable teacher branch to test whether call-side instability caused the fair-contract degradation.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="put_near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_093_policy0_hgb_profit_classifier_put_near_cap2_s42",
            hypothesis="Apply the same causal put-near geometry gate to direct profit classification to test whether the stable bucket is a model-agnostic data slice or only a teacher artifact.",
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=2,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_094_policy0_hgb_teacher_put_near_after0940_cap3_s42",
            hypothesis=(
                "Test whether the put-near teacher branch fails because it enters before enough "
                "causal intraday context has formed; require 09:40 ET or later and allow up to "
                "three serial trades per session to preserve sample size."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="put_near_after_0940",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=3,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_095_policy0_hgb_profit_classifier_put_near_after0940_cap3_s42",
            hypothesis=(
                "Apply the same 09:40 ET causal warmup plus put-near gate to direct profit "
                "classification to test whether the early-context failure is model-agnostic."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=3,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_096_policy0_hgb_profit_classifier_put_near_after0940_vwap_m2_10_cap3_s42",
            hypothesis=(
                "Use the candidate-edge stability audit result: only put-near candidates after "
                "09:40 ET with SPX from 2 points below VWAP to 10 points above VWAP, because "
                "those VWAP-gap buckets were positive in both validation and diagnostic labels."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=3,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_097_policy0_hgb_teacher_put_near_after0940_vwap_m2_10_cap3_s42",
            hypothesis=(
                "Apply the same stable VWAP-gap put-near gate to the Protocol101 teacher branch "
                "to test whether teacher imitation or direct profit classification better extracts "
                "the causal pocket."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="protocol101_teacher_classifier",
            positive_label_threshold=20.0,
            teacher_min_seed_count=1,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=3,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_098_policy0_hgb_top_classifier_put_near_after0940_vwap_m2_10_cap3_s42",
            hypothesis=(
                "Train a decision-top profit classifier inside the stable VWAP-gap put-near pocket, "
                "because direct profit classification found opportunity but selected poor contracts "
                "in diagnostic strict replay."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=3,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_099_policy0_hgb_top_classifier_decision_balanced_put_near_after0940_vwap_m2_10_cap3_s42",
            hypothesis=(
                "Use decision-balanced weighting with the decision-top classifier in the stable "
                "VWAP-gap put-near pocket to prevent dense candidate minutes from dominating "
                "within-decision ranking."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_top_profit_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=3,
            sample_weight_mode="decision_balanced_classifier",
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_100_policy0_hgb_relative_regression_put_near_after0940_vwap_m2_10_cap3_s42",
            hypothesis=(
                "Use decision-relative regression inside the stable VWAP-gap put-near pocket, "
                "because label opportunity exists but classifiers are not ranking candidates "
                "well enough under strict replay."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_relative_regression",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=3,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_101_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_s42",
            hypothesis=(
                "Blend absolute 10-minute PnL with 35% decision-relative advantage inside the "
                "stable VWAP-gap put-near pocket to balance abstention quality and within-minute ranking."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=3,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_102_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil50_s42",
            hypothesis=(
                "Apply a live-causal score-ceiling abstention guard to the near-pass blend35 "
                "relative branch: diagnostics showed extreme positive scores were unstable losers "
                "while sub-50 selected scores preserved validation behavior."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=50.0,
            max_trades_per_session=3,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_103_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil20_s42",
            hypothesis=(
                "Tighten the live-causal score-ceiling abstention guard after attempt 102 showed "
                "sub-50 replacements still carried diagnostic instability; this tests whether "
                "removing all score-20-plus overconfidence restores strict-replay PF without "
                "collapsing validation trade sufficiency."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=20.0,
            max_trades_per_session=3,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_104_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil20_dailyloss500_s42",
            hypothesis=(
                "Layer a live-causal $500 realized daily loss stop onto the near-pass score-ceiling-20 "
                "branch, because its remaining diagnostic PF miss is concentrated in repeated same-day "
                "losses rather than broad feature-contract failure."
            ),
            policy_index=0,
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=20.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_105_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil20_dailyloss500_traintail_s42",
            hypothesis=(
                "Keep the score-ceiling and daily-loss risk controls from the 64-session pass, "
                "but move threshold calibration off March validation and onto the last 20 train "
                "sessions to test whether the 128-session failure is validation-threshold overfit."
            ),
            policy_index=0,
            fit_mode="train_tail20_calibration",
            threshold_rule="daily_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=20.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_106_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil20_dailyloss500_traintail_freq_s42",
            hypothesis=(
                "Use train-tail calibration with the frequency-sufficient threshold rule, because "
                "attempt 105's validation-independent threshold produced high-quality but too-sparse "
                "forward trades."
            ),
            policy_index=0,
            fit_mode="train_tail20_calibration",
            threshold_rule="frequency_sufficient_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=20.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_107_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil50_dailyloss500_plateau_s42",
            hypothesis=(
                "Use a conservative validation-only PnL-plateau threshold with the score-ceiling-50 "
                "guard and daily loss stop, testing whether threshold regularization can avoid the "
                "128-session validation-optimum over-entry failure without using diagnostic labels."
            ),
            policy_index=0,
            threshold_rule="conservative_pnl_plateau_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=50.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_108_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_micromask_dailyloss500_plateau_s42",
            hypothesis=(
                "Remove vendor-sensitive option microstructure fields from model scoring after paired "
                "IBKR-vs-historical replay showed attempt107 action/ranking drift was driven by tiny IV, "
                "spread, and size differences. Keep the same fair labels, put-near VWAP pocket, cap, "
                "daily loss stop, and conservative plateau threshold, but do not use a score ceiling."
            ),
            policy_index=0,
            threshold_rule="conservative_pnl_plateau_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            sample_weight_mode="none",
            feature_transform=FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_109_policy0_mlp_ensemble3_blend35_relative_put_near_after0940_vwap_m2_10_cap3_micromask_dailyloss500_plateau",
            hypothesis=(
                "Use a smoother three-seed MLP ensemble on the same microstructure-masked fair-contract "
                "features to test whether avoiding HGB split brittleness preserves enough edge without "
                "using score ceilings or confirmation-day tuning."
            ),
            policy_index=0,
            threshold_rule="conservative_pnl_plateau_stressed",
            hidden_dim=96,
            seed=17,
            ensemble_seeds="17,23,42",
            model_family="mlp",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=1e-3,
            weight_decay=1e-4,
            min_score_margin=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            sample_weight_mode="none",
            feature_transform=FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
            epochs=8,
        ),
        AttemptConfig(
            attempt_id="attempt_110_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_microbucket_dailyloss500_plateau_s42",
            hypothesis=(
                "Bucket vendor-sensitive option microstructure instead of masking it: retain coarse IV, "
                "spread, and size signal while removing tiny cross-vendor deltas that caused attempt107 "
                "score-ceiling and ranking drift."
            ),
            policy_index=0,
            threshold_rule="conservative_pnl_plateau_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            sample_weight_mode="none",
            feature_transform=FEATURE_TRANSFORM_BUCKET_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
            epochs=6,
        ),
        AttemptConfig(
            attempt_id="attempt_111_policy0_mlp_ensemble3_blend35_relative_put_near_after0940_vwap_m2_10_cap3_microbucket_dailyloss500_plateau",
            hypothesis=(
                "Use a smoother MLP ensemble with bucketed vendor-sensitive option microstructure, "
                "testing whether coarse option-quality signal plus smoother scoring can preserve "
                "diagnostic edge without score-ceiling brittleness."
            ),
            policy_index=0,
            threshold_rule="conservative_pnl_plateau_stressed",
            hidden_dim=96,
            seed=17,
            ensemble_seeds="17,23,42",
            model_family="mlp",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=1e-3,
            weight_decay=1e-4,
            min_score_margin=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            sample_weight_mode="none",
            feature_transform=FEATURE_TRANSFORM_BUCKET_VENDOR_SENSITIVE_OPTION_MICROSTRUCTURE,
            epochs=8,
        ),
        AttemptConfig(
            attempt_id="attempt_112_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_margin5_cap3_dailyloss500_plateau_s42_jittergate",
            hypothesis=(
                "After the feature-jitter gate showed attempt107 was spread/size ranking-fragile, "
                "keep the raw fair-contract option information but remove the brittle score ceiling "
                "and require a minimum 5-point top-vs-runner-up score margin. Run the option-feature "
                "jitter gate as part of model selection."
            ),
            policy_index=0,
            threshold_rule="conservative_pnl_plateau_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=5.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_113_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_margin10_cap3_dailyloss500_plateau_s42_jittergate",
            hypothesis=(
                "Test a stricter 10-point top-vs-runner-up score margin to see whether contract "
                "selection can become robust to spread/size perturbations without deleting the option "
                "microstructure signal or tuning on confirmation days."
            ),
            policy_index=0,
            threshold_rule="conservative_pnl_plateau_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=10.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_114_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_noiseaug_cap3_dailyloss500_plateau_s42_jittergate",
            hypothesis=(
                "Train HGB on deterministic fit-only vendor microstructure jitter so it learns a smoother "
                "mapping over IV, spread, and quote-size differences, while calibration/validation/diagnostic "
                "and June/July confirmation rows remain unaugmented. Require the feature-jitter gate."
            ),
            policy_index=0,
            threshold_rule="conservative_pnl_plateau_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            feature_noise_augmentation=FEATURE_NOISE_AUGMENTATION_VENDOR_MICROSTRUCTURE_JITTER_V1,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_115_policy0_mlp_ensemble3_blend35_relative_put_near_after0940_vwap_m2_10_noiseaug_cap3_dailyloss500_plateau_jittergate",
            hypothesis=(
                "Use a smoother three-seed MLP ensemble trained with fit-only vendor microstructure jitter, "
                "testing whether continuous scoring plus noise augmentation keeps edge and improves spread/size "
                "action stability without confirmation-day tuning."
            ),
            policy_index=0,
            threshold_rule="conservative_pnl_plateau_stressed",
            hidden_dim=96,
            seed=17,
            ensemble_seeds="17,23,42",
            model_family="mlp",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=1e-3,
            weight_decay=1e-4,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=8,
            feature_noise_augmentation=FEATURE_NOISE_AUGMENTATION_VENDOR_MICROSTRUCTURE_JITTER_V1,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_116_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_jitterthreshold_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Select the threshold by development-period jitter-stressed validation behavior instead "
                "of raw validation replay, using raw fair-contract features and requiring the feature-jitter "
                "gate before any confirmation replay."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_117_policy0_mlp_ensemble3_blend35_relative_put_near_after0940_vwap_m2_10_jitterthreshold_cap3_dailyloss500_jittergate",
            hypothesis=(
                "Use the smoother MLP ensemble with raw fair-contract features, but choose its threshold "
                "by jitter-stressed validation behavior to test whether threshold selection, not model "
                "features alone, can recover robust entries."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=17,
            ensemble_seeds="17,23,42",
            model_family="mlp",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            learning_rate=1e-3,
            weight_decay=1e-4,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=8,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_118_policy0_hgb_blend35_relative_put_near_10_20_jitterthreshold_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Broaden beyond the fragile after-09:40/VWAP put pocket by allowing all near-offset puts, "
                "while keeping jitter-stressed threshold selection and the feature-jitter gate."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_119_policy0_hgb_byright_blend35_relative_near_10_20_jitterthreshold_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Broaden side coverage to near-offset calls and puts, but use right-specialized HGB models "
                "plus jitter-stressed threshold selection so side-specific structure can compete under the fair contract."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting_by_right",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="near_10_20_offset",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_120_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_omar_neg_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Use the two-stage selector diagnostic's strongest broad causal context: put-near "
                "09:40+ VWAP -2/+10 candidates only when OMAR is negative. This tests whether "
                "removing positive-OMAR split noise improves strict replay and vendor-jitter stability."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_omar_neg",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_121_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_range20_45_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Use the two-stage selector diagnostic's sample-sufficient moderate-range pocket: "
                "put-near 09:40+ VWAP -2/+10 candidates only when the live session range is 20-45 points."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_range_20_45",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_122_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_nearvwap_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Use the two-stage selector diagnostic's strongest trade-sufficient context gate: "
                "put-near 09:40+ candidates only when SPX is within two points of VWAP, where "
                "oracle opportunity was positive in both validation and diagnostic splits."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_near_vwap",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_123_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_premiumgte7_5_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Use the two-stage selector diagnostic's premium stability result by excluding "
                "sub-7.5 ask candidates while retaining both mid- and high-premium stable buckets. "
                "This tests whether low-premium convex noise drove learned selection fragility."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_premium_gte_7_5",
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_124_policy0_hgb_blend35_relative_nearvwap_stableabs10_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Test aggregate/stable contract selection after attempt122 nearly passed strict replay "
                "but failed spread/size selected-contract jitter. The model decides enter/wait in the "
                "near-VWAP put pocket, but exact contract choice is deterministic: eligible contract "
                "closest to 10-point absolute offset."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_near_vwap",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_10,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_125_policy0_hgb_blend35_relative_nearvwap_stableabs15_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Test aggregate/stable contract selection with the same near-VWAP entry model, but "
                "choose the eligible contract closest to 15-point absolute offset to reduce "
                "spread/size ranking flips while retaining more convexity than the 10-point selector."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_near_vwap",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_15,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_126_policy0_hgb_blend35_relative_nearvwap_stableabs20_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Test aggregate/stable contract selection with the same near-VWAP entry model, but "
                "choose the eligible contract closest to 20-point absolute offset. This checks whether "
                "the profitable diagnostic pocket survives with a deterministic farther-offset contract."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="blended_relative_regression",
            relative_target_weight=0.35,
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_near_vwap",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_127_policy0_hgb_decisionbest_nearvwap_stableabs15_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Train an aggregate decision-level regression target: every candidate in a minute "
                "receives the minute's best fair PnL target, so the model learns whether the setup "
                "is tradable instead of ranking tiny contract differences. Contract choice is then "
                "made by the deterministic 15-point stable selector."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_best_profit_regression",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_near_vwap",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_15,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_128_policy0_hgb_decisionbest_nearvwap_stableabs20_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Repeat the aggregate decision-level regression target with a deterministic "
                "20-point stable selector to test whether a farther put retains enough convexity "
                "without relying on vendor-fragile per-contract score ranking."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_best_profit_regression",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_near_vwap",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_129_policy0_hgb_decisionpresence_nearvwap_stableabs15_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Train an aggregate decision-level classifier where every candidate in a profitable "
                "minute is positive, weighted by decision to avoid wide ladders dominating. This "
                "tests an enter/wait objective separated from exact contract ranking."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_profit_presence_classifier",
            sample_weight_mode="decision_balanced_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_near_vwap",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_15,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_130_policy0_hgb_decisionpresence_nearvwap_stableabs20_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Repeat the aggregate decision-level classifier with the deterministic 20-point "
                "stable selector to test whether action stability improves without sacrificing "
                "too much option convexity."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_profit_presence_classifier",
            sample_weight_mode="decision_balanced_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_near_vwap",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_131_policy0_hgb_decisionpresence_basevwap_stableabs20_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Broaden the best aggregate classifier shape by removing the extra near-VWAP "
                "candidate gate while keeping the causal put-near 09:40+ VWAP pocket and "
                "deterministic 20-point selector. This tests whether aggregate/stable selection "
                "can recover diagnostic trade count without reintroducing contract-ranking fragility."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_profit_presence_classifier",
            sample_weight_mode="decision_balanced_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_132_policy0_hgb_decisionpresence_premiumgte7_5_stableabs20_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Broaden the aggregate classifier while using the premium>=7.5 causal gate from "
                "the two-stage diagnostic. This keeps more trade opportunities than near-VWAP "
                "while avoiding the lowest-premium contracts that previously amplified noise."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_profit_presence_classifier",
            sample_weight_mode="decision_balanced_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_premium_gte_7_5",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_133_policy0_hgb_decisionpresence_mom15nonpos_stableabs20_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Diagnose aggregate edge loss by excluding put entries during positive 15-minute "
                "momentum, which was the largest diagnostic loser bucket in aggregate attempts. "
                "The filter is causal at decision time and keeps the aggregate classifier plus "
                "stable 20-point selector that passed option-feature jitter."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_profit_presence_classifier",
            sample_weight_mode="decision_balanced_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_mom15_nonpos",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_134_policy0_hgb_decisionpresence_omarpos_mom15nonpos_stableabs20_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Test the strongest aggregate diagnostic sub-bucket: positive OMAR with non-positive "
                "15-minute momentum. This keeps the setup causal while checking whether the edge loss "
                "is from entering puts during counter-momentum continuation conditions."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_profit_presence_classifier",
            sample_weight_mode="decision_balanced_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_135_policy0_hgb_decisionpresence_premiumgte7_5_mom15nonpos_stableabs20_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Combine the premium>=7.5 causal gate with non-positive 15-minute momentum to test "
                "whether the mid-premium stable bucket remains robust when the largest diagnostic "
                "loser bucket is removed."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_profit_presence_classifier",
            sample_weight_mode="decision_balanced_classifier",
            positive_label_threshold=20.0,
            entry_filter="put_near_after_0940_vwap_m2_10_premium_gte_7_5_mom15_nonpos",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_136_policy0_hgb_decisionpresence_mom15side_stableabs20_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Test whether aggregate edge loss comes from forcing puts when the same timestamp's "
                "fair label often favors calls. Positive 15-minute momentum allows calls; non-positive "
                "momentum allows puts. Contract choice remains deterministic stable_abs_offset_20."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_profit_presence_classifier",
            sample_weight_mode="decision_balanced_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_after_0940_vwap_m2_10_mom15_side",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
        AttemptConfig(
            attempt_id="attempt_137_policy0_hgb_decisionpresence_mom15side_premiumgte7_5_stableabs20_cap3_dailyloss500_s42_jittergate",
            hypothesis=(
                "Repeat the side-aware aggregate objective with a premium>=7.5 guard to avoid the "
                "lowest-premium contracts while still allowing positive-momentum calls and "
                "non-positive-momentum puts."
            ),
            policy_index=0,
            threshold_rule="jitter_stability_stressed",
            hidden_dim=96,
            seed=42,
            model_family="sklearn_hist_gradient_boosting",
            target_mode="decision_profit_presence_classifier",
            sample_weight_mode="decision_balanced_classifier",
            positive_label_threshold=20.0,
            entry_filter="near_after_0940_vwap_m2_10_mom15_side_premium_gte_7_5",
            selection_mode=SELECTION_MODE_STABLE_ABS_OFFSET_20,
            learning_rate=0.05,
            weight_decay=0.01,
            min_score_margin=0.0,
            max_score_ceiling=0.0,
            max_trades_per_session=3,
            max_daily_loss=500.0,
            epochs=6,
            run_feature_jitter_gate=True,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument(
        "--attempt-ids",
        default="",
        help=(
            "Optional comma-separated attempt ids to run from the preregistered "
            "attempt list. When set, --max-attempts is ignored."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--append-only",
        action="store_true",
        help=(
            "Reuse matching existing registry entries and only execute missing "
            "or changed attempts. This avoids re-running prior gated attempts."
        ),
    )
    return parser.parse_args()


def select_attempts(
    attempts: list[AttemptConfig],
    *,
    max_attempts: int,
    attempt_ids: str = "",
) -> list[AttemptConfig]:
    """Select attempts deterministically from the preregistered attempt list."""
    requested = [item.strip() for item in str(attempt_ids or "").split(",") if item.strip()]
    if not requested:
        return attempts[: max(int(max_attempts), 0)]
    by_id = {attempt.attempt_id: attempt for attempt in attempts}
    missing = [attempt_id for attempt_id in requested if attempt_id not in by_id]
    if missing:
        raise ValueError(f"unknown attempt id(s): {missing}")
    requested_set = set(requested)
    return [attempt for attempt in attempts if attempt.attempt_id in requested_set]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    """Load existing registry rows by attempt id, keeping the last matching row."""
    if not path.exists():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        attempt_id = str(row.get("attempt_id") or "")
        if attempt_id:
            entries[attempt_id] = row
    return entries


def reusable_registry_entry(
    existing: dict[str, Any] | None,
    attempt: AttemptConfig,
) -> dict[str, Any] | None:
    """Return an existing registry entry only when config, versions, and artifacts match."""
    if not existing:
        return None
    if existing.get("schema_version") not in REUSABLE_REGISTRY_ENTRY_VERSIONS:
        return None
    if dict(existing.get("implementation_versions") or {}) != current_implementation_versions():
        return None
    expected = asdict(attempt)
    actual = dict(existing.get("config") or {})
    for key, value in expected.items():
        actual.setdefault(key, value)
    if actual != expected:
        return None
    artifacts = existing.get("artifacts") or {}
    required = (
        "training_result",
        "model",
        "selected_candidates",
        "strict_replay_trades",
        "candidate_validation_report",
        "strict_replay_report",
    )
    if any(not artifacts.get(key) for key in required):
        return None
    if any(not Path(str(artifacts[key])).exists() for key in required):
        return None
    refreshed = dict(existing)
    refreshed["schema_version"] = EXPERIMENT_REGISTRY_ENTRY_VERSION
    refreshed["implementation_versions"] = current_implementation_versions()
    refreshed["objective_score"] = attempt_score(refreshed.get("strict_replay_gate") or {})
    refreshed["reason_accepted_or_rejected"] = reason_for_result(
        validation_gate=refreshed.get("candidate_validation_gate") or {},
        replay_summary=refreshed.get("strict_replay_gate") or {},
    )
    return refreshed


def run_command(args: list[str], *, cwd: Path) -> None:
    subprocess.run(args, cwd=str(cwd), check=True)


def metric(payload: dict[str, Any], split: str, key: str) -> float:
    try:
        return float(((payload.get("metrics") or {}).get(split) or {}).get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def attempt_score(
    replay_summary: dict[str, Any],
    feature_jitter_summary: dict[str, Any] | None = None,
) -> float:
    """Rank by gate-nearness and diagnostic generalization, not headline PnL.

    This score is only a research triage aid. Passing/failing still comes from
    the explicit strict replay checks. Large penalties keep validation-heavy but
    diagnostic-broken attempts from being reported as the "best" branch.
    """
    validation_pnl = metric(replay_summary, "validation", "total_pnl")
    diagnostic_pnl = metric(replay_summary, "diagnostic_test", "total_pnl")
    validation_pf = metric(replay_summary, "validation", "profit_factor")
    diagnostic_pf = metric(replay_summary, "diagnostic_test", "profit_factor")
    validation_dd = abs(metric(replay_summary, "validation", "max_drawdown"))
    diagnostic_dd = abs(metric(replay_summary, "diagnostic_test", "max_drawdown"))
    validation_trades = metric(replay_summary, "validation", "trades")
    diagnostic_trades = metric(replay_summary, "diagnostic_test", "trades")
    blockers = set(str(item) for item in (replay_summary.get("blockers") or []))
    blocker_weights = {
        "no_missing_required_fields": 100_000.0,
        "no_overlap_skips": 100_000.0,
        "no_unaffordable_skips": 100_000.0,
        "diagnostic_positive_pnl": 55_000.0,
        "diagnostic_trade_count": 50_000.0,
        "diagnostic_profit_factor": 40_000.0,
        "diagnostic_drawdown": 35_000.0,
        "validation_positive_pnl": 30_000.0,
        "validation_trade_count": 25_000.0,
        "validation_profit_factor": 22_500.0,
        "validation_drawdown": 20_000.0,
    }
    blocker_penalty = sum(blocker_weights.get(name, 20_000.0) for name in blockers)
    validation_pf_shortfall = max(0.0, 1.25 - validation_pf)
    diagnostic_pf_shortfall = max(0.0, 1.25 - diagnostic_pf)
    validation_trade_shortfall = max(0.0, 20.0 - validation_trades)
    diagnostic_trade_shortfall = max(0.0, 20.0 - diagnostic_trades)
    pass_bonus = 1_000_000.0 if replay_summary.get("status") == "pass" else 0.0
    jitter_penalty = 0.0
    if feature_jitter_summary:
        if feature_jitter_summary.get("status") != "pass":
            jitter_penalty += 250_000.0
        jitter_penalty += 20_000.0 * len(feature_jitter_summary.get("blockers") or [])
    return (
        pass_bonus
        + 0.50 * validation_pnl
        + 4.0 * diagnostic_pnl
        + 2_500.0 * min(max(validation_pf, 0.0), 3.0)
        + 7_500.0 * min(max(diagnostic_pf, 0.0), 3.0)
        + 250.0 * min(validation_trades, 20.0)
        + 500.0 * min(diagnostic_trades, 20.0)
        - 0.25 * validation_dd
        - 0.50 * diagnostic_dd
        - 35_000.0 * validation_pf_shortfall
        - 100_000.0 * diagnostic_pf_shortfall
        - 2_000.0 * validation_trade_shortfall
        - 3_000.0 * diagnostic_trade_shortfall
        - blocker_penalty
        - jitter_penalty
    )


def reason_for_result(
    *,
    validation_gate: dict[str, Any],
    replay_summary: dict[str, Any],
    feature_jitter_summary: dict[str, Any] | None = None,
) -> str:
    if (
        validation_gate.get("status") == "pass"
        and replay_summary.get("status") == "pass"
        and (not feature_jitter_summary or feature_jitter_summary.get("status") == "pass")
    ):
        return "accepted_for_next_shadow_replay_gate"
    jitter_blockers = (
        list(feature_jitter_summary.get("blockers") or [])
        if feature_jitter_summary
        else []
    )
    blockers = sorted(
        set(
            (validation_gate.get("blockers") or [])
            + (replay_summary.get("blockers") or [])
            + jitter_blockers
        )
    )
    if blockers:
        return "rejected:" + ",".join(str(item) for item in blockers)
    if feature_jitter_summary and feature_jitter_summary.get("status") != "pass":
        return f"rejected:feature_jitter_gate={feature_jitter_summary.get('status')}"
    return f"rejected:validation_gate={validation_gate.get('status')};strict_replay={replay_summary.get('status')}"


def run_attempt(
    *,
    attempt: AttemptConfig,
    root: Path,
    design: Path,
    cwd: Path,
    force: bool,
) -> dict[str, Any]:
    attempt_dir = root / "attempts" / attempt.attempt_id
    if force and attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    train_dir = attempt_dir / "training_runner"
    gate_dir = attempt_dir / "candidate_validation_gate"
    export_dir = attempt_dir / "selected_candidate_export"
    replay_dir = attempt_dir / "selected_candidate_replay_gate"
    jitter_dir = attempt_dir / "feature_jitter_gate"
    training_result = train_dir / "training_result.json"
    if not training_result.exists():
        train_command = [
            sys.executable,
            "-m",
            "v4.scripts.run_protocol101_fair_contract_training_runner",
            "--mode",
            "train",
            "--design",
            str(design),
            "--out-dir",
            str(train_dir),
            "--model-out",
            str(train_dir / "model.pt"),
            "--policy-index",
            str(attempt.policy_index),
            "--fit-mode",
            str(attempt.fit_mode),
            "--threshold-rule",
            attempt.threshold_rule,
            "--threshold-stress-per-trade",
            str(attempt.threshold_stress_per_trade),
            "--hidden-dim",
            str(attempt.hidden_dim),
            "--model-family",
            str(attempt.model_family),
            "--learning-rate",
            str(attempt.learning_rate),
            "--weight-decay",
            str(attempt.weight_decay),
            "--target-mode",
            str(attempt.target_mode),
            "--target-clip",
            str(attempt.target_clip),
            "--positive-label-threshold",
            str(attempt.positive_label_threshold),
            "--relative-target-weight",
            str(attempt.relative_target_weight),
            "--teacher-min-seed-count",
            str(attempt.teacher_min_seed_count),
            "--entry-filter",
            str(attempt.entry_filter),
            "--min-score-margin",
            str(attempt.min_score_margin),
            "--max-score-ceiling",
            str(attempt.max_score_ceiling),
            "--max-trades-per-session",
            str(attempt.max_trades_per_session),
            "--max-daily-loss",
            str(attempt.max_daily_loss),
            "--sample-weight-mode",
            str(attempt.sample_weight_mode),
            "--feature-transform",
            str(attempt.feature_transform),
            "--feature-noise-augmentation",
            str(attempt.feature_noise_augmentation),
            "--selection-mode",
            str(attempt.selection_mode),
            "--epochs",
            str(attempt.epochs),
            "--max-train-examples",
            str(attempt.max_train_examples),
            "--batch-size",
            str(attempt.batch_size),
            "--seed",
            str(attempt.seed),
        ]
        if attempt.ensemble_seeds:
            train_command.extend(["--ensemble-seeds", str(attempt.ensemble_seeds)])
        train_command.extend(
            [
                "--owner-approved-model-training",
                "--owner-approved-threshold-selection",
                "--owner-approval-note",
                "Active goal authorizes offline constrained Protocol101 live-v1 model search; no broker, paid data, paper-submit, promotion, or default change.",
            ]
        )
        run_command(train_command, cwd=cwd)
    run_command(
        [
            sys.executable,
            "-m",
            "v4.scripts.run_protocol101_fair_contract_candidate_validation_gate",
            "--runner-plan",
            str(train_dir / "runner_plan.json"),
            "--training-result",
            str(training_result),
            "--out-dir",
            str(gate_dir),
        ],
        cwd=cwd,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "v4.scripts.run_protocol101_fair_contract_selected_candidate_export",
            "--runner-plan",
            str(train_dir / "runner_plan.json"),
            "--training-result",
            str(training_result),
            "--out-dir",
            str(export_dir),
        ],
        cwd=cwd,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "v4.scripts.run_protocol101_fair_contract_selected_candidate_replay_gate",
            "--selected-export",
            str(export_dir / "summary.json"),
            "--out-dir",
            str(replay_dir),
            "--max-trades-per-session",
            str(attempt.max_trades_per_session),
            "--max-daily-loss",
            str(attempt.max_daily_loss),
        ],
        cwd=cwd,
    )
    if attempt.run_feature_jitter_gate:
        run_command(
            [
                sys.executable,
                "-m",
                "v4.scripts.run_protocol101_fair_contract_feature_jitter_gate",
                "--runner-plan",
                str(train_dir / "runner_plan.json"),
                "--training-result",
                str(training_result),
                "--out-dir",
                str(jitter_dir),
            ],
            cwd=cwd,
        )
    training = load_json(training_result)
    design_payload = load_json(design)
    allowed_data = design_payload.get("allowed_data") or {}
    data_scope = (
        f"{allowed_data.get('included_session_count', 'unknown')}_sessions_"
        f"{allowed_data.get('included_first_session', 'unknown')}_to_"
        f"{allowed_data.get('included_last_session', 'unknown')}_manifest_only"
    )
    validation_gate = load_json(gate_dir / "summary.json")
    selected_export = load_json(export_dir / "summary.json")
    replay = load_json(replay_dir / "summary.json")
    feature_jitter_gate = (
        load_json(jitter_dir / "summary.json")
        if attempt.run_feature_jitter_gate
        else None
    )
    entry = {
        "schema_version": EXPERIMENT_REGISTRY_ENTRY_VERSION,
        "implementation_versions": {
            "selected_candidate_export": str(
                selected_export.get("implementation_version")
                or SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION
            ),
            "strict_replay_gate": str(
                replay.get("implementation_version") or STRICT_REPLAY_IMPLEMENTATION_VERSION
            ),
        },
        "attempt_id": attempt.attempt_id,
        "hypothesis": attempt.hypothesis,
        "config": asdict(attempt),
        "feature_contract": "protocol101-live-v1",
        "data_scope": data_scope,
        "forbidden_data_excluded": ["2026-06-30", "2026-07-01", "2026-07-02"],
        "model_training_executed": True,
        "threshold_selection_executed": True,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "chosen_threshold": training.get("chosen_threshold"),
        "training_metrics": (training.get("neural") or {}),
        "candidate_validation_gate": {
            "status": validation_gate.get("status"),
            "decision": validation_gate.get("decision"),
            "blockers": validation_gate.get("blockers") or [],
        },
        "strict_replay_gate": {
            "status": replay.get("status"),
            "decision": replay.get("decision"),
            "blockers": replay.get("blockers") or [],
            "metrics": replay.get("metrics") or {},
        },
        "feature_jitter_gate": (
            {
                "status": feature_jitter_gate.get("status"),
                "decision": feature_jitter_gate.get("decision"),
                "blockers": feature_jitter_gate.get("blockers") or [],
                "requirements": feature_jitter_gate.get("requirements") or {},
            }
            if feature_jitter_gate
            else {
                "status": "not_run",
                "decision": "feature_jitter_gate_not_requested_for_this_attempt",
                "blockers": [],
                "requirements": {},
            }
        ),
        "objective_score": attempt_score(replay, feature_jitter_gate),
        "reason_accepted_or_rejected": reason_for_result(
            validation_gate=validation_gate,
            replay_summary=replay,
            feature_jitter_summary=feature_jitter_gate,
        ),
        "artifacts": {
            "training_result": str(training_result),
            "model": str(train_dir / "model.pt"),
            "selected_candidates": str(export_dir / "selected_candidates.csv"),
            "strict_replay_trades": str(replay_dir / "strict_replay_trades.csv"),
            "candidate_validation_report": str(gate_dir / "report.md"),
            "strict_replay_report": str(replay_dir / "report.md"),
            "feature_jitter_report": str(jitter_dir / "report.md") if feature_jitter_gate else "",
        },
    }
    return entry


def render_report(entries: list[dict[str, Any]]) -> str:
    ranked = sorted(entries, key=lambda item: float(item.get("objective_score") or 0.0), reverse=True)
    lines = [
        "# Protocol101 Fair-Contract Model Search",
        "",
        "## Decision",
        "",
    ]
    best = ranked[0] if ranked else None
    if best:
        lines.extend(
            [
                f"- Best attempt: `{best['attempt_id']}`",
                f"- Best score: `{best['objective_score']:.2f}`",
                f"- Reason: `{best['reason_accepted_or_rejected']}`",
                f"- Strict replay status: `{best['strict_replay_gate']['status']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Attempt Summary",
            "",
            "| Attempt | Family | Fit mode | Weighting | Feature transform | Noise aug | Jitter gate | Policy | Entry filter | Selection | Margin | Score ceiling | Session cap | Daily loss | Threshold rule | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Status |",
            "|---|---|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for entry in ranked:
        metrics = entry["strict_replay_gate"].get("metrics") or {}
        val = metrics.get("validation") or {}
        diag = metrics.get("diagnostic_test") or {}
        cfg = entry["config"]
        jitter = (entry.get("feature_jitter_gate") or {}).get("status", "not_run")
        lines.append(
            f"| `{entry['attempt_id']}` | {cfg.get('model_family', 'mlp')} | {cfg.get('fit_mode', 'full_train')} | "
            f"{cfg.get('sample_weight_mode', 'none')} | {cfg.get('feature_transform', 'none')} | "
            f"{cfg.get('feature_noise_augmentation', 'none')} | "
            f"{jitter} | "
            f"{cfg['policy_index']} | {cfg.get('entry_filter', 'none')} | "
            f"{cfg.get('selection_mode', 'top_score')} | "
            f"{float(cfg.get('min_score_margin') or 0.0):.2f} | "
            f"{float(cfg.get('max_score_ceiling') or 0.0):.2f} | "
            f"{int(cfg.get('max_trades_per_session') or 0)} | {float(cfg.get('max_daily_loss') or 0.0):.0f} | "
            f"{cfg['threshold_rule']} | "
            f"{float(val.get('total_pnl') or 0.0):.2f} | {float(diag.get('total_pnl') or 0.0):.2f} | "
            f"{float(val.get('profit_factor') or 0.0):.3f} | {float(diag.get('profit_factor') or 0.0):.3f} | "
            f"`{entry['reason_accepted_or_rejected']}` |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- June/July IBKR recorder days were excluded from training and threshold selection.",
            "- No broker endpoints, paper-submit, paid downloads, default changes, or promotions are performed by this search.",
            "- Registry reuse requires current selected-export and strict-replay implementation versions.",
            "- Failed attempts are still registry entries, because negative evidence is part of hill climbing safely.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    attempts = select_attempts(
        default_attempts(),
        max_attempts=int(args.max_attempts),
        attempt_ids=str(args.attempt_ids),
    )
    entries: list[dict[str, Any]] = []
    registry = args.out_dir / "experiment_registry.jsonl"
    if args.force and registry.exists():
        registry.unlink()
    existing_by_id = load_registry(registry) if args.append_only and not args.force else {}
    reused_attempts: list[str] = []
    executed_attempts: list[str] = []
    for attempt in attempts:
        existing_entry = existing_by_id.get(attempt.attempt_id)
        entry = reusable_registry_entry(existing_entry, attempt)
        if entry is not None:
            reused_attempts.append(attempt.attempt_id)
        else:
            changed_existing_config = False
            if existing_entry:
                expected = asdict(attempt)
                actual = dict(existing_entry.get("config") or {})
                for key, value in expected.items():
                    actual.setdefault(key, value)
                changed_existing_config = actual != expected
            entry = run_attempt(
                attempt=attempt,
                root=args.out_dir,
                design=args.design,
                cwd=cwd,
                force=bool(args.force or changed_existing_config),
            )
            executed_attempts.append(attempt.attempt_id)
        entries.append(entry)
    registry.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)
    )
    ranked = sorted(entries, key=lambda item: float(item.get("objective_score") or 0.0), reverse=True)
    summary = {
        "schema_version": MODEL_SEARCH_SUMMARY_VERSION,
        "status": "pass" if ranked and ranked[0]["reason_accepted_or_rejected"].startswith("accepted") else "fail",
        "attempts": len(entries),
        "best_attempt": ranked[0] if ranked else None,
        "registry": str(registry),
        "implementation_versions": current_implementation_versions(),
        "append_only": bool(args.append_only),
        "reused_attempts": reused_attempts,
        "executed_attempts": executed_attempts,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "report.md").write_text(render_report(entries))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "attempts": len(entries),
                "best_attempt": ranked[0]["attempt_id"] if ranked else None,
                "reused_attempts": len(reused_attempts),
                "executed_attempts": len(executed_attempts),
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
