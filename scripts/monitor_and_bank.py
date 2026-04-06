#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results.tsv"
CHART_SCRIPT = ROOT / "scripts" / "update_score_chart.py"
BRANCH = "main"
RESULTS_HEADER = [
    "timestamp",
    "run",
    "status",
    "val_balanced_accuracy_score",
    "best_iteration",
    "commit",
    "branch",
    "description",
    "model_config",
    "feature_config",
]


@dataclass(frozen=True)
class Experiment:
    run_name: str
    log_path: Path
    run_args: list[str]
    description: str
    model_config: str
    feature_config: str


EXPERIMENTS = [
    Experiment(
        run_name="logreg_stack_raw_v2",
        log_path=ROOT / "logreg_stack_raw_v2.log",
        run_args=[
            "--run-name",
            "logreg_stack_raw_v2",
            "--decision-policy",
            "logreg_stack",
            "--meta-raw-features",
            "--prediction-cache",
            "cache/refined_base_v1.npz",
        ],
        description=(
            "If we use a repaired logistic stacker on cached model probabilities plus raw drought "
            "signals, balanced accuracy should improve because the meta-model can recalibrate the "
            "Low/Medium boundary more cleanly."
        ),
        model_config="logistic regression stacker on cached LGB/XGB/CAT OOF probabilities + raw meta features",
        feature_config="cached refined base probabilities + raw moisture/temperature/wind/rain meta features",
    ),
    Experiment(
        run_name="ann_stack_full_v1",
        log_path=ROOT / "ann_stack_full_v1.log",
        run_args=[
            "--run-name",
            "ann_stack_full_v1",
            "--decision-policy",
            "mlp_stack",
            "--meta-full-features",
            "--prediction-cache",
            "cache/refined_base_v1.npz",
        ],
        description=(
            "If we let the ANN stacker see the full engineered numeric set plus categorical context on top "
            "of cached OOF probabilities, balanced accuracy should improve because the meta-model can correct "
            "the remaining Medium/High pockets inside stage-mulch-water subgroups."
        ),
        model_config="ANN stacker on cached LGB/XGB/CAT OOF probabilities + full engineered numeric and categorical meta features",
        feature_config="cached refined base probabilities + engineered numeric features + one-hot categorical context",
    ),
    Experiment(
        run_name="xgb_stack_full_v1",
        log_path=ROOT / "xgb_stack_full_v1.log",
        run_args=[
            "--run-name",
            "xgb_stack_full_v1",
            "--decision-policy",
            "xgb_stack",
            "--meta-full-features",
            "--prediction-cache",
            "cache/refined_base_v1.npz",
        ],
        description=(
            "If we use an XGBoost meta-learner on cached OOF probabilities plus full engineered numeric and "
            "categorical context, balanced accuracy should improve because the second stage can learn sharper "
            "non-linear corrections in the ambiguous Medium versus High regions."
        ),
        model_config="XGBoost stacker on cached LGB/XGB/CAT OOF probabilities + full engineered numeric and categorical meta features",
        feature_config="cached refined base probabilities + engineered numeric features + one-hot categorical context",
    ),
    Experiment(
        run_name="ordinal_xgb_stack_full_v1",
        log_path=ROOT / "ordinal_xgb_stack_full_v1.log",
        run_args=[
            "--run-name",
            "ordinal_xgb_stack_full_v1",
            "--decision-policy",
            "ordinal_xgb_stack",
            "--meta-full-features",
            "--prediction-cache",
            "cache/refined_base_v1.npz",
        ],
        description=(
            "If we model the target as an ordered irrigation-severity score with an XGBoost regressor and "
            "learn the class cut points on OOF predictions, balanced accuracy should improve because the "
            "task looks ordinal and most remaining errors are adjacent-class boundary mistakes."
        ),
        model_config="ordinal XGBoost regressor stacker on cached LGB/XGB/CAT OOF probabilities + full engineered numeric and categorical meta features",
        feature_config="cached refined base probabilities + engineered numeric features + one-hot categorical context + learned ordinal thresholds",
    ),
    Experiment(
        run_name="hgb_stack_full_v1",
        log_path=ROOT / "hgb_stack_full_v1.log",
        run_args=[
            "--run-name",
            "hgb_stack_full_v1",
            "--decision-policy",
            "hgb_stack",
            "--meta-full-features",
            "--prediction-cache",
            "cache/refined_base_v1.npz",
        ],
        description=(
            "If we use a histogram boosting meta-learner on cached OOF probabilities plus full engineered "
            "numeric and categorical context, balanced accuracy should improve because tree splits at the "
            "second stage can enforce local corrections around moisture, temperature, and mulch thresholds."
        ),
        model_config="HistGradientBoosting stacker on cached LGB/XGB/CAT OOF probabilities + full engineered numeric and categorical meta features",
        feature_config="cached refined base probabilities + engineered numeric features + one-hot categorical context",
    ),
    Experiment(
        run_name="stress_signals_crosses_v1",
        log_path=ROOT / "stress_signals_crosses.log",
        run_args=[
            "--run-name",
            "stress_signals_crosses_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we add thresholded drought features, stage-conditioned stress flags, and higher-value "
            "categorical crosses, balanced accuracy should improve because the High-risk agronomic pockets "
            "are more explicit for all three base learners."
        ),
        model_config="lgb+xgb+cat refined blend with richer drought thresholds and categorical interaction features",
        feature_config="base numeric + engineered agronomy features + stress buckets + risk flags + categorical crosses",
    ),
    Experiment(
        run_name="stress_signals_class_scale_v1",
        log_path=ROOT / "stress_signals_class_scale.log",
        run_args=[
            "--run-name",
            "stress_signals_class_scale_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "class_scale_search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we retune class decision scales on top of the richer stress-feature ensemble, balanced "
            "accuracy should improve because High and Medium recall can be rebalanced without retraining "
            "the base models."
        ),
        model_config="stress-feature lgb+xgb+cat blend with class-scale search on cached OOF probabilities",
        feature_config="stress buckets + risk flags + categorical crosses + cached ensemble probabilities",
    ),
    Experiment(
        run_name="stress_signals_model_scale_v1",
        log_path=ROOT / "stress_signals_model_scale.log",
        run_args=[
            "--run-name",
            "stress_signals_model_scale_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "model_scale_search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we let each base model keep its own High and Medium probability correction before blending, "
            "balanced accuracy should improve because LightGBM, XGBoost, and CatBoost make different "
            "calibration mistakes that one global class scale cannot fix."
        ),
        model_config="stress-feature lgb+xgb+cat blend with per-model class-scale search and final class-scale tuning",
        feature_config="stress buckets + risk flags + categorical crosses + cached ensemble probabilities with per-model class calibration",
    ),
    Experiment(
        run_name="stress_signals_ann_stack_v1",
        log_path=ROOT / "stress_signals_ann_stack.log",
        run_args=[
            "--run-name",
            "stress_signals_ann_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "mlp_stack",
            "--meta-raw-features",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we let the ANN stacker see the richer stress-feature ensemble plus raw drought signals, "
            "balanced accuracy should improve because the non-linear meta-boundary can adapt to the harder "
            "Medium versus High pockets."
        ),
        model_config="ANN stacker on stress-feature cached OOF probabilities + raw meta features",
        feature_config="stress buckets + risk flags + categorical crosses + raw drought meta features",
    ),
    Experiment(
        run_name="stress_signals_logreg_stack_v1",
        log_path=ROOT / "stress_signals_logreg_stack.log",
        run_args=[
            "--run-name",
            "stress_signals_logreg_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "logreg_stack",
            "--meta-raw-features",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we stack the richer stress-feature ensemble with a repaired logistic meta-model, balanced "
            "accuracy should improve because the calibrated linear combiner may regularize the noisier stress "
            "signals better than argmax."
        ),
        model_config="logistic regression stacker on stress-feature cached OOF probabilities + raw meta features",
        feature_config="stress buckets + risk flags + categorical crosses + raw drought meta features",
    ),
    Experiment(
        run_name="stress_signals_ann_stack_full_v1",
        log_path=ROOT / "stress_signals_ann_stack_full.log",
        run_args=[
            "--run-name",
            "stress_signals_ann_stack_full_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "mlp_stack",
            "--meta-full-features",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we let the ANN stacker see the richer stress-feature ensemble plus the full engineered "
            "numeric and categorical context, balanced accuracy should improve because the meta-model can "
            "correct local Medium/High pockets that still survive the base blend."
        ),
        model_config="ANN stacker on stress-feature cached OOF probabilities + full engineered numeric and categorical meta features",
        feature_config="stress buckets + risk flags + categorical crosses + full engineered numeric features + one-hot categorical context",
    ),
    Experiment(
        run_name="stress_signals_xgb_stack_v1",
        log_path=ROOT / "stress_signals_xgb_stack.log",
        run_args=[
            "--run-name",
            "stress_signals_xgb_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "xgb_stack",
            "--meta-full-features",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we use an XGBoost meta-learner on the richer stress-feature ensemble plus full engineered "
            "numeric and categorical context, balanced accuracy should improve because the second stage can "
            "learn sharper non-linear corrections around the drought-threshold pockets."
        ),
        model_config="XGBoost stacker on stress-feature cached OOF probabilities + full engineered numeric and categorical meta features",
        feature_config="stress buckets + risk flags + categorical crosses + full engineered numeric features + one-hot categorical context",
    ),
    Experiment(
        run_name="stress_signals_hgb_stack_v1",
        log_path=ROOT / "stress_signals_hgb_stack.log",
        run_args=[
            "--run-name",
            "stress_signals_hgb_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "hgb_stack",
            "--meta-full-features",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we use histogram boosting at the meta layer on the richer stress-feature ensemble plus full "
            "engineered numeric and categorical context, balanced accuracy should improve because local splits "
            "can correct threshold-shaped errors the base blend still makes."
        ),
        model_config="HistGradientBoosting stacker on stress-feature cached OOF probabilities + full engineered numeric and categorical meta features",
        feature_config="stress buckets + risk flags + categorical crosses + full engineered numeric features + one-hot categorical context",
    ),
    Experiment(
        run_name="stress_signals_ft_stack_v1",
        log_path=ROOT / "stress_signals_ft_stack.log",
        run_args=[
            "--run-name",
            "stress_signals_ft_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "ft_transformer_stack",
            "--meta-raw-features",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we use an FT-transformer stacker on the richer stress-feature ensemble plus raw drought "
            "signals, balanced accuracy should improve because feature-token attention can model the "
            "remaining Medium versus High interactions more flexibly than the MLP stacker."
        ),
        model_config="FT-transformer stacker on stress-feature cached OOF probabilities + raw meta features",
        feature_config="stress buckets + risk flags + categorical crosses + cached ensemble probabilities + raw drought meta features",
    ),
    Experiment(
        run_name="stress_signals_ann_stack_scaled_v1",
        log_path=ROOT / "stress_signals_ann_stack_scaled.log",
        run_args=[
            "--run-name",
            "stress_signals_ann_stack_scaled_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "mlp_stack",
            "--meta-raw-features",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we search class-specific decision scales on top of the stress-feature ANN stacker's OOF "
            "probabilities, balanced accuracy should improve because the current champion still uses raw "
            "argmax despite Medium being the weakest recall class."
        ),
        model_config="ANN stacker on stress-feature cached OOF probabilities + raw meta features + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + raw drought meta features + calibrated meta probabilities",
    ),
    Experiment(
        run_name="stress_signals_ann_bag_stack_v1",
        log_path=ROOT / "stress_signals_ann_bag_stack.log",
        run_args=[
            "--run-name",
            "stress_signals_ann_bag_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "mlp_bag_stack",
            "--meta-raw-features",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we average several ANN stacker seeds before the final class-scale search, balanced "
            "accuracy should improve because seed bagging can smooth unstable Medium-versus-High pockets "
            "that a single ANN initialization overfits."
        ),
        model_config="seed-bagged ANN stacker on stress-feature cached OOF probabilities + raw meta features + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + raw drought meta features + bagged meta probabilities",
    ),
    Experiment(
        run_name="stress_signals_cnn_stack_v1",
        log_path=ROOT / "stress_signals_cnn_stack.log",
        run_args=[
            "--run-name",
            "stress_signals_cnn_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "cnn_stack",
            "--meta-raw-features",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we use a CNN meta-model on the stress-feature ensemble plus raw drought signals, balanced "
            "accuracy should improve because local convolutional filters can detect recurring probability "
            "and stress-shape motifs that the dense ANN treats independently."
        ),
        model_config="CNN stacker on stress-feature cached OOF probabilities + raw meta features + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + cached ensemble probabilities + raw drought meta features",
    ),
    Experiment(
        run_name="stress_signals_rnn_stack_v1",
        log_path=ROOT / "stress_signals_rnn_stack.log",
        run_args=[
            "--run-name",
            "stress_signals_rnn_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "rnn_stack",
            "--meta-raw-features",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we use a GRU-based RNN meta-model on the stress-feature ensemble plus raw drought signals, "
            "balanced accuracy should improve because the recurrent stacker can accumulate ordered evidence "
            "across model probabilities and drought context before making the final class decision."
        ),
        model_config="GRU-based RNN stacker on stress-feature cached OOF probabilities + raw meta features + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + cached ensemble probabilities + raw drought meta features",
    ),
    Experiment(
        run_name="stress_signals_tabnet_stack_v1",
        log_path=ROOT / "stress_signals_tabnet_stack.log",
        run_args=[
            "--run-name",
            "stress_signals_tabnet_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "tabnet_stack",
            "--meta-raw-features",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we use a TabNet stacker on the stress-feature ensemble plus raw drought signals, balanced "
            "accuracy should improve because sequential feature masking can focus on the strongest Medium "
            "versus High cues instead of treating every meta-feature equally."
        ),
        model_config="TabNet stacker on stress-feature cached OOF probabilities + raw meta features + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + cached ensemble probabilities + raw drought meta features",
    ),
    Experiment(
        run_name="stress_signals_ann_cnn_combo_v1",
        log_path=ROOT / "stress_signals_ann_cnn_combo.log",
        run_args=[
            "--run-name",
            "stress_signals_ann_cnn_combo_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "ann_cnn_combo_stack",
            "--meta-raw-features",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we blend ANN and CNN stacker probabilities on the stress-feature ensemble, balanced "
            "accuracy should improve because the dense model and convolutional model capture different "
            "meta-patterns in the Medium-versus-High boundary."
        ),
        model_config="ANN+CNN combo stacker on stress-feature cached OOF probabilities + raw meta features + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + cached ensemble probabilities + raw drought meta features",
    ),
    Experiment(
        run_name="stress_signals_neural_base_blend_v1",
        log_path=ROOT / "stress_signals_neural_base_blend.log",
        run_args=[
            "--run-name",
            "stress_signals_neural_base_blend_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "neural_base_blend_stack",
            "--meta-raw-features",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we blend LGB, XGB, CatBoost, ANN, and CNN probabilities on the stress-feature ensemble, "
            "balanced accuracy should improve because the neural stackers can contribute complementary "
            "boundary corrections while CatBoost still anchors the strongest base signal."
        ),
        model_config="LGB+XGB+CatBoost+ANN+CNN probability blend on stress-feature cached OOF outputs + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + cached base and neural meta probabilities",
    ),
    Experiment(
        run_name="target_freq_enc_v1",
        log_path=ROOT / "target_freq_enc.log",
        run_args=[
            "--run-name",
            "target_freq_enc_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--frequency-encoding",
            "--target-encoding",
            "--prediction-cache",
            "cache/target_freq_enc_v1.npz",
        ],
        description=(
            "If we add cross-fitted target encoding plus frequency encoding on the full categorical family, "
            "balanced accuracy should improve because LightGBM and XGBoost will finally see category-level "
            "irrigation propensities instead of only ordinal placeholders."
        ),
        model_config="lgb+xgb+cat refined blend with frequency encoding and leakage-safe multiclass target encoding",
        feature_config="stress buckets + risk flags + categorical crosses + frequency encoding + cross-fitted target encoding",
    ),
    Experiment(
        run_name="target_freq_class_scale_v1",
        log_path=ROOT / "target_freq_class_scale.log",
        run_args=[
            "--run-name",
            "target_freq_class_scale_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--frequency-encoding",
            "--target-encoding",
            "--decision-policy",
            "class_scale_search",
            "--prediction-cache",
            "cache/target_freq_enc_v1.npz",
        ],
        description=(
            "If we retune class decision scales on top of the target-encoded ensemble, balanced accuracy "
            "should improve because the encoded category propensities may support a cleaner High-versus-Medium "
            "trade-off."
        ),
        model_config="target-encoded lgb+xgb+cat refined blend with class-scale search on cached OOF probabilities",
        feature_config="stress buckets + risk flags + categorical crosses + frequency encoding + cross-fitted target encoding",
    ),
    Experiment(
        run_name="target_freq_ann_stack_v1",
        log_path=ROOT / "target_freq_ann_stack.log",
        run_args=[
            "--run-name",
            "target_freq_ann_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--frequency-encoding",
            "--target-encoding",
            "--decision-policy",
            "mlp_stack",
            "--meta-raw-features",
            "--prediction-cache",
            "cache/target_freq_enc_v1.npz",
        ],
        description=(
            "If we stack the target-encoded ensemble with the ANN meta-model, balanced accuracy should "
            "improve because the meta-boundary can exploit both encoded categorical propensities and raw "
            "drought features."
        ),
        model_config="ANN stacker on target-encoded cached OOF probabilities + raw meta features",
        feature_config="stress buckets + risk flags + categorical crosses + frequency encoding + cross-fitted target encoding + raw drought meta features",
    ),
    Experiment(
        run_name="target_freq_logreg_stack_v1",
        log_path=ROOT / "target_freq_logreg_stack.log",
        run_args=[
            "--run-name",
            "target_freq_logreg_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--frequency-encoding",
            "--target-encoding",
            "--decision-policy",
            "logreg_stack",
            "--meta-raw-features",
            "--prediction-cache",
            "cache/target_freq_enc_v1.npz",
        ],
        description=(
            "If we use the repaired logistic stacker on the target-encoded ensemble, balanced accuracy "
            "should improve because the encoded probabilities may be easier to calibrate linearly than the "
            "raw ordinal category placeholders."
        ),
        model_config="logistic regression stacker on target-encoded cached OOF probabilities + raw meta features",
        feature_config="stress buckets + risk flags + categorical crosses + frequency encoding + cross-fitted target encoding + raw drought meta features",
    ),
    Experiment(
        run_name="target_freq_ann_stack_full_v1",
        log_path=ROOT / "target_freq_ann_stack_full.log",
        run_args=[
            "--run-name",
            "target_freq_ann_stack_full_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--frequency-encoding",
            "--target-encoding",
            "--decision-policy",
            "mlp_stack",
            "--meta-full-features",
            "--prediction-cache",
            "cache/target_freq_enc_v1.npz",
        ],
        description=(
            "If we let the ANN stacker see the target-encoded ensemble plus the full engineered numeric and "
            "categorical context, balanced accuracy should improve because the meta-model can combine encoded "
            "category propensities with local agronomic threshold signals."
        ),
        model_config="ANN stacker on target-encoded cached OOF probabilities + full engineered numeric and categorical meta features",
        feature_config="stress buckets + risk flags + categorical crosses + frequency encoding + cross-fitted target encoding + full engineered numeric features + one-hot categorical context",
    ),
    Experiment(
        run_name="target_freq_xgb_stack_v1",
        log_path=ROOT / "target_freq_xgb_stack.log",
        run_args=[
            "--run-name",
            "target_freq_xgb_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--frequency-encoding",
            "--target-encoding",
            "--decision-policy",
            "xgb_stack",
            "--meta-full-features",
            "--prediction-cache",
            "cache/target_freq_enc_v1.npz",
        ],
        description=(
            "If we use an XGBoost meta-learner on the target-encoded ensemble plus full engineered numeric "
            "and categorical context, balanced accuracy should improve because the second stage can exploit "
            "non-linear interactions between encoded category propensities and drought thresholds."
        ),
        model_config="XGBoost stacker on target-encoded cached OOF probabilities + full engineered numeric and categorical meta features",
        feature_config="stress buckets + risk flags + categorical crosses + frequency encoding + cross-fitted target encoding + full engineered numeric features + one-hot categorical context",
    ),
    Experiment(
        run_name="target_freq_hgb_stack_v1",
        log_path=ROOT / "target_freq_hgb_stack.log",
        run_args=[
            "--run-name",
            "target_freq_hgb_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--frequency-encoding",
            "--target-encoding",
            "--decision-policy",
            "hgb_stack",
            "--meta-full-features",
            "--prediction-cache",
            "cache/target_freq_enc_v1.npz",
        ],
        description=(
            "If we use histogram boosting at the meta layer on the target-encoded ensemble plus full "
            "engineered numeric and categorical context, balanced accuracy should improve because local tree "
            "splits can refine the remaining threshold-shaped mistakes in encoded category pockets."
        ),
        model_config="HistGradientBoosting stacker on target-encoded cached OOF probabilities + full engineered numeric and categorical meta features",
        feature_config="stress buckets + risk flags + categorical crosses + frequency encoding + cross-fitted target encoding + full engineered numeric features + one-hot categorical context",
    ),
    Experiment(
        run_name="target_freq_ft_stack_v1",
        log_path=ROOT / "target_freq_ft_stack.log",
        run_args=[
            "--run-name",
            "target_freq_ft_stack_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--frequency-encoding",
            "--target-encoding",
            "--decision-policy",
            "ft_transformer_stack",
            "--meta-raw-features",
            "--prediction-cache",
            "cache/target_freq_enc_v1.npz",
        ],
        description=(
            "If we use an FT-transformer stacker on the target-encoded ensemble plus raw drought signals, "
            "balanced accuracy should improve because feature-token attention can combine encoded category "
            "propensities with local agronomic interactions better than the current linear and MLP stackers."
        ),
        model_config="FT-transformer stacker on target-encoded cached OOF probabilities + raw meta features",
        feature_config="stress buckets + risk flags + categorical crosses + frequency encoding + cross-fitted target encoding + raw drought meta features",
    ),
    Experiment(
        run_name="notebook_eda_ann_stack_scaled_v1",
        log_path=ROOT / "notebook_eda_ann_stack_scaled.log",
        run_args=[
            "--run-name",
            "notebook_eda_ann_stack_scaled_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--notebook-eda-features",
            "--decision-policy",
            "mlp_stack",
            "--meta-raw-features",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/notebook_eda_base_v1.npz",
        ],
        description=(
            "If we add the notebook-derived threshold, logit, decimal, and extra agronomy features to the "
            "current stress-feature champion stack, balanced accuracy should improve because the meta-model "
            "gets explicit generator-like irrigation boundaries instead of inferring them from scratch."
        ),
        model_config="ANN stacker on notebook-augmented stress-feature cached OOF probabilities + raw meta features + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + notebook threshold/logit/decimal/domain features + curated notebook bigram crosses",
    ),
    Experiment(
        run_name="rare_group_pairte_base_v1",
        log_path=ROOT / "rare_group_pairte_base_v1.log",
        run_args=[
            "--run-name",
            "rare_group_pairte_base_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--rare-category-bucketing",
            "--group-aggregates",
            "--pairwise-target-encoding",
            "--prediction-cache",
            "cache/rare_group_pairte_base_v1.npz",
        ],
        description=(
            "If we add rare-category bucketing, grouped numeric aggregates, and leakage-safe pairwise "
            "target encoding to the stress-feature ensemble, balanced accuracy should improve because the "
            "base learners will see stable category-pair irrigation propensities and within-group drought "
            "deviations instead of only raw category IDs."
        ),
        model_config="lgb+xgb+cat refined blend with rare-category bucketing, grouped numeric aggregates, and pairwise target encoding",
        feature_config="stress buckets + risk flags + categorical crosses + rare-bucketed categoricals + grouped numeric means/deltas/z-scores + pairwise target encoding",
    ),
    Experiment(
        run_name="rare_group_pairte_ann_scaled_v1",
        log_path=ROOT / "rare_group_pairte_ann_scaled_v1.log",
        run_args=[
            "--run-name",
            "rare_group_pairte_ann_scaled_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--rare-category-bucketing",
            "--group-aggregates",
            "--pairwise-target-encoding",
            "--decision-policy",
            "mlp_stack",
            "--meta-raw-features",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/rare_group_pairte_base_v1.npz",
        ],
        description=(
            "If we stack the new rare-bucketed, group-aggregate, pairwise-target-encoded ensemble with the "
            "ANN meta-model, balanced accuracy should improve because the stacker can combine the stronger "
            "base probabilities with group-level drought anomalies and pairwise category propensities."
        ),
        model_config="ANN stacker on rare-bucketed grouped-aggregate pairwise-target-encoded cached OOF probabilities + raw meta features + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + rare-bucketed categoricals + grouped numeric means/deltas/z-scores + pairwise target encoding + calibrated meta probabilities",
    ),
    Experiment(
        run_name="stress_fs_mlp_hybrid_v1",
        log_path=ROOT / "stress_fs_mlp_hybrid_v1.log",
        run_args=[
            "--run-name",
            "stress_fs_mlp_hybrid_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "mlp_stack",
            "--meta-raw-features",
            "--meta-full-features",
            "--feature-selection",
            "--feature-selection-topk",
            "96",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we run the ANN stacker on a feature-selected hybrid meta set, balanced accuracy should improve "
            "because the stacker can keep the strongest raw and one-hot context while dropping redundant and noisy "
            "meta columns that previously diluted the full-feature run."
        ),
        model_config="ANN stacker on stress-feature cached OOF probabilities + raw/full hybrid meta features + cross-fitted feature selection + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + raw meta features + full one-hot meta context + protected stack core + selected top-k features",
    ),
    Experiment(
        run_name="stress_fs_xgb_hybrid_v1",
        log_path=ROOT / "stress_fs_xgb_hybrid_v1.log",
        run_args=[
            "--run-name",
            "stress_fs_xgb_hybrid_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "xgb_stack",
            "--meta-raw-features",
            "--meta-full-features",
            "--feature-selection",
            "--feature-selection-topk",
            "96",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we run XGBoost on the selected hybrid stacker features, balanced accuracy should improve because "
            "tree splits can focus on the strongest meta interactions after the redundant one-hot tail is removed."
        ),
        model_config="XGBoost stacker on stress-feature cached OOF probabilities + raw/full hybrid meta features + cross-fitted feature selection + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + raw meta features + full one-hot meta context + protected stack core + selected top-k features",
    ),
    Experiment(
        run_name="stress_fs_hgb_hybrid_v1",
        log_path=ROOT / "stress_fs_hgb_hybrid_v1.log",
        run_args=[
            "--run-name",
            "stress_fs_hgb_hybrid_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "hgb_stack",
            "--meta-raw-features",
            "--meta-full-features",
            "--feature-selection",
            "--feature-selection-topk",
            "96",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we run HistGradientBoosting on the selected hybrid stacker features, balanced accuracy should improve "
            "because the boosted tree can exploit the sharper reduced feature set without overfitting the full meta expansion."
        ),
        model_config="HistGradientBoosting stacker on stress-feature cached OOF probabilities + raw/full hybrid meta features + cross-fitted feature selection + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + raw meta features + full one-hot meta context + protected stack core + selected top-k features",
    ),
    Experiment(
        run_name="stress_fs_tabnet_hybrid_v1",
        log_path=ROOT / "stress_fs_tabnet_hybrid_v1.log",
        run_args=[
            "--run-name",
            "stress_fs_tabnet_hybrid_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "tabnet_stack",
            "--meta-raw-features",
            "--meta-full-features",
            "--feature-selection",
            "--feature-selection-topk",
            "96",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we run TabNet on the selected hybrid stacker features, balanced accuracy should improve because "
            "sparse attentive steps should work better once the meta feature set is pruned down to the strongest signals."
        ),
        model_config="TabNet stacker on stress-feature cached OOF probabilities + raw/full hybrid meta features + cross-fitted feature selection + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + raw meta features + full one-hot meta context + protected stack core + selected top-k features",
    ),
    Experiment(
        run_name="stress_fs_neural_blend_hybrid_v1",
        log_path=ROOT / "stress_fs_neural_blend_hybrid_v1.log",
        run_args=[
            "--run-name",
            "stress_fs_neural_blend_hybrid_v1",
            "--categorical-crosses",
            "--risk-flags",
            "--stress-signals",
            "--decision-policy",
            "neural_base_blend_stack",
            "--meta-raw-features",
            "--meta-full-features",
            "--feature-selection",
            "--feature-selection-topk",
            "96",
            "--stack-class-scale-search",
            "--prediction-cache",
            "cache/stress_signals_crosses_v1.npz",
        ],
        description=(
            "If we blend the selected-feature ANN/CNN meta models back with the tree base ensemble, balanced accuracy "
            "should improve because the neural correction layer can stay expressive without carrying the redundant full-meta tail."
        ),
        model_config="ANN+CNN+tree blend on stress-feature cached OOF probabilities + raw/full hybrid meta features + cross-fitted feature selection + final class-scale calibration",
        feature_config="stress buckets + risk flags + categorical crosses + raw meta features + full one-hot meta context + protected stack core + selected top-k features",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)

    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(cmd, cwd=cwd, check=True, text=True, stdout=handle, stderr=subprocess.STDOUT)


def ensure_results_file() -> None:
    if RESULTS_PATH.exists():
        return

    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writeheader()


def read_results() -> list[dict[str, str]]:
    ensure_results_file()
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def write_results(rows: list[dict[str, str]]) -> None:
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def append_result(row: dict[str, str]) -> None:
    rows = read_results()
    rows.append(row)
    write_results(rows)


def find_row(run_name: str) -> dict[str, str] | None:
    for row in read_results():
        if row["run"] == run_name:
            return row
    return None


def champion_score() -> float:
    scores = []
    for row in read_results():
        if row["status"] != "champion":
            continue
        try:
            scores.append(float(row["val_balanced_accuracy_score"]))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else float("-inf")


def extract_text(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_log(log_path: Path) -> dict[str, str | float]:
    if not log_path.exists():
        return {"state": "missing"}

    text = log_path.read_text(encoding="utf-8", errors="replace")
    score = extract_text(r"^val_balanced_accuracy_score:\s*([0-9.]+)$", text)
    best_iteration = extract_text(r"^best_iteration:\s*([0-9]+)$", text)
    weakest_recall = extract_text(r"^weakest_class_recall:\s*(.+)$", text)

    if score is not None:
        return {
            "state": "complete",
            "score": float(score),
            "best_iteration": best_iteration or "",
            "weakest_recall": weakest_recall or "",
        }

    if "Traceback (most recent call last):" in text:
        tail = text.strip().splitlines()[-1] if text.strip() else "Traceback"
        return {"state": "crash", "reason": tail}

    return {"state": "running"}


def timestamp_for(log_path: Path) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(log_path.stat().st_mtime))


def chart_update() -> None:
    run_command(["uv", "run", "python", str(CHART_SCRIPT)], ROOT)


def stage_commit_push(run_name: str) -> str:
    tracked_paths = [
        ".gitignore",
        "README.md",
        "scripts",
        "artifacts/champion_score_history.svg",
    ]
    subprocess.run(["git", "add", *tracked_paths], cwd=ROOT, check=True)

    cached_diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
        check=False,
    )
    if cached_diff.returncode == 0:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    subprocess.run(
        ["git", "commit", "-m", f"Promote {run_name} champion"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "push", "origin", BRANCH], cwd=ROOT, check=True)
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def replace_pending_commit(run_name: str, commit_hash: str) -> None:
    rows = read_results()
    for row in rows:
        if row["run"] == run_name and row["commit"] == "pending":
            row["commit"] = commit_hash
            row["branch"] = BRANCH
    write_results(rows)


def build_row(
    *,
    exp: Experiment,
    status: str,
    timestamp: str,
    score: str = "",
    best_iteration: str = "",
    description: str = "",
) -> dict[str, str]:
    return {
        "timestamp": timestamp,
        "run": exp.run_name,
        "status": status,
        "val_balanced_accuracy_score": score,
        "best_iteration": best_iteration,
        "commit": "pending",
        "branch": BRANCH,
        "description": description,
        "model_config": exp.model_config,
        "feature_config": exp.feature_config,
    }


def handle_complete(exp: Experiment, parsed: dict[str, str | float]) -> None:
    score = float(parsed["score"])
    best = champion_score()
    timestamp = timestamp_for(exp.log_path)
    weakest_recall = str(parsed.get("weakest_recall", ""))

    if score > best + 1e-12:
        description = exp.description
        if weakest_recall:
            description += f" Weakest recall: {weakest_recall}."

        append_result(
            build_row(
                exp=exp,
                status="champion",
                timestamp=timestamp,
                score=f"{score:.6f}",
                best_iteration=str(parsed.get("best_iteration", "")),
                description=description,
            )
        )

        write_log = ROOT / f"{exp.run_name}_write.log"
        run_command(
            ["uv", "run", "scripts/solution.py", *exp.run_args],
            ROOT,
            write_log,
        )
        chart_update()
        commit_hash = stage_commit_push(exp.run_name)
        replace_pending_commit(exp.run_name, commit_hash)
        print(f"[champion] {exp.run_name}: {score:.6f} ({commit_hash})", flush=True)
    else:
        description = (
            f"{exp.description} No lift versus champion {best:.6f}; "
            f"finished at {score:.6f}."
        )
        append_result(
            build_row(
                exp=exp,
                status="discard",
                timestamp=timestamp,
                score=f"{score:.6f}",
                best_iteration=str(parsed.get("best_iteration", "")),
                description=description,
            )
        )
        print(f"[discard] {exp.run_name}: {score:.6f}", flush=True)


def resume_pending_champion(exp: Experiment, parsed: dict[str, str | float]) -> None:
    score = float(parsed["score"])
    write_log = ROOT / f"{exp.run_name}_write.log"
    run_command(
        ["uv", "run", "scripts/solution.py", *exp.run_args],
        ROOT,
        write_log,
    )
    chart_update()
    commit_hash = stage_commit_push(exp.run_name)
    replace_pending_commit(exp.run_name, commit_hash)
    print(f"[champion-resumed] {exp.run_name}: {score:.6f} ({commit_hash})", flush=True)


def handle_crash(exp: Experiment, parsed: dict[str, str | float]) -> None:
    append_result(
        build_row(
            exp=exp,
            status="crash",
            timestamp=timestamp_for(exp.log_path),
            description=f"{exp.description} Crash: {parsed['reason']}",
        )
    )
    print(f"[crash] {exp.run_name}: {parsed['reason']}", flush=True)


def main() -> None:
    args = parse_args()
    deadline = time.time() + args.hours * 3600
    ensure_results_file()

    while time.time() < deadline:
        for exp in EXPERIMENTS:
            row = find_row(exp.run_name)
            if row is not None and not (
                row["status"] == "champion" and row["commit"] == "pending"
            ):
                continue

            parsed = parse_log(exp.log_path)
            state = parsed["state"]
            if row is not None and row["status"] == "champion" and row["commit"] == "pending":
                if state == "complete":
                    resume_pending_champion(exp, parsed)
                continue

            if state == "complete":
                handle_complete(exp, parsed)
            elif state == "crash":
                handle_crash(exp, parsed)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
