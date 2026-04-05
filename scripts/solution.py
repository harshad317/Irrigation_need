"""
Kaggle Playground 2026 - Irrigation Need Prediction
Metric: Balanced Accuracy (multiclass: Low / Medium / High)
High class is only ~3.3% → heavy imbalance, must use class weights.

Run with:
    uv run scripts/solution.py
"""

import argparse
import os, sys

# Fix libomp path on macOS without Homebrew (uses sklearn's bundled libomp)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_libomp_dir = os.path.join(
    _project_root, ".venv", "lib", "python3.13", "site-packages", "sklearn", ".dylibs"
)
if os.path.isdir(_libomp_dir) and "DYLD_LIBRARY_PATH" not in os.environ:
    os.environ["DYLD_LIBRARY_PATH"] = _libomp_dir
    os.execv(sys.executable, [sys.executable] + sys.argv)  # restart with env set

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 45          # fixed validation split seed per evaluation protocol
N_FOLDS = 5
TARGET = "Irrigation_Need"
LABEL_ORDER = ["Low", "Medium", "High"]  # encoded as 0, 1, 2
MODEL_LABELS = ["LGB", "XGB", "CAT"]
MIN_BLEND_WEIGHT = 0.05
STACKER_N_JOBS = max(1, min(4, os.cpu_count() or 1))
torch.set_num_threads(STACKER_N_JOBS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="manual")
    parser.add_argument(
        "--blend-strategy",
        choices=["coarse_grid", "refined_weight_search"],
        default="refined_weight_search",
    )
    parser.add_argument(
        "--decision-policy",
        choices=[
            "argmax",
            "class_scale_search",
            "model_scale_search",
            "logreg_stack",
            "mlp_stack",
            "mlp_bag_stack",
            "tabnet_stack",
            "ann_cnn_combo_stack",
            "neural_base_blend_stack",
            "ft_transformer_stack",
            "cnn_stack",
            "rnn_stack",
            "xgb_stack",
            "hgb_stack",
            "ordinal_xgb_stack",
        ],
        default="argmax",
    )
    parser.add_argument("--prediction-cache", default="")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--categorical-crosses", action="store_true")
    parser.add_argument("--risk-flags", action="store_true")
    parser.add_argument("--stress-signals", action="store_true")
    parser.add_argument("--meta-raw-features", action="store_true")
    parser.add_argument("--meta-full-features", action="store_true")
    parser.add_argument("--frequency-encoding", action="store_true")
    parser.add_argument("--target-encoding", action="store_true")
    parser.add_argument("--target-encoding-smoothing", type=float, default=50.0)
    parser.add_argument("--stack-class-scale-search", action="store_true")
    return parser.parse_args()


ARGS = parse_args()

# Paths
DATA_DIR = os.path.join(_project_root, "Data")
PRED_DIR = os.path.join(_project_root, "Predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub   = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(f"Train: {train.shape}  Test: {test.shape}")
print("Target distribution:\n", train[TARGET].value_counts())
print(f"Run name: {ARGS.run_name}")
print(f"Blend strategy: {ARGS.blend_strategy}")
print(f"Decision policy: {ARGS.decision_policy}")
if ARGS.prediction_cache:
    print(f"Prediction cache: {ARGS.prediction_cache}")
print(f"Categorical crosses: {ARGS.categorical_crosses}")
print(f"Risk flags: {ARGS.risk_flags}")
print(f"Stress signals: {ARGS.stress_signals}")
print(f"Meta raw features: {ARGS.meta_raw_features}")
print(f"Meta full features: {ARGS.meta_full_features}")
print(f"Frequency encoding: {ARGS.frequency_encoding}")
print(f"Target encoding: {ARGS.target_encoding}")
print(f"Stack class scale search: {ARGS.stack_class_scale_search}")
if ARGS.target_encoding:
    print(f"Target encoding smoothing: {ARGS.target_encoding_smoothing:.1f}")

# ── 2. Encode target ──────────────────────────────────────────────────────────

label_enc = LabelEncoder()
label_enc.classes_ = np.array(LABEL_ORDER)
train["label"] = label_enc.transform(train[TARGET])

# ── 3. Feature engineering ────────────────────────────────────────────────────

BASE_CAT_COLS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
CROSS_CAT_COLS = [
    "Crop_Type__Season",
    "Irrigation_Type__Water_Source",
    "Soil_Type__Region",
    "Crop_Growth_Stage__Mulching_Used",
    "Crop_Growth_Stage__Mulching_Used__Water_Source",
    "Crop_Growth_Stage__Mulching_Used__Irrigation_Type",
]
NUM_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
ENG_COLS = [
    "water_stress", "et_proxy", "effective_rain", "rain_per_area",
    "prev_irr_per_ha", "aridity", "heat_index", "soil_quality",
]
RISK_FLAG_COLS = [
    "is_peak_growth_stage",
    "is_peak_growth_without_mulch",
    "is_peak_growth_without_mulch_river",
    "is_peak_growth_without_mulch_canal",
    "is_low_need_mulched_stage",
]
STRESS_NUM_COLS = [
    "moisture_deficit",
    "dry_heat",
    "wind_heat",
    "drought_pressure",
    "rainfall_relief",
    "prev_vs_rain",
    "peak_stage_drought",
    "peak_stage_no_mulch_drought",
    "river_peak_no_mulch_drought",
    "canal_peak_no_mulch_drought",
    "stress_count",
]
STRESS_CAT_COLS = [
    "Soil_Moisture_bin",
    "Temperature_C_bin",
    "Wind_Speed_kmh_bin",
    "Rainfall_mm_bin",
    "stress_count_cat",
    "drought_pressure_bin",
    "peak_stage_bucket",
]

def add_features(df):
    df = df.copy()
    peak_stage = df["Crop_Growth_Stage"].isin(["Vegetative", "Flowering"])
    no_mulch = df["Mulching_Used"].eq("No")
    river_source = df["Water_Source"].eq("River")
    canal_irrigation = df["Irrigation_Type"].eq("Canal")
    # High temp + low moisture + low rainfall → more irrigation needed
    df["water_stress"]    = df["Temperature_C"] / (df["Soil_Moisture"] + 1) / (df["Rainfall_mm"] + 1)
    # Evapotranspiration proxy
    df["et_proxy"]        = df["Temperature_C"] * df["Sunlight_Hours"] * (1 - df["Humidity"] / 100)
    # Effective rainfall after soil absorption
    df["effective_rain"]  = df["Rainfall_mm"] * (1 - df["Soil_Moisture"] / 100)
    # Rainfall per unit area
    df["rain_per_area"]   = df["Rainfall_mm"] / (df["Field_Area_hectare"] + 0.01)
    # Previous irrigation relative to area
    df["prev_irr_per_ha"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 0.01)
    # Aridity index proxy
    df["aridity"]         = (df["Temperature_C"] * df["Wind_Speed_kmh"]) / (df["Rainfall_mm"] + 1)
    # Heat-humidity feel index
    df["heat_index"]      = df["Temperature_C"] * df["Humidity"] / 100
    # Soil quality (organic carbon vs salinity)
    df["soil_quality"]    = df["Organic_Carbon"] / (df["Electrical_Conductivity"] + 0.01)
    if ARGS.risk_flags:
        low_need_stage = df["Crop_Growth_Stage"].isin(["Sowing", "Harvest"]) & df["Mulching_Used"].eq("Yes")

        df["is_peak_growth_stage"] = peak_stage.astype(np.int8)
        df["is_peak_growth_without_mulch"] = (peak_stage & no_mulch).astype(np.int8)
        df["is_peak_growth_without_mulch_river"] = (peak_stage & no_mulch & river_source).astype(np.int8)
        df["is_peak_growth_without_mulch_canal"] = (peak_stage & no_mulch & canal_irrigation).astype(np.int8)
        df["is_low_need_mulched_stage"] = low_need_stage.astype(np.int8)
    if ARGS.stress_signals:
        moisture_deficit = 100 - df["Soil_Moisture"]
        drought_pressure = (
            moisture_deficit * (df["Temperature_C"] + df["Wind_Speed_kmh"])
            / (df["Rainfall_mm"] + 10)
        )
        peak_no_mulch = peak_stage & no_mulch

        df["moisture_deficit"] = moisture_deficit
        df["dry_heat"] = df["Temperature_C"] * moisture_deficit / 100
        df["wind_heat"] = df["Temperature_C"] * df["Wind_Speed_kmh"]
        df["drought_pressure"] = drought_pressure
        df["rainfall_relief"] = df["Rainfall_mm"] / (df["Temperature_C"] + df["Wind_Speed_kmh"] + 1)
        df["prev_vs_rain"] = df["Previous_Irrigation_mm"] / (df["Rainfall_mm"] + 10)
        df["peak_stage_drought"] = drought_pressure * peak_stage.astype(np.int8)
        df["peak_stage_no_mulch_drought"] = drought_pressure * peak_no_mulch.astype(np.int8)
        df["river_peak_no_mulch_drought"] = drought_pressure * (
            peak_no_mulch & river_source
        ).astype(np.int8)
        df["canal_peak_no_mulch_drought"] = drought_pressure * (
            peak_no_mulch & canal_irrigation
        ).astype(np.int8)
        df["stress_count"] = (
            (df["Soil_Moisture"] <= 26).astype(np.int8)
            + (df["Temperature_C"] >= 30).astype(np.int8)
            + (df["Wind_Speed_kmh"] >= 12).astype(np.int8)
            + (df["Rainfall_mm"] <= 1000).astype(np.int8)
            + peak_no_mulch.astype(np.int8)
        )
        df["Soil_Moisture_bin"] = pd.cut(
            df["Soil_Moisture"],
            bins=[-np.inf, 14.0, 20.5, 26.5, 32.5, 40.0, 50.0, np.inf],
            labels=False,
        ).astype(int).astype(str)
        df["Temperature_C_bin"] = pd.cut(
            df["Temperature_C"],
            bins=[-np.inf, 21.0, 27.0, 30.0, 33.0, 36.0, 39.0, np.inf],
            labels=False,
        ).astype(int).astype(str)
        df["Wind_Speed_kmh_bin"] = pd.cut(
            df["Wind_Speed_kmh"],
            bins=[-np.inf, 4.5, 8.5, 10.5, 12.5, 14.5, 18.0, np.inf],
            labels=False,
        ).astype(int).astype(str)
        df["Rainfall_mm_bin"] = pd.cut(
            df["Rainfall_mm"],
            bins=[-np.inf, 650.0, 850.0, 1100.0, 1450.0, 1800.0, 2300.0, np.inf],
            labels=False,
        ).astype(int).astype(str)
        df["drought_pressure_bin"] = pd.cut(
            drought_pressure,
            bins=[-np.inf, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, np.inf],
            labels=False,
        ).astype(int).astype(str)
        df["stress_count_cat"] = df["stress_count"].astype(int).astype(str)
        df["peak_stage_bucket"] = np.where(
            peak_no_mulch,
            "peak_no_mulch",
            np.where(peak_stage, "peak", "other"),
        )
    if ARGS.categorical_crosses:
        df["Crop_Type__Season"] = df["Crop_Type"].astype(str) + "__" + df["Season"].astype(str)
        df["Irrigation_Type__Water_Source"] = (
            df["Irrigation_Type"].astype(str) + "__" + df["Water_Source"].astype(str)
        )
        df["Soil_Type__Region"] = df["Soil_Type"].astype(str) + "__" + df["Region"].astype(str)
        df["Crop_Growth_Stage__Mulching_Used"] = (
            df["Crop_Growth_Stage"].astype(str) + "__" + df["Mulching_Used"].astype(str)
        )
        df["Crop_Growth_Stage__Mulching_Used__Water_Source"] = (
            df["Crop_Growth_Stage"].astype(str)
            + "__"
            + df["Mulching_Used"].astype(str)
            + "__"
            + df["Water_Source"].astype(str)
        )
        df["Crop_Growth_Stage__Mulching_Used__Irrigation_Type"] = (
            df["Crop_Growth_Stage"].astype(str)
            + "__"
            + df["Mulching_Used"].astype(str)
            + "__"
            + df["Irrigation_Type"].astype(str)
        )
    return df


def print_class_diagnostics(y_true, y_pred, label_names):
    class_ids = np.arange(len(label_names))
    class_recalls = recall_score(
        y_true,
        y_pred,
        labels=class_ids,
        average=None,
        zero_division=0,
    )
    recall_summary = " ".join(
        f"{label}={recall:.6f}" for label, recall in zip(label_names, class_recalls)
    )
    weakest_idx = int(np.argmin(class_recalls))

    print(f"per_class_recall: {recall_summary}")
    print(
        "weakest_class_recall: "
        f"{label_names[weakest_idx]}={class_recalls[weakest_idx]:.6f}"
    )

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    for label, row in zip(label_names, cm):
        row_summary = ", ".join(
            f"{pred_label}:{int(count)}" for pred_label, count in zip(label_names, row)
        )
        print(f"confusion_matrix[{label}]: {row_summary}")


def blend_probabilities(prob_matrices, weights):
    blend = np.zeros_like(prob_matrices[0])
    for weight, matrix in zip(weights, prob_matrices):
        blend += weight * matrix
    return blend


def search_coarse_weights(prob_matrices, y_true):
    best_ba, best_w = 0.0, (1 / 3, 1 / 3, 1 / 3)
    for w1 in np.arange(0.1, 0.8, 0.05):
        for w2 in np.arange(0.1, 0.8, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < MIN_BLEND_WEIGHT:
                continue
            weights = (float(w1), float(w2), float(w3))
            blend = blend_probabilities(prob_matrices, weights)
            ba = balanced_accuracy_score(y_true, np.argmax(blend, axis=1))
            if ba > best_ba:
                best_ba, best_w = ba, weights
    return best_ba, best_w


def search_refined_weights(prob_matrices, y_true):
    coarse_ba, coarse_w = search_coarse_weights(prob_matrices, y_true)
    candidate_weights = {tuple(round(w, 6) for w in coarse_w)}

    center = np.array(coarse_w)
    for delta1 in np.arange(-0.08, 0.081, 0.01):
        for delta2 in np.arange(-0.08, 0.081, 0.01):
            weights = np.array([center[0] + delta1, center[1] + delta2, 0.0], dtype=float)
            weights[2] = 1.0 - weights[0] - weights[1]
            if np.any(weights < MIN_BLEND_WEIGHT) or np.any(weights > 0.90):
                continue
            candidate_weights.add(tuple(np.round(weights, 6)))

    rng = np.random.default_rng(SEED)
    random_weights = rng.dirichlet(np.ones(3), size=5000)
    for weights in random_weights:
        if weights.min() < MIN_BLEND_WEIGHT or weights.max() > 0.90:
            continue
        candidate_weights.add(tuple(np.round(weights, 6)))

    best_ba, best_w = coarse_ba, coarse_w
    for weights in sorted(candidate_weights):
        blend = blend_probabilities(prob_matrices, weights)
        ba = balanced_accuracy_score(y_true, np.argmax(blend, axis=1))
        if ba > best_ba:
            best_ba, best_w = ba, weights

    return best_ba, best_w, coarse_ba, coarse_w, len(candidate_weights)


def predict_with_class_scales(prob_matrix, class_scales):
    scaled = prob_matrix * np.asarray(class_scales, dtype=float)
    return np.argmax(scaled, axis=1)


def search_class_scales(prob_matrix, y_true):
    best_scales = (1.0, 1.0, 1.0)
    best_pred = np.argmax(prob_matrix, axis=1)
    best_ba = balanced_accuracy_score(y_true, best_pred)

    for medium_scale in np.arange(0.90, 1.101, 0.01):
        for high_scale in np.arange(0.70, 1.501, 0.01):
            class_scales = (1.0, float(round(medium_scale, 2)), float(round(high_scale, 2)))
            pred = predict_with_class_scales(prob_matrix, class_scales)
            ba = balanced_accuracy_score(y_true, pred)
            if ba > best_ba:
                best_ba = ba
                best_scales = class_scales
                best_pred = pred

    return best_ba, best_scales, best_pred


def format_class_scales(class_scales):
    return (
        f"Low:{class_scales[0]:.2f}  "
        f"Medium:{class_scales[1]:.2f}  "
        f"High:{class_scales[2]:.2f}"
    )


def print_class_scale_summary(prefix, class_scales):
    if any(abs(scale - 1.0) > 1e-12 for scale in class_scales):
        print(f"{prefix} — {format_class_scales(class_scales)}")


def evaluate_probability_policy(prob_matrix, y_true, apply_class_scale_search):
    pred = np.argmax(prob_matrix, axis=1)
    score = balanced_accuracy_score(y_true, pred)
    class_scales = (1.0, 1.0, 1.0)

    if apply_class_scale_search:
        score, class_scales, pred = search_class_scales(prob_matrix, y_true)

    return score, pred, class_scales


def build_model_scaled_blend(prob_matrices, weights, model_scales):
    blend = np.zeros_like(prob_matrices[0])
    for weight, matrix, scales in zip(weights, prob_matrices, model_scales):
        blend += weight * matrix * np.asarray(scales, dtype=float)
    return blend


def search_model_class_scales(prob_matrices, weights, y_true):
    medium_grid = np.arange(0.88, 1.121, 0.02)
    high_grid = np.arange(0.80, 1.601, 0.02)
    weighted = [weight * matrix for weight, matrix in zip(weights, prob_matrices)]
    model_scales = [np.ones(prob_matrices[0].shape[1], dtype=float) for _ in prob_matrices]
    blend = np.zeros_like(prob_matrices[0])
    for weighted_matrix in weighted:
        blend += weighted_matrix

    best_pred = np.argmax(blend, axis=1)
    best_ba = balanced_accuracy_score(y_true, best_pred)

    for _ in range(3):
        improved = False
        for model_idx, weighted_matrix in enumerate(weighted):
            current_scales = model_scales[model_idx].copy()
            base_without_model = blend - weighted_matrix * current_scales

            best_local_scales = current_scales.copy()
            best_local_blend = blend

            for medium_scale in medium_grid:
                candidate_scales = current_scales.copy()
                candidate_scales[1] = float(round(medium_scale, 2))
                candidate_blend = base_without_model + weighted_matrix * candidate_scales
                candidate_pred = np.argmax(candidate_blend, axis=1)
                candidate_ba = balanced_accuracy_score(y_true, candidate_pred)
                if candidate_ba > best_ba + 1e-12:
                    best_ba = candidate_ba
                    best_pred = candidate_pred
                    best_local_scales = candidate_scales
                    best_local_blend = candidate_blend
                    improved = True

            current_scales = best_local_scales.copy()
            blend = best_local_blend
            base_without_model = blend - weighted_matrix * current_scales

            for high_scale in high_grid:
                candidate_scales = current_scales.copy()
                candidate_scales[2] = float(round(high_scale, 2))
                candidate_blend = base_without_model + weighted_matrix * candidate_scales
                candidate_pred = np.argmax(candidate_blend, axis=1)
                candidate_ba = balanced_accuracy_score(y_true, candidate_pred)
                if candidate_ba > best_ba + 1e-12:
                    best_ba = candidate_ba
                    best_pred = candidate_pred
                    best_local_scales = candidate_scales
                    best_local_blend = candidate_blend
                    improved = True

            model_scales[model_idx] = best_local_scales
            blend = best_local_blend

        if not improved:
            break

    class_ba, class_scales, class_pred = search_class_scales(blend, y_true)
    if class_ba > best_ba + 1e-12:
        best_ba = class_ba
        best_pred = class_pred
    else:
        class_scales = (1.0, 1.0, 1.0)

    return best_ba, model_scales, class_scales, best_pred


def predict_with_ordinal_thresholds(score_vector, thresholds):
    lower, upper = thresholds
    pred = np.zeros(len(score_vector), dtype=int)
    pred[score_vector >= lower] = 1
    pred[score_vector >= upper] = 2
    return pred


def search_ordinal_thresholds(score_vector, y_true):
    best_thresholds = (0.85, 1.20)
    best_pred = predict_with_ordinal_thresholds(score_vector, best_thresholds)
    best_ba = balanced_accuracy_score(y_true, best_pred)

    coarse_lower = np.arange(0.50, 1.01, 0.05)
    coarse_upper = np.arange(1.00, 1.71, 0.05)

    for lower in coarse_lower:
        for upper in coarse_upper:
            if upper <= lower + 0.10:
                continue
            thresholds = (float(round(lower, 3)), float(round(upper, 3)))
            pred = predict_with_ordinal_thresholds(score_vector, thresholds)
            ba = balanced_accuracy_score(y_true, pred)
            if ba > best_ba:
                best_ba = ba
                best_thresholds = thresholds
                best_pred = pred

    center_lower, center_upper = best_thresholds
    for lower in np.arange(center_lower - 0.08, center_lower + 0.081, 0.01):
        for upper in np.arange(center_upper - 0.08, center_upper + 0.081, 0.01):
            if upper <= lower + 0.05:
                continue
            thresholds = (float(round(lower, 3)), float(round(upper, 3)))
            pred = predict_with_ordinal_thresholds(score_vector, thresholds)
            ba = balanced_accuracy_score(y_true, pred)
            if ba > best_ba:
                best_ba = ba
                best_thresholds = thresholds
                best_pred = pred

    return best_ba, best_thresholds, best_pred


def resolve_prediction_cache_path(cache_path):
    if not cache_path:
        return ""
    if os.path.isabs(cache_path):
        return cache_path
    return os.path.join(_project_root, cache_path)


def load_prediction_cache(cache_path):
    cache = np.load(cache_path)
    return {key: cache[key] for key in cache.files}


def save_prediction_cache(cache_path, **payload):
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, **payload)


def stack_probabilities(prob_matrices):
    stacked_parts = []
    class_axis = np.arange(prob_matrices[0].shape[1], dtype=float)

    for matrix in prob_matrices:
        stacked_parts.append(matrix)
        stacked_parts.append((matrix @ class_axis).reshape(-1, 1))
        stacked_parts.append((matrix[:, 2] - matrix[:, 0]).reshape(-1, 1))
        stacked_parts.append(np.max(matrix, axis=1, keepdims=True))

    return np.hstack(stacked_parts)


def build_stack_feature_matrices(
    prob_matrices_oof,
    prob_matrices_test,
    *,
    meta_raw_train=None,
    meta_raw_test=None,
    meta_full_train=None,
    meta_full_test=None,
):
    stack_train = stack_probabilities(prob_matrices_oof)
    stack_test = stack_probabilities(prob_matrices_test)

    extra_train_parts = []
    extra_test_parts = []
    if meta_raw_train is not None:
        extra_train_parts.append(meta_raw_train)
        extra_test_parts.append(meta_raw_test)
    if meta_full_train is not None:
        extra_train_parts.append(meta_full_train)
        extra_test_parts.append(meta_full_test)

    if extra_train_parts:
        stack_train = np.hstack([stack_train, *extra_train_parts])
        stack_test = np.hstack([stack_test, *extra_test_parts])

    return stack_train, stack_test


def build_meta_raw_features(df):
    peak_stage = df["Crop_Growth_Stage"].isin(["Vegetative", "Flowering"]).astype(np.int8)
    no_mulch = df["Mulching_Used"].eq("No").astype(np.int8)
    river_source = df["Water_Source"].eq("River").astype(np.int8)
    canal_irrigation = df["Irrigation_Type"].eq("Canal").astype(np.int8)
    moisture_deficit = 100 - df["Soil_Moisture"]
    drought_pressure = (
        moisture_deficit * (df["Temperature_C"] + df["Wind_Speed_kmh"])
        / (df["Rainfall_mm"] + 10)
    )
    stress_count = (
        (df["Soil_Moisture"] <= 26).astype(np.int8)
        + (df["Temperature_C"] >= 30).astype(np.int8)
        + (df["Wind_Speed_kmh"] >= 12).astype(np.int8)
        + (df["Rainfall_mm"] <= 1000).astype(np.int8)
        + (peak_stage & no_mulch).astype(np.int8)
    )
    meta_frame = pd.DataFrame(
        {
            "Soil_Moisture": df["Soil_Moisture"],
            "Temperature_C": df["Temperature_C"],
            "Wind_Speed_kmh": df["Wind_Speed_kmh"],
            "Rainfall_mm": df["Rainfall_mm"],
            "Previous_Irrigation_mm": df["Previous_Irrigation_mm"],
            "Field_Area_hectare": df["Field_Area_hectare"],
            "water_stress": df["water_stress"],
            "effective_rain": df["effective_rain"],
            "aridity": df["aridity"],
            "moisture_deficit": moisture_deficit,
            "drought_pressure": drought_pressure,
            "stress_count": stress_count,
            "is_peak_stage": peak_stage,
            "is_no_mulch": no_mulch,
            "is_river_source": river_source,
            "is_canal_irrigation": canal_irrigation,
            "is_peak_no_mulch": (peak_stage & no_mulch).astype(np.int8),
        }
    )
    return meta_frame.values.astype(float)


def build_meta_full_features(train_df, test_df, numeric_cols, categorical_cols):
    train_num = train_df[numeric_cols].to_numpy(dtype=np.float32, copy=True)
    test_num = test_df[numeric_cols].to_numpy(dtype=np.float32, copy=True)

    if not categorical_cols:
        return train_num, test_num

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    train_cat = encoder.fit_transform(train_df[categorical_cols].astype(str))
    test_cat = encoder.transform(test_df[categorical_cols].astype(str))

    train_meta = np.hstack([train_num, train_cat]).astype(np.float32, copy=False)
    test_meta = np.hstack([test_num, test_cat]).astype(np.float32, copy=False)
    return train_meta, test_meta


def add_frequency_encoding(train_df, test_df, columns):
    feature_names = []
    for col in columns:
        feature_name = f"{col}__freq"
        frequencies = train_df[col].value_counts(normalize=True, dropna=False)
        train_df[feature_name] = train_df[col].map(frequencies).fillna(0.0).astype(float)
        test_df[feature_name] = test_df[col].map(frequencies).fillna(0.0).astype(float)
        feature_names.append(feature_name)
    return feature_names


def add_target_encoding(
    train_df,
    test_df,
    y,
    split_indices,
    columns,
    label_names,
    smoothing,
):
    feature_names = []
    priors = np.bincount(y, minlength=len(label_names)).astype(float) / len(y)

    for col in columns:
        train_series = train_df[col].astype(str)
        test_series = test_df[col].astype(str)

        for class_idx, label_name in enumerate(label_names):
            feature_name = f"{col}__te_{label_name.lower()}"
            train_feature = np.full(len(train_df), priors[class_idx], dtype=float)

            for tr_idx, val_idx in split_indices:
                fold_series = train_series.iloc[tr_idx]
                fold_targets = pd.Series(
                    (y[tr_idx] == class_idx).astype(float),
                    index=fold_series.index,
                )
                fold_counts = fold_series.value_counts(dropna=False)
                fold_positive = fold_targets.groupby(fold_series).sum()
                mapping = (
                    fold_positive.add(smoothing * priors[class_idx], fill_value=smoothing * priors[class_idx])
                    / (fold_counts + smoothing)
                )
                train_feature[val_idx] = (
                    train_series.iloc[val_idx].map(mapping).fillna(priors[class_idx]).to_numpy()
                )

            full_targets = pd.Series(
                (y == class_idx).astype(float),
                index=train_series.index,
            )
            full_counts = train_series.value_counts(dropna=False)
            full_positive = full_targets.groupby(train_series).sum()
            full_mapping = (
                full_positive.add(smoothing * priors[class_idx], fill_value=smoothing * priors[class_idx])
                / (full_counts + smoothing)
            )

            train_df[feature_name] = train_feature
            test_df[feature_name] = (
                test_series.map(full_mapping).fillna(priors[class_idx]).astype(float)
            )
            feature_names.append(feature_name)

    return feature_names


def run_logreg_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    apply_class_scale_search=False,
):
    meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    candidate_cs = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    candidate_weights = [None, "balanced"]

    best_score = -1.0
    best_pred = None
    best_config = None
    best_class_scales = (1.0, 1.0, 1.0)

    for class_weight in candidate_weights:
        for c_value in candidate_cs:
            oof_proba = np.zeros((len(y_true), len(LABEL_ORDER)), dtype=np.float32)
            for tr_idx, val_idx in meta_skf.split(train_features, y_true):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(train_features[tr_idx])
                X_val = scaler.transform(train_features[val_idx])
                model = LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=2000,
                    n_jobs=STACKER_N_JOBS,
                    solver="lbfgs",
                )
                model.fit(
                    X_tr,
                    y_true[tr_idx],
                    sample_weight=sample_weights[tr_idx],
                )
                oof_proba[val_idx] = model.predict_proba(X_val)

            score, oof_pred, class_scales = evaluate_probability_policy(
                oof_proba,
                y_true,
                apply_class_scale_search,
            )
            if score > best_score:
                best_score = score
                best_pred = oof_pred
                best_config = {"C": c_value, "class_weight": class_weight}
                best_class_scales = class_scales

    print(
        "Best logreg stack config — "
        f"C:{best_config['C']:.2f} "
        f"class_weight:{best_config['class_weight'] or 'none'}"
    )
    print_class_scale_summary("Best logreg stack class scales", best_class_scales)

    full_model = LogisticRegression(
        C=best_config["C"],
        class_weight=best_config["class_weight"],
        max_iter=2000,
        n_jobs=STACKER_N_JOBS,
        solver="lbfgs",
    )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    full_model.fit(train_scaled, y_true, sample_weight=sample_weights)
    test_proba = full_model.predict_proba(test_scaled)
    final_test_pred = predict_with_class_scales(test_proba, best_class_scales)

    return best_score, best_pred, final_test_pred


def run_mlp_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    apply_class_scale_search=False,
):
    meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    candidate_configs = [
        {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (128, 64), "alpha": 5e-4, "learning_rate_init": 1e-3},
    ]
    if train_features.shape[1] > 40:
        candidate_configs.extend(
            [
                {
                    "hidden_layer_sizes": (128, 64, 32),
                    "alpha": 3e-4,
                    "learning_rate_init": 7e-4,
                },
                {
                    "hidden_layer_sizes": (192, 96, 32),
                    "alpha": 5e-4,
                    "learning_rate_init": 5e-4,
                },
            ]
        )

    best_score = -1.0
    best_pred = None
    best_config = None
    best_class_scales = (1.0, 1.0, 1.0)

    for config in candidate_configs:
        oof_proba = np.zeros((len(y_true), len(LABEL_ORDER)), dtype=np.float32)
        for tr_idx, val_idx in meta_skf.split(train_features, y_true):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train_features[tr_idx])
            X_val = scaler.transform(train_features[val_idx])
            model = MLPClassifier(
                hidden_layer_sizes=config["hidden_layer_sizes"],
                alpha=config["alpha"],
                learning_rate_init=config["learning_rate_init"],
                batch_size=4096,
                early_stopping=True,
                max_iter=80,
                n_iter_no_change=10,
                random_state=SEED,
            )
            model.fit(X_tr, y_true[tr_idx], sample_weight=sample_weights[tr_idx])
            oof_proba[val_idx] = model.predict_proba(X_val)

        score, oof_pred, class_scales = evaluate_probability_policy(
            oof_proba,
            y_true,
            apply_class_scale_search,
        )
        if score > best_score:
            best_score = score
            best_pred = oof_pred
            best_config = config
            best_class_scales = class_scales

    print(
        "Best mlp stack config — "
        f"hidden:{best_config['hidden_layer_sizes']} "
        f"alpha:{best_config['alpha']:.5f} "
        f"lr:{best_config['learning_rate_init']:.5f}"
    )
    print_class_scale_summary("Best mlp stack class scales", best_class_scales)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    full_model = MLPClassifier(
        hidden_layer_sizes=best_config["hidden_layer_sizes"],
        alpha=best_config["alpha"],
        learning_rate_init=best_config["learning_rate_init"],
        batch_size=4096,
        early_stopping=True,
        max_iter=80,
        n_iter_no_change=10,
        random_state=SEED,
    )
    full_model.fit(train_scaled, y_true, sample_weight=sample_weights)
    test_proba = full_model.predict_proba(test_scaled)
    final_test_pred = predict_with_class_scales(test_proba, best_class_scales)

    return best_score, best_pred, final_test_pred


def run_mlp_bag_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    apply_class_scale_search=False,
):
    meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    candidate_configs = [
        {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (128, 64), "alpha": 5e-4, "learning_rate_init": 1e-3},
    ]
    if train_features.shape[1] > 40:
        candidate_configs.extend(
            [
                {
                    "hidden_layer_sizes": (128, 64, 32),
                    "alpha": 3e-4,
                    "learning_rate_init": 7e-4,
                },
                {
                    "hidden_layer_sizes": (192, 96, 32),
                    "alpha": 5e-4,
                    "learning_rate_init": 5e-4,
                },
            ]
        )
    bag_seeds = [SEED, SEED + 17, SEED + 41]

    best_score = -1.0
    best_pred = None
    best_config = None
    best_class_scales = (1.0, 1.0, 1.0)

    for config in candidate_configs:
        oof_proba = np.zeros((len(y_true), len(LABEL_ORDER)), dtype=np.float32)
        for tr_idx, val_idx in meta_skf.split(train_features, y_true):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train_features[tr_idx])
            X_val = scaler.transform(train_features[val_idx])
            fold_proba = np.zeros((len(val_idx), len(LABEL_ORDER)), dtype=np.float32)

            for random_seed in bag_seeds:
                model = MLPClassifier(
                    hidden_layer_sizes=config["hidden_layer_sizes"],
                    alpha=config["alpha"],
                    learning_rate_init=config["learning_rate_init"],
                    batch_size=4096,
                    early_stopping=True,
                    max_iter=80,
                    n_iter_no_change=10,
                    random_state=random_seed,
                )
                model.fit(X_tr, y_true[tr_idx], sample_weight=sample_weights[tr_idx])
                fold_proba += model.predict_proba(X_val) / len(bag_seeds)

            oof_proba[val_idx] = fold_proba

        score, oof_pred, class_scales = evaluate_probability_policy(
            oof_proba,
            y_true,
            apply_class_scale_search,
        )
        if score > best_score:
            best_score = score
            best_pred = oof_pred
            best_config = config
            best_class_scales = class_scales

    print(
        "Best mlp bag stack config — "
        f"hidden:{best_config['hidden_layer_sizes']} "
        f"alpha:{best_config['alpha']:.5f} "
        f"lr:{best_config['learning_rate_init']:.5f} "
        f"seeds:{bag_seeds}"
    )
    print_class_scale_summary("Best mlp bag stack class scales", best_class_scales)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    test_proba = np.zeros((len(test_features), len(LABEL_ORDER)), dtype=np.float32)

    for random_seed in bag_seeds:
        full_model = MLPClassifier(
            hidden_layer_sizes=best_config["hidden_layer_sizes"],
            alpha=best_config["alpha"],
            learning_rate_init=best_config["learning_rate_init"],
            batch_size=4096,
            early_stopping=True,
            max_iter=80,
            n_iter_no_change=10,
            random_state=random_seed,
        )
        full_model.fit(train_scaled, y_true, sample_weight=sample_weights)
        test_proba += full_model.predict_proba(test_scaled) / len(bag_seeds)

    final_test_pred = predict_with_class_scales(test_proba, best_class_scales)
    return best_score, best_pred, final_test_pred


def collect_mlp_oof_probabilities(
    train_features,
    y_true,
    sample_weights,
    config,
    *,
    random_seeds,
):
    meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_proba = np.zeros((len(y_true), len(LABEL_ORDER)), dtype=np.float32)

    for tr_idx, val_idx in meta_skf.split(train_features, y_true):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_features[tr_idx])
        X_val = scaler.transform(train_features[val_idx])
        fold_proba = np.zeros((len(val_idx), len(LABEL_ORDER)), dtype=np.float32)

        for random_seed in random_seeds:
            model = MLPClassifier(
                hidden_layer_sizes=config["hidden_layer_sizes"],
                alpha=config["alpha"],
                learning_rate_init=config["learning_rate_init"],
                batch_size=4096,
                early_stopping=True,
                max_iter=80,
                n_iter_no_change=10,
                random_state=random_seed,
            )
            model.fit(X_tr, y_true[tr_idx], sample_weight=sample_weights[tr_idx])
            fold_proba += model.predict_proba(X_val) / len(random_seeds)

        oof_proba[val_idx] = fold_proba

    return oof_proba


def train_mlp_test_probabilities(
    train_features,
    y_true,
    test_features,
    sample_weights,
    config,
    *,
    random_seeds,
):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    test_proba = np.zeros((len(test_features), len(LABEL_ORDER)), dtype=np.float32)

    for random_seed in random_seeds:
        full_model = MLPClassifier(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            alpha=config["alpha"],
            learning_rate_init=config["learning_rate_init"],
            batch_size=4096,
            early_stopping=True,
            max_iter=80,
            n_iter_no_change=10,
            random_state=random_seed,
        )
        full_model.fit(train_scaled, y_true, sample_weight=sample_weights)
        test_proba += full_model.predict_proba(test_scaled) / len(random_seeds)

    return test_proba


def predict_torch_probabilities(model, features, batch_size=16384):
    device = next(model.parameters()).device
    model.eval()
    prob_batches = []

    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch_features = torch.from_numpy(
                features[start:start + batch_size].astype(np.float32, copy=False)
            ).to(device)
            logits = model(batch_features)
            prob_batches.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.vstack(prob_batches)


def fit_torch_model(
    model,
    train_features,
    train_labels,
    train_weights,
    val_features,
    val_labels,
    config,
):
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    train_dataset = TensorDataset(
        torch.from_numpy(train_features.astype(np.float32, copy=False)),
        torch.from_numpy(train_labels.astype(np.int64, copy=False)),
        torch.from_numpy(train_weights.astype(np.float32, copy=False)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    best_score = -1.0
    best_epoch = 0
    best_state = None
    patience = 0

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = (loss_fn(logits, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()

        val_proba = predict_torch_probabilities(model, val_features)
        score = balanced_accuracy_score(val_labels, np.argmax(val_proba, axis=1))

        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_score, best_epoch


def train_full_torch_model(
    model,
    train_features,
    train_labels,
    train_weights,
    config,
    epochs,
):
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    train_dataset = TensorDataset(
        torch.from_numpy(train_features.astype(np.float32, copy=False)),
        torch.from_numpy(train_labels.astype(np.int64, copy=False)),
        torch.from_numpy(train_weights.astype(np.float32, copy=False)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = (loss_fn(logits, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()

    return model


def collect_torch_oof_probabilities(
    train_features,
    y_true,
    sample_weights,
    config,
    model_builder,
):
    meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_proba = np.zeros((len(y_true), len(LABEL_ORDER)), dtype=np.float32)
    fold_epochs = []

    for tr_idx, val_idx in meta_skf.split(train_features, y_true):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_features[tr_idx]).astype(np.float32)
        X_val = scaler.transform(train_features[val_idx]).astype(np.float32)

        model, _, best_epoch = fit_torch_model(
            model_builder(X_tr.shape[1], config),
            X_tr,
            y_true[tr_idx],
            sample_weights[tr_idx],
            X_val,
            y_true[val_idx],
            config,
        )
        oof_proba[val_idx] = predict_torch_probabilities(model, X_val)
        fold_epochs.append(best_epoch)

    return oof_proba, fold_epochs


def train_torch_test_probabilities(
    train_features,
    y_true,
    test_features,
    sample_weights,
    config,
    model_builder,
    fold_epochs,
):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features).astype(np.float32)
    test_scaled = scaler.transform(test_features).astype(np.float32)
    full_epochs = max(1, int(round(np.mean(fold_epochs))))
    full_model = train_full_torch_model(
        model_builder(train_scaled.shape[1], config),
        train_scaled,
        y_true,
        sample_weights,
        config,
        full_epochs,
    )
    return predict_torch_probabilities(full_model, test_scaled)


def run_torch_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    *,
    candidate_configs,
    model_builder,
    config_formatter,
    label,
    apply_class_scale_search=False,
):
    best_score = -1.0
    best_pred = None
    best_config = None
    best_class_scales = (1.0, 1.0, 1.0)
    best_epochs = []

    for config in candidate_configs:
        oof_proba, fold_epochs = collect_torch_oof_probabilities(
            train_features,
            y_true,
            sample_weights,
            config,
            model_builder,
        )

        score, oof_pred, class_scales = evaluate_probability_policy(
            oof_proba,
            y_true,
            apply_class_scale_search,
        )
        if score > best_score:
            best_score = score
            best_pred = oof_pred
            best_config = config
            best_class_scales = class_scales
            best_epochs = fold_epochs

    print(f"Best {label} config — {config_formatter(best_config)}")
    print_class_scale_summary(f"Best {label} stack class scales", best_class_scales)

    test_proba = train_torch_test_probabilities(
        train_features,
        y_true,
        test_features,
        sample_weights,
        best_config,
        model_builder,
        best_epochs,
    )
    final_test_pred = predict_with_class_scales(test_proba, best_class_scales)

    return best_score, best_pred, final_test_pred


class FTTransformerStacker(nn.Module):
    def __init__(self, n_features, d_token, n_heads, n_layers, dropout):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, len(LABEL_ORDER)),
        )

    def forward(self, x):
        tokens = (
            x.unsqueeze(-1) * self.feature_weight.unsqueeze(0)
            + self.feature_bias.unsqueeze(0)
        )
        cls = self.cls_token.expand(x.size(0), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0])


class CNNStacker(nn.Module):
    def __init__(self, n_features, channels, kernel_size, dropout):
        super().__init__()
        padding = kernel_size // 2
        self.backbone = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(channels * 4, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, len(LABEL_ORDER)),
        )

    def forward(self, x):
        encoded = self.backbone(x.unsqueeze(1))
        return self.head(encoded)


class RNNStacker(nn.Module):
    def __init__(self, n_features, d_token, hidden_size, n_layers, dropout):
        super().__init__()
        self.feature_projection = nn.Linear(1, d_token)
        self.sequence_bias = nn.Parameter(torch.zeros(1, n_features, d_token))
        self.encoder = nn.GRU(
            input_size=d_token,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        pooled_width = hidden_size * 4
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, len(LABEL_ORDER)),
        )

    def forward(self, x):
        tokens = self.feature_projection(x.unsqueeze(-1)) + self.sequence_bias[:, : x.size(1)]
        encoded, _ = self.encoder(tokens)
        pooled = torch.cat([encoded.mean(dim=1), encoded.max(dim=1).values], dim=1)
        return self.head(pooled)


def run_ft_transformer_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    apply_class_scale_search=False,
):
    candidate_configs = [
        {
            "d_token": 16,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.10,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 4096,
            "max_epochs": 18,
            "patience": 4,
        },
        {
            "d_token": 24,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.10,
            "learning_rate": 2e-4,
            "weight_decay": 2e-4,
            "batch_size": 4096,
            "max_epochs": 20,
            "patience": 4,
        },
    ]
    return run_torch_stack(
        train_features,
        y_true,
        test_features,
        sample_weights,
        candidate_configs=candidate_configs,
        model_builder=lambda n_features, config: FTTransformerStacker(
            n_features=n_features,
            d_token=config["d_token"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
        ),
        config_formatter=lambda config: (
            f"d_token:{config['d_token']} "
            f"layers:{config['n_layers']} "
            f"heads:{config['n_heads']} "
            f"dropout:{config['dropout']:.2f} "
            f"lr:{config['learning_rate']:.5f}"
        ),
        label="ft-transformer",
        apply_class_scale_search=apply_class_scale_search,
    )


def run_cnn_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    apply_class_scale_search=False,
):
    candidate_configs = [
        {
            "channels": 32,
            "kernel_size": 3,
            "dropout": 0.10,
            "learning_rate": 4e-4,
            "weight_decay": 1e-4,
            "batch_size": 4096,
            "max_epochs": 18,
            "patience": 4,
        },
        {
            "channels": 48,
            "kernel_size": 5,
            "dropout": 0.15,
            "learning_rate": 3e-4,
            "weight_decay": 2e-4,
            "batch_size": 4096,
            "max_epochs": 20,
            "patience": 4,
        },
    ]
    return run_torch_stack(
        train_features,
        y_true,
        test_features,
        sample_weights,
        candidate_configs=candidate_configs,
        model_builder=lambda n_features, config: CNNStacker(
            n_features=n_features,
            channels=config["channels"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        ),
        config_formatter=lambda config: (
            f"channels:{config['channels']} "
            f"kernel:{config['kernel_size']} "
            f"dropout:{config['dropout']:.2f} "
            f"lr:{config['learning_rate']:.5f}"
        ),
        label="cnn",
        apply_class_scale_search=apply_class_scale_search,
    )


def run_rnn_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    apply_class_scale_search=False,
):
    candidate_configs = [
        {
            "d_token": 16,
            "hidden_size": 48,
            "n_layers": 1,
            "dropout": 0.10,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "batch_size": 4096,
            "max_epochs": 18,
            "patience": 4,
        },
        {
            "d_token": 24,
            "hidden_size": 64,
            "n_layers": 2,
            "dropout": 0.15,
            "learning_rate": 4e-4,
            "weight_decay": 2e-4,
            "batch_size": 4096,
            "max_epochs": 20,
            "patience": 4,
        },
    ]
    return run_torch_stack(
        train_features,
        y_true,
        test_features,
        sample_weights,
        candidate_configs=candidate_configs,
        model_builder=lambda n_features, config: RNNStacker(
            n_features=n_features,
            d_token=config["d_token"],
            hidden_size=config["hidden_size"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
        ),
        config_formatter=lambda config: (
            f"d_token:{config['d_token']} "
            f"hidden:{config['hidden_size']} "
            f"layers:{config['n_layers']} "
            f"dropout:{config['dropout']:.2f} "
            f"lr:{config['learning_rate']:.5f}"
        ),
        label="rnn",
        apply_class_scale_search=apply_class_scale_search,
    )


def build_tabnet_classifier(input_dim, config, seed):
    return TabNetClassifier(
        input_dim=input_dim,
        output_dim=len(LABEL_ORDER),
        n_d=config["n_d"],
        n_a=config["n_d"],
        n_steps=config["n_steps"],
        gamma=config["gamma"],
        lambda_sparse=config["lambda_sparse"],
        n_independent=2,
        n_shared=2,
        seed=seed,
        verbose=0,
        device_name="cpu",
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": config["learning_rate"]},
        mask_type=config.get("mask_type", "sparsemax"),
    )


def collect_tabnet_oof_probabilities(train_features, y_true, sample_weights, config):
    meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_proba = np.zeros((len(y_true), len(LABEL_ORDER)), dtype=np.float32)
    fold_epochs = []

    for fold_idx, (tr_idx, val_idx) in enumerate(meta_skf.split(train_features, y_true)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_features[tr_idx]).astype(np.float32)
        X_val = scaler.transform(train_features[val_idx]).astype(np.float32)
        model = build_tabnet_classifier(X_tr.shape[1], config, SEED + fold_idx)
        model.fit(
            X_tr,
            y_true[tr_idx],
            eval_set=[(X_val, y_true[val_idx])],
            eval_name=["val"],
            eval_metric=["logloss"],
            weights=sample_weights[tr_idx].astype(np.float32),
            max_epochs=config["max_epochs"],
            patience=config["patience"],
            batch_size=config["batch_size"],
            virtual_batch_size=config["virtual_batch_size"],
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )
        oof_proba[val_idx] = model.predict_proba(X_val).astype(np.float32)
        fold_epochs.append(int(getattr(model, "best_epoch", config["max_epochs"])))

    return oof_proba, fold_epochs


def train_tabnet_test_probabilities(
    train_features,
    y_true,
    test_features,
    sample_weights,
    config,
    fold_epochs,
):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features).astype(np.float32)
    test_scaled = scaler.transform(test_features).astype(np.float32)
    full_epochs = max(5, int(round(np.mean(fold_epochs))))
    model = build_tabnet_classifier(train_scaled.shape[1], config, SEED)
    model.fit(
        train_scaled,
        y_true,
        weights=sample_weights.astype(np.float32),
        max_epochs=full_epochs,
        batch_size=config["batch_size"],
        virtual_batch_size=config["virtual_batch_size"],
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    return model.predict_proba(test_scaled).astype(np.float32)


def run_tabnet_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    apply_class_scale_search=False,
):
    candidate_configs = [
        {
            "n_d": 16,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 1e-4,
            "learning_rate": 2e-2,
            "batch_size": 8192,
            "virtual_batch_size": 1024,
            "max_epochs": 25,
            "patience": 5,
            "mask_type": "sparsemax",
        },
        {
            "n_d": 24,
            "n_steps": 5,
            "gamma": 1.5,
            "lambda_sparse": 5e-5,
            "learning_rate": 1.5e-2,
            "batch_size": 8192,
            "virtual_batch_size": 1024,
            "max_epochs": 30,
            "patience": 6,
            "mask_type": "entmax",
        },
    ]

    best_score = -1.0
    best_pred = None
    best_config = None
    best_class_scales = (1.0, 1.0, 1.0)
    best_epochs = []

    for config in candidate_configs:
        oof_proba, fold_epochs = collect_tabnet_oof_probabilities(
            train_features,
            y_true,
            sample_weights,
            config,
        )
        score, oof_pred, class_scales = evaluate_probability_policy(
            oof_proba,
            y_true,
            apply_class_scale_search,
        )
        if score > best_score:
            best_score = score
            best_pred = oof_pred
            best_config = config
            best_class_scales = class_scales
            best_epochs = fold_epochs

    print(
        "Best tabnet stack config — "
        f"n_d:{best_config['n_d']} "
        f"steps:{best_config['n_steps']} "
        f"gamma:{best_config['gamma']:.2f} "
        f"mask:{best_config['mask_type']} "
        f"lr:{best_config['learning_rate']:.4f}"
    )
    print_class_scale_summary("Best tabnet stack class scales", best_class_scales)

    test_proba = train_tabnet_test_probabilities(
        train_features,
        y_true,
        test_features,
        sample_weights,
        best_config,
        best_epochs,
    )
    final_test_pred = predict_with_class_scales(test_proba, best_class_scales)
    return best_score, best_pred, final_test_pred


def search_two_model_blend(prob_a, prob_b, y_true, apply_class_scale_search):
    best_score = -1.0
    best_weight = 0.50
    best_pred = None
    best_class_scales = (1.0, 1.0, 1.0)

    for weight in np.arange(0.20, 0.81, 0.05):
        blended = weight * prob_a + (1.0 - weight) * prob_b
        score, pred, class_scales = evaluate_probability_policy(
            blended,
            y_true,
            apply_class_scale_search,
        )
        if score > best_score:
            best_score = score
            best_weight = float(round(weight, 3))
            best_pred = pred
            best_class_scales = class_scales

    for weight in np.arange(best_weight - 0.08, best_weight + 0.081, 0.01):
        if weight <= 0.05 or weight >= 0.95:
            continue
        blended = weight * prob_a + (1.0 - weight) * prob_b
        score, pred, class_scales = evaluate_probability_policy(
            blended,
            y_true,
            apply_class_scale_search,
        )
        if score > best_score:
            best_score = score
            best_weight = float(round(weight, 3))
            best_pred = pred
            best_class_scales = class_scales

    return best_score, best_weight, best_pred, best_class_scales


def run_ann_cnn_combo_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    apply_class_scale_search=False,
):
    mlp_candidates = [
        {"hidden_layer_sizes": (128, 64), "alpha": 5e-4, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (192, 96, 32), "alpha": 5e-4, "learning_rate_init": 5e-4},
    ]
    cnn_candidates = [
        {
            "channels": 32,
            "kernel_size": 3,
            "dropout": 0.10,
            "learning_rate": 4e-4,
            "weight_decay": 1e-4,
            "batch_size": 4096,
            "max_epochs": 18,
            "patience": 4,
        },
        {
            "channels": 48,
            "kernel_size": 5,
            "dropout": 0.15,
            "learning_rate": 3e-4,
            "weight_decay": 2e-4,
            "batch_size": 4096,
            "max_epochs": 20,
            "patience": 4,
        },
    ]

    ann_bank = []
    for config in mlp_candidates:
        ann_bank.append(
            {
                "config": config,
                "oof": collect_mlp_oof_probabilities(
                    train_features,
                    y_true,
                    sample_weights,
                    config,
                    random_seeds=[SEED],
                ),
            }
        )

    cnn_bank = []
    for config in cnn_candidates:
        oof_proba, fold_epochs = collect_torch_oof_probabilities(
            train_features,
            y_true,
            sample_weights,
            config,
            lambda n_features, cfg: CNNStacker(
                n_features=n_features,
                channels=cfg["channels"],
                kernel_size=cfg["kernel_size"],
                dropout=cfg["dropout"],
            ),
        )
        cnn_bank.append({"config": config, "oof": oof_proba, "fold_epochs": fold_epochs})

    best_score = -1.0
    best_pred = None
    best_class_scales = (1.0, 1.0, 1.0)
    best_ann = None
    best_cnn = None
    best_weight = 0.50

    for ann_item in ann_bank:
        for cnn_item in cnn_bank:
            score, weight, pred, class_scales = search_two_model_blend(
                ann_item["oof"],
                cnn_item["oof"],
                y_true,
                apply_class_scale_search,
            )
            if score > best_score:
                best_score = score
                best_weight = weight
                best_pred = pred
                best_class_scales = class_scales
                best_ann = ann_item
                best_cnn = cnn_item

    print(
        "Best ann+cnn combo ANN config — "
        f"hidden:{best_ann['config']['hidden_layer_sizes']} "
        f"alpha:{best_ann['config']['alpha']:.5f} "
        f"lr:{best_ann['config']['learning_rate_init']:.5f}"
    )
    print(
        "Best ann+cnn combo CNN config — "
        f"channels:{best_cnn['config']['channels']} "
        f"kernel:{best_cnn['config']['kernel_size']} "
        f"dropout:{best_cnn['config']['dropout']:.2f} "
        f"lr:{best_cnn['config']['learning_rate']:.5f}"
    )
    print(
        f"Best ann+cnn blend weight — ANN:{best_weight:.2f}  CNN:{1.0 - best_weight:.2f}"
    )
    print_class_scale_summary("Best ann+cnn combo class scales", best_class_scales)

    ann_test = train_mlp_test_probabilities(
        train_features,
        y_true,
        test_features,
        sample_weights,
        best_ann["config"],
        random_seeds=[SEED],
    )
    cnn_test = train_torch_test_probabilities(
        train_features,
        y_true,
        test_features,
        sample_weights,
        best_cnn["config"],
        lambda n_features, cfg: CNNStacker(
            n_features=n_features,
            channels=cfg["channels"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
        ),
        best_cnn["fold_epochs"],
    )
    test_proba = best_weight * ann_test + (1.0 - best_weight) * cnn_test
    final_test_pred = predict_with_class_scales(test_proba, best_class_scales)
    return best_score, best_pred, final_test_pred


def search_base_neural_combo(
    base_prob_matrices,
    ann_prob,
    cnn_prob,
    base_weights,
    y_true,
    apply_class_scale_search,
):
    all_probabilities = list(base_prob_matrices) + [ann_prob, cnn_prob]
    best_score = -1.0
    best_pred = None
    best_class_scales = (1.0, 1.0, 1.0)
    best_combo = None

    for ann_weight in np.arange(0.00, 0.31, 0.05):
        for cnn_weight in np.arange(0.00, 0.31, 0.05):
            if ann_weight + cnn_weight >= 0.60:
                continue
            base_scale = 1.0 - ann_weight - cnn_weight
            weights = [base_scale * weight for weight in base_weights] + [ann_weight, cnn_weight]
            blended = blend_probabilities(all_probabilities, weights)
            score, pred, class_scales = evaluate_probability_policy(
                blended,
                y_true,
                apply_class_scale_search,
            )
            if score > best_score:
                best_score = score
                best_pred = pred
                best_class_scales = class_scales
                best_combo = weights

    for ann_weight in np.arange(best_combo[3] - 0.05, best_combo[3] + 0.051, 0.01):
        for cnn_weight in np.arange(best_combo[4] - 0.05, best_combo[4] + 0.051, 0.01):
            if ann_weight < 0.0 or cnn_weight < 0.0 or ann_weight + cnn_weight >= 0.70:
                continue
            base_scale = 1.0 - ann_weight - cnn_weight
            if base_scale <= 0.0:
                continue
            weights = [base_scale * weight for weight in base_weights] + [
                float(round(ann_weight, 3)),
                float(round(cnn_weight, 3)),
            ]
            blended = blend_probabilities(all_probabilities, weights)
            score, pred, class_scales = evaluate_probability_policy(
                blended,
                y_true,
                apply_class_scale_search,
            )
            if score > best_score:
                best_score = score
                best_pred = pred
                best_class_scales = class_scales
                best_combo = weights

    return best_score, best_combo, best_pred, best_class_scales


def run_neural_base_blend_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    prob_matrices_oof,
    prob_matrices_test,
    base_weights,
    apply_class_scale_search=False,
):
    ann_config = {
        "hidden_layer_sizes": (128, 64),
        "alpha": 5e-4,
        "learning_rate_init": 1e-3,
    }
    cnn_candidates = [
        {
            "channels": 32,
            "kernel_size": 3,
            "dropout": 0.10,
            "learning_rate": 4e-4,
            "weight_decay": 1e-4,
            "batch_size": 4096,
            "max_epochs": 18,
            "patience": 4,
        },
        {
            "channels": 48,
            "kernel_size": 5,
            "dropout": 0.15,
            "learning_rate": 3e-4,
            "weight_decay": 2e-4,
            "batch_size": 4096,
            "max_epochs": 20,
            "patience": 4,
        },
    ]

    ann_oof = collect_mlp_oof_probabilities(
        train_features,
        y_true,
        sample_weights,
        ann_config,
        random_seeds=[SEED],
    )

    best_score = -1.0
    best_pred = None
    best_class_scales = (1.0, 1.0, 1.0)
    best_combo = None
    best_cnn = None

    for cnn_config in cnn_candidates:
        cnn_oof, fold_epochs = collect_torch_oof_probabilities(
            train_features,
            y_true,
            sample_weights,
            cnn_config,
            lambda n_features, cfg: CNNStacker(
                n_features=n_features,
                channels=cfg["channels"],
                kernel_size=cfg["kernel_size"],
                dropout=cfg["dropout"],
            ),
        )
        score, combo, pred, class_scales = search_base_neural_combo(
            prob_matrices_oof,
            ann_oof,
            cnn_oof,
            base_weights,
            y_true,
            apply_class_scale_search,
        )
        if score > best_score:
            best_score = score
            best_pred = pred
            best_class_scales = class_scales
            best_combo = combo
            best_cnn = {"config": cnn_config, "fold_epochs": fold_epochs}

    print(
        "Best neural-base ANN config — "
        f"hidden:{ann_config['hidden_layer_sizes']} "
        f"alpha:{ann_config['alpha']:.5f} "
        f"lr:{ann_config['learning_rate_init']:.5f}"
    )
    print(
        "Best neural-base CNN config — "
        f"channels:{best_cnn['config']['channels']} "
        f"kernel:{best_cnn['config']['kernel_size']} "
        f"dropout:{best_cnn['config']['dropout']:.2f} "
        f"lr:{best_cnn['config']['learning_rate']:.5f}"
    )
    print(
        "Best neural-base blend weights — "
        f"LGB:{best_combo[0]:.2f}  XGB:{best_combo[1]:.2f}  CAT:{best_combo[2]:.2f}  "
        f"ANN:{best_combo[3]:.2f}  CNN:{best_combo[4]:.2f}"
    )
    print_class_scale_summary("Best neural-base blend class scales", best_class_scales)

    ann_test = train_mlp_test_probabilities(
        train_features,
        y_true,
        test_features,
        sample_weights,
        ann_config,
        random_seeds=[SEED],
    )
    cnn_test = train_torch_test_probabilities(
        train_features,
        y_true,
        test_features,
        sample_weights,
        best_cnn["config"],
        lambda n_features, cfg: CNNStacker(
            n_features=n_features,
            channels=cfg["channels"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
        ),
        best_cnn["fold_epochs"],
    )
    test_blend = blend_probabilities(
        list(prob_matrices_test) + [ann_test, cnn_test],
        best_combo,
    )
    final_test_pred = predict_with_class_scales(test_blend, best_class_scales)
    return best_score, best_pred, final_test_pred


def run_xgb_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    apply_class_scale_search=False,
):
    meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    candidate_configs = [
        {
            "max_depth": 4,
            "min_child_weight": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
        },
        {
            "max_depth": 5,
            "min_child_weight": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
        },
    ]

    best_score = -1.0
    best_pred = None
    best_config = None
    best_class_scales = (1.0, 1.0, 1.0)

    for config in candidate_configs:
        oof_proba = np.zeros((len(y_true), len(LABEL_ORDER)), dtype=np.float32)
        for tr_idx, val_idx in meta_skf.split(train_features, y_true):
            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                n_estimators=1200,
                learning_rate=0.05,
                max_depth=config["max_depth"],
                min_child_weight=config["min_child_weight"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                reg_alpha=config["reg_alpha"],
                reg_lambda=config["reg_lambda"],
                tree_method="hist",
                random_state=SEED,
                n_jobs=STACKER_N_JOBS,
                verbosity=0,
                early_stopping_rounds=50,
            )
            model.fit(
                train_features[tr_idx],
                y_true[tr_idx],
                sample_weight=sample_weights[tr_idx],
                eval_set=[(train_features[val_idx], y_true[val_idx])],
                verbose=False,
            )
            oof_proba[val_idx] = model.predict_proba(train_features[val_idx])

        score, oof_pred, class_scales = evaluate_probability_policy(
            oof_proba,
            y_true,
            apply_class_scale_search,
        )
        if score > best_score:
            best_score = score
            best_pred = oof_pred
            best_config = config
            best_class_scales = class_scales

    print(
        "Best xgb stack config — "
        f"depth:{best_config['max_depth']} "
        f"min_child_weight:{best_config['min_child_weight']} "
        f"subsample:{best_config['subsample']:.2f} "
        f"colsample:{best_config['colsample_bytree']:.2f}"
    )
    print_class_scale_summary("Best xgb stack class scales", best_class_scales)

    full_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=best_config["max_depth"],
        min_child_weight=best_config["min_child_weight"],
        subsample=best_config["subsample"],
        colsample_bytree=best_config["colsample_bytree"],
        reg_alpha=best_config["reg_alpha"],
        reg_lambda=best_config["reg_lambda"],
        tree_method="hist",
        random_state=SEED,
        n_jobs=STACKER_N_JOBS,
        verbosity=0,
    )
    full_model.fit(train_features, y_true, sample_weight=sample_weights)
    test_proba = full_model.predict_proba(test_features)
    final_test_pred = predict_with_class_scales(test_proba, best_class_scales)

    return best_score, best_pred, final_test_pred


def run_hgb_stack(
    train_features,
    y_true,
    test_features,
    sample_weights,
    apply_class_scale_search=False,
):
    meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    candidate_configs = [
        {
            "learning_rate": 0.05,
            "max_depth": 6,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 200,
            "l2_regularization": 0.1,
        },
        {
            "learning_rate": 0.04,
            "max_depth": 8,
            "max_leaf_nodes": 127,
            "min_samples_leaf": 100,
            "l2_regularization": 0.3,
        },
    ]

    best_score = -1.0
    best_pred = None
    best_config = None
    best_class_scales = (1.0, 1.0, 1.0)

    for config in candidate_configs:
        oof_proba = np.zeros((len(y_true), len(LABEL_ORDER)), dtype=np.float32)
        for tr_idx, val_idx in meta_skf.split(train_features, y_true):
            model = HistGradientBoostingClassifier(
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                max_leaf_nodes=config["max_leaf_nodes"],
                min_samples_leaf=config["min_samples_leaf"],
                l2_regularization=config["l2_regularization"],
                max_iter=300,
                early_stopping=True,
                random_state=SEED,
            )
            model.fit(
                train_features[tr_idx],
                y_true[tr_idx],
                sample_weight=sample_weights[tr_idx],
            )
            oof_proba[val_idx] = model.predict_proba(train_features[val_idx])

        score, oof_pred, class_scales = evaluate_probability_policy(
            oof_proba,
            y_true,
            apply_class_scale_search,
        )
        if score > best_score:
            best_score = score
            best_pred = oof_pred
            best_config = config
            best_class_scales = class_scales

    print(
        "Best hgb stack config — "
        f"depth:{best_config['max_depth']} "
        f"leaf_nodes:{best_config['max_leaf_nodes']} "
        f"min_samples_leaf:{best_config['min_samples_leaf']} "
        f"l2:{best_config['l2_regularization']:.2f}"
    )
    print_class_scale_summary("Best hgb stack class scales", best_class_scales)

    full_model = HistGradientBoostingClassifier(
        learning_rate=best_config["learning_rate"],
        max_depth=best_config["max_depth"],
        max_leaf_nodes=best_config["max_leaf_nodes"],
        min_samples_leaf=best_config["min_samples_leaf"],
        l2_regularization=best_config["l2_regularization"],
        max_iter=300,
        early_stopping=True,
        random_state=SEED,
    )
    full_model.fit(train_features, y_true, sample_weight=sample_weights)
    test_proba = full_model.predict_proba(test_features)
    final_test_pred = predict_with_class_scales(test_proba, best_class_scales)

    return best_score, best_pred, final_test_pred


def run_ordinal_xgb_stack(train_features, y_true, test_features, sample_weights):
    meta_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    candidate_configs = [
        {
            "max_depth": 4,
            "min_child_weight": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
        },
        {
            "max_depth": 5,
            "min_child_weight": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
        },
    ]

    best_score = -1.0
    best_pred = None
    best_config = None
    best_thresholds = None

    for config in candidate_configs:
        oof_score = np.zeros(len(y_true), dtype=float)
        for tr_idx, val_idx in meta_skf.split(train_features, y_true):
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=1200,
                learning_rate=0.05,
                max_depth=config["max_depth"],
                min_child_weight=config["min_child_weight"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                reg_alpha=config["reg_alpha"],
                reg_lambda=config["reg_lambda"],
                tree_method="hist",
                random_state=SEED,
                n_jobs=STACKER_N_JOBS,
                verbosity=0,
                early_stopping_rounds=50,
            )
            model.fit(
                train_features[tr_idx],
                y_true[tr_idx],
                sample_weight=sample_weights[tr_idx],
                eval_set=[(train_features[val_idx], y_true[val_idx])],
                verbose=False,
            )
            oof_score[val_idx] = model.predict(train_features[val_idx])

        score, thresholds, pred = search_ordinal_thresholds(oof_score, y_true)
        if score > best_score:
            best_score = score
            best_pred = pred
            best_config = config
            best_thresholds = thresholds

    print(
        "Best ordinal xgb stack config — "
        f"depth:{best_config['max_depth']} "
        f"min_child_weight:{best_config['min_child_weight']} "
        f"subsample:{best_config['subsample']:.2f} "
        f"colsample:{best_config['colsample_bytree']:.2f}"
    )
    print(
        "Best ordinal thresholds — "
        f"low_to_medium:{best_thresholds[0]:.3f} "
        f"medium_to_high:{best_thresholds[1]:.3f}"
    )

    full_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=best_config["max_depth"],
        min_child_weight=best_config["min_child_weight"],
        subsample=best_config["subsample"],
        colsample_bytree=best_config["colsample_bytree"],
        reg_alpha=best_config["reg_alpha"],
        reg_lambda=best_config["reg_lambda"],
        tree_method="hist",
        random_state=SEED,
        n_jobs=STACKER_N_JOBS,
        verbosity=0,
    )
    full_model.fit(train_features, y_true, sample_weight=sample_weights)
    test_scores = full_model.predict(test_features)
    final_test_pred = predict_with_ordinal_thresholds(test_scores, best_thresholds)

    return best_score, best_pred, final_test_pred

train = add_features(train)
test  = add_features(test)

CAT_COLS = (
    BASE_CAT_COLS
    + (CROSS_CAT_COLS if ARGS.categorical_crosses else [])
    + (STRESS_CAT_COLS if ARGS.stress_signals else [])
)
y = train["label"].values
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
split_indices = list(skf.split(train, y))

ENCODING_NUM_COLS = []
if ARGS.frequency_encoding:
    ENCODING_NUM_COLS.extend(add_frequency_encoding(train, test, CAT_COLS))
if ARGS.target_encoding:
    ENCODING_NUM_COLS.extend(
        add_target_encoding(
            train,
            test,
            y,
            split_indices,
            CAT_COLS,
            LABEL_ORDER,
            ARGS.target_encoding_smoothing,
        )
    )
if ENCODING_NUM_COLS:
    print(f"Generated encoding feature count: {len(ENCODING_NUM_COLS)}")

MODEL_NUM_COLS = (
    NUM_COLS
    + ENG_COLS
    + (RISK_FLAG_COLS if ARGS.risk_flags else [])
    + (STRESS_NUM_COLS if ARGS.stress_signals else [])
    + ENCODING_NUM_COLS
)
FEAT_COLS = MODEL_NUM_COLS + CAT_COLS  # used by CatBoost

# Ordinal-encode categoricals for LGB / XGB
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_cat_enc = oe.fit_transform(train[CAT_COLS])
test_cat_enc  = oe.transform(test[CAT_COLS])

X_num  = train[MODEL_NUM_COLS].values
X_test_num = test[MODEL_NUM_COLS].values

X      = np.hstack([X_num, train_cat_enc])
X_test = np.hstack([X_test_num, test_cat_enc])
meta_train_raw = build_meta_raw_features(train)
meta_test_raw = build_meta_raw_features(test)
meta_train_full = None
meta_test_full = None
if ARGS.meta_full_features and ARGS.decision_policy in {
    "logreg_stack",
    "mlp_stack",
    "mlp_bag_stack",
    "tabnet_stack",
    "ann_cnn_combo_stack",
    "neural_base_blend_stack",
    "ft_transformer_stack",
    "cnn_stack",
    "rnn_stack",
    "xgb_stack",
    "hgb_stack",
    "ordinal_xgb_stack",
}:
    meta_cat_cols = (
        BASE_CAT_COLS
        + (CROSS_CAT_COLS if ARGS.categorical_crosses else [])
        + (STRESS_CAT_COLS if ARGS.stress_signals else [])
    )
    meta_train_full, meta_test_full = build_meta_full_features(
        train,
        test,
        MODEL_NUM_COLS,
        meta_cat_cols,
    )
    print(
        "Meta full feature width: "
        f"{meta_train_full.shape[1]} (numeric {len(MODEL_NUM_COLS)} + categorical expansion)"
    )

lgb_cat_indices = list(range(len(MODEL_NUM_COLS), X.shape[1]))

# ── 4. Class weights ──────────────────────────────────────────────────────────

class_counts  = np.bincount(y)
class_weights = len(y) / (len(class_counts) * class_counts)
sample_weights = class_weights[y]
print(f"\nClass counts:  {dict(zip(LABEL_ORDER, class_counts))}")
print(f"Class weights: {dict(zip(LABEL_ORDER, class_weights.round(3)))}")

# ── 5. CV setup (seed=45, 5-fold ≈ 20% val each fold per protocol) ───────────

n_classes = 3

oof_lgb  = np.zeros((len(X), n_classes))
oof_xgb  = np.zeros((len(X), n_classes))
oof_cat  = np.zeros((len(X), n_classes))
pred_lgb = np.zeros((len(X_test), n_classes))
pred_xgb = np.zeros((len(X_test), n_classes))
pred_cat = np.zeros((len(X_test), n_classes))

best_iters_lgb = []
best_iters_xgb = []
cache_path = resolve_prediction_cache_path(ARGS.prediction_cache)

if cache_path and os.path.exists(cache_path):
    print("\n" + "─"*50)
    print("Loading Prediction Cache")
    print("─"*50)
    cache_payload = load_prediction_cache(cache_path)
    oof_lgb = cache_payload["oof_lgb"]
    oof_xgb = cache_payload["oof_xgb"]
    oof_cat = cache_payload["oof_cat"]
    pred_lgb = cache_payload["pred_lgb"]
    pred_xgb = cache_payload["pred_xgb"]
    pred_cat = cache_payload["pred_cat"]
    best_iters_lgb = cache_payload["best_iters_lgb"].astype(int).tolist()
    best_iters_xgb = cache_payload["best_iters_xgb"].astype(int).tolist()
    print(f"Prediction cache loaded ← {cache_path}")
    print(f"LGB OOF BA: {balanced_accuracy_score(y, np.argmax(oof_lgb, axis=1)):.4f}")
    print(f"XGB OOF BA: {balanced_accuracy_score(y, np.argmax(oof_xgb, axis=1)):.4f}")
    print(f"CAT OOF BA: {balanced_accuracy_score(y, np.argmax(oof_cat, axis=1)):.4f}")
else:
    # ── 6. LightGBM ───────────────────────────────────────────────────────────

    print("\n" + "─"*50)
    print("LightGBM")
    print("─"*50)

    lgb_params = dict(
        objective        = "multiclass",
        num_class        = 3,
        metric           = "multi_logloss",
        n_estimators     = 3000,
        learning_rate    = 0.05,
        num_leaves       = 127,
        min_child_samples= 20,
        subsample        = 0.8,
        subsample_freq   = 1,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        is_unbalance     = True,
        random_state     = SEED,
        n_jobs           = -1,
        verbose          = -1,
    )

    for fold, (tr_idx, val_idx) in enumerate(split_indices):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y_tr,
            sample_weight    = sample_weights[tr_idx],
            eval_set         = [(X_val, y_val)],
            categorical_feature = lgb_cat_indices,
            callbacks        = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
        )
        oof_lgb[val_idx] = model.predict_proba(X_val)
        pred_lgb += model.predict_proba(X_test) / N_FOLDS
        best_iters_lgb.append(model.best_iteration_)

        ba = balanced_accuracy_score(y_val, np.argmax(oof_lgb[val_idx], axis=1))
        print(f"  Fold {fold+1}: BA = {ba:.4f}  (best iter: {model.best_iteration_})")

    print(f"LGB OOF BA: {balanced_accuracy_score(y, np.argmax(oof_lgb, axis=1)):.4f}")

    # ── 7. XGBoost ────────────────────────────────────────────────────────────

    print("\n" + "─"*50)
    print("XGBoost")
    print("─"*50)

    xgb_params = dict(
        objective        = "multi:softprob",
        num_class        = 3,
        eval_metric      = "mlogloss",
        n_estimators     = 3000,
        learning_rate    = 0.05,
        max_depth        = 7,
        min_child_weight = 5,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        tree_method      = "hist",
        random_state     = SEED,
        n_jobs           = -1,
        verbosity        = 0,
        early_stopping_rounds = 100,
    )

    for fold, (tr_idx, val_idx) in enumerate(split_indices):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr, y_tr,
            sample_weight = sample_weights[tr_idx],
            eval_set      = [(X_val, y_val)],
            verbose       = 500,
        )
        oof_xgb[val_idx] = model.predict_proba(X_val)
        pred_xgb += model.predict_proba(X_test) / N_FOLDS
        best_iters_xgb.append(model.best_iteration)

        ba = balanced_accuracy_score(y_val, np.argmax(oof_xgb[val_idx], axis=1))
        print(f"  Fold {fold+1}: BA = {ba:.4f}  (best iter: {model.best_iteration})")

    print(f"XGB OOF BA: {balanced_accuracy_score(y, np.argmax(oof_xgb, axis=1)):.4f}")

    # ── 8. CatBoost ───────────────────────────────────────────────────────────

    print("\n" + "─"*50)
    print("CatBoost")
    print("─"*50)

    X_cb      = train[FEAT_COLS].values
    X_test_cb = test[FEAT_COLS].values
    cat_col_indices = [FEAT_COLS.index(c) for c in CAT_COLS]

    for fold, (tr_idx, val_idx) in enumerate(split_indices):
        X_tr, X_val = X_cb[tr_idx], X_cb[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = CatBoostClassifier(
            iterations         = 3000,
            learning_rate      = 0.05,
            depth              = 7,
            l2_leaf_reg        = 3,
            loss_function      = "MultiClass",
            eval_metric        = "MultiClass",
            cat_features       = cat_col_indices,
            random_seed        = SEED,
            early_stopping_rounds = 100,
            verbose            = 500,
        )
        model.fit(
            X_tr, y_tr,
            sample_weight = sample_weights[tr_idx],
            eval_set      = (X_val, y_val),
        )
        oof_cat[val_idx] = model.predict_proba(X_val)
        pred_cat += model.predict_proba(X_test_cb) / N_FOLDS

        ba = balanced_accuracy_score(y_val, np.argmax(oof_cat[val_idx], axis=1))
        print(f"  Fold {fold+1}: BA = {ba:.4f}")

    print(f"CAT OOF BA: {balanced_accuracy_score(y, np.argmax(oof_cat, axis=1)):.4f}")

    if cache_path:
        save_prediction_cache(
            cache_path,
            oof_lgb=oof_lgb,
            oof_xgb=oof_xgb,
            oof_cat=oof_cat,
            pred_lgb=pred_lgb,
            pred_xgb=pred_xgb,
            pred_cat=pred_cat,
            best_iters_lgb=np.asarray(best_iters_lgb, dtype=int),
            best_iters_xgb=np.asarray(best_iters_xgb, dtype=int),
        )
        print(f"Prediction cache saved → {cache_path}")

# ── 9. Ensemble weight search ─────────────────────────────────────────────────

print("\n" + "─"*50)
print("Ensemble weight search")
print("─"*50)

prob_matrices_oof = [oof_lgb, oof_xgb, oof_cat]
prob_matrices_test = [pred_lgb, pred_xgb, pred_cat]

if ARGS.blend_strategy == "refined_weight_search":
    best_ba, best_w, coarse_ba, coarse_w, n_candidates = search_refined_weights(
        prob_matrices_oof,
        y,
    )
    print(
        f"Coarse seed weights — LGB:{coarse_w[0]:.2f}  "
        f"XGB:{coarse_w[1]:.2f}  CAT:{coarse_w[2]:.2f}"
    )
    print(f"Coarse seed BA: {coarse_ba:.6f}")
    print(f"Refined candidate count: {n_candidates}")
else:
    best_ba, best_w = search_coarse_weights(prob_matrices_oof, y)

best_blend = blend_probabilities(prob_matrices_oof, best_w)
ensemble_oof_pred = np.argmax(best_blend, axis=1)
class_scales = (1.0, 1.0, 1.0)
test_pred_idx = predict_with_class_scales(
    blend_probabilities(prob_matrices_test, best_w),
    class_scales,
)

print(f"Best weights — LGB:{best_w[0]:.2f}  XGB:{best_w[1]:.2f}  CAT:{best_w[2]:.2f}")
if ARGS.decision_policy == "class_scale_search":
    best_ba, class_scales, ensemble_oof_pred = search_class_scales(best_blend, y)
    print(
        f"Best class scales — Low:{class_scales[0]:.2f}  "
        f"Medium:{class_scales[1]:.2f}  High:{class_scales[2]:.2f}"
    )
    test_pred_idx = predict_with_class_scales(
        blend_probabilities(prob_matrices_test, best_w),
        class_scales,
    )
elif ARGS.decision_policy == "model_scale_search":
    best_ba, model_class_scales, class_scales, ensemble_oof_pred = search_model_class_scales(
        prob_matrices_oof,
        best_w,
        y,
    )
    for model_label, scales in zip(MODEL_LABELS, model_class_scales):
        print(
            f"Best model scales [{model_label}] — "
            f"Low:{scales[0]:.2f}  Medium:{scales[1]:.2f}  High:{scales[2]:.2f}"
        )
    print(
        f"Final class scales — Low:{class_scales[0]:.2f}  "
        f"Medium:{class_scales[1]:.2f}  High:{class_scales[2]:.2f}"
    )
    scaled_test_blend = build_model_scaled_blend(prob_matrices_test, best_w, model_class_scales)
    test_pred_idx = predict_with_class_scales(scaled_test_blend, class_scales)
elif ARGS.decision_policy in {
    "logreg_stack",
    "mlp_stack",
    "mlp_bag_stack",
    "tabnet_stack",
    "ann_cnn_combo_stack",
    "neural_base_blend_stack",
    "ft_transformer_stack",
    "cnn_stack",
    "rnn_stack",
    "xgb_stack",
    "hgb_stack",
    "ordinal_xgb_stack",
}:
    stack_train, stack_test = build_stack_feature_matrices(
        prob_matrices_oof,
        prob_matrices_test,
        meta_raw_train=meta_train_raw if ARGS.meta_raw_features else None,
        meta_raw_test=meta_test_raw if ARGS.meta_raw_features else None,
        meta_full_train=meta_train_full if ARGS.meta_full_features else None,
        meta_full_test=meta_test_full if ARGS.meta_full_features else None,
    )

    if ARGS.decision_policy == "logreg_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_logreg_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: multinomial logistic regression on cached OOF probabilities")
    elif ARGS.decision_policy == "mlp_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_mlp_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: ANN stacker on cached OOF probabilities")
    elif ARGS.decision_policy == "mlp_bag_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_mlp_bag_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: seed-bagged ANN stacker on cached OOF probabilities")
    elif ARGS.decision_policy == "tabnet_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_tabnet_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: TabNet stacker on cached OOF probabilities")
    elif ARGS.decision_policy == "ann_cnn_combo_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_ann_cnn_combo_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: ANN+CNN combo stacker on cached OOF probabilities")
    elif ARGS.decision_policy == "neural_base_blend_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_neural_base_blend_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            prob_matrices_oof,
            prob_matrices_test,
            best_w,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: ANN+CNN+tree blend on cached OOF probabilities")
    elif ARGS.decision_policy == "ft_transformer_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_ft_transformer_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: FT-transformer stacker on cached OOF probabilities")
    elif ARGS.decision_policy == "cnn_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_cnn_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: CNN stacker on cached OOF probabilities")
    elif ARGS.decision_policy == "rnn_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_rnn_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: GRU-based RNN stacker on cached OOF probabilities")
    elif ARGS.decision_policy == "xgb_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_xgb_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: XGBoost stacker on cached OOF probabilities")
    elif ARGS.decision_policy == "hgb_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_hgb_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
            apply_class_scale_search=ARGS.stack_class_scale_search,
        )
        print("Meta-model: HistGradientBoosting stacker on cached OOF probabilities")
    elif ARGS.decision_policy == "ordinal_xgb_stack":
        best_ba, ensemble_oof_pred, test_pred_idx = run_ordinal_xgb_stack(
            stack_train,
            y,
            stack_test,
            sample_weights,
        )
        print("Meta-model: ordinal XGBoost regressor stacker on cached OOF probabilities")
print(f"Ensemble OOF BA: {best_ba:.4f}")
print_class_diagnostics(y, ensemble_oof_pred, LABEL_ORDER)

# Summary lines (greppable by lead agent)
avg_best_iter = int(np.mean(best_iters_lgb + best_iters_xgb))
print(f"val_balanced_accuracy_score: {best_ba:.6f}")
print(f"best_iteration: {avg_best_iter}")

# ── 10. Generate submission ───────────────────────────────────────────────────

test_pred  = label_enc.inverse_transform(test_pred_idx)

if ARGS.skip_predictions:
    print("\nPrediction save skipped for challenger safety.")
    print(f"Prediction distribution (unsaved):\n{pd.Series(test_pred).value_counts()}")
else:
    sub["Irrigation_Need"] = test_pred
    pred_path = os.path.join(PRED_DIR, "prediction_irr_need.csv")
    sub.to_csv(pred_path, index=False)
    print(f"\nPrediction saved → {pred_path}")
    print(f"Prediction distribution:\n{pd.Series(test_pred).value_counts()}")
