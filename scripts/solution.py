from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from boruta import BorutaPy
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = REPO_ROOT / "Data" / "train.csv"
DEFAULT_TEST_PATH = REPO_ROOT / "Data" / "test.csv"
DEFAULT_SAMPLE_SUBMISSION_PATH = REPO_ROOT / "Data" / "sample_submission.csv"
DEFAULT_SUBMISSION_PATH = REPO_ROOT / "Predictions" / "prediction_irr_need.csv"
DEFAULT_MODEL_PATH = REPO_ROOT / "artifacts" / "irrigation_need_catboost.cbm"
DEFAULT_ENSEMBLE_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "irrigation_need_extra_trees.pkl"
DEFAULT_METADATA_PATH = REPO_ROOT / "artifacts" / "irrigation_need_metadata.json"
TARGET_ORDER = ["Low", "Medium", "High"]
PEAK_STAGES = {"Vegetative", "Flowering"}
BORUTA_RF_TREES = 400
BORUTA_RF_DEPTH = 8
CERTAIN_FEATURE_PREFIX = "certain_"
STAGE_WATER_DEMAND = {
    "Sowing": 0.92,
    "Vegetative": 1.12,
    "Flowering": 1.22,
    "Harvest": 0.82,
}
IRRIGATION_DELIVERY_FACTOR = {
    "Drip": 1.16,
    "Sprinkler": 1.04,
    "Canal": 0.92,
    "Rainfed": 0.80,
}
WATER_SOURCE_STABILITY = {
    "Groundwater": 1.05,
    "Reservoir": 1.00,
    "River": 0.95,
    "Rainwater": 0.88,
}
MULCH_DEMAND_FACTOR = {
    "Yes": 0.88,
    "No": 1.00,
    "Missing": 1.00,
}
POSITIVE_PRUNING_PROBE_OVERRIDES = {
    "iterations": 450,
    "learning_rate": 0.06,
    "depth": 7,
    "l2_leaf_reg": 8.0,
    "subsample": 0.85,
    "min_data_in_leaf": 24,
    "random_strength": 0.75,
    "border_count": 128,
    "one_hot_max_size": 6,
    "verbose": False,
}
STACKING_SIGNAL_COLUMNS = [
    "net_need_signal",
    "peak_delivery_drought",
    f"{CERTAIN_FEATURE_PREFIX}medium_peak_mulched_wetter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a leakage-safe CatBoost model for irrigation need prediction, "
            "hold out 15%% of train.csv for validation, report validation metrics, "
            "and optionally refit on the full training set for submission."
        )
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=DEFAULT_SAMPLE_SUBMISSION_PATH,
    )
    parser.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--ensemble-artifact-path",
        type=Path,
        default=DEFAULT_ENSEMBLE_ARTIFACT_PATH,
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--target-column", default="Irrigation_Need")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=45)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--loss-function",
        choices=["MultiClass", "MultiClassOneVsAll"],
        default="MultiClass",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=6.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--border-count", type=int, default=128)
    parser.add_argument("--one-hot-max-size", type=int, default=6)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--engineered-features",
        action="store_true",
        help="Enable domain-grounded agronomic feature engineering.",
    )
    parser.add_argument(
        "--boruta",
        action="store_true",
        help="Run Boruta feature selection on the training split only.",
    )
    parser.add_argument(
        "--boruta-positive-pruning",
        action="store_true",
        help=(
            "After Boruta, keep only selected engineered numeric features with "
            "positive permutation contribution on an inner training calibration split."
        ),
    )
    parser.add_argument(
        "--class-scale-search",
        action="store_true",
        help="Learn class probability scales on an inner train/calibration split.",
    )
    parser.add_argument(
        "--rule-overrides",
        action="store_true",
        help="Learn pure class override rules from binary engineered features on the training split only.",
    )
    parser.add_argument(
        "--rule-override-min-purity",
        type=float,
        default=0.975,
        help="Minimum training purity for a learned rule override.",
    )
    parser.add_argument(
        "--rule-override-min-support",
        type=int,
        default=1000,
        help="Minimum training support for a learned rule override.",
    )
    parser.add_argument(
        "--rule-override-max-rules",
        type=int,
        default=5,
        help="Maximum number of learned rule overrides to keep.",
    )
    parser.add_argument(
        "--boruta-max-rows",
        type=int,
        default=120000,
        help="Maximum number of training rows used for Boruta.",
    )
    parser.add_argument(
        "--boruta-max-iter",
        type=int,
        default=20,
        help="Maximum Boruta iterations.",
    )
    parser.add_argument(
        "--positive-pruning-max-rows",
        type=int,
        default=30000,
        help="Maximum number of training rows used for post-Boruta contribution pruning.",
    )
    parser.add_argument(
        "--boruta-perc",
        type=int,
        default=85,
        help="Boruta percentile threshold for shadow importance.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Number of Optuna trials. Set to 0 to disable tuning.",
    )
    parser.add_argument(
        "--optuna-tune-class-weights",
        action="store_true",
        help="Allow Optuna to tune per-class weight multipliers.",
    )
    parser.add_argument(
        "--optuna-timeout-seconds",
        type=int,
        default=0,
        help="Optional Optuna timeout. Set to 0 for no timeout.",
    )
    parser.add_argument(
        "--optuna-train-max-rows",
        type=int,
        default=220000,
        help="Maximum number of training rows used during hyperparameter search.",
    )
    parser.add_argument(
        "--ensemble-mode",
        choices=["none", "cat_high_else_extra"],
        default="none",
        help=(
            "Optional selector ensemble that keeps CatBoost High predictions and "
            "delegates other rows to ExtraTrees."
        ),
    )
    parser.add_argument(
        "--extra-trees-estimators",
        type=int,
        default=700,
        help="Number of trees for the ExtraTrees sidecar used by ensemble mode.",
    )
    parser.add_argument(
        "--extra-trees-max-depth",
        type=int,
        default=24,
        help="Maximum depth for the ExtraTrees sidecar. Use 0 for unlimited depth.",
    )
    parser.add_argument(
        "--extra-trees-min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples per leaf for the ExtraTrees sidecar.",
    )
    parser.add_argument(
        "--extra-trees-max-features",
        default="sqrt",
        help=(
            "Value passed to ExtraTreesClassifier(max_features). "
            "Examples: sqrt, log2, 0.4, 1.0."
        ),
    )
    parser.add_argument(
        "--stacking-mode",
        choices=["none", "catboost_triplet_logreg"],
        default="none",
        help=(
            "Optional leakage-safe probability stacker. "
            "`catboost_triplet_logreg` uses three CatBoost variants and a "
            "multinomial logistic meta-learner."
        ),
    )
    parser.add_argument(
        "--stacking-folds",
        type=int,
        default=3,
        help="Number of OOF folds used by the stacking meta-learner.",
    )
    parser.add_argument(
        "--stacking-max-rows",
        type=int,
        default=220000,
        help="Maximum number of outer-train rows used to fit the stacking meta-learner.",
    )
    parser.add_argument(
        "--stacking-logreg-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the multinomial logistic stacker.",
    )
    parser.add_argument(
        "--skip-refit",
        action="store_true",
        help="Stop after validation instead of refitting on the full training set.",
    )
    args = parser.parse_args()
    if args.stacking_mode != "none":
        if args.ensemble_mode != "none":
            raise SystemExit("Stacking mode cannot be combined with ensemble_mode.")
        if args.class_scale_search:
            raise SystemExit("Stacking mode cannot be combined with class_scale_search.")
        if args.rule_overrides:
            raise SystemExit("Stacking mode cannot be combined with rule_overrides.")
        if args.stacking_folds < 2:
            raise SystemExit("stacking_folds must be at least 2.")
    return args


def json_ready(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: json_ready(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [json_ready(value) for value in payload]
    if isinstance(payload, tuple):
        return [json_ready(value) for value in payload]
    if isinstance(payload, np.integer):
        return int(payload)
    if isinstance(payload, np.floating):
        return float(payload)
    if isinstance(payload, np.bool_):
        return bool(payload)
    return payload


def safe_ratio(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    offset: float = 1.0,
) -> pd.Series | np.ndarray:
    return numerator / (denominator + offset)


def engineer_features(frame: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return frame.copy()

    df = frame.copy()
    soil_moisture = df["Soil_Moisture"].astype(float)
    temperature = df["Temperature_C"].astype(float)
    humidity = df["Humidity"].astype(float)
    rainfall = df["Rainfall_mm"].astype(float)
    sunlight = df["Sunlight_Hours"].astype(float)
    wind = df["Wind_Speed_kmh"].astype(float)
    area = df["Field_Area_hectare"].astype(float)
    previous_irrigation = df["Previous_Irrigation_mm"].astype(float)
    conductivity = df["Electrical_Conductivity"].astype(float)
    organic_carbon = df["Organic_Carbon"].astype(float)
    soil_ph = df["Soil_pH"].astype(float)
    soil_type = df["Soil_Type"].astype("string").fillna("Missing")
    crop_type = df["Crop_Type"].astype("string").fillna("Missing")
    region = df["Region"].astype("string").fillna("Missing")

    water_in = rainfall + previous_irrigation
    humidity_relief = np.clip(1.0 - humidity / 100.0, 0.05, None)
    area_safe = area + 0.1
    soil_safe = soil_moisture + 1.0
    rain_safe = rainfall + 1.0
    per_ha_water_in = water_in / area_safe
    saturation_vapor_pressure = 0.6108 * np.exp(
        (17.27 * temperature) / (temperature + 237.3)
    )
    wind_root = np.sqrt(np.clip(wind, 0.0, None) + 1.0)
    sunlight_plus = np.clip(sunlight, 0.0, None) + 1.0

    peak_stage = df["Crop_Growth_Stage"].isin(PEAK_STAGES)
    growth_stage = df["Crop_Growth_Stage"].astype("string").fillna("Missing")
    no_mulch = df["Mulching_Used"].astype("string").fillna("Missing").eq("No")
    mulch_used = df["Mulching_Used"].astype("string").fillna("Missing")
    mulch_yes = mulch_used.eq("Yes")
    harvest_stage = growth_stage.eq("Harvest")
    sowing_stage = growth_stage.eq("Sowing")
    non_peak_stage = harvest_stage | sowing_stage
    water_source = df["Water_Source"].astype("string").fillna("Missing")
    irrigation_type = df["Irrigation_Type"].astype("string").fillna("Missing")
    season = df["Season"].astype("string").fillna("Missing")
    river_source = water_source.eq("River")
    canal_irrigation = irrigation_type.eq("Canal")
    rainfed_irrigation = irrigation_type.eq("Rainfed")
    drip_irrigation = irrigation_type.eq("Drip")
    kharif_season = season.eq("Kharif")
    stage_demand_factor = growth_stage.map(STAGE_WATER_DEMAND).fillna(1.0).astype(float)
    irrigation_delivery = (
        irrigation_type.map(IRRIGATION_DELIVERY_FACTOR).fillna(1.0).astype(float)
    )
    source_stability = (
        water_source.map(WATER_SOURCE_STABILITY).fillna(1.0).astype(float)
    )
    mulch_demand_factor = (
        mulch_used.map(MULCH_DEMAND_FACTOR).fillna(1.0).astype(float)
    )
    effective_irrigation = previous_irrigation * irrigation_delivery * source_stability
    effective_water_input = rainfall + effective_irrigation
    adjusted_et_demand = (
        safe_ratio(
            temperature * sunlight * (1.0 + wind / 10.0),
            humidity + 1.0,
            offset=0.0,
        )
        * stage_demand_factor
        * mulch_demand_factor
    )
    soil_mid_25_40 = (soil_moisture > 25.0) & (soil_moisture <= 40.0)
    temp_30_35 = (temperature > 30.0) & (temperature <= 35.0)
    wind_10_15 = (wind > 10.0) & (wind <= 15.0)
    rain_500_1200 = (rainfall > 500.0) & (rainfall <= 1200.0)

    df["water_in"] = water_in
    df["water_in_log"] = np.log1p(water_in)
    df["rainfall_log"] = np.log1p(rainfall)
    df["prev_irrigation_log"] = np.log1p(previous_irrigation)
    df["field_area_log"] = np.log1p(area)
    df["moisture_deficit_20"] = np.clip(20.0 - soil_moisture, 0.0, None)
    df["moisture_deficit_18"] = np.clip(18.0 - soil_moisture, 0.0, None)
    df["moisture_deficit_22"] = np.clip(22.0 - soil_moisture, 0.0, None)
    df["moisture_deficit_25"] = np.clip(25.0 - soil_moisture, 0.0, None)
    df["moisture_deficit_30"] = np.clip(30.0 - soil_moisture, 0.0, None)
    df["moisture_le_25"] = (soil_moisture <= 25.0).astype(int)
    df["moisture_le_20"] = (soil_moisture <= 20.0).astype(int)
    df["moisture_le_22"] = (soil_moisture <= 22.0).astype(int)
    df["moisture_le_18"] = (soil_moisture <= 18.0).astype(int)
    df["rain_le_200"] = (rainfall <= 200.0).astype(int)
    df["rain_le_250"] = (rainfall <= 250.0).astype(int)
    df["rain_le_300"] = (rainfall <= 300.0).astype(int)
    df["rain_le_353"] = (rainfall <= 353.0).astype(int)
    df["rain_le_500"] = (rainfall <= 500.0).astype(int)
    df["temp_gt_30"] = (temperature > 30.0).astype(int)
    df["temp_gt_32"] = (temperature > 32.0).astype(int)
    df["temp_gt_35"] = (temperature > 35.0).astype(int)
    df["wind_gt_10"] = (wind > 10.0).astype(int)
    df["wind_gt_12"] = (wind > 12.0).astype(int)
    df["wind_gt_15"] = (wind > 15.0).astype(int)
    df["no_mulch"] = no_mulch.astype(int)
    df["mulch_yes"] = mulch_yes.astype(int)
    df["is_harvest"] = harvest_stage.astype(int)
    df["is_sowing"] = sowing_stage.astype(int)
    df["stage_demand_factor"] = stage_demand_factor
    df["irrigation_delivery_factor"] = irrigation_delivery
    df["water_source_stability"] = source_stability
    df["mulch_demand_factor"] = mulch_demand_factor
    df["dryness_ratio"] = safe_ratio(temperature + wind, soil_safe, offset=0.0)
    df["water_stress"] = safe_ratio(temperature, soil_safe * rain_safe, offset=0.0)
    df["heat_wind_pressure"] = temperature * (1.0 + wind / 10.0)
    df["heat_index"] = temperature * humidity / 100.0
    df["et_proxy"] = (
        temperature
        * sunlight
        * (1.0 + wind / 10.0)
        * humidity_relief
    )
    df["evapotranspiration_proxy"] = safe_ratio(
        temperature * sunlight,
        humidity + 1.0,
        offset=0.0,
    )
    df["heat_stress"] = temperature * humidity_relief
    df["net_water_stress"] = df["et_proxy"] - np.log1p(water_in)
    df["water_stress_ratio"] = safe_ratio(df["et_proxy"], np.log1p(water_in), offset=1.0)
    df["effective_rain"] = rainfall * np.clip(1.0 - soil_moisture / 100.0, 0.0, None)
    df["rain_per_ha"] = safe_ratio(rainfall, area, offset=0.1)
    df["prev_irr_per_ha"] = safe_ratio(previous_irrigation, area, offset=0.1)
    df["water_in_per_ha"] = safe_ratio(water_in, area, offset=0.1)
    df["irrigation_reliance"] = safe_ratio(previous_irrigation, water_in, offset=1.0)
    df["aridity_temp"] = safe_ratio(temperature, rainfall, offset=1.0)
    df["aridity_wind"] = safe_ratio(wind, rainfall, offset=1.0)
    df["aridity_index"] = safe_ratio(temperature * wind, rainfall, offset=1.0)
    df["soil_heat_gap"] = temperature - soil_moisture
    df["humidity_soil_gap"] = humidity - soil_moisture
    df["moisture_deficit_total"] = 100.0 - soil_moisture
    df["dry_heat"] = temperature * df["moisture_deficit_total"] / 100.0
    df["wind_heat"] = temperature * wind
    df["drought_pressure"] = safe_ratio(
        df["moisture_deficit_total"] * (temperature + wind),
        rainfall + 10.0,
        offset=0.0,
    )
    df["rainfall_relief"] = safe_ratio(rainfall, temperature + wind, offset=1.0)
    df["prev_vs_rain"] = safe_ratio(previous_irrigation, rainfall, offset=10.0)
    df["soil_temp_interaction"] = soil_moisture * temperature
    df["soil_wind_interaction"] = soil_moisture * wind
    df["rain_irrigation_product"] = rainfall * previous_irrigation
    df["salinity_organic_ratio"] = safe_ratio(conductivity, organic_carbon, offset=0.1)
    df["soil_quality_index"] = safe_ratio(organic_carbon, conductivity, offset=1.0)
    df["soil_health"] = safe_ratio(organic_carbon * soil_moisture, conductivity, offset=0.1)
    df["salinity_risk"] = safe_ratio(conductivity * temperature, rainfall, offset=1.0)
    df["ph_deviation"] = np.abs(soil_ph - 6.5)
    df["ph_ec_product"] = soil_ph * conductivity
    df["moisture_retention"] = soil_moisture * organic_carbon
    df["mulch_flag"] = mulch_yes.astype(float)
    df["mulch_moisture_buffer"] = soil_moisture * (1.0 + 0.5 * df["mulch_flag"])
    df["mulch_et_saving"] = df["evapotranspiration_proxy"] * (1.0 - 0.3 * df["mulch_flag"])
    df["effective_irrigation_mm"] = effective_irrigation
    df["effective_irrigation_log"] = np.log1p(effective_irrigation)
    df["effective_water_input"] = effective_water_input
    df["effective_water_log"] = np.log1p(effective_water_input)
    df["effective_water_per_ha"] = safe_ratio(effective_water_input, area, offset=0.1)
    df["adjusted_et_demand"] = adjusted_et_demand
    df["adjusted_et_log"] = np.log1p(adjusted_et_demand)
    df["stage_adjusted_moisture"] = safe_ratio(
        soil_moisture,
        stage_demand_factor,
        offset=0.0,
    )
    df["stage_adjusted_water_in_per_ha"] = safe_ratio(
        df["effective_water_per_ha"],
        stage_demand_factor,
        offset=0.0,
    )
    df["effective_supply_demand_ratio"] = safe_ratio(
        np.log1p(effective_water_input),
        np.log1p(adjusted_et_demand),
        offset=1.0,
    )
    df["adjusted_supply_gap"] = adjusted_et_demand - np.log1p(effective_water_input)
    df["soil_buffered_adjusted_gap"] = safe_ratio(
        adjusted_et_demand,
        soil_safe,
        offset=0.0,
    ) - np.log1p(effective_water_input)
    df["delivery_adjusted_prev_irr_per_ha"] = safe_ratio(
        effective_irrigation,
        area,
        offset=0.1,
    )
    df["stage_delivery_drought"] = safe_ratio(
        df["drought_pressure"] * stage_demand_factor,
        irrigation_delivery * source_stability,
        offset=0.1,
    )
    df["peak_delivery_drought"] = df["stage_delivery_drought"] * peak_stage.astype(float)
    df["non_peak_heat_wind"] = df["heat_wind_pressure"] * non_peak_stage.astype(float)
    df["peak_no_mulch_supply_gap"] = df["adjusted_supply_gap"] * (
        peak_stage & no_mulch
    ).astype(float)
    df["rainfed_stage_gap"] = df["adjusted_supply_gap"] * rainfed_irrigation.astype(float)
    df["drip_stage_gap"] = df["adjusted_supply_gap"] * drip_irrigation.astype(float)
    df["moisture_temp_ratio"] = safe_ratio(soil_moisture, temperature, offset=1.0)
    df["moisture_wind_ratio"] = safe_ratio(soil_moisture, wind, offset=1.0)
    df["moisture_rainfall_ratio"] = safe_ratio(soil_moisture, rainfall, offset=1.0)
    df["moisture_et_ratio"] = safe_ratio(soil_moisture, df["evapotranspiration_proxy"], offset=0.1)
    df["vpd_proxy"] = temperature * humidity_relief
    df["saturation_vapor_pressure"] = saturation_vapor_pressure
    df["vpd_kpa"] = saturation_vapor_pressure * humidity_relief
    df["atmospheric_demand"] = df["vpd_kpa"] * sunlight_plus * wind_root
    df["atmospheric_demand_log"] = np.log1p(df["atmospheric_demand"])
    df["atmospheric_demand_per_ha"] = safe_ratio(
        df["atmospheric_demand"],
        area,
        offset=0.1,
    )
    df["water_to_atmospheric_demand"] = safe_ratio(
        water_in,
        df["atmospheric_demand"],
        offset=1.0,
    )
    df["water_per_ha_to_atmospheric_demand"] = safe_ratio(
        per_ha_water_in,
        df["atmospheric_demand"],
        offset=1.0,
    )
    df["soil_buffer_to_atmospheric_demand"] = safe_ratio(
        soil_moisture,
        df["atmospheric_demand"],
        offset=0.5,
    )
    df["organic_buffer_to_atmospheric_demand"] = safe_ratio(
        organic_carbon * soil_moisture,
        df["atmospheric_demand"],
        offset=0.5,
    )
    df["salinity_atmospheric_risk"] = safe_ratio(
        conductivity * df["atmospheric_demand"],
        water_in,
        offset=1.0,
    )
    df["drying_index"] = safe_ratio(sunlight * wind, humidity, offset=1.0)
    df["net_water_need"] = df["evapotranspiration_proxy"] - rainfall / 10.0
    df["drought_risk"] = df["drying_index"] * df["moisture_deficit_total"] / 100.0
    df["water_balance"] = rainfall - previous_irrigation
    df["water_deficit_signal"] = soil_moisture - 0.1 * water_in
    df["rain_et_balance"] = rainfall - 2.0 * df["evapotranspiration_proxy"]
    df["sunlight_temp_ratio"] = safe_ratio(sunlight, temperature, offset=1.0)
    df["water_stress_index"] = safe_ratio(
        df["evapotranspiration_proxy"] - water_in / 10.0,
        soil_moisture,
        offset=1.0,
    )
    df["peak_no_mulch_atmospheric_demand"] = df["atmospheric_demand"] * (
        peak_stage & no_mulch
    ).astype(float)
    df["peak_river_atmospheric_demand"] = df["atmospheric_demand"] * (
        peak_stage & river_source
    ).astype(float)
    df["peak_kharif_atmospheric_demand"] = df["atmospheric_demand"] * (
        peak_stage & kharif_season
    ).astype(float)
    df["peak_no_mulch_demand_supply_ratio"] = safe_ratio(
        df["atmospheric_demand"],
        per_ha_water_in,
        offset=1.0,
    ) * (peak_stage & no_mulch).astype(float)
    df["peak_river_demand_supply_ratio"] = safe_ratio(
        df["atmospheric_demand"],
        per_ha_water_in,
        offset=1.0,
    ) * (peak_stage & river_source).astype(float)
    df["peak_kharif_demand_supply_ratio"] = safe_ratio(
        df["atmospheric_demand"],
        per_ha_water_in,
        offset=1.0,
    ) * (peak_stage & kharif_season).astype(float)
    df["is_peak_stage"] = peak_stage.astype(int)
    df["peak_stage_no_mulch"] = (peak_stage & no_mulch).astype(int)
    df["peak_stage_dry"] = (peak_stage & (soil_moisture < 25.0)).astype(int)
    df["peak_stage_drought"] = df["drought_pressure"] * peak_stage.astype(int)
    df["peak_stage_no_mulch_drought"] = df["drought_pressure"] * (
        peak_stage & no_mulch
    ).astype(int)
    df["river_peak_no_mulch_drought"] = df["drought_pressure"] * (
        peak_stage & no_mulch & river_source
    ).astype(int)
    df["canal_peak_no_mulch_drought"] = df["drought_pressure"] * (
        peak_stage & no_mulch & canal_irrigation
    ).astype(int)
    df["peak_stage_dry_hot"] = (
        peak_stage
        & no_mulch
        & (soil_moisture < 25.0)
        & (temperature > 32.0)
    ).astype(int)
    df["dry_hot_windy_flag"] = (
        (soil_moisture < 25.0)
        & (temperature > 32.0)
        & (wind > 12.0)
    ).astype(int)
    df["high_core"] = (
        (soil_moisture <= 25.0)
        & (rainfall <= 353.0)
        & (temperature > 30.0)
        & (wind > 10.0)
    ).astype(int)
    df["high_super"] = (
        (soil_moisture <= 25.0)
        & (rainfall > 353.0)
        & peak_stage
        & no_mulch
        & (temperature > 30.0)
        & (wind > 12.0)
    ).astype(int)
    df["stress_count"] = (
        (soil_moisture <= 25.0).astype(int)
        + (rainfall <= 500.0).astype(int)
        + (temperature > 30.0).astype(int)
        + (wind > 10.0).astype(int)
        + peak_stage.astype(int)
        + no_mulch.astype(int)
    )
    df["stress_count_alt"] = (
        (soil_moisture <= 26.0).astype(int)
        + (temperature >= 30.0).astype(int)
        + (wind >= 12.0).astype(int)
        + (rainfall <= 1000.0).astype(int)
        + (peak_stage & no_mulch).astype(int)
    )
    df["high_need_signal_count"] = (
        2 * (soil_moisture < 25.0).astype(int)
        + 2 * (rainfall < 300.0).astype(int)
        + (temperature > 30.0).astype(int)
        + (wind > 10.0).astype(int)
    )
    df["peak_no_mulch_moisture_le_22"] = (
        peak_stage & no_mulch & (soil_moisture <= 22.0)
    ).astype(int)
    df["peak_no_mulch_rain_le_300"] = (
        peak_stage & no_mulch & (rainfall <= 300.0)
    ).astype(int)
    df["peak_no_mulch_wind_gt_15"] = (
        peak_stage & no_mulch & (wind > 15.0)
    ).astype(int)
    df["non_peak_hot_dry_windy"] = (
        non_peak_stage
        & (soil_moisture <= 22.0)
        & (temperature > 32.0)
        & (wind > 12.0)
    ).astype(int)
    df["low_need_signal_count"] = (
        2 * harvest_stage.astype(int)
        + 2 * sowing_stage.astype(int)
        + mulch_yes.astype(int)
    )
    df["net_need_signal"] = (
        df["high_need_signal_count"] - df["low_need_signal_count"]
    )
    df[f"{CERTAIN_FEATURE_PREFIX}low_non_peak_mulched_relief"] = (
        non_peak_stage
        & mulch_yes
        & (soil_moisture > 30.0)
        & (temperature < 30.0)
    ).astype(int)
    df[f"{CERTAIN_FEATURE_PREFIX}low_non_peak_mulched_exact"] = (
        non_peak_stage
        & mulch_yes
        & (soil_moisture > 35.0)
        & (temperature < 30.0)
        & (rainfall > 300.0)
    ).astype(int)
    df[f"{CERTAIN_FEATURE_PREFIX}low_non_peak_nomulch_relief"] = (
        non_peak_stage
        & no_mulch
        & (soil_moisture > 30.0)
        & (temperature < 28.0)
    ).astype(int)
    df[f"{CERTAIN_FEATURE_PREFIX}medium_peak_nomulch_warm_midsoil"] = (
        peak_stage
        & no_mulch
        & soil_mid_25_40
        & temp_30_35
    ).astype(int)
    df[f"{CERTAIN_FEATURE_PREFIX}medium_peak_nomulch_windy_midsoil"] = (
        peak_stage
        & no_mulch
        & soil_mid_25_40
        & wind_10_15
    ).astype(int)
    df[f"{CERTAIN_FEATURE_PREFIX}medium_peak_mulched_wet"] = (
        peak_stage
        & mulch_yes
        & (soil_moisture <= 25.0)
        & rain_500_1200
    ).astype(int)
    df[f"{CERTAIN_FEATURE_PREFIX}medium_peak_mulched_wetter"] = (
        peak_stage
        & mulch_yes
        & (soil_moisture <= 25.0)
        & (rainfall > 1200.0)
    ).astype(int)
    df[f"{CERTAIN_FEATURE_PREFIX}high_peak_nomulch_rain400"] = (
        peak_stage
        & no_mulch
        & (soil_moisture <= 25.0)
        & (temperature > 30.0)
        & (wind > 10.0)
        & (rainfall <= 400.0)
    ).astype(int)
    df[f"{CERTAIN_FEATURE_PREFIX}high_peak_nomulch_rain800"] = (
        peak_stage
        & no_mulch
        & (soil_moisture <= 25.0)
        & (temperature > 30.0)
        & (wind > 10.0)
        & (rainfall <= 800.0)
    ).astype(int)
    df[f"{CERTAIN_FEATURE_PREFIX}high_peak_nomulch_hot_core"] = (
        peak_stage
        & no_mulch
        & (soil_moisture <= 20.0)
        & (temperature > 32.0)
        & (wind > 12.0)
    ).astype(int)
    df["soil_moisture_band"] = pd.cut(
        soil_moisture,
        bins=[-np.inf, 15.0, 20.0, 25.0, 30.0, 40.0, np.inf],
        labels=["critical", "very_dry", "dry_gate", "dry_tail", "balanced", "wet"],
        ordered=True,
    )
    df["temperature_band"] = pd.cut(
        temperature,
        bins=[-np.inf, 25.0, 30.0, 32.0, 35.0, np.inf],
        labels=["cool", "warm_gate", "hot_gate", "hot", "very_hot"],
        ordered=True,
    )
    df["rainfall_band"] = pd.cut(
        rainfall,
        bins=[-np.inf, 300.0, 500.0, 800.0, 1200.0, 1600.0, np.inf],
        labels=["extreme_low", "very_low", "low_gate", "moderate", "high", "very_high"],
        ordered=True,
    )
    df["wind_band"] = pd.cut(
        wind,
        bins=[-np.inf, 8.0, 10.0, 12.0, 16.0, np.inf],
        labels=["calm", "moderate", "wind_gate", "breezy", "windy"],
        ordered=True,
    )
    df["drought_pressure_band"] = pd.cut(
        df["drought_pressure"],
        bins=[-np.inf, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, np.inf],
        labels=["trace", "light", "moderate", "elevated", "high", "severe", "extreme"],
        ordered=True,
    )
    df["stress_count_alt_band"] = pd.cut(
        df["stress_count_alt"],
        bins=[-np.inf, 1.0, 2.0, 3.0, 4.0, np.inf],
        labels=["minimal", "guarded", "elevated", "high", "extreme"],
        ordered=True,
    )
    df["peak_stage_bucket"] = np.where(
        peak_stage & no_mulch,
        "peak_no_mulch",
        np.where(peak_stage, "peak", "other"),
    )
    df["Crop_Growth_Stage__Mulching_Used"] = (
        df["Crop_Growth_Stage"].astype("string").fillna("Missing")
        + "__"
        + df["Mulching_Used"].astype("string").fillna("Missing")
    )
    df["Irrigation_Type__Water_Source"] = (
        df["Irrigation_Type"].astype("string").fillna("Missing")
        + "__"
        + df["Water_Source"].astype("string").fillna("Missing")
    )
    df["Crop_Type__Season"] = (
        df["Crop_Type"].astype("string").fillna("Missing")
        + "__"
        + df["Season"].astype("string").fillna("Missing")
    )
    df["Soil_Type__Region"] = (
        df["Soil_Type"].astype("string").fillna("Missing")
        + "__"
        + df["Region"].astype("string").fillna("Missing")
    )
    df["Crop_Growth_Stage__Water_Source"] = (
        df["Crop_Growth_Stage"].astype("string").fillna("Missing")
        + "__"
        + df["Water_Source"].astype("string").fillna("Missing")
    )
    df["Crop_Growth_Stage__Irrigation_Type"] = (
        growth_stage
        + "__"
        + irrigation_type
    )
    df["Mulching_Used__Irrigation_Type"] = (
        mulch_used
        + "__"
        + irrigation_type
    )
    df["Soil_Type__Crop_Type"] = (
        df["Soil_Type"].astype("string").fillna("Missing")
        + "__"
        + df["Crop_Type"].astype("string").fillna("Missing")
    )
    df["Region__Season"] = (
        df["Region"].astype("string").fillna("Missing")
        + "__"
        + season
    )
    df["Crop_Growth_Stage__Mulching_Used__Water_Source"] = (
        growth_stage + "__" + mulch_used + "__" + water_source
    )
    df["Crop_Growth_Stage__Mulching_Used__Irrigation_Type"] = (
        growth_stage + "__" + mulch_used + "__" + irrigation_type
    )
    df["Crop_Growth_Stage__Mulching_Used__Season"] = (
        growth_stage + "__" + mulch_used + "__" + season
    )
    df["Crop_Growth_Stage__Season"] = growth_stage + "__" + season
    df["Crop_Type__Crop_Growth_Stage"] = crop_type + "__" + growth_stage
    df["Mulching_Used__Water_Source"] = mulch_used + "__" + water_source
    df["Water_Source__Region"] = water_source + "__" + region

    return df


def infer_class_names(labels: pd.Series) -> list[str]:
    observed = labels.dropna().astype(str).unique().tolist()
    ordered = [label for label in TARGET_ORDER if label in observed]
    remaining = sorted(label for label in observed if label not in TARGET_ORDER)
    return ordered + remaining


def infer_feature_columns(
    train_df: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> tuple[list[str], list[str]]:
    feature_columns = [
        column
        for column in train_df.columns
        if column not in {target_column, id_column}
    ]
    categorical_columns = (
        train_df[feature_columns]
        .select_dtypes(include=["object", "category", "string"])
        .columns.tolist()
    )
    return feature_columns, categorical_columns


def build_model_params(
    args: argparse.Namespace,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": args.loss_function,
        "eval_metric": "TotalF1:average=Macro",
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "min_data_in_leaf": args.min_data_in_leaf,
        "random_strength": args.random_strength,
        "subsample": args.subsample,
        "bootstrap_type": "Bernoulli",
        "border_count": args.border_count,
        "one_hot_max_size": args.one_hot_max_size,
        "auto_class_weights": "Balanced",
        "random_seed": args.random_state,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 100,
    }
    if overrides:
        params.update(overrides)
    if params.get("class_weights") is not None:
        params.pop("auto_class_weights", None)
    if params.get("auto_class_weights") is None:
        params.pop("auto_class_weights", None)
    return params


def build_pool(
    features: pd.DataFrame,
    labels: pd.Series | None,
    categorical_columns: list[str],
) -> Pool:
    prepared_features = features.copy()
    for column in categorical_columns:
        prepared_features[column] = (
            prepared_features[column].astype("string").fillna("Missing").astype(str)
        )
    if categorical_columns:
        return Pool(prepared_features, label=labels, cat_features=categorical_columns)
    return Pool(prepared_features, label=labels)


def flatten_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.asarray(predictions).reshape(-1)


def resolved_best_iteration(model: CatBoostClassifier) -> int:
    best_iteration = int(model.get_best_iteration())
    if best_iteration < 0:
        return int(model.tree_count_) - 1
    return best_iteration


def print_validation_summary(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: list[str],
    balanced_accuracy: float,
    best_iteration: int,
    tree_count: int,
    train_rows: int,
    validation_rows: int,
    validation_size: float,
    random_state: int,
) -> dict[str, float]:
    per_class_recall_values = recall_score(
        y_true,
        y_pred,
        labels=class_names,
        average=None,
        zero_division=0,
    )
    per_class_recall = {
        label: float(score)
        for label, score in zip(class_names, per_class_recall_values, strict=True)
    }
    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    weakest_label = min(class_names, key=lambda label: per_class_recall[label])
    weakest_score = per_class_recall[weakest_label]

    print(f"train_rows: {train_rows}")
    print(f"validation_rows: {validation_rows}")
    print(f"validation_fraction: {validation_size}")
    print(f"validation_seed: {random_state}")
    print(
        "leakage_guard: fit only sees the training split; "
        "the validation split is held out for early stopping and scoring."
    )
    print(f"val_balanced_accuracy_score: {balanced_accuracy:.10f}")
    print(f"best_iteration: {best_iteration}")
    print(f"tree_count: {tree_count}")
    print("per_class_recall:")
    for label in class_names:
        print(f"  {label}: {per_class_recall[label]:.6f}")
    print(f"weakest_class_recall: {weakest_label}={weakest_score:.6f}")
    print(f"confusion_matrix_labels: {', '.join(class_names)}")
    print("confusion_matrix:")
    for row in matrix:
        print("  " + " ".join(str(int(value)) for value in row))

    return per_class_recall


def stacking_variant_specs() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("mc", {}),
        ("ova", {"loss_function": "MultiClassOneVsAll"}),
        (
            "shallow",
            {
                "depth": 6,
                "l2_leaf_reg": 12.0,
                "subsample": 0.95,
                "min_data_in_leaf": 50,
                "random_strength": 0.4,
            },
        ),
    ]


def merged_model_overrides(
    base_overrides: dict[str, Any] | None,
    extra_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if base_overrides:
        merged.update(base_overrides)
    if extra_overrides:
        merged.update(extra_overrides)
    return merged


def predict_aligned_probabilities(
    model: CatBoostClassifier,
    features: pd.DataFrame,
    categorical_columns: list[str],
    class_names: list[str],
) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(build_pool(features, None, categorical_columns))
    )
    model_classes = [str(label) for label in model.classes_]
    class_lookup = {label: index for index, label in enumerate(model_classes)}
    aligned = np.zeros((len(features), len(class_names)), dtype=np.float32)
    for index, label in enumerate(class_names):
        aligned[:, index] = probabilities[:, class_lookup[label]]
    return aligned


def resolve_stacking_signal_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in STACKING_SIGNAL_COLUMNS if column in features.columns]


def build_stacking_matrix(
    probability_blocks: list[np.ndarray],
    features: pd.DataFrame,
    raw_signal_columns: list[str],
) -> np.ndarray:
    matrix_parts = [np.asarray(block, dtype=np.float32) for block in probability_blocks]
    if raw_signal_columns:
        raw_block = (
            features[raw_signal_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        matrix_parts.append(raw_block)
    return np.hstack(matrix_parts).astype(np.float32, copy=False)


def fit_stacking_bundle(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_columns: list[str],
    class_names: list[str],
    args: argparse.Namespace,
    base_model_overrides: dict[str, Any] | None = None,
    primary_model: CatBoostClassifier | None = None,
    reference_features: pd.DataFrame | None = None,
    reference_labels: pd.Series | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if args.stacking_mode == "none":
        return None, {"enabled": False, "mode": args.stacking_mode}

    sample_limit = len(x_train) if args.stacking_max_rows <= 0 else args.stacking_max_rows
    stack_x_train, stack_y_train = stratified_subsample(
        x_train,
        y_train,
        max_rows=sample_limit,
        random_state=args.random_state + 59,
    )
    stack_x_train = stack_x_train.reset_index(drop=True)
    stack_y_train = stack_y_train.reset_index(drop=True)

    variant_specs = stacking_variant_specs()
    oof_probability_blocks = [
        np.zeros((len(stack_x_train), len(class_names)), dtype=np.float32)
        for _ in variant_specs
    ]
    fold_best_iterations: dict[str, list[int]] = {
        name: [] for name, _ in variant_specs
    }
    fold_scores: list[dict[str, Any]] = []
    splitter = StratifiedKFold(
        n_splits=args.stacking_folds,
        shuffle=True,
        random_state=args.random_state + 59,
    )

    for fold_index, (fold_train_idx, fold_oof_idx) in enumerate(
        splitter.split(stack_x_train, stack_y_train),
        start=1,
    ):
        fold_train_x = stack_x_train.iloc[fold_train_idx].reset_index(drop=True)
        fold_train_y = stack_y_train.iloc[fold_train_idx].reset_index(drop=True)
        fold_oof_x = stack_x_train.iloc[fold_oof_idx].reset_index(drop=True)
        fold_oof_y = stack_y_train.iloc[fold_oof_idx].reset_index(drop=True)
        fold_train_pool = build_pool(fold_train_x, fold_train_y, categorical_columns)
        fold_oof_pool = build_pool(fold_oof_x, fold_oof_y, categorical_columns)

        for block_index, (variant_name, variant_overrides) in enumerate(variant_specs):
            fold_model = CatBoostClassifier(
                **build_model_params(
                    args,
                    overrides=merged_model_overrides(
                        base_model_overrides,
                        variant_overrides,
                    ),
                )
            )
            fold_model.fit(
                fold_train_pool,
                eval_set=fold_oof_pool,
                use_best_model=True,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            oof_probability_blocks[block_index][fold_oof_idx] = predict_aligned_probabilities(
                fold_model,
                fold_oof_x,
                categorical_columns,
                class_names,
            )
            best_iteration = resolved_best_iteration(fold_model)
            fold_best_iterations[variant_name].append(best_iteration)
            fold_predictions = flatten_predictions(fold_model.predict(fold_oof_pool))
            fold_scores.append(
                {
                    "fold": fold_index,
                    "variant": variant_name,
                    "score": float(balanced_accuracy_score(fold_oof_y, fold_predictions)),
                    "best_iteration": best_iteration,
                }
            )

    raw_signal_columns = resolve_stacking_signal_columns(stack_x_train)
    meta_train_matrix = build_stacking_matrix(
        oof_probability_blocks,
        stack_x_train,
        raw_signal_columns,
    )
    meta_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        C=args.stacking_logreg_c,
    )
    meta_model.fit(meta_train_matrix, stack_y_train)

    full_models: dict[str, CatBoostClassifier] = {}
    refit_iterations: dict[str, int] = {}
    for variant_name, variant_overrides in variant_specs:
        if variant_name == "mc" and primary_model is not None:
            full_models[variant_name] = primary_model
            refit_iterations[variant_name] = int(primary_model.tree_count_)
            continue

        full_overrides = merged_model_overrides(base_model_overrides, variant_overrides)
        if reference_features is None or reference_labels is None:
            average_best_iteration = int(
                np.mean(fold_best_iterations[variant_name])
            ) if fold_best_iterations[variant_name] else int(args.iterations // 2)
            full_overrides["iterations"] = max(1, average_best_iteration + 1)
            full_model = CatBoostClassifier(
                **build_model_params(args, overrides=full_overrides)
            )
            full_model.fit(build_pool(x_train, y_train, categorical_columns))
        else:
            full_model = CatBoostClassifier(
                **build_model_params(args, overrides=full_overrides)
            )
            full_model.fit(
                build_pool(x_train, y_train, categorical_columns),
                eval_set=build_pool(
                    reference_features,
                    reference_labels,
                    categorical_columns,
                ),
                use_best_model=True,
                early_stopping_rounds=args.early_stopping_rounds,
            )
        full_models[variant_name] = full_model
        refit_iterations[variant_name] = int(full_model.tree_count_)

    summary = {
        "enabled": True,
        "mode": args.stacking_mode,
        "sample_rows": len(stack_x_train),
        "folds": args.stacking_folds,
        "logreg_c": args.stacking_logreg_c,
        "variant_names": [name for name, _ in variant_specs],
        "raw_signal_columns": raw_signal_columns,
        "meta_feature_count": int(meta_train_matrix.shape[1]),
        "oof_fold_scores": fold_scores,
        "refit_iterations": refit_iterations,
    }
    return (
        {
            "mode": args.stacking_mode,
            "variant_names": [name for name, _ in variant_specs],
            "class_names": class_names[:],
            "categorical_columns": categorical_columns[:],
            "raw_signal_columns": raw_signal_columns,
            "models": full_models,
            "meta_model": meta_model,
            "refit_iterations": refit_iterations,
        },
        summary,
    )


def predict_with_stacking_bundle(
    features: pd.DataFrame,
    bundle: dict[str, Any],
) -> np.ndarray:
    class_names = list(bundle["class_names"])
    categorical_columns = list(bundle["categorical_columns"])
    probability_blocks = [
        predict_aligned_probabilities(
            bundle["models"][variant_name],
            features,
            categorical_columns,
            class_names,
        )
        for variant_name in bundle["variant_names"]
    ]
    meta_matrix = build_stacking_matrix(
        probability_blocks,
        features,
        list(bundle["raw_signal_columns"]),
    )
    predictions = bundle["meta_model"].predict(meta_matrix)
    return np.asarray(predictions, dtype=object)


def predict_with_class_scales(
    probabilities: np.ndarray,
    class_labels: list[str],
    class_scales: dict[str, float],
) -> np.ndarray:
    scales = np.asarray(
        [class_scales.get(label, 1.0) for label in class_labels],
        dtype=float,
    )
    scaled = probabilities * scales
    indices = np.argmax(scaled, axis=1)
    return np.asarray(class_labels, dtype=object)[indices]


def search_class_scales(
    probabilities: np.ndarray,
    y_true: pd.Series,
    class_labels: list[str],
) -> tuple[float, dict[str, float]]:
    best_scales = {label: 1.0 for label in class_labels}
    best_predictions = np.asarray(class_labels, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]
    best_score = balanced_accuracy_score(y_true, best_predictions)

    if "Medium" not in class_labels or "High" not in class_labels:
        return float(best_score), best_scales

    for medium_scale in np.arange(0.90, 1.101, 0.01):
        for high_scale in np.arange(0.70, 1.501, 0.01):
            candidate_scales = {label: 1.0 for label in class_labels}
            candidate_scales["Medium"] = float(round(medium_scale, 2))
            candidate_scales["High"] = float(round(high_scale, 2))
            predictions = predict_with_class_scales(
                probabilities=probabilities,
                class_labels=class_labels,
                class_scales=candidate_scales,
            )
            score = balanced_accuracy_score(y_true, predictions)
            if score > best_score + 1e-12:
                best_score = float(score)
                best_scales = candidate_scales

    return best_score, best_scales


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2)


def stratified_subsample(
    features: pd.DataFrame,
    labels: pd.Series,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows <= 0 or len(features) <= max_rows:
        return features, labels

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=max_rows,
        random_state=random_state,
    )
    indices, _ = next(splitter.split(features, labels))
    return (
        features.iloc[indices].reset_index(drop=True),
        labels.iloc[indices].reset_index(drop=True),
    )


def compute_balanced_class_weights(
    labels: pd.Series,
    class_names: list[str],
) -> list[float]:
    counts = labels.value_counts()
    total = float(len(labels))
    class_count = float(len(class_names))
    weights: list[float] = []
    for class_name in class_names:
        count = float(counts.get(class_name, 0.0))
        if count <= 0:
            weights.append(1.0)
            continue
        weights.append(total / (class_count * count))
    return weights


def parse_extra_trees_max_features(raw_value: str) -> str | float | int | None:
    value = str(raw_value).strip().lower()
    if value in {"sqrt", "log2", "auto"}:
        return value
    if value in {"all", "none"}:
        return None
    numeric = float(value)
    if numeric.is_integer() and numeric >= 1.0:
        return int(numeric)
    return numeric


def build_extra_trees_params(args: argparse.Namespace) -> dict[str, Any]:
    max_depth = None if args.extra_trees_max_depth <= 0 else args.extra_trees_max_depth
    return {
        "n_estimators": args.extra_trees_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": args.extra_trees_min_samples_leaf,
        "max_features": parse_extra_trees_max_features(args.extra_trees_max_features),
        "criterion": "gini",
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": args.random_state,
    }


def fit_extra_trees_bundle(
    features: pd.DataFrame,
    labels: pd.Series,
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.ensemble_mode == "none":
        return None

    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]
    numeric_frame = features[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_fill_values: dict[str, float] = {}
    for column in numeric_columns:
        median = numeric_frame[column].median()
        numeric_fill_values[column] = (
            0.0 if pd.isna(median) else float(median)
        )
    numeric_frame = numeric_frame.fillna(numeric_fill_values)

    encoder: OrdinalEncoder | None = None
    encoded_frame = pd.DataFrame(index=features.index)
    if categorical_columns:
        categorical_frame = (
            features[categorical_columns].astype("string").fillna("Missing")
        )
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
            dtype=np.float32,
        )
        encoded_values = encoder.fit_transform(categorical_frame)
        encoded_frame = pd.DataFrame(
            encoded_values,
            columns=categorical_columns,
            index=features.index,
        )

    prepared = pd.concat([numeric_frame, encoded_frame], axis=1)
    feature_order = prepared.columns.tolist()
    prepared = prepared.astype(np.float32)

    model = ExtraTreesClassifier(**build_extra_trees_params(args))
    model.fit(prepared, labels)
    return {
        "model": model,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns[:],
        "numeric_fill_values": numeric_fill_values,
        "encoder": encoder,
        "feature_order": feature_order,
        "params": build_extra_trees_params(args),
    }


def transform_extra_trees_features(
    features: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    numeric_columns = list(bundle["numeric_columns"])
    categorical_columns = list(bundle["categorical_columns"])
    numeric_frame = features[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.fillna(bundle["numeric_fill_values"])

    encoded_frame = pd.DataFrame(index=features.index)
    encoder = bundle["encoder"]
    if categorical_columns and encoder is not None:
        categorical_frame = (
            features[categorical_columns].astype("string").fillna("Missing")
        )
        encoded_values = encoder.transform(categorical_frame)
        encoded_frame = pd.DataFrame(
            encoded_values,
            columns=categorical_columns,
            index=features.index,
        )

    prepared = pd.concat([numeric_frame, encoded_frame], axis=1)
    prepared = prepared.loc[:, bundle["feature_order"]]
    return prepared.astype(np.float32)


def predict_extra_trees(
    features: pd.DataFrame,
    bundle: dict[str, Any] | None,
) -> np.ndarray | None:
    if bundle is None:
        return None
    prepared = transform_extra_trees_features(features, bundle)
    predictions = bundle["model"].predict(prepared)
    return np.asarray(predictions, dtype=object)


def apply_ensemble_selector(
    catboost_predictions: np.ndarray,
    features: pd.DataFrame,
    bundle: dict[str, Any] | None,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.ensemble_mode == "none" or bundle is None:
        return np.asarray(catboost_predictions, dtype=object)

    extra_predictions = predict_extra_trees(features, bundle)
    if extra_predictions is None:
        return np.asarray(catboost_predictions, dtype=object)

    if args.ensemble_mode == "cat_high_else_extra":
        use_catboost = np.asarray(catboost_predictions, dtype=object) == "High"
    else:
        raise ValueError(f"Unsupported ensemble mode: {args.ensemble_mode}")

    return np.where(use_catboost, catboost_predictions, extra_predictions)


def predict_labels(
    model: CatBoostClassifier,
    features: pd.DataFrame,
    categorical_columns: list[str],
    class_scales: dict[str, float],
    rule_overrides: list[dict[str, Any]],
    args: argparse.Namespace,
    ensemble_bundle: dict[str, Any] | None = None,
    stacking_bundle: dict[str, Any] | None = None,
) -> np.ndarray:
    if stacking_bundle is not None:
        predictions = predict_with_stacking_bundle(features, stacking_bundle)
        return apply_rule_overrides(
            predictions=predictions,
            features=features,
            rules=rule_overrides,
        )

    feature_pool = build_pool(features, None, categorical_columns)
    class_labels = [str(label) for label in model.classes_]
    probabilities = np.asarray(model.predict_proba(feature_pool))
    predictions = predict_with_class_scales(
        probabilities=probabilities,
        class_labels=class_labels,
        class_scales=class_scales,
    )
    predictions = apply_ensemble_selector(
        catboost_predictions=predictions,
        features=features,
        bundle=ensemble_bundle,
        args=args,
    )
    return apply_rule_overrides(
        predictions=predictions,
        features=features,
        rules=rule_overrides,
    )


def learn_rule_overrides(
    features: pd.DataFrame,
    labels: pd.Series,
    class_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not args.rule_overrides:
        return []

    candidate_columns: list[str] = []
    for column in features.columns:
        unique_values = pd.Series(features[column]).dropna().unique().tolist()
        if not unique_values:
            continue
        if set(unique_values).issubset({0, 1, 0.0, 1.0, False, True}):
            candidate_columns.append(column)

    rules: list[dict[str, Any]] = []
    for column in candidate_columns:
        mask = features[column].astype(int) == 1
        support = int(mask.sum())
        if support < args.rule_override_min_support:
            continue
        pocket_labels = labels.loc[mask]
        distribution = pocket_labels.value_counts(normalize=True)
        target_class = distribution.idxmax()
        purity = float(distribution.iloc[0])
        if purity < args.rule_override_min_purity:
            continue
        rules.append(
            {
                "feature": column,
                "target_class": str(target_class),
                "purity": purity,
                "support": support,
            }
        )

    rules.sort(key=lambda item: (-item["purity"], -item["support"], item["feature"]))
    rules = rules[: args.rule_override_max_rules]

    print(f"rule_override_count: {len(rules)}")
    if rules:
        print("rule_overrides:")
        for rule in rules:
            print(
                "  "
                + f"{rule['feature']} -> {rule['target_class']} "
                + f"(purity={rule['purity']:.6f}, support={rule['support']})"
            )
    return rules


def apply_rule_overrides(
    predictions: np.ndarray,
    features: pd.DataFrame,
    rules: list[dict[str, Any]],
) -> np.ndarray:
    if not rules:
        return predictions

    overridden = pd.Series(predictions, index=features.index, dtype="object")
    consumed = pd.Series(False, index=features.index)
    for rule in rules:
        mask = (features[rule["feature"]].astype(int) == 1) & (~consumed)
        if not mask.any():
            continue
        overridden.loc[mask] = rule["target_class"]
        consumed.loc[mask] = True
    return overridden.to_numpy()


def learn_class_scale_map(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_columns: list[str],
    args: argparse.Namespace,
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    if not args.class_scale_search:
        return {}

    scale_train_x, scale_cal_x, scale_train_y, scale_cal_y = train_test_split(
        x_train,
        y_train,
        test_size=args.validation_size,
        random_state=args.random_state + 17,
        stratify=y_train,
    )
    scale_model = CatBoostClassifier(
        **build_model_params(args, overrides=model_overrides)
    )
    scale_train_pool = build_pool(
        scale_train_x.reset_index(drop=True),
        scale_train_y.reset_index(drop=True),
        categorical_columns,
    )
    scale_cal_pool = build_pool(
        scale_cal_x.reset_index(drop=True),
        scale_cal_y.reset_index(drop=True),
        categorical_columns,
    )
    scale_model.fit(
        scale_train_pool,
        eval_set=scale_cal_pool,
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    class_labels = [str(label) for label in scale_model.classes_]
    probabilities = np.asarray(scale_model.predict_proba(scale_cal_pool))
    best_score, class_scales = search_class_scales(
        probabilities=probabilities,
        y_true=scale_cal_y.reset_index(drop=True),
        class_labels=class_labels,
    )
    print(f"class_scale_search_score: {best_score:.10f}")
    print(
        "class_scales: "
        + " ".join(
            f"{label}={class_scales.get(label, 1.0):.2f}" for label in class_labels
        )
    )
    return class_scales


def build_boruta_frame(
    features: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]
    numeric_frame = features[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.fillna(numeric_frame.median())
    proxy_to_original = {column: column for column in numeric_columns}

    if not categorical_columns:
        return numeric_frame, proxy_to_original

    categorical_frame = (
        features[categorical_columns].astype("string").fillna("Missing")
    )
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
        dtype=np.float64,
    )
    ordinal_values = encoder.fit_transform(categorical_frame)
    ordinal_frame = pd.DataFrame(
        ordinal_values,
        columns=[f"{column}__ord" for column in categorical_columns],
        index=features.index,
    )
    frequency_columns: dict[str, pd.Series] = {}
    for column in categorical_columns:
        frequencies = categorical_frame[column].value_counts(normalize=True)
        frequency_columns[f"{column}__freq"] = (
            categorical_frame[column].map(frequencies).fillna(0.0).astype(float)
        )

    frequency_frame = pd.DataFrame(frequency_columns, index=features.index)
    for column in categorical_columns:
        proxy_to_original[f"{column}__ord"] = column
        proxy_to_original[f"{column}__freq"] = column

    boruta_frame = pd.concat(
        [numeric_frame, ordinal_frame, frequency_frame],
        axis=1,
    )
    return boruta_frame, proxy_to_original


def select_features_with_boruta(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
    categorical_columns: list[str],
    class_names: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    if not feature_columns:
        summary = {
            "enabled": True,
            "sample_rows": 0,
            "proxy_source": "no_candidate_features",
            "selected_feature_count": 0,
            "selected_features": [],
            "boruta_max_iter": args.boruta_max_iter,
            "boruta_perc": args.boruta_perc,
        }
        return [], summary

    sample_x, sample_y = stratified_subsample(
        x_train,
        y_train,
        max_rows=args.boruta_max_rows,
        random_state=args.random_state,
    )
    boruta_frame, proxy_to_original = build_boruta_frame(sample_x, categorical_columns)
    label_codes = pd.Categorical(sample_y, categories=class_names).codes
    estimator = RandomForestClassifier(
        n_estimators=BORUTA_RF_TREES,
        max_depth=BORUTA_RF_DEPTH,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.random_state,
    )
    selector = BorutaPy(
        estimator=estimator,
        n_estimators="auto",
        perc=args.boruta_perc,
        alpha=0.05,
        two_step=True,
        max_iter=args.boruta_max_iter,
        random_state=args.random_state,
        verbose=0,
    )
    selector.fit(boruta_frame.to_numpy(dtype=np.float32), label_codes)

    accepted_proxies = [
        column
        for column, keep in zip(boruta_frame.columns, selector.support_, strict=True)
        if keep
    ]
    proxy_source = "confirmed"
    if len(accepted_proxies) < 8:
        tentative = [
            column
            for column, keep in zip(
                boruta_frame.columns,
                selector.support_weak_,
                strict=True,
            )
            if keep
        ]
        accepted_proxies = sorted(set(accepted_proxies + tentative))
        proxy_source = "confirmed_plus_tentative"

    selected_original = [
        column
        for column in feature_columns
        if column in {proxy_to_original[proxy] for proxy in accepted_proxies}
    ]
    if not selected_original:
        selected_original = feature_columns[:]
        proxy_source = "fallback_all_features"

    print(f"boruta_sample_rows: {len(sample_x)}")
    print(f"boruta_proxy_source: {proxy_source}")
    print(f"boruta_selected_feature_count: {len(selected_original)}")
    print("boruta_selected_features:")
    for column in selected_original:
        print(f"  {column}")

    summary = {
        "enabled": True,
        "sample_rows": len(sample_x),
        "proxy_source": proxy_source,
        "selected_feature_count": len(selected_original),
        "selected_features": selected_original,
        "boruta_max_iter": args.boruta_max_iter,
        "boruta_perc": args.boruta_perc,
    }
    return selected_original, summary


def prune_positive_contributor_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    protected_features: set[str],
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    candidate_features = [
        column
        for column in selected_features
        if column not in protected_features and column not in categorical_columns
    ]
    summary: dict[str, Any] = {
        "enabled": True,
        "sample_rows": 0,
        "baseline_score": None,
        "candidate_feature_count": len(candidate_features),
        "kept_feature_count": len(candidate_features),
        "kept_features": candidate_features[:],
        "dropped_features": [],
    }
    if not candidate_features:
        summary["status"] = "no_numeric_engineered_candidates"
        return selected_features, summary

    sample_x, sample_y = stratified_subsample(
        x_train[selected_features],
        y_train,
        max_rows=args.positive_pruning_max_rows,
        random_state=args.random_state + 31,
    )
    prune_train_x, prune_cal_x, prune_train_y, prune_cal_y = train_test_split(
        sample_x,
        sample_y,
        test_size=args.validation_size,
        random_state=args.random_state + 31,
        stratify=sample_y,
    )
    prune_train_x = prune_train_x.reset_index(drop=True)
    prune_cal_x = prune_cal_x.reset_index(drop=True)
    prune_train_y = prune_train_y.reset_index(drop=True)
    prune_cal_y = prune_cal_y.reset_index(drop=True)
    selected_categorical = [
        column for column in categorical_columns if column in selected_features
    ]

    model = CatBoostClassifier(
        **build_model_params(
            args,
            overrides=POSITIVE_PRUNING_PROBE_OVERRIDES,
        )
    )
    model.fit(
        build_pool(prune_train_x, prune_train_y, selected_categorical),
        eval_set=build_pool(prune_cal_x, prune_cal_y, selected_categorical),
        use_best_model=True,
        early_stopping_rounds=max(40, args.early_stopping_rounds // 2),
    )
    baseline_predictions = flatten_predictions(
        model.predict(build_pool(prune_cal_x, None, selected_categorical))
    )
    baseline_score = float(balanced_accuracy_score(prune_cal_y, baseline_predictions))
    rng = np.random.default_rng(args.random_state + 31)
    kept_candidates: list[str] = []
    dropped_features: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []

    for column in candidate_features:
        permuted = prune_cal_x.copy()
        shuffled_values = permuted[column].to_numpy(copy=True)
        rng.shuffle(shuffled_values)
        permuted[column] = shuffled_values
        permuted_predictions = flatten_predictions(
            model.predict(build_pool(permuted, None, selected_categorical))
        )
        permuted_score = float(balanced_accuracy_score(prune_cal_y, permuted_predictions))
        score_delta = baseline_score - permuted_score
        contribution_rows.append(
            {
                "feature": column,
                "score_delta": score_delta,
                "permuted_score": permuted_score,
            }
        )
        if score_delta > 0.0:
            kept_candidates.append(column)
        else:
            dropped_features.append(
                {
                    "feature": column,
                    "score_delta": score_delta,
                    "permuted_score": permuted_score,
                }
            )

    selected_lookup = protected_features | set(kept_candidates)
    pruned_features = [
        column for column in selected_features if column in selected_lookup
    ]
    contribution_rows.sort(
        key=lambda item: (-item["score_delta"], item["feature"])
    )
    dropped_features.sort(key=lambda item: (item["score_delta"], item["feature"]))

    print(f"positive_pruning_sample_rows: {len(sample_x)}")
    print(f"positive_pruning_baseline_score: {baseline_score:.10f}")
    print(f"positive_pruning_candidate_feature_count: {len(candidate_features)}")
    print(f"positive_pruning_kept_feature_count: {len(kept_candidates)}")
    print("positive_pruning_feature_deltas:")
    for row in contribution_rows:
        print(f"  {row['feature']}: {row['score_delta']:.8f}")

    summary.update(
        {
            "sample_rows": len(sample_x),
            "baseline_score": baseline_score,
            "kept_feature_count": len(kept_candidates),
            "kept_features": kept_candidates,
            "dropped_features": dropped_features,
            "feature_deltas": contribution_rows,
            "status": "completed",
        }
    )
    return pruned_features, summary


def tune_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    class_names: list[str],
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.optuna_trials <= 0:
        return {}, {"enabled": False, "completed_trials": 0}

    tune_x_train, tune_y_train = stratified_subsample(
        x_train,
        y_train,
        max_rows=args.optuna_train_max_rows,
        random_state=args.random_state,
    )
    inner_train_x, inner_val_x, inner_train_y, inner_val_y = train_test_split(
        tune_x_train,
        tune_y_train,
        test_size=args.validation_size,
        random_state=args.random_state + 23,
        stratify=tune_y_train,
    )
    inner_train_x = inner_train_x.reset_index(drop=True)
    inner_val_x = inner_val_x.reset_index(drop=True)
    inner_train_y = inner_train_y.reset_index(drop=True)
    inner_val_y = inner_val_y.reset_index(drop=True)
    train_pool = build_pool(inner_train_x, inner_train_y, categorical_columns)
    validation_pool = build_pool(inner_val_x, inner_val_y, categorical_columns)
    base_class_weights = compute_balanced_class_weights(inner_train_y, class_names)

    def objective(trial: optuna.Trial) -> float:
        overrides: dict[str, Any] = {
            "iterations": trial.suggest_int("iterations", 700, 1800),
            "loss_function": trial.suggest_categorical(
                "loss_function",
                ["MultiClass", "MultiClassOneVsAll"],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.02,
                0.12,
                log=True,
            ),
            "depth": trial.suggest_int("depth", 6, 10),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                2.0,
                20.0,
                log=True,
            ),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 80),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.5),
            "border_count": trial.suggest_int("border_count", 64, 255),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 12),
            "verbose": False,
        }
        if args.optuna_tune_class_weights:
            multipliers = [
                trial.suggest_float("class_weight_low_mult", 0.8, 1.4),
                trial.suggest_float("class_weight_medium_mult", 0.8, 1.8),
                trial.suggest_float("class_weight_high_mult", 1.0, 3.0),
            ]
            overrides["class_weights"] = [
                base_weight * multiplier
                for base_weight, multiplier in zip(
                    base_class_weights,
                    multipliers,
                    strict=True,
                )
            ]
            overrides["auto_class_weights"] = None

        params = build_model_params(args, overrides=overrides)
        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=validation_pool,
            use_best_model=True,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        predictions = flatten_predictions(model.predict(validation_pool))
        score = balanced_accuracy_score(inner_val_y, predictions)
        trial.set_user_attr("best_iteration", resolved_best_iteration(model))
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.random_state),
    )
    timeout = args.optuna_timeout_seconds or None
    study.optimize(
        objective,
        n_trials=args.optuna_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    best_params = {key: value for key, value in study.best_params.items()}
    class_weight_multipliers: dict[str, float] | None = None
    if args.optuna_tune_class_weights:
        class_weight_multipliers = {
            "Low": float(best_params.pop("class_weight_low_mult")),
            "Medium": float(best_params.pop("class_weight_medium_mult")),
            "High": float(best_params.pop("class_weight_high_mult")),
        }
        best_params["class_weights"] = [
            base_weight * class_weight_multipliers[class_name]
            for class_name, base_weight in zip(
                class_names,
                base_class_weights,
                strict=True,
            )
        ]
        best_params["auto_class_weights"] = None

    print(f"tuning_train_rows: {len(tune_x_train)}")
    print(f"tuning_inner_train_rows: {len(inner_train_x)}")
    print(f"tuning_inner_validation_rows: {len(inner_val_x)}")
    print(f"optuna_trials_completed: {len(study.trials)}")
    print(f"optuna_best_validation_score: {study.best_value:.10f}")
    print(
        "optuna_best_params: "
        + json.dumps(json_ready(best_params), sort_keys=True)
    )

    summary = {
        "enabled": True,
        "tuning_train_rows": len(tune_x_train),
        "tuning_inner_train_rows": len(inner_train_x),
        "tuning_inner_validation_rows": len(inner_val_x),
        "completed_trials": len(study.trials),
        "best_validation_score": float(study.best_value),
        "best_params": best_params,
        "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
    }
    if class_weight_multipliers is not None:
        summary["class_weight_multipliers"] = class_weight_multipliers
    return best_params, summary


def fit_validation_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    class_names: list[str],
    categorical_columns: list[str],
    args: argparse.Namespace,
    model_overrides: dict[str, Any] | None = None,
) -> tuple[
    CatBoostClassifier,
    dict[str, Any] | None,
    dict[str, Any] | None,
    float,
    dict[str, float],
    list[dict[str, Any]],
    dict[str, float],
    dict[str, Any],
]:
    model = CatBoostClassifier(**build_model_params(args, overrides=model_overrides))
    train_pool = build_pool(x_train, y_train, categorical_columns)
    validation_pool = build_pool(x_val, y_val, categorical_columns)

    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    stacking_bundle: dict[str, Any] | None = None
    stacking_summary: dict[str, Any] = {"enabled": False, "mode": args.stacking_mode}
    ensemble_bundle: dict[str, Any] | None = None
    class_scales: dict[str, float] = {}
    rule_overrides: list[dict[str, Any]] = []
    if args.stacking_mode == "none":
        ensemble_bundle = fit_extra_trees_bundle(
            features=x_train,
            labels=y_train,
            categorical_columns=categorical_columns,
            args=args,
        )
        class_scales = learn_class_scale_map(
            x_train=x_train,
            y_train=y_train,
            categorical_columns=categorical_columns,
            args=args,
            model_overrides=model_overrides,
        )
        rule_overrides = learn_rule_overrides(
            features=x_train,
            labels=y_train,
            class_names=class_names,
            args=args,
        )
    else:
        stacking_bundle, stacking_summary = fit_stacking_bundle(
            x_train=x_train,
            y_train=y_train,
            categorical_columns=categorical_columns,
            class_names=class_names,
            args=args,
            base_model_overrides=model_overrides,
            primary_model=model,
            reference_features=x_val,
            reference_labels=y_val,
        )

    validation_predictions = predict_labels(
        model=model,
        features=x_val,
        categorical_columns=categorical_columns,
        class_scales=class_scales,
        rule_overrides=rule_overrides,
        args=args,
        ensemble_bundle=ensemble_bundle,
        stacking_bundle=stacking_bundle,
    )
    balanced_accuracy = balanced_accuracy_score(y_val, validation_predictions)
    if ensemble_bundle is not None:
        base_predictions = predict_labels(
            model=model,
            features=x_val,
            categorical_columns=categorical_columns,
            class_scales=class_scales,
            rule_overrides=rule_overrides,
            args=argparse.Namespace(**{**vars(args), "ensemble_mode": "none"}),
            ensemble_bundle=None,
            stacking_bundle=None,
        )
        base_score = balanced_accuracy_score(y_val, base_predictions)
        changed_predictions = int(
            np.sum(np.asarray(base_predictions, dtype=object) != validation_predictions)
        )
        print(f"ensemble_mode: {args.ensemble_mode}")
        print(f"ensemble_base_catboost_score: {base_score:.10f}")
        print(f"ensemble_changed_predictions: {changed_predictions}")
    if stacking_bundle is not None:
        print(f"stacking_mode: {args.stacking_mode}")
        print(f"stacking_sample_rows: {stacking_summary['sample_rows']}")
        print(f"stacking_folds: {stacking_summary['folds']}")
        print(f"stacking_logreg_c: {stacking_summary['logreg_c']}")
        print("stacking_variants: " + ", ".join(stacking_summary["variant_names"]))
        if stacking_summary["raw_signal_columns"]:
            print(
                "stacking_raw_signals: "
                + ", ".join(stacking_summary["raw_signal_columns"])
            )
    best_iteration = resolved_best_iteration(model)
    tree_count = int(model.tree_count_)

    per_class_recall = print_validation_summary(
        y_true=y_val,
        y_pred=validation_predictions,
        class_names=class_names,
        balanced_accuracy=balanced_accuracy,
        best_iteration=best_iteration,
        tree_count=tree_count,
        train_rows=len(x_train),
        validation_rows=len(x_val),
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    return (
        model,
        ensemble_bundle,
        stacking_bundle,
        float(balanced_accuracy),
        per_class_recall,
        rule_overrides,
        class_scales,
        stacking_summary,
    )


def refit_full_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    iterations: int,
    args: argparse.Namespace,
    model_overrides: dict[str, Any] | None = None,
) -> CatBoostClassifier:
    model_params = build_model_params(args, overrides=model_overrides)
    model_params["iterations"] = iterations

    full_model = CatBoostClassifier(**model_params)
    full_train_pool = build_pool(
        train_df[feature_columns],
        train_df[args.target_column],
        categorical_columns,
    )
    full_model.fit(full_train_pool)
    return full_model


def write_submission(
    model: CatBoostClassifier,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    rule_overrides: list[dict[str, Any]],
    class_scales: dict[str, float],
    ensemble_bundle: dict[str, Any] | None,
    stacking_bundle: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    submission_df = pd.read_csv(args.sample_submission_path)

    if args.id_column not in test_df.columns:
        raise KeyError(f"Missing id column '{args.id_column}' in {args.test_path}")
    if args.target_column not in submission_df.columns:
        raise KeyError(
            f"Missing target column '{args.target_column}' in {args.sample_submission_path}"
        )

    submission_predictions = predict_labels(
        model=model,
        features=test_df[feature_columns],
        categorical_columns=categorical_columns,
        class_scales=class_scales,
        rule_overrides=rule_overrides,
        args=args,
        ensemble_bundle=ensemble_bundle,
        stacking_bundle=stacking_bundle,
    )

    submission_df[args.id_column] = test_df[args.id_column].to_numpy()
    submission_df[args.target_column] = submission_predictions

    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(args.submission_path, index=False)
    print(f"saved_submission_path: {args.submission_path}")


def main() -> None:
    args = parse_args()

    raw_train_df = pd.read_csv(args.train_path)
    train_df = engineer_features(raw_train_df, enabled=args.engineered_features)
    feature_columns, categorical_columns = infer_feature_columns(
        train_df=train_df,
        target_column=args.target_column,
        id_column=args.id_column,
    )
    class_names = infer_class_names(train_df[args.target_column])
    print(f"engineered_features_enabled: {args.engineered_features}")
    print(f"candidate_feature_count: {len(feature_columns)}")
    print(f"candidate_categorical_feature_count: {len(categorical_columns)}")
    raw_feature_columns, raw_categorical_columns = infer_feature_columns(
        train_df=raw_train_df,
        target_column=args.target_column,
        id_column=args.id_column,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        train_df[feature_columns],
        train_df[args.target_column],
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=train_df[args.target_column],
    )
    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    selected_features = feature_columns[:]
    boruta_summary: dict[str, Any] = {"enabled": False}
    if args.boruta:
        engineered_feature_columns = [
            column for column in feature_columns if column not in raw_feature_columns
        ]
        engineered_categorical_columns = [
            column
            for column in categorical_columns
            if column not in raw_categorical_columns
        ]
        boruta_candidate_features = engineered_feature_columns[:]
        boruta_selected_features, boruta_summary = select_features_with_boruta(
            x_train=x_train[boruta_candidate_features],
            y_train=y_train,
            feature_columns=boruta_candidate_features,
            categorical_columns=engineered_categorical_columns,
            class_names=class_names,
            args=args,
        )
        protected_features = set(raw_feature_columns)
        selected_lookup = protected_features | set(boruta_selected_features)
        selected_features = [
            column for column in feature_columns if column in selected_lookup
        ]
        boruta_summary["protected_feature_count"] = len(protected_features)
        boruta_summary["protected_features"] = [
            column for column in feature_columns if column in protected_features
        ]
        if args.boruta_positive_pruning:
            selected_features, positive_pruning_summary = prune_positive_contributor_features(
                x_train=x_train,
                y_train=y_train,
                selected_features=selected_features,
                protected_features=protected_features,
                categorical_columns=categorical_columns,
                args=args,
            )
            boruta_summary["positive_pruning"] = positive_pruning_summary

    selected_categorical_columns = [
        column for column in categorical_columns if column in selected_features
    ]
    print(f"selected_feature_count: {len(selected_features)}")
    print(f"selected_categorical_feature_count: {len(selected_categorical_columns)}")

    tuning_overrides, tuning_summary = tune_hyperparameters(
        x_train=x_train[selected_features],
        y_train=y_train,
        class_names=class_names,
        categorical_columns=selected_categorical_columns,
        args=args,
    )

    (
        validation_model,
        validation_ensemble_bundle,
        validation_stacking_bundle,
        balanced_accuracy,
        per_class_recall,
        rule_overrides,
        class_scales,
        stacking_summary,
    ) = fit_validation_model(
        x_train=x_train[selected_features],
        y_train=y_train,
        x_val=x_val[selected_features],
        y_val=y_val,
        class_names=class_names,
        categorical_columns=selected_categorical_columns,
        args=args,
        model_overrides=tuning_overrides or None,
    )

    final_model_params = build_model_params(args, overrides=tuning_overrides or None)
    ensemble_summary: dict[str, Any] = {
        "mode": args.ensemble_mode,
        "enabled": args.ensemble_mode != "none",
        "artifact_path": str(args.ensemble_artifact_path),
    }
    if validation_ensemble_bundle is not None:
        ensemble_summary["extra_trees_params"] = validation_ensemble_bundle["params"]
        ensemble_summary["feature_order"] = validation_ensemble_bundle["feature_order"]
    metadata: dict[str, Any] = {
        "target_column": args.target_column,
        "id_column": args.id_column,
        "validation_size": args.validation_size,
        "split_seed": args.random_state,
        "engineered_features_enabled": args.engineered_features,
        "boruta": boruta_summary,
        "optuna": tuning_summary,
        "ensemble": ensemble_summary,
        "stacking": stacking_summary,
        "rule_overrides": rule_overrides,
        "class_scales": class_scales,
        "feature_columns": selected_features,
        "categorical_columns": selected_categorical_columns,
        "class_names": class_names,
        "candidate_feature_count": len(feature_columns),
        "selected_feature_count": len(selected_features),
        "model_params": final_model_params,
        "best_iteration": resolved_best_iteration(validation_model),
        "tree_count": int(validation_model.tree_count_),
        "val_balanced_accuracy_score": float(balanced_accuracy),
        "per_class_recall": per_class_recall,
    }

    if args.skip_refit:
        metadata["refit_status"] = "skipped"
        save_json(metadata, args.metadata_path)
        print("refit_status: skipped")
        print(f"saved_metadata_path: {args.metadata_path}")
        return

    full_model = refit_full_model(
        train_df=train_df,
        feature_columns=selected_features,
        categorical_columns=selected_categorical_columns,
        iterations=int(validation_model.tree_count_),
        args=args,
        model_overrides=tuning_overrides or None,
    )
    full_stacking_bundle: dict[str, Any] | None = None
    full_ensemble_bundle: dict[str, Any] | None = None
    full_rule_overrides: list[dict[str, Any]] = []
    full_class_scales: dict[str, float] = {}
    full_stacking_summary = {"enabled": False, "mode": args.stacking_mode}
    if args.stacking_mode == "none":
        full_ensemble_bundle = fit_extra_trees_bundle(
            features=train_df[selected_features],
            labels=train_df[args.target_column],
            categorical_columns=selected_categorical_columns,
            args=args,
        )
        full_rule_overrides = learn_rule_overrides(
            features=train_df[selected_features],
            labels=train_df[args.target_column],
            class_names=class_names,
            args=args,
        )
        full_class_scales = learn_class_scale_map(
            x_train=train_df[selected_features],
            y_train=train_df[args.target_column],
            categorical_columns=selected_categorical_columns,
            args=args,
            model_overrides=tuning_overrides or None,
        )
    else:
        full_stacking_bundle, full_stacking_summary = fit_stacking_bundle(
            x_train=train_df[selected_features],
            y_train=train_df[args.target_column],
            categorical_columns=selected_categorical_columns,
            class_names=class_names,
            args=args,
            base_model_overrides=tuning_overrides or None,
            primary_model=full_model,
        )
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    full_model.save_model(args.model_path)
    print(f"saved_model_path: {args.model_path}")
    if full_ensemble_bundle is not None or full_stacking_bundle is not None:
        args.ensemble_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with args.ensemble_artifact_path.open("wb") as handle:
            pickle.dump(
                full_stacking_bundle if full_stacking_bundle is not None else full_ensemble_bundle,
                handle,
            )
        print(f"saved_ensemble_artifact_path: {args.ensemble_artifact_path}")

    raw_test_df = pd.read_csv(args.test_path)
    test_df = engineer_features(raw_test_df, enabled=args.engineered_features)
    write_submission(
        model=full_model,
        test_df=test_df,
        feature_columns=selected_features,
        categorical_columns=selected_categorical_columns,
        rule_overrides=full_rule_overrides,
        class_scales=full_class_scales,
        ensemble_bundle=full_ensemble_bundle,
        stacking_bundle=full_stacking_bundle,
        args=args,
    )

    metadata["refit_status"] = "completed"
    metadata["refit_iterations"] = int(validation_model.tree_count_)
    metadata["full_rule_overrides"] = full_rule_overrides
    metadata["full_class_scales"] = full_class_scales
    metadata["full_stacking"] = full_stacking_summary
    if full_ensemble_bundle is not None:
        metadata["ensemble"]["saved_artifact_path"] = str(args.ensemble_artifact_path)
    if full_stacking_bundle is not None:
        metadata["stacking"]["saved_artifact_path"] = str(args.ensemble_artifact_path)
    save_json(metadata, args.metadata_path)
    print(f"saved_metadata_path: {args.metadata_path}")


if __name__ == "__main__":
    main()
