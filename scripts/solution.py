from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from boruta import BorutaPy
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = REPO_ROOT / "Data" / "train.csv"
DEFAULT_TEST_PATH = REPO_ROOT / "Data" / "test.csv"
DEFAULT_SAMPLE_SUBMISSION_PATH = REPO_ROOT / "Data" / "sample_submission.csv"
DEFAULT_SUBMISSION_PATH = REPO_ROOT / "Predictions" / "prediction_irr_need.csv"
DEFAULT_MODEL_PATH = REPO_ROOT / "artifacts" / "irrigation_need_catboost.cbm"
DEFAULT_METADATA_PATH = REPO_ROOT / "artifacts" / "irrigation_need_metadata.json"
TARGET_ORDER = ["Low", "Medium", "High"]
PEAK_STAGES = {"Vegetative", "Flowering"}
BORUTA_RF_TREES = 400
BORUTA_RF_DEPTH = 8


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
        "--skip-refit",
        action="store_true",
        help="Stop after validation instead of refitting on the full training set.",
    )
    return parser.parse_args()


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

    water_in = rainfall + previous_irrigation
    humidity_relief = np.clip(1.0 - humidity / 100.0, 0.05, None)
    area_safe = area + 0.1
    soil_safe = soil_moisture + 1.0
    rain_safe = rainfall + 1.0

    peak_stage = df["Crop_Growth_Stage"].isin(PEAK_STAGES)
    no_mulch = df["Mulching_Used"].astype("string").fillna("Missing").eq("No")

    df["water_in"] = water_in
    df["water_in_log"] = np.log1p(water_in)
    df["rainfall_log"] = np.log1p(rainfall)
    df["prev_irrigation_log"] = np.log1p(previous_irrigation)
    df["field_area_log"] = np.log1p(area)
    df["moisture_deficit_20"] = np.clip(20.0 - soil_moisture, 0.0, None)
    df["moisture_deficit_25"] = np.clip(25.0 - soil_moisture, 0.0, None)
    df["moisture_deficit_30"] = np.clip(30.0 - soil_moisture, 0.0, None)
    df["moisture_le_25"] = (soil_moisture <= 25.0).astype(int)
    df["moisture_le_20"] = (soil_moisture <= 20.0).astype(int)
    df["rain_le_300"] = (rainfall <= 300.0).astype(int)
    df["rain_le_353"] = (rainfall <= 353.0).astype(int)
    df["rain_le_500"] = (rainfall <= 500.0).astype(int)
    df["temp_gt_30"] = (temperature > 30.0).astype(int)
    df["temp_gt_32"] = (temperature > 32.0).astype(int)
    df["wind_gt_10"] = (wind > 10.0).astype(int)
    df["wind_gt_12"] = (wind > 12.0).astype(int)
    df["no_mulch"] = no_mulch.astype(int)
    df["dryness_ratio"] = safe_ratio(temperature + wind, soil_safe, offset=0.0)
    df["heat_wind_pressure"] = temperature * (1.0 + wind / 10.0)
    df["et_proxy"] = (
        temperature
        * sunlight
        * (1.0 + wind / 10.0)
        * humidity_relief
    )
    df["net_water_stress"] = df["et_proxy"] - np.log1p(water_in)
    df["water_stress_ratio"] = safe_ratio(df["et_proxy"], np.log1p(water_in), offset=1.0)
    df["rain_per_ha"] = safe_ratio(rainfall, area, offset=0.1)
    df["prev_irr_per_ha"] = safe_ratio(previous_irrigation, area, offset=0.1)
    df["water_in_per_ha"] = safe_ratio(water_in, area, offset=0.1)
    df["irrigation_reliance"] = safe_ratio(previous_irrigation, water_in, offset=1.0)
    df["aridity_temp"] = safe_ratio(temperature, rainfall, offset=1.0)
    df["aridity_wind"] = safe_ratio(wind, rainfall, offset=1.0)
    df["soil_heat_gap"] = temperature - soil_moisture
    df["humidity_soil_gap"] = humidity - soil_moisture
    df["soil_temp_interaction"] = soil_moisture * temperature
    df["soil_wind_interaction"] = soil_moisture * wind
    df["rain_irrigation_product"] = rainfall * previous_irrigation
    df["salinity_organic_ratio"] = safe_ratio(conductivity, organic_carbon, offset=0.1)
    df["soil_quality_index"] = safe_ratio(organic_carbon, conductivity, offset=1.0)
    df["ph_ec_product"] = soil_ph * conductivity
    df["is_peak_stage"] = peak_stage.astype(int)
    df["peak_stage_no_mulch"] = (peak_stage & no_mulch).astype(int)
    df["peak_stage_dry"] = (peak_stage & (soil_moisture < 25.0)).astype(int)
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


def tune_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
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
    train_pool = build_pool(tune_x_train, tune_y_train, categorical_columns)
    validation_pool = build_pool(x_val, y_val, categorical_columns)
    base_class_weights = compute_balanced_class_weights(y_train, class_names)

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
        score = balanced_accuracy_score(y_val, predictions)
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
    print(f"optuna_trials_completed: {len(study.trials)}")
    print(f"optuna_best_validation_score: {study.best_value:.10f}")
    print(
        "optuna_best_params: "
        + json.dumps(json_ready(best_params), sort_keys=True)
    )

    summary = {
        "enabled": True,
        "tuning_train_rows": len(tune_x_train),
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
    float,
    dict[str, float],
    list[dict[str, Any]],
    dict[str, float],
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
    class_labels = [str(label) for label in model.classes_]
    validation_probabilities = np.asarray(model.predict_proba(validation_pool))
    validation_predictions = predict_with_class_scales(
        probabilities=validation_probabilities,
        class_labels=class_labels,
        class_scales=class_scales,
    )
    validation_predictions = apply_rule_overrides(
        predictions=validation_predictions,
        features=x_val,
        rules=rule_overrides,
    )
    balanced_accuracy = balanced_accuracy_score(y_val, validation_predictions)
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
        float(balanced_accuracy),
        per_class_recall,
        rule_overrides,
        class_scales,
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
    args: argparse.Namespace,
) -> None:
    submission_df = pd.read_csv(args.sample_submission_path)

    if args.id_column not in test_df.columns:
        raise KeyError(f"Missing id column '{args.id_column}' in {args.test_path}")
    if args.target_column not in submission_df.columns:
        raise KeyError(
            f"Missing target column '{args.target_column}' in {args.sample_submission_path}"
        )

    test_pool = build_pool(test_df[feature_columns], None, categorical_columns)
    class_labels = [str(label) for label in model.classes_]
    submission_probabilities = np.asarray(model.predict_proba(test_pool))
    submission_predictions = predict_with_class_scales(
        probabilities=submission_probabilities,
        class_labels=class_labels,
        class_scales=class_scales,
    )
    submission_predictions = apply_rule_overrides(
        predictions=submission_predictions,
        features=test_df[feature_columns],
        rules=rule_overrides,
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
        boruta_candidate_features = [
            column
            for column in engineered_feature_columns
            if column not in engineered_categorical_columns
        ]
        boruta_selected_features, boruta_summary = select_features_with_boruta(
            x_train=x_train[boruta_candidate_features],
            y_train=y_train,
            feature_columns=boruta_candidate_features,
            categorical_columns=[],
            class_names=class_names,
            args=args,
        )
        protected_features = set(raw_feature_columns + engineered_categorical_columns)
        selected_lookup = protected_features | set(boruta_selected_features)
        selected_features = [
            column for column in feature_columns if column in selected_lookup
        ]
        boruta_summary["protected_feature_count"] = len(protected_features)
        boruta_summary["protected_features"] = [
            column for column in feature_columns if column in protected_features
        ]

    selected_categorical_columns = [
        column for column in categorical_columns if column in selected_features
    ]
    print(f"selected_feature_count: {len(selected_features)}")
    print(f"selected_categorical_feature_count: {len(selected_categorical_columns)}")

    tuning_overrides, tuning_summary = tune_hyperparameters(
        x_train=x_train[selected_features],
        y_train=y_train,
        x_val=x_val[selected_features],
        y_val=y_val,
        class_names=class_names,
        categorical_columns=selected_categorical_columns,
        args=args,
    )

    (
        validation_model,
        balanced_accuracy,
        per_class_recall,
        rule_overrides,
        class_scales,
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
    metadata: dict[str, Any] = {
        "target_column": args.target_column,
        "id_column": args.id_column,
        "validation_size": args.validation_size,
        "split_seed": args.random_state,
        "engineered_features_enabled": args.engineered_features,
        "boruta": boruta_summary,
        "optuna": tuning_summary,
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
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    full_model.save_model(args.model_path)
    print(f"saved_model_path: {args.model_path}")

    raw_test_df = pd.read_csv(args.test_path)
    test_df = engineer_features(raw_test_df, enabled=args.engineered_features)
    write_submission(
        model=full_model,
        test_df=test_df,
        feature_columns=selected_features,
        categorical_columns=selected_categorical_columns,
        rule_overrides=full_rule_overrides,
        class_scales=full_class_scales,
        args=args,
    )

    metadata["refit_status"] = "completed"
    metadata["refit_iterations"] = int(validation_model.tree_count_)
    metadata["full_rule_overrides"] = full_rule_overrides
    metadata["full_class_scales"] = full_class_scales
    save_json(metadata, args.metadata_path)
    print(f"saved_metadata_path: {args.metadata_path}")


if __name__ == "__main__":
    main()
