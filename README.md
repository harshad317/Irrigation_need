# Predicting Irrigation Need

This repository contains a tabular classification solution for the Kaggle competition [Predicting Irrigation Need](https://www.kaggle.com/competitions/playground-series-s6e4/overview), part of the Playground Series Season 6 Episode 4. The goal is to predict the categorical target `Irrigation_Need` (`Low`, `Medium`, `High`) from soil, crop, weather, and irrigation-management features.

The current approach is a weighted ensemble of LightGBM, XGBoost, and CatBoost trained with stratified cross-validation, feature engineering, and explicit imbalance handling.

## Competition Summary

- Competition: Kaggle Playground Series S6E4
- Task type: multiclass classification
- Evaluation metric: `Balanced Accuracy Score`
- Submission format: CSV with columns `id` and `Irrigation_Need`

## Repository Layout

```text
.
├── Data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── Predictions/
├── scripts/
│   └── solution.py
├── program.md
├── results.tsv
└── README.md
```

## Data

The training and inference pipeline expects the official Kaggle files in `Data/`:

- `Data/train.csv`: 630,000 rows and 21 columns
- `Data/test.csv`: 270,000 rows and 20 columns
- `Data/sample_submission.csv`: 270,000 rows and 2 columns

The target column is:

- `Irrigation_Need`

The raw feature set is split into:

- Numeric features: `Soil_pH`, `Soil_Moisture`, `Organic_Carbon`, `Electrical_Conductivity`, `Temperature_C`, `Humidity`, `Rainfall_mm`, `Sunlight_Hours`, `Wind_Speed_kmh`, `Field_Area_hectare`, `Previous_Irrigation_mm`
- Categorical features: `Soil_Type`, `Crop_Type`, `Crop_Growth_Stage`, `Season`, `Irrigation_Type`, `Water_Source`, `Mulching_Used`, `Region`

Observed target imbalance in `train.csv`:

- `Low`: about 58.7%
- `Medium`: about 37.9%
- `High`: about 3.3%

This imbalance is important: the script computes inverse-frequency class weights and uses them throughout training.

## Current Modeling Pipeline

`scripts/solution.py` implements the full training, validation, blending, and submission flow:

1. Load `train.csv`, `test.csv`, and `sample_submission.csv`
2. Encode the target with a fixed label order:
   - `Low -> 0`
   - `Medium -> 1`
   - `High -> 2`
3. Create domain-inspired engineered features:
   - `water_stress`
   - `et_proxy`
   - `effective_rain`
   - `rain_per_area`
   - `prev_irr_per_ha`
   - `aridity`
   - `heat_index`
   - `soil_quality`
4. Encode categorical columns with `OrdinalEncoder` for LightGBM and XGBoost
5. Keep raw categorical columns for CatBoost
6. Build inverse-frequency sample weights from the class distribution
7. Train three models with 5-fold `StratifiedKFold` (`shuffle=True`, `random_state=45`)
8. Collect out-of-fold class probabilities for each model
9. Search ensemble weights over the LightGBM, XGBoost, and CatBoost predictions
10. Generate the final Kaggle submission file at `Predictions/prediction_irr_need.csv`

## Models Used

The current ensemble contains:

- LightGBM multiclass classifier
- XGBoost multiclass classifier
- CatBoost multiclass classifier

All three models use early stopping, and the final submission is produced from the best out-of-fold blend found on the validation predictions.

## Validation Protocol

The working evaluation protocol in this repository is aligned with `program.md` and the training script:

- Target: `Irrigation_Need`
- Metric: `balanced_accuracy_score`
- Split strategy: 5-fold stratified cross-validation
- Random seed: `45`
- Validation size per fold: about 20% of the training data

At the end of a run, the script prints greppable summary lines:

```text
val_balanced_accuracy_score: ...
best_iteration: ...
```

## Setup

This repository does not currently include a pinned `pyproject.toml` or `requirements.txt`, so dependencies need to be installed manually.

Example setup with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install numpy pandas scikit-learn lightgbm xgboost catboost
```

## Run Training and Create a Submission

```bash
uv run scripts/solution.py
```

To save the full log and extract the final summary values:

```bash
uv run scripts/solution.py > run.log 2>&1
grep '^val_balanced_accuracy_score:\|^best_iteration:' run.log
```

## Outputs

After a successful run, the repository will contain:

- `Predictions/prediction_irr_need.csv`: submission file in Kaggle format
- `run.log`: optional local training log if you redirect stdout/stderr

The script also prints:

- Train/test shapes
- Target distribution
- Per-fold balanced accuracy for each model
- Overall out-of-fold balanced accuracy for LightGBM, XGBoost, CatBoost, and the final ensemble
- The selected ensemble weights

## macOS Note

`scripts/solution.py` includes a macOS-specific `libomp` workaround that tries to use the bundled OpenMP library from:

```text
.venv/lib/python3.13/site-packages/sklearn/.dylibs
```

If you are using a different Python version or a different environment layout, you may need to adjust that path.

## Workflow Notes

- `program.md` describes the intended experiment workflow for iterating on `scripts/solution.py`
- `results.tsv` is meant for local experiment tracking and is ignored by git
- `Predictions/` is also ignored by git and is intended to store the latest submission file

## Next Improvement Ideas

If you want to push this baseline further, the most promising areas are:

- richer agronomic interaction features
- target-aware categorical encoding experiments
- stronger ensemble weight search or stacking
- hyperparameter tuning per model
- calibration or thresholding experiments optimized for balanced accuracy
