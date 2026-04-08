# Predicting Irrigation Need

This repository is set up for a lean local experiment loop on the Kaggle `Irrigation_Need` task.
The current workflow uses a single CatBoost baseline with a fixed 15% holdout split and a champion-safe runner that only refreshes the canonical prediction when validation improves.

## Current Workflow

- Canonical training entrypoint: `scripts/solution.py`
- Canonical experiment runner: `scripts/run_experiment.py`
- Target: `Irrigation_Need`
- Validation split: `15%`
- Split seed: `45`
- Acceptance metric: `balanced_accuracy_score`

## Environment

The current repo relies on the existing `.venv`.
Activate it before running commands from the repo root:

```bash
source .venv/bin/activate
```

The active environment needs these packages available:

- `catboost`
- `numpy`
- `pandas`
- `scikit-learn`

## Data Layout

Expected local files:

- `Data/train.csv`
- `Data/test.csv`
- `Data/sample_submission.csv`

Generated local outputs:

- `Predictions/prediction_irr_need.csv`
- `artifacts/irrigation_need_catboost.cbm`
- `artifacts/irrigation_need_metadata.json`
- `results.tsv`
- `run.log`

## Low-Level Validation Run

Use the raw training script when you want a direct validation-only run without touching the champion prediction:

```bash
python scripts/solution.py --skip-refit > run.log 2>&1
grep '^val_balanced_accuracy_score:\|^best_iteration:\|^weakest_class_recall:' run.log
```

## Champion-Safe Experiment Run

Use the runner for every official baseline or challenger experiment. It writes `results.tsv`, preserves the current champion on ties/regressions/crashes, and only updates the canonical prediction/model artifacts after a strict improvement.

Baseline example:

```bash
python scripts/run_experiment.py \
  --run-name baseline_holdout_v1 \
  --description "Initial 15% holdout CatBoost baseline." \
  --model-config "CatBoost multiclass holdout baseline" \
  --feature-config "raw features only"
```

Forward additional `solution.py` arguments after `--`:

```bash
python scripts/run_experiment.py \
  --run-name depth6_probe_v1 \
  --description "Test shallower trees for cleaner Low/Medium separation." \
  --model-config "CatBoost depth-6 holdout model" \
  --feature-config "raw features only" \
  -- --depth 6 --iterations 1200
```

## Notes

- `results.tsv`, `Predictions/`, and local artifacts are intentionally ignored by git.
- `program.md` defines the experiment contract and ladder.
- `scripts/run_experiment.py` owns `--skip-refit`, `--submission-path`, `--model-path`, and `--metadata-path` during official runs.
- Better-score promotions belong on `origin/main`, which already points at `https://github.com/harshad317/Irrigation_need.git`.
