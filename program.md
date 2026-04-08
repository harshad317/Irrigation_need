# Predicting Irrigation Need

## Setup

1. Read following files:
   - `README.md`
   - `scripts/solution.py`
   - `scripts/run_experiment.py`
   - `program.md`
2. Verify the existance of data:
   - `Data/train.csv` training data
   - `Data/test.csv` testing data
   - `Data/sample_submission.csv` the final output from the script must be in this format
3. Activate the existing environment from the repo root:
```bash
source .venv/bin/activate
```
4. Run the baseline once through `scripts/run_experiment.py` before making model changes.
5. If `results.tsv` has no successful rows yet, treat the fresh baseline as the initial champion to beat.

## Fixed evaluation protocol

These rules are fixed unless the human explicitly changes them:
   - target: `Irrigation_Need`
   - Validation split: `15% of the train.csv`
   - split seed: `45`
   - metric: `validation balanced_accuracy_score` For more information about balanced_accuracy_score take a look at this url: `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html`
   - More the `val_balanced_accuracy_score` is better

## Baseline run

Run:
```bash
python scripts/run_experiment.py \
  --run-name baseline_holdout_v1 \
  --description "Initial 15% holdout CatBoost baseline." \
  --model-config "CatBoost multiclass holdout baseline" \
  --feature-config "raw features only"
```

Then extract:
```bash
grep "^val_balanced_accuracy_score:\|^best_iteration:\|^weakest_class_recall:" run.log
```

Low-level validation-only debug run:
```bash
python scripts/solution.py --skip-refit > run.log 2>&1
```

After the baseline:

- `CHAMPION_SCORE` = the best successful `val_balanced_accuracy_score` already recorded in `results.tsv`; if `results.tsv` is empty, use the fresh baseline from `run.log`
- `CHAMPION_ITER` = the paired `best_iteration`
- `CHAMPION_PREDICTION` = `Predictions/prediction_irr_need.csv` only when it was produced by the current champion

## Champion contract

The goal is monotonic champion improvement, not a fantasy where every single challenger wins.

- A challenger is allowed to fail, but a failed challenger must never replace the champion.
- The repository best known `val_balanced_accuracy_score` may only stay flat or increase over time.
- Only treat a run as an improvement when `val_balanced_accuracy_score` is strictly greater than `CHAMPION_SCORE`.
- If a run ties the champion or differs only within floating-point noise, keep the existing champion unless the human explicitly prefers the simpler/faster model.
- Never overwrite `Predictions/prediction_irr_need.csv`, the champion row in `results.tsv`, or the working best git state with a weaker run.

## Priority experiment ladder

Always pull the next hypothesis from this ladder, top to bottom. Do not spend more than 2 consecutive experiments on low-signal hyperparameter nudges.

1. Probability-layer innovations
   - fine-grained or constrained ensemble search
   - logit averaging or rank averaging
   - temperature scaling or probability calibration
   - class-specific threshold search optimized on out-of-fold balanced accuracy
2. Meta-model stacking
   - multinomial logistic regression or another simple meta-learner trained on out-of-fold model probabilities
   - stack raw high-signal features with model probabilities only when the setup is leakage-safe
3. Cross-fitted categorical signal
   - target encoding with smoothing and noise
   - frequency or count encoding
   - rare bucket handling
   - high-value categorical crosses such as crop-season, irrigation-water_source, and soil-region
4. Agronomic interaction families
   - water balance and evapotranspiration interactions
   - soil salinity or organic matter interactions
   - area-normalized irrigation and rainfall pressure
   - clipped, logged, binned, or quantile-transformed weather stress features
5. Diversity and bagging
   - seed bagging
   - diverse LightGBM/XGBoost/CatBoost shapes
   - minority-focused objectives or class-sensitive training variants
6. Controlled pseudo-labeling
   - only with high-confidence test predictions
   - only with class caps
   - only with a strict no-regression gate on validation
7. Cleanup hyperparameter tuning
   - only after a new feature family, encoder family, or probability layer has shown promise

## Experiment design rules

- Every experiment must have one bounded hypothesis and one expected mechanism for improving balanced accuracy.
- Balanced accuracy is average recall, so every experiment should explicitly target the weakest class recall, not just overall confidence.
- Report per-class recall and a confusion matrix summary for every run, even though acceptance is still based on `val_balanced_accuracy_score`.
- Any encoding, calibration, threshold search, or stacking must be learned on training folds and evaluated only on out-of-fold predictions.
- Never leak target information from validation into training features or meta-features.
- Do not run blind sweeps of leaf, depth, or learning-rate values without a stronger reason than curiosity.
- If an idea adds major complexity or a new dependency, justify the expected upside first.

## Lead agent workflow

1. Study the current champion, `results.tsv`, `run.log`, confusion matrix, and weakest class recall.
2. Pick exactly one bounded experiment from the priority ladder.
3. Write the hypothesis in one sentence:
   - `If we change X, balanced accuracy should improve because Y.`
4. Spawn a coding worker.
5. Give that worker clear ownership of:
   - `scripts/solution.py`
   - `scripts/run_experiment.py` when experiment bookkeeping or champion-safe writes change
   - optionally `README.md` or `program.md` if documentation needs to be updated
6. Require the worker to return:
   - `val_balanced_accuracy_score`
   - `best_iteration`
   - per-class recall
   - confusion matrix summary
   - exact code/config change
7. Keep and push only strict improvements over the champion.
8. Reject ties, regressions, unstable runs, or leakage-prone ideas immediately and continue to the next hypothesis.

## Coding worker workflow

The coding worker should:

1. Implement only the assigned experiment.
2. Run the experiment and collect:
   - `val_balanced_accuracy_score`
   - `best_iteration`
   - per-class recall
   - confusion matrix summary
   - any crash details if applicable
3. If the run improves the best known `val_balanced_accuracy_score`:
   - keep the code changes
   - update `results.tsv` with the new champion
   - include the experiment hypothesis and the main recall change in `description`
   - rerun through `scripts/run_experiment.py` so the champion write is staged safely before replacing the current prediction/model artifacts
   - Run the model on `test.csv` and save the output as shown in `sample_submission.csv` file in `Predictions` folder.
   - keep the file named `prediction_irr_need.csv`
   - commit the improvement and push it to `origin/main` on `https://github.com/harshad317/Irrigation_need.git` once the winning change set is scoped to the champion run
4. If the run does not improve:
   - do not keep the experiment
   - log it as `discard` or `crash`
   - mention why it failed, for example: no lift, weaker High recall, unstable fold behavior, or leakage risk
   - return control to the lead agent for the next idea

## Results logging

Keep `results.tsv` tab-separated with this schema:

```text
timestamp	run	status	val_balanced_accuracy_score	best_iteration	commit	branch	description	model_config	feature_config
```

`results.tsv` must not be committed.
Always keep the predictions from the model in a saperate folder called: `Predictions`. Keep the file updting with best results. Make sure the file is named as `prediction_irr_need.csv` and is always updated with latest champion results.

Field guidance:

- `status` should clearly distinguish `champion`, `discard`, and `crash`
- `description` should capture the hypothesis and the main effect on class recall
- `model_config` should describe the model family, stacker, calibrator, or bagging setup
- `feature_config` should describe the new feature or encoding family used in that run

## What can change

- experiment orchestration inside `scripts/run_experiment.py`
- feature engineering inside `scripts/solution.py`
- model hyperparameters
- training logic
- stacking, calibration, thresholding, and ensemble logic
- submission generation
- repo docs when the operating workflow changes

## What must not change casually

- dataset contents
- target column
- fixed validation protocol
- `val_balanced_accuracy_score:` summary line

## Fallback mode

There is no `--autoloop` fallback in the lean setup. The local fallback loop is `scripts/run_experiment.py`, which still must respect the champion contract and the priority experiment ladder instead of doing pure random search. The preferred workflow is still:

- `gpt-5.4` with `extra high` reasoning for research/control (lead agent)
- `gpt-5.3-codex` with `extra high` reasoning for coding, committing, and pushing (coding worker)

If fallback mode is used, it should still respect the champion contract and the priority experiment ladder instead of doing pure random search.

## Never stop

Once the experiment loop begins, do not ask whether to continue. Keep researching, delegating, evaluating, and only preserving genuine improvements until the user manually stops the run.
Use subagents, skills if needed.
