# Predicting Irrigation Need

## Setup

1. Read following files:
    - `README.md`
    - `solution.py`
    - `program.md`
2. Verify the existance of data:
    - `Data/train.csv` training data
    - `Data/test.csv` testing data
    - `Data/sample_submission.csv` the final output from the script must be in this format
3. Run the baseline once before making any changes.

## Fixed evaluation protocol

These rules are fixed unless the human explicitly changes them:
    - target: `Irrigation_Need`
    - Validation split: `20% of the train.csv`
    - split seed: `45`
    - metric: ` validation balanced_accuracy_score` For more information about balanced_accuracy_score take a look at this url: `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html`
    - More the `val_balanced_accuracy_score` is better

## Baseline run

Run:
```bash
uv run scripts/solution.py > run.log 2>&1
```

Then extract:
```bash
grep "^val_balanced_accuracy_score:\|^best_iteration:" run.log
```

## Lead agent workflow

1. Study the current baseline, results, and recent failures.
2. Research the next promising idea.
3. Form exactly one bounded experiment hypothesis at a time.
4. Spawn a coding worker.
5. Give that worker clear ownership of:
   - `solution.py`
   - optionally `README.md` or `program.md` if documentation needs to be updated
6. Review the returned metric and code changes.
7. Keep and push only true improvements.
8. Discard regressions or crashes and continue immediately.

## Coding worker workflow

The coding worker should:

1. Implement only the assigned experiment.
2. Run the experiment and collect:
   - `val_balanced_accuracy_score`
   - `best_iteration`
   - any crash details if applicable
3. If the run improves the best known `val_balanced_accuracy_score`:
   - keep the code changes
   - update `results.tsv`
   - Run the model on `test.csv` and save the output as shown in `sample_submission.csv` file in `Predictions` folder.
   - commit the improvement
   - push to the main branch
4. If the run does not improve:
   - do not keep the experiment
   - log it as `discard` or `crash`
   - return control to the lead agent for the next idea

## Results logging

Keep `results.tsv` tab-separated with this schema:

```text
timestamp	run	status	val_balanced_accuracy_score 	best_iteration	commit	branch	description	model_config	feature_config
```

`results.tsv` must not be committed.
Always keep the predictions from the model in a saperate folder called: `Predictions`. Keep the file updting with best results. Make sure the file is named as `prediction_irr_need.csv` and is always updated with latest results.


## What can change

- feature engineering inside `solution.py`
- model hyperparameters
- training logic
- submission generation
- repo docs when the operating workflow changes

## What must not change casually

- dataset contents
- target column
- fixed validation protocol
- `val_balanced_accuracy_score:` summary line

## Fallback mode

`solution.py --autoloop` exists as a non-LLM fallback search loop. It is not the primary workflow. The preferred workflow is still:

- `gpt-5.4` with `extra high` reasoning for research/control (lead agent)
- `gpt-5.3-codex` with `extra high` reasoning for coding, committing, and pushing (coding worker)

## Never stop

Once the experiment loop begins, do not ask whether to continue. Keep researching, delegating, evaluating, and only preserving genuine improvements until the user manually stops the run.