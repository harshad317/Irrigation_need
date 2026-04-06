#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_PATH="$ROOT_DIR/results.tsv"
FEATURE_CACHE="$ROOT_DIR/cache/rare_group_pairte_base_v1.npz"

cd "$ROOT_DIR"

run_experiment() {
  local run_name="$1"
  local log_path="$2"
  shift 2

  if [[ -f "$RESULTS_PATH" ]] && rg -q "$run_name" "$RESULTS_PATH"; then
    return 0
  fi

  if ! PYTHONUNBUFFERED=1 nice -n 10 uv run scripts/solution.py "$@" > "$log_path" 2>&1; then
    echo "Experiment failed: $run_name" >&2
  fi
}

COMMON_FLAGS=(
  --categorical-crosses
  --risk-flags
  --stress-signals
  --rare-category-bucketing
  --group-aggregates
  --pairwise-target-encoding
)

while pgrep -f 'bash scripts/run_requested_methods.sh' >/dev/null \
  || pgrep -f 'scripts/solution.py --run-name stress_signals_(rnn_stack_v1|tabnet_stack_v1|ann_cnn_combo_v1|neural_base_blend_v1)' >/dev/null \
  || pgrep -f 'bash scripts/run_notebook_feature_experiments.sh' >/dev/null \
  || pgrep -f 'scripts/solution.py --run-name notebook_eda_ann_stack_scaled_v1' >/dev/null; do
  sleep 60
done

run_experiment rare_group_pairte_base_v1 rare_group_pairte_base_v1.log \
  --run-name rare_group_pairte_base_v1 \
  "${COMMON_FLAGS[@]}" \
  --prediction-cache "$FEATURE_CACHE" \
  --skip-predictions

run_experiment rare_group_pairte_ann_scaled_v1 rare_group_pairte_ann_scaled_v1.log \
  --run-name rare_group_pairte_ann_scaled_v1 \
  "${COMMON_FLAGS[@]}" \
  --decision-policy mlp_stack \
  --meta-raw-features \
  --stack-class-scale-search \
  --prediction-cache "$FEATURE_CACHE" \
  --skip-predictions
