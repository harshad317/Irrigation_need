#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_PATH="$ROOT_DIR/results.tsv"
NOTEBOOK_CACHE="$ROOT_DIR/cache/notebook_eda_base_v1.npz"

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

while pgrep -f 'scripts/solution.py --run-name stress_signals_(ft_stack_v1|cnn_stack_v1|rnn_stack_v1|ann_bag_stack_v1|tabnet_stack_v1|ann_cnn_combo_v1|neural_base_blend_v1)' >/dev/null \
  || pgrep -f 'bash scripts/run_requested_methods.sh' >/dev/null; do
  sleep 60
done

run_experiment notebook_eda_ann_stack_scaled_v1 notebook_eda_ann_stack_scaled.log \
  --run-name notebook_eda_ann_stack_scaled_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --notebook-eda-features \
  --decision-policy mlp_stack \
  --meta-raw-features \
  --stack-class-scale-search \
  --prediction-cache "$NOTEBOOK_CACHE" \
  --skip-predictions
