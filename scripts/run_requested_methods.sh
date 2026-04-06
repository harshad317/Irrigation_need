#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STRESS_CACHE="$ROOT_DIR/cache/stress_signals_crosses_v1.npz"
RESULTS_PATH="$ROOT_DIR/results.tsv"

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

while [[ ! -f "$STRESS_CACHE" ]]; do
  sleep 60
done

run_experiment stress_signals_cnn_stack_v1 stress_signals_cnn_stack.log \
  --run-name stress_signals_cnn_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy cnn_stack \
  --meta-raw-features \
  --stack-class-scale-search \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions

run_experiment stress_signals_rnn_stack_v1 stress_signals_rnn_stack.log \
  --run-name stress_signals_rnn_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy rnn_stack \
  --meta-raw-features \
  --stack-class-scale-search \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions

run_experiment stress_signals_tabnet_stack_v1 stress_signals_tabnet_stack.log \
  --run-name stress_signals_tabnet_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy tabnet_stack \
  --meta-raw-features \
  --stack-class-scale-search \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions

run_experiment stress_signals_ann_cnn_combo_v1 stress_signals_ann_cnn_combo.log \
  --run-name stress_signals_ann_cnn_combo_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy ann_cnn_combo_stack \
  --meta-raw-features \
  --stack-class-scale-search \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions

run_experiment stress_signals_neural_base_blend_v1 stress_signals_neural_base_blend.log \
  --run-name stress_signals_neural_base_blend_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy neural_base_blend_stack \
  --meta-raw-features \
  --stack-class-scale-search \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions
