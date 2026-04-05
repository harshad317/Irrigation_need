#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASE_CACHE="$ROOT_DIR/cache/refined_base_v1.npz"
STRESS_CACHE="$ROOT_DIR/cache/stress_signals_crosses_v1.npz"

cd "$ROOT_DIR"

run_experiment() {
  local log_path="$1"
  shift

  if ! PYTHONUNBUFFERED=1 uv run scripts/solution.py "$@" > "$log_path" 2>&1; then
    echo "Experiment failed: $*" >&2
  fi
}

while [[ ! -f "$BASE_CACHE" ]]; do
  sleep 60
done

run_experiment logreg_stack_raw_v2.log \
  --run-name logreg_stack_raw_v2 \
  --decision-policy logreg_stack \
  --meta-raw-features \
  --prediction-cache "$BASE_CACHE" \
  --skip-predictions

while [[ ! -f "$STRESS_CACHE" ]]; do
  sleep 60
done

run_experiment stress_signals_logreg_stack.log \
  --run-name stress_signals_logreg_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy logreg_stack \
  --meta-raw-features \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions

run_experiment stress_signals_ann_stack_full.log \
  --run-name stress_signals_ann_stack_full_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy mlp_stack \
  --meta-full-features \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions

run_experiment stress_signals_xgb_stack.log \
  --run-name stress_signals_xgb_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy xgb_stack \
  --meta-full-features \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions

run_experiment stress_signals_hgb_stack.log \
  --run-name stress_signals_hgb_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy hgb_stack \
  --meta-full-features \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions

run_experiment stress_signals_ft_stack.log \
  --run-name stress_signals_ft_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy ft_transformer_stack \
  --meta-raw-features \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions
