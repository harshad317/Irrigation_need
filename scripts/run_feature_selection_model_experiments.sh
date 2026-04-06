#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_PATH="$ROOT_DIR/results.tsv"
STRESS_CACHE="$ROOT_DIR/cache/stress_signals_crosses_v1.npz"

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
  --meta-raw-features
  --meta-full-features
  --feature-selection
  --feature-selection-topk 96
  --prediction-cache "$STRESS_CACHE"
  --stack-class-scale-search
)

while [[ ! -f "$STRESS_CACHE" ]]; do
  sleep 60
done

run_experiment stress_fs_mlp_hybrid_v1 stress_fs_mlp_hybrid_v1.log \
  --run-name stress_fs_mlp_hybrid_v1 \
  "${COMMON_FLAGS[@]}" \
  --decision-policy mlp_stack \
  --skip-predictions

run_experiment stress_fs_xgb_hybrid_v1 stress_fs_xgb_hybrid_v1.log \
  --run-name stress_fs_xgb_hybrid_v1 \
  "${COMMON_FLAGS[@]}" \
  --decision-policy xgb_stack \
  --skip-predictions

run_experiment stress_fs_hgb_hybrid_v1 stress_fs_hgb_hybrid_v1.log \
  --run-name stress_fs_hgb_hybrid_v1 \
  "${COMMON_FLAGS[@]}" \
  --decision-policy hgb_stack \
  --skip-predictions

run_experiment stress_fs_tabnet_hybrid_v1 stress_fs_tabnet_hybrid_v1.log \
  --run-name stress_fs_tabnet_hybrid_v1 \
  "${COMMON_FLAGS[@]}" \
  --decision-policy tabnet_stack \
  --skip-predictions

run_experiment stress_fs_neural_blend_hybrid_v1 stress_fs_neural_blend_hybrid_v1.log \
  --run-name stress_fs_neural_blend_hybrid_v1 \
  "${COMMON_FLAGS[@]}" \
  --decision-policy neural_base_blend_stack \
  --skip-predictions
