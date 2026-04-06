#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASE_CACHE="$ROOT_DIR/cache/refined_base_v1.npz"
RESULTS_PATH="$ROOT_DIR/results.tsv"

cd "$ROOT_DIR"

run_experiment() {
  local run_name="$1"
  local log_path="$2"
  shift 2

  if [[ -f "$RESULTS_PATH" ]] && rg -q "$run_name" "$RESULTS_PATH"; then
    return 0
  fi

  if ! env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    nice -n 10 uv run scripts/solution.py "$@" > "$log_path" 2>&1; then
    echo "Experiment failed: $*" >&2
  fi
}

while [[ ! -f "$BASE_CACHE" ]]; do
  sleep 60
done

run_experiment ann_stack_full_v1 \
  ann_stack_full_v1.log \
  --run-name ann_stack_full_v1 \
  --decision-policy mlp_stack \
  --meta-full-features \
  --prediction-cache "$BASE_CACHE" \
  --skip-predictions

run_experiment xgb_stack_full_v1 \
  xgb_stack_full_v1.log \
  --run-name xgb_stack_full_v1 \
  --decision-policy xgb_stack \
  --meta-full-features \
  --prediction-cache "$BASE_CACHE" \
  --skip-predictions

run_experiment ordinal_xgb_stack_full_v1 \
  ordinal_xgb_stack_full_v1.log \
  --run-name ordinal_xgb_stack_full_v1 \
  --decision-policy ordinal_xgb_stack \
  --meta-full-features \
  --prediction-cache "$BASE_CACHE" \
  --skip-predictions

run_experiment hgb_stack_full_v1 \
  hgb_stack_full_v1.log \
  --run-name hgb_stack_full_v1 \
  --decision-policy hgb_stack \
  --meta-full-features \
  --prediction-cache "$BASE_CACHE" \
  --skip-predictions
