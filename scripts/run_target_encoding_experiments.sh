#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_CACHE="$ROOT_DIR/cache/target_freq_enc_v1.npz"

cd "$ROOT_DIR"

run_experiment() {
  local log_path="$1"
  shift

  if ! PYTHONUNBUFFERED=1 uv run scripts/solution.py "$@" > "$log_path" 2>&1; then
    echo "Experiment failed: $*" >&2
  fi
}

wait_for_stress_branch() {
  while pgrep -f 'scripts/solution.py --run-name stress_signals_' >/dev/null; do
    sleep 60
  done
}

wait_for_stress_branch

run_experiment target_freq_enc.log \
  --run-name target_freq_enc_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --frequency-encoding \
  --target-encoding \
  --prediction-cache "$TARGET_CACHE" \
  --skip-predictions

run_experiment target_freq_class_scale.log \
  --run-name target_freq_class_scale_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --frequency-encoding \
  --target-encoding \
  --decision-policy class_scale_search \
  --prediction-cache "$TARGET_CACHE" \
  --skip-predictions

run_experiment target_freq_ann_stack.log \
  --run-name target_freq_ann_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --frequency-encoding \
  --target-encoding \
  --decision-policy mlp_stack \
  --meta-raw-features \
  --prediction-cache "$TARGET_CACHE" \
  --skip-predictions

run_experiment target_freq_logreg_stack.log \
  --run-name target_freq_logreg_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --frequency-encoding \
  --target-encoding \
  --decision-policy logreg_stack \
  --meta-raw-features \
  --prediction-cache "$TARGET_CACHE" \
  --skip-predictions

run_experiment target_freq_ann_stack_full.log \
  --run-name target_freq_ann_stack_full_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --frequency-encoding \
  --target-encoding \
  --decision-policy mlp_stack \
  --meta-full-features \
  --prediction-cache "$TARGET_CACHE" \
  --skip-predictions

run_experiment target_freq_xgb_stack.log \
  --run-name target_freq_xgb_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --frequency-encoding \
  --target-encoding \
  --decision-policy xgb_stack \
  --meta-full-features \
  --prediction-cache "$TARGET_CACHE" \
  --skip-predictions

run_experiment target_freq_hgb_stack.log \
  --run-name target_freq_hgb_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --frequency-encoding \
  --target-encoding \
  --decision-policy hgb_stack \
  --meta-full-features \
  --prediction-cache "$TARGET_CACHE" \
  --skip-predictions

run_experiment target_freq_ft_stack.log \
  --run-name target_freq_ft_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --frequency-encoding \
  --target-encoding \
  --decision-policy ft_transformer_stack \
  --meta-raw-features \
  --prediction-cache "$TARGET_CACHE" \
  --skip-predictions
