#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STRESS_CACHE="$ROOT_DIR/cache/stress_signals_crosses_v1.npz"

cd "$ROOT_DIR"

run_experiment() {
  local log_path="$1"
  shift

  if ! PYTHONUNBUFFERED=1 nice -n 10 uv run scripts/solution.py "$@" > "$log_path" 2>&1; then
    echo "Experiment failed: $*" >&2
  fi
}

while [[ ! -f "$STRESS_CACHE" ]]; do
  sleep 60
done

while pgrep -f 'scripts/solution.py --run-name stress_signals_(ft_stack_v1|ann_stack_scaled_v1|cnn_stack_v1|rnn_stack_v1)' >/dev/null; do
  sleep 60
done

run_experiment stress_signals_ann_bag_stack.log \
  --run-name stress_signals_ann_bag_stack_v1 \
  --categorical-crosses \
  --risk-flags \
  --stress-signals \
  --decision-policy mlp_bag_stack \
  --meta-raw-features \
  --stack-class-scale-search \
  --prediction-cache "$STRESS_CACHE" \
  --skip-predictions
