#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT_DIR"

run_stage() {
  local script_path="$1"
  local script_name
  script_name="$(basename "$script_path")"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting $script_name"
  if ! bash "$script_path"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] failed $script_name" >&2
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished $script_name"
  fi
}

run_stage scripts/run_next_experiments.sh
run_stage scripts/run_cached_meta_experiments.sh
run_stage scripts/run_more_experiments.sh
run_stage scripts/run_target_encoding_experiments.sh
run_stage scripts/run_requested_methods.sh
run_stage scripts/run_bagged_followup.sh
run_stage scripts/run_notebook_feature_experiments.sh
run_stage scripts/run_feature_engineering_experiments.sh
