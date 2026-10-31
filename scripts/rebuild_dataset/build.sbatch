#!/usr/bin/env bash
#SBATCH --job-name=t4d-dataset-build
#SBATCH --partition=5090
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
# No GPU resource is requested. Override partition only after site-policy checks.
set -euo pipefail
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the t4d project directory}"
: "${PYTHON:?Set PYTHON to an existing Python >=3.10 environment}"
: "${RUN_ID:?Set a unique RUN_ID; reuse only to resume that run}"
cd "$SOURCE_ROOT"
BUILD_CONFIG="${BUILD_CONFIG:-$SOURCE_ROOT/scripts/rebuild_dataset/build_config.json}"
if [[ "${EXECUTE_BUILD:-0}" != "1" ]]; then
  echo 'Plan only. EXECUTE_BUILD=1 is required for actual metadata/download/conversion.'
  exec "$PYTHON" -B -m scripts.rebuild_dataset.pipeline --config "$BUILD_CONFIG" --run-id "$RUN_ID"
fi
exec "$PYTHON" -u -B -m scripts.rebuild_dataset.pipeline --config "$BUILD_CONFIG" --run-id "$RUN_ID" --execute
