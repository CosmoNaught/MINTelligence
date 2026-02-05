#!/bin/bash
# =============================================================================
# SIMPLE STRATIFIED PIPELINE - Fast tuning with proven settings
# =============================================================================
# Based on the complete pipeline that never OOM'd. Uses conservative settings
# that work reliably on 12GB VRAM.
#
# Usage:
#   ./run_simple_stratified_pipeline.sh
#   tmux new -s train './run_simple_stratified_pipeline.sh 2>&1 | tee training.log'
# =============================================================================

set -e

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DB_PATH="/home/cosmo/Documents/Repos/MINT_DATA/malaria_simulations_4096.duckdb"
OUTPUT_BASE="results_simple_stratified"
TUNING_BASE="results_tuned_simple_stratified"

COMMON_ARGS=(
    --db-path "$DB_PATH"
    --num-workers 4
    --parallel-trials 4
    --device cuda
    --sim-limit 4
    --use-cyclical-time
    --window-size 14
    --batch-size 1024
    --run-tuning
    --tuning-trials 16
    --tuning-timeout 43200       # 4 hours max (should finish much faster)
    --epochs 64
    --patience 16
    --seed 42
)

echo "=============================================="
echo "Starting PREVALENCE prediction pipeline"
echo "Database: $DB_PATH"
echo "=============================================="

python main.py \
    "${COMMON_ARGS[@]}" \
    --predictor prevalence \
    --min-prevalence 0.02 \
    --output-dir "${OUTPUT_BASE}_prevalence" \
    --tuning-output-dir "${TUNING_BASE}_prevalence"

echo ""
echo "=============================================="
echo "Starting CASES prediction pipeline"
echo "=============================================="

python main.py \
    "${COMMON_ARGS[@]}" \
    --predictor cases \
    --min-cases 0.1 \
    --output-dir "${OUTPUT_BASE}_cases" \
    --tuning-output-dir "${TUNING_BASE}_cases"

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Results in: ${OUTPUT_BASE}_prevalence/ and ${OUTPUT_BASE}_cases/"
echo "=============================================="
