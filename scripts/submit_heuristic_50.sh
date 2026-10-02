#!/bin/bash
#SBATCH --job-name=heuristic_50
#SBATCH --output=logs/heuristic_50_%A_%a.out
#SBATCH --error=logs/heuristic_50_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch

# ============================================================================
# Heuristic sweep: N = 1..50 (increment 1), all user counts and distributions
#
# Task grid: 50 drone counts × 10 user counts × 3 distributions = 1500 tasks
# Batching:  100 tasks per array slot → 15 array slots
#
# Submit:
#   mkdir -p logs results/heuristic_50
#   sbatch --array=0-14 scripts/submit_heuristic_50.sh
#
# Skip greedy (faster, only kmeans vs analytic variants):
#   sbatch --array=0-14 scripts/submit_heuristic_50.sh --no-greedy
# ============================================================================

set -e

BATCH_SIZE=100
TOTAL_TASKS=1500

echo "=============================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Array ID: $SLURM_ARRAY_JOB_ID  Slot: $SLURM_ARRAY_TASK_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "=============================================="

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

mkdir -p logs results/heuristic_50

SLOT_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_START=$(( SLOT_ID * BATCH_SIZE ))
TASK_END=$(( TASK_START + BATCH_SIZE - 1 ))
if [ "$TASK_END" -ge "$TOTAL_TASKS" ]; then
    TASK_END=$(( TOTAL_TASKS - 1 ))
fi

echo "Slot $SLOT_ID: running tasks $TASK_START..$TASK_END"

for TASK_ID in $(seq "$TASK_START" "$TASK_END"); do
    echo "  Task $TASK_ID / $(( TOTAL_TASKS - 1 ))..."
    python scripts/run_heuristic_experiment.py \
        --task-id "$TASK_ID" \
        --drone-counts $(seq 1 50 | tr '\n' ' ') \
        --output results/heuristic_50 \
        "$@"
done

echo "=============================================="
echo "Finished: $(date)"
echo "=============================================="
