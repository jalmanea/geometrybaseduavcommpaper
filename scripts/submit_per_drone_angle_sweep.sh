#!/bin/bash
#SBATCH --job-name=dronecomm_angle_sweep
#SBATCH --output=logs/angle_sweep_%A_%a.out
#SBATCH --error=logs/angle_sweep_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch

# ============================================================================
# Per-Drone DL Orientation Sweep Experiment
#
# For each heuristic method, sweeps each drone's DL (tilt, azimuth) independently
# to find the best per-drone orientation and measure improvement over defaults.
#
# Default sweep grid:
#   Tilt     : 0-60 deg, step 1 deg  (61 values)
#   Azimuth  : 0-355 deg, step 5 deg (72 values)
#   Per drone: 61 x 72 = 4,392 evaluations per seed
#
# Default task grid: 6 drone counts x 2 user counts x 2 distributions = 24 tasks
#   Drone counts:  5, 10, 15, 20, 25, 30
#   Target users:  200, 400
#   Distributions: clustered, hotspot
#
# Usage:
#   mkdir -p logs results/angle_sweep
#   sbatch --array=0-23 scripts/submit_per_drone_angle_sweep.sh
#
# Custom user/drone counts:
#   sbatch --array=0-47 scripts/submit_per_drone_angle_sweep.sh \
#       --drone-counts 5 10 15 20 25 30 --user-counts 100 200 400 800
#
# Fewer seeds (faster):
#   sbatch --array=0-23 scripts/submit_per_drone_angle_sweep.sh --n-eval-seeds 5
#
# Coarser angle grid (faster):
#   sbatch --array=0-23 scripts/submit_per_drone_angle_sweep.sh \
#       --tilt-step 5 --azimuth-step 15
#
# Specific methods only:
#   sbatch --array=0-23 scripts/submit_per_drone_angle_sweep.sh \
#       --methods analytic repulsive_lloyd
#
# Omit per-drone grids from output (smaller JSON):
#   sbatch --array=0-23 scripts/submit_per_drone_angle_sweep.sh --no-grids
#
# List all tasks:
#   python scripts/run_per_drone_angle_sweep.py --list-tasks
#
# Quick smoke test (local, no SLURM):
#   python scripts/run_per_drone_angle_sweep.py --quick
# ============================================================================

set -e

echo "=============================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Array ID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "=============================================="

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

mkdir -p logs

# Results go under results/angle_sweep/run_<array job id>
RUN_ID="${SLURM_ARRAY_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="results/angle_sweep/run_${RUN_ID}"
mkdir -p "$OUTPUT_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "Output dir: $OUTPUT_DIR"
echo "Running per-drone angle sweep task $TASK_ID..."
python scripts/run_per_drone_angle_sweep.py \
    --task-id "$TASK_ID" \
    --output "$OUTPUT_DIR" \
    "$@"

echo "=============================================="
echo "Finished: $(date)"
echo "=============================================="
