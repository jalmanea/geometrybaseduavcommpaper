#!/bin/bash
#SBATCH --job-name=telecom_stat_batch
#SBATCH --output=logs/telecom_stat_batch_%A_%a.out
#SBATCH --error=logs/telecom_stat_batch_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch

# ============================================================================
# Batched telecom statistical sweep.
#
# This wrapper is for large task grids where one SLURM array element should run
# multiple `run_telecom_statistical.py --task-id ...` tasks sequentially.
#
# Defaults:
#   BATCH_SIZE=100 tasks per array slot
#
# Paper run (4,992 tasks -> 50 array slots of 100):
#   mkdir -p logs
#   BATCH_SIZE=100 sbatch --array=0-49 \
#       scripts/submit_telecom_statistical_batched.sh \
#       --phase main \
#       --snapshot-dir results/telecom_milan/snapshots \
#       --output-dir results/telecom_milan \
#       --drone-counts $(seq 5 30) \
#       --user-counts $(seq 100 100 800) \
#       --n-seeds 20 \
#       --no-greedy \
#       --methods kmeans analytic kmeans_altitude_sweep
#
# Notes:
# - The array range must match the chosen BATCH_SIZE and task grid size.
# - This script computes the total task count dynamically from snapshot metadata
#   plus the provided --drone-counts/--user-counts arguments.
# ============================================================================

set -euo pipefail

echo "=============================================="
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Array ID: ${SLURM_ARRAY_JOB_ID:-local}  Slot: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:     ${SLURMD_NODENAME:-local}"
echo "Started:  $(date)"
echo "=============================================="

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

BATCH_SIZE="${BATCH_SIZE:-100}"
SLOT_ID="${SLURM_ARRAY_TASK_ID:-0}"

TOTAL_TASKS="$({
python - "$@" <<'PY'
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from scripts.run_telecom_statistical import (  # noqa: E402
    DRONE_COUNTS,
    USER_COUNTS,
    generate_main_tasks,
    load_snapshot_metadata,
)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--snapshot-dir", default="results/telecom_milan/snapshots")
parser.add_argument("--drone-counts", type=int, nargs="+", default=None)
parser.add_argument("--user-counts", type=int, nargs="+", default=None)
parser.add_argument("--phase", default="main")
args, _ = parser.parse_known_args(sys.argv[1:])

if args.phase != "main":
    raise SystemExit("submit_telecom_statistical_batched.sh only supports --phase main")

snapshot_dir = Path(args.snapshot_dir)
if not snapshot_dir.is_absolute():
    snapshot_dir = Path.cwd() / snapshot_dir

snapshots = load_snapshot_metadata(snapshot_dir)
drone_counts = args.drone_counts if args.drone_counts is not None else DRONE_COUNTS
user_counts = args.user_counts if args.user_counts is not None else USER_COUNTS
tasks = generate_main_tasks(len(snapshots), drone_counts, user_counts)
print(len(tasks))
PY
})"

TASK_START=$(( SLOT_ID * BATCH_SIZE ))
TASK_END=$(( TASK_START + BATCH_SIZE - 1 ))
if [ "$TASK_END" -ge "$TOTAL_TASKS" ]; then
    TASK_END=$(( TOTAL_TASKS - 1 ))
fi

if [ "$TASK_START" -ge "$TOTAL_TASKS" ]; then
    echo "Slot $SLOT_ID has no work: task start $TASK_START >= total tasks $TOTAL_TASKS"
    exit 0
fi

echo "BATCH_SIZE : $BATCH_SIZE"
echo "TOTAL_TASKS: $TOTAL_TASKS"
echo "Slot $SLOT_ID: running tasks $TASK_START..$TASK_END"

for TASK_ID in $(seq "$TASK_START" "$TASK_END"); do
    echo "  Task $TASK_ID / $(( TOTAL_TASKS - 1 ))..."
    python scripts/run_telecom_statistical.py \
        --task-id "$TASK_ID" \
        "$@"
done

echo "=============================================="
echo "Finished: $(date)"
echo "=============================================="
