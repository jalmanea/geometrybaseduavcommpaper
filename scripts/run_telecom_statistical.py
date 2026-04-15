"""Statistically rigorous telecom heuristic experiment.

Evaluates 6 deployment methods on 24 real-world Telecom Italia snapshots
with proper replication for paired non-parametric statistical tests.

Experimental design
-------------------
- 48 snapshots (6 weeks x weekday/weekend x 4 hours: 08, 12, 16, 20 UTC)
- 3 user densities (200, 400, 800) — subsampled from 800-user pool
- 26 drone counts (5, 6, 7, ..., 30)
- 30 seeds per condition — each seed draws a fresh user subsample AND
  k-means initialisation, so all methods (including greedy) see the
  same users within a seed but different users across seeds
- 6 methods: kmeans, analytic, analytic_pca, repulsive_lloyd,
  altitude_stagger, greedy

Statistical analysis plan
-------------------------
1. Friedman test (non-parametric repeated-measures) per (n_drones, n_users)
2. Post-hoc Wilcoxon signed-rank with Holm-Bonferroni correction
3. Cliff's delta effect sizes for each method pair
4. Bootstrap 95% CIs on mean differences

Phases
------
main         : Full factorial method comparison (3744 SLURM tasks:
               48 snapshots x 26 drone counts x 3 user densities)
sensitivity  : Hyperparameter sweeps for repulsive_lloyd (beta)
               and altitude_stagger (n_tiers) (48 SLURM tasks)

Output
------
Tidy CSV files (one per task), merged in the analysis notebook.

Usage
-----
# List all tasks
python scripts/run_telecom_statistical.py --phase main --list-tasks

# Run one SLURM task
python scripts/run_telecom_statistical.py --phase main --task-id 0

# Quick local smoke test
python scripts/run_telecom_statistical.py --quick

# Sensitivity phase
python scripts/run_telecom_statistical.py --phase sensitivity --task-id 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dronecomm.config import Config
from dronecomm.heuristic import (
    deploy_analytic_heuristic,
    deploy_analytic_pca_heuristic,
    deploy_repulsive_lloyd_heuristic,
    deploy_altitude_staggered_heuristic,
    generate_grid_candidates,
    greedy_discrete_placement,
    mst_backhaul_orientations,
    _gateway_idx,
)
from dronecomm.optimize import _build_models, _evaluate_single
from dronecomm.scenario import Scenario

# ── Constants ─────────────────────────────────────────────────────────────

DEFAULT_SNAPSHOT_DIR = "results/telecom_v2/snapshots"
DEFAULT_OUTPUT_DIR   = "results/telecom_v2"

DRONE_COUNTS   = list(range(5, 31))
USER_COUNTS    = [200, 400, 800]
N_SEEDS        = 30
BASE_SEED      = 1000
GREEDY_RES     = 40

# Sensitivity defaults
SENS_N_DRONES  = 15
SENS_N_USERS   = 400
SENS_N_SEEDS   = 20
BETA_VALUES    = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
TIER_VALUES    = [2, 3, 4, 5]

# CSV column order
MAIN_COLUMNS = [
    "snapshot_idx", "snapshot_date", "snapshot_hour", "day_type",
    "n_drones", "n_users", "method", "seed",
    "coverage_pct", "throughput_mbps",
    "mean_sinr_db", "median_sinr_db", "min_sinr_db", "sinr_5th_pct_db",
    "total_interference_dbm", "worst_pair_interference_dbm",
    "build_time_s",
]

SENS_COLUMNS = [
    "snapshot_idx", "snapshot_date", "snapshot_hour", "day_type",
    "n_drones", "n_users", "method", "param_name", "param_value", "seed",
    "coverage_pct", "throughput_mbps",
    "mean_sinr_db", "median_sinr_db", "min_sinr_db", "sinr_5th_pct_db",
    "total_interference_dbm", "worst_pair_interference_dbm",
    "build_time_s",
]


# ── Helpers ───────────────────────────────────────────────────────────────

def load_snapshot_metadata(snapshot_dir: Path) -> list[dict]:
    meta_path = snapshot_dir / "snapshot_metadata.json"
    if not meta_path.exists():
        sys.exit(f"Metadata not found: {meta_path}\n"
                 f"Run: python scripts/extract_telecom_snapshots.py")
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["snapshots"]


def make_config(n_drones: int, area_size_m: float = 2000.0) -> Config:
    base = Config()
    return replace(
        base,
        network=replace(base.network, n_drones=n_drones),
        scenario=replace(base.scenario, area_size_m=area_size_m),
    )


def subsample_users(
    full_users: np.ndarray, n_users: int, seed: int,
) -> np.ndarray:
    """Deterministic subsample of user positions.

    All methods within a seed see the same subsample.
    Different seeds see different subsamples.
    """
    if n_users >= full_users.shape[0]:
        return full_users.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(full_users.shape[0], size=n_users, replace=False)
    return full_users[idx]


def deploy_kmeans_baseline(
    config: Config, user_positions: np.ndarray, seed: int,
) -> Scenario:
    from scipy.cluster.vq import kmeans2
    n = config.network.n_drones
    xy = user_positions[:, :2]
    centroids, _ = kmeans2(xy, n, minit="points", seed=seed)
    drone_pos = np.zeros((n, 3))
    drone_pos[:, :2] = centroids
    drone_pos[:, 2] = config.network.altitude_m
    gw_idx = _gateway_idx(drone_pos)
    bh_t, bh_a = mst_backhaul_orientations(drone_pos, gateway_idx=gw_idx)
    return Scenario(
        user_positions=user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=np.zeros(n),
        dl_azimuth_rad=np.zeros(n),
        bh_tilt_rad=bh_t,
        bh_azimuth_rad=bh_a,
        area_size_m=config.scenario.area_size_m,
    )


def deploy_method(
    method: str, config: Config, user_positions: np.ndarray,
    seed: int, candidates: np.ndarray | None = None,
    beta: float = 0.15, n_tiers: int = 3,
) -> Scenario:
    if method == "kmeans":
        return deploy_kmeans_baseline(config, user_positions, seed=seed)
    elif method == "analytic":
        return deploy_analytic_heuristic(config, user_positions, seed=seed)
    elif method == "analytic_pca":
        return deploy_analytic_pca_heuristic(config, user_positions, seed=seed)
    elif method == "repulsive_lloyd":
        return deploy_repulsive_lloyd_heuristic(
            config, user_positions, seed=seed, beta=beta)
    elif method == "altitude_stagger":
        return deploy_altitude_staggered_heuristic(
            config, user_positions, seed=seed, n_tiers=n_tiers)
    elif method == "greedy":
        return greedy_discrete_placement(
            config, candidates, user_positions, verbose=False)
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_metrics(m) -> dict:
    return {
        "coverage_pct":                float(m.coverage_fraction * 100),
        "throughput_mbps":             float(m.sum_throughput_mbps),
        "mean_sinr_db":                float(m.mean_sinr_db),
        "median_sinr_db":              float(m.median_sinr_db),
        "min_sinr_db":                 float(m.min_sinr_db),
        "sinr_5th_pct_db":             float(m.sinr_5th_percentile_db),
        "total_interference_dbm":      float(m.total_inter_drone_interference_dbm),
        "worst_pair_interference_dbm": float(m.worst_pair_interference_dbm),
    }


# ── Task grid generators ─────────────────────────────────────────────────

def generate_main_tasks(
    n_snapshots: int,
    drone_counts: list[int],
    user_counts: list[int],
) -> list[dict]:
    """One task per (snapshot, n_drones, n_users)."""
    tasks = []
    for i, (snap_idx, nd, nu) in enumerate(
        product(range(n_snapshots), drone_counts, user_counts)
    ):
        tasks.append({
            "task_id": i,
            "snapshot_idx": snap_idx,
            "n_drones": nd,
            "n_users": nu,
        })
    return tasks


def generate_sensitivity_tasks(n_snapshots: int) -> list[dict]:
    """One task per snapshot (all hyperparams tested within)."""
    return [{"task_id": i, "snapshot_idx": i} for i in range(n_snapshots)]


# ── Phase runners ─────────────────────────────────────────────────────────

def run_main_task(
    task: dict,
    snapshots: list[dict],
    snapshot_dir: Path,
    output_dir: Path,
    n_seeds: int,
    base_seed: int,
    methods: list[str],
    greedy_res: int,
):
    snap = snapshots[task["snapshot_idx"]]
    n_drones = task["n_drones"]
    n_users = task["n_users"]
    full_users = np.load(str(snapshot_dir / snap["file"]))
    config = make_config(n_drones)
    dl_ant, bh_ant, channel = _build_models(config)

    candidates = None
    if "greedy" in methods:
        candidates = generate_grid_candidates(
            config.scenario.area_size_m, resolution=greedy_res)

    rows = []
    for s in range(n_seeds):
        seed = base_seed + s
        user_pos = subsample_users(full_users, n_users, seed)

        for method in methods:
            t0 = time.perf_counter()
            sc = deploy_method(
                method, config, user_pos, seed=seed, candidates=candidates)
            build_t = time.perf_counter() - t0

            m_obj = _evaluate_single(sc, config, dl_ant, bh_ant, channel)
            met = extract_metrics(m_obj)

            rows.append({
                "snapshot_idx":   snap["idx"],
                "snapshot_date":  snap["date"],
                "snapshot_hour":  snap["hour"],
                "day_type":       snap["day_type"],
                "n_drones":       n_drones,
                "n_users":        n_users,
                "method":         method,
                "seed":           seed,
                "build_time_s":   round(build_t, 4),
                **met,
            })

        if (s + 1) % 10 == 0:
            print(f"    seed {seed}: {n_users}u, "
                  f"cov={rows[-len(methods)]['coverage_pct']:.1f}% (kmeans)")

    # Write CSV
    out_path = output_dir / "main" / f"task_{task['task_id']:04d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MAIN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  -> {out_path}  ({len(rows)} rows)")
    return out_path


def run_sensitivity_task(
    task: dict,
    snapshots: list[dict],
    snapshot_dir: Path,
    output_dir: Path,
    n_seeds: int = SENS_N_SEEDS,
    base_seed: int = BASE_SEED,
    n_drones: int = SENS_N_DRONES,
    n_users: int = SENS_N_USERS,
    beta_values: list[float] = BETA_VALUES,
    tier_values: list[int] = TIER_VALUES,
):
    snap = snapshots[task["snapshot_idx"]]
    full_users = np.load(str(snapshot_dir / snap["file"]))
    config = make_config(n_drones)
    dl_ant, bh_ant, channel = _build_models(config)

    rows = []
    for s in range(n_seeds):
        seed = base_seed + s
        user_pos = subsample_users(full_users, n_users, seed)

        # Sweep beta for repulsive_lloyd
        for beta in beta_values:
            t0 = time.perf_counter()
            sc = deploy_method(
                "repulsive_lloyd", config, user_pos, seed=seed, beta=beta)
            build_t = time.perf_counter() - t0
            m_obj = _evaluate_single(sc, config, dl_ant, bh_ant, channel)
            met = extract_metrics(m_obj)
            rows.append({
                "snapshot_idx":   snap["idx"],
                "snapshot_date":  snap["date"],
                "snapshot_hour":  snap["hour"],
                "day_type":       snap["day_type"],
                "n_drones":       n_drones,
                "n_users":        n_users,
                "method":         "repulsive_lloyd",
                "param_name":     "beta",
                "param_value":    beta,
                "seed":           seed,
                "build_time_s":   round(build_t, 4),
                **met,
            })

        # Sweep n_tiers for altitude_stagger
        for nt in tier_values:
            t0 = time.perf_counter()
            sc = deploy_method(
                "altitude_stagger", config, user_pos, seed=seed, n_tiers=nt)
            build_t = time.perf_counter() - t0
            m_obj = _evaluate_single(sc, config, dl_ant, bh_ant, channel)
            met = extract_metrics(m_obj)
            rows.append({
                "snapshot_idx":   snap["idx"],
                "snapshot_date":  snap["date"],
                "snapshot_hour":  snap["hour"],
                "day_type":       snap["day_type"],
                "n_drones":       n_drones,
                "n_users":        n_users,
                "method":         "altitude_stagger",
                "param_name":     "n_tiers",
                "param_value":    nt,
                "seed":           seed,
                "build_time_s":   round(build_t, 4),
                **met,
            })

    out_path = output_dir / "sensitivity" / f"task_{task['task_id']:04d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SENS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  -> {out_path}  ({len(rows)} rows)")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", choices=["main", "sensitivity"],
                        default="main")
    parser.add_argument("--task-id", type=int, default=None,
                        help="SLURM array task index")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 1 snapshot, 3 seeds, N=5, 200 users, no greedy")
    # Main phase overrides
    parser.add_argument("--drone-counts", type=int, nargs="+", default=DRONE_COUNTS)
    parser.add_argument("--user-counts", type=int, nargs="+", default=USER_COUNTS)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--no-greedy", action="store_true")
    parser.add_argument("--greedy-resolution", type=int, default=GREEDY_RES)
    # Sensitivity overrides
    parser.add_argument("--sens-n-drones", type=int, default=SENS_N_DRONES)
    parser.add_argument("--sens-n-users", type=int, default=SENS_N_USERS)
    parser.add_argument("--sens-n-seeds", type=int, default=SENS_N_SEEDS)
    # Paths
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent

    snap_dir = (root / args.snapshot_dir
                if not Path(args.snapshot_dir).is_absolute()
                else Path(args.snapshot_dir))
    out_dir = (root / args.output_dir
               if not Path(args.output_dir).is_absolute()
               else Path(args.output_dir))

    methods = ["kmeans", "analytic", "analytic_pca",
               "repulsive_lloyd", "altitude_stagger"]
    if not args.no_greedy:
        methods.append("greedy")

    # ── Quick smoke test ──────────────────────────────────────────────
    if args.quick:
        snapshots = load_snapshot_metadata(snap_dir)
        task = {"task_id": 0, "snapshot_idx": 0, "n_drones": 5, "n_users": 200}
        print(f"Quick smoke test: snapshot 0, N=5, 200 users, 3 seeds")
        run_main_task(
            task, snapshots, snap_dir, out_dir,
            n_seeds=3, base_seed=BASE_SEED,
            methods=["kmeans", "analytic", "analytic_pca",
                     "repulsive_lloyd", "altitude_stagger"],
            greedy_res=args.greedy_resolution,
        )
        return

    # ── Load metadata ─────────────────────────────────────────────────
    snapshots = load_snapshot_metadata(snap_dir)
    n_snap = len(snapshots)

    if args.phase == "main":
        tasks = generate_main_tasks(n_snap, args.drone_counts, args.user_counts)
    else:
        tasks = generate_sensitivity_tasks(n_snap)

    # ── List tasks ────────────────────────────────────────────────────
    if args.list_tasks:
        print(f"Phase: {args.phase}")
        print(f"Tasks: {len(tasks)}")
        if args.phase == "main":
            print(f"  Snapshots:    {n_snap}")
            print(f"  Drone counts: {args.drone_counts}")
            print(f"  User counts:  {args.user_counts}")
            print(f"  Seeds/task:   {args.n_seeds}")
            print(f"  Methods:      {methods}")
            n_evals = len(tasks) * args.n_seeds * len(methods)
            print(f"  Total evaluations: {n_evals:,}")
        else:
            print(f"  Snapshots:    {n_snap}")
            print(f"  N drones:     {args.sens_n_drones}")
            print(f"  N users:      {args.sens_n_users}")
            print(f"  Seeds/task:   {args.sens_n_seeds}")
            print(f"  Beta values:  {BETA_VALUES}")
            print(f"  Tier values:  {TIER_VALUES}")
            n_evals = n_snap * args.sens_n_seeds * (len(BETA_VALUES) + len(TIER_VALUES))
            print(f"  Total evaluations: {n_evals:,}")
        print()
        for t in tasks[:20]:
            print(f"  task {t['task_id']:4d}: {t}")
        if len(tasks) > 20:
            print(f"  ... ({len(tasks) - 20} more)")
        return

    # ── Run single task ───────────────────────────────────────────────
    if args.task_id is None:
        sys.exit("Specify --task-id N or --list-tasks or --quick")

    if args.task_id < 0 or args.task_id >= len(tasks):
        sys.exit(f"--task-id must be in [0, {len(tasks) - 1}]")

    task = tasks[args.task_id]
    snap = snapshots[task["snapshot_idx"]]
    print(f"\n{'='*60}")
    print(f"Phase: {args.phase}  |  Task: {args.task_id}/{len(tasks) - 1}")
    print(f"Snapshot: {snap['date']} h{snap['hour']:02d} ({snap['day_type']})")

    t0 = time.perf_counter()
    if args.phase == "main":
        print(f"N={task['n_drones']} drones  |  {task['n_users']} users  |  "
              f"{args.n_seeds} seeds  |  methods={methods}")
        print(f"{'='*60}")
        run_main_task(
            task, snapshots, snap_dir, out_dir,
            n_seeds=args.n_seeds, base_seed=BASE_SEED,
            methods=methods, greedy_res=args.greedy_resolution,
        )
    else:
        print(f"N={args.sens_n_drones} drones  |  {args.sens_n_users} users  |  "
              f"{args.sens_n_seeds} seeds")
        print(f"Beta sweep: {BETA_VALUES}")
        print(f"Tier sweep: {TIER_VALUES}")
        print(f"{'='*60}")
        run_sensitivity_task(
            task, snapshots, snap_dir, out_dir,
            n_seeds=args.sens_n_seeds, base_seed=BASE_SEED,
            n_drones=args.sens_n_drones, n_users=args.sens_n_users,
        )

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
