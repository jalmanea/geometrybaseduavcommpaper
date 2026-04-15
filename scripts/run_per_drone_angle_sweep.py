"""Per-drone DL orientation sweep experiment.

Question: Can we improve each heuristic method by optimizing the DL antenna
orientation (tilt + azimuth) of each drone independently?

For each heuristic deployment method:
  1. Deploy drones using the method (positions + default orientations)
  2. For each drone, sweep its DL (tilt, azimuth) while keeping all other
     drones at their default orientations
  3. Find the best per-drone orientation (maximising combined score)
  4. Apply all per-drone best orientations simultaneously and evaluate

Sweep grid per drone (defaults):
  tilt_deg   : [0, 1, ..., 60]   (step 1 deg,  61 values)
  azimuth_deg: [0, 5, ..., 355]  (step 5 deg,  72 values)
  Total      : 61 x 72 = 4 392 evaluations per drone per seed

Note on BH interference
-----------------------
Backhaul interference depends only on BH antenna orientations and drone
positions, NOT on DL orientations.  It is therefore constant across the
DL sweep.  Only DL coverage / SINR / throughput change.  The per-drone
grids store coverage_pct (the metric that varies).

Usage
-----
# Quick test
python scripts/run_per_drone_angle_sweep.py --quick

# Single configuration
python scripts/run_per_drone_angle_sweep.py --n-drones 10 --target-users 200

# SLURM array task
python scripts/run_per_drone_angle_sweep.py --task-id 0 --output results/angle_sweep/run_123

# List all tasks
python scripts/run_per_drone_angle_sweep.py --list-tasks
"""

from __future__ import annotations

import argparse
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
    mst_backhaul_orientations,
    _gateway_idx,
)
from dronecomm.optimize import _build_models, _evaluate_single
from dronecomm.scenario import (
    Scenario,
    generate_users,
    generate_users_uniform,
    generate_users_hotspot,
)


# ── Defaults ─────────────────────────────────────────────────────────

TILT_MAX_DEG = 60
TILT_STEP_DEG = 1
AZIMUTH_STEP_DEG = 5

DRONE_COUNTS_DEFAULT = [5, 10, 15, 20, 25, 30]
TARGET_USERS_DEFAULT = [200, 400]
DISTRIBUTIONS_DEFAULT = ["clustered", "hotspot"]

N_CLUSTERS_CLUSTERED = 5
N_CLUSTERS_HOTSPOT = 3

INTERF_BEST_DBM = -70.0
INTERF_WORST_DBM = -30.0

DEFAULT_METHODS = [
    "kmeans", "analytic", "analytic_pca",
    "repulsive_lloyd", "altitude_stagger",
]

METHOD_LABELS = {
    "kmeans": "K-means",
    "analytic": "Analytic",
    "analytic_pca": "Analytic+PCA",
    "repulsive_lloyd": "Repulsive Lloyd",
    "altitude_stagger": "Alt. Stagger",
}


# ── Helpers (shared with other experiment scripts) ───────────────────


def make_config(n_drones: int, target_users: int, distribution: str = "clustered") -> Config:
    base = Config()
    n_clusters = N_CLUSTERS_HOTSPOT if distribution == "hotspot" else N_CLUSTERS_CLUSTERED
    users_per_cluster_mean = max(10, target_users // n_clusters)
    return replace(
        base,
        network=replace(base.network, n_drones=n_drones),
        scenario=replace(
            base.scenario,
            n_clusters=n_clusters,
            users_per_cluster_mean=users_per_cluster_mean,
            users_per_cluster_std=max(1, users_per_cluster_mean // 4),
        ),
    )


def generate_users_for_distribution(
    distribution: str, target_users: int, config: Config, rng: np.random.Generator,
) -> np.ndarray:
    sc = config.scenario
    if distribution == "clustered":
        return generate_users(
            n_clusters=sc.n_clusters,
            users_per_cluster_mean=sc.users_per_cluster_mean,
            users_per_cluster_std=sc.users_per_cluster_std,
            cluster_spread_m=sc.cluster_spread_m,
            area_size_m=sc.area_size_m,
            rng=rng,
        )
    elif distribution == "uniform":
        return generate_users_uniform(n_users=target_users, area_size_m=sc.area_size_m, rng=rng)
    elif distribution == "hotspot":
        return generate_users_hotspot(
            n_clusters=N_CLUSTERS_HOTSPOT,
            users_per_cluster_mean=sc.users_per_cluster_mean,
            users_per_cluster_std=sc.users_per_cluster_std,
            cluster_spread_m=50.0,
            area_size_m=sc.area_size_m,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def combined_score(coverage_pct: float, total_interference_dbm: float, zeta: float) -> float:
    cov_01 = coverage_pct / 100.0
    interf_01 = (INTERF_WORST_DBM - total_interference_dbm) / (INTERF_WORST_DBM - INTERF_BEST_DBM)
    interf_01 = float(np.clip(interf_01, 0.0, 1.0))
    return float(zeta * cov_01 + (1.0 - zeta) * interf_01)


def metrics_to_dict(m, zeta: float = 0.7) -> dict:
    cov = float(m.coverage_fraction * 100)
    interf = float(m.total_inter_drone_interference_dbm)
    return {
        "coverage_pct": cov,
        "throughput_mbps": float(m.sum_throughput_mbps),
        "mean_sinr_db": float(m.mean_sinr_db),
        "median_sinr_db": float(m.median_sinr_db),
        "min_sinr_db": float(m.min_sinr_db),
        "sinr_5th_pct_db": float(m.sinr_5th_percentile_db),
        "total_interference_dbm": interf,
        "worst_pair_interference_dbm": float(m.worst_pair_interference_dbm),
        "n_users": int(m.n_users),
        "combined_score": combined_score(cov, interf, zeta),
    }


def average_metric_dicts(dicts: list[dict]) -> dict:
    avg = {}
    for key in dicts[0]:
        avg[key] = float(np.mean([d[key] for d in dicts]))
    return avg


# ── Deployment wrapper ───────────────────────────────────────────────


def deploy_kmeans_baseline(config: Config, user_positions: np.ndarray, seed: int = 42) -> Scenario:
    from scipy.cluster.vq import kmeans2
    n = config.network.n_drones
    centroids, _ = kmeans2(user_positions[:, :2], n, minit="points", seed=seed)
    drone_pos = np.zeros((n, 3))
    drone_pos[:, :2] = centroids
    drone_pos[:, 2] = config.network.altitude_m
    gw_idx = _gateway_idx(drone_pos)
    bh_tilt, bh_azimuth = mst_backhaul_orientations(drone_pos, gateway_idx=gw_idx)
    return Scenario(
        user_positions=user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=np.zeros(n),
        dl_azimuth_rad=np.zeros(n),
        bh_tilt_rad=bh_tilt,
        bh_azimuth_rad=bh_azimuth,
        area_size_m=config.scenario.area_size_m,
    )


def deploy_method(
    method: str, config: Config, user_pos: np.ndarray, seed: int,
    repulsive_beta: float = 0.15, n_altitude_tiers: int = 3,
) -> Scenario:
    if method == "kmeans":
        return deploy_kmeans_baseline(config, user_pos, seed)
    elif method == "analytic":
        return deploy_analytic_heuristic(config, user_pos, seed)
    elif method == "analytic_pca":
        return deploy_analytic_pca_heuristic(config, user_pos, seed)
    elif method == "repulsive_lloyd":
        return deploy_repulsive_lloyd_heuristic(config, user_pos, seed, beta=repulsive_beta)
    elif method == "altitude_stagger":
        return deploy_altitude_staggered_heuristic(config, user_pos, seed, n_tiers=n_altitude_tiers)
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Task grid ────────────────────────────────────────────────────────


def generate_task_grid(
    drone_counts: list[int],
    target_users: list[int],
    distributions: list[str],
) -> list[dict]:
    return [
        {"task_id": i, "n_drones": nd, "target_users": tu, "distribution": dist}
        for i, (nd, tu, dist) in enumerate(product(drone_counts, target_users, distributions))
    ]


# ── Core experiment ──────────────────────────────────────────────────


def run_experiment(
    n_drones: int = 10,
    target_users: int = 200,
    distribution: str = "clustered",
    methods: list[str] | None = None,
    tilt_angles_deg: list[float] | None = None,
    azimuths_deg: list[float] | None = None,
    n_eval_seeds: int = 10,
    eval_base_seed: int = 1000,
    zeta: float = 0.7,
    repulsive_beta: float = 0.15,
    n_altitude_tiers: int = 3,
    save_grids: bool = True,
) -> dict:
    if methods is None:
        methods = list(DEFAULT_METHODS)
    if tilt_angles_deg is None:
        tilt_angles_deg = list(range(0, TILT_MAX_DEG + 1, TILT_STEP_DEG))
    if azimuths_deg is None:
        azimuths_deg = list(range(0, 360, AZIMUTH_STEP_DEG))

    n_tilts = len(tilt_angles_deg)
    n_azimuths = len(azimuths_deg)
    n_per_drone = n_tilts * n_azimuths

    config = make_config(n_drones, target_users, distribution)
    dl_antenna, bh_antenna, channel = _build_models(config)

    print(f"Per-Drone Angle Sweep Experiment")
    print(f"  N={n_drones} drones  |  ~{target_users} users  |  dist={distribution}")
    print(f"  Methods      : {methods}")
    print(f"  Seeds        : {n_eval_seeds}")
    print(f"  Tilt angles  : {tilt_angles_deg[0]}-{tilt_angles_deg[-1]} deg"
          f"  (step={tilt_angles_deg[1]-tilt_angles_deg[0] if len(tilt_angles_deg)>1 else 0},"
          f" n={n_tilts})")
    print(f"  Azimuths     : {azimuths_deg[0]}-{azimuths_deg[-1]} deg"
          f"  (step={azimuths_deg[1]-azimuths_deg[0] if len(azimuths_deg)>1 else 0},"
          f" n={n_azimuths})")
    print(f"  Per drone    : {n_per_drone:,} evaluations per seed")
    print(f"  Total evals  : {n_eval_seeds * len(methods) * n_drones * n_per_drone:,}")
    print(f"  Zeta         : {zeta}")
    print()

    # Convert angles to radians (precompute)
    tilt_rad = np.deg2rad(tilt_angles_deg)
    az_rad = np.deg2rad(azimuths_deg)

    all_results = {}

    for method in methods:
        print(f"  Method: {METHOD_LABELS.get(method, method)}")
        t_method_start = time.perf_counter()

        # Accumulators across seeds
        cov_grids_sum = np.zeros((n_drones, n_tilts, n_azimuths))
        default_tilts_sum = np.zeros(n_drones)
        default_azimuths_sum = np.zeros(n_drones)
        baseline_metrics_list = []
        optimized_metrics_list = []

        for s in range(n_eval_seeds):
            seed = eval_base_seed + s
            rng = np.random.default_rng(seed)
            user_pos = generate_users_for_distribution(distribution, target_users, config, rng)

            # Deploy
            scenario = deploy_method(
                method, config, user_pos, seed,
                repulsive_beta=repulsive_beta, n_altitude_tiers=n_altitude_tiers,
            )

            # Baseline evaluation
            baseline_m = _evaluate_single(scenario, config, dl_antenna, bh_antenna, channel)
            baseline_metrics_list.append(metrics_to_dict(baseline_m, zeta))

            # Record default orientations
            default_tilts_sum += np.rad2deg(scenario.dl_tilt_rad)
            default_azimuths_sum += np.rad2deg(scenario.dl_azimuth_rad)

            # Per-drone sweep
            per_drone_best_tilt = scenario.dl_tilt_rad.copy()
            per_drone_best_az = scenario.dl_azimuth_rad.copy()

            for j in range(n_drones):
                best_score = -np.inf

                for ti, t_rad in enumerate(tilt_rad):
                    # Build modified DL arrays (override drone j)
                    dl_tilt_arr = scenario.dl_tilt_rad.copy()
                    dl_tilt_arr[j] = t_rad

                    for ai, a_rad in enumerate(az_rad):
                        dl_az_arr = scenario.dl_azimuth_rad.copy()
                        dl_az_arr[j] = a_rad

                        mod_sc = Scenario(
                            user_positions=user_pos,
                            drone_positions=scenario.drone_positions,
                            dl_tilt_rad=dl_tilt_arr,
                            dl_azimuth_rad=dl_az_arr,
                            bh_tilt_rad=scenario.bh_tilt_rad,
                            bh_azimuth_rad=scenario.bh_azimuth_rad,
                            area_size_m=scenario.area_size_m,
                        )
                        m = _evaluate_single(mod_sc, config, dl_antenna, bh_antenna, channel)
                        cov = float(m.coverage_fraction * 100)
                        cov_grids_sum[j, ti, ai] += cov

                        score = combined_score(
                            cov, float(m.total_inter_drone_interference_dbm), zeta,
                        )
                        if score > best_score:
                            best_score = score
                            per_drone_best_tilt[j] = t_rad
                            per_drone_best_az[j] = a_rad

            # Evaluate optimized: all drones at their per-drone-best angles
            opt_sc = Scenario(
                user_positions=user_pos,
                drone_positions=scenario.drone_positions,
                dl_tilt_rad=per_drone_best_tilt,
                dl_azimuth_rad=per_drone_best_az,
                bh_tilt_rad=scenario.bh_tilt_rad,
                bh_azimuth_rad=scenario.bh_azimuth_rad,
                area_size_m=scenario.area_size_m,
            )
            opt_m = _evaluate_single(opt_sc, config, dl_antenna, bh_antenna, channel)
            optimized_metrics_list.append(metrics_to_dict(opt_m, zeta))

            bl_cov = baseline_metrics_list[-1]["coverage_pct"]
            op_cov = optimized_metrics_list[-1]["coverage_pct"]
            print(f"    seed {seed}: baseline_cov={bl_cov:.1f}%  optimized_cov={op_cov:.1f}%"
                  f"  (delta={op_cov - bl_cov:+.1f}pp)")

        # Average grids and defaults
        cov_grids_avg = cov_grids_sum / n_eval_seeds
        default_tilts_avg = default_tilts_sum / n_eval_seeds
        default_azimuths_avg = default_azimuths_sum / n_eval_seeds

        # Find best angles from averaged grid (robust across seeds)
        per_drone_data = []
        for j in range(n_drones):
            grid = cov_grids_avg[j]
            best_idx = np.unravel_index(np.argmax(grid), grid.shape)
            best_t = float(tilt_angles_deg[best_idx[0]])
            best_a = float(azimuths_deg[best_idx[1]])

            drone_info = {
                "drone_idx": j,
                "avg_default_tilt_deg": round(float(default_tilts_avg[j]), 2),
                "avg_default_azimuth_deg": round(float(default_azimuths_avg[j]), 2),
                "best_tilt_deg": best_t,
                "best_azimuth_deg": best_a,
                "best_coverage_pct": round(float(grid[best_idx]), 2),
            }
            if save_grids:
                drone_info["coverage_grid"] = np.round(grid, 2).tolist()
            per_drone_data.append(drone_info)

        method_elapsed = time.perf_counter() - t_method_start
        baseline_avg = average_metric_dicts(baseline_metrics_list)
        optimized_avg = average_metric_dicts(optimized_metrics_list)

        delta_cov = optimized_avg["coverage_pct"] - baseline_avg["coverage_pct"]
        delta_score = optimized_avg["combined_score"] - baseline_avg["combined_score"]
        print(f"    {METHOD_LABELS.get(method, method)}: "
              f"baseline_cov={baseline_avg['coverage_pct']:.1f}% -> "
              f"optimized_cov={optimized_avg['coverage_pct']:.1f}% "
              f"(delta={delta_cov:+.1f}pp, delta_score={delta_score:+.4f})  "
              f"[{method_elapsed:.1f}s]\n")

        all_results[method] = {
            "baseline": baseline_avg,
            "baseline_per_seed": baseline_metrics_list,
            "optimized": optimized_avg,
            "optimized_per_seed": optimized_metrics_list,
            "per_drone": per_drone_data,
        }

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"{'Method':<18s}  {'Base Cov':>8s}  {'Opt Cov':>8s}  {'Delta':>7s}  "
          f"{'Base Score':>10s}  {'Opt Score':>10s}")
    print(f"{'=' * 70}")
    for method, r in all_results.items():
        bl = r["baseline"]
        op = r["optimized"]
        print(f"  {METHOD_LABELS.get(method, method):<16s}  "
              f"{bl['coverage_pct']:7.1f}%  {op['coverage_pct']:7.1f}%  "
              f"{op['coverage_pct'] - bl['coverage_pct']:+6.1f}  "
              f"{bl['combined_score']:10.4f}  {op['combined_score']:10.4f}")

    return {
        "config": {
            "n_drones": n_drones,
            "target_users": target_users,
            "distribution": distribution,
            "n_eval_seeds": n_eval_seeds,
            "tilt_angles_deg": tilt_angles_deg,
            "azimuths_deg": azimuths_deg,
            "zeta": zeta,
            "methods": methods,
        },
        "results": all_results,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Per-drone DL orientation sweep experiment",
    )
    parser.add_argument("--n-drones", type=int, default=10)
    parser.add_argument("--target-users", type=int, default=200)
    parser.add_argument("--distribution", type=str, default="clustered",
                        choices=["clustered", "uniform", "hotspot"])
    parser.add_argument("--distributions", type=str, nargs="+",
                        default=DISTRIBUTIONS_DEFAULT)
    parser.add_argument("--drone-counts", type=int, nargs="+",
                        default=DRONE_COUNTS_DEFAULT)
    parser.add_argument("--user-counts", type=int, nargs="+",
                        default=TARGET_USERS_DEFAULT)
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help=f"Heuristic methods to sweep (default: {DEFAULT_METHODS})")
    parser.add_argument("--n-eval-seeds", type=int, default=10)
    parser.add_argument("--zeta", type=float, default=0.7)
    parser.add_argument("--tilt-max", type=int, default=TILT_MAX_DEG)
    parser.add_argument("--tilt-step", type=int, default=TILT_STEP_DEG)
    parser.add_argument("--azimuth-step", type=int, default=AZIMUTH_STEP_DEG)
    parser.add_argument("--repulsive-beta", type=float, default=0.15)
    parser.add_argument("--altitude-tiers", type=int, default=3)
    parser.add_argument("--no-grids", action="store_true",
                        help="Omit per-drone coverage grids from output (smaller files)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Run one task by index (for SLURM array jobs)")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: N=5, 100u, 2 seeds, coarse grid")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    tilt_angles = list(range(0, args.tilt_max + 1, args.tilt_step))
    azimuths = list(range(0, 360, args.azimuth_step))
    methods = args.methods or DEFAULT_METHODS
    save_grids = not args.no_grids

    # ── Quick mode ────────────────────────────────────────────────────
    if args.quick:
        data = run_experiment(
            n_drones=5, target_users=100, distribution="clustered",
            methods=["kmeans", "analytic"],
            tilt_angles_deg=list(range(0, 31, 5)),
            azimuths_deg=list(range(0, 360, 30)),
            n_eval_seeds=2, zeta=args.zeta,
            save_grids=save_grids,
        )
        output = args.output or "results/angle_sweep_quick.json"
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {output}")
        return

    # ── Task grid ─────────────────────────────────────────────────────
    task_grid = generate_task_grid(args.drone_counts, args.user_counts, args.distributions)

    if args.list_tasks:
        print(f"Sweep grid: {len(task_grid)} tasks")
        print(f"  Drone counts:   {sorted(set(t['n_drones'] for t in task_grid))}")
        print(f"  Target users:   {sorted(set(t['target_users'] for t in task_grid))}")
        print(f"  Distributions:  {sorted(set(t['distribution'] for t in task_grid))}")
        print()
        for t in task_grid:
            n_evals = args.n_eval_seeds * len(methods) * t["n_drones"] * len(tilt_angles) * len(azimuths)
            print(f"  task {t['task_id']:3d}: drones={t['n_drones']:2d}, "
                  f"users~{t['target_users']:3d}, dist={t['distribution']:<10s}  "
                  f"({n_evals:,} evals)")
        return

    # ── SLURM array task mode ─────────────────────────────────────────
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(task_grid):
            print(f"Error: --task-id must be in [0, {len(task_grid) - 1}]")
            sys.exit(1)
        task = task_grid[args.task_id]
        data = run_experiment(
            n_drones=task["n_drones"],
            target_users=task["target_users"],
            distribution=task["distribution"],
            methods=methods,
            tilt_angles_deg=tilt_angles,
            azimuths_deg=azimuths,
            n_eval_seeds=args.n_eval_seeds,
            zeta=args.zeta,
            repulsive_beta=args.repulsive_beta,
            n_altitude_tiers=args.altitude_tiers,
            save_grids=save_grids,
        )
        output_dir = Path(args.output) if args.output else Path("results/angle_sweep")
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = (output_dir /
                 f"task_{args.task_id:03d}_d{task['n_drones']}_u{task['target_users']}"
                 f"_{task['distribution']}.json")
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {fname}")
        return

    # ── Single run mode ───────────────────────────────────────────────
    data = run_experiment(
        n_drones=args.n_drones,
        target_users=args.target_users,
        distribution=args.distribution,
        methods=methods,
        tilt_angles_deg=tilt_angles,
        azimuths_deg=azimuths,
        n_eval_seeds=args.n_eval_seeds,
        zeta=args.zeta,
        repulsive_beta=args.repulsive_beta,
        n_altitude_tiers=args.altitude_tiers,
        save_grids=save_grids,
    )
    output = args.output or "results/angle_sweep_results.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
