"""Heuristic experiment: user-aware, interference-focused evaluation.

Drones know user positions before deployment. The primary focus is on
interference reduction: each method deploys on the SAME user layout
it is evaluated on.

Methods compared
----------------
kmeans          : k-means (x,y) + fixed altitude + nadir DL + MST BH
analytic        : k-means (x,y) + cluster-spread altitude + centroid DL + MST BH
analytic_pca    : k-means (x,y) + cluster-spread altitude + PCA DL + MST BH
repulsive_lloyd : Lloyd's iteration with repulsive forces + analytic orientations
altitude_stagger: analytic placement + graph-coloured altitude tiers
greedy_fine     : greedy 40x40 grid (1600 candidates) + analytic orientations
ia_greedy       : interference-aware greedy (combined score selection criterion)

User distributions
------------------
clustered       : 5 Gaussian clusters, spread=100 m (default, models shelters)
uniform         : uniformly distributed across area
hotspot         : 3 dense clusters, spread=50 m (concentrated demand)

Usage
-----
# Quick smoke-test
python scripts/run_heuristic_experiment.py --quick

# Single configuration
python scripts/run_heuristic_experiment.py --n-drones 15 --target-users 200

# Single task by index (for SLURM array jobs)
python scripts/run_heuristic_experiment.py --task-id 0 --output results/heuristic/

# List all tasks in sweep grid
python scripts/run_heuristic_experiment.py --list-tasks
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

from dronecomm.config import Config, NetworkConfig, ScenarioConfig
from dronecomm.heuristic import (
    deploy_analytic_heuristic,
    deploy_analytic_pca_heuristic,
    deploy_repulsive_lloyd_heuristic,
    deploy_altitude_staggered_heuristic,
    deploy_interference_aware_greedy,
    generate_grid_candidates,
    greedy_discrete_placement,
    exhaustive_discrete_placement,
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


# ── Default sweep parameters ──────────────────────────────────────────

DRONE_COUNTS_DEFAULT = list(range(5, 31))   # 5, 6, 7, …, 30
TARGET_USERS_DEFAULT = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
DISTRIBUTIONS_DEFAULT = ["clustered", "uniform", "hotspot"]
N_CLUSTERS_CLUSTERED = 5
N_CLUSTERS_HOTSPOT = 3


def target_users_to_cluster_mean(target_users: int, n_clusters: int) -> int:
    return max(10, target_users // n_clusters)


def make_config(n_drones: int, target_users: int, distribution: str = "clustered") -> Config:
    base = Config()
    if distribution == "hotspot":
        n_clusters = N_CLUSTERS_HOTSPOT
    else:
        n_clusters = N_CLUSTERS_CLUSTERED
    users_per_cluster_mean = target_users_to_cluster_mean(target_users, n_clusters)
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


def generate_task_grid(
    drone_counts: list[int],
    target_users: list[int],
    distributions: list[str],
) -> list[dict]:
    return [
        {"task_id": i, "n_drones": nd, "target_users": tu, "distribution": dist}
        for i, (nd, tu, dist) in enumerate(product(drone_counts, target_users, distributions))
    ]


# ── User generation per distribution ─────────────────────────────────


def generate_users_for_distribution(
    distribution: str,
    target_users: int,
    config: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate user positions for the given distribution type."""
    sc_cfg = config.scenario
    if distribution == "clustered":
        return generate_users(
            n_clusters=sc_cfg.n_clusters,
            users_per_cluster_mean=sc_cfg.users_per_cluster_mean,
            users_per_cluster_std=sc_cfg.users_per_cluster_std,
            cluster_spread_m=sc_cfg.cluster_spread_m,
            area_size_m=sc_cfg.area_size_m,
            rng=rng,
        )
    elif distribution == "uniform":
        return generate_users_uniform(
            n_users=target_users,
            area_size_m=sc_cfg.area_size_m,
            rng=rng,
        )
    elif distribution == "hotspot":
        return generate_users_hotspot(
            n_clusters=N_CLUSTERS_HOTSPOT,
            users_per_cluster_mean=sc_cfg.users_per_cluster_mean,
            users_per_cluster_std=sc_cfg.users_per_cluster_std,
            cluster_spread_m=50.0,
            area_size_m=sc_cfg.area_size_m,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ── Metrics extraction ────────────────────────────────────────────────

# Interference normalization bounds (dBm) — used for combined score.
# Based on expected range across drone counts and distributions.
# Lower bound = best case (fewest drones, most separated).
# Upper bound = worst case (most drones, closest packed).
INTERF_BEST_DBM = -70.0   # very low interference (few drones)
INTERF_WORST_DBM = -30.0  # high interference (many drones, dense)


def combined_score(coverage_pct: float, total_interference_dbm: float, zeta: float) -> float:
    """Weighted tradeoff score in [0, 1].

    score = zeta * coverage_01 + (1 - zeta) * interference_01

    where coverage_01  = coverage_pct / 100          (higher is better)
          interference_01 = normalised so that lower interference → higher score

    Parameters
    ----------
    coverage_pct : float
        Coverage percentage [0, 100].
    total_interference_dbm : float
        Total inter-drone interference [dBm]. Lower (more negative) is better.
    zeta : float
        Weight on coverage [0, 1]. zeta=1 → pure coverage, zeta=0 → pure interference.
    """
    cov_01 = coverage_pct / 100.0
    interf_01 = (INTERF_WORST_DBM - total_interference_dbm) / (INTERF_WORST_DBM - INTERF_BEST_DBM)
    interf_01 = float(np.clip(interf_01, 0.0, 1.0))
    return float(zeta * cov_01 + (1.0 - zeta) * interf_01)


def metrics_to_dict(m, zeta: float = 0.7) -> dict:
    """Extract key metrics including interference and combined score."""
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
    """Average a list of metric dicts."""
    avg = {}
    for key in dicts[0]:
        vals = [d[key] for d in dicts]
        avg[key] = float(np.mean(vals))
    return avg


# ── Deploy k-means baseline (user-aware) ──────────────────────────────


def deploy_kmeans_baseline(
    config: Config,
    user_positions: np.ndarray,
    seed: int = 42,
) -> Scenario:
    """K-means placement with fixed altitude, nadir DL, MST BH."""
    from scipy.cluster.vq import kmeans2

    n = config.network.n_drones
    xy = user_positions[:, :2]
    centroids, _ = kmeans2(xy, n, minit="points", seed=seed)

    drone_pos = np.zeros((n, 3))
    drone_pos[:, :2] = centroids
    drone_pos[:, 2] = config.network.altitude_m

    gw_idx = _gateway_idx(drone_pos)
    bh_tilt, bh_azimuth = mst_backhaul_orientations(drone_pos, gateway_idx=gw_idx)

    return Scenario(
        user_positions=user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=np.zeros(n),              # nadir
        dl_azimuth_rad=np.zeros(n),
        bh_tilt_rad=bh_tilt,
        bh_azimuth_rad=bh_azimuth,
        area_size_m=config.scenario.area_size_m,
    )


# ── Core experiment runner ─────────────────────────────────────────────


def run_experiment(
    n_drones: int = 15,
    target_users: int = 200,
    distribution: str = "clustered",
    n_eval_seeds: int = 20,
    eval_base_seed: int = 1000,
    run_exhaustive: bool = False,
    run_greedy: bool = True,
    run_ia_greedy: bool = False,
    greedy_resolution: int = 40,
    zeta: float = 0.7,
    repulsive_beta: float = 0.15,
    n_altitude_tiers: int = 3,
) -> dict:
    config = make_config(n_drones, target_users, distribution)
    area = config.scenario.area_size_m

    print(f"\n{'='*60}")
    print(f"Heuristic Experiment (user-aware)")
    print(f"  N={n_drones} drones  |  ~{target_users} users  |  dist={distribution}")
    print(f"  area={area:.0f}m  |  eval_seeds={n_eval_seeds}")
    print(f"  greedy={run_greedy}  |  ia_greedy={run_ia_greedy}  |  exhaustive={run_exhaustive}")
    print(f"  repulsive_beta={repulsive_beta}  |  altitude_tiers={n_altitude_tiers}")
    print(f"{'='*60}\n")

    dl_antenna, bh_antenna, channel = _build_models(config)

    method_names = [
        "kmeans", "analytic", "analytic_pca",
        "repulsive_lloyd", "altitude_stagger",
    ]
    if run_greedy:
        method_names.append("greedy_fine")
    if run_ia_greedy:
        method_names.append("ia_greedy")
    if run_exhaustive:
        method_names.append("exhaustive_coarse")

    # Pre-generate candidate grids
    fine_candidates = None
    coarse_candidates = None
    if run_greedy or run_ia_greedy:
        fine_candidates = generate_grid_candidates(area, resolution=greedy_resolution)
    if run_exhaustive:
        coarse_candidates = generate_grid_candidates(area, resolution=5)

    # ── Per-seed loop: deploy + evaluate ──────────────────────────────
    all_metrics = {m: [] for m in method_names}
    all_build_times = {m: [] for m in method_names}

    for s in range(n_eval_seeds):
        seed = eval_base_seed + s
        rng = np.random.default_rng(seed)
        user_pos = generate_users_for_distribution(distribution, target_users, config, rng)

        for method in method_names:
            t0 = time.perf_counter()

            if method == "kmeans":
                sc = deploy_kmeans_baseline(config, user_pos, seed=seed)
            elif method == "analytic":
                sc = deploy_analytic_heuristic(config, user_pos, seed=seed)
            elif method == "analytic_pca":
                sc = deploy_analytic_pca_heuristic(config, user_pos, seed=seed)
            elif method == "repulsive_lloyd":
                sc = deploy_repulsive_lloyd_heuristic(
                    config, user_pos, seed=seed, beta=repulsive_beta,
                )
            elif method == "altitude_stagger":
                sc = deploy_altitude_staggered_heuristic(
                    config, user_pos, seed=seed, n_tiers=n_altitude_tiers,
                )
            elif method == "greedy_fine":
                sc = greedy_discrete_placement(
                    config, fine_candidates, user_pos, verbose=False
                )
            elif method == "ia_greedy":
                sc = deploy_interference_aware_greedy(
                    config, fine_candidates, user_pos, zeta=zeta, verbose=False
                )
            elif method == "exhaustive_coarse":
                n_cands = coarse_candidates.shape[0]
                if n_drones > n_cands:
                    continue
                sc = exhaustive_discrete_placement(
                    config, coarse_candidates, user_pos, verbose=False
                )

            build_t = time.perf_counter() - t0
            metrics = _evaluate_single(sc, config, dl_antenna, bh_antenna, channel)
            all_metrics[method].append(metrics_to_dict(metrics, zeta=zeta))
            all_build_times[method].append(build_t)

        if (s + 1) % 5 == 0 or s == 0:
            parts = []
            for m in method_names:
                if all_metrics[m]:
                    parts.append(f"{m}={all_metrics[m][-1]['coverage_pct']:.1f}%")
            print(f"  seed {seed}: {' '.join(parts)}  ({len(user_pos)} users)")

    # ── Aggregate results ─────────────────────────────────────────────
    results = {}
    for method in method_names:
        if not all_metrics[method]:
            continue
        avg = average_metric_dicts(all_metrics[method])
        avg["build_time_s"] = float(np.mean(all_build_times[method]))
        avg["build_time_total_s"] = float(np.sum(all_build_times[method]))
        results[method] = avg

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Method':<22s}  {'Cov%':>6s}  {'SINR':>6s}  {'Interf':>8s}  {'WrstPr':>8s}  {'Build':>7s}")
    print(f"{'─'*70}")
    for method, r in results.items():
        print(f"  {method:<20s}  {r['coverage_pct']:5.1f}%  {r['mean_sinr_db']:5.1f}dB"
              f"  {r['total_interference_dbm']:7.1f}  {r['worst_pair_interference_dbm']:7.1f}"
              f"  {r['build_time_s']:6.2f}s")

    baseline_cov = results["kmeans"]["coverage_pct"]
    baseline_interf = results["kmeans"]["total_interference_dbm"]
    print(f"\nΔ vs k-means:")
    for method, r in results.items():
        if method == "kmeans":
            continue
        dcov = r["coverage_pct"] - baseline_cov
        dintf = r["total_interference_dbm"] - baseline_interf
        print(f"  {method:<20s}  Δcov={dcov:+.1f}pp  Δinterf={dintf:+.1f}dBm")

    return {
        "config": {
            "n_drones": n_drones,
            "target_users": target_users,
            "distribution": distribution,
            "area_size_m": float(area),
            "n_eval_seeds": n_eval_seeds,
            "zeta": zeta,
            "repulsive_beta": repulsive_beta,
            "n_altitude_tiers": n_altitude_tiers,
            "user_aware": True,
        },
        "results": results,
        "per_seed": {m: all_metrics[m] for m in method_names if all_metrics[m]},
    }


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Heuristic drone deployment experiment (user-aware)")
    parser.add_argument("--n-drones", type=int, default=15)
    parser.add_argument("--target-users", type=int, default=200)
    parser.add_argument("--distribution", type=str, default="clustered",
                        choices=["clustered", "uniform", "hotspot"],
                        help="User distribution type")
    parser.add_argument("--distributions", type=str, nargs="+",
                        default=DISTRIBUTIONS_DEFAULT,
                        help="Distribution types for sweep grid")
    parser.add_argument("--drone-counts", type=int, nargs="+", default=DRONE_COUNTS_DEFAULT)
    parser.add_argument("--user-counts", type=int, nargs="+", default=TARGET_USERS_DEFAULT)
    parser.add_argument("--n-eval-seeds", type=int, default=20)
    parser.add_argument("--no-exhaustive", action="store_true",
                        help="Skip exhaustive 5x5 search (default)")
    parser.add_argument("--exhaustive", action="store_true",
                        help="Include exhaustive 5x5 search")
    parser.add_argument("--no-greedy", action="store_true",
                        help="Skip greedy grid search")
    parser.add_argument("--ia-greedy", action="store_true",
                        help="Include interference-aware greedy (slower than regular greedy)")
    parser.add_argument("--greedy-resolution", type=int, default=40,
                        help="Grid resolution for greedy search (default 40 → 40x40=1600 candidates)")
    parser.add_argument("--zeta", type=float, default=0.7,
                        help="Coverage weight in combined score [0,1]. "
                             "zeta=1 → pure coverage, zeta=0 → pure interference (default 0.7)")
    parser.add_argument("--repulsive-beta", type=float, default=0.15,
                        help="Repulsion strength for repulsive Lloyd's [0,1] (default 0.15)")
    parser.add_argument("--altitude-tiers", type=int, default=3,
                        help="Number of altitude tiers for altitude staggering (default 3)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run all drone/user/distribution combinations sequentially")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Run one task by index (for SLURM array jobs)")
    parser.add_argument("--list-tasks", action="store_true",
                        help="Print sweep grid and exit")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: N=5, 100 users, 5 eval seeds, no greedy")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_exhaustive = args.exhaustive
    run_greedy = not args.no_greedy
    run_ia_greedy = args.ia_greedy
    greedy_resolution = args.greedy_resolution
    zeta = args.zeta
    repulsive_beta = args.repulsive_beta
    n_altitude_tiers = args.altitude_tiers

    # ── Quick mode ────────────────────────────────────────────────────
    if args.quick:
        data = run_experiment(
            n_drones=5, target_users=100, distribution="clustered",
            n_eval_seeds=5, run_greedy=False, run_ia_greedy=False,
            run_exhaustive=False, zeta=zeta,
            repulsive_beta=repulsive_beta, n_altitude_tiers=n_altitude_tiers,
        )
        output = args.output or "results/heuristic_quick.json"
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
            print(f"  task {t['task_id']:3d}: drones={t['n_drones']:2d}, "
                  f"users~{t['target_users']:3d}, dist={t['distribution']}")
        return

    # ── SLURM array task mode ─────────────────────────────────────────
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(task_grid):
            print(f"Error: --task-id must be in [0, {len(task_grid)-1}]")
            sys.exit(1)
        task = task_grid[args.task_id]
        data = run_experiment(
            n_drones=task["n_drones"],
            target_users=task["target_users"],
            distribution=task["distribution"],
            n_eval_seeds=args.n_eval_seeds,
            run_exhaustive=run_exhaustive,
            run_greedy=run_greedy,
            run_ia_greedy=run_ia_greedy,
            greedy_resolution=greedy_resolution,
            zeta=zeta,
            repulsive_beta=repulsive_beta,
            n_altitude_tiers=n_altitude_tiers,
        )
        output_dir = Path(args.output) if args.output else Path("results/heuristic")
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = (output_dir /
                 f"task_{args.task_id:03d}_d{task['n_drones']}_u{task['target_users']}"
                 f"_{task['distribution']}.json")
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {fname}")
        return

    # ── Sweep mode ────────────────────────────────────────────────────
    if args.sweep:
        all_results = []
        output_dir = Path(args.output) if args.output else Path("results/heuristic")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Sweep: {len(task_grid)} tasks")
        sweep_start = time.perf_counter()
        for i, task in enumerate(task_grid):
            print(f"\n[{i+1}/{len(task_grid)}] drones={task['n_drones']}, "
                  f"users~{task['target_users']}, dist={task['distribution']}")
            data = run_experiment(
                n_drones=task["n_drones"],
                target_users=task["target_users"],
                distribution=task["distribution"],
                n_eval_seeds=args.n_eval_seeds,
                run_exhaustive=run_exhaustive,
                run_greedy=run_greedy,
                run_ia_greedy=run_ia_greedy,
                greedy_resolution=greedy_resolution,
                zeta=zeta,
                repulsive_beta=repulsive_beta,
                n_altitude_tiers=n_altitude_tiers,
            )
            all_results.append(data)
            fname = (output_dir /
                     f"task_{i:03d}_d{task['n_drones']}_u{task['target_users']}"
                     f"_{task['distribution']}.json")
            with open(fname, "w") as f:
                json.dump(data, f, indent=2)

        sweep_elapsed = time.perf_counter() - sweep_start
        summary_path = output_dir / "sweep_results.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSweep complete: {len(task_grid)} tasks in {sweep_elapsed/60:.1f} min")
        print(f"Summary saved to {summary_path}")
        return

    # ── Single run mode ───────────────────────────────────────────────
    data = run_experiment(
        n_drones=args.n_drones,
        target_users=args.target_users,
        distribution=args.distribution,
        n_eval_seeds=args.n_eval_seeds,
        run_exhaustive=run_exhaustive,
        run_greedy=run_greedy,
        run_ia_greedy=run_ia_greedy,
        greedy_resolution=greedy_resolution,
        zeta=zeta,
        repulsive_beta=repulsive_beta,
        n_altitude_tiers=n_altitude_tiers,
    )
    output = args.output or "results/heuristic_results.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
