"""Geometry-aware analytic heuristic for drone deployment.

Instead of black-box optimization, this module exploits problem structure:
  1. Altitude: set so beam footprint matches cluster spread (physics formula)
  2. Downlink orientation: points boresight at cluster centroid (closed-form)
  3. Backhaul orientation: points each drone toward its MST parent (deterministic)
  4. Brute-force: exhaustive or greedy search over a discrete position grid
     using analytic orientations (no black-box evaluation of orientations)

Zero hyperparameters; zero orientation evaluations for the pure analytic heuristic.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

from .config import Config
from .optimize import (
    _build_models,
    _evaluate_single,
    evaluate_scenario,
)
from .scenario import Scenario, generate_users


# ── Analytic geometry helpers ─────────────────────────────────────────


def altitude_from_cluster_spread(
    cluster_spread_m: float,
    dl_beamwidth_deg: float = 60.0,
    z_min: float = 50.0,
    z_max: float = 300.0,
) -> float:
    """Return altitude so the beam footprint radius matches cluster spread.

    Derivation: the DL antenna -3 dB beam edge is at half-beamwidth from
    boresight. When boresight points straight down, the ground footprint
    radius equals z * tan(θ_3dB / 2). Setting that equal to cluster_spread_m:

        z = cluster_spread_m / tan(θ_3dB / 2)

    Parameters
    ----------
    cluster_spread_m : float
        Empirical std of user XY positions within the cluster [m].
    dl_beamwidth_deg : float
        Full -3 dB beamwidth of the downlink antenna [degrees].
    z_min, z_max : float
        Altitude clamp bounds [m].
    """
    half_bw = np.deg2rad(dl_beamwidth_deg / 2.0)
    z = cluster_spread_m / np.tan(half_bw)
    return float(np.clip(z, z_min, z_max))


def dl_orientation_from_centroid(
    drone_pos: np.ndarray,
    centroid_xy: np.ndarray,
) -> tuple[float, float]:
    """Return (dl_tilt_rad, dl_azimuth_rad) pointing boresight at ground centroid.

    Parameters
    ----------
    drone_pos : np.ndarray, shape (3,)
        Drone position (x, y, z).
    centroid_xy : np.ndarray, shape (2,)
        Ground cluster centroid (x, y), z=0 assumed.

    Returns
    -------
    dl_tilt_rad : float
        Tilt from nadir toward centroid [rad]. 0 = straight down.
    dl_azimuth_rad : float
        Azimuth from +x axis toward centroid [rad], in [0, 2π).
    """
    dx = centroid_xy[0] - drone_pos[0]
    dy = centroid_xy[1] - drone_pos[1]
    horiz = float(np.sqrt(dx**2 + dy**2))
    tilt = float(np.arctan2(horiz, max(float(drone_pos[2]), 1e-3)))
    azimuth = float(np.arctan2(dy, dx)) % (2.0 * np.pi)
    return tilt, azimuth


def mst_backhaul_orientations(
    drone_positions: np.ndarray,
    gateway_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """BH tilt and azimuth for each drone based on MST topology.

    Builds a Minimum Spanning Tree over drone 3D positions, roots it at
    `gateway_idx`, then points each non-gateway drone's BH antenna toward
    its parent in the tree.

    The gateway drone's BH antenna points horizontally toward ground
    infrastructure (tilt = π/2, azimuth = 0).

    The existing mesh interference model is unchanged — this only sets
    antenna pointing directions.

    Parameters
    ----------
    drone_positions : np.ndarray, shape (N, 3)
    gateway_idx : int
        Index of the gateway (root) drone.

    Returns
    -------
    bh_tilt_rad : np.ndarray, shape (N,)
    bh_azimuth_rad : np.ndarray, shape (N,)
    """
    N = drone_positions.shape[0]

    # Pairwise 3D distances for MST
    diff = drone_positions[:, np.newaxis, :] - drone_positions[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)  # (N, N)

    # Build MST (returns sparse upper-triangular matrix of MST edges)
    mst = minimum_spanning_tree(csr_matrix(dist_matrix))
    mst_arr = mst.toarray()
    # Make symmetric so we can traverse in both directions
    adj = (mst_arr + mst_arr.T) > 0  # (N, N) bool

    # BFS from gateway to find parent of each node
    parent = np.full(N, -1, dtype=int)
    visited = np.zeros(N, dtype=bool)
    queue = [gateway_idx]
    visited[gateway_idx] = True
    while queue:
        node = queue.pop(0)
        for neighbor in np.where(adj[node])[0]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                queue.append(int(neighbor))

    # Compute orientations
    bh_tilt = np.full(N, np.pi / 2.0)    # default: horizontal
    bh_azimuth = np.zeros(N)

    for i in range(N):
        if i == gateway_idx:
            # Gateway points horizontally (toward ground base station)
            bh_tilt[i] = np.pi / 2.0
            bh_azimuth[i] = 0.0
        else:
            p = parent[i]
            if p < 0:
                continue  # disconnected (shouldn't happen)
            dx = drone_positions[p, 0] - drone_positions[i, 0]
            dy = drone_positions[p, 1] - drone_positions[i, 1]
            dz = drone_positions[p, 2] - drone_positions[i, 2]
            horiz = float(np.sqrt(dx**2 + dy**2))
            # Tilt from nadir: 0 = down, π/2 = horizontal, π = up
            # Parent can be above or below; arctan2 gives signed elevation
            # We need tilt from nadir so: tilt = π/2 - elevation_angle
            elevation = float(np.arctan2(abs(dz), max(horiz, 1e-6)))
            bh_tilt[i] = np.pi / 2.0 - elevation if dz >= 0 else np.pi / 2.0 + elevation
            bh_tilt[i] = float(np.clip(bh_tilt[i], 0.0, np.pi))
            bh_azimuth[i] = float(np.arctan2(dy, dx)) % (2.0 * np.pi)

    return bh_tilt, bh_azimuth


# ── Cluster statistics ────────────────────────────────────────────────


def assign_users_to_drones(
    drone_positions: np.ndarray,
    user_positions: np.ndarray,
) -> np.ndarray:
    """Return cluster index (nearest drone) for each user.

    Parameters
    ----------
    drone_positions : np.ndarray, shape (N, 3)
    user_positions : np.ndarray, shape (M, 3)

    Returns
    -------
    np.ndarray, shape (M,), dtype int
        Index in [0, N) of the nearest drone for each user.
    """
    # (M, N) distance matrix (2D horizontal only, consistent with sinr.py)
    diff = user_positions[:, np.newaxis, :] - drone_positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)  # (M, N)
    return np.argmin(dists, axis=1)


class ClusterStats(NamedTuple):
    centroid_xy: np.ndarray    # (2,) — ground centroid
    spread_m: float             # empirical std of user xy from centroid
    n_users: int


def compute_cluster_stats(
    drone_positions: np.ndarray,
    user_positions: np.ndarray,
) -> list[ClusterStats]:
    """Centroid, spread, and user count for each drone's cluster.

    Parameters
    ----------
    drone_positions : np.ndarray, shape (N, 3)
    user_positions : np.ndarray, shape (M, 3)

    Returns
    -------
    list of ClusterStats, length N
    """
    N = drone_positions.shape[0]
    assignment = assign_users_to_drones(drone_positions, user_positions)
    stats = []
    for j in range(N):
        mask = assignment == j
        if mask.sum() == 0:
            # No users assigned: use drone xy as centroid, spread = config default
            centroid = drone_positions[j, :2].copy()
            spread = 100.0
            n_users = 0
        else:
            xy = user_positions[mask, :2]
            centroid = xy.mean(axis=0)
            spread = float(np.std(xy - centroid)) if mask.sum() > 1 else 50.0
            spread = max(spread, 10.0)   # avoid near-zero spread
            n_users = int(mask.sum())
        stats.append(ClusterStats(centroid_xy=centroid, spread_m=spread, n_users=n_users))
    return stats


# ── Full analytic deployment ──────────────────────────────────────────


def _gateway_idx(drone_positions: np.ndarray) -> int:
    """Return index of the drone closest to the (0, 0) corner."""
    dists = np.linalg.norm(drone_positions[:, :2], axis=1)
    return int(np.argmin(dists))


def apply_analytic_orientations(
    scenario: Scenario,
    config: Config,
) -> Scenario:
    """Replace orientation arrays in a scenario with analytic values.

    Computes per-drone DL orientation from cluster centroid geometry and BH
    orientation from MST topology. Drone positions and user layout unchanged.

    Also sets per-drone altitude from cluster spread if the scenario has a
    fixed altitude (all drones at same height).

    Parameters
    ----------
    scenario : Scenario
        Input scenario (positions already set).
    config : Config
        For antenna beamwidth.

    Returns
    -------
    Scenario with updated dl_tilt_rad, dl_azimuth_rad, bh_tilt_rad,
    bh_azimuth_rad (and optionally drone_positions[:, 2]).
    """
    N = scenario.drone_positions.shape[0]
    stats = compute_cluster_stats(scenario.drone_positions, scenario.user_positions)

    # Per-drone altitude from cluster spread
    dl_bw = config.antenna.dl_beamwidth_deg
    drone_pos = scenario.drone_positions.copy()
    for j, s in enumerate(stats):
        drone_pos[j, 2] = altitude_from_cluster_spread(s.spread_m, dl_bw)

    # DL orientation: point at cluster centroid
    dl_tilt = np.zeros(N)
    dl_azimuth = np.zeros(N)
    for j, s in enumerate(stats):
        tilt, az = dl_orientation_from_centroid(drone_pos[j], s.centroid_xy)
        dl_tilt[j] = tilt
        dl_azimuth[j] = az

    # BH orientation: MST toward gateway
    gw_idx = _gateway_idx(drone_pos)
    bh_tilt, bh_azimuth = mst_backhaul_orientations(drone_pos, gateway_idx=gw_idx)

    return Scenario(
        user_positions=scenario.user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=dl_tilt,
        dl_azimuth_rad=dl_azimuth,
        bh_tilt_rad=bh_tilt,
        bh_azimuth_rad=bh_azimuth,
        area_size_m=scenario.area_size_m,
    )


def deploy_analytic_heuristic(
    config: Config,
    user_positions: np.ndarray,
    seed: int = 42,
) -> Scenario:
    """Full geometry-aware analytic deployment (user-aware).

    Pipeline:
    1. Place drone (x, y) at k-means cluster centroids of known users
    2. Set altitude per drone to match beam footprint to cluster spread
    3. Set DL orientation analytically from cluster centroid
    4. Set BH orientation from MST (rooted at most-central gateway drone)

    Parameters
    ----------
    config : Config
    user_positions : np.ndarray, shape (M, 3)
        Known user positions used for deployment planning.
    seed : int
        RNG seed for k-means initialisation.

    Returns
    -------
    Scenario
        A fully configured scenario with analytic orientations.
    """
    net_cfg = config.network

    # 1. Place (x, y) via k-means on known users
    n = net_cfg.n_drones
    xy = user_positions[:, :2]
    centroids, _ = kmeans2(xy, n, minit="points", seed=seed)

    drone_pos = np.zeros((n, 3))
    drone_pos[:, :2] = centroids
    drone_pos[:, 2] = net_cfg.altitude_m   # temporary; overwritten below

    # Placeholder scenario to compute cluster stats
    tmp_scenario = Scenario(
        user_positions=user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=np.zeros(n),
        dl_azimuth_rad=np.zeros(n),
        bh_tilt_rad=np.full(n, np.pi / 2),
        bh_azimuth_rad=np.zeros(n),
        area_size_m=config.scenario.area_size_m,
    )

    # 2-4. Apply analytic orientations + altitude
    return apply_analytic_orientations(tmp_scenario, config)


# ── Analytic + PCA heuristic ─────────────────────────────────────────


def pca_dl_orientations(
    drone_positions: np.ndarray,
    user_positions: np.ndarray,
    assignment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-drone DL tilt and azimuth derived from PCA of each user cluster.

    For each drone j, performs PCA on the (x, y) residuals of its assigned
    users relative to the drone footprint:

        xy_rel = user_positions[mask, :2] − drone_positions[j, :2]
        Σ      = Cov(xy_rel)
        λ₁, v₁ = largest eigenvalue / eigenvector of Σ

    The DL beam is then tilted along the major axis by

        θ_tilt = arctan( σ_major / z )   with σ_major = √λ₁

    so the beam centre lands one σ_major away from the nadir point along
    the cluster's principal axis.

    Falls back to nadir (tilt = 0, az = 0) for clusters with fewer than 3
    assigned users (covariance estimate unreliable).

    Parameters
    ----------
    drone_positions : (N, 3)  — drones at their target altitudes
    user_positions  : (M, 3)
    assignment      : (M,) int or None
        Serving-drone index per user.  If None, computed via
        nearest-drone (3-D Euclidean) assignment.

    Returns
    -------
    dl_tilt_rad    : (N,)
    dl_azimuth_rad : (N,)
    """
    N = drone_positions.shape[0]
    if assignment is None:
        assignment = assign_users_to_drones(drone_positions, user_positions)

    dl_tilt    = np.zeros(N)
    dl_azimuth = np.zeros(N)

    for j in range(N):
        mask = assignment == j
        if mask.sum() < 3:
            continue   # nadir fallback

        xy_rel = user_positions[mask, :2] - drone_positions[j, :2]
        cov = np.cov(xy_rel.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)   # ascending order
        major_vec   = eigenvectors[:, -1]
        sigma_major = float(np.sqrt(max(eigenvalues[-1], 1e-3)))
        z           = max(float(drone_positions[j, 2]), 1e-3)

        dl_tilt[j]    = float(np.arctan2(sigma_major, z))
        dl_azimuth[j] = float(np.arctan2(major_vec[1], major_vec[0])) % (2.0 * np.pi)

    return dl_tilt, dl_azimuth


def apply_analytic_pca_orientations(
    scenario: Scenario,
    config: Config,
) -> Scenario:
    """Replace orientations with Analytic+PCA values.

    Identical to ``apply_analytic_orientations`` except the DL beam is aimed
    using PCA of each cluster's user spread rather than pointing at the
    cluster centroid.

    Steps
    -----
    1. Per-drone altitude from cluster spread  (same as pure analytic)
    2. DL orientation via PCA of user cluster  (replaces centroid pointing)
    3. BH orientation from MST topology        (same as pure analytic)

    Parameters
    ----------
    scenario : Scenario
    config   : Config

    Returns
    -------
    Scenario with updated drone altitudes and orientation arrays.
    """
    N = scenario.drone_positions.shape[0]
    stats = compute_cluster_stats(scenario.drone_positions, scenario.user_positions)

    # 1. Per-drone altitude from cluster spread
    dl_bw = config.antenna.dl_beamwidth_deg
    drone_pos = scenario.drone_positions.copy()
    for j, s in enumerate(stats):
        drone_pos[j, 2] = altitude_from_cluster_spread(s.spread_m, dl_bw)

    # 2. DL orientation via PCA
    assignment = assign_users_to_drones(drone_pos, scenario.user_positions)
    dl_tilt, dl_azimuth = pca_dl_orientations(
        drone_pos, scenario.user_positions, assignment
    )

    # 3. BH via MST
    gw_idx = _gateway_idx(drone_pos)
    bh_tilt, bh_azimuth = mst_backhaul_orientations(drone_pos, gateway_idx=gw_idx)

    return Scenario(
        user_positions=scenario.user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=dl_tilt,
        dl_azimuth_rad=dl_azimuth,
        bh_tilt_rad=bh_tilt,
        bh_azimuth_rad=bh_azimuth,
        area_size_m=scenario.area_size_m,
    )


def deploy_analytic_pca_heuristic(
    config: Config,
    user_positions: np.ndarray,
    seed: int = 42,
) -> Scenario:
    """Analytic+PCA deployment heuristic.

    Identical pipeline to ``deploy_analytic_heuristic`` except step 3 uses
    PCA-derived per-drone DL orientation instead of centroid pointing.

    Pipeline
    --------
    1. Place drone (x, y) at K-means cluster centroids
    2. Set per-drone altitude to match beam footprint to cluster spread
    3. Set DL orientation via PCA of user cluster shape
    4. Set BH orientation from MST (rooted at most-central drone)

    Parameters
    ----------
    config         : Config
    user_positions : (M, 3)
    seed           : int  K-means RNG seed

    Returns
    -------
    Scenario  — fully configured with Analytic+PCA orientations
    """
    n = config.network.n_drones
    centroids, _ = kmeans2(user_positions[:, :2], n, minit="points", seed=seed)

    drone_pos = np.zeros((n, 3))
    drone_pos[:, :2] = centroids
    drone_pos[:, 2] = config.network.altitude_m   # temporary; overwritten below

    tmp_scenario = Scenario(
        user_positions=user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=np.zeros(n),
        dl_azimuth_rad=np.zeros(n),
        bh_tilt_rad=np.full(n, np.pi / 2),
        bh_azimuth_rad=np.zeros(n),
        area_size_m=config.scenario.area_size_m,
    )

    return apply_analytic_pca_orientations(tmp_scenario, config)


# ── Discrete grid search ─────────────────────────────────────────────


def generate_grid_candidates(
    area_size_m: float,
    resolution: int = 20,
    altitude_m: float | None = None,
) -> np.ndarray:
    """Generate a regular grid of candidate positions.

    Parameters
    ----------
    area_size_m : float
        Side length of the square area [m].
    resolution : int
        Number of grid points per side (resolution × resolution total).
    altitude_m : float or None
        Fixed altitude for all candidates. If None, altitude is set to 0
        and will be overwritten per drone by the analytic formula.

    Returns
    -------
    np.ndarray, shape (resolution², 3)
    """
    margin = area_size_m / (2.0 * resolution)
    xs = np.linspace(margin, area_size_m - margin, resolution)
    ys = np.linspace(margin, area_size_m - margin, resolution)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])   # (K, 2)
    K = xy.shape[0]
    candidates = np.zeros((K, 3))
    candidates[:, :2] = xy
    if altitude_m is not None:
        candidates[:, 2] = altitude_m
    return candidates


def _build_scenario_from_positions(
    drone_xy: np.ndarray,   # (N, 2) — only x, y
    user_pos: np.ndarray,
    config: Config,
) -> Scenario:
    """Build a scenario with analytic orientations for given drone x,y positions."""
    N = drone_xy.shape[0]
    drone_pos = np.zeros((N, 3))
    drone_pos[:, :2] = drone_xy
    drone_pos[:, 2] = config.network.altitude_m   # temp

    tmp = Scenario(
        user_positions=user_pos,
        drone_positions=drone_pos,
        dl_tilt_rad=np.zeros(N),
        dl_azimuth_rad=np.zeros(N),
        bh_tilt_rad=np.full(N, np.pi / 2),
        bh_azimuth_rad=np.zeros(N),
        area_size_m=config.scenario.area_size_m,
    )
    return apply_analytic_orientations(tmp, config)


def _coverage_for_users(
    drone_xy: np.ndarray,
    user_positions: np.ndarray,
    config: Config,
    dl_antenna,
    bh_antenna,
    channel,
) -> float:
    """Coverage fraction for a specific user layout (user-aware)."""
    scenario = _build_scenario_from_positions(drone_xy, user_positions, config)
    metrics = _evaluate_single(scenario, config, dl_antenna, bh_antenna, channel)
    return metrics.coverage_fraction


def greedy_discrete_placement(
    config: Config,
    candidates: np.ndarray,
    user_positions: np.ndarray,
    verbose: bool = False,
) -> Scenario:
    """Greedy sequential drone placement from a candidate set (user-aware).

    Places drones one at a time; each step picks the candidate position
    (from those not yet used) that maximises coverage for the known users.
    Orientations are set analytically at every step.

    Computational cost: N × K evaluations.
    For N=15, K=400 (20×20 grid): ~6,000 evals.

    Parameters
    ----------
    config : Config
    candidates : np.ndarray, shape (K, 3)
        Candidate (x, y, z) positions. z is overridden by analytic formula.
    user_positions : np.ndarray, shape (M, 3)
        Known user positions.
    verbose : bool

    Returns
    -------
    Scenario
        Fully configured scenario with analytic orientations.
    """
    N = config.network.n_drones
    K = candidates.shape[0]
    dl_antenna, bh_antenna, channel = _build_models(config)

    chosen_xy = np.empty((N, 2))
    available = list(range(K))

    for drone_i in range(N):
        best_cov = -np.inf
        best_k = available[0]
        for k in available:
            trial_xy = np.vstack([chosen_xy[:drone_i], candidates[k, :2]])
            cov = _coverage_for_users(trial_xy, user_positions, config, dl_antenna, bh_antenna, channel)
            if cov > best_cov:
                best_cov = cov
                best_k = k
        chosen_xy[drone_i] = candidates[best_k, :2]
        available.remove(best_k)
        if verbose:
            print(f"  Greedy drone {drone_i+1}/{N}: pos={candidates[best_k,:2]}, cov={best_cov:.3f}")

    return _build_scenario_from_positions(chosen_xy, user_positions, config)


def exhaustive_discrete_placement(
    config: Config,
    candidates: np.ndarray,
    user_positions: np.ndarray,
    verbose: bool = False,
) -> Scenario:
    """Exhaustive search over all C(K, N) candidate combinations (user-aware).

    For each combination of N positions from the candidate set, set analytic
    orientations and evaluate coverage for the known users. Returns the best.

    Feasibility on a 5×5 grid (K=25):
        N=5:  C(25,5)  =    53,130 combos
        N=10: C(25,10) = 3,268,760 combos
        N=15: C(25,15) = 3,268,760 combos

    Parameters
    ----------
    config : Config
    candidates : np.ndarray, shape (K, 3)
        Candidate positions (coarse grid, typically 5×5 = 25).
    user_positions : np.ndarray, shape (M, 3)
        Known user positions.
    verbose : bool

    Returns
    -------
    Scenario
        Best configuration found.
    """
    N = config.network.n_drones
    K = candidates.shape[0]
    n_combos = _comb(K, N)

    dl_antenna, bh_antenna, channel = _build_models(config)

    if verbose:
        print(f"  Exhaustive: C({K},{N}) = {n_combos:,} combinations")

    best_cov = -np.inf
    best_combo = None

    for idx, combo in enumerate(itertools.combinations(range(K), N)):
        xy = candidates[list(combo), :2]
        cov = _coverage_for_users(xy, user_positions, config, dl_antenna, bh_antenna, channel)
        if cov > best_cov:
            best_cov = cov
            best_combo = combo
        if verbose and (idx + 1) % 10_000 == 0:
            pct = (idx + 1) / n_combos * 100
            print(f"    {idx+1:,}/{n_combos:,} ({pct:.1f}%), best so far: {best_cov:.3f}")

    if verbose:
        print(f"  Exhaustive done. Best coverage: {best_cov:.3f}")

    return _build_scenario_from_positions(
        candidates[list(best_combo), :2], user_positions, config
    )


def _comb(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    from math import comb as math_comb
    return math_comb(n, k)


# ── Repulsive Lloyd's heuristic ──────────────────────────────────────


def deploy_repulsive_lloyd_heuristic(
    config: Config,
    user_positions: np.ndarray,
    seed: int = 42,
    beta: float = 0.15,
    max_iters: int = 20,
    tol: float = 1.0,
) -> Scenario:
    """Lloyd's iteration with inter-drone repulsive forces.

    Problem addressed
    -----------------
    The analytic heuristic runs K-means **once** and stops.  K-means
    minimises total user-to-centroid distance — a reasonable coverage
    proxy — but completely ignores inter-drone interference.  In uniform
    or weakly-clustered distributions, a single K-means pass can produce
    Voronoi cells where neighbouring drones end up very close, causing
    high co-channel downlink interference.

    Algorithm
    ---------
    A modified **Lloyd's algorithm** (the iterative procedure K-means
    approximates).  Standard Lloyd's repeats:
      (1) assign each user to the nearest drone (Voronoi partition);
      (2) move each drone to its cell centroid.
    This converges to a centroidal Voronoi tessellation (CVT), which is
    the minimum-energy configuration for user-to-drone assignment.

    The modification adds a **repulsive force** between drones, inspired
    by electrostatics.  Each iteration computes two vectors per drone:

    Attraction — the centroid of the drone's assigned user cluster
    (exactly what standard Lloyd's computes).

    Repulsion — for each drone j, the sum of inverse-square
    displacement vectors from every other drone::

        F_repulse(j) = sum_{i != j}  (p_j - p_i) / ||p_j - p_i||^2

    The 1/d^2 form mirrors how received power (and therefore
    interference) decays with distance in free-space propagation.
    The minimum pairwise distance is clamped to 10 m to avoid blow-up.

    Scale normalisation — the raw repulsive force has arbitrary magnitude
    relative to the attraction displacement.  The code normalises the
    repulsion vector so its average magnitude matches the average
    attraction displacement.  This makes ``beta`` directly
    interpretable: beta=0.15 means "15 % repulsion, 85 % attraction".

    Position update::

        p_j(t+1) = (1 - beta) * centroid_j + beta * (p_j(t) + F_repulse(j))

    When beta=0 this is pure Lloyd's.  When beta=1 drones ignore users
    and only repel each other.

    Convergence — iteration stops when the maximum drone displacement
    falls below ``tol`` metres, or after ``max_iters`` iterations.

    After position convergence, altitude and orientations are set
    analytically (cluster-spread altitude, centroid-pointing DL, MST
    backhaul) — identical to the pure analytic heuristic.

    Why it should work
    ------------------
    * Clustered distributions: K-means already converges well in one
      pass (clusters are well-separated), so repulsive Lloyd's will
      match or marginally improve over the analytic heuristic.
    * Uniform distributions: K-means produces irregular Voronoi cells
      with some neighbouring drones very close.  Repulsion pushes these
      apart, yielding a more regular tessellation with better spatial
      reuse.
    * Hotspot distributions with unequal cluster sizes: some K-means
      centroids can collapse toward the same dense cluster.  Repulsion
      prevents this collapse while Lloyd's iterations rebalance the
      assignment.

    Complexity
    ----------
    O(T × (M·N + N²)) per call.  With M=300 users, N=15 drones,
    T=20 iterations this is ~100 k scalar operations — sub-millisecond.

    Parameters
    ----------
    config : Config
    user_positions : (M, 3)
    seed : int
        K-means RNG seed for initialisation.
    beta : float
        Repulsion strength in [0, 1].  0 = pure Lloyd's (no repulsion),
        1 = pure repulsion (ignores users).  Default 0.15.
    max_iters : int
        Maximum number of Lloyd iterations.
    tol : float
        Convergence threshold: stop when max drone displacement < tol [m].

    Returns
    -------
    Scenario — fully configured with analytic orientations.
    """
    n = config.network.n_drones
    area = config.scenario.area_size_m
    xy_users = user_positions[:, :2]

    # 1. Initialise via K-means
    centroids, _ = kmeans2(xy_users, n, minit="points", seed=seed)
    drone_xy = centroids.copy()  # (N, 2)

    # 2. Iterative refinement
    for _ in range(max_iters):
        # a. Assign users to nearest drone (2D)
        diff = xy_users[:, np.newaxis, :] - drone_xy[np.newaxis, :, :]  # (M, N, 2)
        dists = np.linalg.norm(diff, axis=-1)  # (M, N)
        assignment = np.argmin(dists, axis=1)  # (M,)

        # b. Attraction: cluster centroids
        attract = np.zeros_like(drone_xy)
        for j in range(n):
            mask = assignment == j
            if mask.sum() > 0:
                attract[j] = xy_users[mask].mean(axis=0)
            else:
                attract[j] = drone_xy[j]

        # c. Repulsion: pairwise 1/d² displacement
        repulse = np.zeros_like(drone_xy)
        for j in range(n):
            delta = drone_xy[j] - drone_xy  # (N, 2), vector from others to j
            d = np.linalg.norm(delta, axis=1, keepdims=True)  # (N, 1)
            d = np.maximum(d, 10.0)  # avoid division by zero
            # Exclude self (d[j]=0 → clamped to 10, but delta[j]=0 anyway)
            force = delta / (d ** 2)  # (N, 2), inverse-square repulsion
            repulse[j] = force.sum(axis=0)

        # Normalise repulsion to have similar scale as attraction displacement
        repulse_norm = np.linalg.norm(repulse, axis=1, keepdims=True)
        attract_scale = np.linalg.norm(attract - drone_xy, axis=1).mean()
        repulse_scale = repulse_norm.mean()
        if repulse_scale > 1e-6:
            repulse = repulse * (attract_scale / repulse_scale)

        # d. Combined displacement
        new_xy = (1.0 - beta) * attract + beta * (drone_xy + repulse)

        # e. Clamp to area
        margin = 10.0
        new_xy = np.clip(new_xy, margin, area - margin)

        # Check convergence
        max_shift = np.max(np.linalg.norm(new_xy - drone_xy, axis=1))
        drone_xy = new_xy
        if max_shift < tol:
            break

    # 3. Build scenario with analytic orientations
    drone_pos = np.zeros((n, 3))
    drone_pos[:, :2] = drone_xy
    drone_pos[:, 2] = config.network.altitude_m

    tmp = Scenario(
        user_positions=user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=np.zeros(n),
        dl_azimuth_rad=np.zeros(n),
        bh_tilt_rad=np.full(n, np.pi / 2),
        bh_azimuth_rad=np.zeros(n),
        area_size_m=area,
    )
    return apply_analytic_orientations(tmp, config)


# ── Altitude-staggered heuristic ─────────────────────────────────────


def deploy_altitude_staggered_heuristic(
    config: Config,
    user_positions: np.ndarray,
    seed: int = 42,
    n_tiers: int = 3,
) -> Scenario:
    """Analytic placement with altitude staggering via graph colouring.

    Problem addressed
    -----------------
    All existing heuristics compute each drone's altitude independently
    from a local formula: z = cluster_spread / tan(beamwidth/2).  Drones
    serving clusters of similar spread end up at nearly the **same
    altitude**.  When two such drones are horizontally close, their DL
    beams create maximum mutual interference because the interfering
    signal arrives at victim users within the main lobe elevation range.

    Key physical insight: if two neighbouring drones are at **different
    altitudes**, the path from drone A to drone B's users arrives at a
    steeper or shallower elevation angle than drone B's own beam.  With
    a 60° beamwidth antenna (3GPP model), even a 20–30° elevation angle
    difference pushes the interfering signal into the sidelobe region,
    where gain is 20 dB below peak.  This is the **vertical-dimension
    analogue of frequency reuse**: instead of assigning different
    frequencies to adjacent cells, assign different altitudes.

    Algorithm
    ---------
    Step 1 — Analytic baseline: run the full analytic heuristic to get
    positions, altitude, and orientations.

    Step 2 — Build interference proximity graph: two drones are
    "adjacent" (edge exists) if their beam footprints overlap on the
    ground.  The footprint radius of drone j at altitude z_j with
    beamwidth θ_3dB is::

        r_j = z_j · tan(θ_3dB / 2)

    An edge exists between i and j when their horizontal distance is
    less than r_i + r_j (footprints overlap).

    Step 3 — Greedy graph colouring with K colours (K = ``n_tiers``,
    default 3), using a largest-degree-first ordering.  Each node
    receives the smallest colour not used by any already-coloured
    neighbour, capped at ``n_tiers - 1``.

    Step 4 — Map colours to altitude tier bands within [50 m, 300 m]::

        n_tiers=3 → tier 0: [50, 133] m
                     tier 1: [133, 217] m
                     tier 2: [217, 300] m

    Step 5 — Per-drone altitude: reapply the cluster-spread formula
    z = spread / tan(bw/2) but **clamped to the assigned tier band**.
    This preserves beam-footprint matching while enforcing vertical
    separation between neighbours.

    Step 6 — Recompute orientations: with new altitudes the DL tilt
    angles change (tilt = arctan(horiz_dist / z)), so the full analytic
    orientation pipeline is rerun.

    Why it should work
    ------------------
    Consider two adjacent drones A (tier 0, z=80 m) and B (tier 2,
    z=250 m) serving clusters 400 m apart.  Drone A's beam points at
    its own users.  From drone A's antenna to drone B's users the
    elevation angle is arctan((250-0)/400) ≈ 32°, well outside A's
    main beam tilt — putting the interference power into the sidelobe
    region (−20 dB).  Without staggering, both at z=150 m, the
    elevation to each other's users is arctan(150/400) ≈ 21° and both
    beams are tilted similarly, keeping the interfering path closer to
    the main lobe.  The benefit grows with drone density (N=20–30)
    where footprint overlap is most severe.

    Complexity
    ----------
    O(N²) for graph construction and colouring, plus one analytic
    heuristic call.  Total: sub-millisecond.

    Parameters
    ----------
    config : Config
    user_positions : (M, 3)
    seed : int
    n_tiers : int
        Number of altitude tiers (2 or 3 recommended).  2 tiers give
        maximum vertical separation but fewer options; 3 tiers are
        generally sufficient since most interference graphs are
        3-colourable for typical drone layouts.

    Returns
    -------
    Scenario
    """
    # 1. Analytic baseline
    scenario = deploy_analytic_heuristic(config, user_positions, seed=seed)
    n = scenario.drone_positions.shape[0]

    if n <= 1:
        return scenario

    # 2. Build proximity graph: edge if drones' beam footprints overlap
    drone_pos = scenario.drone_positions.copy()
    xy = drone_pos[:, :2]
    dist_matrix = np.linalg.norm(
        xy[:, np.newaxis, :] - xy[np.newaxis, :, :], axis=-1
    )
    # Overlap threshold: sum of beam footprint radii (z * tan(bw/2))
    dl_bw_rad = np.deg2rad(config.antenna.dl_beamwidth_deg / 2.0)
    footprint_radii = drone_pos[:, 2] * np.tan(dl_bw_rad)
    overlap_threshold = (
        footprint_radii[:, np.newaxis] + footprint_radii[np.newaxis, :]
    )
    # Adjacency: overlapping footprints (exclude diagonal)
    adj = (dist_matrix < overlap_threshold) & (dist_matrix > 0)

    # 3. Greedy graph colouring (largest-degree-first)
    degrees = adj.sum(axis=1)
    order = np.argsort(-degrees)  # colour high-degree nodes first
    colours = np.full(n, -1, dtype=int)
    for node in order:
        neighbour_colours = set(colours[adj[node]] [colours[adj[node]] >= 0])
        # Pick smallest colour not used by neighbours
        c = 0
        while c in neighbour_colours:
            c += 1
        colours[node] = min(c, n_tiers - 1)  # cap at n_tiers

    # 4. Map colours to altitude tier bands
    # Define tier bands within [50, 300] m
    z_min, z_max = 50.0, 300.0
    tier_width = (z_max - z_min) / n_tiers
    tier_centres = [z_min + (t + 0.5) * tier_width for t in range(n_tiers)]

    # 5. Per-drone altitude: cluster-spread formula clamped to tier band
    stats = compute_cluster_stats(drone_pos, user_positions)
    dl_bw_deg = config.antenna.dl_beamwidth_deg
    for j in range(n):
        tier = colours[j]
        tier_lo = z_min + tier * tier_width
        tier_hi = tier_lo + tier_width
        z_analytic = altitude_from_cluster_spread(
            stats[j].spread_m, dl_bw_deg, z_min=tier_lo, z_max=tier_hi
        )
        drone_pos[j, 2] = z_analytic

    # 6. Recompute orientations with new altitudes
    tmp = Scenario(
        user_positions=user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=np.zeros(n),
        dl_azimuth_rad=np.zeros(n),
        bh_tilt_rad=np.full(n, np.pi / 2),
        bh_azimuth_rad=np.zeros(n),
        area_size_m=config.scenario.area_size_m,
    )
    return apply_analytic_orientations(tmp, config)


# ── Interference-aware greedy heuristic ──────────────────────────────


def _objective_for_users(
    drone_xy: np.ndarray,
    user_positions: np.ndarray,
    config: Config,
    dl_antenna,
    bh_antenna,
    channel,
    zeta: float = 0.7,
) -> float:
    """Combined score (coverage + interference) for a specific user layout.

    Builds a full scenario with analytic orientations, runs the complete
    SINR + backhaul-interference pipeline, and returns a scalar score::

        score = zeta * coverage_fraction
              + (1 - zeta) * clip((-30 - interf_dBm) / 40, 0, 1)

    Used as the per-candidate selection criterion in
    ``deploy_interference_aware_greedy``.
    """
    scenario = _build_scenario_from_positions(drone_xy, user_positions, config)
    metrics = _evaluate_single(scenario, config, dl_antenna, bh_antenna, channel)

    cov_01 = metrics.coverage_fraction
    interf_dbm = metrics.total_inter_drone_interference_dbm
    # Normalise interference to [0, 1] (higher = better = less interference)
    interf_best = -70.0
    interf_worst = -30.0
    interf_01 = float(np.clip(
        (interf_worst - interf_dbm) / (interf_worst - interf_best), 0.0, 1.0
    ))
    return zeta * cov_01 + (1.0 - zeta) * interf_01


def deploy_interference_aware_greedy(
    config: Config,
    candidates: np.ndarray,
    user_positions: np.ndarray,
    zeta: float = 0.7,
    verbose: bool = False,
) -> Scenario:
    """Greedy sequential placement maximising combined coverage+interference score.

    Problem addressed
    -----------------
    The existing ``greedy_discrete_placement`` places drones one at a
    time, each time picking the grid candidate that **maximises
    coverage**.  This works well for coverage but is blind to
    interference: drones cluster around dense user groups, multiple
    drones end up close together, and inter-drone interference rises by
    +10 dB compared to the analytic heuristic.

    Root cause: coverage is monotonically non-decreasing when adding
    drones (an extra drone never reduces a user's best serving SINR),
    but interference is also monotonically non-decreasing.  A
    coverage-only criterion has no mechanism to penalise the
    interference cost of placing a drone near existing ones.

    Algorithm
    ---------
    Structurally identical to the existing greedy — sequential placement
    from a K×K candidate grid — but the **selection criterion** changes.

    Each candidate is scored by::

        score = ζ · C + (1 − ζ) · I_norm

    where:
      C      = coverage_fraction (users with SINR ≥ 3 dB), in [0, 1]
      I_norm = normalised interference quality:
               clip( (-30 − I_dBm) / (-30 − (−70)), 0, 1 )
               Maps total inter-drone interference from [−70, −30] dBm
               to [1, 0]: lower interference → higher score.
      ζ      = coverage weight (default 0.7).
               ζ=1.0 reduces to the original greedy.
               ζ=0.0 purely minimises interference.

    For each candidate position the code builds a complete scenario
    (analytic orientations, altitude, MST backhaul), runs the full
    SINR + interference pipeline, and computes metrics.  This is what
    makes each evaluation expensive.

    Why it should work
    ------------------
    The interference penalty discourages placing each subsequent drone
    near already-placed ones.  Instead, the greedy spreads drones more
    widely across the area, accepting slightly lower coverage for
    dramatically reduced mutual interference.  The ``zeta`` parameter
    directly controls where you land on the Pareto frontier.

    Complexity
    ----------
    O(N × K) full pipeline evaluations.  With N=15 drones and K=1600
    candidates (40×40 grid): 24 000 evaluations, each running the full
    antenna + channel + SINR + interference pipeline.  Roughly 1–5
    minutes per scenario depending on user count — significantly slower
    than the sub-millisecond analytic methods.

    Parameters
    ----------
    config : Config
    candidates : (K, 3)
        Candidate positions (z overridden by analytic formula).
    user_positions : (M, 3)
        Known user positions.
    zeta : float
        Coverage weight [0, 1].  Default 0.7.  Shared with the
        experiment framework's combined score.
    verbose : bool

    Returns
    -------
    Scenario
    """
    N = config.network.n_drones
    dl_antenna, bh_antenna, channel = _build_models(config)

    chosen_xy = np.empty((N, 2))
    available = list(range(candidates.shape[0]))

    for drone_i in range(N):
        best_score = -np.inf
        best_k = available[0]
        for k in available:
            trial_xy = np.vstack([chosen_xy[:drone_i], candidates[k, :2].reshape(1, 2)])
            score = _objective_for_users(
                trial_xy, user_positions, config,
                dl_antenna, bh_antenna, channel, zeta=zeta,
            )
            if score > best_score:
                best_score = score
                best_k = k
        chosen_xy[drone_i] = candidates[best_k, :2]
        available.remove(best_k)
        if verbose:
            print(
                f"  IA-Greedy drone {drone_i+1}/{N}: "
                f"pos={candidates[best_k,:2]}, score={best_score:.3f}"
            )

    return _build_scenario_from_positions(chosen_xy, user_positions, config)
