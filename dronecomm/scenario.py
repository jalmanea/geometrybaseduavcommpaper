"""Disaster scenario generation: ground users and drone placement.

Users are generated as clustered distributions (Gaussian mixture) to model
shelters, collapsed buildings, and evacuation routes. Drones are placed
using K-means, grid, or random strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2


@dataclass
class Scenario:
    """A complete disaster scenario."""

    user_positions: np.ndarray  # (M, 3), z=0 for ground users
    drone_positions: np.ndarray  # (N, 3)
    dl_tilt_rad: np.ndarray  # (N,) downlink antenna tilt
    dl_azimuth_rad: np.ndarray  # (N,) downlink antenna azimuth
    bh_tilt_rad: np.ndarray  # (N,) backhaul antenna tilt
    bh_azimuth_rad: np.ndarray  # (N,) backhaul antenna azimuth
    area_size_m: float


def generate_users(
    n_clusters: int = 5,
    users_per_cluster_mean: int = 40,
    users_per_cluster_std: int = 10,
    cluster_spread_m: float = 100.0,
    area_size_m: float = 2000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate clustered ground users for a disaster scenario.

    Parameters
    ----------
    n_clusters : int
        Number of user clusters (demand hotspots).
    users_per_cluster_mean : int
        Mean number of users per cluster.
    users_per_cluster_std : int
        Std dev of users per cluster.
    cluster_spread_m : float
        Spatial spread (std dev) of users within each cluster [m].
    area_size_m : float
        Side length of the square disaster area [m].
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    positions : np.ndarray, shape (M, 3)
        Ground user positions (z = 0).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Cluster centers uniformly in the area
    centers = rng.uniform(0.0, area_size_m, size=(n_clusters, 2))

    all_xy = []
    for center in centers:
        n_users = max(1, int(rng.normal(users_per_cluster_mean, users_per_cluster_std)))
        xy = rng.normal(loc=center, scale=cluster_spread_m, size=(n_users, 2))
        all_xy.append(xy)

    xy = np.vstack(all_xy)

    # Clip to area bounds
    xy = np.clip(xy, 0.0, area_size_m)

    M = xy.shape[0]
    positions = np.zeros((M, 3))
    positions[:, :2] = xy

    return positions


def generate_users_uniform(
    n_users: int = 200,
    area_size_m: float = 2000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate uniformly distributed ground users.

    Parameters
    ----------
    n_users : int
        Number of users to generate.
    area_size_m : float
        Side length of the square area [m].
    rng : np.random.Generator or None

    Returns
    -------
    positions : np.ndarray, shape (n_users, 3)
        Ground user positions (z = 0).
    """
    if rng is None:
        rng = np.random.default_rng()

    xy = rng.uniform(0.0, area_size_m, size=(n_users, 2))
    positions = np.zeros((n_users, 3))
    positions[:, :2] = xy
    return positions


def generate_users_hotspot(
    n_clusters: int = 3,
    users_per_cluster_mean: int = 67,
    users_per_cluster_std: int = 10,
    cluster_spread_m: float = 50.0,
    area_size_m: float = 2000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate hotspot users: fewer, denser clusters than the default.

    Parameters
    ----------
    n_clusters : int
        Number of dense hotspots (default 3).
    users_per_cluster_mean : int
        Mean users per hotspot (default 67 for ~200 total with 3 clusters).
    users_per_cluster_std : int
    cluster_spread_m : float
        Tighter spread than default clustered (50 m vs 100 m).
    area_size_m : float
    rng : np.random.Generator or None

    Returns
    -------
    positions : np.ndarray, shape (M, 3)
        Ground user positions (z = 0).
    """
    return generate_users(
        n_clusters=n_clusters,
        users_per_cluster_mean=users_per_cluster_mean,
        users_per_cluster_std=users_per_cluster_std,
        cluster_spread_m=cluster_spread_m,
        area_size_m=area_size_m,
        rng=rng,
    )


def generate_users_telecom(
    npy_path: str,
    n_users: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Load user positions from a pre-extracted Telecom Italia .npy file.

    The file must be shape (M, 3) with z=0, as produced by
    ``scripts/extract_telecom_users.py``.

    Parameters
    ----------
    npy_path : str
        Path to the .npy file.
    n_users : int or None
        If given, sub-sample (without replacement) to exactly this many users.
        If larger than the file, users are sampled with replacement.
    rng : np.random.Generator or None

    Returns
    -------
    positions : np.ndarray, shape (n_users or M, 3)
    """
    positions = np.load(npy_path)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected shape (M, 3), got {positions.shape}")

    if n_users is not None and n_users != positions.shape[0]:
        if rng is None:
            rng = np.random.default_rng()
        replace = n_users > positions.shape[0]
        idx = rng.choice(positions.shape[0], size=n_users, replace=replace)
        positions = positions[idx]

    return positions


def deploy_drones_kmeans(
    user_positions: np.ndarray,
    n_drones: int = 15,
    altitude_m: float = 150.0,
    dl_tilt_deg: float = 0.0,
    bh_tilt_deg: float = 90.0,
    min_separation_m: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deploy drones above user cluster centroids using K-means.

    Parameters
    ----------
    user_positions : np.ndarray, shape (M, 3)
    n_drones : int
    altitude_m : float
    dl_tilt_deg : float
        Downlink antenna tilt from nadir [degrees]. 0 = straight down.
    bh_tilt_deg : float
        Backhaul antenna tilt from nadir [degrees]. 90 = horizontal.
    min_separation_m : float
        Minimum inter-drone distance (post-hoc enforcement).

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
    dl_tilt_rad : np.ndarray, shape (N,)
    dl_azimuth_rad : np.ndarray, shape (N,)
    bh_tilt_rad : np.ndarray, shape (N,)
    bh_azimuth_rad : np.ndarray, shape (N,)
    """
    xy = user_positions[:, :2]
    centroids, _ = kmeans2(xy, n_drones, minit="points", seed=42)  # (N, 2)

    positions = np.zeros((n_drones, 3))
    positions[:, :2] = centroids
    positions[:, 2] = altitude_m

    # Enforce minimum separation by nudging close drones
    _enforce_min_separation(positions, min_separation_m)

    dl_tilt_rad = np.full(n_drones, np.deg2rad(dl_tilt_deg))
    dl_azimuth_rad = np.zeros(n_drones)
    bh_tilt_rad = np.full(n_drones, np.deg2rad(bh_tilt_deg))
    bh_azimuth_rad = np.zeros(n_drones)

    return positions, dl_tilt_rad, dl_azimuth_rad, bh_tilt_rad, bh_azimuth_rad


def deploy_drones_grid(
    n_drones: int = 16,
    area_size_m: float = 2000.0,
    altitude_m: float = 150.0,
    dl_tilt_deg: float = 0.0,
    bh_tilt_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deploy drones on a regular grid.

    Parameters
    ----------
    n_drones : int
        Approximate number of drones. Rounded to nearest perfect square.
    area_size_m : float
    altitude_m : float
    dl_tilt_deg : float
    bh_tilt_deg : float

    Returns
    -------
    Same as deploy_drones_kmeans.
    """
    side = int(np.ceil(np.sqrt(n_drones)))
    margin = area_size_m / (2 * side)
    xs = np.linspace(margin, area_size_m - margin, side)
    ys = np.linspace(margin, area_size_m - margin, side)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])[:n_drones]

    actual_n = xy.shape[0]
    positions = np.zeros((actual_n, 3))
    positions[:, :2] = xy
    positions[:, 2] = altitude_m

    dl_tilt_rad = np.full(actual_n, np.deg2rad(dl_tilt_deg))
    dl_azimuth_rad = np.zeros(actual_n)
    bh_tilt_rad = np.full(actual_n, np.deg2rad(bh_tilt_deg))
    bh_azimuth_rad = np.zeros(actual_n)

    return positions, dl_tilt_rad, dl_azimuth_rad, bh_tilt_rad, bh_azimuth_rad


def deploy_drones_random(
    n_drones: int = 15,
    area_size_m: float = 2000.0,
    altitude_m: float = 150.0,
    dl_tilt_deg: float = 0.0,
    bh_tilt_deg: float = 90.0,
    min_separation_m: float = 100.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deploy drones uniformly at random within the area.

    Parameters
    ----------
    Same as above.

    Returns
    -------
    Same as deploy_drones_kmeans.
    """
    if rng is None:
        rng = np.random.default_rng()

    xy = rng.uniform(0.0, area_size_m, size=(n_drones, 2))
    positions = np.zeros((n_drones, 3))
    positions[:, :2] = xy
    positions[:, 2] = altitude_m

    _enforce_min_separation(positions, min_separation_m)

    dl_tilt_rad = np.full(n_drones, np.deg2rad(dl_tilt_deg))
    dl_azimuth_rad = np.zeros(n_drones)
    bh_tilt_rad = np.full(n_drones, np.deg2rad(bh_tilt_deg))
    bh_azimuth_rad = np.zeros(n_drones)

    return positions, dl_tilt_rad, dl_azimuth_rad, bh_tilt_rad, bh_azimuth_rad


def _enforce_min_separation(positions: np.ndarray, min_dist: float) -> None:
    """Nudge drones apart until all pairs satisfy minimum separation.

    Modifies positions in-place. Uses iterative repulsion (simple approach).
    """
    max_iter = 100
    for _ in range(max_iter):
        N = positions.shape[0]
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist and dist > 1e-6:
                    # Push apart equally
                    direction = diff / dist
                    push = (min_dist - dist) / 2.0 + 1.0
                    positions[i] += direction * push
                    positions[j] -= direction * push
                    changed = True
        if not changed:
            break


def create_scenario(
    n_drones: int = 15,
    placement: str = "kmeans",
    altitude_m: float = 150.0,
    dl_tilt_deg: float = 0.0,
    bh_tilt_deg: float = 90.0,
    n_clusters: int = 5,
    users_per_cluster_mean: int = 40,
    cluster_spread_m: float = 100.0,
    area_size_m: float = 2000.0,
    min_separation_m: float = 100.0,
    seed: int | None = None,
) -> Scenario:
    """Create a complete disaster scenario.

    Parameters
    ----------
    placement : str
        Drone placement strategy: "kmeans", "grid", or "random".
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    Scenario
    """
    rng = np.random.default_rng(seed)

    user_positions = generate_users(
        n_clusters=n_clusters,
        users_per_cluster_mean=users_per_cluster_mean,
        cluster_spread_m=cluster_spread_m,
        area_size_m=area_size_m,
        rng=rng,
    )

    if placement == "kmeans":
        drone_pos, dl_t, dl_a, bh_t, bh_a = deploy_drones_kmeans(
            user_positions, n_drones, altitude_m, dl_tilt_deg, bh_tilt_deg,
            min_separation_m,
        )
    elif placement == "grid":
        drone_pos, dl_t, dl_a, bh_t, bh_a = deploy_drones_grid(
            n_drones, area_size_m, altitude_m, dl_tilt_deg, bh_tilt_deg,
        )
    elif placement == "random":
        drone_pos, dl_t, dl_a, bh_t, bh_a = deploy_drones_random(
            n_drones, area_size_m, altitude_m, dl_tilt_deg, bh_tilt_deg,
            min_separation_m, rng,
        )
    else:
        raise ValueError(f"Unknown placement strategy: {placement}")

    return Scenario(
        user_positions=user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=dl_t,
        dl_azimuth_rad=dl_a,
        bh_tilt_rad=bh_t,
        bh_azimuth_rad=bh_a,
        area_size_m=area_size_m,
    )
