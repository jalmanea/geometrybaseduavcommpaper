"""3D vector math, angle computations, and coordinate transforms.

All functions are vectorized with NumPy for efficient batch computation.

Coordinate system:
- Right-handed Cartesian (x, y, z), z pointing up.
- Ground plane at z = 0.
- Antenna orientation defined by tilt (from nadir) and azimuth (from +x axis, CCW).
"""

from __future__ import annotations

import numpy as np


def boresight_vector(tilt_rad: np.ndarray, azimuth_rad: np.ndarray) -> np.ndarray:
    """Compute antenna boresight unit vector from tilt and azimuth angles.

    Parameters
    ----------
    tilt_rad : array-like, shape (...,)
        Tilt angle from nadir in radians (0 = straight down, pi/2 = horizontal).
    azimuth_rad : array-like, shape (...,)
        Azimuth angle in radians from +x axis, measured CCW in x-y plane.

    Returns
    -------
    np.ndarray, shape (..., 3)
        Unit vector(s) pointing in the boresight direction.
    """
    tilt_rad = np.asarray(tilt_rad)
    azimuth_rad = np.asarray(azimuth_rad)
    return np.stack(
        [
            np.sin(tilt_rad) * np.cos(azimuth_rad),
            np.sin(tilt_rad) * np.sin(azimuth_rad),
            -np.cos(tilt_rad),
        ],
        axis=-1,
    )


def off_boresight_angle(
    boresight: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Angle between antenna boresight and a direction vector.

    Parameters
    ----------
    boresight : np.ndarray, shape (..., 3)
        Antenna boresight unit vector(s).
    direction : np.ndarray, shape (..., 3)
        Direction unit vector(s) from antenna to target.

    Returns
    -------
    np.ndarray, shape (...)
        Off-boresight angle(s) in radians, range [0, pi].
    """
    cos_theta = np.sum(boresight * direction, axis=-1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def direction_and_distance(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Direction unit vector and Euclidean distance from source to target.

    Parameters
    ----------
    source : np.ndarray, shape (..., 3)
        Source position(s).
    target : np.ndarray, shape (..., 3)
        Target position(s).

    Returns
    -------
    direction : np.ndarray, shape (..., 3)
        Unit vector(s) from source to target.
    distance : np.ndarray, shape (...)
        3D Euclidean distance(s). Clamped to min 1e-6 m to avoid division by zero.
    """
    diff = target - source
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    dist = np.maximum(dist, 1e-6)
    direction = diff / dist
    return direction, dist[..., 0]


def elevation_angle(drone_pos: np.ndarray, ground_pos: np.ndarray) -> np.ndarray:
    """Elevation angle from a ground point looking up toward a drone.

    Parameters
    ----------
    drone_pos : np.ndarray, shape (..., 3)
        Drone position(s) (z > 0).
    ground_pos : np.ndarray, shape (..., 3)
        Ground point(s) (typically z = 0).

    Returns
    -------
    np.ndarray, shape (...)
        Elevation angle(s) in radians, range [0, pi/2].
    """
    dz = np.abs(drone_pos[..., 2] - ground_pos[..., 2])
    horiz = np.sqrt(
        (drone_pos[..., 0] - ground_pos[..., 0]) ** 2
        + (drone_pos[..., 1] - ground_pos[..., 1]) ** 2
    )
    return np.arctan2(dz, np.maximum(horiz, 1e-6))


def pairwise_directions_and_distances(
    sources: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute direction vectors and distances for all (source, target) pairs.

    Parameters
    ----------
    sources : np.ndarray, shape (N, 3)
    targets : np.ndarray, shape (M, 3)

    Returns
    -------
    directions : np.ndarray, shape (N, M, 3)
        Unit vector from each source to each target.
    distances : np.ndarray, shape (N, M)
        Euclidean distance for each pair.
    """
    # (N, 1, 3) - (1, M, 3) -> (N, M, 3)
    diff = sources[:, np.newaxis, :] - targets[np.newaxis, :, :]
    # Note: direction is source -> target, so negate diff
    diff = -diff
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)  # (N, M, 1)
    dist = np.maximum(dist, 1e-6)
    directions = diff / dist
    return directions, dist[..., 0]


def pairwise_elevation_angles(
    drone_pos: np.ndarray, ground_pos: np.ndarray
) -> np.ndarray:
    """Elevation angles for all (drone, ground user) pairs.

    Parameters
    ----------
    drone_pos : np.ndarray, shape (N, 3)
    ground_pos : np.ndarray, shape (M, 3)

    Returns
    -------
    np.ndarray, shape (N, M)
        Elevation angle from each ground user to each drone, in radians.
    """
    dz = np.abs(
        drone_pos[:, np.newaxis, 2] - ground_pos[np.newaxis, :, 2]
    )  # (N, M)
    horiz = np.sqrt(
        (drone_pos[:, np.newaxis, 0] - ground_pos[np.newaxis, :, 0]) ** 2
        + (drone_pos[:, np.newaxis, 1] - ground_pos[np.newaxis, :, 1]) ** 2
    )  # (N, M)
    return np.arctan2(dz, np.maximum(horiz, 1e-6))


def pairwise_off_boresight_angles(
    boresights: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Off-boresight angles for all (drone, target) pairs.

    Parameters
    ----------
    boresights : np.ndarray, shape (N, 3)
        Antenna boresight unit vectors.
    directions : np.ndarray, shape (N, M, 3)
        Direction unit vectors from each drone to each target.

    Returns
    -------
    np.ndarray, shape (N, M)
        Off-boresight angle for each (drone, target) pair, in radians.
    """
    # (N, 1, 3) * (N, M, 3) -> sum over axis=-1 -> (N, M)
    cos_theta = np.sum(boresights[:, np.newaxis, :] * directions, axis=-1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)
