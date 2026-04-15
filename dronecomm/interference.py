"""Interference computation between drones and ground users.

Computes the received-power matrix P[j, u] = power at user u from drone j,
combining antenna directional gain and channel path loss.

Two interference types:
- Downlink (drone -> ground user): air-to-ground channel with LoS/NLoS.
- Backhaul (drone -> drone): air-to-air free-space channel.
Cross-interference between DL and BH is assumed zero (frequency separation).
"""

from __future__ import annotations

import numpy as np

from .antenna import AntennaModel
from .channel import ChannelModel
from .geometry import (
    boresight_vector,
    pairwise_directions_and_distances,
    pairwise_elevation_angles,
    pairwise_off_boresight_angles,
)


def downlink_power_matrix(
    drone_positions: np.ndarray,
    dl_tilt_rad: np.ndarray,
    dl_azimuth_rad: np.ndarray,
    user_positions: np.ndarray,
    dl_antenna: AntennaModel,
    channel: ChannelModel,
    p_tx_dl_w: float,
) -> np.ndarray:
    """Received power at each ground user from each drone (downlink).

    Parameters
    ----------
    drone_positions : np.ndarray, shape (N, 3)
        3D positions of N drones.
    dl_tilt_rad : np.ndarray, shape (N,)
        Downlink antenna tilt per drone (radians from nadir).
    dl_azimuth_rad : np.ndarray, shape (N,)
        Downlink antenna azimuth per drone (radians).
    user_positions : np.ndarray, shape (M, 3)
        3D positions of M ground users (z typically 0).
    dl_antenna : AntennaModel
        Downlink antenna radiation model.
    channel : ChannelModel
        Air-to-ground channel model.
    p_tx_dl_w : float
        Downlink transmit power in Watts.

    Returns
    -------
    np.ndarray, shape (N, M)
        power_matrix[j, u] = received power at user u from drone j [Watts].
    """
    N = drone_positions.shape[0]
    M = user_positions.shape[0]

    # Boresight vectors: (N, 3)
    boresights = boresight_vector(dl_tilt_rad, dl_azimuth_rad)

    # Directions and distances from each drone to each user: (N, M, 3) and (N, M)
    directions, distances = pairwise_directions_and_distances(
        drone_positions, user_positions
    )

    # Off-boresight angles: (N, M)
    theta = pairwise_off_boresight_angles(boresights, directions)

    # Antenna gain (linear): (N, M)
    gain = dl_antenna.gain_linear(theta)

    # Elevation angles: (N, M)
    elev = pairwise_elevation_angles(drone_positions, user_positions)

    # Path loss (linear attenuation): (N, M)
    pl = channel.path_loss_linear(distances, elev)

    return p_tx_dl_w * gain * pl


def backhaul_interference_matrix(
    drone_positions: np.ndarray,
    bh_tilt_rad: np.ndarray,
    bh_azimuth_rad: np.ndarray,
    bh_antenna: AntennaModel,
    channel: ChannelModel,
    p_tx_bh_w: float,
) -> np.ndarray:
    """Interference power between each pair of drones on the backhaul link.

    Parameters
    ----------
    drone_positions : np.ndarray, shape (N, 3)
    bh_tilt_rad : np.ndarray, shape (N,)
        Backhaul antenna tilt per drone.
    bh_azimuth_rad : np.ndarray, shape (N,)
        Backhaul antenna azimuth per drone.
    bh_antenna : AntennaModel
        Backhaul antenna radiation model.
    channel : ChannelModel
        Channel model (only a2a path loss is used).
    p_tx_bh_w : float
        Backhaul transmit power in Watts.

    Returns
    -------
    np.ndarray, shape (N, N)
        interference[j, i] = interference power drone j creates at drone i [Watts].
        Diagonal is zero.
    """
    N = drone_positions.shape[0]

    # Boresight vectors for backhaul antennas: (N, 3)
    boresights = boresight_vector(bh_tilt_rad, bh_azimuth_rad)

    # Directions and distances between all drone pairs: (N, N, 3), (N, N)
    directions, distances = pairwise_directions_and_distances(
        drone_positions, drone_positions
    )

    # Tx gain: gain of drone j's BH antenna in direction of drone i
    # Off-boresight from j toward i: (N, N)
    theta_tx = pairwise_off_boresight_angles(boresights, directions)
    gain_tx = bh_antenna.gain_linear(theta_tx)

    # Rx gain: gain of drone i's BH antenna in direction of drone j
    # directions_rev[j, i] = direction from drone i toward drone j (arrival at j from i)
    # pairwise_off_boresight_angles indexes boresight by the first axis, so
    # theta_rx[j, i] = angle at drone j for arrival from drone i.
    # For interference[j, i] we need the rx gain at drone *i*, i.e. theta_rx[i, j].
    directions_rev = -directions
    theta_rx = pairwise_off_boresight_angles(boresights, directions_rev)
    gain_rx = bh_antenna.gain_linear(theta_rx)

    # Air-to-air path loss (free-space): (N, N)
    pl = channel.a2a_path_loss_linear(distances)

    # interference[j, i] = P_tx * G_tx(j→i) * G_rx(i←j) * PL(j,i)
    # gain_rx must be transposed: gain_rx[i, j] is the rx gain at drone i from drone j
    interference = p_tx_bh_w * gain_tx * gain_rx.T * pl

    # Zero out diagonal (no self-interference)
    np.fill_diagonal(interference, 0.0)

    return interference
