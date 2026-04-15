"""SINR and throughput computation for ground users.

Given a received-power matrix and user-drone association, computes:
- Per-user SINR (signal / (noise + interference))
- Per-user Shannon throughput
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dbm_to_watts(dbm: float) -> float:
    """Convert dBm to Watts."""
    return 10.0 ** ((dbm - 30.0) / 10.0)


def watts_to_dbm(watts: np.ndarray) -> np.ndarray:
    """Convert Watts to dBm."""
    return 10.0 * np.log10(np.maximum(watts, 1e-30)) + 30.0


def linear_to_db(x: np.ndarray) -> np.ndarray:
    """Convert linear ratio to dB."""
    return 10.0 * np.log10(np.maximum(x, 1e-30))


@dataclass
class SINRResult:
    """Per-user SINR computation result."""

    sinr_linear: np.ndarray  # (M,)
    sinr_db: np.ndarray  # (M,)
    signal_power: np.ndarray  # (M,) Watts
    interference: np.ndarray  # (M,) Watts
    noise_power: float  # Watts
    throughput_bps: np.ndarray  # (M,) bits/sec


def nearest_drone_association(
    drone_positions: np.ndarray, user_positions: np.ndarray
) -> np.ndarray:
    """Associate each user to the nearest drone (min 3D distance).

    Parameters
    ----------
    drone_positions : np.ndarray, shape (N, 3)
    user_positions : np.ndarray, shape (M, 3)

    Returns
    -------
    np.ndarray, shape (M,), dtype int
        Index of the serving drone for each user.
    """
    # (N, M) distance matrix
    diff = drone_positions[:, np.newaxis, :] - user_positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (N, M)
    return np.argmin(dist, axis=0)  # (M,)


def compute_sinr(
    power_matrix: np.ndarray,
    association: np.ndarray,
    noise_psd_dbm_hz: float = -174.0,
    bandwidth_hz: float = 10.0e6,
) -> SINRResult:
    """Compute SINR for all ground users.

    Parameters
    ----------
    power_matrix : np.ndarray, shape (N, M)
        power_matrix[j, u] = received power at user u from drone j [Watts].
    association : np.ndarray, shape (M,), dtype int
        Index of serving drone for each user.
    noise_psd_dbm_hz : float
        Noise power spectral density in dBm/Hz (default: -174 dBm/Hz at 290 K).
    bandwidth_hz : float
        Bandwidth per user in Hz.

    Returns
    -------
    SINRResult
    """
    N, M = power_matrix.shape
    user_indices = np.arange(M)

    # Signal power: power from the serving drone
    signal_power = power_matrix[association, user_indices]  # (M,)

    # Total received power from all drones
    total_power = np.sum(power_matrix, axis=0)  # (M,)

    # Interference = total - signal
    interference = total_power - signal_power  # (M,)

    # Thermal noise
    noise_psd_w = dbm_to_watts(noise_psd_dbm_hz)  # W/Hz
    noise_power = noise_psd_w * bandwidth_hz  # W

    # SINR
    sinr_linear = signal_power / (noise_power + interference)
    sinr_db = linear_to_db(sinr_linear)

    # Shannon throughput
    throughput = bandwidth_hz * np.log2(1.0 + sinr_linear)

    return SINRResult(
        sinr_linear=sinr_linear,
        sinr_db=sinr_db,
        signal_power=signal_power,
        interference=interference,
        noise_power=noise_power,
        throughput_bps=throughput,
    )
