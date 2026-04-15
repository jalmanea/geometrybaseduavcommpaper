"""Coverage, throughput, and SINR metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sinr import SINRResult


@dataclass
class SimulationMetrics:
    """Aggregate metrics from a single trial."""

    # System-level
    sum_throughput_mbps: float  # Total throughput [Mbps]
    mean_sinr_db: float
    median_sinr_db: float
    min_sinr_db: float
    sinr_5th_percentile_db: float
    coverage_fraction: float  # Fraction of users above SINR threshold

    # Interference
    total_inter_drone_interference_dbm: float
    worst_pair_interference_dbm: float

    # Counts
    n_users: int
    n_drones: int


def compute_metrics(
    sinr_result: SINRResult,
    sinr_threshold_db: float = 3.0,
    backhaul_interference_matrix: np.ndarray | None = None,
) -> SimulationMetrics:
    """Compute all metrics from SINR results.

    Parameters
    ----------
    sinr_result : SINRResult
    sinr_threshold_db : float
        SINR threshold for coverage [dB].
    backhaul_interference_matrix : np.ndarray, shape (N, N) or None
        Inter-drone backhaul interference. If None, interference metrics are set to -inf.

    Returns
    -------
    SimulationMetrics
    """
    sinr_db = sinr_result.sinr_db

    # Coverage (fraction of users above threshold)
    coverage = float(np.mean(sinr_db >= sinr_threshold_db))

    # Throughput
    sum_throughput_mbps = float(np.sum(sinr_result.throughput_bps) / 1e6)

    # Inter-drone interference
    if backhaul_interference_matrix is not None:
        # Upper triangle (avoid double counting)
        N = backhaul_interference_matrix.shape[0]
        triu_idx = np.triu_indices(N, k=1)
        pair_interference = backhaul_interference_matrix[triu_idx]
        total_interference_w = float(np.sum(pair_interference))
        worst_pair_w = float(np.max(pair_interference)) if len(pair_interference) > 0 else 0.0
        total_idb = 10.0 * np.log10(max(total_interference_w, 1e-30)) + 30.0
        worst_idb = 10.0 * np.log10(max(worst_pair_w, 1e-30)) + 30.0
        n_drones = N
    else:
        total_idb = float("-inf")
        worst_idb = float("-inf")
        n_drones = 0

    return SimulationMetrics(
        sum_throughput_mbps=sum_throughput_mbps,
        mean_sinr_db=float(np.mean(sinr_db)),
        median_sinr_db=float(np.median(sinr_db)),
        min_sinr_db=float(np.min(sinr_db)),
        sinr_5th_percentile_db=float(np.percentile(sinr_db, 5)),
        coverage_fraction=coverage,
        total_inter_drone_interference_dbm=total_idb,
        worst_pair_interference_dbm=worst_idb,
        n_users=len(sinr_db),
        n_drones=n_drones,
    )
