"""Tests for SINR module."""

import numpy as np
import pytest

from dronecomm.sinr import (
    compute_sinr,
    dbm_to_watts,
    nearest_drone_association,
    watts_to_dbm,
)


def test_dbm_to_watts():
    """30 dBm = 1 W, 0 dBm = 1 mW."""
    assert dbm_to_watts(30.0) == pytest.approx(1.0, rel=1e-6)
    assert dbm_to_watts(0.0) == pytest.approx(1e-3, rel=1e-6)


def test_watts_to_dbm():
    assert watts_to_dbm(np.array(1.0)) == pytest.approx(30.0, rel=1e-6)


def test_nearest_association():
    """Each user should be assigned to the closest drone."""
    drones = np.array([[0, 0, 100], [1000, 0, 100]], dtype=float)
    users = np.array([[100, 0, 0], [900, 0, 0], [500, 0, 0]], dtype=float)
    assoc = nearest_drone_association(drones, users)
    np.testing.assert_array_equal(assoc, [0, 1, 0])  # user at 500 is closer to drone 0


def test_sinr_no_interference():
    """With a single drone, SINR = signal / noise (no interference)."""
    power_matrix = np.array([[1e-8]])  # 1 drone, 1 user
    assoc = np.array([0])
    result = compute_sinr(power_matrix, assoc, noise_psd_dbm_hz=-174.0, bandwidth_hz=1e6)
    # Noise = 10^((-174-30)/10) * 1e6 = 10^(-20.4) * 1e6 ≈ 3.98e-15
    # SINR = 1e-8 / 3.98e-15 ≈ 2.51e6 -> ~64 dB
    assert result.sinr_db[0] > 50  # Should be very high with no interference
    assert result.interference[0] == pytest.approx(0.0, abs=1e-20)


def test_sinr_with_interference():
    """With equal signal and interference, SINR should be around 0 dB."""
    power_matrix = np.array([
        [1e-9, 1e-9],  # Drone 0: equal power to both users
        [1e-9, 1e-9],  # Drone 1: equal power to both users
    ])
    assoc = np.array([0, 1])  # User 0 -> drone 0, user 1 -> drone 1
    result = compute_sinr(power_matrix, assoc, noise_psd_dbm_hz=-174.0, bandwidth_hz=1e6)
    # SINR ≈ signal / interference ≈ 1 (0 dB), since noise is negligible
    assert abs(result.sinr_db[0]) < 1.0  # Near 0 dB
    assert abs(result.sinr_db[1]) < 1.0


def test_throughput_positive():
    """Throughput should always be positive."""
    power_matrix = np.array([[1e-8, 1e-10], [1e-10, 1e-8]])
    assoc = np.array([0, 1])
    result = compute_sinr(power_matrix, assoc)
    assert np.all(result.throughput_bps > 0)
