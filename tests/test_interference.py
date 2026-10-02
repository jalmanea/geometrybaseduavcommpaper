"""Tests for interference module."""

import numpy as np
import pytest

from dronecomm.antenna import ParametricAntenna
from dronecomm.channel import ChannelModel, URBAN
from dronecomm.interference import downlink_power_matrix, backhaul_interference_matrix


@pytest.fixture
def setup():
    """Common test setup: 2 drones, 2 users."""
    drone_pos = np.array([
        [500.0, 500.0, 150.0],
        [1500.0, 500.0, 150.0],
    ])
    user_pos = np.array([
        [500.0, 500.0, 0.0],   # directly below drone 0
        [1500.0, 500.0, 0.0],  # directly below drone 1
    ])
    tilt = np.array([0.0, 0.0])  # nadir-pointing
    azimuth = np.array([0.0, 0.0])
    antenna = ParametricAntenna(g_max_dbi=18.0, beamwidth_deg=60.0, sla_db=20.0)
    channel = ChannelModel(env=URBAN, f_c=2.0e9)
    return drone_pos, user_pos, tilt, azimuth, antenna, channel


def test_power_matrix_shape(setup):
    drone_pos, user_pos, tilt, az, ant, ch = setup
    pm = downlink_power_matrix(drone_pos, tilt, az, user_pos, ant, ch, 1.0)
    assert pm.shape == (2, 2)


def test_signal_stronger_than_interference(setup):
    """Power from serving drone should exceed power from interfering drone."""
    drone_pos, user_pos, tilt, az, ant, ch = setup
    pm = downlink_power_matrix(drone_pos, tilt, az, user_pos, ant, ch, 1.0)
    # User 0 below drone 0: pm[0,0] > pm[1,0]
    assert pm[0, 0] > pm[1, 0], "Serving drone signal should be stronger"
    # User 1 below drone 1: pm[1,1] > pm[0,1]
    assert pm[1, 1] > pm[0, 1], "Serving drone signal should be stronger"


def test_symmetry(setup):
    """Symmetric setup should give symmetric power matrix."""
    drone_pos, user_pos, tilt, az, ant, ch = setup
    pm = downlink_power_matrix(drone_pos, tilt, az, user_pos, ant, ch, 1.0)
    # Signal to own user should be equal for both drones (same altitude, nadir-pointing)
    assert pm[0, 0] == pytest.approx(pm[1, 1], rel=1e-3)
    # Cross-interference should be equal
    assert pm[0, 1] == pytest.approx(pm[1, 0], rel=1e-3)


def test_tilt_increases_interference(setup):
    """Tilting drone 1 toward user 0 should increase interference at user 0."""
    drone_pos, user_pos, _, az, ant, ch = setup

    # Nadir pointing
    tilt_nadir = np.array([0.0, 0.0])
    pm_nadir = downlink_power_matrix(drone_pos, tilt_nadir, az, user_pos, ant, ch, 1.0)

    # Tilt drone 1 toward user 0 (tilt from nadir, azimuth pointing left = pi)
    tilt_tilted = np.array([0.0, np.deg2rad(45.0)])
    az_tilted = np.array([0.0, np.pi])  # pointing toward -x (toward drone 0 / user 0)
    pm_tilted = downlink_power_matrix(drone_pos, tilt_tilted, az_tilted, user_pos, ant, ch, 1.0)

    # Interference from drone 1 at user 0 should increase
    assert pm_tilted[1, 0] > pm_nadir[1, 0], \
        "Tilting toward victim should increase interference"


def test_backhaul_matrix_shape():
    """Backhaul interference matrix should be (N, N) with zero diagonal."""
    drone_pos = np.array([
        [0.0, 0.0, 150.0],
        [500.0, 0.0, 150.0],
        [250.0, 400.0, 150.0],
    ])
    tilt = np.full(3, np.pi / 2)  # horizontal
    az = np.zeros(3)
    ant = ParametricAntenna(g_max_dbi=12.0, beamwidth_deg=30.0, sla_db=25.0)
    ch = ChannelModel(env=URBAN)

    bh = backhaul_interference_matrix(drone_pos, tilt, az, ant, ch, 0.5)
    assert bh.shape == (3, 3)
    np.testing.assert_array_equal(np.diag(bh), 0.0)


def test_backhaul_all_positive():
    """Off-diagonal backhaul interference should be positive."""
    drone_pos = np.array([
        [0.0, 0.0, 150.0],
        [500.0, 0.0, 150.0],
    ])
    tilt = np.full(2, np.pi / 2)
    az = np.zeros(2)
    ant = ParametricAntenna()
    ch = ChannelModel()

    bh = backhaul_interference_matrix(drone_pos, tilt, az, ant, ch, 0.5)
    assert bh[0, 1] > 0
    assert bh[1, 0] > 0
