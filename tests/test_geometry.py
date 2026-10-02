"""Tests for geometry module."""

import numpy as np
import pytest

from dronecomm.geometry import (
    boresight_vector,
    direction_and_distance,
    elevation_angle,
    off_boresight_angle,
    pairwise_directions_and_distances,
    pairwise_elevation_angles,
    pairwise_off_boresight_angles,
)


def test_boresight_nadir():
    """Tilt=0 should point straight down."""
    b = boresight_vector(0.0, 0.0)
    np.testing.assert_allclose(b, [0, 0, -1], atol=1e-10)


def test_boresight_horizontal():
    """Tilt=pi/2, azimuth=0 should point along +x."""
    b = boresight_vector(np.pi / 2, 0.0)
    np.testing.assert_allclose(b, [1, 0, 0], atol=1e-10)


def test_boresight_unit_length():
    """Boresight vectors should be unit length."""
    tilts = np.array([0, 0.3, 0.7, np.pi / 2, np.pi])
    azimuths = np.array([0, 0.5, 1.2, np.pi, 2.5])
    bs = boresight_vector(tilts, azimuths)
    norms = np.linalg.norm(bs, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_off_boresight_same_direction():
    """Angle between identical vectors should be 0."""
    b = np.array([0, 0, -1.0])
    d = np.array([0, 0, -1.0])
    assert off_boresight_angle(b, d) == pytest.approx(0.0, abs=1e-10)


def test_off_boresight_orthogonal():
    """Angle between orthogonal vectors should be pi/2."""
    b = np.array([0, 0, -1.0])
    d = np.array([1, 0, 0.0])
    assert off_boresight_angle(b, d) == pytest.approx(np.pi / 2, abs=1e-10)


def test_off_boresight_opposite():
    """Angle between opposite vectors should be pi."""
    b = np.array([0, 0, -1.0])
    d = np.array([0, 0, 1.0])
    assert off_boresight_angle(b, d) == pytest.approx(np.pi, abs=1e-10)


def test_direction_and_distance():
    """Basic distance and direction check."""
    source = np.array([0.0, 0.0, 100.0])
    target = np.array([300.0, 400.0, 0.0])
    direction, dist = direction_and_distance(source, target)
    # Distance should be sqrt(300^2 + 400^2 + 100^2) = sqrt(260000) ≈ 509.9
    expected_dist = np.sqrt(300**2 + 400**2 + 100**2)
    assert dist == pytest.approx(expected_dist, rel=1e-6)
    # Direction should be unit length
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-10)


def test_elevation_directly_below():
    """User directly below drone -> elevation = pi/2."""
    drone = np.array([500.0, 500.0, 150.0])
    user = np.array([500.0, 500.0, 0.0])
    elev = elevation_angle(drone, user)
    assert elev == pytest.approx(np.pi / 2, abs=1e-6)


def test_elevation_far_horizontal():
    """User very far away horizontally -> elevation close to 0."""
    drone = np.array([0.0, 0.0, 150.0])
    user = np.array([100000.0, 0.0, 0.0])
    elev = elevation_angle(drone, user)
    assert elev < 0.01  # Near horizontal


def test_pairwise_shapes():
    """Verify output shapes of pairwise computations."""
    drones = np.array([[0, 0, 100], [500, 500, 150]], dtype=float)
    users = np.array([[100, 100, 0], [200, 200, 0], [300, 300, 0]], dtype=float)

    dirs, dists = pairwise_directions_and_distances(drones, users)
    assert dirs.shape == (2, 3, 3)
    assert dists.shape == (2, 3)

    elevs = pairwise_elevation_angles(drones, users)
    assert elevs.shape == (2, 3)

    boresights = np.array([[0, 0, -1], [0, 0, -1]], dtype=float)
    thetas = pairwise_off_boresight_angles(boresights, dirs)
    assert thetas.shape == (2, 3)
