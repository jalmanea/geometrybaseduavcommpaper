"""Tests for heuristic deployment baselines."""

import numpy as np
import pytest

from dronecomm.config import Config, NetworkConfig, ScenarioConfig
from dronecomm.heuristic import (
    deploy_kmeans_baseline,
    deploy_kmeans_global_altitude_sweep,
    mst_backhaul_orientations,
    sweep_kmeans_global_altitude,
)


@pytest.fixture
def small_config():
    """Compact config for fast baseline tests."""
    return Config(
        network=NetworkConfig(n_drones=2, altitude_m=120.0, min_separation_m=40.0),
        scenario=ScenarioConfig(
            area_size_m=500.0,
            n_clusters=2,
            users_per_cluster_mean=6,
            users_per_cluster_std=0,
            cluster_spread_m=30.0,
        ),
    )


@pytest.fixture
def fixed_users():
    """Two compact user clusters with deterministic positions."""
    cluster_a = np.array(
        [
            [90.0, 95.0, 0.0],
            [100.0, 100.0, 0.0],
            [110.0, 105.0, 0.0],
            [95.0, 110.0, 0.0],
            [105.0, 90.0, 0.0],
            [115.0, 100.0, 0.0],
        ]
    )
    cluster_b = np.array(
        [
            [390.0, 395.0, 0.0],
            [400.0, 400.0, 0.0],
            [410.0, 405.0, 0.0],
            [395.0, 410.0, 0.0],
            [405.0, 390.0, 0.0],
            [415.0, 400.0, 0.0],
        ]
    )
    return np.vstack([cluster_a, cluster_b])


def test_deploy_kmeans_baseline_sets_common_altitude_and_mst(small_config, fixed_users):
    """The reusable K-means baseline should preserve a fixed common altitude."""
    scenario = deploy_kmeans_baseline(
        small_config,
        fixed_users,
        seed=7,
        altitude_m=90.0,
    )

    np.testing.assert_allclose(scenario.drone_positions[:, 2], 90.0)
    gateway_idx = int(np.argmin(np.linalg.norm(scenario.drone_positions[:, :2], axis=1)))
    exp_tilt, exp_azimuth = mst_backhaul_orientations(
        scenario.drone_positions,
        gateway_idx=gateway_idx,
    )
    np.testing.assert_allclose(scenario.bh_tilt_rad, exp_tilt)
    np.testing.assert_allclose(scenario.bh_azimuth_rad, exp_azimuth)


def test_sweep_kmeans_global_altitude_picks_best_candidate(small_config, fixed_users):
    """The sweep should return the best altitude from the candidate set."""
    altitudes = [60.0, 120.0, 180.0]
    scenario, entries = sweep_kmeans_global_altitude(
        small_config,
        fixed_users,
        seed=7,
        altitudes_m=altitudes,
    )

    assert len(entries) == len(altitudes)
    best_entry = max(
        entries,
        key=lambda entry: (
            entry.metrics.coverage_fraction,
            entry.metrics.sinr_5th_percentile_db,
            -entry.metrics.total_inter_drone_interference_dbm,
            -entry.altitude_m,
        ),
    )

    chosen_altitude = float(scenario.drone_positions[0, 2])
    np.testing.assert_allclose(scenario.drone_positions[:, 2], chosen_altitude)
    assert chosen_altitude in altitudes
    assert chosen_altitude == best_entry.altitude_m


def test_deploy_kmeans_global_altitude_sweep_matches_sweep_result(small_config, fixed_users):
    """The convenience wrapper should return the same best scenario as the sweep."""
    altitudes = [60.0, 120.0, 180.0]
    best_scenario, _ = sweep_kmeans_global_altitude(
        small_config,
        fixed_users,
        seed=7,
        altitudes_m=altitudes,
    )
    wrapped_scenario = deploy_kmeans_global_altitude_sweep(
        small_config,
        fixed_users,
        seed=7,
        altitudes_m=altitudes,
    )

    np.testing.assert_allclose(wrapped_scenario.drone_positions, best_scenario.drone_positions)
    np.testing.assert_allclose(wrapped_scenario.bh_tilt_rad, best_scenario.bh_tilt_rad)
    np.testing.assert_allclose(wrapped_scenario.bh_azimuth_rad, best_scenario.bh_azimuth_rad)


def test_sweep_kmeans_global_altitude_rejects_empty_grid(small_config, fixed_users):
    """The altitude sweep should reject an empty candidate grid."""
    with pytest.raises(ValueError, match="at least one altitude"):
        sweep_kmeans_global_altitude(
            small_config,
            fixed_users,
            seed=7,
            altitudes_m=[],
        )
