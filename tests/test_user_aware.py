"""Tests for the user-aware deployment engine."""

import numpy as np
import pytest

from dronecomm.config import Config, NetworkConfig, ScenarioConfig
from dronecomm.heuristic import mst_backhaul_orientations
from dronecomm.user_aware import (
    UserAwareObjectiveFunction,
    build_user_aware_scenario,
    compare_user_aware_methods,
    optimize_user_aware_deployment,
)


@pytest.fixture
def small_config():
    """Compact config for fast user-aware tests."""
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


def test_build_user_aware_scenario_analytic_sets_mst_backhaul(small_config, fixed_users):
    """Analytic altitude mode should derive BH pointing from the MST."""
    drone_xy = np.array([[100.0, 100.0], [400.0, 400.0]])
    scenario = build_user_aware_scenario(small_config, fixed_users, drone_xy, dl_mode="centroid")

    gateway_idx = int(np.argmin(np.linalg.norm(scenario.drone_positions[:, :2], axis=1)))
    exp_tilt, exp_azimuth = mst_backhaul_orientations(
        scenario.drone_positions, gateway_idx=gateway_idx
    )

    assert scenario.drone_positions.shape == (2, 3)
    np.testing.assert_allclose(scenario.bh_tilt_rad, exp_tilt)
    np.testing.assert_allclose(scenario.bh_azimuth_rad, exp_azimuth)


def test_build_user_aware_scenario_fixed_altitude_preserves_z(small_config, fixed_users):
    """When altitude is optimized externally, the builder must preserve the input z."""
    drone_xy = np.array([[100.0, 100.0], [400.0, 400.0]])
    drone_z = np.array([80.0, 150.0])
    scenario = build_user_aware_scenario(
        small_config,
        fixed_users,
        drone_xy,
        drone_z=drone_z,
        dl_mode="pca",
    )

    np.testing.assert_allclose(scenario.drone_positions[:, 2], drone_z)
    gateway_idx = int(np.argmin(np.linalg.norm(scenario.drone_positions[:, :2], axis=1)))
    exp_tilt, exp_azimuth = mst_backhaul_orientations(
        scenario.drone_positions, gateway_idx=gateway_idx
    )
    np.testing.assert_allclose(scenario.bh_tilt_rad, exp_tilt)
    np.testing.assert_allclose(scenario.bh_azimuth_rad, exp_azimuth)


def test_user_aware_objective_penalizes_bunched_drones(small_config, fixed_users):
    """The objective should worsen when drones violate separation heavily."""
    objective_fn = UserAwareObjectiveFunction(
        config=small_config,
        user_positions=fixed_users,
        n_drones=2,
        altitude_mode="analytic",
        dl_mode="centroid",
    )
    spread = build_user_aware_scenario(
        small_config,
        fixed_users,
        np.array([[100.0, 100.0], [400.0, 400.0]]),
    )
    bunched = build_user_aware_scenario(
        small_config,
        fixed_users,
        np.array([[250.0, 250.0], [250.0, 250.0]]),
    )

    _, spread_cost, _ = objective_fn.evaluate_scenario(spread)
    _, bunched_cost, _ = objective_fn.evaluate_scenario(bunched)

    assert bunched_cost > spread_cost


def test_pattern_search_supports_both_altitude_modes(small_config, fixed_users):
    """Pattern search should run in analytic-z and optimize-z modes."""
    analytic_result = optimize_user_aware_deployment(
        config=small_config,
        user_positions=fixed_users,
        n_drones=2,
        method="pattern_search",
        altitude_mode="analytic",
        pattern_step_schedule_xy=(40.0, 20.0),
        pattern_max_cycles_per_step=1,
        verbose=False,
    )
    optimize_z_result = optimize_user_aware_deployment(
        config=small_config,
        user_positions=fixed_users,
        n_drones=2,
        method="pattern_search",
        altitude_mode="optimize",
        z_bounds=(60.0, 180.0),
        pattern_step_schedule_xy=(40.0, 20.0),
        pattern_step_schedule_z=(20.0, 10.0),
        pattern_max_cycles_per_step=1,
        verbose=False,
    )

    assert np.isfinite(analytic_result.objective_cost)
    assert analytic_result.n_objective_evaluations > 0
    assert np.isfinite(optimize_z_result.objective_cost)
    assert np.all(optimize_z_result.best_scenario.drone_positions[:, 2] >= 60.0)
    assert np.all(optimize_z_result.best_scenario.drone_positions[:, 2] <= 180.0)


def test_cma_es_returns_valid_user_aware_result(small_config, fixed_users):
    """CMA-ES should produce a finite deployment result on a small problem."""
    result = optimize_user_aware_deployment(
        config=small_config,
        user_positions=fixed_users,
        n_drones=2,
        method="cma_es",
        altitude_mode="analytic",
        cma_maxiter=2,
        cma_popsize=4,
        cma_sigma0=40.0,
        verbose=False,
    )

    assert result.method == "cma_es"
    assert np.isfinite(result.objective_cost)
    assert result.n_objective_evaluations > 0
    assert result.best_scenario.drone_positions.shape == (2, 3)


def test_compare_user_aware_methods_runs_multiple_methods(small_config, fixed_users):
    """Method comparison should return one result per requested method."""
    results = compare_user_aware_methods(
        config=small_config,
        user_positions=fixed_users,
        n_drones=2,
        methods=("analytic", "repulsive_lloyd"),
    )

    assert set(results) == {"analytic", "repulsive_lloyd"}
    assert np.isfinite(results["analytic"].objective_cost)
    assert np.isfinite(results["repulsive_lloyd"].objective_cost)


def test_analytic_method_rejects_optimize_altitude_mode(small_config, fixed_users):
    """Direct analytic methods should reject incompatible altitude search mode."""
    with pytest.raises(ValueError, match="only supports altitude_mode='analytic'"):
        optimize_user_aware_deployment(
            config=small_config,
            user_positions=fixed_users,
            n_drones=2,
            method="analytic",
            altitude_mode="optimize",
        )
