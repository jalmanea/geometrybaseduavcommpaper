"""Tests for the optimization module."""

import numpy as np
import pytest

from dronecomm.config import Config
from dronecomm.optimize import (
    ObjectiveFunction,
    OptimizationProblem,
    build_bounds,
    evaluate_scenario,
    flatten,
    optimize_de,
    optimize_pso,
    unflatten,
)
from dronecomm.scenario import create_scenario


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def small_config():
    """Config with 3 drones for fast tests."""
    return Config(
        network=Config.__dataclass_fields__["network"].default_factory()
        .__class__(n_drones=3, altitude_m=120.0, min_separation_m=50.0),
        scenario=Config.__dataclass_fields__["scenario"].default_factory()
        .__class__(
            area_size_m=500.0,
            n_clusters=2,
            users_per_cluster_mean=15,
            cluster_spread_m=50.0,
        ),
    )


@pytest.fixture
def small_scenario(small_config):
    """Baseline scenario with 3 drones."""
    net = small_config.network
    sc = small_config.scenario
    return create_scenario(
        n_drones=net.n_drones,
        placement="kmeans",
        altitude_m=net.altitude_m,
        dl_tilt_deg=net.dl_tilt_deg,
        bh_tilt_deg=net.bh_tilt_deg,
        n_clusters=sc.n_clusters,
        users_per_cluster_mean=sc.users_per_cluster_mean,
        cluster_spread_m=sc.cluster_spread_m,
        area_size_m=sc.area_size_m,
        min_separation_m=net.min_separation_m,
        seed=42,
    )


# ── Problem Definition Tests ─────────────────────────────────────────


def test_vars_per_drone_full():
    """All variable groups enabled: 2 + 2 + 2 = 6 vars per drone."""
    prob = OptimizationProblem(n_drones=5)
    assert prob.vars_per_drone == 6
    assert prob.dimension == 30


def test_vars_per_drone_position_only():
    """Position only: 2 vars per drone."""
    prob = OptimizationProblem(
        n_drones=3,
        optimize_positions=True,
        optimize_dl_orientation=False,
        optimize_bh_orientation=False,
    )
    assert prob.vars_per_drone == 2
    assert prob.dimension == 6


def test_vars_per_drone_orientation_only():
    """Orientation only: 4 vars per drone (dl_tilt, dl_az, bh_tilt, bh_az)."""
    prob = OptimizationProblem(
        n_drones=4,
        optimize_positions=False,
        optimize_dl_orientation=True,
        optimize_bh_orientation=True,
    )
    assert prob.vars_per_drone == 4
    assert prob.dimension == 16


def test_vars_per_drone_with_altitude():
    """Position + altitude + orientations: 2 + 1 + 2 + 2 = 7."""
    prob = OptimizationProblem(
        n_drones=2,
        optimize_altitude=True,
    )
    assert prob.vars_per_drone == 7
    assert prob.dimension == 14


# ── Encoding Roundtrip Tests ─────────────────────────────────────────


def test_flatten_unflatten_roundtrip_full(small_scenario):
    """flatten -> unflatten should reproduce the original scenario."""
    prob = OptimizationProblem(n_drones=3)
    x = flatten(small_scenario, prob)

    assert x.shape == (prob.dimension,)

    recovered = unflatten(x, small_scenario, prob)

    np.testing.assert_allclose(
        recovered.drone_positions, small_scenario.drone_positions, atol=1e-10
    )
    np.testing.assert_allclose(
        recovered.dl_tilt_rad, small_scenario.dl_tilt_rad, atol=1e-10
    )
    np.testing.assert_allclose(
        recovered.dl_azimuth_rad, small_scenario.dl_azimuth_rad, atol=1e-10
    )
    np.testing.assert_allclose(
        recovered.bh_tilt_rad, small_scenario.bh_tilt_rad, atol=1e-10
    )
    np.testing.assert_allclose(
        recovered.bh_azimuth_rad, small_scenario.bh_azimuth_rad, atol=1e-10
    )
    # Users should be unchanged
    np.testing.assert_array_equal(
        recovered.user_positions, small_scenario.user_positions
    )


def test_flatten_unflatten_roundtrip_position_only(small_scenario):
    """Roundtrip with position-only optimization preserves positions."""
    prob = OptimizationProblem(
        n_drones=3,
        optimize_positions=True,
        optimize_dl_orientation=False,
        optimize_bh_orientation=False,
    )
    x = flatten(small_scenario, prob)
    assert x.shape == (6,)  # 3 drones * 2 vars

    recovered = unflatten(x, small_scenario, prob)
    np.testing.assert_allclose(
        recovered.drone_positions[:, :2],
        small_scenario.drone_positions[:, :2],
        atol=1e-10,
    )
    # Non-optimized vars should come from baseline
    np.testing.assert_allclose(
        recovered.dl_tilt_rad, small_scenario.dl_tilt_rad, atol=1e-10
    )


def test_unflatten_modifies_only_selected_vars(small_scenario):
    """Unflatten with a modified vector changes only the optimized vars."""
    prob = OptimizationProblem(
        n_drones=3,
        optimize_positions=False,
        optimize_dl_orientation=True,
        optimize_bh_orientation=False,
    )
    x = flatten(small_scenario, prob)
    x_mod = x + 0.1  # shift all DL orientations
    recovered = unflatten(x_mod, small_scenario, prob)

    # Positions unchanged (not optimized)
    np.testing.assert_allclose(
        recovered.drone_positions, small_scenario.drone_positions, atol=1e-10
    )
    # BH orientations unchanged (not optimized)
    np.testing.assert_allclose(
        recovered.bh_tilt_rad, small_scenario.bh_tilt_rad, atol=1e-10
    )
    # DL orientations should be modified
    assert not np.allclose(recovered.dl_tilt_rad, small_scenario.dl_tilt_rad)


def test_flatten_vector_values(small_scenario):
    """Verify the flat vector contains the expected interleaved values."""
    prob = OptimizationProblem(n_drones=3)
    x = flatten(small_scenario, prob)
    # First drone: x, y, dl_tilt, dl_az, bh_tilt, bh_az
    assert x[0] == small_scenario.drone_positions[0, 0]
    assert x[1] == small_scenario.drone_positions[0, 1]
    assert x[2] == small_scenario.dl_tilt_rad[0]
    assert x[3] == small_scenario.dl_azimuth_rad[0]
    assert x[4] == small_scenario.bh_tilt_rad[0]
    assert x[5] == small_scenario.bh_azimuth_rad[0]


# ── Bounds Tests ─────────────────────────────────────────────────────


def test_bounds_length():
    """Bounds list should have one entry per decision variable."""
    prob = OptimizationProblem(n_drones=5)
    bounds = build_bounds(prob, area_size=2000.0)
    assert len(bounds) == prob.dimension


def test_bounds_position_range():
    """Position bounds should be [0, area_size]."""
    prob = OptimizationProblem(
        n_drones=2,
        optimize_positions=True,
        optimize_dl_orientation=False,
        optimize_bh_orientation=False,
    )
    bounds = build_bounds(prob, area_size=1000.0)
    # Each drone has x, y bounds
    assert bounds[0] == (0.0, 1000.0)
    assert bounds[1] == (0.0, 1000.0)
    assert bounds[2] == (0.0, 1000.0)
    assert bounds[3] == (0.0, 1000.0)


def test_bounds_orientation_range():
    """DL tilt in [0, pi/2], azimuth in [0, 2pi], BH tilt in [0, pi]."""
    prob = OptimizationProblem(
        n_drones=1,
        optimize_positions=False,
        optimize_dl_orientation=True,
        optimize_bh_orientation=True,
    )
    bounds = build_bounds(prob, area_size=2000.0)
    assert len(bounds) == 4
    assert bounds[0] == (0.0, np.pi / 2)  # dl_tilt
    assert bounds[1] == (0.0, 2 * np.pi)  # dl_azimuth
    assert bounds[2] == (0.0, np.pi)  # bh_tilt
    assert bounds[3] == (0.0, 2 * np.pi)  # bh_azimuth


# ── Objective Function Tests ─────────────────────────────────────────


def test_objective_returns_finite(small_config, small_scenario):
    """Objective function should return a finite float."""
    prob = OptimizationProblem(n_drones=3)
    obj = ObjectiveFunction(
        config=small_config, problem=prob, baseline_scenario=small_scenario
    )
    x = flatten(small_scenario, prob)
    cost = obj(x)
    assert np.isfinite(cost)
    assert isinstance(cost, float)


def test_objective_tracks_history(small_config, small_scenario):
    """History should grow with each evaluation."""
    prob = OptimizationProblem(n_drones=3)
    obj = ObjectiveFunction(
        config=small_config, problem=prob, baseline_scenario=small_scenario
    )
    x = flatten(small_scenario, prob)
    obj(x)
    obj(x)
    obj(x)
    assert len(obj.history) == 3
    assert obj.eval_count == 3


def test_objective_reset(small_config, small_scenario):
    """Reset should clear history and eval count."""
    prob = OptimizationProblem(n_drones=3)
    obj = ObjectiveFunction(
        config=small_config, problem=prob, baseline_scenario=small_scenario
    )
    x = flatten(small_scenario, prob)
    obj(x)
    obj(x)
    obj.reset()
    assert obj.eval_count == 0
    assert len(obj.history) == 0
    assert obj.best_cost == np.inf


def test_objective_penalty_separation(small_config, small_scenario):
    """Placing drones too close should increase the cost via penalty."""
    prob = OptimizationProblem(n_drones=3)
    obj = ObjectiveFunction(
        config=small_config, problem=prob, baseline_scenario=small_scenario
    )
    x_baseline = flatten(small_scenario, prob)
    cost_baseline = obj(x_baseline)

    # Bunch all drones to the same position -> separation penalty
    obj.reset()
    x_bunched = x_baseline.copy()
    for i in range(3):
        offset = i * prob.vars_per_drone
        x_bunched[offset] = 250.0  # x
        x_bunched[offset + 1] = 250.0  # y
    cost_bunched = obj(x_bunched)

    assert cost_bunched > cost_baseline


# ── Optimizer Tests ──────────────────────────────────────────────────


def test_de_improves_over_baseline(small_config, small_scenario):
    """DE should improve (or at least match) the baseline cost."""
    prob = OptimizationProblem(n_drones=3)
    bounds = build_bounds(prob, small_config.scenario.area_size_m)
    x0 = flatten(small_scenario, prob)

    # Evaluate baseline cost
    obj_base = ObjectiveFunction(
        config=small_config, problem=prob, baseline_scenario=small_scenario
    )
    baseline_cost = obj_base(x0)

    # Run DE with minimal iterations
    obj_de = ObjectiveFunction(
        config=small_config, problem=prob, baseline_scenario=small_scenario
    )
    result = optimize_de(
        obj_de, bounds, x0=x0,
        maxiter=10, popsize=5, seed=42, verbose=False,
    )

    assert result.best_cost <= baseline_cost + 1e-6
    assert result.method == "differential_evolution"
    assert result.n_evaluations > 0
    assert len(result.history) > 0
    assert result.elapsed_seconds > 0
    assert result.final_metrics is not None


def test_pso_returns_valid_result(small_config, small_scenario):
    """PSO should return a valid OptimizationResult."""
    prob = OptimizationProblem(n_drones=3)
    bounds = build_bounds(prob, small_config.scenario.area_size_m)
    x0 = flatten(small_scenario, prob)

    obj = ObjectiveFunction(
        config=small_config, problem=prob, baseline_scenario=small_scenario
    )
    result = optimize_pso(
        obj, bounds, x0=x0,
        n_particles=5, max_iters=10, seed=42, verbose=False,
    )

    assert np.isfinite(result.best_cost)
    assert result.method == "pso"
    assert result.best_vector.shape == (prob.dimension,)
    assert result.n_evaluations > 0
    assert result.final_metrics is not None
    # Best vector should be within bounds
    for i, (lo, hi) in enumerate(bounds):
        assert lo - 1e-10 <= result.best_vector[i] <= hi + 1e-10


def test_evaluate_scenario_multi_seed(small_config, small_scenario):
    """Multi-seed evaluation should return averaged metrics."""
    metrics = evaluate_scenario(small_scenario, small_config, n_seeds=3)
    assert np.isfinite(metrics.coverage_fraction)
    assert 0.0 <= metrics.coverage_fraction <= 1.0
    assert np.isfinite(metrics.mean_sinr_db)
    assert metrics.n_drones == 3
