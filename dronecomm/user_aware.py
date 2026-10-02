"""User-aware deployment engine for fixed user placements.

This module optimizes drone deployment for a known set of user positions.
Unlike ``dronecomm.optimize``, it does not search over downlink and backhaul
antenna angles directly. Instead:

1. The optimizer searches only over drone positions ``(x, y)`` or ``(x, y, z)``.
2. The downlink antenna orientation is derived from the user geometry.
3. The backhaul antenna orientation is always induced by the drone MST.

This keeps the search space small and aligned with the deployment problem.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Sequence

import cma
import numpy as np

from .config import Config
from .heuristic import (
    apply_analytic_orientations,
    apply_analytic_pca_orientations,
    assign_users_to_drones,
    compute_cluster_stats,
    deploy_analytic_heuristic,
    deploy_analytic_pca_heuristic,
    deploy_repulsive_lloyd_heuristic,
    dl_orientation_from_centroid,
    mst_backhaul_orientations,
    pca_dl_orientations,
)
from .metrics import SimulationMetrics
from .optimize import _build_models, _evaluate_single
from .scenario import Scenario


VALID_METHODS = ("analytic", "repulsive_lloyd", "pattern_search", "cma_es")
VALID_ALTITUDE_MODES = ("analytic", "optimize")
VALID_DL_MODES = ("centroid", "pca")


@dataclass(frozen=True)
class UserAwarePositionProblem:
    """Decision-variable structure for the user-aware engine."""

    n_drones: int
    optimize_altitude: bool = False

    @property
    def vars_per_drone(self) -> int:
        return 3 if self.optimize_altitude else 2

    @property
    def dimension(self) -> int:
        return self.n_drones * self.vars_per_drone


@dataclass(frozen=True)
class UserAwareObjective:
    """Objective weights and normalization constants.

    The deployment cost minimized by this engine is:

        cost = -(w_c * coverage
                 + w_t * throughput_norm
                 + w_e * edge_sinr_norm)
               + w_i * bh_interference_norm
               + lambda_sep * separation_penalty
               + lambda_empty * empty_cluster_fraction

    Lower is better.
    """

    coverage_weight: float = 1.0
    throughput_weight: float = 0.2
    edge_sinr_weight: float = 0.3
    interference_weight: float = 0.4
    separation_penalty_lambda: float = 5.0
    empty_cluster_penalty_lambda: float = 0.25
    throughput_reference_mbps: float = 20000.0
    edge_sinr_target_db: float = 3.0
    edge_sinr_span_db: float = 10.0
    interference_good_dbm: float = -70.0
    interference_bad_dbm: float = -30.0


@dataclass
class UserAwareDeploymentResult:
    """Result of a user-aware deployment run."""

    best_scenario: Scenario
    metrics: SimulationMetrics
    objective_cost: float
    objective_terms: dict[str, float]
    method: str
    altitude_mode: str
    dl_mode: str
    n_objective_evaluations: int
    history: list[float]
    elapsed_seconds: float


def _config_with_n_drones(config: Config, n_drones: int) -> Config:
    """Return a config copy with an overridden drone count."""
    return replace(config, network=replace(config.network, n_drones=n_drones))


def _validate_user_positions(user_positions: np.ndarray) -> np.ndarray:
    """Validate the user position array shape."""
    arr = np.asarray(user_positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected user_positions shape (M, 3), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("user_positions must contain at least one user")
    return arr


def _validate_modes(method: str, altitude_mode: str, dl_mode: str) -> None:
    """Validate method and orientation mode names."""
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Expected one of {VALID_METHODS}")
    if altitude_mode not in VALID_ALTITUDE_MODES:
        raise ValueError(
            f"Unknown altitude_mode '{altitude_mode}'. Expected one of {VALID_ALTITUDE_MODES}"
        )
    if dl_mode not in VALID_DL_MODES:
        raise ValueError(f"Unknown dl_mode '{dl_mode}'. Expected one of {VALID_DL_MODES}")


def _gateway_idx(drone_positions: np.ndarray) -> int:
    """Return the gateway drone index, anchored to the most central corner."""
    return int(np.argmin(np.linalg.norm(drone_positions[:, :2], axis=1)))


def _build_empty_scenario(
    user_positions: np.ndarray,
    drone_positions: np.ndarray,
    area_size_m: float,
) -> Scenario:
    """Build a scenario shell before analytic orientation derivation."""
    n_drones = drone_positions.shape[0]
    return Scenario(
        user_positions=user_positions,
        drone_positions=drone_positions,
        dl_tilt_rad=np.zeros(n_drones),
        dl_azimuth_rad=np.zeros(n_drones),
        bh_tilt_rad=np.full(n_drones, np.pi / 2.0),
        bh_azimuth_rad=np.zeros(n_drones),
        area_size_m=area_size_m,
    )


def _build_fixed_altitude_scenario(
    user_positions: np.ndarray,
    drone_positions: np.ndarray,
    config: Config,
    dl_mode: str,
) -> Scenario:
    """Build a scenario when altitude is part of the decision variables."""
    if dl_mode == "centroid":
        stats = compute_cluster_stats(drone_positions, user_positions)
        dl_tilt = np.zeros(drone_positions.shape[0])
        dl_azimuth = np.zeros(drone_positions.shape[0])
        for idx, stat in enumerate(stats):
            tilt, azimuth = dl_orientation_from_centroid(
                drone_positions[idx], stat.centroid_xy
            )
            dl_tilt[idx] = tilt
            dl_azimuth[idx] = azimuth
    else:
        assignment = assign_users_to_drones(drone_positions, user_positions)
        dl_tilt, dl_azimuth = pca_dl_orientations(
            drone_positions, user_positions, assignment
        )

    gateway_idx = _gateway_idx(drone_positions)
    bh_tilt, bh_azimuth = mst_backhaul_orientations(
        drone_positions, gateway_idx=gateway_idx
    )
    return Scenario(
        user_positions=user_positions,
        drone_positions=drone_positions,
        dl_tilt_rad=dl_tilt,
        dl_azimuth_rad=dl_azimuth,
        bh_tilt_rad=bh_tilt,
        bh_azimuth_rad=bh_azimuth,
        area_size_m=config.scenario.area_size_m,
    )


def build_user_aware_scenario(
    config: Config,
    user_positions: np.ndarray,
    drone_xy: np.ndarray,
    drone_z: np.ndarray | None = None,
    dl_mode: str = "centroid",
) -> Scenario:
    """Build a scenario from user-aware position decisions.

    Parameters
    ----------
    config : Config
    user_positions : np.ndarray, shape (M, 3)
    drone_xy : np.ndarray, shape (N, 2)
    drone_z : np.ndarray or None, shape (N,)
        If ``None``, altitude is derived analytically from cluster spread.
        Otherwise the provided altitudes are used directly.
    dl_mode : {"centroid", "pca"}
        Downlink orientation derivation rule.
    """
    if dl_mode not in VALID_DL_MODES:
        raise ValueError(f"Unknown dl_mode '{dl_mode}'. Expected one of {VALID_DL_MODES}")

    user_positions = _validate_user_positions(user_positions)
    drone_xy = np.asarray(drone_xy, dtype=float)
    if drone_xy.ndim != 2 or drone_xy.shape[1] != 2:
        raise ValueError(f"Expected drone_xy shape (N, 2), got {drone_xy.shape}")

    drone_positions = np.zeros((drone_xy.shape[0], 3), dtype=float)
    drone_positions[:, :2] = drone_xy
    if drone_z is None:
        drone_positions[:, 2] = config.network.altitude_m
        scenario = _build_empty_scenario(
            user_positions, drone_positions, config.scenario.area_size_m
        )
        if dl_mode == "centroid":
            return apply_analytic_orientations(scenario, config)
        return apply_analytic_pca_orientations(scenario, config)

    drone_z = np.asarray(drone_z, dtype=float)
    if drone_z.shape != (drone_xy.shape[0],):
        raise ValueError(
            f"Expected drone_z shape ({drone_xy.shape[0]},), got {drone_z.shape}"
        )
    drone_positions[:, 2] = drone_z
    return _build_fixed_altitude_scenario(user_positions, drone_positions, config, dl_mode)


def compute_user_aware_cost(
    metrics: SimulationMetrics,
    scenario: Scenario,
    objective: UserAwareObjective,
    min_separation_m: float,
) -> tuple[float, dict[str, float]]:
    """Compute the deployment objective for a fixed-user scenario."""
    throughput_norm = metrics.sum_throughput_mbps / objective.throughput_reference_mbps
    edge_sinr_norm = float(
        np.clip(
            (metrics.sinr_5th_percentile_db - objective.edge_sinr_target_db)
            / objective.edge_sinr_span_db,
            0.0,
            1.0,
        )
    )
    interference_norm = float(
        np.clip(
            (metrics.total_inter_drone_interference_dbm - objective.interference_good_dbm)
            / (objective.interference_bad_dbm - objective.interference_good_dbm),
            0.0,
            1.0,
        )
    )

    stats = compute_cluster_stats(scenario.drone_positions, scenario.user_positions)
    empty_cluster_fraction = float(
        sum(stat.n_users == 0 for stat in stats) / max(len(stats), 1)
    )

    pos = scenario.drone_positions
    diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    violations = np.maximum(0.0, min_separation_m - dists) / max(min_separation_m, 1e-6)
    separation_penalty = float(np.sum(violations**2) / 2.0)

    reward = (
        objective.coverage_weight * metrics.coverage_fraction
        + objective.throughput_weight * throughput_norm
        + objective.edge_sinr_weight * edge_sinr_norm
    )
    cost = -reward
    cost += objective.interference_weight * interference_norm
    cost += objective.separation_penalty_lambda * separation_penalty
    cost += objective.empty_cluster_penalty_lambda * empty_cluster_fraction

    terms = {
        "coverage": metrics.coverage_fraction,
        "throughput_norm": throughput_norm,
        "edge_sinr_norm": edge_sinr_norm,
        "bh_interference_norm": interference_norm,
        "separation_penalty": separation_penalty,
        "empty_cluster_fraction": empty_cluster_fraction,
        "reward": reward,
        "cost": cost,
    }
    return cost, terms


def flatten_position_variables(
    scenario: Scenario,
    problem: UserAwarePositionProblem,
) -> np.ndarray:
    """Flatten the position decision variables from a scenario."""
    if problem.optimize_altitude:
        return scenario.drone_positions.reshape(-1).copy()
    return scenario.drone_positions[:, :2].reshape(-1).copy()


def unflatten_position_variables(
    x: np.ndarray,
    n_drones: int,
    optimize_altitude: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a flat position vector into ``(xy, z)`` arrays."""
    x = np.asarray(x, dtype=float)
    if optimize_altitude:
        positions = x.reshape(n_drones, 3)
        return positions[:, :2].copy(), positions[:, 2].copy()
    xy = x.reshape(n_drones, 2)
    return xy.copy(), None


def build_position_bounds(
    problem: UserAwarePositionProblem,
    area_size_m: float,
    z_bounds: tuple[float, float] = (50.0, 300.0),
) -> list[tuple[float, float]]:
    """Build bounds for the user-aware position problem."""
    per_drone = [(0.0, area_size_m), (0.0, area_size_m)]
    if problem.optimize_altitude:
        per_drone.append(z_bounds)
    return per_drone * problem.n_drones


@dataclass
class UserAwareObjectiveFunction:
    """Callable objective for continuous user-aware optimization."""

    config: Config
    user_positions: np.ndarray
    n_drones: int
    altitude_mode: str = "analytic"
    dl_mode: str = "centroid"
    objective: UserAwareObjective = field(default_factory=UserAwareObjective)
    z_bounds: tuple[float, float] = (50.0, 300.0)

    eval_count: int = field(default=0, init=False)
    best_cost: float = field(default=np.inf, init=False)
    history: list[float] = field(default_factory=list, init=False)

    _dl_antenna: object = field(default=None, init=False, repr=False)
    _bh_antenna: object = field(default=None, init=False, repr=False)
    _channel: object = field(default=None, init=False, repr=False)
    _problem: UserAwarePositionProblem = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_modes("cma_es", self.altitude_mode, self.dl_mode)
        self.user_positions = _validate_user_positions(self.user_positions)
        self.config = _config_with_n_drones(self.config, self.n_drones)
        self._problem = UserAwarePositionProblem(
            n_drones=self.n_drones,
            optimize_altitude=self.altitude_mode == "optimize",
        )
        self._dl_antenna, self._bh_antenna, self._channel = _build_models(self.config)

    @property
    def problem(self) -> UserAwarePositionProblem:
        """Return the current optimization problem definition."""
        return self._problem

    def reset(self) -> None:
        """Reset optimizer bookkeeping."""
        self.eval_count = 0
        self.best_cost = np.inf
        self.history = []

    def scenario_from_vector(self, x: np.ndarray) -> Scenario:
        """Build a full scenario from the position vector."""
        drone_xy, drone_z = unflatten_position_variables(
            x, self.n_drones, self.problem.optimize_altitude
        )
        if drone_z is not None:
            drone_z = np.clip(drone_z, self.z_bounds[0], self.z_bounds[1])
        drone_xy = np.clip(drone_xy, 0.0, self.config.scenario.area_size_m)
        return build_user_aware_scenario(
            self.config,
            self.user_positions,
            drone_xy,
            drone_z=drone_z,
            dl_mode=self.dl_mode,
        )

    def evaluate_scenario(
        self,
        scenario: Scenario,
    ) -> tuple[SimulationMetrics, float, dict[str, float]]:
        """Evaluate a full scenario under the fixed user layout."""
        metrics = _evaluate_single(
            scenario,
            self.config,
            self._dl_antenna,
            self._bh_antenna,
            self._channel,
        )
        cost, terms = compute_user_aware_cost(
            metrics,
            scenario,
            self.objective,
            self.config.network.min_separation_m,
        )
        return metrics, cost, terms

    def __call__(self, x: np.ndarray) -> float:
        scenario = self.scenario_from_vector(x)
        _, cost, _ = self.evaluate_scenario(scenario)
        self.eval_count += 1
        if cost < self.best_cost:
            self.best_cost = cost
        self.history.append(self.best_cost)
        return cost


def _result_from_scenario(
    objective_fn: UserAwareObjectiveFunction,
    scenario: Scenario,
    method: str,
    altitude_mode: str,
    dl_mode: str,
    elapsed_seconds: float,
) -> UserAwareDeploymentResult:
    """Build a deployment result from a finished scenario."""
    metrics, cost, terms = objective_fn.evaluate_scenario(scenario)
    return UserAwareDeploymentResult(
        best_scenario=scenario,
        metrics=metrics,
        objective_cost=cost,
        objective_terms=terms,
        method=method,
        altitude_mode=altitude_mode,
        dl_mode=dl_mode,
        n_objective_evaluations=objective_fn.eval_count,
        history=list(objective_fn.history),
        elapsed_seconds=elapsed_seconds,
    )


def _base_seed_scenario(
    config: Config,
    user_positions: np.ndarray,
    dl_mode: str,
    seed: int,
    init_method: str,
    repulsive_beta: float,
    repulsive_max_iters: int,
) -> Scenario:
    """Build the initial deployment seed used by search methods."""
    if init_method == "analytic":
        if dl_mode == "pca":
            return deploy_analytic_pca_heuristic(config, user_positions, seed=seed)
        return deploy_analytic_heuristic(config, user_positions, seed=seed)

    if init_method != "repulsive_lloyd":
        raise ValueError("init_method must be 'analytic' or 'repulsive_lloyd'")

    repulsive = deploy_repulsive_lloyd_heuristic(
        config,
        user_positions,
        seed=seed,
        beta=repulsive_beta,
        max_iters=repulsive_max_iters,
    )
    return build_user_aware_scenario(
        config,
        user_positions,
        repulsive.drone_positions[:, :2],
        dl_mode=dl_mode,
    )


def _pattern_search(
    objective_fn: UserAwareObjectiveFunction,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    step_schedule_xy: Sequence[float],
    step_schedule_z: Sequence[float] | None,
    max_cycles_per_step: int,
) -> np.ndarray:
    """Coordinate-style mesh search around an initial deployment."""
    x = x0.copy()
    best_cost = objective_fn(x)
    lb = np.array([lo for lo, _ in bounds], dtype=float)
    ub = np.array([hi for _, hi in bounds], dtype=float)

    if step_schedule_z is None:
        step_schedule_z = [0.0] * len(step_schedule_xy)
    if len(step_schedule_xy) != len(step_schedule_z):
        raise ValueError("step_schedule_xy and step_schedule_z must have the same length")

    moves_xy = [
        (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
        (0.0, -1.0), (0.0, 1.0),
        (1.0, -1.0), (1.0, 0.0), (1.0, 1.0),
    ]

    for step_xy, step_z in zip(step_schedule_xy, step_schedule_z):
        for _ in range(max_cycles_per_step):
            improved = False
            for drone_idx in range(objective_fn.n_drones):
                best_local_x = x.copy()
                best_local_cost = best_cost
                base = drone_idx * objective_fn.problem.vars_per_drone
                for dx, dy in moves_xy:
                    candidate = x.copy()
                    candidate[base] += dx * step_xy
                    candidate[base + 1] += dy * step_xy
                    candidate = np.clip(candidate, lb, ub)
                    cost = objective_fn(candidate)
                    if cost < best_local_cost:
                        best_local_cost = cost
                        best_local_x = candidate
                if objective_fn.problem.optimize_altitude and step_z > 0.0:
                    for dz in (-step_z, step_z):
                        candidate = best_local_x.copy()
                        candidate[base + 2] += dz
                        candidate = np.clip(candidate, lb, ub)
                        cost = objective_fn(candidate)
                        if cost < best_local_cost:
                            best_local_cost = cost
                            best_local_x = candidate
                if best_local_cost < best_cost:
                    x = best_local_x
                    best_cost = best_local_cost
                    improved = True
            if not improved:
                break
    return x


def optimize_user_aware_deployment(
    config: Config,
    user_positions: np.ndarray,
    n_drones: int,
    method: str = "cma_es",
    altitude_mode: str = "analytic",
    dl_mode: str = "centroid",
    objective: UserAwareObjective | None = None,
    z_bounds: tuple[float, float] = (50.0, 300.0),
    seed: int = 42,
    init_method: str = "repulsive_lloyd",
    repulsive_beta: float = 0.15,
    repulsive_max_iters: int = 20,
    pattern_step_schedule_xy: Sequence[float] | None = None,
    pattern_step_schedule_z: Sequence[float] | None = None,
    pattern_max_cycles_per_step: int = 2,
    cma_sigma0: float | None = None,
    cma_maxiter: int = 80,
    cma_popsize: int | None = None,
    verbose: bool = False,
) -> UserAwareDeploymentResult:
    """Optimize deployment for a fixed user layout.

    Methods
    -------
    analytic
        K-means placement with analytic altitude and DL orientation.
    repulsive_lloyd
        Repulsive Lloyd refinement with analytic altitude and DL orientation.
    pattern_search
        Local mesh search over positions with analytic or optimized altitude.
    cma_es
        Continuous global search over positions with analytic or optimized altitude.
    """
    _validate_modes(method, altitude_mode, dl_mode)
    if method in {"analytic", "repulsive_lloyd"} and altitude_mode != "analytic":
        raise ValueError(
            f"method='{method}' only supports altitude_mode='analytic'"
        )

    local_config = _config_with_n_drones(config, n_drones)
    objective_fn = UserAwareObjectiveFunction(
        config=local_config,
        user_positions=_validate_user_positions(user_positions),
        n_drones=n_drones,
        altitude_mode=altitude_mode,
        dl_mode=dl_mode,
        objective=objective or UserAwareObjective(),
        z_bounds=z_bounds,
    )

    t0 = time.time()

    if method == "analytic":
        if dl_mode == "pca":
            scenario = deploy_analytic_pca_heuristic(local_config, user_positions, seed=seed)
        else:
            scenario = deploy_analytic_heuristic(local_config, user_positions, seed=seed)
        objective_fn(scenario.drone_positions[:, :2].reshape(-1))
        return _result_from_scenario(
            objective_fn, scenario, method, altitude_mode, dl_mode, time.time() - t0
        )

    if method == "repulsive_lloyd":
        scenario = _base_seed_scenario(
            local_config,
            user_positions,
            dl_mode,
            seed,
            init_method="repulsive_lloyd",
            repulsive_beta=repulsive_beta,
            repulsive_max_iters=repulsive_max_iters,
        )
        objective_fn(flatten_position_variables(scenario, objective_fn.problem))
        return _result_from_scenario(
            objective_fn, scenario, method, altitude_mode, dl_mode, time.time() - t0
        )

    seed_scenario = _base_seed_scenario(
        local_config,
        user_positions,
        dl_mode,
        seed,
        init_method=init_method,
        repulsive_beta=repulsive_beta,
        repulsive_max_iters=repulsive_max_iters,
    )
    x0 = flatten_position_variables(seed_scenario, objective_fn.problem)
    bounds = build_position_bounds(
        objective_fn.problem, local_config.scenario.area_size_m, z_bounds=z_bounds
    )

    if method == "pattern_search":
        if pattern_step_schedule_xy is None:
            area = local_config.scenario.area_size_m
            pattern_step_schedule_xy = (area / 8.0, area / 16.0, area / 32.0)
        if objective_fn.problem.optimize_altitude and pattern_step_schedule_z is None:
            z_span = z_bounds[1] - z_bounds[0]
            pattern_step_schedule_z = (z_span / 6.0, z_span / 12.0, z_span / 24.0)
        best_x = _pattern_search(
            objective_fn,
            x0,
            bounds,
            pattern_step_schedule_xy,
            pattern_step_schedule_z,
            pattern_max_cycles_per_step,
        )
        best_scenario = objective_fn.scenario_from_vector(best_x)
        return _result_from_scenario(
            objective_fn,
            best_scenario,
            method,
            altitude_mode,
            dl_mode,
            time.time() - t0,
        )

    if cma_sigma0 is None:
        span = np.array([hi - lo for lo, hi in bounds], dtype=float)
        cma_sigma0 = float(np.mean(span) / 5.0)

    cma_options = {
        "bounds": [
            [lo for lo, _ in bounds],
            [hi for _, hi in bounds],
        ],
        "seed": seed,
        "maxiter": cma_maxiter,
        "verbose": 1 if verbose else -9,
    }
    if cma_popsize is not None:
        cma_options["popsize"] = cma_popsize

    objective_fn.reset()
    es = cma.CMAEvolutionStrategy(x0.tolist(), cma_sigma0, cma_options)
    es.optimize(objective_fn)
    best_x = np.asarray(es.result.xbest, dtype=float)
    best_scenario = objective_fn.scenario_from_vector(best_x)
    return _result_from_scenario(
        objective_fn,
        best_scenario,
        method,
        altitude_mode,
        dl_mode,
        time.time() - t0,
    )


def compare_user_aware_methods(
    config: Config,
    user_positions: np.ndarray,
    n_drones: int,
    methods: Sequence[str] = VALID_METHODS,
    **kwargs,
) -> dict[str, UserAwareDeploymentResult]:
    """Run several user-aware deployment methods on the same fixed users."""
    results: dict[str, UserAwareDeploymentResult] = {}
    for method in methods:
        results[method] = optimize_user_aware_deployment(
            config=config,
            user_positions=user_positions,
            n_drones=n_drones,
            method=method,
            **kwargs,
        )
    return results
