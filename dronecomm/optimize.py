"""Gradient-free optimization of drone deployment.

Provides three optimization methods:
  - Differential Evolution (via scipy)
  - Alternating Optimization (position <-> orientation)
  - Particle Swarm Optimization (custom implementation)

All methods work with the existing simulation pipeline without
modifying core modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .antenna import ParametricAntenna, IsotropicAntenna
from .channel import ChannelModel, ENVIRONMENTS
from .config import Config
from .interference import downlink_power_matrix, backhaul_interference_matrix
from .metrics import SimulationMetrics, compute_metrics
from .scenario import Scenario, create_scenario, generate_users
from .sinr import compute_sinr, nearest_drone_association, dbm_to_watts


# ── Problem Definition ───────────────────────────────────────────────


@dataclass(frozen=True)
class OptimizationProblem:
    """Defines which decision variables to optimize and their structure."""

    n_drones: int
    optimize_positions: bool = True
    optimize_altitude: bool = False
    optimize_dl_orientation: bool = True
    optimize_bh_orientation: bool = True

    @property
    def vars_per_drone(self) -> int:
        n = 0
        if self.optimize_positions:
            n += 2  # x, y
        if self.optimize_altitude:
            n += 1  # z
        if self.optimize_dl_orientation:
            n += 2  # dl_tilt, dl_azimuth
        if self.optimize_bh_orientation:
            n += 2  # bh_tilt, bh_azimuth
        return n

    @property
    def dimension(self) -> int:
        return self.vars_per_drone * self.n_drones


# ── Variable Encoding ────────────────────────────────────────────────


def flatten(scenario: Scenario, problem: OptimizationProblem) -> np.ndarray:
    """Extract decision variables from a Scenario into a flat vector."""
    parts: list[float] = []
    for i in range(problem.n_drones):
        if problem.optimize_positions:
            parts.append(scenario.drone_positions[i, 0])
            parts.append(scenario.drone_positions[i, 1])
        if problem.optimize_altitude:
            parts.append(scenario.drone_positions[i, 2])
        if problem.optimize_dl_orientation:
            parts.append(scenario.dl_tilt_rad[i])
            parts.append(scenario.dl_azimuth_rad[i])
        if problem.optimize_bh_orientation:
            parts.append(scenario.bh_tilt_rad[i])
            parts.append(scenario.bh_azimuth_rad[i])
    return np.array(parts)


def unflatten(
    x: np.ndarray, baseline: Scenario, problem: OptimizationProblem
) -> Scenario:
    """Create a new Scenario by injecting decision variables from a flat vector."""
    drone_pos = baseline.drone_positions.copy()
    dl_tilt = baseline.dl_tilt_rad.copy()
    dl_az = baseline.dl_azimuth_rad.copy()
    bh_tilt = baseline.bh_tilt_rad.copy()
    bh_az = baseline.bh_azimuth_rad.copy()

    vpd = problem.vars_per_drone
    for i in range(problem.n_drones):
        offset = i * vpd
        j = 0
        if problem.optimize_positions:
            drone_pos[i, 0] = x[offset + j]; j += 1
            drone_pos[i, 1] = x[offset + j]; j += 1
        if problem.optimize_altitude:
            drone_pos[i, 2] = x[offset + j]; j += 1
        if problem.optimize_dl_orientation:
            dl_tilt[i] = x[offset + j]; j += 1
            dl_az[i] = x[offset + j]; j += 1
        if problem.optimize_bh_orientation:
            bh_tilt[i] = x[offset + j]; j += 1
            bh_az[i] = x[offset + j]; j += 1

    return Scenario(
        user_positions=baseline.user_positions,
        drone_positions=drone_pos,
        dl_tilt_rad=dl_tilt,
        dl_azimuth_rad=dl_az,
        bh_tilt_rad=bh_tilt,
        bh_azimuth_rad=bh_az,
        area_size_m=baseline.area_size_m,
    )


def build_bounds(
    problem: OptimizationProblem,
    area_size: float,
    z_min: float = 30.0,
    z_max: float = 120.0,
) -> list[tuple[float, float]]:
    """Build (lower, upper) bounds for each variable in the flat vector."""
    per_drone: list[tuple[float, float]] = []
    if problem.optimize_positions:
        per_drone.extend([(0.0, area_size), (0.0, area_size)])
    if problem.optimize_altitude:
        per_drone.append((z_min, z_max))
    if problem.optimize_dl_orientation:
        per_drone.extend([(0.0, np.pi / 2), (0.0, 2 * np.pi)])
    if problem.optimize_bh_orientation:
        per_drone.extend([(0.0, np.pi), (0.0, 2 * np.pi)])
    return per_drone * problem.n_drones


# ── Results ──────────────────────────────────────────────────────────


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    best_scenario: Scenario
    best_cost: float
    best_vector: np.ndarray
    history: list[float]
    n_evaluations: int
    method: str
    elapsed_seconds: float
    final_metrics: SimulationMetrics


# ── Pipeline Evaluation ──────────────────────────────────────────────


def _build_models(config: Config):
    """Build antenna and channel models from config."""
    ant = config.antenna
    ch_cfg = config.channel
    dl_antenna = ParametricAntenna(
        g_max_dbi=ant.dl_g_max_dbi,
        beamwidth_deg=ant.dl_beamwidth_deg,
        sla_db=ant.dl_sla_db,
    )
    bh_antenna = ParametricAntenna(
        g_max_dbi=ant.bh_g_max_dbi,
        beamwidth_deg=ant.bh_beamwidth_deg,
        sla_db=ant.bh_sla_db,
    )
    env = ENVIRONMENTS[ch_cfg.environment]
    channel = ChannelModel(env=env, f_c=ch_cfg.carrier_freq_hz)
    return dl_antenna, bh_antenna, channel


def _evaluate_single(
    scenario: Scenario, config: Config,
    dl_antenna: ParametricAntenna, bh_antenna: ParametricAntenna,
    channel: ChannelModel,
) -> SimulationMetrics:
    """Run the full pipeline on a scenario and return metrics."""
    net = config.network
    p_tx_dl_w = dbm_to_watts(net.p_tx_dl_dbm)
    p_tx_bh_w = dbm_to_watts(net.p_tx_bh_dbm)

    power_mat = downlink_power_matrix(
        drone_positions=scenario.drone_positions,
        dl_tilt_rad=scenario.dl_tilt_rad,
        dl_azimuth_rad=scenario.dl_azimuth_rad,
        user_positions=scenario.user_positions,
        dl_antenna=dl_antenna,
        channel=channel,
        p_tx_dl_w=p_tx_dl_w,
    )
    association = nearest_drone_association(
        scenario.drone_positions, scenario.user_positions
    )
    sinr_result = compute_sinr(
        power_mat, association,
        noise_psd_dbm_hz=net.noise_psd_dbm_hz,
        bandwidth_hz=net.bandwidth_hz,
    )
    bh_interf = backhaul_interference_matrix(
        drone_positions=scenario.drone_positions,
        bh_tilt_rad=scenario.bh_tilt_rad,
        bh_azimuth_rad=scenario.bh_azimuth_rad,
        bh_antenna=bh_antenna,
        channel=channel,
        p_tx_bh_w=p_tx_bh_w,
    )
    return compute_metrics(
        sinr_result,
        sinr_threshold_db=net.sinr_threshold_db,
        backhaul_interference_matrix=bh_interf,
    )


def evaluate_scenario(
    scenario: Scenario,
    config: Config,
    n_seeds: int = 10,
    base_seed: int = 42,
) -> SimulationMetrics:
    """Evaluate a scenario robustly with multiple random user distributions.

    Re-generates users for each seed while keeping drone deployment fixed.
    Averages scalar metrics across seeds.
    """
    dl_antenna, bh_antenna, channel = _build_models(config)
    sc_cfg = config.scenario

    metrics_list: list[SimulationMetrics] = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        user_pos = generate_users(
            n_clusters=sc_cfg.n_clusters,
            users_per_cluster_mean=sc_cfg.users_per_cluster_mean,
            users_per_cluster_std=sc_cfg.users_per_cluster_std,
            cluster_spread_m=sc_cfg.cluster_spread_m,
            area_size_m=sc_cfg.area_size_m,
            rng=rng,
        )
        eval_sc = Scenario(
            user_positions=user_pos,
            drone_positions=scenario.drone_positions,
            dl_tilt_rad=scenario.dl_tilt_rad,
            dl_azimuth_rad=scenario.dl_azimuth_rad,
            bh_tilt_rad=scenario.bh_tilt_rad,
            bh_azimuth_rad=scenario.bh_azimuth_rad,
            area_size_m=scenario.area_size_m,
        )
        metrics_list.append(
            _evaluate_single(eval_sc, config, dl_antenna, bh_antenna, channel)
        )

    return _average_metrics(metrics_list)


def _average_metrics(metrics_list: list[SimulationMetrics]) -> SimulationMetrics:
    """Average scalar fields across a list of SimulationMetrics."""
    n = len(metrics_list)
    return SimulationMetrics(
        sum_throughput_mbps=sum(m.sum_throughput_mbps for m in metrics_list) / n,
        mean_sinr_db=sum(m.mean_sinr_db for m in metrics_list) / n,
        median_sinr_db=sum(m.median_sinr_db for m in metrics_list) / n,
        min_sinr_db=sum(m.min_sinr_db for m in metrics_list) / n,
        sinr_5th_percentile_db=sum(m.sinr_5th_percentile_db for m in metrics_list) / n,
        coverage_fraction=sum(m.coverage_fraction for m in metrics_list) / n,
        total_inter_drone_interference_dbm=sum(
            m.total_inter_drone_interference_dbm for m in metrics_list
        ) / n,
        worst_pair_interference_dbm=sum(
            m.worst_pair_interference_dbm for m in metrics_list
        ) / n,
        n_users=metrics_list[0].n_users,
        n_drones=metrics_list[0].n_drones,
    )


# ── Standalone Cost Function ─────────────────────────────────────────


def compute_cost(
    metrics: SimulationMetrics,
    scenario: Scenario,
    weights: dict[str, float],
    penalty_lambdas: dict[str, float],
    min_separation_m: float,
) -> float:
    """Compute optimization cost from metrics (lower is better)."""
    w = weights
    obj = -(
        w["coverage"] * metrics.coverage_fraction
        + w.get("throughput", 0.0) * metrics.sum_throughput_mbps / 20000.0
    )

    # Minimum inter-drone separation penalty (vectorized)
    lam = penalty_lambdas
    pos = scenario.drone_positions
    diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    violations = np.maximum(0.0, min_separation_m - dists) / min_separation_m
    obj += lam["separation"] * np.sum(violations) / 2

    return obj


# ── Objective Function ───────────────────────────────────────────────


@dataclass
class ObjectiveFunction:
    """Wraps the simulation pipeline as a callable for optimizers.

    Calling an instance with a flat variable vector returns a scalar cost
    (to minimize).

    Parameters
    ----------
    n_train_samples : int
        Number of random user distributions to sample per evaluation.
        Default is 1 (original behavior). Set to 5+ for stochastic training
        that prevents overfitting to a single user distribution.
    train_seed : int
        Base seed for reproducible random user sampling during training.
    """

    config: Config
    problem: OptimizationProblem
    baseline_scenario: Scenario
    weights: dict[str, float] = field(
        default_factory=lambda: {"coverage": 1.0, "throughput": 0.0}
    )
    penalty_lambdas: dict[str, float] = field(
        default_factory=lambda: {"separation": 5.0}
    )
    n_train_samples: int = 1
    train_seed: int = 42

    # Internal state
    eval_count: int = field(default=0, init=False)
    best_cost: float = field(default=np.inf, init=False)
    history: list[float] = field(default_factory=list, init=False)
    _rng: np.random.Generator = field(default=None, init=False, repr=False)

    # Cached models (built once)
    _dl_antenna: ParametricAntenna = field(default=None, init=False, repr=False)
    _bh_antenna: ParametricAntenna = field(default=None, init=False, repr=False)
    _channel: ChannelModel = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._dl_antenna, self._bh_antenna, self._channel = _build_models(self.config)
        self._rng = np.random.default_rng(self.train_seed)

    def __call__(self, x: np.ndarray) -> float:
        if self.n_train_samples == 1:
            # Original behavior: evaluate on fixed baseline users
            obj = self._evaluate_single_distribution(x)
        else:
            # Stochastic: evaluate on K random user distributions
            costs = []
            for _ in range(self.n_train_samples):
                seed = int(self._rng.integers(0, 100000))
                cost = self._evaluate_with_random_users(x, seed)
                costs.append(cost)
            obj = float(np.mean(costs))

        self.eval_count += 1
        if obj < self.best_cost:
            self.best_cost = obj
        self.history.append(self.best_cost)
        return obj

    def _evaluate_single_distribution(self, x: np.ndarray) -> float:
        """Evaluate on the fixed baseline user distribution (original behavior)."""
        scenario = unflatten(x, self.baseline_scenario, self.problem)
        metrics = _evaluate_single(
            scenario, self.config, self._dl_antenna, self._bh_antenna, self._channel
        )
        return self._compute_objective(metrics, scenario)

    def _evaluate_with_random_users(self, x: np.ndarray, seed: int) -> float:
        """Evaluate candidate with randomly generated users."""
        rng = np.random.default_rng(seed)
        sc_cfg = self.config.scenario

        user_pos = generate_users(
            n_clusters=sc_cfg.n_clusters,
            users_per_cluster_mean=sc_cfg.users_per_cluster_mean,
            users_per_cluster_std=sc_cfg.users_per_cluster_std,
            cluster_spread_m=sc_cfg.cluster_spread_m,
            area_size_m=sc_cfg.area_size_m,
            rng=rng,
        )

        # Create scenario with new users but optimized drone config
        base = unflatten(x, self.baseline_scenario, self.problem)
        scenario = Scenario(
            user_positions=user_pos,
            drone_positions=base.drone_positions,
            dl_tilt_rad=base.dl_tilt_rad,
            dl_azimuth_rad=base.dl_azimuth_rad,
            bh_tilt_rad=base.bh_tilt_rad,
            bh_azimuth_rad=base.bh_azimuth_rad,
            area_size_m=base.area_size_m,
        )

        metrics = _evaluate_single(
            scenario, self.config, self._dl_antenna, self._bh_antenna, self._channel
        )
        return self._compute_objective(metrics, scenario)

    def _compute_objective(
        self, metrics: SimulationMetrics, scenario: Scenario
    ) -> float:
        """Compute objective value from metrics."""
        return compute_cost(
            metrics, scenario, self.weights, self.penalty_lambdas,
            self.config.network.min_separation_m,
        )

    def reset(self):
        """Reset evaluation counter and history for a fresh optimization run."""
        self.eval_count = 0
        self.best_cost = np.inf
        self.history = []
        self._rng = np.random.default_rng(self.train_seed)


# ── Optimizer 1: Differential Evolution ──────────────────────────────


def optimize_de(
    objective: ObjectiveFunction,
    bounds: list[tuple[float, float]],
    x0: np.ndarray | None = None,
    maxiter: int = 200,
    popsize: int = 15,
    tol: float = 1e-6,
    mutation: tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int = 42,
    verbose: bool = True,
) -> OptimizationResult:
    """Differential Evolution optimization (scipy)."""
    objective.reset()
    t0 = time.time()

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        x0=x0,
        polish=False,
        disp=verbose,
    )

    elapsed = time.time() - t0
    best_scenario = unflatten(result.x, objective.baseline_scenario, objective.problem)

    # Final robust evaluation
    final_metrics = evaluate_scenario(best_scenario, objective.config, n_seeds=10)

    return OptimizationResult(
        best_scenario=best_scenario,
        best_cost=result.fun,
        best_vector=result.x.copy(),
        history=list(objective.history),
        n_evaluations=objective.eval_count,
        method="differential_evolution",
        elapsed_seconds=elapsed,
        final_metrics=final_metrics,
    )


# ── Optimizer 2: Particle Swarm Optimization ─────────────────────────


def optimize_pso(
    objective: ObjectiveFunction,
    bounds: list[tuple[float, float]],
    x0: np.ndarray | None = None,
    n_particles: int = 50,
    max_iters: int = 200,
    w_start: float = 0.9,
    w_end: float = 0.4,
    c1: float = 1.5,
    c2: float = 1.5,
    v_max_fraction: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> OptimizationResult:
    """Particle Swarm Optimization with linear inertia decay and velocity clamping.

    Parameters
    ----------
    objective : ObjectiveFunction
    bounds : list of (lower, upper) per dimension
    x0 : optional initial solution seeded as first particle
    n_particles : int
        Swarm size. Should be >= dimension for high-D problems.
    max_iters : int
    w_start, w_end : float
        Inertia weight decays linearly from w_start to w_end over iterations.
        High initial inertia encourages exploration; low final inertia
        encourages exploitation.
    c1, c2 : float
        Cognitive and social acceleration coefficients.
    v_max_fraction : float
        Maximum velocity per dimension as a fraction of the variable range.
        Prevents particles from overshooting and sticking to boundaries.
    seed : int
    verbose : bool
    """
    objective.reset()
    t0 = time.time()

    rng = np.random.default_rng(seed)
    D = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    span = ub - lb

    # Velocity clamp: prevent particles from overshooting
    v_max = v_max_fraction * span

    # Initialize particles
    positions = rng.uniform(lb, ub, size=(n_particles, D))
    if x0 is not None:
        positions[0] = x0  # seed first particle with baseline
    velocities = rng.uniform(-v_max, v_max, size=(n_particles, D))

    # Evaluate initial population
    pbest_cost = np.array([objective(p) for p in positions])
    pbest_pos = positions.copy()

    gbest_idx = np.argmin(pbest_cost)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_cost = float(pbest_cost[gbest_idx])

    for iteration in range(max_iters):
        # Linear inertia weight decay
        w = w_start - (w_start - w_end) * iteration / max(max_iters - 1, 1)

        r1 = rng.uniform(0, 1, size=(n_particles, D))
        r2 = rng.uniform(0, 1, size=(n_particles, D))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )

        # Clamp velocities
        velocities = np.clip(velocities, -v_max, v_max)

        positions = np.clip(positions + velocities, lb, ub)

        costs = np.array([objective(p) for p in positions])

        improved = costs < pbest_cost
        pbest_pos[improved] = positions[improved]
        pbest_cost[improved] = costs[improved]

        gen_best_idx = np.argmin(pbest_cost)
        if pbest_cost[gen_best_idx] < gbest_cost:
            gbest_pos = pbest_pos[gen_best_idx].copy()
            gbest_cost = float(pbest_cost[gen_best_idx])

        if verbose and (iteration + 1) % 20 == 0:
            print(f"  PSO iter {iteration + 1}/{max_iters}: best = {gbest_cost:.4f}")

    elapsed = time.time() - t0
    best_scenario = unflatten(gbest_pos, objective.baseline_scenario, objective.problem)
    final_metrics = evaluate_scenario(best_scenario, objective.config, n_seeds=10)

    return OptimizationResult(
        best_scenario=best_scenario,
        best_cost=gbest_cost,
        best_vector=gbest_pos.copy(),
        history=list(objective.history),
        n_evaluations=objective.eval_count,
        method="pso",
        elapsed_seconds=elapsed,
        final_metrics=final_metrics,
    )


# ── Optimizer 3: Alternating Optimization ────────────────────────────


def optimize_alternating(
    config: Config,
    baseline_scenario: Scenario,
    max_outer_iters: int = 10,
    de_maxiter_per_phase: int = 100,
    de_popsize: int = 10,
    optimize_altitude: bool = False,
    tol: float = 1e-4,
    weights: dict[str, float] | None = None,
    penalty_lambdas: dict[str, float] | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> OptimizationResult:
    """Alternating optimization: positions <-> orientations.

    Outer loop alternates between:
      Phase A: Fix positions, optimize all orientations with DE
      Phase B: Fix orientations, optimize positions (x, y, and optionally z) with DE

    Parameters
    ----------
    optimize_altitude : bool
        If True, altitude (z) is included in the position phase.
    """
    t0 = time.time()
    area_size = baseline_scenario.area_size_m
    n_drones = baseline_scenario.drone_positions.shape[0]
    kw = {}
    if weights is not None:
        kw["weights"] = weights
    if penalty_lambdas is not None:
        kw["penalty_lambdas"] = penalty_lambdas

    current_scenario = baseline_scenario
    full_history: list[float] = []
    total_evals = 0

    # Full problem for consistent convergence evaluation
    problem_full = OptimizationProblem(
        n_drones=n_drones, optimize_altitude=optimize_altitude,
    )

    # Evaluate baseline with a fresh objective on the full problem
    obj_eval = ObjectiveFunction(
        config=config, problem=problem_full,
        baseline_scenario=current_scenario, **kw,
    )
    best_cost = obj_eval(flatten(current_scenario, problem_full))
    total_evals += 1
    full_history.append(best_cost)

    if verbose:
        print(f"  Alternating: baseline cost = {best_cost:.4f}")

    for outer in range(max_outer_iters):
        # Phase A: Fix positions, optimize orientations
        problem_orient = OptimizationProblem(
            n_drones=n_drones,
            optimize_positions=False,
            optimize_altitude=False,
            optimize_dl_orientation=True,
            optimize_bh_orientation=True,
        )
        obj_orient = ObjectiveFunction(
            config=config, problem=problem_orient,
            baseline_scenario=current_scenario, **kw,
        )
        bounds_orient = build_bounds(problem_orient, area_size)
        x0_orient = flatten(current_scenario, problem_orient)

        res_orient = differential_evolution(
            obj_orient, bounds=bounds_orient,
            maxiter=de_maxiter_per_phase, popsize=de_popsize,
            seed=seed + outer * 2, x0=x0_orient,
            polish=False, disp=False,
        )
        current_scenario = unflatten(
            res_orient.x, current_scenario, problem_orient
        )
        total_evals += obj_orient.eval_count
        full_history.extend(obj_orient.history)

        # Phase B: Fix orientations, optimize positions (and optionally altitude)
        problem_pos = OptimizationProblem(
            n_drones=n_drones,
            optimize_positions=True,
            optimize_altitude=optimize_altitude,
            optimize_dl_orientation=False,
            optimize_bh_orientation=False,
        )
        obj_pos = ObjectiveFunction(
            config=config, problem=problem_pos,
            baseline_scenario=current_scenario, **kw,
        )
        bounds_pos = build_bounds(problem_pos, area_size)
        x0_pos = flatten(current_scenario, problem_pos)

        res_pos = differential_evolution(
            obj_pos, bounds=bounds_pos,
            maxiter=de_maxiter_per_phase, popsize=de_popsize,
            seed=seed + outer * 2 + 1, x0=x0_pos,
            polish=False, disp=False,
        )
        current_scenario = unflatten(
            res_pos.x, current_scenario, problem_pos
        )
        total_evals += obj_pos.eval_count
        full_history.extend(obj_pos.history)

        # Evaluate combined cost on the full problem for consistent comparison
        obj_check = ObjectiveFunction(
            config=config, problem=problem_full,
            baseline_scenario=current_scenario, **kw,
        )
        combined_cost = obj_check(flatten(current_scenario, problem_full))
        total_evals += 1
        full_history.append(combined_cost)

        if verbose:
            print(
                f"  Alternating iter {outer + 1}: "
                f"orient={res_orient.fun:.4f}, pos={res_pos.fun:.4f}, "
                f"combined={combined_cost:.4f}"
            )
        if abs(best_cost - combined_cost) < tol:
            if verbose:
                print("  Converged.")
            break
        best_cost = min(best_cost, combined_cost)

    elapsed = time.time() - t0

    final_metrics = evaluate_scenario(current_scenario, config, n_seeds=10)

    return OptimizationResult(
        best_scenario=current_scenario,
        best_cost=best_cost,
        best_vector=flatten(current_scenario, problem_full),
        history=full_history,
        n_evaluations=total_evals,
        method="alternating",
        elapsed_seconds=elapsed,
        final_metrics=final_metrics,
    )


# ── Optimizer 4: Two-Phase (Positions then Per-Drone Orientations) ───


def _optimize_single_drone_orientation(
    drone_idx: int,
    current_scenario: Scenario,
    config: Config,
    dl_antenna: ParametricAntenna,
    bh_antenna: ParametricAntenna,
    channel: ChannelModel,
    user_seeds: list[int],
    weights: dict[str, float],
    penalty_lambdas: dict[str, float],
    method: str = "Powell",
    maxiter: int = 200,
) -> tuple[Scenario, int, list[float]]:
    """Optimize one drone's 4 orientation variables while all others stay fixed.

    Returns (updated_scenario, eval_count, cost_history).
    """
    sc_cfg = config.scenario
    min_sep = config.network.min_separation_m
    eval_count = 0
    cost_history: list[float] = []

    # Extract current orientation as x0
    x0 = np.array([
        current_scenario.dl_tilt_rad[drone_idx],
        current_scenario.dl_azimuth_rad[drone_idx],
        current_scenario.bh_tilt_rad[drone_idx],
        current_scenario.bh_azimuth_rad[drone_idx],
    ])

    bounds_4d = [
        (0.0, np.pi / 2),   # dl_tilt
        (0.0, 2 * np.pi),   # dl_azimuth
        (0.0, np.pi),        # bh_tilt
        (0.0, 2 * np.pi),   # bh_azimuth
    ]

    def local_objective(orient_4d: np.ndarray) -> float:
        nonlocal eval_count
        # Build modified scenario with only this drone's orientation changed
        dl_tilt = current_scenario.dl_tilt_rad.copy()
        dl_az = current_scenario.dl_azimuth_rad.copy()
        bh_tilt = current_scenario.bh_tilt_rad.copy()
        bh_az = current_scenario.bh_azimuth_rad.copy()

        dl_tilt[drone_idx] = orient_4d[0]
        dl_az[drone_idx] = orient_4d[1]
        bh_tilt[drone_idx] = orient_4d[2]
        bh_az[drone_idx] = orient_4d[3]

        costs = []
        for seed in user_seeds:
            rng = np.random.default_rng(seed)
            user_pos = generate_users(
                n_clusters=sc_cfg.n_clusters,
                users_per_cluster_mean=sc_cfg.users_per_cluster_mean,
                users_per_cluster_std=sc_cfg.users_per_cluster_std,
                cluster_spread_m=sc_cfg.cluster_spread_m,
                area_size_m=sc_cfg.area_size_m,
                rng=rng,
            )
            scenario = Scenario(
                user_positions=user_pos,
                drone_positions=current_scenario.drone_positions,
                dl_tilt_rad=dl_tilt,
                dl_azimuth_rad=dl_az,
                bh_tilt_rad=bh_tilt,
                bh_azimuth_rad=bh_az,
                area_size_m=current_scenario.area_size_m,
            )
            metrics = _evaluate_single(scenario, config, dl_antenna, bh_antenna, channel)
            costs.append(compute_cost(metrics, scenario, weights, penalty_lambdas, min_sep))

        eval_count += 1
        avg_cost = float(np.mean(costs))
        cost_history.append(avg_cost)
        return avg_cost

    result = minimize(
        local_objective, x0, method=method,
        bounds=bounds_4d,
        options={"maxiter": maxiter, "maxfev": maxiter * 4},
    )

    # Apply best orientation to scenario
    dl_tilt = current_scenario.dl_tilt_rad.copy()
    dl_az = current_scenario.dl_azimuth_rad.copy()
    bh_tilt = current_scenario.bh_tilt_rad.copy()
    bh_az = current_scenario.bh_azimuth_rad.copy()

    dl_tilt[drone_idx] = result.x[0]
    dl_az[drone_idx] = result.x[1]
    bh_tilt[drone_idx] = result.x[2]
    bh_az[drone_idx] = result.x[3]

    updated = Scenario(
        user_positions=current_scenario.user_positions,
        drone_positions=current_scenario.drone_positions,
        dl_tilt_rad=dl_tilt,
        dl_azimuth_rad=dl_az,
        bh_tilt_rad=bh_tilt,
        bh_azimuth_rad=bh_az,
        area_size_m=current_scenario.area_size_m,
    )
    return updated, eval_count, cost_history


def optimize_two_phase(
    config: Config,
    baseline_scenario: Scenario,
    run_phase1: bool = True,
    cma_sigma0: float = 200.0,
    cma_maxiter: int = 100,
    cma_popsize: int | None = None,
    coord_descent_cycles: int = 2,
    local_method: str = "Powell",
    local_maxiter: int = 200,
    n_train_samples: int = 5,
    train_seed: int = 42,
    weights: dict[str, float] | None = None,
    penalty_lambdas: dict[str, float] | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> OptimizationResult:
    """Two-phase optimization: global positions (CMA-ES) then per-drone orientations (Powell).

    Phase 1 (optional): Optimize drone positions using CMA-ES while keeping
    default orientations. Uses existing ObjectiveFunction with stochastic
    user sampling.

    Phase 2: Coordinate descent over per-drone orientations. Each drone's
    4 orientation variables (dl_tilt, dl_azimuth, bh_tilt, bh_azimuth) are
    optimized independently via Powell, Gauss-Seidel style.

    Parameters
    ----------
    run_phase1 : bool
        If True, run CMA-ES position optimization first.
    cma_sigma0 : float
        Initial step size for CMA-ES.
    cma_maxiter : int
        Maximum CMA-ES iterations.
    cma_popsize : int or None
        CMA-ES population size. If None, uses CMA default.
    coord_descent_cycles : int
        Number of coordinate descent cycles over all drones.
    local_method : str
        Scipy minimize method for per-drone orientation optimization.
    local_maxiter : int
        Max iterations per drone in orientation optimization.
    n_train_samples : int
        Number of user distributions to average over per evaluation.
    train_seed : int
        Base seed for user distribution generation.
    weights, penalty_lambdas : dict or None
        Objective weights and penalty coefficients. Uses defaults if None.
    """
    t0 = time.time()
    n_drones = baseline_scenario.drone_positions.shape[0]

    if weights is None:
        weights = {"coverage": 1.0, "throughput": 0.1}
    if penalty_lambdas is None:
        penalty_lambdas = {"separation": 5.0}

    dl_antenna, bh_antenna, channel = _build_models(config)
    current_scenario = baseline_scenario
    full_history: list[float] = []
    total_evals = 0

    # ── Phase 1: Position optimization via CMA-ES ────────────────────
    if run_phase1:
        if verbose:
            print("  Phase 1: CMA-ES position optimization...")

        problem_pos = OptimizationProblem(
            n_drones=n_drones,
            optimize_positions=True,
            optimize_altitude=False,
            optimize_dl_orientation=False,
            optimize_bh_orientation=False,
        )
        obj_pos = ObjectiveFunction(
            config=config,
            problem=problem_pos,
            baseline_scenario=current_scenario,
            weights=weights,
            penalty_lambdas=penalty_lambdas,
            n_train_samples=n_train_samples,
            train_seed=train_seed,
        )
        x0_pos = flatten(current_scenario, problem_pos)
        bounds_pos = build_bounds(problem_pos, config.scenario.area_size_m)
        lb = np.array([b[0] for b in bounds_pos])
        ub = np.array([b[1] for b in bounds_pos])

        import cma

        cma_opts = {
            "bounds": [lb.tolist(), ub.tolist()],
            "maxiter": cma_maxiter,
            "seed": seed,
            "verbose": -9 if not verbose else 1,
        }
        if cma_popsize is not None:
            cma_opts["popsize"] = cma_popsize

        es = cma.CMAEvolutionStrategy(x0_pos.tolist(), cma_sigma0, cma_opts)
        es.optimize(obj_pos)

        best_x_pos = es.result.xbest
        current_scenario = unflatten(
            np.array(best_x_pos), current_scenario, problem_pos
        )
        total_evals += obj_pos.eval_count
        full_history.extend(obj_pos.history)

        if verbose:
            print(f"    Phase 1 done: {obj_pos.eval_count} evals, best cost = {obj_pos.best_cost:.4f}")

    # ── Phase 2: Per-drone orientation via coordinate descent ────────
    if verbose:
        print("  Phase 2: Per-drone orientation optimization (coordinate descent)...")

    prev_cycle_cost = np.inf
    convergence_threshold = 1e-4

    for cycle in range(coord_descent_cycles):
        # Generate user seeds for this cycle
        cycle_rng = np.random.default_rng(train_seed + cycle)
        user_seeds = [int(cycle_rng.integers(0, 100000)) for _ in range(n_train_samples)]

        cycle_evals = 0
        for drone_i in range(n_drones):
            current_scenario, n_evals, drone_hist = _optimize_single_drone_orientation(
                drone_idx=drone_i,
                current_scenario=current_scenario,
                config=config,
                dl_antenna=dl_antenna,
                bh_antenna=bh_antenna,
                channel=channel,
                user_seeds=user_seeds,
                weights=weights,
                penalty_lambdas=penalty_lambdas,
                method=local_method,
                maxiter=local_maxiter,
            )
            cycle_evals += n_evals
            full_history.extend(drone_hist)

        total_evals += cycle_evals

        # Evaluate cycle cost for convergence check
        cycle_cost = full_history[-1] if full_history else np.inf

        if verbose:
            print(f"    Cycle {cycle + 1}/{coord_descent_cycles}: {cycle_evals} evals, cost = {cycle_cost:.4f}")

        if abs(prev_cycle_cost - cycle_cost) < convergence_threshold:
            if verbose:
                print("    Converged early.")
            break
        prev_cycle_cost = cycle_cost

    # ── Finalize ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    final_metrics = evaluate_scenario(current_scenario, config, n_seeds=10)

    # Build a full-problem vector for the result
    problem_full = OptimizationProblem(
        n_drones=n_drones,
        optimize_positions=True,
        optimize_altitude=False,
        optimize_dl_orientation=True,
        optimize_bh_orientation=True,
    )

    best_cost = full_history[-1] if full_history else np.inf

    if verbose:
        print(f"  Two-phase done: {total_evals} total evals in {elapsed:.1f}s")
        print(f"    Final coverage: {final_metrics.coverage_fraction:.1%}")

    return OptimizationResult(
        best_scenario=current_scenario,
        best_cost=best_cost,
        best_vector=flatten(current_scenario, problem_full),
        history=full_history,
        n_evaluations=total_evals,
        method="two_phase",
        elapsed_seconds=elapsed,
        final_metrics=final_metrics,
    )


# ── Convenience: run full comparison ─────────────────────────────────


def run_comparison(
    config: Config,
    seed: int = 42,
    methods: list[str] | None = None,
    de_maxiter: int = 200,
    pso_maxiter: int = 200,
    pso_n_particles: int = 50,
    alt_outer_iters: int = 10,
    optimize_altitude: bool = False,
    weights: dict[str, float] | None = None,
    penalty_lambdas: dict[str, float] | None = None,
    verbose: bool = True,
) -> dict[str, OptimizationResult]:
    """Run all optimization methods from the same K-means baseline.

    Parameters
    ----------
    config : Config
    seed : int
    methods : list of method names, default ["baseline", "de", "alternating", "pso"]
    pso_n_particles : int
        Number of PSO particles (should be >= problem dimension).
    optimize_altitude : bool
        If True, include altitude as a decision variable.
    weights : dict or None
        Objective function weights. If None, uses defaults.
    penalty_lambdas : dict or None
        Penalty coefficients. If None, uses defaults.
    verbose : bool

    Returns
    -------
    dict mapping method name to OptimizationResult.
    """
    if methods is None:
        methods = ["baseline", "de", "alternating", "pso"]

    # Generate baseline scenario
    net = config.network
    sc_cfg = config.scenario
    baseline = create_scenario(
        n_drones=net.n_drones,
        placement="kmeans",
        altitude_m=net.altitude_m,
        dl_tilt_deg=net.dl_tilt_deg,
        bh_tilt_deg=net.bh_tilt_deg,
        n_clusters=sc_cfg.n_clusters,
        users_per_cluster_mean=sc_cfg.users_per_cluster_mean,
        cluster_spread_m=sc_cfg.cluster_spread_m,
        area_size_m=sc_cfg.area_size_m,
        min_separation_m=net.min_separation_m,
        seed=seed,
    )

    n_drones = baseline.drone_positions.shape[0]
    problem = OptimizationProblem(
        n_drones=n_drones, optimize_altitude=optimize_altitude,
    )
    bounds = build_bounds(problem, sc_cfg.area_size_m)

    obj_kw: dict = {}
    if weights is not None:
        obj_kw["weights"] = weights
    if penalty_lambdas is not None:
        obj_kw["penalty_lambdas"] = penalty_lambdas

    results: dict[str, OptimizationResult] = {}

    # Baseline (no optimization)
    if "baseline" in methods:
        if verbose:
            print("Evaluating baseline (K-means)...")
        t0 = time.time()
        final_metrics = evaluate_scenario(baseline, config, n_seeds=10, base_seed=seed)
        obj_tmp = ObjectiveFunction(
            config=config, problem=problem,
            baseline_scenario=baseline, **obj_kw,
        )
        baseline_cost = obj_tmp(flatten(baseline, problem))
        results["baseline"] = OptimizationResult(
            best_scenario=baseline,
            best_cost=baseline_cost,
            best_vector=flatten(baseline, problem),
            history=[baseline_cost],
            n_evaluations=1,
            method="baseline",
            elapsed_seconds=time.time() - t0,
            final_metrics=final_metrics,
        )
        if verbose:
            print(f"  Baseline coverage: {final_metrics.coverage_fraction:.1%}")

    # Differential Evolution
    if "de" in methods:
        if verbose:
            print(f"\nRunning Differential Evolution (maxiter={de_maxiter})...")
        obj_de = ObjectiveFunction(
            config=config, problem=problem,
            baseline_scenario=baseline, **obj_kw,
        )
        x0 = flatten(baseline, problem)
        results["de"] = optimize_de(
            obj_de, bounds, x0=x0,
            maxiter=de_maxiter, seed=seed, verbose=verbose,
        )
        if verbose:
            m = results["de"].final_metrics
            print(f"  DE coverage: {m.coverage_fraction:.1%} "
                  f"({results['de'].n_evaluations} evals, "
                  f"{results['de'].elapsed_seconds:.1f}s)")

    # Alternating Optimization
    if "alternating" in methods:
        if verbose:
            print(f"\nRunning Alternating Optimization (max {alt_outer_iters} outer iters)...")
        results["alternating"] = optimize_alternating(
            config, baseline,
            max_outer_iters=alt_outer_iters,
            optimize_altitude=optimize_altitude,
            weights=weights,
            penalty_lambdas=penalty_lambdas,
            seed=seed, verbose=verbose,
        )
        if verbose:
            m = results["alternating"].final_metrics
            print(f"  Alternating coverage: {m.coverage_fraction:.1%} "
                  f"({results['alternating'].n_evaluations} evals, "
                  f"{results['alternating'].elapsed_seconds:.1f}s)")

    # PSO
    if "pso" in methods:
        if verbose:
            print(f"\nRunning PSO (n_particles={pso_n_particles}, "
                  f"max_iters={pso_maxiter})...")
        obj_pso = ObjectiveFunction(
            config=config, problem=problem,
            baseline_scenario=baseline, **obj_kw,
        )
        x0 = flatten(baseline, problem)
        results["pso"] = optimize_pso(
            obj_pso, bounds, x0=x0,
            n_particles=pso_n_particles,
            max_iters=pso_maxiter, seed=seed, verbose=verbose,
        )
        if verbose:
            m = results["pso"].final_metrics
            print(f"  PSO coverage: {m.coverage_fraction:.1%} "
                  f"({results['pso'].n_evaluations} evals, "
                  f"{results['pso'].elapsed_seconds:.1f}s)")

    return results


# ── Isotropic vs Directional Comparison ──────────────────────────────


def _build_isotropic_models(config: Config):
    """Build isotropic antenna models for comparison.

    Uses true isotropic antennas with 0 dBi gain to model what happens when
    the optimizer ignores antenna directionality entirely (assumes omnidirectional).
    """
    ch_cfg = config.channel
    # True isotropic: 0 dBi in all directions
    dl_antenna = IsotropicAntenna(g_dbi=0.0)
    bh_antenna = IsotropicAntenna(g_dbi=0.0)
    env = ENVIRONMENTS[ch_cfg.environment]
    channel = ChannelModel(env=env, f_c=ch_cfg.carrier_freq_hz)
    return dl_antenna, bh_antenna, channel


@dataclass
class IsotropicVsDirectionalResult:
    """Result of comparing isotropic-optimized vs directional-optimized deployments."""

    # Optimized assuming isotropic, evaluated with directional
    isotropic_opt_scenario: Scenario
    isotropic_opt_metrics_iso_eval: SimulationMetrics  # Evaluated with isotropic model
    isotropic_opt_metrics_dir_eval: SimulationMetrics  # Evaluated with directional model

    # Optimized assuming directional, evaluated with directional
    directional_opt_scenario: Scenario
    directional_opt_metrics: SimulationMetrics

    # Baseline (no optimization)
    baseline_scenario: Scenario
    baseline_metrics: SimulationMetrics

    # Optimization stats
    n_evaluations: int
    elapsed_seconds: float


def _evaluate_with_model(
    scenario: Scenario,
    config: Config,
    dl_antenna,
    bh_antenna,
    channel: ChannelModel,
    n_seeds: int = 10,
    base_seed: int = 42,
) -> SimulationMetrics:
    """Evaluate a scenario with specific antenna/channel models."""
    net = config.network
    sc_cfg = config.scenario
    p_tx_dl_w = dbm_to_watts(net.p_tx_dl_dbm)
    p_tx_bh_w = dbm_to_watts(net.p_tx_bh_dbm)

    metrics_list: list[SimulationMetrics] = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        user_pos = generate_users(
            n_clusters=sc_cfg.n_clusters,
            users_per_cluster_mean=sc_cfg.users_per_cluster_mean,
            users_per_cluster_std=sc_cfg.users_per_cluster_std,
            cluster_spread_m=sc_cfg.cluster_spread_m,
            area_size_m=sc_cfg.area_size_m,
            rng=rng,
        )
        eval_sc = Scenario(
            user_positions=user_pos,
            drone_positions=scenario.drone_positions,
            dl_tilt_rad=scenario.dl_tilt_rad,
            dl_azimuth_rad=scenario.dl_azimuth_rad,
            bh_tilt_rad=scenario.bh_tilt_rad,
            bh_azimuth_rad=scenario.bh_azimuth_rad,
            area_size_m=scenario.area_size_m,
        )

        power_mat = downlink_power_matrix(
            drone_positions=eval_sc.drone_positions,
            dl_tilt_rad=eval_sc.dl_tilt_rad,
            dl_azimuth_rad=eval_sc.dl_azimuth_rad,
            user_positions=eval_sc.user_positions,
            dl_antenna=dl_antenna,
            channel=channel,
            p_tx_dl_w=p_tx_dl_w,
        )
        association = nearest_drone_association(
            eval_sc.drone_positions, eval_sc.user_positions
        )
        sinr_result = compute_sinr(
            power_mat, association,
            noise_psd_dbm_hz=net.noise_psd_dbm_hz,
            bandwidth_hz=net.bandwidth_hz,
        )
        bh_interf = backhaul_interference_matrix(
            drone_positions=eval_sc.drone_positions,
            bh_tilt_rad=eval_sc.bh_tilt_rad,
            bh_azimuth_rad=eval_sc.bh_azimuth_rad,
            bh_antenna=bh_antenna,
            channel=channel,
            p_tx_bh_w=p_tx_bh_w,
        )
        metrics_list.append(compute_metrics(
            sinr_result,
            sinr_threshold_db=net.sinr_threshold_db,
            backhaul_interference_matrix=bh_interf,
        ))

    return _average_metrics(metrics_list)


class ObjectiveFunctionIsotropic(ObjectiveFunction):
    """Objective function using isotropic antenna models for optimization."""

    def __post_init__(self):
        self._dl_antenna, self._bh_antenna, self._channel = _build_isotropic_models(
            self.config
        )


def compare_isotropic_vs_directional(
    config: Config,
    seed: int = 42,
    max_evals: int = 50000,
    verbose: bool = True,
) -> IsotropicVsDirectionalResult:
    """Compare deployment optimized with isotropic vs directional antenna models.

    This is the key experiment showing the value of geometry-aware optimization:
    1. Optimize assuming isotropic antennas (ignoring directionality)
    2. Optimize assuming directional antennas (geometry-aware)
    3. Evaluate both deployments with the TRUE directional model

    Parameters
    ----------
    config : Config
    seed : int
    max_evals : int
        Maximum function evaluations budget (same for both methods).
    verbose : bool
    """
    t0 = time.time()
    net = config.network
    sc_cfg = config.scenario

    weights = {"coverage": 1.0, "throughput": 0.5}
    penalty_lambdas = {"separation": 5.0}

    baseline = create_scenario(
        n_drones=net.n_drones,
        placement="kmeans",
        altitude_m=net.altitude_m,
        dl_tilt_deg=net.dl_tilt_deg,
        bh_tilt_deg=net.bh_tilt_deg,
        n_clusters=sc_cfg.n_clusters,
        users_per_cluster_mean=sc_cfg.users_per_cluster_mean,
        cluster_spread_m=sc_cfg.cluster_spread_m,
        area_size_m=sc_cfg.area_size_m,
        min_separation_m=net.min_separation_m,
        seed=seed,
    )

    n_drones = baseline.drone_positions.shape[0]
    problem = OptimizationProblem(n_drones=n_drones, optimize_altitude=False)
    bounds = build_bounds(problem, sc_cfg.area_size_m)

    # Compute iterations to match evaluation budget
    # DE evaluations ≈ popsize * maxiter (roughly)
    popsize = 15
    de_maxiter = max(10, max_evals // (popsize * problem.dimension))

    # Build models
    dl_dir, bh_dir, ch_dir = _build_models(config)
    dl_iso, bh_iso, ch_iso = _build_isotropic_models(config)

    # ── 1. Optimize with ISOTROPIC model ──
    if verbose:
        print(f"Optimizing with ISOTROPIC antenna model (budget={max_evals} evals)...")

    obj_iso = ObjectiveFunctionIsotropic(
        config=config, problem=problem, baseline_scenario=baseline,
        weights=weights, penalty_lambdas=penalty_lambdas,
    )
    x0 = flatten(baseline, problem)
    result_iso = differential_evolution(
        obj_iso, bounds=bounds,
        maxiter=de_maxiter, popsize=popsize,
        seed=seed, x0=x0, polish=False, disp=verbose,
    )
    iso_scenario = unflatten(result_iso.x, baseline, problem)
    iso_evals = obj_iso.eval_count

    if verbose:
        print(f"  Isotropic optimization: {iso_evals} evals")

    # ── 2. Optimize with DIRECTIONAL model ──
    if verbose:
        print(f"Optimizing with DIRECTIONAL antenna model (budget={max_evals} evals)...")

    obj_dir = ObjectiveFunction(
        config=config, problem=problem, baseline_scenario=baseline,
        weights=weights, penalty_lambdas=penalty_lambdas,
    )
    result_dir = differential_evolution(
        obj_dir, bounds=bounds,
        maxiter=de_maxiter, popsize=popsize,
        seed=seed, x0=x0, polish=False, disp=verbose,
    )
    dir_scenario = unflatten(result_dir.x, baseline, problem)
    dir_evals = obj_dir.eval_count

    if verbose:
        print(f"  Directional optimization: {dir_evals} evals")

    # ── 3. Evaluate all scenarios with DIRECTIONAL (true) model ──
    if verbose:
        print("Evaluating all deployments with directional (true) model...")

    baseline_metrics = _evaluate_with_model(
        baseline, config, dl_dir, bh_dir, ch_dir, n_seeds=10, base_seed=seed
    )
    iso_metrics_iso = _evaluate_with_model(
        iso_scenario, config, dl_iso, bh_iso, ch_iso, n_seeds=10, base_seed=seed
    )
    iso_metrics_dir = _evaluate_with_model(
        iso_scenario, config, dl_dir, bh_dir, ch_dir, n_seeds=10, base_seed=seed
    )
    dir_metrics = _evaluate_with_model(
        dir_scenario, config, dl_dir, bh_dir, ch_dir, n_seeds=10, base_seed=seed
    )

    elapsed = time.time() - t0

    if verbose:
        print(f"\n{'Method':<25} {'Coverage':>10} {'Throughput':>12} {'Mean SINR':>10}")
        print("-" * 60)
        print(f"{'Baseline (K-means)':<25} {baseline_metrics.coverage_fraction:>9.1%} "
              f"{baseline_metrics.sum_throughput_mbps:>10.0f} Mb "
              f"{baseline_metrics.mean_sinr_db:>9.1f} dB")
        print(f"{'Isotropic-opt (iso eval)':<25} {iso_metrics_iso.coverage_fraction:>9.1%} "
              f"{iso_metrics_iso.sum_throughput_mbps:>10.0f} Mb "
              f"{iso_metrics_iso.mean_sinr_db:>9.1f} dB")
        print(f"{'Isotropic-opt (dir eval)':<25} {iso_metrics_dir.coverage_fraction:>9.1%} "
              f"{iso_metrics_dir.sum_throughput_mbps:>10.0f} Mb "
              f"{iso_metrics_dir.mean_sinr_db:>9.1f} dB")
        print(f"{'Directional-opt':<25} {dir_metrics.coverage_fraction:>9.1%} "
              f"{dir_metrics.sum_throughput_mbps:>10.0f} Mb "
              f"{dir_metrics.mean_sinr_db:>9.1f} dB")

    return IsotropicVsDirectionalResult(
        isotropic_opt_scenario=iso_scenario,
        isotropic_opt_metrics_iso_eval=iso_metrics_iso,
        isotropic_opt_metrics_dir_eval=iso_metrics_dir,
        directional_opt_scenario=dir_scenario,
        directional_opt_metrics=dir_metrics,
        baseline_scenario=baseline,
        baseline_metrics=baseline_metrics,
        n_evaluations=iso_evals + dir_evals,
        elapsed_seconds=elapsed,
    )


# ── Drone Count Sweep ────────────────────────────────────────────────


@dataclass
class DroneCountSweepResult:
    """Result of sweeping over different drone counts with multi-seed averaging."""

    drone_counts: list[int]
    n_seeds: int

    # Mean coverage fractions across seeds
    baseline_coverage_mean: list[float]
    baseline_coverage_std: list[float]
    directional_coverage_mean: list[float]
    directional_coverage_std: list[float]
    isotropic_coverage_mean: list[float]
    isotropic_coverage_std: list[float]

    # Mean throughput across seeds
    baseline_throughput_mean: list[float]
    directional_throughput_mean: list[float]
    isotropic_throughput_mean: list[float]

    # Gap statistics
    gap_mean: list[float]  # directional - isotropic
    gap_std: list[float]

    elapsed_seconds: float


def run_drone_count_sweep(
    config: Config,
    drone_counts: list[int] | None = None,
    base_seed: int = 42,
    n_seeds: int = 3,
    max_evals_per_run: int = 30000,
    verbose: bool = True,
) -> DroneCountSweepResult:
    """Run isotropic vs directional comparison for different drone counts.

    Runs multiple seeds per drone count and reports mean ± std.

    Parameters
    ----------
    config : Config
        Base configuration (n_drones will be overridden).
    drone_counts : list of int
        Drone counts to test. Default: [5, 10, 15, 20].
    base_seed : int
        Base random seed. Seeds used will be base_seed, base_seed+1, ...
    n_seeds : int
        Number of seeds to average over for statistical robustness.
    max_evals_per_run : int
        Evaluation budget per optimization run.
    verbose : bool
    """
    if drone_counts is None:
        drone_counts = [5, 10, 15, 20]

    t0 = time.time()

    # Results storage: [drone_count][seed]
    baseline_cov_all: list[list[float]] = []
    dir_cov_all: list[list[float]] = []
    iso_cov_all: list[list[float]] = []
    baseline_tput_all: list[list[float]] = []
    dir_tput_all: list[list[float]] = []
    iso_tput_all: list[list[float]] = []

    for n_drones in drone_counts:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running sweep for {n_drones} drones ({n_seeds} seeds)")
            print('='*60)

        # Create modified config with different drone count
        modified_config = Config(
            antenna=config.antenna,
            channel=config.channel,
            network=config.network.__class__(
                n_drones=n_drones,
                altitude_m=config.network.altitude_m,
                p_tx_dl_dbm=config.network.p_tx_dl_dbm,
                p_tx_bh_dbm=config.network.p_tx_bh_dbm,
                bandwidth_hz=config.network.bandwidth_hz,
                noise_psd_dbm_hz=config.network.noise_psd_dbm_hz,
                sinr_threshold_db=config.network.sinr_threshold_db,
                min_separation_m=config.network.min_separation_m,
                dl_tilt_deg=config.network.dl_tilt_deg,
                bh_tilt_deg=config.network.bh_tilt_deg,
            ),
            scenario=config.scenario,
            simulation=config.simulation,
        )

        baseline_seeds: list[float] = []
        dir_seeds: list[float] = []
        iso_seeds: list[float] = []
        baseline_tput_seeds: list[float] = []
        dir_tput_seeds: list[float] = []
        iso_tput_seeds: list[float] = []

        for s in range(n_seeds):
            seed = base_seed + s * 100  # Spread seeds apart
            if verbose:
                print(f"  Seed {s+1}/{n_seeds} (seed={seed})...")

            result = compare_isotropic_vs_directional(
                modified_config,
                seed=seed,
                max_evals=max_evals_per_run,
                verbose=False,  # Suppress per-seed output
            )

            baseline_seeds.append(result.baseline_metrics.coverage_fraction)
            dir_seeds.append(result.directional_opt_metrics.coverage_fraction)
            iso_seeds.append(result.isotropic_opt_metrics_dir_eval.coverage_fraction)
            baseline_tput_seeds.append(result.baseline_metrics.sum_throughput_mbps)
            dir_tput_seeds.append(result.directional_opt_metrics.sum_throughput_mbps)
            iso_tput_seeds.append(result.isotropic_opt_metrics_dir_eval.sum_throughput_mbps)

        baseline_cov_all.append(baseline_seeds)
        dir_cov_all.append(dir_seeds)
        iso_cov_all.append(iso_seeds)
        baseline_tput_all.append(baseline_tput_seeds)
        dir_tput_all.append(dir_tput_seeds)
        iso_tput_all.append(iso_tput_seeds)

        if verbose:
            mean_gap = np.mean(dir_seeds) - np.mean(iso_seeds)
            print(f"  -> Mean gap: {mean_gap:+.1%}")

    elapsed = time.time() - t0

    # Compute statistics
    baseline_cov_mean = [float(np.mean(x)) for x in baseline_cov_all]
    baseline_cov_std = [float(np.std(x)) for x in baseline_cov_all]
    dir_cov_mean = [float(np.mean(x)) for x in dir_cov_all]
    dir_cov_std = [float(np.std(x)) for x in dir_cov_all]
    iso_cov_mean = [float(np.mean(x)) for x in iso_cov_all]
    iso_cov_std = [float(np.std(x)) for x in iso_cov_all]

    baseline_tput_mean = [float(np.mean(x)) for x in baseline_tput_all]
    dir_tput_mean = [float(np.mean(x)) for x in dir_tput_all]
    iso_tput_mean = [float(np.mean(x)) for x in iso_tput_all]

    # Gap statistics
    gap_per_seed = [
        [d - i for d, i in zip(dir_cov_all[j], iso_cov_all[j])]
        for j in range(len(drone_counts))
    ]
    gap_mean = [float(np.mean(x)) for x in gap_per_seed]
    gap_std = [float(np.std(x)) for x in gap_per_seed]

    if verbose:
        print(f"\n{'='*70}")
        print("SWEEP SUMMARY (mean ± std across seeds)")
        print('='*70)
        print(f"{'Drones':>8} {'Baseline':>12} {'Isotropic':>14} {'Directional':>14} {'Gap':>14}")
        print("-" * 70)
        for i, n in enumerate(drone_counts):
            print(f"{n:>8} {baseline_cov_mean[i]:>9.1%}±{baseline_cov_std[i]:.1%} "
                  f"{iso_cov_mean[i]:>9.1%}±{iso_cov_std[i]:.1%} "
                  f"{dir_cov_mean[i]:>9.1%}±{dir_cov_std[i]:.1%} "
                  f"{gap_mean[i]:>+8.1%}±{gap_std[i]:.1%}")

    return DroneCountSweepResult(
        drone_counts=list(drone_counts),
        n_seeds=n_seeds,
        baseline_coverage_mean=baseline_cov_mean,
        baseline_coverage_std=baseline_cov_std,
        directional_coverage_mean=dir_cov_mean,
        directional_coverage_std=dir_cov_std,
        isotropic_coverage_mean=iso_cov_mean,
        isotropic_coverage_std=iso_cov_std,
        baseline_throughput_mean=baseline_tput_mean,
        directional_throughput_mean=dir_tput_mean,
        isotropic_throughput_mean=iso_tput_mean,
        gap_mean=gap_mean,
        gap_std=gap_std,
        elapsed_seconds=elapsed,
    )
