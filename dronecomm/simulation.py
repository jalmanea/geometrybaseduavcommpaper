"""Monte Carlo simulation runner.

Orchestrates: scenario generation -> interference computation -> SINR -> metrics,
repeated across multiple trials with random user distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .antenna import ParametricAntenna
from .channel import ChannelModel, ENVIRONMENTS
from .config import Config
from .interference import downlink_power_matrix, backhaul_interference_matrix
from .metrics import SimulationMetrics, compute_metrics
from .scenario import Scenario, create_scenario
from .sinr import SINRResult, compute_sinr, nearest_drone_association, dbm_to_watts


@dataclass
class TrialResult:
    """Result of a single Monte Carlo trial."""

    trial_id: int
    metrics: SimulationMetrics
    scenario: Scenario
    association: np.ndarray
    sinr_result: SINRResult


@dataclass
class MonteCarloResult:
    """Aggregated results across all Monte Carlo trials."""

    trial_results: list[TrialResult] = field(default_factory=list)
    config: Config = field(default_factory=Config)

    @property
    def n_trials(self) -> int:
        return len(self.trial_results)

    def mean_metric(self, attr: str) -> float:
        """Mean of a scalar metric across trials."""
        values = [getattr(tr.metrics, attr) for tr in self.trial_results]
        return float(np.mean(values))

    def std_metric(self, attr: str) -> float:
        """Std dev of a scalar metric across trials."""
        values = [getattr(tr.metrics, attr) for tr in self.trial_results]
        return float(np.std(values))

    def summary(self) -> dict[str, dict[str, float]]:
        """Summary statistics for key metrics."""
        keys = [
            "sum_throughput_mbps",
            "mean_sinr_db",
            "min_sinr_db",
            "sinr_5th_percentile_db",
            "coverage_fraction",
        ]
        return {
            k: {"mean": self.mean_metric(k), "std": self.std_metric(k)}
            for k in keys
        }


def run_trial(config: Config, trial_id: int, seed: int) -> TrialResult:
    """Run a single Monte Carlo trial.

    Parameters
    ----------
    config : Config
    trial_id : int
    seed : int
        RNG seed for this trial.

    Returns
    -------
    TrialResult
    """
    net = config.network
    sc = config.scenario
    ant = config.antenna
    ch = config.channel

    # Build models
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
    env = ENVIRONMENTS.get(ch.environment)
    if env is None:
        raise ValueError(f"Unknown environment: {ch.environment}")
    channel = ChannelModel(env=env, f_c=ch.carrier_freq_hz)

    # Generate scenario
    scenario = create_scenario(
        n_drones=net.n_drones,
        placement=config.simulation.placement_strategy,
        altitude_m=net.altitude_m,
        dl_tilt_deg=net.dl_tilt_deg,
        bh_tilt_deg=net.bh_tilt_deg,
        n_clusters=sc.n_clusters,
        users_per_cluster_mean=sc.users_per_cluster_mean,
        cluster_spread_m=sc.cluster_spread_m,
        area_size_m=sc.area_size_m,
        min_separation_m=net.min_separation_m,
        seed=seed,
    )

    # Compute downlink power matrix
    p_tx_dl_w = dbm_to_watts(net.p_tx_dl_dbm)
    power_mat = downlink_power_matrix(
        drone_positions=scenario.drone_positions,
        dl_tilt_rad=scenario.dl_tilt_rad,
        dl_azimuth_rad=scenario.dl_azimuth_rad,
        user_positions=scenario.user_positions,
        dl_antenna=dl_antenna,
        channel=channel,
        p_tx_dl_w=p_tx_dl_w,
    )

    # User-drone association
    association = nearest_drone_association(
        scenario.drone_positions, scenario.user_positions
    )

    # SINR
    sinr_result = compute_sinr(
        power_matrix=power_mat,
        association=association,
        noise_psd_dbm_hz=net.noise_psd_dbm_hz,
        bandwidth_hz=net.bandwidth_hz,
    )

    # Backhaul interference
    p_tx_bh_w = dbm_to_watts(net.p_tx_bh_dbm)
    bh_interf = backhaul_interference_matrix(
        drone_positions=scenario.drone_positions,
        bh_tilt_rad=scenario.bh_tilt_rad,
        bh_azimuth_rad=scenario.bh_azimuth_rad,
        bh_antenna=bh_antenna,
        channel=channel,
        p_tx_bh_w=p_tx_bh_w,
    )

    # Metrics
    metrics = compute_metrics(
        sinr_result=sinr_result,
        sinr_threshold_db=net.sinr_threshold_db,
        backhaul_interference_matrix=bh_interf,
    )

    return TrialResult(
        trial_id=trial_id,
        metrics=metrics,
        scenario=scenario,
        association=association,
        sinr_result=sinr_result,
    )


def run_monte_carlo(config: Config, verbose: bool = True) -> MonteCarloResult:
    """Run Monte Carlo simulation.

    Parameters
    ----------
    config : Config
    verbose : bool
        Print progress.

    Returns
    -------
    MonteCarloResult
    """
    n_trials = config.simulation.n_trials
    base_seed = config.simulation.seed

    results = []
    for i in range(n_trials):
        seed = base_seed + i
        trial = run_trial(config, trial_id=i, seed=seed)
        results.append(trial)
        if verbose and (i + 1) % 10 == 0:
            print(f"  Trial {i + 1}/{n_trials} done")

    mc_result = MonteCarloResult(trial_results=results, config=config)

    if verbose:
        print("\n=== Monte Carlo Summary ===")
        for key, stats in mc_result.summary().items():
            print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    return mc_result
