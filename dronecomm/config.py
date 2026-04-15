"""Dataclass-based configuration with YAML loading support."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AntennaConfig:
    """Antenna parameters for downlink and backhaul."""

    dl_g_max_dbi: float = 18.0
    dl_beamwidth_deg: float = 60.0
    dl_sla_db: float = 20.0
    bh_g_max_dbi: float = 12.0
    bh_beamwidth_deg: float = 30.0
    bh_sla_db: float = 25.0


@dataclass(frozen=True)
class ChannelConfig:
    """Channel model parameters."""

    environment: str = "urban"
    carrier_freq_hz: float = 2.0e9


@dataclass(frozen=True)
class NetworkConfig:
    """Network and power parameters."""

    n_drones: int = 15
    altitude_m: float = 150.0
    p_tx_dl_dbm: float = 30.0  # 1 W
    p_tx_bh_dbm: float = 27.0  # 0.5 W
    bandwidth_hz: float = 10.0e6  # 10 MHz
    noise_psd_dbm_hz: float = -174.0
    sinr_threshold_db: float = 3.0
    min_separation_m: float = 100.0
    dl_tilt_deg: float = 0.0
    bh_tilt_deg: float = 90.0


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario generation parameters."""

    area_size_m: float = 2000.0
    n_clusters: int = 5
    users_per_cluster_mean: int = 40
    users_per_cluster_std: int = 10
    cluster_spread_m: float = 100.0


@dataclass(frozen=True)
class SimulationConfig:
    """Monte Carlo simulation parameters."""

    n_trials: int = 100
    seed: int = 42
    placement_strategy: str = "kmeans"


@dataclass(frozen=True)
class Config:
    """Top-level configuration combining all sub-configs."""

    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            antenna=AntennaConfig(**data.get("antenna", {})),
            channel=ChannelConfig(**data.get("channel", {})),
            network=NetworkConfig(**data.get("network", {})),
            scenario=ScenarioConfig(**data.get("scenario", {})),
            simulation=SimulationConfig(**data.get("simulation", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
