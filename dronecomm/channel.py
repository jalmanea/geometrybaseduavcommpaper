"""Air-to-ground channel model with probabilistic LoS/NLoS.

Implements the Al-Hourani / ITU-R model:
    P_LoS(theta) = 1 / (1 + a * exp(-b * (theta_deg - a)))
    PL(d, theta) = FSPL(d, f_c) + P_LoS * eta_LoS + P_NLoS * eta_NLoS   [dB]

For air-to-air links (drone-to-drone), pure free-space path loss is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvironmentParams:
    """Propagation environment parameters for LoS probability model."""

    name: str
    a: float  # Sigmoid shape parameter
    b: float  # Sigmoid rate parameter
    eta_los_db: float  # Excess path loss for LoS [dB]
    eta_nlos_db: float  # Excess path loss for NLoS [dB]


# ── Environment presets (Al-Hourani / ITU-R) ────────────────────────

SUBURBAN = EnvironmentParams("suburban", a=4.88, b=0.43, eta_los_db=1.0, eta_nlos_db=20.0)
URBAN = EnvironmentParams("urban", a=9.61, b=0.16, eta_los_db=1.0, eta_nlos_db=20.0)
DENSE_URBAN = EnvironmentParams("dense_urban", a=12.08, b=0.11, eta_los_db=1.6, eta_nlos_db=23.0)

ENVIRONMENTS = {"suburban": SUBURBAN, "urban": URBAN, "dense_urban": DENSE_URBAN}

# Speed of light [m/s]
C = 3.0e8


@dataclass(frozen=True)
class ChannelModel:
    """Air-to-ground channel with probabilistic LoS/NLoS path loss.

    Parameters
    ----------
    env : EnvironmentParams
        Propagation environment.
    f_c : float
        Carrier frequency in Hz. Default: 2.0 GHz.
    """

    env: EnvironmentParams = URBAN
    f_c: float = 2.0e9

    def los_probability(self, theta_elev_rad: np.ndarray) -> np.ndarray:
        """Probability of Line-of-Sight.

        Parameters
        ----------
        theta_elev_rad : np.ndarray
            Elevation angle(s) in radians (0 = horizontal, pi/2 = overhead).

        Returns
        -------
        np.ndarray
            P(LoS) in [0, 1].
        """
        theta_deg = np.rad2deg(np.asarray(theta_elev_rad, dtype=np.float64))
        return 1.0 / (1.0 + self.env.a * np.exp(-self.env.b * (theta_deg - self.env.a)))

    def fspl_db(self, distance: np.ndarray) -> np.ndarray:
        """Free-space path loss in dB.

        Parameters
        ----------
        distance : np.ndarray
            3D distance(s) in meters.

        Returns
        -------
        np.ndarray
            FSPL in dB.
        """
        distance = np.maximum(np.asarray(distance, dtype=np.float64), 1e-6)
        return 20.0 * np.log10(4.0 * np.pi * distance * self.f_c / C)

    def path_loss_db(
        self, distance: np.ndarray, theta_elev_rad: np.ndarray
    ) -> np.ndarray:
        """Average path loss combining LoS and NLoS contributions.

        Parameters
        ----------
        distance : np.ndarray
            3D distance(s) in meters.
        theta_elev_rad : np.ndarray
            Elevation angle(s) in radians.

        Returns
        -------
        np.ndarray
            Path loss in dB.
        """
        p_los = self.los_probability(theta_elev_rad)
        fspl = self.fspl_db(distance)
        return fspl + p_los * self.env.eta_los_db + (1.0 - p_los) * self.env.eta_nlos_db

    def path_loss_linear(
        self, distance: np.ndarray, theta_elev_rad: np.ndarray
    ) -> np.ndarray:
        """Path loss as a linear attenuation factor (< 1).

        Parameters
        ----------
        distance : np.ndarray
            3D distance(s) in meters.
        theta_elev_rad : np.ndarray
            Elevation angle(s) in radians.

        Returns
        -------
        np.ndarray
            Linear attenuation factor.
        """
        return 10.0 ** (-self.path_loss_db(distance, theta_elev_rad) / 10.0)

    def a2a_path_loss_db(self, distance: np.ndarray) -> np.ndarray:
        """Air-to-air path loss (free-space only, for drone-to-drone links).

        Parameters
        ----------
        distance : np.ndarray
            3D distance(s) between drones in meters.

        Returns
        -------
        np.ndarray
            Path loss in dB.
        """
        return self.fspl_db(distance)

    def a2a_path_loss_linear(self, distance: np.ndarray) -> np.ndarray:
        """Air-to-air path loss as a linear attenuation factor."""
        return 10.0 ** (-self.a2a_path_loss_db(distance) / 10.0)
