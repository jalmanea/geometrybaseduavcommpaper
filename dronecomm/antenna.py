"""Parametric antenna radiation model.

Implements a 3GPP-inspired single-element pattern:
    G(theta) = G_max - min(12 * (theta / theta_3dB)^2, SLA)   [dBi]

where theta is the off-boresight angle, theta_3dB is the full -3 dB beamwidth,
and SLA is the sidelobe level attenuation in dB.

The model is azimuthally symmetric (depends only on the off-boresight angle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class AntennaModel(Protocol):
    """Interface for any antenna radiation pattern model."""

    def gain_dbi(self, theta: np.ndarray) -> np.ndarray:
        """Gain in dBi for off-boresight angle(s) theta in radians."""
        ...

    def gain_linear(self, theta: np.ndarray) -> np.ndarray:
        """Gain as a linear power ratio for off-boresight angle(s) theta."""
        ...

    @property
    def g_max_dbi(self) -> float:
        """Maximum (boresight) gain in dBi."""
        ...

    @property
    def beamwidth_rad(self) -> float:
        """Full -3 dB beamwidth in radians."""
        ...


@dataclass(frozen=True)
class ParametricAntenna:
    """Simple parametric antenna: parabolic main-lobe rolloff + sidelobe floor.

    Based on 3GPP TR 38.901 Table 7.3-1 single-element vertical pattern.
    The gain drops by 3 dB at theta_3dB / 2 from boresight.

    Parameters
    ----------
    g_max_dbi : float
        Boresight (peak) gain in dBi.
    beamwidth_deg : float
        Full -3 dB beamwidth in degrees.
    sla_db : float
        Sidelobe level attenuation in dB (positive value).
    """

    g_max_dbi: float = 18.0
    beamwidth_deg: float = 60.0
    sla_db: float = 20.0

    @property
    def beamwidth_rad(self) -> float:
        return np.deg2rad(self.beamwidth_deg)

    def gain_dbi(self, theta: np.ndarray) -> np.ndarray:
        """Compute antenna gain in dBi.

        Parameters
        ----------
        theta : np.ndarray
            Off-boresight angle(s) in radians. Any shape.

        Returns
        -------
        np.ndarray
            Gain in dBi. Same shape as theta.
        """
        theta = np.asarray(theta, dtype=np.float64)
        ratio = theta / self.beamwidth_rad
        attenuation = np.minimum(12.0 * ratio**2, self.sla_db)
        return self.g_max_dbi - attenuation

    def gain_linear(self, theta: np.ndarray) -> np.ndarray:
        """Compute antenna gain as a linear power ratio.

        Parameters
        ----------
        theta : np.ndarray
            Off-boresight angle(s) in radians.

        Returns
        -------
        np.ndarray
            Gain as linear ratio (not in dB).
        """
        return 10.0 ** (self.gain_dbi(theta) / 10.0)


@dataclass(frozen=True)
class IsotropicAntenna:
    """Isotropic (omnidirectional) antenna with constant gain in all directions.

    Used as a baseline to demonstrate the impact of ignoring directional
    antenna patterns when optimizing drone deployment.

    Parameters
    ----------
    g_dbi : float
        Constant gain in dBi (default 0 dBi = unity gain).
    """

    g_dbi: float = 0.0

    @property
    def g_max_dbi(self) -> float:
        return self.g_dbi

    @property
    def beamwidth_rad(self) -> float:
        return np.pi  # Full hemisphere

    def gain_dbi(self, theta: np.ndarray) -> np.ndarray:
        """Return constant gain regardless of angle."""
        theta = np.asarray(theta, dtype=np.float64)
        return np.full_like(theta, self.g_dbi)

    def gain_linear(self, theta: np.ndarray) -> np.ndarray:
        """Return constant linear gain regardless of angle."""
        theta = np.asarray(theta, dtype=np.float64)
        return np.full_like(theta, 10.0 ** (self.g_dbi / 10.0))


# ── Default antenna presets ──────────────────────────────────────────

DOWNLINK_ANTENNA = ParametricAntenna(g_max_dbi=18.0, beamwidth_deg=60.0, sla_db=20.0)
BACKHAUL_ANTENNA = ParametricAntenna(g_max_dbi=12.0, beamwidth_deg=30.0, sla_db=25.0)
ISOTROPIC_ANTENNA = IsotropicAntenna(g_dbi=0.0)
