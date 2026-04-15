"""Visualization: 3D scenario plots, SINR heat maps, antenna patterns, CDFs."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .antenna import AntennaModel, ParametricAntenna
from .channel import ChannelModel
from .interference import downlink_power_matrix
from .sinr import compute_sinr, nearest_drone_association


def plot_scenario_3d(
    drone_positions: np.ndarray,
    user_positions: np.ndarray,
    association: np.ndarray | None = None,
    title: str = "Disaster Scenario",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """3D scatter of drones and ground users.

    Parameters
    ----------
    drone_positions : (N, 3)
    user_positions : (M, 3)
    association : (M,) or None, serving drone indices
    title : str
    ax : matplotlib 3D axes or None
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Users
    ax.scatter(
        user_positions[:, 0],
        user_positions[:, 1],
        user_positions[:, 2],
        c="steelblue", s=8, alpha=0.5, label="Users",
    )
    # Drones
    ax.scatter(
        drone_positions[:, 0],
        drone_positions[:, 1],
        drone_positions[:, 2],
        c="black", s=80, marker="^", label="Drones",
    )
    # Association lines
    if association is not None:
        for u_idx in range(len(user_positions)):
            d_idx = association[u_idx]
            ax.plot(
                [user_positions[u_idx, 0], drone_positions[d_idx, 0]],
                [user_positions[u_idx, 1], drone_positions[d_idx, 1]],
                [user_positions[u_idx, 2], drone_positions[d_idx, 2]],
                c="gray", alpha=0.1, linewidth=0.3,
            )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return fig


def _drone_colors(n: int) -> list:
    return [plt.cm.tab10(d % 10) for d in range(n)]


def plot_drone_placement_3d(
    drone_positions: np.ndarray,
    user_positions: np.ndarray,
    association: np.ndarray,
    title: str = "",
    figsize: tuple[int, int] = (9, 7),
) -> plt.Figure:
    """3-D view of drone placement and user assignment.

    Parameters
    ----------
    drone_positions : (N, 3)
    user_positions  : (M, 3)
    association     : (M,)  serving-drone index per user
    title           : str
    figsize         : tuple
    """
    N = drone_positions.shape[0]
    colors = _drone_colors(N)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for d_idx in range(N):
        mask = association == d_idx
        if not mask.any():
            continue
        ax.scatter(
            user_positions[mask, 0], user_positions[mask, 1], user_positions[mask, 2],
            color=colors[d_idx], s=10, alpha=0.65, zorder=2,
        )
        for u_idx in np.where(mask)[0]:
            ax.plot(
                [user_positions[u_idx, 0], drone_positions[d_idx, 0]],
                [user_positions[u_idx, 1], drone_positions[d_idx, 1]],
                [user_positions[u_idx, 2], drone_positions[d_idx, 2]],
                color=colors[d_idx], alpha=0.08, linewidth=0.4,
            )

    ax.scatter(
        drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2],
        c="black", s=120, marker="^", zorder=5, label="Drones",
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_drone_placement_2d(
    drone_positions: np.ndarray,
    user_positions: np.ndarray,
    association: np.ndarray,
    dl_beamwidth_deg: float = 60.0,
    title: str = "",
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Top-down view of drone placement with nadir beam footprint circles.

    Parameters
    ----------
    drone_positions  : (N, 3)
    user_positions   : (M, 3)
    association      : (M,)  serving-drone index per user
    dl_beamwidth_deg : float  full DL beamwidth — used to draw footprint circles
    title            : str
    figsize          : tuple
    """
    N = drone_positions.shape[0]
    colors = _drone_colors(N)

    fig, ax = plt.subplots(figsize=figsize)

    for d_idx in range(N):
        mask = association == d_idx
        if not mask.any():
            continue
        ax.scatter(
            user_positions[mask, 0], user_positions[mask, 1],
            color=colors[d_idx], s=12, alpha=0.65, zorder=2,
        )

    half_bw = np.deg2rad(dl_beamwidth_deg / 2.0)
    for d_idx in range(N):
        r = drone_positions[d_idx, 2] * np.tan(half_bw)
        ax.add_patch(plt.Circle(
            (drone_positions[d_idx, 0], drone_positions[d_idx, 1]),
            r,
            color=colors[d_idx], fill=False, linewidth=0.9, linestyle="--", alpha=0.45,
        ))

    ax.scatter(
        drone_positions[:, 0], drone_positions[:, 1],
        c="black", s=120, marker="^", zorder=5,
    )

    for d_idx in range(N):
        mask = association == d_idx
        n = int(mask.sum())
        z = drone_positions[d_idx, 2]
        spread = float(np.std(user_positions[mask, :2] -
                              drone_positions[d_idx, :2], axis=0).mean()) if n > 0 else 0.0
        ax.annotate(
            f"D{d_idx}  z={z:.0f}m\nσ={spread:.0f}m  ({n}u)",
            (drone_positions[d_idx, 0], drone_positions[d_idx, 1]),
            fontsize=6, ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points",
        )

    xy_max = max(user_positions[:, 0].max(), user_positions[:, 1].max(),
                 drone_positions[:, 0].max(), drone_positions[:, 1].max())
    pad = xy_max * 0.05
    ax.set_xlim(-pad, xy_max + pad)
    ax.set_ylim(-pad, xy_max + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Top-down view\n(dashed = nadir beam footprint at analytic altitude)")
    ax.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_drone_placement(
    drone_positions: np.ndarray,
    user_positions: np.ndarray,
    association: np.ndarray,
    dl_beamwidth_deg: float = 60.0,
    title: str = "",
    figsize: tuple[int, int] = (16, 7),
) -> tuple[plt.Figure, plt.Figure]:
    """Return (fig_3d, fig_2d) for drone placement and user assignment.

    Parameters
    ----------
    drone_positions  : (N, 3)
    user_positions   : (M, 3)
    association      : (M,)  serving-drone index per user
    dl_beamwidth_deg : float
    title            : str   shared title prefix
    figsize          : tuple ignored (kept for back-compat); each figure uses its own default
    """
    fig_3d = plot_drone_placement_3d(
        drone_positions, user_positions, association,
        title=title,
    )
    fig_2d = plot_drone_placement_2d(
        drone_positions, user_positions, association,
        dl_beamwidth_deg=dl_beamwidth_deg,
        title=title,
    )
    return fig_3d, fig_2d


def plot_pca_orientations(
    drone_positions: np.ndarray,
    user_positions: np.ndarray,
    association: np.ndarray,
    title: str = "",
    figsize: tuple[int, int] = (9, 9),
) -> plt.Figure:
    """Top-down view showing per-drone PCA orientation.

    For each drone's user cluster:
    - Scatter of users (coloured by drone)
    - 2-σ covariance ellipse centred on the drone's XY position
    - Arrow in the PCA major-axis direction, with length = σ_major
      (= the ground displacement of the PCA-tilt beam centre at that altitude)

    Parameters
    ----------
    drone_positions : (N, 3)
    user_positions  : (M, 3)
    association     : (M,)  K-means labels or nearest-drone indices
    title           : str
    figsize         : tuple
    """
    from matplotlib.patches import Ellipse

    N = drone_positions.shape[0]
    colors = _drone_colors(N)

    fig, ax = plt.subplots(figsize=figsize)

    for d_idx in range(N):
        mask = association == d_idx
        if not mask.any():
            continue

        xy = user_positions[mask, :2]
        cx, cy = drone_positions[d_idx, :2]
        z = drone_positions[d_idx, 2]
        color = colors[d_idx]

        ax.scatter(xy[:, 0], xy[:, 1], color=color, s=12, alpha=0.5, zorder=2)

        if len(xy) >= 3:
            xy_rel = xy - np.array([cx, cy])
            cov = np.cov(xy_rel.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending
            # major axis = last column
            major_vec = eigenvectors[:, -1]
            sigma_major = float(np.sqrt(max(eigenvalues[-1], 1e-3)))
            sigma_minor = float(np.sqrt(max(eigenvalues[-2], 1e-3)))

            angle_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
            ellipse = Ellipse(
                (cx, cy),
                width=4 * sigma_major,   # 2-sigma half-width → full axis = 4σ
                height=4 * sigma_minor,
                angle=angle_deg,
                color=color, fill=False, linewidth=1.4, linestyle="-", alpha=0.7, zorder=3,
            )
            ax.add_patch(ellipse)

            # Arrow: direction = major eigenvector, length = σ_major
            # (ground displacement of PCA-tilt beam = z·tan(arctan(σ/z)) = σ)
            tilt_deg = float(np.degrees(np.arctan2(sigma_major, max(z, 1e-3))))
            dx = major_vec[0] * sigma_major
            dy = major_vec[1] * sigma_major
            ax.annotate(
                "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0),
                zorder=4,
            )
            ax.annotate(
                f"D{d_idx}  z={z:.0f}m\ntilt={tilt_deg:.1f}°",
                (cx, cy), fontsize=6, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points",
            )
        else:
            ax.annotate(
                f"D{d_idx}  z={z:.0f}m\n(nadir)",
                (cx, cy), fontsize=6, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points",
            )

        ax.scatter(cx, cy, c="black", s=120, marker="^", zorder=5)

    xy_max = max(user_positions[:, 0].max(), user_positions[:, 1].max(),
                 drone_positions[:, 0].max(), drone_positions[:, 1].max())
    pad = xy_max * 0.05
    ax.set_xlim(-pad, xy_max + pad)
    ax.set_ylim(-pad, xy_max + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(
        "PCA tilt orientations\n"
        "ellipse = 2σ cluster spread  |  arrow = beam displacement at PCA tilt"
    )
    ax.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_antenna_pattern_polar(
    antenna: AntennaModel,
    title: str = "Antenna Radiation Pattern",
) -> plt.Figure:
    """Polar plot of antenna gain vs off-boresight angle."""
    theta = np.linspace(0, np.pi, 500)
    gain_dbi = antenna.gain_dbi(theta)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    ax.plot(theta, gain_dbi, linewidth=2, color="darkblue")
    # Mirror for symmetric pattern
    ax.plot(-theta, gain_dbi, linewidth=2, color="darkblue")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.set_title(title, pad=20)
    ax.set_xlabel("Gain [dBi]")
    return fig


def plot_sinr_heatmap(
    drone_positions: np.ndarray,
    dl_tilt_rad: np.ndarray,
    dl_azimuth_rad: np.ndarray,
    dl_antenna: AntennaModel,
    channel: ChannelModel,
    p_tx_dl_w: float,
    area_size_m: float,
    noise_psd_dbm_hz: float = -174.0,
    bandwidth_hz: float = 10.0e6,
    grid_resolution: int = 100,
    title: str = "SINR Heat Map [dB]",
) -> plt.Figure:
    """SINR heat map on the ground plane.

    Computes SINR on a regular grid and plots as a colormesh.
    """
    xs = np.linspace(0, area_size_m, grid_resolution)
    ys = np.linspace(0, area_size_m, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_positions = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    power_mat = downlink_power_matrix(
        drone_positions=drone_positions,
        dl_tilt_rad=dl_tilt_rad,
        dl_azimuth_rad=dl_azimuth_rad,
        user_positions=grid_positions,
        dl_antenna=dl_antenna,
        channel=channel,
        p_tx_dl_w=p_tx_dl_w,
    )

    association = nearest_drone_association(drone_positions, grid_positions)
    sinr_result = compute_sinr(power_mat, association, noise_psd_dbm_hz, bandwidth_hz)
    sinr_map = sinr_result.sinr_db.reshape(grid_resolution, grid_resolution)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.pcolormesh(xx, yy, sinr_map, cmap="RdYlGn", shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SINR [dB]")

    # Mark drone positions
    ax.scatter(
        drone_positions[:, 0], drone_positions[:, 1],
        c="black", s=60, marker="^", zorder=5, label="Drones",
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    return fig


def plot_sinr_cdf(
    sinr_db: np.ndarray,
    sinr_threshold_db: float = 3.0,
    title: str = "SINR CDF",
) -> plt.Figure:
    """CDF of SINR for all users."""
    fig, ax = plt.subplots(figsize=(8, 5))

    vals = np.sort(sinr_db)
    cdf = np.arange(1, len(vals) + 1) / len(vals)
    ax.step(vals, cdf, where="post", color="steelblue", linewidth=2, label="Users")

    ax.axvline(sinr_threshold_db, color="steelblue", linestyle="--", alpha=0.5,
               label=f"Threshold ({sinr_threshold_db} dB)")

    ax.set_xlabel("SINR [dB]")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig


def plot_interference_graph(
    drone_positions: np.ndarray,
    bh_interference: np.ndarray,
    title: str = "Inter-Drone Backhaul Interference",
) -> plt.Figure:
    """Top-down view of drones with edges colored by mutual interference."""
    N = drone_positions.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))

    # Edges (upper triangle)
    triu_i, triu_j = np.triu_indices(N, k=1)
    interf_vals = bh_interference[triu_i, triu_j]
    interf_db = 10.0 * np.log10(np.maximum(interf_vals, 1e-30)) + 30.0

    if len(interf_db) > 0:
        norm = Normalize(vmin=np.min(interf_db), vmax=np.max(interf_db))
        cmap = plt.cm.YlOrRd

        for idx in range(len(triu_i)):
            i, j = triu_i[idx], triu_j[idx]
            color = cmap(norm(interf_db[idx]))
            ax.plot(
                [drone_positions[i, 0], drone_positions[j, 0]],
                [drone_positions[i, 1], drone_positions[j, 1]],
                color=color, linewidth=1.5, alpha=0.7,
            )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Interference [dBm]")

    ax.scatter(
        drone_positions[:, 0], drone_positions[:, 1],
        c="black", s=80, marker="^", zorder=5,
    )
    for i in range(N):
        ax.annotate(str(i), (drone_positions[i, 0], drone_positions[i, 1]),
                    fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    return fig
