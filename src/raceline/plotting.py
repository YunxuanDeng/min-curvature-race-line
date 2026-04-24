"""Visualization helpers for tracks, racing lines, and lap telemetry.

All matplotlib code is isolated in this module. The core library
(track, vehicle, optimizer, speed_profile, simulator) has no
plotting dependencies and can be used in environments without
matplotlib.

Every function follows the ``ax=None`` pattern: if no axes object
is provided, a new figure and axes are created; otherwise the
function draws onto the provided axes and returns them. This lets
callers compose multi-panel figures easily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from raceline import PointMassVehicle

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from raceline.optimizer import RacingLine
    from raceline.simulator import LapResult
    from raceline.track import Track


def plot_track(
    track: Track,
    *,
    ax: Axes | None = None,
    show_centerline: bool = True,
) -> Axes:
    """Plot the track boundaries and (optionally) the centerline.

    Args:
        track: A ``Track`` instance.
        ax: Matplotlib axes to draw on. Created if ``None``.
        show_centerline: If ``True``, draw the centerline as a
            dashed grey line.

    Returns:
        The axes with the track drawn.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Compute boundary points from centerline + normals
    cl = track.centerline
    dx = np.gradient(cl[:, 0])
    dy = np.gradient(cl[:, 1])
    lengths = np.sqrt(dx**2 + dy**2)
    lengths = np.where(lengths == 0, 1.0, lengths)
    nx, ny = -dy / lengths, dx / lengths

    left_x = cl[:, 0] + track.width_left * nx
    left_y = cl[:, 1] + track.width_left * ny
    right_x = cl[:, 0] - track.width_right * nx
    right_y = cl[:, 1] - track.width_right * ny

    if track.closed:
        left_x = np.append(left_x, left_x[0])
        left_y = np.append(left_y, left_y[0])
        right_x = np.append(right_x, right_x[0])
        right_y = np.append(right_y, right_y[0])

    ax.plot(left_x, left_y, "k-", linewidth=1.2, label="Track limit")
    ax.plot(right_x, right_y, "k-", linewidth=1.2)

    if show_centerline:
        cx = cl[:, 0]
        cy = cl[:, 1]
        if track.closed:
            cx = np.append(cx, cx[0])
            cy = np.append(cy, cy[0])
        ax.plot(
            cx,
            cy,
            "--",
            color="0.6",
            linewidth=0.8,
            label="Centerline",
        )

    # Start/finish marker
    ax.plot(
        cl[0, 0],
        cl[0, 1],
        "s",
        color="red",
        markersize=7,
        zorder=10,
        label="Start/Finish",
    )

    # Direction arrow right after the start/finish line
    n_pts = len(cl)
    arrow_idx = 20
    step = max(1, n_pts // 100)
    arrow_tail = cl[arrow_idx]
    arrow_head = cl[arrow_idx + step]
    ax.annotate(
        "",
        xy=arrow_head,
        xytext=arrow_tail,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "green",
            "lw": 2.0,
            "mutation_scale": 20,
        },
        zorder=10,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Track Layout")
    return ax


def plot_racing_line(
    track: Track,
    line: RacingLine,
    speeds: NDArray[np.float64] | None = None,
    *,
    ax: Axes | None = None,
    cmap: str = "plasma",
    show_centerline: bool = True,
) -> Axes:
    """Overlay the racing line on the track, color-coded by speed.

    Args:
        track: A ``Track`` instance (for drawing boundaries).
        line: A ``RacingLine`` to overlay.
        speeds: Optional speed array for color-coding. If ``None``,
            the line is drawn in a single color.
        ax: Matplotlib axes to draw on. Created if ``None``.
        cmap: Colormap name for speed coloring.
        show_centerline: Whether to draw the centerline.

    Returns:
        The axes with the track and racing line drawn.
    """
    from matplotlib.collections import LineCollection

    ax = plot_track(
        track,
        ax=ax,
        show_centerline=show_centerline,
    )

    pts = line.points

    if speeds is not None:
        # Build line segments for LineCollection: each segment
        # connects consecutive points. Color is set by the
        # average speed of the two endpoints.
        n = len(pts)
        idx = np.arange(n)
        if track.closed:
            idx_next = np.roll(idx, -1)
        else:
            idx_next = np.concatenate([idx[1:], idx[-1:]])
        segments = np.stack([pts[idx], pts[idx_next]], axis=1)
        sp_kmh = speeds * 3.6
        seg_colors = 0.5 * (sp_kmh[idx] + sp_kmh[idx_next])

        lc = LineCollection(
            segments,
            cmap=cmap,
            linewidths=1.2,
            zorder=5,
        )
        lc.set_array(seg_colors)
        ax.add_collection(lc)
        cb: Colorbar = plt.colorbar(lc, ax=ax, pad=0.02)
        cb.set_label("Speed (km/h)")
    else:
        close_pts = pts
        if track.closed:
            close_pts = np.vstack([pts, pts[0:1]])
        ax.plot(
            close_pts[:, 0],
            close_pts[:, 1],
            "-",
            color="blue",
            linewidth=1.5,
            label="Racing line",
            zorder=5,
        )

    ax.set_title("Racing Line")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_speed_profile(
    result: LapResult,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot speed vs. arc length along the racing line.

    Args:
        result: A ``LapResult`` from ``simulate_lap``.
        ax: Matplotlib axes to draw on. Created if ``None``.

    Returns:
        The axes with the speed profile drawn.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 4))

    s = result.speed_profile.arc_lengths
    v = result.speed_profile.speeds * 3.6  # m/s → km/h

    ax.plot(s, v, "-", color="steelblue", linewidth=1.2)
    ax.fill_between(s, 0, v, alpha=0.15, color="steelblue")

    ax.set_xlabel("Arc length (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(
        f"Hypothetical Best Speed Profile | Lap time: {result.lap_time:.2f} s"
    )
    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    return ax


def plot_gg_diagram(
    result: LapResult,
    *,
    ax: Axes | None = None,
    show_friction_circle: bool = True,
    grip_limit: float | None = None,
) -> Axes:
    """Plot the g-g diagram (lateral vs. longitudinal acceleration).

    Shows how fully the vehicle uses its available grip. Points near
    the friction-circle boundary indicate the vehicle is at the
    limit; points near the center indicate grip in reserve.

    Args:
        result: A ``LapResult`` from ``simulate_lap``.
        ax: Matplotlib axes to draw on. Created if ``None``.
        show_friction_circle: If ``True``, overlay the grip limit
            circle for reference.
        grip_limit: The vehicle's max grip (m/s^2) for drawing the
            friction circle. If ``None``, estimated from the data.

    Returns:
        The axes with the g-g diagram drawn.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    g = 9.81  # m/s^2 per G
    a_lat = result.lateral_g / g
    # Negate so braking (deceleration) is positive/upward
    a_long = -result.longitudinal_g / g

    ax.scatter(
        a_lat,
        a_long,
        s=6,
        alpha=0.15,
        c="steelblue",
        zorder=5,
    )

    if show_friction_circle:
        if grip_limit is None:
            grip_limit = float(
                np.sqrt(result.lateral_g**2 + result.longitudinal_g**2).max()
                * 1.05
            )
        grip_g = grip_limit / g
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            grip_g * np.cos(theta),
            grip_g * np.sin(theta),
            "--",
            color="red",
            linewidth=1.0,
            label=f"Grip limit ({grip_g:.1f} G)",
        )

    ax.set_xlabel("Lateral acceleration (G)")
    ax.set_ylabel(
        "Longitudinal acceleration (G)\n+ braking / \u2212 acceleration"
    )
    ax.set_title("Simulated g-g Diagram")
    ax.set_aspect("equal")
    ax.axhline(0, color="0.8", linewidth=0.5)
    ax.axvline(0, color="0.8", linewidth=0.5)
    if show_friction_circle:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_lap_summary(
    track: Track,
    result: LapResult,
    vehicle: PointMassVehicle | None = None,
) -> Figure:
    """Create a summary figure for a lap simulation.

    Layout: large racing line panel on the left, g-g diagram on the
    upper right, speed profile across the bottom.

    Args:
        track: The ``Track`` used for simulation
        result: A ``LapResult`` from ``simulate_lap``
        vehicle: A ``PointMassVehicle`` from ``point_mass``

    Returns:
        The matplotlib ``Figure`` containing all panels.
    """
    fig = plt.figure(figsize=(16, 10))

    # Left panel (tall): racing line.
    ax_track = fig.add_axes((0.05, 0.32, 0.55, 0.60))
    plot_racing_line(
        track,
        result.racing_line,
        speeds=result.speed_profile.speeds,
        ax=ax_track,
        show_centerline=False,
    )

    # Upper right: g-g diagram.
    ax_gg = fig.add_axes((0.68, 0.38, 0.28, 0.50))
    plot_gg_diagram(result, ax=ax_gg)

    # Bottom (full width): speed profile.
    ax_speed = fig.add_axes((0.05, 0.06, 0.90, 0.22))
    plot_speed_profile(result, ax=ax_speed)

    fig.suptitle(
        f"Lap Summary  |  {result.lap_time:.2f} s",
        fontsize=14,
        fontweight="bold",
    )

    # Vehicle parameters text box.
    if vehicle is not None:
        info = (
            f"Mass: {vehicle.mass:.0f} kg\n"
            f"Grip: {vehicle.max_grip:.1f} m/s² "
            f"({vehicle.max_grip / 9.81:.2f} G)\n"
            f"Engine: {vehicle.max_engine_acceleration:.1f}"
            f" m/s²\n"
            f"Brakes: {vehicle.max_brake_deceleration:.1f}"
            f" m/s²\n"
            f"Top speed: {vehicle.max_speed:.1f} m/s "
            f"({vehicle.max_speed * 3.6:.0f} km/h)"
        )
        fig.text(
            0.68,
            0.32,
            info,
            fontsize=9,
            fontfamily="monospace",
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "wheat",
                "alpha": 0.8,
            },
        )

    fig.suptitle(
        f"Lap Summary  |  {result.lap_time:.2f} s",
        fontsize=14,
        fontweight="bold",
    )
    return fig
