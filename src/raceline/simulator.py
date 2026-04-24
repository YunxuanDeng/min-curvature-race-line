"""End-to-end lap simulation.

Ties together a ``Track``, the racing-line optimizer, and a
``VehicleModel`` to produce a ``LapResult`` containing timing,
speed trace, and the racing line used.

This module is a thin composition layer: all physics lives in
``speed_profile.py``, all geometry in ``track.py``, and all
optimization in ``optimizer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from raceline.optimizer import RacingLine, optimize_line
from raceline.speed_profile import SpeedProfile, compute_speed_profile

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from raceline.track import Track
    from raceline.vehicle import VehicleModel


@dataclass
class LapResult:
    """Complete result of a lap simulation.

    Attributes:
        lap_time: Total lap time in seconds.
        speed_profile: The full speed profile along the path.
        racing_line: The racing line used (optimized or centerline).
        lateral_g: Lateral acceleration at each station (m/s^2).
        longitudinal_g: Longitudinal acceleration at each station
            (m/s^2), computed from speed differences.
    """

    lap_time: float
    speed_profile: SpeedProfile
    racing_line: RacingLine
    lateral_g: NDArray[np.float64]
    longitudinal_g: NDArray[np.float64]


def _compute_lateral_g(
    speeds: NDArray[np.float64],
    curvature: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute signed lateral acceleration: a_lat = v^2 * kappa.

    The sign preserves the turn direction: positive curvature (left
    turn) gives positive lateral g, negative curvature (right turn)
    gives negative lateral g. This is needed for the g-g diagram to
    show scatter in both lateral directions.
    """
    return speeds**2 * curvature


def _compute_longitudinal_g(
    speeds: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    closed: bool,
) -> NDArray[np.float64]:
    """Compute longitudinal acceleration from speed differences.

    Uses forward differences: the acceleration at station ``i`` is
    derived from the speed change between station ``i`` and ``i+1``.
    This pairs each acceleration value with the curvature at the
    same station, preserving the friction-circle relationship.

    Sign convention: positive = accelerating, negative = braking.
    """
    n = len(speeds)
    ds = float(arc_lengths[1] - arc_lengths[0]) if n > 1 else 1.0

    if closed:
        v_next = np.roll(speeds, -1)
        a_long: NDArray[np.float64] = (
            (v_next**2 - speeds**2) / (2.0 * ds)
        ).astype(np.float64)
    else:
        a_long = np.zeros(n, dtype=np.float64)
        if n > 1:
            a_long[:-1] = (speeds[1:] ** 2 - speeds[:-1] ** 2) / (2.0 * ds)
            a_long[-1] = a_long[-2]

    return a_long


def _centerline_as_racing_line(track: Track) -> RacingLine:
    """Wrap the track centerline as a RacingLine with zero offsets."""
    return RacingLine(
        points=track.centerline.copy(),
        offsets=np.zeros(len(track), dtype=np.float64),
        arc_lengths=track.arc_lengths.copy(),
        closed=track.closed,
    )


def simulate_lap(
    track: Track,
    vehicle: VehicleModel,
    *,
    optimize: bool = True,
    length_weight: float = 0.0001,
) -> LapResult:
    """Simulate a complete lap and return timing and telemetry.

    This is the main entry point for end-to-end simulation. It:

    1. Optionally optimizes the racing line (or uses the centerline).
    2. Computes the speed profile via the forward-backward pass.
    3. Derives lateral and longitudinal g-forces from the result.

    Args:
        track: A ``Track`` instance (should be uniformly resampled).
        vehicle: A ``VehicleModel`` providing physics constraints.
        optimize: If ``True`` (default), run the minimum-curvature
            optimizer to find the best line. If ``False``, simulate
            on the centerline.
        length_weight: Weight for the path-length regularizer in the
            optimizer. Only used when ``optimize=True``.

    Returns:
        A ``LapResult`` with lap time, speed profile, racing line,
        and g-force traces.
    """
    if optimize:
        line = optimize_line(track, length_weight=length_weight)
    else:
        line = _centerline_as_racing_line(track)

    profile = compute_speed_profile(
        line.curvature,
        line.arc_lengths,
        vehicle,
        closed=track.closed,
    )

    lat_g = _compute_lateral_g(profile.speeds, line.curvature)
    long_g = _compute_longitudinal_g(
        profile.speeds, line.arc_lengths, track.closed
    )

    # Clamp longitudinal g to the friction circle. The speed profile
    # was computed respecting the friction circle at each step, but
    # post-hoc finite differences can slightly exceed it at
    # corner-entry/exit transitions due to discretization.
    # Infer the grip limit from the maximum lateral g observed
    # (which is set by the vehicle's cornering speed limit).
    grip = float(np.abs(lat_g).max()) * 1.001
    lat_mag = np.abs(lat_g)
    long_budget = np.sqrt(np.maximum(0.0, grip**2 - lat_mag**2))
    long_g = np.clip(long_g, -long_budget, long_budget)

    return LapResult(
        lap_time=profile.lap_time,
        speed_profile=profile,
        racing_line=line,
        lateral_g=lat_g,
        longitudinal_g=long_g,
    )
