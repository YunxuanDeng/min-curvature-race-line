"""Speed profile computation via the forward-backward pass.

Given a path with known curvature at each arc-length station and a vehicle
model, this module computes the maximum sustainable speed at each station
and integrates it to produce a lap time.

Algorithm
---------
The achievable speed limit at each station is the minimum of three constraints:

1. **Cornering speed limit** (from the friction circle):
   ``v_corner = sqrt(max_grip / |kappa|)``.

2. **Braking constraint** (backward pass): to reach the cornering speed
   ``v_corner`` at station ``s_i``, the vehicle must have been at most
   ``sqrt(v_corner^2 + 2 * a_brake * (s_j - s_i))`` at an earlier station
   ``s_j``, accounting for the friction-circle coupling at each step.

3. **Acceleration constraint** (forward pass): leaving the cornering
   speed ``v_corner`` at station ``s_i``, the vehicle can reach at most
   ``sqrt(v_corner^2 + 2 * a_engine * (s_j - s_i))`` at a later station
   ``s_j``, again respecting the friction-circle coupling.

The combined speed profile is the minimum of the forward and backward passes,
capped by ``max_speed``. Lap time is then the integral of ``ds / v(s)`` around
the loop.

For closed tracks, the forward and backward passes are run twice so the
wrap-around is resolved without needing a seed. For open tracks a single
pass in each direction suffices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from raceline.vehicle import VehicleModel


@dataclass
class SpeedProfile:
    """The achievable speed at each station along a path, and the lap time.

    Attributes:
        speeds: Array of shape ``(N,)`` of achievable speeds (m/s) at each
            arc-length station.
        arc_lengths: Array of shape ``(N,)`` of cumulative arc-length
            positions (m), same indexing as ``speeds``.
        lap_time: Total time to traverse the path (s). For closed tracks,
            this is the lap time; for open tracks, the traversal time.
        closed: Whether the path is a closed loop.
    """

    speeds: NDArray[np.float64]
    arc_lengths: NDArray[np.float64]
    lap_time: float
    closed: bool

    @property
    def min_speed(self) -> float:
        """Minimum speed anywhere on the path (m/s).

        Typically occurs at the slowest corner.
        """
        return float(self.speeds.min())

    @property
    def max_speed_achieved(self) -> float:
        """Maximum speed achieved anywhere on the path (m/s)."""
        return float(self.speeds.max())


def _cornering_speed_limits(
    curvature: NDArray[np.float64], vehicle: VehicleModel
) -> NDArray[np.float64]:
    """Return the cornering speed limit at each station."""
    # Vectorized via a loop since max_speed_at_curvature is scalar-valued.
    # N is typically <= 10000, so this is fast enough.
    return np.array(
        [vehicle.max_speed_at_curvature(k) for k in curvature],
        dtype=np.float64,
    )


def _segment_lengths(
    arc_lengths: NDArray[np.float64], closed: bool
) -> NDArray[np.float64]:
    """Return the length of each segment (from station i to i+1).

    For closed tracks, the last segment wraps from the final station back
    to the first. Returns an array of shape ``(N,)`` where entry ``i`` is
    the length of the segment starting at station ``i``.
    """
    n = len(arc_lengths)
    segs = np.empty(n, dtype=np.float64)
    segs[:-1] = np.diff(arc_lengths)
    if closed:
        # Wrap segment: assume uniform spacing (which resample() guarantees).
        # The closing segment length equals the typical inter-station spacing.
        segs[-1] = segs[:-1].mean()
    else:
        # For open tracks there's no segment after the last point.
        segs[-1] = 0.0
    return segs


def _forward_pass(
    speeds: NDArray[np.float64],
    curvature: NDArray[np.float64],
    segs: NDArray[np.float64],
    vehicle: VehicleModel,
    closed: bool,
) -> NDArray[np.float64]:
    """Enforce forward-in-time acceleration constraint.

    Walks forward through stations, limiting the speed at each to what's
    reachable from the previous station under the engine's acceleration
    given the current lateral-grip usage. For closed loops, walks twice
    so the wrap-around fully resolves.
    """
    n = len(speeds)
    result = speeds.copy()
    n_passes = 2 if closed else 1
    for _ in range(n_passes):
        for i in range(n):
            prev = (i - 1) % n if closed else i - 1
            if prev < 0:
                continue
            v_prev = result[prev]
            a_lat = v_prev**2 * abs(curvature[prev])
            a_long = vehicle.max_longitudinal_acceleration(v_prev, a_lat)
            v_reachable = float(np.sqrt(v_prev**2 + 2 * a_long * segs[prev]))
            if v_reachable < result[i]:
                result[i] = v_reachable
    return result


def _backward_pass(
    speeds: NDArray[np.float64],
    curvature: NDArray[np.float64],
    segs: NDArray[np.float64],
    vehicle: VehicleModel,
    closed: bool,
) -> NDArray[np.float64]:
    """Enforce backward-in-time braking constraint.

    Walks backward through stations, limiting the speed at each to what
    allows the vehicle to slow down to the next station's speed limit
    under the brake's deceleration given the current lateral-grip usage.
    For closed loops, walks twice so the wrap-around fully resolves.
    """
    n = len(speeds)
    result = speeds.copy()
    n_passes = 2 if closed else 1
    for _ in range(n_passes):
        for i in range(n - 1, -1, -1):
            nxt = (i + 1) % n if closed else i + 1
            if nxt >= n:
                continue
            v_next = result[nxt]
            a_lat = v_next**2 * abs(curvature[nxt])
            a_brake = vehicle.max_longitudinal_deceleration(v_next, a_lat)
            v_reachable = float(np.sqrt(v_next**2 + 2 * a_brake * segs[i]))
            if v_reachable < result[i]:
                result[i] = v_reachable
    return result


def _integrate_lap_time(
    speeds: NDArray[np.float64],
    segs: NDArray[np.float64],
    closed: bool,
) -> float:
    """Integrate dt = ds / v using the trapezoidal rule between stations."""
    n = len(speeds)
    total = 0.0
    last = n if closed else n - 1
    for i in range(last):
        j = (i + 1) % n
        v_avg = 0.5 * (speeds[i] + speeds[j])
        total += segs[i] / v_avg
    return float(total)


def compute_speed_profile(
    curvature: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    vehicle: VehicleModel,
    *,
    closed: bool = True,
) -> SpeedProfile:
    """Compute the achievable speed profile along a path.

    Uses the forward-backward pass algorithm: compute the cornering speed
    limit at each station, then propagate acceleration and braking
    constraints through the path, returning the minimum of the two passes
    at each station.

    Args:
        curvature: Array of shape ``(N,)`` of signed curvature (1/m) at
            each station. Sign is ignored (only magnitude affects grip).
        arc_lengths: Array of shape ``(N,)`` of cumulative arc-length
            positions (m), monotonically increasing. Should be uniformly
            spaced for best numerical behavior.
        vehicle: A ``VehicleModel`` implementation providing the
            physics-of-motion methods.
        closed: Whether the path is a closed loop (wraparound).
            Defaults to True.

    Returns:
        A ``SpeedProfile`` containing speeds at each station and total
        lap time.

    Raises:
        ValueError: If inputs have mismatched shapes or fewer than 2
            stations.
    """
    if curvature.shape != arc_lengths.shape:
        raise ValueError(
            f"curvature and arc_lengths must have the same shape, "
            f"got {curvature.shape} and {arc_lengths.shape}"
        )
    if curvature.ndim != 1:
        raise ValueError(f"curvature must be 1D, got shape {curvature.shape}")
    if len(curvature) < 2:
        raise ValueError(f"need at least 2 stations, got {len(curvature)}")

    # Step 1: cornering limits.
    v_corner = _cornering_speed_limits(curvature, vehicle)

    # Step 2: segment lengths.
    segs = _segment_lengths(arc_lengths, closed)

    # Step 3: forward pass (acceleration).
    v_after_forward = _forward_pass(v_corner, curvature, segs, vehicle, closed)

    # Step 4: backward pass (braking).
    v_final = _backward_pass(v_after_forward, curvature, segs, vehicle, closed)

    # Step 5: lap time.
    lap_time = _integrate_lap_time(v_final, segs, closed)

    return SpeedProfile(
        speeds=v_final,
        arc_lengths=arc_lengths,
        lap_time=lap_time,
        closed=closed,
    )
