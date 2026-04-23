"""Minimum-curvature racing line optimization.

Formulates the racing line as a quadratic program (QP): at each discretization
station along the centerline, a lateral offset variable determines how far the
racing line deviates from the centerline. The objective minimizes the sum of
squared discrete curvatures (with an optional path-length regularizer), subject
to the constraint that each point stays within the track boundaries.

The QP is convex and has a unique solution, solved via ``cvxpy``.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from raceline.track import Track


@dataclass
class RacingLine:
    """An optimized racing line on a track.

    Attributes:
        points: Array of shape ``(N, 2)`` giving (x, y) coordinates of the
        racing line in meters.
        offsets: Array of shape ``(N,)`` of lateral offsets from the centerline
        (m). Positive = left of centerline.
        arc_lengths: Cumulative arc length along the racing line (m).
        closed: Whether the line forms a closed loop.
    """

    points: NDArray[np.float64]
    offsets: NDArray[np.float64]
    arc_lengths: NDArray[np.float64]
    closed: bool

    @property
    def curvature(self) -> NDArray[np.float64]:
        """Signed curvature at each point of the racing line (1/m).

        Computed from the (x, y) points using the same parametric formula as
        ``Track.curvature``.
        """
        x = self.points[:, 0]
        y = self.points[:, 1]
        if self.closed:
            x_w = np.concatenate([x[-1:], x, x[:1]])
            y_w = np.concatenate([y[-1:], y, y[:1]])
            s_w = np.concatenate(
                [
                    [
                        self.arc_lengths[0]
                        - np.linalg.norm(self.points[0] - self.points[-1])
                    ],
                    self.arc_lengths,
                    [
                        self.arc_lengths[-1]
                        + np.linalg.norm(self.points[0] - self.points[-1])
                    ],
                ]
            )
            xp = np.gradient(x_w, s_w)[1:-1]
            yp = np.gradient(y_w, s_w)[1:-1]
            xpp = np.gradient(xp, self.arc_lengths)
            ypp = np.gradient(yp, self.arc_lengths)
        else:
            xp = np.gradient(x, self.arc_lengths)
            yp = np.gradient(y, self.arc_lengths)
            xpp = np.gradient(xp, self.arc_lengths)
            ypp = np.gradient(yp, self.arc_lengths)

        num = xp * ypp - yp * xpp
        den = (xp**2 + yp**2) ** 1.5
        den = np.where(den == 0, 1.0, den)
        result: NDArray[np.float64] = (num / den).astype(np.float64)
        return result


def _compute_normals(
    centerline: NDArray[np.float64], closed: bool
) -> NDArray[np.float64]:
    """Compute unit normal vectors at each centerline point.

    The normal is the 90-degree counter-clock-wise (CCW) rotation of the unit
    tangent:
    ``n = (-ty, tx)`` where ``(tx, ty)`` is the tangent direction.
    This means positive offsets go to the left of the travel direction.
    """
    tangents = np.empty_like(centerline)
    if closed:
        tangents = np.roll(centerline, -1, axis=0) - np.roll(
            centerline, 1, axis=0
        )
    else:
        tangents[1:-1] = centerline[2:] - centerline[:-2]
        tangents[0] = centerline[1] - centerline[0]
        tangents[-1] = centerline[-1] - centerline[-2]

    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1.0, lengths)
    tangents = tangents / lengths

    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    return normals


def _compute_arc_lengths_from_points(
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute cumulative arc lengths from an (N, 2) point array."""
    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def optimize_line(
    track: Track,
    *,
    length_weight: float = 0.01,
) -> RacingLine:
    """Compute the minimum-curvature racing line on a track.

    Solves a convex quadratic program: the decision variables are
    lateral offsets from the centerline at each station. The
    objective minimizes the sum of squared discrete curvatures
    (approximated by second differences of the path) plus an
    optional path-length regularizer.

    The track should be uniformly resampled before calling this
    function (via ``Track.resample``). Non-uniform spacing degrades
    the accuracy of the curvature approximation.

    Args:
        track: A ``Track`` instance (uniformly resampled).
        length_weight: Weight of the path-length regularizer relative
            to the curvature term. 0.0 gives pure minimum curvature;
            larger values bias toward shorter paths. Typical range:
            0.001 to 0.1. Defaults to 0.01.

    Returns:
        A ``RacingLine`` containing the optimized path, offsets, and
        arc lengths.

    Raises:
        ValueError: If the solver fails to find a solution.
    """
    n = len(track)
    normals = _compute_normals(track.centerline, track.closed)

    # Decision variable: lateral offset at each station.
    alpha = cp.Variable(n)

    # Build the path points: p_i = c_i + alpha_i * n_i.
    # Since cvxpy doesn't support element-wise multiply with 2D
    # indexing naturally, we build the x and y components separately.
    cx = track.centerline[:, 0]
    cy = track.centerline[:, 1]
    nx = normals[:, 0]
    ny = normals[:, 1]

    px = cx + cp.multiply(alpha, nx)
    py = cy + cp.multiply(alpha, ny)

    # Curvature objective: sum of squared second differences.
    # For closed tracks, indices wrap around.
    if track.closed:
        idx_prev = np.roll(np.arange(n), 1)
        idx_next = np.roll(np.arange(n), -1)
    else:
        idx_prev = np.concatenate([[0], np.arange(n - 1)])
        idx_next = np.concatenate([np.arange(1, n), [n - 1]])

    ddx = px[idx_prev] - 2 * px + px[idx_next]
    ddy = py[idx_prev] - 2 * py + py[idx_next]
    curvature_cost = cp.sum_squares(ddx) + cp.sum_squares(ddy)

    # Path-length objective: sum of squared first differences.
    if track.closed:
        dx = px[idx_next] - px
        dy = py[idx_next] - py
    else:
        dx = px[1:] - px[:-1]
        dy = py[1:] - py[:-1]
    length_cost = cp.sum_squares(dx) + cp.sum_squares(dy)

    objective = cp.Minimize(curvature_cost + length_weight * length_cost)

    # Constraints: stay within track boundaries.
    # Positive alpha = left, so:
    #   alpha_i <= w_left(i)   (can't go further left than left boundary)
    #   alpha_i >= -w_right(i) (can't go further right than right boundary)
    constraints = [
        alpha <= track.width_left,
        alpha >= -track.width_right,
    ]

    problem = cp.Problem(objective, constraints)  # type: ignore
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"Optimizer failed: solver status = {problem.status}")

    alpha_opt: NDArray[np.float64] = np.asarray(alpha.value, dtype=np.float64)

    # Build the optimized path.
    points = track.centerline + alpha_opt[:, np.newaxis] * normals
    arc_lengths = _compute_arc_lengths_from_points(points)

    return RacingLine(
        points=points,
        offsets=alpha_opt,
        arc_lengths=arc_lengths,
        closed=track.closed,
    )
