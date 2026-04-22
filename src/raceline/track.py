"""Track geometry representation and I/O.

Supported input format: TUM racetrack-database CSV format
(https://github.com/TUMFTM/racetrack-database)

# x_m, y_m, w_tr_right_m, w_tr_left_m
# 0.960975, 4.022273, 7.565, 7.361
# 4.935182, 0.985988, 7.584, 7.382
# ...

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Track:
    """A racetrack defined by a centerline and variable track widths.

    The centerline is a 2D polyline in a planar coordinate system (meters).
    At each centerline point, the track extends ``width_right`` to the right
    and ``width_left`` to the left, where "right" and "left" are defined
    relative to the direction of travel along the centerline.

    Closed-loop tracks are represented implicitly: the last centerline point
    connects back to the first, and the stored arrays do not duplicate the
    starting point.

    Attributes:
        centerline: Array of shape ``(N, 2)`` giving (x, y) coordinates in
        meters.
        width_right: Array of shape ``(N,)`` giving right-side track width in
        meters.
        width_left: Array of shape ``(N,)`` giving left-side track width in
        meters.
        closed: ``True`` if the track forms a closed loop.

    Raises:
        ValueError: If the input arrays have inconsistent shapes, if any width
            is non-positive, or if the track has too few points.
    """

    centerline: NDArray[np.float64]
    width_right: NDArray[np.float64]
    width_left: NDArray[np.float64]
    closed: bool = True

    def __post_init__(self) -> None:
        """Validate shapes and values of the input arrays."""
        if self.centerline.ndim != 2 or self.centerline.shape[1] != 2:
            raise ValueError(
                f"Centerline must have shape (N, 2),\n"
                f"got {self.centerline.shape}."
            )

        n = self.centerline.shape[0]
        if self.width_right.shape != (n,):
            raise ValueError(
                f"Width_right must have shape ({n}, ), \n"
                f"got {self.width_right.shape}."
            )

        if self.width_left.shape != (n,):
            raise ValueError(
                f"Width_left must have shape ({n}, ), \n"
                f"got {self.width_left.shape}."
            )

        min_points = 3 if self.closed else 2
        if n < min_points:
            raise ValueError(
                f"Track must have at least {min_points} points "
                f"({'closed' if self.closed else 'open'}), got {n}"
            )

        if np.any(self.width_right <= 0) or np.any(self.width_left <= 0):
            raise ValueError("Track widths must be strictly positive.")

    def __len__(self) -> int:
        """Return the number of centerline points."""
        return int(self.centerline.shape[0])

    @cached_property
    def arc_lengths(self) -> NDArray[np.float64]:
        """Cumulative arc length along the centerline at each point.

        The first entry is always 0. For a track with ``N`` points, the
        returned array has length ``N`` and represents arc length at each
        stored centerline point (not including the closing segment of a closed
        track; use :attr`total_length` for that).

        Returns:
            Array of shape ``(N, )`` of cumulative arc lengths, in meters.
        """
        diffs = np.diff(self.centerline, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return np.concatenate([[0.0], np.cumsum(segment_lengths)])

    @cached_property
    def total_length(self) -> float:
        """Total arc length of the track, in meters.

        For open tracks this is the distance from the first to last centerline
        point along the polyline. For closed tracks it additionally includes
        the closing segment from the last stored point back to the first.
        """
        open_length = float(self.arc_lengths[-1])
        if self.closed:
            closing = float(
                np.linalg.norm(self.centerline[0] - self.centerline[-1])
            )
            return open_length + closing
        return open_length

    def resample(self, spacing: float) -> Track:
        """Return a new Track with uniform arc-length spacing.

        The new track has the same geometry (to the precision of linear
        interpolation between centerline points) but with points spaced
        uniformly along arc length. This is a prerequisite for curvature
        computation and for the optimization formulation, where the
        discretization stations are expected to be evenly spaced.

        For closed tracks, the number of points is chosen so that the spacing
        between the last point and the first is also uniform.

        Args:
            spacing: Target spacing between consecutive centerline points, in
            meters. Must be positive. The actual spacing in the returned track
            will be very close to this but may differ slightly since the number
            of points is rounded to an integer.

        Returns:
            A new ``Track`` with uniformly spaced centerline points and
            interpolated widths.

        Raises:
            ValueError: if ``Spacing`` is not strictly positive.
        """
        if spacing <= 0:
            raise ValueError(f"Spacing must be positive, got {spacing}.")

        # For closed tracks, augment the arrays with the first point appended
        # so linear interpolation naturally handles wrap-around
        if self.closed:
            points_aug = np.vstack([self.centerline, self.centerline[0:1]])
            wr_aug = np.concatenate([self.width_right, self.width_right[0:1]])
            wl_aug = np.concatenate([self.width_left, self.width_left[0:1]])
        else:
            points_aug = self.centerline
            wr_aug = self.width_right
            wl_aug = self.width_left

        segment_lengths = np.linalg.norm(np.diff(points_aug, axis=0), axis=1)
        s_original = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total = float(s_original[-1])

        # Choose the number of new points. For closed tracks the N points are
        # placed at [0, L/N, 2L/N, ..., (N-1)L/N], giving uniform spacing L/N
        # including the closing segment. For open tracks, include both
        # endpoints, so N points produce N-1 intervals
        if self.closed:
            n_new = max(3, round(total / spacing))
            s_new = np.linspace(0.0, total, n_new, endpoint=False)
        else:
            n_new = max(2, round(total / spacing) + 1)
            s_new = np.linspace(0.0, total, n_new)

        x_new = np.interp(s_new, s_original, points_aug[:, 0])
        y_new = np.interp(s_new, s_original, points_aug[:, 1])
        wr_new = np.interp(s_new, s_original, wr_aug)
        wl_new = np.interp(s_new, s_original, wl_aug)

        return Track(
            centerline=np.column_stack([x_new, y_new]),
            width_right=wr_new,
            width_left=wl_new,
            closed=self.closed,
        )

    @classmethod
    def from_csv(cls, path: Path | str, *, closed: bool = True) -> Track:  # noqa
        """Load a track from a TUM-format CSV file.

        The file must be a comma-separated file with four columns:
        ``x_m``, ``y_m``, ``w_tr_right_m``, ``w_tr_left_m``. Lines beginning
        with ``#`` are treated as comments and skipped, which conveniently
        handles the header row used by the TUM racetrack-database.

        Args:
            path: Path to the CSV file.
            closed: Whether the track is a closed loop. Defaults to ``True``.

        Returns:
            A new ``Track`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed as a 4-column numeric CSV,
                or if the resulting arrays fail validation.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"track file not found: {path}")

        data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)

        # np.loadtxt returns a 1D array if the file has a single data row;
        # reshape so downstream checks are uniform.
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != 4:
            raise ValueError(
                f"expected a CSV with 4 columns (x, y, w_right, w_left), "
                f"got {data.shape[1]} columns"
            )

        return cls(
            centerline=data[:, 0:2].copy(),
            width_right=data[:, 2].copy(),
            width_left=data[:, 3].copy(),
            closed=closed,
        )
