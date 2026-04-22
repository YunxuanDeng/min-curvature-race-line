"""Unit tests for the Track class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from track import Track


# Helpers
def _make_square_track(width: float = 5.0) -> Track:
    """Return a simple square closed track, useful as a valid fixture."""
    centerline = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float64
    )
    widths = np.full(4, width, dtype=np.float64)
    return Track(
        centerline=centerline,
        width_right=widths,
        width_left=widths.copy(),
        closed=True,
    )


def _write_csv(
    path: Path,
    rows: list[tuple[float, float, float, float]],
    *,
    header: str | None = "# x_m,y_m,w_tr_right_m,w_tr_left_m",
) -> None:
    """Write a CSV file in TUM format with the given rows."""
    with path.open("w") as f:
        if header is not None:
            f.write(header + "\n")
        for x, y, wr, wl in rows:
            f.write(f"{x},{y},{wr},{wl}\n")


# Constructor tests
class TestTrackConstruction:
    """Tests for the ``Track`` constructor and ``__post_init__`` validation."""

    def test_valid_closed_track(self) -> None:
        """Test that a 4-point square closed track constructs successfully."""
        track = _make_square_track()
        assert len(track) == 4
        assert track.closed is True

    def test_valid_open_track(self) -> None:
        """Test that a minimal 2-point open track constructs successfully."""
        centerline = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        widths = np.array([3.0, 3.0], dtype=np.float64)
        track = Track(
            centerline=centerline,
            width_right=widths,
            width_left=widths.copy(),
            closed=False,
        )
        assert len(track) == 2
        assert track.closed is False

    def test_rejects_wrong_centerline_dimensions(self) -> None:
        """Test that a 1D centerline (instead of (N,2)) gets rejected."""
        bad = np.array([0.0, 1.0, 2.0], dtype=np.float64)  # 1D
        with pytest.raises(ValueError, match="[Cc]enterline must have shape"):
            Track(
                centerline=bad,
                width_right=np.array([1.0]),
                width_left=np.array([1.0]),
            )

    def test_rejects_centerline_with_wrong_column_count(self) -> None:
        """Test that a centerline with shape (N,3) gets rejected."""
        bad = np.zeros((5, 3), dtype=np.float64)  # (N, 3) instead of (N, 2)
        with pytest.raises(ValueError, match="[Cc]enterline must have shape"):
            Track(
                centerline=bad,
                width_right=np.ones(5),
                width_left=np.ones(5),
            )

    def test_rejects_width_right_length_mismatch(self) -> None:
        """Test that a width_right array with length != N gets rejected."""
        centerline = np.zeros((4, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="[Ww]idth_right must have shape"):
            Track(
                centerline=centerline,
                width_right=np.ones(3),  # wrong length
                width_left=np.ones(4),
            )

    def test_rejects_width_left_length_mismatch(self) -> None:
        """Test that a width_left array with length != N gets rejected."""
        centerline = np.zeros((4, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="[Ww]idth_left must have shape"):
            Track(
                centerline=centerline,
                width_right=np.ones(4),
                width_left=np.ones(5),  # wrong length
            )

    def test_rejects_non_positive_widths(self) -> None:
        """Test that a track with any zero width gets rejected."""
        centerline = np.zeros((4, 2), dtype=np.float64)
        with pytest.raises(
            ValueError, match="widths must be strictly positive"
        ):
            Track(
                centerline=centerline,
                width_right=np.array([1.0, 0.0, 1.0, 1.0]),  # contains zero
                width_left=np.ones(4),
            )

    def test_rejects_negative_widths(self) -> None:
        """Test that a track with any negative width gets rejected."""
        centerline = np.zeros((4, 2), dtype=np.float64)
        with pytest.raises(
            ValueError, match="widths must be strictly positive"
        ):
            Track(
                centerline=centerline,
                width_right=np.ones(4),
                width_left=np.array([1.0, 1.0, -0.5, 1.0]),  # negative
            )

    def test_closed_track_requires_three_points(self) -> None:
        """Test that any closed track needs at least 3 points."""
        centerline = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="at least 3 points"):
            Track(
                centerline=centerline,
                width_right=np.ones(2),
                width_left=np.ones(2),
                closed=True,
            )

    def test_open_track_requires_two_points(self) -> None:
        """Test that any open track needs at least 2 points."""
        centerline = np.array([[0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="at least 2 points"):
            Track(
                centerline=centerline,
                width_right=np.ones(1),
                width_left=np.ones(1),
                closed=False,
            )


# CSV loading tests
class TestTrackFromCsv:
    """Tests for the ``Track.from_csv`` classmethod and file parsing."""

    def test_loads_valid_csv_with_header(self, tmp_path: Path) -> None:
        """Test that a 4-row CSV with ``#`` parses into expected arrays."""
        csv = tmp_path / "track.csv"
        rows = [
            (0.0, 0.0, 5.0, 5.0),
            (1.0, 0.0, 5.1, 4.9),
            (2.0, 0.5, 5.2, 4.8),
            (1.0, 1.0, 5.1, 4.9),
        ]
        _write_csv(csv, rows)

        track = Track.from_csv(csv)

        assert len(track) == 4
        np.testing.assert_allclose(track.centerline[0], [0.0, 0.0])
        np.testing.assert_allclose(track.centerline[-1], [1.0, 1.0])
        np.testing.assert_allclose(track.width_right, [5.0, 5.1, 5.2, 5.1])
        np.testing.assert_allclose(track.width_left, [5.0, 4.9, 4.8, 4.9])

    def test_loads_valid_csv_without_header(self, tmp_path: Path) -> None:
        """Test that a CSV with no header parses successfully."""
        csv = tmp_path / "track.csv"
        rows = [
            (0.0, 0.0, 5.0, 5.0),
            (1.0, 0.0, 5.0, 5.0),
            (1.0, 1.0, 5.0, 5.0),
        ]
        _write_csv(csv, rows, header=None)

        track = Track.from_csv(csv)

        assert len(track) == 3

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        """Test that ``from_csv`` accepts a str path, not just an object."""
        csv = tmp_path / "track.csv"
        rows = [
            (0.0, 0.0, 5.0, 5.0),
            (1.0, 0.0, 5.0, 5.0),
            (1.0, 1.0, 5.0, 5.0),
        ]
        _write_csv(csv, rows)

        track = Track.from_csv(str(csv))

        assert len(track) == 3

    def test_open_loop_flag_is_respected(self, tmp_path: Path) -> None:
        """Passing ``closed=False`` to ``from_csv`` produces an open track."""
        csv = tmp_path / "track.csv"
        rows = [(0.0, 0.0, 5.0, 5.0), (10.0, 0.0, 5.0, 5.0)]
        _write_csv(csv, rows)

        track = Track.from_csv(csv, closed=False)

        assert not track.closed

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Test that loading a nonexistent path raises the right error."""
        with pytest.raises(FileNotFoundError, match="track file not found"):
            Track.from_csv(tmp_path / "does_not_exist.csv")

    def test_rejects_wrong_column_count(self, tmp_path: Path) -> None:
        """Test that a CSV with the wrong number of col get rejected."""
        csv = tmp_path / "track.csv"
        csv.write_text("# x,y,w\n0,0,5\n1,0,5\n2,0,5\n")

        with pytest.raises(ValueError, match="expected a CSV with 4 columns"):
            Track.from_csv(csv)

    def test_loaded_arrays_are_independent_copies(
        self, tmp_path: Path
    ) -> None:
        """Mutating the Track arrays should not leak into anything else."""
        csv = tmp_path / "track.csv"
        rows = [
            (0.0, 0.0, 5.0, 5.0),
            (1.0, 0.0, 5.0, 5.0),
            (1.0, 1.0, 5.0, 5.0),
        ]
        _write_csv(csv, rows)

        track_a = Track.from_csv(csv)
        track_b = Track.from_csv(csv)

        track_a.centerline[0, 0] = 999.0
        assert track_b.centerline[0, 0] == 0.0


# Arc-length parameterization tests
class TestArcLengths:
    """Tests for the ``Track.arc_lengths`` cached property."""

    def test_first_entry_is_zero(self) -> None:
        """Test that the cumulative arc length at first point is always 0."""
        track = _make_square_track()
        assert track.arc_lengths[0] == 0.0

    def test_cumulative_lengths_on_square(self) -> None:
        """A 10m square has arc lengths [0, 10, 20, 30] along the polyline."""
        # Square with 10m sides: arc lengths along the stored polyline
        # should be [0, 10, 20, 30] (not including closing segment).
        track = _make_square_track()
        np.testing.assert_allclose(track.arc_lengths, [0.0, 10.0, 20.0, 30.0])

    def test_cumulative_lengths_monotonic(self) -> None:
        """Arc lengths are increasing for a track with non-duplicate points."""
        track = _make_square_track()
        assert np.all(np.diff(track.arc_lengths) > 0)


class TestTotalLength:
    """Tests for the ``Track.total_length`` cached property."""

    def test_closed_square_total_length(self) -> None:
        """Test that a closed 10m square's total_length == its perimeter."""
        # 10m sides, closed loop: perimeter = 40m
        track = _make_square_track()
        assert track.total_length == pytest.approx(40.0)

    def test_open_track_total_length(self) -> None:
        """An open straight from (0,0) to (10,0) has total_length = 10m."""
        # Straight of length 10m from (0,0) to (10,0), open
        centerline = np.array(
            [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float64
        )
        widths = np.full(3, 3.0, dtype=np.float64)
        track = Track(
            centerline=centerline,
            width_right=widths,
            width_left=widths.copy(),
            closed=False,
        )
        assert track.total_length == pytest.approx(10.0)

    def test_closed_vs_open_differ_by_closing_segment(self) -> None:
        """Closed-track length exceeds open-track by the closing segment."""
        centerline = np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float64
        )
        widths = np.full(3, 3.0, dtype=np.float64)
        open_track = Track(
            centerline=centerline,
            width_right=widths,
            width_left=widths.copy(),
            closed=False,
        )
        closed_track = Track(
            centerline=centerline,
            width_right=widths,
            width_left=widths.copy(),
            closed=True,
        )
        closing_segment = float(np.linalg.norm(centerline[0] - centerline[-1]))
        assert closed_track.total_length == pytest.approx(
            open_track.total_length + closing_segment
        )


# Resampling tests
class TestResample:
    """Tests for the ``Track.resample`` method. Uniform arc-length sampling."""

    def test_rejects_non_positive_spacing(self) -> None:
        """``resample`` rejects zero or negative spacing values."""
        track = _make_square_track()
        with pytest.raises(ValueError, match="[Ss]pacing must be positive"):
            track.resample(0.0)
        with pytest.raises(ValueError, match="[Ss]pacing must be positive"):
            track.resample(-1.0)

    def test_square_resample_gives_expected_points(self) -> None:
        """A 10m square resampled at 5m yields 8 points (corners + 4midpts)."""
        # 10m square (perimeter 40m) resampled at 5m spacing should produce
        # exactly 8 points: 4 corners and 4 edge midpoints.
        track = _make_square_track()
        resampled = track.resample(spacing=5.0)

        assert len(resampled) == 8
        expected = np.array(
            [
                [0.0, 0.0],  # corner
                [5.0, 0.0],  # midpoint of bottom edge
                [10.0, 0.0],  # corner
                [10.0, 5.0],  # midpoint of right edge
                [10.0, 10.0],  # corner
                [5.0, 10.0],  # midpoint of top edge
                [0.0, 10.0],  # corner
                [0.0, 5.0],  # midpoint of left edge
            ]
        )
        np.testing.assert_allclose(resampled.centerline, expected, atol=1e-10)

    def test_resample_preserves_total_length(self) -> None:
        """Resampling at a divisor of the perimeter preserves total_length."""
        # Resampling a polyline onto itself (no corners cut) preserves length.
        track = _make_square_track()
        resampled = track.resample(spacing=5.0)
        assert resampled.total_length == pytest.approx(track.total_length)

    def test_resample_widths_interpolated(self) -> None:
        """Test that verify widths at midpoints."""
        # Give each corner a distinct width and verify midpoints get the
        # average of the neighboring corners.
        centerline = np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            dtype=np.float64,
        )
        wr = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
        wl = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float64)
        track = Track(
            centerline=centerline, width_right=wr, width_left=wl, closed=True
        )

        resampled = track.resample(spacing=5.0)

        # At edge midpoints, widths should be the average of the bracketing
        # corners (including wrap for the last midpoint).
        expected_wr = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0])
        np.testing.assert_allclose(
            resampled.width_right, expected_wr, atol=1e-10
        )

    def test_resample_is_pure(self) -> None:
        """resample() returns a new Track; the original is unchanged."""
        track = _make_square_track()
        original_centerline = track.centerline.copy()
        original_wr = track.width_right.copy()
        _ = track.resample(spacing=2.5)
        np.testing.assert_array_equal(track.centerline, original_centerline)
        np.testing.assert_array_equal(track.width_right, original_wr)

    def test_resample_preserves_closed_flag(self) -> None:
        """Resampling a closed track yields a closed track."""
        track = _make_square_track()
        resampled = track.resample(spacing=5.0)
        assert resampled.closed is True

    def test_resample_open_track(self) -> None:
        """An open straight resamples to evenly-spaced points."""
        # Straight from (0,0) to (10,0), open. Resample at 2.5m spacing:
        # expect 5 points at x = 0, 2.5, 5, 7.5, 10.
        centerline = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        widths = np.full(2, 3.0, dtype=np.float64)
        track = Track(
            centerline=centerline,
            width_right=widths,
            width_left=widths.copy(),
            closed=False,
        )
        resampled = track.resample(spacing=2.5)
        assert len(resampled) == 5
        np.testing.assert_allclose(
            resampled.centerline[:, 0], [0.0, 2.5, 5.0, 7.5, 10.0]
        )
        np.testing.assert_allclose(resampled.centerline[:, 1], np.zeros(5))

    def test_resample_on_smooth_curve_gives_near_uniform_spacing(self) -> None:
        """On a smooth curve, Euclidean distance approximates arc length."""
        # 200-point approximation of a unit circle centered at origin
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        centerline = np.column_stack([np.cos(theta), np.sin(theta)])
        widths = np.full(200, 0.3, dtype=np.float64)
        track = Track(
            centerline=centerline,
            width_right=widths,
            width_left=widths.copy(),
            closed=True,
        )

        target_spacing = 0.1
        resampled = track.resample(spacing=target_spacing)

        diffs = np.diff(resampled.centerline, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        wrap_length = float(
            np.linalg.norm(resampled.centerline[0] - resampled.centerline[-1])
        )
        all_segs = np.concatenate([seg_lengths, [wrap_length]])

        # Each Euclidean distance should be close to the effective spacing.
        # On a smooth curve the two agree to O(spacing^2 / R) — here ~1e-3.
        effective_spacing = resampled.total_length / len(resampled)
        np.testing.assert_allclose(all_segs, effective_spacing, rtol=1e-2)

    def test_resample_with_spacing_larger_than_perimeter(self) -> None:
        """Very large spacing is clamped to the minimum point count."""
        track = _make_square_track()
        resampled = track.resample(spacing=1000.0)
        assert len(resampled) == 3  # minimum for a closed track
