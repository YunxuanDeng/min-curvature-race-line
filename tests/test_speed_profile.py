"""Unit tests for the speed profile computation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from raceline import (
    PointMassVehicle,
    SpeedProfile,
    Track,
    compute_speed_profile,
)


# Helpers
def _circle_track(radius: float, n_points: int = 500) -> Track:
    """Build a closed circular track of the given radius."""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    centerline = radius * np.column_stack([np.cos(theta), np.sin(theta)])
    widths = np.full(n_points, max(1.0, radius * 0.1))
    return Track(
        centerline=centerline,
        width_right=widths,
        width_left=widths.copy(),
        closed=True,
    )


def _tire_limited_vehicle() -> PointMassVehicle:
    """A vehicle whose engine and brakes exceed tire grip: tire-limited."""
    return PointMassVehicle(
        mass=1000.0,
        max_grip=10.0,
        max_engine_acceleration=100.0,  # way higher than grip
        max_brake_deceleration=100.0,  # way higher than grip
        max_speed_value=200.0,  # essentially unbounded for test cases
    )


def _balanced_vehicle() -> PointMassVehicle:
    """A vehicle with engine and brakes smaller than tire grip."""
    return PointMassVehicle(
        mass=1000.0,
        max_grip=10.0,
        max_engine_acceleration=4.0,
        max_brake_deceleration=8.0,
        max_speed_value=60.0,
    )


# SpeedProfile construction tests
class TestSpeedProfileDataclass:
    """Tests for the SpeedProfile container dataclass."""

    def test_construction_stores_values(self) -> None:
        """SpeedProfile stores the provided arrays and lap time verbatim."""
        speeds = np.array([10.0, 20.0, 30.0])
        arcs = np.array([0.0, 1.0, 2.0])
        profile = SpeedProfile(
            speeds=speeds, arc_lengths=arcs, lap_time=0.5, closed=True
        )
        np.testing.assert_array_equal(profile.speeds, speeds)
        np.testing.assert_array_equal(profile.arc_lengths, arcs)
        assert profile.lap_time == 0.5
        assert profile.closed is True

    def test_min_speed_property(self) -> None:
        """min_speed returns the smallest speed in the profile."""
        profile = SpeedProfile(
            speeds=np.array([30.0, 10.0, 20.0]),
            arc_lengths=np.array([0.0, 1.0, 2.0]),
            lap_time=1.0,
            closed=True,
        )
        assert profile.min_speed == 10.0

    def test_max_speed_achieved_property(self) -> None:
        """max_speed_achieved returns the largest speed in the profile."""
        profile = SpeedProfile(
            speeds=np.array([30.0, 10.0, 20.0]),
            arc_lengths=np.array([0.0, 1.0, 2.0]),
            lap_time=1.0,
            closed=True,
        )
        assert profile.max_speed_achieved == 30.0


# Input validation tests
class TestInputValidation:
    """Tests for argument validation in compute_speed_profile."""

    def test_rejects_mismatched_shapes(self) -> None:
        """Curvature and arc_lengths with different shapes are rejected."""
        vehicle = _tire_limited_vehicle()
        with pytest.raises(ValueError, match="same shape"):
            compute_speed_profile(
                curvature=np.zeros(5),
                arc_lengths=np.zeros(10),
                vehicle=vehicle,
            )

    def test_rejects_2d_curvature(self) -> None:
        """2D curvature array is rejected."""
        vehicle = _tire_limited_vehicle()
        with pytest.raises(ValueError, match="1D"):
            compute_speed_profile(
                curvature=np.zeros((5, 2)),
                arc_lengths=np.zeros((5, 2)),
                vehicle=vehicle,
            )

    def test_rejects_too_few_stations(self) -> None:
        """A single station cannot form a speed profile."""
        vehicle = _tire_limited_vehicle()
        with pytest.raises(ValueError, match="at least 2 stations"):
            compute_speed_profile(
                curvature=np.array([0.0]),
                arc_lengths=np.array([0.0]),
                vehicle=vehicle,
            )


# Oracle tests: constant-radius circle (analytical ground truth)
class TestCircleOracle:
    """Tests comparing computed speed profiles to analytical formulas."""

    def test_uniform_speed_on_circle(self) -> None:
        """On a constant-radius circle, the achievable speed is constant."""
        radius = 50.0
        track = _circle_track(radius)
        vehicle = _tire_limited_vehicle()

        profile = compute_speed_profile(
            track.curvature, track.arc_lengths, vehicle, closed=True
        )

        # Speeds should be (numerically) identical everywhere.
        assert profile.speeds.std() < 1e-6

    def test_circle_speed_matches_analytical_formula(self) -> None:
        """Circle speed equals sqrt(max_grip * R)."""
        radius = 50.0
        track = _circle_track(radius)
        vehicle = _tire_limited_vehicle()

        profile = compute_speed_profile(
            track.curvature, track.arc_lengths, vehicle, closed=True
        )

        expected = math.sqrt(vehicle.max_grip * radius)
        assert profile.min_speed == pytest.approx(expected, rel=1e-4)
        assert profile.max_speed_achieved == pytest.approx(expected, rel=1e-4)

    def test_circle_lap_time_matches_analytical_formula(self) -> None:
        """Circle lap time equals 2*pi*sqrt(R / max_grip)."""
        radius = 50.0
        track = _circle_track(radius)
        vehicle = _tire_limited_vehicle()

        profile = compute_speed_profile(
            track.curvature, track.arc_lengths, vehicle, closed=True
        )

        expected = 2 * math.pi * math.sqrt(radius / vehicle.max_grip)
        # Trapezoidal integration on a constant-speed curve is essentially
        # exact; any error is floating-point noise.
        assert profile.lap_time == pytest.approx(expected, rel=1e-4)

    def test_larger_circle_gives_higher_speed_and_longer_lap(self) -> None:
        """A larger circle allows faster cornering but takes longer to lap."""
        vehicle = _tire_limited_vehicle()

        track_small = _circle_track(25.0)
        track_large = _circle_track(100.0)
        profile_small = compute_speed_profile(
            track_small.curvature,
            track_small.arc_lengths,
            vehicle,
            closed=True,
        )
        profile_large = compute_speed_profile(
            track_large.curvature,
            track_large.arc_lengths,
            vehicle,
            closed=True,
        )

        assert profile_large.min_speed > profile_small.min_speed
        assert profile_large.lap_time > profile_small.lap_time

    def test_cap_at_max_speed_on_very_large_circle(self) -> None:
        """On a very large circle, the vehicle's top speed caps the profile."""
        # Huge radius → grip-limited speed would exceed top speed.
        radius = 10000.0
        track = _circle_track(radius)
        # Use a vehicle with modest top speed but huge engine/brakes.
        vehicle = PointMassVehicle(
            mass=1000.0,
            max_grip=10.0,
            max_engine_acceleration=100.0,
            max_brake_deceleration=100.0,
            max_speed_value=50.0,  # moderately small top speed
        )

        profile = compute_speed_profile(
            track.curvature, track.arc_lengths, vehicle, closed=True
        )

        # sqrt(10 * 10000) = sqrt(100000) ≈ 316 m/s, far above 50 cap.
        assert profile.max_speed_achieved == pytest.approx(50.0, rel=1e-3)


# Physical consistency checks
class TestPhysicalConsistency:
    """Sanity checks for physical sensibility."""

    def test_all_speeds_positive(self) -> None:
        """Speeds are strictly positive on a non-degenerate track."""
        track = _circle_track(50.0)
        vehicle = _balanced_vehicle()
        profile = compute_speed_profile(
            track.curvature, track.arc_lengths, vehicle, closed=True
        )
        assert np.all(profile.speeds > 0)

    def test_speeds_do_not_exceed_max_speed(self) -> None:
        """No station's speed exceeds the vehicle's top speed."""
        track = _circle_track(1000.0)  # fast, but capped by top speed
        vehicle = _balanced_vehicle()
        profile = compute_speed_profile(
            track.curvature, track.arc_lengths, vehicle, closed=True
        )
        assert profile.max_speed_achieved <= vehicle.max_speed + 1e-9

    def test_speeds_respect_friction_circle_at_all_stations(self) -> None:
        """At every station, lateral g = v^2 * |kappa| does not exceed grip."""
        track = _circle_track(50.0)
        vehicle = _tire_limited_vehicle()
        profile = compute_speed_profile(
            track.curvature, track.arc_lengths, vehicle, closed=True
        )

        lateral_g = profile.speeds**2 * np.abs(track.curvature)
        # Allow small numerical slack.
        assert np.all(lateral_g <= vehicle.max_grip + 1e-6)

    def test_lap_time_is_positive(self) -> None:
        """Lap time is a positive number for a valid track."""
        track = _circle_track(50.0)
        vehicle = _balanced_vehicle()
        profile = compute_speed_profile(
            track.curvature, track.arc_lengths, vehicle, closed=True
        )
        assert profile.lap_time > 0

    def test_slower_vehicle_takes_longer_lap(self) -> None:
        """Less powerful vehicle produces longer lap time on same track."""
        track = _circle_track(50.0)
        fast = PointMassVehicle(
            mass=1000.0,
            max_grip=20.0,
            max_engine_acceleration=20.0,
            max_brake_deceleration=20.0,
            max_speed_value=200.0,
        )
        slow = PointMassVehicle(
            mass=1000.0,
            max_grip=5.0,
            max_engine_acceleration=5.0,
            max_brake_deceleration=5.0,
            max_speed_value=200.0,
        )

        profile_fast = compute_speed_profile(
            track.curvature, track.arc_lengths, fast, closed=True
        )
        profile_slow = compute_speed_profile(
            track.curvature, track.arc_lengths, slow, closed=True
        )

        assert profile_slow.lap_time > profile_fast.lap_time


# Curvature tests (part of this phase since Track.curvature is added)
class TestTrackCurvature:
    """Tests for the Track.curvature cached property."""

    def test_unit_circle_curvature_is_one(self) -> None:
        """Curvature of a unit circle is 1.0 everywhere."""
        track = _circle_track(1.0, n_points=200)
        np.testing.assert_allclose(track.curvature, 1.0, atol=1e-6)

    def test_radius_R_circle_curvature_is_inverse_R(self) -> None:
        """Curvature of a radius-R circle is 1/R everywhere."""
        for radius in (5.0, 50.0, 500.0):
            track = _circle_track(radius, n_points=300)
            np.testing.assert_allclose(
                track.curvature, 1.0 / radius, atol=1e-5 / radius
            )

    def test_straight_curvature_is_zero(self) -> None:
        """An open straight track has zero curvature everywhere."""
        x = np.linspace(0, 100, 101)
        y = np.zeros_like(x)
        centerline = np.column_stack([x, y])
        widths = np.full(101, 3.0)
        track = Track(
            centerline=centerline,
            width_right=widths,
            width_left=widths.copy(),
            closed=False,
        )
        np.testing.assert_allclose(track.curvature, 0.0, atol=1e-10)

    def test_curvature_sign_differs_for_opposite_rotations(self) -> None:
        """CCW and CW circles have opposite-sign curvature."""
        n = 200
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ccw = np.column_stack([np.cos(theta), np.sin(theta)])
        cw = np.column_stack([np.cos(-theta), np.sin(-theta)])
        widths = np.full(n, 0.1)

        track_ccw = Track(
            centerline=ccw,
            width_right=widths,
            width_left=widths.copy(),
            closed=True,
        )
        track_cw = Track(
            centerline=cw,
            width_right=widths,
            width_left=widths.copy(),
            closed=True,
        )

        # Both have magnitude 1.0 (unit circles), opposite signs.
        assert track_ccw.curvature.mean() > 0
        assert track_cw.curvature.mean() < 0
        np.testing.assert_allclose(
            track_ccw.curvature, -track_cw.curvature, atol=1e-6
        )
