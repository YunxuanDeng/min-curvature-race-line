"""Unit tests for the end-to-end lap simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from raceline import (
    LapResult,
    PointMassVehicle,
    Track,
    simulate_lap,
)


# Helpers
def _circle_track(
    radius: float, n_points: int = 300, width: float = 5.0
) -> Track:
    """Build a closed circular track."""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cl = radius * np.column_stack([np.cos(theta), np.sin(theta)])
    w = np.full(n_points, width)
    return Track(
        centerline=cl,
        width_right=w,
        width_left=w.copy(),
        closed=True,
    )


def _stadium_track(width: float = 5.0) -> Track:
    """Build a stadium: two straights + two half-circles."""
    r, L = 30.0, 150.0
    n_s, n_a = 100, 100
    x_b = np.linspace(-L / 2, L / 2, n_s, endpoint=False)
    y_b = np.full(n_s, -r)
    t_r = np.linspace(-np.pi / 2, np.pi / 2, n_a, endpoint=False)
    x_r = L / 2 + r * np.cos(t_r)
    y_r = r * np.sin(t_r)
    x_t = np.linspace(L / 2, -L / 2, n_s, endpoint=False)
    y_t = np.full(n_s, r)
    t_l = np.linspace(np.pi / 2, 3 * np.pi / 2, n_a, endpoint=False)
    x_l = -L / 2 + r * np.cos(t_l)
    y_l = r * np.sin(t_l)
    cl = np.column_stack(
        [
            np.concatenate([x_b, x_r, x_t, x_l]),
            np.concatenate([y_b, y_r, y_t, y_l]),
        ]
    )
    n = len(cl)
    w = np.full(n, width)
    track = Track(
        centerline=cl,
        width_right=w,
        width_left=w.copy(),
        closed=True,
    )
    return track.resample(spacing=1.0)


def _default_vehicle() -> PointMassVehicle:
    """A balanced test vehicle."""
    return PointMassVehicle(
        mass=1000.0,
        max_grip=10.0,
        max_engine_acceleration=5.0,
        max_brake_deceleration=8.0,
        max_speed_value=60.0,
    )


# LapResult construction
class TestLapResult:
    """Tests for the LapResult dataclass."""

    def test_stores_attributes(self) -> None:
        """LapResult stores all provided fields."""
        speeds = np.array([10.0, 20.0])
        arcs = np.array([0.0, 1.0])
        from raceline import RacingLine, SpeedProfile

        sp = SpeedProfile(
            speeds=speeds,
            arc_lengths=arcs,
            lap_time=1.0,
            closed=True,
        )
        rl = RacingLine(
            points=np.zeros((2, 2)),
            offsets=np.zeros(2),
            arc_lengths=arcs,
            closed=True,
        )
        result = LapResult(
            lap_time=1.0,
            speed_profile=sp,
            racing_line=rl,
            lateral_g=np.zeros(2),
            longitudinal_g=np.zeros(2),
        )
        assert result.lap_time == 1.0
        assert len(result.lateral_g) == 2


# Centerline simulation (optimize=False)
class TestSimulateCenterline:
    """Tests for simulation on the centerline (no optimization)."""

    def test_circle_centerline_lap_time(self) -> None:
        """Centerline lap on a circle matches the analytical formula."""
        radius = 50.0
        track = _circle_track(radius)
        vehicle = PointMassVehicle(
            mass=1000.0,
            max_grip=10.0,
            max_engine_acceleration=100.0,
            max_brake_deceleration=100.0,
            max_speed_value=200.0,
        )
        result = simulate_lap(track, vehicle, optimize=False)
        expected = 2 * math.pi * math.sqrt(radius / vehicle.max_grip)
        assert result.lap_time == pytest.approx(expected, rel=1e-3)

    def test_centerline_offsets_are_zero(self) -> None:
        """Centerline simulation uses zero lateral offsets."""
        track = _circle_track(50.0)
        vehicle = _default_vehicle()
        result = simulate_lap(track, vehicle, optimize=False)
        np.testing.assert_allclose(result.racing_line.offsets, 0.0)

    def test_lateral_g_within_grip(self) -> None:
        """Lateral g magnitude never exceeds the vehicle's grip limit."""
        track = _stadium_track()
        vehicle = _default_vehicle()
        result = simulate_lap(track, vehicle, optimize=False)
        assert np.all(np.abs(result.lateral_g) <= vehicle.max_grip + 1e-6)

    def test_all_speeds_positive(self) -> None:
        """All speeds in the result are positive."""
        track = _stadium_track()
        vehicle = _default_vehicle()
        result = simulate_lap(track, vehicle, optimize=False)
        assert np.all(result.speed_profile.speeds > 0)


# Optimized simulation (optimize=True)
class TestSimulateOptimized:
    """Tests for simulation with the racing-line optimizer."""

    def test_optimized_is_faster_than_centerline(self) -> None:
        """Optimized lap is faster than centerline lap."""
        track = _stadium_track()
        vehicle = _default_vehicle()
        cl = simulate_lap(track, vehicle, optimize=False)
        opt = simulate_lap(track, vehicle, optimize=True)
        assert opt.lap_time < cl.lap_time

    def test_optimized_offsets_within_bounds(self) -> None:
        """Optimized offsets stay within track boundaries."""
        track = _stadium_track()
        vehicle = _default_vehicle()
        result = simulate_lap(track, vehicle, optimize=True)
        # OSQP has ~1e-4 constraint tolerance, so allow slack.
        assert np.all(result.racing_line.offsets >= -track.width_right - 1e-3)
        assert np.all(result.racing_line.offsets <= track.width_left + 1e-3)

    def test_result_has_g_force_arrays(self) -> None:
        """LapResult includes lateral and longitudinal g arrays."""
        track = _stadium_track()
        vehicle = _default_vehicle()
        result = simulate_lap(track, vehicle, optimize=True)
        assert len(result.lateral_g) == len(track)
        assert len(result.longitudinal_g) == len(track)

    def test_length_weight_affects_result(self) -> None:
        """Different length_weight produces different lap times."""
        track = _stadium_track()
        vehicle = _default_vehicle()
        r1 = simulate_lap(track, vehicle, length_weight=0.0)
        r2 = simulate_lap(track, vehicle, length_weight=1.0)
        assert r1.lap_time != pytest.approx(r2.lap_time)


# G-force consistency
class TestGForces:
    """Tests for the g-force computation."""

    def test_lateral_g_on_circle_is_constant(self) -> None:
        """On a circle, lateral g is constant (v^2 * kappa)."""
        radius = 50.0
        track = _circle_track(radius)
        vehicle = PointMassVehicle(
            mass=1000.0,
            max_grip=10.0,
            max_engine_acceleration=100.0,
            max_brake_deceleration=100.0,
            max_speed_value=200.0,
        )
        result = simulate_lap(track, vehicle, optimize=False)
        # On a constant-speed constant-radius lap, lateral g
        # should be uniform and equal to v^2/R = grip.
        assert result.lateral_g.std() < 0.1
        assert result.lateral_g.mean() == pytest.approx(
            vehicle.max_grip, rel=1e-2
        )

    def test_longitudinal_g_on_circle_is_near_zero(self) -> None:
        """On a circle at constant speed, longitudinal g is ~0."""
        radius = 50.0
        track = _circle_track(radius)
        vehicle = PointMassVehicle(
            mass=1000.0,
            max_grip=10.0,
            max_engine_acceleration=100.0,
            max_brake_deceleration=100.0,
            max_speed_value=200.0,
        )
        result = simulate_lap(track, vehicle, optimize=False)
        # Constant speed -> no longitudinal acceleration.
        np.testing.assert_allclose(result.longitudinal_g, 0.0, atol=0.1)
