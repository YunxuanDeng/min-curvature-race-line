"""Unit tests for the PointMassVehicle class."""

from __future__ import annotations

import math

import pytest

from raceline import PointMassVehicle, VehicleModel


def _f1_like_vehicle() -> PointMassVehicle:
    """An F1-like vehicle with high grip (slicks) and high engine output."""
    return PointMassVehicle(
        mass=740.0,
        max_grip=30.0,
        max_engine_acceleration=15.0,
        max_brake_deceleration=50.0,
        max_speed_value=95.0,
    )


def _road_car() -> PointMassVehicle:
    """A road-car-like sedan: MUCH lower grip, weaker engine, weaker brakes."""
    return PointMassVehicle(
        mass=1500.0,
        max_grip=15.0,
        max_engine_acceleration=4.0,
        max_brake_deceleration=8.0,
        max_speed_value=60.0,
    )


# Construction tests
class TestPointMassVehicleConstruction:
    """Tests for the constructor and validation of PointMassVehicle."""

    def test_valid_construction(self) -> None:
        """A vehicle with all-positive parameters constructs successfully."""
        v = _f1_like_vehicle()
        assert isinstance(v, VehicleModel)
        assert v.mass == 740.0
        assert v.max_speed == 95.0

    def test_rejects_zero_mass(self) -> None:
        """Mass of zero is rejected."""
        with pytest.raises(ValueError, match="mass"):
            PointMassVehicle(0.0, 10.0, 4.0, 8.0, 60.0)

    def test_rejects_negative_grip(self) -> None:
        """Negative grip is rejected."""
        with pytest.raises(ValueError, match="max_grip"):
            PointMassVehicle(1500.0, -5.0, 4.0, 8.0, 60.0)

    def test_rejects_zero_engine_acceleration(self) -> None:
        """Zero engine acceleration is rejected."""
        with pytest.raises(ValueError, match="max_engine_acceleration"):
            PointMassVehicle(1500.0, 10.0, 0.0, 8.0, 60.0)

    def test_rejects_zero_brake(self) -> None:
        """Zero brake deceleration is rejected."""
        with pytest.raises(ValueError, match="max_brake_deceleration"):
            PointMassVehicle(1500.0, 10.0, 4.0, 0.0, 60.0)

    def test_rejects_zero_max_speed(self) -> None:
        """Zero max speed is rejected."""
        with pytest.raises(ValueError, match="max_speed_value"):
            PointMassVehicle(1500.0, 10.0, 4.0, 8.0, 0.0)


# Cornering speed tests (max_speed_at_curvature)
class TestMaxSpeedAtCurvature:
    """Tests for the cornering speed limit formula."""

    def test_straight_returns_max_speed(self) -> None:
        """Straight: cornering doesn't constrain; cap at max_speed."""
        v = _f1_like_vehicle()
        assert v.max_speed_at_curvature(0.0) == 95.0

    def test_circle_of_radius_R_gives_sqrt_grip_R(self) -> None:
        """For curvature 1/R, max speed equals sqrt(max_grip * R)."""
        v = _f1_like_vehicle()  # max_grip = 30 m/s^2
        # Pick R = 100 m. Expected: sqrt(30 * 100) = sqrt(3000) ≈ 54.77 m/s.
        # This is below max_speed (95), so grip is the binding constraint.
        radius = 100.0
        kappa = 1.0 / radius
        expected = math.sqrt(v.max_grip * radius)
        assert v.max_speed_at_curvature(kappa) == pytest.approx(expected)

    def test_capped_by_max_speed_on_gentle_curve(self) -> None:
        """A very gentle curve where grip allows > max_speed gets capped."""
        v = _f1_like_vehicle()  # max_grip = 30, max_speed = 95
        # R = 1000 m → grip-limited speed is sqrt(30 * 1000) ≈ 173 m/s,
        # well above max_speed = 95. The cap should bind.
        kappa = 1.0 / 1000.0
        assert v.max_speed_at_curvature(kappa) == 95.0

    def test_sign_of_curvature_does_not_matter(self) -> None:
        """Left vs right turns of equal magnitude give equal speed limits."""
        v = _f1_like_vehicle()
        assert v.max_speed_at_curvature(0.01) == v.max_speed_at_curvature(
            -0.01
        )

    def test_tighter_curve_gives_lower_speed(self) -> None:
        """Increasing curvature monotonically decreases the speed limit."""
        v = _f1_like_vehicle()
        speeds = [
            v.max_speed_at_curvature(kappa)
            for kappa in (0.005, 0.01, 0.02, 0.05, 0.1)
        ]
        # Strictly decreasing
        assert all(speeds[i] > speeds[i + 1] for i in range(len(speeds) - 1))


# Friction-circle acceleration tests
class TestMaxLongitudinalAcceleration:
    """Tests for the longitudinal acceleration with friction-circle."""

    def test_no_lateral_grip_used_returns_engine_cap_when_engine_limited(
        self,
    ) -> None:
        """With zero lateral g, the engine cap is the binding constraint."""
        v = _f1_like_vehicle()
        # max_grip=30, max_engine=15. With no cornering, tires allow 30
        # but engine only delivers 15. Engine binds.
        assert v.max_longitudinal_acceleration(50.0, 0.0) == 15.0

    def test_no_lateral_grip_used_returns_grip_cap_when_grip_limited(
        self,
    ) -> None:
        """No cornering: if engine > grip, grip is the binding cap."""
        v = PointMassVehicle(
            mass=1000.0,
            max_grip=10.0,
            max_engine_acceleration=20.0,  # higher than grip
            max_brake_deceleration=20.0,
            max_speed_value=60.0,
        )
        assert v.max_longitudinal_acceleration(30.0, 0.0) == 10.0

    def test_friction_circle_at_full_lateral_grip_gives_zero(self) -> None:
        """At lateral g = max_grip, no longitudinal grip remains."""
        v = _f1_like_vehicle()
        assert v.max_longitudinal_acceleration(50.0, v.max_grip) == 0.0

    def test_friction_circle_at_half_grip_gives_correct_remainder(
        self,
    ) -> None:
        """Lateral g = G*sin(theta) -> longitudinal max = G*cos(theta)."""
        # Vehicle where engine is not the constraint.
        v = PointMassVehicle(
            mass=1000.0,
            max_grip=10.0,
            max_engine_acceleration=100.0,  # huge: tires bind
            max_brake_deceleration=100.0,
            max_speed_value=200.0,
        )
        # At lateral g = 6, expected longitudinal = sqrt(100 - 36) = 8.
        assert v.max_longitudinal_acceleration(30.0, 6.0) == pytest.approx(8.0)

    def test_excessive_lateral_grip_returns_zero(self) -> None:
        """Lateral g exceeding max_grip means no remainder."""
        v = _f1_like_vehicle()  # max_grip = 30
        assert v.max_longitudinal_acceleration(30.0, 999.0) == 0.0

    def test_at_max_speed_returns_zero(self) -> None:
        """At top speed the vehicle cannot accelerate further."""
        v = _f1_like_vehicle()  # max_speed = 95
        assert v.max_longitudinal_acceleration(95.0, 0.0) == 0.0

    def test_above_max_speed_returns_zero(self) -> None:
        """Above top speed (e.g. downhill) the vehicle cannot accelerate."""
        v = _f1_like_vehicle()
        assert v.max_longitudinal_acceleration(100.0, 0.0) == 0.0


# Friction-circle braking tests
class TestMaxLongitudinalDeceleration:
    """Tests for the braking constraint with friction-circle coupling."""

    def test_no_lateral_returns_brake_cap_when_brake_limited(self) -> None:
        """No cornering: brake cap binds when brakes are weaker than tires."""
        v = _road_car()  # grip=10, brake=8
        assert v.max_longitudinal_deceleration(30.0, 0.0) == 8.0

    def test_no_lateral_returns_grip_cap_when_grip_limited(self) -> None:
        """No cornering: grip binds when brakes are stronger than tires."""
        v = _f1_like_vehicle()  # grip=30, brake=50
        assert v.max_longitudinal_deceleration(50.0, 0.0) == 30.0

    def test_friction_circle_at_half_grip(self) -> None:
        """Braking remainder under cornering: sqrt(G^2 - a_lat^2)."""
        v = PointMassVehicle(
            mass=1000.0,
            max_grip=10.0,
            max_engine_acceleration=100.0,
            max_brake_deceleration=100.0,  # huge: tires bind
            max_speed_value=200.0,
        )
        # Same geometry as before: a_lat = 6, expected = sqrt(100-36) = 8.
        assert v.max_longitudinal_deceleration(30.0, 6.0) == pytest.approx(8.0)

    def test_full_lateral_grip_gives_zero_braking(self) -> None:
        """At max lateral g, no braking grip is available."""
        v = _f1_like_vehicle()
        assert v.max_longitudinal_deceleration(50.0, v.max_grip) == 0.0


# "Interaction" tests
class TestPhysicsConsistency:
    """Tests exercising the model from a higher-level physics view."""

    def test_circle_constant_radius_lap_time_matches_analytical(self) -> None:
        """Steady-state lap time on a circle matches 2*pi*R / v_max formula."""
        # On a circular track of radius R driven at the cornering speed
        # limit, the lap time is the perimeter divided by the speed.
        v = _f1_like_vehicle()
        radius = 50.0
        v_max = v.max_speed_at_curvature(1.0 / radius)
        expected_lap_time = 2 * math.pi * radius / v_max
        # The cornering-limited speed is sqrt(max_grip * R) = sqrt(1500),
        # so expected lap time is 2*pi*R / sqrt(max_grip * R)
        #                       = 2*pi*sqrt(R / max_grip).
        analytical = 2 * math.pi * math.sqrt(radius / v.max_grip)
        assert expected_lap_time == pytest.approx(analytical)

    def test_acceleration_and_braking_symmetric_under_grip_limit(
        self,
    ) -> None:
        """When tires bind both, accel and braking remainders are equal."""
        # Vehicle with engine and brakes both stronger than tires.
        v = PointMassVehicle(
            mass=1000.0,
            max_grip=10.0,
            max_engine_acceleration=100.0,
            max_brake_deceleration=100.0,
            max_speed_value=200.0,
        )
        a_lat = 5.0
        accel = v.max_longitudinal_acceleration(50.0, a_lat)
        brake = v.max_longitudinal_deceleration(50.0, a_lat)
        assert accel == pytest.approx(brake)
