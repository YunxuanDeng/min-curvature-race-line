"""Point-mass vehicle model with friction-circle physics.

A ``PointMassVehicle`` treats the car as a single point with a tire grip
limit that constrains the total acceleration vector. We assume that the
combined lateral and longitudinal acceleration cannot exceed the grip limit:

    sqrt(a_lat^2 + a_long^2) <= max_grip

Power and brake limits are independent caps on top of the grip constraint:
the actual available acceleration or deceleration is the minimum of
"what the tires allow" and "what the engine/brakes can deliver".

This model captures the essential trade-off behind racing-line strategy
(trail braking, late apex) without modeling complicated physics like weight
transfer, downforce, or suspension dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from raceline.vehicle import VehicleModel


@dataclass
class PointMassVehicle(VehicleModel):
    """A vehicle modeled as a point mass with a friction-circle grip limit.

    Attributes:
        mass: Vehicle mass in kg. Stored for documentation and downstream
            extensions; does not appear in any of the kinematic formulas
            below (mass cancels out under the friction-circle assumption).
        max_grip: Maximum total acceleration the tires can provide (m/s^2),
            i.e. mu * g where mu is the tire-pavement friction coefficient and
            g is gravitational acceleration. Typical values: ~10 for a
            road car (1g), ~25 for a sports car (2.5g), ~35+ for an
            F1 car with downforce treated as constant.
        max_engine_acceleration: Engine/power-limited forward acceleration cap
            (m/s^2). The actual forward acceleration is the minimum of
            this and the friction-circle remainder. Set this large
            (>= max_grip) to model a vehicle that is always tire-limited
            in acceleration.
        max_brake_deceleration: Brake-limited deceleration cap (m/s^2),
            stored as a positive number. Like the engine cap, the actual
            available deceleration is the minimum of this and the
            friction-circle remainder.
        max_speed: Top speed in m/s, set by aerodynamic drag and gearing.

    Raises:
        ValueError: If any parameter is non-positive.
    """

    mass: float
    max_grip: float
    max_engine_acceleration: float
    max_brake_deceleration: float
    max_speed_value: float  # backing field for the max_speed property

    def __post_init__(self) -> None:
        """Validate that all physical parameters are strictly positive."""
        for name, value in [
            ("mass", self.mass),
            ("max_grip", self.max_grip),
            ("max_engine_acceleration", self.max_engine_acceleration),
            ("max_brake_deceleration", self.max_brake_deceleration),
            ("max_speed_value", self.max_speed_value),
        ]:
            if value <= 0:
                raise ValueError(
                    f"{name} must be strictly positive, got {value}"
                )

    @property
    def max_speed(self) -> float:
        """Top speed of the vehicle (m/s)."""
        return self.max_speed_value

    def max_speed_at_curvature(self, curvature: float) -> float:
        """Return the maximum sustainable cornering speed.

        On a straight, no or very little lateral acceleration is required, so
        the limiting factor is ``max_speed``.

        On a curve of curvature ``kappa``, following the curve at speed
        ``v`` requires lateral acceleration ``v^2 * |kappa|``. Setting
        this equal to the grip limit gives:

            v_max = sqrt(max_grip / |kappa|)

        The returned value is then capped at ``max_speed``.

        Args:
            curvature: Signed curvature of the path (1/m). The sign is
                ignored; only magnitude affects the grip requirement.

        Returns:
            Maximum sustainable speed at this curvature (m/s).
        """
        kappa = abs(curvature)
        if kappa == 0.0:  # Straight line case
            return self.max_speed_value
        v_grip = math.sqrt(self.max_grip / kappa)  # Curvature
        return min(v_grip, self.max_speed_value)

    def max_longitudinal_acceleration(
        self, speed: float, lateral_acceleration: float
    ) -> float:
        """Return the maximum longitudinal acceleration available.

        Computed as the minimum of:

        - The friction-circle remainder:
          ``sqrt(max_grip^2 - lateral_acceleration^2)`` if cornering is
          within the grip budget; otherwise zero.
        - The engine/power unit acceleration cap.

        Returns 0 if the vehicle is at or above its top speed (cannot
        accelerate further).

        Args:
            speed: Current speed of the vehicle (m/s).
            lateral_acceleration: Lateral acceleration in use to follow
                the path (m/s^2). Must be non-negative (magnitude only).

        Returns:
            Maximum forward acceleration available (m/s^2), >= 0.
        """
        if speed >= self.max_speed_value:
            return 0.0
        tire_remainder = self._friction_circle_remainder(lateral_acceleration)
        return min(tire_remainder, self.max_engine_acceleration)

    def max_longitudinal_deceleration(
        self, speed: float, lateral_acceleration: float
    ) -> float:
        """Return the maximum longitudinal deceleration (braking) available.

        Computed as the minimum of:

        - The friction-circle remainder (same formula as for acceleration).
        - The brake deceleration cap.

        Args:
            speed: Current speed of the vehicle (m/s). Currently unused
                (brakes are modeled as speed-independent), but accepted
                to match the ``VehicleModel`` interface.
            lateral_acceleration: Lateral acceleration in use to follow
                the path (m/s^2). Must be non-negative.

        Returns:
            Maximum braking deceleration available (m/s^2), >= 0.
        """
        del speed  # unused in this simple model
        tire_remainder = self._friction_circle_remainder(lateral_acceleration)
        return min(tire_remainder, self.max_brake_deceleration)

    def _friction_circle_remainder(self, lateral_acceleration: float) -> float:
        """Return the longitudinal grip remaining given lateral grip used.

        Solves ``sqrt(a_lat^2 + a_long^2) <= max_grip`` for the maximum
        ``a_long`` given a fixed ``a_lat``. If the requested lateral
        acceleration already exceeds the grip budget, returns 0.
        """
        a_lat = abs(lateral_acceleration)
        if a_lat >= self.max_grip:
            return 0.0
        return math.sqrt(self.max_grip**2 - a_lat**2)
