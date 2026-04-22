"""Vehicle model interface.

Defines the ``VehicleModel`` abstract base class, which specifies the
interface that concrete vehicle models must implement for use by the
speed-profile computation and the racing-line optimizer.

Concrete implementations (see ``point_mass.py``) decide how to compute these
values from their own physics using a simple friction-circle model. Richer
models can further account for things like weight transfer, aerodynamic
downforce, or engine torque curves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class VehicleModel(ABC):
    """Abstract interface for a vehicle used in speed-profile computation.

    A ``VehicleModel`` answers three questions at any point along a racing
    line:
    1. What is the maximum speed at which the vehicle can follow a curve of
       a given curvature without exceeding its grip limit? (Cornering speed
       limit.)
    2. At the current speed, how fast can the vehicle accelerate
       longitudinally?
    3. At the current speed, how fast can the vehicle decelerate by braking?

    The speed-profile computation calls these methods at every station
    along the path. The optimizer uses the same model when evaluating how
    different candidate racing lines trade off path length against cornering
    speed.

    Units:
    - Curvature: 1/m
    - Speed: m/s
    - Acceleration, deceleration: m/s^2

    Note:
        Curvature is the reciprocal of the turn radius. A straight has
        curvature 0; a circle of radius R has curvature 1/R.
    """

    @property
    @abstractmethod
    def max_speed(self) -> float:
        """Top speed of the vehicle (m/s).

        The speed-profile computation caps all speeds at this value, even
        on straights where grip and power would otherwise allow more. This
        represents the engine/drag-limited top speed.
        """

    @abstractmethod
    def max_speed_at_curvature(self, curvature: float) -> float:
        """Return the maximum speed at which the vehicle can sustain a curve.

        This is the cornering speed limit: the highest speed at which the
        lateral acceleration required to follow a curve of the given
        curvature does not exceed the vehicle's grip limit.

        Args:
            curvature: Signed curvature of the path at this station (1/m).
                Only the magnitude matters for grip; sign indicates turn
                direction and is ignored here.

        Returns:
            Maximum sustainable speed in m/s. For a straight
            (``curvature == 0``) this is ``max_speed``.
        """

    @abstractmethod
    def max_longitudinal_acceleration(
        self, speed: float, lateral_acceleration: float
    ) -> float:
        """Return the maximum longitudinal acceleration available.

        Accounts for the friction-circle coupling: when the vehicle is
        cornering at a given lateral acceleration, some of its grip budget
        is already “spent”, leaving less available for forward acceleration.

        Args:
            speed: Current speed of the vehicle (m/s).
            lateral_acceleration: Lateral acceleration currently being used
                by the vehicle to follow the path (m/s^2). Must be
                non-negative.

        Returns:
            Maximum additional forward acceleration the vehicle can produce
            at this speed and lateral-grip usage (m/s^2). Returns 0 if the
            vehicle is already using all of its grip for cornering.
        """

    @abstractmethod
    def max_longitudinal_deceleration(
        self, speed: float, lateral_acceleration: float
    ) -> float:
        """Return the maximum longitudinal deceleration (braking) available.

        Like ``max_longitudinal_acceleration``, accounts for friction-circle
        coupling: cornering consumes part of the grip budget and reduces
        the deceleration available from the brakes.

        Args:
            speed: Current speed of the vehicle (m/s).
            lateral_acceleration: Lateral acceleration currently being used
                by the vehicle (m/s^2). Must be non-negative.

        Returns:
            Maximum braking deceleration available at this speed and
            lateral-grip usage (m/s^2). Returned as a positive value.
        """
