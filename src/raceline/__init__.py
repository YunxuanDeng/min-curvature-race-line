"""min-curvature-race-line: minimum-curvature racing line optimization."""

from raceline.optimizer import RacingLine, optimize_line
from raceline.point_mass import PointMassVehicle
from raceline.simulator import LapResult, simulate_lap
from raceline.speed_profile import SpeedProfile, compute_speed_profile
from raceline.track import Track
from raceline.vehicle import VehicleModel

__version__ = "0.1.0"

__all__ = [
    "LapResult",
    "PointMassVehicle",
    "RacingLine",
    "SpeedProfile",
    "Track",
    "VehicleModel",
    "compute_speed_profile",
    "optimize_line",
    "simulate_lap",
]
