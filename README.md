# min-curvature-race-line
Python library to find the minimum curvature racing line given track layout and basic vehicle parameters

On a racetrack, the path a car takes through a corner has a large effect on lap time. The *minimum-curvature line* is a good approximation to the optimal racing line as it minimizes the integral of squared path curvature subject to staying inside the track boundaries. Because maximum cornering speed scales with the square root of the turn radius, minimizing curvature approximates maximizing average speed, and the resulting line visibly resembles the late-apex lines that skilled racing drivers use.

This library implements this optimization from scratch, along with the
supporting machinery needed to turn a track description and a vehicle model into a lap time estimate:

- A `Track` representation that loads a centerline and track widths from a CSV file, resamples to uniform arc length, and computes curvature.
- A `VehicleModel` abstraction with a point-mass + friction-circle
  implementation.
- A minimum-curvature path optimizer formulated as a quadratic program.
- A forward-backward speed-profile computation that produces a speed trace and lap time for a given path and vehicle.
- A command-line (can change to more visual interface time permitting) interface for running the full pipeline on a track file.

## Architecture
