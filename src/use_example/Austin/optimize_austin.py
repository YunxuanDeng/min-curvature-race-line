"""End-to-end example: optimize and simulate a lap on Austin COTA.

Usage:
    python src/use_example/Austin/optimize_austin.py

Requires a TUM-format track CSV at ``src/use_example/Austin/Austin.csv``.
Produces ``./austin_summary.png``.
"""

from pathlib import Path

from raceline import PointMassVehicle, Track, simulate_lap
from raceline.plotting import plot_lap_summary

# Paths
OUTPUT_DIR = Path("src/use_example/Austin")
OUTPUT_DIR.mkdir(exist_ok=True)
TRACK_PATH = Path("src/use_example/Austin/Austin.csv")

# Load and resample the track
track = Track.from_csv(TRACK_PATH).resample(spacing=5.0)
print(f"Track: {TRACK_PATH.stem}")
print(f"  {len(track)} stations, {track.total_length:.0f} m")

# Define your vehicle vehicle
vehicle = PointMassVehicle(
    mass=740,
    max_grip=30.0,
    max_engine_acceleration=15.0,
    max_brake_deceleration=50.0,
    max_speed_value=95.0,
)

# Baseline: centerline lap
result_cl = simulate_lap(track, vehicle, optimize=False)
print(f"\nCenterline lap: {result_cl.lap_time:.2f} s")
v_min = result_cl.speed_profile.min_speed * 3.6
v_max = result_cl.speed_profile.max_speed_achieved * 3.6
print(f"  Min speed: {v_min:.0f} km/h")
print(f"  Max speed: {v_max:.0f} km/h")

# Optimized: minimum-curvature racing line
result_opt = simulate_lap(track, vehicle, optimize=True)
improvement = result_cl.lap_time - result_opt.lap_time
print(f"\nOptimized lap: {result_opt.lap_time:.2f} s")
v_min = result_opt.speed_profile.min_speed * 3.6
v_max = result_opt.speed_profile.max_speed_achieved * 3.6
print(f"  Min speed: {v_min:.0f} km/h")
print(f"  Max speed: {v_max:.0f} km/h")
pct = improvement / result_cl.lap_time * 100
print(f"  Improvement: {improvement:.2f} s ({pct:.1f}%)")

# Generate summary plot
fig = plot_lap_summary(
    track, result_opt, vehicle_note="A F1-like car", vehicle=vehicle
)
out = OUTPUT_DIR / "austin_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
