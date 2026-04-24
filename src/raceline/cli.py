"""Command-line interface for raceline package without writing/modifying code.

Usage:
    raceline optimize --track TRACK.csv [options]
    raceline simulate --track TRACK.csv [options]

The CLI is a thin wrapper over the library. All business logic
lives in the other modules.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="raceline",
        description="Minimum-curvature racing line optimizer.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- optimize subcommand ---
    opt = sub.add_parser(
        "optimize",
        help="Optimize and simulate a lap.",
    )
    opt.add_argument(
        "--track",
        required=True,
        help="Path to a TUM-format track CSV.",
    )
    opt.add_argument(
        "--spacing",
        type=float,
        default=5.0,
        help="Resample spacing in meters (default: 5).",
    )
    opt.add_argument(
        "--mass",
        type=float,
        default=740,
        help="Vehicle mass in kg (default: 740).",
    )
    opt.add_argument(
        "--grip",
        type=float,
        default=30.0,
        help="Max grip in m/s^2 (default: 30).",
    )
    opt.add_argument(
        "--engine",
        type=float,
        default=15.0,
        help="Max engine accel in m/s^2 (default: 15).",
    )
    opt.add_argument(
        "--brake",
        type=float,
        default=50.0,
        help="Max brake decel in m/s^2 (default: 50).",
    )
    opt.add_argument(
        "--top-speed",
        type=float,
        default=95.0,
        help="Top speed in m/s (default: 95).",
    )
    opt.add_argument(
        "--length-weight",
        type=float,
        default=0.0001,
        help="Length regularizer weight (default: 0.0001).",
    )
    opt.add_argument(
        "--plot",
        default=None,
        help="Save summary plot to this path (e.g. out.png).",
    )
    opt.add_argument(
        "--no-optimize",
        action="store_true",
        help="Simulate on centerline only (no optimization).",
    )

    return parser


def main() -> None:
    """Entry point for the ``raceline`` command."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "optimize":
        _run_optimize(args)


def _run_optimize(args: argparse.Namespace) -> None:
    """Run the optimize subcommand."""
    from raceline import (
        PointMassVehicle,
        Track,
        simulate_lap,
    )

    track = Track.from_csv(args.track).resample(spacing=args.spacing)
    vehicle = PointMassVehicle(
        mass=args.mass,
        max_grip=args.grip,
        max_engine_acceleration=args.engine,
        max_brake_deceleration=args.brake,
        max_speed_value=args.top_speed,
    )

    do_optimize = not args.no_optimize
    result = simulate_lap(
        track,
        vehicle,
        optimize=do_optimize,
        length_weight=args.length_weight,
    )

    mode = "Optimized" if do_optimize else "Centerline"
    print(f"Track:     {args.track}")
    print(f"Stations:  {len(track)}")
    print(f"Length:    {track.total_length:.0f} m")
    print(f"Mode:      {mode}")
    print(f"Lap time:  {result.lap_time:.2f}s")
    print(f"Min speed: {result.speed_profile.min_speed * 3.6:.0f} km/h")
    print(
        f"Max speed: {result.speed_profile.max_speed_achieved * 3.6:.0f} km/h"
    )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        from raceline.plotting import plot_lap_summary

        fig = plot_lap_summary(track, result, vehicle=vehicle)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"Plot:      {args.plot}")


if __name__ == "__main__":
    main()
