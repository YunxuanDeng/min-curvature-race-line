"""Unit tests for the plotting module.

These are smoke tests: they verify that each plotting function
runs without errors and returns the expected matplotlib objects.
Visual correctness is not asserted (which requires human).

The ``Agg`` backend is used so no GUI window pops up during tests.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raceline import (
    PointMassVehicle,
    Track,
    simulate_lap,
)
from raceline.plotting import (
    plot_gg_diagram,
    plot_lap_summary,
    plot_racing_line,
    plot_speed_profile,
    plot_track,
)


# Fixtures
@pytest.fixture()
def circle_track() -> Track:
    """A simple circular track for testing."""
    n = 200
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cl = 50.0 * np.column_stack([np.cos(theta), np.sin(theta)])
    w = np.full(n, 5.0)
    return Track(
        centerline=cl,
        width_right=w,
        width_left=w.copy(),
        closed=True,
    )


@pytest.fixture()
def vehicle() -> PointMassVehicle:
    """A balanced test vehicle."""
    return PointMassVehicle(
        mass=1000.0,
        max_grip=10.0,
        max_engine_acceleration=5.0,
        max_brake_deceleration=8.0,
        max_speed_value=60.0,
    )


@pytest.fixture()
def _close_figures() -> Iterator[None]:
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


pytestmark = pytest.mark.usefixtures("_close_figures")


# plot_track
class TestPlotTrack:
    """Smoke tests for plot_track."""

    def test_returns_axes(self, circle_track: Track) -> None:
        """plot_track returns a matplotlib Axes."""
        ax = plot_track(circle_track)
        assert ax is not None

    def test_draws_on_provided_axes(self, circle_track: Track) -> None:
        """plot_track draws on a pre-existing axes."""
        _, ax = plt.subplots()
        returned = plot_track(circle_track, ax=ax)
        assert returned is ax

    def test_without_centerline(self, circle_track: Track) -> None:
        """plot_track works with show_centerline=False."""
        ax = plot_track(circle_track, show_centerline=False)
        assert ax is not None


# plot_racing_line
class TestPlotRacingLine:
    """Smoke tests for plot_racing_line."""

    def test_without_speeds(
        self,
        circle_track: Track,
        vehicle: PointMassVehicle,
    ) -> None:
        """Racing line without speed coloring renders OK."""
        result = simulate_lap(circle_track, vehicle, optimize=False)
        ax = plot_racing_line(circle_track, result.racing_line)
        assert ax is not None

    def test_with_speeds(
        self,
        circle_track: Track,
        vehicle: PointMassVehicle,
    ) -> None:
        """Racing line with speed coloring renders OK."""
        result = simulate_lap(circle_track, vehicle, optimize=False)
        ax = plot_racing_line(
            circle_track,
            result.racing_line,
            speeds=result.speed_profile.speeds,
        )
        assert ax is not None


# plot_speed_profile
class TestPlotSpeedProfile:
    """Smoke tests for plot_speed_profile."""

    def test_returns_axes(
        self,
        circle_track: Track,
        vehicle: PointMassVehicle,
    ) -> None:
        """plot_speed_profile returns axes."""
        result = simulate_lap(circle_track, vehicle, optimize=False)
        ax = plot_speed_profile(result)
        assert ax is not None

    def test_draws_on_provided_axes(
        self,
        circle_track: Track,
        vehicle: PointMassVehicle,
    ) -> None:
        """plot_speed_profile draws on a pre-existing axes."""
        _, ax = plt.subplots()
        result = simulate_lap(circle_track, vehicle, optimize=False)
        returned = plot_speed_profile(result, ax=ax)
        assert returned is ax


# plot_gg_diagram
class TestPlotGGDiagram:
    """Smoke tests for plot_gg_diagram."""

    def test_returns_axes(
        self,
        circle_track: Track,
        vehicle: PointMassVehicle,
    ) -> None:
        """plot_gg_diagram returns axes."""
        result = simulate_lap(circle_track, vehicle, optimize=False)
        ax = plot_gg_diagram(result)
        assert ax is not None

    def test_with_explicit_grip(
        self,
        circle_track: Track,
        vehicle: PointMassVehicle,
    ) -> None:
        """plot_gg_diagram with explicit grip_limit renders OK."""
        result = simulate_lap(circle_track, vehicle, optimize=False)
        ax = plot_gg_diagram(result, grip_limit=vehicle.max_grip)
        assert ax is not None

    def test_without_friction_circle(
        self,
        circle_track: Track,
        vehicle: PointMassVehicle,
    ) -> None:
        """plot_gg_diagram without friction circle overlay."""
        result = simulate_lap(circle_track, vehicle, optimize=False)
        ax = plot_gg_diagram(result, show_friction_circle=False)
        assert ax is not None


# plot_lap_summary
class TestPlotLapSummary:
    """Smoke tests for the combined summary figure."""

    def test_returns_figure(
        self,
        circle_track: Track,
        vehicle: PointMassVehicle,
    ) -> None:
        """plot_lap_summary returns a matplotlib Figure."""
        result = simulate_lap(circle_track, vehicle, optimize=False)
        fig = plot_lap_summary(circle_track, result)
        assert fig is not None
        assert len(fig.axes) >= 3  # track, gg, speed (+ colorbar)
