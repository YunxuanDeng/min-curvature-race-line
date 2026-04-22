"""Unit tests for the VehicleModel abstract base class."""

from __future__ import annotations

import pytest

from raceline import VehicleModel


class _FakeVehicle(VehicleModel):
    """Minimal concrete subclass used to exercise the ABC contract."""

    @property
    def max_speed(self) -> float:
        return 100.0  # 100 m/s is equivalent to about 223 mph

    def max_speed_at_curvature(self, curvature: float) -> float:
        return 50.0

    def max_longitudinal_acceleration(
        self, speed: float, lateral_acceleration: float
    ) -> float:
        return 20.0  # 20 m/s^2 is equivalent to about 2 g units

    def max_longitudinal_deceleration(
        self, speed: float, lateral_acceleration: float
    ) -> float:
        return 20.0


class TestVehicleModelIsAbstract:
    """Tests that VehicleModel behaves as a proper abstract base class."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """Instantiating ``VehicleModel`` directly raises TypeError."""
        with pytest.raises(TypeError, match="abstract"):
            VehicleModel()  # type: ignore[abstract]

    def test_subclass_must_implement_max_speed(self) -> None:
        """A subclass missing ``max_speed`` cannot be instantiated."""

        class _Incomplete(VehicleModel):
            def max_speed_at_curvature(self, curvature: float) -> float:
                return 0.0

            def max_longitudinal_acceleration(
                self, speed: float, lateral_acceleration: float
            ) -> float:
                return 0.0

            def max_longitudinal_deceleration(
                self, speed: float, lateral_acceleration: float
            ) -> float:
                return 0.0

        with pytest.raises(TypeError, match="abstract"):
            _Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_max_speed_at_curvature(self) -> None:
        """A subclass without ``max_speed_at_curvature`` cannot instantiate."""

        class _Incomplete(VehicleModel):
            @property
            def max_speed(self) -> float:
                return 100.0

            def max_longitudinal_acceleration(
                self, speed: float, lateral_acceleration: float
            ) -> float:
                return 0.0

            def max_longitudinal_deceleration(
                self, speed: float, lateral_acceleration: float
            ) -> float:
                return 0.0

        with pytest.raises(TypeError, match="abstract"):
            _Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_max_long_acceleration(self) -> None:
        """A subclass missing the acceleration method cannot instantiate."""

        class _Incomplete(VehicleModel):
            @property
            def max_speed(self) -> float:
                return 100.0

            def max_speed_at_curvature(self, curvature: float) -> float:
                return 0.0

            def max_longitudinal_deceleration(
                self, speed: float, lateral_acceleration: float
            ) -> float:
                return 0.0

        with pytest.raises(TypeError, match="abstract"):
            _Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_max_long_deceleration(self) -> None:
        """A subclass missing the deceleration method cannot instantiate."""

        class _Incomplete(VehicleModel):
            @property
            def max_speed(self) -> float:
                return 100.0

            def max_speed_at_curvature(self, curvature: float) -> float:
                return 0.0

            def max_longitudinal_acceleration(
                self, speed: float, lateral_acceleration: float
            ) -> float:
                return 0.0

        with pytest.raises(TypeError, match="abstract"):
            _Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self) -> None:
        """A subclass implementing all abstract methods instantiates OK."""
        vehicle = _FakeVehicle()
        assert isinstance(vehicle, VehicleModel)

    def test_complete_subclass_methods_callable(self) -> None:
        """A concrete subclass's methods return the expected values."""
        vehicle = _FakeVehicle()
        assert vehicle.max_speed == 100.0
        assert vehicle.max_speed_at_curvature(0.01) == 50.0
        assert vehicle.max_longitudinal_acceleration(30.0, 5.0) == 20.0
        assert vehicle.max_longitudinal_deceleration(30.0, 5.0) == 20.0
