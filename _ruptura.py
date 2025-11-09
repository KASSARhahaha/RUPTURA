"""Pure Python fallbacks that emulate the compiled `_ruptura` bindings used by the
ruptura Python API.

The real project ships a pybind11 extension that provides these classes.  The
kata environment does not build native extensions, so we supply small Python
stand-ins exposing the same surface that the unit tests exercise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Isotherm:
    """Lightweight container matching the C++ isotherm object."""

    name: str
    parameters: Sequence[float]
    size: int


@dataclass
class Component:
    """Mimic the attributes provided by the native Component class."""

    idx: int
    MoleculeName: str
    isotherms: Sequence[Isotherm]
    GasPhaseMolFraction: float
    MassTransferCoefficient: float
    AxialDispersionCoefficient: float
    CarrierGas: bool


class Fitting:
    """Simplified fitting wrapper used by the Python facade."""

    def __init__(self, display_name: str, components: Iterable[Component], pressure_scale: int) -> None:
        self.display_name = display_name
        self.components = list(components)
        self.pressure_scale = pressure_scale

    def compute(self, data):  # pragma: no cover - patched in tests
        return np.asarray(data)

    def evaluate(self, p):  # pragma: no cover - exercised indirectly
        p = np.asarray(p)
        if p.ndim == 1:
            return np.tile(p[:, None], (1, max(1, len(self.components))))
        return p


class MixturePrediction:
    """Return structured arrays describing mixture predictions."""

    def __init__(
        self,
        display_name: str,
        components: Iterable[Component],
        has_carrier_gas: int,
        carrier_gas_index: int,
        temperature: float,
        pressure_start: float,
        pressure_end: float,
        number_of_pressure_points: int,
        pressure_scale: int,
        method: int,
        iast_method: int,
    ) -> None:
        self.display_name = display_name
        self.components = list(components)
        self.has_carrier_gas = has_carrier_gas
        self.carrier_gas_index = carrier_gas_index
        self.temperature = temperature
        self.pressure_start = pressure_start
        self.pressure_end = pressure_end
        self.number_of_pressure_points = number_of_pressure_points
        self.pressure_scale = pressure_scale
        self.method = method
        self.iast_method = iast_method

    def compute(self):  # pragma: no cover - patched in tests
        ncomp = max(1, len(self.components))
        return np.zeros((self.number_of_pressure_points, ncomp, 6), dtype=float)


class Breakthrough:
    """Produce placeholder breakthrough curves for the API surface."""

    def __init__(
        self,
        display_name: str,
        components: Iterable[Component],
        carrier_gas_index: int,
        number_of_grid_points: int,
        print_every: int,
        write_every: int,
        temperature: float,
        total_pressure: float,
        column_void_fraction: float,
        pressure_gradient: float,
        particle_density: float,
        column_entrance_velocity: float,
        column_length: float,
        time_step: float,
        number_of_time_steps: int,
        auto_steps: bool,
        pulse: bool,
        pulse_time: float,
        mixture_prediction: MixturePrediction,
    ) -> None:
        self.display_name = display_name
        self.components = list(components)
        self.carrier_gas_index = carrier_gas_index
        self.number_of_grid_points = number_of_grid_points
        self.print_every = print_every
        self.write_every = write_every
        self.temperature = temperature
        self.total_pressure = total_pressure
        self.column_void_fraction = column_void_fraction
        self.pressure_gradient = pressure_gradient
        self.particle_density = particle_density
        self.column_entrance_velocity = column_entrance_velocity
        self.column_length = column_length
        self.time_step = time_step
        self.number_of_time_steps = number_of_time_steps
        self.auto_steps = auto_steps
        self.pulse = pulse
        self.pulse_time = pulse_time
        self.mixture_prediction = mixture_prediction

    def compute(self):  # pragma: no cover - patched in tests
        steps = self.number_of_time_steps or self.number_of_grid_points
        ncomp = max(1, len(self.components))
        columns = 8 + ncomp * 6
        return np.zeros((steps, self.number_of_grid_points, columns), dtype=float)
