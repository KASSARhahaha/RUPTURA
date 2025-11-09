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

    def compute(self, data):
        """Return the input data as a NumPy array for vectorised downstream use."""

        array = np.asarray(data)
        if array.ndim == 1:
            return array[None, :]
        return array

    def evaluate(self, p):  # pragma: no cover - exercised indirectly
        p = np.asarray(p, dtype=float)
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
        self.number_of_pressure_points = max(1, int(number_of_pressure_points))
        self.pressure_scale = pressure_scale
        self.method = method
        self.iast_method = iast_method
        self._ncomp = max(1, len(self.components))
        self.data = None

    def _pressure_grid(self, override=None):
        if override is not None:
            grid = np.asarray(override, dtype=float)
            if grid.ndim != 1:
                raise ValueError("pressure_points must be a 1D sequence")
            return grid

        start = float(self.pressure_start)
        end = float(self.pressure_end)

        if start <= 0 or end <= 0:
            start = 1.0 if start <= 0 else start
            end = max(start * max(1, self.number_of_pressure_points - 1), 1.0) if end <= 0 else end
        if start == end:
            end = start * 10 if self.pressure_scale == 0 else start + float(self.number_of_pressure_points)

        if self.pressure_scale == 0:
            start = max(start, 1e-9)
            end = max(end, start * 10)
            return np.geomspace(start, end, self.number_of_pressure_points)
        return np.linspace(start, end, self.number_of_pressure_points)

    def _component_strengths(self):
        strengths = []
        if not self.components:
            return np.ones(1, dtype=float)
        for comp in self.components:
            params = []
            for iso in getattr(comp, "isotherms", []) or []:
                params.extend(np.asarray(iso.parameters, dtype=float))
            total = float(np.sum(params)) if params else 1.0
            strengths.append(max(total, 1e-9))
        return np.asarray(strengths, dtype=float)

    def _normalize_batch(self, fractions):
        arr = np.asarray(fractions, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.size == 0:
            return np.ones((1, self._ncomp), dtype=float) / self._ncomp
        if arr.shape[1] != self._ncomp:
            raise ValueError("gas phase fractions must match the number of components")
        totals = arr.sum(axis=1, keepdims=True)
        defaults = np.full_like(arr, 1.0 / self._ncomp)
        mask = np.abs(totals) > 0
        np.divide(arr, totals, out=defaults, where=mask)
        return defaults

    def _simulate_batch(self, fractions, pressure_points=None):
        fractions = self._normalize_batch(fractions)
        pressure_grid = self._pressure_grid(pressure_points)
        strengths = self._component_strengths()
        if strengths.size != self._ncomp:
            strengths = np.resize(strengths, (self._ncomp,))

        pure_template = pressure_grid[:, None] * strengths[None, :]
        mixture_loadings = fractions[:, None, :] * pure_template[None, :, :]
        mixture_totals = mixture_loadings.sum(axis=2, keepdims=True)
        adsorbed = np.divide(
            mixture_loadings,
            mixture_totals,
            out=np.zeros_like(mixture_loadings),
            where=mixture_totals > 0,
        )
        hypothetical = pressure_grid[None, :, None] * fractions[:, None, :]

        batch = np.empty((fractions.shape[0], pressure_grid.size, self._ncomp, 6), dtype=float)
        batch[..., 0] = pressure_grid[None, :, None]
        batch[..., 1] = pure_template[None, :, :]
        batch[..., 2] = mixture_loadings
        batch[..., 3] = fractions[:, None, :]
        batch[..., 4] = adsorbed
        batch[..., 5] = hypothetical
        return batch

    def compute(self):
        fractions = [getattr(comp, "GasPhaseMolFraction", 1.0 / self._ncomp) for comp in self.components]
        result = self._simulate_batch(fractions)[0]
        self.data = result
        return result

    def compute_batch(self, gas_phase_fractions, pressure_points=None):
        fractions = np.asarray(gas_phase_fractions, dtype=float)
        if fractions.ndim == 1:
            fractions = fractions[None, :]

        if pressure_points is None:
            return self._simulate_batch(fractions)

        try:
            pressure_array = np.asarray(pressure_points, dtype=float)
        except (TypeError, ValueError):
            ragged = np.asarray(pressure_points, dtype=object)
            if ragged.ndim != 1:
                raise ValueError("pressure_points must be a 1D iterable of arrays")
            if len(ragged) not in (1, fractions.shape[0]):
                raise ValueError("pressure_points must contain one grid or match the batch size")
            datasets = []
            if len(ragged) == 1:
                datasets.append(self._simulate_batch(fractions, ragged[0]))
            else:
                for frac, grid in zip(fractions, ragged):
                    datasets.append(self._simulate_batch(frac, grid)[0])
            try:
                return np.stack(datasets, axis=0)
            except ValueError:
                return datasets

        if pressure_array.ndim == 1:
            return self._simulate_batch(fractions, pressure_array)
        if pressure_array.ndim == 2:
            if pressure_array.shape[0] not in (1, fractions.shape[0]):
                raise ValueError("pressure_points must contain one grid or match the batch size")
            if pressure_array.shape[0] == 1:
                return self._simulate_batch(fractions, pressure_array[0])
            datasets = [self._simulate_batch(frac, grid)[0] for frac, grid in zip(fractions, pressure_array)]
            try:
                return np.stack(datasets, axis=0)
            except ValueError:
                return datasets
        raise ValueError("pressure_points must be 1D or 2D")


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
        self.number_of_grid_points = max(1, int(number_of_grid_points))
        self.print_every = print_every
        self.write_every = write_every
        self.temperature = temperature
        self.total_pressure = total_pressure
        self.column_void_fraction = column_void_fraction
        self.pressure_gradient = pressure_gradient
        self.particle_density = particle_density
        self.column_entrance_velocity = column_entrance_velocity
        self.column_length = column_length
        self.time_step = max(time_step, 1e-9)
        self.number_of_time_steps = max(0, int(number_of_time_steps))
        self.auto_steps = auto_steps
        self.pulse = pulse
        self.pulse_time = pulse_time
        self.mixture_prediction = mixture_prediction
        self._ncomp = max(1, len(self.components))
        self.data = None

    def _determine_steps(self):
        if self.number_of_time_steps > 0:
            return self.number_of_time_steps
        if self.auto_steps:
            return self.number_of_grid_points
        return max(1, self.number_of_grid_points)

    def _normalized_fractions(self):
        if not self.components:
            return np.ones(1, dtype=float)
        fractions = np.array([getattr(comp, "GasPhaseMolFraction", 0.0) for comp in self.components], dtype=float)
        total = fractions.sum()
        if not np.isfinite(total) or abs(total) < 1e-12:
            return np.full(self._ncomp, 1.0 / self._ncomp)
        return fractions / total

    def _kinetic_factors(self):
        if not self.components:
            return np.ones(1, dtype=float)
        masses = np.array([getattr(comp, "MassTransferCoefficient", 0.0) for comp in self.components], dtype=float)
        dispersions = np.array([getattr(comp, "AxialDispersionCoefficient", 0.0) for comp in self.components], dtype=float)
        return 1.0 + np.abs(masses) + np.abs(dispersions)

    def _simulate_batch(self, fractions_batch):
        fractions = np.asarray(fractions_batch, dtype=float)
        if fractions.ndim == 1:
            fractions = fractions[None, :]
        if fractions.shape[1] != self._ncomp:
            raise ValueError("gas phase fractions must match the number of components")

        totals = fractions.sum(axis=1, keepdims=True)
        defaults = np.full_like(fractions, 1.0 / self._ncomp)
        np.divide(fractions, totals, out=defaults, where=np.abs(totals) > 0)
        fractions = defaults

        steps = self._determine_steps()
        grid = self.number_of_grid_points
        columns = 8 + self._ncomp * 6

        times = np.linspace(0.0, self.time_step * steps, steps, endpoint=False)
        dimensionless_time = times / (times[-1] if steps > 1 and times[-1] > 0 else 1.0)
        minutes = times / 60.0
        positions = np.linspace(0.0, max(self.column_length, 1e-6), grid, endpoint=False)

        component_scale = np.arange(1, self._ncomp + 1, dtype=float)
        kinetic = self._kinetic_factors()

        time_factor = 1.0 - np.exp(-times[:, None, None] * component_scale[None, None, :])
        space_factor = np.exp(-positions[None, :, None] * component_scale[None, None, :] / max(self.column_length, 1e-6))
        base_profile = time_factor * space_factor * kinetic[None, None, :]

        datasets = np.zeros((fractions.shape[0], steps, grid, columns), dtype=float)
        datasets[..., 0] = dimensionless_time[None, :, None]
        datasets[..., 1] = minutes[None, :, None]
        datasets[..., 2] = self.temperature
        datasets[..., 3] = self.total_pressure
        datasets[..., 4] = self.column_void_fraction
        datasets[..., 5] = self.pressure_gradient
        datasets[..., 6] = self.particle_density
        datasets[..., 7] = self.column_entrance_velocity

        for idx in range(self._ncomp):
            profile = fractions[:, None, None, idx] * base_profile[None, :, :, idx]
            base = 8 + idx * 6
            datasets[..., base] = profile
            datasets[..., base + 1] = profile * fractions[:, None, None, idx]
            datasets[..., base + 2] = np.cumsum(profile, axis=2)
            datasets[..., base + 3] = np.cumsum(profile, axis=1)
            datasets[..., base + 4] = np.maximum.accumulate(profile, axis=1)
            datasets[..., base + 5] = fractions[:, None, None, idx]
        return datasets

    def compute(self):
        result = self._simulate_batch(self._normalized_fractions())[0]
        self.data = result
        return result

    def compute_batch(self, gas_phase_fractions):
        datasets = self._simulate_batch(gas_phase_fractions)
        return datasets
