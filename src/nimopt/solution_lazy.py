"""Lazy-loading solution with memory-mapped storage."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .model import Model
    from .solvers.base import Solver


@dataclass
class LazySolution:
    """Solution with lazy-loaded memory-mapped arrays.

    Data stays on disk until accessed. Efficient for large models.
    """
    directory: Path
    objective_value: Optional[float] = None
    _var_meta: dict = field(default_factory=dict)
    _con_meta: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    def var(self, name: str) -> "LazyVariable":
        """Get variable solution (lazy-loaded)."""
        if name not in self._var_meta:
            raise KeyError(f"Variable '{name}' not in solution")
        if name not in self._cache:
            self._cache[name] = LazyVariable(self.directory, name, self._var_meta[name])
        return self._cache[name]

    def con(self, name: str) -> "LazyConstraint":
        """Get constraint solution (lazy-loaded)."""
        if name not in self._con_meta:
            raise KeyError(f"Constraint '{name}' not in solution")
        key = f"con_{name}"
        if key not in self._cache:
            meta = self._con_meta[name]
            self._cache[key] = LazyConstraint(self.directory, name, meta)
        return self._cache[key]

    @property
    def variables(self) -> list[str]:
        return list(self._var_meta.keys())

    @property
    def constraints(self) -> list[str]:
        return list(self._con_meta.keys())


@dataclass
class LazyVariable:
    """Lazy-loaded variable solution."""
    directory: Path
    name: str
    meta: dict
    _values: Optional[np.ndarray] = field(default=None, repr=False)
    _duals: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def dims(self) -> list[str]:
        return self.meta['dims']

    @property
    def elements(self) -> list[list]:
        return self.meta['elements']

    @property
    def shape(self) -> tuple:
        return tuple(self.meta['shape'])

    @property
    def values(self) -> np.ndarray:
        """Load values on first access (memory-mapped)."""
        if self._values is None:
            path = self.directory / f"var_{self.name}_values.npy"
            self._values = np.load(path, mmap_mode='r').reshape(self.shape)
        return self._values

    @property
    def duals(self) -> np.ndarray:
        """Load duals on first access (memory-mapped)."""
        if self._duals is None:
            path = self.directory / f"var_{self.name}_duals.npy"
            self._duals = np.load(path, mmap_mode='r').reshape(self.shape)
        return self._duals

    def __getitem__(self, idx):
        """Direct indexing into values."""
        return self.values[idx]


@dataclass
class LazyConstraint:
    """Lazy-loaded constraint solution."""
    directory: Path
    name: str
    meta: dict
    _duals: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def dims(self) -> list[str]:
        return self.meta['dims']

    @property
    def elements(self) -> list[list]:
        return self.meta['elements']

    @property
    def shape(self) -> tuple:
        return tuple(self.meta['shape'])

    @property
    def duals(self) -> np.ndarray:
        if self._duals is None:
            path = self.directory / f"con_{self.name}_duals.npy"
            self._duals = np.load(path, mmap_mode='r').reshape(self.shape)
        return self._duals

    def __getitem__(self, idx):
        return self.duals[idx]


def save_solution(
    solver: "Solver", model: "Model", directory: str | Path, use_rust: bool = True
) -> LazySolution:
    """Extract and save solution to disk with memory-mapped arrays.

    Fast binary format. Data loaded lazily on access.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Try to use Rust for faster file I/O
    write_npy = None
    if use_rust:
        try:
            import nimopt_rust
            write_npy = nimopt_rust.write_npy_f64
        except ImportError:
            pass

    # Get solution arrays (single conversion from solver)
    var_values = np.asarray(solver.get_variable_values(), dtype=np.float64)
    var_duals = np.asarray(solver.get_variable_duals(), dtype=np.float64)
    con_duals = np.asarray(solver.get_constraint_duals(), dtype=np.float64)

    var_meta = {}
    con_meta = {}

    # Save variables
    var_offset = 0
    for var_name, var in model.variables.items():
        size = var.size
        shape = list(var.shape) if var.sets else [1]
        dims = var.dims
        elements = [list(s.elements) for s in var.sets] if var.sets else []

        # Save arrays
        vals_path = str(directory / f"var_{var_name}_values.npy")
        duals_path = str(directory / f"var_{var_name}_duals.npy")
        vals_slice = var_values[var_offset:var_offset + size]
        duals_slice = var_duals[var_offset:var_offset + size]

        if write_npy:
            write_npy(vals_path, np.ascontiguousarray(vals_slice))
            write_npy(duals_path, np.ascontiguousarray(duals_slice))
        else:
            np.save(vals_path, vals_slice)
            np.save(duals_path, duals_slice)
        var_offset += size

        var_meta[var_name] = {'dims': dims, 'elements': elements, 'shape': shape}

    # Save constraints
    con_offset = 0
    for con_name, constraint in model._constraints.items():
        free_sets = constraint.free_sets
        if not free_sets:
            size, shape, dims, elements = 1, [1], [], []
        else:
            size = 1
            for s in free_sets:
                size *= len(s)
            shape = [len(s) for s in free_sets]
            dims = [s.name for s in free_sets]
            elements = [list(s.elements) for s in free_sets]

        duals_path = str(directory / f"con_{con_name}_duals.npy")
        duals_slice = con_duals[con_offset:con_offset + size]

        if write_npy:
            write_npy(duals_path, np.ascontiguousarray(duals_slice))
        else:
            np.save(duals_path, duals_slice)
        con_offset += size

        con_meta[con_name] = {'dims': dims, 'elements': elements, 'shape': shape}

    # Save metadata
    obj_val = None
    if hasattr(solver, 'get_objective_value'):
        obj_val = solver.get_objective_value()
    meta = {
        'objective_value': obj_val,
        'variables': var_meta,
        'constraints': con_meta,
    }
    with open(directory / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return LazySolution(
        directory=directory,
        objective_value=meta['objective_value'],
        _var_meta=var_meta,
        _con_meta=con_meta,
    )


def load_solution(directory: str | Path) -> LazySolution:
    """Load a saved solution (lazy - no data loaded until accessed)."""
    directory = Path(directory)

    with open(directory / 'meta.json') as f:
        meta = json.load(f)

    return LazySolution(
        directory=directory,
        objective_value=meta.get('objective_value'),
        _var_meta=meta['variables'],
        _con_meta=meta['constraints'],
    )
