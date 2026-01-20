"""Solution extraction and storage."""

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .model import Model
    from .solvers.base import Solver


@dataclass
class VariableSolution:
    """Solution data for a single variable."""

    name: str
    dims: list[str]
    elements: list[list]  # Elements for each dimension
    values: np.ndarray  # Shape matches variable shape
    duals: np.ndarray  # Reduced costs, same shape

    def to_array(self):
        """Convert values to nimblend Array."""
        import nimblend as nb

        if not self.dims:
            return nb.Array(self.values.flat[0])
        coords = {dim: self.elements[i] for i, dim in enumerate(self.dims)}
        return nb.Array(self.values, dims=self.dims, coords=coords)

    def to_dual_array(self):
        """Convert reduced costs to nimblend Array."""
        import nimblend as nb

        if not self.dims:
            return nb.Array(self.duals.flat[0])
        coords = {dim: self.elements[i] for i, dim in enumerate(self.dims)}
        return nb.Array(self.duals, dims=self.dims, coords=coords)


@dataclass
class ConstraintSolution:
    """Solution data for a single constraint."""

    name: str
    dims: list[str]
    elements: list[list]  # Elements for each dimension
    duals: np.ndarray  # Shadow prices

    def to_array(self):
        """Convert duals (shadow prices) to nimblend Array."""
        import nimblend as nb

        if not self.dims:
            return nb.Array(self.duals.flat[0])
        coords = {dim: self.elements[i] for i, dim in enumerate(self.dims)}
        return nb.Array(self.duals, dims=self.dims, coords=coords)


@dataclass
class Solution:
    """Complete solution with per-variable and per-constraint data."""

    objective_value: Optional[float] = None
    variables: dict[str, VariableSolution] = field(default_factory=dict)
    constraints: dict[str, ConstraintSolution] = field(default_factory=dict)

    def var(self, name: str):
        """Get variable solution values as nimblend Array."""
        return self.variables[name].to_array()

    def var_dual(self, name: str):
        """Get variable reduced costs as nimblend Array."""
        return self.variables[name].to_dual_array()

    def con(self, name: str):
        """Get constraint duals (shadow prices) as nimblend Array."""
        return self.constraints[name].to_array()


def extract_solution_python(solver: "Solver", model: "Model") -> Solution:
    """Extract solution from solver using model metadata.

    Uses model structure to map solver values directly to arrays.
    No string parsing - uses same iteration order as LP generation.
    """
    sol = Solution()

    # Get objective value if available
    if hasattr(solver, '_h'):
        # HiGHS direct solver
        sol.objective_value = solver._h.getInfo().objective_function_value
    elif hasattr(solver, 'get_objective_value'):
        sol.objective_value = solver.get_objective_value()

    # Get flat arrays from solver
    var_values = np.array(solver.get_variable_values(), dtype=np.float64)
    var_duals = np.array(solver.get_variable_duals(), dtype=np.float64)
    con_duals = np.array(solver.get_constraint_duals(), dtype=np.float64)

    # Variables: use model metadata to map directly
    var_offset = 0
    for var_name, var in model.variables.items():
        size = var.size
        dims = var.dims
        elements = [list(s.elements) for s in var.sets] if var.sets else []
        shape = var.shape if var.sets else (1,)

        # Slice values in LP order (same as itertools.product)
        values = var_values[var_offset : var_offset + size].reshape(shape)
        duals = var_duals[var_offset : var_offset + size].reshape(shape)
        var_offset += size

        sol.variables[var_name] = VariableSolution(
            name=var_name,
            dims=dims,
            elements=elements,
            values=values,
            duals=duals,
        )

    # Constraints: use model metadata
    con_offset = 0
    for con_name, constraint in model._constraints.items():
        free_sets = constraint.free_sets
        if not free_sets:
            size = 1
            shape = (1,)
            dims = []
            elements = []
        else:
            size = 1
            for s in free_sets:
                size *= len(s)
            shape = tuple(len(s) for s in free_sets)
            dims = [s.name for s in free_sets]
            elements = [list(s.elements) for s in free_sets]

        duals = con_duals[con_offset : con_offset + size].reshape(shape)
        con_offset += size

        sol.constraints[con_name] = ConstraintSolution(
            name=con_name,
            dims=dims,
            elements=elements,
            duals=duals,
        )

    return sol


def write_solution_csv(
    sol: Solution, directory: str | Path, use_rust: bool = True
) -> None:
    """Write solution to CSV files, one per variable and constraint."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if use_rust:
        try:
            import nimopt_rust

            _write_solution_csv_rust(sol, directory, nimopt_rust)
            return
        except ImportError:
            pass

    _write_solution_csv_python(sol, directory)


def _write_solution_csv_rust(sol: Solution, directory: Path, nimopt_rust) -> None:
    """Write solution CSVs using Rust."""
    for var_name, var_sol in sol.variables.items():
        path = str(directory / f"var_{var_name}.csv")
        nimopt_rust.write_var_solution_csv(
            path,
            var_sol.dims,
            var_sol.elements,
            var_sol.values.flatten().astype("float64"),
            var_sol.duals.flatten().astype("float64"),
        )

    for con_name, con_sol in sol.constraints.items():
        path = str(directory / f"con_{con_name}.csv")
        nimopt_rust.write_con_solution_csv(
            path,
            con_sol.dims,
            con_sol.elements,
            con_sol.duals.flatten().astype("float64"),
        )


def _write_solution_csv_python(sol: Solution, directory: Path) -> None:
    """Write solution CSVs using Python (fallback)."""
    # Write variables
    for var_name, var_sol in sol.variables.items():
        path = directory / f"var_{var_name}.csv"
        with open(path, "w") as f:
            header = var_sol.dims + ["value", "dual"]
            f.write(",".join(header) + "\n")

            if not var_sol.dims:
                f.write(f"{var_sol.values.flat[0]},{var_sol.duals.flat[0]}\n")
            else:
                ranges = [range(len(e)) for e in var_sol.elements]
                for idx in itertools.product(*ranges):
                    row = [var_sol.elements[d][idx[d]] for d in range(len(idx))]
                    row.append(str(var_sol.values[idx]))
                    row.append(str(var_sol.duals[idx]))
                    f.write(",".join(str(x) for x in row) + "\n")

    # Write constraints
    for con_name, con_sol in sol.constraints.items():
        path = directory / f"con_{con_name}.csv"
        with open(path, "w") as f:
            header = con_sol.dims + ["dual"]
            f.write(",".join(header) + "\n")

            if not con_sol.dims:
                f.write(f"{con_sol.duals.flat[0]}\n")
            else:
                ranges = [range(len(e)) for e in con_sol.elements]
                for idx in itertools.product(*ranges):
                    row = [con_sol.elements[d][idx[d]] for d in range(len(idx))]
                    row.append(str(con_sol.duals[idx]))
                    f.write(",".join(str(x) for x in row) + "\n")
