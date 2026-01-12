"""Unified solution interface with automatic memory management."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from .solution import (
    ConstraintSolution,
    Solution,
    VariableSolution,
    extract_solution_python,
    write_solution_csv,
)
from .solution_lazy import (
    LazyConstraint,
    LazySolution,
    LazyVariable,
    load_solution,
    save_solution,
)

if TYPE_CHECKING:
    from .model import Model
    from .solvers.base import Solver

# Threshold: switch to lazy storage above this many variables
DEFAULT_LAZY_THRESHOLD = 500_000  # 500k variables ~ 8MB per array


def extract_solution(
    solver: "Solver",
    model: "Model",
    lazy: Optional[bool] = None,
    directory: Optional[str | Path] = None,
    threshold: int = DEFAULT_LAZY_THRESHOLD,
) -> Union[Solution, LazySolution]:
    """Extract solution from solver with automatic memory management.

    Parameters
    ----------
    solver : Solver
        Solved solver instance.
    model : Model
        The nimopt Model.
    lazy : bool, optional
        Force lazy (disk-backed) or in-memory storage.
        If None, automatically choose based on model size.
    directory : str or Path, optional
        Directory for lazy storage. Required if lazy=True.
        Auto-generated temp directory if lazy and not provided.
    threshold : int
        Variable count threshold for automatic lazy switch.
        Default: 500,000 variables.

    Returns
    -------
    Solution or LazySolution
        In-memory Solution for small models, LazySolution for large ones.
    """
    # Count total variables
    n_vars = sum(v.size for v in model.variables.values())

    # Decide storage mode
    if lazy is None:
        use_lazy = n_vars > threshold
    else:
        use_lazy = lazy

    if use_lazy:
        if directory is None:
            import tempfile
            directory = tempfile.mkdtemp(prefix='nimopt_sol_')
        return save_solution(solver, model, directory)
    else:
        return extract_solution_python(solver, model)


def get_variable(
    sol: Union[Solution, LazySolution], name: str
) -> Union[VariableSolution, "LazyVariable"]:
    """Get variable from solution (works with both types)."""
    if isinstance(sol, Solution):
        return sol.variables[name]
    else:
        return sol.var(name)


def get_constraint(
    sol: Union[Solution, LazySolution], name: str
) -> Union[ConstraintSolution, "LazyConstraint"]:
    """Get constraint from solution (works with both types)."""
    if isinstance(sol, Solution):
        return sol.constraints[name]
    else:
        return sol.con(name)


def to_csv(
    sol: Union[Solution, LazySolution],
    directory: str | Path,
) -> None:
    """Export solution to CSV files.

    Works with both in-memory and lazy solutions.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if isinstance(sol, Solution):
        write_solution_csv(sol, directory)
    else:
        # Convert lazy to CSV
        _lazy_to_csv(sol, directory)


def _lazy_to_csv(sol: LazySolution, directory: Path) -> None:
    """Convert lazy solution to CSV files."""
    try:
        import nimopt_rust
        use_rust = True
    except ImportError:
        use_rust = False

    for var_name in sol.variables:
        var = sol.var(var_name)
        path = str(directory / f"var_{var_name}.csv")

        if use_rust and var.dims:
            nimopt_rust.write_var_solution_csv(
                path,
                var.dims,
                var.elements,
                var.values.flatten().astype('float64'),
                var.duals.flatten().astype('float64'),
            )
        else:
            _write_var_csv_python(var, path)

    for con_name in sol.constraints:
        con = sol.con(con_name)
        path = str(directory / f"con_{con_name}.csv")

        if use_rust and con.dims:
            nimopt_rust.write_con_solution_csv(
                path,
                con.dims,
                con.elements,
                con.duals.flatten().astype('float64'),
            )
        else:
            _write_con_csv_python(con, path)


def _write_var_csv_python(var, path: str) -> None:
    """Write variable CSV (Python fallback)."""
    import itertools
    with open(path, 'w') as f:
        header = var.dims + ['value', 'dual']
        f.write(','.join(header) + '\n')

        if not var.dims:
            f.write(f"{var.values.flat[0]},{var.duals.flat[0]}\n")
        else:
            ranges = [range(len(e)) for e in var.elements]
            for idx in itertools.product(*ranges):
                row = [var.elements[d][idx[d]] for d in range(len(idx))]
                row.extend([str(var.values[idx]), str(var.duals[idx])])
                f.write(','.join(str(x) for x in row) + '\n')


def _write_con_csv_python(con, path: str) -> None:
    """Write constraint CSV (Python fallback)."""
    import itertools
    with open(path, 'w') as f:
        header = con.dims + ['dual']
        f.write(','.join(header) + '\n')

        if not con.dims:
            f.write(f"{con.duals.flat[0]}\n")
        else:
            ranges = [range(len(e)) for e in con.elements]
            for idx in itertools.product(*ranges):
                row = [con.elements[d][idx[d]] for d in range(len(idx))]
                row.append(str(con.duals[idx]))
                f.write(','.join(str(x) for x in row) + '\n')


# Re-export for convenience
__all__ = [
    'extract_solution',
    'get_variable',
    'get_constraint',
    'to_csv',
    'load_solution',
    'Solution',
    'LazySolution',
    'DEFAULT_LAZY_THRESHOLD',
]
