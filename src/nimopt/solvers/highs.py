"""The HiGHS adapter.

HiGHS reads a row-wise matrix as the same three arrays `Assembled` contains.
The adapter passes those arrays without building another form. The column
kinds are passed as one array over the columns.

`change_column_bounds`, `append_rows` and `resume` change and solve the model
HiGHS loaded, and copy no part of it out of HiGHS. `resume` reports the
iterations of its own run.

`OUTCOME` maps a HiGHS model status into `STATUS`. `FAILED` lists the HiGHS
statuses this adapter raises on. A status in neither raises.
"""

import weakref
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from nimopt.assembly import Assembled

from nimopt.solvers.base import (
    Capabilities,
    Result,
    check_reported,
    check_sense,
    importable,
    proved_bound,
)
from nimopt.solvers.options import translated

BACKEND = "highspy"

CAPABILITIES = Capabilities(
    "highs",
    {
        "integrality": "native",
        "duals": "native",
        "conflict": "native",
        "ray": "native",
        "mutation": "native",
    },
    rejected=(("integrality", "duals"),),
)

OUTCOME = {
    "kOptimal": "optimal",
    "kInfeasible": "infeasible",
    "kUnbounded": "unbounded",
    "kUnboundedOrInfeasible": "unbounded_or_infeasible",
    "kModelEmpty": "empty",
    "kTimeLimit": "time_limit",
    "kIterationLimit": "iteration_limit",
    "kSolutionLimit": "solution_limit",
    "kObjectiveBound": "objective_bound",
    "kObjectiveTarget": "objective_target",
    "kMemoryLimit": "memory_limit",
    "kInterrupt": "interrupt",
    "kHighsInterrupt": "interrupt",
}

SENSES = {
    "min": "kMinimize",
    "max": "kMaximize",
}

IRREDUCIBLE = 4

FAILED = (
    "kNotset",
    "kLoadError",
    "kModelError",
    "kPresolveError",
    "kSolveError",
    "kPostsolveError",
    "kUnknown",
)

OPTION_NAMES = {
    "time_limit": "time_limit",
    "iteration_limit": "simplex_iteration_limit",
    "node_limit": "mip_max_nodes",
    "mip_gap": "mip_rel_gap",
    "mip_abs_gap": "mip_abs_gap",
    "feasibility_tol": "primal_feasibility_tolerance",
    "optimality_tol": "dual_feasibility_tolerance",
    "threads": "threads",
    "seed": "random_seed",
    "log": "output_flag",
    "presolve": "presolve",
    "method": "solver",
    "newton_system": "hipo_system",
    "crossover": "run_crossover",
    "pdlp_tol": "pdlp_optimality_tolerance",
    "separate_process": None,
}

OPTION_VALUES = {
    "method": {
        "choose": "choose",
        "simplex": "simplex",
        "barrier": "ipm",
        "hipo": "hipo",
        "pdlp": "pdlp",
    },
}

_PROBED = {}

_INTEGER: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def hipo_available() -> bool:
    """Return whether this HiGHS provides HiPO's extras library.

    HiGHS loads HiPO's orderings and its BLAS from a library at run time.
    Without that library HiGHS logs an error and runs simplex. This function
    solves a two-column model with the log captured, once per process.
    """
    if "hipo" not in _PROBED:
        import highspy

        lines = []
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", True)
        highs.setOptionValue("log_to_console", False)
        highs.cbLogging.subscribe(
            lambda event: lines.append(str(getattr(event, "message", event)))
        )
        inf = highspy.kHighsInf
        highs.addVars(2, np.zeros(2), np.full(2, inf))
        highs.changeColsCost(2, np.arange(2, dtype=np.int32), np.ones(2))
        highs.addRow(1.0, inf, 2, np.arange(2, dtype=np.int32), np.ones(2))
        highs.setOptionValue("solver", "hipo")
        highs.run()
        _PROBED["hipo"] = not any("unavailable" in line for line in lines)
    return _PROBED["hipo"]


def _bound(integer: bool, status: str, objective: float, info: Any) -> float | None:
    """Return the bound HiGHS proved on the optimal objective, or None.

    For a model with integer columns the bound is `mip_dual_bound`. It is None
    where HiGHS reports an infinite value. For a model without integer columns
    HiGHS defines no dual bound. The bound is then the objective at status
    `optimal` and None at any other status.
    """
    return proved_bound(
        status, objective, float(info.mip_dual_bound) if integer else None
    )


def installed() -> bool:
    """Return whether `highspy` can be imported."""
    return importable(BACKEND)


def _settings(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the HiGHS options `solve` sets for `options`.

    Raises RuntimeError for a method of `hipo` on a HiGHS without HiPO's
    extras library.
    """
    settings = translated(options, OPTION_NAMES, OPTION_VALUES, solver="highs")
    if settings.get("solver") == "hipo" and not hipo_available():
        raise RuntimeError(
            "method is 'hipo' and this HiGHS provides no HiPO extras library; "
            "install a HiGHS with `libhighs_extras` beside `libhighs`, or "
            "choose another method"
        )
    return settings


def _apply(highs: Any, settings: Mapping[str, Any]) -> None:
    """Set `settings` on the HiGHS handle `highs`.

    Raises ValueError for an option HiGHS rejects.
    """
    import highspy

    highs.setOptionValue("output_flag", False)
    for key, value in settings.items():
        if highs.setOptionValue(key, value) != highspy.HighsStatus.kOk:
            raise ValueError(
                f"HiGHS rejected option {key!r} at {value!r}; pass a name and "
                f"a value HiGHS accepts"
            )


def _threads(options: Mapping[str, Any] | None) -> str:
    """Return the note on the thread count for a solve that sets `threads`."""
    if "threads" not in (options or {}):
        return ""
    return (
        " HiGHS fixes its thread count at the first solve of a process. A "
        "later `threads` value does not change it."
    )


def _check_vectors(assembled: "Assembled") -> None:
    """Raise RuntimeError for a vector of a length or a kind HiGHS misreads.

    HiGHS reads each array by the count of rows, columns or nonzeros it is
    given, and checks no length. It reads an integrality flag above 1 as a
    column kind other than continuous and integer.
    """
    rows, columns = assembled.n_rows, assembled.n_cols
    nonzeros = assembled.matrix.nnz
    expected = {
        "col_cost": columns,
        "col_lower": columns,
        "col_upper": columns,
        "integrality": columns,
        "row_lower": rows,
        "row_upper": rows,
        "indptr": rows + 1,
        "indices": nonzeros,
        "values": nonzeros,
    }
    wrong = [
        f"{name} has length {getattr(assembled, name).size}, not {length}"
        for name, length in expected.items()
        if getattr(assembled, name).size != length
    ]
    if wrong:
        raise RuntimeError(
            f"the assembled vectors differ in length from the matrix of {rows} "
            f"rows, {columns} columns and {nonzeros} nonzeros: "
            f"{', '.join(wrong)}; pass vectors of the lengths the matrix declares"
        )
    flags = assembled.integrality
    if flags.size and (flags.min() < 0 or flags.max() > 1):
        raise RuntimeError(
            "integrality has a value other than 0 and 1; pass 0 for a "
            "continuous column and 1 for an integer column"
        )


def solve(
    assembled: "Assembled", sense: str, options: Mapping[str, Any] | None = None
) -> Result:
    """Solve an assembled model and return its status, its values and its bound.

    `Result.backend` is the model HiGHS built. A session reads a conflict or a
    ray from that solved instance. `Result.row_dual` is None for a model with
    integer columns. `resume` solves that same model again.
    """
    import highspy

    check_sense(sense)
    settings = _settings(options)
    _check_vectors(assembled)
    highs = highspy.Highs()
    _apply(highs, settings)
    passed = highs.passModel(
        assembled.n_cols,
        assembled.n_rows,
        int(assembled.values.size),
        int(highspy.MatrixFormat.kRowwise),
        int(getattr(highspy.ObjSense, SENSES[sense])),
        0.0,
        assembled.col_cost,
        assembled.col_lower,
        assembled.col_upper,
        assembled.row_lower,
        assembled.row_upper,
        assembled.indptr,
        assembled.indices,
        assembled.values,
        assembled.integrality,
    )
    if passed == highspy.HighsStatus.kError:
        raise RuntimeError(
            f"HiGHS rejected the model it was passed: {assembled.n_rows} rows, "
            f"{assembled.n_cols} columns and {assembled.values.size} nonzeros; "
            f"solve with `log=True` and read the reason in the HiGHS log"
        )
    integer = bool(assembled.integrality.any())
    _INTEGER[highs] = integer
    threads = _threads(options)
    if highs.run() == highspy.HighsStatus.kError:
        raise RuntimeError(
            f"HiGHS returned an error running the model and reported no "
            f"outcome; check the model and the options.{threads}"
        )
    return _solved(highs, integer, threads)


def resume(backend: Any, options: Mapping[str, Any] | None = None) -> Result:
    """Solve the model HiGHS loaded again and return its status and its values.

    `backend` is the `Result.backend` of a `solve` of this adapter. `options`
    are applied to the handle as `solve` applies them. `Result.iterations` is
    the simplex iterations of this run. `Result.row_dual` is None where the
    loaded model has integer columns.
    """
    import highspy

    settings = _settings(options)
    _apply(backend, settings)
    threads = _threads(options)
    if backend.run() == highspy.HighsStatus.kError:
        raise RuntimeError(
            f"HiGHS returned an error running the model and reported no "
            f"outcome; check the model and the options.{threads}"
        )
    return _solved(backend, _integer(backend), threads)


def _solved(highs: Any, integer: bool, threads: str) -> Result:
    """Return the result of the solve HiGHS ran on the model it loaded."""
    import highspy

    reported = str(highs.getModelStatus()).split(".")[-1]
    check_reported(
        reported, FAILED, OUTCOME, "HiGHS", "model status", tail=f".{threads}"
    )
    solution = highs.getSolution()
    duals = None
    if CAPABILITIES.reports_duals(integer):
        duals = np.asarray(solution.row_dual, dtype=np.float64)
    info = highs.getInfo()
    status = OUTCOME[reported]
    objective = float(info.objective_function_value)
    feasible = (
        info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
    )
    return Result(
        status,
        feasible,
        objective,
        _bound(integer, status, objective, info),
        np.asarray(solution.col_value, dtype=np.float64),
        duals,
        highs,
        int(info.simplex_iteration_count),
    )


def _integer(backend: Any) -> bool:
    """Return whether the model this adapter passed to `backend` has integer columns.

    Raises RuntimeError for a backend this adapter did not solve.
    """
    try:
        return _INTEGER[backend]
    except KeyError:
        raise RuntimeError(
            "backend is not a model this adapter solved; pass the backend of "
            "a Result that `solve` returned"
        ) from None


def _length(name: str, found: int, expected: int, action: str) -> None:
    """Raise RuntimeError where `found` differs from `expected`."""
    if found != expected:
        raise RuntimeError(f"{name} has length {found}, not {expected}; {action}")


def change_column_bounds(backend: Any, columns: Any, lower: Any, upper: Any) -> None:
    """Change the bounds of `columns` on the model HiGHS loaded.

    `columns` are absolute column indices, and `lower` and `upper` have one
    value per column. Raises RuntimeError for arrays of other lengths and
    where HiGHS rejects the change.
    """
    import highspy

    count = int(columns.size)
    _length("lower", int(lower.size), count, "pass one lower bound per column")
    _length("upper", int(upper.size), count, "pass one upper bound per column")
    if backend.changeColsBounds(count, columns, lower, upper) == (
        highspy.HighsStatus.kError
    ):
        raise RuntimeError(
            f"HiGHS rejected the change of {count} column bounds; pass columns "
            f"of the loaded model"
        )


def append_rows(
    backend: Any,
    row_lower: Any,
    row_upper: Any,
    indptr: Any,
    indices: Any,
    values: Any,
) -> None:
    """Append rows to the model HiGHS loaded.

    `indptr`, `indices` and `values` are the CSR form of the appended rows
    over the columns of the loaded model. Raises RuntimeError for arrays of
    other lengths and where HiGHS rejects the rows.
    """
    import highspy

    rows = int(row_lower.size)
    _length(
        "row_upper",
        int(row_upper.size),
        rows,
        "pass one upper bound per row of row_lower",
    )
    _length(
        "indptr",
        int(indptr.size),
        rows + 1,
        "pass one entry per row of row_lower and one more",
    )
    nonzeros = int(indptr[-1])
    _length(
        "indices",
        int(indices.size),
        nonzeros,
        "pass one column index per entry of indptr",
    )
    _length("values", int(values.size), nonzeros, "pass one value per entry of indptr")
    if (
        backend.addRows(
            rows, row_lower, row_upper, nonzeros, indptr[:-1], indices, values
        )
        == highspy.HighsStatus.kError
    ):
        raise RuntimeError(
            f"HiGHS rejected {rows} appended rows; pass rows over the columns "
            f"of the loaded model"
        )


def conflict(backend: Any) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the rows and columns of an irreducible infeasible subsystem.

    The strategy option is set here, not at the solve. HiGHS computes the
    subsystem over the model's linear relaxation. An invalid subsystem raises
    RuntimeError.
    """
    backend.setOptionValue("iis_strategy", IRREDUCIBLE)
    reported, iis = backend.getIis()
    if not iis.valid_:
        raise RuntimeError(
            f"HiGHS computed no conflict over the feasible linear relaxation "
            f"and reported {str(reported).split('.')[-1]}; use a solver whose "
            f"conflict covers the integrality"
        )
    return (
        np.asarray(iis.row_index_, dtype=np.int64),
        np.asarray(iis.col_index_, dtype=np.int64),
    )


def ray(backend: Any) -> npt.NDArray[np.float64] | None:
    """Return the primal ray of an unbounded model, or None where none exists."""
    _, found, values = backend.getPrimalRay()
    if not found:
        return None
    return np.asarray(values, dtype=np.float64)
