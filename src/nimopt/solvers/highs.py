"""The HiGHS adapter.

HiGHS reads a row-wise matrix as the same three arrays `Assembled` contains.
The adapter passes those arrays without building another form. The column
kinds are passed as one array over the columns.

`OUTCOME` maps a HiGHS model status into `STATUS`. `FAILED` lists the HiGHS
statuses this adapter raises on. A status in neither raises.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from nimopt.model import Assembled

from nimopt.solvers.base import Capabilities, Result
from nimopt.solvers.options import translated

BACKEND = "highspy"

CAPABILITIES = Capabilities(
    "highs",
    {
        "integrality": "native",
        "duals": "native",
        "conflict": "native",
        "ray": "native",
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
    if not integer:
        return objective if status == "optimal" else None
    value = float(info.mip_dual_bound)
    return value if np.isfinite(value) else None


def solve(
    assembled: "Assembled", sense: str, options: Mapping[str, Any] | None = None
) -> Result:
    """Solve an assembled model and return its status, its values and its bound.

    `Result.backend` is the model HiGHS built. A session reads a conflict or a
    ray from that solved instance. `Result.row_dual` is None for a model with
    integer columns.
    """
    import highspy

    if sense not in SENSES:
        raise ValueError(f"sense is 'min' or 'max'; got {sense!r}")
    settings = translated(options, OPTION_NAMES, OPTION_VALUES, solver="highs")
    if settings.get("solver") == "hipo" and not hipo_available():
        raise RuntimeError(
            "method is 'hipo' and this HiGHS provides no HiPO extras library; "
            "install a HiGHS with `libhighs_extras` beside `libhighs`, or "
            "choose another method"
        )
    lp = highspy.HighsLp()
    lp.num_col_ = assembled.n_cols
    lp.num_row_ = assembled.n_rows
    lp.col_cost_ = assembled.col_cost
    lp.col_lower_ = assembled.col_lower
    lp.col_upper_ = assembled.col_upper
    lp.row_lower_ = assembled.row_lower
    lp.row_upper_ = assembled.row_upper
    lp.sense_ = getattr(highspy.ObjSense, SENSES[sense])
    if assembled.integrality.any():
        lp.integrality_ = np.where(
            assembled.integrality.astype(bool),
            highspy.HighsVarType.kInteger,
            highspy.HighsVarType.kContinuous,
        )

    matrix = lp.a_matrix_
    matrix.format_ = highspy.MatrixFormat.kRowwise
    matrix.num_col_ = assembled.n_cols
    matrix.num_row_ = assembled.n_rows
    matrix.start_ = assembled.indptr
    matrix.index_ = assembled.indices
    matrix.value_ = assembled.values

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    for key, value in settings.items():
        if highs.setOptionValue(key, value) != highspy.HighsStatus.kOk:
            raise ValueError(
                f"HiGHS rejected option {key!r} at {value!r}; pass a name and "
                f"a value HiGHS accepts"
            )
    if highs.passModel(lp) == highspy.HighsStatus.kError:
        raise RuntimeError(
            f"HiGHS rejected the model it was passed: {assembled.n_cols} "
            f"columns against {assembled.col_cost.size} costs, "
            f"{assembled.col_lower.size} lower and {assembled.col_upper.size} "
            f"upper column bounds, {assembled.n_rows} rows against "
            f"{assembled.row_lower.size} lower and {assembled.row_upper.size} "
            f"upper row bounds; pass vectors of the declared column and row "
            f"counts"
        )
    threads = (
        " HiGHS fixes its thread count at the first solve of a process. A "
        "later `threads` value does not change it."
        if "threads" in (options or {})
        else ""
    )
    if highs.run() == highspy.HighsStatus.kError:
        raise RuntimeError(
            f"HiGHS returned an error running the model and reported no "
            f"outcome; check the model and the options.{threads}"
        )

    reported = str(highs.getModelStatus()).split(".")[-1]
    if reported in FAILED:
        raise RuntimeError(
            f"HiGHS stopped at model status {reported} without solving the "
            f"model; check the model and the options.{threads}"
        )
    if reported not in OUTCOME:
        raise RuntimeError(
            f"HiGHS reported the model status {reported!r}; this adapter maps "
            f"no outcome to it, report it as a defect"
        )
    solution = highs.getSolution()
    integer = bool(assembled.integrality.any())
    duals = None
    if not (integer and CAPABILITIES.rejects("integrality", "duals")):
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
            f"HiGHS computed no conflict and reported "
            f"{str(reported).split('.')[-1]}; HiGHS computes a conflict over "
            f"the linear relaxation alone, and this relaxation is feasible"
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
