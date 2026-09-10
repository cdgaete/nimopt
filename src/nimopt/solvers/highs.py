"""The HiGHS adapter.

HiGHS takes a row-wise matrix as the same three arrays `Assembled` carries,
so the handoff passes them without building another form. The column kinds
cross the same way, as an array rather than a value stated per column.

Every model status HiGHS can report is either named in `OUTCOME`, which maps
into the seam's `STATUS`, or listed in `FAILED` and raised. A status the
adapter does not know is refused rather than folded into a catch-all, because
a caller reading a status it was never given cannot tell an answer from the
absence of one.
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
    """Whether this HiGHS carries HiPO's extras library.

    HiGHS keeps HiPO's orderings and its BLAS in a library it loads at run
    time, and asked for HiPO without it, it logs an error and runs simplex.
    A two-column probe with the log captured answers once per process.
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

    The model HiGHS built is returned in `Result.backend`. A session holding it
    reads a conflict or a ray from the same solved instance. A model with
    integer columns is returned with no duals. HiGHS reports the relaxation's
    duals for one, and this adapter does not report that pair.
    """
    import highspy

    if sense not in SENSES:
        raise ValueError(f"sense is 'min' or 'max'; got {sense!r}")
    settings = translated(options, OPTION_NAMES, OPTION_VALUES, solver="highs")
    if settings.get("solver") == "hipo" and not hipo_available():
        raise RuntimeError(
            "this HiGHS was built without HiPO's extras library, so "
            "method='hipo' would log an error and run simplex instead; install "
            "a HiGHS carrying `libhighs_extras` beside `libhighs`, or choose "
            "another method"
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
                f"HiGHS refused option {key!r} at {value!r}; it states a name "
                f"and a type, and answers an error rather than raising"
            )
    if highs.passModel(lp) == highspy.HighsStatus.kError:
        raise RuntimeError(
            f"HiGHS refused the model it was passed: {assembled.n_cols} "
            f"columns against {assembled.col_cost.size} costs, "
            f"{assembled.col_lower.size} lower and {assembled.col_upper.size} "
            f"upper column bounds; {assembled.n_rows} rows against "
            f"{assembled.row_lower.size} lower and {assembled.row_upper.size} "
            f"upper row bounds"
        )
    threads = (
        " HiGHS fixes its thread count at the first solve a process runs, "
        "so `threads` states the same count from that solve onwards."
        if "threads" in (options or {})
        else ""
    )
    if highs.run() == highspy.HighsStatus.kError:
        raise RuntimeError(
            f"HiGHS answered an error running the model it was passed, and "
            f"reached no outcome to report.{threads}"
        )

    reported = str(highs.getModelStatus()).split(".")[-1]
    if reported in FAILED:
        raise RuntimeError(
            f"HiGHS stopped without solving the model: {reported}. The model "
            f"was passed to the solver but no outcome was reached.{threads}"
        )
    if reported not in OUTCOME:
        raise RuntimeError(
            f"HiGHS reported model status {reported!r}, which this adapter "
            f"does not read; nimopt names an outcome or refuses it"
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
    """The rows and columns of an irreducible infeasible subsystem.

    The strategy is set here rather than at the solve, so a solve nobody asks
    a question of pays nothing for one. HiGHS computes the set over the
    model's linear relaxation and reports an invalid one where that relaxation
    is feasible; an invalid set is refused rather than handed back, because a
    conflict nothing proved minimal is one an agent would act on.
    """
    backend.setOptionValue("iis_strategy", IRREDUCIBLE)
    reported, iis = backend.getIis()
    if not iis.valid_:
        raise RuntimeError(
            f"HiGHS computed no conflict and reported "
            f"{str(reported).split('.')[-1]}: it could not prove this model "
            f"infeasible from its linear relaxation, which is what its "
            f"irreducible set is computed over. A model infeasible only "
            f"through its integrality reaches this"
        )
    return (
        np.asarray(iis.row_index_, dtype=np.int64),
        np.asarray(iis.col_index_, dtype=np.int64),
    )


def ray(backend: Any) -> npt.NDArray[np.float64] | None:
    """The direction an unbounded model runs off in, or None where there is none."""
    _, found, values = backend.getPrimalRay()
    if not found:
        return None
    return np.asarray(values, dtype=np.float64)
