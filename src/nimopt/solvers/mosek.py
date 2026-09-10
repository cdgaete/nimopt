"""The Mosek adapter.

Mosek's task interface reads a row-wise matrix as the three arrays `Assembled`
contains. The row pointer is split into where each row starts and where it
ends. A bound is passed as a key beside its two numbers, derived from the
bound arrays. An infinite bound is passed as the absence of a bound.

A solve reports through three signals. The termination code identifies a limit
the optimizer stopped at. The solution status identifies an optimum. The
problem status identifies infeasibility and unboundedness. `LIMITS`, `OPTIMAL`
and `OUTCOME` map each into `STATUS`. `FAILED` lists the termination codes this
adapter raises on. A signal in none of them raises.

Mosek runs only its mixed-integer optimizer on a model with integer columns.
The adapter raises on the error code `err_inv_optimizer`.
"""

import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from nimopt.model import Assembled

from nimopt.solvers.base import Capabilities, Result
from nimopt.solvers.options import translated

BACKEND = "mosek"

CAPABILITIES = Capabilities(
    "mosek",
    {
        "integrality": "native",
        "duals": "native",
        "conflict": "absent",
        "ray": "native",
    },
    rejected=(("integrality", "duals"),),
)

OPTIMAL = ("optimal", "integer_optimal")

FEASIBLE = ("optimal", "integer_optimal", "prim_feas", "prim_and_dual_feas")

OUTCOME = {
    "prim_infeas": "infeasible",
    "prim_and_dual_infeas": "infeasible",
    "dual_infeas": "unbounded",
    "prim_infeas_or_unbounded": "unbounded_or_infeasible",
}

LIMITS = {
    "trm_max_time": "time_limit",
    "trm_max_iterations": "iteration_limit",
    "trm_mio_num_branches": "node_limit",
    "trm_num_max_num_int_solutions": "solution_limit",
    "trm_objective_range": "objective_bound",
    "trm_user_callback": "interrupt",
}

FAILED = (
    "trm_stall",
    "trm_numerical_problem",
    "trm_internal",
    "trm_internal_stop",
    "trm_lost_race",
    "trm_max_num_setbacks",
    "trm_mio_num_relaxs",
    "trm_server_max_memory",
    "trm_server_max_time",
)

SENSES = {"min": "minimize", "max": "maximize"}

OPTION_NAMES = {
    "time_limit": "optimizer_max_time",
    "iteration_limit": "sim_max_iterations",
    "node_limit": "mio_max_num_branches",
    "mip_gap": "mio_tol_rel_gap",
    "mip_abs_gap": "mio_tol_abs_gap",
    "feasibility_tol": "basis_tol_x",
    "optimality_tol": "basis_tol_s",
    "threads": "num_threads",
    "seed": "mio_seed",
    "log": "log",
    "presolve": "presolve_use",
    "method": "optimizer",
    "newton_system": None,
    "crossover": "intpnt_basis",
    "pdlp_tol": None,
}

OPTION_VALUES = {
    "presolve": {"off": "off", "choose": "free", "on": "on"},
    "method": {"choose": "free", "simplex": "free_simplex", "barrier": "intpnt"},
    "crossover": {"choose": "always", "off": "never", "on": "always"},
}

ENUMS = {
    "presolve_use": "presolvemode",
    "optimizer": "optimizertype",
    "intpnt_basis": "basindtype",
}


def _name(member: Any) -> str:
    """Return the name of an enum member, without its enum."""
    return str(member).split(".")[-1]


def _keys(
    lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64], keys: Any
) -> npt.NDArray[np.int32]:
    """Return Mosek's bound key per pair, from which bounds are finite."""
    below = np.isneginf(lower)
    above = np.isposinf(upper)
    held = np.full(lower.shape, int(keys.ra), dtype=np.int32)
    held[below & above] = int(keys.fr)
    held[below & ~above] = int(keys.up)
    held[~below & above] = int(keys.lo)
    held[lower == upper] = int(keys.fx)
    return held


def _configured(mosek: Any, settings: dict[str, Any]) -> Any:
    """Return a task with `settings` applied under Mosek's parameter names."""
    task = mosek.Task()
    if settings.pop("log", False):
        task.set_Stream(mosek.streamtype.log, sys.stdout.write)
    for key, value in settings.items():
        if key in ENUMS:
            value = getattr(getattr(mosek, ENUMS[key]), value)
        if hasattr(mosek.dparam, key):
            task.putdouparam(getattr(mosek.dparam, key), float(value))
        else:
            task.putintparam(getattr(mosek.iparam, key), int(value))
    return task


def _loaded(mosek: Any, task: Any, assembled: "Assembled", sense: str) -> Any:
    """Return `task` with the assembled model loaded, the matrix row-wise."""
    keys = mosek.boundkey
    task.appendvars(assembled.n_cols)
    task.appendcons(assembled.n_rows)
    task.putcslice(0, assembled.n_cols, assembled.col_cost)
    task.putvarboundslice(
        0,
        assembled.n_cols,
        _keys(assembled.col_lower, assembled.col_upper, keys),
        assembled.col_lower,
        assembled.col_upper,
    )
    task.putconboundslice(
        0,
        assembled.n_rows,
        _keys(assembled.row_lower, assembled.row_upper, keys),
        assembled.row_lower,
        assembled.row_upper,
    )
    pointer = assembled.indptr.astype(np.int64)
    task.putarowslice(
        0,
        assembled.n_rows,
        pointer[:-1],
        pointer[1:],
        assembled.indices,
        assembled.values,
    )
    integer = np.flatnonzero(assembled.integrality).astype(np.int32)
    if integer.size:
        task.putvartypelist(integer, [mosek.variabletype.type_int] * integer.size)
    task.putobjsense(getattr(mosek.objsense, SENSES[sense]))
    return task


def _defined(mosek: Any, task: Any, integer: Any) -> Any:
    """Return the solution type the task defines for this model, or None.

    A model with integer columns defines the integer solution. A model without
    them defines the basic solution where the optimizer built one, and the
    interior solution otherwise.
    """
    which = (mosek.soltype.itg,) if integer else (mosek.soltype.bas, mosek.soltype.itr)
    for candidate in which:
        if task.solutiondef(candidate):
            return candidate
    return None


def _bound(
    mosek: Any, task: Any, integer: bool, status: str, objective: float
) -> float | None:
    """Return the bound Mosek proved on the optimal objective, or None.

    For a model with integer columns the bound is `mio_obj_bound`. Mosek
    defines that item after it solves a relaxation, and `mio_num_relax` counts
    the relaxations it solved. The bound is None where Mosek reports an
    infinite value. Mosek defines no dual bound at a relaxation count of zero
    or for a model without integer columns. The bound is then the objective at
    status `optimal` and None at any other status.
    """
    if not integer or task.getintinf(mosek.iinfitem.mio_num_relax) == 0:
        return objective if status == "optimal" else None
    value = float(task.getdouinf(mosek.dinfitem.mio_obj_bound))
    return value if np.isfinite(value) else None


def solve(
    assembled: "Assembled", sense: str, options: Mapping[str, Any] | None = None
) -> Result:
    """Solve an assembled model and return its status, its values and its bound.

    `Result.backend` is the task Mosek built. A session reads a ray from that
    solved instance. `Result.row_dual` is None for a model with integer
    columns.
    """
    import mosek

    if sense not in SENSES:
        raise ValueError(f"sense is 'min' or 'max'; got {sense!r}")
    settings = translated(options, OPTION_NAMES, OPTION_VALUES, solver="mosek")
    task = _loaded(mosek, _configured(mosek, settings), assembled, sense)
    try:
        reported = _name(task.optimize())
    except mosek.Error as error:
        if error.errno == mosek.rescode.err_inv_optimizer:
            raise RuntimeError(
                "Mosek runs only its mixed-integer optimizer on a model with "
                "integer columns; leave `method` at 'choose' for this model"
            ) from error
        raise
    if reported in FAILED:
        raise RuntimeError(
            f"Mosek stopped at termination code {reported} without solving the "
            f"model; check the model and the options"
        )
    if reported != "ok" and reported not in LIMITS:
        raise RuntimeError(
            f"Mosek reported the termination code {reported!r}; this adapter "
            f"maps no outcome to it, report it as a defect"
        )
    integer = bool(assembled.integrality.any())
    duals = None
    if not (integer and CAPABILITIES.rejects("integrality", "duals")):
        duals = np.zeros(assembled.n_rows, dtype=np.float64)
    which = _defined(mosek, task, integer)
    if which is None:
        if reported in LIMITS:
            return Result(
                LIMITS[reported],
                False,
                0.0,
                _bound(mosek, task, integer, LIMITS[reported], 0.0),
                np.zeros(assembled.n_cols, dtype=np.float64),
                duals,
                task,
            )
        raise RuntimeError(
            "Mosek reported no solution and no limit; check the model and the options"
        )
    solution = _name(task.getsolsta(which))
    problem = _name(task.getprosta(which))
    if solution in OPTIMAL:
        status = "optimal"
    elif reported in LIMITS:
        status = LIMITS[reported]
    elif problem in OUTCOME:
        status = OUTCOME[problem]
    else:
        raise RuntimeError(
            f"Mosek reported the solution status {solution!r} and the problem "
            f"status {problem!r}; this adapter maps no outcome to them, report "
            f"them as a defect"
        )
    col_value = np.zeros(assembled.n_cols, dtype=np.float64)
    objective = 0.0
    feasible = solution in FEASIBLE
    if feasible:
        objective = float(task.getprimalobj(which))
        task.getxx(which, col_value)
        if duals is not None:
            task.gety(which, duals)
    return Result(
        status,
        feasible,
        objective,
        _bound(mosek, task, integer, status, objective),
        col_value,
        duals,
        task,
    )


def ray(backend: Any) -> npt.NDArray[np.float64] | None:
    """Return the primal ray of an unbounded model, or None where none exists.

    Mosek's certificate of dual infeasibility is a primal direction along
    which the objective falls without bound. The solution with the status
    `dual_infeas_cer` contains it.
    """
    import mosek

    for which in (mosek.soltype.bas, mosek.soltype.itr):
        if backend.solutiondef(which) and (
            _name(backend.getsolsta(which)) == "dual_infeas_cer"
        ):
            found = np.empty(backend.getnumvar(), dtype=np.float64)
            backend.getxx(which, found)
            return found
    return None
