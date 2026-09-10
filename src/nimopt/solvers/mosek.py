"""The Mosek adapter.

Mosek's task interface takes a row-wise matrix as the three arrays `Assembled`
carries, with the row pointer split into where each row starts and where it
ends. A bound crosses as a key beside its two numbers, derived from the bound
arrays, so an infinite bound is stated as the absence of one rather than as a
number the solver would read.

A solve reports through three signals. The termination code names a limit the
optimizer stopped at; the solution status names an optimum; the problem
status names infeasibility and unboundedness. Each is mapped into the seam's
`STATUS` through `LIMITS`, `OPTIMAL` and `OUTCOME`, or listed in `FAILED` and
raised. A signal the adapter does not know is refused rather than folded into
a catch-all, because a caller reading a status it was never given cannot tell
an answer from the absence of one.

Mosek runs only its mixed-integer optimizer on a model with integer columns
and refuses another; the adapter names that cause rather than passing the
solver's code through.
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
    refused=(("integrality", "duals"),),
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
    """What Mosek calls an enum member, without its enum."""
    return str(member).split(".")[-1]


def _keys(
    lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64], keys: Any
) -> npt.NDArray[np.int32]:
    """Mosek's bound key per pair, from which of its two bounds are finite."""
    below = np.isneginf(lower)
    above = np.isposinf(upper)
    held = np.full(lower.shape, int(keys.ra), dtype=np.int32)
    held[below & above] = int(keys.fr)
    held[below & ~above] = int(keys.up)
    held[~below & above] = int(keys.lo)
    held[lower == upper] = int(keys.fx)
    return held


def _configured(mosek: Any, settings: dict[str, Any]) -> Any:
    """A task carrying `settings`, each under the enum Mosek reads it from."""
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
    """`task` holding the assembled model, with the matrix passed row-wise."""
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
    """The solution the task holds for this model, or None where it holds none.

    A mixed-integer model answers through its integer solution. A continuous
    one answers through the basic solution where the optimizer built one and
    through the interior solution otherwise, which is what an interior point
    method without a basis identification leaves.
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
    the relaxations it solved. The bound is None at a count of zero and None
    where Mosek reports an infinite value. For a model without integer columns
    Mosek defines no dual bound. The bound is then the objective at status
    `optimal` and None at any other status.
    """
    if not integer:
        return objective if status == "optimal" else None
    if task.getintinf(mosek.iinfitem.mio_num_relax) == 0:
        return None
    value = float(task.getdouinf(mosek.dinfitem.mio_obj_bound))
    return value if np.isfinite(value) else None


def solve(
    assembled: "Assembled", sense: str, options: Mapping[str, Any] | None = None
) -> Result:
    """Solve an assembled model and return its status, its values and its bound.

    The task Mosek built is returned in `Result.backend`. A session holding it
    reads a ray from the same solved instance. A model with integer columns is
    returned with no duals. Mosek defines none for an integer solution, and
    this adapter does not report that pair.
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
                "integer columns and refuses another rather than solving the "
                "relaxation; leave `method` at 'choose' for this model"
            ) from error
        raise
    if reported in FAILED:
        raise RuntimeError(
            f"Mosek stopped without solving the model: {reported}. The model "
            f"was passed to the solver but no outcome was reached."
        )
    if reported != "ok" and reported not in LIMITS:
        raise RuntimeError(
            f"Mosek reported termination code {reported!r}, which this "
            f"adapter does not read; nimopt names an outcome or refuses it"
        )
    integer = bool(assembled.integrality.any())
    duals = None
    if not (integer and CAPABILITIES.refuses("integrality", "duals")):
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
            "Mosek reported no solution and no limit; the model was passed "
            "to the solver but no outcome was reached"
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
            f"Mosek reported solution status {solution!r} and problem status "
            f"{problem!r}, which this adapter does not read; nimopt names an "
            f"outcome or refuses it"
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
    """The direction an unbounded model runs off in, or None where there is none.

    Mosek's certificate of dual infeasibility is a primal direction along
    which the objective falls without bound, and the solution holding it
    reports that status.
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
