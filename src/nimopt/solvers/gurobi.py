"""The Gurobi adapter.

Gurobi's matrix interface reads a constraint matrix as a dense array or a
SciPy sparse matrix. The three arrays `Assembled` contains are CSR already.
The adapter wraps them in a `csr_matrix` over the same buffers and adds every
row in one call. No part of the matrix is copied.

The adapter sets `InfUnbdInfo` to 1, and declares a native ray. Without that
parameter Gurobi reports `INF_OR_UNBD` for an unbounded model and computes no
ray. `DualReductions` keeps the value Gurobi sets. A model Gurobi does not
separate is reported as `unbounded_or_infeasible`.

`OUTCOME` maps a Gurobi model status into `STATUS`. `FAILED` lists the Gurobi
statuses this adapter raises on. A status in neither raises.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from nimopt.model import Assembled

from nimopt.row import senses
from nimopt.solvers.base import Capabilities, Result
from nimopt.solvers.options import translated

BACKEND = "gurobipy"

CAPABILITIES = Capabilities(
    "gurobi",
    {
        "integrality": "native",
        "duals": "native",
        "conflict": "native",
        "ray": "native",
    },
    rejected=(("integrality", "duals"),),
)

OUTCOME = {
    "OPTIMAL": "optimal",
    "INFEASIBLE": "infeasible",
    "UNBOUNDED": "unbounded",
    "INF_OR_UNBD": "unbounded_or_infeasible",
    "ITERATION_LIMIT": "iteration_limit",
    "NODE_LIMIT": "node_limit",
    "TIME_LIMIT": "time_limit",
    "SOLUTION_LIMIT": "solution_limit",
    "WORK_LIMIT": "work_limit",
    "CUTOFF": "objective_bound",
    "USER_OBJ_LIMIT": "objective_target",
    "MEM_LIMIT": "memory_limit",
    "INTERRUPTED": "interrupt",
}

SENSES = {"min": "MINIMIZE", "max": "MAXIMIZE"}

RELATIONS = {"<=": "<", ">=": ">", "==": "="}

FAILED = (
    "LOADED",
    "NUMERIC",
    "SUBOPTIMAL",
    "INPROGRESS",
    "LOCALLY_SOLVED",
    "LOCALLY_OPTIMAL",
    "LOCALLY_INFEASIBLE",
)

OPTION_NAMES = {
    "time_limit": "TimeLimit",
    "iteration_limit": "IterationLimit",
    "node_limit": "NodeLimit",
    "mip_gap": "MIPGap",
    "mip_abs_gap": "MIPGapAbs",
    "feasibility_tol": "FeasibilityTol",
    "optimality_tol": "OptimalityTol",
    "threads": "Threads",
    "seed": "Seed",
    "log": "OutputFlag",
    "presolve": "Presolve",
    "method": "Method",
    "newton_system": None,
    "crossover": "Crossover",
    "pdlp_tol": None,
}

OPTION_VALUES = {
    "presolve": {"off": 0, "choose": -1, "on": 2},
    "method": {"choose": -1, "simplex": 1, "barrier": 2},
    "crossover": {"choose": -1, "off": 0, "on": 1},
}


def _matrix(assembled: "Assembled") -> Any:
    """Return the assembled matrix as the `csr_matrix` Gurobi reads.

    `Assembled` contains the column indices, the values and the row pointer of
    a row-wise matrix. The wrapper shares those buffers and copies nothing.
    """
    from scipy.sparse import csr_matrix

    return csr_matrix(
        (assembled.values, assembled.indices, assembled.indptr),
        shape=(assembled.n_rows, assembled.n_cols),
    )


def _sense(relations: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return each row's relation as the character Gurobi uses for it."""
    named = np.empty(relations.shape, dtype="<U1")
    for relation, character in RELATIONS.items():
        named[relations == relation] = character
    return named


def _named(statuses: Any, code: Any) -> str:
    """Return the name Gurobi defines for a status code."""
    for name in dir(statuses):
        if name.isupper() and getattr(statuses, name) == code:
            return name
    return str(code)


def _bound(model: Any, integer: bool, status: str, objective: float) -> float | None:
    """Return the bound Gurobi proved on the optimal objective, or None.

    For a model with integer columns the bound is `ObjBound`. It is None where
    Gurobi reports an infinite value. For a model without integer columns
    `ObjBound` is not a dual bound. The bound is then the objective at status
    `optimal` and None at any other status.
    """
    if not integer:
        return objective if status == "optimal" else None
    value = float(model.ObjBound)
    return value if np.isfinite(value) else None


def solve(
    assembled: "Assembled", sense: str, options: Mapping[str, Any] | None = None
) -> Result:
    """Solve an assembled model and return its status, its values and its bound.

    `Result.backend` is the model Gurobi built. A session reads a conflict or a
    ray from that solved instance. `Result.row_dual` is None for a model with
    integer columns.
    """
    import gurobipy as gp
    from gurobipy import GRB

    if sense not in SENSES:
        raise ValueError(f"sense is 'min' or 'max'; got {sense!r}")
    env = gp.Env(
        params={
            "OutputFlag": 0,
            **translated(options, OPTION_NAMES, OPTION_VALUES, solver="gurobi"),
            "InfUnbdInfo": 1,
        }
    )
    model = gp.Model(env=env)
    kinds = np.where(assembled.integrality.astype(bool), GRB.INTEGER, GRB.CONTINUOUS)
    columns = model.addMVar(
        assembled.n_cols,
        lb=assembled.col_lower,
        ub=assembled.col_upper,
        obj=assembled.col_cost,
        vtype=kinds,
    )
    relations = senses(assembled.row_lower, assembled.row_upper)
    rhs = np.where(relations == ">=", assembled.row_lower, assembled.row_upper)
    rows = model.addMConstr(_matrix(assembled), columns, _sense(relations), rhs)
    model.ModelSense = getattr(GRB, SENSES[sense])
    model.optimize()

    reported = _named(GRB.Status, model.Status)
    if reported in FAILED:
        raise RuntimeError(
            f"Gurobi stopped at model status {reported} without solving the "
            f"model; check the model and the options"
        )
    if reported not in OUTCOME:
        raise RuntimeError(
            f"Gurobi reported the model status {reported!r}; this adapter maps "
            f"no outcome to it, report it as a defect"
        )
    integer = bool(assembled.integrality.any())
    status = OUTCOME[reported]
    duals = None
    if not (integer and CAPABILITIES.rejects("integrality", "duals")):
        duals = np.zeros(assembled.n_rows, dtype=np.float64)
    if not model.SolCount:
        return Result(
            status,
            False,
            0.0,
            _bound(model, integer, status, 0.0),
            np.zeros(assembled.n_cols, dtype=np.float64),
            duals,
            model,
        )
    if duals is not None:
        duals = np.asarray(rows.Pi, dtype=np.float64)
    objective = float(model.ObjVal)
    return Result(
        status,
        True,
        objective,
        _bound(model, integer, status, objective),
        np.asarray(columns.X, dtype=np.float64),
        duals,
        model,
    )


def conflict(backend: Any) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the rows and columns of an irreducible infeasible subsystem.

    The subsystem is computed here, not at the solve. Gurobi computes it over
    the model with its integrality. A model feasible as a relaxation and
    infeasible as a mixed-integer program yields a subsystem.
    """
    backend.computeIIS()
    rows = [at for at, row in enumerate(backend.getConstrs()) if row.IISConstr]
    columns = [
        at
        for at, column in enumerate(backend.getVars())
        if column.IISLB or column.IISUB
    ]
    return np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)


def ray(backend: Any) -> npt.NDArray[np.float64] | None:
    """Return the primal ray of an unbounded model, or None where none exists."""
    from gurobipy import GurobiError

    try:
        found = backend.getAttr("UnbdRay", backend.getVars())
    except (AttributeError, GurobiError):
        return None
    return np.asarray(found, dtype=np.float64)
