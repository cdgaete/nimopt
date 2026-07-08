"""Regression tests: additive numeric constant on the LHS of a constraint.

`z + 3 <= 9` must be solved as `z <= 6` -- the LHS constant has to be moved to
the RHS (RHS' = RHS - lhs.const). The expression layer captures it correctly
(`con.lhs.const == 3.0`), but the direct solver's matrix builder silently
dropped a bare *literal* constant on the LHS of a scalar (dimensionless)
constraint, so `z + 3 <= 9` was built as `z <= 9` and maximised to 9 instead
of 6.

(The bare-*param* LHS case -- `x[I] + off[I] <= cap[I]` -- was already fixed;
this covers the pure numeric-constant scalar path that still leaked.)
"""

import numpy as np

import nimopt as no
from nimopt.solvers import HiGHSDirectSolver


def _solve(m):
    solver = HiGHSDirectSolver(use_rust=True)
    solver.load_model(m)
    result = solver.solve()
    assert result.status.value == "optimal", f"status={result.status}"
    return result, solver.get_solution()


def test_scalar_lhs_const_plus():
    """maximize z s.t. z + 3 <= 9  =>  z == 6."""
    m = no.Model(sense="maximize")
    z = m.var("z", [], lb=0, ub=100)
    m.eq("only", z + 3 <= 9)
    m.set_objective(z)

    result, sol = _solve(m)
    assert result.objective_value == 6.0
    # read the raw values array (sol.var() routes a scalar through a broken
    # to_array() path unrelated to this bug)
    assert float(np.asarray(sol.variables["z"].values).ravel()[0]) == 6.0


def test_scalar_lhs_const_minus():
    """maximize z s.t. z - 3 <= 6  =>  z == 9 (const moves as +3)."""
    m = no.Model(sense="maximize")
    z = m.var("z", [], lb=0, ub=100)
    m.eq("only", z - 3 <= 6)
    m.set_objective(z)

    result, _ = _solve(m)
    assert result.objective_value == 9.0


def test_scalar_lhs_const_geq():
    """minimize z s.t. z + 3 >= 12  =>  z == 9."""
    m = no.Model(sense="minimize")
    z = m.var("z", [], lb=0, ub=100)
    m.eq("only", z + 3 >= 12)
    m.set_objective(z)

    result, _ = _solve(m)
    assert result.objective_value == 9.0
