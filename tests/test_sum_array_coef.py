"""Regression tests for Bug 1: array coefficient over a *summed* axis.

`Sum(j, a[i,j] * x[j])` (i free, j summed, coef `a` has both axes, variable `x`
has only the summed axis) must compute the true column sum `Σ_j a[i,j]·x[j]`.

The bug: the coefficient's summed axis was pinned to index 0 in the direct
solver's Rust CSR builder (`build_multi_term_csr`), yielding `a[i,0]·Σ_j x[j]`.
This silently corrupts any transport/supply/assignment/blending constraint of
the shape `Sum(j, a[i,j]*x[j]) <=/== b[i]`.

These go through HiGHSDirectSolver(use_rust=True) on purpose: that is the path
that carries the defect (the LP writer is a separate, already-fixed path).
"""

import numpy as np

import nimopt as no
from nimopt import Sum
from nimopt.solvers import HiGHSDirectSolver


def _solve(m):
    solver = HiGHSDirectSolver(use_rust=True)
    solver.load_model(m)
    result = solver.solve()
    assert result.status.value == "optimal", f"status={result.status}"
    return solver.get_solution()


def test_sum_array_coef_over_summed_axis_column_sums():
    """y[R] == Sum(P, u[P,R]*x[P]) with x fixed to 1 must give column sums of u."""
    P = no.Set("P", ["p1", "p2", "p3"])
    R = no.Set("R", ["r1", "r2"])
    # column sums of u = [1+2+3, 10+20+30] = [6, 60]
    u = no.Param("u", [P, R], [[1, 10], [2, 20], [3, 30]])

    m = no.Model(sense="minimize")
    x = m.var("x", [P], lb=1, ub=1)  # x fixed to 1
    y = m.var("y", [R], lb=0, ub=1000)

    m.eq("ydef", y[R] == Sum(P, u[P, R] * x[P]))
    m.set_objective(Sum(R, y[R]))

    sol = _solve(m)
    y_vals = np.asarray(sol.var("y").values, dtype=float).ravel()

    np.testing.assert_allclose(y_vals, [6.0, 60.0])


def test_matrix_vector_product_Ax_equals_b():
    """Canonical LP standard form: Σ_j A[i,j]*x[j] == b[i] (free i, summed j).

    The summed variable x[j] has ONLY the summed axis, so the coef A[i,j] has a
    free axis (i) the variable lacks -- exactly the shape that triggers Bug 1.
    (Contrast the classic transport shape x[i,j], where the variable carries the
    free axis and the correct path is taken.)

    A = [[1,2],[3,4]], b = [5,11]  =>  unique solution x = [1,2].

    This single-term Sum routes through a *different* Rust builder than the
    two-variable case above (`build_sum_csr`, not `build_multi_term_csr`).
    Under the bug that builder indexes a coef whose free axis (i) the variable
    lacks and panics with index-out-of-bounds -- so the same root defect must be
    fixed in both builders.
    """
    I = no.Set("I", ["i1", "i2"])
    J = no.Set("J", ["j1", "j2"])
    A = no.Param("A", [I, J], [[1.0, 2.0], [3.0, 4.0]])
    b = no.Param("b", [I], [5.0, 11.0])

    m = no.Model(sense="minimize")
    x = m.var("x", [J], lb=0, ub=100)
    m.eq("Ax", Sum(J, A[I, J] * x[J]) == b[I])
    m.set_objective(Sum(J, x[J]))

    sol = _solve(m)  # under the bug this is infeasible -> assertion in _solve fires
    x_vals = np.asarray(sol.var("x").values, dtype=float).ravel()

    np.testing.assert_allclose(x_vals, [1.0, 2.0])
