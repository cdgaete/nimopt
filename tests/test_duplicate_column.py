"""Regression tests for the duplicate-column CSR pathology.

Background
----------
When a single constraint row references the same variable through two terms
(e.g. a ``Sum`` over a set plus a literal element of that set, or ``x[I] +
x[I]``), both terms resolve to the *same* solver column. Before the fix the
CSR assembler emitted that column index twice in one row; HiGHS rejects a row
with a repeated column index (it silently drops the whole row), which later
surfaced as a length-0 dual slice and a ``cannot reshape array of size 0 into
shape (n,)`` crash in solution extraction.

The bug affected two assembly paths independently:

* ``_expand_constraint`` -- the no-free-set / Python fallback, reached by
  ``Sum(BND, b[BND]) - b[opc]`` (the literal ``b[opc]`` carries a fixed index
  so the constraint fails the Rust gates).
* ``_build_multi_term_csr_rust`` -- the Rust multi-term path, reached by
  ``x[I] + x[I]`` and ``Sum(J, a[I,J]) - Sum(J, a[I,J])`` (free set, all terms
  indexed, no fixed index).

The fix accumulates contributions per (row, column) and drops exact zeros, so
cancellation collapses a column, doubling sums it, and a fully-cancelling row
becomes a valid empty ``0 <= rhs`` row -- never a duplicate index or a
length-0 array reaching the solver.

These tests assert the *coefficients are correct*, not merely that no crash
occurs (a dropped row previously produced silently-wrong models).
"""

import numpy as np

import nimopt as no
from nimopt.solvers import HiGHSDirectSolver, SolverStatus
from nimopt.solvers.highs_direct import _build_matrices_rust

BND_ELEMS = ["opc", "flyash", "ggbs", "silica", "limestone", "pozzolan"]


def _col_names(model):
    """Column name in solver order: scalar -> name, indexed -> name_e1_e2..."""
    import itertools

    names = []
    for var in model.variables.values():
        if not var.sets:
            names.append(var.name)
        else:
            for combo in itertools.product(*(s.elements for s in var.sets)):
                names.append(var.name + "_" + "_".join(str(e) for e in combo))
    return names


def _decode_rows(model):
    """Return one {colname: coef} dict per constraint row of ``model``.

    Also asserts no row repeats a column index -- the exact defect under test.
    """
    names = _col_names(model)
    mats = _build_matrices_rust(model)
    indptr = np.asarray(mats["indptr"])
    indices = np.asarray(mats["indices"])
    data = np.asarray(mats["data"])
    rows = []
    for r in range(mats["n_cons"]):
        cols = indices[indptr[r] : indptr[r + 1]]
        vals = data[indptr[r] : indptr[r + 1]]
        assert len(cols) == len(set(cols.tolist())), (
            f"row {r} has a duplicate column index: {cols.tolist()}"
        )
        rows.append({names[c]: float(v) for c, v in zip(cols, vals)})
    return rows


def _bnd_model():
    m = no.Model(name="bnd", sense="maximize")
    BND = no.Set("BND", BND_ELEMS)
    b = m.var("b", [BND], lb=0.0, ub=100.0)
    keffb = no.Param("keffb", [BND], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    m.set_objective(no.Sum(BND, b[BND]))
    return m, BND, b, keffb


# --------------------------------------------------------------------------
# Fallback path (_expand_constraint): Sum-over-set + literal element, no free set
# --------------------------------------------------------------------------


def test_fallback_sum_minus_literal_drops_column():
    """Sum(BND, b[BND]) - b[opc] <= 0  =>  opc coefficient cancels to 0."""
    m, BND, b, keffb = _bnd_model()
    m.eq("c", no.Sum(BND, b[BND]) - b["opc"] <= 0)
    (row,) = _decode_rows(m)
    assert "b_opc" not in row  # exactly cancelled and dropped
    for e in BND_ELEMS[1:]:
        assert row[f"b_{e}"] == 1.0
    assert set(row) == {f"b_{e}" for e in BND_ELEMS[1:]}


def test_fallback_sum_plus_literal_doubles_column():
    """Sum(BND, b[BND]) + b[opc] <= 1000  =>  opc coefficient is +2 (not zero).

    This is the case that falsifies a 'zero-filtering' explanation: the
    coefficient never becomes zero yet the pre-fix code still crashed.
    """
    m, BND, b, keffb = _bnd_model()
    m.eq("c", no.Sum(BND, b[BND]) + b["opc"] <= 1000)
    (row,) = _decode_rows(m)
    assert row["b_opc"] == 2.0
    for e in BND_ELEMS[1:]:
        assert row[f"b_{e}"] == 1.0


def test_fallback_weighted_cancellation():
    """Sum(keffb[BND]*b[BND]) - keffb[opc]*b[opc] <= 0  =>  opc term cancels."""
    m, BND, b, keffb = _bnd_model()
    m.eq("c", no.Sum(BND, keffb[BND] * b[BND]) - keffb["opc"] * b["opc"] <= 0)
    (row,) = _decode_rows(m)
    assert "b_opc" not in row
    # remaining keffb weights untouched
    assert row == {
        "b_flyash": 2.0,
        "b_ggbs": 3.0,
        "b_silica": 4.0,
        "b_limestone": 5.0,
        "b_pozzolan": 6.0,
    }


def test_fallback_two_literals_dropped():
    """Sum(BND, b[BND]) - b[opc] - b[flyash] <= 0  =>  both literals drop."""
    m, BND, b, keffb = _bnd_model()
    m.eq("c", no.Sum(BND, b[BND]) - b["opc"] - b["flyash"] <= 0)
    (row,) = _decode_rows(m)
    assert set(row) == {"b_ggbs", "b_silica", "b_limestone", "b_pozzolan"}
    assert all(v == 1.0 for v in row.values())


def test_fallback_distinct_variable_not_merged():
    """A literal on a different variable must NOT collide with the block."""
    m = no.Model(name="two", sense="maximize")
    BND = no.Set("BND", BND_ELEMS)
    T = no.Set("T", ["a", "b", "c"])
    b = m.var("b", [BND], lb=0.0, ub=100.0)
    x = m.var("x", [T], lb=0.0, ub=10.0)
    m.set_objective(no.Sum(BND, b[BND]))
    m.eq("c", no.Sum(BND, b[BND]) - x["a"] <= 0)
    (row,) = _decode_rows(m)
    assert row["b_opc"] == 1.0  # NOT cancelled - different column
    assert row["x_a"] == -1.0


# --------------------------------------------------------------------------
# Rust multi-term path (_build_multi_term_csr_rust): free set, same var twice
# --------------------------------------------------------------------------


def _ij_model():
    m = no.Model(name="ij", sense="maximize")
    I = no.Set("I", ["i1", "i2", "i3"])
    J = no.Set("J", ["j1", "j2"])
    a = m.var("a", [I, J], lb=0.0, ub=10.0)
    x = m.var("x", [I], lb=0.0, ub=10.0)
    m.set_objective(no.Sum(I, x[I]))
    return m, I, J, a, x


def test_rust_multiterm_same_var_doubles():
    """x[I] + x[I] <= 10 (free I)  =>  each row has x_i coefficient +2."""
    m, I, J, a, x = _ij_model()
    m.eq("c", x[I] + x[I] <= 10)
    rows = _decode_rows(m)
    assert len(rows) == 3
    for i, row in zip(["i1", "i2", "i3"], rows):
        assert row == {f"x_{i}": 2.0}


def test_rust_multiterm_full_cancellation_empty_rows():
    """Sum(J, a[I,J]) - Sum(J, a[I,J]) <= 0 (free I)  =>  every row empty."""
    m, I, J, a, x = _ij_model()
    m.eq("c", no.Sum(J, a[I, J]) - no.Sum(J, a[I, J]) <= 0)
    rows = _decode_rows(m)
    assert len(rows) == 3
    assert all(row == {} for row in rows)


def test_rust_multiterm_scalar_weighted_double():
    """3*x[I] - x[I] <= 5 (free I)  =>  each row has x_i coefficient +2."""
    m, I, J, a, x = _ij_model()
    m.eq("c", 3 * x[I] - x[I] <= 5)
    rows = _decode_rows(m)
    assert len(rows) == 3
    for i, row in zip(["i1", "i2", "i3"], rows):
        assert row == {f"x_{i}": 2.0}


# --------------------------------------------------------------------------
# End-to-end: the model must solve, not crash, and reflect the merged row
# --------------------------------------------------------------------------


def _solve(model):
    solver = HiGHSDirectSolver()
    solver.load_model(model)
    result = solver.solve()
    return solver, result


def test_solve_flagship_case_correct_optimum():
    """max Sum(b) s.t. Sum(BND,b) - b[opc] <= 0, b in [0,100].

    The row means b_flyash+...+b_pozzolan <= 0, so those are pinned to 0 while
    b_opc is free up to its bound -> objective 100, b_opc = 100.
    """
    m, BND, b, keffb = _bnd_model()
    m.eq("c", no.Sum(BND, b[BND]) - b["opc"] <= 0)
    solver, result = _solve(m)
    assert result.status == SolverStatus.OPTIMAL
    assert abs(result.objective_value - 100.0) < 1e-6
    values = solver.get_variable_values()
    assert abs(values[0] - 100.0) < 1e-6  # b_opc (first column)
    assert all(abs(v) < 1e-6 for v in values[1:])  # all other binders 0


def test_solve_rust_multiterm_no_crash():
    """x[I] + x[I] <= 10 must solve: 2*x_i <= 10 -> x_i = 5, objective 15."""
    m, I, J, a, x = _ij_model()
    m.eq("c", x[I] + x[I] <= 10)
    solver, result = _solve(m)
    assert result.status == SolverStatus.OPTIMAL
    assert abs(result.objective_value - 15.0) < 1e-6


def test_solve_full_cancellation_no_crash():
    """A fully-cancelling row is a valid trivial constraint, not a crash."""
    m, I, J, a, x = _ij_model()
    m.eq("c", no.Sum(J, a[I, J]) - no.Sum(J, a[I, J]) <= 0)
    solver, result = _solve(m)
    assert result.status == SolverStatus.OPTIMAL
