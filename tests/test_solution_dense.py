import numpy as np
from nimblend import DenseArray, SparseArray

from nimopt import Model, Param, Set, Sum, subset


def transport(sparse_arcs=False):
    """A 2x3 transport model, over the full product or over three arcs."""
    m = Model("t")
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    arcs = subset(
        (P, W),
        {"P": np.array(["p1", "p1", "p2"]), "W": np.array(["w1", "w2", "w3"])},
    )
    x = m.var("x", (P, W), subset=arcs if sparse_arcs else None)
    cost = Param.from_dense("c", (P, W), np.arange(1.0, 7.0).reshape(2, 3))
    one = Param.from_dense("one", (P, W), np.ones((2, 3)))
    m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))
    m.eq(
        "supply",
        Sum(W, one[P, W] * x[P, W])
        >= Param.from_dense("a", (P,), np.array([1.0, 2.0]))[P],
    )
    return m


def test_a_full_product_primal_is_dense():
    sol = transport().solve()
    values = sol.primal("x")
    assert isinstance(values, DenseArray)
    assert values.dims == ("P", "W")
    assert values.shape == (2, 3)
    assert np.array_equal(values.to_dense(), sol._col_value[:6].reshape(2, 3))


def test_a_subset_primal_stays_sparse():
    sol = transport(sparse_arcs=True).solve()
    values = sol.primal("x")
    assert isinstance(values, SparseArray)
    # three arcs of the six cells, so the grid is never materialised
    assert values.nnz == 3
    assert values.shape == (2, 3)


def test_a_dense_primal_declares_that_absence_is_unknown():
    values = transport().solve().primal("x")
    assert values.absence == "unknown"
    assert values.mask is None
    assert values.nnz == 6


def test_a_dual_covering_its_frame_is_dense():
    sol = transport().solve()
    duals = sol.dual("supply")
    assert isinstance(duals, DenseArray)
    assert duals.dims == ("P",)
    assert np.array_equal(duals.to_dense(), sol._row_dual[:2])


def test_a_dual_over_dropped_rows_stays_sparse():
    m = Model("lag", sense="max")
    S = Set("S", np.array(["s1", "s2"]))
    T = Set("T", np.array(["t0", "t1", "t2"]))
    level = m.var(
        "level", (S, T), upper=Param.from_dense("cap", (S,), np.array([5.0, 5.0]))
    )
    one = Param.from_dense("one", (S, T), np.ones((2, 3)))
    m.set_objective(Sum(S, T, one[S, T] * level[S, T]))
    # the lag drops t0, so the constraint states four of the six rows
    m.eq(
        "bal",
        level[S, T] - level[S, T - 1]
        <= Param.from_dense("step", (S, T), np.ones((2, 3)))[S, T],
    )
    sol = m.solve()
    # the sense is load-bearing: minimising this model answers 0.0 and leaves
    # every assertion below standing, so the objective is asserted beside them
    assert sol.objective == 30.0
    duals = sol.dual("bal")
    assert isinstance(duals, SparseArray)
    assert duals.nnz == 4
    assert duals.shape == (2, 3)


def test_the_two_routes_carry_the_same_labels():
    dense = transport().solve().primal("x")
    sparse = transport(sparse_arcs=True).solve().primal("x")
    assert dense.dims == sparse.dims
    named = np.array(["p1", "p2"])
    assert np.array_equal(dense.coords["P"].to_index(np.arange(2)), named)
    assert np.array_equal(sparse.coords["P"].to_index(np.arange(2)), named)
