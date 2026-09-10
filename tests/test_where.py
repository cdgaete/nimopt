import numpy as np
import pytest

from nimopt import Model, Param, Set, Sum, product, subset


def arcs_model():
    """A 2x3 grid where only three of the six arcs exist."""
    m = Model("arcs")
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    x = m.var("x", (P, W))
    one = Param.from_dense("one", (P, W), np.ones((2, 3)))
    cap = Param.from_dense("cap", (P,), np.array([10.0, 20.0]))
    arcs = subset(
        (P, W),
        {
            "P": np.array(["p1", "p1", "p2"]),
            "W": np.array(["w1", "w2", "w3"]),
        },
    )
    return m, P, W, x, one, cap, arcs


def test_a_sum_over_a_condition_keeps_only_the_named_coordinates():
    m, P, W, x, one, cap, arcs = arcs_model()
    m.constraint("supply", Sum(W, one[P, W] * x[P, W], where=arcs) <= cap[P])
    # three arcs across two rows, against six on the full product
    assert m.nnz == 3
    assert m.n_rows == 2


def test_a_condition_leaves_the_full_product_unrestricted():
    m, P, W, x, one, cap, arcs = arcs_model()
    m.constraint("supply", Sum(W, one[P, W] * x[P, W]) <= cap[P])
    assert m.nnz == 6


def test_a_condition_names_the_columns_the_arcs_stand_for():
    m, P, W, x, one, cap, arcs = arcs_model()
    m.constraint("supply", Sum(W, one[P, W] * x[P, W], where=arcs) <= cap[P])
    assembled = m.assemble()
    # p1 reaches w1 and w2, which are columns 0 and 1; p2 reaches w3, column 5
    assert np.array_equal(assembled.indices, np.array([0, 1, 5], dtype=np.int32))


def test_a_condition_over_a_dimension_the_variable_lacks_is_refused():
    m, P, W, x, one, cap, arcs = arcs_model()
    other = Set("Z", np.array(["z1", "z2"]))
    wrong = subset((other,), {"Z": np.array(["z1"])})
    with pytest.raises(ValueError, match="does not carry"):
        Sum(W, one[P, W] * x[P, W], where=wrong)


def test_a_condition_survives_a_coefficient_and_a_scale():
    m, P, W, x, one, cap, arcs = arcs_model()
    expression = 2.0 * Sum(W, one[P, W] * x[P, W], where=arcs)
    m.constraint("supply", expression <= cap[P])
    assembled = m.assemble()
    assert m.nnz == 3
    assert np.array_equal(assembled.values, np.array([2.0, 2.0, 2.0]))


def test_a_constraint_condition_drops_the_rows_it_omits():
    m, P, W, x, one, cap, arcs = arcs_model()
    only_p1 = subset((P,), {"P": np.array(["p1"])})
    m.constraint("supply", Sum(W, one[P, W] * x[P, W]) <= cap[P], where=only_p1)
    # p2's row is not stated, so its three coefficients are not either
    assert m.n_rows == 1
    assert m.nnz == 3


def test_a_constraint_condition_over_the_wrong_frame_is_refused():
    m, P, W, x, one, cap, arcs = arcs_model()
    with pytest.raises(ValueError, match="free dimensions"):
        m.constraint("supply", Sum(W, one[P, W] * x[P, W]) <= cap[P], where=arcs)


def test_a_constraint_condition_and_a_sum_condition_compose():
    m, P, W, x, one, cap, arcs = arcs_model()
    only_p1 = subset((P,), {"P": np.array(["p1"])})
    m.constraint(
        "supply", Sum(W, one[P, W] * x[P, W], where=arcs) <= cap[P], where=only_p1
    )
    # of the three arcs, only p1's two stand once p2's row is dropped
    assert m.n_rows == 1
    assert m.nnz == 2


def test_a_condition_may_name_a_dimension_the_coefficient_introduces():
    B = Set("B", np.array(["b0", "b1"]))
    L = Set("L", np.array(["l0", "l1"]))
    T = Set("T", np.arange(2))
    m = Model("m")
    p = m.var("p", (L, T), lower=-np.inf)
    grid = np.zeros((2, 2, 2))
    grid[0, 0, :] = -1.0
    grid[1, 1, :] = -1.0
    inc = Param.from_dense("inc", (B, L, T), grid)
    live = subset(
        (B, L, T),
        {
            "B": np.array(["b0", "b0"]),
            "L": np.array(["l0", "l0"]),
            "T": np.array([0, 1]),
        },
    )
    rhs = Param.from_dense("rhs", (B, T), np.zeros((2, 2)))
    m.constraint("balance", Sum(L, inc[B, L, T] * p[L, T], where=live) >= rhs[B, T])
    assembled = m.assemble()
    # only b0's own link entries survive the condition
    assert int(assembled.values.size) == 2


def network(rows):
    """A bus balance whose terms each reach some of its rows.

    `rows` is called with the two sets and answers what states them, so one
    model is stated four ways and the spellings are compared on equal terms.
    """
    B = Set("B", np.array(["b0", "b1"]))
    T = Set("T", np.array([0, 1, 2]))
    L = Set("L", np.array(["l0", "l1"]))
    m = Model("network")
    flow = m.var("flow", (L, T), lower=-np.inf)
    inc = Param.from_long(
        "inc",
        (B, L, T),
        {
            "B": np.array(["b0", "b1", "b0", "b1"]),
            "L": np.array(["l0", "l0", "l1", "l1"]),
            "T": np.array([0, 0, 1, 1]),
        },
        np.array([1.0, -1.0, 1.0, -1.0]),
    )
    zero = Param.from_dense("zero", (B, T), np.zeros((2, 3)))
    m.constraint(
        "balance", Sum(L, inc[B, L, T] * flow[L, T]) == zero[B, T], over=rows(B, T)
    )
    return m


def marked(B, T):
    """Two of the six bus-hours, as label columns."""
    return {"B": np.array(["b0", "b1"]), "T": np.array([0, 0])}


def shape(model):
    """The rows and the coefficients a model built, as one pair."""
    return model.n_rows, model.nnz


def test_rows_stated_as_sets_are_the_rows_the_product_states():
    # the full product, given as a domain and given as the sets it spans;
    # the two spellings agree on rows and on nonzeros
    assert shape(network(lambda B, T: product((B, T)))) == (6, 4)
    assert shape(network(lambda B, T: (B, T))) == (6, 4)


def test_rows_stated_as_a_parameter_are_the_coordinates_it_carries():
    # a parameter's support is a set of coordinates, which is what a row
    # domain is; an over= is not always a full product, so this is the shape
    # that needs the second spelling
    assert shape(network(lambda B, T: subset((B, T), marked(B, T)))) == (2, 2)
    assert shape(
        network(lambda B, T: Param.from_long("mark", (B, T), marked(B, T), np.ones(2)))
    ) == (2, 2)


def test_a_condition_named_as_a_parameter_restricts_a_sum():
    P = Set("P", np.array(["p0", "p1"]))
    W = Set("W", np.array(["w0", "w1"]))
    arcs = Param.from_long(
        "arcs",
        (P, W),
        {"P": np.array(["p0", "p0", "p1"]), "W": np.array(["w0", "w1", "w0"])},
        np.ones(3),
    )
    m = Model("m")
    x = m.var("x", (P, W), upper=10.0)
    one = Param.from_dense("one", (P, W), np.ones((2, 2)))
    cap = Param.from_dense("cap", (P,), np.array([5.0, 5.0]))
    m.constraint("capacity", Sum(W, one[P, W] * x[P, W], where=arcs) <= cap[P])
    # three arcs across two rows, against four on the full product
    assert m.nnz == 3
    assert m.n_rows == 2


def test_rows_over_the_wrong_dimensions_are_refused_however_they_are_stated():
    with pytest.raises(ValueError, match="its over= is over"):
        network(lambda B, T: (T,))
