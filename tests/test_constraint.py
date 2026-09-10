import numpy as np
import pytest
from nimblend import EntryBuffer

from nimopt.constraint import Constraint
from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set, product, subset
from nimopt.term import Sum


def model():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    x = m.var("x", (P, W))
    d = Param.from_dense("d", (P, W), np.arange(1, 7, dtype=float).reshape(2, 3))
    cap = Param.from_dense("cap", (P,), np.array([100.0, 200.0]))
    return m, P, W, x, d, cap


def test_a_constraint_measures_its_rows_and_coefficients():
    _, P, W, x, d, cap = model()
    c = Constraint("supply", Sum(W, d[P, W] * x[P, W]) <= cap[P])
    assert c.n_rows == 2
    assert c.nnz == 6


def bounds_of(constraint):
    """The pair a constraint writes, in vectors the caller owns."""
    lower = np.empty(constraint.n_rows)
    upper = np.empty(constraint.n_rows)
    constraint.write_bounds(lower, upper)
    return lower, upper


def test_the_bounds_follow_the_sense():
    _, P, W, x, d, cap = model()
    lower, upper = bounds_of(Constraint("c", Sum(W, d[P, W] * x[P, W]) <= cap[P]))
    assert list(upper) == [100.0, 200.0]
    assert list(lower) == [-np.inf, -np.inf]
    lower, upper = bounds_of(Constraint("c", Sum(W, d[P, W] * x[P, W]) >= cap[P]))
    assert list(lower) == [100.0, 200.0]
    assert list(upper) == [np.inf, np.inf]
    lower, upper = bounds_of(Constraint("c", Sum(W, d[P, W] * x[P, W]) == cap[P]))
    assert list(lower) == list(upper) == [100.0, 200.0]


def test_a_number_on_the_right_bounds_every_row_alike():
    _, P, W, x, d, _ = model()
    _, upper = bounds_of(Constraint("c", Sum(W, d[P, W] * x[P, W]) <= 7.0))
    assert list(upper) == [7.0, 7.0]


def test_write_bounds_fills_the_caller_s_vectors_and_allocates_nothing():
    _, P, W, x, d, cap = model()
    c = Constraint("c", Sum(W, d[P, W] * x[P, W]) <= cap[P])
    lower = np.zeros(c.n_rows + 2)
    upper = np.zeros(c.n_rows + 2)
    c.write_bounds(lower[1:-1], upper[1:-1])
    # the rows outside the slice are the caller's and stay untouched
    assert lower[0] == 0.0 and lower[-1] == 0.0
    assert list(upper[1:-1]) == [100.0, 200.0]


def test_a_row_the_right_hand_side_does_not_carry_is_not_stated():
    _, P, W, x, d, _ = model()
    partial = Param.from_long("cap", (P,), {"P": np.array(["p2"])}, np.array([9.0]))
    c = Constraint("c", Sum(W, d[P, W] * x[P, W]) <= partial[P])
    assert c.n_rows == 1
    assert list(c.rows.labels()["P"]) == ["p2"]
    assert c.nnz == 3


def test_the_block_is_written_into_the_buffer_it_is_given():
    _, P, W, x, d, cap = model()
    c = Constraint("supply", Sum(W, d[P, W] * x[P, W]) <= cap[P])
    buffer = EntryBuffer(2, c.nnz)
    block = c.write_into(buffer, 0)
    assert block.dims == ("__row__", "__column__")
    assert block.to_dense().tolist() == [
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
    ]
    _, written = buffer.written()
    assert list(written) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_the_block_is_numbered_from_the_row_it_is_given():
    _, P, W, x, d, cap = model()
    c = Constraint("supply", Sum(W, d[P, W] * x[P, W]) <= cap[P])
    buffer = EntryBuffer(2, c.nnz)
    block = c.write_into(buffer, 7)
    assert block.coordinates()[0].tolist() == [7, 7, 7, 8, 8, 8]


def test_a_right_hand_side_over_the_wrong_dimensions_raises():
    _, P, W, x, d, _ = model()
    wrong = Param.from_dense("w", (W,), np.ones(3))
    with pytest.raises(ValueError, match="free dimensions"):
        Constraint("c", Sum(W, d[P, W] * x[P, W]) <= wrong[W])


def test_something_that_is_not_a_comparison_raises():
    _, P, W, x, d, _ = model()
    with pytest.raises(TypeError, match="comparison"):
        Constraint("c", Sum(W, d[P, W] * x[P, W]))


def _two_bus_model():
    """A generator at b0 and a link from b0 to b1, over two hours."""
    B = Set("B", np.array(["b0", "b1"]))
    G = Set("G", np.array(["g0"]))
    L = Set("L", np.array(["l0"]))
    T = Set("T", np.arange(2))
    m = Model("network")
    at_bus = subset(
        (B, G, T),
        {
            "B": np.array(["b0", "b0"]),
            "G": np.array(["g0", "g0"]),
            "T": np.array([0, 1]),
        },
    )
    gen = m.var("gen", (B, G, T), subset=at_bus)
    flow = m.var("flow", (L, T))
    ones = Param.from_dense("ones", (B, G, T), np.ones((2, 1, 2)))
    grid = np.zeros((2, 1, 2))
    grid[0, 0, :] = -1.0
    grid[1, 0, :] = 0.9
    inc = Param.from_dense("inc", (B, L, T), grid)
    expression = Sum(G, ones[B, G, T] * gen[B, G, T]) + Sum(
        L, inc[B, L, T] * flow[L, T]
    )
    return m, B, T, expression, ones


def test_rows_derived_from_the_terms_drop_a_bus_a_term_misses():
    m, B, T, expression, _ = _two_bus_model()
    load = Param.from_dense("load", (B, T), np.zeros((2, 2)))
    constraint = m.constraint("balance", expression == load[B, T])
    # the link has an entry at both buses and the generator at b0 only; the
    # intersection keeps b0's two hours and drops b1 entirely
    assert constraint.n_rows == 2


def test_a_stated_row_domain_carries_every_bus():
    m, B, T, expression, _ = _two_bus_model()
    load = Param.from_dense("load", (B, T), np.zeros((2, 2)))
    constraint = m.constraint("balance", expression == load[B, T], over=product((B, T)))
    assert constraint.n_rows == 4
    # the generator's two entries and the link's four
    assert constraint.nnz == 6


def test_stating_rows_and_narrowing_them_together_is_refused():
    m, B, T, expression, _ = _two_bus_model()
    load = Param.from_dense("load", (B, T), np.zeros((2, 2)))
    rows = product((B, T))
    with pytest.raises(ValueError, match="pass one of them"):
        m.constraint("balance", expression == load[B, T], where=rows, over=rows)


def test_a_stated_row_domain_over_other_dimensions_is_refused():
    m, B, T, expression, _ = _two_bus_model()
    load = Param.from_dense("load", (B, T), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="free dimensions"):
        m.constraint("balance", expression == load[B, T], over=product((T,)))


def test_a_right_hand_side_missing_a_stated_row_is_refused():
    m, B, T, expression, _ = _two_bus_model()
    # both hours of b0 alone, so b1's two rows have no right-hand side
    partial = Param.from_long(
        "load",
        (B, T),
        {"B": np.array(["b0", "b0"]), "T": np.array([0, 1])},
        np.zeros(2),
    )
    with pytest.raises(ValueError, match="a value at every row"):
        m.constraint("balance", expression == partial[B, T], over=product((B, T)))


def test_a_constraint_keeps_the_row_domain_it_was_given():
    # absent() re-runs the narrowing, which it cannot do from inputs the
    # constraint discarded
    P = Set("P", np.array(["p1", "p2", "p3"]))
    m = Model("m")
    x = m.var("x", (P,), upper=5.0)
    one = Param.from_dense("one", (P,), np.ones(3))
    keep = subset((P,), {"P": np.array(["p1", "p3"])})
    narrowed = m.constraint("cap", one[P] * x[P] <= 1.0, where=keep)
    stated = m.constraint("all", one[P] * x[P] <= 1.0, over=product((P,)))
    assert narrowed.where is keep
    assert narrowed.over is None
    assert stated.over is not None
    assert stated.where is None


def test_a_constraint_narrows_to_the_same_shape_when_it_is_re_run():
    # the recorder re-runs this function; a re-run that answered differently
    # would attribute drops the model does not carry
    from nimopt.constraint import narrow

    P = Set("P", np.array(["p1", "p2", "p3"]))
    m = Model("m")
    x = m.var("x", (P,), upper=5.0)
    one = Param.from_dense("one", (P,), np.ones(3))
    rhs = Param.from_long("rhs", (P,), {"P": np.array(["p1", "p2"])}, np.ones(2))
    c = m.constraint("cap", one[P] * x[P] <= rhs[P])
    rows, values, nnz = narrow(c)
    assert rows.size == c.n_rows
    assert nnz == c.nnz
    assert list(values) == [1.0, 1.0]
