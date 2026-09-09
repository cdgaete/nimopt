import numpy as np
import pytest

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import LaggedSet, Set
from nimopt.term import Sum


def model():
    m = Model()
    T = Set("T", np.array([2030, 2031, 2032]))
    x = m.var("x", (T,))
    one = Param.from_dense("a", (T,), np.ones(3))
    return m, T, x, one


def test_subtracting_from_a_set_answers_a_lagged_reference():
    _, T, _, _ = model()
    lagged = T - 1
    assert isinstance(lagged, LaggedSet)
    assert lagged.name == "T"
    # the entry written at t-1 lands on row t, so [T-1] is a shift of +1
    assert lagged.shift == 1
    assert lagged.mode == "drop"


def test_adding_to_a_set_reads_the_following_member():
    _, T, _, _ = model()
    assert (T + 1).shift == -1
    assert (T + 1).mode == "drop"


def test_a_cyclic_set_lags_by_wrapping():
    _, T, _, _ = model()
    assert (T.cyclic - 1).shift == 1
    assert (T.cyclic - 1).mode == "wrap"
    assert (T.cyclic + 1).shift == -1


def test_a_lagged_reference_records_the_shift_on_the_term():
    _, T, x, _ = model()
    e = x[T - 1]
    assert e.terms[0].shifts == {"T": (1, "drop")}
    assert e.frame == ("T",)


def test_an_unlagged_reference_records_no_shift():
    _, T, x, _ = model()
    assert x[T].terms[0].shifts == {}


def test_a_lagged_term_puts_the_previous_columns_value_on_each_row():
    _, T, x, _ = model()
    block, rows = x[T - 1].materialise()
    # row t carries the column of t-1; the first period's row is not stated
    assert list(rows.labels()["T"]) == [2031, 2032]
    assert block.to_dense().tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


def test_a_lag_past_the_start_of_a_set_drops_the_row():
    _, T, x, one = model()
    _, rows = (one[T] * x[T] - one[T] * x[T - 1]).materialise()
    assert rows.size == 2
    assert list(rows.labels()["T"]) == [2031, 2032]


def test_a_cyclic_lag_states_every_row():
    _, T, x, one = model()
    _, rows = (one[T] * x[T] - one[T] * x[T.cyclic - 1]).materialise()
    assert rows.size == 3
    assert list(rows.labels()["T"]) == [2030, 2031, 2032]


def test_a_cyclic_lag_wraps_the_first_row_onto_the_last_column():
    _, T, x, _ = model()
    block, _ = x[T.cyclic - 1].materialise()
    assert block.to_dense().tolist() == [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


def test_a_coefficient_multiplies_the_lagged_term_at_the_rows_own_coordinate():
    m = Model()
    T = Set("T", np.array([2030, 2031, 2032]))
    x = m.var("x", (T,))
    a = Param.from_dense("a", (T,), np.array([10.0, 20.0, 30.0]))
    block, _ = (a[T] * x[T - 1]).materialise()
    # row 2031 carries a's value at 2031, on the column of 2030
    assert block.to_dense().tolist() == [
        [0.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [0.0, 30.0, 0.0],
    ]


def test_summing_over_a_lagged_set_is_refused():
    _, T, x, one = model()
    with pytest.raises(ValueError, match="lag"):
        Sum(T - 1, one[T] * x[T])


def test_a_lagged_parameter_reference_is_refused():
    _, T, _, one = model()
    with pytest.raises(ValueError, match="lag"):
        one[T - 1]


def test_a_lag_carries_through_a_scale_and_a_sum():
    m = Model()
    G = Set("G", np.array(["g1", "g2"]))
    T = Set("T", np.array([2030, 2031]))
    y = m.var("y", (G, T))
    e = Sum(G, 2.0 * y[G, T - 1])
    assert e.terms[0].shifts == {"T": (1, "drop")}
    assert e.terms[0].summed == ("G",)
    assert e.terms[0].scale == 2.0
