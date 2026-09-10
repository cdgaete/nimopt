import numpy as np
import pytest

from nimopt.constraint import Constraint
from nimopt.param import Param
from nimopt.sets import Set
from nimopt.term import Sum
from nimopt.variable import Variable


def test_a_coefficient_absent_inside_a_sum_drops_the_term_not_the_row():
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    x = Variable("x", (P, W), start=0, total_columns=6)
    # no coefficient at (p1, w2): that arc does not exist
    cost = Param.from_long(
        "c",
        (P, W),
        {
            "P": np.array(["p1", "p1", "p2", "p2", "p2"]),
            "W": np.array(["w1", "w3", "w1", "w2", "w3"]),
        },
        np.ones(5),
    )
    con = Constraint("supply", Sum(W, cost[P, W] * x[P, W]) <= 1.0)
    # both rows are present; p1 has two terms, not three
    assert con.n_rows == 2
    assert con.nnz == 5


def test_a_reference_reaching_past_the_start_of_a_set_drops_the_row():
    T = Set("T", np.array([2030, 2031, 2032]))
    x = Variable("x", (T,), start=0, total_columns=3)
    one = Param.from_dense("a", (T,), np.ones(3))
    # a balance relating each period to the one before it
    con = Constraint("balance", one[T] * x[T] - one[T] * x[T - 1] == 0.0)
    # 2030 has no predecessor, so its row is not stated at all
    assert con.n_rows == 2
    assert list(con.rows.labels()["T"]) == [2031, 2032]


def test_the_dropped_row_is_absent_rather_than_trivially_satisfied():
    T = Set("T", np.array([2030, 2031]))
    x = Variable("x", (T,), start=0, total_columns=2)
    one = Param.from_dense("a", (T,), np.ones(2))
    con = Constraint("balance", one[T] * x[T] - one[T] * x[T - 1] == 0.0)
    assert con.n_rows == 1


def test_a_lag_narrows_the_row_domain():
    T = Set("T", np.array([2030, 2031, 2032]))
    x = Variable("x", (T,), start=0, total_columns=3)
    one = Param.from_dense("a", (T,), np.ones(3))
    _, rows = (one[T] * x[T - 1]).materialise()
    # 2030 reads a member before the set, so its row is not stated
    assert list(rows.labels()["T"]) == [2031, 2032]


def test_a_cyclic_lag_keeps_every_row():
    T = Set("T", np.array([2030, 2031, 2032]))
    x = Variable("x", (T,), start=0, total_columns=3)
    one = Param.from_dense("a", (T,), np.ones(3))
    _, rows = (one[T] * x[T.cyclic - 1]).materialise()
    # 2030 wraps onto 2032, so every row stands
    assert list(rows.labels()["T"]) == [2030, 2031, 2032]


def test_a_reference_naming_a_set_the_variable_lacks_is_refused():
    T = Set("T", np.array([2030, 2031]))
    other = Set("Z", np.array([1, 2]))
    x = Variable("x", (T,), start=0, total_columns=2)
    with pytest.raises(ValueError, match="declared over"):
        x[other - 1]


def test_a_row_the_right_hand_side_does_not_carry_is_not_stated():
    P = Set("P", np.array(["p1", "p2"]))
    x = Variable("x", (P,), start=0, total_columns=2)
    one = Param.from_dense("a", (P,), np.ones(2))
    partial = Param.from_long("cap", (P,), {"P": np.array(["p1"])}, np.array([4.0]))
    con = Constraint("cap", one[P] * x[P] <= partial[P])
    assert con.n_rows == 1
    assert list(con.rows.labels()["P"]) == ["p1"]
