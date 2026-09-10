import numpy as np
import pytest

from nimopt import Model, Param, Set, Sum


def test_a_constant_in_a_row_moves_to_the_right_hand_side():
    # x + 1 <= 5 declares the row x <= 4
    P = Set("P", np.array(["a"]))
    m = Model("m", sense="max")
    x = m.var("x", (P,), upper=100.0)
    one = Param.from_dense("one", (P,), np.ones(1))
    m.constraint("cap", one[P] * x[P] + 1.0 <= 5.0)
    m.set_objective(Sum(P, one[P] * x[P]))
    assert m.solve().objective == pytest.approx(4.0)


def test_a_constant_on_both_sides_nets_out():
    P = Set("P", np.array(["a"]))
    m = Model("m", sense="max")
    x = m.var("x", (P,), upper=100.0)
    one = Param.from_dense("one", (P,), np.ones(1))
    m.constraint("cap", one[P] * x[P] + 1.0 <= one[P] * x[P] * 0.0 + 5.0)
    m.set_objective(Sum(P, one[P] * x[P]))
    assert m.solve().objective == pytest.approx(4.0)


def test_a_constant_in_the_objective_is_a_fixed_cost():
    P = Set("P", np.array(["a"]))
    m = Model("m", sense="min")
    x = m.var("x", (P,), lower=2.0, upper=100.0)
    one = Param.from_dense("one", (P,), np.ones(1))
    m.constraint("floor", one[P] * x[P] >= 2.0)
    m.set_objective(Sum(P, one[P] * x[P]) + 7.0)
    assert m.solve().objective == pytest.approx(9.0)


def test_the_matrix_carries_no_column_for_a_constant():
    P = Set("P", np.array(["a"]))
    m = Model("m", sense="min")
    x = m.var("x", (P,), lower=2.0)
    one = Param.from_dense("one", (P,), np.ones(1))
    m.constraint("floor", one[P] * x[P] >= 2.0)
    m.set_objective(Sum(P, one[P] * x[P]) + 7.0)
    assert m.assemble().n_cols == 1


def test_a_constant_alone_is_not_an_objective():
    m = Model("m", sense="min")
    with pytest.raises(TypeError, match="an objective is an expression"):
        m.set_objective(7.0)


def test_a_constant_is_reported_beside_the_objective_terms():
    P = Set("P", np.array(["a"]))
    m = Model("m", sense="min")
    x = m.var("x", (P,), lower=2.0)
    one = Param.from_dense("one", (P,), np.ones(1))
    m.constraint("floor", one[P] * x[P] >= 2.0)
    m.set_objective(Sum(P, one[P] * x[P]) + 7.0)
    found = m.explain()
    assert found.objective_constant == pytest.approx(7.0)
    assert "Sum(P, one[P] * x[P]) + 7" in repr(found)


def test_a_number_adds_to_an_expression_from_either_side():
    P = Set("P", np.array(["a"]))
    m = Model("m")
    x = m.var("x", (P,))
    one = Param.from_dense("one", (P,), np.ones(1))
    assert (1.0 + one[P] * x[P]).constant == pytest.approx(1.0)
    assert (one[P] * x[P] - 1.0).constant == pytest.approx(-1.0)
    assert (3.0 - one[P] * x[P]).constant == pytest.approx(3.0)
    assert (2.0 * (one[P] * x[P] + 1.5)).constant == pytest.approx(3.0)
