import numpy as np
import pytest

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set, subset
from nimopt.term import Sum
from nimopt.variable import Variable


def test_variables_take_successive_column_ranges():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    x = m.var("x", (P, W))
    y = m.var("y", (P,))
    assert (x.start, x.n_columns) == (0, 6)
    assert (y.start, y.n_columns) == (6, 2)
    assert m.n_columns == 8


def test_declaring_a_variable_renumbers_the_earlier_ones():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    x = m.var("x", (P,))
    assert x.total_columns == 2
    m.var("y", (P,))
    assert x.total_columns == 4
    assert x.terms().shape[-1] == 4


def test_a_subset_variable_costs_only_its_members():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    arcs = subset((P, W), {"P": np.array(["p1", "p2"]), "W": np.array(["w1", "w3"])})
    z = m.var("z", (P, W), subset=arcs)
    assert z.n_columns == 2
    assert m.n_columns == 2


def test_a_repeated_variable_name_raises():
    m = Model()
    P = Set("P", np.array(["p1"]))
    m.var("x", (P,))
    with pytest.raises(ValueError, match="already"):
        m.var("x", (P,))


def test_column_bounds_come_from_the_variables():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    m.var("x", (P,), lower=0.0, upper=5.0)
    m.var("b", (P,), lower=0.0, upper=1.0, integer=True)
    lower, upper = m.column_bounds()
    assert list(lower) == [0.0, 0.0, 0.0, 0.0]
    assert list(upper) == [5.0, 5.0, 1.0, 1.0]
    assert list(m.integrality()) == [0, 0, 1, 1]


def test_constraints_take_successive_row_ranges():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    x = m.var("x", (P,))
    one = Param.from_dense("a", (P,), np.ones(2))
    first = m.eq("c1", one[P] * x[P] <= 1.0)
    second = m.eq("c2", one[P] * x[P] >= 0.0)
    assert (first.n_rows, second.n_rows) == (2, 2)
    assert m.n_rows == 4
    assert m.nnz == 4


def test_the_objective_is_an_expression_over_the_column_space_only():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    x = m.var("x", (P,))
    cost = Param.from_dense("c", (P,), np.array([3.0, 4.0]))
    m.set_objective(Sum(P, cost[P] * x[P]))
    assert list(m.objective_coefficients()) == [3.0, 4.0]
    assert m.sense == "min"


def test_a_model_declared_to_maximise_records_that_sense():
    m = Model(sense="max")
    P = Set("P", np.array(["p1"]))
    x = m.var("x", (P,))
    cost = Param.from_dense("c", (P,), np.array([2.0]))
    m.set_objective(Sum(P, cost[P] * x[P]))
    assert m.sense == "max"


def test_an_objective_with_free_dimensions_left_raises():
    m = Model()
    P = Set("P", np.array(["p1"]))
    x = m.var("x", (P,))
    cost = Param.from_dense("c", (P,), np.array([2.0]))
    with pytest.raises(ValueError, match="free dimensions"):
        m.set_objective(cost[P] * x[P])


def test_the_objective_covers_columns_declared_after_it_is_set():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    x = m.var("x", (P,))
    cost = Param.from_dense("c", (P,), np.array([1.0, 2.0]))
    m.set_objective(Sum(P, cost[P] * x[P]))
    m.var("y", (P,))
    assert m.n_columns == 4
    assert list(m.objective_coefficients()) == [1.0, 2.0, 0.0, 0.0]


def test_a_model_states_its_sense_where_it_is_named():
    assert Model("m").sense == "min"
    assert Model("m", sense="max").sense == "max"


def test_a_sense_that_is_neither_is_refused_at_construction():
    with pytest.raises(ValueError, match="sense is 'min' or 'max'"):
        Model("m", sense="minimize")


def test_setting_an_objective_does_not_restate_the_sense():
    T = Set("T", np.array([0, 1]))
    m = Model("m", sense="max")
    x = m.var("x", (T,), upper=1.0)
    one = Param.from_dense("one", (T,), np.ones(2))
    m.set_objective(Sum(T, one[T] * x[T]))
    assert m.sense == "max"


def test_a_model_carries_no_second_way_to_set_a_sense():
    assert not hasattr(Model("m"), "minimize")
    assert not hasattr(Model("m"), "maximize")
    # a sense read from an attribute anything can assign has two homes, so
    # the one home is read-only
    with pytest.raises(AttributeError):
        Model("m").sense = "max"


def test_an_equation_is_stated_with_eq_for_every_sense():
    T = Set("T", np.array([0, 1]))
    m = Model("m")
    x = m.var("x", (T,), upper=5.0)
    one = Param.from_dense("one", (T,), np.ones(2))
    # eq is the equation, not the equality: every sense goes through it
    assert m.eq("le", one[T] * x[T] <= 1.0).sense == "<="
    assert m.eq("ge", one[T] * x[T] >= 0.0).sense == ">="
    assert m.eq("en", one[T] * x[T] == 1.0).sense == "=="


def test_a_model_carries_no_second_way_to_state_an_equation():
    assert not hasattr(Model("m"), "add")


def test_a_model_adopts_a_variable_declared_outside_it():
    S = Set("S")
    x = Variable("x", (S,), upper=3.0)
    S._bind(np.array(["s1", "s2"]))
    m = Model("m")
    m._adopt(x)
    assert m.variables["x"] is x
    assert (x.start, x.n_columns, m.n_columns) == (0, 2, 2)
    y = Variable("y", (S,))
    m._adopt(y)
    # the adopted variable takes the next range, and the first is renumbered
    assert (y.start, m.n_columns, x.total_columns) == (2, 4, 4)


def test_a_model_refuses_to_adopt_a_name_it_already_carries():
    S = Set("S", np.array(["s1"]))
    m = Model("m")
    m._adopt(Variable("x", (S,)))
    with pytest.raises(ValueError, match="variable 'x' is already declared"):
        m._adopt(Variable("x", (S,)))
