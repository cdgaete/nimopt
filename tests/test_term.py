import numpy as np
import pytest

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set
from nimopt.term import Expression, Relation, Sum


def model():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    x = m.var("x", (P, W))
    y = m.var("y", (P,))
    c = Param.from_dense("c", (P, W), np.arange(6, dtype=float).reshape(2, 3))
    a = Param.from_dense("a", (P,), np.array([10.0, 20.0]))
    return m, P, W, x, y, c, a


def test_indexing_a_variable_answers_a_one_term_expression():
    _, P, W, x, _, _, _ = model()
    e = x[P, W]
    assert isinstance(e, Expression)
    assert len(e.terms) == 1
    assert e.terms[0].coefficient is None
    assert e.terms[0].scale == 1.0
    assert e.frame == ("P", "W")


def test_indexing_a_variable_at_the_wrong_sets_raises():
    _, P, _, x, _, _, _ = model()
    with pytest.raises(ValueError, match="declared over"):
        x[P]


def test_a_parameter_times_a_variable_records_the_coefficient():
    _, P, W, x, _, c, _ = model()
    e = c[P, W] * x[P, W]
    assert e.terms[0].coefficient.param is c
    assert e.frame == ("P", "W")


def test_a_coefficient_over_fewer_dimensions_than_the_variable_is_accepted():
    _, P, W, x, _, _, a = model()
    e = a[P] * x[P, W]
    assert e.terms[0].coefficient.param is a
    assert e.frame == ("P", "W")


def test_a_coefficient_over_a_dimension_the_variable_lacks_widens_the_term():
    _, P, W, _, y, c, _ = model()
    term = (c[P, W] * y[P]).terms[0]
    # the coefficient pairs each of y's P columns to the W rows that read it
    assert term.carried_dims == ("P", "W")
    assert term.free_dims == ("P", "W")


def test_a_second_coefficient_on_one_term_multiplies_the_two():
    # a linear term has one coefficient, and a product of coefficients is
    # one: the outer factor comes first, as it was written
    _, P, W, x, _, c, a = model()
    coefficient = (a[P] * (c[P, W] * x[P, W])).terms[0].coefficient
    assert coefficient.name == "(a * c)"
    assert coefficient.dims == ("P", "W")


def test_summing_records_the_dimension_and_narrows_the_frame():
    _, P, W, x, _, c, _ = model()
    e = Sum(W, c[P, W] * x[P, W])
    assert e.terms[0].summed == ("W",)
    assert e.terms[0].free_dims == ("P",)
    assert e.frame == ("P",)


def test_summing_several_dimensions_in_one_call():
    _, P, W, x, _, c, _ = model()
    e = Sum(P, W, c[P, W] * x[P, W])
    assert e.terms[0].summed == ("P", "W")
    assert e.frame == ()


def test_summing_a_dimension_no_term_carries_raises():
    _, P, W, _, y, _, a = model()
    with pytest.raises(ValueError, match="is not over"):
        Sum(W, a[P] * y[P])


def test_adding_expressions_concatenates_their_terms():
    _, P, W, x, y, c, a = model()
    e = Sum(W, c[P, W] * x[P, W]) + a[P] * y[P]
    assert len(e.terms) == 2
    assert e.frame == ("P",)


def test_the_frame_is_the_union_ordered_by_the_term_that_introduces_each():
    _, P, W, x, y, _, _ = model()
    e = y[P] + x[P, W]
    assert e.frame == ("P", "W")


def test_subtracting_negates_the_scale_of_the_right_hand_terms():
    _, P, _, _, y, _, a = model()
    e = a[P] * y[P] - y[P]
    assert [t.scale for t in e.terms] == [1.0, -1.0]


def test_negating_negates_every_scale():
    _, P, W, x, y, _, _ = model()
    e = -(y[P] + x[P, W])
    assert [t.scale for t in e.terms] == [-1.0, -1.0]


def test_multiplying_by_a_number_scales_every_term():
    _, P, W, x, y, _, _ = model()
    e = 2.0 * (y[P] + x[P, W])
    assert [t.scale for t in e.terms] == [2.0, 2.0]
    assert [t.scale for t in (e * 0.5).terms] == [1.0, 1.0]


def test_multiplying_two_expressions_raises():
    _, P, _, _, y, _, _ = model()
    with pytest.raises(TypeError, match="linear"):
        y[P] * y[P]


def test_an_expression_holds_no_array():
    _, P, W, x, _, c, _ = model()
    e = Sum(W, c[P, W] * x[P, W])
    held = [type(v).__name__ for v in vars(e.terms[0]).values()]
    assert "SparseArray" not in held


def test_a_comparison_answers_a_relation_carrying_its_sense():
    _, P, W, x, _, c, _ = model()
    r = Sum(W, c[P, W] * x[P, W]) <= 5.0
    assert isinstance(r, Relation)
    assert r.sense == "<="
    assert r.rhs == 5.0


def test_each_sense_is_answered():
    _, P, _, _, y, _, a = model()
    assert (a[P] * y[P] >= 0.0).sense == ">="
    assert (a[P] * y[P] == 0.0).sense == "=="


def test_a_parameter_read_on_the_right_is_kept_as_the_right_hand_side():
    _, P, W, x, _, c, a = model()
    r = Sum(W, c[P, W] * x[P, W]) <= a[P]
    assert r.rhs.parameters() == (a,)
    assert r.rhs.dims == ("P",)
    assert len(r.expression.terms) == 1


def test_a_variable_on_the_right_moves_left_against_zero():
    _, P, W, x, y, c, _ = model()
    r = Sum(W, c[P, W] * x[P, W]) == y[P]
    assert r.rhs == 0.0
    assert [t.scale for t in r.expression.terms] == [1.0, -1.0]


def test_an_expression_is_not_a_dictionary_key():
    _, P, _, _, y, _, _ = model()
    assert Expression.__hash__ is None
    with pytest.raises(TypeError):
        {y[P]: 1}


def test_a_chained_comparison_raises_rather_than_keeping_half_of_itself():
    _, P, _, _, y, _, _ = model()
    with pytest.raises(TypeError, match="each bound in its own constraint"):
        0.0 <= y[P] <= 10.0


def test_a_coefficient_may_carry_a_dimension_the_variable_lacks():
    G = Set("G", np.array(["g0", "g1"]))
    T = Set("T", np.arange(3))
    m = Model("m")
    cap = m.var("cap", (G,))
    avail = Param.from_dense("avail", (G, T), np.full((2, 3), 0.5))
    term = (avail[G, T] * cap[G]).terms[0]
    assert term.carried_dims == ("G", "T")
    assert term.free_dims == ("G", "T")


def test_a_coefficient_nesting_in_the_variable_keeps_the_variable_order():
    G = Set("G", np.array(["g0", "g1"]))
    T = Set("T", np.arange(3))
    m = Model("m")
    gen = m.var("gen", (G, T))
    price = Param.from_dense("price", (T,), np.ones(3))
    term = (price[T] * gen[G, T]).terms[0]
    assert term.carried_dims == ("G", "T")
    assert term.free_dims == ("G", "T")


def test_a_dimension_the_coefficient_introduces_can_be_summed_away():
    B = Set("B", np.array(["b0", "b1"]))
    L = Set("L", np.array(["l0"]))
    T = Set("T", np.arange(2))
    m = Model("m")
    p = m.var("p", (L, T), lower=-np.inf)
    inc = Param.from_dense("inc", (B, L, T), np.zeros((2, 1, 2)))
    expression = Sum(L, inc[B, L, T] * p[L, T])
    assert expression.frame == ("B", "T")
