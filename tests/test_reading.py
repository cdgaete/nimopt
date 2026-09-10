"""A symbol's bracket lists the dimensions it carries."""

import numpy as np
import pytest

from nimopt import Definition, Model, Param, Set, Sum
from nimopt.spelling import read, spell


def model():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    x = m.var("x", (P, W))
    d = Param.from_dense("d", (P, W), np.arange(1, 7, dtype=float).reshape(2, 3))
    cap = Param.from_dense("cap", (P,), np.array([100.0, 200.0]))
    return m, P, W, x, d, cap


def test_a_parameter_states_a_right_hand_side_only_when_it_is_read():
    _, P, W, x, d, cap = model()
    with pytest.raises(TypeError, match=r"read it at its sets as cap\[P\]"):
        Sum(W, d[P, W] * x[P, W]) <= cap


def test_every_sense_refuses_a_right_hand_side_that_is_not_read():
    _, P, W, x, d, cap = model()
    for compare in (
        lambda e: e <= cap,
        lambda e: e >= cap,
        lambda e: e == cap,
    ):
        with pytest.raises(TypeError, match=r"read it at its sets as cap\[P\]"):
            compare(Sum(W, d[P, W] * x[P, W]))


def test_the_reading_states_the_rows_the_bare_name_did():
    m, P, W, x, d, cap = model()
    m.constraint("supply", Sum(W, d[P, W] * x[P, W]) <= cap[P])
    assert m.constraints["supply"].n_rows == 2
    assert m.constraints["supply"].nnz == 6


def test_a_reading_in_another_order_than_the_declaration_is_refused():
    _, P, W, x, d, _ = model()
    refusal = r"declared over \('P', 'W'\); got \('W', 'P'\)"
    with pytest.raises(ValueError, match=refusal):
        Sum(W, d[P, W] * x[P, W]) == d[W, P]


def test_a_relation_built_by_hand_reads_its_right_hand_side_too():
    from nimopt.term import Relation

    _, P, W, x, d, cap = model()
    with pytest.raises(TypeError, match=r"read it at its sets as cap\[P\]"):
        Relation(Sum(W, d[P, W] * x[P, W]), "<=", cap)


def test_a_coefficient_on_the_left_states_the_row_the_flipped_spelling_does():
    m, P, W, x, d, cap = model()
    m.constraint("flipped", cap[P] >= Sum(W, d[P, W] * x[P, W]))
    m.constraint("plain", Sum(W, d[P, W] * x[P, W]) <= cap[P])
    flipped, plain = m.constraints["flipped"], m.constraints["plain"]
    assert flipped.sense == plain.sense == "<="
    assert flipped.n_rows == plain.n_rows
    assert spell(flipped.relation) == spell(plain.relation)


def test_a_coefficient_on_the_left_of_an_equality_states_the_same_row():
    m, P, W, x, d, cap = model()
    m.constraint("flipped", cap[P] == Sum(W, d[P, W] * x[P, W]))
    assert m.constraints["flipped"].sense == "=="
    assert m.constraints["flipped"].n_rows == 2


def test_a_coefficient_compared_with_a_coefficient_still_states_no_row():
    _, P, _, _, _, cap = model()
    with pytest.raises(TypeError, match="states no row"):
        cap[P] <= cap[P]


def scalars():
    """A definition carrying a variable and a parameter over no dimension."""
    d = Definition("scalar", sense="min")
    S = d.set("S")
    theta = d.var("theta", (), lower=-np.inf)
    k = d.param("k", ())
    x = d.var("x", (S,))
    a = d.param("a", (S,))
    return d, S, theta, k, x, a


def test_a_variable_over_no_dimension_enters_an_expression_bare():
    _, S, theta, _, x, _ = scalars()
    assert spell(theta + x[S]) == "theta + x[S]"
    assert spell(x[S] - theta) == "x[S] - theta"
    assert spell(2.0 * theta) == "2 * theta"
    assert spell(-theta) == "-theta"


def test_a_variable_over_no_dimension_is_the_same_term_bracketed_or_bare():
    _, S, theta, _, x, _ = scalars()
    assert spell(theta[()]) == "theta"
    assert spell(theta + x[S]) == spell(theta[()] + x[S])


def test_a_parameter_over_no_dimension_multiplies_a_variable_bare():
    _, S, _, k, x, _ = scalars()
    assert spell(k * x[S]) == "k * x[S]"
    assert spell(x[S] * k) == "k * x[S]"


def test_a_parameter_over_no_dimension_states_a_right_hand_side_bare():
    d, _, theta, k, _, _ = scalars()
    d.constraint("lid", theta <= k)
    built = d.build({"S": np.array(["s1"]), "k": np.array(4.0), "a": np.array([1.0])})
    assert built.constraints["lid"].n_rows == 1
    lower = np.empty(1)
    upper = np.empty(1)
    built.constraints["lid"].write_bounds(lower, upper)
    assert upper[0] == 4.0


def test_a_variable_over_dimensions_used_bare_names_its_reading():
    _, S, _, _, x, _ = scalars()
    with pytest.raises(TypeError, match=r"read it at its sets as x\[S\]"):
        x + x[S]


def test_a_parameter_over_dimensions_used_bare_names_its_reading():
    _, S, _, _, x, a = scalars()
    with pytest.raises(TypeError, match=r"read it at its sets as a\[S\]"):
        a * x[S]


def test_a_symbol_stays_hashable():
    _, S, theta, k, x, a = scalars()
    assert len({theta, k, x, a, S}) == 5


def test_a_scalar_round_trips_through_its_spelling():
    d, S, theta, k, x, a = scalars()
    symbols = {"S": S, "theta": theta, "k": k, "x": x, "a": a}
    for text in (
        "theta + Sum(S, a[S] * x[S]) >= 0",
        "k * theta <= 3",
        "Sum(S, x[S]) + theta == k",
    ):
        assert spell(read(text, symbols)) == text


def test_an_objective_takes_a_variable_over_no_dimension_bare():
    d, _, theta, _, _, _ = scalars()
    d.set_objective(theta)
    assert spell(d.objective) == "theta"


def test_an_objective_names_the_reading_a_variable_over_dimensions_wants():
    d, _, _, _, x, _ = scalars()
    with pytest.raises(TypeError, match=r"read it at its sets as x\[S\]"):
        d.set_objective(x)


def test_a_parameter_over_no_dimension_compares_as_the_coefficient_it_is():
    _, S, theta, k, x, a = scalars()
    assert type(k == theta).__name__ == "Relation"
    assert type(theta == k).__name__ == "Relation"
    with pytest.raises(TypeError, match="states no row"):
        k <= 3
    with pytest.raises(TypeError, match="states no row"):
        k == 3


def test_a_parameter_over_dimensions_compared_bare_names_its_reading():
    _, S, _, _, _, a = scalars()
    for compare in (
        lambda p: p == 3,
        lambda p: p <= 3,
        lambda p: p >= 3,
        lambda p: p != 3,
    ):
        with pytest.raises(TypeError, match=r"read it at its sets as a\[S\]"):
            compare(a)


def test_an_equation_takes_a_comparison_and_refuses_anything_else():
    d, S, _, _, x, a = scalars()
    with pytest.raises(TypeError, match="takes a comparison of an expression"):
        d.constraint("c", True)


def test_a_relation_folds_an_expression_on_the_right_however_it_is_built():
    from nimopt.term import Relation

    _, S, theta, _, x, _ = scalars()
    built = Relation(x[S], "<=", x[S] + theta)
    assert built.rhs == 0.0
    assert spell(built) == spell(x[S] <= x[S] + theta)
    assert "Relation" in repr(type(built))
