import numpy as np
import pytest

from nimopt import Model, Param, Set, subset


def test_a_scalar_bound_reaches_every_column():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b", "c"]))
    m.var("x", (S,), lower=1.0, upper=5.0)
    lower, upper = m.column_bounds()
    assert np.array_equal(lower, np.array([1.0, 1.0, 1.0]))
    assert np.array_equal(upper, np.array([5.0, 5.0, 5.0]))


def test_a_parameter_bound_reaches_the_column_of_each_member():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b", "c"]))
    cap = Param.from_dense("cap", (S,), np.array([10.0, 20.0, 30.0]))
    m.var("x", (S,), upper=cap)
    lower, upper = m.column_bounds()
    assert np.array_equal(upper, np.array([10.0, 20.0, 30.0]))
    assert np.array_equal(lower, np.zeros(3))


def test_a_parameter_bound_lands_on_the_right_columns_beside_another_variable():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b"]))
    m.var("y", (S,))
    cap = Param.from_dense("cap", (S,), np.array([7.0, 8.0]))
    m.var("x", (S,), upper=cap)
    _, upper = m.column_bounds()
    # y takes columns 0 and 1, so x's bounds land on 2 and 3
    assert np.array_equal(upper, np.array([np.inf, np.inf, 7.0, 8.0]))


def test_a_parameter_bound_over_a_two_dimensional_variable():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b"]))
    J = Set("J", np.array(["p", "q", "r"]))
    cap = Param.from_dense("cap", (S, J), np.arange(6.0).reshape(2, 3))
    m.var("x", (S, J), upper=cap)
    _, upper = m.column_bounds()
    assert np.array_equal(upper, np.arange(6.0))


def test_a_bound_that_does_not_cover_the_variable_is_refused():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b", "c"]))
    cap = Param.from_long(
        "cap", (S,), {"S": np.array(["a", "c"])}, np.array([1.0, 3.0])
    )
    m.var("x", (S,), upper=cap)
    with pytest.raises(ValueError, match="'b'"):
        m.column_bounds()


def test_a_bound_over_the_wrong_dimensions_is_refused():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b"]))
    J = Set("J", np.array(["p", "q"]))
    cap = Param.from_dense("cap", (J,), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="declared over"):
        m.var("x", (S,), upper=cap)


def test_a_bound_over_more_members_than_a_subset_variable_holds_bounds_those_it_holds():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b"]))
    J = Set("J", np.array(["p", "q"]))
    members = subset((S, J), {"S": np.array(["a"]), "J": np.array(["p"])})
    cap = Param.from_dense("cap", (S, J), np.array([[1.0, 2.0], [3.0, 4.0]]))
    m.var("x", (S, J), subset=members, upper=cap)
    _, upper = m.column_bounds()
    # the variable has the member (a,p) alone, and the bound is 1.0 there
    assert np.array_equal(upper, np.array([1.0]))


def test_a_bound_over_fewer_dimensions_reaches_every_column():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b"]))
    J = Set("J", np.array(["p", "q", "r"]))
    cap = Param.from_dense("cap", (S,), np.array([7.0, 8.0]))
    m.var("x", (S, J), upper=cap)
    _, upper = m.column_bounds()
    # every column of a takes 7.0 and every column of b takes 8.0
    assert np.array_equal(upper, np.array([7.0, 7.0, 7.0, 8.0, 8.0, 8.0]))


def test_a_bound_over_the_inner_dimension_broadcasts_across_the_outer():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b"]))
    J = Set("J", np.array(["p", "q", "r"]))
    cap = Param.from_dense("cap", (J,), np.array([1.0, 2.0, 3.0]))
    m.var("x", (S, J), upper=cap)
    _, upper = m.column_bounds()
    assert np.array_equal(upper, np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))


def test_a_broadcast_bound_reaches_a_subset_variables_columns():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b"]))
    J = Set("J", np.array(["p", "q"]))
    members = subset((S, J), {"S": np.array(["a", "b"]), "J": np.array(["q", "p"])})
    cap = Param.from_dense("cap", (S,), np.array([7.0, 8.0]))
    m.var("x", (S, J), subset=members, upper=cap)
    _, upper = m.column_bounds()
    # the members are (a,q) and (b,p), in that code order
    assert np.array_equal(upper, np.array([7.0, 8.0]))


def test_a_broadcast_bound_missing_a_member_is_still_refused():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b", "c"]))
    J = Set("J", np.array(["p", "q"]))
    cap = Param.from_long(
        "cap", (S,), {"S": np.array(["a", "c"])}, np.array([1.0, 3.0])
    )
    m.var("x", (S, J), upper=cap)
    with pytest.raises(ValueError, match="'b'"):
        m.column_bounds()


def test_a_bound_over_a_dimension_the_variable_lacks_is_refused():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b"]))
    Z = Set("Z", np.array(["z1", "z2"]))
    cap = Param.from_dense("cap", (Z,), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="is not over"):
        m.var("x", (S,), upper=cap)


def test_a_bound_missing_a_member_names_it_under_the_product_rule():
    m = Model("bounds")
    S = Set("S", np.array(["p1", "p2", "p3"]))
    cap = Param.from_long(
        "cap", (S,), {"S": np.array(["p1", "p3"])}, np.array([1.0, 2.0])
    )
    m.var("x", (S,), upper=cap)
    with pytest.raises(ValueError, match=r"no value at member \('p2',\)"):
        m.column_bounds()


def test_a_bound_missing_a_member_names_it_under_the_subset_rule():
    m = Model("bounds")
    S = Set("S", np.array(["p1", "p2"]))
    J = Set("J", np.array(["w1", "w2"]))
    arcs = subset(
        (S, J),
        {"S": np.array(["p1", "p2"]), "J": np.array(["w1", "w2"])},
    )
    cap = Param.from_long(
        "cap", (S, J), {"S": np.array(["p1"]), "J": np.array(["w1"])}, np.array([1.0])
    )
    m.var("y", (S, J), subset=arcs, upper=cap)
    with pytest.raises(ValueError, match=r"no value at member \('p2', 'w2'\)"):
        m.column_bounds()


def test_a_bound_covering_every_member_is_taken_whole():
    m = Model("bounds")
    S = Set("S", np.array(["p1", "p2", "p3"]))
    cap = Param.from_dense("cap", (S,), np.array([1.0, 2.0, 3.0]))
    m.var("x", (S,), upper=cap)
    _, upper = m.column_bounds()
    assert list(upper) == [1.0, 2.0, 3.0]


def test_a_bound_stating_the_variables_dimensions_in_another_order_is_transposed():
    # a bound over the variable's own dimensions, transposed. Scattering it as
    # written sends each value to another column and leaves one never written
    T = Set("T", np.arange(3))
    G = Set("G", np.array(["a", "b"]))
    capacity = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    m = Model("m")
    m.var("x", (T, G), upper=Param.from_dense("cap", (G, T), capacity))
    _, upper = m.column_bounds()
    # the columns run (t0,a), (t0,b), (t1,a) ... so the bounds are the transpose
    assert list(upper) == list(capacity.T.ravel())


def test_a_scalar_bound_that_is_not_a_number_is_refused():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b"]))
    with pytest.raises(ValueError, match="not a number"):
        m.var("x", (S,), upper=np.nan)


def test_a_parameter_bound_carrying_no_number_is_refused_at_the_member():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b", "c"]))
    cap = Param.from_dense("cap", (S,), np.array([10.0, np.nan, 30.0]))
    m.var("x", (S,), upper=cap)
    with pytest.raises(ValueError, match=r"'cap'.*\('b',\)|\('b',\).*'cap'"):
        m.column_bounds()


def test_an_infinite_bound_is_not_refused():
    m = Model("bounds")
    S = Set("S", np.array(["a", "b"]))
    cap = Param.from_dense("cap", (S,), np.array([np.inf, 5.0]))
    m.var("x", (S,), lower=-np.inf, upper=cap)
    lower, upper = m.column_bounds()
    assert np.array_equal(lower, np.array([-np.inf, -np.inf]))
    assert np.array_equal(upper, np.array([np.inf, 5.0]))
