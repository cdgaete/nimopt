import numpy as np
import pytest

from nimopt import Model, Param, Set, Sum, subset
from nimopt.bounds import Bounds, target_of


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


def plant():
    """Return x over (P, W) in [0, 100], an integer n over P, and their Bounds."""
    P = Set("P", np.array(["p0", "p1"]))
    W = Set("W", np.array(["w0", "w1", "w2"]))
    m = Model("plant")
    x = m.var("x", (P, W), lower=0.0, upper=100.0)
    n = m.var("n", (P,), integer=True)
    lower, upper = m.column_bounds()
    return m, P, W, x, n, lower, upper, Bounds(lower, upper)


def test_a_point_fix_pins_one_column():
    m, P, W, x, n, lower, upper, bounds = plant()
    bounds.fix(x, np.array([0]), 4.0, whole=False)
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x.tolist() == [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert upper_x.tolist() == [4.0, 100.0, 100.0, 100.0, 100.0, 100.0]


def test_a_fix_is_not_clamped_into_the_declared_bounds():
    m, P, W, x, n, lower, upper, bounds = plant()
    bounds.fix(x, np.array([0]), -5.0, whole=False)
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x[0] == upper_x[0] == -5.0


def test_a_later_fix_replaces_an_earlier_one():
    m, P, W, x, n, lower, upper, bounds = plant()
    bounds.fix(x, np.array([0]), 4.0, whole=False)
    bounds.fix(x, np.array([0]), 7.0, whole=False)
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x[0] == upper_x[0] == 7.0


def test_a_fix_that_is_not_finite_raises():
    m, P, W, x, n, lower, upper, bounds = plant()
    for value in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match="x"):
            bounds.fix(x, np.array([0]), value, whole=False)


def test_a_fix_on_an_integer_column_is_integral():
    m, P, W, x, n, lower, upper, bounds = plant()
    with pytest.raises(ValueError):
        bounds.fix(n, np.array([0]), 2.5, whole=False)
    bounds.fix(n, np.array([0]), 2.0, whole=False)
    lower_n, upper_n = bounds.bounds_of(n)
    assert lower_n[0] == upper_n[0] == 2.0


def test_a_parameter_fix_applies_to_a_whole_variable():
    m, P, W, x, n, lower, upper, bounds = plant()
    cap = Param.from_dense("v", (P,), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        bounds.fix(x, np.array([0, 1, 2]), cap, whole=False)


def test_a_parameter_fix_over_one_set_pins_every_member():
    m, P, W, x, n, lower, upper, bounds = plant()
    v = Param.from_dense("v", (P,), np.array([1.0, 2.0]))
    bounds.fix(x, np.arange(x.n_columns), v, whole=True)
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert upper_x.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]


def test_a_parameter_fix_with_fewer_members_pins_those():
    m, P, W, x, n, lower, upper, bounds = plant()
    v = Param.from_long("v", (P, W), {"P": ["p1"], "W": ["w0"]}, [9.0])
    bounds.fix(x, np.arange(x.n_columns), v, whole=True)
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x.tolist() == [0.0, 0.0, 0.0, 9.0, 0.0, 0.0]
    assert upper_x.tolist() == [100.0, 100.0, 100.0, 9.0, 100.0, 100.0]


def test_unfix_of_a_column_that_is_not_pinned_is_accepted():
    m, P, W, x, n, lower, upper, bounds = plant()
    bounds.fix(x, np.array([0]), 4.0, whole=False)
    bounds.unfix(x, np.array([1]))
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x[0] == upper_x[0] == 4.0


def test_unfix_of_a_variable_with_no_pin_raises():
    m, P, W, x, n, lower, upper, bounds = plant()
    with pytest.raises(ValueError, match="has no fixed column"):
        bounds.unfix(x, np.arange(x.n_columns))


def test_unfix_restores_the_bounds_exactly():
    m, P, W, x, n, lower, upper, bounds = plant()
    before = bounds.bounds_of(x)
    bounds.fix(x, np.array([0]), 4.0, whole=False)
    bounds.unfix(x, np.array([0]))
    after = bounds.bounds_of(x)
    assert after[0].tolist() == before[0].tolist()
    assert after[1].tolist() == before[1].tolist()


def test_set_bound_requires_a_side():
    m, P, W, x, n, lower, upper, bounds = plant()
    with pytest.raises(ValueError):
        bounds.set_bound(x, np.arange(x.n_columns), None, None, whole=True)


def test_a_bound_on_one_element_keeps_the_others():
    Pr = Set("Pr", np.array(["bands", "coils", "plate"]))
    m = Model("products")
    v = m.var("v", (Pr,), lower=1.0, upper=9.0)
    lower, upper = m.column_bounds()
    bounds = Bounds(lower, upper)
    bounds.set_bound(v, np.array([2]), None, 3.0, whole=False)
    lower_v, upper_v = bounds.bounds_of(v)
    assert upper_v.tolist() == [9.0, 9.0, 3.0]
    assert lower_v.tolist() == [1.0, 1.0, 1.0]


def test_a_fix_survives_set_bound():
    m, P, W, x, n, lower, upper, bounds = plant()
    bounds.fix(x, np.array([0]), 4.0, whole=False)
    bounds.set_bound(x, np.arange(x.n_columns), 1.0, 50.0, whole=True)
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x[0] == upper_x[0] == 4.0
    assert lower_x[1] == 1.0
    assert upper_x[1] == 50.0
    bounds.unfix(x, np.array([0]))
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x[0] == 1.0
    assert upper_x[0] == 50.0


def test_a_parameter_bound_has_a_value_at_every_member():
    m, P, W, x, n, lower, upper, bounds = plant()
    part = Param.from_long("cap", (P, W), {"P": ["p1"], "W": ["w0"]}, [9.0])
    with pytest.raises(ValueError):
        bounds.set_bound(x, np.arange(x.n_columns), None, part, whole=True)


def test_a_parameter_bound_over_one_set_bounds_every_member():
    m, P, W, x, n, lower, upper, bounds = plant()
    cap = Param.from_dense("cap", (P,), np.array([10.0, 20.0]))
    bounds.set_bound(x, np.arange(x.n_columns), None, cap, whole=True)
    lower_x, upper_x = bounds.bounds_of(x)
    assert upper_x.tolist() == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    assert lower_x.tolist() == [0.0] * 6


def test_a_parameter_bound_applies_to_a_whole_variable():
    m, P, W, x, n, lower, upper, bounds = plant()
    cap = Param.from_dense("cap", (P,), np.array([10.0, 20.0]))
    with pytest.raises(ValueError):
        bounds.set_bound(x, np.array([0]), None, cap, whole=False)


def test_a_bound_that_is_not_a_number_raises():
    m, P, W, x, n, lower, upper, bounds = plant()
    with pytest.raises(ValueError):
        bounds.set_bound(x, np.arange(x.n_columns), float("nan"), None, whole=True)


def test_a_variable_that_is_not_changed_has_no_entry():
    m, P, W, x, n, lower, upper, bounds = plant()
    bounds.fix(x, np.array([0]), 4.0, whole=False)
    lower_n, upper_n = bounds.bounds_of(n)
    assert lower_n.tolist() == lower[6:8].tolist()
    assert upper_n.tolist() == upper[6:8].tolist()
    lower_x, upper_x = bounds.bounds_of(x)
    assert not np.shares_memory(lower_x, lower)
    assert not np.shares_memory(upper_x, upper)
    assert lower[0] == 0.0
    assert upper[0] == 100.0


def test_a_target_is_a_variable_or_one_plain_reference():
    m, P, W, x, n, lower, upper, bounds = plant()
    held, positions, whole = target_of(m.variables, x)
    assert held is x
    assert positions.tolist() == [0, 1, 2, 3, 4, 5]
    assert whole is True
    held, positions, whole = target_of(m.variables, x[P, W])
    assert held is x
    assert positions.tolist() == [0, 1, 2, 3, 4, 5]
    assert whole is True
    held, positions, whole = target_of(m.variables, x["p0", W])
    assert held is x
    assert positions.tolist() == [0, 1, 2]
    assert whole is False
    for other in (2.0 * x[P, W], Sum(W, x[P, W]), x[P, W - 1]):
        with pytest.raises(TypeError):
            target_of(m.variables, other)


def test_a_target_of_another_model_resolves_by_name():
    m, P, W, x, n, lower, upper, bounds = plant()
    other, _, _, other_x, _, _, _, _ = plant()
    assert other_x is not x
    assert target_of(m.variables, other_x)[0] is x


def test_a_target_the_model_does_not_have_raises():
    m, P, W, x, n, lower, upper, bounds = plant()
    other, other_P, _, _, _, _, _, _ = plant()
    y = other.var("y", (other_P,))
    with pytest.raises(ValueError) as raised:
        target_of(m.variables, y)
    assert "'x'" in str(raised.value)
    assert "'n'" in str(raised.value)


def test_unfix_restores_the_bounds_after_the_given_vectors_are_written():
    # a session writes bounds_of into the vectors it gave to Bounds
    _, _, _, x, _, lower, upper, bounds = plant()
    bounds.fix(x, np.array([0]), 4.0, whole=False)
    lower[0:6], upper[0:6] = bounds.bounds_of(x)
    bounds.unfix(x, np.array([0]))
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x.tolist() == [0.0] * 6
    assert upper_x.tolist() == [100.0] * 6


def test_a_parameter_fix_is_restored_after_the_given_vectors_are_written():
    _, P, _, x, _, lower, upper, bounds = plant()
    held = Param.from_dense("v", (P,), np.array([1.0, 2.0]))
    bounds.fix(x, np.arange(6), held, whole=True)
    lower[0:6], upper[0:6] = bounds.bounds_of(x)
    bounds.unfix(x, np.arange(6))
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x.tolist() == [0.0] * 6
    assert upper_x.tolist() == [100.0] * 6


def test_a_change_that_is_not_rolled_back_is_recorded():
    m, P, W, x, n, lower, upper, bounds = plant()
    with bounds.changing(x):
        bounds.fix(x, np.array([0]), 4.0, whole=False)
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x[0] == upper_x[0] == 4.0


def test_a_rolled_back_change_removes_an_entry_that_was_absent():
    m, P, W, x, n, lower, upper, bounds = plant()
    with pytest.raises(RuntimeError, match="rejected"), bounds.changing(x):
        bounds.fix(x, np.array([0]), 4.0, whole=False)
        raise RuntimeError("rejected")
    lower_x, upper_x = bounds.bounds_of(x)
    assert (lower_x.tolist(), upper_x.tolist()) == ([0.0] * 6, [100.0] * 6)
    with pytest.raises(ValueError, match="has no fixed column"):
        bounds.unfix(x, np.arange(x.n_columns))
    # no declared layer remains, and a later change starts from the given vectors
    lower[0:6] = 7.0
    assert bounds.bounds_of(x)[0].tolist() == [7.0] * 6


def test_a_rolled_back_change_keeps_an_earlier_fix_and_bound():
    m, P, W, x, n, lower, upper, bounds = plant()
    bounds.fix(x, np.array([0]), 4.0, whole=False)
    bounds.set_bound(x, np.arange(x.n_columns), 1.0, 50.0, whole=True)
    with pytest.raises(RuntimeError, match="rejected"), bounds.changing(x):
        bounds.fix(x, np.array([1]), 9.0, whole=False)
        bounds.set_bound(x, np.arange(x.n_columns), 2.0, 60.0, whole=True)
        raise RuntimeError("rejected")
    lower_x, upper_x = bounds.bounds_of(x)
    assert (lower_x[0], upper_x[0]) == (4.0, 4.0)
    assert (lower_x[1], upper_x[1]) == (1.0, 50.0)


def test_an_unfixed_entry_comes_back_where_the_body_raises():
    m, P, W, x, n, lower, upper, bounds = plant()
    bounds.fix(x, np.array([0]), 4.0, whole=False)
    with pytest.raises(RuntimeError, match="rejected"), bounds.changing(x):
        bounds.unfix(x, np.array([0]))
        raise RuntimeError("rejected")
    lower_x, upper_x = bounds.bounds_of(x)
    assert lower_x[0] == upper_x[0] == 4.0


def test_a_fix_of_a_variable_with_no_column_is_recorded():
    E = Set("E", np.array([], dtype=str))
    m = Model("empty")
    m.var("x", (E,), lower=0.0, upper=1.0)
    held = m.variables["x"]
    lower, upper = m.column_bounds()
    bounds = Bounds(lower, upper)
    bounds.fix(held, np.arange(0), 5.0, whole=True)
    assert [vector.size for vector in bounds.bounds_of(held)] == [0, 0]
    bounds.unfix(held, np.arange(0))
    with pytest.raises(ValueError, match="has no fixed column"):
        bounds.unfix(held, np.arange(0))


def test_a_parameter_fix_at_no_member_is_recorded():
    _, P, W, x, _, _, _, bounds = plant()
    nowhere = Param.from_long("nowhere", (P, W), {"P": [], "W": []}, [])
    bounds.fix(x, np.arange(6), nowhere, whole=True)
    assert bounds.bounds_of(x)[0].tolist() == [0.0] * 6
    assert bounds.bounds_of(x)[1].tolist() == [100.0] * 6
    bounds.unfix(x, np.arange(6))
    with pytest.raises(ValueError, match="has no fixed column"):
        bounds.unfix(x, np.arange(6))
