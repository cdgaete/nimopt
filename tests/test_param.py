import nimblend as nb
import numpy as np
import pytest

from nimopt.param import Param
from nimopt.sets import Set


def sets_2x3():
    return Set("P", np.array(["p1", "p2"])), Set("W", np.array(["w1", "w2", "w3"]))


def test_from_dense_carries_every_cell():
    P, W = sets_2x3()
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    c = Param.from_dense("c", (P, W), values)
    assert c.dims == ("P", "W")
    assert c.nnz == 6
    assert np.array_equal(c.array.to_dense(), values)


def test_from_long_carries_only_the_rows_it_is_given():
    P, W = sets_2x3()
    c = Param.from_long(
        "c",
        (P, W),
        {"P": np.array(["p1", "p2"]), "W": np.array(["w3", "w1"])},
        np.array([5.0, 7.0]),
    )
    assert c.nnz == 2
    assert c.array.to_dense()[0, 2] == 5.0
    assert c.array.to_dense()[1, 0] == 7.0


def test_a_param_declares_empty_absence():
    P, W = sets_2x3()
    assert Param.from_dense("c", (P, W), np.zeros((2, 3))).array.absence == "empty"


def test_a_stored_zero_stays_present():
    P, W = sets_2x3()
    c = Param.from_long(
        "c", (P, W), {"P": np.array(["p1"]), "W": np.array(["w1"])}, np.array([0.0])
    )
    assert c.nnz == 1


def test_from_long_refuses_a_value_column_of_the_wrong_length():
    P, W = sets_2x3()
    with pytest.raises(ValueError, match="length"):
        Param.from_long(
            "c",
            (P, W),
            {"P": np.array(["p1", "p2"]), "W": np.array(["w1", "w2"])},
            np.array([1.0]),
        )


def test_from_dense_refuses_values_of_the_wrong_shape():
    P, W = sets_2x3()
    with pytest.raises(ValueError, match="shape"):
        Param.from_dense("c", (P, W), np.zeros((3, 2)))


def test_a_param_wrapping_an_array_of_the_wrong_dimensions_raises():
    from nimblend import from_dense

    P, W = sets_2x3()
    arr = from_dense(np.zeros((2, 3)), {"A": np.arange(2), "B": np.arange(3)})
    with pytest.raises(ValueError, match="its array is over"):
        Param("c", (P, W), arr)


def test_a_solved_models_values_become_another_models_coefficients():
    """The claim `Param` makes: a solution's arrays reach a constraint directly.

    A dual and a full-product primal are DenseArrays and a subset primal is a
    SparseArray, so all three shapes are fed back rather than the one that
    happens to match the column block's implementation.
    """
    from nimopt import Model, Set, Sum, subset

    T = Set("T", np.arange(3))
    G = Set("G", np.array(["g1", "g2"]))
    m = Model("m")
    x = m.var("x", (T,), upper=5.0)
    m.var(
        "z",
        (T, G),
        subset=subset((T, G), {"T": np.array([0, 1]), "G": np.array(["g1", "g2"])}),
        upper=1.0,
    )
    one = Param.from_dense("one", (T,), np.ones(3))
    rhs = Param.from_dense("rhs", (T,), np.array([1.0, 2.0, 3.0]))
    m.constraint("c", one[T] * x[T] >= rhs[T])
    m.set_objective(Sum(T, one[T] * x[T]))
    solved = m.solve()

    # duals are 1.0 at each of the three rows; primals are the right-hand side
    for values, expected in ((solved.dual("c"), 3.0), (solved.primal("x"), 6.0)):
        priced = Param("price", (T,), values.as_empty())
        second = Model("second", sense="max")
        y = second.var("y", (T,), upper=1.0)
        second.set_objective(Sum(T, priced[T] * y[T]))
        assert second.solve().objective == pytest.approx(expected)

    # the subset primal enters a constraint by the same route. The columns
    # read in step: the subset names (0, g1) and (1, g2) -- two members
    members = solved.primal("z").as_empty()
    assert Param("held", (T, G), members).nnz == 2


def test_a_parameter_declares_without_values():
    p = Param("cost", (Set("G"),))
    assert p.declared
    # dims come from the sets; they are readable with nothing bound
    assert p.dims == ("G",)


def test_a_declared_parameter_refuses_to_materialise():
    p = Param("cost", (Set("G"),))
    with pytest.raises(
        ValueError, match="parameter 'cost' is declared and has no values"
    ):
        p.materialise()


def test_binding_gives_a_declared_parameter_its_values():
    G = Set("G")
    p = Param("cost", (G,))
    G._bind(np.array(["a", "b"]))
    p._bind(Param.from_dense("cost", (G,), np.array([1.0, 2.0])).array)
    assert not p.declared
    assert p.nnz == 2


def test_a_declared_parameter_refuses_wherever_its_values_are_read():
    # reading the array behind materialise would return None; the caller then
    # gets a TypeError about NoneType instead of one about the parameter
    G = Set("G")
    p = Param("cost", (G,))
    with pytest.raises(ValueError, match="has no values"):
        p[G].materialise()


def test_from_positions_equals_from_long_at_the_same_members():
    P, W = sets_2x3()
    # columns in no particular order: (p2, w3), (p1, w1), (p2, w1)
    by_position = Param.from_positions(
        "c", (P, W), [[1, 0, 1], [2, 0, 0]], [3.0, 1.0, 2.0]
    )
    by_label = Param.from_long(
        "c",
        (P, W),
        {"P": ["p2", "p1", "p2"], "W": ["w3", "w1", "w1"]},
        [3.0, 1.0, 2.0],
    )
    assert (
        by_position.array.coordinates().tolist()
        == by_label.array.coordinates().tolist()
    )
    assert by_position.array.values().tolist() == by_label.array.values().tolist()


def test_from_positions_raises_for_a_position_outside_a_set():
    P, W = sets_2x3()
    with pytest.raises(
        ValueError, match="parameter 'c': row 1 of the index has positions"
    ):
        Param.from_positions("c", (P, W), [[0], [3]], [1.0])


def test_from_positions_raises_for_a_member_given_twice():
    P, W = sets_2x3()
    with pytest.raises(ValueError, match="appears twice"):
        Param.from_positions("c", (P, W), [[0, 0], [1, 1]], [1.0, 2.0])


def test_from_positions_raises_for_one_value_per_column_missing():
    P, W = sets_2x3()
    with pytest.raises(ValueError, match="has 2 index columns and 3 values"):
        Param.from_positions("c", (P, W), [[0, 1], [1, 2]], [1.0, 2.0, 3.0])


def sets_abc():
    return Set("S", np.array(["a", "b", "c"]))


def bound_values(param, S):
    """Solve max sum x subject to x <= param and return x by label."""
    from nimopt import Model, Sum

    m = Model("m", sense="max")
    x = m.var("x", (S,), upper=100.0)
    m.constraint("cap", x[S] <= param[S])
    m.set_objective(Sum(S, x[S]))
    solved = m.solve()
    return dict(zip(S.labels.tolist(), solved.primal("x").to_dense().tolist()))


@pytest.mark.parametrize(
    ("labels", "values", "expected"),
    [
        (["a", "b", "c"], [1.0, 2.0, 3.0], {"a": 1.0, "b": 2.0, "c": 3.0}),
        (["c", "b", "a"], [3.0, 2.0, 1.0], {"a": 1.0, "b": 2.0, "c": 3.0}),
        # b has no coefficient: its row is absent and x is at its upper bound
        (["a", "c"], [1.0, 3.0], {"a": 1.0, "b": 100.0, "c": 3.0}),
    ],
)
def test_from_array_places_values_by_label(labels, values, expected):
    S = sets_abc()
    array = nb.from_dense(np.array(values), {"S": np.array(labels)})
    p = Param.from_array("p", (S,), array)
    assert p.nnz == len(labels)
    assert bound_values(p, S) == expected


def test_from_array_reads_dimensions_in_any_order():
    P, W = sets_2x3()
    array = nb.from_long(
        ("W", "P"),
        {
            "W": nb.StoredCoord(np.array(["w3", "w1"])),
            "P": nb.StoredCoord(np.array(["p2", "p1"])),
        },
        {"W": np.array(["w3", "w1"]), "P": np.array(["p1", "p2"])},
        np.array([5.0, 7.0]),
    )
    p = Param.from_array("c", (P, W), array)
    # (p1, w3) = 5 and (p2, w1) = 7, in the sets' order
    np.testing.assert_array_equal(
        p.materialise().to_dense(), [[0.0, 0.0, 5.0], [7.0, 0.0, 0.0]]
    )


def test_from_array_stores_only_present_entries():
    rows = Set("R", np.arange(1000))
    cols = Set("C", np.arange(1000))
    rng = np.random.default_rng(0)
    flat = rng.choice(1_000_000, 10_000, replace=False)
    array = nb.from_long(
        ("R", "C"),
        {"R": nb.StoredCoord(np.arange(1000)), "C": nb.StoredCoord(np.arange(1000))},
        {"R": flat // 1000, "C": flat % 1000},
        np.ones(10_000),
    )
    assert Param.from_array("p", (rows, cols), array).nnz == 10_000


def test_from_array_raises_for_a_foreign_label():
    S = sets_abc()
    array = nb.from_dense(np.array([1.0, 9.0]), {"S": np.array(["a", "z"])})
    with pytest.raises(ValueError, match=r"parameter 'p' has label 'z' on 'S'"):
        Param.from_array("p", (S,), array)


def test_from_array_raises_for_other_dimensions():
    S = sets_abc()
    array = nb.from_dense(np.array([1.0]), {"T": np.array(["x"])})
    with pytest.raises(
        ValueError, match=r"declared over \('S',\) and its array is over \('T',\)"
    ):
        Param.from_array("p", (S,), array)


def test_from_array_raises_for_a_list():
    with pytest.raises(TypeError, match="pass a nimblend DenseArray or SparseArray"):
        Param.from_array("p", (sets_abc(),), [1.0, 2.0, 3.0])


def test_from_array_raises_for_a_declared_set():
    array = nb.from_dense(np.array([1.0]), {"G": np.array(["g1"])})
    with pytest.raises(ValueError, match="set 'G' of parameter 'p' has no members"):
        Param.from_array("p", (Set("G"),), array)


def test_from_array_of_an_empty_array():
    S = sets_abc()
    array = nb.from_long(
        ("S",),
        {"S": nb.StoredCoord(np.array(["a"]))},
        {"S": np.array([], dtype="<U1")},
        np.array([]),
    )
    assert Param.from_array("p", (S,), array).nnz == 0


def test_from_array_converts_datetime_labels():
    T = Set("T", np.array(["2030-01-01T00", "2030-01-01T01"], dtype="datetime64[h]"))
    array = nb.from_dense(
        np.array([1.0, 2.0]), {"T": np.array(["2030-01-01T01", "2030-01-01T00"])}
    )
    p = Param.from_array("p", (T,), array)
    np.testing.assert_array_equal(p.materialise().to_dense(), [2.0, 1.0])


def test_from_array_takes_a_subset_primal():
    from nimopt import Model, Sum, subset

    T = Set("T", np.arange(2))
    G = Set("G", np.array(["g1", "g2"]))
    m = Model("m")
    z = m.var(
        "z",
        (T, G),
        subset=subset((T, G), {"T": np.array([0, 1]), "G": np.array(["g1", "g2"])}),
        lower=1.0,
    )
    m.set_objective(Sum(T, G, z[T, G]))
    primal = m.solve().primal("z")
    assert primal.absence == "unknown"
    assert Param.from_array("held", (T, G), primal).nnz == 2


def test_a_dual_and_a_primal_become_coefficients():
    from nimopt import Model, Sum

    T = Set("T", np.arange(3))
    m = Model("m")
    x = m.var("x", (T,), upper=5.0)
    rhs = Param.from_dense("rhs", (T,), np.array([1.0, 2.0, 3.0]))
    m.constraint("c", x[T] >= rhs[T])
    m.set_objective(Sum(T, x[T]))
    solved = m.solve()
    # duals are 1 at each row; primals equal the right-hand side
    for values, expected in ((solved.dual("c"), 3.0), (solved.primal("x"), 6.0)):
        priced = Param.from_array("price", (T,), values)
        second = Model("second", sense="max")
        y = second.var("y", (T,), upper=1.0)
        second.set_objective(Sum(T, priced[T] * y[T]))
        assert second.solve().objective == pytest.approx(expected)
