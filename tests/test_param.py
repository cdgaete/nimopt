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
    with pytest.raises(ValueError, match="dimensions"):
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
    m.eq("c", one[T] * x[T] >= rhs[T])
    m.set_objective(Sum(T, one[T] * x[T]))
    solved = m.solve()

    # duals are 1.0 at each of the three rows; primals are the right-hand side
    for values, expected in ((solved.dual("c"), 3.0), (solved.primal("x"), 6.0)):
        priced = Param("price", (T,), values.as_empty())
        second = Model("second", sense="max")
        y = second.var("y", (T,), upper=1.0)
        second.set_objective(Sum(T, priced[T] * y[T]))
        assert second.solve().objective == pytest.approx(expected)

    # the subset primal reaches a constraint by the same route. The columns
    # read in step, so the subset names (0, g1) and (1, g2) -- two members
    members = solved.primal("z").as_empty()
    assert Param("held", (T, G), members).nnz == 2


def test_a_parameter_declares_without_values():
    p = Param("cost", (Set("G"),))
    assert p.declared
    # dims come from the sets, so they answer with nothing bound
    assert p.dims == ("G",)


def test_a_declared_parameter_refuses_to_materialise():
    p = Param("cost", (Set("G"),))
    with pytest.raises(
        ValueError, match="parameter 'cost' is declared and carries no values"
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
    # a reader reaching the array behind materialise hands out None, which the
    # caller meets as a TypeError naming NoneType rather than the parameter
    G = Set("G")
    p = Param("cost", (G,))
    with pytest.raises(ValueError, match="carries no values"):
        p[G].materialise()
