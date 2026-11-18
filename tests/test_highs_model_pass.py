import numpy as np
import pytest

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set
from nimopt.solvers import highs
from nimopt.term import Sum


def model(integer=False):
    """Return a model of two columns and three rows, with a known optimum."""
    S = Set("S", np.array(["a", "b"]))
    R = Set("R", np.array(["r0", "r1", "r2"]))
    weight = Param.from_dense(
        "weight", (R, S), np.array([[1.0, 2.0], [3.0, 1.0], [1.0, 1.0]])
    )
    need = Param.from_dense("need", (R,), np.array([4.0, 6.0, 3.0]))
    m = Model("cover")
    x = m.var("x", (S,), lower=0.0, upper=10.0, integer=integer)
    m.constraint("cover", Sum(S, weight[R, S] * x[S]) >= need[R])
    m.set_objective(Sum(S, x[S]))
    return m


def test_the_adapter_passes_the_arrays_and_builds_no_lp_object(monkeypatch):
    import highspy

    def built(*args, **kwargs):
        raise AssertionError("a HighsLp object was built")

    monkeypatch.setattr(highspy, "HighsLp", built)
    solution = model().solve()
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(3.0)


def test_the_integrality_flags_are_the_variable_types_highs_reads():
    import highspy

    # `Assembled.integrality` is 0 for a continuous column and 1 for an integer
    assert int(highspy.HighsVarType.kContinuous) == 0
    assert int(highspy.HighsVarType.kInteger) == 1


def test_an_integer_model_solves_through_the_arrays():
    solution = model(integer=True).solve()
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(3.0)
    assert not solution.has_duals


VECTORS = (
    "col_cost",
    "col_lower",
    "col_upper",
    "integrality",
    "row_lower",
    "row_upper",
    "indptr",
    "indices",
    "values",
)


@pytest.mark.parametrize("name", VECTORS)
def test_a_vector_of_another_length_than_the_matrix_is_refused(name):
    # HiGHS reads each vector by the declared count and checks no length
    assembled = model().assemble()
    setattr(assembled, name, getattr(assembled, name)[:-1])
    with pytest.raises(RuntimeError, match=f"{name} has"):
        highs.solve(assembled, "min")


def test_a_refused_vector_states_both_lengths():
    assembled = model().assemble()
    assembled.col_cost = assembled.col_cost[:-1]
    with pytest.raises(RuntimeError, match="col_cost has length 1, not 2"):
        highs.solve(assembled, "min")


@pytest.mark.parametrize("flag", [2, -1])
def test_an_integrality_flag_other_than_0_and_1_is_refused(flag):
    # HiGHS reads 2 as a semi-continuous column
    assembled = model().assemble()
    assembled.integrality = np.array([0, flag], dtype=np.int32)
    with pytest.raises(RuntimeError, match="integrality has a value other than 0"):
        highs.solve(assembled, "min")
