import numpy as np
import pytest

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set
from nimopt.term import Sum

highspy = pytest.importorskip("highspy")


def transport():
    m = Model("transport")
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2"]))
    x = m.var("x", (P, W))
    one = Param.from_dense("a", (P, W), np.ones((2, 2)))
    cost = Param.from_dense("c", (P, W), np.array([[1.0, 2.0], [3.0, 1.0]]))
    supply = Param.from_dense("supply", (P,), np.array([10.0, 10.0]))
    demand = Param.from_dense("demand", (W,), np.array([6.0, 8.0]))
    m.constraint("supply", Sum(W, one[P, W] * x[P, W]) <= supply[P])
    m.constraint("demand", Sum(P, one[P, W] * x[P, W]) >= demand[W])
    m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))
    return m


def test_a_transport_model_solves_to_its_known_optimum():
    got = transport().solve()
    # ship w1 from p1 at 1 and w2 from p2 at 1: 6 + 8 = 14
    assert got.status == "optimal"
    assert got.objective == pytest.approx(14.0)


def test_the_primal_comes_back_as_a_labeled_array():
    m = transport()
    got = m.solve()
    x = got.primal("x")
    assert x.dims == ("P", "W")
    assert x.to_dense()[0, 0] == pytest.approx(6.0)
    assert x.to_dense()[1, 1] == pytest.approx(8.0)


def test_a_solution_array_declares_unknown_absence():
    got = transport().solve()
    assert got.primal("x").absence == "unknown"
    assert got.dual("demand").absence == "unknown"


def test_the_dual_is_labeled_by_the_constraints_free_sets():
    got = transport().solve()
    d = got.dual("demand")
    assert d.dims == ("W",)
    assert d.nnz == 2


def test_a_dual_is_usable_as_a_coefficient_without_conversion():
    m = transport()
    got = m.solve()
    W = Set("W", np.array(["w1", "w2"]))
    cut = Param("cut", (W,), got.dual("demand").as_empty())
    assert cut.nnz == 2


def test_a_maximised_model_flips_the_optimum():
    m = Model(sense="max")
    P = Set("P", np.array(["p1"]))
    x = m.var("x", (P,), lower=0.0, upper=4.0)
    one = Param.from_dense("a", (P,), np.ones(1))
    m.constraint("cap", one[P] * x[P] <= 3.0)
    m.set_objective(Sum(P, one[P] * x[P]))
    assert m.solve().objective == pytest.approx(3.0)


def test_an_unknown_solver_raises():
    with pytest.raises(ValueError, match="highs"):
        transport().solve(solver="glpk")


def test_a_model_solved_through_a_session_names_the_solver():
    assert transport().solve().solver == "highs"


def test_a_dual_is_labelled_by_the_sets_its_constraint_is_over():
    m = Model()
    P = Set("P", np.array(["seattle", "sandiego"]))
    x = m.var("x", (P,), lower=0.0, upper=10.0)
    one = Param.from_dense("one", (P,), np.ones(2))
    cap = Param.from_dense("cap", (P,), np.array([3.0, 4.0]))
    m.constraint("supply", one[P] * x[P] <= cap[P])
    cost = Param.from_dense("c", (P,), np.array([-1.0, -1.0]))
    m.set_objective(Sum(P, cost[P] * x[P]))
    solution = m.solve()
    dual = solution.dual("supply")
    assert dual.dims == ("P",)
    # the dual has the set's own coordinate, not a positional index
    assert dual.coords["P"] == P.coord


def infeasible():
    """A model whose only constraint asks for more than its bounds allow."""
    m = Model("infeasible")
    P = Set("P", np.array(["p1", "p2"]))
    x = m.var("x", (P,), lower=0.0, upper=1.0)
    one = Param.from_dense("one", (P,), np.ones(2))
    floor = Param.from_dense("floor", (P,), np.array([5.0, 5.0]))
    m.constraint("floor", one[P] * x[P] >= floor[P])
    m.set_objective(Sum(P, one[P] * x[P]))
    return m


def test_an_infeasible_model_reports_its_status():
    assert infeasible().solve().status == "infeasible"


def test_an_infeasible_model_refuses_to_hand_back_a_primal():
    got = infeasible().solve()
    with pytest.raises(ValueError, match="infeasible"):
        got.primal("x")


def test_an_infeasible_model_refuses_to_hand_back_a_dual():
    got = infeasible().solve()
    with pytest.raises(ValueError, match="infeasible"):
        got.dual("floor")


def test_an_infeasible_model_refuses_to_hand_back_an_objective():
    got = infeasible().solve()
    with pytest.raises(ValueError, match="infeasible"):
        got.objective


def test_the_repr_of_an_unsolved_model_states_its_status():
    assert "infeasible" in repr(infeasible().solve())


def test_a_limit_the_solver_stops_on_is_named_rather_than_called_other():
    m = transport()
    got = m.solve(options={"time_limit": 0.0})
    # a time limit is an outcome the model reports, not an unnamed 'other'
    assert got.status in ("optimal", "time_limit")
    assert got.status != "other"


def test_every_status_highs_can_report_is_either_named_or_refused():
    from nimopt.solvers import highs

    reported = {n for n in dir(highspy.HighsModelStatus) if n.startswith("k")}
    unaccounted = reported - set(highs.OUTCOME) - set(highs.FAILED)
    assert unaccounted == set(), unaccounted
