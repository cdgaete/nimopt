import numpy as np
import pytest

from nimopt import Model, Param, Set, Solution, Sum
from nimopt.solvers.base import Result


def knapsack(n=120, k=10, seed=1):
    """A multidimensional knapsack: binary columns over k capacity rows."""
    rng = np.random.default_rng(seed)
    ITEM = Set("I", np.array([f"i{t}" for t in range(n)]))
    K = Set("K", np.array([f"k{t}" for t in range(k)]))
    w = Param.from_dense("w", (K, ITEM), rng.uniform(1, 50, (k, n)))
    v = Param.from_dense("v", (ITEM,), rng.uniform(1, 100, n))
    cap = Param.from_dense("cap", (K,), np.full(k, 0.3 * 25.5 * n))
    m = Model("mdknap", sense="max")
    x = m.var("x", (ITEM,), integer=True, upper=1.0)
    m.eq("cap", Sum(ITEM, w[K, ITEM] * x[ITEM]) <= cap[K])
    m.set_objective(Sum(ITEM, v[ITEM] * x[ITEM]))
    return m


def tiny_knapsack():
    """Four binary columns under one capacity row; the optimum is 12."""
    m = Model("tinyknap", sense="max")
    J = Set("J", np.array(["a", "b", "c", "d"]))
    v = Param.from_dense("v", (J,), np.array([7.0, 5.0, 4.0, 3.0]))
    w = Param.from_dense("w", (J,), np.array([5.0, 4.0, 3.0, 2.0]))
    x = m.var("x", (J,), integer=True, upper=1.0)
    m.eq("cap", Sum(J, w[J] * x[J]) <= 9.0)
    m.set_objective(Sum(J, v[J] * x[J]))
    return m


def dense_lp(n=60, seed=3):
    """A dense covering LP that takes several simplex iterations."""
    rng = np.random.default_rng(seed)
    ITEM = Set("I", np.array([f"i{t}" for t in range(n)]))
    K = Set("K", np.array([f"k{t}" for t in range(n)]))
    a = Param.from_dense("a", (K, ITEM), rng.uniform(1, 9, (n, n)))
    c = Param.from_dense("c", (ITEM,), rng.uniform(1, 9, n))
    b = Param.from_dense("b", (K,), rng.uniform(50, 90, n))
    m = Model("denselp", sense="min")
    x = m.var("x", (ITEM,), lower=0.0, upper=100.0)
    m.eq("need", Sum(ITEM, a[K, ITEM] * x[ITEM]) >= b[K])
    m.set_objective(Sum(ITEM, c[ITEM] * x[ITEM]))
    return m


def transport(constant=0.0):
    """The two-plant transport model; its optimum is 135 plus the constant."""
    P = Set("P", np.array(["lisbon", "porto"]))
    W = Set("W", np.array(["berlin", "paris", "rome"]))
    cost = Param.from_dense(
        "cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]])
    )
    supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
    demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))
    m = Model("transport")
    x = m.var("x", (P, W))
    m.eq("supply", Sum(W, x[P, W]) <= supply[P])
    m.eq("demand", Sum(P, x[P, W]) >= demand[W])
    m.set_objective(Sum(P, W, cost[P, W] * x[P, W]) + constant)
    return m


def solution_over(model, status, feasible, objective, bound):
    """A Solution over `model` holding the fields a repr reads."""
    assembled = model.assemble()
    rows_of = {name: assembled.row_of(name) for name in model.constraints}
    return Solution(
        model,
        status,
        feasible,
        objective,
        bound,
        np.zeros(assembled.n_cols),
        None,
        rows_of,
        "highs",
    )


def test_a_limit_that_binds_with_an_incumbent_reads_its_values():
    # HiGHS stops after one node and reports the incumbent its root
    # heuristics found: objective 2970.81 under a bound of 2990.44.
    # `mip_max_nodes` is what HiGHS reports as kSolutionLimit.
    solution = knapsack().solve(options={"node_limit": 1})
    assert solution.status == "solution_limit"
    assert solution.feasible is True
    assert 2000.0 < solution.objective < solution.bound
    assert np.isfinite(solution.bound)
    assert 0.0 < solution.gap < 0.05
    assert solution.primal("x").to_dense().sum() >= 1.0
    assert repr(solution).startswith("Solution('solution_limit', objective 29")
    assert repr(solution).endswith("%)")
    with pytest.raises(ValueError, match="duals are defined at status 'optimal' only"):
        solution.dual("cap")


def test_a_limit_that_binds_with_no_incumbent_raises_on_every_value():
    # HiGHS stops before its first node and reports primal_solution_status
    # kSolutionStatusNone, with mip_dual_bound infinite
    solution = tiny_knapsack().solve(options={"node_limit": 0})
    assert solution.status == "solution_limit"
    assert solution.feasible is False
    assert solution.bound is None
    assert solution.gap is None
    assert repr(solution) == "Solution('solution_limit', no values)"
    with pytest.raises(ValueError, match="reports no feasible point"):
        solution.objective
    with pytest.raises(ValueError, match="reports no feasible point"):
        solution.primal("x")
    with pytest.raises(ValueError, match="duals are defined at status 'optimal' only"):
        solution.dual("cap")


def test_an_lp_stopped_at_an_iteration_limit_reports_no_feasible_point():
    # one simplex iteration from a basis that misses every row: HiGHS
    # reports primal_solution_status kSolutionStatusInfeasible
    solution = dense_lp().solve(
        options={"iteration_limit": 1, "presolve": "off", "method": "simplex"}
    )
    assert solution.status == "iteration_limit"
    assert solution.feasible is False
    assert solution.bound is None
    assert solution.gap is None
    assert repr(solution) == "Solution('iteration_limit', no values)"
    with pytest.raises(ValueError, match="reports no feasible point"):
        solution.objective


def test_a_solve_stopped_at_its_gap_reports_optimal_beside_the_gap():
    # HiGHS reports kOptimal at a mip_rel_gap stop; the gap it stopped at
    # is 0.121 against the 0.2 asked for
    solution = knapsack().solve(options={"mip_gap": 0.2})
    assert solution.status == "optimal"
    assert solution.feasible is True
    assert 0.0 < solution.gap <= 0.2
    assert np.isfinite(solution.bound)
    assert solution.objective > 2000.0
    assert solution.primal("x").to_dense().sum() >= 1.0


def test_an_lp_at_an_optimum_bounds_its_objective_by_the_objective():
    solution = transport().solve()
    assert solution.status == "optimal"
    assert solution.feasible is True
    assert solution.objective == pytest.approx(135.0)
    assert solution.bound == pytest.approx(135.0)
    assert solution.gap == 0.0
    assert repr(solution) == "Solution('optimal', objective 135)"


def test_a_constant_in_the_objective_shifts_the_bound_with_the_objective():
    solution = transport(constant=10.0).solve()
    assert solution.objective == pytest.approx(145.0)
    assert solution.bound == pytest.approx(145.0)
    assert solution.gap == 0.0


def test_a_mixed_integer_optimum_bounds_its_objective_by_the_dual_bound():
    solution = tiny_knapsack().solve()
    assert solution.status == "optimal"
    assert solution.feasible is True
    assert solution.objective == pytest.approx(12.0)
    assert solution.bound == pytest.approx(12.0)
    assert solution.gap == 0.0
    assert repr(solution) == "Solution('optimal', objective 12)"


def test_a_repr_reports_the_objective_the_bound_and_the_gap():
    m = transport()
    assert repr(solution_over(m, "time_limit", False, 0.0, None)) == (
        "Solution('time_limit', no values)"
    )
    assert repr(solution_over(m, "time_limit", True, 9190.19, None)) == (
        "Solution('time_limit', objective 9190.19, no bound)"
    )
    assert repr(solution_over(m, "optimal", True, 135.0, 135.0)) == (
        "Solution('optimal', objective 135)"
    )
    assert repr(solution_over(m, "time_limit", True, 9190.19, 9745.64)) == (
        "Solution('time_limit', objective 9190.19, gap 6.04%)"
    )


def test_a_gap_over_a_zero_objective_is_zero_or_infinite():
    m = transport()
    assert solution_over(m, "time_limit", True, 0.0, 0.0).gap == 0.0
    assert solution_over(m, "time_limit", True, 0.0, 4.0).gap == float("inf")
    assert solution_over(m, "time_limit", False, 0.0, 4.0).gap is None
    assert solution_over(m, "time_limit", True, 4.0, None).gap is None


def test_a_result_validates_what_an_adapter_reports():
    values = np.zeros(1)
    with pytest.raises(ValueError, match="the statuses are"):
        Result("solved", True, 1.0, None, values, None, None)
    with pytest.raises(ValueError, match="status is 'optimal' and feasible is False"):
        Result("optimal", False, 1.0, None, values, None, None)
    with pytest.raises(ValueError, match="bound is inf"):
        Result("time_limit", True, 1.0, float("inf"), values, None, None)
    with pytest.raises(ValueError, match="objective is inf"):
        Result("time_limit", True, float("inf"), None, values, None, None)
    stopped = Result("time_limit", False, float("inf"), None, values, None, None)
    assert stopped.feasible is False
    assert stopped.bound is None
