import tracemalloc

import numpy as np
import pytest

from nimopt import Model, Param, Set, Sum, subset
from nimopt.models import commitment, transport
from session_models import loop, plant_model, products, two

pytest.importorskip("highspy")


def swept():
    """Return the loop model and its session after the sweep of one fix."""
    m, _, H, g = loop()
    session = m.session()
    objectives = []
    for value in (0.0, 40.0, 80.0):
        session.fix(g["wind", 1], value)
        solution = session.solve()
        objectives.append(solution.objective)
    return m, H, g, session, objectives, solution


def first_member(variable):
    """Return a reference to the first member of a variable."""
    labels = variable.labels_at([0])
    return variable[tuple(labels[d][0] for d in variable.dims)]


def test_a_sweep_of_one_fix_solves_each_point():
    _, _, _, session, objectives, solution = swept()
    assert (session.n_columns, session.n_rows) == (36, 48)
    assert objectives == pytest.approx([37050.0, 34650.0, 32700.0])
    # the dual of demand at hour 1
    assert solution.dual("demand").to_dense()[0] == pytest.approx(30.0)


def test_an_appended_row_is_solved_and_read_by_name():
    _, H, g, session, _, _ = swept()
    session.add_row("coal_cap_total", Sum(H, g["coal", H]) <= 1000.0)
    assert session.n_rows == 49
    solution = session.solve()
    assert solution.objective == pytest.approx(35400.0)
    assert solution.dual("coal_cap_total").to_dense().shape == ()


def test_a_session_changes_nothing_in_the_model_it_was_opened_on():
    m, H, g, session, _, _ = swept()
    session.add_row("coal_cap_total", Sum(H, g["coal", H]) <= 1000.0)
    session.solve()
    assert session.model is m
    assert m.n_rows == 48
    assert "coal_cap_total" not in m.constraints
    assert m.solve().objective == pytest.approx(31500.0)
    # the session solves on after the model it was opened on is solved
    assert session.solve().objective == pytest.approx(35400.0)


def test_a_point_fix_is_solved():
    m, _, _, x = plant_model()
    with m.session() as session:
        assert session.solve().objective == pytest.approx(30.0)
        session.fix(x["p0", "w0"], 4.0)
        # p0 at 4 costs 20, p1 covers 6 at 1, and two weeks cost 10 each
        assert session.solve().objective == pytest.approx(46.0)


def test_a_bound_on_one_element_is_solved():
    m, _, make = products()
    with m.session() as session:
        assert session.solve().objective == pytest.approx(180000.0)
        session.set_bound(make["coils"], upper=1000.0)
        solution = session.solve()
        assert solution.objective == pytest.approx(175000.0)
        made = solution.primal("make").to_dense().tolist()
        assert made == pytest.approx([0.0, 1000.0, 5000.0])


def test_the_dual_of_an_appended_row_is_read_by_its_name():
    m, i, x = two()
    with m.session() as session:
        assert session.solve().objective == pytest.approx(4.0)
        session.add_row("cut", Sum(i, x[i]) >= 7.0)
        assert session.n_rows == 3
        solution = session.solve()
        assert solution.objective == pytest.approx(7.0)
        assert float(solution.dual("cut").to_dense()) == pytest.approx(1.0)
        assert float(solution.dual("total").to_dense()) == pytest.approx(0.0)
        assert solution.dual("x").to_dense().tolist() == pytest.approx([0.0, 0.0])


def test_a_fix_and_an_appended_row_apply_together():
    m, i, x = two()
    with m.session() as session:
        session.solve()
        session.fix(x["a"], 1.0)
        session.add_row("cut", Sum(i, x[i]) >= 6.0)
        solution = session.solve()
        assert solution.objective == pytest.approx(6.0)
        assert solution.primal("x").to_dense().tolist() == pytest.approx([1.0, 5.0])


def test_the_duals_of_rows_appended_across_several_growths_are_read():
    m, i, x = two()
    with m.session() as session:
        session.solve()
        for k in range(40):
            session.add_row(f"cut{k}", Sum(i, x[i]) >= 4.0 + 0.1 * k)
        assert session.n_rows == 42
        solution = session.solve()
        assert solution.objective == pytest.approx(7.9)
        assert float(solution.dual("cut39").to_dense()) == pytest.approx(1.0)
        assert float(solution.dual("cut0").to_dense()) == pytest.approx(0.0)


def test_a_resumed_solve_under_barrier_returns_the_optimum():
    m, _, _, g = loop()
    with m.session(options={"method": "barrier"}) as session:
        for value, objective in ((0.0, 37050.0), (40.0, 34650.0), (80.0, 32700.0)):
            session.fix(g["wind", 1], value)
            assert session.solve().objective == pytest.approx(objective, rel=1e-6)


def test_a_resumed_solve_takes_fewer_iterations_than_a_fresh_session():
    # measured: 1 simplex iteration resumed, 38 in the fresh session
    model = transport.definition().build(transport.data(10))
    target = first_member(model.variables["flow"])
    with model.session() as resumed:
        resumed.solve()
        resumed.fix(target, 0.0)
        warm = resumed.solve()
    with model.session() as fresh:
        fresh.fix(target, 0.0)
        cold = fresh.solve()
    assert warm.iterations < cold.iterations
    assert warm.objective == pytest.approx(cold.objective)


def test_solving_twice_with_no_change_returns_the_same_objective():
    m, _, _, _ = loop()
    with m.session() as session:
        first = session.solve().objective
        assert first == pytest.approx(31500.0)
        assert session.solve().objective == pytest.approx(first)


def test_a_change_before_the_first_solve_is_solved():
    m, i, x = two()
    with m.session() as session:
        session.fix(x["a"], 1.0)
        session.add_row("cut", Sum(i, x[i]) >= 6.0)
        assert not session.backend_open
        solution = session.solve()
        assert solution.objective == pytest.approx(6.0)
        assert solution.primal("x").to_dense().tolist() == pytest.approx([1.0, 5.0])
        assert float(solution.dual("cut").to_dense()) == pytest.approx(1.0)


def test_a_closed_session_loads_the_changed_model_again():
    m, i, x = two()
    session = m.session()
    session.solve()
    session.fix(x["a"], 1.0)
    session.add_row("cut", Sum(i, x[i]) >= 6.0)
    session.close()
    assert not session.backend_open
    solution = session.solve()
    assert solution.objective == pytest.approx(6.0)
    assert float(solution.dual("cut").to_dense()) == pytest.approx(1.0)
    assert session.n_rows == 3
    session.close()


def test_every_method_raises_after_the_model_declares_a_variable():
    m, i, x = two()
    with m.session() as session:
        session.solve()
        m.var("late", (i,))
        calls = (
            lambda: session.fix(x["a"], 1.0),
            lambda: session.unfix(x["a"]),
            lambda: session.set_bound(x["a"], upper=2.0),
            lambda: session.add_row("cut", Sum(i, x[i]) >= 6.0),
            session.solve,
        )
        for call in calls:
            with pytest.raises(ValueError, match="open a new Session"):
                call()


def test_a_row_with_a_name_the_model_declares_raises():
    m, i, x = two()
    with m.session() as session:
        with pytest.raises(ValueError, match="total"):
            session.add_row("total", Sum(i, x[i]) >= 6.0)
        session.add_row("cut", Sum(i, x[i]) >= 6.0)
        with pytest.raises(ValueError, match="cut"):
            session.add_row("cut", Sum(i, x[i]) >= 7.0)
        assert session.n_rows == 3


def test_a_row_over_a_variable_of_another_model_raises():
    m, i, _ = two()
    _, other_i, other_x = two()
    other = Model("other")
    y = other.var("y", (i,))
    with m.session() as session:
        # the message lists the variables of the session
        with pytest.raises(ValueError, match="'x'"):
            session.add_row("cut", Sum(i, y[i]) >= 6.0)
        # a variable of the same name in another model is another variable
        with pytest.raises(ValueError, match="'x'"):
            session.add_row("cut", Sum(other_i, other_x[other_i]) >= 6.0)
        assert session.n_rows == 2
        assert session.solve().objective == pytest.approx(4.0)


def test_a_conflict_reports_an_appended_row_and_the_current_bounds():
    m, i, x = two()
    with m.session() as session:
        session.solve()
        session.set_bound(x, upper=2.0)
        session.add_row("cut", Sum(i, x[i]) >= 7.0)
        assert session.solve().status == "infeasible"
        diagnosis = session.diagnose()
        cut = next(row for row in diagnosis.conflict if row.constraint == "cut")
        assert cut.index == 2
        assert [term.coefficient for term in cut.terms] == [1.0, 1.0]
        assert (cut.sense, cut.lower) == (">=", 7.0)
        assert [held.upper for held in diagnosis.columns] == [2.0, 2.0]
        assert "cut" in repr(diagnosis)


def test_a_lower_bound_above_an_upper_bound_is_infeasible():
    m, _, x = two()
    with m.session() as session:
        session.solve()
        session.set_bound(x["a"], lower=5.0, upper=1.0)
        assert session.solve().status == "infeasible"


def test_a_change_after_an_infeasible_solve_is_solved():
    m, _, x = two()
    with m.session() as session:
        session.solve()
        session.set_bound(x["a"], lower=5.0, upper=1.0)
        assert session.solve().status == "infeasible"
        session.set_bound(x["a"], lower=0.0, upper=10.0)
        assert session.solve().objective == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("position", "value", "objective"),
    # the unit at column 0 off leaves the optimum; the unit at column 1 on costs 500
    [(0, 0.0, 121500.0), (1, 1.0, 122000.0)],
)
def test_a_model_with_integer_columns_is_solved_again(position, value, objective):
    model = commitment.definition().build(commitment.data(3))
    on = model.variables["on"]
    labels = on.labels_at([position])
    target = on[tuple(labels[d][0] for d in on.dims)]
    with model.session() as resumed:
        resumed.solve()
        resumed.fix(target, value)
        again = resumed.solve()
    with model.session() as fresh:
        fresh.fix(target, value)
        cold = fresh.solve()
    assert again.status == "optimal"
    assert again.objective == pytest.approx(cold.objective)
    assert again.objective == pytest.approx(objective)
    assert not again.has_duals


def test_the_assembled_bounds_are_the_current_bounds():
    m, _, x = two()
    with m.session() as session:
        session.fix(x["a"], 1.0)
        assert session.assembled.col_lower[0] == 1.0
        assert session.assembled.col_upper[0] == 1.0


def test_an_adapter_without_mutation_raises_at_the_change():
    pytest.importorskip("gurobipy")
    m, i, x = two()
    with m.session("gurobi") as session:
        calls = (
            lambda: session.fix(x["a"], 1.0),
            lambda: session.unfix(x["a"]),
            lambda: session.set_bound(x["a"], upper=2.0),
            lambda: session.add_row("cut", Sum(i, x[i]) >= 6.0),
        )
        for call in calls:
            with pytest.raises(ValueError, match="capabilities"):
                call()
        assert session.solve().objective == pytest.approx(4.0)
        assert session.solve().objective == pytest.approx(4.0)
        assert session.n_rows == 2
        assert session.assembled.col_upper.tolist() == [10.0, 10.0]


def test_an_appended_row_allocates_less_than_a_tenth_of_the_matrix():
    R = Set("R", np.arange(5000))
    K = Set("K", np.arange(200))
    weight = Param.from_dense("weight", (R, K), np.ones((5000, 200)))
    m = Model("wide")
    x = m.var("x", (K,), lower=0.0, upper=1.0)
    m.constraint("rows", Sum(K, weight[R, K] * x[K]) <= 150.0)
    m.set_objective(Sum(K, x[K]))
    with m.session() as session:
        assert session.assembled.values.size == 1_000_000
        session.solve()
        tracemalloc.start()
        session.add_row("cut", Sum(K, x[K]) <= 100.0)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert session.n_rows == 5001
    # the matrix has 1,000,000 nonzeros of 16 bytes each
    assert peak < 0.1 * 16 * 1_000_000


def test_unfix_restores_the_declared_bounds():
    m, _, x = two()
    with m.session() as session:
        session.fix(x["a"], 6.0)
        assert session.solve().objective == pytest.approx(7.0)
        session.unfix(x["a"])
        assert session.solve().objective == pytest.approx(4.0)
        assert session.assembled.col_lower.tolist() == [0.0, 0.0]
        assert session.assembled.col_upper.tolist() == [10.0, 10.0]


def test_a_parameter_fix_on_a_whole_variable_is_solved():
    m, P, W, x = plant_model()
    held = Param.from_long("held", (P, W), {"P": ["p0"], "W": ["w0"]}, [4.0])
    with m.session() as session:
        session.fix(x, held)
        assert session.solve().objective == pytest.approx(46.0)
        with pytest.raises(ValueError, match="'x'"):
            session.fix(x["p0", W], held)


def test_a_rejected_change_changes_nothing():
    m, i, x = two()
    with m.session() as session:
        session.solve()
        with pytest.raises(ValueError):
            session.fix(x["a"], float("inf"))
        with pytest.raises(TypeError):
            session.fix(2.0 * x[i], 1.0)
        with pytest.raises(ValueError):
            session.set_bound(x["a"])
        with pytest.raises(ValueError, match="has no fixed column"):
            session.unfix(x)
        assert session.assembled.col_lower.tolist() == [0.0, 0.0]
        assert session.assembled.col_upper.tolist() == [10.0, 10.0]
        assert session.solve().objective == pytest.approx(4.0)


def test_a_row_over_a_set_appends_one_row_per_member():
    m, i, x = two()
    with m.session() as session:
        session.solve()
        session.add_row("floor", x[i] >= 3.0)
        assert session.n_rows == 4
        solution = session.solve()
        assert solution.objective == pytest.approx(6.0)
        assert solution.dual("floor").to_dense().tolist() == pytest.approx([1.0, 1.0])


def test_a_condition_narrows_the_rows_of_an_appended_constraint():
    m, i, x = two()
    only_b = subset((i,), {"i": ["b"]})
    with m.session() as session:
        session.add_row("floor", x[i] >= 5.0, where=only_b)
        assert session.n_rows == 3
        assert session.solve().objective == pytest.approx(5.0)


def test_a_row_with_no_coefficient_is_loaded_at_its_place():
    m, Pr, make = products()
    only_coils = Param.from_long("only_coils", (Pr,), {"Pr": ["coils"]}, [1.0])
    every = subset((Pr,), {"Pr": ["bands", "coils", "plate"]})
    session = m.session()
    session.solve()
    # the rows at bands and at plate have no coefficient
    session.add_row("limit", only_coils[Pr] * make[Pr] <= 1000.0, over=every)
    assert session.n_rows == 4
    for _ in range(2):
        solution = session.solve()
        assert solution.objective == pytest.approx(175000.0)
        duals = solution.dual("limit").to_dense().tolist()
        # coils earns 30 and plate 29: one more hour of coils is worth 1
        assert [abs(dual) for dual in duals] == pytest.approx([0.0, 1.0, 0.0])
        # the second pass loads the model again, with the appended rows
        session.close()


def test_a_solution_of_a_session_reads_the_rows_and_names_of_its_own_solve():
    m, i, x = two()
    with m.session() as session:
        before = session.solve()
        session.add_row("cut", Sum(i, x[i]) >= 7.0)
        after = session.solve()
    # the solution read before the row was appended has no dual for it
    with pytest.raises(KeyError, match="cut"):
        before.dual("cut")
    assert float(before.dual("total").to_dense()) == pytest.approx(1.0)
    assert float(after.dual("cut").to_dense()) == pytest.approx(1.0)
    # the model the session was opened on has no row named cut
    with pytest.raises(KeyError, match="cut"):
        m.solve().dual("cut")
