import numpy as np
import pytest

from nimopt import Model, Param, Session, Set, Sum


def dispatch(load=(25.0, 20.0, 5.0)):
    """Three snapshots against a fleet carrying 30."""
    SNAP = Set("snapshot", np.arange(len(load)))
    GEN = Set("generator", np.array(["wind", "gas"]))
    p_max = Param.from_dense("p_max", (GEN,), np.array([10.0, 20.0]))
    want = Param.from_dense("load", (SNAP,), np.asarray(load))
    cost = Param.from_dense("cost", (GEN,), np.array([1.0, 5.0]))
    m = Model("dispatch", sense="min")
    p = m.var("p", (SNAP, GEN), lower=0.0, upper=p_max)
    m.eq("balance", Sum(GEN, p[SNAP, GEN]) == want[SNAP])
    m.set_objective(Sum(SNAP, GEN, cost[GEN] * p[SNAP, GEN]))
    return m


def infeasible():
    """The second snapshot asks for 100 against a fleet carrying 30."""
    return dispatch(load=(25.0, 100.0, 5.0))


def unbounded():
    """Two columns with no upper bound and a negative cost."""
    T = Set("T", np.arange(2))
    one = Param.from_dense("one", (T,), np.ones(2))
    price = Param.from_dense("price", (T,), np.array([-1.0, -1.0]))
    m = Model("unbounded", sense="min")
    x = m.var("x", (T,), lower=0.0, upper=np.inf)
    m.eq("floor", one[T] * x[T] >= 1.0)
    m.set_objective(Sum(T, price[T] * x[T]))
    return m


def odd():
    """Feasible as an LP and infeasible with an integer column: 2x == 5."""
    T = Set("T", np.arange(1))
    two = Param.from_dense("two", (T,), np.full(1, 2.0))
    one = Param.from_dense("one", (T,), np.ones(1))
    m = Model("odd", sense="min")
    x = m.var("x", (T,), lower=0.0, upper=10.0, integer=True)
    m.eq("odd", two[T] * x[T] == 5.0)
    m.set_objective(Sum(T, one[T] * x[T]))
    return m


def test_a_session_holds_the_model_it_was_opened_on():
    m = dispatch()
    with m.session() as session:
        assert isinstance(session, Session)
        assert session.model is m
        assert session.solver == "highs"
        assert session.assembled.n_rows == m.n_rows


def test_a_session_assembles_once_and_solves_from_that_matrix():
    # the matrix is the session's, so a solve after a solve costs a solve
    with dispatch().session() as session:
        first = session.assembled
        answer = session.solve()
        assert session.assembled is first
        assert answer.objective == pytest.approx(session.solve().objective)


def test_a_solution_names_the_solver_that_produced_it():
    assert dispatch().solve().solver == "highs"


def test_a_session_reports_the_status_of_its_last_solve():
    with dispatch().session() as session:
        assert session.status is None
        session.solve()
        assert session.status == "optimal"


def test_a_closed_session_holds_no_backend():
    session = dispatch().session()
    session.solve()
    assert session.backend_open is True
    session.close()
    assert session.backend_open is False


def test_a_solve_through_a_session_answers_what_a_model_solve_answers():
    m = dispatch()
    with m.session() as session:
        assert session.solve().objective == pytest.approx(m.solve().objective)


def test_the_capabilities_a_session_reads_are_its_adapters():
    from nimopt import capabilities

    with dispatch().session() as session:
        assert session.capabilities is capabilities("highs")


def test_a_solver_the_seam_does_not_adapt_is_refused_where_it_is_named():
    with pytest.raises(ValueError, match="the solvers are"):
        dispatch().session(solver="glpk")


def test_a_mixed_integer_model_carries_no_duals_to_read():
    # a mixed-integer model's duals are not the relaxation's duals, and a
    # vector of zeros handed over as duals is indistinguishable from an answer
    m = Model("m")
    T = Set("T", np.arange(2))
    one = Param.from_dense("one", (T,), np.ones(2))
    x = m.var("x", (T,), upper=3.0, integer=True)
    m.eq("cap", one[T] * x[T] <= 2.0)
    m.set_objective(Sum(T, one[T] * x[T]))
    solved = m.solve()
    assert solved.status == "optimal"
    assert solved.primal("x").to_dense().sum() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="refuses duals"):
        solved.dual("cap")


def test_a_continuous_model_still_reads_its_duals():
    assert dispatch().solve().dual("balance").to_dense().shape == (3,)


def test_a_model_infeasible_only_through_its_integrality_is_infeasible():
    assert odd().solve().status == "infeasible"


def test_a_model_with_no_upper_bound_and_a_negative_cost_is_unbounded():
    assert unbounded().solve().status == "unbounded"


def test_a_conflict_names_the_rows_that_cannot_hold_together():
    # the second snapshot asks for 100 against a fleet carrying 30
    with infeasible().session() as session:
        assert session.solve().status == "infeasible"
        found = session.diagnose()
    assert found.status == "infeasible"
    assert found.solver == "highs"
    assert found.method == "native"
    assert [row.constraint for row in found.conflict] == ["balance"]
    assert [row.coordinate for row in found.conflict] == [{"snapshot": 1}]
    assert found.conflict[0].index == 1


def test_a_conflicting_row_renders_as_model_row_renders_it():
    # one format, because a conflict holds the same Row
    m = infeasible()
    with m.session() as session:
        session.solve()
        found = session.diagnose()
    assert repr(found.conflict[0]) == repr(m.row("balance", snapshot=1))


def test_the_columns_of_a_conflict_carry_the_bounds_the_model_states():
    with infeasible().session() as session:
        session.solve()
        found = session.diagnose()
    assert [held.variable for held in found.columns] == ["p", "p"]
    assert [held.coordinate["generator"] for held in found.columns] == ["wind", "gas"]
    assert [held.upper for held in found.columns] == [10.0, 20.0]


def test_removing_the_conflict_makes_the_model_feasible():
    # each returned set conflicts. Two solvers may return different
    # irreducible sets, so this is what is checked
    from nimopt.solvers import highs

    m = infeasible()
    with m.session() as session:
        session.solve()
        found = session.diagnose()
        assembled = session.assembled
    at = [row.index for row in found.conflict]
    assembled.row_lower[at] = -np.inf
    assembled.row_upper[at] = np.inf
    assert highs.solve(assembled, m.sense)[0] == "optimal"


def test_a_model_that_solved_carries_no_conflict():
    with dispatch().session() as session:
        session.solve()
        found = session.diagnose()
    assert found.status == "optimal"
    assert found.conflict is None
    assert found.columns == ()
    assert found.ray is None


def test_an_unbounded_model_names_the_direction_it_runs_off_in():
    with unbounded().session() as session:
        assert session.solve().status == "unbounded"
        found = session.diagnose()
    assert found.conflict is None
    assert [held.variable for held in found.ray] == ["x"]
    assert found.ray[0].direction > 0.0


def test_a_diagnosis_before_a_solve_is_refused():
    with dispatch().session() as session:
        with pytest.raises(ValueError, match="solve before diagnosing"):
            session.diagnose()


def test_highs_refuses_a_conflict_it_could_not_prove():
    # HiGHS computes its conflict over the linear relaxation, so a model
    # infeasible only through its integrality reaches no conflict at all;
    # the rows it names then are not one, and are not handed over
    with odd().session() as session:
        assert session.solve().status == "infeasible"
        with pytest.raises(RuntimeError, match="linear relaxation"):
            session.diagnose()


def test_a_diagnosis_reads_as_the_rows_that_conflict():
    with infeasible().session() as session:
        session.solve()
        rendered = repr(session.diagnose())
    assert rendered.startswith("infeasible  highs  conflict native")
    assert "balance[snapshot=1]" in rendered
