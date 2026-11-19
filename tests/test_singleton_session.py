import tracemalloc

import numpy as np
import pytest

from nimopt import Model, Set, Sum
from optimality import residuals
from singleton_models import plant


def opened(singleton_rows):
    model = plant()
    return (
        model,
        model.variables["p"],
        model.session("highs", singleton_rows=singleton_rows),
    )


@pytest.mark.parametrize("singleton_rows", ["rows", "bounds"])
def test_a_session_bound_does_not_loosen_a_converted_row(singleton_rows):
    model, p, session = opened(singleton_rows)
    with session:
        assert session.solve().objective == pytest.approx(43.0)
        session.set_bound(p, lower=-5.0)
        solution = session.solve()
        assert solution.objective == pytest.approx(43.0)
        found = residuals(model, solution)
        assert found["stationarity"] < 1e-6 and found["wrong_sign"] == 0, found


@pytest.mark.parametrize("singleton_rows", ["rows", "bounds"])
def test_fix_inside_and_outside_the_converted_bounds_and_unfix(singleton_rows):
    _, p, session = opened(singleton_rows)
    with session:
        session.fix(p["a", 0], 4.0)
        assert session.solve().objective == pytest.approx(45.0)
        session.fix(p["a", 0], 7.0)
        assert session.solve().status == "infeasible"
        session.unfix(p["a", 0])
        assert session.solve().objective == pytest.approx(43.0)


def test_n_rows_reports_the_loaded_rows_and_an_appended_row_is_a_row():
    model, p, session = opened("bounds")
    G, T = (held for held in model.variables["p"].sets)
    with session:
        assert (session.n_rows, model.n_rows) == (3, 15)
        session.add_row("extra", p[G, T] <= 9.0)
        assert session.n_rows == 9
        solution = session.solve()
        assert solution.objective == pytest.approx(43.0)
        assert np.asarray(solution.dual("cap").to_dense()).shape == (2, 3)
        assert np.asarray(solution.dual("cap").to_dense())[0, 0] == pytest.approx(-2.0)


def test_a_closed_session_reloads_the_converted_bounds():
    _, _, session = opened("bounds")
    assert session.solve().objective == pytest.approx(43.0)
    session.close()
    assert session.solve().objective == pytest.approx(43.0)


def crossing():
    T = Set("T", np.arange(2))
    m = Model("crossing")
    x = m.var("x", (T,), lower=-np.inf)
    y = m.var("y", (T,), upper=10.0)
    m.constraint("lo", x[T] >= 2.0)
    m.constraint("up", x[T] <= 1.0)
    m.constraint("link", x[T] + y[T] >= 1.0)
    m.set_objective(Sum(T, x[T] + y[T]))
    return m


@pytest.mark.parametrize("singleton_rows", ["rows", "bounds"])
def test_a_conflict_of_converted_rows_reports_their_constraints(singleton_rows):
    model = crossing()
    with model.session("highs", singleton_rows=singleton_rows) as session:
        assert session.solve().status == "infeasible"
        found = session.diagnose()
    assert {row.constraint for row in found.conflict} == {"lo", "up"}
    for row in found.conflict:
        assert row.index == model.row(row.constraint, **row.coordinate).index


def test_a_kept_row_of_a_conflict_has_its_declared_index():
    T = Set("T", np.arange(2))
    m = Model("kept")
    x = m.var("x", (T,), lower=-np.inf)
    y = m.var("y", (T,), lower=-np.inf)
    m.constraint("floor", x[T] >= 0.0)
    m.constraint("a", x[T] + y[T] >= 5.0)
    m.constraint("b", x[T] + y[T] <= 4.0)
    m.set_objective(Sum(T, x[T] + y[T]))
    with m.session("highs", singleton_rows="bounds") as session:
        session.solve()
        found = session.diagnose()
    assert found.conflict
    for row in found.conflict:
        assert row.index == m.row(row.constraint, **row.coordinate).index
        assert row.index >= 2


@pytest.mark.parametrize("singleton_rows", ["rows", "bounds"])
def test_an_unbounded_model_reports_its_ray(singleton_rows):
    T = Set("T", np.arange(1))
    m = Model("ray")
    x = m.var("x", (T,), lower=-np.inf)
    y = m.var("y", (T,), lower=-np.inf)
    m.constraint("top", x[T] <= 5.0)
    m.constraint("pair", x[T] + y[T] >= 0.0)
    m.set_objective(Sum(T, x[T]))
    with m.session("highs", singleton_rows=singleton_rows) as session:
        assert session.solve().status == "unbounded"
        ray = {term.variable: term.direction for term in session.diagnose().ray}
    assert ray["x"] < 0 and ray["y"] > 0


@pytest.mark.parametrize("singleton_rows", ["rows", "bounds"])
def test_a_model_of_one_singleton_row_alone_is_unbounded(singleton_rows):
    T = Set("T", np.arange(1))
    m = Model("ray")
    x = m.var("x", (T,), lower=-np.inf)
    m.constraint("top", x[T] <= 5.0)
    m.set_objective(Sum(T, x[T]))
    with m.session("highs", singleton_rows=singleton_rows) as session:
        assert session.solve().status in ("unbounded", "unbounded_or_infeasible")


def one_sided():
    T = Set("T", np.arange(1))
    m = Model("side")
    x = m.var("x", (T,), lower=-np.inf)
    y = m.var("y", (T,), lower=-np.inf)
    m.constraint("xlo", x[T] >= 0.0)
    m.constraint("xup", x[T] <= 5.0)
    m.constraint("ylo", y[T] >= 0.0)
    m.constraint("yup", y[T] <= 10.0)
    m.constraint("need", x[T] + y[T] >= 20.0)
    m.set_objective(Sum(T, x[T] + y[T]))
    return m


@pytest.mark.parametrize("singleton_rows", ["rows", "bounds"])
def test_a_conflict_reports_the_converted_rows_whose_bound_is_in_it(singleton_rows):
    model = one_sided()
    with model.session("highs", singleton_rows=singleton_rows) as session:
        assert session.solve().status == "infeasible"
        found = session.diagnose()
    assert {row.constraint for row in found.conflict} == {"xup", "yup", "need"}


def test_model_row_is_the_same_under_both_values():
    m = plant()
    assert m.row("demand", T=1).index == 13
    assert m.row("cap", G="b", T=2).index == 11


def test_an_appended_row_of_a_conflict_has_the_index_of_the_declared_rows():
    model, p, session = opened("bounds")
    G, _ = (held for held in model.variables["p"].sets)
    with session:
        assert session.solve().objective == pytest.approx(43.0)
        session.add_row("cut", Sum(G, p[G, 0]) <= 4.0)
        assert session.solve().status == "infeasible"
        found = session.diagnose()
    cut = next(row for row in found.conflict if row.constraint == "cut")
    assert cut.index == model.n_rows


def test_a_bound_of_one_variable_computes_no_bound_of_another(monkeypatch):
    T = Set("T", np.arange(2))
    m = Model("two", sense="max")
    x = m.var("x", (T,), lower=-np.inf)
    y = m.var("y", (T,), lower=-np.inf)
    m.constraint("x_top", x[T] <= 4.0)
    m.constraint("y_top", y[T] <= 9.0)
    m.set_objective(Sum(T, x[T] + y[T]))
    with m.session("highs", singleton_rows="bounds") as session:

        def must_not_run():
            raise AssertionError("the bounds of another variable were computed")

        monkeypatch.setattr(m.constraints["y_top"], "singleton_rhs", must_not_run)
        session.set_bound(x, lower=0.0)
        monkeypatch.undo()
        assert session.solve().objective == pytest.approx(26.0)


def test_a_clip_allocates_no_vector_over_the_whole_record():
    # ten variables over 1,000 members: the record holds 10,000 converted rows,
    # and one float64 vector over them is 80,000 bytes. One clip of one
    # variable's columns peaks at 42,952 bytes; the code that computed the
    # bound of every converted row of the record at each clip peaked at
    # 212,672 bytes on this model
    T = Set("T", np.arange(1000))
    m = Model("wide")
    variables = [m.var(f"v{at}", (T,), lower=-np.inf) for at in range(10)]
    for at, held in enumerate(variables):
        m.constraint(f"top{at}", held[T] <= 4.0)
    with m.session("highs", singleton_rows="bounds") as session:
        singletons = session.assembled.singletons
        assert singletons.n_rows == 10_000
        columns = variables[0]._columns()
        lower = np.full(columns.stop - columns.start, -np.inf)
        upper = np.full(columns.stop - columns.start, np.inf)
        tracemalloc.start()
        singletons.clip(columns.start, columns.stop, lower, upper)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    assert peak < 80_000, peak
