import tracemalloc

import numpy as np
import pytest

from nimopt import Model, Param, Set, Sum, dumps, loads
from nimopt.solution import reduced_costs
from singleton_models import neg_max, plant


def dense(array):
    return np.asarray(array.to_dense()).tolist()


@pytest.mark.parametrize("singleton_rows", ["rows", "bounds"])
def test_the_plant_solves_to_the_same_values(singleton_rows):
    s = plant().solve(singleton_rows=singleton_rows)
    assert s.status == "optimal"
    assert s.objective == pytest.approx(43.0)
    assert dense(s.primal("p")) == pytest.approx(
        np.array([[5.0, 5.0, 3.0], [3.0, 7.0, 0.0]])
    )
    assert dense(s.dual("floor")) == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    )
    assert dense(s.dual("cap")) == pytest.approx(
        np.array([[-2.0, -2.0, 0.0], [0.0, 0.0, 0.0]])
    )
    assert dense(s.dual("demand")) == pytest.approx([3.0, 3.0, 1.0])
    assert dense(s.dual("p", kind="variable")) == pytest.approx(
        np.array([[0.0] * 3] * 2)
    )


@pytest.mark.parametrize("singleton_rows", ["rows", "bounds"])
def test_sense_max_with_a_negative_coefficient(singleton_rows):
    s = neg_max().solve(singleton_rows=singleton_rows)
    assert s.objective == pytest.approx(12.0)
    assert dense(s.dual("neg")) == pytest.approx([-0.5] * 3)
    assert dense(s.dual("x", kind="variable")) == pytest.approx([0.0] * 3)


def one_column(rows, lower=-np.inf, integer=False, sense="min"):
    """Minimize or maximize one column `x` under the given rows.

    Each row is `(name, coefficient, sense, rhs)`.
    """
    T = Set("T", np.arange(1))
    m = Model("m", sense=sense)
    x = m.var("x", (T,), lower=lower, integer=integer)
    for name, coefficient, relation, rhs in rows:
        k = Param.from_dense(f"k_{name}", (T,), np.array([coefficient]))
        left = k[T] * x[T]
        m.constraint(
            name, {">=": left >= rhs, "<=": left <= rhs, "==": left == rhs}[relation]
        )
    m.set_objective(Sum(T, x[T]))
    return m


CASES = [
    # rows, variable lower bound, objective, duals by constraint, reduced cost of x
    ([("c", 2.0, ">=", 4.0)], -np.inf, 2.0, {"c": 0.5}, 0.0),
    ([("eq", 1.0, "==", 3.0)], -np.inf, 3.0, {"eq": 1.0}, 0.0),
    (
        [("low1", 1.0, ">=", 1.0), ("low2", 1.0, ">=", 2.0)],
        -np.inf,
        2.0,
        {"low1": 0.0, "low2": 1.0},
        0.0,
    ),
    (
        [("first", 1.0, ">=", 2.0), ("second", 1.0, ">=", 2.0)],
        -np.inf,
        2.0,
        {"first": 1.0, "second": 0.0},
        0.0,
    ),
    (
        [
            ("loose", 1.0, ">=", 1.0),
            ("high", 1.0, ">=", 2.0),
            ("same", 1.0, ">=", 2.0),
        ],
        -np.inf,
        2.0,
        {"loose": 0.0, "high": 1.0, "same": 0.0},
        0.0,
    ),
    ([("row", 1.0, ">=", 2.0)], 2.0, 2.0, {"row": 1.0}, 0.0),
    ([("row", 1.0, ">=", 2.0)], 3.0, 3.0, {"row": 0.0}, 1.0),
]


@pytest.mark.parametrize("rows, lower, objective, duals, reduced", CASES)
def test_each_rule_of_the_duals(rows, lower, objective, duals, reduced):
    s = one_column(rows, lower).solve(singleton_rows="bounds")
    assert s.objective == pytest.approx(objective)
    for name, value in duals.items():
        assert dense(s.dual(name)) == pytest.approx([value]), name
    assert dense(s.dual("x", kind="variable")) == pytest.approx([reduced])


def test_each_constraint_is_one_block_of_the_record():
    a = one_column([("first", 1.0, ">=", 2.0), ("second", 1.0, ">=", 2.0)]).assemble(
        singleton_rows="bounds"
    )
    assert (a.singletons.n_rows, a.singletons.rows_of("first")) == (2, slice(0, 1))
    assert a.singletons.rows_of("second") == slice(1, 2)


def ten_thousand_columns():
    """Ten constraints over a set of 1,000 members: one converted row per column."""
    J = Set("J", np.arange(10))
    T = Set("T", np.arange(1000))
    m = Model("m")
    x = m.var("x", (J, T), lower=0.0)
    for j in range(10):
        m.constraint(f"floor_{j}", x[j, T] >= 1.0)
    cost = Param.from_dense("cost", (J, T), np.full((10, 1000), 2.0))
    m.set_objective(Sum(J, T, cost[J, T] * x[J, T]))
    return m


def test_the_duals_allocate_by_the_block_and_not_by_the_record():
    m = ten_thousand_columns()
    assert m.solve(singleton_rows="bounds").status == "optimal"
    a = m.assemble(singleton_rows="bounds")
    # The loaded matrix has no row, so a reduced cost is the cost of its column.
    reduced = reduced_costs(a, np.zeros(a.n_rows))
    assert (a.n_cols, a.singletons.n_rows) == (10000, 10000)
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    shares, declared = a.singletons.duals(reduced, a.col_lower, a.col_upper, "min")
    peak = tracemalloc.get_traced_memory()[1] - before
    tracemalloc.stop()
    assert (shares.tolist()[:2], declared.tolist()[:2]) == ([2.0, 2.0], [0.0, 0.0])
    # Reading the two bound vectors, the two parts and the masks at the size
    # of the record peaks at 989,944 bytes on this model; reading one block at
    # a time peaks at 274,888 bytes.
    assert peak < 500_000, f"the call peaks {peak} bytes above the start"


def test_a_fractional_bound_on_an_integer_column():
    for singleton_rows in ("rows", "bounds"):
        m = one_column([("top", 1.0, "<=", 4.5)], integer=True, sense="max")
        s = m.solve(singleton_rows=singleton_rows)
        assert s.objective == pytest.approx(4.0)
        assert s.has_duals is False


def test_a_model_read_from_text_solves_under_both_values():
    for singleton_rows in ("rows", "bounds"):
        s = loads(dumps(plant(), inline=True)).solve(singleton_rows=singleton_rows)
        assert s.objective == pytest.approx(43.0)


def test_the_cost_model_solves_to_the_same_objective():
    from test_piecewise import cost_model

    rows = cost_model().solve(singleton_rows="rows")
    bounds = cost_model().solve(singleton_rows="bounds")
    assert rows.status == bounds.status == "optimal"
    assert bounds.objective == pytest.approx(rows.objective)


@pytest.mark.parametrize("singleton_rows", ["rows", "bounds"])
def test_a_coefficient_highs_reads_as_zero_is_not_converted(singleton_rows):
    T = Set("T", np.arange(1))
    k = Param.from_dense("k", (T,), np.array([1e-21]))
    m = Model("m")
    x = m.var("x", (T,), lower=-10.0)
    m.constraint("c", k[T] * x[T] >= 5e-21)
    m.set_objective(Sum(T, x[T]))
    assert m.solve(singleton_rows=singleton_rows).objective == pytest.approx(-10.0)


@pytest.mark.parametrize("call", ["session", "solve"])
def test_an_unknown_value_raises(call):
    with pytest.raises(ValueError, match="singleton_rows is 'bounds' or 'rows'"):
        getattr(plant(), call)(singleton_rows="columns")
