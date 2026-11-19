import numpy as np
import pytest

from nimopt import Model, Param, Set
from singleton_models import mixed, neg_max, plant


def test_a_constraint_of_one_term_marks_every_row():
    m = plant()
    assert m.constraints["floor"].n_singleton_rows == 6
    assert m.constraints["floor"].singleton_positions().tolist() == [0, 1, 2, 3, 4, 5]
    assert m.constraints["floor"].kept_positions().tolist() == []
    assert m.constraints["cap"].singleton_rhs().tolist() == [5.0] * 3 + [10.0] * 3
    assert m.constraints["demand"].n_singleton_rows == 0
    assert m.constraints["demand"].kept_positions().tolist() == [0, 1, 2]


def test_the_declared_counts_include_the_singleton_rows():
    m = plant()
    assert (m.n_rows, m.nnz) == (15, 18)
    assert (m.constraints["floor"].n_rows, m.constraints["floor"].nnz) == (6, 6)


def test_a_mixed_constraint_marks_its_rows_of_one_coefficient():
    held = mixed().constraints["mixed"]
    assert held.n_singleton_rows == 5
    assert held.singleton_positions().tolist() == [1, 2, 3, 4, 5]
    assert held.kept_positions().tolist() == [0]


def one_term(coefficients, rhs, sense="<="):
    T = Set("T", np.arange(3))
    k = Param.from_dense("k", (T,), np.array(coefficients))
    r = Param.from_dense("r", (T,), np.array(rhs))
    m = Model("m")
    x = m.var("x", (T,), lower=-np.inf)
    relation = k[T] * x[T] <= r[T] if sense == "<=" else k[T] * x[T] >= r[T]
    return m.constraint("c", relation)


def test_a_coefficient_of_zero_is_not_a_singleton_row():
    assert one_term([0.0, 1.0, 1.0], [1.0] * 3).singleton_positions().tolist() == [1, 2]


def test_an_infinite_right_hand_side_is_not_a_singleton_row():
    held = one_term([1.0] * 3, [np.inf, 1.0, 1.0])
    assert held.singleton_positions().tolist() == [1, 2]


def test_a_bound_of_1e20_or_more_is_not_a_singleton_row():
    held = one_term([1e-21, 1.0, 1.0], [1.0] * 3, sense=">=")
    assert held.singleton_positions().tolist() == [1, 2]


def test_a_negative_coefficient_does_not_change_the_marking():
    held = neg_max().constraints["neg"]
    assert held.n_singleton_rows == 3
    assert held.singleton_positions().tolist() == [0, 1, 2]
    assert held.singleton_rhs().tolist() == [-8.0] * 3


def test_the_marking_emits_no_warning_for_a_quotient_that_overflows():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert one_term(
            [5e-324, 1.0, 1.0], [1.0] * 3, sense=">="
        ).singleton_positions().tolist() == [1, 2]
        assert one_term(
            [np.inf, 1.0, 1.0], [np.inf, 1.0, 1.0]
        ).singleton_positions().tolist() == [1, 2]


INF = np.inf


def test_bounds_writes_the_kept_rows_and_the_column_bounds():
    a = plant().assemble(singleton_rows="bounds")
    assert (a.n_rows, a.n_cols, int(a.values.size)) == (3, 6, 6)
    assert a.indptr.tolist() == [0, 2, 4, 6]
    assert a.indices.tolist() == [0, 3, 1, 4, 2, 5]
    assert a.row_lower.tolist() == [8.0, 12.0, 3.0]
    assert a.row_upper.tolist() == [8.0, 12.0, 3.0]
    assert a.col_lower.tolist() == [0.0] * 6
    assert a.col_upper.tolist() == [5.0, 5.0, 5.0, 10.0, 10.0, 10.0]
    assert a.row_of("floor") == slice(0, 0)
    assert a.row_of("cap") == slice(0, 0)
    assert a.row_of("demand") == slice(0, 3)
    assert a.singletons.n_rows == 12
    assert a.singletons.rows_of("cap") == slice(6, 12)
    assert a.singletons.rows_of("demand") is None


def test_rows_builds_the_arrays_of_the_declared_model():
    a = plant().assemble(singleton_rows="rows")
    assert (a.n_rows, int(a.values.size)) == (15, 18)
    assert np.diff(a.indptr).tolist() == [1] * 12 + [2] * 3
    assert a.col_lower.tolist() == [-INF] * 6
    assert a.col_upper.tolist() == [INF] * 6
    assert a.row_of("cap") == slice(6, 12)
    assert a.singletons.n_rows == 0


def test_a_mixed_constraint_keeps_its_row_of_two_coefficients():
    a = mixed().assemble(singleton_rows="bounds")
    assert (a.n_rows, int(a.values.size)) == (1, 2)
    assert a.indices.tolist() == [0, 6]
    assert a.col_upper.tolist() == [INF, 9.0, 9.0, 9.0, 9.0, 9.0, INF, INF, INF]
    assert a.col_lower.tolist() == [-INF] * 6 + [0.0] * 3
    assert a.singletons.kept("mixed").tolist() == [0]


def test_a_negative_coefficient_exchanges_the_two_sides():
    a = neg_max().assemble(singleton_rows="bounds")
    assert (a.n_rows, int(a.values.size)) == (0, 0)
    assert a.indptr.tolist() == [0]
    assert a.col_lower.tolist() == [-INF] * 3
    assert a.col_upper.tolist() == [4.0] * 3


def bounded(lower, row_lower):
    T = Set("T", np.arange(1))
    m = Model("m")
    x = m.var("x", (T,), lower=lower)
    m.constraint("low", x[T] >= row_lower)
    return m.assemble(singleton_rows="bounds")


def test_the_column_bound_is_the_narrowest_of_the_variable_and_the_rows():
    assert bounded(1.0, 0.0).col_lower.tolist() == [1.0]
    assert bounded(1.0, 2.0).col_lower.tolist() == [2.0]


def test_a_condition_bounds_only_the_columns_of_its_rows():
    from nimopt import subset

    T = Set("T", np.arange(3))
    m = Model("m")
    y = m.var("y", (T,), lower=-np.inf)
    m.constraint("some", y[T] <= 3.0, where=subset((T,), {"T": np.array([0, 2])}))
    a = m.assemble(singleton_rows="bounds")
    assert a.n_rows == 0
    assert a.col_upper.tolist() == [3.0, INF, 3.0]


def test_two_rows_that_cross_are_passed_as_bounds():
    T = Set("T", np.arange(1))
    m = Model("m")
    x = m.var("x", (T,), lower=-np.inf)
    m.constraint("lo", x[T] >= 2.0)
    m.constraint("up", x[T] <= 1.0)
    a = m.assemble(singleton_rows="bounds")
    assert (a.col_lower.tolist(), a.col_upper.tolist()) == ([2.0], [1.0])


def test_an_unknown_value_raises_at_assemble():
    with pytest.raises(
        ValueError, match="singleton_rows is 'bounds' or 'rows'; got 'bound'"
    ):
        plant().assemble(singleton_rows="bound")


def test_a_row_of_one_coefficient_that_is_not_a_singleton_row_is_loaded():
    T = Set("T", np.arange(3))
    k = Param.from_dense("k", (T,), np.array([1e-21, 1.0, 1.0]))
    m = Model("m")
    x = m.var("x", (T,), lower=-np.inf)
    m.constraint("c", k[T] * x[T] >= 1.0)
    a = m.assemble(singleton_rows="bounds")
    assert (a.n_rows, a.values.tolist()) == (1, [1e-21])
    assert a.col_lower.tolist() == [-INF, 1.0, 1.0]


def test_a_marked_row_whose_coefficient_becomes_zero_raises_at_assembly():
    T = Set("T", np.arange(3))
    k = Param.from_dense("k", (T,), np.array([2.0, 3.0, 4.0]))
    m = Model("m")
    x = m.var("x", (T,), lower=-np.inf)
    m.constraint("c", k[T] * x[T] >= 1.0)
    k.array = Param.from_dense("k", (T,), np.array([0.0, 3.0, 4.0])).array
    with pytest.raises(ValueError, match="marked 3 singleton rows and built 2"):
        m.assemble(singleton_rows="bounds")
