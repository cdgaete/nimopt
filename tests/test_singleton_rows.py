import numpy as np

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
