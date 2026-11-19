import numpy as np
import pytest

from nimopt.solvers import highs
from session_models import two

pytest.importorskip("highspy")


def two_columns():
    return two()[0].assemble(singleton_rows="rows")


def test_a_changed_bound_is_solved_on_the_loaded_model():
    first = highs.solve(two_columns(), "min")
    assert first.objective == pytest.approx(4.0)
    highs.change_column_bounds(
        first.backend, np.array([0], dtype=np.int32), np.array([5.0]), np.array([5.0])
    )
    again = highs.resume(first.backend)
    assert again.status == "optimal"
    assert again.objective == pytest.approx(6.0)
    assert again.backend is first.backend


def test_an_appended_row_is_solved_on_the_loaded_model():
    first = highs.solve(two_columns(), "min")
    highs.append_rows(
        first.backend,
        np.array([7.0]),
        np.array([np.inf]),
        np.array([0, 2], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.array([1.0, 1.0]),
    )
    again = highs.resume(first.backend)
    assert again.objective == pytest.approx(7.0)
    assert again.row_dual.tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_a_resumed_solve_takes_fewer_iterations_than_the_first():
    from nimopt.models import transport

    model = transport.definition().build(transport.data(10))
    first = highs.solve(model.assemble(), "min")
    highs.change_column_bounds(
        first.backend, np.array([0], dtype=np.int32), np.array([0.0]), np.array([0.0])
    )
    assert highs.resume(first.backend).iterations < first.iterations


def test_a_column_outside_the_loaded_model_raises():
    first = highs.solve(two_columns(), "min")
    with pytest.raises(RuntimeError, match="HiGHS rejected"):
        highs.change_column_bounds(
            first.backend,
            np.array([9], dtype=np.int32),
            np.array([0.0]),
            np.array([1.0]),
        )


def test_an_appended_row_over_a_column_outside_the_loaded_model_raises():
    first = highs.solve(two_columns(), "min")
    with pytest.raises(RuntimeError, match="HiGHS rejected"):
        highs.append_rows(
            first.backend,
            np.array([7.0]),
            np.array([np.inf]),
            np.array([0, 2], dtype=np.int32),
            np.array([0, 5], dtype=np.int32),
            np.array([1.0, 1.0]),
        )


def test_bounds_of_another_length_than_the_columns_raise():
    first = highs.solve(two_columns(), "min")
    with pytest.raises(RuntimeError, match="length"):
        highs.change_column_bounds(
            first.backend,
            np.array([0, 1], dtype=np.int32),
            np.array([0.0]),
            np.array([1.0, 1.0]),
        )


def test_appended_rows_of_inconsistent_lengths_raise():
    first = highs.solve(two_columns(), "min")
    with pytest.raises(RuntimeError, match="length"):
        highs.append_rows(
            first.backend,
            np.array([7.0]),
            np.array([np.inf, np.inf]),
            np.array([0, 2], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
            np.array([1.0, 1.0]),
        )


def test_a_resumed_solve_of_a_model_with_integer_columns_reports_no_duals():
    from nimopt.models import commitment

    model = commitment.definition().build(commitment.data(3))
    first = highs.solve(model.assemble(), model.sense)
    again = highs.resume(first.backend)
    assert again.row_dual is None
    assert again.objective == pytest.approx(first.objective)
