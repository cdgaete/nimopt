import numpy as np
import pytest

from nimopt import Param, Sum, subset
from nimopt.assembly import Appended
from nimopt.constraint import Constraint
from nimopt.solution import reduced_costs
from session_models import two

m, i, x = two()


def two_columns():
    return m


def cut():
    return Constraint("cut", Sum(i, x[i]) >= 7.0)


def test_an_appended_constraint_returns_its_own_csr_block():
    appended = Appended(n_columns=2, first_row=2)
    lower, upper, indptr, indices, values = appended.add("cut", cut())
    assert (lower.tolist(), upper.tolist()) == ([7.0], [np.inf])
    assert (indptr.tolist(), indices.tolist(), values.tolist()) == (
        [0, 2],
        [0, 1],
        [1.0, 1.0],
    )
    assert appended.n_rows == 1
    assert appended.row_of("cut") == slice(2, 3)


def test_a_row_is_read_by_its_index_in_the_loaded_model():
    appended = Appended(n_columns=2, first_row=2)
    appended.add("cut", cut())
    name, position, columns, values, lower, upper = appended.row(2)
    assert (name, position, columns.tolist(), values.tolist(), lower, upper) == (
        "cut",
        0,
        [0, 1],
        [1.0, 1.0],
        7.0,
        np.inf,
    )
    with pytest.raises(ValueError, match="row 3"):
        appended.row(3)


def test_the_reduced_costs_include_the_appended_rows():
    # cost 1 per column; duals 0, 0 for the base rows and 1 for the cut
    assembled = two_columns().assemble()
    appended = Appended(n_columns=2, first_row=2)
    appended.add("cut", cut())
    assert reduced_costs(assembled, np.array([0.0, 0.0, 1.0]), appended).tolist() == [
        0.0,
        0.0,
    ]


def test_rows_appended_across_several_growths_are_all_read():
    appended = Appended(n_columns=2, first_row=2)
    for at in range(50):
        appended.add(f"cut{at}", Constraint(f"cut{at}", Sum(i, x[i]) >= float(at)))
    lower, _, indptr, _, _ = appended.csr()
    assert lower.tolist() == [float(at) for at in range(50)]
    assert indptr.tolist() == list(range(0, 101, 2))
    assert appended.row_of("cut0") == slice(2, 3)
    assert appended.row_of("cut49") == slice(51, 52)
    assert appended.row(51)[0] == "cut49"


def test_a_constraint_over_a_set_appends_one_row_per_member():
    appended = Appended(n_columns=2, first_row=2)
    lower, _, indptr, _, _ = appended.add("each", Constraint("each", x[i] >= 1.0))
    assert appended.n_rows == 2
    assert indptr.tolist() == [0, 1, 2]


def test_a_name_appended_twice_raises():
    appended = Appended(n_columns=2, first_row=2)
    appended.add("cut", cut())
    with pytest.raises(ValueError, match="cut"):
        appended.add("cut", cut())


def test_the_arrays_of_an_append_are_unchanged_by_later_appends():
    appended = Appended(n_columns=2, first_row=2)
    _, _, _, indices, values = appended.add("cut", cut())
    for at in range(50):
        appended.add(f"cut{at}", Constraint(f"cut{at}", 2.0 * x["a"] >= float(at)))
    assert (indices.tolist(), values.tolist()) == ([0, 1], [1.0, 1.0])


def test_the_whole_block_is_returned_with_its_columns_and_values():
    appended = Appended(n_columns=2, first_row=2)
    appended.add("cut", cut())
    appended.add("double_a", Constraint("double_a", 2.0 * x["a"] <= 9.0))
    lower, upper, indptr, indices, values = appended.csr()
    assert (lower.tolist(), upper.tolist()) == ([7.0, -np.inf], [np.inf, 9.0])
    assert indptr.tolist() == [0, 2, 3]
    assert (indices.tolist(), values.tolist()) == ([0, 1, 0], [1.0, 1.0, 2.0])
    assert appended.names == ("cut", "double_a")
    assert appended.n_rows == 2


def test_no_appended_row_gives_an_empty_block_and_no_weighted_sum():
    appended = Appended(n_columns=2, first_row=2)
    lower, upper, indptr, indices, values = appended.csr()
    assert (lower.size, upper.size, indices.size, values.size) == (0, 0, 0, 0)
    assert indptr.tolist() == [0]
    assert appended.weighted(np.empty(0)) is None
    assembled = two_columns().assemble()
    # cost 1 per column and duals of zero: the reduced costs are the costs
    assert reduced_costs(assembled, np.zeros(2), appended).tolist() == [1.0, 1.0]


def test_the_weighted_sum_is_over_the_columns_of_the_model():
    appended = Appended(n_columns=2, first_row=2)
    appended.add("cut", cut())
    appended.add("double_a", Constraint("double_a", 2.0 * x["a"] <= 9.0))
    # 3 * (1, 1) + 5 * (2, 0)
    assert appended.weighted(np.array([3.0, 5.0])).to_dense().tolist() == [13.0, 3.0]


def test_duals_of_another_length_than_the_rows_raise():
    assembled = two_columns().assemble()
    appended = Appended(n_columns=2, first_row=2)
    appended.add("cut", cut())
    with pytest.raises(ValueError, match="row_dual"):
        reduced_costs(assembled, np.zeros(2), appended)


def gap():
    """Return a constraint of two rows whose row at a has no coefficient."""
    only_b = Param.from_long("c", (i,), {"i": ["b"]}, [2.0])
    return Constraint(
        "gap", only_b[i] * x[i] >= 1.0, over=subset((i,), {"i": ["a", "b"]})
    )


def test_a_row_with_no_coefficient_keeps_its_place_in_the_block():
    appended = Appended(n_columns=2, first_row=2)
    appended.add("cut", cut())
    _, _, indptr, indices, values = appended.add("gap", gap())
    assert (indptr.tolist(), indices.tolist(), values.tolist()) == (
        [0, 0, 1],
        [1],
        [2.0],
    )
    appended.add("double_a", Constraint("double_a", 2.0 * x["a"] <= 9.0))
    _, _, indptr, indices, values = appended.csr()
    # rows: cut, gap at a with no coefficient, gap at b, double_a
    assert indptr.tolist() == [0, 2, 2, 3, 4]
    assert (indices.tolist(), values.tolist()) == ([0, 1, 1, 0], [1.0, 1.0, 2.0, 2.0])
    # column 0: 1 * 1 + 1000 * 2; column 1: 1 * 1 + 100 * 2
    weighted = appended.weighted(np.array([1.0, 10.0, 100.0, 1000.0]))
    assert weighted.to_dense().tolist() == [2001.0, 201.0]
    assert appended.row_of("gap") == slice(3, 5)
    name, position, columns, values, lower, _ = appended.row(3)
    assert (name, position, columns.tolist(), values.tolist(), lower) == (
        "gap",
        0,
        [],
        [],
        1.0,
    )
    assert appended.row(4)[:2] == ("gap", 1)
    assert appended.row(4)[2].tolist() == [1]


def test_a_constraint_with_no_coefficient_appends_its_rows():
    only_b = Param.from_long("c", (i,), {"i": ["b"]}, [2.0])
    none = Constraint("none", only_b[i] * x[i] >= 1.0, over=subset((i,), {"i": ["a"]}))
    appended = Appended(n_columns=2, first_row=2)
    lower, upper, indptr, indices, values = appended.add("none", none)
    assert (lower.tolist(), upper.tolist()) == ([1.0], [np.inf])
    assert (indptr.tolist(), indices.size, values.size) == ([0, 0], 0, 0)
    appended.add("cut", cut())
    _, _, indptr, indices, _ = appended.csr()
    assert indptr.tolist() == [0, 0, 2]
    assert indices.tolist() == [0, 1]
    assert appended.weighted(np.array([5.0, 3.0])).to_dense().tolist() == [3.0, 3.0]


def test_a_solution_reads_reduced_costs_through_the_appended_rows():
    from nimopt.solution import Solution

    model = two_columns()
    assembled = model.assemble()
    appended = Appended(n_columns=2, first_row=2)
    appended.add("cut", cut())
    solution = Solution(
        model,
        "optimal",
        True,
        7.0,
        7.0,
        np.array([6.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        assembled,
        {name: assembled.row_of(name) for name in model.constraints},
        "highs",
        1,
        appended,
    )
    assert solution.dual("x").to_dense().tolist() == [0.0, 0.0]


def test_the_accept_call_reads_the_arrays_before_the_rows_are_recorded():
    appended = Appended(n_columns=2, first_row=2)
    seen = []

    def accept(lower, upper, indptr, indices, values):
        seen.append((appended.n_rows, lower.tolist(), values.tolist()))

    appended.add("cut", cut(), accept=accept)
    assert seen == [(0, [7.0], [1.0, 1.0])]
    assert appended.n_rows == 1


def test_an_accept_call_that_raises_records_no_row():
    appended = Appended(n_columns=2, first_row=2)

    def reject(*arrays):
        raise RuntimeError("rows rejected")

    with pytest.raises(RuntimeError, match="rows rejected"):
        appended.add("cut", cut(), accept=reject)
    assert (appended.n_rows, appended.names) == (0, ())
    assert appended.csr()[2].tolist() == [0]
    appended.add("cut", cut())
    assert appended.row_of("cut") == slice(2, 3)


def test_a_row_bound_that_is_not_a_number_raises():
    appended = Appended(n_columns=2, first_row=2)
    nan = Constraint("nan", Sum(i, x[i]) >= float("nan"))
    with pytest.raises(ValueError, match="not a number"):
        appended.add("nan", nan)
    assert appended.n_rows == 0


def test_a_coefficient_that_is_not_finite_raises():
    huge = Param.from_dense("huge", (i,), np.array([np.inf, 1.0]))
    appended = Appended(n_columns=2, first_row=2)
    with pytest.raises(ValueError, match="not finite"):
        appended.add("bad", Constraint("bad", Sum(i, huge[i] * x[i]) >= 1.0))
    assert appended.n_rows == 0
