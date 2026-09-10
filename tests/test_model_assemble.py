import numpy as np

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set
from nimopt.term import Sum
from reference import dense_matrix


def transport():
    """A two-plant, three-warehouse transport model."""
    m = Model("transport")
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    x = m.var("x", (P, W))
    one = Param.from_dense("a", (P, W), np.ones((2, 3)))
    cost = Param.from_dense("c", (P, W), np.array([[2.0, 3.0, 1.0], [5.0, 4.0, 8.0]]))
    supply = Param.from_dense("supply", (P,), np.array([30.0, 40.0]))
    demand = Param.from_dense("demand", (W,), np.array([20.0, 25.0, 15.0]))
    m.constraint("supply", Sum(W, one[P, W] * x[P, W]) <= supply[P])
    m.constraint("demand", Sum(P, one[P, W] * x[P, W]) >= demand[W])
    m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))
    return m, P, W


def test_the_assembled_matrix_matches_the_dense_reference():
    m, _, _ = transport()
    got = m.assemble()
    assert np.array_equal(got.to_dense(), dense_matrix(m))


def test_the_assembled_matrix_is_the_labeled_array_the_model_built():
    import nimblend as nb

    from nimopt import COLUMN, ROW

    m, _, _ = transport()
    got = m.assemble()
    assert isinstance(got.matrix, nb.Array)
    assert got.matrix.dims == (ROW, COLUMN)
    # the extents are read from the matrix and are not stored a second time
    assert got.matrix.shape == (got.n_rows, got.n_cols)
    assert np.array_equal(got.to_dense(), got.matrix.to_dense())


def test_the_csr_arrays_and_the_matrix_describe_one_set_of_entries():
    # the model allocates one buffer: the CSR arrays are views into it and
    # the matrix is the same entries under labels, not a second copy
    m, _, _ = transport()
    got = m.assemble()
    assert got.values.size == got.indices.size == got.matrix.nnz
    assert got.values.base is not None
    assert got.indices.base is not None


def test_the_assembled_shape_is_the_declared_one():
    m, _, _ = transport()
    got = m.assemble()
    assert (got.n_rows, got.n_cols) == (5, 6)
    assert got.indices.size == got.values.size == 12
    assert list(got.indptr) == [0, 3, 6, 8, 10, 12]


def test_supply_rows_carry_one_entry_per_warehouse():
    m, _, _ = transport()
    dense = m.assemble().to_dense()
    assert list(dense[0]) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert list(dense[1]) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_demand_rows_carry_one_entry_per_plant():
    m, _, _ = transport()
    dense = m.assemble().to_dense()
    assert list(dense[2]) == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_row_bounds_follow_each_constraint_sense():
    m, _, _ = transport()
    got = m.assemble()
    assert list(got.row_upper[:2]) == [30.0, 40.0]
    assert list(got.row_lower[:2]) == [-np.inf, -np.inf]
    assert list(got.row_lower[2:]) == [20.0, 25.0, 15.0]


def test_the_objective_reaches_every_column():
    m, _, _ = transport()
    assert list(m.assemble().col_cost) == [2.0, 3.0, 1.0, 5.0, 4.0, 8.0]


def test_row_of_names_each_constraints_range():
    m, _, _ = transport()
    got = m.assemble()
    assert got.row_of("supply") == slice(0, 2)
    assert got.row_of("demand") == slice(2, 5)


def test_indices_and_values_are_views_of_one_buffer():
    m, _, _ = transport()
    got = m.assemble()
    assert np.shares_memory(got.indices, got.values) is False
    assert got.indices.dtype == np.int32


def test_assembling_a_model_with_no_constraints_gives_an_empty_matrix():
    m = Model()
    P = Set("P", np.array(["p1"]))
    m.var("x", (P,))
    got = m.assemble()
    assert got.n_rows == 0
    assert got.indices.size == 0
    assert list(got.indptr) == [0]


def test_a_model_with_no_objective_costs_nothing():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    x = m.var("x", (P,))
    one = Param.from_dense("a", (P,), np.ones(2))
    m.constraint("c", one[P] * x[P] <= 1.0)
    assert list(m.assemble().col_cost) == [0.0, 0.0]


def _assembly_peak(n_constraints):
    """Peak bytes of assembling a model with `n_constraints` identical rows."""
    import tracemalloc

    m = Model()
    P = Set("P", np.array([f"p{i}" for i in range(400)]))
    W = Set("W", np.array([f"w{i}" for i in range(200)]))
    x = m.var("x", (P, W))
    one = Param.from_dense("a", (P, W), np.ones((400, 200)))
    for i in range(n_constraints):
        m.constraint(f"c{i}", Sum(W, one[P, W] * x[P, W]) <= 1.0)
    tracemalloc.start()
    got = m.assemble()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    matrix = got.indices.nbytes + got.values.nbytes
    return {"matrix_mb": matrix / 1e6, "excess_mb": (peak - matrix) / 1e6}


def test_assembly_does_not_hold_a_block_per_constraint():
    three = _assembly_peak(3)
    nine = _assembly_peak(9)
    # the matrix triples; the memory beside it is one constraint's expression
    # and does not grow with the number of constraints
    assert nine["matrix_mb"] > three["matrix_mb"] * 2.5, (three, nine)
    assert nine["excess_mb"] < three["excess_mb"] * 1.5, (three, nine)
