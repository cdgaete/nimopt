import numpy as np
import pytest

from nimopt import Model, Param, Set, Sum


def storage_shapes():
    """Two units over three periods, with a parameter over both."""
    m = Model("storage")
    S = Set("S", np.array(["s1", "s2"]))
    T = Set("T", np.array(["t0", "t1", "t2"]))
    inflow = Param.from_dense("inflow", (S, T), np.arange(6.0).reshape(2, 3))
    return m, S, T, inflow


def test_a_member_reference_drops_the_dimension_from_the_frame():
    m, S, T, inflow = storage_shapes()
    x = m.var("x", (S, T))
    expression = x[S, "t0"]
    assert expression.frame == ("S",)


def test_a_member_reference_carries_that_members_columns():
    m, S, T, inflow = storage_shapes()
    x = m.var("x", (S, T))
    block, rows = x[S, "t0"].materialise()
    # t0 is the first of three periods, so s1 takes column 0 and s2 column 3
    assert np.array_equal(block.coordinates()[-1], np.array([0, 3]))
    assert rows.size == 2


def test_a_parameter_at_a_member_supplies_the_coefficient():
    m, S, T, inflow = storage_shapes()
    x = m.var("x", (S, T))
    m.constraint("pin", inflow[S, "t1"] * x[S, "t1"] == 0.0)
    assembled = m.assemble()
    # column 1 carries inflow[s1,t1] = 1.0 and column 4 carries inflow[s2,t1] = 4.0
    assert np.array_equal(assembled.values, np.array([1.0, 4.0]))
    assert np.array_equal(assembled.indices, np.array([1, 4], dtype=np.int32))


def test_two_variables_at_the_same_member_share_a_row():
    m, S, T, inflow = storage_shapes()
    level = m.var("level", (S, T))
    release = m.var("release", (S, T))
    m.constraint("boundary", level[S, "t0"] + release[S, "t0"] == 0.0)
    assert m.n_rows == 2
    assert m.nnz == 4


def test_a_member_reference_composes_with_a_sum_over_the_other_dimension():
    m, S, T, inflow = storage_shapes()
    x = m.var("x", (S, T))
    m.constraint("total", Sum(S, x[S, "t2"]) == 0.0)
    assert m.n_rows == 1
    assert m.nnz == 2


def test_a_member_the_set_does_not_carry_is_refused():
    m, S, T, inflow = storage_shapes()
    x = m.var("x", (S, T))
    with pytest.raises(ValueError, match="t9"):
        x[S, "t9"]


def test_a_parameter_at_a_member_the_set_does_not_carry_is_refused():
    m, S, T, inflow = storage_shapes()
    with pytest.raises(ValueError, match="t9"):
        inflow[S, "t9"]


def test_a_reference_of_the_wrong_length_is_still_refused():
    m, S, T, inflow = storage_shapes()
    x = m.var("x", (S, T))
    with pytest.raises(ValueError, match="declared over"):
        x["t0"]


def test_a_parameter_reference_is_a_right_hand_side():
    m, S, T, inflow = storage_shapes()
    x = m.var("x", (S, T))
    m.constraint("row", x[S, T] == inflow[S, T])
    assembled = m.assemble()
    assert m.n_rows == 6
    assert np.array_equal(assembled.row_lower, np.arange(6.0))


def test_a_right_hand_side_at_a_member_is_read_there():
    m, S, T, inflow = storage_shapes()
    x = m.var("x", (S, T))
    m.constraint("row", x[S, "t1"] == inflow[S, "t1"])
    assembled = m.assemble()
    # inflow[s1,t1] = 1.0 and inflow[s2,t1] = 4.0
    assert np.array_equal(assembled.row_lower, np.array([1.0, 4.0]))


def test_a_right_hand_side_over_the_wrong_frame_is_refused():
    m, S, T, inflow = storage_shapes()
    x = m.var("x", (S, T))
    with pytest.raises(ValueError, match="free dimensions"):
        m.constraint("row", x[S, "t1"] == inflow[S, T])
