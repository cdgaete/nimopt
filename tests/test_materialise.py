import numpy as np

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set
from nimopt.term import Sum


def model():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    x = m.var("x", (P, W))
    y = m.var("y", (P,))
    d = Param.from_dense("d", (P, W), np.arange(1, 7, dtype=float).reshape(2, 3))
    a = Param.from_dense("a", (P,), np.array([10.0, 20.0]))
    return m, P, W, x, y, d, a


def test_a_summed_term_carries_one_row_per_free_coordinate():
    _, P, W, x, _, d, _ = model()
    block, rows = Sum(W, d[P, W] * x[P, W]).materialise()
    assert block.dims == ("P", "__column__")
    assert block.nnz == 6
    assert rows.size == 2


def test_two_terms_over_one_frame_land_in_one_block():
    _, P, W, x, y, d, a = model()
    e = Sum(W, d[P, W] * x[P, W]) + a[P] * y[P]
    block, rows = e.materialise()
    assert block.nnz == 8
    assert rows.size == 2
    assert block.to_dense().tolist() == [
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 20.0],
    ]


def test_a_term_narrower_than_the_frame_is_replicated_across_it():
    # x[P,W] + y[P]: y's column stands on every row sharing its P
    _, P, W, x, y, _, _ = model()
    block, rows = (x[P, W] + y[P]).materialise()
    assert block.dims == ("P", "W", "__column__")
    assert rows.size == 6
    assert block.nnz == 12


def test_a_scale_multiplies_the_coefficients():
    _, P, W, x, _, d, _ = model()
    block, _ = Sum(W, 2.0 * (d[P, W] * x[P, W])).materialise()
    assert block.to_dense()[0].tolist() == [2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_a_term_with_no_coefficient_carries_ones():
    _, P, W, x, _, _, _ = model()
    block, _ = Sum(W, x[P, W]).materialise()
    assert block.to_dense()[0].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_a_coefficient_absent_inside_a_sum_drops_the_term_not_the_row():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2"]))
    x = m.var("x", (P, W))
    sparse = Param.from_long(
        "d",
        (P, W),
        {"P": np.array(["p1", "p2"]), "W": np.array(["w1", "w1"])},
        np.array([1.0, 2.0]),
    )
    block, rows = Sum(W, sparse[P, W] * x[P, W]).materialise()
    # both rows stand; each carries one coefficient rather than two
    assert rows.size == 2
    assert block.nnz == 2


def test_a_row_no_term_reaches_is_not_stated():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    x = m.var("x", (P,))
    only_first = Param.from_long("a", (P,), {"P": np.array(["p1"])}, np.array([5.0]))
    _, rows = (only_first[P] * x[P]).materialise()
    assert rows.size == 1
    assert list(rows.labels()["P"]) == ["p1"]


def test_the_row_domain_is_the_intersection_of_the_terms_domains():
    m = Model()
    P = Set("P", np.array(["p1", "p2", "p3"]))
    x = m.var("x", (P,))
    y = m.var("y", (P,))
    first = Param.from_long(
        "a", (P,), {"P": np.array(["p1", "p2"])}, np.array([1.0, 1.0])
    )
    second = Param.from_long(
        "b", (P,), {"P": np.array(["p2", "p3"])}, np.array([1.0, 1.0])
    )
    _, rows = (first[P] * x[P] + second[P] * y[P]).materialise()
    assert list(rows.labels()["P"]) == ["p2"]


def test_materialising_twice_answers_the_same_block():
    _, P, W, x, _, d, _ = model()
    e = Sum(W, d[P, W] * x[P, W])
    first, _ = e.materialise()
    second, _ = e.materialise()
    assert first.to_dense().tolist() == second.to_dense().tolist()
