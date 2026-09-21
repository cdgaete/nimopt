import nimblend as nb
import numpy as np
import pytest

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set, subset
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
    # both rows are present; each has one coefficient, not two
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


def test_the_terms_are_added_in_one_merge(monkeypatch):
    # a pairwise fold merges the running total once per term
    _, P, W, x, y, d, a = model()
    e = Sum(W, d[P, W] * x[P, W]) + a[P] * y[P] + 3.0 * y[P]
    expected = e.materialise()[0].to_dense().tolist()

    def pairwise(self, other):
        raise AssertionError("the term blocks are added pairwise")

    monkeypatch.setattr(nb.SparseArray, "__add__", pairwise)
    assert e.materialise()[0].to_dense().tolist() == expected


def test_the_terms_at_one_coordinate_are_added_in_one_merge(monkeypatch):
    _, P, W, x, y, d, a = model()
    e = Sum(W, d[P, W] * x[P, W]) + a[P] * y[P]
    expected = e.materialise_at({"P": "p2"}).to_dense().tolist()

    def pairwise(self, other):
        raise AssertionError("the term blocks are added pairwise")

    monkeypatch.setattr(nb.SparseArray, "__add__", pairwise)
    assert e.materialise_at({"P": "p2"}).to_dense().tolist() == expected


def lagged():
    m = Model()
    P = Set("P", np.array(["p1", "p2"]))
    T = Set("T", np.array([2030, 2031, 2032, 2033]))
    return P, T, m.var("x", (P, T))


def summed_terms():
    _, P, W, x, y, d, a = model()
    return Sum(W, d[P, W] * x[P, W]) + a[P] * y[P], subset((P,), {"P": ["p2"]})


def replicated_term():
    _, P, W, x, y, _, _ = model()
    within = subset((P, W), {"P": ["p1", "p2"], "W": ["w1", "w3"]})
    return x[P, W] + y[P], within


def widening_coefficient():
    _, P, W, _, y, d, _ = model()
    within = subset((P, W), {"P": ["p1", "p2"], "W": ["w2", "w1"]})
    return d[P, W] * y[P], within


def lag():
    P, T, x = lagged()
    within = subset((P, T), {"P": ["p1", "p2"], "T": [2031, 2033]})
    return x[P, T] - x[P, T - 1], within


def fixed_member():
    _, P, _, x, y, _, _ = model()
    return x[P, "w2"] + y[P], subset((P,), {"P": ["p1"]})


def kept(block, within):
    inside = block.restrict(within)
    return inside.coordinates().tolist(), inside.values().tolist()


@pytest.mark.parametrize(
    "case",
    [summed_terms, replicated_term, widening_coefficient, lag, fixed_member],
)
def test_a_block_within_rows_has_the_entries_of_the_full_block_there(case):
    expression, within = case()
    full, full_rows = expression.materialise()
    part, part_rows = expression.materialise(within=within)
    assert kept(part, within) == kept(full, within)
    assert (
        part_rows.intersect(within).coordinates().tolist()
        == full_rows.intersect(within).coordinates().tolist()
    )


def test_a_block_within_a_product_of_members_has_no_entry_outside_it():
    expression, within = summed_terms()
    part, _ = expression.materialise(within=within)
    assert part.nnz == expression.materialise()[0].restrict(within).nnz == 4


def test_rows_over_other_dimensions_than_the_frame_raise():
    _, P, W, x, _, d, _ = model()
    with pytest.raises(ValueError, match="pass a domain over the frame"):
        Sum(W, d[P, W] * x[P, W]).materialise(within=subset((W,), {"W": ["w1"]}))
