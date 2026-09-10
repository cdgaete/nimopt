import nimblend as nb
import numpy as np
import pytest

from nimopt.names import COLUMN
from nimopt.param import Param
from nimopt.sets import Set, subset
from nimopt.variable import Variable


def sets_2x3():
    return Set("P", np.array(["p1", "p2"])), Set("W", np.array(["w1", "w2", "w3"]))


def test_a_dense_variable_spans_the_full_product():
    P, W = sets_2x3()
    x = Variable("x", (P, W), start=0, total_columns=6)
    assert x.n_columns == 6
    assert x.dims == ("P", "W")


def test_terms_place_one_entry_at_each_member_column():
    P, W = sets_2x3()
    x = Variable("x", (P, W), start=0, total_columns=6)
    arr = x.terms()
    assert arr.dims == ("P", "W", COLUMN)
    assert arr.nnz == 6
    assert list(arr.values()) == [1.0] * 6
    # C order over (p, w): column = p * |W| + w
    at = arr.coordinates()
    assert list(at[0]) == [0, 0, 0, 1, 1, 1]
    assert list(at[1]) == [0, 1, 2, 0, 1, 2]
    assert list(at[2]) == [0, 1, 2, 3, 4, 5]


def test_a_start_offset_shifts_every_column():
    P, W = sets_2x3()
    y = Variable("y", (P, W), start=6, total_columns=12)
    assert list(y.terms().coordinates()[2]) == [6, 7, 8, 9, 10, 11]


def test_a_subset_variable_carries_only_its_members():
    P, W = sets_2x3()
    arcs = subset((P, W), {"P": np.array(["p1", "p2"]), "W": np.array(["w3", "w1"])})
    z = Variable("z", (P, W), start=0, total_columns=2, subset=arcs)
    arr = z.terms()
    assert z.n_columns == 2
    assert arr.nnz == 2
    at = arr.coordinates()
    assert list(at[0]) == [0, 1]
    assert list(at[1]) == [2, 0]
    assert list(at[2]) == [0, 1]


def test_terms_are_canonical():
    P, W = sets_2x3()
    arr = Variable("x", (P, W), start=0, total_columns=6).terms()
    assert nb.is_canonical(arr.coordinates(), arr.shape)


def test_terms_declare_empty_absence():
    P, W = sets_2x3()
    assert Variable("x", (P, W), start=0, total_columns=6).terms().absence == "empty"


def test_getitem_checks_the_sets_it_is_given():
    P, W = sets_2x3()
    x = Variable("x", (P, W), start=0, total_columns=6)
    assert len(x[P, W].terms) == 1
    with pytest.raises(ValueError, match="declared over"):
        x[W, P]


def test_a_subset_whose_sets_differ_from_the_variable_raises():
    P, W = sets_2x3()
    other = Set("Z", np.array(["z1"]))
    arcs = subset((P, other), {"P": np.array(["p1"]), "Z": np.array(["z1"])})
    with pytest.raises(ValueError, match="its members span"):
        Variable("z", (P, W), start=0, total_columns=1, subset=arcs)


def test_bounds_and_integrality_are_carried():
    P, W = sets_2x3()
    b = Variable(
        "b", (P, W), start=0, total_columns=6, lower=0.0, upper=1.0, integer=True
    )
    assert (b.lower, b.upper, b.integer) == (0.0, 1.0, True)


def test_a_variable_declares_over_declared_sets():
    S, G = Set("S"), Set("G")
    v = Variable("x", (S, G))
    assert v.declared
    # dims are the set names; they are readable with nothing bound
    assert v.dims == ("S", "G")


def test_a_declared_variable_refuses_to_report_its_columns():
    with pytest.raises(ValueError, match="variable 'x' is declared and has no columns"):
        Variable("x", (Set("S"),)).n_columns


def test_binding_builds_the_column_rule_from_the_bound_sets():
    S, G = Set("S"), Set("G")
    v = Variable("x", (S, G))
    S._bind(np.arange(4))
    G._bind(np.array(["a", "b", "c"]))
    v._bind(start=10, total_columns=22)
    assert not v.declared
    assert v.n_columns == 12
    assert v.start == 10


def test_a_bound_variable_still_names_the_parameter_it_took_its_members_from():
    # binding resolves the members and keeps the parameter it read them from;
    # an explanation reports the parameter and not the domain
    P, W = Set("P"), Set("W")
    cost = Param("cost", (P, W))
    v = Variable("flow", (P, W), subset=cost)
    P._bind(np.array(["p1"]))
    W._bind(np.array(["w1"]))
    cost._bind(Param.from_dense("cost", (P, W), np.ones((1, 1))).array)
    v._bind(start=0, total_columns=1)
    assert v.subset is cost


def test_a_subset_named_as_a_parameter_becomes_its_members_when_bound():
    # the arcs are the parameter's coefficients; a sparse variable takes one
    # column per arc and not one per cell of the product
    P, W = Set("P"), Set("W")
    cost = Param("cost", (P, W))
    v = Variable("flow", (P, W), subset=cost)
    P._bind(np.array(["p1", "p2", "p3"]))
    W._bind(np.array(["w1", "w2"]))
    columns = {"P": np.array(["p1", "p1", "p2"]), "W": np.array(["w1", "w2", "w1"])}
    cost._bind(Param.from_long("cost", (P, W), columns, np.ones(3)).array)
    v._bind(start=0, total_columns=3)
    assert v.n_columns == 3
    assert v.domain().size == 3


def test_a_subset_named_as_a_parameter_that_carries_nothing_yet_is_refused():
    P, W = Set("P"), Set("W")
    cost = Param("cost", (P, W))
    v = Variable("flow", (P, W), subset=cost)
    P._bind(np.array(["p1"]))
    W._bind(np.array(["w1"]))
    with pytest.raises(ValueError, match="has no values"):
        v._bind(start=0, total_columns=1)
