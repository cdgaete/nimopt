import numpy as np
import pytest

from nimopt import Absence, Model, Param, Set, Sum, product, subset
from test_definition import nodal, nodal_data


def dropped_by(absence, rule):
    """The coordinates one rule accounts for, as dicts."""
    return [d.coordinate for d in absence.dropped_rows if d.rule == rule]


def test_a_row_no_term_reaches_is_dropped_and_says_so():
    # the incidence has no entry in the third hour; no term covers those
    # bus-hours, and those rows are absent
    B = Set("B", np.array(["b0", "b1"]))
    T = Set("T", np.array([0, 1, 2]))
    L = Set("L", np.array(["l0", "l1"]))
    m = Model("n")
    flow = m.var("flow", (L, T), lower=-1.0)
    inc = Param.from_long(
        "inc",
        (B, L, T),
        {
            "B": np.array(["b0", "b1", "b0", "b1"]),
            "L": np.array(["l0", "l0", "l1", "l1"]),
            "T": np.array([0, 0, 1, 1]),
        },
        np.array([1.0, -1.0, 1.0, -1.0]),
    )
    zero = Param.from_dense("zero", (B, T), np.zeros((2, 3)))
    m.constraint("balance", Sum(L, inc[B, L, T] * flow[L, T]) == zero[B, T])
    a = m.absent("balance")
    assert isinstance(a, Absence)
    assert a.stated_by == "terms"
    assert (a.expected, a.standing) == (6, 4)
    assert dropped_by(a, "term-does-not-reach") == [
        {"B": "b0", "T": 2},
        {"B": "b1", "T": 2},
    ]
    assert {d.detail for d in a.dropped_rows} == {"flow"}


def test_a_row_a_condition_omits_is_dropped_and_says_so():
    P = Set("P", np.array(["p1", "p2", "p3"]))
    m = Model("m")
    x = m.var("x", (P,), upper=5.0)
    one = Param.from_dense("one", (P,), np.ones(3))
    m.constraint(
        "cap", one[P] * x[P] <= 1.0, where=subset((P,), {"P": np.array(["p1", "p3"])})
    )
    a = m.absent("cap")
    assert (a.expected, a.standing) == (3, 2)
    assert dropped_by(a, "where") == [{"P": "p2"}]


def test_a_row_the_right_hand_side_misses_is_dropped_and_says_so():
    P = Set("P", np.array(["p1", "p2", "p3"]))
    m = Model("m")
    x = m.var("x", (P,), upper=5.0)
    one = Param.from_dense("one", (P,), np.ones(3))
    rhs = Param.from_long("rhs", (P,), {"P": np.array(["p1", "p2"])}, np.ones(2))
    m.constraint("cap", one[P] * x[P] <= rhs[P])
    a = m.absent("cap")
    assert (a.expected, a.standing) == (3, 2)
    assert dropped_by(a, "absent-rhs") == [{"P": "p3"}]
    assert [d.detail for d in a.dropped_rows] == ["rhs"]


def test_a_coefficient_absent_inside_a_sum_drops_a_term_not_a_row():
    # the split agents get wrong: the row stands, one of its terms is gone
    P = Set("P", np.array(["p1", "p2"]))
    W = Set("W", np.array(["w1", "w2", "w3"]))
    m = Model("t")
    flow = m.var("flow", (P, W))
    cost = Param.from_long(
        "cost",
        (P, W),
        {"P": np.array(["p1", "p1", "p2"]), "W": np.array(["w1", "w2", "w1"])},
        np.array([1.0, 2.0, 3.0]),
    )
    supply = Param.from_dense("supply", (P,), np.array([3.0, 3.0]))
    m.constraint("supply", Sum(W, cost[P, W] * flow[P, W]) <= supply[P])
    a = m.absent("supply")
    assert (a.expected, a.standing) == (2, 2)
    assert a.dropped_rows == ()
    assert [(d.coordinate, d.variable, d.rule, d.detail) for d in a.dropped_terms] == [
        ({"P": "p1", "W": "w3"}, "flow", "absent-coefficient", "cost"),
        ({"P": "p2", "W": "w2"}, "flow", "absent-coefficient", "cost"),
        ({"P": "p2", "W": "w3"}, "flow", "absent-coefficient", "cost"),
    ]


def test_rows_stated_outright_drop_nothing_and_say_so():
    # under over= the rows are declared, an empty dropped_rows is structural,
    # and stated_by reports which branch produced the rows
    a = nodal().build(nodal_data()).absent("live")
    assert a.stated_by == "over"
    assert a.dropped_rows == ()
    assert a.expected == a.standing == 2


def test_a_row_stated_outright_still_reports_the_terms_that_are_missing():
    # link l1 touches no bus in the first hour, so both stated rows stand
    # with one term each rather than two
    a = nodal().build(nodal_data()).absent("live")
    assert [(d.coordinate, d.detail) for d in a.dropped_terms] == [
        ({"B": "b0", "L": "l1", "T": 0}, "inc"),
        ({"B": "b1", "L": "l1", "T": 0}, "inc"),
    ]


def test_every_dropped_row_is_accounted_for_by_exactly_one_rule():
    # the arithmetic an agent relies on: what was expected, less what fell,
    # is what stands
    P = Set("P", np.array(["p1", "p2", "p3", "p4"]))
    m = Model("m")
    x = m.var("x", (P,), upper=5.0)
    one = Param.from_dense("one", (P,), np.ones(4))
    rhs = Param.from_long("rhs", (P,), {"P": np.array(["p1", "p2", "p3"])}, np.ones(3))
    m.constraint(
        "cap",
        one[P] * x[P] <= rhs[P],
        where=subset((P,), {"P": np.array(["p1", "p3", "p4"])}),
    )
    a = m.absent("cap")
    assert a.expected - len(a.dropped_rows) == a.standing
    assert (a.expected, a.standing) == (4, 2)
    # p2 falls to the condition, p4 to the right-hand side; each named once
    assert dropped_by(a, "where") == [{"P": "p2"}]
    assert dropped_by(a, "absent-rhs") == [{"P": "p4"}]


def test_a_constraint_that_drops_nothing_says_so_without_an_empty_rule():
    P = Set("P", np.array(["p1", "p2"]))
    m = Model("m")
    x = m.var("x", (P,), upper=5.0)
    one = Param.from_dense("one", (P,), np.ones(2))
    m.constraint("cap", one[P] * x[P] <= 1.0)
    a = m.absent("cap")
    assert (a.expected, a.standing) == (2, 2)
    assert a.dropped_rows == ()
    assert a.dropped_terms == ()


def test_asking_a_constraint_that_is_not_declared_is_refused():
    with pytest.raises(KeyError, match="has no constraint 'nowhere'; use one of"):
        nodal().build(nodal_data()).absent("nowhere")


def test_recording_leaves_the_constraint_it_re_ran_unchanged():
    # absent() re-runs the shape pass; a re-run that moved the model would
    # answer for a matrix nobody assembled
    m = nodal().build(nodal_data())
    before = (m.n_rows, m.nnz, m.constraints["live"].n_rows)
    m.absent("live")
    assert (m.n_rows, m.nnz, m.constraints["live"].n_rows) == before


ROW_RULES = ("term-does-not-reach", "where", "absent-rhs")
TERM_RULES = ("absent-coefficient",)


def every_absence():
    """One absence per fixture below, covering both stated_by branches."""
    P = Set("P", np.array(["p1", "p2", "p3"]))
    one = Param.from_dense("one", (P,), np.ones(3))

    absent_term = Model("absent_term")
    y = absent_term.var("y", (P,), upper=5.0)
    sparse = Param.from_long("sparse", (P,), {"P": np.array(["p1", "p2"])}, np.ones(2))
    absent_term.constraint("rows", sparse[P] * y[P] <= 1.0)

    condition = Model("condition")
    c = condition.var("x", (P,), upper=5.0)
    condition.constraint(
        "rows", one[P] * c[P] <= 1.0, where=subset((P,), {"P": np.array(["p1"])})
    )

    rhs = Model("rhs")
    r = rhs.var("x", (P,), upper=5.0)
    cap = Param.from_long("cap", (P,), {"P": np.array(["p1"])}, np.ones(1))
    rhs.constraint("rows", one[P] * r[P] <= cap[P])

    stated = Model("stated")
    s = stated.var("x", (P,), upper=5.0)
    stated.constraint("rows", one[P] * s[P] <= 1.0, over=product((P,)))

    W = Set("W", np.array(["w1", "w2"]))
    summed = Model("summed")
    f = summed.var("f", (P, W))
    arcs = Param.from_long(
        "arcs",
        (P, W),
        {"P": np.array(["p1", "p2", "p3"]), "W": np.array(["w1", "w1", "w1"])},
        np.ones(3),
    )
    summed.constraint("rows", Sum(W, arcs[P, W] * f[P, W]) <= 1.0)

    return [m.absent("rows") for m in (absent_term, condition, rhs, stated, summed)]


def test_every_absence_rule_is_produced_by_some_model():
    # a rule no model produces is dead vocabulary an agent branches on and
    # never runs
    found = every_absence()
    rows = {d.rule for a in found for d in a.dropped_rows}
    terms = {d.rule for a in found for d in a.dropped_terms}
    assert rows == set(ROW_RULES)
    assert terms == set(TERM_RULES)


def test_both_ways_a_constraint_states_its_rows_are_produced():
    assert {a.stated_by for a in every_absence()} == {"terms", "over"}


def test_no_absence_names_a_rule_outside_the_vocabulary():
    # the closed set: a rule the documentation does not carry is one an agent
    # meets with no way to read it
    for a in every_absence():
        assert all(d.rule in ROW_RULES for d in a.dropped_rows)
        assert all(d.rule in TERM_RULES for d in a.dropped_terms)


def test_the_arithmetic_holds_for_every_model_in_the_corpus():
    for a in every_absence():
        assert a.expected - len(a.dropped_rows) == a.standing
