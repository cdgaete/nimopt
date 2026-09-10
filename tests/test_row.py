import numpy as np
import pytest

from nimopt import Model, Param, Row, Set
from nimopt.row import resolve
from test_definition import data, dispatch, nodal, nodal_data


def test_a_row_reads_back_the_matrix_row_it_names():
    # read from what was built: a render from a second walk of the
    # expression passes inspection and fails this
    m = dispatch().build(data())
    a = m.assemble()
    row = m.row("balance", snapshot=2)
    dense = a.to_dense()[row.index]
    assert isinstance(row, Row)
    assert row.index == 2
    assert [t.column for t in row.terms] == list(np.nonzero(dense)[0])
    assert [t.coefficient for t in row.terms] == list(dense[np.nonzero(dense)])


def test_a_row_names_its_constraint_and_its_own_coordinate():
    row = dispatch().build(data()).row("balance", snapshot=2)
    assert row.constraint == "balance"
    assert row.coordinate == {"snapshot": 2}


def test_a_rows_terms_name_the_variable_and_the_coordinate_of_each_column():
    row = dispatch().build(data()).row("balance", snapshot=2)
    assert [t.variable for t in row.terms] == ["p", "p", "p"]
    assert [t.coordinate["generator"] for t in row.terms] == ["wind", "solar", "gas"]
    assert {t.coordinate["snapshot"] for t in row.terms} == {2}


def test_a_rows_bounds_and_sense_come_from_the_matrix_it_was_read_from():
    m = dispatch().build(data())
    a = m.assemble()
    row = m.row("balance", snapshot=3)
    assert row.sense == "=="
    assert (row.lower, row.upper) == (a.row_lower[3], a.row_upper[3])
    assert row.lower == 180.0


def test_each_sense_is_read_from_the_bounds_the_row_carries():
    P = Set("P", np.array(["p1"]))
    m = Model("m")
    x = m.var("x", (P,), upper=9.0)
    one = Param.from_dense("one", (P,), np.ones(1))
    m.constraint("le", one[P] * x[P] <= 4.0)
    m.constraint("ge", one[P] * x[P] >= 1.0)
    m.constraint("en", one[P] * x[P] == 2.0)
    assert m.row("le", P="p1").sense == "<="
    assert m.row("ge", P="p1").sense == ">="
    assert m.row("en", P="p1").sense == "=="


def test_a_bound_pair_no_constraint_writes_is_refused_rather_than_named():
    # write_bounds writes -inf, +inf or an equal pair, so a finite unequal
    # pair reaches no row; naming it would be vocabulary nothing produces
    from nimopt.row import _sense_of

    assert _sense_of(-np.inf, 4.0) == "<="
    assert _sense_of(1.0, np.inf) == ">="
    assert _sense_of(2.0, 2.0) == "=="
    with pytest.raises(ValueError, match="no constraint states a range"):
        _sense_of(1.0, 4.0)


def test_a_row_over_several_dimensions_names_every_one_of_them():
    row = nodal().build(nodal_data()).row("balance", B="b1", T=1)
    assert row.coordinate == {"B": "b1", "T": 1}
    assert [t.variable for t in row.terms] == ["flow"]
    assert row.terms[0].coordinate == {"L": "l1", "T": 1}
    assert row.terms[0].coefficient == -1.0


def test_a_row_the_constraint_does_not_state_is_refused():
    # the live rows are stated for the first hour alone, so naming another is
    # a question absent() answers and row() cannot
    m = nodal().build(nodal_data())
    with pytest.raises(ValueError, match="states no row at"):
        m.row("live", B="b0", T=2)


def test_a_coordinate_the_constraint_is_not_free_over_is_refused():
    m = dispatch().build(data())
    with pytest.raises(ValueError, match="is free over"):
        m.row("balance", generator="wind")


def test_a_row_of_a_constraint_that_is_not_declared_is_refused():
    with pytest.raises(KeyError):
        dispatch().build(data()).row("nowhere", snapshot=0)


def test_an_empty_row_stated_by_over_reads_back_with_no_terms():
    # the balance states every bus-hour, and no link touches the third
    row = nodal().build(nodal_data()).row("balance", B="b0", T=2)
    assert row.terms == ()
    assert row.sense == "=="


def test_every_rows_sense_is_read_by_one_rule():
    # the adapters read the whole vector and a Row reads one pair; a second
    # rule for one fact is what disagrees silently
    from nimopt.row import senses

    lower = np.array([-np.inf, 1.0, 2.0])
    upper = np.array([4.0, np.inf, 2.0])
    assert list(senses(lower, upper)) == ["<=", ">=", "=="]


def test_a_range_is_refused_wherever_the_rule_is_read():
    from nimopt.row import senses

    with pytest.raises(ValueError, match="no constraint states a range"):
        senses(np.array([-np.inf, 1.0]), np.array([4.0, 4.0]))


def test_a_column_resolves_to_the_variable_and_coordinate_it_stands_for():
    m = dispatch().build(data())
    resolved = resolve(m, np.array([4, 1]))
    assert [(c, v) for _, c, v, _ in resolved] == [(1, "p"), (4, "p")]
    assert [at["generator"] for *_, at in resolved] == ["solar", "solar"]
    # the position each column stood at in what was given, so a caller can
    # follow it back to a value it was handed beside
    assert [position for position, *_ in resolved] == [1, 0]


def test_a_column_outside_every_variable_is_refused():
    m = dispatch().build(data())
    with pytest.raises(ValueError, match="belongs to no variable"):
        resolve(m, np.array([m.n_columns]))
