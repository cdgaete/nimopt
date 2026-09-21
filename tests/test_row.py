import importlib

import nimblend as nb
import numpy as np
import pytest

from nimopt import Expression, Model, Param, Row, Set, Sum
from nimopt.models import MODELS
from nimopt.row import read, resolve
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
    # write_bounds writes -inf, +inf or an equal pair; a finite unequal pair
    # belongs to no row the builder writes
    from nimopt.row import _sense_of

    assert _sense_of(-np.inf, 4.0) == "<="
    assert _sense_of(1.0, np.inf) == ">="
    assert _sense_of(2.0, 2.0) == "=="
    with pytest.raises(ValueError, match="are a range"):
        _sense_of(1.0, 4.0)


def test_a_row_over_several_dimensions_names_every_one_of_them():
    row = nodal().build(nodal_data()).row("balance", B="b1", T=1)
    assert row.coordinate == {"B": "b1", "T": 1}
    assert [t.variable for t in row.terms] == ["flow"]
    assert row.terms[0].coordinate == {"L": "l1", "T": 1}
    assert row.terms[0].coefficient == -1.0


def test_a_row_the_constraint_does_not_contain_raises():
    # the live rows cover the first hour alone; absent() reports the rule
    # that dropped any other row, and row() does not
    m = nodal().build(nodal_data())
    with pytest.raises(ValueError, match="has no row at"):
        m.row("live", B="b0", T=2)


def test_a_coordinate_the_constraint_is_not_free_over_is_refused():
    m = dispatch().build(data())
    with pytest.raises(ValueError, match="is free over"):
        m.row("balance", generator="wind")


def test_a_row_of_a_constraint_that_is_not_declared_is_refused():
    with pytest.raises(KeyError, match="has no constraint 'nowhere'; use one of"):
        dispatch().build(data()).row("nowhere", snapshot=0)


def test_an_empty_row_stated_by_over_reads_back_with_no_terms():
    # the balance covers every bus-hour, and no link touches the third
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

    with pytest.raises(ValueError, match="are a range"):
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


def test_every_row_equals_the_row_of_the_assembled_matrix():
    # the second constraint's rows are numbered after the first constraint's
    m = nodal().build(nodal_data())
    a = m.assemble()
    assert list(m.constraints) == ["balance", "live"]
    for index in range(a.n_rows):
        expected = read(m, a, index)
        assert m.row(expected.constraint, **expected.coordinate) == expected


def test_a_row_decodes_the_labels_of_that_row_alone(monkeypatch):
    # the constraint has 6 rows; decoding every row reads all 6, and each
    # decode here reads the one row
    m = nodal().build(nodal_data())
    rows = m.constraints["balance"].rows
    decoded = []
    labels = nb.Domain.labels

    def counted(self, positions=None):
        if self is rows:
            decoded.append(None if positions is None else len(positions))
        return labels(self, positions)

    monkeypatch.setattr(nb.Domain, "labels", counted)
    m.row("balance", B="b1", T=2)
    assert decoded
    assert set(decoded) == {1}


def rows_equal_their_assembled_rows(m):
    # a dozen rows of each constraint, spread over its range
    a = m.assemble()
    for name in m.constraints:
        at = a.row_of(name)
        if at.stop == at.start:
            continue
        for index in np.unique(np.linspace(at.start, at.stop - 1, 12).astype(int)):
            expected = read(m, a, int(index))
            assert m.row(name, **expected.coordinate) == expected


@pytest.mark.parametrize("name", MODELS)
def test_a_row_of_every_corpus_constraint_equals_its_assembled_row(name):
    held = importlib.import_module(f"nimopt.models.{name}")
    rows_equal_their_assembled_rows(held.definition().build(held.data()))


def test_a_row_of_every_piecewise_constraint_equals_its_assembled_row():
    # the generated rows read the segment after the current one
    from test_piecewise import cost_model

    rows_equal_their_assembled_rows(cost_model())


def test_a_row_of_a_sum_over_a_free_dimension_sums_every_member():
    # the second term sums G, over which the first term is free
    G = Set("G", np.array(["a", "b", "c"]))
    T = Set("T", np.array([0, 1]))
    m = Model("m")
    p = m.var("p", (G, T))
    m.constraint("share", p[G, T] - 0.5 * Sum(G, p[G, T]) <= 0.0)
    row = m.row("share", G="b", T=1)
    assert [t.coefficient for t in row.terms] == [-0.5, 0.5, -0.5]
    rows_equal_their_assembled_rows(m)


def test_a_row_materialises_no_whole_expression(monkeypatch):
    # the row is computed at its coordinate
    m = nodal().build(nodal_data())

    def whole(self, *args, **kwargs):
        raise AssertionError("Model.row materialized a whole expression")

    monkeypatch.setattr(Expression, "materialise", whole)
    assert m.row("balance", B="b1", T=2).constraint == "balance"


def test_row_at_returns_the_columns_coefficients_and_bounds_of_a_row():
    m = nodal().build(nodal_data())
    a = m.assemble()
    index = a.row_of("live").start + 1
    columns, values, lower, upper = m.constraints["live"].row_at(1)
    span = slice(a.indptr[index], a.indptr[index + 1])
    assert columns.tolist() == a.indices[span].tolist()
    assert values.tolist() == a.values[span].tolist()
    assert (lower, upper) == (a.row_lower[index], a.row_upper[index])


def test_materialise_at_other_dimensions_than_the_frame_raises():
    m = nodal().build(nodal_data())
    with pytest.raises(ValueError, match="expression is free over"):
        m.constraints["balance"].expression.materialise_at({"B": "b0"})
