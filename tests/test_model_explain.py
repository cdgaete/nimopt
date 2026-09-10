import numpy as np
import pytest

from nimopt import Explanation, Model, Param, Set, Sum
from test_definition import (
    data,
    dispatch,
    nodal,
    nodal_data,
    transport,
    transport_data,
)


def test_a_built_model_explains_that_it_is_built():
    e = dispatch().build(data()).explain()
    assert isinstance(e, Explanation)
    assert e.name == "dispatch"
    assert e.sense == "min"
    assert e.built is True


def test_the_counts_a_model_explains_are_the_ones_it_assembles():
    # what is reported is read from what was built
    m = dispatch().build(data())
    a = m.assemble()
    e = m.explain()
    assert (e.columns, e.rows, e.nonzeros) == (a.n_cols, a.n_rows, a.values.size)
    assert (e.columns, e.rows, e.nonzeros) == (18, 6, 18)


def test_a_built_explanation_reports_a_count_for_every_declaration():
    e = dispatch().build(data()).explain()
    assert [(s.name, s.size) for s in e.sets] == [("snapshot", 6), ("generator", 3)]
    assert [(v.name, v.columns) for v in e.variables] == [("p", 18)]
    assert [(q.name, q.rows, q.nonzeros) for q in e.constraints] == [("balance", 6, 18)]


def test_a_model_explains_a_set_no_variable_carries():
    # nodal's B is introduced by the coefficient inc and by no variable, so a
    # walk of the variables alone reports a dimension short
    e = nodal().build(nodal_data()).explain()
    assert sorted(s.name for s in e.sets) == ["B", "L", "T"]


def test_a_model_explains_a_parameter_used_only_to_state_rows():
    # nodal's live is the over= of one equation and nothing else
    e = nodal().build(nodal_data()).explain()
    assert "live" in [p.name for p in e.parameters]
    assert [p.entries for p in e.parameters if p.name == "live"] == [2]


def test_a_built_model_still_names_the_parameter_a_sparse_variable_reads():
    # members is a declared fact and survives binding, so a built sparse
    # variable and a built dense one still explain differently
    e = transport().build(transport_data()).explain()
    assert e.variables[0].members == "cost"
    assert e.variables[0].columns == 3
    assert "over cost" in repr(e)


def test_a_built_explanation_renders_its_shape_in_its_header():
    text = repr(dispatch().build(data()).explain())
    assert "not built" not in text
    assert "18 columns" in text and "6 rows" in text and "18 nonzeros" in text


def test_a_definition_and_the_model_it_builds_explain_the_same_declarations():
    # one rule renders a symbol either way; only the counts differ
    d = dispatch()
    before, after = d.explain(), d.build(data()).explain()
    assert [s.name for s in before.sets] == [s.name for s in after.sets]
    assert [(v.name, v.dims) for v in before.variables] == [
        (v.name, v.dims) for v in after.variables
    ]
    assert [(q.name, q.free, q.sense) for q in before.constraints] == [
        (q.name, q.free, q.sense) for q in after.constraints
    ]
    written = "Sum(snapshot, generator, cost[generator] * p[snapshot, generator])"
    assert before.objective == after.objective == written


def test_a_model_declared_directly_explains_without_a_definition():
    from nimopt import Model, Param, Set, Sum

    P = Set("P", np.array(["p1", "p2"]))
    m = Model("direct", sense="max")
    x = m.var("x", (P,), upper=4.0)
    one = Param.from_dense("one", (P,), np.ones(2))
    m.constraint("cap", one[P] * x[P] <= 3.0)
    m.set_objective(Sum(P, one[P] * x[P]))
    e = m.explain()
    assert (e.name, e.sense, e.built) == ("direct", "max", True)
    assert (e.columns, e.rows, e.nonzeros) == (2, 2, 2)
    assert [p.name for p in e.parameters] == ["one"]


def test_a_count_a_built_model_reports_is_zero_where_it_is_zero():
    # absent means unbound; a model with no constraints has none, which is a
    # fact a caller acts on and is reported as the number it is
    from nimopt import Model, Set

    P = Set("P", np.array(["p1"]))
    m = Model("bare")
    m.var("x", (P,))
    e = m.explain()
    assert e.rows == 0
    assert e.nonzeros == 0
    assert e.constraints == ()
    assert e.objective is None


def test_a_model_reads_its_objective():
    P = Set("P", np.array(["a", "b"]))
    m = Model("m")
    x = m.var("x", (P,))
    assert m.objective is None
    m.set_objective(Sum(P, x[P]))
    assert repr(m.objective) == "Sum(P, x[P])"


def test_a_constraint_reads_back_as_the_relation_it_was_declared_from():
    P = Set("P", np.array(["a", "b"]))
    m = Model("m")
    x = m.var("x", (P,))
    held = m.constraint("cap", x[P] <= 1.0)
    assert repr(held.relation) == "x[P] <= 1"


def test_two_parameters_sharing_a_name_are_refused_where_the_model_walks_them():
    P = Set("P", np.array(["a", "b"]))
    first = Param.from_dense("cost", (P,), np.ones(2))
    second = Param.from_dense("cost", (P,), np.ones(2))
    m = Model("m")
    x = m.var("x", (P,))
    m.constraint("one", first[P] * x[P] <= 1.0)
    m.constraint("two", second[P] * x[P] <= 1.0)
    with pytest.raises(ValueError, match="two parameters named 'cost'"):
        m.explain()


def test_two_sets_sharing_a_name_are_refused_where_the_model_walks_them():
    m = Model("m")
    m.var("x", (Set("P", np.array(["a"])),))
    m.var("y", (Set("P", np.array(["b"])),))
    with pytest.raises(ValueError, match="two sets named 'P'"):
        m.explain()


def test_a_built_explanation_spells_its_relations_beside_its_counts():
    P = Set("P", np.array(["a", "b"]))
    one = Param.from_dense("one", (P,), np.ones(2))
    m = Model("m")
    x = m.var("x", (P,))
    m.constraint("cap", one[P] * x[P] + 1 <= 5.0)
    m.set_objective(Sum(P, x[P]) + 7.0)
    lines = repr(m.explain()).splitlines()
    assert lines[-2] == "  constraint  cap (P)  one[P] * x[P] + 1 <= 5  2 rows  2 nz"
    assert lines[-1] == "  objective   min  Sum(P, x[P]) + 7"
