from nimopt import Explanation
from test_definition import dispatch


def test_a_definition_explains_with_nothing_bound():
    e = dispatch().explain()
    assert isinstance(e, Explanation)
    assert e.name == "dispatch"
    assert e.sense == "min"
    assert e.built is False


def test_an_unbuilt_explanation_reports_absent_counts_not_zero():
    e = dispatch().explain()
    # zero is a fact a caller acts on; an unbound declaration has none
    assert e.columns is None
    assert e.rows is None
    assert e.nonzeros is None
    assert [s.size for s in e.sets] == [None, None]
    assert [v.columns for v in e.variables] == [None]


def test_an_explanation_reads_free_dimensions_and_sense_off_the_relation():
    balance = dispatch().explain().constraints[0]
    assert balance.name == "balance"
    assert balance.free == ("snapshot",)
    assert balance.sense == "=="
    # rows and nonzeros are facts about data, so they are absent
    assert balance.rows is None
    assert balance.nonzeros is None


def test_an_explanation_states_the_shape_of_every_declaration():
    e = dispatch().explain()
    assert [(p.name, p.dims) for p in e.parameters] == [
        ("p_max", ("generator",)),
        ("load", ("snapshot",)),
        ("cost", ("generator",)),
    ]
    assert e.variables[0].dims == ("snapshot", "generator")


def test_a_sparse_variable_explains_differently_from_a_dense_one():
    # columns are absent until data binds, so without the members a sparse
    # declaration and a dense one would render identically
    from test_definition import transport

    assert transport().explain().variables[0].members == "cost"
    assert dispatch().explain().variables[0].members is None
    assert "over cost" in repr(transport().explain())


def test_an_unbuilt_explanation_says_so_when_it_renders():
    assert "not built" in repr(dispatch().explain())


def test_an_explanation_spells_the_objective():
    assert (
        dispatch().explain().objective
        == "Sum(snapshot, generator, cost[generator] * p[snapshot, generator])"
    )


def test_an_explanation_spells_each_constraint():
    written = "Sum(generator, p[snapshot, generator]) == load[snapshot]"
    balance = dispatch().explain().constraints[0]
    assert balance.relation == written
    assert written in repr(dispatch().explain())
