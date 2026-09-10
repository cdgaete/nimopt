import importlib

import numpy as np
import pytest

from nimopt import Definition, Model, Param, Relation, Set, Sum, subset
from nimopt.models import MODELS
from nimopt.syntax import read, render


def fleet():
    G = Set("G", np.array(["base", "peak"]))
    T = Set("T", np.array(["t0", "t1", "t2"]))
    m = Model("m")
    gen = m.var("gen", (G, T))
    cost = Param.from_dense("cost", (G, T), np.ones((2, 3)))
    eta = Param.from_dense("eta", (G, T), np.full((2, 3), 0.5))
    load = Param.from_dense("load", (T,), np.ones(3))
    return G, T, gen, cost, eta, load


def test_a_term_spells_its_variable_at_its_sets():
    G, T, gen, *_ = fleet()
    assert render(gen[G, T]) == "gen[G, T]"


def test_a_coefficient_spells_with_its_brackets():
    G, T, gen, cost, *_ = fleet()
    assert render(cost[G, T] * gen[G, T]) == "cost[G, T] * gen[G, T]"


def test_a_sum_wraps_each_term_it_reduces():
    G, T, gen, cost, *_ = fleet()
    # the DSL distributes a sum over its terms, and the spelling states that
    assert (
        render(Sum(G, cost[G, T] * gen[G, T] + gen[G, T]))
        == "Sum(G, cost[G, T] * gen[G, T]) + Sum(G, gen[G, T])"
    )
    assert render(Sum(G, T, gen[G, T])) == "Sum(G, T, gen[G, T])"


def test_a_lag_and_a_cyclic_lag_spell_as_typed():
    G, T, gen, *_ = fleet()
    assert render(gen[G, T - 1]) == "gen[G, T - 1]"
    assert render(gen[G, T + 2]) == "gen[G, T + 2]"
    assert render(gen[G, T.cyclic - 1]) == "gen[G, T.cyclic - 1]"


def test_a_fixed_member_spells_as_its_label():
    G, T, gen, cost, *_ = fleet()
    assert render(cost[G, "t0"] * gen[G, "t0"]) == "cost[G, 't0'] * gen[G, 't0']"


def test_a_numeric_label_spells_bare_whatever_its_type():
    K = Set("K", np.arange(3))
    m = Model("m")
    x = m.var("x", (K,))
    assert render(x[0]) == "x[0]"
    assert render(x[np.int64(2)]) == "x[2]"


def test_a_derived_coefficient_spells_as_its_tree():
    G, T, gen, cost, eta, _ = fleet()
    unit = cost[G, T] / eta[G, T]
    assert render(unit * gen[G, T]) == "(cost[G, T] / eta[G, T]) * gen[G, T]"
    assert render(2 * unit * gen[G, T]) == "((cost[G, T] / eta[G, T]) * 2) * gen[G, T]"
    assert render((-cost[G, T]) * gen[G, T]) == "(-cost[G, T]) * gen[G, T]"
    assert render((2 - cost[G, T]) * gen[G, T]) == "(2 - cost[G, T]) * gen[G, T]"
    assert render((cost[G, T] ** 2) * gen[G, T]) == "(cost[G, T] ** 2) * gen[G, T]"


def test_a_derived_coefficient_read_at_its_sets_spells_once():
    G, T, gen, cost, eta, _ = fleet()
    unit = cost[G, T] / eta[G, T]
    assert render(unit[G, T] * gen[G, T]) == "(cost[G, T] / eta[G, T]) * gen[G, T]"
    assert (
        render(unit[G, "t0"] * gen[G, "t0"])
        == "(cost[G, T] / eta[G, T])[G, 't0'] * gen[G, 't0']"
    )


def test_a_scale_leads_and_parenthesises_a_product():
    G, T, gen, cost, *_ = fleet()
    assert render(2 * gen[G, T]) == "2 * gen[G, T]"
    assert render(0.5 * gen[G, T]) == "0.5 * gen[G, T]"
    assert render(2 * (cost[G, T] * gen[G, T])) == "2 * (cost[G, T] * gen[G, T])"
    assert (
        render(2 * Sum(G, cost[G, T] * gen[G, T]))
        == "2 * Sum(G, cost[G, T] * gen[G, T])"
    )


def test_a_negative_term_leads_with_a_minus_and_follows_with_one():
    G, T, gen, cost, *_ = fleet()
    assert render(-gen[G, T]) == "-gen[G, T]"
    assert render(-(cost[G, T] * gen[G, T])) == "-(cost[G, T] * gen[G, T])"
    assert render(-2 * (cost[G, T] * gen[G, T])) == "-2 * (cost[G, T] * gen[G, T])"
    assert (
        render(gen[G, T] - cost[G, T] * gen[G, T])
        == "gen[G, T] - cost[G, T] * gen[G, T]"
    )
    assert render(gen[G, T] - 2 * gen[G, T]) == "gen[G, T] - 2 * gen[G, T]"


def test_a_constant_stands_last():
    G, T, gen, *_ = fleet()
    assert render(gen[G, T] + 1) == "gen[G, T] + 1"
    assert render(gen[G, T] - 1.5) == "gen[G, T] - 1.5"


def test_a_relation_spells_its_sense_and_right_hand_side():
    G, T, gen, cost, eta, load = fleet()
    assert render(Sum(G, gen[G, T]) == load[T]) == "Sum(G, gen[G, T]) == load[T]"
    assert render(gen[G, T] <= 5) == "gen[G, T] <= 5"
    assert render(gen[G, T] >= cost[G, T]) == "gen[G, T] >= cost[G, T]"
    assert render(gen[G, T] >= cost[G, T] - 1) == "gen[G, T] >= (cost[G, T] - 1)"
    assert render(gen[G, T] <= gen[G, T - 1]) == "gen[G, T] - gen[G, T - 1] <= 0"


def test_a_sum_condition_spells_a_parameter_or_a_tuple_of_sets():
    G, T, gen, cost, *_ = fleet()
    assert render(Sum(G, gen[G, T], where=cost)) == "Sum(G, gen[G, T], where=cost)"
    assert render(Sum(G, gen[G, T], where=(G, T))) == "Sum(G, gen[G, T], where=(G, T))"
    assert render(Sum(T, gen[G, T], where=(T,))) == "Sum(T, gen[G, T], where=(T,))"


def test_a_condition_with_no_name_is_refused():
    G, T, gen, *_ = fleet()
    arcs = subset((G, T), {"G": np.array(["base"]), "T": np.array(["t0"])})
    with pytest.raises(ValueError, match="no name"):
        render(Sum(G, gen[G, T], where=arcs))


def test_a_coefficient_spells_on_its_own():
    G, T, _, cost, eta, _ = fleet()
    assert render(cost[G, T]) == "cost[G, T]"
    assert render(cost[G, T] / eta[G, T]) == "(cost[G, T] / eta[G, T])"


def test_an_expression_and_a_relation_print_as_their_spelling():
    G, T, gen, *_ = fleet()
    assert repr(gen[G, T] + 1) == "gen[G, T] + 1"
    assert repr(gen[G, T] <= 1) == "gen[G, T] <= 1"


def symbols(definition):
    return {**definition.sets, **definition.parameters, **definition.variables}


def declared():
    d = Definition("d")
    G, T = d.set("G"), d.set("T")
    d.param("cost", (G, T))
    d.param("eta", (G, T))
    d.param("load", (T,))
    d.param("live", (G,))
    d.var("gen", (G, T))
    return d


def test_a_relation_reads_to_a_relation_over_the_declared_symbols():
    d = declared()
    held = read("Sum(G, gen[G, T]) == load[T]", symbols(d))
    assert isinstance(held, Relation)
    assert held.sense == "=="
    assert held.rhs.parameters() == (d.parameters["load"],)
    assert held.expression.terms[0].variable is d.variables["gen"]


def test_reading_calls_the_dsl_so_its_normalisation_applies():
    held = read("gen[G, T] <= gen[G, T - 1]", symbols(declared()))
    assert [t.scale for t in held.expression.terms] == [1.0, -1.0]
    assert held.expression.terms[1].shifts == {"T": (1, "drop")}
    assert held.rhs == 0.0


CANONICAL = [
    "gen[G, T]",
    "cost[G, T] * gen[G, T]",
    "Sum(G, cost[G, T] * gen[G, T]) + Sum(G, gen[G, T])",
    "Sum(G, T, gen[G, T])",
    "gen[G, T - 1]",
    "gen[G, T + 2]",
    "gen[G, T.cyclic - 1]",
    "cost[G, 't0'] * gen[G, 't0']",
    "(cost[G, T] / eta[G, T]) * gen[G, T]",
    "((cost[G, T] / eta[G, T]) * 2) * gen[G, T]",
    "(-cost[G, T]) * gen[G, T]",
    "(2 - cost[G, T]) * gen[G, T]",
    "(cost[G, T] ** 2) * gen[G, T]",
    "(cost[G, T] / eta[G, T])[G, 't0'] * gen[G, 't0']",
    "2 * gen[G, T]",
    "0.5 * gen[G, T]",
    "2 * (cost[G, T] * gen[G, T])",
    "2 * Sum(G, cost[G, T] * gen[G, T])",
    "-gen[G, T]",
    "-(cost[G, T] * gen[G, T])",
    "-2 * (cost[G, T] * gen[G, T])",
    "gen[G, T] - cost[G, T] * gen[G, T]",
    "gen[G, T] - 2 * gen[G, T]",
    "gen[G, T] + 1",
    "gen[G, T] - 1.5",
    "Sum(G, gen[G, T], where=live)",
    "Sum(G, gen[G, T], where=(G, T))",
    "Sum(T, gen[G, T], where=(T,))",
    "Sum(G, gen[G, T]) == load[T]",
    "gen[G, T] <= 5",
    "gen[G, T] >= cost[G, T]",
    "gen[G, T] >= (cost[G, T] - 1)",
    "gen[G, T] - gen[G, T - 1] <= 0",
]


@pytest.mark.parametrize("text", CANONICAL)
def test_a_spelling_is_a_fixed_point_of_reading(text):
    assert render(read(text, symbols(declared()))) == text


EDITED = [
    ("2 * cost[G, T] * gen[G, T]", "(cost[G, T] * 2) * gen[G, T]"),
    ("gen[G, T] * 2 + 1 - 3", "2 * gen[G, T] - 2"),
    ("-cost[G, T] * gen[G, T]", "(-cost[G, T]) * gen[G, T]"),
    ("5 >= gen[G, T]", "gen[G, T] <= 5"),
    (
        "Sum(G, gen[G, T] + cost[G, T] * gen[G, T])",
        "Sum(G, gen[G, T]) + Sum(G, cost[G, T] * gen[G, T])",
    ),
    ("gen[G, T] / 4", "0.25 * gen[G, T]"),
    ("(gen[G, T] + 1) * 2 <= load[T]", "2 * gen[G, T] + 2 <= load[T]"),
]


@pytest.mark.parametrize(("edited", "canonical"), EDITED)
def test_a_hand_edit_reads_as_the_same_edit_in_python(edited, canonical):
    assert render(read(edited, symbols(declared()))) == canonical


def test_the_dsl_refuses_in_a_file_what_it_refuses_in_python():
    held = symbols(declared())
    with pytest.raises(TypeError, match="strict inequality"):
        read("gen[G, T] < 5", held)
    with pytest.raises(TypeError, match="not linear"):
        read("gen[G, T] * gen[G, T]", held)
    with pytest.raises(ValueError, match="takes the set and not a lag"):
        read("Sum(T - 1, gen[G, T])", held)
    with pytest.raises(ValueError, match="declared over"):
        read("gen[T, G]", held)


def test_a_chained_comparison_is_refused_by_the_reader():
    with pytest.raises(ValueError, match="chained comparison"):
        read("0 <= gen[G, T] <= 5", symbols(declared()))


def test_a_name_outside_the_declarations_is_refused():
    with pytest.raises(ValueError, match="'demand'.*names no declared"):
        read("Sum(G, gen[G, T]) == demand", symbols(declared()))


def test_a_call_other_than_sum_is_refused():
    held = symbols(declared())
    with pytest.raises(ValueError, match="Sum is the one call"):
        read("abs(gen[G, T])", held)
    with pytest.raises(ValueError, match="Sum is the one call"):
        read("max(gen[G, T], 1)", held)
    with pytest.raises(ValueError, match="where"):
        read("Sum(G, gen[G, T], over=live)", held)


def test_an_attribute_other_than_cyclic_is_refused():
    held = symbols(declared())
    with pytest.raises(ValueError, match="cyclic"):
        read("gen[G, T.labels]", held)
    with pytest.raises(ValueError, match="cyclic"):
        read("cost.array", held)
    with pytest.raises(ValueError, match="cyclic"):
        read("gen[G, T].cyclic", held)


@pytest.mark.parametrize(
    ("text", "node"),
    [
        ("lambda: 1", "Lambda"),
        ("[gen[G, T]]", "List"),
        ("gen[G, T] if 1 else 2", "IfExp"),
        ("gen[G, T] and 1", "BoolOp"),
        ("gen[G, T] % 2", "Mod"),
        ("Sum(*G, gen[G, T])", "Starred"),
        ("gen[G, T] <= True", "bool"),
        ("gen[G, T] <= None", "None"),
    ],
)
def test_a_construct_outside_the_spelling_is_refused_by_name(text, node):
    with pytest.raises(ValueError, match=node):
        read(text, symbols(declared()))


def test_text_that_is_not_an_expression_is_refused():
    with pytest.raises(ValueError, match="not an expression"):
        read("gen[G, T] =", symbols(declared()))


@pytest.mark.parametrize("name", MODELS)
def test_every_worked_model_spells_to_a_fixed_point(name):
    d = importlib.import_module(f"nimopt.models.{name}").definition()
    held = symbols(d)
    for relation, _, _ in d.constraints.values():
        text = render(relation)
        assert render(read(text, held)) == text
    text = render(d.objective)
    assert render(read(text, held)) == text
