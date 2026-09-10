import importlib
import re
import textwrap

import numpy as np
import pytest

from nimopt import Definition, Model, Param, Set, Sum, load, loads, save, subset
from nimopt.files import CONSTRAINT_KEYS, INSTRUCTIONS, KEYS, VARIABLE_KEYS, dumps
from nimopt.models import MODELS
from nimopt.models import transport as worked_transport
from reference import dense_matrix


def module(name):
    return importlib.import_module(f"nimopt.models.{name}")


DISPATCH = textwrap.dedent(
    """\
    version: 2
    name: dispatch
    sense: min
    sets: [G, T]
    parameters:
      price: [G, T]
      eta: [G, T]
      cap: [G, T]
      ramp_limit: [G, T]
      load: [T]
      budget: [G]
      live: [G]
    variables:
      gen:
        sets: [G, T]
        upper: cap
    constraints:
      balance:
        relation: Sum(G, gen[G, T]) == load[T]
      ramp:
        relation: gen[G, T] - gen[G, T - 1] <= ramp_limit[G, T]
      annual_cap:
        relation: Sum(T, gen[G, T]) <= budget[G]
        where: live
    objective: Sum(G, T, (price[G, T] / eta[G, T]) * gen[G, T])
    """
)


def dispatch():
    d = Definition("dispatch", sense="min")
    G, T = d.set("G"), d.set("T")
    price, eta = d.param("price", (G, T)), d.param("eta", (G, T))
    cap, ramp_limit = d.param("cap", (G, T)), d.param("ramp_limit", (G, T))
    load_ = d.param("load", (T,))
    budget, live = d.param("budget", (G,)), d.param("live", (G,))
    gen = d.var("gen", (G, T), upper=cap)
    d.eq("balance", Sum(G, gen[G, T]) == load_[T])
    d.eq("ramp", gen[G, T] - gen[G, T - 1] <= ramp_limit[G, T])
    d.eq("annual_cap", Sum(T, gen[G, T]) <= budget[G], where=live)
    d.set_objective(Sum(G, T, (price[G, T] / eta[G, T]) * gen[G, T]))
    return d


def test_a_definition_writes_the_file_the_spec_shows():
    assert dispatch().to_yaml() == DISPATCH


def test_the_file_reads_back_to_a_definition_that_writes_the_same_file():
    d = loads(DISPATCH)
    assert isinstance(d, Definition)
    assert list(d.sets) == ["G", "T"]
    assert d.variables["gen"].upper is d.parameters["cap"]
    assert d.constraints["annual_cap"][1] is d.parameters["live"]
    assert d.to_yaml() == DISPATCH


def test_defaults_are_omitted_and_read_back_as_defaults():
    d = Definition("d")
    S = d.set("S")
    d.var("x", (S,))
    d.var("z", (S,), lower=-np.inf, upper=3.0, integer=True)
    text = d.to_yaml()
    assert (
        "  x:\n    sets: [S]\n"
        "  z:\n    sets: [S]\n    lower: -.inf\n    upper: 3.0\n    integer: true\n"
    ) in text
    back = loads(text)
    assert (back.variables["x"].lower, back.variables["x"].upper) == (0.0, np.inf)
    assert (back.variables["z"].lower, back.variables["z"].upper) == (-np.inf, 3.0)
    assert back.variables["z"].integer is True


def test_a_subset_and_stated_rows_write_as_the_definition_takes_them():
    d = Definition("d")
    P, W = d.set("P"), d.set("W")
    cost = d.param("cost", (P, W))
    flow = d.var("flow", (P, W), subset=cost)
    d.eq("all", Sum(W, flow[P, W]) <= 1.0, over=(P,))
    text = d.to_yaml()
    assert "    subset: cost\n" in text
    assert "    over: [P]\n" in text
    back = loads(text)
    assert back.variables["flow"].subset is back.parameters["cost"]
    assert [s.name for s in back.constraints["all"][2]] == ["P"]


def test_a_model_writes_its_structure_in_order_of_first_appearance():
    P = Set("P", np.array(["a", "b"]))
    cost = Param.from_dense("cost", (P,), np.ones(2))
    m = Model("m", sense="max")
    x = m.var("x", (P,), upper=cost)
    m.eq("cap", Sum(P, cost[P] * x[P]) <= 3.0)
    assert m.to_yaml() == textwrap.dedent(
        """\
        version: 2
        name: m
        sense: max
        sets: [P]
        parameters:
          cost: [P]
        variables:
          x:
            sets: [P]
            upper: cost
        constraints:
          cap:
            relation: Sum(P, cost[P] * x[P]) <= 3
        """
    )


def test_a_model_over_a_domain_with_no_name_is_refused_by_name():
    P = Set("P", np.array(["a", "b"]))
    m = Model("m")
    x = m.var("x", (P,))
    m.eq("cap", x[P] <= 1.0, where=subset((P,), {"P": np.array(["a"])}))
    with pytest.raises(ValueError, match="constraint 'cap'.*no name.*parameter"):
        m.to_yaml()
    n = Model("n")
    n.var("y", (P,), subset=subset((P,), {"P": np.array(["a"])}))
    with pytest.raises(ValueError, match="variable 'y'.*no name.*parameter"):
        n.to_yaml()


def test_a_model_whose_symbol_the_spelling_cannot_address_is_refused():
    P = Set("my-set", np.array(["a"]))
    m = Model("m")
    m.var("x", (P,))
    with pytest.raises(ValueError, match="'my-set'.*identifier"):
        m.to_yaml()


HEAD = "version: 2\nname: d\nsense: min\nsets: [S]\n"
ONE_VARIABLE = f"{HEAD}variables:\n  x: {{sets: [S]}}\n"


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("name: d\nsense: min\n", "version"),
        ("version: 1\nname: d\nsense: min\n", "version 1"),
        ("version: 2\nname: d\nsense: min\nextra: 1\n", "extra"),
        (f"{HEAD}variables:\n  x: {{sets: [S], bound: 1}}\n", "bound"),
        (
            f"{ONE_VARIABLE}constraints:\n  c: {{relation: 'x[S] <= 1', on: S}}\n",
            "on",
        ),
        (
            f"{ONE_VARIABLE}constraints:\n  c: {{relation: 'x[S] + 1'}}\n",
            "no comparison",
        ),
        (f"{ONE_VARIABLE}objective: 'x[S] <= 1'\n", "comparison"),
        (f"{HEAD}parameters:\n  c: [Q]\n", "'Q'"),
        (f"{HEAD}variables:\n  x: {{sets: [S], upper: cap}}\n", "'cap'"),
        ("- 1\n", "mapping"),
    ],
)
def test_a_file_the_format_does_not_carry_is_refused_naming_what(text, names):
    with pytest.raises(ValueError, match=names):
        loads(text)


def same_model(a, b):
    assert (a.n_rows, a.n_columns, a.nnz) == (b.n_rows, b.n_columns, b.nnz)
    assert np.array_equal(dense_matrix(a), dense_matrix(b))
    assert np.array_equal(a.objective_coefficients(), b.objective_coefficients())
    assert [c.n_rows for c in a.constraints.values()] == [
        c.n_rows for c in b.constraints.values()
    ]


@pytest.mark.parametrize("name", MODELS)
def test_every_worked_model_round_trips_through_its_file(name):
    held = module(name)
    d = held.definition()
    text = d.to_yaml()
    back = loads(text)
    assert back.to_yaml() == text
    inputs = held.data()
    same_model(d.build(inputs), back.build(inputs))
    assert back.build(inputs).solve().objective == pytest.approx(
        d.build(inputs).solve().objective
    )


def test_a_file_is_saved_and_loaded_as_a_definition(tmp_path):
    path = tmp_path / "dispatch.yaml"
    save(dispatch(), path)
    assert path.read_text() == DISPATCH
    assert load(path).to_yaml() == DISPATCH


def test_a_definition_has_nothing_to_inline(tmp_path):
    with pytest.raises(ValueError, match="no data to inline"):
        save(dispatch(), tmp_path / "d.yaml", inline=True)


def built_transport():
    inputs = worked_transport.data()
    return worked_transport.definition().build(inputs), inputs


def test_a_model_inlines_its_data_in_the_three_shapes():
    P = Set("P", np.array(["a", "b"]))
    W = Set("W", np.array(["u", "v", "w"]))
    cost = Param.from_long(
        "cost",
        (P, W),
        {"P": np.array(["a", "b"]), "W": np.array(["u", "w"])},
        np.array([1.0, 2.0]),
    )
    cap = Param.from_dense("cap", (P,), np.array([4.0, 9.0]))
    m = Model("m")
    x = m.var("x", (P, W), subset=cost, upper=cap)
    m.eq("cap", Sum(W, cost[P, W] * x[P, W]) <= cap[P])
    text = m.to_yaml(inline=True)
    # `cap` leads `cost` because a variable's bounds are walked before its subset
    assert text.endswith(
        textwrap.dedent(
            """\
            data:
              P: [a, b]
              W: [u, v, w]
              cap: [4.0, 9.0]
              cost:
                columns: [P, W, value]
                rows:
                - [a, u, 1.0]
                - [b, w, 2.0]
            """
        )
    )
    back = loads(text)
    assert isinstance(back, Model)
    same_model(m, back)


def test_a_long_parameter_covering_its_product_inlines_as_a_grid():
    P = Set("P", np.array(["a", "b"]))
    full = Param.from_long(
        "full", (P,), {"P": np.array(["b", "a"])}, np.array([2.0, 1.0])
    )
    m = Model("m")
    x = m.var("x", (P,), upper=full)
    m.eq("cap", x[P] <= full[P])
    assert "  full: [1.0, 2.0]\n" in m.to_yaml(inline=True)


def test_a_file_with_an_inline_block_loads_to_a_built_model(tmp_path):
    m, _ = built_transport()
    path = tmp_path / "transport.yaml"
    save(m, path, inline=True)
    back = load(path)
    assert isinstance(back, Model)
    same_model(m, back)
    assert back.solve().objective == pytest.approx(m.solve().objective)


def test_a_model_saves_its_data_beside_the_file_and_loads_back(tmp_path):
    m, _ = built_transport()
    path = tmp_path / "transport.yaml"
    save(m, path)
    assert path.read_text().endswith("data: transport.npz\n")
    assert (tmp_path / "transport.npz").is_file()
    back = load(path)
    assert isinstance(back, Model)
    same_model(m, back)
    a, b = m.assemble(), back.assemble()
    assert np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.col_lower, b.col_lower)
    assert np.array_equal(a.col_upper, b.col_upper)
    assert np.array_equal(a.col_cost, b.col_cost)


def test_the_sidecar_holds_the_three_shapes_under_the_symbols_names(tmp_path):
    m, inputs = built_transport()
    save(m, tmp_path / "t.yaml")
    with np.load(tmp_path / "t.npz", allow_pickle=False) as held:
        assert set(held.files) == set(inputs)
        assert held["cost"].dtype.names == ("P", "W", "value")
        assert held["supply"].shape == inputs["supply"].shape
        assert held["P"].tolist() == inputs["P"].tolist()


def test_a_definition_builds_from_a_sidecar_named_by_the_caller(tmp_path):
    m, _ = built_transport()
    save(m, tmp_path / "t.yaml")
    d = worked_transport.definition()
    save(d, tmp_path / "bare.yaml")
    back = load(tmp_path / "bare.yaml", data=tmp_path / "t.npz")
    assert isinstance(back, Model)
    same_model(m, back)
    also = load(tmp_path / "bare.yaml", data=worked_transport.data())
    same_model(m, also)


def test_two_sources_for_one_model_are_refused(tmp_path):
    m, inputs = built_transport()
    save(m, tmp_path / "t.yaml")
    with pytest.raises(ValueError, match="two sources"):
        load(tmp_path / "t.yaml", data=inputs)
    with pytest.raises(ValueError, match="two sources"):
        loads(m.to_yaml(inline=True), data=inputs)


def test_text_cannot_name_a_sidecar():
    with pytest.raises(ValueError, match="no directory"):
        loads(DISPATCH + "data: dispatch.npz\n")


def test_a_sidecar_name_carries_no_directory(tmp_path):
    (tmp_path / "d.yaml").write_text(DISPATCH + "data: ../elsewhere.npz\n")
    with pytest.raises(ValueError, match="no directory"):
        load(tmp_path / "d.yaml")


def test_an_inline_table_states_the_dimensions_then_value():
    text = DISPATCH + textwrap.dedent(
        """\
        data:
          G: [base]
          T: [0]
          price: [[1.0]]
          eta: [[1.0]]
          cap: [[1.0]]
          ramp_limit: [[1.0]]
          load: [1.0]
          budget: [1.0]
          live:
            columns: [value, G]
            rows:
            - [1.0, base]
        """
    )
    with pytest.raises(ValueError, match="'live'.*\\['G', 'value'\\]"):
        loads(text)


def test_an_object_array_is_refused_before_the_sidecar_is_written(tmp_path):
    P = Set("P", np.array(["a", 1], dtype=object))
    m = Model("m")
    m.var("x", (P,))
    with pytest.raises(ValueError, match="'P'.*object"):
        save(m, tmp_path / "m.yaml")
    assert not (tmp_path / "m.npz").exists()


def test_the_data_a_sidecar_carries_is_named_by_the_file(tmp_path):
    m, inputs = built_transport()
    save(m, tmp_path / "t.yaml")
    with np.load(tmp_path / "t.npz", allow_pickle=False) as held:
        keep = {name: held[name] for name in held.files if name != "cost"}
    np.savez(tmp_path / "t.npz", **keep)
    with pytest.raises(ValueError, match="does not cover \\['cost'\\]"):
        load(tmp_path / "t.yaml")


def test_a_sidecar_keeps_the_dtype_each_label_column_carries(tmp_path):
    # a structured array read row by row hands back a datetime64 as the
    # integer behind it, and the label then resolves against nothing
    hours = np.array(["2030-01-01", "2030-01-02", "2030-01-03"], dtype="datetime64[ns]")
    P = Set("P", np.array(["a", "b"]))
    T = Set("T", hours)
    cost = Param.from_long(
        "cost",
        (P, T),
        {"P": np.array(["a", "b"]), "T": hours[[0, 2]]},
        np.array([1.0, 2.0]),
    )
    m = Model("m")
    x = m.var("x", (P, T), subset=cost)
    m.eq("cap", Sum(T, cost[P, T] * x[P, T]) <= 1.0)
    path = tmp_path / "m.yaml"
    save(m, path)
    with np.load(tmp_path / "m.npz", allow_pickle=False) as held:
        assert held["cost"].dtype["T"] == np.dtype("datetime64[ns]")
    back = load(path)
    assert isinstance(back, Model)
    same_model(m, back)


def test_an_objective_over_no_dimension_survives_the_file():
    d = Definition("scalar", sense="min")
    S = d.set("S")
    theta = d.var("theta", (), lower=-np.inf)
    x = d.var("x", (S,))
    d.eq("tail", theta - Sum(S, x[S]) >= 0.0)
    d.set_objective(theta)
    text = d.to_yaml()
    assert "objective: theta\n" in text
    assert loads(text).to_yaml() == text


def test_dumps_prefixes_the_block_where_instructions_is_asked():
    assert dumps({"version": 2}, instructions=True) == INSTRUCTIONS + "version: 2\n"
    assert dumps({"version": 2}) == "version: 2\n"


def test_a_definition_writes_the_block_only_where_instructions_is_asked():
    d = dispatch()
    assert d.to_yaml(instructions=True) == INSTRUCTIONS + d.to_yaml()
    assert "#" not in d.to_yaml()


def test_a_model_writes_the_block_above_its_inline_data():
    m = worked_transport.definition().build(worked_transport.data())
    text = m.to_yaml(inline=True, instructions=True)
    assert text == INSTRUCTIONS + m.to_yaml(inline=True)
    block, head, data = (
        text.index(s) for s in ("Reading this file", "version: 2", "data:")
    )
    assert block < head < data


def test_save_writes_a_definition_with_the_block(tmp_path):
    d = dispatch()
    save(d, tmp_path / "d.yaml", instructions=True)
    assert (tmp_path / "d.yaml").read_text() == INSTRUCTIONS + d.to_yaml()


def test_save_writes_a_model_with_the_block_above_the_sidecar_line(tmp_path):
    m = worked_transport.definition().build(worked_transport.data())
    save(m, tmp_path / "m.yaml", instructions=True)
    text = (tmp_path / "m.yaml").read_text()
    assert text == INSTRUCTIONS + m.to_yaml() + "data: m.npz\n"


def test_a_file_carrying_the_block_reads_to_the_definition_the_rest_states():
    d = dispatch()
    assert loads(d.to_yaml(instructions=True)).to_yaml() == d.to_yaml()


def test_a_definition_with_the_block_round_trips_to_the_same_text():
    text = dispatch().to_yaml(instructions=True)
    assert loads(text).to_yaml(instructions=True) == text


def test_a_saved_model_with_the_block_loads_and_saves_to_the_same_file(tmp_path):
    m = worked_transport.definition().build(worked_transport.data())
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    save(m, tmp_path / "a" / "m.yaml", instructions=True)
    save(load(tmp_path / "a" / "m.yaml"), tmp_path / "b" / "m.yaml", instructions=True)
    again = (tmp_path / "b" / "m.yaml").read_text()
    assert again == (tmp_path / "a" / "m.yaml").read_text()


def test_the_block_names_every_key_of_the_format():
    # a key added to the format without a line in the block fails here
    for key in (*KEYS, *VARIABLE_KEYS, *CONSTRAINT_KEYS):
        assert re.search(rf"\b{key}\b", INSTRUCTIONS), key


def test_the_block_is_ascii_comment_lines():
    # every line is a YAML comment, so the loader never sees it, and the
    # text stays ASCII so the file is written the same under any locale
    lines = INSTRUCTIONS.splitlines()
    assert lines and all(line.startswith("#") for line in lines)
    assert INSTRUCTIONS.endswith("\n")
    assert INSTRUCTIONS.isascii()
