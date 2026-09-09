import numpy as np
import pytest

from nimopt import Definition, Sum


def dispatch():
    """Least-cost dispatch of a generator fleet against a load, declared."""
    d = Definition("dispatch", sense="min")
    snapshot = d.set("snapshot")
    generator = d.set("generator")
    p_max = d.param("p_max", (generator,))
    load = d.param("load", (snapshot,))
    cost = d.param("cost", (generator,))
    p = d.var("p", (snapshot, generator), lower=0.0, upper=p_max)
    d.eq("balance", Sum(generator, p[snapshot, generator]) == load[snapshot])
    d.set_objective(Sum(snapshot, generator, cost[generator] * p[snapshot, generator]))
    return d


def test_a_definition_registers_every_symbol_it_declares():
    d = dispatch()
    assert list(d.sets) == ["snapshot", "generator"]
    assert list(d.parameters) == ["p_max", "load", "cost"]
    assert list(d.variables) == ["p"]
    assert list(d.constraints) == ["balance"]


def test_an_equation_carries_the_free_dimensions_and_sense_of_its_relation():
    # both are properties of the expression, so neither is declared
    relation, _, _ = dispatch().constraints["balance"]
    assert relation.expression.frame == ("snapshot",)
    assert relation.sense == "=="


def test_a_definition_states_its_sense_where_it_is_named():
    assert Definition("d").sense == "min"
    assert Definition("d", sense="max").sense == "max"
    with pytest.raises(ValueError, match="sense is 'min' or 'max'"):
        Definition("d", sense="minimize")


def test_a_symbol_declared_twice_is_refused():
    d = Definition("d")
    d.set("S")
    with pytest.raises(ValueError, match="set 'S' is already declared"):
        d.set("S")


def test_a_set_and_a_parameter_do_not_share_a_name():
    # build's data is keyed by declared name, so one would shadow the other
    d = Definition("d")
    S = d.set("S")
    with pytest.raises(ValueError, match="already declared as a set"):
        d.param("S", (S,))


def test_an_equation_may_carry_the_name_of_the_parameter_that_bounds_it():
    # only a set and a parameter share a key space; an equation is in no
    # data mapping, and naming a row for its right-hand side is the ordinary
    # way to write one
    d = Definition("d")
    S = d.set("S")
    supply = d.param("supply", (S,))
    one = d.param("one", (S,))
    x = d.var("x", (S,))
    d.eq("supply", Sum(S, one[S] * x[S]) <= supply[S])
    assert list(d.parameters) == ["supply", "one"]
    assert list(d.constraints) == ["supply"]


def test_a_definition_carries_no_second_way_to_set_a_sense():
    with pytest.raises(AttributeError):
        Definition("d").sense = "max"


def test_an_objective_carrying_free_dimensions_is_refused_where_it_is_set():
    # the frame is a property of the expression, so a definition sees it as
    # well as a model does; refusing at build names a line nobody wrote
    d = Definition("d")
    S = d.set("S")
    one = d.param("one", (S,))
    x = d.var("x", (S,))
    with pytest.raises(ValueError, match=r"free dimensions \('S',\)"):
        d.set_objective(one[S] * x[S])


def data():
    return {
        "snapshot": np.arange(6),
        "generator": np.array(["wind", "solar", "gas"]),
        "p_max": np.array([100.0, 60.0, 200.0]),
        "cost": np.array([1.0, 2.0, 50.0]),
        "load": np.array([80.0, 120.0, 150.0, 180.0, 140.0, 100.0]),
    }


def test_a_definition_builds_a_model_that_solves():
    m = dispatch().build(data())
    assert (m.n_columns, m.n_rows, m.nnz) == (18, 6, 18)
    solution = m.solve()
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(1920.0)


def test_a_definition_builds_again_against_other_data():
    d = dispatch()
    first = d.build(data()).solve().objective
    cheaper = data() | {"cost": np.array([1.0, 2.0, 5.0])}
    second = d.build(cheaper).solve().objective
    assert first == pytest.approx(1920.0)
    assert second == pytest.approx(1020.0)
    # the declaration is unchanged by either build
    assert d.sets["generator"].declared


def test_data_that_misses_a_declaration_is_refused():
    with pytest.raises(ValueError, match="does not cover"):
        dispatch().build({"snapshot": np.arange(6)})


def test_data_naming_something_undeclared_is_refused():
    with pytest.raises(ValueError, match="names undeclared"):
        dispatch().build(data() | {"wind_speed": np.zeros(6)})


def test_a_tuple_that_does_not_state_the_long_form_is_refused():
    # a tuple is recognised by its type, so it is the one form that can be
    # given by accident; the refusal names the parameter and both forms
    with pytest.raises(ValueError, match="states its coefficients the long way"):
        dispatch().build(data() | {"cost": (1.0, 2.0)})


def test_a_model_built_from_a_definition_equals_one_built_directly():
    from nimopt import Model, Param, Set

    built = dispatch().build(data())
    snapshot = Set("snapshot", np.arange(6))
    generator = Set("generator", np.array(["wind", "solar", "gas"]))
    m = Model("dispatch")
    p_max = Param.from_dense("p_max", (generator,), np.array([100.0, 60.0, 200.0]))
    load = Param.from_dense("load", (snapshot,), data()["load"])
    cost = Param.from_dense("cost", (generator,), np.array([1.0, 2.0, 50.0]))
    p = m.var("p", (snapshot, generator), lower=0.0, upper=p_max)
    m.eq("balance", Sum(generator, p[snapshot, generator]) == load[snapshot])
    m.set_objective(Sum(snapshot, generator, cost[generator] * p[snapshot, generator]))
    assert (built.n_columns, built.n_rows, built.nnz) == (
        m.n_columns,
        m.n_rows,
        m.nnz,
    )
    assert built.solve().objective == pytest.approx(m.solve().objective)


def transport():
    """A variable over a sparse subset of a set product, declared.

    The arcs are the coefficients `cost` carries, so `flow` takes its members
    from that parameter rather than spanning the whole product.
    """
    d = Definition("transport", sense="min")
    P, W = d.set("P"), d.set("W")
    cost = d.param("cost", (P, W))
    supply = d.param("supply", (P,))
    demand = d.param("demand", (W,))
    flow = d.var("flow", (P, W), subset=cost, lower=0.0)
    d.eq("supply", Sum(W, cost[P, W] * flow[P, W]) <= supply[P])
    d.eq("demand", Sum(P, cost[P, W] * flow[P, W]) >= demand[W])
    d.set_objective(Sum(P, W, cost[P, W] * flow[P, W]))
    return d


def transport_data():
    """Three of the six arcs, stated as one label column per set."""
    return {
        "P": np.array(["p1", "p2"]),
        "W": np.array(["w1", "w2", "w3"]),
        "cost": (
            {
                "P": np.array(["p1", "p1", "p2"]),
                "W": np.array(["w1", "w2", "w1"]),
            },
            np.array([1.0, 2.0, 3.0]),
        ),
        "supply": np.array([3.0, 3.0]),
        "demand": np.array([1.0, 1.0, 1.0]),
    }


def test_a_sparse_shape_declares_and_builds():
    d = transport()
    e = d.explain()
    assert [q.free for q in e.constraints] == [("P",), ("W",)]
    # the columns are absent until data names the arcs
    assert e.variables[0].columns is None

    m = d.build(transport_data())
    # one column per arc, not one per cell of a 2x3 product
    assert m.n_columns == 3
    # w3 is reached by no arc, so its demand row is not stated
    assert (m.constraints["supply"].n_rows, m.constraints["demand"].n_rows) == (2, 2)
    solution = m.solve()
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(2.0)


def test_a_sparse_variable_takes_its_members_from_the_parameter_named():
    # the check that this shape is masked at all: a cost carrying every cell
    # gives the same declaration the whole product
    dense = transport_data() | {"cost": np.ones((2, 3))}
    assert transport().build(dense).n_columns == 6


def storage():
    """A model coupling each period to the one before it, declared."""
    d = Definition("storage", sense="min")
    T = d.set("T")
    one = d.param("one", (T,))
    inflow = d.param("inflow", (T,))
    level = d.var("level", (T,), lower=0.0)
    # the lag is stated where the variable is referenced
    d.eq("balance", one[T] * level[T] - one[T] * level[T - 1] == inflow[T])
    d.set_objective(Sum(T, one[T] * level[T]))
    return d


def test_a_temporally_coupled_shape_declares_and_builds():
    d = storage()
    assert d.explain().constraints[0].free == ("T",)
    m = d.build(
        {
            "T": np.arange(4),
            "one": np.ones(4),
            "inflow": np.array([0.0, 1.0, 1.0, 1.0]),
        }
    )
    # the first period has no predecessor, so its row is not stated
    assert m.n_rows == 3
    assert m.solve().status == "optimal"


def nodal():
    """A balance whose terms each reach some of its rows, declared.

    An incidence carries an entry only where a link touches a bus, so no term
    reaches every bus-hour and the rows are stated rather than derived. `live`
    names them: a parameter whose coefficients are the bus-hours the balance
    is written for.
    """
    d = Definition("nodal", sense="min")
    B, L, T = d.set("B"), d.set("L"), d.set("T")
    inc = d.param("inc", (B, L, T))
    zero = d.param("zero", (B, T))
    live = d.param("live", (B, T))
    cost = d.param("cost", (L, T))
    flow = d.var("flow", (L, T), lower=-1.0, upper=1.0)
    d.eq("balance", Sum(L, inc[B, L, T] * flow[L, T]) == zero[B, T], over=(B, T))
    d.eq("live", Sum(L, inc[B, L, T] * flow[L, T]) == zero[B, T], over=live)
    d.set_objective(Sum(L, T, cost[L, T] * flow[L, T]))
    return d


def nodal_data():
    """Two links joining two buses, live in the first hour of three."""
    return {
        "B": np.array(["b0", "b1"]),
        "L": np.array(["l0", "l1"]),
        "T": np.array([0, 1, 2]),
        "inc": (
            {
                "B": np.array(["b0", "b1", "b0", "b1"]),
                "L": np.array(["l0", "l0", "l1", "l1"]),
                "T": np.array([0, 0, 1, 1]),
            },
            np.array([1.0, -1.0, 1.0, -1.0]),
        ),
        "zero": np.zeros((2, 3)),
        "cost": np.ones((2, 3)),
        "live": (
            {"B": np.array(["b0", "b1"]), "T": np.array([0, 0])},
            np.ones(2),
        ),
    }


def test_a_shape_that_states_its_rows_declares_and_builds():
    m = nodal().build(nodal_data())
    # the sets state every bus-hour; the parameter states the two it carries
    assert m.constraints["balance"].n_rows == 6
    assert m.constraints["live"].n_rows == 2
    assert m.solve().status == "optimal"


def test_a_variable_does_not_take_the_name_of_a_set_or_a_parameter():
    d = Definition("d")
    S = d.set("S")
    d.param("cost", (S,))
    with pytest.raises(ValueError, match="variable 'S' is already declared as a set"):
        d.var("S", (S,))
    with pytest.raises(
        ValueError, match="variable 'cost' is already declared as a parameter"
    ):
        d.var("cost", (S,))
    d.var("x", (S,))
    with pytest.raises(ValueError, match="set 'x' is already declared as a variable"):
        d.set("x")


def test_a_symbol_is_named_by_an_identifier_other_than_sum():
    d = Definition("d")
    with pytest.raises(ValueError, match="identifier"):
        d.set("my-set")
    with pytest.raises(ValueError, match="Sum"):
        d.set("Sum")
    S = d.set("S")
    x = d.var("x", (S,))
    # a constraint stands in no expression, so its name is free
    d.eq("supply-1", x[S] <= 1.0)
    assert "supply-1" in d.constraints


def stock():
    d = Definition("stock")
    G, T = d.set("G"), d.set("T")
    start, rate = d.param("start", (G,)), d.param("rate", (G, T))
    x = d.var("x", (G, T))
    return d, G, T, start, rate, x


def stock_data():
    return {
        "G": np.array(["g0", "g1"]),
        "T": np.array(["t0", "t1"]),
        "start": np.array([5.0, 7.0]),
        "rate": np.ones((2, 2)),
    }


def test_a_definition_states_a_fixed_member_and_checks_it_when_the_set_binds():
    d, G, T, start, rate, x = stock()
    d.eq("initial", rate[G, "t0"] * x[G, "t0"] == start[G])
    m = d.build(stock_data())
    assert repr(m.row("initial", G="g1")) == "initial[G='g1']  row 1\n  1·x[g1,t0] == 7"
    missing = dict(stock_data(), T=np.array(["t1", "t2"]))
    with pytest.raises(
        ValueError, match="variable 'x' is read at member 't0' of dimension 'T'"
    ):
        d.build(missing)


def test_a_fixed_member_of_a_coefficient_is_checked_when_the_set_binds():
    d, G, T, start, rate, x = stock()
    d.eq("scaled", rate[G, "t0"] * x[G, T] <= 1.0)
    missing = dict(stock_data(), T=np.array(["t1", "t2"]))
    with pytest.raises(
        ValueError, match="parameter 'rate' is read at member 't0' of dimension 'T'"
    ):
        d.build(missing)


def test_a_fixed_member_in_the_objective_is_checked_when_the_set_binds():
    d, G, T, start, rate, x = stock()
    d.set_objective(Sum(G, x[G, "t0"]))
    missing = dict(stock_data(), T=np.array(["t1", "t2"]))
    with pytest.raises(ValueError, match="variable 'x' is read at member 't0'"):
        d.build(missing)
