import re

import numpy as np
import pytest

import nimopt as no


def table(G, B, rows, name):
    """A parameter over (G, B) from (member, breakpoint, value) rows."""
    g, b, v = zip(*rows)
    return no.Param.from_long(
        name, (G, B), {"G": np.array(g), "B": np.array(b)}, np.array(v, float)
    )


def cost_model(**changes):
    # a: four breakpoints, not convex; b: three breakpoints, the last absent;
    # c: no breakpoint, and a capacity of zero
    G = no.Set("G", np.array(["a", "b", "c"]))
    T = no.Set("T", np.array(["t0", "t1"]))
    B = no.Set("B", np.array(["b0", "b1", "b2", "b3"]))
    xs = [
        ("a", "b0", 0),
        ("a", "b1", 10),
        ("a", "b2", 20),
        ("a", "b3", 30),
        ("b", "b0", 0),
        ("b", "b1", 10),
        ("b", "b2", 20),
    ]
    ys = [
        ("a", "b0", 0),
        ("a", "b1", 5),
        ("a", "b2", 30),
        ("a", "b3", 35),
        ("b", "b0", 0),
        ("b", "b1", 20),
        ("b", "b2", 30),
    ]
    xp = table(G, B, changes.pop("xs", xs), "xp")
    yp = table(G, B, changes.pop("ys", ys), "yp")
    cap = no.Param.from_dense("cap", (G,), np.array([np.inf, np.inf, 0.0]))
    demand = no.Param.from_dense("demand", (T,), np.array([25.0, 40.0]))
    m = no.Model("cost")
    p = m.var("p", (G, T), upper=cap)
    c = m.var("c", (G, T))
    m.constraint("balance", no.Sum(G, p[G, T]) == demand[T])
    arguments = dict(
        x=p[G, T],
        x_points=xp[G, B],
        y=c[G, T],
        y_points=yp[G, B],
        sign=">=",
        method="incremental",
    )
    arguments.update(changes)
    m.piecewise("curve", **arguments)
    m.set_objective(no.Sum(G, T, c[G, T]))
    return m


def test_incremental_solves_a_curve_that_is_not_convex():
    # t0 splits 25 as a=10, b=15 at cost 5 + 25; t1 splits 40 as a=30,
    # b=10 at cost 35 + 20
    m = cost_model()
    s = m.solve()
    assert s.status == "optimal"
    assert s.objective == pytest.approx(85.0)
    assert s.primal("p").to_dense().tolist() == [[10.0, 30.0], [15.0, 10.0], [0.0, 0.0]]


def test_incremental_generates_the_declarations_of_its_names():
    m = cost_model()
    assert list(m.variables) == ["p", "c", "curve_fill", "curve_order"]
    assert list(m.constraints) == [
        "balance",
        "curve_x",
        "curve_y",
        "curve_order_bound",
        "curve_fill_order",
        "curve_order_link",
    ]
    assert m.variables["curve_order"].integer
    d = m.piecewise_declarations["curve"]
    assert d.generated["sets"] == ("curve_segment",)
    assert d.generated["variables"] == ("curve_fill", "curve_order")
    assert d.generated_names() >= {"curve_members", "curve_x_step", "curve_x"}


def test_an_entity_with_no_breakpoint_has_no_rows_and_no_columns():
    m = cost_model()
    # a and b in each of two periods; c has none
    assert m.constraints["curve_x"].n_rows == 4
    # a has three segments and b two, in each of two periods
    assert m.variables["curve_fill"].n_columns == 10


def test_the_generated_columns_are_read_by_their_names():
    s = cost_model().solve()
    fill = s.primal("curve_fill")
    assert fill.nnz == 10


def test_incremental_with_active_holds_x_at_zero_where_active_is_zero():
    # t0: demand 5 is below the minimum output 10, so a is off and s serves
    # it at 6; t1: a on at 20 costs 15 + 2 and s serves 5 at 6
    G = no.Set("G", np.array(["a"]))
    T = no.Set("T", np.array(["t0", "t1"]))
    B = no.Set("B", np.array(["b0", "b1", "b2"]))
    xp = no.Param.from_dense("xp", (G, B), [[10.0, 20.0, 30.0]])
    yp = no.Param.from_dense("yp", (G, B), [[10.0, 15.0, 30.0]])
    demand = no.Param.from_dense("demand", (T,), [5.0, 25.0])
    m = no.Model("committed")
    p = m.var("p", (G, T))
    c = m.var("c", (G, T))
    u = m.var("u", (G, T), upper=1.0, integer=True)
    s = m.var("s", (T,))
    m.constraint("balance", no.Sum(G, p[G, T]) + s[T] == demand[T])
    m.piecewise(
        "curve",
        x=p[G, T],
        x_points=xp[G, B],
        y=c[G, T],
        y_points=yp[G, B],
        sign=">=",
        method="incremental",
        active=u[G, T],
    )
    m.set_objective(
        no.Sum(G, T, c[G, T]) + 1.2 * no.Sum(T, s[T]) + 2.0 * no.Sum(G, T, u[G, T])
    )
    solved = m.solve()
    assert solved.objective == pytest.approx(29.0)
    assert solved.primal("u").to_dense().tolist() == [[0.0, 1.0]]
    assert solved.primal("p").to_dense().tolist() == [[0.0, 20.0]]
    assert "curve_active" in m.constraints


def one_generator(yp_values, sense="min"):
    G = no.Set("G", np.array(["a"]))
    B = no.Set("B", np.array(["b0", "b1", "b2"]))
    xp = no.Param.from_dense("xp", (G, B), [[0.0, 10.0, 20.0]])
    yp = no.Param.from_dense("yp", (G, B), [yp_values])
    return G, B, xp, yp, no.Model("one", sense=sense)


def test_tangent_solves_a_convex_curve_bounded_below():
    # p = 10 costs 10 and s = 5 costs 7.5; the slope above 10 is 2 > 1.5
    G, B, xp, yp, m = one_generator([0.0, 10.0, 30.0])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    s = m.var("s", ())
    m.constraint("balance", no.Sum(G, p[G]) + s == 15.0)
    m.piecewise(
        "curve",
        x=p[G],
        x_points=xp[G, B],
        y=c[G],
        y_points=yp[G, B],
        sign=">=",
        method="tangent",
    )
    m.set_objective(no.Sum(G, c[G]) + 1.5 * s)
    solved = m.solve()
    assert solved.objective == pytest.approx(17.5)
    assert solved.primal("p").to_dense().tolist() == [10.0]
    assert list(m.variables) == ["p", "c", "s"]
    rows = {n: m.constraints[n].n_rows for n in m.constraints if n != "balance"}
    assert rows == {"curve_tangent": 2, "curve_x_min": 1, "curve_x_max": 1}
    assert solved.dual("curve_tangent").to_dense().shape == (1, 2)


def test_tangent_solves_a_concave_curve_bounded_above():
    # revenue less 1.5 per unit is 5 at p = 10 on either segment
    G, B, xp, yp, m = one_generator([0.0, 20.0, 30.0], sense="max")
    p = m.var("p", (G,))
    r = m.var("r", (G,))
    m.piecewise(
        "curve",
        x=p[G],
        x_points=xp[G, B],
        y=r[G],
        y_points=yp[G, B],
        sign="<=",
        method="tangent",
    )
    m.set_objective(no.Sum(G, r[G]) - 1.5 * no.Sum(G, p[G]))
    assert m.solve().objective == pytest.approx(5.0)


def test_incremental_equality_pins_y_to_the_curve():
    # y == f(x) with f not convex; minimizing y - x picks x = 10, y = 5
    G, B, xp, yp, m = one_generator([0.0, 5.0, 30.0])
    x = m.var("x", (G,))
    y = m.var("y", (G,))
    m.piecewise(
        "curve",
        x=x[G],
        x_points=xp[G, B],
        y=y[G],
        y_points=yp[G, B],
        sign="==",
        method="incremental",
    )
    m.set_objective(no.Sum(G, y[G]) - no.Sum(G, x[G]))
    solved = m.solve()
    assert solved.objective == pytest.approx(-5.0)
    assert solved.primal("x").to_dense().tolist() == [10.0]


def test_decreasing_breakpoints_describe_the_same_curve():
    rising = cost_model()
    xs = [
        ("a", "b0", 30),
        ("a", "b1", 20),
        ("a", "b2", 10),
        ("a", "b3", 0),
        ("b", "b0", 20),
        ("b", "b1", 10),
        ("b", "b2", 0),
    ]
    ys = [
        ("a", "b0", 35),
        ("a", "b1", 30),
        ("a", "b2", 5),
        ("a", "b3", 0),
        ("b", "b0", 30),
        ("b", "b1", 20),
        ("b", "b2", 0),
    ]
    falling = cost_model(xs=xs, ys=ys)
    assert falling.solve().objective == pytest.approx(rising.solve().objective)


def call_error(message, error=ValueError, **changes):
    with pytest.raises(error, match=re.escape(message)):
        cost_model(**changes)


def test_a_name_that_is_not_an_identifier_raises():
    m = no.Model("m")
    G = no.Set("G", np.array(["a"]))
    B = no.Set("B", np.array(["b0", "b1"]))
    x = m.var("x", (G,))
    xp = no.Param.from_dense("xp", (G, B), [[0.0, 1.0]])
    with pytest.raises(ValueError, match="piecewise name 'cost curve' is not"):
        m.piecewise("cost curve", x[G], xp[G, B], x[G], xp[G, B], "==", "incremental")


def test_an_unknown_method_raises():
    call_error(
        "method of piecewise 'curve' is one of ('incremental', 'tangent'); got 'sos2'",
        method="sos2",
    )


def test_an_unknown_sign_raises():
    call_error(
        "sign of piecewise 'curve' is one of ('==', '<=', '>='); got '='", sign="="
    )


def test_tangent_with_equality_raises():
    call_error(
        "piecewise 'curve' uses method 'tangent' with sign '=='; use sign "
        "'<=' or '>=', or method 'incremental'",
        method="tangent",
        sign="==",
    )


def test_tangent_with_active_raises():
    G, B, xp, yp, m = one_generator([0.0, 10.0, 30.0])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    u = m.var("u", (G,), upper=1.0, integer=True)
    with pytest.raises(ValueError, match="uses method 'tangent' with active"):
        m.piecewise("curve", p[G], xp[G, B], c[G], yp[G, B], ">=", "tangent", u[G])


def test_expressions_over_different_sets_raise():
    G, B, xp, yp, m = one_generator([0.0, 10.0, 30.0])
    T = no.Set("T", np.array(["t0"]))
    p = m.var("p", (G, T))
    c = m.var("c", (G,))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "y of piecewise 'curve' is over ('G',) and x is over ('G', 'T'); "
            "give both over the same sets"
        ),
    ):
        m.piecewise("curve", p[G, T], xp[G, B], c[G], yp[G, B], ">=", "incremental")


def test_points_without_one_breakpoint_set_raise():
    G, B, xp, yp, m = one_generator([0.0, 10.0, 30.0])
    p = m.var("p", (G, B))
    c = m.var("c", (G, B))
    with pytest.raises(ValueError, match="give x_points exactly one set that x"):
        m.piecewise("curve", p[G, B], xp[G, B], c[G, B], yp[G, B], ">=", "tangent")


def test_points_over_different_sets_raise():
    G, B, xp, _, m = one_generator([0.0, 10.0, 30.0])
    C = no.Set("C", np.array(["c0", "c1", "c2"]))
    other = no.Param.from_dense("other", (G, C), [[0.0, 10.0, 30.0]])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "y_points of piecewise 'curve' is over ('G', 'C') and x_points is "
            "over ('G', 'B'); give both over the same sets"
        ),
    ):
        m.piecewise("curve", p[G], xp[G, B], c[G], other[G, C], ">=", "tangent")


def test_a_bare_parameter_as_points_raises():
    G, B, xp, yp, m = one_generator([0.0, 10.0, 30.0])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    with pytest.raises(
        TypeError,
        match="x_points of piecewise 'curve' is a parameter read at its sets; "
        "got Param",
    ):
        m.piecewise("curve", p[G], xp, c[G], yp[G, B], ">=", "tangent")


def test_a_number_as_x_raises():
    G, B, xp, yp, m = one_generator([0.0, 10.0, 30.0])
    c = m.var("c", (G,))
    with pytest.raises(TypeError, match="x of piecewise 'curve' is an expression"):
        m.piecewise("curve", 3.0, xp[G, B], c[G], yp[G, B], ">=", "tangent")


def test_a_second_declaration_of_one_name_raises():
    G, B, xp, yp, m = one_generator([0.0, 10.0, 30.0])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    m.piecewise("curve", p[G], xp[G, B], c[G], yp[G, B], ">=", "tangent")
    with pytest.raises(ValueError, match="piecewise 'curve' is already declared"):
        m.piecewise("curve", p[G], xp[G, B], c[G], yp[G, B], ">=", "tangent")


def test_a_generated_name_the_model_declares_raises_before_any_declaration():
    G, B, xp, yp, m = one_generator([0.0, 10.0, 30.0])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    m.var("curve_fill", (G,))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "piecewise 'curve' generates ['curve_fill'], already declared in "
            "model 'one'; rename the piecewise declaration"
        ),
    ):
        m.piecewise("curve", p[G], xp[G, B], c[G], yp[G, B], ">=", "incremental")
    assert list(m.variables) == ["p", "c", "curve_fill"]
    assert m.piecewise_declarations == {}


def data_error(message, **changes):
    with pytest.raises(ValueError, match=re.escape(message)):
        cost_model(**changes)


def test_points_present_at_different_breakpoints_raise():
    ys = [
        ("a", "b0", 0),
        ("a", "b1", 5),
        ("a", "b2", 30),
        ("b", "b0", 0),
        ("b", "b1", 20),
        ("b", "b2", 30),
    ]
    data_error(
        "piecewise 'curve' has x_points and y_points present at different "
        "breakpoints at {'G': 'a'}; give both a value at the same breakpoints",
        ys=ys,
    )


def test_an_entity_with_one_breakpoint_raises():
    xs = [("a", "b0", 0), ("b", "b0", 0), ("b", "b1", 10)]
    ys = [("a", "b0", 0), ("b", "b0", 0), ("b", "b1", 20)]
    data_error(
        "piecewise 'curve' has one breakpoint at {'G': 'a'}; give two or more",
        xs=xs,
        ys=ys,
    )


def test_an_absent_breakpoint_before_a_present_one_raises():
    xs = [("a", "b0", 0), ("a", "b2", 10), ("b", "b0", 0), ("b", "b1", 10)]
    ys = [("a", "b0", 0), ("a", "b2", 20), ("b", "b0", 0), ("b", "b1", 20)]
    data_error(
        "piecewise 'curve' has an absent breakpoint before a present one at "
        "{'G': 'a'}; leave only the last breakpoints absent",
        xs=xs,
        ys=ys,
    )


def test_breakpoints_that_are_not_monotonic_raise():
    xs = [("a", "b0", 0), ("a", "b1", 20), ("a", "b2", 10)]
    ys = [("a", "b0", 0), ("a", "b1", 20), ("a", "b2", 30)]
    data_error(
        "piecewise 'curve' has x_points that are not strictly monotonic at "
        "{'G': 'a'}; give strictly increasing or strictly decreasing "
        "breakpoints; nimopt does not support special ordered sets",
        xs=xs,
        ys=ys,
    )


def test_a_breakpoint_that_is_not_finite_raises():
    xs = [("a", "b0", 0), ("a", "b1", np.inf)]
    ys = [("a", "b0", 0), ("a", "b1", 20)]
    data_error(
        "x_points of piecewise 'curve' contains a value that is not finite; "
        "give finite breakpoints",
        xs=xs,
        ys=ys,
    )


def test_points_with_no_breakpoint_raise():
    G, B, _, _, m = one_generator([0.0, 10.0, 30.0])
    empty = no.Param.from_long(
        "empty", (G, B), {"G": np.array([], str), "B": np.array([], str)}, []
    )
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    with pytest.raises(ValueError, match="piecewise 'curve' has no breakpoint"):
        m.piecewise("curve", p[G], empty[G, B], c[G], empty[G, B], ">=", "tangent")


def test_tangent_on_a_curve_that_is_not_convex_raises():
    data_error(
        "piecewise 'curve' has points that are not convex, required by sign "
        "'>=' at {'G': 'a'}; use method 'incremental'",
        method="tangent",
    )


def test_tangent_on_a_curve_that_is_not_concave_raises():
    data_error(
        "piecewise 'curve' has points that are not concave, required by sign "
        "'<=' at {'G': 'a'}; use method 'incremental'",
        method="tangent",
        sign="<=",
    )


def test_a_failed_check_leaves_the_model_unchanged():
    G, B, _, _, m = one_generator([0.0, 10.0, 30.0])
    bad = no.Param.from_dense("bad", (G, B), [[0.0, 20.0, 10.0]])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    with pytest.raises(ValueError, match="not strictly monotonic"):
        m.piecewise("curve", p[G], bad[G, B], c[G], bad[G, B], ">=", "incremental")
    assert list(m.variables) == ["p", "c"]
    assert list(m.constraints) == []
    assert m.piecewise_declarations == {}


def test_incremental_reads_a_constant_in_x():
    # x is p + 5 with p fixed at 5, so x is 10, the second breakpoint, and
    # the curve is 5 there
    G, B, xp, yp, m = one_generator([0.0, 5.0, 20.0])
    p = m.var("p", (G,), lower=5.0, upper=5.0)
    c = m.var("c", (G,))
    m.piecewise("curve", p[G] + 5.0, xp[G, B], c[G], yp[G, B], ">=", "incremental")
    m.set_objective(no.Sum(G, c[G]))
    assert m.solve().objective == pytest.approx(5.0)


def test_tangent_with_a_constant_in_x_raises():
    G, B, xp, yp, m = one_generator([0.0, 5.0, 20.0])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "x of piecewise 'curve' has the constant 5.0 and the method is "
            "'tangent'; subtract the constant from x_points, or use method "
            "'incremental'"
        ),
    ):
        m.piecewise("curve", p[G] + 5.0, xp[G, B], c[G], yp[G, B], ">=", "tangent")


def test_an_active_expression_with_a_constant_raises():
    G, B, xp, yp, m = one_generator([0.0, 5.0, 20.0])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    u = m.var("u", (G,), upper=1.0, integer=True)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "active of piecewise 'curve' has the constant 1.0; give active as "
            "an expression with no constant"
        ),
    ):
        m.piecewise(
            "curve", p[G], xp[G, B], c[G], yp[G, B], ">=", "incremental", u[G] + 1.0
        )


def test_points_declared_over_the_same_sets_in_another_order_agree():
    G, B, xp, yp, m = one_generator([0.0, 5.0, 20.0])
    other = no.Param.from_dense("other", (B, G), [[0.0], [5.0], [20.0]])

    def solved(points):
        held = no.Model("order")
        p = held.var("p", (G,), lower=15.0, upper=15.0)
        c = held.var("c", (G,))
        held.piecewise("curve", p[G], xp[G, B], c[G], points, ">=", "tangent")
        held.set_objective(no.Sum(G, c[G]))
        return held.solve().objective

    assert solved(other[B, G]) == pytest.approx(solved(yp[G, B]))


def test_one_curve_covers_every_entity_where_the_points_have_no_entity_set():
    # the points are over the breakpoints alone, and both generators read the
    # same curve: 5 at 10 and 1.5 per unit above it
    G = no.Set("G", np.array(["a", "b"]))
    B = no.Set("B", np.array(["b0", "b1", "b2"]))
    xp = no.Param.from_dense("xp", (B,), [0.0, 10.0, 20.0])
    yp = no.Param.from_dense("yp", (B,), [0.0, 5.0, 20.0])
    m = no.Model("shared")
    p = m.var("p", (G,), lower=15.0, upper=15.0)
    c = m.var("c", (G,))
    m.piecewise("curve", p[G], xp[B], c[G], yp[B], ">=", "tangent")
    m.set_objective(no.Sum(G, c[G]))
    assert m.solve().objective == pytest.approx(25.0)
    assert m.constraints["curve_tangent"].n_rows == 4


def test_a_curve_over_no_set_relates_two_scalar_variables():
    B = no.Set("B", np.array(["b0", "b1", "b2"]))
    xp = no.Param.from_dense("xp", (B,), [0.0, 10.0, 20.0])
    yp = no.Param.from_dense("yp", (B,), [0.0, 5.0, 20.0])
    for method, rows in (("tangent", 4), ("incremental", 6)):
        m = no.Model(method)
        x = m.var("x", (), lower=15.0, upper=15.0)
        y = m.var("y", ())
        m.piecewise("curve", x[()], xp[B], y[()], yp[B], ">=", method)
        m.set_objective(y[()])
        assert m.solve().objective == pytest.approx(12.5), method
        assert m.n_rows == rows, method


def partial_domain_model(same_domain):
    """A curve of slope 2 where y is declared over fewer members than x.

    Demand of 12 in each period costs 48.0 when the relation is enforced at
    every member of x. Where y is absent the y row is dropped and production
    there is free.
    """
    G = no.Set("G", np.array(["a", "b"]))
    T = no.Set("T", np.array(["t0", "t1"]))
    B = no.Set("B", np.array(["b0", "b1"]))
    xp = table(
        G, B, [(g, b, v) for g in "ab" for b, v in (("b0", 0), ("b1", 20))], "xp"
    )
    yp = table(
        G, B, [(g, b, v) for g in "ab" for b, v in (("b0", 0), ("b1", 40))], "yp"
    )
    demand = no.Param.from_dense("demand", (T,), np.array([12.0, 12.0]))
    m = no.Model("partial")
    members = {"G": np.array(["a", "a", "b"]), "T": np.array(["t0", "t1", "t0"])}
    held = no.subset((G, T), members)
    p = m.var("p", (G, T), subset=held if same_domain else None, upper=20.0)
    c = m.var("c", (G, T), subset=held, lower=-1e4)
    m.constraint("balance", no.Sum(G, p[G, T]) >= demand[T])
    m.piecewise(
        "curve",
        x=p[G, T],
        x_points=xp[G, B],
        y=c[G, T],
        y_points=yp[G, B],
        sign=">=",
        method="incremental",
    )
    m.set_objective(no.Sum(G, T, c[G, T]))
    return m


def test_a_curve_whose_y_is_over_fewer_members_than_x_raises():
    # x has a column at (b, t1) and y has none, so the curve there relates a
    # column to nothing. The declaration is ambiguous and is rejected.
    with pytest.raises(ValueError, match="curve"):
        partial_domain_model(same_domain=False)


def test_a_curve_over_one_subset_is_enforced_at_every_member_of_it():
    # the same declaration with x and y over the same members: three of the
    # four member pairs, and the relation binds at each one
    solved = partial_domain_model(same_domain=True).solve()
    assert solved.objective == pytest.approx(48.0)


@pytest.mark.parametrize(
    "name", ["curve_segment", "curve_members", "curve_fill", "curve_x"]
)
def test_a_name_a_declaration_generated_is_not_declared_after_it(name):
    # the reverse order raises already. A model file writes a set, a
    # parameter, a variable and a constraint under four keys and reads the
    # first three into one table of symbols, so a generated name a variable
    # takes later is a file that cannot be read back.
    G, B, xp, yp, m = one_generator([0.0, 10.0, 30.0])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    m.piecewise("curve", p[G], xp[G, B], c[G], yp[G, B], ">=", "incremental")
    with pytest.raises(ValueError, match=re.escape(f"{name!r}")):
        m.var(name, (G,))


def test_a_constraint_does_not_take_a_name_a_declaration_generated():
    G, B, xp, yp, m = one_generator([0.0, 10.0, 30.0])
    p = m.var("p", (G,))
    c = m.var("c", (G,))
    m.piecewise("curve", p[G], xp[G, B], c[G], yp[G, B], ">=", "incremental")
    with pytest.raises(ValueError, match=re.escape("'curve_x'")):
        m.constraint("curve_x", p[G] >= 0.0)


def curve_with_active(active_of):
    """A curve switched by `active_of(model, G)`, over one generator."""
    G, B, xp, yp, m = one_generator([20.0, 35.0, 80.0])
    p = m.var("p", (G,), upper=20.0)
    c = m.var("c", (G,), lower=-1e4)
    m.piecewise(
        "curve",
        p[G],
        xp[G, B],
        c[G],
        yp[G, B],
        ">=",
        "incremental",
        active_of(m, G),
    )
    return m


def test_a_continuous_active_is_not_a_switch():
    # a value between 0 and 1 scales every breakpoint, so the curve is met at
    # a fraction of its first breakpoint and at a fraction of its cost
    with pytest.raises(ValueError, match="active"):
        curve_with_active(lambda m, G: m.var("u", (G,), upper=1.0)[G])


def test_an_integer_active_above_one_is_not_a_switch():
    with pytest.raises(ValueError, match="active"):
        curve_with_active(lambda m, G: m.var("u", (G,), upper=5.0, integer=True)[G])


def test_a_scaled_active_is_not_a_switch():
    with pytest.raises(ValueError, match="active"):
        curve_with_active(
            lambda m, G: 2.0 * m.var("u", (G,), upper=1.0, integer=True)[G]
        )


def test_a_binary_active_switches_the_curve():
    m = curve_with_active(lambda m, G: m.var("u", (G,), upper=1.0, integer=True)[G])
    assert "curve_active" in m.piecewise_declarations["curve"].generated["constraints"]
