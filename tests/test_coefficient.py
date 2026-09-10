import numpy as np
import pytest

from nimopt import Definition, Model, Param, Set, Sum


def fleet():
    G = Set("G", np.array(["base", "peak"]))
    T = Set("T", np.arange(3))
    price = Param.from_dense("fuel_price", (G, T), np.full((2, 3), 30.0))
    eta = Param.from_dense("efficiency", (G, T), np.array([[0.5] * 3, [0.4] * 3]))
    return G, T, price, eta


def test_a_coefficient_states_its_dimensions_without_materialising():
    G, T, price, eta = fleet()
    unit_cost = price[G, T] / eta[G, T]
    assert unit_cost.dims == ("G", "T")
    assert unit_cost.name == "(fuel_price / efficiency)"


def test_a_coefficient_is_read_at_its_sets_as_a_parameter_is():
    G, T, price, eta = fleet()
    unit_cost = price[G, T] / eta[G, T]
    assert unit_cost[G, T].dims == ("G", "T")
    with pytest.raises(ValueError, match=r"is over \('G', 'T'\)"):
        unit_cost[T, G]
    with pytest.raises(ValueError, match=r"is over \('G', 'T'\)"):
        unit_cost[G]


def test_a_derived_coefficient_builds_the_model_its_arithmetic_states():
    G, T, price, eta = fleet()
    load = Param.from_dense("load", (T,), np.full(3, 100.0))
    cap = Param.from_dense("capacity", (G, T), np.full((2, 3), 80.0))
    unit_cost = price[G, T] / eta[G, T]

    m = Model("dispatch", sense="min")
    gen = m.var("gen", (G, T), lower=0.0, upper=cap)
    m.constraint("balance", Sum(G, gen[G, T]) == load[T])
    m.set_objective(Sum(G, T, unit_cost[G, T] * gen[G, T]))
    answer = m.solve()
    assert answer.status == "optimal"
    # base runs at 30/0.5 = 60, peak at 30/0.4 = 75; capacity 80 each,
    # load 100 an hour for three hours
    assert answer.objective == pytest.approx(3 * (80 * 60.0 + 20 * 75.0))


@pytest.mark.parametrize(
    "form,expected",
    [
        ("a[G, T] + b[G, T]", "(a + b)"),
        ("a[G, T] - b[G, T]", "(a - b)"),
        ("a[G, T] * b[G, T]", "(a * b)"),
        ("a[G, T] / b[G, T]", "(a / b)"),
        ("-a[G, T]", "(-a)"),
        ("a[G, T] * 2.0", "(a * 2.0)"),
        ("2.0 * a[G, T]", "(a * 2.0)"),
        ("a[G, T] ** 2", "(a ** 2)"),
    ],
)
def test_each_combination_names_itself(form, expected):
    G, T, _, _ = fleet()
    named = {
        "G": G,
        "T": T,
        "a": Param.from_dense("a", (G, T), np.full((2, 3), 4.0)),
        "b": Param.from_dense("b", (G, T), np.full((2, 3), 2.0)),
    }
    assert eval(form, named).name == expected  # noqa: S307


def test_a_coefficient_over_a_wider_frame_aligns_on_what_they_share():
    G, T, _, _ = fleet()
    per_unit = Param.from_dense("per_unit", (G,), np.array([2.0, 5.0]))
    grid = Param.from_dense("grid", (G, T), np.ones((2, 3)))
    assert (grid[G, T] * per_unit[G]).dims == ("G", "T")
    assert (grid[G, T] / per_unit[G]).dims == ("G", "T")


def test_two_coefficients_sharing_no_dimension_are_refused():
    G, T, _, _ = fleet()
    over_g = Param.from_dense("over_g", (G,), np.ones(2))
    over_t = Param.from_dense("over_t", (T,), np.ones(3))
    with pytest.raises(ValueError, match="share no dimension"):
        over_g[G] * over_t[T]


def test_a_divisor_carrying_a_zero_raises_where_python_raises():
    G, T, price, _ = fleet()
    holed = Param.from_dense("holed", (G, T), np.array([[0.5, 0.0, 0.5], [0.4] * 3]))
    with pytest.raises(ZeroDivisionError, match="holed"):
        (price[G, T] / holed[G, T]).materialise()


@pytest.mark.parametrize("zero", [0.0, 0, np.float64(0.0)])
def test_a_divisor_that_is_zero_raises_however_it_is_spelled(zero):
    # a numpy scalar divides to infinity rather than raising, so the check is
    # stated rather than left to the arithmetic
    G, T, price, _ = fleet()
    with pytest.raises(ZeroDivisionError):
        price[G, T] / zero


def test_a_term_carrying_two_coefficients_multiplies_them():
    G, T, price, eta = fleet()
    assert (price[G, T] * (eta[G, T] * _gen(G, T))).terms[0].coefficient.dims == (
        "G",
        "T",
    )


def _gen(G, T):
    m = Model("m")
    gen = m.var("gen", (G, T))
    return gen[G, T]


def test_a_coefficient_never_adds_a_column_a_variable_excluded():
    # the variable's members are the model's columns; a coefficient
    # multiplying them can only narrow what survives
    P = Set("P", np.array(["a", "b"]))
    Q = Set("Q", np.array(["x", "y", "z"]))
    arcs = Param.from_long(
        "arcs",
        (P, Q),
        {"P": np.array(["a", "b"]), "Q": np.array(["x", "z"])},
        np.ones(2),
    )
    m = Model("m")
    v = m.var("v", (P, Q), subset=arcs)
    dense = Param.from_dense("dense", (P, Q), np.ones((2, 3)))
    term = (dense[P, Q] * v[P, Q]).terms[0]
    block = term.materialise(("P", "Q"), {"P": P.coord, "Q": Q.coord})
    assert v.n_columns == 2
    assert block.nnz == 2


def test_a_coefficient_is_written_before_its_data_exists():
    d = Definition("dispatch", sense="min")
    G, T = d.set("G"), d.set("T")
    price = d.param("fuel_price", (G, T))
    eta = d.param("efficiency", (G, T))
    load = d.param("load", (T,))
    cap = d.param("capacity", (G, T))
    gen = d.var("gen", (G, T), lower=0.0, upper=cap)
    unit_cost = price[G, T] / eta[G, T]
    d.constraint("balance", Sum(G, gen[G, T]) == load[T])
    d.set_objective(Sum(G, T, unit_cost[G, T] * gen[G, T]))

    assert "(fuel_price[G, T] / efficiency[G, T])" in repr(d.explain())
    inputs = {
        "G": np.array(["base", "peak"]),
        "T": np.arange(3),
        "fuel_price": np.full((2, 3), 30.0),
        "efficiency": np.array([[0.5] * 3, [0.4] * 3]),
        "load": np.full(3, 100.0),
        "capacity": np.full((2, 3), 80.0),
    }
    assert d.build(inputs).solve().objective == pytest.approx(18900.0)
    worse = dict(inputs, efficiency=np.array([[0.25] * 3, [0.4] * 3]))
    # base at 30/0.25 = 120 is dearer than peak at 75, so the order flips
    assert d.build(worse).solve().objective == pytest.approx(25200.0)


def test_a_power_takes_a_number_and_not_a_coefficient():
    G, T, price, eta = fleet()
    with pytest.raises(TypeError, match="a power takes a number"):
        price[G, T] ** eta[G, T]


def test_a_bare_parameter_names_the_reading_its_arithmetic_needs():
    G, T, price, eta = fleet()
    with pytest.raises(TypeError, match=r"read it at its sets"):
        price * 2.0
    with pytest.raises(TypeError, match=r"read it at its sets"):
        price + eta


def test_a_model_reports_every_parameter_a_derived_coefficient_reads():
    # what is reported is read from what was built: a coefficient standing
    # for two parameters accounts for both, and for the sets they introduce
    G, T, price, eta = fleet()
    load = Param.from_dense("load", (T,), np.full(3, 100.0))
    m = Model("dispatch", sense="min")
    gen = m.var("gen", (G, T), upper=80.0)
    m.constraint("balance", Sum(G, gen[G, T]) == load[T])
    m.set_objective(Sum(G, T, (price[G, T] / eta[G, T])[G, T] * gen[G, T]))
    found = m.explain()
    assert [p.name for p in found.parameters] == ["load", "fuel_price", "efficiency"]
    assert [s.name for s in found.sets] == ["G", "T"]
    assert found.objective == (
        "Sum(G, T, (fuel_price[G, T] / efficiency[G, T]) * gen[G, T])"
    )


def test_everything_a_term_reads_as_a_coefficient_answers_one_surface():
    # `Absence`, `Explanation` and `Row` each read a coefficient's name; the
    # surface they read is stated here rather than in prose
    from nimopt import Coefficient

    G, T, price, eta = fleet()
    for held in (price[G, T], price[G, T] / eta[G, T], (price[G, T] * eta[G, T])[G, T]):
        assert isinstance(held, Coefficient)
        assert isinstance(held.name, str)
        assert held.dims == ("G", "T")
        assert held.materialise().dims == ("G", "T")
        assert all(isinstance(p, Param) for p in held.parameters())


def test_a_coefficient_reads_the_same_term_from_either_side():
    G, T, price, _ = fleet()
    gen = _gen(G, T)
    left = (price[G, T] * gen).terms[0]
    right = (gen * price[G, T]).terms[0]
    assert left.coefficient.name == right.coefficient.name
    assert left.variable is right.variable


def test_a_variable_divided_by_a_coefficient_is_the_reciprocal_times_it():
    G, T, price, eta = fleet()
    load = Param.from_dense("load", (T,), np.full(3, 100.0))
    m = Model("m", sense="min")
    gen = m.var("gen", (G, T), lower=0.0, upper=200.0)
    # meeting the load with the energy each unit delivers per unit of fuel
    m.constraint("balance", Sum(G, gen[G, T] / eta[G, T]) == load[T])
    m.set_objective(Sum(G, T, price[G, T] * gen[G, T]))
    assert m.solve().status == "optimal"


def test_a_variable_divided_by_a_number_scales_it():
    G, T, _, _ = fleet()
    assert (_gen(G, T) / 2.0).terms[0].scale == pytest.approx(0.5)


@pytest.mark.parametrize("zero", [0.0, 0, np.float64(0.0)])
def test_a_variable_divided_by_zero_raises_however_the_zero_is_spelled(zero):
    # one exception for every divisor that is zero, so a caller reads the
    # error Python already means by it
    G, T, _, _ = fleet()
    with pytest.raises(ZeroDivisionError):
        _gen(G, T) / zero


def test_a_variable_in_a_denominator_is_refused():
    G, T, price, _ = fleet()
    gen = _gen(G, T)
    for form in (lambda: 2.0 / gen, lambda: price[G, T] / gen, lambda: gen / gen):
        with pytest.raises(TypeError, match="linear"):
            form()
    # the divisor is an expression; the message calls it a variable
    with pytest.raises(TypeError, match="by a variable: expressions are"):
        gen / gen


def test_a_divisor_that_is_not_a_number_reports_its_type():
    # a str is neither a number nor a Coefficient; a message calling it a
    # variable would describe the wrong mistake
    G, T, _, _ = fleet()
    gen = _gen(G, T)
    with pytest.raises(TypeError, match="by a str; divide by a number"):
        gen / "a"
    with pytest.raises(TypeError, match="by a list; divide by a number"):
        gen / [1, 2]
