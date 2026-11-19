"""Two sets of one name with different labels raise where a model reads both."""

import numpy as np
import pytest

from nimopt import Model, Param, Set, Sum, load, save, subset

FORMS = ("parameter", "tuple", "domain")
READS = r"constraint 'c' reads different labels on \['A'\]"


def two_sets():
    """Return two sets named A with the labels a0 and a1 in opposite order."""
    return Set("A", np.array(["a0", "a1"])), Set("A", np.array(["a1", "a0"]))


def condition(form, held):
    """Return a condition over the set `held`, in one of the three forms."""
    if form == "parameter":
        return Param.from_long("q", (held,), {"A": ["a1"]}, [1.0])
    if form == "tuple":
        return (held,)
    return subset((held,), {"A": ["a1"]})


def test_a_bound_over_another_set_of_the_name_raises_at_the_declaration():
    A, other = two_sets()
    # cap is 10 at a1 and 20 at a0; read by position it bounds a0 at 10
    cap = Param.from_dense("cap", (other,), np.array([10.0, 20.0]))
    m = Model("bounds")
    with pytest.raises(
        ValueError,
        match=r"variable 'x' and parameter 'cap' have different labels on \['A'\]",
    ):
        m.var("x", (A,), upper=cap)
    assert m.n_columns == 0


def test_align_with_a_parameter_over_another_set_of_the_name_raises():
    A, other = two_sets()
    held = Param.from_dense("held", (other,), np.array([10.0, 20.0]))
    x = Model("align").var("x", (A,))
    with pytest.raises(ValueError, match="parameter 'held' have different labels"):
        x.align(held)


@pytest.mark.parametrize("change", ["fix", "set_bound"])
def test_a_session_change_with_a_parameter_over_another_set_raises(change):
    pytest.importorskip("highspy")
    A, other = two_sets()
    held = Param.from_dense("held", (other,), np.array([10.0, 20.0]))
    m = Model("session", sense="max")
    x = m.var("x", (A,), upper=100.0)
    m.set_objective(Sum(A, x[A]))
    with m.session() as session:
        with pytest.raises(ValueError, match="parameter 'held' have different labels"):
            if change == "fix":
                session.fix(x, held)
            else:
                session.set_bound(x, upper=held)
        # both columns keep the declared upper bound of 100
        assert session.solve().objective == pytest.approx(200.0)


@pytest.mark.parametrize("form", FORMS)
def test_a_subset_over_another_set_of_the_name_raises(form):
    A, other = two_sets()
    m = Model("subset")
    with pytest.raises(
        ValueError,
        match=r"variable 'x' and its subset= have different labels on \['A'\]",
    ):
        m.var("x", (A,), subset=condition(form, other))
    assert m.n_columns == 0


@pytest.mark.parametrize("form", FORMS)
def test_rows_declared_over_another_set_of_the_name_raise(form):
    A, other = two_sets()
    m = Model("rows")
    x = m.var("x", (A,))
    with pytest.raises(ValueError, match=READS):
        m.constraint("c", x[A] <= 1.0, over=condition(form, other))
    assert m.n_rows == 0


def test_a_sum_condition_over_another_set_of_the_name_raises():
    A, other = two_sets()
    m = Model("sum")
    x = m.var("x", (A,))
    with pytest.raises(ValueError, match=READS):
        m.constraint("c", Sum(A, x[A], where=condition("domain", other)) <= 1.0)


def test_a_coefficient_over_another_set_than_the_declared_rows_raises():
    A, other = two_sets()
    T = Set("T", np.array(["t0", "t1"]))
    # k introduces the dimension A, and the variable is over T alone
    k = Param.from_dense("k", (other,), np.array([10.0, 20.0]))
    m = Model("rows")
    y = m.var("y", (T,))
    with pytest.raises(ValueError, match=READS):
        m.constraint("c", k[other] * y[T] <= 1.0, over=(A, T))


@pytest.mark.parametrize(
    "part", ["coefficient", "right-hand side", "condition", "second variable"]
)
def test_a_constraint_over_two_sets_of_one_name_raises_with_its_name(part):
    A, other = two_sets()
    k = Param.from_dense("k", (other,), np.array([10.0, 20.0]))
    m = Model("parts")
    x = m.var("x", (A,))
    where = None
    if part == "coefficient":
        relation = k[other] * x[A] <= 1.0
    elif part == "right-hand side":
        relation = x[A] <= k[other]
    elif part == "condition":
        relation, where = x[A] <= 1.0, (other,)
    else:
        relation = x[A] + m.var("y", (other,))[other] <= 1.0
    with pytest.raises(ValueError, match=READS):
        m.constraint("c", relation, where=where)
    assert m.n_rows == 0


def test_a_set_read_at_one_fixed_member_is_compared():
    A = Set("A", np.array(["a0", "a1"]))
    T = Set("T", np.array(["t0", "t1"]))
    other = Set("T", np.array(["t1", "t0"]))
    p = Param.from_dense("p", (other,), np.array([10.0, 20.0]))
    m = Model("fixed")
    x = m.var("x", (A, T))
    # both readings fix T at t0, and T is no free dimension of the row
    with pytest.raises(
        ValueError, match=r"constraint 'c' reads different labels on \['T'\]"
    ):
        m.constraint("c", p["t0"] * x[A, "t0"] <= 1.0)
    assert m.n_rows == 0


def test_an_objective_over_two_sets_of_one_name_raises_where_it_is_set():
    A, other = two_sets()
    k = Param.from_dense("k", (other,), np.array([10.0, 20.0]))
    m = Model("objective")
    x = m.var("x", (A,))
    with pytest.raises(
        ValueError, match=r"the objective reads different labels on \['A'\]"
    ):
        m.set_objective(Sum(A, k[other] * x[A]))
    assert m.objective is None


def test_an_appended_row_over_two_sets_of_one_name_raises_with_its_name():
    pytest.importorskip("highspy")
    A, other = two_sets()
    k = Param.from_dense("k", (other,), np.array([10.0, 20.0]))
    m = Model("session", sense="max")
    x = m.var("x", (A,), upper=100.0)
    m.set_objective(Sum(A, x[A]))
    with m.session() as session:
        with pytest.raises(
            ValueError, match=r"constraint 'r' reads different labels on \['A'\]"
        ):
            session.add_row("r", x[A] <= k[other])
        assert session.n_rows == 0
        assert session.solve().objective == pytest.approx(200.0)


def test_two_sets_of_one_name_with_equal_labels_are_accepted():
    A = Set("A", np.array(["a0", "a1"]))
    same = Set("A", np.array(["a0", "a1"]))
    cap = Param.from_dense("cap", (same,), np.array([10.0, 20.0]))
    m = Model("equal")
    x = m.var("x", (A,), subset=(same,), upper=cap)
    m.constraint("c", x[A] <= 1.0, over=(same,))
    assert m.column_bounds()[1].tolist() == [10.0, 20.0]
    assert x.align(cap)[1].tolist() == [10.0, 20.0]
    assert m.n_rows == 2


def equal_sets_model():
    """Return a model over two sets named A with equal labels: optimum 30."""
    A = Set("A", np.array(["a0", "a1"]))
    same = Set("A", np.array(["a0", "a1"]))
    cap = Param.from_dense("cap", (same,), np.array([10.0, 20.0]))
    m = Model("equal", sense="max")
    x = m.var("x", (A,), upper=cap)
    m.set_objective(Sum(A, x[A]))
    return m


def test_explain_reports_two_sets_of_one_name_with_equal_labels_once():
    explanation = equal_sets_model().explain()
    assert [(held.name, held.size) for held in explanation.sets] == [("A", 2)]


def test_a_model_over_two_sets_of_one_name_with_equal_labels_is_saved(tmp_path):
    pytest.importorskip("highspy")
    path = tmp_path / "equal.yaml"
    save(equal_sets_model(), path)
    assert load(path).solve().objective == pytest.approx(30.0)
