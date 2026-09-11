import re

import numpy as np
import pytest

import nimopt as no

pytest.importorskip("highspy")


def model(name="t"):
    P = no.Set("P", np.array(["a", "b"]))
    m = no.Model(name)
    x = m.var("x", (P,))
    m.constraint("cap", x[P] <= 1.0)
    m.set_objective(no.Sum(P, x[P]))
    return m


def raises(message):
    return pytest.raises(KeyError, match=re.escape(message))


def test_an_unknown_variable_raises_with_the_declared_variables():
    with raises("model 't' has no variable 'nosuch'; use one of ('x',)"):
        model().solve().primal("nosuch")


def test_a_constraint_read_as_a_variable_raises_with_the_method_to_use():
    with raises("'cap' is a constraint, not a variable; read it with dual()"):
        model().solve().primal("cap")


def test_an_unknown_constraint_raises_with_the_declared_constraints():
    m = model()
    message = "model 't' has no constraint 'nosuch'; use one of ('cap',)"
    with raises(message):
        m.solve().dual("nosuch")
    with raises(message):
        m.row("nosuch")
    with raises(message):
        m.absent("nosuch")


def test_a_variable_read_as_a_constraint_raises_with_the_method_to_use():
    m = model()
    with raises("'x' is a variable, not a constraint; read it with primal()"):
        m.solve().dual("x")
    # row and absent have no counterpart for a variable
    with raises("'x' is a variable, not a constraint; use one of ('cap',)"):
        m.row("x", P="a")
    with raises("'x' is a variable, not a constraint; use one of ('cap',)"):
        m.absent("x")


def test_an_unknown_constraint_of_an_assembled_matrix_raises():
    with raises("assembled matrix has no constraint 'nosuch'; use one of ('cap',)"):
        model().assemble().row_of("nosuch")


def test_an_unknown_name_raises_before_the_status_is_checked():
    P = no.Set("P", np.array(["a", "b"]))
    m = no.Model("short")
    x = m.var("x", (P,))
    m.constraint("cap", x[P] <= 1.0)
    m.constraint("need", no.Sum(P, x[P]) >= 5.0)
    m.set_objective(no.Sum(P, x[P]))
    s = m.solve()
    assert s.status == "infeasible"
    with raises("model 'short' has no variable 'y'; use one of ('x',)"):
        s.primal("y")
    with raises("model 'short' has no constraint 'y'; use one of ('cap', 'need')"):
        s.dual("y")


def transport_like(name="t"):
    P = no.Set("P", np.array(["a", "b"]))
    m = no.Model(name)
    x = m.var("x", (P,))
    m.constraint("cap", x[P] <= 1.0)
    m.set_objective(no.Sum(P, x[P]))
    return m, P, x


def test_a_name_declared_after_the_solve_is_not_in_the_solution():
    m, P, x = transport_like()
    s = m.solve()
    late = m.var("late", (P,))
    m.constraint("later", late[P] <= 2.0)
    with raises("variable 'late' is not in the solved matrix; solve the model again"):
        s.primal("late")
    with raises(
        "constraint 'later' is not in the solved matrix; solve the model again"
    ):
        s.dual("later")


def test_an_empty_model_asks_for_a_declaration():
    m = no.Model("empty")
    with raises("model 'empty' has no variable 'x'; declare a variable first"):
        m.solve().primal("x")
    with raises("model 'empty' has no constraint 'c'; declare a constraint first"):
        m.row("c")
    with raises("model 'empty' has no constraint 'c'; declare a constraint first"):
        m.absent("c")
    with raises(
        "assembled matrix has no constraint 'c'; declare a constraint and "
        "assemble the model again"
    ):
        m.assemble().row_of("c")


def test_a_session_whose_model_changed_after_it_opened_raises():
    m, P, x = transport_like()
    message = re.escape(
        "model 't' declares variables or constraints that are not in the "
        "assembled matrix; open a new Session"
    )
    with m.session() as session:
        assert session.solve().status == "optimal"
        m.constraint("later", x[P] >= 0.0)
        with pytest.raises(ValueError, match=message):
            session.solve()
        with pytest.raises(ValueError, match=message):
            session.diagnose()


def test_a_variable_with_no_columns_declared_after_the_solve_raises():
    m, P, x = transport_like()
    s = m.solve()
    E = no.Set("E", np.array([], dtype=str))
    m.var("late", (E,))
    with raises("variable 'late' is not in the solved matrix; solve the model again"):
        s.primal("late")


def test_a_row_at_a_label_outside_its_set_identifies_the_set():
    m, P, x = transport_like()
    with raises(
        "member 'zz' is not in dimension 'P' of constraint 'cap'; pass a member of 'P'"
    ):
        m.row("cap", P="zz")
