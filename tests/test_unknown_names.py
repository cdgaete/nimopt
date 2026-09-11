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
