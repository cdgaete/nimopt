"""Models with singleton rows, with values computed by hand."""

import numpy as np

from nimopt import Model, Param, Set, Sum, product


def plant():
    """Two generators over three hours: 15 rows, 18 coefficients, optimum 43."""
    G = Set("G", np.array(["a", "b"]))
    T = Set("T", np.arange(3))
    cap = Param.from_dense("cap", (G, T), np.array([[5.0] * 3, [10.0] * 3]))
    cost = Param.from_dense("cost", (G,), np.array([1.0, 3.0]))
    d = Param.from_dense("d", (T,), np.array([8.0, 12.0, 3.0]))
    m = Model("plant")
    p = m.var("p", (G, T), lower=-np.inf)
    m.constraint("floor", p[G, T] >= 0.0)
    m.constraint("cap", p[G, T] <= cap[G, T])
    m.constraint("demand", Sum(G, p[G, T]) == d[T])
    m.set_objective(Sum(G, T, cost[G] * p[G, T]))
    return m


def mixed():
    """One constraint of six rows: the first has two coefficients, the others one."""
    G = Set("G", np.array(["a", "b"]))
    T = Set("T", np.arange(3))
    w = Param.from_long(
        "w", (G, T), {"G": np.array(["a"]), "T": np.array([0])}, np.array([2.0])
    )
    m = Model("mixed")
    p = m.var("p", (G, T), lower=-np.inf)
    s = m.var("s", (T,))
    m.constraint("mixed", p[G, T] + w[G, T] * s[T] <= 9.0, over=product((G, T)))
    return m


def neg_max():
    """`-2 x >= -8` under sense max: every row is a singleton row, optimum 12."""
    T = Set("T", np.arange(3))
    two = Param.from_dense("two", (T,), np.full(3, -2.0))
    m = Model("neg", sense="max")
    x = m.var("x", (T,), lower=-np.inf)
    m.constraint("neg", two[T] * x[T] >= -8.0)
    m.set_objective(Sum(T, x[T]))
    return m
