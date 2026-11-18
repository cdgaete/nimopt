"""Models the session tests share."""

import numpy as np

from nimopt import Model, Param, Set, Sum


def two():
    """Two columns in [0, 10], rows total and floor_b, minimized: optimum 4."""
    i = Set("i", np.array(["a", "b"]))
    m = Model("two")
    x = m.var("x", (i,), lower=0.0, upper=10.0)
    m.constraint("total", Sum(i, x[i]) >= 4.0)
    m.constraint("floor_b", x["b"] >= 1.0)
    m.set_objective(Sum(i, x[i]))
    return m, i, x


def loop():
    """Three plants over twelve hours: 36 columns, 48 rows, optimum 31,500."""
    P = Set("P", np.array(["coal", "gas", "wind"]))
    H = Set("H", np.arange(1, 13))
    cost = Param.from_dense("cost", (P,), np.array([30.0, 60.0, 0.0]))
    cap = Param.from_dense(
        "cap", (P, H), np.repeat(np.array([[140.0], [80.0], [120.0]]), 12, axis=1)
    )
    need = Param.from_dense(
        "need", (H,), np.array([200.0 + 5.0 * (h % 4) for h in range(1, 13)])
    )
    m = Model("loop")
    g = m.var("g", (P, H), lower=0.0)
    m.set_objective(Sum(P, H, cost[P] * g[P, H]))
    m.constraint("capacity", g[P, H] <= cap[P, H])
    m.constraint("demand", Sum(P, g[P, H]) == need[H])
    return m, P, H, g


def products():
    """Three products sharing 6000 hours, maximized: optimum 180,000."""
    Pr = Set("Pr", np.array(["bands", "coils", "plate"]))
    profit = Param.from_dense("profit", (Pr,), np.array([25.0, 30.0, 29.0]))
    m = Model("products", sense="max")
    make = m.var("make", (Pr,), lower=0.0)
    m.constraint("time", Sum(Pr, make[Pr]) <= 6000.0)
    m.set_objective(Sum(Pr, profit[Pr] * make[Pr]))
    return m, Pr, make


def plant_model():
    """Two plants serving three weeks at 10 each, minimized: optimum 30."""
    P = Set("P", np.array(["p0", "p1"]))
    W = Set("W", np.array(["w0", "w1", "w2"]))
    cost = Param.from_dense("cost", (P,), np.array([5.0, 1.0]))
    m = Model("plant")
    x = m.var("x", (P, W), lower=0.0, upper=100.0)
    m.constraint("demand", Sum(P, x[P, W]) >= 10.0)
    m.set_objective(Sum(P, W, cost[P] * x[P, W]))
    return m, P, W, x
