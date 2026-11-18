"""Models the session tests share."""

import numpy as np

from nimopt import Model, Set, Sum


def two():
    """Two columns in [0, 10], rows total and floor_b, minimized: optimum 4."""
    i = Set("i", np.array(["a", "b"]))
    m = Model("two")
    x = m.var("x", (i,), lower=0.0, upper=10.0)
    m.constraint("total", Sum(i, x[i]) >= 4.0)
    m.constraint("floor_b", x["b"] >= 1.0)
    m.set_objective(Sum(i, x[i]))
    return m, i, x
