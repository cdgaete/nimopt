import nimblend as nb
import numpy as np

import nimopt as no
from nimopt.piecewise import _bounds_of


def test_bounds_of_reduces_a_variables_bounds_through_nimblend_min_and_max(
    monkeypatch,
):
    # floor and ceiling are Params, so bound_array returns a SparseArray for
    # both "lower" and "upper"; _bounds_of must call nimblend to reduce them.
    G = no.Set("G", np.array(["a", "b", "c"]))
    floor = no.Param.from_dense("floor", (G,), np.array([1.0, 2.0, 0.5]))
    ceiling = no.Param.from_dense("ceiling", (G,), np.array([3.0, 7.0, 2.0]))
    m = no.Model("bounds")
    u = m.var("u", (G,), lower=floor, upper=ceiling)

    calls = []
    original_min = nb.SparseArray.min
    original_max = nb.SparseArray.max

    def spy_min(self, *args, **kwargs):
        calls.append("min")
        return original_min(self, *args, **kwargs)

    def spy_max(self, *args, **kwargs):
        calls.append("max")
        return original_max(self, *args, **kwargs)

    monkeypatch.setattr(nb.SparseArray, "min", spy_min)
    monkeypatch.setattr(nb.SparseArray, "max", spy_max)

    low, high = _bounds_of(u)

    assert calls == ["min", "max"]
    assert (low, high) == (0.5, 7.0)
