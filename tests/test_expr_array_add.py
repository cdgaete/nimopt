"""Regression tests for Bug 5: adding a bare param/nb.Array to a LinearExpr.

`LinearExpr.__add__` accepted int/float, LinearExpr and `.array`-bearing objects
but not a raw `nb.Array`, so `param + var` (param on the left, which routes a raw
nb.Array into `LinearExpr + nb.Array`) raised TypeError. `__sub__` was worse: it
handled only int/float and LinearExpr and had no trailing `return NotImplemented`,
so `LinearExpr - nb.Array` fell off the end and silently returned None.
"""

import numpy as np

import nimopt as no
from nimopt.expression import LinearExpr


def _model():
    I = no.Set("I", ["a", "b"])
    m = no.Model()
    x = m.var("x", [I])
    p = no.Param("p", [I], [1.0, 2.0])
    return I, x, p


def test_param_plus_var_param_on_left():
    """param + var (param on the left) must build a LinearExpr, not TypeError."""
    I, x, p = _model()
    expr = p[I] + x[I]
    assert isinstance(expr, LinearExpr)


def test_linexpr_plus_raw_array():
    I, x, p = _model()
    expr = x[I] + p[I]  # LinearExpr
    out = expr + p.array  # LinearExpr + raw nb.Array
    assert isinstance(out, LinearExpr)


def test_linexpr_minus_array_not_none():
    """LinearExpr - nb.Array must not silently return None."""
    I, x, p = _model()
    expr = x[I] + p[I]  # LinearExpr
    out = expr - p.array  # LinearExpr - raw nb.Array
    assert out is not None
    assert isinstance(out, LinearExpr)
