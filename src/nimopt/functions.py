"""
Mathematical functions for optimization modeling.

Includes Sum for vectorized reduction and math functions (sqrt, exp, log, abs)
that work with nimblend arrays and parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import nimblend as nb
import numpy as np

from .expression import LinearExpr
from .sets import Set

if TYPE_CHECKING:
    from .param import Param
    from .variable import VarRef


# =============================================================================
# Mathematical Functions (vectorized via nimblend/numpy)
# =============================================================================


def sqrt(x: Union["Param", nb.Array, float]) -> Union["Param", nb.Array, float]:
    """
    Square root - vectorized for parameters and arrays.

    Parameters
    ----------
    x : Param, nb.Array, or scalar
        Input value(s).

    Returns
    -------
    Same type as input with sqrt applied element-wise.

    Examples
    --------
    >>> eff = Param('eff', [storages], [0.81, 0.9])
    >>> sqrt_eff = sqrt(eff)  # Returns Param with [0.9, 0.949...]
    """
    from .param import Param

    if isinstance(x, Param):
        new_data = np.sqrt(x.values)
        return Param(f"sqrt_{x.name}", x.sets, new_data)
    elif isinstance(x, nb.Array):
        return nb.Array(np.sqrt(x.values), x.coords, x.dims, x.name)
    else:
        return np.sqrt(x)


def exp(x: Union["Param", nb.Array, float]) -> Union["Param", nb.Array, float]:
    """Exponential function - vectorized for parameters and arrays."""
    from .param import Param

    if isinstance(x, Param):
        new_data = np.exp(x.values)
        return Param(f"exp_{x.name}", x.sets, new_data)
    elif isinstance(x, nb.Array):
        return nb.Array(np.exp(x.values), x.coords, x.dims, x.name)
    else:
        return np.exp(x)


def log(x: Union["Param", nb.Array, float]) -> Union["Param", nb.Array, float]:
    """Natural logarithm - vectorized for parameters and arrays."""
    from .param import Param

    if isinstance(x, Param):
        new_data = np.log(x.values)
        return Param(f"log_{x.name}", x.sets, new_data)
    elif isinstance(x, nb.Array):
        return nb.Array(np.log(x.values), x.coords, x.dims, x.name)
    else:
        return np.log(x)


def abs_(x: Union["Param", nb.Array, float]) -> Union["Param", nb.Array, float]:
    """
    Absolute value - vectorized for parameters and arrays.

    Note: Named abs_ to avoid shadowing builtin abs().
    """
    from .param import Param

    if isinstance(x, Param):
        new_data = np.abs(x.values)
        return Param(f"abs_{x.name}", x.sets, new_data)
    elif isinstance(x, nb.Array):
        return nb.Array(np.abs(x.values), x.coords, x.dims, x.name)
    else:
        return np.abs(x)


def power(
    x: Union["Param", nb.Array, float], n: float
) -> Union["Param", nb.Array, float]:
    """Power function - vectorized for parameters and arrays."""
    from .param import Param

    if isinstance(x, Param):
        new_data = np.power(x.values, n)
        return Param(f"pow_{x.name}_{n}", x.sets, new_data)
    elif isinstance(x, nb.Array):
        return nb.Array(np.power(x.values, n), x.coords, x.dims, x.name)
    else:
        return np.power(x, n)


# =============================================================================
# Sum Function (vectorized via nimblend reduction)
# =============================================================================


def Sum(*args) -> LinearExpr:
    """
    Summation over sets using nimblend reduction.

    Args:
        *args: One or more Set objects followed by an expression.
               The last argument is the expression to sum.
               All preceding arguments are sets to sum over.

    Examples
    --------
    >>> Sum(i, j, x[i,j])           # Sum over all i,j -> scalar
    >>> Sum(j, cost[i,j] * x[i,j])  # Sum over j -> indexed by i
    """
    from .param import Param
    from .variable import VarRef

    if len(args) < 2:
        raise TypeError("Sum requires at least one set and an expression")

    # Last argument is the expression, rest are sets
    *sets, expr = args

    # Validate sets
    for s in sets:
        if not isinstance(s, Set):
            raise TypeError(f"Expected Set, got {type(s).__name__}")

    # Process expression based on type
    if isinstance(expr, VarRef):
        return _sum_varref(sets, expr)
    elif isinstance(expr, LinearExpr):
        return _sum_linear(sets, expr)
    elif isinstance(expr, Param):
        return _sum_param(sets, expr)
    elif isinstance(expr, (int, float)):
        count = 1
        for s in sets:
            count *= len(s)
        return LinearExpr([], expr * count, [])
    else:
        raise TypeError(f"Cannot sum {type(expr)}")


def _sum_varref(sets: list, var_ref: "VarRef") -> LinearExpr:
    """Sum a variable reference: Sum(j, x[i,j])."""
    sum_set_ids = {id(s) for s in sets}
    remaining_free = [s for s in var_ref.free_sets if id(s) not in sum_set_ids]

    return LinearExpr(
        terms=[(var_ref.var, 1.0, var_ref.fixed_indices, var_ref.lagged_indices)],
        const=0.0,
        free_sets=remaining_free,
    )


def _sum_param(sets: list, param: "Param") -> LinearExpr:
    """Sum a parameter: Sum(i, j, cost[i,j]) -> scalar."""
    result = param.array
    for s in sets:
        if s.name in result.dims:
            result = result.sum(s.name)

    if not result.dims:
        return LinearExpr([], float(result.values), [])

    remaining_sets = [s for s in param.sets if s.name in result.dims]
    return LinearExpr([], result, remaining_sets)


def _sum_linear(sets: list, expr: LinearExpr) -> LinearExpr:
    """Sum a linear expression: Sum(j, cost[i,j] * x[i,j])."""
    sum_set_ids = {id(s) for s in sets}

    new_terms = []
    for term in expr.terms:
        # Handle both 3-tuple and 4-tuple formats
        var, coef, fixed = term[0], term[1], term[2]
        lagged = term[3] if len(term) > 3 else []
        new_terms.append((var, coef, fixed, lagged))

    remaining_free = [s for s in expr.free_sets if id(s) not in sum_set_ids]

    new_const = expr.const
    if isinstance(new_const, nb.Array):
        for s in sets:
            if s.name in new_const.dims:
                new_const = new_const.sum(s.name)
        if not new_const.dims:
            new_const = float(new_const.values)

    return LinearExpr(
        terms=new_terms,
        const=new_const,
        free_sets=remaining_free,
    )
