"""
Sum function using nimblend for vectorized reduction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nimblend as nb

from .expression import LinearExpr
from .sets import Set

if TYPE_CHECKING:
    from .param import Param
    from .variable import VarRef


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
        terms=[(var_ref.var, 1.0, var_ref.fixed_indices)],
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
    for var, coef, fixed in expr.terms:
        new_terms.append((var, coef, fixed))

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
