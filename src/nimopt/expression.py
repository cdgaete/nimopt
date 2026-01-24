"""
Expression system using nimblend for coefficient arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import nimblend as nb

from .sets import Set

if TYPE_CHECKING:
    from .variable import Variable, VarRef


class LinearExpr:
    """
    Linear expression with nimblend-backed coefficients.

    Stores terms as (Variable, coef, fixed_indices, lagged_indices) tuples.
    - fixed_indices: List of (position, value) for concrete indices
    - lagged_indices: List of (position, LaggedSet) for lagged indices
    Coefficients can be scalars or nimblend Arrays for broadcasting.
    """

    __slots__ = ("terms", "const", "free_sets")

    def __init__(
        self,
        terms: Optional[List[Tuple["Variable", Any, List, List]]] = None,
        const: Any = 0.0,
        free_sets: Optional[List[Set]] = None,
    ):
        self.terms = terms or []
        self.const = const
        self.free_sets = free_sets or []

    @classmethod
    def from_term(cls, var_ref: "VarRef", coef: Any) -> "LinearExpr":
        """Create expression from single term: coef * var_ref."""
        from .param import Param

        free_sets = var_ref.free_sets
        fixed = var_ref.fixed_indices
        lagged = var_ref.lagged_indices

        if isinstance(coef, Param):
            coef_val = coef.array
        elif isinstance(coef, (int, float)):
            coef_val = float(coef)
        elif isinstance(coef, nb.Array):
            coef_val = coef
        else:
            coef_val = coef

        return cls(
            terms=[(var_ref.var, coef_val, fixed, lagged)],
            const=0.0,
            free_sets=free_sets,
        )

    def __repr__(self):
        parts = []
        for term in self.terms:
            var, coef = term[0], term[1]
            if isinstance(coef, nb.Array):
                parts.append(f"{var.name}[coef:{coef.shape}]")
            else:
                parts.append(f"{coef}*{var.name}")
        if self.const != 0:
            parts.append(str(self.const))
        return " + ".join(parts) if parts else "0"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return LinearExpr(
                self.terms.copy(),
                _add_const(self.const, other),
                self.free_sets.copy(),
            )
        elif isinstance(other, LinearExpr):
            new_terms = self.terms + other.terms
            new_free = _merge_sets(self.free_sets, other.free_sets)
            new_const = _add_const(self.const, other.const)
            return LinearExpr(new_terms, new_const, new_free)
        elif hasattr(other, "array"):
            return LinearExpr(
                self.terms.copy(),
                _add_const(self.const, other.array),
                self.free_sets.copy(),
            )
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.__add__(-other)
        elif isinstance(other, LinearExpr):
            neg_terms = [
                (t[0], _negate(t[1]), t[2], t[3] if len(t) > 3 else [])
                for t in other.terms
            ]
            return LinearExpr(
                self.terms + neg_terms,
                _add_const(self.const, _negate(other.const)),
                _merge_sets(self.free_sets, other.free_sets),
            )
        return NotImplemented

    def __mul__(self, other):
        from .param import Param

        # Convert Param to Array
        if isinstance(other, Param):
            other = other.array

        if isinstance(other, (int, float)):
            new_terms = [
                (t[0], _scale(t[1], other), t[2], t[3] if len(t) > 3 else [])
                for t in self.terms
            ]
            new_const = _scale(self.const, other)
            return LinearExpr(new_terms, new_const, self.free_sets.copy())
        elif isinstance(other, nb.Array):
            # Scale all coefficients by the Array
            new_terms = [
                (t[0], _scale(t[1], other), t[2], t[3] if len(t) > 3 else [])
                for t in self.terms
            ]
            new_const = _scale(self.const, other)
            return LinearExpr(new_terms, new_const, self.free_sets.copy())
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __le__(self, other):
        return Constraint(self, "<=", other)

    def __ge__(self, other):
        return Constraint(self, ">=", other)

    def __eq__(self, other):
        return Constraint(self, "==", other)


class Constraint:
    """Linear constraint: lhs sense rhs."""

    __slots__ = ("lhs", "sense", "rhs")

    def __init__(self, lhs: LinearExpr, sense: str, rhs: Any):
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs

    def __repr__(self):
        return f"{self.lhs} {self.sense} {self.rhs}"

    @property
    def free_sets(self) -> List[Set]:
        """All free sets in this constraint."""
        result = list(self.lhs.free_sets)
        seen = {id(s) for s in result}

        if isinstance(self.rhs, LinearExpr):
            for s in self.rhs.free_sets:
                if id(s) not in seen:
                    result.append(s)
                    seen.add(id(s))
        elif hasattr(self.rhs, "sets"):
            for s in self.rhs.sets:
                if id(s) not in seen:
                    result.append(s)
                    seen.add(id(s))

        return result


def _add_const(a, b):
    if isinstance(a, nb.Array) and isinstance(b, nb.Array):
        return a + b
    elif isinstance(a, nb.Array):
        return a + b if b != 0 else a
    elif isinstance(b, nb.Array):
        return b + a if a != 0 else b
    return a + b


def _negate(x):
    if isinstance(x, nb.Array):
        return x * (-1)
    return -x


def _scale(x, s):
    if isinstance(x, nb.Array):
        return x * s
    return x * s


def _merge_sets(a: List[Set], b: List[Set]) -> List[Set]:
    """Merge two set lists, preserving order, no duplicates."""
    result = list(a)
    seen = {id(s) for s in a}
    for s in b:
        if id(s) not in seen:
            result.append(s)
            seen.add(id(s))
    return result
