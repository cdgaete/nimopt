"""
Variable class for decision variables.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from .sets import Set

if TYPE_CHECKING:
    from .model import Model


class Variable:
    """
    Decision variable indexed by sets.

    Parameters
    ----------
    name : str
        Variable name.
    sets : list of Set
        Sets that index this variable.
    lb : float, optional
        Lower bound (default: -inf).
    ub : float, optional
        Upper bound (default: +inf).
    vtype : str
        Variable type: 'continuous', 'binary', 'integer'.
    """

    __slots__ = ("name", "sets", "lb", "ub", "vtype", "model")

    def __init__(
        self,
        name: str,
        sets: Optional[List[Set]] = None,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
        vtype: str = "continuous",
        model: Optional["Model"] = None,
    ):
        self.name = name
        self.sets = list(sets) if sets else []
        self.lb = lb
        self.ub = ub
        self.vtype = vtype
        self.model = model

    def __repr__(self):
        if self.sets:
            dims = ", ".join(s.name for s in self.sets)
            return f"Variable('{self.name}', [{dims}])"
        return f"Variable('{self.name}')"

    def __getitem__(self, indices) -> "VarRef":
        """Index the variable, returns VarRef for expressions."""
        from .sets import LaggedSet

        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != len(self.sets):
            raise IndexError(
                f"Variable '{self.name}' has {len(self.sets)} dims, got {len(indices)}"
            )

        for idx, s in zip(indices, self.sets):
            if isinstance(idx, Set):
                if idx is not s:
                    raise IndexError(f"Expected set '{s.name}', got '{idx.name}'")
            elif isinstance(idx, LaggedSet):
                # LaggedSet must reference the same base set
                if idx.base_set is not s:
                    raise IndexError(
                        f"Expected set '{s.name}', got lagged '{idx.base_set.name}'"
                    )
            elif idx not in s:
                raise IndexError(f"'{idx}' not in set '{s.name}'")

        return VarRef(self, indices)

    @property
    def dims(self) -> List[str]:
        return [s.name for s in self.sets]

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(len(s) for s in self.sets)

    @property
    def size(self) -> int:
        if not self.sets:
            return 1
        n = 1
        for s in self.sets:
            n *= len(s)
        return n

    @property
    def is_scalar(self) -> bool:
        return len(self.sets) == 0

    def all_names(self) -> List[str]:
        """Generate all concrete variable names for LP output."""
        if not self.sets:
            return [self.name]
        names = []
        for combo in itertools.product(*(s.elements for s in self.sets)):
            names.append(self.name + "_" + "_".join(str(e) for e in combo))
        return names

    # Arithmetic operators for scalar variables
    def _as_varref(self) -> "VarRef":
        """Convert scalar Variable to VarRef."""
        if self.sets:
            msg = f"Cannot use indexed variable '{self.name}' directly. "
            msg += "Use x[i,j] syntax."
            raise ValueError(msg)
        return VarRef(self, ())

    def __mul__(self, other):
        return self._as_varref() * other

    def __rmul__(self, other):
        return self._as_varref() * other

    def __add__(self, other):
        return self._as_varref() + other

    def __radd__(self, other):
        return self._as_varref() + other

    def __sub__(self, other):
        return self._as_varref() - other

    def __rsub__(self, other):
        return other + (-self._as_varref())

    def __neg__(self):
        return -self._as_varref()

    def __le__(self, other):
        return self._as_varref() <= other

    def __ge__(self, other):
        return self._as_varref() >= other

    def __eq__(self, other):
        return self._as_varref() == other


class VarRef:
    """
    Reference to a variable with indexing.

    Represents x[i,j] in expressions. Tracks which indices are
    symbolic (Set) vs concrete (specific element).
    """

    __slots__ = ("var", "indices")

    def __init__(self, var: Variable, indices: Tuple):
        self.var = var
        self.indices = indices

    def __repr__(self):
        from .sets import LaggedSet

        parts = []
        for s in self.indices:
            if isinstance(s, Set):
                parts.append(s.name)
            elif isinstance(s, LaggedSet):
                sign = "-" if s.offset < 0 else "+"
                parts.append(f"{s.base_set.name}{sign}{abs(s.offset)}")
            else:
                parts.append(repr(s))
        return f"{self.var.name}[{', '.join(parts)}]"

    @property
    def name(self):
        return self.var.name

    @property
    def free_sets(self) -> List[Set]:
        """Sets used symbolically (not fixed to specific element)."""
        from .sets import LaggedSet

        result = []
        for s in self.indices:
            if isinstance(s, Set):
                result.append(s)
            elif isinstance(s, LaggedSet):
                result.append(s.base_set)
        return result

    @property
    def lagged_indices(self) -> List[Tuple[int, Any]]:
        """List of (position, LaggedSet) for lagged indices."""
        from .sets import LaggedSet

        return [(i, s) for i, s in enumerate(self.indices) if isinstance(s, LaggedSet)]

    @property
    def has_lag(self) -> bool:
        """Check if this VarRef has any lagged indices."""
        from .sets import LaggedSet

        return any(isinstance(s, LaggedSet) for s in self.indices)

    @property
    def fixed_indices(self) -> List[Tuple[int, Any]]:
        """List of (position, value) for fixed indices."""
        from .sets import LaggedSet

        return [
            (i, v)
            for i, v in enumerate(self.indices)
            if not isinstance(v, (Set, LaggedSet))
        ]

    def __mul__(self, other):
        from .expression import LinearExpr

        return LinearExpr.from_term(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division: x / coef = (1/coef) * x."""
        import nimblend as nb

        from .expression import LinearExpr
        from .param import Param

        # Compute 1/other as the coefficient
        if isinstance(other, Param):
            inv_coef = Param(f"inv_{other.name}", other.sets, 1.0 / other.values)
            return LinearExpr.from_term(self, inv_coef)
        elif isinstance(other, nb.Array):
            inv_coef = nb.Array(1.0 / other.values, other.coords, other.dims)
            return LinearExpr.from_term(self, inv_coef)
        elif isinstance(other, (int, float)):
            return LinearExpr.from_term(self, 1.0 / other)
        else:
            return NotImplemented

    def __add__(self, other):
        from .expression import LinearExpr

        return LinearExpr.from_term(self, 1.0) + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from .expression import LinearExpr

        return LinearExpr.from_term(self, 1.0) - other

    def __neg__(self):
        from .expression import LinearExpr

        return LinearExpr.from_term(self, -1.0)

    def __le__(self, other):
        from .expression import Constraint, LinearExpr

        return Constraint(LinearExpr.from_term(self, 1.0), "<=", other)

    def __ge__(self, other):
        from .expression import Constraint, LinearExpr

        return Constraint(LinearExpr.from_term(self, 1.0), ">=", other)

    def __eq__(self, other):
        from .expression import Constraint, LinearExpr

        return Constraint(LinearExpr.from_term(self, 1.0), "==", other)

    def __hash__(self):
        return hash((id(self.var), self.indices))
