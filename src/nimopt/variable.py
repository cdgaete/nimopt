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
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != len(self.sets):
            raise IndexError(
                f"Variable '{self.name}' has {len(self.sets)} dims, "
                f"got {len(indices)}"
            )

        for idx, s in zip(indices, self.sets):
            if isinstance(idx, Set):
                if idx is not s:
                    raise IndexError(f"Expected set '{s.name}', got '{idx.name}'")
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
        idx_str = ", ".join(
            s.name if isinstance(s, Set) else repr(s) for s in self.indices
        )
        return f"{self.var.name}[{idx_str}]"

    @property
    def name(self):
        return self.var.name

    @property
    def free_sets(self) -> List[Set]:
        """Sets used symbolically (not fixed to specific element)."""
        return [s for s in self.indices if isinstance(s, Set)]

    @property
    def fixed_indices(self) -> List[Tuple[int, Any]]:
        """List of (position, value) for fixed indices."""
        return [
            (i, v) for i, v in enumerate(self.indices) if not isinstance(v, Set)
        ]

    def __mul__(self, other):
        from .expression import LinearExpr
        return LinearExpr.from_term(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

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
