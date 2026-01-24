"""
Parameter class backed by nimblend Array.

Parameters store constant data indexed by Sets. Internally uses
nimblend for labeled N-dimensional arrays with automatic broadcasting.
"""

from typing import Any, List, Tuple, Union

import nimblend as nb
import numpy as np

from .sets import LaggedSet, Set


class ParamRef:
    """
    Reference to a parameter with symbolic indexing.

    Similar to VarRef, tracks which sets were used for indexing.
    This allows proper free_set detection when subsets are used
    to index parameters defined over supersets.
    """

    __slots__ = ("param", "indices")

    def __init__(self, param: "Param", indices: Tuple):
        self.param = param
        self.indices = indices

    def __repr__(self):
        parts = []
        for s in self.indices:
            if isinstance(s, Set):
                parts.append(s.name)
            elif isinstance(s, LaggedSet):
                sign = "-" if s.offset < 0 else "+"
                parts.append(f"{s.base_set.name}{sign}{abs(s.offset)}")
            else:
                parts.append(repr(s))
        return f"{self.param.name}[{', '.join(parts)}]"

    @property
    def name(self):
        return self.param.name

    @property
    def array(self):
        """Return underlying nimblend array for coefficient extraction."""
        return self.param.array

    @property
    def sets(self) -> List[Set]:
        """Return the SYMBOLIC sets used for indexing (not underlying param sets)."""
        result = []
        for idx in self.indices:
            if isinstance(idx, Set):
                result.append(idx)
            elif isinstance(idx, LaggedSet):
                result.append(idx.base_set)
        return result

    @property
    def values(self):
        return self.param.values

    # Arithmetic - delegate to param's array
    def __add__(self, other):
        return self.param.array + other

    def __radd__(self, other):
        return other + self.param.array

    def __sub__(self, other):
        return self.param.array - other

    def __rsub__(self, other):
        return other - self.param.array

    def __mul__(self, other):
        return self.param.array * other

    def __rmul__(self, other):
        return other * self.param.array

    def __truediv__(self, other):
        return self.param.array / other

    def __rtruediv__(self, other):
        return other / self.param.array

    def __neg__(self):
        return -self.param.array


class Param:
    """
    Named parameter indexed by sets, backed by nimblend Array.

    Parameters hold constant data (costs, demands, capacities).
    The nimblend backend enables automatic broadcasting when
    combining parameters with different dimensions.

    Parameters
    ----------
    name : str
        Name of the parameter.
    sets : list of Set
        Sets that index this parameter.
    data : array-like
        Parameter values. Shape must match set dimensions.

    Examples
    --------
    >>> i = Set('i', ['seattle', 'sandiego'])
    >>> j = Set('j', ['newyork', 'chicago'])
    >>> cost = Param('cost', [i, j], [[2.5, 1.7], [2.5, 1.8]])
    >>> cost.array.shape
    (2, 2)
    >>> cost['seattle', 'newyork']
    2.5
    """

    __slots__ = ("name", "sets", "array")

    def __init__(self, name: str, sets: List[Set], data: Any):
        self.name = name
        self.sets = list(sets)

        # Build nimblend coords from sets
        coords = {s.name: s.elements for s in self.sets}
        self.array = nb.Array(np.asarray(data, dtype=np.float64), coords)

        # Validate shape
        expected = tuple(len(s) for s in self.sets)
        if self.array.shape != expected:
            raise ValueError(
                f"Data shape {self.array.shape} doesn't match set dimensions {expected}"
            )

    def __repr__(self):
        dims = ", ".join(s.name for s in self.sets)
        return f"Param('{self.name}', [{dims}], shape={self.array.shape})"

    def __getitem__(self, indices) -> Union[float, "ParamRef"]:
        """
        Index the parameter.

        Returns scalar if all indices concrete, or ParamRef if any symbolic.
        """
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != len(self.sets):
            raise IndexError(
                f"Param '{self.name}' has {len(self.sets)} dims, got {len(indices)}"
            )

        # Check if any indices are symbolic (Set or LaggedSet objects)
        has_symbolic = any(isinstance(idx, (Set, LaggedSet)) for idx in indices)

        if has_symbolic:
            # Return ParamRef that tracks the symbolic indices
            return ParamRef(self, indices)

        # All concrete - return scalar value
        int_indices = []
        for idx, s in zip(indices, self.sets):
            int_indices.append(s.index(idx))

        return float(self.array.values[tuple(int_indices)])

    @property
    def dims(self) -> List[str]:
        """Dimension names."""
        return [s.name for s in self.sets]

    @property
    def values(self) -> np.ndarray:
        """Raw numpy array of values."""
        return self.array.values

    @property
    def shape(self):
        return self.array.shape

    def __add__(self, other):
        """Add to param - returns nimblend Array or delegates to other."""
        if isinstance(other, Param):
            return self.array + other.array
        if hasattr(other, "__radd__"):
            result = other.__radd__(self)
            if result is not NotImplemented:
                return result
        return self.array + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract from param - returns nimblend Array or delegates to other."""
        if isinstance(other, Param):
            return self.array - other.array
        if hasattr(other, "__rsub__"):
            result = other.__rsub__(self)
            if result is not NotImplemented:
                return result
        return self.array - other

    def __rsub__(self, other):
        if isinstance(other, Param):
            return other.array - self.array
        return other - self.array

    def __mul__(self, other):
        """Multiply param - returns nimblend Array or delegates to other."""
        if isinstance(other, Param):
            return self.array * other.array
        if hasattr(other, "__rmul__"):
            result = other.__rmul__(self)
            if result is not NotImplemented:
                return result
        return self.array * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide param - returns nimblend Array or delegates to other."""
        if isinstance(other, Param):
            return self.array / other.array
        if hasattr(other, "__rtruediv__"):
            result = other.__rtruediv__(self)
            if result is not NotImplemented:
                return result
        return self.array / other

    def __rtruediv__(self, other):
        if isinstance(other, Param):
            return other.array / self.array
        return other / self.array

    def __neg__(self):
        """Negate param - returns nimblend Array."""
        return -self.array
