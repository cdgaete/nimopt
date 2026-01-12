"""
Parameter class backed by nimblend Array.

Parameters store constant data indexed by Sets. Internally uses
nimblend for labeled N-dimensional arrays with automatic broadcasting.
"""

from typing import Any, List, Union

import nimblend as nb
import numpy as np

from .sets import Set


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
                f"Data shape {self.array.shape} doesn't match "
                f"set dimensions {expected}"
            )

    def __repr__(self):
        dims = ", ".join(s.name for s in self.sets)
        return f"Param('{self.name}', [{dims}], shape={self.array.shape})"

    def __getitem__(self, indices) -> Union[float, "Param"]:
        """
        Index the parameter.

        Returns scalar if all indices concrete, or view if partial.
        """
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != len(self.sets):
            raise IndexError(
                f"Param '{self.name}' has {len(self.sets)} dims, got {len(indices)}"
            )

        # Check if all concrete (not Set objects)
        int_indices = []
        for idx, s in zip(indices, self.sets):
            if isinstance(idx, Set):
                # Symbolic - return self for expression building
                return self
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
