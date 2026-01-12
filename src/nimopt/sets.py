"""
Set class for defining index domains.

Sets are the foundation of algebraic modeling - they define the indices
over which variables, parameters, and constraints are defined.
"""

from typing import Any, Iterator, List


class Set:
    """
    Named set of elements for indexing variables and parameters.

    In nimopt, Sets define dimensions for nimblend arrays.
    They provide the coordinate labels for each axis.

    Parameters
    ----------
    name : str
        Name of the set (becomes dimension name in nimblend).
    elements : list
        Elements of the set (become coordinate labels).

    Examples
    --------
    >>> i = Set('i', ['seattle', 'sandiego'])
    >>> j = Set('j', ['newyork', 'chicago', 'topeka'])
    >>> len(i)
    2
    >>> 'seattle' in i
    True
    """

    __slots__ = ("name", "elements", "_index_map")

    def __init__(self, name: str, elements: List[Any]):
        self.name = name
        self.elements = list(elements)
        self._index_map = {e: idx for idx, e in enumerate(self.elements)}

    def __repr__(self):
        if len(self.elements) <= 5:
            return f"Set('{self.name}', {self.elements})"
        return f"Set('{self.name}', n={len(self)})"

    def __len__(self):
        return len(self.elements)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.elements)

    def __contains__(self, item):
        return item in self._index_map

    def __eq__(self, other):
        if isinstance(other, Set):
            return self.name == other.name and self.elements == other.elements
        return False

    def __hash__(self):
        return hash((self.name, tuple(self.elements)))

    def index(self, element: Any) -> int:
        """Get the integer index of an element."""
        if element not in self._index_map:
            raise KeyError(f"Element '{element}' not in set '{self.name}'")
        return self._index_map[element]
