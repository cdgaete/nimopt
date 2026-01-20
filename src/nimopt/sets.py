"""
Set class for defining index domains.

Sets are the foundation of algebraic modeling - they define the indices
over which variables, parameters, and constraints are defined.
"""

from typing import Any, Callable, Iterator, List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


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

    def lag(self, n: int = 1, cyclic: bool = False) -> "LaggedSet":
        """
        Create a lagged index reference (h-n).

        Parameters
        ----------
        n : int
            Number of periods to lag (default 1).
        cyclic : bool
            If True, wrap around (h=1 references h=last).

        Returns
        -------
        LaggedSet
            Lagged set reference for use in constraints.

        Examples
        --------
        >>> hours = Set('h', list(range(1, 25)))
        >>> x[hours.lag(1)]  # References x[h-1]
        """
        return LaggedSet(self, -n, cyclic)

    def lead(self, n: int = 1, cyclic: bool = False) -> "LaggedSet":
        """
        Create a lead index reference (h+n).

        Parameters
        ----------
        n : int
            Number of periods to lead (default 1).
        cyclic : bool
            If True, wrap around (h=last references h=1).

        Returns
        -------
        LaggedSet
            Lead set reference for use in constraints.
        """
        return LaggedSet(self, n, cyclic)

    def subset(self, name: str, mask: List[bool]) -> "Set":
        """
        Create a subset from a boolean mask.

        Parameters
        ----------
        name : str
            Name for the new subset.
        mask : list of bool
            Boolean mask, same length as elements.

        Returns
        -------
        Set
            New set containing only elements where mask is True.

        Examples
        --------
        >>> tech = Set('tech', ['coal', 'gas', 'wind', 'solar'])
        >>> renewable_mask = [False, False, True, True]
        >>> renewables = tech.subset('renewables', renewable_mask)
        >>> list(renewables)
        ['wind', 'solar']
        """
        if len(mask) != len(self.elements):
            raise ValueError(
                f"Mask length {len(mask)} != set length {len(self.elements)}"
            )
        filtered = [e for e, m in zip(self.elements, mask) if m]
        return Set(name, filtered)

    def filter(
        self,
        name: str,
        elements: Optional[List[Any]] = None,
        predicate: Optional[Callable[[Any], bool]] = None,
    ) -> "Set":
        """
        Create a subset by filtering elements.

        Parameters
        ----------
        name : str
            Name for the new subset.
        elements : list, optional
            Explicit list of elements to keep (must be in this set).
        predicate : callable, optional
            Function that takes an element and returns True to keep it.

        Returns
        -------
        Set
            New set containing filtered elements.

        Examples
        --------
        >>> hours = Set('h', list(range(1, 25)))
        >>> peak = hours.filter('peak', elements=[17, 18, 19, 20])
        >>> offpeak = hours.filter('offpeak', predicate=lambda h: h < 7 or h > 22)
        """
        if elements is not None:
            filtered = [e for e in elements if e in self]
        elif predicate is not None:
            filtered = [e for e in self.elements if predicate(e)]
        else:
            raise ValueError("Must provide either 'elements' or 'predicate'")
        return Set(name, filtered)

    @classmethod
    def from_dataframe(
        cls,
        name: str,
        df: "pd.DataFrame",
        column: str,
        filter_col: Optional[str] = None,
        filter_val: Optional[Any] = None,
        filter_func: Optional[Callable] = None,
    ) -> "Set":
        """
        Create a set from a DataFrame column, optionally filtered.

        Parameters
        ----------
        name : str
            Name for the set.
        df : pd.DataFrame
            Source DataFrame.
        column : str
            Column containing set elements.
        filter_col : str, optional
            Column to filter on.
        filter_val : any, optional
            Value to match in filter_col (equality filter).
        filter_func : callable, optional
            Function that takes a Series and returns boolean mask.

        Returns
        -------
        Set
            New set with unique elements from the column.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'tech': ['coal', 'gas', 'wind', 'solar'],
        ...     'renewable': [0, 0, 1, 1],
        ... })
        >>> all_tech = Set.from_dataframe('tech', df, 'tech')
        >>> renewables = Set.from_dataframe('renewables', df, 'tech',
        ...                                  filter_col='renewable', filter_val=1)
        >>> list(renewables)
        ['wind', 'solar']
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for from_dataframe()")

        # Apply filters
        if filter_col is not None and filter_val is not None:
            df = df[df[filter_col] == filter_val]
        elif filter_func is not None:
            mask = filter_func(df[filter_col] if filter_col else df)
            df = df[mask]

        # Get unique elements preserving order
        elements = df[column].drop_duplicates().tolist()
        return cls(name, elements)

    @classmethod
    def from_csv(
        cls,
        name: str,
        filepath: str,
        column: str,
        filter_col: Optional[str] = None,
        filter_val: Optional[Any] = None,
    ) -> "Set":
        """
        Create a set from a CSV file column, optionally filtered.

        Parameters
        ----------
        name : str
            Name for the set.
        filepath : str
            Path to CSV file.
        column : str
            Column containing set elements.
        filter_col : str, optional
            Column to filter on.
        filter_val : any, optional
            Value to match in filter_col.

        Returns
        -------
        Set
            New set with unique elements from the column.

        Examples
        --------
        >>> tech = Set.from_csv('tech', 'technologies.csv', 'name')
        >>> renewables = Set.from_csv('renewables', 'technologies.csv', 'name',
        ...                           filter_col='renewable', filter_val=1)
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for from_csv()")

        df = pd.read_csv(filepath)
        return cls.from_dataframe(name, df, column, filter_col, filter_val)


        df = pd.read_csv(filepath)
        return cls.from_dataframe(name, df, column, filter_col, filter_val)


class LaggedSet:
    """
    Represents a lagged/lead set index like h-1 or h+1.

    Used in time-indexed constraints where we need to reference
    the previous or next time period.

    Parameters
    ----------
    base_set : Set
        The original set being lagged.
    offset : int
        Lag offset. Negative for lag (h-1), positive for lead (h+1).
    cyclic : bool
        If True, wrap around (last->first). If False, boundary is excluded.

    Examples
    --------
    >>> hours = Set('h', list(range(1, 25)))
    >>> h_lag = hours.lag(1)      # h-1
    >>> h_lead = hours.lead(1)    # h+1
    >>> h_lag_cyclic = hours.lag(1, cyclic=True)  # wraps h=1 -> h=24
    """

    __slots__ = ("base_set", "offset", "cyclic")

    def __init__(self, base_set: "Set", offset: int, cyclic: bool = False):
        self.base_set = base_set
        self.offset = offset
        self.cyclic = cyclic

    def __repr__(self):
        sign = "-" if self.offset < 0 else "+"
        cyc = " (cyclic)" if self.cyclic else ""
        return f"LaggedSet({self.base_set.name}{sign}{abs(self.offset)}{cyc})"

    @property
    def name(self):
        """Return base set name for dimension matching."""
        return self.base_set.name

    @property
    def elements(self):
        """Return base set elements."""
        return self.base_set.elements

    def __len__(self):
        return len(self.base_set)

    def __iter__(self):
        return iter(self.base_set)

    def __contains__(self, item):
        return item in self.base_set

    def get_lagged_element(self, element):
        """
        Get the lagged element corresponding to the given element.

        For lag(1) with element at index i, returns element at index i-1.
        Returns None if out of bounds and not cyclic.
        """
        idx = self.base_set.index(element)
        lagged_idx = idx + self.offset  # offset is negative for lag

        if self.cyclic:
            lagged_idx = lagged_idx % len(self.base_set)
        elif lagged_idx < 0 or lagged_idx >= len(self.base_set):
            return None  # Out of bounds

        return self.base_set.elements[lagged_idx]
