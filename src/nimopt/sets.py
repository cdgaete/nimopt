"""The index sets a model is declared over."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from nimblend import Domain, StoredCoord


def _periods(given: Any) -> int:
    """Return `given` as a whole number of members.

    Raises TypeError for a value that is not an integer or a float, and
    ValueError for a float with a fractional part.
    """
    if isinstance(given, bool) or not isinstance(given, (int, float, np.integer)):
        raise TypeError(f"a lag is a whole number of members; got {given!r}")
    if float(given) != int(given):
        raise ValueError(f"a lag is a whole number of members; got {given!r}")
    return int(given)


class Set:
    """A named dimension, with labels or without them.

    A set declared without labels has a name and no members. A model is
    declared over such a set before its data exists.
    """

    def __init__(self, name: str, labels: npt.ArrayLike | None = None) -> None:
        self.name = str(name)
        self.labels = None
        self.coord = None
        if labels is not None:
            self._bind(labels)

    @property
    def declared(self) -> bool:
        """True where this set has no members."""
        return self.labels is None

    def _bind(self, labels: npt.ArrayLike) -> None:
        self.labels = np.asarray(labels)
        self.coord = StoredCoord(self.labels)

    def _members(self) -> StoredCoord:
        if self.coord is None:
            raise ValueError(
                f"set {self.name!r} is declared and has no members; bind it "
                f"before reading them"
            )
        return self.coord

    def __len__(self) -> int:
        return len(self._members())

    def __repr__(self) -> str:
        if self.declared:
            return f"Set({self.name!r}, declared)"
        return f"Set({self.name!r}, {len(self)} members)"

    def position_of(self, labels: npt.ArrayLike) -> npt.NDArray[np.integer]:
        """Return the positions the given labels occupy in this set."""
        return self._members().to_position(labels)

    @property
    def cyclic(self) -> "CyclicSet":
        """Return this set as a cyclic axis, where a lag past an end wraps."""
        return CyclicSet(self)

    def __sub__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self, _periods(periods), "drop")

    def __add__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self, -_periods(periods), "drop")


class Alias:
    """A second name for a set, sharing its labels and its coordinate.

    A parameter over a set and its alias is a two-dimensional array. The
    alias copies no labels. It reads the coordinate the set already built.
    """

    def __init__(self, name: str, base: "Set") -> None:
        self.name = str(name)
        self.base = base

    @property
    def labels(self) -> npt.NDArray[Any] | None:
        """Return the labels of the set this alias refers to."""
        return self.base.labels

    @property
    def coord(self) -> StoredCoord | None:
        """Return the coordinate of the set this alias refers to."""
        return self.base.coord

    def __len__(self) -> int:
        return len(self.base)

    def __repr__(self) -> str:
        return f"Alias({self.name!r}, {self.base.name!r})"

    @property
    def cyclic(self) -> "CyclicSet":
        """Return this alias as a cyclic axis, where a lag past an end wraps."""
        return CyclicSet(self)

    def __sub__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self, _periods(periods), "drop")

    def __add__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self, -_periods(periods), "drop")


class CyclicSet:
    """A set whose lag arithmetic wraps, keeping every row."""

    def __init__(self, base: "Set | Alias") -> None:
        self.base = base

    @property
    def name(self) -> str:
        """Return the name of the set this refers to."""
        return self.base.name

    def __repr__(self) -> str:
        return f"CyclicSet({self.name!r})"

    def __sub__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self.base, _periods(periods), "wrap")

    def __add__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self.base, -_periods(periods), "wrap")


class LaggedSet:
    """A set referenced at an offset from the row's own member.

    `T - 1` reads the previous member. That moves an entry from `t` to
    `t + 1`, a shift of `+1`. Under mode `drop` a reference outside the set
    removes the row. Under mode `wrap`, written `T.cyclic - 1`, the reference
    wraps and every row is kept.
    """

    def __init__(self, base: "Set | Alias", shift: int, mode: str) -> None:
        self.base = base
        self.shift = int(shift)
        self.mode = mode

    @property
    def name(self) -> str:
        """Return the name of the set this refers to."""
        return self.base.name

    def __len__(self) -> int:
        return len(self.base)

    @property
    def coord(self) -> StoredCoord | None:
        """Return the coordinate of the set this refers to."""
        return self.base.coord

    def __repr__(self) -> str:
        return f"LaggedSet({self.name!r}, shift={self.shift}, mode={self.mode!r})"

    def _one_lag(self, periods: Any) -> Any:
        raise TypeError(
            f"set {self.name!r} is already read at a lag of {self.shift} and a "
            f"reference takes one lag; write the total as a single lag, not "
            f"{periods!r} more"
        )

    __sub__ = _one_lag
    __add__ = _one_lag


SET_LIKE = (Set, Alias, CyclicSet, LaggedSet)


def reference(
    sets: Any, dims: Sequence[str]
) -> tuple[tuple[str, ...], dict[str, tuple[int, str]], dict[str, Any]]:
    """Return the dimension names of a reference, its lags and its fixed members.

    An entry that is not a set is a label. It fixes the dimension at its
    position in `dims`. `x[S, "t0"]` selects the rows at `t0`, and that
    dimension is not in the frame.
    """
    given = sets if isinstance(sets, tuple) else (sets,)
    names = []
    shifts = {}
    fixed = {}
    for position, item in enumerate(given):
        if isinstance(item, SET_LIKE):
            names.append(item.name)
            if isinstance(item, LaggedSet):
                shifts[item.name] = (item.shift, item.mode)
        elif position < len(dims):
            names.append(dims[position])
            fixed[dims[position]] = item
        else:
            names.append(repr(item))
    return tuple(names), shifts, fixed


def check_members(sets: Iterable[Any], fixed: Mapping[str, Any], owner: str) -> None:
    """Check each fixed member against the labels of its own dimension's set.

    Raises ValueError for a label the set does not contain. A declared set
    has no members, and the check skips it.
    """
    by_name = {s.name: s for s in sets}
    for dim, label in fixed.items():
        held = by_name[dim]
        if getattr(held, "declared", False):
            continue
        try:
            held.coord.to_position(np.asarray([label]))
        except KeyError:
            raise ValueError(
                f"{owner} is read at member {label!r} of dimension {dim!r}; "
                f"read it at a member that set contains"
            ) from None


def reading(name: str, dims: Sequence[str]) -> str:
    """Return the text that reads a symbol named `name` over `dims`.

    A symbol over no dimension requires no bracket. The name is returned
    alone.
    """
    return f"{name}[{', '.join(dims)}]" if dims else str(name)


def coords_of(sets: Iterable[Any]) -> dict[str, Any]:
    """Return each set's coordinate, keyed by the dimension name."""
    return {s.name: s.coord for s in sets}


def _frame(sets: Iterable[Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    return tuple(s.name for s in sets), coords_of(sets)


def subset(sets: Iterable[Any], columns: Mapping[str, npt.ArrayLike]) -> Domain:
    """Return the members of a set product, selected by label.

    `columns` contains one label column per set, keyed by the set's name. The
    columns are read in step: the k-th entry of each identifies one member.
    """
    dims, coords = _frame(tuple(sets))
    return Domain.from_labels(dims, coords, columns)


def product(sets: Iterable[Any]) -> Domain:
    """Return every member of a set product, as the rows of a constraint.

    A constraint declares its rows with this instead of deriving them from
    its terms.
    """
    dims, coords = _frame(tuple(sets))
    return Domain.full(dims, coords)


def subset_of(sets: Iterable[Any], index: npt.ArrayLike) -> Domain:
    """Return the members of a set product, selected by position.

    Each column of `index` identifies one member. A caller with positions
    passes them directly, with no labels to resolve.
    """
    dims, coords = _frame(tuple(sets))
    return Domain.from_coordinates(dims, coords, index)


def rows_of(given: Any, dims: tuple[str, ...] | None, owner: str, what: str) -> Domain:
    """Return the domain `given` specifies over `dims`.

    A tuple of sets resolves to their full product, a parameter to the
    coordinates of its array, and a domain to itself. `dims` are the
    dimensions the domain is required to span, or `None` where a caller has
    already checked them. Raises ValueError where the domain spans other
    dimensions.
    """
    from nimopt.param import Param

    if isinstance(given, Param):
        resolved = given.materialise().domain(given.dims)
    elif isinstance(given, tuple):
        resolved = product(given)
    else:
        resolved = given
    if dims is not None and resolved.dims != dims:
        raise ValueError(
            f"{owner} has free dimensions {dims}; its {what} is over {resolved.dims}"
        )
    return resolved
