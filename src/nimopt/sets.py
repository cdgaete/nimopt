"""The index sets a model is declared over."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from nimblend import Domain, StoredCoord


def _periods(given: Any) -> int:
    """A lag as the whole number of members it steps.

    A lag that is not whole states a different lag once truncated, and a model
    built from it reports success, so it is refused rather than rounded.
    """
    if isinstance(given, bool) or not isinstance(given, (int, float, np.integer)):
        raise TypeError(f"a lag is a whole number of members; got {given!r}")
    if float(given) != int(given):
        raise ValueError(f"a lag is a whole number of members; got {given!r}")
    return int(given)


class Set:
    """A named dimension, with labels or awaiting them.

    A set declared without labels names a dimension and carries no members
    until `_bind` gives it some, which is what lets a model be declared
    before its data exists.
    """

    def __init__(self, name: str, labels: npt.ArrayLike | None = None) -> None:
        self.name = str(name)
        self.labels = None
        self.coord = None
        if labels is not None:
            self._bind(labels)

    @property
    def declared(self) -> bool:
        """Whether this set is still awaiting its members."""
        return self.labels is None

    def _bind(self, labels: npt.ArrayLike) -> None:
        self.labels = np.asarray(labels)
        self.coord = StoredCoord(self.labels)

    def _members(self) -> StoredCoord:
        if self.coord is None:
            raise ValueError(
                f"set {self.name!r} is declared and carries no members; bind "
                f"it before reading them"
            )
        return self.coord

    def __len__(self) -> int:
        return len(self._members())

    def __repr__(self) -> str:
        if self.declared:
            return f"Set({self.name!r}, declared)"
        return f"Set({self.name!r}, {len(self)} members)"

    def position_of(self, labels: npt.ArrayLike) -> npt.NDArray[np.integer]:
        """Positions the given labels occupy in this set."""
        return self._members().to_position(labels)

    @property
    def cyclic(self) -> "CyclicSet":
        """This set read as a cyclic axis, where a lag past an end wraps."""
        return CyclicSet(self)

    def __sub__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self, _periods(periods), "drop")

    def __add__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self, -_periods(periods), "drop")


class Alias:
    """A second name for a set, sharing its labels and its coordinate.

    A parameter over a set and its alias is an ordinary two-dimensional
    array, so a model relating a set to itself states it without a second
    set. No labels are copied: the alias reads the coordinate the set
    already built.
    """

    def __init__(self, name: str, base: "Set") -> None:
        self.name = str(name)
        self.base = base

    @property
    def labels(self) -> npt.NDArray[Any] | None:
        """The labels of the set this names."""
        return self.base.labels

    @property
    def coord(self) -> StoredCoord | None:
        """The coordinate of the set this names."""
        return self.base.coord

    def __len__(self) -> int:
        return len(self.base)

    def __repr__(self) -> str:
        return f"Alias({self.name!r}, {self.base.name!r})"

    @property
    def cyclic(self) -> "CyclicSet":
        """This alias read as a cyclic axis, where a lag past an end wraps."""
        return CyclicSet(self)

    def __sub__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self, _periods(periods), "drop")

    def __add__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self, -_periods(periods), "drop")


class CyclicSet:
    """A set whose lag arithmetic wraps, so every row stays stated."""

    def __init__(self, base: "Set | Alias") -> None:
        self.base = base

    @property
    def name(self) -> str:
        """The name of the set this reads."""
        return self.base.name

    def __repr__(self) -> str:
        return f"CyclicSet({self.name!r})"

    def __sub__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self.base, _periods(periods), "wrap")

    def __add__(self, periods: Any) -> "LaggedSet":
        return LaggedSet(self.base, -_periods(periods), "wrap")


class LaggedSet:
    """A set referenced at an offset from the row's own member.

    `T - 1` reads the previous member, which moves an entry from `t` to
    `t + 1`, so it is a shift of `+1`. Under `drop` a reference reaching
    outside the set removes the row rather than leaving it standing with a
    term missing; `T.cyclic - 1` wraps instead and states every row.
    """

    def __init__(self, base: "Set | Alias", shift: int, mode: str) -> None:
        self.base = base
        self.shift = int(shift)
        self.mode = mode

    @property
    def name(self) -> str:
        """The name of the set this reads."""
        return self.base.name

    def __len__(self) -> int:
        return len(self.base)

    @property
    def coord(self) -> StoredCoord | None:
        """The coordinate of the set this reads."""
        return self.base.coord

    def __repr__(self) -> str:
        return f"LaggedSet({self.name!r}, shift={self.shift}, mode={self.mode!r})"

    def _one_lag(self, periods: Any) -> Any:
        raise TypeError(
            f"set {self.name!r} is already read at a lag of {self.shift}; a "
            f"reference carries one lag, so write the total as a single lag "
            f"rather than {periods!r} more"
        )

    __sub__ = _one_lag
    __add__ = _one_lag


SET_LIKE = (Set, Alias, CyclicSet, LaggedSet)


def reference(
    sets: Any, dims: Sequence[str]
) -> tuple[tuple[str, ...], dict[str, tuple[int, str]], dict[str, Any]]:
    """The dimension names a reference states, its lags, and its fixed members.

    An entry that is not a set is a label, and it fixes the dimension standing
    at its position in `dims`: `x[S, "t0"]` is the row at `t0` alone, and the
    dimension it names leaves the frame.
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
    """Refuse a fixed member the dimension's own set does not carry.

    A declared set carries no members yet, so its check waits until it binds.
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
                f"{owner} is read at member {label!r} of dimension {dim!r}, "
                f"which that set does not carry"
            ) from None


def reading(name: str, dims: Sequence[str]) -> str:
    """How a symbol over `dims` is read, as the spelling states it.

    A symbol over no dimension is already read, so it carries no bracket.
    """
    return f"{name}[{', '.join(dims)}]" if dims else str(name)


def coords_of(sets: Iterable[Any]) -> dict[str, Any]:
    """Each set's coordinate, keyed by the name of the dimension it names."""
    return {s.name: s.coord for s in sets}


def _frame(sets: Iterable[Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    return tuple(s.name for s in sets), coords_of(sets)


def subset(sets: Iterable[Any], columns: Mapping[str, npt.ArrayLike]) -> Domain:
    """The members of a set product a model carries, named by label.

    `columns` holds one label column per set, keyed by the set's name, read
    in step: the k-th entry of each names one member.
    """
    dims, coords = _frame(tuple(sets))
    return Domain.from_labels(dims, coords, columns)


def product(sets: Iterable[Any]) -> Domain:
    """Every member of a set product, as the rows a constraint states.

    A constraint whose terms each reach some of its rows states them with
    this rather than having them derived from the terms.
    """
    dims, coords = _frame(tuple(sets))
    return Domain.full(dims, coords)


def subset_of(sets: Iterable[Any], index: npt.ArrayLike) -> Domain:
    """The members of a set product a model carries, named by position.

    Each column of `index` names one member. A caller holding positions
    states them directly rather than building labels to resolve back.
    """
    dims, coords = _frame(tuple(sets))
    return Domain.from_coordinates(dims, coords, index)


def rows_of(given: Any, dims: tuple[str, ...] | None, owner: str, what: str) -> Domain:
    """The domain `given` states over `dims`.

    A tuple of sets is their full product, a parameter is the coordinates it
    carries, and a domain is itself. A domain resolves labels through each
    set's coordinate and a declared set carries none, so a definition names
    the sets or the parameter and both spellings reach the same domain.

    `dims` are the dimensions the domain is required to span, or `None` where
    a caller has already checked them against something else.
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
