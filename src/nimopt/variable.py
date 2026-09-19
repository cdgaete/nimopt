"""A decision variable, whose columns are a dimension of the model's arrays."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from nimblend import Domain, ProductCoord, SparseArray

if TYPE_CHECKING:
    from nimopt.term import Expression

from nimopt.names import COLUMN
from nimopt.param import Param
from nimopt.sets import as_sets, condition_dims, coords_of, label_text, rows_of
from nimopt.symbol import Symbol


class Variable(Symbol):
    """A variable over a set product, or over a subset of one.

    The variable's columns are a virtual coordinate. For a full product a
    member's column is computed from its multi-index by stride arithmetic.
    For a subset it is the member's rank among the subset's codes. No column
    index is stored.
    """

    def __init__(
        self,
        name: str,
        sets: Iterable[Any],
        start: int | None = None,
        total_columns: int | None = None,
        subset: Any = None,
        lower: float | Param = 0.0,
        upper: float | Param = np.inf,
        integer: bool = False,
    ) -> None:
        self.name = str(name)
        self.sets = as_sets(sets, f"variable {self.name!r}")
        if subset is not None:
            condition_dims(subset, f"variable {self.name!r}", "subset=")
        self.subset = subset
        self.lower = self._bound(lower, "lower")
        self.upper = self._bound(upper, "upper")
        self.integer = bool(integer)
        self.start = None
        self.total_columns = None
        self.coord = None
        self._domain = None
        if start is not None:
            if total_columns is None:
                raise ValueError(
                    f"variable {self.name!r} is numbered from {start} and has "
                    f"no total; give a start and a total together"
                )
            self._bind(start, total_columns)

    @property
    def declared(self) -> bool:
        """True where this variable has no numbering."""
        return self.coord is None

    def _bind(self, start: int, total_columns: int) -> None:
        self.start = int(start)
        self.total_columns = int(total_columns)
        sizes = tuple(len(s) for s in self.sets)
        if self.subset is None:
            self.coord = ProductCoord(sizes, self.start)
            return
        members = rows_of(self.subset, None, f"variable {self.name!r}", "subset=")
        if members.dims != self.dims or members.shape != sizes:
            raise ValueError(
                f"variable {self.name!r} is over {self.dims} of sizes {sizes} "
                f"and its members span {members.dims} of sizes "
                f"{members.shape}; give members over {self.dims}"
            )
        self._domain = members
        self.coord = members.as_coord(self.start)

    def _bound(self, bound: float | Param, which: str) -> float | Param:
        """Return the bound as given, as a float or a Param.

        A Param over fewer of the variable's dimensions bounds every column
        that shares those coordinates. Raises ValueError for a Param over any
        other dimension, and for a float that is NaN.
        """
        if isinstance(bound, Param):
            lacking = [d for d in bound.dims if d not in self.dims]
            if lacking:
                raise ValueError(
                    f"variable {self.name!r} is declared over {self.dims} and "
                    f"is not over {lacking}; its {which} bound "
                    f"{bound.name!r} is declared over {bound.dims}"
                )
            return bound
        value = float(bound)
        if np.isnan(value):
            raise ValueError(
                f"the {which} bound of variable {self.name!r} is not a "
                f"number; give a finite value or an infinity"
            )
        return value

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the names of the sets this variable is declared over."""
        return tuple(s.name for s in self.sets)

    @property
    def coords(self) -> dict[str, Any]:
        """Return the coordinate of each set this variable is declared over."""
        return coords_of(self.sets)

    def domain(self) -> Domain:
        """Return the members of this variable, over its own sets."""
        if self._domain is None:
            return Domain.full(self.dims, self.coords)
        return self._domain

    @property
    def _numbered(self) -> tuple[Any, int, int]:
        """Return this variable's coordinate, start and total column count.

        Raises ValueError where the variable is declared and has no columns.
        """
        if self.coord is None or self.start is None or self.total_columns is None:
            raise ValueError(
                f"variable {self.name!r} is declared and has no columns; "
                f"bind it before reading them"
            )
        return self.coord, self.start, self.total_columns

    @property
    def n_columns(self) -> int:
        """Number of columns this variable occupies."""
        return len(self._numbered[0])

    def __len__(self) -> int:
        return self.n_columns

    def _columns(self) -> slice:
        """Return the slice of the column indices of this variable.

        Raises ValueError for a variable with no numbering.
        """
        if self.start is None:
            raise ValueError(
                f"variable {self.name!r} has no numbering; declare it with Model.var"
            )
        return slice(self.start, self.start + self.n_columns)

    def __repr__(self) -> str:
        if self.declared:
            return f"Variable({self.name!r}, {self.dims}, declared)"
        return f"Variable({self.name!r}, {self.dims}, {self.n_columns} columns)"

    def __getitem__(self, sets: Any) -> "Expression":
        """Return a one-term expression over this variable, at the sets given.

        A set given as `T - 1` reads the previous member, and the lag is
        recorded on the term. A label in place of a set fixes that dimension
        at one member, and that dimension is not in the frame.
        """
        from nimopt.sets import read_at
        from nimopt.term import Expression, Term

        shifts, fixed = read_at(
            f"variable {self.name!r}", self.dims, self.sets, sets, lags=True
        )
        return Expression([Term(self, shifts=shifts, fixed=fixed)])

    kind = "variable"
    expresses = "term"

    def terms(self) -> SparseArray:
        """Return one entry per column of this variable, each valued 1.0.

        The array is over `(*dims, COLUMN)`. A member's column is its rank
        among this variable's members plus the variable's start. That rule
        covers a full product and a subset alike.
        """
        _, start, total = self._numbered
        return self.domain().identity(COLUMN, ProductCoord((total,)), start=start)

    def write_bounds(
        self, lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64]
    ) -> None:
        """Write this variable's columns of `lower` and `upper` in place.

        A scalar bound fills the range. A Param is scattered through the
        variable's own coordinate, mapping each member's multi-index to its
        column. The two vectors belong to the model. No array is allocated
        per variable.
        """
        _, start, _total = self._numbered
        at = slice(start, start + self.n_columns)
        self._write_bound(lower[at], "lower")
        self._write_bound(upper[at], "upper")

    def bound_array(self, which: str) -> float | SparseArray:
        """Return the lower or the upper bound at each member of this variable.

        `which` is "lower" or "upper". A number is returned as a float. A
        Param is returned as an array over the variable's dimensions,
        replicated over the dimensions it lacks and restricted to the
        variable's members. Raises ValueError for a Param with no value or a
        value that is not a number at a member.
        """
        if which not in ("lower", "upper"):
            raise ValueError(f"which is 'lower' or 'upper'; got {which!r}")
        bound = self.lower if which == "lower" else self.upper
        if not isinstance(bound, Param):
            return bound
        array = bound.materialise().broadcast(self.dims, self.coords)
        if self._domain is not None:
            array = array.restrict(self._domain)
        covered = array.domain(self.dims)
        if covered.size != self.n_columns:
            self._reject_uncovered(covered, bound, which)
        unstated = np.flatnonzero(np.isnan(array.values()))
        if unstated.size:
            self._reject_not_a_number(covered, bound, which, int(unstated[0]))
        return array

    def _write_bound(self, target: npt.NDArray[np.float64], which: str) -> None:
        """Fill `target`, a view of one variable's columns, from one bound."""
        array = self.bound_array(which)
        if not isinstance(array, SparseArray):
            target[:] = array
            return
        coord, start, _total = self._numbered
        at = coord.to_position(array.coordinates()) - start
        target[at] = array.values()

    def _reject_not_a_number(
        self, covered: Domain, bound: Param, which: str, at: int
    ) -> None:
        """Raise ValueError for a bound that is not a number at one member."""
        labels = covered.labels()
        named = tuple(label_text(labels[d][at]) for d in self.dims)
        raise ValueError(
            f"the {which} bound {bound.name!r} is not a number at member "
            f"{named} of variable {self.name!r}; give a finite value or an "
            f"infinity"
        )

    def _reject_uncovered(self, covered: Domain, bound: Param, which: str) -> None:
        """Raise ValueError for a bound with no value at one column."""
        labels = self.domain().difference(covered).labels()
        named = tuple(label_text(labels[d][0]) for d in self.dims)
        raise ValueError(
            f"the {which} bound {bound.name!r} has no value at member "
            f"{named} of variable {self.name!r}; give the bound a value at "
            f"every member of the variable"
        )
