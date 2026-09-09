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
from nimopt.sets import coords_of, rows_of
from nimopt.symbol import Symbol


class Variable(Symbol):
    """A variable over a set product, or over a subset of one.

    The variable's columns are a virtual coordinate: a member's column is
    computed from its multi-index by stride arithmetic for a full product, or
    is its rank among a subset's codes. Nothing stores a column index, which
    is what makes a variable over millions of columns cost only its members.
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
        self.sets = tuple(sets)
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
                    f"variable {self.name!r} is numbered from {start} and "
                    f"states no total; a start and a total are given together"
                )
            self._bind(start, total_columns)

    @property
    def declared(self) -> bool:
        """Whether this variable is still awaiting its numbering."""
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
                f"the members given span {members.dims} of sizes "
                f"{members.shape}; variable {self.name!r} is over "
                f"{self.dims} of sizes {sizes}"
            )
        self._domain = members
        self.coord = members.as_coord(self.start)

    def _bound(self, bound: float | Param, which: str) -> float | Param:
        """The bound as given, refusing a parameter over other dimensions.

        A parameter over fewer of the variable's dimensions is a bound every
        column sharing those coordinates takes.
        """
        if isinstance(bound, Param):
            lacking = [d for d in bound.dims if d not in self.dims]
            if lacking:
                raise ValueError(
                    f"variable {self.name!r} is declared over {self.dims} and "
                    f"does not carry {lacking}; its {which} bound "
                    f"{bound.name!r} is declared over {bound.dims}"
                )
            return bound
        value = float(bound)
        if np.isnan(value):
            raise ValueError(
                f"the {which} bound of variable {self.name!r} is not a "
                f"number; a bound states a finite value or an infinity"
            )
        return value

    @property
    def dims(self) -> tuple[str, ...]:
        """The names of the sets this variable is declared over."""
        return tuple(s.name for s in self.sets)

    @property
    def coords(self) -> dict[str, Any]:
        """Each dimension this variable carries, by its set's coordinate."""
        return coords_of(self.sets)

    def domain(self) -> Domain:
        """The members this variable carries, over its own sets."""
        if self._domain is None:
            return Domain.full(self.dims, self.coords)
        return self._domain

    @property
    def _numbered(self) -> tuple[Any, int, int]:
        """This variable's coordinate, start and extent, once it has them."""
        if self.coord is None or self.start is None or self.total_columns is None:
            raise ValueError(
                f"variable {self.name!r} is declared and carries no columns; "
                f"bind it before reading them"
            )
        return self.coord, self.start, self.total_columns

    @property
    def n_columns(self) -> int:
        """Number of columns this variable occupies."""
        return len(self._numbered[0])

    def __len__(self) -> int:
        return self.n_columns

    def __repr__(self) -> str:
        if self.declared:
            return f"Variable({self.name!r}, {self.dims}, declared)"
        return f"Variable({self.name!r}, {self.dims}, {self.n_columns} columns)"

    def __getitem__(self, sets: Any) -> "Expression":
        """A one-term expression over this variable, at the sets given.

        A set given as `T - 1` reads the previous member, and the lag is
        recorded on the term rather than applied to an array here. A label in
        place of a set fixes that dimension at one member, which leaves the
        frame.
        """
        from nimopt.sets import check_members, reference
        from nimopt.term import Expression, Term

        given, shifts, fixed = reference(sets, self.dims)
        if given != self.dims:
            raise ValueError(
                f"variable {self.name!r} is declared over {self.dims}; got {given}"
            )
        check_members(self.sets, fixed, f"variable {self.name!r}")
        return Expression([Term(self, shifts=shifts, fixed=fixed)])

    kind = "variable"
    states = "term"

    def terms(self) -> SparseArray:
        """One entry per column of this variable, valued 1.0.

        Over `(*dims, COLUMN)`. A member's column is its rank among the
        members this variable carries plus the variable's start, which is
        both numbering rules, so the domain pairs each member with its own
        column and the entries are canonical as written.
        """
        _, start, total = self._numbered
        return self.domain().identity(COLUMN, ProductCoord((total,)), start=start)

    def write_bounds(
        self, lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64]
    ) -> None:
        """Write this variable's columns of `lower` and `upper` in place.

        A scalar bound fills the range; a parameter is scattered through the
        variable's own coordinate, which maps a member's multi-index to its
        column under both numbering rules. The vectors are the model's, so
        nothing is allocated per variable.
        """
        _, start, _total = self._numbered
        at = slice(start, start + self.n_columns)
        self._write_bound(lower[at], self.lower, "lower")
        self._write_bound(upper[at], self.upper, "upper")

    def _write_bound(
        self, target: npt.NDArray[np.float64], bound: float | Param, which: str
    ) -> None:
        """Fill `target`, a view of one variable's columns, from `bound`.

        A bound over fewer dimensions than the variable is replicated across
        the extent of the rest before it is scattered, so one value reaches
        every column that shares its coordinates. It is then read in the
        variable's own dimension order, because a bound stating the same
        dimensions in another order scatters into other columns. A variable
        over a subset takes the members it carries: replication states the
        whole product, and only the columns the variable holds are bounded.
        """
        if not isinstance(bound, Param):
            target[:] = bound
            return
        array = bound.materialise()
        missing = tuple(d for d in self.dims if d not in bound.dims)
        if missing:
            array = array.expand(missing, {d: self.coords[d] for d in missing})
        array = array.transpose(*self.dims)
        if self._domain is not None:
            array = array.restrict(self._domain)
        carried = array.domain(self.dims)
        if carried.size != self.n_columns:
            self._refuse_uncovered(carried, bound, which)
        values = array.values()
        unstated = np.flatnonzero(np.isnan(values))
        if unstated.size:
            self._refuse_not_a_number(carried, bound, which, int(unstated[0]))
        coord, start, _total = self._numbered
        at = coord.to_position(array.coordinates()) - start
        target[at] = values

    def _refuse_not_a_number(
        self, carried: Domain, bound: Param, which: str, at: int
    ) -> None:
        """Refuse a bound whose value at one member is not a number.

        A solver reads NaN as neither a value nor an infinity, so it is
        refused where the bound is read rather than passed to the seam.
        """
        labels = carried.labels()
        named = tuple(str(labels[d][at]) for d in self.dims)
        raise ValueError(
            f"the {which} bound {bound.name!r} states no number for member "
            f"{named} of variable {self.name!r}; a bound states a finite "
            f"value or an infinity"
        )

    def _refuse_uncovered(self, carried: Domain, bound: Param, which: str) -> None:
        """Refuse a bound that leaves a column of the variable unbounded."""
        labels = self.domain().difference(carried).labels()
        named = tuple(str(labels[d][0]) for d in self.dims)
        raise ValueError(
            f"the {which} bound {bound.name!r} carries no value for member "
            f"{named} of variable {self.name!r}; a bound covers every "
            f"column of the variable it bounds"
        )
