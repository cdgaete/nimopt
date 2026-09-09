"""A named coefficient array."""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from nimblend import Domain, SparseArray, from_long

from nimopt.sets import coords_of
from nimopt.symbol import Symbol

if TYPE_CHECKING:
    from nimopt.term import ParamRef


class Param(Symbol):
    """Coefficients over a set product.

    A `Param` is an array with a name: its values reach a constraint through
    the array's own product and reduction, with no marshalling, which is what
    lets a solved model's duals become another model's coefficients.
    """

    def __init__(
        self, name: str, sets: Iterable[Any], array: SparseArray | None = None
    ) -> None:
        self.name = str(name)
        self.sets = tuple(sets)
        self.array = None
        if array is not None:
            self._bind(array)

    @property
    def declared(self) -> bool:
        """Whether this parameter is still awaiting its values."""
        return self.array is None

    def _bind(self, array: SparseArray) -> None:
        if array.dims != self.dims:
            raise ValueError(
                f"parameter {self.name!r} is declared over dimensions "
                f"{self.dims}; its array carries {array.dims}"
            )
        if array.absence != "empty":
            raise ValueError(
                f"a coefficient absent from parameter {self.name!r} is a term "
                f"that is not there, so its array declares absence 'empty'; "
                f"got {array.absence!r}"
            )
        self.array = array

    @classmethod
    def from_dense(
        cls, name: str, sets: Iterable[Any], values: npt.ArrayLike
    ) -> "Param":
        """Every cell of `values` as a coefficient.

        The cells are read in row-major order, which is the order the full
        product carries its members, so the values state the grid without an
        index being built for them.
        """
        sets = tuple(sets)
        values = np.array(values, dtype=np.float64)
        dims = tuple(s.name for s in sets)
        shape = tuple(len(s) for s in sets)
        if values.shape != shape:
            raise ValueError(
                f"parameter {name!r} is over sets of shape {shape}; got values "
                f"of shape {values.shape}"
            )
        members = Domain.full(dims, coords_of(sets))
        return cls(name, sets, members.array(values.ravel()))

    @classmethod
    def from_long(
        cls,
        name: str,
        sets: Iterable[Any],
        columns: Mapping[str, npt.ArrayLike],
        values: npt.ArrayLike,
    ) -> "Param":
        """Coefficients from one label column per set and one value column.

        Each label resolves through the coordinate its set already holds, so
        a dimension's labels are looked up once however many parameters are
        declared over it.
        """
        sets = tuple(sets)
        dims = tuple(s.name for s in sets)
        try:
            array = from_long(dims, coords_of(sets), columns, values)
        except ValueError as refusal:
            raise ValueError(f"parameter {name!r}: {refusal}") from None
        return cls(name, sets, array)

    @property
    def dims(self) -> tuple[str, ...]:
        """The names of the sets this parameter is declared over."""
        return tuple(s.name for s in self.sets)

    def __getitem__(self, sets: Any) -> "ParamRef":
        """A reference to this parameter, checking the sets given.

        A label in place of a set fixes that dimension at one member, so the
        coefficient is read there and the dimension leaves the reference.
        """
        from nimopt.sets import check_members, reference
        from nimopt.term import ParamRef

        given, shifts, fixed = reference(sets, self.dims)
        if shifts:
            raise ValueError(
                f"parameter {self.name!r} is read at a lag {sorted(shifts)}; "
                f"state the lag at the variable's reference, where a "
                f"coefficient multiplies the row it lands on"
            )
        if given != self.dims:
            raise ValueError(
                f"parameter {self.name!r} is declared over {self.dims}; got {given}"
            )
        check_members(self.sets, fixed, f"parameter {self.name!r}")
        return ParamRef(self, fixed)

    def materialise(self) -> SparseArray:
        """The array this parameter carries."""
        if self.array is None:
            raise ValueError(
                f"parameter {self.name!r} is declared and carries no values; "
                f"bind it before reading them"
            )
        return self.array

    @property
    def nnz(self) -> int:
        """Number of coefficients this parameter carries."""
        return self.materialise().nnz

    kind = "parameter"
    states = "coefficient"

    def __repr__(self) -> str:
        if self.declared:
            return f"Param({self.name!r}, {self.dims}, declared)"
        return f"Param({self.name!r}, {self.dims}, {self.nnz} coefficients)"
