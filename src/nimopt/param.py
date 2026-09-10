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

    A `Param` is a named `nimblend` array. Its values enter a constraint
    through the array's own product and reduction, with no conversion step.
    A solved model's duals are usable directly as another model's
    coefficients.
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
        """True where this parameter has no values."""
        return self.array is None

    def _bind(self, array: SparseArray) -> None:
        if array.dims != self.dims:
            raise ValueError(
                f"parameter {self.name!r} is declared over {self.dims} and its "
                f"array is over {array.dims}; bind an array over {self.dims}"
            )
        if array.absence != "empty":
            raise ValueError(
                f"parameter {self.name!r} is bound to an array declaring "
                f"absence {array.absence!r}; bind an array declaring absence "
                f"'empty'"
            )
        self.array = array

    @classmethod
    def from_dense(
        cls, name: str, sets: Iterable[Any], values: npt.ArrayLike
    ) -> "Param":
        """Return a parameter holding every cell of `values` as a coefficient.

        `values` has the shape of the set product. The cells are read in
        row-major order, the order of the full product's members. Raises
        ValueError for any other shape.
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
        """Return a parameter from one label column per set and one value column.

        Each label resolves through the coordinate its set already stores.
        """
        sets = tuple(sets)
        dims = tuple(s.name for s in sets)
        try:
            array = from_long(dims, coords_of(sets), columns, values)
        except ValueError as error:
            raise ValueError(f"parameter {name!r}: {error}") from None
        return cls(name, sets, array)

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the names of the sets this parameter is declared over."""
        return tuple(s.name for s in self.sets)

    def __getitem__(self, sets: Any) -> "ParamRef":
        """Return a reference to this parameter at the sets given.

        A label in place of a set fixes that dimension at one member, and the
        reference is not over that dimension. Raises ValueError for a lag and
        for a set list that differs from the declared dimensions.
        """
        from nimopt.sets import check_members, reference
        from nimopt.term import ParamRef

        given, shifts, fixed = reference(sets, self.dims)
        if shifts:
            raise ValueError(
                f"parameter {self.name!r} is read at a lag {sorted(shifts)}; "
                f"write the lag at the variable's reference"
            )
        if given != self.dims:
            raise ValueError(
                f"parameter {self.name!r} is declared over {self.dims}; got {given}"
            )
        check_members(self.sets, fixed, f"parameter {self.name!r}")
        return ParamRef(self, fixed)

    def materialise(self) -> SparseArray:
        """Return the array bound to this parameter.

        Raises ValueError where the parameter is declared and has no array.
        """
        if self.array is None:
            raise ValueError(
                f"parameter {self.name!r} is declared and has no values; "
                f"bind it before reading them"
            )
        return self.array

    @property
    def nnz(self) -> int:
        """Return the number of coefficients this parameter contains."""
        return self.materialise().nnz

    kind = "parameter"
    expresses = "coefficient"

    def __repr__(self) -> str:
        if self.declared:
            return f"Param({self.name!r}, {self.dims}, declared)"
        return f"Param({self.name!r}, {self.dims}, {self.nnz} coefficients)"
