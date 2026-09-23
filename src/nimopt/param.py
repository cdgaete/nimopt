"""A named coefficient array."""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from nimblend import DenseArray, Domain, SparseArray, from_long

from nimopt.sets import as_members, as_sets, coords_of, subset_of
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
        self.sets = as_sets(sets, f"parameter {self.name!r}")
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
        sets = as_sets(sets, f"parameter {name!r}")
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
        sets = as_sets(sets, f"parameter {name!r}")
        dims = tuple(s.name for s in sets)
        converted = dict(columns)
        for held in sets:
            if held.name in converted and held.labels is not None:
                converted[held.name] = as_members(
                    converted[held.name],
                    held.labels.dtype,
                    f"column {held.name!r} of parameter {name!r}",
                )
        try:
            array = from_long(dims, coords_of(sets), converted, values)
        except ValueError as error:
            raise ValueError(f"parameter {name!r}: {error}") from None
        return cls(name, sets, array)

    @classmethod
    def from_positions(
        cls,
        name: str,
        sets: Iterable[Any],
        index: npt.ArrayLike,
        values: npt.ArrayLike,
    ) -> "Param":
        """Return a parameter from an index matrix and one value column.

        `index` has one row per set, and each column identifies one member by
        its position in each set. A caller with positions resolves no labels.
        Raises ValueError for a position outside its set, a member given
        twice, and a value count that differs from the column count.
        """
        sets = as_sets(sets, f"parameter {name!r}")
        index = np.asarray(index)
        values = np.asarray(values, dtype=np.float64)
        try:
            members = subset_of(sets, index)
        except ValueError as error:
            raise ValueError(f"parameter {name!r}: {error}") from None
        if values.shape != (index.shape[1],):
            raise ValueError(
                f"parameter {name!r} has {index.shape[1]} index columns and "
                f"{values.size} values; pass one value per column"
            )
        ordered = np.empty(members.size, dtype=np.float64)
        ordered[members.positions_of_coordinates(index)] = values
        return cls(name, sets, members.array(ordered))

    @classmethod
    def from_array(
        cls, name: str, sets: Iterable[Any], array: DenseArray | SparseArray
    ) -> "Param":
        """Return a parameter holding the present entries of a nimblend array.

        The array is over the sets' names, in any order. Its labels are
        members of the sets, in any order and extent. An absent coordinate
        has no coefficient, under absence 'empty' and 'unknown' alike. Raises
        TypeError for an object that is not a nimblend array, and ValueError
        for other dimension names, a set with no members and a label outside
        its set.
        """
        if not isinstance(array, DenseArray | SparseArray):
            raise TypeError(
                f"parameter {name!r} is given {type(array).__name__}; pass a "
                f"nimblend DenseArray or SparseArray"
            )
        sets = as_sets(sets, f"parameter {name!r}")
        dims = tuple(s.name for s in sets)
        if sorted(array.dims) != sorted(dims):
            raise ValueError(
                f"parameter {name!r} is declared over {dims} and its array is "
                f"over {tuple(array.dims)}; pass an array over {dims}"
            )
        labels = array.domain().labels()
        for held in sets:
            if held.labels is None:
                raise ValueError(
                    f"set {held.name!r} of parameter {name!r} has no members; "
                    f"bind the set before the parameter"
                )
            members = np.asarray(
                as_members(
                    labels[held.name],
                    held.labels.dtype,
                    f"array of parameter {name!r}",
                )
            )
            foreign = ~np.isin(members, held.labels)
            if foreign.any():
                first = members[foreign].tolist()[0]
                raise ValueError(
                    f"parameter {name!r} has label {first!r} on "
                    f"{held.name!r} outside the set; pass labels of the set"
                )
            labels[held.name] = members
        return cls.from_long(name, sets, labels, array.values())

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
        from nimopt.sets import read_at
        from nimopt.term import ParamRef

        _, fixed = read_at(
            f"parameter {self.name!r}", self.dims, self.sets, sets, lags=False
        )
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
