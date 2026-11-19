"""The matrix a model assembles into, with its row and column data."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from nimblend import Domain, EntryBuffer, ProductCoord, SparseArray

from nimopt.names import COLUMN, ROW
from nimopt.progress import reporter

if TYPE_CHECKING:
    from nimopt.constraint import Constraint
    from nimopt.model import Model


class Written:
    """The nonzeros a pass has written, reported to a progress reporter.

    A term's own count is known only once its constraint finishes. A term
    advances the report by nothing, and a constraint advances it by its own
    coefficients. Reporting the term keeps the line moving while a large
    constraint is built.
    """

    def __init__(self, held: Any, total: int, what: str) -> None:
        self.held = held
        self.done = 0
        self.what = ""
        if held is not None:
            held.start(total, what)

    def term(self, term: Any) -> None:
        """Report that a term of the constraint being written is built."""
        if self.held is not None:
            self.what = term.variable.name
            self.held.step(self.done, self.what)

    def constraint(self, name: str, nnz: int) -> None:
        """Report that a constraint of `nnz` coefficients is written."""
        self.done += nnz
        if self.held is not None:
            self.held.step(self.done, name)

    def close(self) -> None:
        """Report that the pass is finished."""
        if self.held is not None:
            self.held.done()


def check_singleton_rows(value: str) -> None:
    """Raise ValueError for a value of `singleton_rows` that selects no matrix."""
    if value not in ("bounds", "rows"):
        raise ValueError(f"singleton_rows is 'bounds' or 'rows'; got {value!r}")


@dataclass(frozen=True)
class _Block:
    """One constraint's converted rows in the record.

    `at` is their range in the record. `first_column` and `last_column` are
    the smallest and the largest column they contain.
    """

    constraint: "Constraint"
    at: slice
    first_column: int
    last_column: int


class Singletons:
    """The rows an assembly stores as column bounds.

    The record contains the constraint of each converted row and the column and
    the coefficient of that row, in the order of the constraints and of the rows
    inside a constraint. The bound a converted row sets is computed from the
    right-hand side the constraint stores when it is read. The columns are one
    int32 array and the coefficients one float64 array, both allocated at the
    number of converted rows the constraints mark.
    """

    def __init__(self, n_rows: int = 0) -> None:
        self._n_rows = int(n_rows)
        self._columns = np.empty(self._n_rows, dtype=np.int32)
        self._coefficients = np.empty(self._n_rows, dtype=np.float64)
        self._blocks: dict[str, _Block] = {}
        self._written = 0

    def __repr__(self) -> str:
        return f"Singletons({self._n_rows} rows, {len(self._blocks)} constraints)"

    @property
    def n_rows(self) -> int:
        """Return the number of converted rows."""
        return self._n_rows

    def add(
        self,
        name: str,
        constraint: "Constraint",
        columns: npt.NDArray[np.int32],
        coefficients: npt.NDArray[np.float64],
    ) -> None:
        """Write the column and the coefficient of each converted row.

        The rows are numbered after the converted rows already written. Raises
        ValueError where the record has no room for them.
        """
        start = self._written
        columns = np.asarray(columns, dtype=np.int32)
        stop = start + int(columns.size)
        if stop > self._n_rows:
            raise ValueError(
                f"constraint {name!r} writes {stop - start} converted rows at "
                f"{start} and the record has {self._n_rows} rows; allocate the "
                f"record at the number of rows the constraints mark"
            )
        self._columns[start:stop] = columns
        self._coefficients[start:stop] = coefficients
        self._blocks[name] = _Block(
            constraint, slice(start, stop), int(columns.min()), int(columns.max())
        )
        self._written = stop

    def rows_of(self, name: str) -> slice | None:
        """Return the range of the converted rows of a constraint.

        Returns None for a constraint with no converted row.
        """
        held = self._blocks.get(name)
        return None if held is None else held.at

    def kept(self, name: str) -> npt.NDArray[np.int64] | None:
        """Return the positions of the kept rows of a constraint.

        The positions are among the rows of the constraint. Returns None for a
        constraint with no converted row.
        """
        held = self._blocks.get(name)
        if held is None:
            return None
        return held.constraint.kept_positions()

    def sides_set(
        self,
        column: int,
        col_lower: float,
        col_upper: float,
        lower: bool,
        upper: bool,
    ) -> int:
        """Return the requested sides of a column whose bound a converted row sets.

        `lower` and `upper` select the sides to check. Bit 1 is set where a
        converted row sets the lower bound of the column and bit 2 where a
        converted row sets its upper bound. The bounds of the blocks that
        contain `column` are computed.
        """
        found = 0
        wanted = int(bool(lower)) + int(bool(upper))
        for held in self._touching(column, column + 1):
            if found.bit_count() == wanted:
                break
            block_lower, block_upper = self._block_bounds(held)
            on_column = np.flatnonzero(self._columns[held.at] == column)
            for side, bound, merged in (
                (1, block_lower, col_lower),
                (2, block_upper, col_upper),
            ):
                if found & side or not (lower if side == 1 else upper):
                    continue
                at = bound[on_column]
                if np.any(np.isfinite(at) & (at == merged)):
                    found |= side
        return found

    def setting(
        self,
        column: int,
        col_lower: float,
        col_upper: float,
        lower: bool,
        upper: bool,
    ) -> tuple[tuple[str, int], ...]:
        """Return the constraint and the position of each row that sets a bound.

        `lower` and `upper` select the sides to report. The result contains the
        first converted row of `column` whose bound on a requested side equals
        the bound of the column on that side. A position is among the rows of
        that constraint. A row that sets both requested bounds appears once. The
        bounds of the blocks that contain `column` are computed.
        """
        blocks = self._touching(column, column + 1)
        if not blocks or not (lower or upper):
            return ()
        first: dict[int, int] = {}
        wanted = int(bool(lower)) + int(bool(upper))
        for held in blocks:
            if len(first) == wanted:
                break
            block_lower, block_upper = self._block_bounds(held)
            on_column = np.flatnonzero(self._columns[held.at] == column)
            for side, bound, merged in (
                (1, block_lower, col_lower),
                (2, block_upper, col_upper),
            ):
                if side in first or not (lower if side == 1 else upper):
                    continue
                at = bound[on_column]
                sets = on_column[np.isfinite(at) & (at == merged)]
                if sets.size:
                    first[side] = held.at.start + int(sets[0])
        found: list[tuple[str, int]] = []
        for side in (1, 2):
            if side in first:
                placed = self._placed(first[side])
                if placed not in found:
                    found.append(placed)
        return tuple(found)

    def clip(
        self,
        start: int,
        stop: int,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
    ) -> None:
        """Narrow the bounds of the columns `start` to `stop` in place.

        A converted row whose column is in the range sets a lower or an upper
        bound. The lower bound of a column is the largest of its bound and of
        the lower bounds set on it; the upper bound is the smallest. Each block
        that contains a column of the range is read at the size of that block.
        """
        for held in self._touching(start, stop):
            block_lower, block_upper = self._block_bounds(held)
            columns = self._columns[held.at]
            inside = (columns >= start) & (columns < stop)
            at = columns[inside] - start
            np.maximum.at(lower, at, block_lower[inside])
            np.minimum.at(upper, at, block_upper[inside])

    def duals(
        self,
        reduced: npt.NDArray[np.float64],
        col_lower: npt.NDArray[np.float64],
        col_upper: npt.NDArray[np.float64],
        sense: str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the dual of each converted row and the declared reduced costs.

        `reduced` contains the reduced cost of each column of the loaded
        model, and is not changed. Under sense `min` the nonnegative part of a
        reduced cost belongs to the lower bound of the column and the
        nonpositive part to the upper bound; under sense `max` the two parts
        are exchanged. On each side the part belongs to the first converted
        row whose bound equals the bound of the column on that side, and every
        other converted row on that side has the dual zero. Where no converted
        row has that bound, the part remains in the reduced cost of the
        declared model. The duals are in the order of the record.
        """
        declared = reduced.copy()
        shares = np.zeros(self._n_rows, dtype=np.float64)
        if self._written == 0:
            return shares, declared
        positive = np.maximum(reduced, 0.0)
        lower_part = positive if sense == "min" else reduced - positive
        lower, upper = self._bounds()
        columns = self._columns[: self._written]
        for part, bound, merged in (
            (lower_part, lower, col_lower),
            (reduced - lower_part, upper, col_upper),
        ):
            sets = np.flatnonzero(np.isfinite(bound) & (bound == merged[columns]))
            first = np.unique(columns[sets], return_index=True)[1]
            at = columns[sets][first]
            shares[sets[first]] += part[at]
            declared[at] -= part[at]
        return shares / self._coefficients, declared

    def _touching(self, start: int, stop: int) -> list[_Block]:
        """Return the blocks that contain a column from `start` to `stop`."""
        return [
            held
            for held in self._blocks.values()
            if held.first_column < stop and held.last_column >= start
        ]

    def _placed(self, index: int) -> tuple[str, int]:
        """Return the constraint and the position of the converted row `index`."""
        for name, held in self._blocks.items():
            if held.at.start <= index < held.at.stop:
                position = held.constraint.singleton_positions()[index - held.at.start]
                return name, int(position)
        raise ValueError(
            f"converted row {index} is outside the {self._written} rows of the record"
        )

    def _block_bounds(
        self, held: _Block
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the lower and the upper bound the rows of one block set."""
        constraint, at = held.constraint, held.at
        coefficients = self._coefficients[at]
        value = constraint.singleton_rhs() / coefficients
        lower = np.full(value.size, -np.inf, dtype=np.float64)
        upper = np.full(value.size, np.inf, dtype=np.float64)
        if constraint.sense == "==":
            lower[:] = value
            upper[:] = value
            return lower, upper
        positive = coefficients > 0
        sets_lower = positive if constraint.sense == ">=" else ~positive
        lower[sets_lower] = value[sets_lower]
        upper[~sets_lower] = value[~sets_lower]
        return lower, upper

    def _bounds(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the lower and upper bound each converted row sets."""
        lower = np.full(self._written, -np.inf, dtype=np.float64)
        upper = np.full(self._written, np.inf, dtype=np.float64)
        for held in self._blocks.values():
            lower[held.at], upper[held.at] = self._block_bounds(held)
        return lower, upper


def matrix_of(
    constraints: Mapping[str, "Constraint"],
    n_columns: int,
    report: Written,
    singleton_rows: str = "rows",
) -> tuple[
    SparseArray,
    dict[str, slice],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    Singletons,
]:
    """Return the matrix, the row ranges, the row bounds and the `Singletons`.

    Rows are numbered from zero in the order of `constraints`. Each constraint
    writes its block into one buffer. The row bounds are written after the
    matrix is built. Under `singleton_rows="bounds"` a row of one coefficient
    is stored as a bound of its column and is absent from the matrix.
    """
    check_singleton_rows(singleton_rows)
    converted = singleton_rows == "bounds"
    blocks = [
        (
            name,
            constraint,
            constraint.n_rows - constraint.n_singleton_rows,
            constraint.nnz - constraint.n_singleton_rows,
        )
        if converted
        else (name, constraint, constraint.n_rows, constraint.nnz)
        for name, constraint in constraints.items()
    ]
    n_rows = sum(kept for _, _, kept, _ in blocks)
    n_converted = (
        sum(c.n_singleton_rows for c in constraints.values()) if converted else 0
    )
    buffer = EntryBuffer(2, sum(nnz for _, _, _, nnz in blocks))
    rows = ProductCoord((n_rows,))
    rows_of = {}
    singletons = Singletons(n_converted)
    row_start = 0
    for name, constraint, kept, _ in blocks:
        if kept == constraint.n_rows:
            constraint.write_into(buffer, rows, row_start, report)
        else:
            columns, coefficients = constraint.write_kept(
                buffer, rows, row_start, report
            )
            singletons.add(name, constraint, columns, coefficients)
        report.constraint(name, constraint.nnz)
        rows_of[name] = slice(row_start, row_start + kept)
        row_start += kept
    report.close()
    matrix = buffer.array(
        {ROW: rows, COLUMN: ProductCoord((n_columns,))},
        (ROW, COLUMN),
    )
    row_lower = np.empty(n_rows, dtype=np.float64)
    row_upper = np.empty(n_rows, dtype=np.float64)
    for name, constraint, kept, _ in blocks:
        at = rows_of[name]
        if kept == constraint.n_rows:
            constraint.write_bounds(row_lower[at], row_upper[at])
        else:
            constraint.write_kept_bounds(row_lower[at], row_upper[at])
    return matrix, rows_of, row_lower, row_upper, singletons


def assemble_model(
    model: "Model", progress: Any = False, singleton_rows: str = "bounds"
) -> "Assembled":
    """Return a model's matrix and its row and column data, in CSR form.

    Each constraint rebuilds its expression, writes it into its slice of one
    buffer and releases it. One expression is live at a time. The row bounds
    are written into the model's own vectors after the matrix is built, and
    no row is copied twice. `singleton_rows` is `"bounds"` or `"rows"`, and
    raises ValueError for another value.
    """
    check_singleton_rows(singleton_rows)
    report = Written(reporter(progress), model.nnz, f"assembling {model.name}")
    matrix, rows_of, row_lower, row_upper, singletons = matrix_of(
        model.constraints, model.n_columns, report, singleton_rows
    )
    indices, values, indptr = matrix.to_csr()
    col_lower, col_upper = model.column_bounds()
    singletons.clip(0, model.n_columns, col_lower, col_upper)
    return Assembled(
        matrix,
        indices,
        values,
        indptr,
        row_lower,
        row_upper,
        model.objective_coefficients(),
        col_lower,
        col_upper,
        model.integrality(),
        rows_of,
        singletons,
    )


class Appended:
    """The rows a session appends to the matrix it loaded.

    One resizable buffer contains the entries over `(ROW, COLUMN)`, with rows
    numbered from zero inside the block. Each appended constraint keeps its
    name, its row range and its own row bounds. `first_row` is the number of
    rows of the base matrix, and the appended rows follow them in the loaded
    model.
    """

    def __init__(self, n_columns: int, first_row: int) -> None:
        self.first_row = int(first_row)
        self._n_columns = int(n_columns)
        self._buffer = EntryBuffer(2, 0, resizable=True)
        self._n_rows = 0
        self._blocks: list[tuple[str, Constraint, int]] = []
        self._bounds: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        self._rows_of: dict[str, slice] = {}

    def __repr__(self) -> str:
        return f"Appended({self._n_rows} rows, {len(self._blocks)} constraints)"

    @property
    def n_rows(self) -> int:
        """Return the number of appended rows."""
        return self._n_rows

    @property
    def names(self) -> tuple[str, ...]:
        """Return the names of the appended constraints, in the order added."""
        return tuple(name for name, _, _ in self._blocks)

    def add(
        self,
        name: str,
        constraint: "Constraint",
        accept: Callable[..., None] | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
        npt.NDArray[np.int32],
        npt.NDArray[np.float64],
    ]:
        """Append the rows of a constraint; return its bounds and CSR arrays.

        `Constraint.write_into` writes the block once into a buffer sized to the
        constraint, and `SparseArray.group` places its entries in the
        resizable buffer of the appended rows, numbered after the rows already
        appended. The domain of the group is every row of the block. A row
        with no coefficient keeps its position in the constraint. The five
        arrays describe the constraint alone, with its rows numbered from zero.
        They are views of the buffer of `write_into`, and a later append does
        not change them. `accept` is called with the five arrays before the
        rows are recorded, and a call that raises records no row. Raises
        ValueError for a name that is already appended, for a row with a
        coefficient that is not finite, for a row bound that is NaN and for a
        lower bound of infinity or an upper bound of minus infinity.
        """
        if name in self._rows_of:
            raise ValueError(f"row {name!r} is already appended; append each row once")
        buffer = EntryBuffer(2, constraint.nnz)
        rows = ProductCoord((constraint.n_rows,))
        constraint.write_into(buffer, rows, 0)
        block = buffer.array(
            {ROW: rows, COLUMN: ProductCoord((self._n_columns,))}, (ROW, COLUMN)
        )
        indices, values, indptr = block.to_csr()
        row_lower = np.empty(constraint.n_rows, dtype=np.float64)
        row_upper = np.empty(constraint.n_rows, dtype=np.float64)
        constraint.write_bounds(row_lower, row_upper)
        if not np.isfinite(values).all():
            raise ValueError(
                f"row {name!r} has a coefficient that is not finite; give finite "
                f"coefficients"
            )
        if np.isnan(row_lower).any() or np.isnan(row_upper).any():
            raise ValueError(
                f"row {name!r} has a bound that is not a number; give a number or "
                f"an infinity"
            )
        if np.isposinf(row_lower).any() or np.isneginf(row_upper).any():
            raise ValueError(
                f"row {name!r} has a lower bound of infinity or an upper bound of "
                f"minus infinity; give a bound a solution can satisfy"
            )
        if accept is not None:
            accept(row_lower, row_upper, indptr, indices, values)
        total = self._n_rows + constraint.n_rows
        block.group(
            (ROW,),
            into=ROW,
            domain=Domain.full((ROW,), block.coords),
            coord=ProductCoord((total,)),
            start=self._n_rows,
            reserve=self._buffer.reserve,
        )
        self._rows_of[name] = slice(
            self.first_row + self._n_rows, self.first_row + total
        )
        self._blocks.append((name, constraint, self._n_rows))
        self._bounds.append((row_lower, row_upper))
        self._n_rows = total
        return row_lower, row_upper, indptr, indices, values

    def row_of(self, name: str) -> slice:
        """Return the range of rows a constraint occupies in the loaded model.

        Raises KeyError for a name that is not appended.
        """
        if name not in self._rows_of:
            valid = (
                f"use one of {tuple(self._rows_of)}"
                if self._rows_of
                else "append a row with add()"
            )
            raise KeyError(f"appended rows have no constraint {name!r}; {valid}")
        return self._rows_of[name]

    def row(
        self, index: int
    ) -> tuple[
        str,
        int,
        npt.NDArray[np.int32],
        npt.NDArray[np.float64],
        float,
        float,
    ]:
        """Return the name, position, columns, values and bounds of a row.

        `index` is the row's index in the loaded model, at `first_row` or
        above. Raises ValueError for an index outside the appended rows.
        """
        local = int(index) - self.first_row
        for name, constraint, start in reversed(self._blocks):
            if start <= local < start + constraint.n_rows:
                position = local - start
                columns, values, lower, upper = constraint.row_at(position)
                return name, position, columns, values, lower, upper
        raise ValueError(
            f"row {index} is not an appended row; {self._n_rows} row(s) are "
            f"appended after row {self.first_row - 1}"
        )

    def weighted(self, row_dual: npt.NDArray[np.float64]) -> SparseArray | None:
        """Return each column's sum over the appended rows, weighted by duals.

        `row_dual` has one value per appended row. Returns None where no row
        is appended.
        """
        if self._n_rows == 0:
            return None
        return self._matrix().weighted_sum(ROW, row_dual)

    def csr(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
        npt.NDArray[np.int32],
        npt.NDArray[np.float64],
    ]:
        """Return the row bounds and the CSR arrays of every appended row.

        The rows are numbered from zero inside the block. With no row appended
        the five arrays are empty and `indptr` is `[0]`.
        """
        row_lower = (
            np.concatenate([lower for lower, _ in self._bounds])
            if self._bounds
            else np.empty(0, dtype=np.float64)
        )
        row_upper = (
            np.concatenate([upper for _, upper in self._bounds])
            if self._bounds
            else np.empty(0, dtype=np.float64)
        )
        indices, values, indptr = self._matrix().to_csr()
        return row_lower, row_upper, indptr, indices, values

    def _matrix(self) -> SparseArray:
        """Return the appended entries as an array over `(ROW, COLUMN)`."""
        return self._buffer.array(
            {
                ROW: ProductCoord((self._n_rows,)),
                COLUMN: ProductCoord((self._n_columns,)),
            },
            (ROW, COLUMN),
        )


class Assembled:
    """A model's matrix over `(ROW, COLUMN)`, with its row and column data.

    `matrix` is the labeled array the model built. `indices` and `values` are
    views of the one buffer behind it, and only `indptr` is built. The extents
    are read from `matrix`. `singletons` contains the rows stored as column
    bounds.
    """

    def __init__(
        self,
        matrix: SparseArray,
        indices: npt.NDArray[np.int32],
        values: npt.NDArray[np.float64],
        indptr: npt.NDArray[np.int32],
        row_lower: npt.NDArray[np.float64],
        row_upper: npt.NDArray[np.float64],
        col_cost: npt.NDArray[np.float64],
        col_lower: npt.NDArray[np.float64],
        col_upper: npt.NDArray[np.float64],
        integrality: npt.NDArray[np.int32],
        rows_of: Mapping[str, slice],
        singletons: Singletons,
    ) -> None:
        self.matrix = matrix
        self.indices = indices
        self.values = values
        self.indptr = indptr
        self.row_lower = row_lower
        self.row_upper = row_upper
        self.col_cost = col_cost
        self.col_lower = col_lower
        self.col_upper = col_upper
        self.integrality = integrality
        self.singletons = singletons
        self._rows_of = rows_of

    def __repr__(self) -> str:
        return (
            f"Assembled({self.n_rows} rows, {self.n_cols} columns, "
            f"{self.values.size} nonzeros)"
        )

    @property
    def n_rows(self) -> int:
        """Return the number of rows of the matrix."""
        return self.matrix.shape[0]

    @property
    def n_cols(self) -> int:
        """Return the number of columns of the matrix."""
        return self.matrix.shape[1]

    def row_of(self, name: str) -> slice:
        """Return the range of rows the named constraint occupies.

        Raises KeyError for a name that is not a constraint of the matrix.
        """
        if name not in self._rows_of:
            valid = (
                f"use one of {tuple(self._rows_of)}"
                if self._rows_of
                else "declare a constraint and assemble the model again"
            )
            raise KeyError(f"assembled matrix has no constraint {name!r}; {valid}")
        return self._rows_of[name]

    def to_dense(self) -> npt.NDArray[np.float64]:
        """Return the matrix as a dense ndarray."""
        return self.matrix.to_dense()
