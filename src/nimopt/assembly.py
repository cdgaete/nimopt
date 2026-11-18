"""The matrix a model assembles into, with its row and column data."""

from collections.abc import Mapping
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


def matrix_of(
    constraints: Mapping[str, "Constraint"], n_columns: int, report: Written
) -> tuple[
    SparseArray, dict[str, slice], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Return the matrix of `constraints`, their row ranges and their row bounds.

    Rows are numbered from zero in the order of `constraints`. Each constraint
    writes its block into one buffer. The row bounds are written after the
    matrix is built.
    """
    n_rows = sum(c.n_rows for c in constraints.values())
    buffer = EntryBuffer(2, sum(c.nnz for c in constraints.values()))
    rows = ProductCoord((n_rows,))
    rows_of = {}
    row_start = 0
    for name, constraint in constraints.items():
        constraint.write_into(buffer, rows, row_start, report)
        report.constraint(name, constraint.nnz)
        rows_of[name] = slice(row_start, row_start + constraint.n_rows)
        row_start += constraint.n_rows
    report.close()
    matrix = buffer.array(
        {ROW: rows, COLUMN: ProductCoord((n_columns,))},
        (ROW, COLUMN),
    )
    row_lower = np.empty(n_rows, dtype=np.float64)
    row_upper = np.empty(n_rows, dtype=np.float64)
    for name, constraint in constraints.items():
        constraint.write_bounds(row_lower[rows_of[name]], row_upper[rows_of[name]])
    return matrix, rows_of, row_lower, row_upper


def assemble_model(model: "Model", progress: Any = False) -> "Assembled":
    """Return a model's matrix and its row and column data, in CSR form.

    Each constraint rebuilds its expression, writes it into its slice of one
    buffer and releases it. One expression is live at a time. The row bounds
    are written into the model's own vectors after the matrix is built, and
    no row is copied twice.
    """
    report = Written(reporter(progress), model.nnz, f"assembling {model.name}")
    matrix, rows_of, row_lower, row_upper = matrix_of(
        model.constraints, model.n_columns, report
    )
    indices, values, indptr = matrix.to_csr()
    col_lower, col_upper = model.column_bounds()
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
        self, name: str, constraint: "Constraint"
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
        not change them. Raises ValueError for a name that is already appended.
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
    are read from `matrix`.
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
