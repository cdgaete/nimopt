"""The matrix a model assembles into, with its row and column data."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from nimblend import EntryBuffer, ProductCoord, SparseArray

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
