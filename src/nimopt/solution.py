"""A solved model's primals and duals, as labeled arrays."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from nimblend import DenseArray, SparseArray

if TYPE_CHECKING:
    from nimopt.model import Model


class Solution:
    """Primal and dual values, read back onto the sets they were declared over.

    Every array declares `absence="unknown"`: a coordinate a model did not
    carry has no value, and combining two instances' results must not invent
    a zero for it.

    Which implementation carries them follows what the model declared. A
    variable over a full product has a value at every cell of its frame, and
    the solver returns them in the order the columns are numbered, so the
    values reshape into a `DenseArray` with no index built at all. A variable
    over a subset carries a value at its members alone, and a dense frame
    would be the grid it was declared to avoid, so those stay a `SparseArray`.
    A dual follows its constraint's row domain by the same rule.

    A model the solver did not bring to an optimum carries no answer, and
    the objective, the primals and the duals refuse to be read: a vector the
    solver left behind is not a solution, and returning it would be
    indistinguishable from one. `status` is readable either way, and is what
    a caller reads first.

    A model carrying integer columns holds no duals at all. A mixed-integer
    model's duals are not the relaxation's, and a solver reporting the
    relaxation's reports a number that means nothing, so the seam hands none
    over and `dual` refuses the read.
    """

    def __init__(
        self,
        model: "Model",
        status: str,
        objective: float,
        col_value: npt.NDArray[np.float64],
        row_dual: npt.NDArray[np.float64] | None,
        rows_of: Mapping[str, Any],
        solver: str,
    ) -> None:
        self.model = model
        self.status = status
        self.solver = solver
        self._objective = objective
        self._col_value = col_value
        self._row_dual = row_dual
        self._rows_of = rows_of

    def __repr__(self) -> str:
        if self.status != "optimal":
            return f"Solution({self.status!r}, no values)"
        return f"Solution({self.status!r}, objective {self._objective:g})"

    def _refuse_unsolved(self, what: str) -> None:
        """Refuse a read of values a non-optimal solve did not produce."""
        if self.status != "optimal":
            raise ValueError(
                f"the model's status is {self.status!r}, so it carries no "
                f"{what}; read `status` before reading values"
            )

    @property
    def objective(self) -> float:
        """The optimal objective value."""
        self._refuse_unsolved("objective")
        return self._objective

    def primal(self, name: str) -> DenseArray | SparseArray:
        """The named variable's values over its own sets."""
        self._refuse_unsolved(f"value for variable {name!r}")
        variable = self.model.variables[name]
        at = slice(variable.start, variable.start + variable.n_columns)
        values = self._col_value[at]
        if variable.subset is None:
            shape = tuple(len(s) for s in variable.sets)
            return DenseArray(
                values.reshape(shape), variable.coords, variable.dims, absence="unknown"
            )
        return variable.domain().array(values.copy(), absence="unknown")

    def dual(self, name: str) -> DenseArray | SparseArray:
        """The named constraint's duals over its free sets."""
        self._refuse_unsolved(f"dual for constraint {name!r}")
        if self._row_dual is None:
            raise ValueError(
                f"this model carries integer columns and {self.solver!r} "
                f"refuses duals for a model with integrality, so there is no "
                f"dual for constraint {name!r} to read: a mixed-integer "
                f"model's duals are not its relaxation's"
            )
        constraint = self.model.constraints[name]
        rows = constraint.rows
        values = self._row_dual[self._rows_of[name]]
        if rows.is_full:
            return DenseArray(
                values.reshape(rows.shape),
                rows.coords,
                rows.dims,
                absence="unknown",
            )
        return rows.array(values.copy(), absence="unknown")
