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

    Every array declares `absence="unknown"`. A coordinate the model does not
    define has no value, and combining two instances' results does not add a
    zero for it.

    A variable over a full product has a value at every cell of its frame. The
    solver returns those values in the order the columns are numbered, and they
    reshape into a `DenseArray` with no index built at all. A variable over a
    subset has a value at its members alone and stays a `SparseArray`. A dual
    follows its constraint's row domain by the same rule.

    `status` and `feasible` are readable after any solve. `objective` and
    `primal` raise ValueError where `feasible` is False. `dual` raises
    ValueError where `status` is not `optimal`, and raises for a model with
    integer columns. A mixed-integer model has no duals of its own, and the
    adapters report none for one.
    """

    def __init__(
        self,
        model: "Model",
        status: str,
        feasible: bool,
        objective: float,
        bound: float | None,
        col_value: npt.NDArray[np.float64],
        row_dual: npt.NDArray[np.float64] | None,
        rows_of: Mapping[str, Any],
        solver: str,
    ) -> None:
        self.model = model
        self.status = status
        self.solver = solver
        self.feasible = feasible
        self._objective = objective
        self._bound = bound
        self._col_value = col_value
        self._row_dual = row_dual
        self._rows_of = rows_of

    def __repr__(self) -> str:
        if not self.feasible:
            return f"Solution({self.status!r}, no values)"
        head = f"Solution({self.status!r}, objective {self._objective:g}"
        gap = self.gap
        if gap is None:
            return f"{head}, no bound)"
        if gap == 0.0:
            return f"{head})"
        return f"{head}, gap {gap:.2%})"

    def _require_feasible(self) -> None:
        """Raise where the solver reports no feasible point."""
        if not self.feasible:
            raise ValueError(
                f"status is {self.status!r} and the solver reports no feasible "
                f"point; read `status` before reading values"
            )

    @property
    def objective(self) -> float:
        """Return the objective value of the point the solver reported.

        Raises ValueError where `feasible` is False.
        """
        self._require_feasible()
        return self._objective

    @property
    def bound(self) -> float | None:
        """Return the bound on the optimal objective the solver proved.

        The bound is a lower bound under sense `min` and an upper bound under
        sense `max`. It is None where the solver reports none.
        """
        return self._bound

    @property
    def gap(self) -> float | None:
        """Return the relative distance from the objective to the bound.

        It is None where `feasible` is False or `bound` is None. For an
        objective of zero it is 0.0 under a bound of zero and infinity under
        any other bound.
        """
        if not self.feasible or self._bound is None:
            return None
        if self._objective == 0.0:
            return 0.0 if self._bound == 0.0 else float("inf")
        return abs(self._objective - self._bound) / abs(self._objective)

    def primal(self, name: str) -> DenseArray | SparseArray:
        """Return the named variable's values over its own sets.

        Raises ValueError where `feasible` is False.
        """
        self._require_feasible()
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
        """Return the named constraint's duals over its free sets.

        Raises ValueError where `status` is not `optimal`. Raises ValueError
        for a model with integer columns.
        """
        if self.status != "optimal":
            raise ValueError(
                f"status is {self.status!r}; duals are defined at status 'optimal' only"
            )
        if self._row_dual is None:
            raise ValueError(
                f"model {self.model.name!r} has integer columns and "
                f"{self.solver!r} reports no duals for it; read primal values "
                f"only"
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
