"""A solved model's primals and duals, as labeled arrays."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from nimblend import DenseArray, SparseArray

if TYPE_CHECKING:
    from nimopt.model import Model

NO_FINITE_OPTIMUM = ("unbounded", "unbounded_or_infeasible")


class Solution:
    """Primal and dual values, read back onto the sets they were declared over.

    Every array declares `absence="unknown"`. A coordinate the model does not
    define has no value.

    A variable over a full product has a value at every cell of its frame, and
    reshapes into a `DenseArray`. A variable over a subset has a value at its
    members alone and is a `SparseArray`. A dual follows its constraint's row
    domain by the same rule.

    `status` and `feasible` are readable after any solve. `objective` and
    `primal` raise ValueError where `feasible` is False. `objective`, `primal`
    and `gap` raise ValueError at status `unbounded` and
    `unbounded_or_infeasible`, whatever `feasible` reports. `dual` raises
    ValueError where `status` is not `optimal`, and for a model with integer
    columns.
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
        if not self.feasible or self.status in NO_FINITE_OPTIMUM:
            return f"Solution({self.status!r}, no values)"
        head = f"Solution({self.status!r}, objective {self._objective:g}"
        gap = self.gap
        if gap is None:
            return f"{head}, no bound)"
        if gap == 0.0:
            return f"{head})"
        return f"{head}, gap {gap:.2%})"

    def _reject_unbounded(self) -> None:
        """Raise ValueError at a status reporting no finite optimum."""
        if self.status == "unbounded":
            raise ValueError(
                "status is 'unbounded' and the objective has no finite "
                "optimum; read `Session.diagnose()` for the ray"
            )
        if self.status == "unbounded_or_infeasible":
            raise ValueError(
                "status is 'unbounded_or_infeasible' and the model has no "
                "finite optimum; read `status` and `Session.diagnose()`"
            )

    def _require_feasible(self) -> None:
        """Raise ValueError where the solution defines no value to read."""
        self._reject_unbounded()
        if not self.feasible:
            raise ValueError(
                f"status is {self.status!r} and the solver reports no feasible "
                f"point; read `status` before reading values"
            )

    @property
    def objective(self) -> float:
        """Return the objective value of the point the solver reported.

        Raises ValueError where `feasible` is False. Raises ValueError at
        status `unbounded` and `unbounded_or_infeasible`.
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
        any other bound. Raises ValueError at status `unbounded` and
        `unbounded_or_infeasible`.
        """
        self._reject_unbounded()
        if not self.feasible or self._bound is None:
            return None
        if self._objective == 0.0:
            return 0.0 if self._bound == 0.0 else float("inf")
        return abs(self._objective - self._bound) / abs(self._objective)

    def primal(self, name: str) -> DenseArray | SparseArray:
        """Return the named variable's values over its own sets.

        Raises KeyError for a name that is not a declared variable. Raises
        ValueError where `feasible` is False. Raises ValueError at status
        `unbounded` and `unbounded_or_infeasible`.
        """
        variable = self.model._variable(name, "read it with dual()")
        self._require_feasible()
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

        Raises KeyError for a name that is not a declared constraint. Raises
        ValueError where `status` is not `optimal`. Raises ValueError for a
        model with integer columns.
        """
        constraint = self.model._constraint(name, "read it with primal()")
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
