"""A solved model's primals and duals, as labeled arrays."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from nimblend import DenseArray, Domain, SparseArray

from nimopt.names import ROW

if TYPE_CHECKING:
    from nimopt.assembly import Assembled
    from nimopt.model import Model

NO_FINITE_OPTIMUM = ("unbounded", "unbounded_or_infeasible")


def reduced_costs(
    assembled: "Assembled", row_dual: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return each column's cost less the duals of the rows it appears in.

    The matrix is summed over its rows, weighted by `row_dual`, with no
    temporary of the size of the matrix.
    """
    weighted = assembled.matrix.weighted_sum(ROW, row_dual)
    return assembled.col_cost - weighted.to_dense()


class Solution:
    """Primal and dual values, read back onto the sets they were declared over.

    Every array declares `absence="unknown"`. A coordinate the model does not
    define has no value.

    A variable over a full product has a value at every cell of its frame, and
    reshapes into a `DenseArray`. A variable over a subset has a value at its
    members alone and is a `SparseArray`. A dual follows its constraint's row
    domain by the same rule.

    `status` and `feasible` are readable after any solve. `objective` and
    `primal` raise ValueError where `feasible` is False. They raise ValueError
    at status `unbounded` and `unbounded_or_infeasible`, whatever `feasible`
    reports. `bound` and `gap` are None where no value is defined, at those two
    statuses included. `dual` raises ValueError where `status` is not
    `optimal`, and for a model with integer columns.

    The reduced costs are computed from `assembled` and the row duals when
    the solution is constructed. The solution stores no reference to
    `assembled`.
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
        assembled: "Assembled",
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
        self._col_dual = (
            None if row_dual is None else reduced_costs(assembled, row_dual)
        )
        self._rows_of = rows_of
        self._variables = frozenset(model.variables)

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

        It is None where `feasible` is False, where `bound` is None, and at
        status `unbounded` and `unbounded_or_infeasible`. For an objective of
        zero it is 0.0 under a bound of zero and infinity under any other bound.
        """
        if self.status in NO_FINITE_OPTIMUM or not self.feasible or self._bound is None:
            return None
        if self._objective == 0.0:
            return 0.0 if self._bound == 0.0 else float("inf")
        return abs(self._objective - self._bound) / abs(self._objective)

    def primal(self, name: str) -> DenseArray | SparseArray:
        """Return the named variable's values over its own sets.

        Raises KeyError for a name that is not a declared variable, and for a
        variable declared after the solve. Raises ValueError where `feasible`
        is False. Raises ValueError at status `unbounded` and
        `unbounded_or_infeasible`.
        """
        variable = self.model._variable(name, "read it with dual()")
        if name not in self._variables:
            raise KeyError(
                f"variable {name!r} is not in the solved matrix; solve the model again"
            )
        self._require_feasible()
        return self._over_variable(variable, self._col_value[variable._columns()])

    def _over_variable(
        self, variable: Any, values: npt.NDArray[np.float64]
    ) -> DenseArray | SparseArray:
        """Return one value per column, read back onto the variable's sets."""
        return _on_members(variable.domain(), values)

    @property
    def has_duals(self) -> bool:
        """Return True where `dual` returns values for this solve.

        Duals are defined at status `optimal`, for a solver that reports them.
        A solver reports no duals for a model with integer columns.
        """
        return (
            self.status == "optimal"
            and self._row_dual is not None
            and self._col_dual is not None
        )

    def _require_dual(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the row duals and the column duals.

        Raises ValueError where the solve defines no dual value to read.
        """
        if self.status != "optimal":
            raise ValueError(
                f"status is {self.status!r}; duals are defined at status 'optimal' only"
            )
        if self._row_dual is None or self._col_dual is None:
            raise ValueError(
                f"model {self.model.name!r} has integer columns and "
                f"{self.solver!r} reports no duals for it; read primal values "
                f"only"
            )
        return self._row_dual, self._col_dual

    def dual(self, name: str, kind: str | None = None) -> DenseArray | SparseArray:
        """Return a constraint's duals over its free sets, or a variable's
        reduced costs over its own sets.

        A model declares its constraints and its variables in two registries,
        and one name identifies one of each. `kind` is `"constraint"` or
        `"variable"` and selects which to read. A reduced cost is the
        variable's objective coefficient less the duals of the rows it
        appears in, weighted by its coefficients in them, in the model's own
        objective. Raises ValueError for a name that identifies both with no
        `kind`, and for a `kind` that is neither. Raises KeyError for a name
        the model does not declare, and for one declared after the solve.
        Raises ValueError where `status` is not `optimal`, and for a model
        with integer columns.
        """
        if kind is not None and kind not in ("constraint", "variable"):
            raise ValueError(f"kind is 'constraint' or 'variable'; got {kind!r}")
        declares_variable = name in self.model.variables
        declares_constraint = name in self.model.constraints
        if kind is None:
            if declares_variable and declares_constraint:
                raise ValueError(
                    f"{name!r} is both a constraint and a variable of model "
                    f"{self.model.name!r}; pass kind='constraint' or "
                    f"kind='variable'"
                )
            kind = "variable" if declares_variable else "constraint"
        if kind == "variable":
            if not declares_variable:
                raise KeyError(
                    f"model {self.model.name!r} has no variable {name!r}; use "
                    f"one of {tuple(self.model.variables)}"
                )
            variable = self.model.variables[name]
            if name not in self._variables:
                raise KeyError(
                    f"variable {name!r} is not in the solved matrix; solve the "
                    f"model again"
                )
            _, col_dual = self._require_dual()
            return self._over_variable(variable, col_dual[variable._columns()])
        if not declares_constraint:
            valid = tuple(self.model.constraints) + tuple(
                n for n in self.model.variables if n not in self.model.constraints
            )
            raise KeyError(
                f"model {self.model.name!r} has no constraint or variable "
                f"{name!r}; use one of {valid}"
            )
        constraint = self.model.constraints[name]
        if name not in self._rows_of:
            raise KeyError(
                f"constraint {name!r} is not in the solved matrix; solve the model "
                f"again"
            )
        row_dual, _ = self._require_dual()
        return _on_members(constraint.rows, row_dual[self._rows_of[name]])


def _on_members(
    members: Domain, values: npt.NDArray[np.float64]
) -> DenseArray | SparseArray:
    """Return one value per member of `members`, as an array of absence unknown.

    A domain that covers its product returns a `DenseArray`, and any other
    domain a `SparseArray`.
    """
    if members.is_full:
        return DenseArray(
            values.reshape(members.shape),
            members.coords,
            members.dims,
            absence="unknown",
        )
    return members.array(values.copy(), absence="unknown")
