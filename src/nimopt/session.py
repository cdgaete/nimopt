"""One open solver model: its solves, the changes to it, and the conflict or ray."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from nimopt.assembly import Appended
from nimopt.bounds import Bounds, target_of
from nimopt.param import Param
from nimopt.row import Row, read, resolve, written
from nimopt.sets import shown
from nimopt.term import Relation
from nimopt.variable import Variable

if TYPE_CHECKING:
    from nimopt.model import Model
    from nimopt.solvers.base import Capabilities
from nimopt.solution import Solution
from nimopt.solvers import adapter


@dataclass(frozen=True)
class ColumnBound:
    """One column of a conflict: its variable, coordinate and two bounds."""

    variable: str
    coordinate: dict
    lower: float
    upper: float


@dataclass(frozen=True)
class RayTerm:
    """One column of a ray: its variable, coordinate and direction."""

    variable: str
    coordinate: dict
    direction: float


@dataclass(frozen=True)
class Diagnosis:
    """Why a model did not solve, as the rows and columns that report it.

    `conflict` contains `Row` objects, and a conflicting row renders as
    `Model.row` renders it. `method` is `"native"` where the solver computed
    the conflict and `"computed"` where nimopt computed it.

    `conflict` is None where the model is not infeasible. `ray` is None where
    the model is not unbounded. An empty tuple reports that the solver
    returned nothing.
    """

    status: str
    solver: str
    method: str
    conflict: tuple | None
    columns: tuple[ColumnBound, ...]
    ray: tuple[RayTerm, ...] | None

    def __repr__(self) -> str:
        lines = [f"{self.status}  {self.solver}  conflict {self.method}"]
        for row in self.conflict or ():
            lines.extend(repr(row).split("\n"))
        for held in self.columns:
            at = ", ".join(f"{d}={shown(v)}" for d, v in held.coordinate.items())
            lines.append(
                f"  bound  {held.variable}[{at}]  [{held.lower:g}, {held.upper:g}]"
            )
        for held in self.ray or ():
            at = ", ".join(f"{d}={shown(v)}" for d, v in held.coordinate.items())
            lines.append(f"  ray    {held.variable}[{at}]  {held.direction:+g}")
        return "\n".join(lines)


class Session:
    """One solver's model, opened on one assembled model and kept open.

    A solve passes the matrix to the solver and reads the values back. A
    conflict or a ray is read from that same solved instance. The matrix is
    assembled when the session opens.

    `fix`, `unfix`, `set_bound` and `add_row` change the model the solver
    loaded, and `solve` then solves that model again on the same solver
    instance. With the simplex method the solver starts from the basis of its
    previous solve. `n_rows` and `n_columns` report the size of the loaded
    model.

    The session keeps a private copy of the model for the rows it appends,
    and the model the session was opened on is not changed.

    The solver's own model is passed only to the adapter that built it.
    """

    def __init__(
        self,
        model: "Model",
        solver: str = "highs",
        options: Mapping[str, Any] | None = None,
        progress: Any = False,
    ) -> None:
        self.model = model
        self.solver = solver
        self.assembled = model.assemble(progress=progress)
        self._declared = self._declarations()
        self._objective = model.objective
        self.options = options
        self._adapter = adapter(solver)
        self._backend = None
        self._status = None
        self._copy = model._session_copy()
        self._bounds = Bounds(self.assembled.col_lower, self.assembled.col_upper)
        self._appended = Appended(self.assembled.n_cols, self.assembled.n_rows)

    def __repr__(self) -> str:
        held = "open" if self._backend is not None else "not solved"
        return f"Session({self.model.name!r}, {self.solver!r}, {held})"

    def __enter__(self) -> "Session":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    @property
    def capabilities(self) -> "Capabilities":
        """Return what this session's adapter does, as shipped."""
        return self._adapter.CAPABILITIES

    @property
    def status(self) -> str | None:
        """Return the status of the last solve, or None before one."""
        return self._status

    @property
    def backend_open(self) -> bool:
        """Return whether this session still stores the solver's model."""
        return self._backend is not None

    @property
    def n_rows(self) -> int:
        """Return the number of rows of the loaded model."""
        return self.assembled.n_rows + self._appended.n_rows

    @property
    def n_columns(self) -> int:
        """Return the number of columns of the loaded model."""
        return self.assembled.n_cols

    def close(self) -> None:
        """Release the solver's model."""
        self._backend = None

    def fix(self, target: Any, value: float | Param) -> None:
        """Pin the columns a target selects to `value`.

        The next `solve` solves the loaded model with those pins. Raises
        TypeError for a target that is not a variable or one reference to a
        variable. Raises ValueError for a variable the session's model does
        not declare, for a value that is not finite, for a value that is not
        integral on a column of an integer variable, and for a Param given for
        a target with a label. Raises ValueError where the model changed after
        the session opened, and where the adapter does not change a loaded
        model. A rejected call changes no bound.
        """
        self._require_current()
        self._require_mutation()
        variable, positions, whole = target_of(self._copy.variables, target)
        self._bounds.fix(variable, positions, value, whole)
        self._write_bounds(variable)

    def unfix(self, target: Any) -> None:
        """Remove the pin of the columns a target selects.

        The next `solve` solves the loaded model with the declared bounds
        restored. Raises TypeError for a target that is not a variable or one
        reference to a variable. Raises ValueError for a variable the session's
        model does not declare, and for a variable with no fixed column. Raises
        ValueError where the model changed after the session opened, and where
        the adapter does not change a loaded model. A rejected call changes no
        bound.
        """
        self._require_current()
        self._require_mutation()
        variable, positions, _ = target_of(self._copy.variables, target)
        self._bounds.unfix(variable, positions)
        self._write_bounds(variable)

    def set_bound(
        self,
        target: Any,
        lower: float | Param | None = None,
        upper: float | Param | None = None,
    ) -> None:
        """Replace the bounds of the columns a target selects.

        The next `solve` solves the loaded model with those bounds. Raises
        TypeError for a target that is not a variable or one reference to a
        variable. Raises ValueError for a variable the session's model does
        not declare, for neither bound given, for a bound that is NaN, for a
        Param given for a target with a label, and for a Param with no value at
        a member of the variable. Raises ValueError where the model changed
        after the session opened, and where the adapter does not change a
        loaded model. A rejected call changes no bound.
        """
        self._require_current()
        self._require_mutation()
        variable, positions, whole = target_of(self._copy.variables, target)
        self._bounds.set_bound(variable, positions, lower, upper, whole)
        self._write_bounds(variable)

    def add_row(
        self,
        name: str,
        relation: Any,
        where: Any = None,
        over: Any = None,
    ) -> None:
        """Append the rows of a constraint to the model the solver loaded.

        `where=` and `over=` have the meaning they have in `Model.constraint`.
        A relation free over a set appends one row per member, after the rows
        already appended. Raises ValueError where the model changed after the
        session opened, where the adapter does not change a loaded model, for
        a name the session's model declares, and for a term over a variable
        the session's model does not declare. A rejected call appends no row.
        """
        self._require_current()
        self._require_mutation()
        self._check_variables(name, relation)
        constraint = self._copy.constraint(name, relation, where, over)
        arrays = self._appended.add(name, constraint)
        if self._backend is not None:
            self._adapter.append_rows(self._backend, *arrays)

    def solve(self) -> Solution:
        """Solve the loaded model and read the values back onto its sets.

        A solve with a loaded model calls `resume` on an adapter that changes
        it. Any other solve passes the assembled model to `solve`. Where rows
        were appended before that call, it appends them to the solver's model
        and resumes, and the result of the resume is the result of the solve.
        Raises ValueError where the model declares a variable or a constraint,
        or sets its objective, after the session opened.
        """
        self._require_current()
        if self._backend is not None and self.capabilities.supports("mutation"):
            result = self._adapter.resume(self._backend, self.options)
        else:
            result = self._adapter.solve(self.assembled, self.model.sense, self.options)
            self._backend = result.backend
            if self._appended.n_rows:
                self._adapter.append_rows(self._backend, *self._appended.csr())
                result = self._adapter.resume(self._backend, self.options)
        self._status = result.status
        constant = self.model.objective_constant
        return Solution(
            self._copy,
            result.status,
            result.feasible,
            result.objective + constant,
            None if result.bound is None else result.bound + constant,
            result.col_value,
            result.row_dual,
            self.assembled,
            self._rows_of(),
            self.solver,
            iterations=result.iterations,
            appended=self._appended,
        )

    def diagnose(self) -> Diagnosis:
        """Return why this session's last solve returned no optimal point.

        The conflict and the ray are read from the solved backend. No
        model is solved a second time.

        Raises ValueError before the first solve, and where the model declares
        a variable or a constraint, or sets its objective, after the session
        opened.
        """
        self._require_current()
        if self._status is None:
            raise ValueError(
                f"session on {self.model.name!r} has not solved; call `solve` "
                f"before `diagnose`"
            )
        conflict, columns, ray = None, (), None
        if self._status == "infeasible":
            conflict, columns = self._conflict()
        if self._status == "unbounded":
            ray = self._ray()
        return Diagnosis(
            status=self._status,
            solver=self.solver,
            method="native",
            conflict=conflict,
            columns=columns,
            ray=ray,
        )

    def _declarations(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the names of the model's variables and constraints."""
        return tuple(self.model.variables), tuple(self.model.constraints)

    def _require_current(self) -> None:
        """Raise ValueError where the model changed after the session opened.

        A change is a declared variable or constraint, or a set objective.
        """
        if self._declarations() != self._declared:
            raise ValueError(
                f"model {self.model.name!r} declares variables or constraints "
                f"that are not in the assembled matrix; open a new Session"
            )
        if self.model.objective is not self._objective:
            raise ValueError(
                f"model {self.model.name!r} sets its objective after the "
                f"session opened; open a new Session"
            )

    def _require_mutation(self) -> None:
        """Raise ValueError where this session's adapter changes no loaded model."""
        if not self.capabilities.supports("mutation"):
            raise ValueError(
                f"{self.solver!r} does not change a loaded model; read "
                f"`capabilities({self.solver!r})`, or open the session with a "
                f"solver that does"
            )

    def _check_variables(self, name: str, relation: Any) -> None:
        """Raise ValueError for a relation over a variable of another model.

        A relation that is not a comparison raises TypeError at
        `Model.constraint`.
        """
        if not isinstance(relation, Relation):
            return
        for term in relation.expression.terms:
            if self._copy.variables.get(term.variable.name) is not term.variable:
                raise ValueError(
                    f"row {name!r} is declared over variable "
                    f"{term.variable.name!r}, not a variable of model "
                    f"{self._copy.name!r}; use one of "
                    f"{tuple(self._copy.variables)}"
                )

    def _write_bounds(self, variable: Variable) -> None:
        """Write a variable's bounds into the assembled vectors and the backend."""
        lower, upper = self._bounds.bounds_of(variable)
        columns = variable._columns()
        self.assembled.col_lower[columns] = lower
        self.assembled.col_upper[columns] = upper
        if self._backend is not None:
            self._adapter.change_column_bounds(
                self._backend,
                np.arange(columns.start, columns.stop, dtype=np.int32),
                lower,
                upper,
            )

    def _rows_of(self) -> dict[str, slice]:
        """Return the row range of every constraint of the loaded model."""
        rows_of = {name: self.assembled.row_of(name) for name in self.model.constraints}
        rows_of.update(
            {name: self._appended.row_of(name) for name in self._appended.names}
        )
        return rows_of

    def _row(self, index: int) -> Row:
        """Return the row at an index of the loaded model."""
        if index < self.assembled.n_rows:
            return read(self._copy, self.assembled, index)
        name, position, columns, values, lower, upper = self._appended.row(index)
        return written(self._copy, name, index, position, columns, values, lower, upper)

    def _conflict(self) -> tuple[tuple[Row, ...], tuple[ColumnBound, ...]]:
        """Return the rows that cannot all be satisfied and the columns they bind."""
        if not self.capabilities.supports("conflict"):
            raise ValueError(
                f"{self.solver!r} computes no conflict; read "
                f"`capabilities({self.solver!r})` for what this adapter does"
            )
        rows, columns = self._adapter.conflict(self._backend)
        return (
            tuple(self._row(int(at)) for at in rows),
            tuple(
                ColumnBound(
                    variable,
                    coordinate,
                    float(self.assembled.col_lower[column]),
                    float(self.assembled.col_upper[column]),
                )
                for _, column, variable, coordinate in resolve(self._copy, columns)
            ),
        )

    def _ray(self) -> tuple[RayTerm, ...] | None:
        """Return the columns of the ray and how far each one moves."""
        if not self.capabilities.supports("ray"):
            raise ValueError(
                f"{self.solver!r} computes no ray; read "
                f"`capabilities({self.solver!r})` for what this adapter does"
            )
        direction = self._adapter.ray(self._backend)
        if direction is None:
            return None
        moving = np.nonzero(direction)[0]
        return tuple(
            RayTerm(variable, coordinate, float(direction[column]))
            for _, column, variable, coordinate in resolve(self._copy, moving)
        )
