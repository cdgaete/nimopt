"""One open solver model, across a solve and the conflict or ray after it."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from nimopt.row import Row, read, resolve

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
            at = ", ".join(f"{d}={v!r}" for d, v in held.coordinate.items())
            lines.append(
                f"  bound  {held.variable}[{at}]  [{held.lower:g}, {held.upper:g}]"
            )
        for held in self.ray or ():
            at = ", ".join(f"{d}={v!r}" for d, v in held.coordinate.items())
            lines.append(f"  ray    {held.variable}[{at}]  {held.direction:+g}")
        return "\n".join(lines)


class Session:
    """One solver's model, opened on one assembled model and kept open.

    A solve passes the matrix to the solver and reads the values back. A
    conflict or a ray is read from that same solved instance. The matrix is
    assembled when the session opens, and a session that solves twice
    assembles once.

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
        self.options = options
        self._adapter = adapter(solver)
        self._backend = None
        self._status = None

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

    def close(self) -> None:
        """Release the solver's model."""
        self._backend = None

    def solve(self) -> Solution:
        """Solve this session's matrix and read the values back onto its sets."""
        result = self._adapter.solve(self.assembled, self.model.sense, self.options)
        self._backend = result.backend
        self._status = result.status
        rows_of = {name: self.assembled.row_of(name) for name in self.model.constraints}
        constant = self.model.objective_constant
        return Solution(
            self.model,
            result.status,
            result.feasible,
            result.objective + constant,
            None if result.bound is None else result.bound + constant,
            result.col_value,
            result.row_dual,
            rows_of,
            self.solver,
        )

    def diagnose(self) -> Diagnosis:
        """Return why this session's last solve returned no optimal point.

        The conflict and the ray are read from the solved backend. No
        model is solved a second time.

        Raises ValueError before the first solve.
        """
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

    def _conflict(self) -> tuple[tuple[Row, ...], tuple[ColumnBound, ...]]:
        """Return the rows that cannot all be satisfied and the columns they bind."""
        if not self.capabilities.supports("conflict"):
            raise ValueError(
                f"{self.solver!r} computes no conflict; read "
                f"`capabilities({self.solver!r})` for what this adapter does"
            )
        rows, columns = self._adapter.conflict(self._backend)
        return (
            tuple(read(self.model, self.assembled, int(at)) for at in rows),
            tuple(
                ColumnBound(
                    variable,
                    coordinate,
                    float(self.assembled.col_lower[column]),
                    float(self.assembled.col_upper[column]),
                )
                for _, column, variable, coordinate in resolve(self.model, columns)
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
            for _, column, variable, coordinate in resolve(self.model, moving)
        )
