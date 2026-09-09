"""A live backend holding one model, across a solve and the questions after it."""

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
    """One column of a conflict: what it stands for, and the bounds it carries."""

    variable: str
    coordinate: dict
    lower: float
    upper: float


@dataclass(frozen=True)
class RayTerm:
    """One column of a ray: what it stands for, and the direction it moves in."""

    variable: str
    coordinate: dict
    direction: float


@dataclass(frozen=True)
class Diagnosis:
    """Why a model did not solve, as the rows and columns that say so.

    `conflict` holds `Row` objects, so a conflicting row renders exactly as
    `Model.row` renders it. `method` states who computed the conflict:
    `"native"` means the solver did, `"computed"` that nimopt did, so a caller
    is never handed a minimal-looking conflict nothing proved minimal.

    `conflict` is `None` where the model is not infeasible and `ray` is `None`
    where it is not unbounded, so an empty tuple means the solver named
    nothing rather than that nothing was asked.
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
    """One solver's model, opened on one assembled model and held.

    A solve hands the matrix across and reads an answer back; a question asked
    afterwards is asked of the same solved instance rather than of a second
    one. The matrix is assembled when the session opens, so a session that
    solves twice assembles once.

    The solver's own model is the adapter's, and is handed back to the adapter
    that made it. Nothing above the seam reads it.
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
        """What this session's adapter does, as shipped."""
        return self._adapter.CAPABILITIES

    @property
    def status(self) -> str | None:
        """What the last solve reached, or None before one."""
        return self._status

    @property
    def backend_open(self) -> bool:
        """Whether this session still holds the solver's model."""
        return self._backend is not None

    def close(self) -> None:
        """Release the solver's model."""
        self._backend = None

    def solve(self) -> Solution:
        """Solve this session's matrix and read the answer back onto its sets."""
        status, objective, col_value, row_dual, backend = self._adapter.solve(
            self.assembled, self.model.sense, self.options
        )
        self._backend = backend
        self._status = status
        rows_of = {name: self.assembled.row_of(name) for name in self.model.constraints}
        return Solution(
            self.model,
            status,
            objective + self.model.objective_constant,
            col_value,
            row_dual,
            rows_of,
            self.solver,
        )

    def diagnose(self) -> Diagnosis:
        """Why this session's last solve did not reach an answer.

        A conflict and a ray are questions asked of the backend that solved,
        so nothing is solved a second time and the rows named are the rows it
        was given.
        """
        if self._status is None:
            raise ValueError(
                f"session on {self.model.name!r} has not solved; solve before "
                f"diagnosing, because a conflict is a question about a solved "
                f"model"
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
        """The rows that cannot hold together, and the columns they bind."""
        if not self.capabilities.supports("conflict"):
            raise ValueError(
                f"{self.solver!r} computes no conflict, so there is none to "
                f"read; `capabilities({self.solver!r})` states what it does"
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
        """The columns an unbounded model runs off along, and how far each moves."""
        if not self.capabilities.supports("ray"):
            raise ValueError(
                f"{self.solver!r} computes no ray, so there is none to read; "
                f"`capabilities({self.solver!r})` states what it does"
            )
        direction = self._adapter.ray(self._backend)
        if direction is None:
            return None
        moving = np.nonzero(direction)[0]
        return tuple(
            RayTerm(variable, coordinate, float(direction[column]))
            for _, column, variable, coordinate in resolve(self.model, moving)
        )
