"""The interface between a model and a solver adapter.

An adapter is a module. There is no base class to inherit. Each adapter
defines

    BACKEND       the import name of the solver library it calls
    CAPABILITIES  a `Capabilities` describing the adapter as shipped

and three functions

    solve(assembled, sense, options=None) -> Result
    conflict(backend) -> (rows, columns)
    ray(backend) -> ndarray | None

An adapter that declares neither a conflict nor a ray is never called for one.

`Result.status` is a member of `STATUS`. `Result.feasible` is True where the
solver reports `col_value` as a primal-feasible point. Each adapter reads that
flag from its solver. `Result.objective` is the objective value of `col_value`
and is defined where `feasible` is True. `Result.bound` is the bound on the
optimal objective the solver proved. It is a lower bound under sense `min`
and an upper bound under sense `max`. `Result.bound` is None where the solver
reports none. `Result.row_dual` is None for a model whose duals the adapter
does not report. `Result.backend` is the adapter's own solver model. The
session stores it and passes it to the adapter that built it.

A descriptor describes the adapter, not the library behind it. A solver
feature the adapter does not call is `absent`. Support has two values. No
adapter reformulates a model.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

STATUS = (
    "optimal",
    "infeasible",
    "unbounded",
    "unbounded_or_infeasible",
    "empty",
    "time_limit",
    "iteration_limit",
    "node_limit",
    "solution_limit",
    "work_limit",
    "objective_bound",
    "objective_target",
    "memory_limit",
    "interrupt",
)

CAPABILITIES = ("integrality", "duals", "conflict", "ray")

SUPPORT = ("native", "absent")


@dataclass(frozen=True)
class Result:
    """The values one adapter returns from a solve.

    `status` is a member of `STATUS`. `feasible` is True where the solver
    reports `col_value` as a primal-feasible point. `objective` is the
    objective value of `col_value`, and is finite where `feasible` is True.
    `bound` is the bound on the optimal objective the solver proved, or None
    where the solver reports none. `row_dual` is None for a model whose duals
    the adapter does not report. `backend` is the solver's own model.
    """

    status: str
    feasible: bool
    objective: float
    bound: float | None
    col_value: npt.NDArray[np.float64]
    row_dual: npt.NDArray[np.float64] | None
    backend: Any

    def __post_init__(self) -> None:
        if self.status not in STATUS:
            raise ValueError(f"status is {self.status!r}; pass one of {STATUS}")
        if self.status == "optimal" and not self.feasible:
            raise ValueError(
                "status is 'optimal' and feasible is False; report a feasible "
                "point at status 'optimal'"
            )
        if self.status == "infeasible" and self.feasible:
            raise ValueError(
                "status is 'infeasible' and feasible is True; report no "
                "feasible point at status 'infeasible'"
            )
        if self.bound is not None and not np.isfinite(self.bound):
            raise ValueError(f"bound is {self.bound}; pass a finite bound or pass None")
        if self.feasible and not np.isfinite(self.objective):
            raise ValueError(
                f"feasible is True and objective is {self.objective}; pass the "
                f"objective value of col_value"
            )


@dataclass(frozen=True)
class Capabilities:
    """What one adapter does, and which pairs of capabilities it rejects.

    An adapter can support two capabilities and reject their combination.
    `rejected` contains those pairs.
    """

    solver: str
    support: dict[str, str]
    rejected: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        unknown = sorted(set(self.support) - set(CAPABILITIES))
        if unknown:
            raise ValueError(
                f"{self.solver!r} declares the unknown capabilities {unknown}; "
                f"declare capabilities of {CAPABILITIES}"
            )
        missing = sorted(set(CAPABILITIES) - set(self.support))
        if missing:
            raise ValueError(
                f"{self.solver!r} declares no support for {missing}; declare "
                f"'native' or 'absent' for every capability"
            )
        for capability, support in self.support.items():
            if support not in SUPPORT:
                raise ValueError(
                    f"{self.solver!r} declares support {support!r} for "
                    f"{capability!r}; declare one of {SUPPORT}"
                )
        for pair in self.rejected:
            unknown = sorted(set(pair) - set(CAPABILITIES))
            if len(pair) != 2 or unknown:
                raise ValueError(
                    f"{self.solver!r} declares the rejected pair {pair}; pass "
                    f"two capabilities of {CAPABILITIES}"
                )

    def __repr__(self) -> str:
        does = " · ".join(f"{c} {self.support[c]}" for c in CAPABILITIES)
        if not self.rejected:
            return f"{self.solver}  {does}"
        rejects = " · ".join("+".join(sorted(p)) for p in self.rejected)
        return f"{self.solver}  {does}  rejects {rejects}"

    def supports(self, capability: str) -> bool:
        """Return whether the adapter supports `capability`.

        Raises ValueError for a capability outside `CAPABILITIES`.
        """
        if capability not in self.support:
            raise ValueError(
                f"capability is {capability!r}; pass one of {CAPABILITIES}"
            )
        return self.support[capability] != "absent"

    def rejects(self, one: str, other: str) -> bool:
        """Return whether the adapter rejects two capabilities together."""
        return tuple(sorted((one, other))) in tuple(
            tuple(sorted(pair)) for pair in self.rejected
        )
