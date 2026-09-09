"""What crosses the boundary between a model and a solver.

An adapter is a module, not a class: there is no base to inherit. The contract
is what crosses the seam. Each adapter states

    BACKEND       the import name of the library it drives
    CAPABILITIES  a `Capabilities`, describing the adapter as shipped

and answers

    solve(assembled, sense, options=None)
        -> (status, objective, col_value, row_dual, backend)
    conflict(backend) -> (rows, columns)
    ray(backend) -> ndarray | None

`status` is a member of `STATUS`. `row_dual` is `None` where the adapter
refuses duals for the model it was given. `backend` is the adapter's own
model, held by the session and handed back to the adapter that made it;
nothing above the seam reads it. An adapter declaring neither a conflict nor a
ray is never asked for one.

A descriptor describes the adapter, not the library behind it. A solver
feature the adapter does not call is `absent`, and claiming otherwise answers a
different model's optimum. Support is three-valued in principle -- a
reformulation is an answer rather than a missing implementation -- and is
two-valued here because no adapter reformulates anything.
"""

from dataclasses import dataclass

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
class Capabilities:
    """What one adapter does, and which of its capabilities refuse each other.

    A flat set is insufficient: a solver can hold two capabilities and refuse
    their combination. `refused` carries those pairs, so a caller reads what a
    solver will not do for a given model rather than discovering it in a
    vector that means nothing.
    """

    solver: str
    support: dict[str, str]
    refused: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        unknown = sorted(set(self.support) - set(CAPABILITIES))
        if unknown:
            raise ValueError(
                f"the capabilities are {CAPABILITIES}; {self.solver!r} names {unknown}"
            )
        missing = sorted(set(CAPABILITIES) - set(self.support))
        if missing:
            raise ValueError(
                f"{self.solver!r} says nothing about {missing}; a descriptor "
                f"states every capability, so `absent` is stated rather than "
                f"left out"
            )
        for capability, support in self.support.items():
            if support not in SUPPORT:
                raise ValueError(
                    f"support is one of {SUPPORT}; {self.solver!r} states "
                    f"{support!r} for {capability!r}"
                )
        for pair in self.refused:
            unknown = sorted(set(pair) - set(CAPABILITIES))
            if len(pair) != 2 or unknown:
                raise ValueError(
                    f"a refused pair names two capabilities of {CAPABILITIES}; "
                    f"{self.solver!r} states {pair}"
                )

    def __repr__(self) -> str:
        does = " · ".join(f"{c} {self.support[c]}" for c in CAPABILITIES)
        if not self.refused:
            return f"{self.solver}  {does}"
        refuses = " · ".join("+".join(sorted(p)) for p in self.refused)
        return f"{self.solver}  {does}  refuses {refuses}"

    def supports(self, capability: str) -> bool:
        """Whether this adapter answers for `capability` at all."""
        if capability not in self.support:
            raise ValueError(f"the capabilities are {CAPABILITIES}; got {capability!r}")
        return self.support[capability] != "absent"

    def refuses(self, one: str, other: str) -> bool:
        """Whether this adapter refuses two capabilities together."""
        return tuple(sorted((one, other))) in tuple(
            tuple(sorted(pair)) for pair in self.refused
        )
