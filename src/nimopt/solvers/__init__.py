"""Adapters handing an assembled model to a solver.

`available()` reports which adapters can run here and what each declares, so
an agent choosing a solver reads it rather than guessing. `capabilities(name)`
answers for an adapter whether or not its backend is installed: a descriptor
states what the adapter does as shipped, and reading one is how a caller
decides what to install.
"""

from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

from nimopt.solvers.base import Capabilities
from nimopt.solvers.options import OPTIONS
from nimopt.solvers.options import Option as Option

ADAPTERS = ("highs", "gurobi", "mosek")


def adapter(name: str) -> ModuleType:
    """The module adapting the named solver."""
    if name not in ADAPTERS:
        raise ValueError(f"the solvers are {ADAPTERS}; got {name!r}")
    return import_module(f"nimopt.solvers.{name}")


def capabilities(name: str) -> Capabilities:
    """What the named adapter does, as shipped.

    Readable whether or not the backend is installed: an adapter names its
    library only where it drives it.
    """
    return adapter(name).CAPABILITIES


def available() -> dict[str, Capabilities]:
    """Every adapter whose backend can be imported here, with what it declares."""
    return {
        name: capabilities(name)
        for name in ADAPTERS
        if find_spec(adapter(name).BACKEND) is not None
    }


def options(solver: str | None = None) -> tuple[Option, ...]:
    """Every option a caller can set, or those under one solver's own names.

    Named without a solver it is the vocabulary itself. Named with one, each
    option carries that solver's own spelling and its own values, which is
    what a caller reads to follow an option into the solver's documentation.
    An option the solver does not carry reads back with `native` of `None`,
    and a choice it lacks is absent from `native_choices`.
    """
    if solver is None:
        return OPTIONS
    held = adapter(solver)

    def own(option: Option) -> tuple[tuple[str, object], ...]:
        values = held.OPTION_VALUES.get(option.name)
        if values is None:
            return tuple((choice, choice) for choice in option.choices)
        return tuple(
            (choice, values[choice]) for choice in option.choices if choice in values
        )

    return tuple(
        replace(
            option,
            native=held.OPTION_NAMES[option.name],
            native_choices=own(option),
        )
        for option in OPTIONS
    )
