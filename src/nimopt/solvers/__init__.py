"""Adapters passing an assembled model to a solver.

`available()` reports which adapters run in this environment and what each
declares. `capabilities(name)` reports for one adapter whether or not its
backend is installed. A descriptor describes the adapter as shipped.
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
    """Return the module adapting the named solver."""
    if name not in ADAPTERS:
        raise ValueError(f"the solvers are {ADAPTERS}; got {name!r}")
    return import_module(f"nimopt.solvers.{name}")


def capabilities(name: str) -> Capabilities:
    """Return what the named adapter does, as shipped.

    The descriptor is readable whether or not the backend is installed.
    """
    return adapter(name).CAPABILITIES


def available() -> dict[str, Capabilities]:
    """Return every adapter whose backend can be imported, with its descriptor."""
    return {
        name: capabilities(name)
        for name in ADAPTERS
        if find_spec(adapter(name).BACKEND) is not None
    }


def options(solver: str | None = None) -> tuple[Option, ...]:
    """Return every option a caller can set, or those under one solver's names.

    Called without a solver it returns `OPTIONS`. Called with one, each option
    reports that solver's own name in `native` and its own values in
    `native_choices`. An option the solver lacks has `native` of None, and a
    choice it lacks is absent from `native_choices`.
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
