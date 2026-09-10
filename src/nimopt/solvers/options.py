"""The options a caller sets, under one name per option.

`OPTIONS` defines every option. Each adapter translates a name into the one
its solver uses. A model therefore sets a time limit under one name for every
solver. A name outside `OPTIONS` raises.

An adapter supports a subset of the options and a subset of each option's
choices. An option or a choice its solver lacks raises, and the message
identifies it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Option:
    """One option a caller sets, the type it takes, and what it does.

    `native` and `native_choices` are empty until `options(solver)` reads the
    option for one solver. That call fills them with the solver's own names.
    """

    name: str
    kind: str
    does: str
    choices: tuple[str, ...] = ()
    native: str = ""
    native_choices: tuple[tuple[str, object], ...] = ()


OPTIONS = (
    Option("time_limit", "float", "seconds the solver may run for"),
    Option("iteration_limit", "int", "simplex iterations the solver may take"),
    Option("node_limit", "int", "branch-and-bound nodes the solver may explore"),
    Option("mip_gap", "float", "relative gap at which a mixed-integer solve stops"),
    Option("mip_abs_gap", "float", "absolute gap at which a mixed-integer solve stops"),
    Option("feasibility_tol", "float", "how far a primal solution may miss a row"),
    Option("optimality_tol", "float", "how far a dual solution may miss a bound"),
    Option("threads", "int", "threads the solver may use; 0 leaves it the choice"),
    Option("seed", "int", "the seed the solver randomizes from"),
    Option("log", "bool", "whether the solver writes its own iteration log"),
    Option(
        "presolve", "choice", "how hard the solver presolves", ("off", "choose", "on")
    ),
    Option(
        "method",
        "choice",
        "the algorithm the solver runs",
        ("choose", "simplex", "barrier", "hipo", "pdlp"),
    ),
    Option(
        "newton_system",
        "choice",
        "the Newton system an interior point method factorizes",
        ("choose", "augmented", "normaleq"),
    ),
    Option(
        "crossover",
        "choice",
        "whether an interior point is moved to a vertex after the solve",
        ("choose", "off", "on"),
    ),
    Option(
        "pdlp_tol", "float", "relative tolerance at which the first-order method stops"
    ),
)

BY_NAME = {option.name: option for option in OPTIONS}


def checked(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return `options` as a plain dict, validated against `OPTIONS`.

    Raises ValueError for a name outside `OPTIONS` and for a value outside a
    choice's set. Raises TypeError for a value of the wrong type. Each message
    reports what the option accepts.
    """
    if not options:
        return {}
    unknown = sorted(set(options) - set(BY_NAME))
    if unknown:
        raise ValueError(f"the options are {tuple(BY_NAME)}; got {unknown}")
    for name, value in options.items():
        option = BY_NAME[name]
        if option.kind == "bool":
            if not isinstance(value, bool):
                raise TypeError(
                    f"option {name!r} takes a bool; got {type(value).__name__}"
                )
        elif option.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"option {name!r} takes an int; got {type(value).__name__}"
                )
        elif option.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"option {name!r} takes a number; got {type(value).__name__}"
                )
        elif value not in option.choices:
            raise ValueError(
                f"option {name!r} takes one of {option.choices}; got {value!r}"
            )
    return dict(options)


def translated(
    options: Mapping[str, Any] | None,
    names: Mapping[str, str | None],
    values: Mapping[str, Mapping[str, Any]],
    solver: str = "the solver",
) -> dict[str, Any]:
    """Return `options` under a solver's own names and its own values.

    `names` maps every option to the solver's name for it, or to None where
    the solver has no such option. `values` maps the choices of an option whose
    values the solver writes differently. A choice absent from that map is one
    the solver lacks. Either raises ValueError.
    """
    held = {}
    for name, value in checked(options).items():
        native = names[name]
        if native is None:
            raise ValueError(
                f"{solver} has no option {name!r}; remove it or solve with "
                f"another solver"
            )
        own = values.get(name)
        if own is not None and value not in own:
            raise ValueError(
                f"{solver} has no {value!r} for option {name!r}; pass one of "
                f"{tuple(own)}"
            )
        held[native] = value if own is None else own[value]
    return held
