"""The options a caller sets, in nimopt's own words.

An option is named once here and translated by the adapter driving each
solver, so a model states a time limit one way whatever solves it. A name the
vocabulary does not carry is refused rather than passed on, because a solver
that ignores what it was asked answers a question the caller did not write.

Not every solver carries every option, and a solver carries some choices and
not others: an adapter names what it lacks, and asking a solver for an option
or a choice it lacks is refused by name rather than answered by a different
algorithm.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Option:
    """One option a caller sets, what it takes, and what it means.

    `native` and `native_choices` are empty until the option is read for a
    named solver, which is what fills them with that solver's own spelling.
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
    Option("seed", "int", "the seed the solver randomises from"),
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
        "the Newton system an interior point method factorises",
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
    """`options` as a plain mapping, refusing anything the vocabulary lacks.

    A name outside the vocabulary, a value of the wrong kind, and a value
    outside a choice's set are each refused naming what is accepted, so a
    caller reads the correction rather than a solve that ignored the option.
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
    """`options` under a solver's own names, and its own values where they differ.

    `names` maps every name in the vocabulary to the solver's, or to `None`
    where the solver carries no such option; `values` maps the choices of an
    option whose values the solver states differently, and a choice missing
    from that map is one the solver lacks. Either is refused by name.
    """
    held = {}
    for name, value in checked(options).items():
        native = names[name]
        if native is None:
            raise ValueError(f"{solver} carries no option {name!r}")
        own = values.get(name)
        if own is not None and value not in own:
            raise ValueError(
                f"{solver} has no {value!r} for option {name!r}; it takes {tuple(own)}"
            )
        held[native] = value if own is None else own[value]
    return held
