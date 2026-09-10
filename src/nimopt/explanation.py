"""What a model or a definition declares, and what it built."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nimopt.param import Param
from nimopt.syntax import render

if TYPE_CHECKING:
    from nimopt.variable import Variable


@dataclass(frozen=True)
class SetShape:
    """A dimension, and how many members it carries."""

    name: str
    size: int | None


@dataclass(frozen=True)
class ParamShape:
    """A parameter, its dimensions, and how many coefficients it carries."""

    name: str
    dims: tuple[str, ...]
    entries: int | None


@dataclass(frozen=True)
class VariableShape:
    """A variable, its dimensions, its members, its columns and its bounds."""

    name: str
    dims: tuple[str, ...]
    members: str | None
    columns: int | None
    lower: str
    upper: str
    integer: bool


@dataclass(frozen=True)
class ConstraintShape:
    """A constraint, the dimensions it is free over, its relation, and what it built."""

    name: str
    free: tuple[str, ...]
    sense: str
    rows: int | None
    nonzeros: int | None
    relation: str


@dataclass(frozen=True)
class Explanation:
    """What is declared, and where it is built, what was built from it.

    A count is `None` where nothing is bound. It is never zero: a count of
    zero is a fact a caller acts on, and reporting one for a declaration
    would state something false.
    """

    name: str
    sense: str
    built: bool
    sets: tuple[SetShape, ...]
    parameters: tuple[ParamShape, ...]
    variables: tuple[VariableShape, ...]
    constraints: tuple[ConstraintShape, ...]
    objective: str | None
    columns: int | None
    rows: int | None
    nonzeros: int | None
    objective_constant: float = 0.0
    aliases: tuple[tuple[str, str], ...] = ()

    def __repr__(self) -> str:
        return "\n".join(self._lines())

    def _lines(self) -> list[str]:
        if self.built:
            head = (
                f"{self.name}  {self.sense}  {self.columns} columns · "
                f"{self.rows} rows · {self.nonzeros} nonzeros"
            )
        else:
            head = f"{self.name}  {self.sense}  not built"
        lines = [head]
        lines.append("  sets        " + " · ".join(_set(s) for s in self.sets))
        lines.append("  parameters  " + " · ".join(_param(s) for s in self.parameters))
        lines.append(
            "  variables   " + " · ".join(_variable(s) for s in self.variables)
        )
        if self.aliases:
            named = " · ".join(f"{name} of {base}" for name, base in self.aliases)
            lines.insert(2, f"  aliases     {named}")
        for constraint in self.constraints:
            lines.append("  constraint  " + _constraint(constraint))
        if self.objective is not None:
            lines.append(f"  objective   {self.sense}  {self.objective}")
        return lines


def _set(shape: SetShape) -> str:
    return shape.name if shape.size is None else f"{shape.name} {shape.size}"


def _param(shape: ParamShape) -> str:
    dims = ",".join(shape.dims)
    if shape.entries is None:
        return f"{shape.name} ({dims})"
    return f"{shape.name} ({dims}) {shape.entries}"


def _variable(shape: VariableShape) -> str:
    dims = "×".join(shape.dims)
    over = "" if shape.members is None else f" over {shape.members}"
    columns = "" if shape.columns is None else f" {shape.columns} cols"
    kind = " integer" if shape.integer else ""
    return f"{shape.name} ({dims}){over}{columns} [{shape.lower}, {shape.upper}]{kind}"


def _constraint(shape: ConstraintShape) -> str:
    free = ",".join(shape.free)
    built = "" if shape.rows is None else f"  {shape.rows} rows  {shape.nonzeros} nz"
    return f"{shape.name} ({free})  {shape.relation}{built}"


def set_shape(dimension: Any, size: int | None) -> SetShape:
    """The shape of a dimension, with `size` where its members are known."""
    return SetShape(dimension.name, size)


def param_shape(parameter: Param, entries: int | None) -> ParamShape:
    """The shape of a parameter, with `entries` where its values are known."""
    return ParamShape(parameter.name, parameter.dims, entries)


def variable_shape(
    name: str, variable: "Variable", columns: int | None
) -> VariableShape:
    """The shape of a variable, with `columns` where its numbering is known."""
    return VariableShape(
        name,
        variable.dims,
        _members(variable.subset),
        columns,
        _bound(variable.lower),
        _bound(variable.upper),
        variable.integer,
    )


def objective_constant(expression: Any) -> float:
    """The fixed cost the objective states, zero where it states none."""
    return 0.0 if expression is None else expression.constant


def objective_text(expression: Any) -> str | None:
    """The objective's spelling, or None for no objective."""
    return None if expression is None else render(expression)


def _members(subset: Any) -> str | None:
    if isinstance(subset, Param):
        return subset.name
    return None


def _bound(bound: Any) -> str:
    if isinstance(bound, Param):
        return bound.name
    return str(bound)
