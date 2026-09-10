"""A definition: what a model is declared from, before its data exists."""

import copy
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from nimopt.coefficient import Coefficient, Derived, DerivedRef

if TYPE_CHECKING:
    from nimopt.model import Model

from nimopt.explanation import (
    ConstraintShape,
    Explanation,
    objective_constant,
    objective_text,
    param_shape,
    set_shape,
    variable_shape,
)
from nimopt.model import Model
from nimopt.param import Param
from nimopt.progress import reporter
from nimopt.sets import Alias, Set, check_members
from nimopt.symbol import read_at_its_sets
from nimopt.syntax import render
from nimopt.term import Expression, ParamRef, Relation
from nimopt.variable import Variable


class Definition:
    """The symbols and constraints a model is declared from, without its data.

    A definition declares its sets and parameters by name. It declares its
    constraints in the expression syntax a model uses. An expression contains
    handles, not arrays. A constraint's free dimensions and its sense are read
    from the relation.
    """

    def __init__(self, name: str = "definition", sense: str = "min") -> None:
        if sense not in ("min", "max"):
            raise ValueError(f"sense is 'min' or 'max'; got {sense!r}")
        self.name = str(name)
        self._sense = sense
        self.sets = {}
        self.aliases = {}
        self.parameters = {}
        self.variables = {}
        self.constraints = {}
        self.objective = None

    @property
    def sense(self) -> str:
        """Return the direction this definition's objective is optimized in."""
        return self._sense

    def __repr__(self) -> str:
        return (
            f"Definition({self.name!r}, {len(self.variables)} variables, "
            f"{len(self.constraints)} constraints)"
        )

    def _fresh(self, name: str, registry: Mapping[str, Any], what: str) -> str:
        name = str(name)
        if name in registry:
            raise ValueError(f"{what} {name!r} is already declared")
        if registry is self.constraints:
            return name
        if not name.isidentifier() or name == "Sum":
            raise ValueError(
                f"{what} {name!r} is not a name an expression can address; "
                f"declare a Python identifier other than Sum"
            )
        held = (
            ("set", self.sets),
            ("alias", self.aliases),
            ("parameter", self.parameters),
            ("variable", self.variables),
        )
        for kind, other in held:
            if other is not registry and name in other:
                raise ValueError(
                    f"{what} {name!r} is already declared as a {kind}; "
                    f"declare another name"
                )
        return name

    def set(self, name: str) -> Set:
        """Declare a dimension, whose members arrive with the data."""
        name = self._fresh(name, self.sets, "set")
        self.sets[name] = Set(name)
        return self.sets[name]

    def alias(self, name: str, base: Any) -> Alias:
        """Declare a second name for `base`, over the members it binds.

        An alias has no data of its own. It reads the labels and the
        coordinate of its base set, and binds when that set binds. A model
        relating a set to itself is declared over the set and its alias.
        """
        if base not in self.sets.values():
            raise ValueError(
                f"base {getattr(base, 'name', base)!r} of alias {name!r} is "
                f"not a set of definition {self.name!r}; pass a set of it"
            )
        name = self._fresh(name, self.aliases, "alias")
        self.aliases[name] = Alias(name, base)
        return self.aliases[name]

    def param(self, name: str, sets: Any) -> Param:
        """Declare a parameter over `sets`, whose values arrive with the data."""
        name = self._fresh(name, self.parameters, "parameter")
        self.parameters[name] = Param(name, sets)
        return self.parameters[name]

    def var(
        self,
        name: str,
        sets: Any,
        subset: Any = None,
        lower: float | Param = 0.0,
        upper: float | Param = np.inf,
        integer: bool = False,
    ) -> Variable:
        """Declare a variable over `sets`, or over `subset` of them."""
        name = self._fresh(name, self.variables, "variable")
        self.variables[name] = Variable(
            name, sets, subset=subset, lower=lower, upper=upper, integer=integer
        )
        return self.variables[name]

    def constraint(
        self, name: str, relation: Any, where: Any = None, over: Any = None
    ) -> None:
        """Declare an equation from a comparison of an expression.

        `where=` narrows the rows and `over=` declares them. Each is a tuple
        of this definition's sets, or one of its parameters whose coefficients
        are the coordinates. Both resolve when the declaration binds.
        """
        if not isinstance(relation, Relation):
            raise TypeError(
                f"constraint {str(name)!r} takes a comparison of an "
                f"expression, such as `expr <= rhs`; got "
                f"{type(relation).__name__}"
            )
        name = self._fresh(name, self.constraints, "constraint")
        self.constraints[name] = (relation, where, over)

    def explain(self) -> Explanation:
        """Return what this definition declares, with nothing bound."""
        return Explanation(
            name=self.name,
            sense=self.sense,
            built=False,
            sets=tuple(set_shape(s, None) for s in self.sets.values()),
            aliases=tuple((a.name, a.base.name) for a in self.aliases.values()),
            parameters=tuple(param_shape(p, None) for p in self.parameters.values()),
            variables=tuple(
                variable_shape(name, v, None) for name, v in self.variables.items()
            ),
            constraints=tuple(
                ConstraintShape(
                    name,
                    relation.expression.frame,
                    relation.sense,
                    None,
                    None,
                    render(relation),
                )
                for name, (relation, _, _) in self.constraints.items()
            ),
            objective=objective_text(self.objective),
            objective_constant=objective_constant(self.objective),
            columns=None,
            rows=None,
            nonzeros=None,
        )

    def to_yaml(self, instructions: bool = False) -> str:
        """Return this definition as file text, its structure and no data.

        `instructions=True` prefixes the comment block that explains the
        format.
        """
        from nimopt.files import dumps, structure

        return dumps(structure(self), instructions)

    def build(self, data: Mapping[str, Any], progress: Any = False) -> "Model":
        """Return a model over this definition's declarations, bound to `data`.

        `data` maps a declared set's name to its members and a declared
        parameter's name to its values. A set and a parameter never share a
        name, and the two registries merge into one key set. The declarations
        are copied before they are bound, and the definition is unchanged.

        `progress=` reports each constraint as it is measured. The count is
        the number of constraints.
        """
        declared = {**self.sets, **self.parameters}
        missing = [name for name in declared if name not in data]
        if missing:
            raise ValueError(f"data does not cover {missing}; add an entry for each")
        undeclared = [name for name in data if name not in declared]
        if undeclared:
            raise ValueError(
                f"data contains the undeclared {undeclared}; declare them or "
                f"remove them"
            )

        bound = copy.deepcopy(self)
        for name, dimension in bound.sets.items():
            dimension._bind(np.asarray(data[name]))
        for name, parameter in bound.parameters.items():
            parameter._bind(_values(name, parameter.sets, data[name]).array)
        for owner, sets, fixed in _fixed_members(bound):
            check_members(sets, fixed, owner)

        model = Model(bound.name, sense=bound.sense)
        for variable in bound.variables.values():
            model._adopt(variable)
        held = reporter(progress)
        if held is not None:
            held.start(len(bound.constraints), f"building {bound.name}")
        for at, (name, (relation, where, over)) in enumerate(
            bound.constraints.items(), start=1
        ):
            model.constraint(name, relation, where, over)
            if held is not None:
                held.step(at, name)
        if held is not None:
            held.done()
        if bound.objective is not None:
            model.set_objective(bound.objective)
        return model

    def set_objective(self, expression: Any) -> None:
        """Set the objective the definition's sense optimizes."""
        expression = read_at_its_sets(expression)
        if not isinstance(expression, Expression):
            raise TypeError(
                f"an objective is an expression over the model's columns; got "
                f"{type(expression).__name__}"
            )
        if expression.frame:
            raise ValueError(
                f"objective expression has free dimensions {expression.frame}; "
                f"sum the expression over them"
            )
        self.objective = expression


def _values(name: str, sets: Any, given: Any) -> Param:
    """Return a parameter over `given`, dense over the product or long.

    A pair is one mapping of label columns and one value column. It defines a
    coefficient at some coordinates of the product and none at the rest. Any
    other value is read as values over every cell.
    """
    if isinstance(given, tuple):
        if len(given) != 2 or not hasattr(given[0], "keys"):
            raise ValueError(
                f"parameter {name!r} is given a tuple of {len(given)} item(s); "
                f"pass one mapping of label columns and one value column, or "
                f"pass an array over every cell of the product"
            )
        columns, values = given
        return Param.from_long(name, sets, columns, values)
    return Param.from_dense(name, sets, np.asarray(given, dtype=np.float64))


def _readings(coefficient: Any) -> Iterator[tuple[str, Any, Mapping[str, Any]]]:
    """Yield every reading of a parameter or a combination in a coefficient."""
    if isinstance(coefficient, ParamRef):
        yield (
            f"parameter {coefficient.param.name!r}",
            coefficient.param.sets,
            coefficient.fixed,
        )
    elif isinstance(coefficient, DerivedRef):
        yield (
            f"coefficient {coefficient.name}",
            coefficient.derived.sets,
            coefficient.fixed,
        )
        yield from _readings(coefficient.derived)
    elif isinstance(coefficient, Derived):
        for operand in (coefficient.left, coefficient.right):
            if isinstance(operand, Coefficient):
                yield from _readings(operand)


def _fixed_members(
    definition: "Definition",
) -> Iterator[tuple[str, Any, Mapping[str, Any]]]:
    """Yield each fixed member a definition reads, with the sets it is over.

    A reading with nothing fixed is skipped.
    """
    expressions = [
        relation.expression for relation, _, _ in definition.constraints.values()
    ]
    rhs = [relation.rhs for relation, _, _ in definition.constraints.values()]
    if definition.objective is not None:
        expressions.append(definition.objective)
    for expression in expressions:
        for term in expression.terms:
            if term.fixed:
                yield (
                    f"variable {term.variable.name!r}",
                    term.variable.sets,
                    term.fixed,
                )
            if term.coefficient is not None:
                yield from _fixed(_readings(term.coefficient))
    for held in rhs:
        if isinstance(held, Coefficient):
            yield from _fixed(_readings(held))


def _fixed(
    readings: Iterable[tuple[str, Any, Mapping[str, Any]]],
) -> Iterator[tuple[str, Any, Mapping[str, Any]]]:
    for owner, sets, fixed in readings:
        if fixed:
            yield owner, sets, fixed
