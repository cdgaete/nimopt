"""The DSL as text: rendering an object to text, and reading text back."""

import ast
import operator
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

import numpy as np

from nimopt.coefficient import Coefficient, Derived, DerivedRef
from nimopt.param import Param
from nimopt.sets import Alias, Set
from nimopt.term import Expression, ParamRef, Relation, Sum


def number(value: float) -> str:
    """Return a number as text: a whole value as an integer, otherwise repr."""
    value = float(value)
    if np.isfinite(value) and value == int(value):
        return str(int(value))
    return repr(value)


def label(value: Any) -> str:
    """Return a member's label as text: a string quoted, a number bare."""
    if isinstance(value, np.generic):
        value = value.item()
    return repr(value)


def _items(
    dims: Sequence[str],
    shifts: Mapping[str, tuple[int, str]],
    fixed: Mapping[str, Any],
) -> str:
    """Return the bracket of a reference: each dimension, lagged or fixed."""
    out = []
    for dim in dims:
        if dim in shifts:
            shift, mode = shifts[dim]
            base = f"{dim}.cyclic" if mode == "wrap" else dim
            out.append(f"{base} - {shift}" if shift > 0 else f"{base} + {-shift}")
        elif dim in fixed:
            out.append(label(fixed[dim]))
        else:
            out.append(dim)
    return ", ".join(out)


def _domain(held: Any) -> str:
    """Return a condition as its name: a parameter's, or a tuple of set names."""
    if isinstance(held, Param):
        return held.name
    if isinstance(held, tuple):
        names = ", ".join(s.name for s in held)
        return f"({names},)" if len(held) == 1 else f"({names})"
    raise ValueError(
        "a condition with no name has no text form; declare its members as a "
        "parameter and use that name"
    )


def _operand(held: Any) -> str:
    return _coefficient(held) if isinstance(held, Coefficient) else number(held)


def _coefficient(held: Any) -> str:
    """Return a coefficient as the text that rebuilds it, parentheses included."""
    if isinstance(held, ParamRef):
        items = _items(held.param.dims, {}, held.fixed)
        return f"{held.param.name}[{items}]" if items else held.param.name
    if isinstance(held, DerivedRef):
        text = _coefficient(held.derived)
        if held.fixed:
            text = f"{text}[{_items(held.derived.dims, {}, held.fixed)}]"
        return text
    if isinstance(held, Derived):
        left = _operand(held.left)
        if held.right is None:
            return f"(-{left})"
        return f"({left} {held.symbol} {_operand(held.right)})"
    raise TypeError(
        f"{type(held).__name__} is not a coefficient; render a ParamRef, a "
        f"Derived or a DerivedRef"
    )


def _term(term: Any) -> tuple[str, bool]:
    """Return a term's body, and whether a minus must parenthesize it."""
    items = _items(term.variable.dims, term.shifts, term.fixed)
    body = f"{term.variable.name}[{items}]" if items else term.variable.name
    product = term.coefficient is not None
    if product:
        body = f"{_coefficient(term.coefficient)} * {body}"
    if term.summed:
        where = "" if term.where is None else f", where={_domain(term.where)}"
        body = f"Sum({', '.join(term.summed)}, {body}{where})"
        product = False
    scale = abs(term.scale)
    if scale != 1.0:
        body = f"{number(scale)} * ({body})" if product else f"{number(scale)} * {body}"
        product = False
    return body, product


def _expression(expression: Any) -> str:
    parts = []
    for k, term in enumerate(expression.terms):
        body, bare = _term(term)
        negative = term.scale < 0
        if k == 0:
            head = f"({body})" if negative and bare else body
            parts.append(f"-{head}" if negative else head)
        else:
            parts.append(f" - {body}" if negative else f" + {body}")
    constant = expression.constant
    if not parts:
        return number(constant)
    if constant < 0:
        parts.append(f" - {number(-constant)}")
    elif constant > 0:
        parts.append(f" + {number(constant)}")
    return "".join(parts)


def _rhs(rhs: Any) -> str:
    if isinstance(rhs, Coefficient):
        return _coefficient(rhs)
    return number(rhs)


def render(held: Any) -> str:
    """Return `held` as the text that reads back to it.

    `held` is a coefficient, an expression or a relation. The text is built
    from the object, not from what was written: each term has its own sum,
    sign and scale, a combination has its parentheses, and the constant comes
    last.
    """
    if isinstance(held, Relation):
        return f"{_expression(held.expression)} {held.sense} {_rhs(held.rhs)}"
    if isinstance(held, Expression):
        return _expression(held)
    if isinstance(held, Coefficient):
        return _coefficient(held)
    raise TypeError(
        f"render takes a coefficient, an expression or a relation; got "
        f"{type(held).__name__}"
    )


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_COMPARE = {
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.NotEq: operator.ne,
}

_CHAINED = (
    "a chained comparison keeps only its second bound; write each bound as "
    "its own relation"
)


class _Reader(ast.NodeVisitor):
    """The walk over the nodes of a parsed expression, applied through the DSL."""

    def __init__(self, text: str, symbols: Mapping[str, Any]) -> None:
        self.text = text
        self.symbols = symbols

    def reject(self, what: str) -> NoReturn:
        raise ValueError(f"{self.text!r}: {what}")

    def generic_visit(self, node: ast.AST) -> NoReturn:
        self.reject(
            f"{type(node).__name__} is not part of the syntax; the syntax is "
            f"names, brackets, labels, Sum, + - * / **, a unary minus, .cyclic "
            f"and one comparison"
        )

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id == "Sum":
            return Sum
        if node.id not in self.symbols:
            self.reject(f"{node.id!r} names no declared set, parameter or variable")
        return self.symbols[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            self.reject(
                f"{value!r} is a {type(value).__name__}; a constant is a number "
                f"or a label"
            )
        return value

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(element) for element in node.elts)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        return self.visit(node.value)[self.visit(node.slice)]

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        held = self.visit(node.value)
        if node.attr != "cyclic" or not isinstance(held, (Set, Alias)):
            self.reject(
                f".{node.attr} is not read; .cyclic on a set is the one attribute"
            )
        return held.cyclic

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        apply = _BINARY.get(type(node.op))
        if apply is None:
            self.reject(f"{type(node.op).__name__} is not an operator of the syntax")
        return apply(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, ast.USub):
            self.reject(f"{type(node.op).__name__} is not an operator of the syntax")
        return -self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> Any:
        if not (isinstance(node.func, ast.Name) and node.func.id == "Sum"):
            self.reject("Sum is the one call the syntax supports")
        args = [self.visit(argument) for argument in node.args]
        held = {}
        for keyword in node.keywords:
            if keyword.arg != "where":
                self.reject(f"Sum takes where= and no {keyword.arg}=")
            held["where"] = self.visit(keyword.value)
        return Sum(*args, **held)

    def visit_Compare(self, node: ast.Compare) -> Any:
        if len(node.ops) != 1:
            self.reject(_CHAINED)
        apply = _COMPARE.get(type(node.ops[0]))
        if apply is None:
            self.reject(
                f"{type(node.ops[0]).__name__} is not an equation operator; an "
                f"equation is <=, >= or =="
            )
        return apply(self.visit(node.left), self.visit(node.comparators[0]))


def read(text: str, symbols: Mapping[str, Any]) -> Any:
    """Return the object `text` denotes, evaluated over `symbols` through the DSL.

    `symbols` maps each declared set, parameter and variable to its object.
    The text is parsed by Python and each node is applied as the operator it
    is. The DSL accepts and rejects the same forms here as in a script, with
    the same messages.
    """
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"{text!r} is not an expression: {error.msg}") from None
    return _Reader(text, symbols).visit(tree.body)
