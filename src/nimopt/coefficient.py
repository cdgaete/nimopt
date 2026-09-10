"""A coefficient: a parameter read at its sets, or a combination of them."""

from typing import Any

import numpy as np
from nimblend import combined_dims

from nimopt.sets import check_members, reference
from nimopt.symbol import Symbol, read_bare

NOT_A_COEFFICIENT = (
    "a coefficient is a parameter; build one with `Param.from_dense` or "
    "`Param.from_long` and read it at its sets. A product of two expressions "
    "is not linear."
)

_BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "**": lambda a, b: a**b,
}


def _rendered(operand: Any) -> str:
    """Return the text of an operand inside a combination's name."""
    return operand.name if isinstance(operand, Coefficient) else str(operand)


def _finite_divisor(divisor: Any, named: str) -> None:
    """Raise ZeroDivisionError for a divisor of zero, including a numpy zero."""
    if float(divisor) == 0.0:
        raise ZeroDivisionError(
            f"{named} is divided by zero; divide by a non-zero number"
        )


def _finite_quotient(array: Any, name: str) -> None:
    """Raise ZeroDivisionError for a divisor array that contains a zero."""
    at = np.flatnonzero(array.values() == 0.0)
    if at.size:
        where = {
            d: labels[at[0]].item() for d, labels in array.domain().labels().items()
        }
        raise ZeroDivisionError(
            f"divisor {name} is zero at {at.size} coordinate(s), first at "
            f"{where}; remove the zeros or divide by another parameter"
        )


class Coefficient:
    """The coefficient of a term: a parameter reference or a combination.

    A coefficient has a `name`, the `dims` it is over, an array from
    `materialise()`, and a reading at its sets. Multiplying a coefficient by
    an expression multiplies that expression's terms. Combining it with
    another coefficient or a number returns a `Derived`, computed where the
    matrix is built. Every other operand returns `NotImplemented`.
    """

    __array_ufunc__ = None
    __hash__ = None

    @property
    def name(self) -> str:
        """Return the name this coefficient is reported under."""
        raise NotImplementedError

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the dimensions this coefficient is over."""
        raise NotImplementedError

    def materialise(self) -> Any:
        """Return the array this coefficient computes to."""
        raise NotImplementedError

    def held(self) -> Any:
        """Return the array this coefficient already stores, or None.

        A bound parameter stores its array. A combination stores none, and
        computing it is the build.
        """
        return None

    def parameters(self) -> tuple[Any, ...]:
        """Return the parameters this coefficient reads, in order of appearance."""
        raise NotImplementedError

    def __getitem__(self, sets: Any) -> Any:
        """Raise TypeError: a coefficient is read at its sets once."""
        raise TypeError(
            f"coefficient {self.name} is already read at {self.dims}; read a "
            f"coefficient at its sets once"
        )

    def _read(self, sets: Any, holder: Any) -> dict[str, Any]:
        """Return the members `sets` fixes, checked against the dimensions."""
        given, shifts, fixed = reference(sets, self.dims)
        if shifts:
            raise ValueError(
                f"coefficient {self.name} is read at a lag {sorted(shifts)}; "
                f"write the lag at the variable's reference"
            )
        if given != self.dims:
            raise ValueError(
                f"coefficient {self.name} is over {self.dims}; got {given}"
            )
        check_members(holder.sets, fixed, f"coefficient {self.name}")
        return fixed

    def _combine(self, other: Any, symbol: str, flip: bool = False) -> Any:
        if isinstance(other, np.ndarray):
            raise TypeError(NOT_A_COEFFICIENT)
        if not isinstance(other, (Coefficient, int, float, np.number)):
            return NotImplemented
        return Derived(other, self, symbol) if flip else Derived(self, other, symbol)

    def _applied(self, other: Any) -> Any:
        from nimopt.term import Expression

        other = read_bare(other)
        if isinstance(other, Expression):
            return Expression([t.with_coefficient(self) for t in other.terms])
        return None

    def __mul__(self, other: Any) -> Any:
        applied = self._applied(other)
        return self._combine(other, "*") if applied is None else applied

    def __rmul__(self, other: Any) -> Any:
        applied = self._applied(other)
        if applied is not None:
            return applied
        if isinstance(other, (int, float, np.number)):
            return Derived(self, other, "*")
        return self._combine(other, "*", flip=True)

    def __add__(self, other: Any) -> Any:
        return self._combine(other, "+")

    def __radd__(self, other: Any) -> Any:
        return self._combine(other, "+", flip=True)

    def __sub__(self, other: Any) -> Any:
        return self._combine(other, "-")

    def __rsub__(self, other: Any) -> Any:
        return self._combine(other, "-", flip=True)

    def __truediv__(self, other: Any) -> Any:
        other = read_bare(other)
        if isinstance(other, (int, float, np.number)):
            _finite_divisor(other, f"coefficient {self.name}")
        elif isinstance(other, Coefficient):
            in_hand = other.held()
            if in_hand is not None:
                _finite_quotient(in_hand, other.name)
        return self._combine(other, "/")

    def __rtruediv__(self, other: Any) -> Any:
        return self._combine(other, "/", flip=True)

    def __neg__(self) -> Any:
        return Derived(self, None, "-")

    def __rpow__(self, other: Any) -> Any:
        raise TypeError(
            f"a power takes a number and coefficient {self.name} varies by "
            f"coordinate; compute {_rendered(other)} raised to it as data and "
            f"declare a parameter over the result"
        )

    def __pow__(self, other: Any) -> Any:
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(
                f"a power takes a number and {_rendered(other)} is over "
                f"dimensions; compute the exponent as data and declare a "
                f"parameter over the result"
            )
        return Derived(self, other, "**")

    def __abs__(self) -> Any:
        raise TypeError(
            f"coefficient {self.name} has no absolute value here; compute the "
            f"magnitude as data and declare a parameter over the result"
        )

    def _no_row(self, other: Any) -> Any:
        """Raise TypeError for a comparison of two coefficients.

        Returns `NotImplemented` for an expression or a symbol. Python then
        applies the reflected operator and the sense reverses:
        `capacity[G, T] >= gen[G, T]` is the row `gen[G, T] <= capacity[G, T]`.
        """
        from nimopt.term import Expression

        if isinstance(other, (Expression, Symbol)):
            return NotImplemented
        raise TypeError(
            f"coefficient {self.name} compared with {_rendered(other)} is not "
            f"a row; put a variable on one side of the equation"
        )

    __le__ = _no_row
    __ge__ = _no_row
    __lt__ = _no_row
    __gt__ = _no_row
    __eq__ = _no_row


class Derived(Coefficient):
    """Two coefficients and an operator, or one coefficient and a number.

    The combination stores handles. It derives its dimensions from its
    operands and computes once, where the term it multiplies materialises. A
    combination is written in a definition before any data exists.
    """

    def __init__(self, left: Any, right: Any, symbol: str) -> None:
        self.left = left
        self.right = right
        self.symbol = symbol
        operand_dims = [
            operand.dims
            for operand in (left, right)
            if isinstance(operand, Coefficient)
        ]
        self._dims = (
            combined_dims(*operand_dims) if len(operand_dims) == 2 else operand_dims[0]
        )

    @property
    def name(self) -> str:
        """Return the arithmetic of this combination, as it was written."""
        if self.right is None:
            return f"({self.symbol}{self.left.name})"
        return f"({_rendered(self.left)} {self.symbol} {_rendered(self.right)})"

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the dimensions of this combination, from its operands'.

        The dimensions are computed where the combination is written. Two
        operands sharing no dimension raise there.
        """
        return self._dims

    @property
    def sets(self) -> tuple[Any, ...]:
        """Return the sets this combination's dimensions are declared over."""
        held = {s.name: s for p in self.parameters() for s in p.sets}
        return tuple(held[d] for d in self.dims)

    def parameters(self) -> tuple[Any, ...]:
        """Return the parameters this combination reads, in order of appearance."""
        found = {}
        for operand in (self.left, self.right):
            if isinstance(operand, Coefficient):
                for parameter in operand.parameters():
                    found.setdefault(parameter.name, parameter)
        return tuple(found.values())

    def __repr__(self) -> str:
        return f"Derived({self.name!r}, {self.dims})"

    def materialise(self) -> Any:
        """Return the array this combination computes to, over its own frame."""
        left = self.left
        if isinstance(left, Coefficient):
            left = left.materialise()
        if self.right is None:
            return -left
        right = self.right
        if isinstance(right, Coefficient):
            right = right.materialise()
        if self.symbol == "/":
            if isinstance(right, (int, float, np.number)):
                _finite_divisor(right, f"coefficient {_rendered(self.left)}")
            else:
                _finite_quotient(right, self.right.name)
            return left / right
        return _BINARY[self.symbol](left, right)

    def __getitem__(self, sets: Any) -> "DerivedRef":
        """Return this combination read at its sets.

        The reading is checked against the dimensions of the combination. A
        transposed or short reading raises where it is written.
        """
        return DerivedRef(self, self._read(sets, self))


class DerivedRef(Coefficient):
    """A derived coefficient read at its sets, and at the members fixed."""

    def __init__(self, derived: Any, fixed: dict[str, Any] | None = None) -> None:
        self.derived = derived
        self.fixed = dict(fixed) if fixed else {}

    @property
    def name(self) -> str:
        """Return the arithmetic of the combination this reference reads."""
        return self.derived.name

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the dimensions of this reference, without those it fixes."""
        return tuple(d for d in self.derived.dims if d not in self.fixed)

    @property
    def sets(self) -> tuple[Any, ...]:
        """Return the sets this reference's dimensions are declared over."""
        return self.derived.sets

    def parameters(self) -> tuple[Any, ...]:
        """Return the parameters the combination this reference reads."""
        return self.derived.parameters()

    def __repr__(self) -> str:
        return f"DerivedRef({self.name!r}, {self.dims})"

    def materialise(self) -> Any:
        """Return the combination's array, read at the members this fixes."""
        array = self.derived.materialise()
        if not self.fixed:
            return array
        return array.sel(self.fixed)
