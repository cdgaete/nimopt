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
    """How an operand reads inside a combination's name."""
    return operand.name if isinstance(operand, Coefficient) else str(operand)


def _finite_divisor(divisor: Any, named: str) -> None:
    """Refuse a divisor that is zero, naming what is divided.

    A numpy scalar divides to infinity where a Python number raises, so the
    rule is stated rather than left to the arithmetic: a coefficient reaching
    a solver is finite.
    """
    if float(divisor) == 0.0:
        raise ZeroDivisionError(
            f"{named} is divided by zero; a divisor of zero states a "
            f"coefficient no solver can read, so it is prepared before it "
            f"reaches an expression"
        )


def _finite_quotient(array: Any, name: str) -> None:
    """Refuse a divisor carrying a zero, naming the first coordinate it sits at."""
    at = np.flatnonzero(array.values() == 0.0)
    if at.size:
        where = {
            d: labels[at[0]].item() for d, labels in array.domain().labels().items()
        }
        raise ZeroDivisionError(
            f"divisor {name} carries a zero at {at.size} coordinate(s), the "
            f"first at {where}; a quotient there states a coefficient no "
            f"solver can read"
        )


class Coefficient:
    """What a term reads as its coefficient.

    A `name` to report, the `dims` it carries, the array it `materialise()`s
    to, and a reading at its sets. A parameter's reference and a combination
    of coefficients both answer that surface, so a verb reporting a
    coefficient reads one thing whichever it is handed.

    The arithmetic is here because both answer it alike: a coefficient
    meeting an expression multiplies its terms, one meeting another
    coefficient or a number states a combination that computes where the
    matrix is built, and everything else is declined so the operand on the
    right is offered its turn.
    """

    __array_ufunc__ = None
    __hash__ = None

    @property
    def name(self) -> str:
        """What this coefficient is called where it is reported."""
        raise NotImplementedError

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions this coefficient carries."""
        raise NotImplementedError

    def materialise(self) -> Any:
        """The array this coefficient computes to."""
        raise NotImplementedError

    def held(self) -> Any:
        """The array this coefficient already holds, or None where it holds none.

        A parameter that is bound holds its array, so a fact about its values
        is in hand where the arithmetic is written. A combination holds none:
        computing one to read a fact off it is the build, done early.
        """
        return None

    def parameters(self) -> tuple[Any, ...]:
        """The parameters this coefficient reads, in order of appearance."""
        raise NotImplementedError

    def __getitem__(self, sets: Any) -> Any:
        """Refuse a second reading of a coefficient already read."""
        raise TypeError(
            f"coefficient {self.name} is already read at {self.dims}; a "
            f"coefficient is read at its sets once"
        )

    def _read(self, sets: Any, holder: Any) -> dict[str, Any]:
        """The members `sets` fixes, checked against the dimensions carried."""
        given, shifts, fixed = reference(sets, self.dims)
        if shifts:
            raise ValueError(
                f"coefficient {self.name} is read at a lag {sorted(shifts)}; "
                f"state the lag at the variable's reference, where a "
                f"coefficient multiplies the row it lands on"
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
            f"a power takes a number, and {_rendered(other)} raised to "
            f"coefficient {self.name} varies by coordinate; nimopt expresses a "
            f"linear term, and such a value is data a caller prepares"
        )

    def __pow__(self, other: Any) -> Any:
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(
                f"a power takes a number, and {_rendered(other)} carries "
                f"dimensions; an exponent that varies by coordinate is data a "
                f"caller prepares before a parameter exists"
            )
        return Derived(self, other, "**")

    def __abs__(self) -> Any:
        raise TypeError(
            f"coefficient {self.name} has no absolute value here; nimopt "
            f"reduces with `Sum` over its sets, and a magnitude is prepared "
            f"before a parameter exists"
        )

    def _no_row(self, other: Any) -> Any:
        """Refuse a comparison of two coefficients, declining one that states a row.

        An expression or a symbol answers the comparison itself, so this
        declines and Python offers it the reflected operator, which reverses
        the sense: `capacity[G, T] >= gen[G, T]` is the row
        `gen[G, T] <= capacity[G, T]`.
        """
        from nimopt.term import Expression

        if isinstance(other, (Expression, Symbol)):
            return NotImplemented
        raise TypeError(
            f"coefficient {self.name} compared with {_rendered(other)} states "
            f"no row; an equation needs a variable on one side of it"
        )

    __le__ = _no_row
    __ge__ = _no_row
    __lt__ = _no_row
    __gt__ = _no_row
    __eq__ = _no_row


class Derived(Coefficient):
    """Two coefficients and an operator, or one coefficient and a number.

    The combination holds handles: it states its dimensions from its
    operands' and computes once, where the term it multiplies materialises.
    A coefficient is therefore written in a definition before any data
    exists.
    """

    def __init__(self, left: Any, right: Any, symbol: str) -> None:
        self.left = left
        self.right = right
        self.symbol = symbol
        carried = [
            operand.dims
            for operand in (left, right)
            if isinstance(operand, Coefficient)
        ]
        self._dims = combined_dims(*carried) if len(carried) == 2 else carried[0]

    @property
    def name(self) -> str:
        """The arithmetic this combination states, as it was written."""
        if self.right is None:
            return f"({self.symbol}{self.left.name})"
        return f"({_rendered(self.left)} {self.symbol} {_rendered(self.right)})"

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions the combination carries, read from its operands'.

        They are settled where the combination is written, so two operands
        sharing no dimension are refused there rather than at build.
        """
        return self._dims

    @property
    def sets(self) -> tuple[Any, ...]:
        """The sets the dimensions this carries are declared over."""
        held = {s.name: s for p in self.parameters() for s in p.sets}
        return tuple(held[d] for d in self.dims)

    def parameters(self) -> tuple[Any, ...]:
        """The parameters this combination reads, in order of appearance."""
        found = {}
        for operand in (self.left, self.right):
            if isinstance(operand, Coefficient):
                for parameter in operand.parameters():
                    found.setdefault(parameter.name, parameter)
        return tuple(found.values())

    def __repr__(self) -> str:
        return f"Derived({self.name!r}, {self.dims})"

    def materialise(self) -> Any:
        """The array this combination computes to, over the frame it states."""
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
        """This combination read at its sets, as a parameter is read at its.

        The reading is checked against the dimensions the combination
        carries, so a transposed or short spelling is refused where it is
        written rather than a layer away.
        """
        return DerivedRef(self, self._read(sets, self))


class DerivedRef(Coefficient):
    """A derived coefficient read at its sets, and at the members fixed."""

    def __init__(self, derived: Any, fixed: dict[str, Any] | None = None) -> None:
        self.derived = derived
        self.fixed = dict(fixed) if fixed else {}

    @property
    def name(self) -> str:
        """The arithmetic the combination this reads states."""
        return self.derived.name

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions the reference carries, without those it fixes."""
        return tuple(d for d in self.derived.dims if d not in self.fixed)

    @property
    def sets(self) -> tuple[Any, ...]:
        """The sets the dimensions this carries are declared over."""
        return self.derived.sets

    def parameters(self) -> tuple[Any, ...]:
        """The parameters the combination this reads carries."""
        return self.derived.parameters()

    def __repr__(self) -> str:
        return f"DerivedRef({self.name!r}, {self.dims})"

    def materialise(self) -> Any:
        """The combination's array, read at the members this reference fixes."""
        array = self.derived.materialise()
        if not self.fixed:
            return array
        return array.sel(self.fixed)
