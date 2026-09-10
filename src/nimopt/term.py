"""A linear expression as the terms that were typed, holding no array."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from nimblend import Domain, SparseArray

from nimopt.coefficient import NOT_A_COEFFICIENT, Coefficient, Derived
from nimopt.names import COLUMN
from nimopt.sets import LaggedSet, rows_of
from nimopt.symbol import Symbol, read_at_its_sets, read_bare


def _names(sets: Any) -> tuple[str, ...]:
    given = sets if isinstance(sets, tuple) else (sets,)
    lagged = [s.name for s in given if isinstance(s, LaggedSet)]
    if lagged:
        raise ValueError(
            f"a sum is over the members of {lagged}, so it takes the set and "
            f"not a lag of it; state the lag at the variable's reference"
        )
    return tuple(s.name for s in given)


def _dims_of(given: Any) -> tuple[str, ...]:
    """The dimensions a condition names, however it is spelled."""
    if isinstance(given, tuple):
        return tuple(s.name for s in given)
    return given.dims


_CHAINED = (
    "a chained comparison such as 0 <= expr <= 10 reads as two comparisons "
    "joined by `and` and keeps only the second, so state each bound separately"
)


class Term:
    """One variable, an optional coefficient, the dimensions summed, a scale.

    The term is the recipe for a block of coefficients and holds handles
    rather than arrays, so writing it costs what was typed: an expression
    over a million columns costs the same as one over ten.
    """

    def __init__(
        self,
        variable: Any,
        coefficient: Any = None,
        summed: Iterable[str] = (),
        scale: float = 1.0,
        shifts: Mapping[str, tuple[int, str]] | None = None,
        where: Any = None,
        fixed: Mapping[str, Any] | None = None,
    ) -> None:
        self.variable = variable
        self.coefficient = coefficient
        self.summed = tuple(summed)
        self.scale = float(scale)
        self.shifts = dict(shifts) if shifts else {}
        self.where = where
        self.fixed = dict(fixed) if fixed else {}

    @property
    def carried_dims(self) -> tuple[str, ...]:
        """Every dimension the term reads, before the reductions.

        A coefficient introducing a dimension leads, because that is the
        order the product of the two arrays returns.
        """
        dims = self.variable.dims
        coefficient = self.coefficient
        if coefficient is not None and not set(coefficient.dims) <= set(dims):
            dims = coefficient.dims + tuple(
                d for d in dims if d not in coefficient.dims
            )
        return dims

    @property
    def free_dims(self) -> tuple[str, ...]:
        """The dimensions surviving the reductions and the members fixed."""
        return tuple(
            d for d in self.carried_dims if d not in self.summed and d not in self.fixed
        )

    def __repr__(self) -> str:
        name = "" if self.coefficient is None else f"{self.coefficient.name} * "
        where = "" if self.where is None else f", where={_dims_of(self.where)}"
        return (
            f"Term({name}{self.variable.name}, summed={self.summed}, "
            f"shifts={self.shifts}{where})"
        )

    def with_coefficient(self, coefficient: Any) -> "Term":
        """The term reading `coefficient` as its coefficient.

        A coefficient carrying dimensions the variable does not states which
        rows read the variable's column, so its support is the pairing of
        rows to columns and those dimensions are free dimensions of the term.
        A term already reading one multiplies the two, because a linear term
        carries a single coefficient and a product of coefficients is one.
        """
        if self.coefficient is not None:
            coefficient = Derived(coefficient, self.coefficient, "*")
        return Term(
            self.variable,
            coefficient,
            self.summed,
            self.scale,
            self.shifts,
            self.where,
            self.fixed,
        )

    def summing(self, dims: Iterable[str]) -> "Term":
        """The term with `dims` added to the dimensions summed away."""
        repeated = [d for d in dims if d in self.summed]
        if repeated:
            raise ValueError(
                f"term {self.variable.name!r} already sums over {repeated}; a "
                f"dimension is reduced once, and a second reduction has "
                f"nothing left to reduce"
            )
        lacking = [d for d in dims if d not in self.free_dims]
        if lacking:
            raise ValueError(
                f"term {self.variable.name!r} is free over {self.free_dims} "
                f"and does not carry {lacking}"
            )
        return Term(
            self.variable,
            self.coefficient,
            self.summed + tuple(dims),
            self.scale,
            self.shifts,
            self.where,
            self.fixed,
        )

    def scaled(self, by: float) -> "Term":
        """The term with its scale multiplied by `by`."""
        return Term(
            self.variable,
            self.coefficient,
            self.summed,
            self.scale * by,
            self.shifts,
            self.where,
            self.fixed,
        )

    def restricted_to(self, domain: Any) -> "Term":
        """The term reading only the coordinates `domain` carries."""
        if self.where is not None:
            raise ValueError(
                f"term {self.variable.name!r} already reads a condition over "
                f"{_dims_of(self.where)}; a term carries one condition"
            )
        given = _dims_of(domain)
        lacking = [d for d in given if d not in self.carried_dims]
        if lacking:
            raise ValueError(
                f"the condition is over {given} and term "
                f"{self.variable.name!r} reads {self.carried_dims} and does "
                f"not carry {lacking}; a condition names the coordinates of "
                f"the term it restricts"
            )
        return Term(
            self.variable,
            self.coefficient,
            self.summed,
            self.scale,
            self.shifts,
            domain,
            self.fixed,
        )

    def materialise(
        self, frame: Sequence[str], coords: Mapping[str, Any], record: Any = None
    ) -> SparseArray:
        """The term's coefficients over `(*frame, COLUMN)`.

        The variable's columns carry the value 1.0; the coefficient
        multiplies them by broadcasting over the column dimension only the
        variable carries; each summed dimension is reduced away. A frame
        wider than the term's own dimensions is reached by replication,
        which is what a coefficient over fewer dimensions than the rows
        means.

        A lag moves the variable's columns onto the rows that read them
        before the coefficient multiplies, so the coefficient is read at the
        row's own coordinate rather than at the lagged one. A condition
        restricts the entries once the coefficient has placed them, so it
        names any coordinate the term reads, including one only the
        coefficient carries. A fixed member is read first, because it names
        the entries the rest applies to.
        """
        array = self.variable.terms()
        for dim, (amount, mode) in self.shifts.items():
            array = array.shift({dim: amount}, mode=mode)
        if self.fixed:
            array = array.sel(self.fixed)
        if self.coefficient is not None:
            multiplied = self.coefficient.materialise() * array
            if record is not None:
                record.coefficient(self, array, multiplied)
            array = multiplied
        if self.where is not None:
            array = array.restrict(
                rows_of(self.where, None, f"term {self.variable.name!r}", "condition")
            )
        if self.scale != 1.0:
            array = array * self.scale
        for dim in self.summed:
            array = array.sum(dim)
        missing = tuple(d for d in frame if d not in array.dims)
        if missing:
            array = array.expand(missing, {d: coords[d] for d in missing})
        return array.transpose(*frame, COLUMN)


class ParamRef(Coefficient):
    """A parameter read at the sets given, and at the members fixed."""

    def __init__(self, param: Any, fixed: Mapping[str, Any] | None = None) -> None:
        self.param = param
        self.fixed = dict(fixed) if fixed else {}

    @property
    def name(self) -> str:
        """The parameter's name."""
        return self.param.name

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions the reference carries, without those it fixes."""
        return tuple(d for d in self.param.dims if d not in self.fixed)

    def __repr__(self) -> str:
        return f"ParamRef({self.name!r}, {self.dims})"

    @property
    def sets(self) -> tuple[Any, ...]:
        """The sets this reference's parameter is declared over."""
        return self.param.sets

    def parameters(self) -> tuple[Any, ...]:
        """The parameter this reference reads."""
        return (self.param,)

    def held(self) -> Any:
        """The parameter's array, or None while it is declared and carries none."""
        return None if self.param.declared else self.materialise()

    def materialise(self) -> Any:
        """The coefficient array, read at the members this reference fixes."""
        array = self.param.materialise()
        if not self.fixed:
            return array
        return array.sel(self.fixed)


class Expression:
    """A list of terms, a constant, and the frame the terms agree on.

    The frame is the union of the terms' free dimensions, ordered by the term
    that introduces each. A term narrower than the frame reaches it by
    replication when the expression materialises.

    The constant carries no dimension and no column: it folds into the
    right-hand side where a constraint is built, and into the reported
    objective as a fixed cost.
    """

    __array_ufunc__ = None

    def __init__(self, terms: Iterable["Term"], constant: float = 0.0) -> None:
        self.terms = tuple(terms)
        self.constant = float(constant)

    @property
    def frame(self) -> tuple[str, ...]:
        """The dimensions the terms are free over, in order of introduction."""
        seen = []
        for t in self.terms:
            for d in t.free_dims:
                if d not in seen:
                    seen.append(d)
        return tuple(seen)

    def __repr__(self) -> str:
        from nimopt.syntax import render

        return render(self)

    def __add__(self, other: Any) -> Any:
        """The expression with `other` added, a symbol read first.

        A symbol still carrying dimensions declines rather than refusing, so
        the operand on the right is offered its turn and names its own
        reading.
        """
        other = read_bare(other)
        if isinstance(other, (int, float, np.number)):
            return Expression(self.terms, self.constant + float(other))
        if isinstance(other, Symbol):
            return NotImplemented
        if not isinstance(other, Expression):
            raise TypeError(
                f"an expression adds an expression or a number; a coefficient "
                f"states no row until a variable multiplies it, so "
                f"{type(other).__name__} does not add to one"
            )
        return Expression(self.terms + other.terms, self.constant + other.constant)

    __radd__ = __add__

    def __neg__(self) -> Any:
        return Expression([t.scaled(-1.0) for t in self.terms], -self.constant)

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, (int, float, np.number)):
            return self.__add__(-float(other))
        return self.__add__(-other)

    def __rsub__(self, other: Any) -> Any:
        return (-self).__add__(other)

    def __mul__(self, other: Any) -> Any:
        """The expression scaled by a number, or read by a coefficient.

        A coefficient answers the product itself, so this declines rather
        than refusing and the operand on the right is offered its turn.
        """
        other = read_bare(other)
        if isinstance(other, (Coefficient, Symbol)):
            return NotImplemented
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(NOT_A_COEFFICIENT)
        by = float(other)
        return Expression([t.scaled(by) for t in self.terms], self.constant * by)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        """The expression scaled by the reciprocal of a number or coefficient.

        A divisor of zero raises `ZeroDivisionError` however it is spelled,
        because a numpy scalar divides to infinity where a Python number
        raises and a column reaching a solver with an infinite cost is a
        model nobody wrote.
        """
        other = read_bare(other)
        if isinstance(other, Coefficient):
            return (1.0 / other) * self
        if isinstance(other, Symbol):
            return NotImplemented
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(
                "an expression divides by a number or by a coefficient; "
                "nimopt expresses a linear term, so a variable in a "
                "denominator is not one"
            )
        if float(other) == 0.0:
            raise ZeroDivisionError(
                "an expression is divided by zero; a column scaled by "
                "infinity states a model no solver can read"
            )
        by = 1.0 / float(other)
        return Expression([t.scaled(by) for t in self.terms], self.constant * by)

    def __rtruediv__(self, other: Any) -> Any:
        raise TypeError(
            "nimopt expresses a linear term, so a variable in a denominator "
            "is not one; state the reciprocal as a coefficient the variable "
            "multiplies"
        )

    def _relate(self, other: Any, sense: str) -> "Relation":
        return Relation(self, sense, other)

    def __ne__(self, other: Any) -> Any:
        raise TypeError(
            "an equation states one bound, with `<=`, `>=` or `==`; an LP has "
            "no row for `!=`"
        )

    def __le__(self, other: Any) -> Any:
        return self._relate(other, "<=")

    def __ge__(self, other: Any) -> Any:
        return self._relate(other, ">=")

    def __eq__(self, other: Any) -> Any:
        return self._relate(other, "==")

    __hash__ = None

    def __bool__(self) -> bool:
        raise TypeError(f"an expression has no truth value; {_CHAINED}")

    def __len__(self) -> int:
        raise TypeError(
            "an expression has no length; the terms it carries are `terms` "
            "and the dimensions it is free over are `frame`"
        )

    def __pow__(self, other: Any) -> Any:
        raise TypeError(
            "nimopt expresses a linear term, so a variable raised to a power "
            "is not one; a coefficient takes the power instead, and a "
            "variable multiplies it"
        )

    __rpow__ = __pow__

    def __abs__(self) -> Any:
        raise TypeError(
            "nimopt expresses a linear term, so the absolute value of one is "
            "not linear; reduce with `Sum` over its sets, or state the "
            "magnitude with two rows bounding the expression"
        )

    def _no_strict(self, other: Any) -> Any:
        raise TypeError(
            "an LP has no row for a strict inequality; state `<=` or `>=`. "
            "`min` and `max` compare two expressions this way and are not "
            "linear either, so reduce with `Sum` over the sets instead"
        )

    __lt__ = _no_strict
    __gt__ = _no_strict

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse a reduction that states no set.

        `numpy.sum` reaches this, and without it returns the expression
        unchanged, having reduced nothing.
        """
        raise TypeError(
            "an expression is reduced over the sets it is summed across; "
            "state them with `Sum(I, J, expression)`"
        )

    @property
    def coords(self) -> dict[str, Any]:
        """Each dimension's coordinate, from the sets its variable is over."""
        found = {}
        for t in self.terms:
            for s in t.variable.sets:
                found[s.name] = s.coord
        return found

    def materialise(
        self, record: Any = None, progress: Any = None
    ) -> tuple[SparseArray, Domain]:
        """The block over `(*frame, COLUMN)`, and the rows it states.

        Every entry is produced by a `nimblend` operation. The row domain is the
        intersection of the terms' domains over the frame: a row one term
        does not reach is a row the expression does not state, because a row
        missing one of its terms says something that was not written.

        A reporter is told as each term is built, which is the finest the
        frame divides into: a term is one `nimblend` operation chain, and
        nothing above it can say how far through one it is.
        """
        frame = self.frame
        coords = self.coords
        blocks = []
        for term in self.terms:
            blocks.append(term.materialise(frame, coords, record))
            if progress is not None:
                progress.term(term)
        rows = blocks[0].domain(frame)
        total = blocks[0]
        if record is not None:
            record.term_rows(self.terms[0], rows)
        for term, block in zip(self.terms[1:], blocks[1:]):
            reaches = block.domain(frame)
            if record is not None:
                record.term_rows(term, reaches)
            rows = rows.intersect(reaches)
            total = total + block
        return total, rows


def Sum(*args: Any, where: Any = None) -> "Expression":
    """`Sum(I, J, expression)` — the expression reduced over the named sets.

    `where=` takes a domain and restricts each term's entries before the
    reduction, so a sum states the coordinates it runs over rather than every
    coordinate of the product.
    """
    if len(args) < 2:
        raise ValueError("Sum takes one or more sets and then an expression")
    *sets, expression = args
    if not isinstance(expression, Expression):
        raise TypeError(
            f"Sum reduces an expression over its sets, and a coefficient "
            f"states no row until a variable multiplies it; got "
            f"{type(expression).__name__} last"
        )
    dims = _names(tuple(sets))
    terms = [t.summing(dims) for t in expression.terms]
    if where is not None:
        terms = [t.restricted_to(where) for t in terms]
    return Expression(terms)


class Relation:
    """An expression, a sense and a right-hand side."""

    __array_ufunc__ = None

    def __init__(self, expression: "Expression", sense: str, rhs: Any) -> None:
        """An expression bounded by `rhs`, which is read and then folded.

        A right-hand side carrying columns states no bound of its own, so it
        moves left against zero. Both happen here rather than at the
        comparison, so a relation assembled directly reads as one written
        with `<=`.
        """
        rhs = read_at_its_sets(rhs)
        if isinstance(rhs, Expression):
            expression, rhs = expression - rhs, 0.0
        self.expression = expression
        self.sense = sense
        self.rhs = rhs

    def __repr__(self) -> str:
        from nimopt.syntax import render

        return render(self)

    def __bool__(self) -> bool:
        raise TypeError(f"a relation has no truth value; {_CHAINED}")

    def _one_bound(self, other: Any) -> Any:
        raise TypeError(
            "a relation is already an equation and states one bound; compare "
            "the expression a second time in its own equation rather than "
            "comparing the relation"
        )

    __le__ = _one_bound
    __ge__ = _one_bound
    __lt__ = _one_bound
    __gt__ = _one_bound
    __eq__ = _one_bound
    __ne__ = _one_bound
    __hash__ = None
