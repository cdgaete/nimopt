"""A linear expression as the terms that were written, with no array."""

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
            f"a sum is over the members of {lagged} and takes the set, not a "
            f"lag of it; write the lag at the variable's reference"
        )
    return tuple(s.name for s in given)


def _dims_of(given: Any) -> tuple[str, ...]:
    """Return the dimensions a condition is over."""
    if isinstance(given, tuple):
        return tuple(s.name for s in given)
    return given.dims


_EACH_BOUND = "write each bound in its own equation"


class Term:
    """One variable, an optional coefficient, the dimensions summed, a scale.

    The term stores handles, not arrays. Building it costs the size of what
    was written, not the size of the block it produces.
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
        """Return every dimension the term reads, before the reductions.

        A dimension introduced by the coefficient comes first, the order the
        product of the two arrays returns.
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
        """Return the dimensions left after the reductions and the fixed members."""
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
        """Return the term with `coefficient` as its coefficient.

        A coefficient over dimensions the variable is not over pairs rows to
        columns, and those dimensions are free dimensions of the term. A term
        that already has a coefficient multiplies the two into one `Derived`.
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
        """Return the term with `dims` added to the dimensions summed away."""
        repeated = [d for d in dims if d in self.summed]
        if repeated:
            raise ValueError(
                f"term {self.variable.name!r} already sums over {repeated}; "
                f"sum over each dimension once"
            )
        lacking = [d for d in dims if d not in self.free_dims]
        if lacking:
            raise ValueError(
                f"term {self.variable.name!r} is free over {self.free_dims} "
                f"and is not over {lacking}; sum over its free dimensions"
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
        """Return the term with its scale multiplied by `by`."""
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
        """Return the term restricted to the coordinates of `domain`."""
        if self.where is not None:
            raise ValueError(
                f"term {self.variable.name!r} already reads a condition over "
                f"{_dims_of(self.where)}; a term takes one condition"
            )
        given = _dims_of(domain)
        lacking = [d for d in given if d not in self.carried_dims]
        if lacking:
            raise ValueError(
                f"the condition is over {given} and term "
                f"{self.variable.name!r} reads {self.carried_dims} and is not "
                f"over {lacking}; restrict the term with a condition over its "
                f"own coordinates"
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
        """Return the term's coefficients over `(*frame, COLUMN)`.

        The variable's columns have the value 1.0. The operations run in this
        order: the fixed members are selected, the lag moves the columns onto
        the rows that read them, the coefficient multiplies, the condition
        restricts, the scale multiplies, and each summed dimension is
        reduced. A frame wider than the term's own dimensions is filled by
        replication.
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
        """Return the parameter's name."""
        return self.param.name

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the dimensions of this reference, without those it fixes."""
        return tuple(d for d in self.param.dims if d not in self.fixed)

    def __repr__(self) -> str:
        return f"ParamRef({self.name!r}, {self.dims})"

    @property
    def sets(self) -> tuple[Any, ...]:
        """Return the sets this reference's parameter is declared over."""
        return self.param.sets

    def parameters(self) -> tuple[Any, ...]:
        """Return the parameter this reference reads."""
        return (self.param,)

    def held(self) -> Any:
        """Return the parameter's array, or None where the parameter is declared."""
        return None if self.param.declared else self.materialise()

    def materialise(self) -> Any:
        """Return the coefficient array, read at the members this fixes."""
        array = self.param.materialise()
        if not self.fixed:
            return array
        return array.sel(self.fixed)


class Expression:
    """A list of terms, a constant, and the frame the terms agree on.

    The frame is the union of the terms' free dimensions, ordered by the term
    that introduces each. A term over fewer dimensions than the frame is
    replicated when the expression materialises. The constant is over no
    dimension and no column. It folds into the right-hand side of a
    constraint, and into the reported objective as a fixed cost.
    """

    __array_ufunc__ = None

    def __init__(self, terms: Iterable["Term"], constant: float = 0.0) -> None:
        self.terms = tuple(terms)
        self.constant = float(constant)

    @property
    def frame(self) -> tuple[str, ...]:
        """Return the dimensions the terms are free over, in order of introduction."""
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
        """Return the expression with `other` added, a symbol read first.

        Returns `NotImplemented` for a symbol over one or more dimensions.
        That symbol then raises from its own reflected operator.
        """
        other = read_bare(other)
        if isinstance(other, (int, float, np.number)):
            return Expression(self.terms, self.constant + float(other))
        if isinstance(other, Symbol):
            return NotImplemented
        if not isinstance(other, Expression):
            raise TypeError(
                f"an expression adds an expression or a number and got "
                f"{type(other).__name__}; multiply a coefficient by a "
                f"variable first"
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
        """Return the expression scaled by a number.

        Returns `NotImplemented` for a coefficient or a symbol. The
        coefficient then multiplies the expression's terms through its own
        operator.
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
        """Return the expression divided by a number or a Coefficient.

        Raises ZeroDivisionError for a divisor of zero, including a numpy
        zero. Raises TypeError for a divisor that is not a number or a
        Coefficient.
        """
        other = read_bare(other)
        if isinstance(other, Coefficient):
            return (1.0 / other) * self
        if isinstance(other, Symbol):
            return NotImplemented
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(
                "cannot divide an expression by a variable: expressions are "
                "linear; divide by a number or a Coefficient"
            )
        if float(other) == 0.0:
            raise ZeroDivisionError(
                "expression divided by zero; divide by a non-zero number"
            )
        by = 1.0 / float(other)
        return Expression([t.scaled(by) for t in self.terms], self.constant * by)

    def __rtruediv__(self, other: Any) -> Any:
        raise TypeError(
            "cannot divide by an expression: expressions are linear; declare "
            "the reciprocal as a coefficient the variable multiplies"
        )

    def _relate(self, other: Any, sense: str) -> "Relation":
        return Relation(self, sense, other)

    def __ne__(self, other: Any) -> Any:
        raise TypeError(
            "an LP has no row for `!=`; write one bound with `<=`, `>=` or `==`"
        )

    def __le__(self, other: Any) -> Any:
        return self._relate(other, "<=")

    def __ge__(self, other: Any) -> Any:
        return self._relate(other, ">=")

    def __eq__(self, other: Any) -> Any:
        return self._relate(other, "==")

    __hash__ = None

    def __bool__(self) -> bool:
        raise TypeError(f"an expression has no truth value; {_EACH_BOUND}")

    def __len__(self) -> int:
        raise TypeError(
            "an expression has no length; read `terms` for its terms and "
            "`frame` for the dimensions it is free over"
        )

    def __pow__(self, other: Any) -> Any:
        raise TypeError(
            "cannot raise an expression to a power: expressions are linear; "
            "raise a coefficient to the power and multiply it by a variable"
        )

    __rpow__ = __pow__

    def __abs__(self) -> Any:
        raise TypeError(
            "an expression has no absolute value: expressions are linear; "
            "bound the expression with two rows, or reduce it with `Sum` "
            "over its sets"
        )

    def _no_strict(self, other: Any) -> Any:
        raise TypeError(
            "an LP has no row for a strict inequality; write `<=` or `>=`, "
            "and reduce with `Sum` in place of `min` or `max`"
        )

    __lt__ = _no_strict
    __gt__ = _no_strict

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        """Raise TypeError for a reduction that names no set.

        `numpy.sum` calls this method. Without it, `numpy.sum` returns the
        expression unchanged and reduces nothing.
        """
        raise TypeError(
            "an expression is reduced over the sets it is summed across; "
            "name them with `Sum(I, J, expression)`"
        )

    @property
    def coords(self) -> dict[str, Any]:
        """Return each dimension's coordinate, from the sets its variable is over."""
        found = {}
        for t in self.terms:
            for s in t.variable.sets:
                found[s.name] = s.coord
        return found

    def materialise(
        self, record: Any = None, progress: Any = None
    ) -> tuple[SparseArray, Domain]:
        """Return the block over `(*frame, COLUMN)` and the rows it spans.

        Every entry is produced by a `nimblend` operation. The row domain is
        the intersection of the terms' domains over the frame. A row absent
        from one term is absent from the expression. `progress` is called
        once per term, the finest division of the work available.
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
            spanned = block.domain(frame)
            if record is not None:
                record.term_rows(term, spanned)
            rows = rows.intersect(spanned)
            total = total + block
        return total, rows


def Sum(*args: Any, where: Any = None) -> "Expression":
    """Return the expression reduced over the named sets.

    `Sum(I, J, expression)` sums over `I` and `J`. `where=` takes a domain
    and restricts each term's entries before the reduction.
    """
    if len(args) < 2:
        raise ValueError("Sum takes one or more sets and then an expression")
    *sets, expression = args
    if not isinstance(expression, Expression):
        raise TypeError(
            f"Sum takes an expression last and got "
            f"{type(expression).__name__}; multiply a coefficient by a "
            f"variable to form an expression"
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
        """Store an expression bounded by `rhs`, read at its sets and folded.

        A right-hand side that contains columns is subtracted from the
        expression and the bound becomes zero. A relation built directly is
        identical to one written with `<=`.
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
        raise TypeError(f"a relation has no truth value; {_EACH_BOUND}")

    def _one_bound(self, other: Any) -> Any:
        raise TypeError(
            "a relation is already an equation with one bound; compare the "
            "expression again in its own equation"
        )

    __le__ = _one_bound
    __ge__ = _one_bound
    __lt__ = _one_bound
    __gt__ = _one_bound
    __eq__ = _one_bound
    __ne__ = _one_bound
    __hash__ = None
