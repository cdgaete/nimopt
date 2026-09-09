"""A constraint: an expression, a sense and a right-hand side."""

from typing import Any

import numpy as np
import numpy.typing as npt
from nimblend import Domain, EntryBuffer, SparseArray

from nimopt.coefficient import Coefficient
from nimopt.names import ROW
from nimopt.param import Param
from nimopt.sets import rows_of
from nimopt.term import Relation

SENSES = ("<=", ">=", "==")


class Constraint:
    """Rows over an expression's frame, bounded by a right-hand side.

    A row derived from the terms exists where every term reaches and the
    right-hand side carries a value. A coefficient absent inside a sum
    removes a term and leaves the row standing; a term absent along a free
    dimension removes the row, because a row missing one of its terms states
    something that was not written. `over=` states the rows outright instead,
    so a term reaching some of them contributes where it reaches.

    A condition intersects the row domain, so a row the condition omits is a
    row the constraint does not state.

    A constant the expression carries folds into the right-hand side, so
    `x + 1 <= 5` states the row `x <= 4` and the matrix gains nothing.

    The expression is symbolic, so the constraint holds the recipe rather
    than a block: it materialises once to measure its shape and once to
    write it, and holds nothing between.
    """

    def __init__(
        self, name: str, relation: Any, where: Any = None, over: Any = None
    ) -> None:
        self.name = str(name)
        if not isinstance(relation, Relation):
            raise TypeError(
                f"constraint {self.name!r} takes a comparison of an "
                f"expression, such as `expr <= rhs`; got "
                f"{type(relation).__name__}"
            )
        if relation.sense not in SENSES:
            raise ValueError(f"sense is one of {SENSES}; got {relation.sense!r}")
        self.expression = relation.expression
        self.sense = relation.sense
        self.rhs = relation.rhs
        self.frame = self.expression.frame
        self.where = where
        self.over = over
        if over is not None and where is not None:
            raise ValueError(
                f"constraint {self.name!r} states its rows with over= and "
                f"narrows them with where=; state one"
            )
        self.rows, self._rhs_values, self._nnz = narrow(self)

    def __repr__(self) -> str:
        return (
            f"Constraint({self.name!r}, {self.frame}, {self.n_rows} rows, "
            f"{self.nnz} coefficients)"
        )

    @property
    def n_rows(self) -> int:
        """Number of rows this constraint contributes."""
        return self.rows.size

    @property
    def nnz(self) -> int:
        """Number of coefficients this constraint contributes."""
        return self._nnz

    @property
    def relation(self) -> Relation:
        """The comparison this constraint was declared from."""
        return Relation(self.expression, self.sense, self.rhs)

    def write_bounds(
        self, lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64]
    ) -> None:
        """Write this constraint's rows of `lower` and `upper` in place.

        The vectors are the model's, so nothing is allocated per constraint.
        """
        values = self._rhs_values
        if self.sense == "<=":
            lower[:] = -np.inf
            upper[:] = values
        elif self.sense == ">=":
            lower[:] = values
            upper[:] = np.inf
        else:
            lower[:] = values
            upper[:] = values

    def write_into(
        self, buffer: EntryBuffer, row_start: int, progress: Any = None
    ) -> SparseArray:
        """The block over `(ROW, COLUMN)`, its entries written into `buffer`.

        The block is grouped straight into the slice the buffer reserves, so
        it never exists as a second object. The frame precedes the column
        dimension in canonical order, so the grouping reads a leading prefix
        and the result is canonical as written.
        """
        block, _ = self.expression.materialise(progress=progress)
        n = int((self.rows.positions_of(block) >= 0).sum())
        if n != self._nnz:
            raise ValueError(
                f"constraint {self.name!r} measured {self._nnz} coefficients "
                f"and built {n}; the data its parameters read changed between "
                f"the two"
            )
        return block.group(
            self.frame,
            into=ROW,
            domain=self.rows,
            offset=row_start,
            out=buffer.reserve(n),
        )


def narrow(
    constraint: "Constraint", record: Any = None
) -> tuple[Domain, npt.NDArray[np.float64], int]:
    """The rows a constraint states, its right-hand side values and its nonzeros.

    The rows a constraint's terms reach, narrowed by its condition and by the
    coverage of its right-hand side, or stated outright by its `over=`. A
    recorder given here is told what each narrowing dropped, which is how a
    dropped row — leaving no trace in the matrix — is attributed at all.
    """
    frame = constraint.frame
    block, rows = constraint.expression.materialise(record)
    if record is not None:
        record.reached(frame, rows)
    if constraint.over is not None:
        rows = rows_of(
            constraint.over, frame, f"constraint {constraint.name!r}", "over="
        )
        if record is not None:
            record.stated(rows)
    elif constraint.where is not None:
        condition = rows_of(
            constraint.where, frame, f"constraint {constraint.name!r}", "condition"
        )
        narrowed = rows.intersect(condition)
        if record is not None:
            record.dropped(rows, narrowed, "where", _named(constraint.where))
        rows = narrowed
    rhs = constraint.rhs
    if isinstance(rhs, Coefficient):
        if rhs.dims != frame:
            raise ValueError(
                f"constraint {constraint.name!r} has free dimensions "
                f"{frame}; its right-hand side {rhs.name!r} is over {rhs.dims}"
            )
        values = rhs.materialise()
        covered = values.domain(frame)
        if constraint.over is None:
            narrowed = rows.intersect(covered)
            if record is not None:
                record.dropped(rows, narrowed, "absent-rhs", rhs.name)
            rows = narrowed
        else:
            missing = rows.difference(covered)
            if missing.size:
                first = {d: v[0] for d, v in missing.labels().items()}
                raise ValueError(
                    f"constraint {constraint.name!r} states {rows.size} rows "
                    f"and right-hand side {rhs.name!r} misses {missing.size} "
                    f"of them, the first at {first}; a right-hand side covers "
                    f"every row the constraint states"
                )
        rhs_values = values.restrict(rows).values()
    elif isinstance(rhs, (int, float, np.number)):
        rhs_values = np.full(rows.size, float(rhs))
    else:
        raise TypeError(
            "a right-hand side is a number or a coefficient read at the "
            f"constraint's free dimensions; got {type(rhs).__name__}"
        )
    rhs_values = rhs_values - constraint.expression.constant
    if record is not None:
        record.settled(rows)
    return rows, rhs_values, int((rows.positions_of(block) >= 0).sum())


def _named(given: Any) -> str:
    """What a row domain was given as, for a report to name."""
    if isinstance(given, Param):
        return given.name
    if isinstance(given, tuple):
        return ",".join(s.name for s in given)
    return ",".join(given.dims)
