"""A constraint: an expression, a sense and a right-hand side."""

from typing import Any

import numpy as np
import numpy.typing as npt
from nimblend import Domain, EntryBuffer, SparseArray

from nimopt.coefficient import Coefficient
from nimopt.names import ROW
from nimopt.param import Param
from nimopt.sets import displayed, rows_of
from nimopt.term import Relation

SENSES = ("<=", ">=", "==")


class Constraint:
    """Rows over an expression's frame, bounded by a right-hand side.

    A row derived from the terms exists where every term is present and the
    right-hand side has a value. A coefficient absent inside a sum removes a
    term and keeps the row. A term absent along a free dimension removes the
    row. `over=` declares the rows instead. A condition intersects the row
    domain, and a row the condition omits is not a row of the constraint.

    A constant in the expression folds into the right-hand side. `x + 1 <= 5`
    declares the row `x <= 4` and adds nothing to the matrix.

    The expression is symbolic. The constraint materialises it once to measure
    its shape and once to write it, and stores no block between.
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
                f"constraint {self.name!r} is given over= and where= together; "
                f"pass one of them"
            )
        self.rows, self._rhs_values, self._nnz = narrow(self)

    def __repr__(self) -> str:
        return (
            f"Constraint({self.name!r}, {self.frame}, {self.n_rows} rows, "
            f"{self.nnz} coefficients)"
        )

    @property
    def n_rows(self) -> int:
        """Return the number of rows this constraint contributes."""
        return self.rows.size

    @property
    def nnz(self) -> int:
        """Return the number of coefficients this constraint contributes."""
        return self._nnz

    @property
    def relation(self) -> Relation:
        """Return the comparison this constraint was declared from."""
        return Relation(self.expression, self.sense, self.rhs)

    def write_bounds(
        self, lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64]
    ) -> None:
        """Write this constraint's rows of `lower` and `upper` in place.

        The vectors are the model's. No vector is allocated per constraint.
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
        """Return the block over `(ROW, COLUMN)`, its entries written to `buffer`.

        The block is grouped into the slice the buffer reserves, and exists as
        no second object. The frame precedes the column dimension in canonical
        order. The grouping reads a leading prefix, and the result is canonical
        as written.
        """
        block, _ = self.expression.materialise(progress=progress)
        n = int((self.rows.positions_of(block) >= 0).sum())
        if n != self._nnz:
            raise ValueError(
                f"constraint {self.name!r} measured {self._nnz} coefficients "
                f"and built {n}; keep the parameter data unchanged between "
                f"measuring and writing"
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
    """Return a constraint's rows, its right-hand side values and its nonzeros.

    The rows are the coordinates at which every term is present, narrowed by
    the condition and by the coverage of the right-hand side. `over=` declares
    the rows instead. A recorder passed here records what each narrowing
    dropped.
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
                first = {d: displayed(v[0]) for d, v in missing.labels().items()}
                raise ValueError(
                    f"constraint {constraint.name!r} declares {rows.size} rows "
                    f"and right-hand side {rhs.name!r} misses {missing.size} "
                    f"of them, the first at {first}; give the right-hand side "
                    f"a value at every row"
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
    """Return the name of the object a row domain was given as."""
    if isinstance(given, Param):
        return given.name
    if isinstance(given, tuple):
        return ",".join(s.name for s in given)
    return ",".join(given.dims)
