"""A constraint: an expression, a sense and a right-hand side."""

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from nimblend import Domain, EntryBuffer, SparseArray

from nimopt.coefficient import Coefficient
from nimopt.names import COLUMN, ROW
from nimopt.sets import (
    as_member,
    check_labels,
    condition_coords,
    condition_dims,
    condition_name,
    coord_dtype,
    displayed,
    rows_of,
    shown,
)
from nimopt.term import Expression, Relation

SENSES = ("<=", ">=", "==")

SINGLETON_LIMIT = 1e20
SINGLETON_COEFFICIENT_LOW = 1e-9
SINGLETON_COEFFICIENT_HIGH = 1e15


def _convertible(coefficients: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Return one flag per coefficient, True where one can become a column bound.

    HiGHS treats a coefficient of 1e-9 or less in magnitude as zero, and it
    rejects one of 1e15 or more. A NaN is outside the range.
    """
    magnitude = np.abs(coefficients)
    return (magnitude > SINGLETON_COEFFICIENT_LOW) & (
        magnitude < SINGLETON_COEFFICIENT_HIGH
    )


def _integer_variable_columns(expression: Expression) -> tuple[slice, ...]:
    """Return the slice of column indices of each integer variable of the terms."""
    return tuple(
        term.variable._columns() for term in expression.terms if term.variable.integer
    )


def _on_integer_columns(
    columns: npt.NDArray[np.integer], held: tuple[slice, ...]
) -> npt.NDArray[np.bool_]:
    """Return one flag per column, True where the column is of an integer variable."""
    inside = np.zeros(columns.size, dtype=bool)
    for at in held:
        inside |= (columns >= at.start) & (columns < at.stop)
    return inside


def _singleton_rows(
    coefficients: npt.NDArray[np.float64],
    bound: npt.NDArray[np.float64] | float,
    integer: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.bool_]:
    """Return one flag per row, True where the row is a singleton row.

    A singleton row has a coefficient whose magnitude is above
    `SINGLETON_COEFFICIENT_LOW` and below `SINGLETON_COEFFICIENT_HIGH`, a
    finite right-hand side, a magnitude of the right-hand side divided by the
    coefficient below `SINGLETON_LIMIT` and, where `integer` marks the row, a
    quotient that is an integer. `bound` is one value per row or one value for
    every row. `integer` is one flag per row, or None where no row is on a
    column of an integer variable.
    """
    divisor = np.where(coefficients == 0.0, 1.0, coefficients)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        quotient = bound / divisor
    usable = (
        _convertible(coefficients)
        & np.isfinite(bound)
        & (np.abs(quotient) < SINGLETON_LIMIT)
    )
    if integer is not None:
        usable &= (~integer) | (quotient == np.rint(quotient))
    return usable


def check_relation(name: str, relation: Any) -> None:
    """Raise TypeError for a relation that is not a comparison of an expression."""
    if not isinstance(relation, Relation):
        raise TypeError(
            f"constraint {name!r} takes a comparison of an expression, such as "
            f"`expr <= rhs`; got {type(relation).__name__}"
        )


def _write_row_bounds(
    sense: str,
    values: npt.NDArray[np.float64] | float,
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> None:
    """Write the bounds of rows of one sense into `lower` and `upper`.

    `values` contains one right-hand side per row or one number for every row.
    """
    if sense == "<=":
        lower[:] = -np.inf
        upper[:] = values
    elif sense == ">=":
        lower[:] = values
        upper[:] = np.inf
    else:
        lower[:] = values
        upper[:] = values


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
        check_relation(self.name, relation)
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
        check_labels(f"constraint {self.name!r}", self._coordinates())
        self.rows, self._rhs_values, self._nnz, self._singletons = narrow(self)

    def _coordinates(self) -> Iterator[tuple[str, Any]]:
        """Yield the dimension and the coordinate of each set this constraint reads.

        The constraint reads the sets of its expression, the dimensions of
        `over=` or `where=`, and the sets of each parameter of its right-hand
        side.
        """
        owner = f"constraint {self.name!r}"
        yield from self.expression._coordinates()
        for given, what in ((self.over, "over="), (self.where, "condition")):
            if given is not None:
                yield from condition_coords(given, owner, what).items()
        if isinstance(self.rhs, Coefficient):
            for parameter in self.rhs.parameters():
                for held in parameter.sets:
                    yield held.name, held.coord

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
    def n_singleton_rows(self) -> int:
        """Return the number of singleton rows of this constraint."""
        marked = self._singletons
        if marked is None:
            return 0
        if marked is True:
            return self.rows.size
        return int(np.count_nonzero(marked))

    def singleton_positions(self) -> npt.NDArray[np.int64]:
        """Return the positions of the singleton rows, ascending."""
        return np.flatnonzero(self._marked()).astype(np.int64)

    def kept_positions(self) -> npt.NDArray[np.int64]:
        """Return the positions of the rows that are not singleton rows, ascending."""
        return np.flatnonzero(~self._marked()).astype(np.int64)

    def singleton_rhs(self) -> npt.NDArray[np.float64]:
        """Return the right-hand side of each singleton row.

        The values are in the order of the rows.
        """
        return self._rhs_at(self.singleton_positions())

    def _marked(self) -> npt.NDArray[np.bool_]:
        """Return one flag per row, True at a singleton row."""
        marked = self._singletons
        if marked is None:
            return np.zeros(self.rows.size, dtype=bool)
        if isinstance(marked, bool):
            return np.full(self.rows.size, marked)
        return marked

    def _rhs_at(self, positions: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Return the right-hand side at `positions`, in the order given."""
        held = self._rhs_values
        if isinstance(held, np.ndarray):
            return held[positions].astype(np.float64, copy=True)
        return np.full(positions.size, float(held))

    @property
    def relation(self) -> Relation:
        """Return the comparison this constraint was declared from."""
        return Relation(self.expression, self.sense, self.rhs)

    def labels_at(self, positions: npt.ArrayLike) -> dict[str, npt.NDArray[Any]]:
        """Return the labels of the rows at `positions`, per free dimension.

        A position is the index of a row among this constraint's rows, from 0.
        The labels are in the order of `positions`. Raises ValueError for a
        position that is not an integer or is outside the rows.
        """
        return self.rows.labels(positions)

    def position_of(self, coords: Mapping[str, Any]) -> int:
        """Return the position of the row at a coordinate.

        `coords` maps each free dimension to one label. Each label is converted
        to the dtype of its dimension's members. A constraint with no free
        dimension has its one row at an empty `coords`. Raises ValueError for
        other dimensions, for a label that is not a member of its dimension,
        for a label that does not convert and for a coordinate with no row.
        """
        rows = self.rows
        if tuple(coords) != rows.dims:
            raise ValueError(
                f"constraint {self.name!r} is free over {rows.dims}; got "
                f"{tuple(coords)}"
            )
        positions = []
        for d in rows.dims:
            dtype = coord_dtype(rows.coords[d])
            where = f"dimension {d!r} of constraint {self.name!r}"
            label = coords[d] if dtype is None else as_member(coords[d], dtype, where)
            try:
                at = rows.coords[d].to_position(np.asarray([label]))
            except KeyError:
                raise ValueError(
                    f"member {shown(coords[d])} is not in dimension {d!r} of "
                    f"constraint {self.name!r}; pass a member of {d!r}"
                ) from None
            positions.append(at.astype(np.int32))
        index = np.asarray(positions, dtype=np.int32).reshape(len(rows.dims), 1)
        at = int(rows.positions_of_coordinates(index)[0])
        if at < 0:
            raise ValueError(
                f"constraint {self.name!r} has no row at {dict(coords)}; "
                f"read `absent({self.name!r})` for the rule that dropped it"
            )
        return at

    def row_at(
        self, position: int
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], float, float]:
        """Return the columns, coefficients and bounds of the row at `position`.

        The columns ascend. The coefficients are the entries `write_into`
        writes for that row. Raises ValueError for a position outside the
        rows.
        """
        labels = self.labels_at([position])
        block = self.expression.materialise_at(
            {d: held[0] for d, held in labels.items()}
        )
        held = self._rhs_values
        value = float(held[position] if isinstance(held, np.ndarray) else held)
        lower = value if self.sense in (">=", "==") else -np.inf
        upper = value if self.sense in ("<=", "==") else np.inf
        return block.coordinates()[0], block.values(), lower, upper

    def write_bounds(
        self, lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64]
    ) -> None:
        """Write this constraint's rows of `lower` and `upper` in place.

        The vectors are the model's. No vector is allocated per constraint.
        """
        _write_row_bounds(self.sense, self._rhs_values, lower, upper)

    def write_kept_bounds(
        self, lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64]
    ) -> None:
        """Write the bounds of the rows that are not singleton rows in place.

        The vectors contain one entry per kept row. The bound of a row is that
        of `write_bounds`.
        """
        _write_row_bounds(self.sense, self._rhs_at(self.kept_positions()), lower, upper)

    def write_kept(
        self, buffer: EntryBuffer, rows: Any, row_start: int, progress: Any = None
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
        """Write the entries of the rows that are not singleton rows to `buffer`.

        `rows` is the coordinate of every loaded row of the model, and this
        constraint's kept rows are numbered from `row_start` inside it. Returns
        the column and the coefficient of each singleton row, in the order of
        the rows. The expression is materialised once. Raises ValueError where
        a marked row fails the definition of a singleton row: it has a number
        of coefficients other than one, the magnitude of its coefficient is
        1e-9 or less or 1e15 or more, its right-hand side is not finite, the
        magnitude of the right-hand side divided by the coefficient is 1e20 or
        more, or its column is a column of an integer variable and the quotient
        is not an integer. Raises ValueError where the kept rows have a number
        of coefficients different from the number measured at declaration.
        Nothing is written then.
        """
        narrowed = self.over is not None or self.where is not None
        within = self.rows if narrowed else None
        block = self.expression.block(progress=progress, within=within)
        marked = self.rows.take(self.singleton_positions())
        kept = self.rows.take(self.kept_positions())
        held = block.restrict(marked)
        columns = held.coordinates((COLUMN,))[0]
        coefficients = held.values()
        at = marked.positions_of(held)
        count = np.bincount(at, minlength=marked.size)
        usable = count == 1
        value = np.zeros(marked.size, dtype=np.float64)
        value[at] = coefficients
        stored = self._rhs_values
        bound = self.singleton_rhs() if isinstance(stored, np.ndarray) else stored
        integer_columns = _integer_variable_columns(self.expression)
        integer: npt.NDArray[np.bool_] | None = None
        if integer_columns:
            column = np.zeros(marked.size, dtype=np.int64)
            column[at] = columns
            integer = _on_integer_columns(column, integer_columns)
        usable &= _singleton_rows(value, bound, integer)
        found = int(np.count_nonzero(usable))
        if found != marked.size:
            raise ValueError(
                f"constraint {self.name!r} marked {marked.size} singleton rows "
                f"and built {found}; keep the parameter data unchanged between "
                f"measuring and writing"
            )
        if kept.size:
            expected = self._nnz - marked.size

            def reserve(n: int) -> Any:
                if n != expected:
                    raise ValueError(
                        f"constraint {self.name!r} measured {self._nnz} "
                        f"coefficients and built {n + marked.size}; keep the "
                        f"parameter data unchanged between measuring and writing"
                    )
                return buffer.reserve(n)

            block.group(
                self.frame,
                into=ROW,
                domain=kept,
                coord=rows,
                start=row_start,
                reserve=reserve,
            )
        return columns, coefficients

    def write_into(
        self, buffer: EntryBuffer, rows: Any, row_start: int, progress: Any = None
    ) -> SparseArray:
        """Return the block over `(ROW, COLUMN)`, its entries written to `buffer`.

        `rows` is the coordinate of every row of the model, and this
        constraint's rows are numbered from `row_start` inside it. The block
        is grouped into the slice the buffer reserves, and exists as no second
        object. The frame precedes the column dimension in canonical order.
        The grouping reads a leading prefix, and the result is canonical as
        written. Raises ValueError where the coefficients differ in number
        from those measured at declaration, and writes nothing then.
        """
        narrowed = self.over is not None or self.where is not None
        within = self.rows if narrowed else None
        block = self.expression.block(progress=progress, within=within)

        def reserve(n: int) -> Any:
            if n != self._nnz:
                raise ValueError(
                    f"constraint {self.name!r} measured {self._nnz} coefficients "
                    f"and built {n}; keep the parameter data unchanged between "
                    f"measuring and writing"
                )
            return buffer.reserve(n)

        return block.group(
            self.frame,
            into=ROW,
            domain=self.rows,
            coord=rows,
            start=row_start,
            reserve=reserve,
        )


def narrow(
    constraint: "Constraint", record: Any = None
) -> tuple[
    Domain, npt.NDArray[np.float64] | float, int, bool | npt.NDArray[np.bool_] | None
]:
    """Return a constraint's rows, right-hand side values, nonzeros and marking.

    The rows are the coordinates at which every term is present, narrowed by
    the condition and by the coverage of the right-hand side. `over=` declares
    the rows instead. The right-hand side is one value per row for a
    coefficient, and one float for a number. The marking is `None` where no
    row is a singleton row and where a recorder is passed, `True` where every
    row is one, and one flag per row otherwise. Without a recorder, the
    expression is materialised within the declared rows or the condition. A
    recorder passed here records what each narrowing dropped, over the whole
    expression.
    """
    frame = constraint.frame
    owner = f"constraint {constraint.name!r}"
    declared = condition = None
    if constraint.over is not None:
        declared = rows_of(constraint.over, frame, owner, "over=")
    elif constraint.where is not None:
        condition = rows_of(constraint.where, frame, owner, "condition")
    within = declared if declared is not None else condition
    if record is not None:
        within = None
    if declared is not None and record is None:
        block, rows = constraint.expression.block(within=declared), declared
    else:
        block, rows = constraint.expression.materialise(record, within=within)
    if record is not None:
        record.reached(frame, rows)
    if declared is not None:
        rows = declared
        if record is not None:
            record.stated(rows)
    elif condition is not None:
        narrowed = rows.intersect(condition)
        if record is not None:
            record.dropped(
                rows,
                narrowed,
                "where",
                _named(constraint.where, f"constraint {constraint.name!r}"),
            )
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
        rhs_values = float(rhs)
    else:
        raise TypeError(
            "a right-hand side is a number or a coefficient read at the "
            f"constraint's free dimensions; got {type(rhs).__name__}"
        )
    rhs_values = rhs_values - constraint.expression.constant
    if record is not None:
        record.settled(rows)
    at = rows.positions_of(block)
    marking = (
        None
        if record is not None
        else _marking(block, rows, at, rhs_values, constraint.expression)
    )
    return rows, rhs_values, int((at >= 0).sum()), marking


def _marking(
    block: SparseArray,
    rows: Domain,
    at: npt.NDArray[np.integer],
    rhs_values: npt.NDArray[np.float64] | float,
    expression: Expression,
) -> bool | npt.NDArray[np.bool_] | None:
    """Return the singleton marking of a block over `rows`.

    The result is None where no row of `rows` is a singleton row, True where
    every row is one, and one flag per row otherwise. A singleton row has one
    coefficient inside `rows` whose magnitude is above
    `SINGLETON_COEFFICIENT_LOW` and below `SINGLETON_COEFFICIENT_HIGH`, a
    finite right-hand side, a magnitude of the right-hand side divided by the
    coefficient below `SINGLETON_LIMIT` and, where its column is a column of an
    integer variable of `expression`, a quotient that is an integer. Where no
    term of `expression` has an integer variable, no column coordinate is
    read. The entries of `at` are the positions of the entries of the block in
    `rows`, and the entries ascend by row.
    """
    keep = at >= 0
    every = keep.all()
    inside = at if every else at[keep]
    marked = np.bincount(inside, minlength=rows.size) == 1
    if not marked.any():
        return None
    positions = np.flatnonzero(marked)
    entries = block.values()
    if not every:
        entries = entries[keep]
    coefficients = entries[marked[inside]]
    held = rhs_values
    bound = held[positions] if isinstance(held, np.ndarray) else float(held)
    integer_columns = _integer_variable_columns(expression)
    integer: npt.NDArray[np.bool_] | None = None
    if integer_columns:
        columns = block.coordinates((COLUMN,))[0]
        if not every:
            columns = columns[keep]
        integer = _on_integer_columns(columns[marked[inside]], integer_columns)
    marked[positions] = _singleton_rows(coefficients, bound, integer)
    if marked.all():
        return True
    if not marked.any():
        return None
    return marked


def _named(given: Any, owner: str) -> str:
    """Return the name of the object a row domain was given as."""
    name = condition_name(given, owner, "condition")
    if isinstance(name, str):
        return name
    return ",".join(condition_dims(given, owner, "condition") if name is None else name)
