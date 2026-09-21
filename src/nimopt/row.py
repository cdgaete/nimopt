"""One row of a built model, read back from the matrix it was assembled into."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from nimopt.sets import as_label, label_text, shown

if TYPE_CHECKING:
    from nimopt.model import Assembled, Model


@dataclass(frozen=True)
class RowTerm:
    """One coefficient of a row: its column, its variable and its coordinate."""

    column: int
    variable: str
    coordinate: dict[str, Any]
    coefficient: float


@dataclass(frozen=True)
class Row:
    """A row as the matrix stores it: its terms, its sense and its bounds.

    The row is read from a matrix the assembly pass writes, not from a second
    walk of the expression.
    """

    constraint: str
    coordinate: dict[str, Any]
    index: int
    terms: tuple[RowTerm, ...]
    sense: str
    lower: float
    upper: float

    def __repr__(self) -> str:
        at = ", ".join(f"{d}={shown(v)}" for d, v in self.coordinate.items())
        terms = " + ".join(_term(t) for t in self.terms)
        bound = self.upper if self.sense != ">=" else self.lower
        return (
            f"{self.constraint}[{at}]  row {self.index}\n"
            f"  {terms or '0'} {self.sense} {bound:g}"
        )


def _term(term: RowTerm) -> str:
    at = ",".join(label_text(v) for v in term.coordinate.values())
    return f"{term.coefficient:g}·{term.variable}[{at}]"


def senses(lower: npt.ArrayLike, upper: npt.ArrayLike) -> npt.NDArray[np.str_]:
    """Return the sense of each pair of row bounds.

    Equal bounds are `==`, an infinite lower bound is `<=`, and an infinite
    upper bound is `>=`. Raises ValueError for a finite unequal pair.
    """
    lower = np.atleast_1d(np.asarray(lower, dtype=np.float64))
    upper = np.atleast_1d(np.asarray(upper, dtype=np.float64))
    equal = lower == upper
    below = lower == -np.inf
    above = upper == np.inf
    ranged = ~(equal | below | above)
    if ranged.any():
        first = int(np.nonzero(ranged)[0][0])
        raise ValueError(
            f"row bounds [{lower[first]}, {upper[first]}] are a range; give "
            f"equal bounds, or an infinite bound on one side"
        )
    named = np.full(lower.shape, "==", dtype="<U2")
    named[below & ~equal] = "<="
    named[above & ~equal] = ">="
    return named


def _sense_of(lower: float, upper: float) -> str:
    """Return the sense of one pair of row bounds."""
    return str(senses(lower, upper)[0])


def resolve(
    model: "Model", columns: npt.ArrayLike
) -> tuple[tuple[int, int, str, dict[str, Any]], ...]:
    """Return each column as its variable and coordinate, ordered by column.

    Each entry also contains the position the column occupied in `columns`.
    A variable owns a contiguous range of the column space from its `start`.
    A column resolves to its variable by that range, and to a coordinate
    through `Variable.labels_at`. Each label is the Python value. A datetime64
    or a timedelta64 label is the numpy scalar. Raises ValueError for a column
    no variable owns.
    """
    columns = np.asarray(columns, dtype=np.int64)
    found = []
    for variable in model.variables.values():
        span = variable._columns()
        at = np.nonzero((columns >= span.start) & (columns < span.stop))[0]
        if not at.size:
            continue
        labels = variable.labels_at(columns[at] - span.start)
        for k, position in enumerate(at):
            coordinate = {d: as_label(held[k]) for d, held in labels.items()}
            found.append(
                (int(position), int(columns[position]), variable.name, coordinate)
            )
    if len(found) != columns.size:
        resolved = {column for _, column, _, _ in found}
        stray = sorted(set(columns.tolist()) - resolved)
        raise ValueError(
            f"column {stray[0]} belongs to no variable of model "
            f"{model.name!r}; pass a column below {model.n_columns}"
        )
    return tuple(sorted(found, key=lambda held: held[1]))


def read(model: "Model", assembled: "Assembled", index: int) -> Row:
    """Return the row at `index` of an assembled model.

    `index` is the solver's own row number, the number a solver reports for a
    conflict. Raises ValueError for an index outside the assembled rows.
    """
    for name in model.constraints:
        at = assembled.row_of(name)
        if at.start <= index < at.stop:
            span = slice(assembled.indptr[index], assembled.indptr[index + 1])
            return written(
                model,
                name,
                index,
                index - at.start,
                assembled.indices[span],
                assembled.values[span],
                float(assembled.row_lower[index]),
                float(assembled.row_upper[index]),
            )
    raise ValueError(
        f"row {index} is outside the {assembled.n_rows} rows this model assembles"
    )


def written(
    model: "Model",
    name: str,
    index: int,
    position: int,
    columns: npt.NDArray[np.int32],
    values: npt.NDArray[np.float64],
    lower: float,
    upper: float,
) -> Row:
    """Return row `index` of the matrix, the row at `position` of a constraint.

    `name` identifies the constraint. `columns` and `values` are the row's
    column indices and coefficients. `lower` and `upper` are its bounds. Each
    label of the coordinate is the Python value. A datetime64 or a
    timedelta64 label is the numpy scalar.
    """
    labels = model.constraints[name].labels_at([position])
    return Row(
        constraint=name,
        coordinate={d: as_label(held[0]) for d, held in labels.items()},
        index=int(index),
        terms=tuple(
            RowTerm(column, variable, coordinate, float(values[at]))
            for at, column, variable, coordinate in resolve(model, columns)
        ),
        sense=_sense_of(lower, upper),
        lower=lower,
        upper=upper,
    )
