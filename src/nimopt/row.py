"""One row of a built model, read back from the matrix it was assembled into."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from nimopt.constraint import Constraint
    from nimopt.model import Assembled, Model


@dataclass(frozen=True)
class RowTerm:
    """One coefficient of a row: its column, and what that column stands for."""

    column: int
    variable: str
    coordinate: dict[str, Any]
    coefficient: float


@dataclass(frozen=True)
class Row:
    """A row as the matrix carries it: its terms, its sense and its bounds.

    Read from the assembled matrix rather than from a second walk of the
    expression, so what is shown is what the solver is given.
    """

    constraint: str
    coordinate: dict[str, Any]
    index: int
    terms: tuple[RowTerm, ...]
    sense: str
    lower: float
    upper: float

    def __repr__(self) -> str:
        at = ", ".join(f"{d}={v!r}" for d, v in self.coordinate.items())
        terms = " + ".join(_term(t) for t in self.terms)
        bound = self.upper if self.sense != ">=" else self.lower
        return (
            f"{self.constraint}[{at}]  row {self.index}\n"
            f"  {terms or '0'} {self.sense} {bound:g}"
        )


def _term(term: RowTerm) -> str:
    at = ",".join(str(v) for v in term.coordinate.values())
    return f"{term.coefficient:g}·{term.variable}[{at}]"


def senses(lower: npt.ArrayLike, upper: npt.ArrayLike) -> npt.NDArray[np.str_]:
    """The sense each pair of row bounds states.

    Equal bounds are `==`, an infinite lower is `<=`, an infinite upper is
    `>=`. `Constraint.write_bounds` writes one of those three, so a finite
    unequal pair reaches no row and is refused rather than named: a rule
    nothing produces is vocabulary a caller branches on and never reaches.
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
            f"row bounds [{lower[first]}, {upper[first]}] state a range, and "
            f"no constraint states a range; a row carries one sense"
        )
    named = np.full(lower.shape, "==", dtype="<U2")
    named[below & ~equal] = "<="
    named[above & ~equal] = ">="
    return named


def _sense_of(lower: float, upper: float) -> str:
    """The sense one pair of row bounds states."""
    return str(senses(lower, upper)[0])


def _coordinate(domain: Any, position: int) -> dict[str, Any]:
    """The label of each dimension at one member of a domain.

    A label is handed over as the Python value it stands for, so a caller
    comparing one against a literal, and a rendering of it, both read plainly.
    """
    labels = domain.labels()
    return {d: labels[d][position].item() for d in domain.dims}


def resolve(
    model: "Model", columns: npt.ArrayLike
) -> tuple[tuple[int, int, str, dict[str, Any]], ...]:
    """Each column as the variable and the coordinate it stands for.

    Ordered by column, and carrying the position each stood at in `columns`,
    so a caller holding a value per column follows it back. A variable owns a
    contiguous range of the column space from its `start`, so a column
    resolves to its variable by that range and to a coordinate through the
    variable's own numbering rule. A column no variable carries is refused
    rather than dropped.
    """
    columns = np.asarray(columns, dtype=np.int64)
    found = []
    for variable in model.variables.values():
        end = variable.start + variable.n_columns
        at = np.nonzero((columns >= variable.start) & (columns < end))[0]
        if not at.size:
            continue
        index = variable.coord.to_index(columns[at])
        for k, position in enumerate(at):
            coordinate = {
                d: s.coord.to_index(np.asarray([index[j][k]]))[0].item()
                for j, (d, s) in enumerate(zip(variable.dims, variable.sets))
            }
            found.append(
                (int(position), int(columns[position]), variable.name, coordinate)
            )
    if len(found) != columns.size:
        carried = {column for _, column, _, _ in found}
        stray = sorted(set(columns.tolist()) - carried)
        raise ValueError(
            f"column {stray[0]} belongs to no variable of model "
            f"{model.name!r}, which numbers {model.n_columns} columns"
        )
    return tuple(sorted(found, key=lambda held: held[1]))


def read(model: "Model", assembled: "Assembled", index: int) -> Row:
    """The row at `index` of an assembled model, as the matrix carries it.

    `index` is the solver's own row number, which is what a backend reporting
    a conflict hands back.
    """
    for name, constraint in model.constraints.items():
        at = assembled.row_of(name)
        if at.start <= index < at.stop:
            span = slice(assembled.indptr[index], assembled.indptr[index + 1])
            lower = float(assembled.row_lower[index])
            upper = float(assembled.row_upper[index])
            values = assembled.values[span]
            return Row(
                constraint=name,
                coordinate=_coordinate(constraint.rows, index - at.start),
                index=int(index),
                terms=tuple(
                    RowTerm(column, variable, coordinate, float(values[position]))
                    for position, column, variable, coordinate in resolve(
                        model, assembled.indices[span]
                    )
                ),
                sense=_sense_of(lower, upper),
                lower=lower,
                upper=upper,
            )
    raise ValueError(
        f"row {index} is outside the {assembled.n_rows} rows this model assembles"
    )


def position_of(constraint: "Constraint", coords: Mapping[str, Any]) -> int:
    """Where in a constraint's rows the named coordinate stands.

    Each label resolves through its dimension's own coordinate, and the row
    domain answers where the member it names sits.
    """
    rows = constraint.rows
    if tuple(coords) != rows.dims:
        raise ValueError(
            f"constraint {constraint.name!r} is free over {rows.dims}; got "
            f"{tuple(coords)}"
        )
    index = np.stack(
        [
            rows.coords[d].to_position(np.asarray([coords[d]])).astype(np.int32)
            for d in rows.dims
        ]
    )
    at = int(rows.positions_of_coordinates(index)[0])
    if at < 0:
        raise ValueError(
            f"constraint {constraint.name!r} states no row at {dict(coords)}; "
            f"`absent({constraint.name!r})` names the rule that dropped it"
        )
    return at
