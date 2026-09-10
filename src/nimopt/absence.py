"""The coordinates a constraint dropped, and the rule that dropped each."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nimblend import Domain, SparseArray

from nimopt.names import COLUMN


@dataclass(frozen=True)
class DroppedRow:
    """A row a constraint does not contain, and the rule that removed it."""

    coordinate: dict[str, Any]
    rule: str
    detail: str


@dataclass(frozen=True)
class DroppedTerm:
    """A term missing from a row that remains, and the rule that removed it."""

    coordinate: dict[str, Any]
    variable: str
    rule: str
    detail: str


@dataclass(frozen=True)
class Absence:
    """The rows a constraint expected, the rows it has, and the rows dropped.

    `stated_by` is `"terms"` where the rows are derived from the terms, and
    `"over"` where they are declared with `over=`. Under `"over"` nothing is
    dropped and `dropped_rows` is empty. A coefficient absent inside a sum
    removes a term and keeps the row. A term absent along a free dimension
    removes the row.
    """

    constraint: str
    stated_by: str
    expected: int
    standing: int
    dropped_rows: tuple[DroppedRow, ...]
    dropped_terms: tuple[DroppedTerm, ...]

    def __repr__(self) -> str:
        lines = [
            f"{self.constraint}  {self.standing} of {self.expected} rows  "
            f"stated by {self.stated_by}"
        ]
        for dropped in self.dropped_rows:
            lines.append(
                f"  row absent  {_at(dropped.coordinate)}  "
                f"{dropped.rule} ({dropped.detail})"
            )
        for dropped in self.dropped_terms:
            lines.append(
                f"  term absent {_at(dropped.coordinate)}  {dropped.variable}  "
                f"{dropped.rule} ({dropped.detail})"
            )
        return "\n".join(lines)


def _at(coordinate: Mapping[str, Any]) -> str:
    return ", ".join(f"{d}={v!r}" for d, v in coordinate.items())


def _coordinates(domain: Domain) -> list[dict[str, Any]]:
    """Return one dict of dimension to label per member of a domain.

    Each label is the Python value, not a numpy scalar.
    """
    labels = domain.labels()
    return [{d: labels[d][k].item() for d in domain.dims} for k in range(domain.size)]


def _absent_terms(
    rows: Domain, term: Any, before: SparseArray, after: SparseArray
) -> list[DroppedTerm]:
    """Return the terms a coefficient removed from rows that remain.

    A term is identified by its coordinate after the coefficient multiplies.
    The full set of such coordinates is the variable's entries crossed with
    each member of a dimension the coefficient introduces. The coordinates
    absent from the product are the removed terms.
    """
    if not term.summed:
        return []
    keep = tuple(d for d in after.dims if d != COLUMN)
    placed = after.domain(keep)
    brought = tuple(d for d in before.dims if d != COLUMN)
    entries = before.domain(brought)
    introduced = tuple(d for d in keep if d not in brought)
    whole = entries.expand(introduced, placed.coords).transpose(*keep)
    lost = whole.difference(placed)
    at = lost.coordinates()[[keep.index(d) for d in rows.dims]]
    standing = rows.positions_of_coordinates(at) >= 0
    variable = term.variable.name
    detail = term.coefficient.name
    return [
        DroppedTerm(coordinate, variable, "absent-coefficient", detail)
        for coordinate, keeping in zip(_coordinates(lost), standing)
        if keeping
    ]


class Recorder:
    """The rows and terms a constraint's shape pass drops, gathered as it runs.

    A dropped row is not present in the matrix. It is attributed while the
    narrowing runs. Each narrowing reads only what the one before it kept,
    and each coordinate is attributed to one rule.
    """

    def __init__(self) -> None:
        self.stated_by = "terms"
        self.expected: int | None = None
        self.rows = []
        self.terms = []
        self._reached = []
        self._coefficients = []
        self._standing: Domain | None = None

    def term_rows(self, term: Any, domain: Domain) -> None:
        """Record the rows one term of the expression spans."""
        self._reached.append((term, domain))

    def reached(self, frame: Sequence[str], rows: Domain) -> None:
        """Record the rows the terms span, against the product of the frame.

        A coordinate absent from every term is attributed to the first term
        that lacks it.
        """
        whole = Domain.full(frame, rows.coords)
        self.expected = whole.size
        standing = whole
        for term, domain in self._reached:
            reduced = standing.intersect(domain)
            self._record_rows(
                standing, reduced, "term-does-not-reach", term.variable.name
            )
            standing = reduced

    def stated(self, rows: Domain) -> None:
        """Record the rows `over=` declares, dropping none."""
        self.stated_by = "over"
        self.expected = rows.size
        self.rows = []

    def dropped(self, before: Domain, after: Domain, rule: str, detail: str) -> None:
        """Record the rows one narrowing removed."""
        self._record_rows(before, after, rule, detail)

    def _record_rows(
        self, before: Domain, after: Domain, rule: str, detail: str
    ) -> None:
        lost = before.difference(after)
        self.rows.extend(DroppedRow(at, rule, detail) for at in _coordinates(lost))

    def coefficient(self, term: Any, before: SparseArray, after: SparseArray) -> None:
        """Record a term's entries before and after its coefficient multiplies."""
        self._coefficients.append((term, before, after))

    def settled(self, rows: Domain) -> None:
        """Record the constraint's rows, once every narrowing has run."""
        self._standing = rows
        for held in self._coefficients:
            self.terms.extend(_absent_terms(rows, *held))

    def absence(self, name: str) -> Absence:
        """Return what this recorder gathered, as an `Absence`.

        Raises ValueError before the shape pass has run.
        """
        if self.expected is None or self._standing is None:
            raise ValueError(
                f"constraint {name!r} has not settled its rows; an absence is "
                f"read once the shape pass has run"
            )
        return Absence(
            constraint=name,
            stated_by=self.stated_by,
            expected=self.expected,
            standing=self._standing.size,
            dropped_rows=tuple(self.rows),
            dropped_terms=tuple(self.terms),
        )
