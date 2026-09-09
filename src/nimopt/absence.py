"""Which coordinates fell out of a constraint, and by which rule."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nimblend import Domain, SparseArray

from nimopt.names import COLUMN


@dataclass(frozen=True)
class DroppedRow:
    """A row a constraint did not state, and what removed it."""

    coordinate: dict[str, Any]
    rule: str
    detail: str


@dataclass(frozen=True)
class DroppedTerm:
    """A term missing from a row that stands, and what removed it."""

    coordinate: dict[str, Any]
    variable: str
    rule: str
    detail: str


@dataclass(frozen=True)
class Absence:
    """What a constraint set out to state, what it states, and what fell.

    `stated_by` is `"terms"` where the rows are derived from what the terms
    reach and `"over"` where they are stated outright. Under `"over"` nothing
    is dropped and an empty `dropped_rows` is structural rather than a
    constraint that happened to lose nothing.

    A coefficient absent inside a sum removes a term and leaves the row
    standing; a term absent along a free dimension removes the row, because a
    row missing one of its terms states something that was not written.
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
    """One dict of dimension to label per member of a domain.

    A label is handed over as the Python value it stands for, so a caller
    comparing one against a literal, and a rendering of it, both read plainly.
    """
    labels = domain.labels()
    return [{d: labels[d][k].item() for d in domain.dims} for k in range(domain.size)]


def _absent_terms(
    rows: Domain, term: Any, before: SparseArray, after: SparseArray
) -> list[DroppedTerm]:
    """Terms a coefficient removed from rows that still stand.

    A term is identified by the coordinate it carries once its coefficient has
    placed it. Every such coordinate is the entries the variable brought,
    crossed with each member of a dimension the coefficient introduced; the
    ones the coefficient does not carry are the terms that are gone.
    """
    if not term.summed:
        return []
    keep = tuple(d for d in after.dims if d != COLUMN)
    carried = after.domain(keep)
    brought = tuple(d for d in before.dims if d != COLUMN)
    entries = before.domain(brought)
    introduced = tuple(d for d in keep if d not in brought)
    whole = entries.expand(introduced, carried.coords).transpose(*keep)
    lost = whole.difference(carried)
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
    """What a constraint's shape pass dropped, gathered as it narrows.

    A dropped row leaves no trace in the matrix, so it is attributed while the
    narrowing runs rather than read back from it. Each narrowing sees only what
    survived the one before, so a coordinate is attributed to one rule.
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
        """The rows one term of the expression reaches."""
        self._reached.append((term, domain))

    def reached(self, frame: Sequence[str], rows: Domain) -> None:
        """The rows the terms reach, against the product the frame spans.

        A coordinate no term reaches is attributed to the first term that
        misses it, so each is named once and by the term that lost it.
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
        """The rows an `over=` states outright, which drops nothing."""
        self.stated_by = "over"
        self.expected = rows.size
        self.rows = []

    def dropped(self, before: Domain, after: Domain, rule: str, detail: str) -> None:
        """The rows one narrowing removed."""
        self._record_rows(before, after, rule, detail)

    def _record_rows(
        self, before: Domain, after: Domain, rule: str, detail: str
    ) -> None:
        lost = before.difference(after)
        self.rows.extend(DroppedRow(at, rule, detail) for at in _coordinates(lost))

    def coefficient(self, term: Any, before: SparseArray, after: SparseArray) -> None:
        """A term's entries either side of the coefficient that multiplied it."""
        self._coefficients.append((term, before, after))

    def settled(self, rows: Domain) -> None:
        """The rows the constraint states, once every narrowing has run."""
        self._standing = rows
        for held in self._coefficients:
            self.terms.extend(_absent_terms(rows, *held))

    def absence(self, name: str) -> Absence:
        """What this recorder gathered, as the answer a caller reads."""
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
