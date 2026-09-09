"""A declared name over a set product, and what it answers before it is read."""

from collections.abc import Callable
from typing import Any

from nimopt.sets import reading


class Symbol:
    """A name declared over dimensions, whose bracket lists the ones it carries.

    A symbol over no dimension carries no bracket: it is already read, and
    answers arithmetic and comparison by delegating to that reading. One over
    dimensions states nothing until it is read, and every operator refuses
    alike, naming the reading it wants.

    `kind` and `states` are what the refusal calls this symbol and what it
    would state once read, so one sentence serves both a parameter and a
    variable in whatever slot they stand in.
    """

    kind: str | None = None
    states: str | None = None
    dims: tuple[str, ...]
    name: str
    __getitem__: Callable[..., Any]

    __array_ufunc__ = None
    __hash__ = object.__hash__

    def _read(self) -> Any:
        """This symbol as the reading it is, where it carries no dimension."""
        if self.dims:
            raise TypeError(
                f"{self.kind} {self.name!r} carries {self.dims} and states no "
                f"{self.states} until it is read; read it at its sets as "
                f"{reading(self.name, self.dims)}"
            )
        return self[()]

    def __add__(self, other: Any) -> Any:
        return self._read().__add__(other)

    def __radd__(self, other: Any) -> Any:
        return self._read().__radd__(other)

    def __sub__(self, other: Any) -> Any:
        return self._read().__sub__(other)

    def __rsub__(self, other: Any) -> Any:
        return self._read().__rsub__(other)

    def __mul__(self, other: Any) -> Any:
        return self._read().__mul__(other)

    def __rmul__(self, other: Any) -> Any:
        return self._read().__rmul__(other)

    def __truediv__(self, other: Any) -> Any:
        return self._read().__truediv__(other)

    def __rtruediv__(self, other: Any) -> Any:
        return self._read().__rtruediv__(other)

    def __pow__(self, other: Any) -> Any:
        return self._read().__pow__(other)

    def __rpow__(self, other: Any) -> Any:
        return self._read().__rpow__(other)

    def __neg__(self) -> Any:
        return self._read().__neg__()

    def __abs__(self) -> Any:
        return self._read().__abs__()

    def __le__(self, other: Any) -> Any:
        return self._read().__le__(other)

    def __ge__(self, other: Any) -> Any:
        return self._read().__ge__(other)

    def __lt__(self, other: Any) -> Any:
        return self._read().__lt__(other)

    def __gt__(self, other: Any) -> Any:
        return self._read().__gt__(other)

    def __eq__(self, other: Any) -> Any:
        return self._read().__eq__(other)

    def __ne__(self, other: Any) -> Any:
        return self._read().__ne__(other)


def read_bare(held: Any) -> Any:
    """A symbol over no dimension, read; anything else as it was given.

    A symbol over dimensions is left alone so that its own operator states
    which reading it wants, rather than a message from here naming neither.
    """
    if isinstance(held, Symbol) and not held.dims:
        return held[()]
    return held


def read_at_its_sets(held: Any) -> Any:
    """A symbol read at its sets, refusing one whose dimensions are unread."""
    return held._read() if isinstance(held, Symbol) else held
