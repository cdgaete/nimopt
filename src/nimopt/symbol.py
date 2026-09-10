"""A declared name over a set product, and its operators before it is read."""

from collections.abc import Callable
from typing import Any

from nimopt.sets import reading


class Symbol:
    """A name declared over dimensions, read at its sets through a bracket.

    A symbol over no dimension requires no bracket. Its arithmetic and
    comparison operators delegate to the reading at the empty tuple. A symbol
    over one or more dimensions raises TypeError from every operator until it
    is read. `kind` identifies the class of symbol. `expresses` identifies
    what a reading of it produces. The error message contains both.
    """

    kind: str | None = None
    expresses: str | None = None
    dims: tuple[str, ...]
    name: str
    __getitem__: Callable[..., Any]

    __array_ufunc__ = None
    __hash__ = object.__hash__

    def _read(self) -> Any:
        """Return this symbol read at the empty tuple.

        Raises TypeError for a symbol declared over one or more dimensions.
        """
        if self.dims:
            raise TypeError(
                f"{self.kind} {self.name!r} is over {self.dims} and expresses "
                f"no {self.expresses} until it is read; read it at its sets as "
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
    """Return a symbol over no dimension read at the empty tuple.

    Any other value is returned unchanged, including a symbol over one or
    more dimensions. Such a symbol raises from its own operator. The message
    identifies it.
    """
    if isinstance(held, Symbol) and not held.dims:
        return held[()]
    return held


def read_at_its_sets(held: Any) -> Any:
    """Return a symbol read at its sets, and any other value unchanged.

    Raises TypeError for a symbol declared over one or more dimensions.
    """
    return held._read() if isinstance(held, Symbol) else held
