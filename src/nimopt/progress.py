"""What a caller is told while a model is built.

A report states what it knows: the pass it is in, the work it has done
against the total where a total exists, and how long it has taken. A pass
that cannot state a total states none rather than a fraction of a number it
guessed.
"""

import sys
import time
from typing import Any, Protocol, TextIO

WIDTH = 19


class Reporter(Protocol):
    """What a build calls to say where it has got to."""

    def start(self, total: int | None, what: str) -> None: ...
    def step(self, done: int, what: str) -> None: ...
    def done(self) -> None: ...


class Progress:
    """A report drawn where a terminal reads it, and nowhere else.

    A redirected or piped run carries no bar and no carriage returns, so the
    text a caller keeps is the text it meant to keep.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stderr if stream is None else stream
        self.total = None
        self.what = ""
        self.started = 0.0
        self.drawing = False

    def start(self, total: int | None, what: str) -> None:
        """Begin a pass over `total` units of work, or over an unknown amount."""
        self.total = total
        self.what = what
        self.started = time.perf_counter()
        self.drawing = bool(getattr(self.stream, "isatty", lambda: False)())
        self._draw(0, "")

    def step(self, done: int, what: str) -> None:
        """`done` units of the pass are finished, the last of them `what`."""
        self._draw(done, what)

    def done(self) -> None:
        """End the pass, leaving its last line in place."""
        if self.drawing:
            self.stream.write("\n")
            self.stream.flush()

    def _draw(self, done: int, what: str) -> None:
        """One line, rewritten in place."""
        if not self.drawing:
            return
        elapsed = time.perf_counter() - self.started
        if self.total:
            share = min(max(done / self.total, 0.0), 1.0)
            filled = int(share * WIDTH)
            bar = "█" * filled + "░" * (WIDTH - filled)
            head = f"{self.what}  [{bar}]  {share * 100:3.0f}%"
            tail = f"  {what}  {done:,} / {self.total:,}  {elapsed:.1f}s"
        else:
            head = f"{self.what}"
            tail = f"  {what}  {done:,}  {elapsed:.1f}s"
        self.stream.write(f"\r\033[K{head}{tail}")
        self.stream.flush()


def reporter(progress: Any) -> Reporter | None:
    """The reporter `progress` asks for, or None where it asks for none.

    `False` reports nothing, `True` reports through the built-in one, and a
    reporter of the caller's own is taken as it is, so a caller with a log or
    an interface of its own writes there.
    """
    if progress is None or progress is False:
        return None
    if progress is True:
        return Progress()
    return progress
