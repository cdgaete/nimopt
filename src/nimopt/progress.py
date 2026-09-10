"""Progress reporting for a model build.

A report contains the name of the pass, the number of units finished, the
total where one is known, and the elapsed time. A pass with no known total
reports no total and no percentage.
"""

import sys
import time
from typing import Any, Protocol, TextIO

WIDTH = 19


class Reporter(Protocol):
    """The interface a build calls to report its progress."""

    def start(self, total: int | None, what: str) -> None: ...
    def step(self, done: int, what: str) -> None: ...
    def done(self) -> None: ...


class Progress:
    """A progress bar written to `stream` when `stream` is a terminal.

    A redirected or piped stream receives no bar and no carriage returns.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stderr if stream is None else stream
        self.total = None
        self.what = ""
        self.started = 0.0
        self.drawing = False

    def start(self, total: int | None, what: str) -> None:
        """Start a pass over `total` units of work, or over an unknown total."""
        self.total = total
        self.what = what
        self.started = time.perf_counter()
        self.drawing = bool(getattr(self.stream, "isatty", lambda: False)())
        self._draw(0, "")

    def step(self, done: int, what: str) -> None:
        """Report `done` finished units of the pass, the last of them `what`."""
        self._draw(done, what)

    def done(self) -> None:
        """End the pass and keep its last line."""
        if self.drawing:
            self.stream.write("\n")
            self.stream.flush()

    def _draw(self, done: int, what: str) -> None:
        """Write one line of the bar, replacing the line before it."""
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
    """Return the reporter `progress` selects, or None for no reporting.

    `None` and `False` select no reporter. `True` selects `Progress`. Any
    other value is returned unchanged and is called as a `Reporter`.
    """
    if progress is None or progress is False:
        return None
    if progress is True:
        return Progress()
    return progress
