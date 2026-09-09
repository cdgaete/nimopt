"""Attribute every allocation of a build to the package whose frame made it."""

import tracemalloc
from pathlib import Path

import nimblend as nb
import numpy

import nimopt as no

_ROOTS = (
    ("nimopt", str(Path(no.__file__).parent)),
    ("nimblend", str(Path(nb.__file__).parent)),
)
_NUMPY = str(Path(numpy.__file__).parent)


def _owner(traceback):
    """The package that ordered the allocation.

    A traceback runs oldest frame first, so a package's innermost frame is
    found by walking it in reverse, and a package is recognised by its own
    source root rather than by a name appearing anywhere in the path — an
    interpreter installed under a package's directory carries that package's
    name in every filename.

    numpy is transparent. An array a package asks numpy to build belongs to
    the package that asked, whether the call lands in C or in numpy's own
    Python — `np.empty` allocates under its caller's frame and `np.resize`
    under numpy's, and the two must not attribute differently.
    """
    for frame in reversed(traceback):
        for package, root in _ROOTS:
            if frame.filename.startswith(root):
                return package
    for frame in reversed(traceback):
        if frame.filename.startswith(_NUMPY):
            return "numpy"
    return "caller"


def attribute(build, depth=25):
    """Bytes each package still holds once `build` has returned.

    A snapshot carries the blocks alive when it is taken, so this weighs what
    a build retains and not what it touched on the way. `peak_bytes` answers
    the other half.
    """
    tracemalloc.start(depth)
    kept = build()
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    owned = {}
    for stat in snapshot.statistics("traceback"):
        package = _owner(stat.traceback)
        owned[package] = owned.get(package, 0) + stat.size
    del kept
    return owned


def peak_bytes(build, depth=25):
    """The highest traced total reached while `build` runs, over every package.

    `get_traced_memory` keeps a high-water mark, so this counts an allocation
    a build frees before returning, which a snapshot cannot see. It is one
    number for the process and not a figure per package: a snapshot carries no
    peak per traceback, so the peak is attributed by holding everything but
    the nonzeros fixed and reading what the nonzeros add.

    An allocation freed before the high-water mark is reached does not appear,
    which is correct rather than a gap: peak is a maximum, and memory returned
    before it costs nothing.
    """
    tracemalloc.start(depth)
    kept = build()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del kept
    return peak
