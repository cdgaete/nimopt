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
    """Return the package that allocated the block.

    A traceback is ordered from the oldest frame. The reverse walk finds the
    innermost frame of a package. A package is identified by its source root
    and not by a name appearing anywhere in the path. An interpreter
    installed under a package directory contains that package name in every
    filename.

    numpy is transparent. An array built by a numpy call is attributed to the
    calling package. `np.empty` allocates under the frame of its caller.
    `np.resize` allocates under a numpy frame. Both are attributed to the
    calling package.
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
    """Return the bytes each package retains after `build` returns.

    The snapshot contains the blocks alive when it is taken. The result
    measures what a build retains and not what it allocates and frees.
    `peak_bytes` measures the peak.
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
    """Return the highest traced total over every package while `build` runs.

    `get_traced_memory` reports the peak traced total. The result includes an
    allocation a build frees before returning. A snapshot does not report that
    allocation. The result is one number for the process and not a figure per
    package. A snapshot has no peak per traceback. The peak of a package is
    measured by changing the nonzero count and reading the difference.

    An allocation freed before the peak does not appear in the result. The
    peak is a maximum, and memory returned before it does not raise the
    maximum.
    """
    tracemalloc.start(depth)
    kept = build()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del kept
    return peak
