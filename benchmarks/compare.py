"""Measure what a model costs to build, solve and read back.

A `Case` defines how one builder expresses one model. `measure` runs its
phases in this process. `run` runs one case in a fresh child process.
Resident memory includes the import cost of the library loaded, and each side
is given a process of its own.

Resident size is sampled and not traced. `tracemalloc` records only the
allocations of the Python allocator. Two libraries allocating by different
routes would be compared on the route and not on the model. The traced peak
is reported beside the sampled peak for the build phase, in the units of the
single-library benchmarks.
"""

import json
import os
import resource
import subprocess
import sys
import threading
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")
_STATM = Path("/proc/self/statm")
_HERE = Path(__file__).resolve().parent
_RESULT = "RESULT "


@dataclass
class Case:
    """One model, expressed by one builder, as the phases a benchmark measures.

    `build` returns the value the later phases use. `describe` returns the
    shape of the problem. `solve` returns the objective. `read` returns a
    summary of the primals and duals read back onto their labels.
    """

    build: Callable[[], Any]
    describe: Callable[[Any], dict]
    solve: Callable[[Any], float]
    read: Callable[[Any], dict]


def _rss_mb():
    return int(_STATM.read_text().split()[1]) * _PAGE_BYTES / 1e6


class _Watermark:
    """Resident size through a phase, and the highest sample.

    The sampling interval is short enough to record a peak between two
    coarser readings. The trace is written at a coarser cadence than the
    sampling: one reading every `every` seconds, and one whenever the size
    changes by more than `step`. An hour-long solve produces a curve of a few
    thousand rows, and a peak of one millisecond is recorded.
    """

    def __init__(self, interval=0.0005, every=0.25, step=1.0):
        self.interval = interval
        self.every = every
        self.step = step
        self.baseline = _rss_mb()
        self.high = self.baseline
        self.peak_at = 0.0
        self.trace = [(0.0, self.baseline)]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        started = time.perf_counter()
        last_at, last_rss = 0.0, self.baseline
        while not self._stop.is_set():
            rss = _rss_mb()
            now = time.perf_counter() - started
            if rss > self.high:
                self.high, self.peak_at = rss, now
            if now - last_at >= self.every or abs(rss - last_rss) >= self.step:
                self.trace.append((now, rss))
                last_at, last_rss = now, rss
            time.sleep(self.interval)

    def __enter__(self):
        self._started = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        rss = _rss_mb()
        if rss > self.high:
            self.high, self.peak_at = rss, time.perf_counter() - self._started
        self.trace.append((time.perf_counter() - self._started, rss))

    @property
    def over_baseline(self):
        return self.high - self.baseline


def _phase(got, name, work, traces=None):
    """Run one phase, recording resident size through it and its wall time.

    The peak is read from the trace. A solver that allocates a factorization
    and frees it produces a curve. A single number at the end of the phase
    does not report that allocation.
    """
    started = time.perf_counter()
    with _Watermark() as mark:
        result = work()
    got[f"{name}_ms"] = (time.perf_counter() - started) * 1e3
    got[f"{name}_rss_mb"] = mark.over_baseline
    got[f"{name}_rss_peak_mb"] = mark.high
    got[f"{name}_rss_peak_at_s"] = mark.peak_at
    if traces is not None:
        traces[name] = mark.trace
    return result


def write_trace(path, trace):
    """Write the resident size of one phase over time, in seconds and megabytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        out.write("seconds,rss_mb\n")
        for at, rss in trace:
            out.write(f"{at:.4f},{rss:.3f}\n")


def measure(case, traced=True, solve=True, trace_dir=None, label="", on_phase=None):
    """Build, solve and read back one case, reporting the cost of each phase.

    `solve=False` measures the build alone. A problem too large to solve
    reports the cost of the build and no objective.

    A phase trace is written and `on_phase(name, got)` is called when that
    phase ends. A run stopped during a long solve retains the numbers and the
    curve of the build.
    """
    got = {}
    traces = {}

    def done(name):
        if trace_dir is not None:
            write_trace(Path(trace_dir) / f"{label}-{name}.csv", traces[name])
            got[f"{name}_trace"] = f"{label}-{name}.csv"
        if on_phase is not None:
            on_phase(name, got)

    built = _phase(got, "build", case.build, traces)
    got.update(case.describe(built))
    done("build")
    if solve:
        got["objective"] = _phase(got, "solve", lambda: case.solve(built), traces)
        done("solve")
        got.update(_phase(got, "read", lambda: case.read(built), traces))
        done("read")
    # getrusage counts kibibytes, so a megabyte is 1024 of them
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    got["process_rss_mb"] = peak_kib * 1024 / 1e6
    built = None
    if traced:
        tracemalloc.start()
        traced_built = case.build()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del traced_built
        got["build_traced_mb"] = peak / 1e6
    return got


def run(case, side, sizes, traced=True, solve=True, trace_dir=None):
    """Measure one case in a fresh process with one library loaded."""
    label = "-".join([case, side] + [str(v) for v in sizes.values()])
    spec = json.dumps(
        {
            "case": case,
            "side": side,
            "sizes": sizes,
            "traced": traced,
            "solve": solve,
            "trace_dir": None if trace_dir is None else str(trace_dir),
            "label": label,
        }
    )
    done = subprocess.run(
        [sys.executable, str(_HERE / "compare.py"), spec],
        capture_output=True,
        text=True,
        cwd=str(_HERE),
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"measuring {case} on {side} at {sizes} failed; read the error "
            f"below\n{done.stderr.strip()}"
        )
    for line in done.stdout.splitlines():
        if line.startswith(_RESULT):
            return json.loads(line[len(_RESULT) :])
    raise RuntimeError(
        f"measuring {case} on {side} at {sizes} printed no result; read the "
        f"output below\n{done.stdout}"
    )


def _main(argv):
    spec = json.loads(argv[1])
    sys.path.insert(0, str(_HERE))
    from bench_vs_linopy import CASES

    factory = CASES[spec["case"]][spec["side"]]
    got = measure(
        factory(**spec["sizes"]),
        traced=spec["traced"],
        solve=spec.get("solve", True),
        trace_dir=spec.get("trace_dir"),
        label=spec.get("label", ""),
    )
    print(_RESULT + json.dumps(got))


if __name__ == "__main__":
    _main(sys.argv)
