"""Measure what a model costs to build, solve and read back.

A `Case` states how one builder expresses one model. `measure` runs its
phases in this process; `run` runs one in a fresh child, which is what makes
two libraries comparable — resident memory carries the import cost of
whichever library is loaded, so each side is given a process of its own.

Resident size is sampled rather than traced: `tracemalloc` sees only what
passes through Python's allocator, and two libraries that reach memory by
different routes would be compared on the route rather than on the model.
The traced peak is reported beside it for the build phase, in the units the
single-library benchmarks use.
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

    `build` returns the state the later phases work on, `describe` states the
    problem's shape from it, `solve` returns the objective, and `read` returns
    a summary of the primals and duals read back onto their labels.
    """

    build: Callable[[], Any]
    describe: Callable[[Any], dict]
    solve: Callable[[Any], float]
    read: Callable[[Any], dict]


def _rss_mb():
    return int(_STATM.read_text().split()[1]) * _PAGE_BYTES / 1e6


class _Watermark:
    """Resident size through a phase, and the highest it reached.

    Sampling is fast enough that a peak between two coarser readings is not
    missed. The trace is kept at a coarser cadence than the sampling — a
    reading every `every` seconds, and one whenever the size moves by more
    than `step` — so an hour-long solve leaves a curve rather than millions
    of rows, while a peak that lasts a millisecond is still recorded.
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

    The trace is where the peak is read from rather than assumed: a solver
    that holds its factorization for a moment and frees it leaves a curve
    that says so, and a single number at the end would not.
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
    """One phase's resident size through time, as seconds and megabytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        out.write("seconds,rss_mb\n")
        for at, rss in trace:
            out.write(f"{at:.4f},{rss:.3f}\n")


def measure(case, traced=True, solve=True, trace_dir=None, label="", on_phase=None):
    """Build, solve and read back one case, reporting what each phase cost.

    `solve=False` measures the build alone. A problem too large to solve
    still states what it cost to build, and reports no objective rather than
    a number it did not reach.

    A phase's trace is written and `on_phase(name, got)` is called as soon
    as that phase ends, so a run stopped during a long solve leaves the
    build's numbers and curve behind it.
    """
    got = {}
    traces = {}

    def done(name):
        if trace_dir is not None:
            write_trace(Path(trace_dir) / f"{label}-{name}.csv", traces[name])
            got[f"{name}_trace"] = f"{label}-{name}.csv"
        if on_phase is not None:
            on_phase(name, got)

    state = _phase(got, "build", case.build, traces)
    got.update(case.describe(state))
    done("build")
    if solve:
        got["objective"] = _phase(got, "solve", lambda: case.solve(state), traces)
        done("solve")
        got.update(_phase(got, "read", lambda: case.read(state), traces))
        done("read")
    # getrusage counts kibibytes, so a megabyte is 1024 of them. Dividing by
    # a thousand instead puts the process peak below phases it contains.
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    got["process_rss_mb"] = peak_kib * 1024 / 1e6
    state = None
    if traced:
        tracemalloc.start()
        traced_state = case.build()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del traced_state
        got["build_traced_mb"] = peak / 1e6
    return got


def run(case, side, sizes, traced=True, solve=True, trace_dir=None):
    """Measure one case in a fresh process, so only one library is loaded."""
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
            f"measuring {case} on {side} at {sizes} failed:\n{done.stderr.strip()}"
        )
    for line in done.stdout.splitlines():
        if line.startswith(_RESULT):
            return json.loads(line[len(_RESULT) :])
    raise RuntimeError(
        f"measuring {case} on {side} at {sizes} printed no result:\n{done.stdout}"
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
