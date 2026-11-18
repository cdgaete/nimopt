"""The time and the simplex iterations of a sweep, and the cost of an appended row.

`sweep` changes one column bound of the corpus storage model per value, once
through one open session and once by building the model again for each value.
The values are fractions of the declared capacity of that column. The
benchmark reports the mean time of an iteration on each side, the simplex
iterations of every solve, and the largest difference between the two
objectives.

`append` appends one row of 200 coefficients to the European network. The
unloaded phase first measures the cost outside HiGHS, in a process that has
loaded no solver model. The loaded phase then measures the same rows with the
model loaded in HiGHS.

    python benchmarks/bench_session.py
    python benchmarks/bench_session.py --snapshots 24
    python benchmarks/bench_session.py --skip-append
"""

import argparse
import gc
import threading
import time

import numpy as np
from bench_storage import UNIT_MW, build, profiles
from compare import _rss_mb

from nimopt import Param, Sum


def _first_member(model):
    """Return a reference to the first column of the model's `gen` variable."""
    gen = model.variables["gen"]
    labels = gen.labels_at([0])
    return gen[tuple(labels[d][0] for d in gen.dims)]


def sweep(n_generators, n_storage, n_hours, iterations):
    """Return the time and the simplex iterations of one sweep, both ways.

    The session side builds the model once, solves it once, and then changes
    the upper bound of one column of `gen` and solves again `iterations`
    times. The rebuild side builds the model and opens a session for each
    value, changes the bound before the first solve, and closes the session.
    Each value is a fraction of the declared capacity of that column.
    """
    availability, demand, price = profiles(n_generators, n_hours)
    data = (n_generators, n_storage, n_hours, availability, demand, price)

    started = time.perf_counter()
    model = build(*data)
    session = model.session()
    first = session.solve()
    session_first_ms = (time.perf_counter() - started) * 1e3

    target = _first_member(model)
    v0 = float(first.primal("gen").to_dense().reshape(-1)[0])
    capacity = UNIT_MW * float(availability[0, 0])
    values = [
        capacity * (iterations - k) / (iterations + 1) for k in range(1, iterations + 1)
    ]

    session_ms = []
    session_iterations = [int(first.iterations)]
    objectives = []
    for value in values:
        started = time.perf_counter()
        session.set_bound(target, upper=value)
        solution = session.solve()
        session_ms.append((time.perf_counter() - started) * 1e3)
        session_iterations.append(int(solution.iterations))
        objectives.append(float(solution.objective))
    rows = session.n_rows
    cols = session.n_columns
    nnz = session.assembled.values.size
    session.close()

    rebuild_ms = []
    rebuild_iterations = []
    rebuilt = []
    for value in values:
        started = time.perf_counter()
        again = build(*data)
        with again.session() as other:
            other.set_bound(_first_member(again), upper=value)
            result = other.solve()
        rebuild_ms.append((time.perf_counter() - started) * 1e3)
        rebuild_iterations.append(int(result.iterations))
        rebuilt.append(float(result.objective))

    return {
        "first_value": v0,
        "binding": int(sum(1 for value in values if value < v0)),
        "rebuild_ms": float(np.mean(rebuild_ms)),
        "session_ms": float(np.mean(session_ms)),
        "session_first_ms": float(session_first_ms),
        "rebuild_iterations": rebuild_iterations,
        "session_iterations": session_iterations,
        "objectives": objectives,
        "largest_disagreement": float(
            max(abs(a - b) for a, b in zip(objectives, rebuilt))
        ),
        "rows": int(rows),
        "cols": int(cols),
        "nnz": int(nnz),
    }


class _Sampled:
    """The highest resident size a thread samples while a block runs."""

    def __init__(self, interval=0.002):
        self.interval = interval
        self.high = _rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        while not self._stop.is_set():
            self.high = max(self.high, _rss_mb())
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.high = max(self.high, _rss_mb())


def _appended(session, name, relation):
    """Return the elapsed time of one `add_row` call, in milliseconds."""
    started = time.perf_counter()
    session.add_row(name, relation)
    return (time.perf_counter() - started) * 1e3


def append(snapshots, later=8):
    """Return the time and the resident memory of one appended row of 200 coefficients.

    The rows are `Sum(Link, coef[Link] * Link_p_nom[Link]) <= 1e9` and
    `Sum(Link, coef[Link] * Link_p[Link, <the first snapshot>]) <= 1e9`.
    The unloaded phase runs first, on a session that was never solved. The
    loaded phase then solves the model under a time limit and appends the same
    rows to the solver's model. `snapshots=None` loads every snapshot of the
    network.
    """
    import bench_pypsa_backend as network

    n = network.load("nimopt", snapshots)
    model = n.optimize.create_model(backend="nimopt").model
    nom = model.variables["Link_p_nom"]
    wide = model.variables["Link_p"]
    link = nom.sets[0]
    labels = nom.labels_at(np.arange(nom.n_columns))
    chosen = labels[link.name][:200]
    if chosen.size < 200:
        raise ValueError(
            f"variable {nom.name!r} has {chosen.size} members; append a row of "
            f"200 coefficients to a model of 200 members or more"
        )
    rng = np.random.default_rng(0)
    coef = Param.from_long("coef", (link,), {link.name: chosen}, rng.random(200))
    first_snapshot = wide.labels_at([0])[wide.sets[1].name][0]

    def narrow():
        return Sum(link, coef[link] * nom[link]) <= 1e9

    def wide_row():
        return Sum(link, coef[link] * wide[link, first_snapshot]) <= 1e9

    unloaded = model.session()
    rows = unloaded.n_rows
    cols = unloaded.n_columns
    nnz = unloaded.assembled.values.size

    gc.collect()
    before = _rss_mb()
    unloaded_ms = 0.0
    with _Sampled() as sample:
        for k in range(later):
            unloaded_ms = max(
                unloaded_ms, _appended(unloaded, f"bench_nom_unloaded{k}", narrow())
            )
    added_rss_mb = _rss_mb() - before
    peak_rss_mb = sample.high - before

    before = _rss_mb()
    wide_unloaded_ms = 0.0
    with _Sampled() as sample:
        for k in range(later):
            wide_unloaded_ms = max(
                wide_unloaded_ms,
                _appended(unloaded, f"bench_wide_unloaded{k}", wide_row()),
            )
    wide_added_rss_mb = _rss_mb() - before
    wide_peak_rss_mb = sample.high - before

    del unloaded
    gc.collect()

    session = model.session(options={"time_limit": 1.0})
    status = session.solve().status
    first_ms = _appended(session, "bench_nom_first", narrow())
    later_ms = 0.0
    for k in range(later):
        later_ms = max(later_ms, _appended(session, f"bench_nom_later{k}", narrow()))
    wide_later_ms = 0.0
    for k in range(later):
        wide_later_ms = max(
            wide_later_ms, _appended(session, f"bench_wide_later{k}", wide_row())
        )
    session.close()

    return {
        "status": status,
        "rows": int(rows),
        "cols": int(cols),
        "nnz": int(nnz),
        "snapshots": int(len(n.snapshots)),
        "first_ms": first_ms,
        "later_ms": later_ms,
        "wide_later_ms": wide_later_ms,
        "added_rss_mb": added_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "wide_added_rss_mb": wide_added_rss_mb,
        "wide_peak_rss_mb": wide_peak_rss_mb,
        "unloaded_ms": unloaded_ms,
        "wide_unloaded_ms": wide_unloaded_ms,
    }


def _number(value):
    """Return one number of a report line, in its own precision."""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return f"{value:,}"
    if value and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _report(title, got, units):
    """Print one labeled line per number of `got`, in the order of `units`."""
    print(title)
    for name, unit in units:
        value = got[name]
        if isinstance(value, list):
            shown = ", ".join(_number(one) for one in value)
        else:
            shown = _number(value)
        print(f"  {name:<22} {shown:>12} {unit}")
    print(flush=True)


_SWEEP_UNITS = (
    ("first_value", "value of the first column at the first solve"),
    ("binding", "sweep values below the first solve"),
    ("rebuild_ms", "ms per rebuild iteration"),
    ("session_ms", "ms per resumed iteration"),
    ("session_first_ms", "ms for the build, the open and the first solve"),
    ("rebuild_iterations", "simplex iterations of each rebuild solve"),
    ("session_iterations", "simplex iterations of the session solves"),
    ("objectives", "objective of each resumed solve"),
    ("largest_disagreement", "largest absolute difference of the objectives"),
    ("rows", "rows"),
    ("cols", "columns"),
    ("nnz", "nonzeros"),
)

_APPEND_UNITS = (
    ("status", "what the loaded solve reports"),
    ("first_ms", "ms for the first row over Link_p_nom, with the HiGHS call"),
    ("later_ms", "ms for the slowest row over Link_p_nom, with the HiGHS call"),
    ("wide_later_ms", "ms for the slowest row over Link_p, with the HiGHS call"),
    ("unloaded_ms", "ms for the slowest row over Link_p_nom, no HiGHS model"),
    ("wide_unloaded_ms", "ms for the slowest row over Link_p, no HiGHS model"),
    ("added_rss_mb", "MB resident after the rows over Link_p_nom, no HiGHS model"),
    ("peak_rss_mb", "MB the highest sample less the start, Link_p_nom, no HiGHS model"),
    ("wide_added_rss_mb", "MB resident after the rows over Link_p, no HiGHS model"),
    (
        "wide_peak_rss_mb",
        "MB the highest sample less the start, Link_p, no HiGHS model",
    ),
    ("rows", "rows"),
    ("cols", "columns"),
    ("nnz", "nonzeros"),
    ("snapshots", "snapshots"),
)


def main(argv=None):
    """Measure the sweep and the appended row, and print one line per number."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--snapshots", type=int, default=None, help="first N snapshots of the network"
    )
    parser.add_argument(
        "--skip-append", action="store_true", help="measure the sweep alone"
    )
    args = parser.parse_args(argv)
    _report(
        "sweep, 40 generators, 8 storage units, 2190 hours, 8 values",
        sweep(40, 8, 2190, 8),
        _SWEEP_UNITS,
    )
    if not args.skip_append:
        _report(
            "appended rows, one row of 200 coefficients",
            append(args.snapshots),
            _APPEND_UNITS,
        )


if __name__ == "__main__":
    main()
