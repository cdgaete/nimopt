"""The time of the declaration shape pass and the share of it spent on values.

A constraint materializes its expression at declaration, to find its rows and
count its coefficients, and again at assembly, to write them. The first pass
reads the coordinates of the entries and not their values. This benchmark
times both passes on the corpus storage model and profiles a second build. It
reports two bounds on the value time of the build: the functions that compute
values alone, and those plus the functions that compute indices and values
together.
"""

import cProfile
import pstats
import time
from pathlib import Path

from bench_storage import inputs, profiles

from nimopt.models import storage

VALUE_ONLY = frozenset(
    {
        ("kernel.py", "take_filled"),
        ("kernel.py", "_sum_runs"),
        ("sparse.py", "_assemble"),
        ("sparse.py", "_scalar"),
    }
)
INDEX_AND_VALUE = frozenset(
    ("kernel.py", name)
    for name in (
        "multiply_join",
        "multiply_lookup",
        "reduce_axis",
        "weighted_sum_axis",
        "gather",
        "compress",
        "select_axis",
        "shift_axis",
        "canonicalize",
        "regroup",
    )
)


def value_time(stats):
    """Return the seconds in value-only functions, and with the mixed ones."""
    low = mixed = 0.0
    for (path, _, name), row in stats.stats.items():
        key = (Path(path).name, name)
        if key in VALUE_ONLY:
            low += row[2]
        elif key in INDEX_AND_VALUE:
            mixed += row[2]
    return low, low + mixed


def measure(n_generators, n_storage, n_hours, seed=0):
    """Return the time of each pass and the bounds on the value time.

    `saving_low` and `saving_high` bound the share of the declaration and
    assembly time that a shape pass computing no values saves.
    """
    availability, demand, price = profiles(n_generators, n_hours, seed)
    data = inputs(n_generators, n_storage, n_hours, availability, demand, price)
    definition = storage.definition()
    start = time.perf_counter()
    model = definition.build(data)
    declare = time.perf_counter() - start
    start = time.perf_counter()
    model.assemble()
    assemble = time.perf_counter() - start
    profiler = cProfile.Profile()
    profiler.enable()
    definition.build(data)
    profiler.disable()
    stats = pstats.Stats(profiler)
    low, high = value_time(stats)
    profiled = stats.total_tt
    weight = declare / (declare + assemble)
    return {
        "rows": model.n_rows,
        "nnz": model.nnz,
        "declare_s": declare,
        "assemble_s": assemble,
        "profiled_s": profiled,
        "value_low_s": low,
        "value_high_s": high,
        "saving_low": low / profiled * weight,
        "saving_high": high / profiled * weight,
    }


if __name__ == "__main__":
    print(
        f"{'gens':>5} {'store':>6} {'hours':>6} {'rows':>9} {'nnz':>10} "
        f"{'declare s':>10} {'assemble s':>11} {'saving':>15}"
    )
    for generators, batteries, hours in ((10, 2, 168), (40, 8, 720), (80, 20, 8760)):
        got = measure(generators, batteries, hours)
        print(
            f"{generators:>5} {batteries:>6} {hours:>6} {got['rows']:>9} "
            f"{got['nnz']:>10} {got['declare_s']:>10.3f} {got['assemble_s']:>11.3f} "
            f"{got['saving_low']:>7.1%} to {got['saving_high']:.1%}"
        )
