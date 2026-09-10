"""Peak RAM and time of building the corpus's storage model at a size.

The model is `nimopt.models.storage`, so the benchmark and the documentation
describe one definition rather than two that resemble each other. What differs
here is the data: a thermal fleet, a solar fleet and a price spread wide enough
that the batteries are worth cycling, where the corpus's own `data` states a
spread narrow enough for its reference to be arithmetic.

The hourly profiles are the input a time series store hands over; everything
nimopt does with them is measured.
"""

import time
import tracemalloc

import numpy as np

from nimopt.models import storage
from nimopt.models.storage import CHARGE_EFFICIENCY, DISCHARGE_EFFICIENCY
from nimopt.solvers import highs

UNIT_MW = 100.0
STORE_MW = 50.0
STORE_MWH = 200.0
RAMP_FRACTION = 0.5


def profiles(n_generators, n_hours, seed=0):
    """Availability per generator and hour, hourly demand, and marginal costs.

    Half the fleet is thermal and available at every hour; the rest is solar
    and follows daylight. Peak demand stays under the thermal fleet, so the
    dispatch clears whatever the solar profile does.
    """
    rng = np.random.default_rng(seed)
    n_thermal = max(1, n_generators // 2)
    hour = np.arange(n_hours) % 24
    availability = np.ones((n_generators, n_hours))
    availability[n_thermal:] = np.clip(np.sin(2 * np.pi * (hour - 6) / 24), 0.0, 1.0)
    base = 0.5 * n_thermal * UNIT_MW
    demand = base * (1 + 0.3 * np.sin(2 * np.pi * (hour - 18) / 24))
    price = np.concatenate(
        [rng.uniform(20.0, 80.0, n_thermal), np.full(n_generators - n_thermal, 0.1)]
    )
    return availability, demand, np.tile(price[:, None], (1, n_hours))


def inputs(n_generators, n_storage, n_hours, availability, demand, price):
    """The corpus model's data, at this benchmark's size and price spread."""
    gen_shape = (n_generators, n_hours)
    store_shape = (n_storage, n_hours)
    return {
        "T": np.arange(n_hours),
        "G": np.array([f"g{i}" for i in range(n_generators)]),
        "S": np.array([f"s{i}" for i in range(n_storage)]),
        "cost": price,
        "capacity": UNIT_MW * availability,
        "ramp_limit": np.full(gen_shape, RAMP_FRACTION * UNIT_MW),
        "load": demand,
        "power": np.full(store_shape, STORE_MW),
        "energy": np.full(store_shape, STORE_MWH),
        "charge_eta": np.full(store_shape, CHARGE_EFFICIENCY),
        "discharge_eta": np.full(store_shape, 1 / DISCHARGE_EFFICIENCY),
    }


def build(n_generators, n_storage, n_hours, availability, demand, price):
    """The corpus's storage model, bound to this benchmark's data."""
    return storage.definition().build(
        inputs(n_generators, n_storage, n_hours, availability, demand, price)
    )


def measure(n_generators, n_storage, n_hours, solve=False, seed=0):
    """Peak bytes and elapsed time of one build, assembly and optional solve."""
    availability, demand, price = profiles(n_generators, n_hours, seed)

    tracemalloc.start()
    started = time.perf_counter()
    model = build(n_generators, n_storage, n_hours, availability, demand, price)
    assembled = model.assemble()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    matrix = assembled.indices.nbytes + assembled.values.nbytes
    got = {
        "rows": assembled.n_rows,
        "cols": assembled.n_cols,
        "nnz": int(assembled.values.size),
        "matrix_mb": matrix / 1e6,
        "peak_mb": peak / 1e6,
        "ratio": peak / matrix,
        "build_ms": elapsed * 1e3,
    }
    if solve:
        started = time.perf_counter()
        result = highs.solve(assembled, model.sense)
        got["solve_ms"] = (time.perf_counter() - started) * 1e3
        got["status"] = result.status
        got["objective"] = result.objective
    return got


if __name__ == "__main__":
    print(
        f"{'gens':>5} {'store':>6} {'hours':>6} {'rows':>9} {'cols':>9} "
        f"{'nnz':>10} {'matrix MB':>10} {'peak MB':>9} {'ratio':>7} "
        f"{'build ms':>9}"
    )
    for generators, batteries, hours in ((10, 2, 168), (40, 8, 720), (80, 20, 8760)):
        got = measure(generators, batteries, hours)
        print(
            f"{generators:>5} {batteries:>6} {hours:>6} {got['rows']:>9} "
            f"{got['cols']:>9} {got['nnz']:>10} {got['matrix_mb']:>10.2f} "
            f"{got['peak_mb']:>9.2f} {got['ratio']:>7.2f} {got['build_ms']:>9.1f}"
        )
    solved = measure(10, 2, 168, solve=True)
    print(
        f"\na week on 10 generators and 2 batteries solves {solved['status']} "
        f"at {solved['objective']:.1f} in {solved['solve_ms']:.1f} ms"
    )
