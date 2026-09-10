"""Peak RAM and time of building the corpus transport model at a size.

The model is `nimopt.models.transport`. The benchmark and the documentation
describe one definition. The data differs. This network draws each band from
every warehouse. The corpus network excludes the last warehouse.

The arc list and its labels are the input of a columnar store. Every
computation nimopt performs on them is measured.
"""

import time
import tracemalloc

import numpy as np

from nimopt.models import transport
from nimopt.solvers import highs


def network(n_plants, n_warehouses, arcs_per_plant, seed=0):
    """Return the arc index of plant and warehouse positions, one column per arc.

    Each plant serves a band of `arcs_per_plant` consecutive warehouses.
    """
    if arcs_per_plant > n_warehouses:
        raise ValueError(
            f"arcs_per_plant is {arcs_per_plant} and n_warehouses is "
            f"{n_warehouses}; set arcs_per_plant to at most n_warehouses"
        )
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, n_warehouses, n_plants)
    plants = np.repeat(np.arange(n_plants, dtype=np.int32), arcs_per_plant)
    band = np.arange(arcs_per_plant, dtype=np.int64)
    warehouses = ((offsets[:, None] + band[None, :]) % n_warehouses).ravel()
    return np.stack([plants, warehouses.astype(np.int32)])


def inputs(n_plants, n_warehouses, arcs_per_plant, arc_index, unit_cost, integer):
    """Return the corpus model data over the given arc network."""
    plants = np.array([f"p{i}" for i in range(n_plants)])
    warehouses = np.array([f"w{i}" for i in range(n_warehouses)])
    labels = {"P": plants[arc_index[0]], "W": warehouses[arc_index[1]]}

    # supply per plant is the band width and demand per warehouse is at most
    # that width, so the model is feasible
    per_plant = float(arcs_per_plant)
    served = np.bincount(arc_index[1], minlength=n_warehouses).astype(np.float64)
    data = {
        "P": plants,
        "W": warehouses,
        "cost": (labels, unit_cost),
        "supply": np.full(n_plants, per_plant),
        "demand": np.minimum(served, per_plant),
    }
    if integer:
        data["capacity"] = (labels, np.full(arc_index.shape[1], per_plant))
    return data


def build(n_plants, n_warehouses, arcs_per_plant, arc_index, unit_cost, integer=False):
    """Return the corpus transport model over this benchmark's network.

    `integer` makes the flow an integer column bounded by the band width. The
    model is then a MILP.
    """
    return transport.definition(integer=integer).build(
        inputs(n_plants, n_warehouses, arcs_per_plant, arc_index, unit_cost, integer)
    )


def measure(n_plants, n_warehouses, arcs_per_plant, solve=False, seed=0):
    """Return the peak bytes and the elapsed time of one build and assembly.

    `solve` adds the status, the objective and the solve time.
    """
    arc_index = network(n_plants, n_warehouses, arcs_per_plant, seed)
    unit_cost = np.random.default_rng(seed + 1).uniform(1.0, 9.0, arc_index.shape[1])

    tracemalloc.start()
    started = time.perf_counter()
    model = build(n_plants, n_warehouses, arcs_per_plant, arc_index, unit_cost)
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
        f"{'plants':>7} {'wh':>6} {'arcs/p':>7} {'rows':>8} {'cols':>9} "
        f"{'nnz':>9} {'matrix MB':>10} {'peak MB':>9} {'ratio':>7} "
        f"{'build ms':>9}"
    )
    for plants, warehouses, per_plant in (
        (200, 100, 10),
        (2000, 500, 20),
        (10000, 2000, 40),
    ):
        got = measure(plants, warehouses, per_plant)
        print(
            f"{plants:>7} {warehouses:>6} {per_plant:>7} {got['rows']:>8} "
            f"{got['cols']:>9} {got['nnz']:>9} {got['matrix_mb']:>10.2f} "
            f"{got['peak_mb']:>9.2f} {got['ratio']:>7.2f} {got['build_ms']:>9.1f}"
        )
    solved = measure(200, 100, 10, solve=True)
    print(
        f"\n200x100: status {solved['status']}, objective "
        f"{solved['objective']:.1f}, solve {solved['solve_ms']:.1f} ms"
    )
