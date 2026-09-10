"""Unit commitment: a binary on-off column per generator and snapshot.

A committed unit runs between its minimum and its maximum and adds a no-load
charge for being on. An uncommitted unit produces nothing. The rows `capacity`
and `minimum` express that pair against the binary column.

No row couples one snapshot to the next. `reference` finds the cheapest
commitment per snapshot by enumerating every on-off subset of three units.
"""

import itertools
from collections.abc import Mapping
from typing import Any

import numpy as np

from nimopt import Definition, Sum

FLEET = (
    ("base", 120.0, 40.0, 20.0, 500.0),
    ("mid", 80.0, 20.0, 35.0, 200.0),
    ("peak", 60.0, 0.0, 70.0, 50.0),
)
LOAD = (60.0, 140.0, 200.0, 95.0)


def definition() -> Definition:
    """Return the commitment model, with no data bound."""
    d = Definition("commitment", sense="min")
    T, G = d.set("T"), d.set("G")
    p_max = d.param("p_max", (G,))
    p_min = d.param("p_min", (G,))
    cost = d.param("cost", (G,))
    no_load = d.param("no_load", (G,))
    load = d.param("load", (T,))

    on = d.var("on", (T, G), lower=0.0, upper=1.0, integer=True)
    gen = d.var("gen", (T, G), lower=0.0)
    d.constraint("capacity", gen[T, G] - p_max[G] * on[T, G] <= 0.0)
    d.constraint("minimum", gen[T, G] - p_min[G] * on[T, G] >= 0.0)
    d.constraint("balance", Sum(G, gen[T, G]) == load[T])
    d.set_objective(Sum(T, G, cost[G] * gen[T, G]) + Sum(T, G, no_load[G] * on[T, G]))
    return d


def data(scale: int = 1) -> dict[str, Any]:
    """Return inputs for `scale` copies of the fleet over the scaled hours."""
    fleet = [
        (f"{name}{k}", p_max, p_min, cost, no_load)
        for k in range(scale)
        for name, p_max, p_min, cost, no_load in FLEET
    ]
    return {
        "T": np.arange(len(LOAD) * scale),
        "G": np.array([name for name, _, _, _, _ in fleet]),
        "p_max": np.array([v for _, v, _, _, _ in fleet]),
        "p_min": np.array([v for _, _, v, _, _ in fleet]),
        "cost": np.array([v for _, _, _, v, _ in fleet]),
        "no_load": np.array([v for _, _, _, _, v in fleet]),
        "load": np.tile(np.array(LOAD), scale) * scale,
    }


def reference(data: Mapping[str, Any]) -> float:
    """Return the cheapest commitment per snapshot, over every on-off subset.

    A committed unit runs at least its minimum. The remainder of the load is
    filled cheapest first among the units that are on.
    """
    p_max, p_min = data["p_max"], data["p_min"]
    cost, no_load = data["cost"], data["no_load"]
    units = range(len(p_max))
    total = 0.0
    for want in data["load"]:
        best = np.inf
        for mask in itertools.product((0, 1), repeat=len(p_max)):
            up = [g for g in units if mask[g]]
            floor = sum(p_min[g] for g in up)
            if not up or floor > want + 1e-9:
                continue
            if sum(p_max[g] for g in up) < want - 1e-9:
                continue
            spend = sum(no_load[g] + p_min[g] * cost[g] for g in up)
            left = want - floor
            for g in sorted(up, key=lambda g: cost[g]):
                taken = min(left, p_max[g] - p_min[g])
                spend += taken * cost[g]
                left -= taken
            best = min(best, spend)
        if not np.isfinite(best):
            raise ValueError(
                f"no commitment of this fleet meets a load of {want}; pass a load "
                f"the fleet meets"
            )
        total += best
    return float(total)
