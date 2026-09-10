"""Least-cost dispatch of a generator fleet against a load.

The baseline shape: one variable over snapshots crossed with generators, one
balance row per snapshot, and a cost per generator. Every other model in the
corpus is this one with an axis added.

The balance row states `Sum(generator, p[snapshot, generator]) == load[snapshot]`
and carries no coefficient, because a term summed over a dimension needs none.

Each snapshot is independent, so the optimum is the hourly merit order and
`reference` computes it.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from nimopt import Definition, Sum
from nimopt.models._arithmetic import merit_order

FLEET = ("wind", "solar", "gas")
CAPACITY = (100.0, 60.0, 200.0)
COST = (1.0, 2.0, 50.0)
LOAD = (80.0, 120.0, 150.0, 180.0, 140.0, 100.0)


def definition() -> Definition:
    """The dispatch model, with no data bound."""
    d = Definition("dispatch", sense="min")
    snapshot = d.set("snapshot")
    generator = d.set("generator")
    p_max = d.param("p_max", (generator,))
    load = d.param("load", (snapshot,))
    cost = d.param("cost", (generator,))
    p = d.var("p", (snapshot, generator), lower=0.0, upper=p_max)
    d.constraint("balance", Sum(generator, p[snapshot, generator]) == load[snapshot])
    d.set_objective(Sum(snapshot, generator, cost[generator] * p[snapshot, generator]))
    return d


def data(scale: int = 1) -> dict[str, Any]:
    """Inputs for a fleet and horizon `scale` times the base size.

    The fleet repeats and the horizon tiles, so the load per generator stays
    what it is at scale 1 and the merit order is the same shape at any size.
    """
    fleet = [f"{name}{k}" for k in range(scale) for name in FLEET]
    return {
        "snapshot": np.arange(len(LOAD) * scale),
        "generator": np.array(fleet),
        "p_max": np.tile(np.array(CAPACITY), scale),
        "cost": np.tile(np.array(COST), scale),
        "load": np.tile(np.array(LOAD), scale) * scale,
    }


def reference(data: Mapping[str, Any]) -> float:
    """What the dispatch costs, by merit order over each snapshot."""
    return merit_order(data["p_max"], data["cost"], data["load"])
