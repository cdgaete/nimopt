"""Least-cost dispatch of a generator fleet against a load.

One variable over snapshots crossed with generators, one balance row per
snapshot, and a cost per generator. Every other model in the corpus is this
one with an axis added. The balance row is
`Sum(generator, p[snapshot, generator]) == load[snapshot]`, with no
coefficient. Each snapshot is independent, and `reference` computes the
hourly merit order.
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
    """Return the dispatch model, with no data bound."""
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
    """Return inputs for a fleet and horizon `scale` times the base size.

    The fleet repeats and the horizon tiles. The load per generator is what it
    is at scale 1, and the merit order has the same shape at any size.
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
    """Return what the dispatch costs, by merit order over each snapshot."""
    return merit_order(data["p_max"], data["cost"], data["load"])
