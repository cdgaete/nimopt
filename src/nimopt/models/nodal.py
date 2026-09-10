"""Generators sited at buses, each bus meeting its own demand.

The grouping is a lookup. `at[G, B]` has an entry where generator `g` is sited
at bus `b`. Multiplying the generation by it turns a row over generators into
a row over buses. The coefficient introduces `B`, which no variable is
declared over, and the balance is free over the dimensions of the lookup. Each
bus meets its own demand, and `reference` computes a merit order per bus and
hour.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from nimopt import Definition, Sum
from nimopt.models._arithmetic import merit_order

SITED = (
    ("g0", "b0", 100.0, 10.0),
    ("g1", "b0", 80.0, 40.0),
    ("g2", "b1", 90.0, 15.0),
    ("g3", "b1", 70.0, 45.0),
)
HOURS = 3
DEMAND = {"b0": (120.0, 150.0, 90.0), "b1": (100.0, 130.0, 80.0)}


def definition() -> Definition:
    """Return the nodal model, with no data bound."""
    d = Definition("nodal", sense="min")
    T, G, B = d.set("T"), d.set("G"), d.set("B")
    at = d.param("at", (G, B))
    p_max = d.param("p_max", (G,))
    cost = d.param("cost", (G,))
    demand = d.param("demand", (B, T))
    gen = d.var("gen", (T, G), lower=0.0, upper=p_max)
    d.constraint("balance", Sum(G, at[G, B] * gen[T, G]) == demand[B, T])
    d.set_objective(Sum(T, G, cost[G] * gen[T, G]))
    return d


def _sited(scale: int) -> list[tuple[str, str, float, float]]:
    """Return the generator-bus pairs, repeated `scale` times."""
    return [
        (f"{unit}_{k}", f"{bus}_{k}", capacity, price)
        for k in range(scale)
        for unit, bus, capacity, price in SITED
    ]


def data(scale: int = 1) -> dict[str, Any]:
    """Return inputs for `scale` copies of the network over the scaled hours."""
    sited = _sited(scale)
    hours = np.arange(HOURS * scale)
    buses = list(dict.fromkeys(bus for _, bus, _, _ in sited))
    return {
        "T": hours,
        "G": np.array([unit for unit, _, _, _ in sited]),
        "B": np.array(buses),
        "at": (
            {
                "G": np.array([unit for unit, _, _, _ in sited]),
                "B": np.array([bus for _, bus, _, _ in sited]),
            },
            np.ones(len(sited)),
        ),
        "p_max": np.array([capacity for _, _, capacity, _ in sited]),
        "cost": np.array([price for _, _, _, price in sited]),
        "demand": np.array(
            [np.tile(np.array(DEMAND[bus.split("_")[0]]), scale) for bus in buses]
        ),
    }


def reference(data: Mapping[str, Any]) -> float:
    """Return what the dispatch costs, by merit order over each bus and hour."""
    columns, _ = data["at"]
    total = 0.0
    for position, bus in enumerate(data["B"]):
        here = columns["B"] == bus
        total += merit_order(
            data["p_max"][here], data["cost"][here], data["demand"][position]
        )
    return float(total)
