"""Technologies sited in some regions, running in every hour.

Mixed density: the region-technology map is sparse, and every sited pair runs
in every hour. The generation variable takes its columns from a parameter
holding the sparse pairs crossed with the whole horizon.

Each region meets its own demand from the technologies sited in it, and
`reference` computes a merit order per region and hour.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from nimopt import Definition, Sum
from nimopt.models._arithmetic import merit_order

SITED = (
    ("north", "hydro", 60.0, 5.0),
    ("north", "gas", 90.0, 40.0),
    ("south", "gas", 70.0, 45.0),
    ("south", "coal", 80.0, 25.0),
)
HOURS = 4
DEMAND = {"north": (100.0, 130.0, 80.0, 50.0), "south": (90.0, 140.0, 60.0, 110.0)}


def definition() -> Definition:
    """Return the sector model, with no data bound."""
    d = Definition("sector", sense="min")
    R, K, T = d.set("R"), d.set("K"), d.set("T")
    sited = d.param("sited", (R, K, T))
    capacity = d.param("capacity", (R, K))
    cost = d.param("cost", (R, K))
    demand = d.param("demand", (R, T))
    gen = d.var("gen", (R, K, T), subset=sited, lower=0.0, upper=capacity)
    d.constraint("balance", Sum(K, gen[R, K, T]) == demand[R, T])
    d.set_objective(Sum(R, K, T, cost[R, K] * gen[R, K, T]))
    return d


def _pairs(scale: int) -> list[tuple[str, str, float, float]]:
    """Return the sited region-technology pairs, repeated `scale` times."""
    return [
        (f"{region}{k}", technology, capacity, price)
        for k in range(scale)
        for region, technology, capacity, price in SITED
    ]


def data(scale: int = 1) -> dict[str, Any]:
    """Return inputs for `scale` copies of the regions, over the scaled hours."""
    pairs = _pairs(scale)
    hours = np.arange(HOURS * scale)
    regions = list(dict.fromkeys(region for region, _, _, _ in pairs))
    technologies = list(dict.fromkeys(t for _, t, _, _ in pairs))
    return {
        "R": np.array(regions),
        "K": np.array(technologies),
        "T": hours,
        "sited": (
            {
                "R": np.array([r for r, _, _, _ in pairs for _ in hours]),
                "K": np.array([k for _, k, _, _ in pairs for _ in hours]),
                "T": np.array([t for _ in pairs for t in hours]),
            },
            np.ones(len(pairs) * len(hours)),
        ),
        "capacity": (
            {
                "R": np.array([r for r, _, _, _ in pairs]),
                "K": np.array([k for _, k, _, _ in pairs]),
            },
            np.array([c for _, _, c, _ in pairs]),
        ),
        "cost": (
            {
                "R": np.array([r for r, _, _, _ in pairs]),
                "K": np.array([k for _, k, _, _ in pairs]),
            },
            np.array([p for _, _, _, p in pairs]),
        ),
        "demand": np.array(
            [
                np.tile(np.array(DEMAND[region.rstrip("0123456789")]), scale)
                for region in regions
            ]
        ),
    }


def reference(data: Mapping[str, Any]) -> float:
    """Return what the sector costs, by merit order over each region and hour."""
    columns, capacity = data["capacity"]
    prices = data["cost"][1]
    total = 0.0
    for position, region in enumerate(data["R"]):
        here = columns["R"] == region
        total += merit_order(capacity[here], prices[here], data["demand"][position])
    return float(total)
