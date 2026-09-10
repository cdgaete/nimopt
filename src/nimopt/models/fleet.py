"""One dispatch, declared as a variable per unit rather than one over a product.

`dispatch` states a single variable over snapshots crossed with generators.
This states the same problem as one variable per unit over the snapshots
alone, and adds their terms into one balance row. The answer is the same merit
order; what differs is the cost of declaring it, which is the axis this model
stresses.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from nimopt import Definition, Sum
from nimopt.models._arithmetic import merit_order

FLEET = (("g0", 100.0, 12.0), ("g1", 80.0, 30.0), ("g2", 90.0, 55.0))
LOAD = (90.0, 140.0, 200.0, 60.0)


def units(scale: int) -> list[str]:
    """The unit names this model declares a variable for, at a scale."""
    return [f"{name}_{k}" for k in range(scale) for name, _, _ in FLEET]


def definition(scale: int = 1) -> Definition:
    """The fleet model, with no data bound.

    The number of variables is a property of the declaration rather than of
    the data, so the scale is stated here as well as in `data`.
    """
    d = Definition("fleet", sense="min")
    T = d.set("T")
    load = d.param("load", (T,))
    generation, spend = None, None
    for name in units(scale):
        capacity = d.param(f"p_max_{name}", (T,))
        price = d.param(f"cost_{name}", (T,))
        unit = d.var(name, (T,), lower=0.0, upper=capacity)
        generation = unit[T] if generation is None else generation + unit[T]
        term = price[T] * unit[T]
        spend = term if spend is None else spend + term
    d.constraint("balance", generation == load[T])
    d.set_objective(Sum(T, spend))
    return d


def data(scale: int = 1) -> dict[str, Any]:
    """Inputs for `scale` copies of the fleet over `len(LOAD) * scale` hours."""
    hours = np.arange(len(LOAD) * scale)
    inputs = {"T": hours, "load": np.tile(np.array(LOAD), scale) * scale}
    for k in range(scale):
        for name, capacity, price in FLEET:
            inputs[f"p_max_{name}_{k}"] = np.full(len(hours), capacity)
            inputs[f"cost_{name}_{k}"] = np.full(len(hours), price)
    return inputs


def reference(data: Mapping[str, Any]) -> float:
    """What the dispatch costs, by merit order over each hour."""
    names = [key[len("p_max_") :] for key in data if key.startswith("p_max_")]
    capacity = np.array([data[f"p_max_{name}"][0] for name in names])
    cost = np.array([data[f"cost_{name}"][0] for name in names])
    return merit_order(capacity, cost, data["load"])
