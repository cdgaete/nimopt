"""Plants shipping to warehouses over a network that is not complete.

A plant reaches a band of nearby warehouses rather than all of them, so the
arcs are a subset of the plant-warehouse product. The cost carries one
coefficient per arc and the flow takes its columns from that parameter, so the
model spends a column per arc and not one per cell.

Supply is twice what a plant's whole band can demand, so no plant binds. Each
warehouse therefore buys from the cheapest plant that reaches it, which is what
`reference` computes. A binding supply would make the optimum a transport LP,
and the corpus would have no answer of its own to check against.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from nimopt import Definition, Sum

PLANTS = 4
WAREHOUSES = 6
BAND = 3
DEMAND = 10.0


def definition(integer: bool = False) -> Definition:
    """The transport model, with no data bound.

    `integer` makes the flow an integer column bounded by `capacity`, so the
    same network states a MILP rather than an LP. The bound is declared only
    then, because a parameter a definition declares is one its data must
    cover.
    """
    d = Definition("transport", sense="min")
    P = d.set("P")
    W = d.set("W")
    cost = d.param("cost", (P, W))
    supply = d.param("supply", (P,))
    demand = d.param("demand", (W,))
    upper = d.param("capacity", (P, W)) if integer else np.inf
    flow = d.var("flow", (P, W), subset=cost, lower=0.0, upper=upper, integer=integer)
    d.eq("supply", Sum(W, flow[P, W]) <= supply[P])
    d.eq("demand", Sum(P, flow[P, W]) >= demand[W])
    d.set_objective(Sum(P, W, cost[P, W] * flow[P, W]))
    return d


def arcs(
    n_plants: int, n_warehouses: int, seed: int = 0
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """One arc per plant and warehouse in its band, with a cost for each.

    A band is drawn from every warehouse but the last, so the last is reached
    by no plant at any size. Its demand row is therefore never stated, which
    is the absence this shape carries and not an accident of the seed.
    """
    rng = np.random.default_rng(seed)
    served = n_warehouses - 1
    offsets = rng.integers(0, served, n_plants)
    plants, warehouses = [], []
    for plant, offset in enumerate(offsets):
        for step in range(BAND):
            plants.append(plant)
            warehouses.append((offset + step) % served)
    return (
        np.array(plants),
        np.array(warehouses),
        rng.uniform(1.0, 9.0, len(plants)),
    )


def data(scale: int = 1, integer: bool = False) -> dict[str, Any]:
    """Inputs for a network `scale` times the base size.

    `integer` adds the capacity the integer flow is bounded by, which the
    definition declares only in that form.
    """
    n_plants, n_warehouses = PLANTS * scale, WAREHOUSES * scale
    plants, warehouses, unit_cost = arcs(n_plants, n_warehouses)
    labels = {
        "P": np.array([f"p{i}" for i in plants]),
        "W": np.array([f"w{i}" for i in warehouses]),
    }
    inputs = {
        "P": np.array([f"p{i}" for i in range(n_plants)]),
        "W": np.array([f"w{i}" for i in range(n_warehouses)]),
        "cost": (labels, unit_cost),
        "supply": np.full(n_plants, 2.0 * BAND * DEMAND),
        "demand": np.full(n_warehouses, DEMAND),
    }
    if integer:
        inputs["capacity"] = (labels, np.full(len(unit_cost), BAND * DEMAND))
    return inputs


def reference(data: Mapping[str, Any]) -> float:
    """What the shipping costs, each warehouse buying its cheapest arc.

    A warehouse no arc reaches states no demand row, so it buys nothing and
    adds nothing.
    """
    columns, unit_cost = data["cost"]
    cheapest = {}
    for warehouse, price in zip(columns["W"], unit_cost):
        cheapest[warehouse] = min(cheapest.get(warehouse, np.inf), price)
    return float(
        sum(
            cheapest[name] * want
            for name, want in zip(data["W"], data["demand"])
            if name in cheapest
        )
    )
