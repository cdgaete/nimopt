"""A fleet and a battery meeting an hourly load, the hours coupled by the store.

The `state_of_charge` row reads the previous hour through `T.cyclic - 1`,
and the row at the first hour reads the last. The ramp row reads `T - 1`. The
first hour has no predecessor, and that row is absent. One model uses both lag
rules. Every limit is a row, not a bound, and a row has a dual.

`data` gives a fleet whose costs span 50.0 to 55.0 and a store whose round
trip returns `0.95 * 0.93` of what it takes. A megawatt-hour bought at 50.0
comes back as 0.8835 of one and displaces at most 48.59. Shifting energy never
pays, the store stays idle, and the optimum is the hourly merit order. The
ramp limit is the whole capacity, and that row never binds. Data with a wider
spread gives a store that moves energy.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from nimopt import Definition, Sum
from nimopt.models._arithmetic import merit_order

CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.93
FLEET = ("base", "mid", "peak")
CAPACITY = (100.0, 100.0, 100.0)
COST = (50.0, 52.0, 55.0)
HOURS = 24
STORE_MW = 50.0
STORE_MWH = 200.0


def definition() -> Definition:
    """Return the storage model, with no data bound."""
    d = Definition("storage", sense="min")
    T, G, S = d.set("T"), d.set("G"), d.set("S")
    cost = d.param("cost", (G, T))
    capacity = d.param("capacity", (G, T))
    ramp_limit = d.param("ramp_limit", (G, T))
    load = d.param("load", (T,))
    power = d.param("power", (S, T))
    energy = d.param("energy", (S, T))
    charge_eta = d.param("charge_eta", (S, T))
    discharge_eta = d.param("discharge_eta", (S, T))

    gen = d.var("gen", (G, T), lower=0.0)
    charge = d.var("charge", (S, T), lower=0.0)
    discharge = d.var("discharge", (S, T), lower=0.0)
    soc = d.var("soc", (S, T), lower=0.0)

    d.constraint(
        "balance",
        Sum(G, gen[G, T]) + Sum(S, discharge[S, T]) - Sum(S, charge[S, T]) == load[T],
    )
    d.constraint(
        "state_of_charge",
        soc[S, T]
        - soc[S, T.cyclic - 1]
        - charge_eta[S, T] * charge[S, T]
        + discharge_eta[S, T] * discharge[S, T]
        == 0.0,
    )
    d.constraint("generation_limit", gen[G, T] <= capacity[G, T])
    d.constraint("ramp", gen[G, T] - gen[G, T - 1] <= ramp_limit[G, T])
    d.constraint("charge_limit", charge[S, T] <= power[S, T])
    d.constraint("discharge_limit", discharge[S, T] <= power[S, T])
    d.constraint("energy_limit", soc[S, T] <= energy[S, T])
    d.set_objective(Sum(G, T, cost[G, T] * gen[G, T]))
    return d


def data(scale: int = 1) -> dict[str, Any]:
    """Return inputs for `scale` copies of the fleet over the scaled hours.

    The ramp limit is a generator's whole capacity. The ramp rows never bind,
    and the optimum is the hourly merit order.
    """
    hours = np.arange(HOURS * scale)
    fleet = [f"{name}{k}" for k in range(scale) for name in FLEET]
    shape = (len(fleet), len(hours))
    store = (scale, len(hours))
    capacity = np.tile(np.array(CAPACITY), scale)[:, None] * np.ones(shape)
    return {
        "T": hours,
        "G": np.array(fleet),
        "S": np.array([f"b{k}" for k in range(scale)]),
        "cost": np.tile(np.array(COST), scale)[:, None] * np.ones(shape),
        "capacity": capacity,
        "ramp_limit": capacity.copy(),
        "load": (120.0 + 60.0 * np.sin(2 * np.pi * (hours - 18) / 24)) * scale,
        "power": np.full(store, STORE_MW),
        "energy": np.full(store, STORE_MWH),
        "charge_eta": np.full(store, CHARGE_EFFICIENCY),
        "discharge_eta": np.full(store, 1.0 / DISCHARGE_EFFICIENCY),
    }


def reference(data: Mapping[str, Any]) -> float:
    """Return what the dispatch costs, hour by hour against each capacity.

    The store is idle at the optimum. It moves no energy and adds no cost.
    """
    capacity, cost, load = data["capacity"], data["cost"], data["load"]
    return float(
        sum(
            merit_order(capacity[:, hour], cost[:, hour], np.array([want]))
            for hour, want in enumerate(load)
        )
    )
