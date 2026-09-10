"""A dispatch whose capacity is a profile: one bound per generator and hour.

`dispatch` bounds a generator by a single number. Here `p_max` is a parameter
over generators crossed with snapshots, and the bound has a different value in
every hour. The profile is declared over `(G, T)` and the variable over
`(T, G)`. A bound is read in the dimension order of the variable it bounds,
and both orders resolve to the same columns. Each snapshot is independent, and the
optimum is the merit order against that hour's capacities.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from nimopt import Definition, Sum
from nimopt.models._arithmetic import merit_order

FLEET = ("base", "solar", "peak")
RATING = (100.0, 60.0, 80.0)
COST = (25.0, 0.5, 70.0)
HOURS = 8
LOAD = (60.0, 90.0, 110.0, 150.0, 160.0, 130.0, 100.0, 70.0)


def definition() -> Definition:
    """Return the profiled model, with no data bound."""
    d = Definition("profiled", sense="min")
    T, G = d.set("T"), d.set("G")
    profile = d.param("profile", (G, T))
    cost = d.param("cost", (G,))
    load = d.param("load", (T,))
    gen = d.var("gen", (T, G), lower=0.0, upper=profile)
    d.constraint("balance", Sum(G, gen[T, G]) == load[T])
    d.set_objective(Sum(T, G, cost[G] * gen[T, G]))
    return d


def _profile(scale: int) -> npt.NDArray[np.float64]:
    """Return one capacity row per generator, flat thermal and daylight solar."""
    hours = np.arange(HOURS * scale)
    daylight = np.clip(np.sin(2 * np.pi * (hours % 24 - 6) / 24), 0.0, 1.0)
    rows = []
    for _ in range(scale):
        for name, rating in zip(FLEET, RATING):
            shape = daylight if name == "solar" else np.ones(len(hours))
            rows.append(rating * shape)
    return np.array(rows)


def data(scale: int = 1) -> dict[str, Any]:
    """Return inputs for `scale` copies of the fleet over the scaled hours."""
    hours = np.arange(HOURS * scale)
    fleet = [f"{name}{k}" for k in range(scale) for name in FLEET]
    return {
        "T": hours,
        "G": np.array(fleet),
        "profile": _profile(scale),
        "cost": np.tile(np.array(COST), scale),
        "load": np.tile(np.array(LOAD), scale) * scale,
    }


def reference(data: Mapping[str, Any]) -> float:
    """Return what the dispatch costs, hour by hour against each capacity."""
    profile, cost, load = data["profile"], data["cost"], data["load"]
    return float(
        sum(
            merit_order(profile[:, hour], cost, np.array([want]))
            for hour, want in enumerate(load)
        )
    )
