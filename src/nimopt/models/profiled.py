"""A dispatch whose capacity is a profile: one bound per generator and hour.

`dispatch` bounds a generator by a single number. Here `p_max` is a parameter
over generators crossed with snapshots, so a solar unit is bounded by daylight
and a thermal one by its rating, and the bound reaches a different value in
every hour.

The profile is stated over `(G, T)` and the variable is over `(T, G)`; a bound
is read in the dimension order of the variable it bounds, so both spellings
reach the same columns.

Each snapshot is still independent, so the optimum is the merit order against
that hour's capacities.
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
    """The profiled model, with no data bound."""
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
    """One capacity row per generator: flat for thermal, daylight for solar."""
    hours = np.arange(HOURS * scale)
    daylight = np.clip(np.sin(2 * np.pi * (hours % 24 - 6) / 24), 0.0, 1.0)
    rows = []
    for _ in range(scale):
        for name, rating in zip(FLEET, RATING):
            shape = daylight if name == "solar" else np.ones(len(hours))
            rows.append(rating * shape)
    return np.array(rows)


def data(scale: int = 1) -> dict[str, Any]:
    """Inputs for `scale` copies of the fleet over `HOURS * scale` hours."""
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
    """What the dispatch costs, hour by hour against that hour's capacities."""
    profile, cost, load = data["profile"], data["cost"], data["load"]
    return float(
        sum(
            merit_order(profile[:, hour], cost, np.array([want]))
            for hour, want in enumerate(load)
        )
    )
