"""A commitment fixed before the scenario is known, dispatched once it is.

`commitment` decides which units are on against one demand. Here the demand
and the fuel price are a scenario, and the on-off decision is taken before
either is revealed: `on` is indexed by hour and unit alone, while `p` and
`shed` carry the scenario. One commitment therefore has to serve every
scenario, which is what makes the binary a hedge rather than a schedule.

A committed unit runs between its minimum and its maximum, and there is
nowhere to put unwanted energy. A unit whose minimum exceeds the demand of
the mildest scenario therefore cannot be committed at all, however cheap it
is to run in the others: the row it would break is that scenario's balance.

Nothing couples one hour to the next, so the commitment is chosen hour by
hour and `reference` enumerates every on-off subset of the fleet, scoring
each by its expected recourse across the scenarios. Three units make eight
subsets, which is exact and cheap at every scale the corpus uses.
"""

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from nimopt import Definition, Sum

FLEET = (
    ("base", 120.0, 40.0, 20.0, 400.0),
    ("mid", 80.0, 20.0, 45.0, 150.0),
    ("peak", 60.0, 0.0, 90.0, 40.0),
)
SCENARIO = (("mild", 0.85, 0.5), ("normal", 1.0, 0.3), ("cold", 1.25, 0.2))
DEMAND = (45.0, 140.0, 215.0, 95.0)
SPREAD = 0.02
VOLL = 300.0


def definition() -> Definition:
    """The recourse model, with no data bound."""
    d = Definition("recourse", sense="min")
    S, G, T = d.set("S"), d.set("G"), d.set("T")
    p_max = d.param("p_max", (G,))
    p_min = d.param("p_min", (G,))
    no_load = d.param("no_load", (G,))
    cost = d.param("cost", (S, G))
    demand = d.param("demand", (S, T))
    weight = d.param("weight", (S,))
    voll = d.param("voll", (S,))

    on = d.var("on", (T, G), lower=0.0, upper=1.0, integer=True)
    p = d.var("p", (S, G, T), lower=0.0)
    shed = d.var("shed", (S, T), lower=0.0)

    d.eq("capacity", p[S, G, T] - p_max[G] * on[T, G] <= 0.0)
    d.eq("minimum", p[S, G, T] - p_min[G] * on[T, G] >= 0.0)
    d.eq("balance", Sum(G, p[S, G, T]) + shed[S, T] == demand[S, T])
    d.set_objective(
        Sum(T, G, no_load[G] * on[T, G])
        + Sum(S, G, T, weight[S] * cost[S, G] * p[S, G, T])
        + Sum(S, T, weight[S] * voll[S] * shed[S, T])
    )
    return d


def _scenarios(
    scale: int,
) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The scenario names, their level and their probability, at a scale.

    Each of `SCENARIO` splits into `scale` draws spread by `SPREAD` about its
    level and sharing its probability, so the probabilities sum to one at
    every scale and scale 1 is the three the module names.
    """
    names, level, weight = [], [], []
    for name, centre, share in SCENARIO:
        for k in range(scale):
            names.append(f"{name}{k}")
            level.append(centre * (1.0 + SPREAD * (k - (scale - 1) / 2)))
            weight.append(share / scale)
    return names, np.array(level), np.array(weight)


def data(scale: int = 1) -> dict[str, Any]:
    """Inputs for `3 * scale` scenarios over `len(DEMAND) * scale` hours.

    A scenario's level moves its demand and its fuel price together, so the
    merit order is the same one in every scenario. The first hour is 45.0
    before the level is applied, so the mildest scenario asks for 38.25 there
    and the base unit, whose minimum is 40.0, cannot be committed in it. The
    fleet carries 260.0 against a coldest third hour of 268.75, so that hour
    sheds whatever is committed.
    """
    names, level, weight = _scenarios(scale)
    return {
        "S": np.array(names),
        "G": np.array([name for name, _, _, _, _ in FLEET]),
        "T": np.arange(len(DEMAND) * scale),
        "p_max": np.array([value for _, value, _, _, _ in FLEET]),
        "p_min": np.array([value for _, _, value, _, _ in FLEET]),
        "cost": level[:, None] * np.array([value for _, _, _, value, _ in FLEET]),
        "no_load": np.array([value for _, _, _, _, value in FLEET]),
        "demand": level[:, None] * np.tile(np.array(DEMAND), scale),
        "weight": weight,
        "voll": np.full(len(names), VOLL),
    }


def _serve(
    up: Sequence[int],
    want: float,
    p_max: npt.NDArray[np.float64],
    p_min: npt.NDArray[np.float64],
    cost: npt.NDArray[np.float64],
    voll: float,
) -> float:
    """What a committed subset costs to meet `want`, cheapest headroom first.

    Every committed unit runs at least its minimum; the remainder is filled
    from the cheapest headroom and what no unit reaches is shed at `voll`.
    """
    spend = sum(p_min[g] * cost[g] for g in up)
    left = want - sum(p_min[g] for g in up)
    for g in sorted(up, key=lambda g: cost[g]):
        taken = min(left, p_max[g] - p_min[g])
        spend += taken * cost[g]
        left -= taken
    return spend + left * voll


def reference(data: Mapping[str, Any]) -> float:
    """The cheapest commitment per hour, over every on-off subset.

    A subset whose minimum output exceeds the mildest scenario's demand
    breaks that scenario's balance and is not a commitment at all, so it is
    passed over. Committing nothing always serves, by shedding, so every hour
    has a subset to compare against.
    """
    weight, voll = data["weight"], data["voll"]
    p_max, p_min, no_load = data["p_max"], data["p_min"], data["no_load"]
    cost, demand = data["cost"], data["demand"]
    units = range(len(p_max))
    total = 0.0
    for want in demand.T:
        best = np.inf
        for mask in itertools.product((0, 1), repeat=len(p_max)):
            up = [g for g in units if mask[g]]
            if sum(p_min[g] for g in up) > want.min() + 1e-9:
                continue
            spend = sum(no_load[g] for g in up)
            for s, need in enumerate(want):
                spend += weight[s] * _serve(up, need, p_max, p_min, cost[s], voll[s])
            best = min(best, spend)
        total += best
    return float(total)
