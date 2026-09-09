"""Capacity built before the scenario is known, dispatched once it is.

The first stage is `cap`, a capacity per technology carrying no scenario
dimension. The second stage is `p` and `shed`, which carry one. A decision
that cannot depend on the outcome is stated by leaving the scenario dimension
off the variable, so the model is non-anticipative by the shape of `cap`
rather than by a row tying copies of it together.

A scenario states a demand and a fuel price, so `cost` carries the scenario
as well as the technology. The objective is the capital the first stage
commits to plus the second stage weighted by `weight`: the expected cost of
the recourse.

A technology is available in full wherever it is built and nothing couples
one hour to the next, so the recourse is the merit order of the built
capacity against that hour's demand, with the remainder unserved at `voll`.
Written as bands of cumulative capacity, the total separates into one term
per technology, each convex in that band's level, and `reference` minimises
them one at a time.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from nimopt import Definition, Sum

TECHNOLOGY = (("base", 133.0, 8.0), ("mid", 95.0, 25.0), ("peak", 45.0, 70.0))
SCENARIO = (("mild", 0.85, 0.5), ("normal", 1.0, 0.3), ("cold", 1.25, 0.2))
DEMAND = (60.0, 90.0, 130.0, 180.0, 140.0, 80.0)
SPREAD = 0.02
VOLL = 200.0


def definition() -> Definition:
    """The expansion model, with no data bound."""
    d = Definition("expansion", sense="min")
    S, G, T = d.set("S"), d.set("G"), d.set("T")
    capital = d.param("capital", (G,))
    cost = d.param("cost", (S, G))
    demand = d.param("demand", (S, T))
    weight = d.param("weight", (S,))
    voll = d.param("voll", (S,))

    cap = d.var("cap", (G,), lower=0.0)
    p = d.var("p", (S, G, T), lower=0.0)
    shed = d.var("shed", (S, T), lower=0.0)

    d.eq("capacity", p[S, G, T] - cap[G] <= 0.0)
    d.eq("balance", Sum(G, p[S, G, T]) + shed[S, T] == demand[S, T])
    d.set_objective(
        Sum(G, capital[G] * cap[G])
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

    A scenario's level moves its demand and its fuel price together, so a
    cold draw is dear as well as long, and the merit order is the same one in
    every scenario.
    """
    names, level, weight = _scenarios(scale)
    return {
        "S": np.array(names),
        "G": np.array([name for name, _, _ in TECHNOLOGY]),
        "T": np.arange(len(DEMAND) * scale),
        "capital": np.array([value for _, value, _ in TECHNOLOGY]),
        "cost": level[:, None] * np.array([value for _, _, value in TECHNOLOGY]),
        "demand": level[:, None] * np.tile(np.array(DEMAND), scale),
        "weight": weight,
        "voll": np.full(len(names), VOLL),
    }


def _band(
    demand: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
    a: float,
) -> tuple[float, float]:
    """The level of a cumulative capacity band and what that band costs.

    The band's cost is `a * X` less the fuel it displaces, whose weight per
    scenario is already folded into `weight`. That is convex in `X` and turns
    at a demand, so a demand level or nothing built is where it is least.
    """
    order = np.argsort(demand.ravel(), kind="stable")
    level = demand.ravel()[order]
    share = np.repeat(weight, demand.shape[1])[order]
    below = np.cumsum(share * level)
    above = share.sum() - np.cumsum(share)
    spend = a * level - (below + level * above)
    best = int(np.argmin(spend))
    if spend[best] >= 0.0:
        return 0.0, 0.0
    return float(level[best]), float(spend[best])


def reference(data: Mapping[str, Any]) -> float:
    """What the expansion costs, one band of cumulative capacity at a time.

    Each band sits between two technologies, or between the dearest one and
    lost load, and the merit order writes the total as a sum over bands that
    share no variable. Data whose technologies are not a cost frontier, or
    whose bands do not stack, is refused: the separation is what makes this
    the optimum, and neither holds without it.
    """
    weight, voll = data["weight"], data["voll"]
    capital, cost, demand = data["capital"], data["cost"], data["demand"]
    order = np.argsort(cost[0], kind="stable")
    if any(np.argsort(row, kind="stable").tolist() != order.tolist() for row in cost):
        raise ValueError("the merit order is not the same in every scenario")
    capital, cost = capital[order], cost[:, order]

    total = float((weight * voll * demand.sum(axis=1)).sum())
    built = []
    for k, dearer in enumerate(list(cost.T[1:]) + [voll]):
        name = data["G"][order][k]
        step = capital[k] - (capital[k + 1] if k + 1 < len(capital) else 0.0)
        displaced = weight * (dearer - cost[:, k])
        if step <= 0.0:
            raise ValueError(f"capital does not fall from {name} to what follows it")
        if displaced.sum() <= 0.0:
            raise ValueError(f"running cost does not rise from {name} to what follows")
        level, spend = _band(demand, displaced, step)
        built.append(level)
        total += spend
    if built != sorted(built):
        raise ValueError(f"the cumulative capacity does not stack: {built}")
    return total
