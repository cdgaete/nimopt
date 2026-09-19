"""The arithmetic a model's reference is computed with, independent of nimopt."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def scenarios(
    scenario: Sequence[tuple[str, float, float]], spread: float, scale: int
) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the names, the levels and the probabilities of a scenario table.

    Each entry of `scenario` is a name, a center and a probability. It splits
    into `scale` draws spread by `spread` about its center, each with an equal
    share of its probability. The probabilities sum to one at every scale.
    """
    names, level, weight = [], [], []
    for name, center, share in scenario:
        for k in range(scale):
            names.append(f"{name}{k}")
            level.append(center * (1.0 + spread * (k - (scale - 1) / 2)))
            weight.append(share / scale)
    return names, np.array(level), np.array(weight)


def merit_order(
    capacity: npt.ArrayLike, cost: npt.ArrayLike, load: npt.ArrayLike
) -> float:
    """Return what meeting each load costs, cheapest capacity first.

    `capacity` and `cost` have one entry per unit and `load` one per period.
    Each period is met independently. The result is the optimum of a dispatch
    whose only constraint is that generation meets the load.

    Raises ValueError for a load the capacity cannot meet.
    """
    capacity = np.asarray(capacity, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    order = np.argsort(cost, kind="stable")
    total = 0.0
    for want in np.asarray(load, dtype=np.float64).ravel():
        left = float(want)
        for unit in order:
            taken = min(left, capacity[unit])
            total += taken * cost[unit]
            left -= taken
            if left <= 0.0:
                break
        if left > 1e-9:
            raise ValueError(
                f"the units total {capacity.sum()} against a load of {want}; "
                f"pass a load the capacity meets"
            )
    return total
