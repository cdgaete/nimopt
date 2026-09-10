"""The arithmetic a model's reference is computed with, independent of nimopt."""

import numpy as np
import numpy.typing as npt


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
