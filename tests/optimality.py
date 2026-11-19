"""The optimality conditions of a declared model, measured on a solution."""

import numpy as np


def check(model, solution):
    """Assert that a solution meets the optimality conditions of `model`."""
    found = residuals(model, solution)
    assert found["primal"] < 1e-6, found
    assert found["stationarity"] < 1e-6, found
    assert found["wrong_sign"] == 0, found
    assert found["gap"] < 1e-6, found


def residuals(model, solution, tolerance=1e-7):
    """Return how far a solution is from the optimality conditions of `model`."""
    a = model.assemble(singleton_rows="rows")
    x = np.concatenate([solution.primal(n).values() for n in model.variables])
    y = np.concatenate(
        [solution.dual(n, kind="constraint").values() for n in model.constraints]
    )
    d = np.concatenate(
        [solution.dual(n, kind="variable").values() for n in model.variables]
    )
    row = np.repeat(np.arange(a.n_rows), np.diff(a.indptr))
    activity = np.bincount(row, weights=a.values * x[a.indices], minlength=a.n_rows)
    transposed = np.bincount(a.indices, weights=a.values * y[row], minlength=a.n_cols)
    sign = 1.0 if model.sense == "min" else -1.0

    def lower_part(values):
        return sign * np.maximum(sign * values, 0.0)

    def finite(values):
        return np.where(np.isfinite(values), values, 0.0)

    primal = max(
        float(np.max(a.row_lower - activity, initial=0.0)),
        float(np.max(activity - a.row_upper, initial=0.0)),
        float(np.max(a.col_lower - x, initial=0.0)),
        float(np.max(x - a.col_upper, initial=0.0)),
    )
    wrong = 0
    for duals, low, high in (
        (y, a.row_lower, a.row_upper),
        (d, a.col_lower, a.col_upper),
    ):
        wrong += int(((sign * duals > tolerance) & ~np.isfinite(low)).sum())
        wrong += int(((sign * duals < -tolerance) & ~np.isfinite(high)).sum())
    dual_objective = model.objective_constant + sum(
        float(
            (lower_part(duals) * finite(low)).sum()
            + ((duals - lower_part(duals)) * finite(high)).sum()
        )
        for duals, low, high in (
            (y, a.row_lower, a.row_upper),
            (d, a.col_lower, a.col_upper),
        )
    )
    return {
        "primal": primal,
        "stationarity": float(np.abs(a.col_cost - transposed - d).max(initial=0.0)),
        "wrong_sign": wrong,
        "gap": abs(dual_objective - solution.objective)
        / (1.0 + abs(solution.objective)),
    }
