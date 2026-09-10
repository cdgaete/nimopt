"""The benchmark models as linopy builds them.

Each factory returns a `compare.Case`. The models are those of
`bench_transport.py` and `bench_storage.py`, formulated as linopy requires. A
variable spans the full product of its coordinates and a mask removes the
members the model excludes. nimopt gives the variable a subset to span.

The two sides agree row for row. The ramp block is masked to the hours with a
predecessor, and the `state_of_charge` block rolls. Both follow the absence
rules of those constraints.
"""

import linopy
import numpy as np
import pandas as pd
import xarray as xr
from bench_storage import (
    CHARGE_EFFICIENCY,
    DISCHARGE_EFFICIENCY,
    STORE_MW,
    STORE_MWH,
    UNIT_MW,
    profiles,
)
from bench_transport import network
from compare import Case


def _describe(model):
    """Return the rows, columns and nonzeros of the matrix passed to a solver."""
    matrix = model.matrices.A
    return {
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "matrix_mb": (matrix.data.nbytes + matrix.indices.nbytes) / 1e6,
    }


def _solve(model):
    model.solve(
        solver_name="highs",
        io_api="direct",
        output_flag=False,
        log_to_console=False,
    )
    if model.status != "ok":
        raise RuntimeError(
            f"linopy returned status {model.status!r}; solve a model that returns 'ok'"
        )
    return float(model.objective.value)


def transport(n_plants, n_warehouses, arcs_per_plant, seed=0, integer=False):
    """Return the transport case, with the flow masked over the full product."""
    arc_index = network(n_plants, n_warehouses, arcs_per_plant, seed)
    unit_cost = np.random.default_rng(seed + 1).uniform(1.0, 9.0, arc_index.shape[1])
    per_plant = float(arcs_per_plant)

    P = pd.Index([f"p{i}" for i in range(n_plants)], name="P")
    W = pd.Index([f"w{i}" for i in range(n_warehouses)], name="W")
    arc_mask = np.zeros((n_plants, n_warehouses), dtype=bool)
    arc_mask[arc_index[0], arc_index[1]] = True
    cost_grid = np.zeros((n_plants, n_warehouses))
    cost_grid[arc_index[0], arc_index[1]] = unit_cost
    served = np.bincount(arc_index[1], minlength=n_warehouses).astype(np.float64)

    def build():
        model = linopy.Model()
        flow = model.add_variables(
            lower=0.0,
            upper=per_plant if integer else np.inf,
            coords=[P, W],
            name="flow",
            mask=arc_mask,
            integer=integer,
        )
        one = xr.DataArray(arc_mask.astype(np.float64), coords=[P, W])
        model.add_constraints((one * flow).sum("W") <= per_plant, name="supply")
        model.add_constraints(
            (one * flow).sum("P")
            >= xr.DataArray(np.minimum(served, per_plant), coords=[W]),
            name="demand",
        )
        model.add_objective((xr.DataArray(cost_grid, coords=[P, W]) * flow).sum())
        model.matrices.A
        return model

    def read(model):
        got = {"primal_sum": float(np.nansum(model.solution["flow"].values))}
        # a mixed-integer model has no duals, and both sides read the same
        # fields
        if not integer:
            got["dual_sum"] = float(np.nansum(model.dual["demand"].values))
        return got

    return Case(build=build, describe=_describe, solve=_solve, read=read)


def storage(n_generators, n_storage, n_hours, seed=0):
    """Return the storage case, with a cyclic `state_of_charge` row and ramps."""
    availability, demand, price = profiles(n_generators, n_hours, seed)
    G = pd.Index([f"g{i}" for i in range(n_generators)], name="G")
    S = pd.Index([f"s{i}" for i in range(n_storage)], name="S")
    T = pd.Index(np.arange(n_hours), name="T")

    def build():
        model = linopy.Model()
        gen = model.add_variables(lower=0.0, coords=[G, T], name="gen")
        charge = model.add_variables(lower=0.0, coords=[S, T], name="charge")
        discharge = model.add_variables(lower=0.0, coords=[S, T], name="discharge")
        soc = model.add_variables(lower=0.0, coords=[S, T], name="soc")

        model.add_constraints(
            gen.sum("G") + discharge.sum("S") - charge.sum("S")
            == xr.DataArray(demand, coords=[T]),
            name="balance",
        )
        model.add_constraints(
            soc
            - soc.roll(T=1)
            - CHARGE_EFFICIENCY * charge
            + (1 / DISCHARGE_EFFICIENCY) * discharge
            == 0.0,
            name="state_of_charge",
        )
        model.add_constraints(
            gen <= xr.DataArray(UNIT_MW * availability, coords=[G, T]),
            name="generation_limit",
        )
        # the ramp row at the first hour has no predecessor and is absent
        model.add_constraints(
            gen - gen.shift(T=1) <= 0.5 * UNIT_MW,
            name="ramp",
            mask=xr.DataArray(
                np.tile(np.arange(n_hours) > 0, (n_generators, 1)), coords=[G, T]
            ),
        )
        model.add_constraints(charge <= STORE_MW, name="charge_limit")
        model.add_constraints(discharge <= STORE_MW, name="discharge_limit")
        model.add_constraints(soc <= STORE_MWH, name="energy_limit")
        model.add_objective((xr.DataArray(price, coords=[G, T]) * gen).sum())
        model.matrices.A
        return model

    def read(model):
        return {
            "primal_sum": float(np.nansum(model.solution["gen"].values)),
            "dual_sum": float(np.nansum(model.dual["balance"].values)),
        }

    return Case(build=build, describe=_describe, solve=_solve, read=read)


def transport_milp(n_plants, n_warehouses, arcs_per_plant, seed=0):
    """Return the transport case with integer flows."""
    return transport(n_plants, n_warehouses, arcs_per_plant, seed, integer=True)
