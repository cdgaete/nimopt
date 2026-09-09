"""A PyPSA network as plain arrays, and what its own model carries.

`nimopt` does not import PyPSA. This module reads a network once and writes
two files beside it: the arrays a restatement is built from, and the rows,
columns and nonzeros PyPSA's model holds per constraint family, which is what
the restatement is checked against.

The cycle basis is PyPSA's own rather than recomputed. A different basis spans
the same space and states different rows, so Kirchhoff's law could not be
compared against a basis chosen here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

STATIC = {
    "generators": (
        "bus",
        "carrier",
        "p_nom",
        "p_nom_extendable",
        "p_nom_min",
        "p_nom_max",
        "p_min_pu",
        "p_max_pu",
        "marginal_cost",
        "capital_cost",
        "efficiency",
        "e_sum_min",
        "e_sum_max",
    ),
    "lines": (
        "bus0",
        "bus1",
        "s_nom",
        "s_nom_extendable",
        "s_nom_min",
        "s_nom_max",
        "s_max_pu",
        "x_pu_eff",
        "capital_cost",
    ),
    "links": (
        "bus0",
        "bus1",
        "bus2",
        "bus3",
        "bus4",
        "efficiency",
        "efficiency2",
        "efficiency3",
        "efficiency4",
        "p_nom",
        "p_nom_extendable",
        "p_nom_min",
        "p_nom_max",
        "p_min_pu",
        "p_max_pu",
        "marginal_cost",
        "capital_cost",
    ),
    "stores": (
        "bus",
        "carrier",
        "e_nom",
        "e_nom_extendable",
        "e_nom_min",
        "e_nom_max",
        "e_min_pu",
        "e_max_pu",
        "standing_loss",
        "e_cyclic",
        "capital_cost",
        "marginal_cost",
    ),
    "storage_units": (
        "bus",
        "p_nom",
        "p_nom_extendable",
        "p_nom_min",
        "p_nom_max",
        "p_min_pu",
        "p_max_pu",
        "efficiency_store",
        "efficiency_dispatch",
        "max_hours",
        "standing_loss",
        "cyclic_state_of_charge",
        "capital_cost",
        "marginal_cost",
    ),
    "loads": ("bus", "p_set"),
}

TIME_VARYING = {
    "generators": ("p_min_pu", "p_max_pu"),
    "links": ("efficiency", "p_min_pu", "p_max_pu"),
    "stores": ("e_min_pu", "e_max_pu"),
    "storage_units": ("inflow",),
    "loads": ("p_set",),
}


def _family_nnz(constraint):
    """Nonzeros a family contributes, with repeated terms summed as one.

    A row naming one variable twice holds one nonzero in the matrix, so the
    terms are grouped before they are counted, and a term whose coefficients
    cancel holds none.
    """
    flat = constraint.flat
    frame = pd.DataFrame(
        {"row": flat["labels"], "var": flat["vars"], "coef": flat["coeffs"]}
    )
    frame = frame[(frame["var"] != -1) & (frame["row"] != -1)]
    return int(frame.groupby(["row", "var"])["coef"].sum().ne(0).sum())


def _cycles(network):
    """The `(line, cycle)` basis PyPSA states Kirchhoff's law over."""
    lines = list(network.lines.index)
    blocks = []
    for sub in network.sub_networks.obj:
        basis = getattr(sub, "C", None)
        if basis is None or basis.shape[1] == 0:
            continue
        dense = np.asarray(basis.todense())
        rows = [lines.index(name) for name in sub.branches().index.get_level_values(1)]
        block = np.zeros((len(lines), dense.shape[1]))
        block[rows] = dense
        blocks.append(block)
    if not blocks:
        return np.zeros((len(lines), 0))
    return np.hstack(blocks)


def _column(frame, name):
    """One component column as an array, strings kept as labels.

    A component class that does not carry the column answers an empty label
    for every member, because a network whose links reach two buses states
    nothing at the third rather than failing to be read.
    """
    if name not in frame.columns:
        return np.full(len(frame), "", dtype=str)
    values = frame[name].to_numpy()
    if values.dtype == object or values.dtype.kind in "US":
        return values.astype(str)
    if values.dtype == bool:
        return values.astype(np.int8)
    return values.astype(np.float64)


def _grid(network, component, attr):
    """A time-varying attribute as `(component, snapshot)`, and its names.

    An attribute no member varies over answers no names, and the static
    column stands for every member instead.
    """
    frame = getattr(network, component + "_t")[attr]
    names = np.array(list(frame.columns), dtype=str)
    return names, np.ascontiguousarray(frame.to_numpy().T, dtype=np.float64)


def _put(out, key, value):
    """Store one array, refusing a key another array already holds.

    A static column and a time-varying grid can carry the same attribute
    name, and the grid silently replacing the column would leave a component
    without the value it is stated over.
    """
    if key in out:
        raise ValueError(f"two arrays claim the key {key!r}")
    out[key] = value


def arrays(network):
    """Every array a restatement of `network` is built from."""
    out = {
        "snapshots": np.arange(len(network.snapshots), dtype=np.int64),
        "buses": np.array(list(network.buses.index), dtype=str),
        "carriers": np.array(list(network.carriers.index), dtype=str),
        "carriers_co2": network.carriers.co2_emissions.to_numpy().astype(np.float64),
        "cycles": _cycles(network),
    }
    for column in network.snapshot_weightings.columns:
        out[f"weighting_{column}"] = (
            network.snapshot_weightings[column].to_numpy().astype(np.float64)
        )
    for component, columns in STATIC.items():
        frame = getattr(network, component)
        _put(out, f"{component}_names", np.array(list(frame.index), dtype=str))
        for column in columns:
            _put(out, f"{component}_{column}", _column(frame, column))
    for component, attrs in TIME_VARYING.items():
        for attr in attrs:
            names, grid = _grid(network, component, attr)
            if names.size == 0:
                continue
            _put(out, f"{component}_{attr}_t_names", names)
            _put(out, f"{component}_{attr}_t", grid)
    limits = network.global_constraints
    out["global_sense"] = np.array(list(limits.sense), dtype=str)
    out["global_type"] = np.array(list(limits.type), dtype=str)
    out["global_carrier"] = np.array(list(limits.carrier_attribute), dtype=str)
    out["global_constant"] = limits.constant.to_numpy().astype(np.float64)
    out["global_names"] = np.array(list(limits.index), dtype=str)
    return out


def _reference(network, model):
    """What PyPSA's own model carries, overall and per constraint family."""
    matrix = model.matrices.A
    families = {
        name: {
            "rows": int((np.asarray(c.labels) != -1).sum()),
            "nnz": _family_nnz(c),
        }
        for name, c in model.constraints.items()
    }
    return {
        "pypsa": pypsa.__version__,
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "families": families,
    }


def extract(
    path, out_dir, stem="elec_s_10", solve=True, snapshots=None, reference=True
):
    """Write the arrays and the reference PyPSA's model states for `path`.

    `snapshots` takes the first that many, so a reference can be written at a
    horizon that fits where the whole one would not.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    network = pypsa.Network(str(path))
    if snapshots is not None:
        network.set_snapshots(network.snapshots[:snapshots])
    network.determine_network_topology()
    if not reference:
        np.savez_compressed(out_dir / f"{stem}.npz", **arrays(network))
        return {"stem": stem, "snapshots": len(network.snapshots)}
    model = network.optimize.create_model()

    got = _reference(network, model)
    if solve:
        status, condition = network.optimize.solve_model(
            solver_name="highs", log_to_console=False
        )
        if status != "ok" or condition != "optimal":
            raise RuntimeError(f"the reference network solved {status}/{condition}")
        got["objective"] = float(network.objective)
        got["constant"] = float(network.objective_constant)

    np.savez_compressed(out_dir / f"{stem}.npz", **arrays(network))
    (out_dir / f"{stem}_reference.json").write_text(json.dumps(got, indent=2) + "\n")
    return got


if __name__ == "__main__":
    import sys

    here = Path(__file__).resolve().parent
    got = extract(sys.argv[1], here / "data")
    print(
        f"{got['rows']} rows, {got['cols']} cols, {got['nnz']} nonzeros "
        f"across {len(got['families'])} families"
    )
