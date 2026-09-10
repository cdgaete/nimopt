"""A European capacity expansion network as a case both builders declare.

The `linopy` side is the PyPSA model itself. The comparison is over the model
a modeler runs. Each side reads its input before the measurement starts: the
arrays on one side, the network and its horizon on the other. The build phase
then measures building and not reading a 212 MB file.

The two sides start from different resident baselines. Importing PyPSA costs
about 323 MB and importing numpy costs about 32 MB. The watermark of each
phase is taken over a baseline captured when that phase starts. The difference
is outside the deltas and is reported beside them.
"""

from pathlib import Path

import numpy as np
from compare import Case

HERE = Path(__file__).resolve().parent
LARGE = HERE / "data" / "large"
ARRAYS = LARGE / "eu.npz"
NETWORK = LARGE / "interconnected-transport.nc"

BALANCE = "Bus-nodal_balance"
OUTPUT = "Generator-p"

# linopy adds the objective constant as a column fixed to its value, with no
# matrix entry; nimopt reports the constant as a number. Excluding the column
# compares the two matrices.
CONSTANT_COLUMNS = 1


def _shape(rows, cols, nnz, index_bytes, value_bytes, columns_of, snapshots):
    """Return the fields every side reports about the matrix it built.

    `dense_cols` counts the columns with at least as many nonzeros as the rung
    has snapshots. Those are the capacity columns, with one operating row per
    snapshot. They drive the fill-in of a barrier factorization. The count is
    below the number of nominal columns. A capacity term is absent at every
    hour the component is unavailable.
    """
    per_column = np.bincount(columns_of, minlength=int(cols))
    return {
        "rows": int(rows),
        "cols": int(cols),
        "nnz": int(nnz),
        "matrix_mb": (index_bytes + value_bytes) / 1e6,
        "dense_cols": int((per_column >= snapshots).sum()),
    }


def nimopt(snapshots):
    """Return the nimopt case for the network over its first `snapshots` hours."""
    from pypsa_network import Data, constant, model_from

    data = Data(ARRAYS, snapshots)
    offset = constant(ARRAYS)

    def build():
        model = model_from(data)
        return {"model": model, "session": model.session()}

    def describe(built):
        assembled = built["session"].assembled
        return _shape(
            assembled.n_rows,
            assembled.n_cols,
            assembled.values.size,
            assembled.indices.nbytes,
            assembled.values.nbytes,
            assembled.indices,
            snapshots,
        )

    def solve(built):
        solution = built["session"].solve()
        if solution.status != "optimal":
            raise RuntimeError(
                f"nimopt returned status {solution.status!r}; solve a model "
                f"that returns 'optimal'"
            )
        built["solution"] = solution
        # PyPSA excludes the cost of the existing capacity from its objective
        return float(solution.objective) - offset

    def read(built):
        solution = built["solution"]
        return {
            "primal_sum": float(solution.primal(OUTPUT).values().sum()),
            "dual_sum": float(solution.dual(BALANCE).values().sum()),
        }

    return Case(build=build, describe=describe, solve=solve, read=read)


def linopy(snapshots):
    """Return the PyPSA case for the network over its first `snapshots` hours."""
    import pypsa

    network = pypsa.Network(str(NETWORK))
    network.set_snapshots(network.snapshots[:snapshots])
    network.determine_network_topology()

    def build():
        model = network.optimize.create_model()
        model.matrices.A
        return model

    def describe(model):
        matrix = model.matrices.A
        return _shape(
            matrix.shape[0],
            matrix.shape[1] - CONSTANT_COLUMNS,
            matrix.nnz,
            matrix.indices.nbytes,
            matrix.data.nbytes,
            matrix.indices,
            snapshots,
        )

    def solve(model):
        # io_api="direct" passes the matrix to the solver in memory; the
        # default writes an LP file of about a gigabyte at these horizons,
        # and that measures the disk
        model.solve(
            solver_name="highs",
            io_api="direct",
            output_flag=False,
            log_to_console=False,
        )
        if model.status != "ok":
            raise RuntimeError(
                f"linopy returned status {model.status!r}; solve a model that "
                f"returns 'ok'"
            )
        return float(model.objective.value)

    def read(model):
        primal = np.nansum(model.solution[OUTPUT].values)
        dual = sum(
            float(np.nansum(model.dual[name].values))
            for name in model.dual
            if name.endswith(BALANCE)
        )
        return {"primal_sum": float(primal), "dual_sum": dual}

    return Case(build=build, describe=describe, solve=solve, read=read)
