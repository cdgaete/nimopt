"""A European capacity expansion network as a case both builders state.

The `linopy` side is PyPSA's own model rather than a restatement of it, so
what is compared is what a modeller would actually run. Each side reads its
input before the measurement begins — the arrays on one, the network and its
horizon on the other — so the build phase measures building and not reading a
212 MB file.

The two sides start from different resident baselines, because importing PyPSA
costs about 323 MB against numpy's 32 MB. Every phase's watermark is taken
over a baseline captured when that phase begins, so the difference sits
outside the deltas and is reported beside them rather than corrected for.
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

# linopy carries the objective's constant as a column fixed to its value,
# holding no matrix entry; nimopt states that constant as a number. Excluding
# it compares the two matrices rather than the two ways of holding a scalar.
CONSTANT_COLUMNS = 1


def _shape(rows, cols, nnz, index_bytes, value_bytes, columns_of, snapshots):
    """The fields every side reports about the matrix it built.

    `dense_cols` counts the columns holding at least as many nonzeros as the
    rung has snapshots. Those are the capacity columns, which reach one
    operating row per snapshot, and they are what drives fill-in in a barrier
    solver's factorization. The count sits a little under the number of
    nominal columns because a capacity term drops at every hour its component
    is unavailable.
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
    """The network as `nimopt` states it, over its first `snapshots` hours."""
    from pypsa_network import Data, constant, model_from

    data = Data(ARRAYS, snapshots)
    offset = constant(ARRAYS)

    def build():
        model = model_from(data)
        return {"model": model, "session": model.session()}

    def describe(state):
        assembled = state["session"].assembled
        return _shape(
            assembled.n_rows,
            assembled.n_cols,
            assembled.values.size,
            assembled.indices.nbytes,
            assembled.values.nbytes,
            assembled.indices,
            snapshots,
        )

    def solve(state):
        solution = state["session"].solve()
        if solution.status != "optimal":
            raise RuntimeError(f"nimopt returned status {solution.status!r}")
        state["solution"] = solution
        # PyPSA reports its objective less what the existing capacity costs
        return float(solution.objective) - offset

    def read(state):
        solution = state["solution"]
        return {
            "primal_sum": float(solution.primal(OUTPUT).values().sum()),
            "dual_sum": float(solution.dual(BALANCE).values().sum()),
        }

    return Case(build=build, describe=describe, solve=solve, read=read)


def linopy(snapshots):
    """The network as PyPSA states it, over its first `snapshots` hours."""
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
        # io_api="direct" hands the matrix to the solver in memory; the
        # default writes an LP file, which measures the disk rather than the
        # solver and reaches a gigabyte at the horizons this sweep runs
        model.solve(
            solver_name="highs",
            io_api="direct",
            output_flag=False,
            log_to_console=False,
        )
        if model.status != "ok":
            raise RuntimeError(f"linopy returned status {model.status!r}")
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
