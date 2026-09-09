"""The European network through PyPSA itself, on either optimisation backend.

`bench_pypsa` compares nimopt's restatement of the network with PyPSA's linopy
model. This module compares the two paths a PyPSA user runs: the same
network, the same `n.optimize` accessor, and the backend switched between
`nimopt` and `linopy`. Each phase is the one PyPSA runs — building the model,
handing it to the solver, and assigning the solution and duals back onto the
network — measured with `compare.measure`, which samples resident size
through each phase and writes the curve beside the numbers.

Run one backend per process, so the resident figures carry one library:

    python benchmarks/bench_pypsa_backend.py nimopt
    python benchmarks/bench_pypsa_backend.py linopy --snapshots 24
    python benchmarks/bench_pypsa_backend.py nimopt --build-only
    python benchmarks/bench_pypsa_backend.py nimopt --solver mosek --threads 8
    python benchmarks/bench_pypsa_backend.py nimopt --method barrier --crossover off

`--solver` names the adapter the nimopt backend hands the matrix to; `--method`
speaks nimopt's vocabulary and reaches whichever solver runs. The linopy
backend takes HiGHS alone here, because its method names are HiGHS's.

The network's 2,920 three-hourly snapshots cover 8,760 hours; `--snapshots`
takes the first that many, and the default is every one of them.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from compare import Case, measure

HERE = Path(__file__).resolve().parent
LARGE = HERE / "data" / "large"
NETWORK = LARGE / "interconnected-transport.nc"
TRACES = LARGE / "traces"

METHODS = ("choose", "simplex", "barrier")
CROSSOVER = ("choose", "off", "on")
SOLVERS = ("highs", "gurobi", "mosek")
LINOPY_METHOD = {"choose": "choose", "simplex": "simplex", "barrier": "ipm"}


def load(backend, snapshots, network=NETWORK):
    """The network with its horizon set, ready for either backend to build.

    Reading the file and resolving the topology happen here, before any
    phase is measured, so the build phase measures building alone.
    """
    import pypsa

    pypsa.options.params.optimize.include_objective_constant = True
    started = time.perf_counter()
    n = pypsa.Network(str(network))
    if snapshots is not None:
        n.set_snapshots(n.snapshots[:snapshots])
    n.determine_network_topology()
    print(
        f"loaded {network.name}: {len(n.snapshots)} snapshots, "
        f"backend {backend}, {time.perf_counter() - started:.1f} s",
        flush=True,
    )
    return n


def _dense_columns(columns_of, cols, snapshots):
    """Columns reaching at least one row per snapshot: the capacity columns."""
    per_column = np.bincount(columns_of, minlength=int(cols))
    return int((per_column >= snapshots).sum())


def case(
    n,
    backend,
    method=None,
    threads=None,
    log=False,
    time_limit=None,
    solver="highs",
    crossover=None,
):
    """PyPSA's three phases on `backend`, as the harness measures them.

    A solve stopped at `time_limit` seconds still reports its phase: the
    objective is then `NaN`, the status is carried in the result, and the
    read phase is skipped, so a run bounded for its peak leaves its numbers.
    """
    snapshots = len(n.snapshots)
    options = {}
    if backend == "nimopt":
        if method is not None:
            options["method"] = method
        if crossover is not None:
            options["crossover"] = crossover
        if threads is not None:
            options["threads"] = threads
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        options["log"] = log
    elif backend == "linopy":
        if solver != "highs":
            raise ValueError(f"the linopy backend runs HiGHS here; got {solver!r}")
        if crossover is not None:
            raise ValueError("crossover is set through the nimopt backend here")
        # io_api="direct" hands the matrix to the solver in memory; the
        # default writes an LP file, which measures the disk rather than the
        # solver and reaches gigabytes at the whole horizon
        options["io_api"] = "direct"
        options["log_to_console"] = log
        if method is not None:
            options["solver"] = LINOPY_METHOD[method]
        if threads is not None:
            options["threads"] = threads
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
    else:
        raise ValueError(f"backend is 'nimopt' or 'linopy'; got {backend!r}")

    def build():
        model = n.optimize.create_model(backend=backend)
        if backend == "nimopt":
            assembled = model.model.assemble()
            shape = {
                "rows": int(assembled.n_rows),
                "cols": int(assembled.n_cols),
                "nnz": int(assembled.values.size),
                "matrix_mb": (assembled.indices.nbytes + assembled.values.nbytes) / 1e6,
                "dense_cols": _dense_columns(
                    assembled.indices, assembled.n_cols, snapshots
                ),
            }
            del assembled
        else:
            matrix = model.matrices.A
            shape = {
                "rows": int(matrix.shape[0]),
                "cols": int(matrix.shape[1]),
                "nnz": int(matrix.nnz),
                "matrix_mb": (
                    matrix.indices.nbytes + matrix.nnz * matrix.dtype.itemsize
                )
                / 1e6,
                "dense_cols": _dense_columns(
                    matrix.indices, matrix.shape[1], snapshots
                ),
            }
        return {"model": model, "shape": shape}

    def describe(state):
        return state["shape"]

    def solve(state):
        status, condition = state["model"].solve(solver_name=solver, **options)
        state["status"] = f"{status}, {condition}"
        # linopy reports a time limit as "ok" and hands the iterate over;
        # only an optimum is an answer here, on either side
        if condition != "optimal":
            print(f"solve stopped: {state['status']}", flush=True)
            return float("nan")
        return float(state["model"].objective.value)

    def read(state):
        if not state["status"].endswith("optimal"):
            return {"status": state["status"]}
        n.optimize.assign_solution()
        n.optimize.assign_duals()
        n.optimize.post_processing()
        return {
            "primal_sum": float(np.nansum(n.c.generators.dynamic.p.to_numpy())),
            "dual_sum": float(np.nansum(n.c.buses.dynamic.marginal_price.to_numpy())),
            "network_objective": float(n.objective),
            "status": state["status"],
        }

    return Case(build=build, describe=describe, solve=solve, read=read)


def report_phase(phase, got):
    """One phase's time and resident size, printed as soon as it ends."""
    print(
        f"{phase:6s} {got[f'{phase}_ms'] / 1e3:10.1f} s   "
        f"+{got[f'{phase}_rss_mb']:8.1f} MB over its baseline   "
        f"peak {got[f'{phase}_rss_peak_mb']:8.1f} MB "
        f"at {got[f'{phase}_rss_peak_at_s']:.1f} s"
        + (f"   trace {got[phase + '_trace']}" if phase + "_trace" in got else ""),
        flush=True,
    )
    if phase == "build":
        print(
            f"matrix {got['rows']:,} rows x {got['cols']:,} cols, {got['nnz']:,} "
            f"nonzeros, {got['matrix_mb']:.1f} MB, {got['dense_cols']:,} dense "
            "columns",
            flush=True,
        )


def report(got):
    """The answer and the process peak, once every phase has run."""
    if "status" in got:
        print(f"status {got['status']}")
    if "objective" in got:
        print(f"objective {got['objective']:.4f}")
    print(f"process peak {got['process_rss_mb']:.1f} MB")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("backend", choices=("nimopt", "linopy"))
    parser.add_argument("--snapshots", type=int, default=None, help="first N snapshots")
    parser.add_argument("--build-only", action="store_true", help="skip the solve")
    parser.add_argument("--solver", choices=SOLVERS, default="highs")
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--crossover", choices=CROSSOVER, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--time-limit", type=float, default=None, help="seconds the solver may run"
    )
    parser.add_argument("--log", action="store_true", help="let the solver log")
    parser.add_argument("--traced", action="store_true", help="add a tracemalloc build")
    parser.add_argument("--trace-dir", type=Path, default=TRACES)
    parser.add_argument("--network", type=Path, default=NETWORK)
    args = parser.parse_args(argv)

    n = load(args.backend, args.snapshots, args.network)
    label = "-".join(
        str(part)
        for part in (
            "pypsa-backend",
            args.backend,
            args.solver,
            args.method,
            args.crossover,
            args.threads,
            len(n.snapshots),
        )
        if part is not None
    )
    got = measure(
        case(
            n,
            args.backend,
            args.method,
            args.threads,
            args.log,
            args.time_limit,
            args.solver,
            args.crossover,
        ),
        traced=args.traced,
        solve=not args.build_only,
        trace_dir=args.trace_dir,
        label=label,
        on_phase=report_phase,
    )
    report(got)
    print("RESULT " + json.dumps(got))
    return got


if __name__ == "__main__":
    main(sys.argv[1:])
