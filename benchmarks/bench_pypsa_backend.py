"""The European network through PyPSA itself, on either optimization backend.

`bench_pypsa` compares the nimopt model of the network with the PyPSA linopy
model. This module compares the two paths a PyPSA user runs: the same network,
the same `n.optimize` accessor, and the backend switched between `nimopt` and
`linopy`. The phases are the phases PyPSA runs: building the model, passing it
to the solver, and assigning the solution and the duals back onto the network.
`compare.measure` samples resident size through each phase and writes the
curve beside the numbers.

Run one backend per process, and the resident figures then cover one library:

    python benchmarks/bench_pypsa_backend.py nimopt
    python benchmarks/bench_pypsa_backend.py linopy --snapshots 24
    python benchmarks/bench_pypsa_backend.py nimopt --build-only
    python benchmarks/bench_pypsa_backend.py nimopt --solver mosek --threads 8
    python benchmarks/bench_pypsa_backend.py nimopt --method barrier --crossover off

`--solver` selects the adapter the nimopt backend passes the matrix to.
`--method` uses the nimopt option names and applies to every solver. The
linopy backend runs HiGHS alone here, and its method names are the HiGHS names.

The 2,920 three-hourly snapshots of the network cover 8,760 hours.
`--snapshots` takes the first N, and the default is every snapshot.
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
    """Return the network with its horizon set, for either backend to build.

    The file is read and the topology is resolved here, before any phase is
    measured. The build phase then measures building alone.
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
    """Return the number of columns with at least one nonzero per snapshot."""
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
    """Return the three PyPSA phases on `backend`, as the harness measures them.

    A solve stopped at `time_limit` seconds reports its phase. The objective is
    then `NaN`, the result contains the status, and the read phase is skipped.
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
            raise ValueError(
                "crossover is not supported by the linopy backend; run the "
                "nimopt backend"
            )
        # io_api="direct" passes the matrix to the solver in memory; the
        # default writes an LP file of several gigabytes at the whole horizon,
        # and that measures the disk
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

    def describe(built):
        return built["shape"]

    def solve(built):
        status, condition = built["model"].solve(solver_name=solver, **options)
        built["status"] = f"{status}, {condition}"
        # linopy reports a time limit as "ok" and returns the iterate; only
        # status optimal is accepted here, on either side
        if condition != "optimal":
            print(f"solve stopped: {built['status']}", flush=True)
            return float("nan")
        return float(built["model"].objective.value)

    def read(built):
        if not built["status"].endswith("optimal"):
            return {"status": built["status"]}
        n.optimize.assign_solution()
        n.optimize.assign_duals()
        n.optimize.post_processing()
        return {
            "primal_sum": float(np.nansum(n.c.generators.dynamic.p.to_numpy())),
            "dual_sum": float(np.nansum(n.c.buses.dynamic.marginal_price.to_numpy())),
            "network_objective": float(n.objective),
            "status": built["status"],
        }

    return Case(build=build, describe=describe, solve=solve, read=read)


def report_phase(phase, got):
    """Print the time and the resident size of one phase."""
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
    """Print the status, the objective and the process peak."""
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
    parser.add_argument("--log", action="store_true", help="print the solver log")
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
