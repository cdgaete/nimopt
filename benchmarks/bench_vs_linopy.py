"""nimopt against linopy: what the same model costs each of them to build.

Three models over four axes. Transport puts a variable on a sparse arc
network — a subset of the product in nimopt, a masked full product in linopy. Storage
couples neighbouring hours through a ramp limit and a cyclic state of charge.
The integer transport model states the same network as a MILP. Every case
also reads its primals and duals back onto their labels, which is the fourth
axis.

Each side runs in a process of its own, and a row is printed only once the
two agree on rows, columns, nonzeros and the solved objective: a comparison
of two different problems is worth nothing.
"""

from pathlib import Path

import bench_pypsa
import linopy_models
import numpy as np
from bench_storage import build as storage_build
from bench_storage import profiles
from bench_transport import build as transport_build
from bench_transport import network
from compare import Case, run


def _describe(state):
    """Rows, columns and nonzeros of the matrix a solver would be given."""
    assembled = state["session"].assembled
    return {
        "rows": assembled.n_rows,
        "cols": assembled.n_cols,
        "nnz": int(assembled.values.size),
        "matrix_mb": (assembled.indices.nbytes + assembled.values.nbytes) / 1e6,
    }


def _solve(state):
    solution = state["session"].solve()
    if solution.status != "optimal":
        raise RuntimeError(f"nimopt returned status {solution.status!r}")
    state["solution"] = solution
    return float(solution.objective)


def _reader(variable, constraint, duals=True):
    def read(state):
        solution = state["solution"]
        got = {"primal_sum": float(solution.primal(variable).values().sum())}
        # a mixed-integer model carries no duals: the seam refuses the pair,
        # and the number both sides reported for one was zero
        if duals:
            got["dual_sum"] = float(solution.dual(constraint).values().sum())
        return got

    return read


def transport(n_plants, n_warehouses, arcs_per_plant, seed=0, integer=False):
    """Flow over an arc network, as a variable spanning the arcs."""
    arc_index = network(n_plants, n_warehouses, arcs_per_plant, seed)
    unit_cost = np.random.default_rng(seed + 1).uniform(1.0, 9.0, arc_index.shape[1])

    def build():
        model = transport_build(
            n_plants, n_warehouses, arcs_per_plant, arc_index, unit_cost, integer
        )
        return {"model": model, "session": model.session()}

    return Case(
        build=build,
        describe=_describe,
        solve=_solve,
        read=_reader("flow", "demand", duals=not integer),
    )


def storage(n_generators, n_storage, n_hours, seed=0):
    """Dispatch with a cyclic state of charge and a ramp limit."""
    availability, demand, price = profiles(n_generators, n_hours, seed)

    def build():
        model = storage_build(
            n_generators, n_storage, n_hours, availability, demand, price
        )
        return {"model": model, "session": model.session()}

    return Case(
        build=build,
        describe=_describe,
        solve=_solve,
        read=_reader("gen", "balance"),
    )


def transport_milp(n_plants, n_warehouses, arcs_per_plant, seed=0):
    """The transport model with integer flows."""
    return transport(n_plants, n_warehouses, arcs_per_plant, seed, integer=True)


CASES = {
    "pypsa": {"nimopt": bench_pypsa.nimopt, "linopy": bench_pypsa.linopy},
    "transport": {"nimopt": transport, "linopy": linopy_models.transport},
    "storage": {"nimopt": storage, "linopy": linopy_models.storage},
    "transport_milp": {
        "nimopt": transport_milp,
        "linopy": linopy_models.transport_milp,
    },
}

SIZES = {
    "pypsa": (
        dict(snapshots=24),
        dict(snapshots=192),
        dict(snapshots=730),
        dict(snapshots=2920),
    ),
    "transport": (
        dict(n_plants=200, n_warehouses=100, arcs_per_plant=10),
        dict(n_plants=2000, n_warehouses=500, arcs_per_plant=20),
        dict(n_plants=10000, n_warehouses=2000, arcs_per_plant=40),
    ),
    "storage": (
        dict(n_generators=10, n_storage=2, n_hours=168),
        dict(n_generators=40, n_storage=8, n_hours=720),
        dict(n_generators=80, n_storage=20, n_hours=8760),
    ),
    "transport_milp": (
        dict(n_plants=200, n_warehouses=100, arcs_per_plant=10),
        dict(n_plants=2000, n_warehouses=500, arcs_per_plant=20),
    ),
}


def agree(sides, sizes):
    """The shape and objective both builders reached, or a raised mismatch.

    Rows, columns and nonzeros are compared at every rung. A rung neither
    side solved is compared on its shape alone, because what a build costs
    does not wait on a solve that will not finish.
    """
    nimopt, linopy = sides["nimopt"], sides["linopy"]
    for field in ("rows", "cols", "nnz"):
        if nimopt[field] != linopy[field]:
            raise ValueError(
                f"at {sizes} the two builders state different problems: "
                f"{field} is {nimopt[field]} for nimopt and {linopy[field]} for linopy"
            )
    solved = ["objective" in side for side in (nimopt, linopy)]
    if not any(solved):
        return nimopt
    if not all(solved):
        raise ValueError(
            f"at {sizes} one builder solved and the other did not; a rung is "
            f"solved on both sides or on neither"
        )
    if abs(nimopt["objective"] - linopy["objective"]) > 1e-6 * max(
        1.0, abs(nimopt["objective"])
    ):
        raise ValueError(
            f"at {sizes} the two builders solve to different objectives: "
            f"{nimopt['objective']} and {linopy['objective']}"
        )
    return nimopt


def compare(case, sizes, traced=True, solve=True, trace_dir=None):
    """Both builders on one case at one size, each in a process of its own."""
    sides = {
        side: run(case, side, sizes, traced, solve, trace_dir)
        for side in ("nimopt", "linopy")
    }
    agree(sides, sizes)
    return sides


# How many of a case's rungs are solved. A rung past this is built and
# measured but not solved, because what a build costs does not wait on a
# solve that will not finish. A case named here solves at least one rung.
SOLVED = {"pypsa": 3}


def solves(case, rung):
    """Whether the rung at this position is solved on both sides."""
    return rung < SOLVED.get(case, len(SIZES[case]))


TRACES = Path(__file__).resolve().parent / "data" / "large" / "traces"


def sweep(cases=None, traced=True, trace_dir=TRACES):
    """Every rung of every case, printed as both builders measured it.

    Each phase's resident size through time is written beside the table, so
    a peak is read off the curve rather than taken on trust.
    """
    header = (
        f"{'case':>8} {'size':>20} {'side':>7} {'rows':>10} {'cols':>10} "
        f"{'nnz':>11} {'nnz/col':>8} {'dense':>7} {'matrix':>9} "
        f"{'build RSS':>10} {'solve RSS':>10} {'solve peak':>11} "
        f"{'peak at s':>10} {'peak RSS':>10} {'build ms':>10} "
        f"{'solve ms':>10} {'objective':>22}"
    )
    for case in cases or SIZES:
        print(f"\n{header}")
        solved_any = False
        for rung, sizes in enumerate(SIZES[case]):
            wanted = solves(case, rung)
            sides = compare(
                case, sizes, traced=traced, solve=wanted, trace_dir=trace_dir
            )
            solved_any = solved_any or "objective" in sides["nimopt"]
            label = "x".join(str(v) for v in sizes.values())
            for side in ("nimopt", "linopy"):
                got = sides[side]
                timed = "solve_ms" in got
                solve_ms = f"{got['solve_ms']:.1f}" if timed else "not solved"
                value = f"{got['objective']:,.2f}" if "objective" in got else ""
                per_col = got["nnz"] / max(1, got["cols"])
                dense = got.get("dense_cols", "")
                solved_here = "solve_rss_mb" in got
                solve_rss = f"{got['solve_rss_mb']:.1f}M" if solved_here else ""
                peak = f"{got['solve_rss_peak_mb']:.1f}M" if solved_here else ""
                peak_at = f"{got['solve_rss_peak_at_s']:.1f}" if solved_here else ""
                print(
                    f"{case:>8} {label:>20} {side:>7} {got['rows']:>10} "
                    f"{got['cols']:>10} {got['nnz']:>11} {per_col:>8.2f} "
                    f"{dense:>7} {got['matrix_mb']:>8.1f}M "
                    f"{got['build_rss_mb']:>9.1f}M {solve_rss:>10} "
                    f"{peak:>11} {peak_at:>10} "
                    f"{got['process_rss_mb']:>9.1f}M {got['build_ms']:>10.1f} "
                    f"{solve_ms:>10} {value:>22}"
                )
        if not solved_any:
            raise RuntimeError(
                f"no rung of {case!r} was solved; a shape both builders agree "
                f"on is not evidence they state the same problem"
            )


if __name__ == "__main__":
    import sys

    sweep(sys.argv[1:] or None)
