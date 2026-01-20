"""Benchmark: LP file vs Direct HiGHS solver."""

import os
import tempfile
import time

import numpy as np

# Check if Rust extension is available
try:
    import nimopt_rust  # noqa: F401
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def create_transport_model(n_sources: int, n_dests: int):
    """Create a transport model with given dimensions."""
    import nimopt as no

    sources = [f"s{i}" for i in range(n_sources)]
    dests = [f"d{j}" for j in range(n_dests)]

    i = no.Set('i', sources)
    j = no.Set('j', dests)

    np.random.seed(42)
    supply_vals = np.random.uniform(100, 1000, n_sources).tolist()
    demand_vals = np.random.uniform(50, 200, n_dests).tolist()

    total_supply = sum(supply_vals)
    total_demand = sum(demand_vals)
    demand_vals = [d * total_supply / total_demand * 0.9 for d in demand_vals]

    cost_vals = np.random.uniform(1, 10, (n_sources, n_dests)).tolist()

    supply = no.Param('supply', [i], supply_vals)
    demand = no.Param('demand', [j], demand_vals)
    cost = no.Param('cost', [i, j], cost_vals)

    m = no.Model(sense='minimize')
    x = m.var('x', [i, j], lb=0)

    m.set_objective(no.Sum(i, j, cost[i, j] * x[i, j]))
    m.eq('supply', no.Sum(j, x[i, j]) <= supply[i])
    m.eq('demand', no.Sum(i, x[i, j]) >= demand[j])

    return m


def bench_lp_file(model, use_rust_lp: bool = True, n_runs: int = 3):
    """Benchmark LP file based solving."""
    from nimopt.solvers import HiGHSSolver

    times = []
    for _ in range(n_runs):
        with tempfile.NamedTemporaryFile(suffix='.lp', delete=False) as f:
            lp_path = f.name

        try:
            t0 = time.perf_counter()
            model.to_lp(lp_path, use_rust=use_rust_lp)
            solver = HiGHSSolver()
            solver.read_lp(lp_path)
            result = solver.solve()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        finally:
            os.unlink(lp_path)

    return np.median(times), result.objective_value


def bench_direct(model, use_rust_direct: bool = True, n_runs: int = 3):
    """Benchmark direct HiGHS solving."""
    from nimopt.solvers import HiGHSDirectSolver

    times = []
    for _ in range(n_runs):
        solver = HiGHSDirectSolver(use_rust=use_rust_direct)
        t0 = time.perf_counter()
        solver.load_model(model)
        result = solver.solve()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times), result.objective_value


def run_benchmark():
    """Run benchmark comparing all approaches."""
    print("=" * 75)
    print("Benchmark: LP File vs Direct HiGHS Solver")
    print(f"Rust extension available: {HAS_RUST}")
    print("=" * 75)

    sizes = [
        (10, 10),
        (30, 30),
        (100, 100),
        (200, 200),
        (300, 300),
    ]

    print("\n" + "-" * 75)
    print(f"{'Size':<10} {'Vars':<8} {'LP(Py)':<12} {'LP(Rust)':<12} "
          f"{'Direct(Py)':<12} {'Direct(Rust)':<12}")
    print("-" * 75)

    for n_src, n_dst in sizes:
        n_vars = n_src * n_dst
        model = create_transport_model(n_src, n_dst)

        lp_py, _ = bench_lp_file(model, use_rust_lp=False)
        lp_rust, _ = bench_lp_file(model, use_rust_lp=True) if HAS_RUST else (0, 0)
        direct_py, _ = bench_direct(model, use_rust_direct=False)
        if HAS_RUST:
            direct_rust, _ = bench_direct(model, use_rust_direct=True)
        else:
            direct_rust = 0

        print(f"{n_src}x{n_dst:<6} {n_vars:<8} "
              f"{lp_py*1000:>8.1f} ms  "
              f"{lp_rust*1000:>8.1f} ms  "
              f"{direct_py*1000:>8.1f} ms  "
              f"{direct_rust*1000:>8.1f} ms")

    print("-" * 75)
    print("\nSummary for 300x300 (90k vars):")
    model = create_transport_model(300, 300)

    lp_py, _ = bench_lp_file(model, use_rust_lp=False)
    lp_rust, _ = bench_lp_file(model, use_rust_lp=True)
    direct_py, _ = bench_direct(model, use_rust_direct=False)
    direct_rust, _ = bench_direct(model, use_rust_direct=True)

    print(f"  LP (Python):      {lp_py*1000:>8.1f} ms")
    print(f"  LP (Rust):        {lp_rust*1000:>8.1f} ms")
    print(f"  Direct (Python):  {direct_py*1000:>8.1f} ms")
    print(f"  Direct (Rust):    {direct_rust*1000:>8.1f} ms  <- FASTEST")
    print(f"\n  Speedup vs LP(Py):   {lp_py/direct_rust:.1f}x")
    print(f"  Speedup vs LP(Rust): {lp_rust/direct_rust:.2f}x")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
