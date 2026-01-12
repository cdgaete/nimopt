"""Benchmark solution extraction."""

import tempfile
import time

import numpy as np

import nimopt as no
from nimopt.solution import extract_solution_python, write_solution_csv
from nimopt.solvers import HiGHSSolver


def benchmark(n_sources: int, n_destinations: int, n_runs: int = 3):
    """Run benchmark."""
    n_vars = n_sources * n_destinations
    print(f"Benchmark: {n_sources}x{n_destinations} = {n_vars} variables")

    # Build model
    i = no.Set('i', [f'S{k}' for k in range(n_sources)])
    j = no.Set('j', [f'D{k}' for k in range(n_destinations)])
    supply_data = np.random.randint(50, 150, n_sources).tolist()
    demand_data = np.random.randint(30, 100, n_destinations).tolist()
    supply = no.Param('supply', [i], supply_data)
    demand = no.Param('demand', [j], demand_data)
    cost = no.Param('cost', [i, j], np.random.rand(n_sources, n_destinations) * 10)

    m = no.Model(name='transport', sense='minimize')
    x = m.var('x', [i, j], lb=0)
    m.eq('supply', no.Sum(j, x[i, j]) <= supply[i])
    m.eq('demand', no.Sum(i, x[i, j]) >= demand[j])
    m.objective = no.Sum(i, j, cost[i, j] * x[i, j])

    # Export and solve
    with tempfile.NamedTemporaryFile(suffix='.lp', delete=False) as f:
        lp_file = f.name
    m.to_lp(lp_file)

    solver = HiGHSSolver()
    solver.read_lp(lp_file)
    t0 = time.perf_counter()
    solver.solve()
    solve_time = time.perf_counter() - t0
    print(f"  Solve time: {solve_time*1000:.1f}ms")

    # Extract solution
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sol = extract_solution_python(solver, m)
        times.append(time.perf_counter() - t0)
    avg_time = sum(times) / len(times)
    print(f"  Extract (Python): {avg_time*1000:.2f}ms")

    # Write CSV
    out_dir = '/tmp/bench_sol'
    t0 = time.perf_counter()
    write_solution_csv(sol, out_dir)
    write_time = time.perf_counter() - t0
    print(f"  Write CSV: {write_time*1000:.2f}ms")

    return avg_time


if __name__ == "__main__":
    print("Solution Extraction Benchmark (Python - no parsing)")
    print("=" * 55)
    benchmark(10, 10)
    benchmark(50, 50)
    benchmark(100, 100)
    benchmark(200, 200)
    benchmark(500, 500)
