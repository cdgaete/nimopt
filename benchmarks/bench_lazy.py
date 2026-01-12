"""Benchmark lazy solution storage."""

import tempfile
import time

import numpy as np

import nimopt as no
from nimopt.solution_lazy import load_solution, save_solution
from nimopt.solvers import HiGHSSolver


def benchmark(n: int):
    """Benchmark save/load for n x n transport model."""
    n_vars = n * n
    print(f"\n{n}x{n} = {n_vars:,} variables")

    i = no.Set('i', [f'S{k}' for k in range(n)])
    j = no.Set('j', [f'D{k}' for k in range(n)])
    supply = no.Param('supply', [i], np.random.randint(50, 150, n).tolist())
    demand = no.Param('demand', [j], np.random.randint(30, 100, n).tolist())
    cost = no.Param('cost', [i, j], np.random.rand(n, n) * 10)

    m = no.Model(name='transport', sense='minimize')
    x = m.var('x', [i, j], lb=0)
    m.eq('supply', no.Sum(j, x[i, j]) <= supply[i])
    m.eq('demand', no.Sum(i, x[i, j]) >= demand[j])
    m.objective = no.Sum(i, j, cost[i, j] * x[i, j])

    with tempfile.NamedTemporaryFile(suffix='.lp', delete=False) as f:
        lp_file = f.name
    m.to_lp(lp_file)

    solver = HiGHSSolver()
    solver.read_lp(lp_file)
    solver.solve()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        t0 = time.perf_counter()
        save_solution(solver, m, tmpdir)
        save_time = time.perf_counter() - t0
        print(f"  Save: {save_time*1000:.2f}ms")

        # Load (just metadata)
        t0 = time.perf_counter()
        sol2 = load_solution(tmpdir)
        load_time = time.perf_counter() - t0
        print(f"  Load (meta only): {load_time*1000:.2f}ms")

        # Access one variable
        t0 = time.perf_counter()
        _ = sol2.var('x').values
        access_time = time.perf_counter() - t0
        print(f"  Access x.values: {access_time*1000:.2f}ms")

        # Random access
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = sol2.var('x')[np.random.randint(n), np.random.randint(n)]
        rand_time = time.perf_counter() - t0
        print(f"  1000 random accesses: {rand_time*1000:.2f}ms")


if __name__ == "__main__":
    print("Lazy Solution Benchmark")
    print("=" * 40)
    benchmark(100)
    benchmark(500)
    benchmark(1000)
