"""Benchmark: nimopt vs linopy."""

import sys
import time

import numpy as np

# Suppress linopy solver output
import os
os.environ['HIGHS_OUTPUT'] = '0'


def bench_linopy(n_src, n_dst, supply_vals, demand_vals, cost_vals, n_runs=3):
    """Benchmark linopy."""
    import linopy
    import xarray as xr

    sources = [f's{i}' for i in range(n_src)]
    dests = [f'd{j}' for j in range(n_dst)]

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        m = linopy.Model()
        x = m.add_variables(lower=0, coords=[sources, dests], name='x')
        supply = xr.DataArray(supply_vals, coords=[sources], dims=['dim_0'])
        demand = xr.DataArray(demand_vals, coords=[dests], dims=['dim_1'])
        cost = xr.DataArray(cost_vals, coords=[sources, dests],
                           dims=['dim_0', 'dim_1'])
        m.add_objective((cost * x).sum())
        m.add_constraints(x.sum('dim_1') <= supply, name='supply')
        m.add_constraints(x.sum('dim_0') >= demand, name='demand')
        m.solve(solver_name='highs', log_fn=None, io_api='direct')
        times.append(time.perf_counter() - t0)
    return np.median(times)


def bench_nimopt(n_src, n_dst, supply_vals, demand_vals, cost_vals, n_runs=3):
    """Benchmark nimopt with Rust-accelerated direct solver."""
    import nimopt as no
    from nimopt.solvers import HiGHSDirectSolver

    sources = [f's{i}' for i in range(n_src)]
    dests = [f'd{j}' for j in range(n_dst)]

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        i = no.Set('i', sources)
        j = no.Set('j', dests)
        supply_p = no.Param('supply', [i], supply_vals.tolist())
        demand_p = no.Param('demand', [j], demand_vals.tolist())
        cost_p = no.Param('cost', [i, j], cost_vals.tolist())

        m = no.Model(sense='minimize')
        x = m.var('x', [i, j], lb=0)
        m.set_objective(no.Sum(i, j, cost_p[i, j] * x[i, j]))
        m.eq('supply', no.Sum(j, x[i, j]) <= supply_p[i])
        m.eq('demand', no.Sum(i, x[i, j]) >= demand_p[j])

        solver = HiGHSDirectSolver(use_rust=True)
        solver.load_model(m)
        solver.solve()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def main():
    print("=" * 70)
    print("Benchmark: nimopt vs linopy (transport problem)")
    print("=" * 70)

    sizes = [(30, 30), (100, 100), (200, 200), (300, 300), (400, 400)]

    print()
    print(f"{'Size':<12} {'Vars':<10} {'linopy':<15} {'nimopt':<15} {'Speedup':<10}")
    print("-" * 70)

    for n_src, n_dst in sizes:
        np.random.seed(42)

        supply_vals = np.random.uniform(100, 1000, n_src)
        demand_vals = np.random.uniform(50, 200, n_dst)
        total_supply = supply_vals.sum()
        total_demand = demand_vals.sum()
        demand_vals = demand_vals * total_supply / total_demand * 0.9
        cost_vals = np.random.uniform(1, 10, (n_src, n_dst))

        # Suppress output during benchmarks
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            lp_time = bench_linopy(n_src, n_dst, supply_vals, demand_vals,
                                   cost_vals)
            no_time = bench_nimopt(n_src, n_dst, supply_vals, demand_vals,
                                   cost_vals)
        finally:
            sys.stdout = old_stdout

        n_vars = n_src * n_dst
        speedup = lp_time / no_time
        print(f"{n_src}x{n_dst:<8} {n_vars:<10} {lp_time*1000:>10.1f} ms   "
              f"{no_time*1000:>10.1f} ms   {speedup:>6.1f}x")

    print("-" * 70)
    print()
    print("Note: Times include model building + solving")
    print("=" * 70)


if __name__ == "__main__":
    main()
