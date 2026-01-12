"""Benchmark nimopt vs linopy LP generation."""

import tempfile
import time


def benchmark_nimopt(n_sources: int, n_destinations: int):
    """Benchmark nimopt LP generation."""
    import nimopt as no

    t0 = time.perf_counter()

    # Sets
    i = no.Set('i', [f'S{k}' for k in range(n_sources)])
    j = no.Set('j', [f'D{k}' for k in range(n_destinations)])

    # Parameters
    import numpy as np
    supply_data = np.random.randint(50, 150, n_sources).tolist()
    demand_data = np.random.randint(30, 100, n_destinations).tolist()
    cost_data = np.random.rand(n_sources, n_destinations) * 10

    supply = no.Param('supply', [i], supply_data)
    demand = no.Param('demand', [j], demand_data)
    cost = no.Param('cost', [i, j], cost_data)

    # Model
    m = no.Model(name='transport', sense='minimize')
    x = m.var('x', [i, j], lb=0)
    m.eq('supply', no.Sum(j, x[i, j]) <= supply[i])
    m.eq('demand', no.Sum(i, x[i, j]) >= demand[j])
    m.objective = no.Sum(i, j, cost[i, j] * x[i, j])

    build_time = time.perf_counter() - t0

    # Write LP
    with tempfile.NamedTemporaryFile(suffix='.lp', delete=False) as f:
        lp_file = f.name

    t0 = time.perf_counter()
    m.to_lp(lp_file)
    write_time = time.perf_counter() - t0

    return build_time, write_time


def benchmark_linopy(n_sources: int, n_destinations: int):
    """Benchmark linopy LP generation."""
    import linopy
    import numpy as np
    import pandas as pd
    import xarray as xr

    t0 = time.perf_counter()

    sources = [f'S{k}' for k in range(n_sources)]
    dests = [f'D{k}' for k in range(n_destinations)]

    supply_data = np.random.randint(50, 150, n_sources)
    demand_data = np.random.randint(30, 100, n_destinations)
    cost_data = np.random.rand(n_sources, n_destinations) * 10

    m = linopy.Model()

    x = m.add_variables(lower=0, coords=[sources, dests], name='x')

    m.add_constraints(
        x.sum('dim_1') <= pd.Series(supply_data, index=sources),
        name='supply'
    )
    m.add_constraints(
        x.sum('dim_0') >= pd.Series(demand_data, index=dests),
        name='demand'
    )

    cost = xr.DataArray(cost_data, coords=[sources, dests], dims=['dim_0', 'dim_1'])
    m.add_objective((cost * x).sum())

    build_time = time.perf_counter() - t0

    with tempfile.NamedTemporaryFile(suffix='.lp', delete=False) as f:
        lp_file = f.name

    t0 = time.perf_counter()
    m.to_file(lp_file)
    write_time = time.perf_counter() - t0

    return build_time, write_time


def run_benchmark(sizes):
    """Run benchmarks for different sizes."""
    print(f"{'Size':<15} {'nimopt build':<15} {'nimopt write':<15} "
          f"{'linopy build':<15} {'linopy write':<15}")
    print("=" * 75)

    for n in sizes:
        n_vars = n * n

        # nimopt
        nimopt_build, nimopt_write = benchmark_nimopt(n, n)

        # linopy
        linopy_build, linopy_write = benchmark_linopy(n, n)

        print(f"{n}x{n} ({n_vars:,})"
              f"  {nimopt_build*1000:>10.1f}ms"
              f"  {nimopt_write*1000:>12.1f}ms"
              f"  {linopy_build*1000:>12.1f}ms"
              f"  {linopy_write*1000:>12.1f}ms")


if __name__ == "__main__":
    print("LP Generation Benchmark: nimopt vs linopy")
    print()
    run_benchmark([10, 50, 100, 200, 300])
