"""Benchmark Python 3.11 vs 3.14."""
import sys
import time
import numpy as np

# Add paths
sys.path.insert(0, '/home/carlos/projects/nimblend/src')
sys.path.insert(0, '/home/carlos/projects/nimopt/src')

import nimopt as no
from nimopt.solvers import HiGHSDirectSolver

print(f"Python {sys.version.split()[0]}")
print("=" * 50)

sizes = [(300, 300), (500, 500), (1000, 1000), (1000, 2000)]

for n_src, n_dst in sizes:
    times = []
    for _ in range(3):
        np.random.seed(42)
        sources = [f's{i}' for i in range(n_src)]
        dests = [f'd{j}' for j in range(n_dst)]
        supply_vals = 100 + 900 * np.random.rand(n_src)
        demand_vals = 50 + 150 * np.random.rand(n_dst)
        demand_vals = demand_vals * (supply_vals.sum() / demand_vals.sum() * 0.9)
        cost_vals = 1 + 9 * np.random.rand(n_src, n_dst)

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

    median = sorted(times)[1]
    n_vars = n_src * n_dst
    print(f"{n_src}x{n_dst} ({n_vars:,} vars): {median*1000:.0f} ms")
