#!/usr/bin/env python3
"""
Comprehensive DIETER benchmark: nimopt vs linopy vs Pyomo

Compares:
- nimopt Direct (Rust)
- nimopt LP (Rust) + HiGHS
- linopy + HiGHS
- Pyomo + appsi_highs
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "dieter"))

import numpy as np


def benchmark_nimopt_direct(n_hours: int, corr_factor: float):
    """Benchmark nimopt with direct HiGHS solver."""
    from dieter import load_sets, load_params, build_model
    from nimopt.solvers import HiGHSDirectSolver

    # Load
    t0 = time.time()
    sets = load_sets(n_hours)
    params = load_params(sets)
    t_load = time.time() - t0

    # Build
    t0 = time.time()
    model = build_model(sets, params, corr_factor)
    t_build = time.time() - t0

    # Solve
    t0 = time.time()
    solver = HiGHSDirectSolver(use_rust=True)
    solver.load_model(model)
    t_solver_load = time.time() - t0

    t0 = time.time()
    result = solver.solve()
    t_solve = time.time() - t0

    return {
        'name': 'nimopt Direct (Rust)',
        'load': t_load,
        'build': t_build,
        'solver_load': t_solver_load,
        'solve': t_solve,
        'total': t_load + t_build + t_solver_load + t_solve,
        'objective': result.objective_value,
        'status': result.status.name,
    }


def benchmark_nimopt_lp(n_hours: int, corr_factor: float):
    """Benchmark nimopt with LP file + HiGHS."""
    import tempfile
    import highspy
    from dieter import load_sets, load_params, build_model

    # Load
    t0 = time.time()
    sets = load_sets(n_hours)
    params = load_params(sets)
    t_load = time.time() - t0

    # Build
    t0 = time.time()
    model = build_model(sets, params, corr_factor)
    t_build = time.time() - t0

    # Write LP
    lp_path = tempfile.mktemp(suffix='.lp')
    t0 = time.time()
    model.to_lp(lp_path, use_rust=True)
    t_write = time.time() - t0

    # Read and solve
    t0 = time.time()
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.readModel(lp_path)
    t_read = time.time() - t0

    t0 = time.time()
    h.run()
    t_solve = time.time() - t0

    obj = h.getObjectiveValue()
    status = h.getModelStatus().name

    # Cleanup
    import os
    os.unlink(lp_path)

    return {
        'name': 'nimopt LP (Rust)',
        'load': t_load,
        'build': t_build,
        'solver_load': t_write + t_read,
        'solve': t_solve,
        'total': t_load + t_build + t_write + t_read + t_solve,
        'objective': obj,
        'status': status,
    }


def benchmark_linopy(n_hours: int, corr_factor: float):
    """Benchmark linopy."""
    from dieter_linopy import load_data, build_model

    # Load
    t0 = time.time()
    data = load_data(n_hours)
    t_load = time.time() - t0

    # Build
    t0 = time.time()
    model = build_model(data, corr_factor)
    t_build = time.time() - t0

    # Solve
    t0 = time.time()
    model.solve(solver_name='highs', log_fn=None)
    t_solve = time.time() - t0

    return {
        'name': 'linopy',
        'load': t_load,
        'build': t_build,
        'solver_load': 0,
        'solve': t_solve,
        'total': t_load + t_build + t_solve,
        'objective': model.objective.value,
        'status': model.status,
    }


def benchmark_pyomo(n_hours: int, corr_factor: float):
    """Benchmark Pyomo."""
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    from dieter_pyomo import load_data, build_model

    # Load
    t0 = time.time()
    data = load_data(n_hours)
    t_load = time.time() - t0

    # Build
    t0 = time.time()
    model = build_model(data, corr_factor)
    t_build = time.time() - t0

    # Solve
    t0 = time.time()
    solver = SolverFactory('appsi_highs')
    result = solver.solve(model, tee=False)
    t_solve = time.time() - t0

    return {
        'name': 'Pyomo',
        'load': t_load,
        'build': t_build,
        'solver_load': 0,
        'solve': t_solve,
        'total': t_load + t_build + t_solve,
        'objective': pyo.value(model.objective),
        'status': str(result.solver.termination_condition),
    }


def print_results(results: list, n_hours: int):
    """Print benchmark results as a table."""
    print(f"\n{'='*80}")
    print(f"DIETER Benchmark Results - {n_hours} hours ({n_hours/24:.1f} days)")
    print(f"{'='*80}")
    
    # Header
    print(f"\n{'Framework':<20} {'Load':>8} {'Build':>8} {'LP/Load':>8} {'Solve':>8} {'Total':>10} {'Objective':>18}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['name']:<20} {r['load']*1000:>7.0f}ms {r['build']*1000:>7.0f}ms "
              f"{r['solver_load']*1000:>7.0f}ms {r['solve']*1000:>7.0f}ms "
              f"{r['total']*1000:>9.0f}ms {r['objective']:>17,.0f}")
    
    # Speedup vs slowest
    slowest = max(r['total'] for r in results)
    print("\nSpeedup vs slowest:")
    for r in results:
        speedup = slowest / r['total']
        print(f"  {r['name']:<20} {speedup:>5.1f}x")
    
    # Verify objectives match
    objs = [r['objective'] for r in results]
    if max(objs) - min(objs) < 1.0:
        print("\n✓ All objectives match")
    else:
        print("\n✗ WARNING: Objectives differ!")
        for r in results:
            print(f"  {r['name']}: {r['objective']:,.2f}")


def main():
    parser = argparse.ArgumentParser(description='DIETER Benchmark')
    parser.add_argument('--hours', type=int, default=168,
                        help='Number of hours (default: 168)')
    parser.add_argument('--skip-pyomo', action='store_true',
                        help='Skip Pyomo benchmark')
    parser.add_argument('--skip-linopy', action='store_true',
                        help='Skip linopy benchmark')
    args = parser.parse_args()

    n_hours = args.hours
    corr_factor = n_hours / 8760

    print(f"\nRunning DIETER benchmark with {n_hours} hours...")
    
    results = []

    # nimopt Direct
    print("\n[1/4] nimopt Direct (Rust)...")
    try:
        results.append(benchmark_nimopt_direct(n_hours, corr_factor))
        print(f"      Done: {results[-1]['total']*1000:.0f}ms")
    except Exception as e:
        print(f"      Failed: {e}")

    # nimopt LP
    print("\n[2/4] nimopt LP (Rust)...")
    try:
        results.append(benchmark_nimopt_lp(n_hours, corr_factor))
        print(f"      Done: {results[-1]['total']*1000:.0f}ms")
    except Exception as e:
        print(f"      Failed: {e}")

    # linopy
    if not args.skip_linopy:
        print("\n[3/4] linopy...")
        try:
            results.append(benchmark_linopy(n_hours, corr_factor))
            print(f"      Done: {results[-1]['total']*1000:.0f}ms")
        except Exception as e:
            print(f"      Failed: {e}")
    else:
        print("\n[3/4] linopy... SKIPPED")

    # Pyomo
    if not args.skip_pyomo:
        print("\n[4/4] Pyomo...")
        try:
            results.append(benchmark_pyomo(n_hours, corr_factor))
            print(f"      Done: {results[-1]['total']*1000:.0f}ms")
        except Exception as e:
            print(f"      Failed: {e}")
    else:
        print("\n[4/4] Pyomo... SKIPPED")

    print_results(results, n_hours)


if __name__ == '__main__':
    main()
