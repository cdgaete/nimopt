#!/usr/bin/env python3
"""Benchmark HiGHS vs PDLP GPU on DIETER model."""

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dieter import load_sets, load_params, build_model

import nimopt as no
from nimopt.solvers import HiGHSDirectSolver, PDLPSolver


def main():
    n_hours = 8760
    corr_factor = 1.0
    
    print("=" * 60)
    print("DIETER 8760h - Solver Benchmark")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    t0 = time.time()
    sets = load_sets(n_hours)
    params = load_params(sets)
    print(f"  Data loaded: {time.time() - t0:.2f}s")
    
    # Build model
    print("\nBuilding model...")
    t0 = time.time()
    model = build_model(sets, params, corr_factor)
    print(f"  Model built: {time.time() - t0:.2f}s")
    
    n_vars = sum(v.size for v in model.variables.values())
    n_cons = len(model._constraints)
    print(f"  Variables: {n_vars:,}")
    print(f"  Constraints: {n_cons}")
    
    # HiGHS
    print("\n" + "=" * 60)
    print("HiGHS (Simplex)")
    print("=" * 60)
    solver = HiGHSDirectSolver(use_rust=True)
    t0 = time.time()
    solver.load_model(model)
    load_time = time.time() - t0
    print(f"Load time: {load_time:.2f}s")
    result = solver.solve()
    print(f"Solve time: {result.solve_time:.2f}s")
    print(f"Total: {load_time + result.solve_time:.2f}s")
    print(f"Objective: €{result.objective_value:,.0f}")
    print(f"Iterations: {result.iterations}")
    highs_obj = result.objective_value
    highs_total = load_time + result.solve_time
    
    # PDLP GPU
    print("\n" + "=" * 60)
    print("PDLP GPU (cuPDLPx)")
    print("=" * 60)
    solver = PDLPSolver(backend='cupdlp', tolerance=1e-4, verbose=False)
    t0 = time.time()
    solver.load_model(model)
    load_time = time.time() - t0
    print(f"Load time: {load_time:.2f}s")
    result = solver.solve()
    print(f"Solve time: {result.solve_time:.2f}s")
    print(f"Total: {load_time + result.solve_time:.2f}s")
    print(f"Objective: €{result.objective_value:,.0f}")
    print(f"Iterations: {result.iterations}")
    pdlp_total = load_time + result.solve_time
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"HiGHS total:  {highs_total:.2f}s")
    print(f"PDLP GPU:     {pdlp_total:.2f}s")
    print(f"Speedup:      {highs_total/pdlp_total:.1f}x")
    print(f"Obj diff:     {abs(result.objective_value - highs_obj)/highs_obj*100:.4f}%")


if __name__ == '__main__':
    main()
