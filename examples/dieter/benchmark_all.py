#!/usr/bin/env python3
"""
Comprehensive Solver Benchmark for DIETER Model

Tests all available solvers and reports detailed timing for:
- Model building
- Solver loading (matrix construction)
- Solve time
- Solution extraction (primal and dual values)

Usage:
    python benchmark_all.py                 # Default 168 hours
    python benchmark_all.py --hours 24      # Quick test
    python benchmark_all.py --hours 8760    # Full year
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dieter import load_sets, load_params, build_model

import numpy as np


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}µs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds/60:.1f}min"


def format_currency(value: float) -> str:
    """Format currency value."""
    if value >= 1e12:
        return f"€{value/1e12:.2f}T"
    elif value >= 1e9:
        return f"€{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"€{value/1e6:.2f}M"
    else:
        return f"€{value:,.0f}"


class BenchmarkResult:
    """Store benchmark results for a solver."""
    
    def __init__(self, name: str):
        self.name = name
        self.load_time = 0.0
        self.solve_time = 0.0
        self.extract_time = 0.0
        self.objective = None
        self.status = None
        self.iterations = 0
        self.n_primal = 0
        self.n_dual = 0
        self.error = None
    
    @property
    def total_time(self) -> float:
        return self.load_time + self.solve_time + self.extract_time
    
    def __repr__(self):
        if self.error:
            return f"{self.name}: ERROR - {self.error}"
        return (f"{self.name}: obj={format_currency(self.objective)}, "
                f"total={format_time(self.total_time)}")


def benchmark_lp_file(model, use_rust: bool = True) -> BenchmarkResult:
    """Benchmark solving via LP file."""
    import os
    import highspy
    
    result = BenchmarkResult("LP File + HiGHS")
    
    try:
        # Write LP file
        lp_path = tempfile.mktemp(suffix='.lp')
        t0 = time.time()
        model.to_lp(lp_path, use_rust=use_rust)
        write_time = time.time() - t0
        
        # Load into HiGHS
        h = highspy.Highs()
        h.setOptionValue('output_flag', False)
        t0 = time.time()
        h.readModel(lp_path)
        read_time = time.time() - t0
        
        result.load_time = write_time + read_time
        
        # Solve
        t0 = time.time()
        h.run()
        result.solve_time = time.time() - t0
        
        # Extract solution
        t0 = time.time()
        primal = h.allVariableValues()
        dual_vars = h.allVariableDuals()
        dual_cons = h.allConstrDuals()
        result.extract_time = time.time() - t0
        
        result.objective = h.getObjectiveValue()
        result.status = str(h.getModelStatus())
        result.iterations = h.getInfo().simplex_iteration_count
        result.n_primal = len(primal)
        result.n_dual = len(dual_cons)
        
        # Clean up
        os.unlink(lp_path)
        
    except Exception as e:
        result.error = str(e)
    
    return result



def benchmark_highs_direct(model, use_rust: bool = True) -> BenchmarkResult:
    """Benchmark HiGHS direct solver."""
    from nimopt.solvers import HiGHSDirectSolver
    
    result = BenchmarkResult("HiGHS Direct")
    
    try:
        solver = HiGHSDirectSolver(use_rust=use_rust)
        
        # Load model
        t0 = time.time()
        solver.load_model(model)
        result.load_time = time.time() - t0
        
        # Solve
        t0 = time.time()
        solve_result = solver.solve()
        result.solve_time = time.time() - t0
        
        # Extract solution
        t0 = time.time()
        primal = solver.get_variable_values()
        dual_vars = solver.get_variable_duals()
        dual_cons = solver.get_constraint_duals()
        # Also test nimblend Array extraction
        sol = solver.get_solution()
        _ = sol.var('G_D')  # Test variable extraction
        result.extract_time = time.time() - t0
        
        result.objective = solve_result.objective_value
        result.status = solve_result.status.name
        result.iterations = solve_result.iterations
        result.n_primal = len(primal)
        result.n_dual = len(dual_cons)
        
    except Exception as e:
        result.error = str(e)
    
    return result


def benchmark_pdlp_gpu(model, tolerance: float = 1e-4) -> BenchmarkResult:
    """Benchmark PDLP GPU solver (cuPDLPx)."""
    result = BenchmarkResult("PDLP GPU (cuPDLPx)")
    
    try:
        from nimopt.solvers import PDLPSolver
        solver = PDLPSolver(backend='cupdlp', tolerance=tolerance, verbose=False)
        
        # Load model
        t0 = time.time()
        solver.load_model(model)
        result.load_time = time.time() - t0
        
        # Solve
        t0 = time.time()
        solve_result = solver.solve()
        result.solve_time = time.time() - t0
        
        # Extract solution
        t0 = time.time()
        primal = solver.get_variable_values()
        dual_cons = solver.get_constraint_duals()
        # Also test nimblend Array extraction
        sol = solver.get_solution()
        _ = sol.var('G_D')
        result.extract_time = time.time() - t0
        
        result.objective = solve_result.objective_value
        result.status = solve_result.status.name
        result.iterations = solve_result.iterations
        result.n_primal = len(primal)
        result.n_dual = len(dual_cons)
        
    except ImportError:
        result.error = "cuPDLPx not installed"
    except Exception as e:
        result.error = str(e)
    
    return result


def print_results_table(results: list, build_time: float, n_vars: int, n_cons: int):
    """Print results in a formatted table."""
    
    # Find reference (first successful result)
    ref_obj = None
    for r in results:
        if r.objective is not None:
            ref_obj = r.objective
            break
    
    # Header
    print("\n" + "=" * 100)
    print(f"{'Solver':<25} {'Load':>10} {'Solve':>10} {'Extract':>10} {'Total':>10} {'Objective':>15} {'Diff':>8}")
    print("=" * 100)
    
    for r in results:
        if r.error:
            print(f"{r.name:<25} {'ERROR: ' + r.error[:60]}")
            continue
        
        diff = ""
        if ref_obj and r.objective:
            diff_pct = abs(r.objective - ref_obj) / ref_obj * 100
            diff = f"{diff_pct:.4f}%"
        
        print(f"{r.name:<25} "
              f"{format_time(r.load_time):>10} "
              f"{format_time(r.solve_time):>10} "
              f"{format_time(r.extract_time):>10} "
              f"{format_time(r.total_time):>10} "
              f"{format_currency(r.objective):>15} "
              f"{diff:>8}")
    
    print("=" * 100)
    
    # Speedup comparison
    if len([r for r in results if r.objective]) >= 2:
        print("\nSpeedup Analysis:")
        base = results[0]
        if base.objective:
            for r in results[1:]:
                if r.objective:
                    speedup = base.total_time / r.total_time
                    print(f"  {r.name} vs {base.name}: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")



def main():
    parser = argparse.ArgumentParser(description='Comprehensive Solver Benchmark')
    parser.add_argument('--hours', type=int, default=168,
                        help='Number of hours to model (default: 168)')
    parser.add_argument('--no-lp', action='store_true',
                        help='Skip LP file benchmark')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Skip PDLP GPU benchmark')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='PDLP tolerance (default: 1e-4)')
    args = parser.parse_args()
    
    n_hours = args.hours
    corr_factor = n_hours / 8760
    
    print("=" * 100)
    print("DIETER Solver Benchmark")
    print("=" * 100)
    print(f"Hours: {n_hours} ({n_hours/24:.1f} days)")
    print(f"Correction factor: {corr_factor:.4f}")
    
    # Load data
    print("\n[1/3] Loading data...")
    t0 = time.time()
    sets = load_sets(n_hours)
    params = load_params(sets)
    data_time = time.time() - t0
    print(f"      Data loading: {format_time(data_time)}")
    
    # Build model
    print("\n[2/3] Building model...")
    t0 = time.time()
    model = build_model(sets, params, corr_factor)
    build_time = time.time() - t0
    
    n_vars = sum(v.size for v in model.variables.values())
    n_cons = len(model._constraints)
    print(f"      Model building: {format_time(build_time)}")
    print(f"      Variables: {n_vars:,}")
    print(f"      Constraints: {n_cons}")
    
    # Run benchmarks
    print("\n[3/3] Running solver benchmarks...")
    results = []
    
    # LP File benchmark
    if not args.no_lp:
        print(f"\n      Testing LP File + HiGHS...")
        results.append(benchmark_lp_file(model))
        print(f"      Done: {results[-1]}")
    
    # HiGHS Direct benchmark
    print(f"\n      Testing HiGHS Direct...")
    results.append(benchmark_highs_direct(model))
    print(f"      Done: {results[-1]}")
    
    # PDLP GPU benchmark
    if not args.no_gpu:
        print(f"\n      Testing PDLP GPU...")
        results.append(benchmark_pdlp_gpu(model, tolerance=args.tolerance))
        print(f"      Done: {results[-1]}")
    
    # Print results table
    print_results_table(results, build_time, n_vars, n_cons)
    
    # Detailed timing breakdown
    print("\nDetailed Timing Breakdown:")
    print(f"  Data loading:    {format_time(data_time)}")
    print(f"  Model building:  {format_time(build_time)}")
    for r in results:
        if not r.error:
            print(f"\n  {r.name}:")
            print(f"    Matrix build:  {format_time(r.load_time)}")
            print(f"    Solve:         {format_time(r.solve_time)}")
            print(f"    Extract sol:   {format_time(r.extract_time)}")
            print(f"    Status:        {r.status}")
            print(f"    Iterations:    {r.iterations:,}")
            print(f"    Primal vars:   {r.n_primal:,}")
            print(f"    Dual cons:     {r.n_dual:,}")


if __name__ == '__main__':
    main()
