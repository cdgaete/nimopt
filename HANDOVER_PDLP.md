# HANDOVER: PDLP GPU Solver Integration

## Current Status: FIXED ✅
The PDLP GPU solver is now working correctly with the DIETER energy model.

## Benchmark Results (DIETER 8760h - Full Year)

| Solver | Load | Solve | Extract | **Total** | Objective | Diff |
|--------|------|-------|---------|-----------|-----------|------|
| LP File + HiGHS | 2.51s | 13.86s | 17.3ms | **16.39s** | €32.14B | 0.00% |
| HiGHS Direct | 220ms | 50.39s | 51.1ms | **50.66s** | €32.14B | 0.00% |
| **PDLP GPU** | 139ms | 3.18s | 65.7ms | **3.39s** | €32.13B | 0.0054% |
| GAMS/CPLEX | 405ms | 17.04s | 3µs | **17.44s** | €32.14B | 0.00% |

**PDLP GPU is 4.8x faster than the next best solver!**

## The Bug Found & Fixed

**cuPDLPx does not properly enforce variable bounds** (`lb <= x <= ub`).

The solver correctly satisfies constraint bounds (`row_lower <= Ax <= row_upper`) but
ignores variable bounds. This caused variables constrained to `x >= 0` to go negative,
leading to incorrect solutions (42% lower objective).

### The Fix
Convert variable bounds to explicit constraint rows in `_solve_cupdlp()`:
- For `x >= lb`: Add constraint row with coefficient 1 and bounds `[lb, inf]`
- For `x <= ub`: Add constraint row with coefficient 1 and bounds `[-inf, ub]`

Location: `/home/carlos/projects/nimopt/src/nimopt/solvers/pdlp.py` lines ~220-290

## Available Solvers

| Solver | Class | Backend | Notes |
|--------|-------|---------|-------|
| LP File + HiGHS | - | highspy | Write LP file, load into HiGHS |
| HiGHS Direct | `HiGHSDirectSolver` | highspy | Build matrices directly in memory |
| PDLP GPU | `PDLPSolver` | cuPDLPx | GPU-accelerated first-order method |
| GAMS/CPLEX | `GamsCplexSolver` | GAMS + CPLEX | Commercial solver via GAMS |

## Quick Test Commands
```bash
cd /home/carlos/projects/nimopt && . .venv/bin/activate

# Full benchmark with all solvers (8760 hours)
python3 examples/dieter/benchmark_all.py --hours 8760

# Quick test (24 hours)
python3 examples/dieter/benchmark_all.py --hours 24

# Skip specific solvers
python3 examples/dieter/benchmark_all.py --hours 168 --no-gams
python3 examples/dieter/benchmark_all.py --hours 168 --no-gpu

# Test simple model with all solvers
python3 -c "
import nimopt as no
from nimopt import Sum
from nimopt.solvers import PDLPSolver, HiGHSDirectSolver, GamsCplexSolver

i = no.Set('i', ['a', 'b'])
j = no.Set('j', ['x', 'y', 'z'])
d = no.Param('d', [i, j], [[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]])

m = no.Model(sense='minimize')
x = m.var('x', [i, j], lb=0)
m.eq('supply', Sum(j, x[i,j]) <= 100)
m.eq('demand', Sum(i, x[i,j]) >= 50)
m.set_objective(Sum(i, j, d[i,j] * x[i,j]))

for name, solver in [
    ('HiGHS', HiGHSDirectSolver()),
    ('PDLP GPU', PDLPSolver(backend='cupdlp')),
]:
    solver.load_model(m)
    r = solver.solve()
    print(f'{name}: {r.objective_value:.2f}')
"
```

## Files

| File | Description |
|------|-------------|
| `src/nimopt/solvers/pdlp.py` | PDLP solver with cuPDLPx fix |
| `src/nimopt/solvers/highs_direct.py` | HiGHS direct solver |
| `src/nimopt/solvers/gams_cplex.py` | GAMS/CPLEX wrapper |
| `examples/dieter/benchmark_all.py` | Comprehensive solver benchmark |
| `examples/dieter/dieter.py` | DIETER model (load_sets, load_params, build_model) |

## Environment
- Server: pangui (192.168.1.86)
- GPU: NVIDIA RTX 2080 SUPER
- CUDA: 13.1 / Driver: 590.48.01
- Python: 3.11 in `/home/carlos/projects/nimopt/.venv`
- GAMS: /opt/gams/gams46.4_linux_x64_64_sfx

## Note on GPU Persistence
After reboot, run: `sudo modprobe nvidia` to load the driver.

## Known Issues

1. **OR-Tools PDLP backend**: Returns objective=0 for DIETER (works on simple problems).
   Not a priority since cuPDLPx is much faster.

2. **HiGHS Direct slower than LP file**: Different iteration counts suggest different
   presolve behavior. LP file: 104,257 iterations vs Direct: 181,859 iterations.

## Potential Future Improvements
1. File an issue on cuPDLPx GitHub about the variable bounds bug
2. Investigate HiGHS Direct vs LP file iteration difference
3. Add warm-start support for PDLP
4. Profile the bound constraint overhead (currently adds n_vars rows)
