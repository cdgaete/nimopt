# HANDOVER: PDLP GPU Solver Integration

## Current Status: FIXED ✅
The PDLP GPU solver is now working correctly with the DIETER energy model.

**Results (DIETER 8760h):**
- HiGHS: €32,136,324,411 in 49.37s
- PDLP GPU: €32,134,586,383 in 1.64s  
- **Difference: 0.0054%** (within tolerance)
- **Speedup: 30.1x faster!**

## The Bug Found
**cuPDLPx does not properly enforce variable bounds** (`lb <= x <= ub`).

The solver correctly satisfies constraint bounds (`row_lower <= Ax <= row_upper`) but
ignores variable bounds. This causes variables that should be `>= 0` to go negative,
leading to incorrect solutions.

### Evidence:
- cuPDLPx reported "Primal infeas: 6.3e-9" (very small)
- Actual variable bound violations: up to 46,701 (massive!)
- Variables like `CU_pv_nw_18` were -46,701 instead of 0

## The Fix
Convert variable bounds to explicit constraint rows in `_solve_cupdlp()`:

```python
# WORKAROUND: cuPDLPx doesn't properly enforce variable bounds
# Convert x >= lb to explicit constraints: row with coef 1, bounds [lb, inf]
# Convert x <= ub to explicit constraints: row with coef 1, bounds [-inf, ub]
```

The fix is implemented in `/home/carlos/projects/nimopt/src/nimopt/solvers/pdlp.py`
in the `_solve_cupdlp()` method (lines ~220-290).

## OR-Tools PDLP Backend
The OR-Tools PDLP backend (`_solve_ortools()`) has a separate issue:
- Returns objective=0 and all-zero solution for DIETER
- Works correctly on simple transport problems
- Issue appears related to large RHS values (up to 1e12) in constraints
- Not a priority since cuPDLPx is much faster

## What Works Now
1. **GPU acceleration**: NVIDIA RTX 2080 SUPER working
2. **cuPDLPx backend**: Fixed with bound constraint workaround
3. **Simple models**: Both backends work correctly
4. **DIETER model**: Matches HiGHS within 0.01% tolerance


## Quick Test Commands
```bash
cd /home/carlos/projects/nimopt && . .venv/bin/activate

# Test DIETER benchmark (should show ~30x speedup)
python3 examples/dieter/benchmark_solvers.py

# Test simple model (all solvers should match)
python3 -c "
import nimopt as no
from nimopt import Sum
from nimopt.solvers import PDLPSolver, HiGHSDirectSolver

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

## Environment
- Server: pangui (192.168.1.86)
- GPU: NVIDIA RTX 2080 SUPER
- CUDA: 13.1 / Driver: 590.48.01
- Python: 3.11 in `/home/carlos/projects/nimopt/.venv`

## Note on GPU Persistence
After reboot, run: `sudo modprobe nvidia` to load the driver.

## Potential Future Improvements
1. File an issue on cuPDLPx GitHub about the variable bounds bug
2. Investigate OR-Tools PDLP issue with large RHS values
3. Consider presolve/scaling options for better numerical stability
4. Profile the bound constraint overhead (currently adds n_vars rows)
