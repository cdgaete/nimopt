# nimopt Development Handover

## Project Overview

**nimopt** is an algebraic optimization modeling library with GAMS-like syntax, using **nimblend** for labeled N-dimensional arrays.

Location: `/home/carlos/projects/nimopt`

## Installation

```bash
cd /home/carlos/projects/nimopt
python3 -m venv .venv
source .venv/bin/activate
pip install -e /home/carlos/projects/nimblend
pip install -e .
pip install highspy pandas  # For DIETER example
```

## Key Features

- **Lazy constraint expansion**: O(1) model building time
- **nimblend backend**: Outer-join/zero-fill semantics for parameters
- **Rust acceleration**: 100x+ speedup for LP writing (full year DIETER)
- **Multiple solvers**: HiGHS (open-source), GAMS/CPLEX (commercial)
- **Smart solution storage**: Auto lazy/eager based on model size
- **Math functions**: `sqrt`, `exp`, `log`, `abs_`, `power` for parameters
- **Subset support**: `Set.from_csv()` with filtering for DIETER-style subsets
- **Lag/lead indexing**: Time-shifted constraints (e.g., `STO[h] == STO[h-1] + IN - OUT`)

## DIETER Energy Model Example

Full implementation of DIETER (Dispatch and Investment Evaluation Tool with Endogenous Renewables):

```bash
source .venv/bin/activate
python examples/dieter/dieter.py --hours 8760       # Full year, direct solver
python examples/dieter/dieter.py --hours 168        # One week
python examples/dieter/dieter.py --hours 168 --lp   # Via LP file
```

### Full Year Results (8760 hours)
- **Objective**: €32.1 billion
- **Solve time**: ~14s via LP file, ~50s direct solver

### Solver Comparison (Full Year)

| Solver Path | Write/Load | Read | Solve | Total |
|-------------|------------|------|-------|-------|
| Direct HiGHS (Rust) | 206ms | - | 49.5s | 49.7s |
| LP (Rust) + HiGHS | 396ms | 2.1s | 13.9s | 16.4s |

Note: LP file path is faster because HiGHS optimizes better from LP format. Direct solver has extra non-zeros (matrix deduplication issue to investigate).

### LP Writer Performance

| Model | Python LP | Rust LP | Speedup |
|-------|-----------|---------|---------|
| 168 hours | 59ms | 10ms | 6x |
| 8760 hours | 40.3s | 394ms | **102x** |

## API Usage

```python
import nimopt as no
from nimopt.solvers import HiGHSDirectSolver

# Sets and parameters
i = no.Set('i', ['seattle', 'sandiego'])
j = no.Set('j', ['newyork', 'chicago', 'topeka'])
cost = no.Param('cost', [i, j], [[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]])

# Math functions on parameters
eff = no.Param('eff', [i], [0.81, 0.64])
sqrt_eff = no.sqrt(eff)  # Returns Param with sqrt applied

# Subsets from CSV
Tech = no.Set.from_csv('Tech', 'tech.csv', 'name')
Renewables = no.Set.from_csv('Renewables', 'tech.csv', 'name', 
                              filter_col='renewable', filter_val=1)

# Model
m = no.Model(sense='minimize')
x = m.var('x', [i, j], lb=0)
m.eq('supply', no.Sum(j, x[i,j]) <= supply[i])
m.objective = no.Sum(i, j, cost[i,j] * x[i,j])

# Solve (direct - no LP file)
solver = HiGHSDirectSolver(use_rust=True)
solver.load_model(m)
result = solver.solve()

# Or via LP file
m.to_lp('model.lp', use_rust=True)  # Rust-accelerated
import highspy
h = highspy.Highs()
h.readModel('model.lp')
h.run()

# Get solution as nimblend Arrays
sol = solver.get_solution()
x_values = sol.var('x')           # nimblend Array
x_values.sel({'i': 'seattle'})    # coordinate selection
```

## File Structure

```
src/nimopt/
├── model.py              # Model class, lazy constraint storage
├── param.py              # Param (nimblend Array wrapper) + arithmetic ops
├── sets.py               # Set class with from_csv(), subset support
├── variable.py           # Variable, VarRef (4-tuple terms with lagged support)
├── expression.py         # LinearExpr, Constraint (4-tuple: var, coef, fixed, lagged)
├── functions.py          # Sum(), sqrt(), exp(), log(), abs_(), power()
├── solution.py           # Solution extraction
├── solvers/
│   ├── highs_direct.py   # Direct HiGHS (fastest) - Python & Rust paths
│   ├── highs.py          # HiGHS via LP file
│   └── gams_cplex.py     # GAMS/CPLEX
└── writers/
    ├── lp.py             # Python LP writer
    └── lp_rust.py        # Rust-accelerated LP generation
rust/
└── src/lib.rs            # Rust extension (LP writing, matrix building)
examples/
└── dieter/               # DIETER energy system model
    ├── dieter.py         # Main model implementation
    ├── prepare_data.py   # Data preparation script
    └── data/             # CSV data files
```

## Development Commands

```bash
cd /home/carlos/projects/nimopt
source .venv/bin/activate
pytest tests/ --tb=short          # Run tests (28 passing)
ruff check src/ tests/            # Lint

# Rebuild Rust extension
export PATH="$HOME/.cargo/bin:$PATH"
cd rust && maturin develop --release
```

## Recent Fixes (January 2025)

### Rust LP Writer Fixes
1. **Scalar coefficients for constraints**: When constraint terms had scalar coefficients (e.g., `-1`), they were lost (defaulted to `1.0`). Fixed by adding `term_scalar_coefs` parameter to Rust function.

2. **Array RHS constant handling**: When `con.rhs.const` was an Array (like demand in energy balance), it was incorrectly negated. The Python LP writer doesn't negate array constants - fixed to match.

3. **Objective coefficient deduplication**: Variables appearing multiple times in objective (e.g., from multiple Sum terms) needed coefficient accumulation. Added `coef_map` dictionary.

### Coefficient Broadcasting
Fixed parameter broadcasting when dimensions differ from variable:
- `MarginalCost[Tech]` × `G[Tech, Hours]` now broadcasts correctly
- Affects both objective and constraint coefficient extraction

### Variable Shadowing Bug
Fixed loop variable `c` overwriting cost vector in `_build_matrices()`.

## Architecture Notes

### Expression Terms (4-tuple format)
Terms stored as `(Variable, coef, fixed_indices, lagged_indices)`:
- `fixed_indices`: List of `(position, value)` for concrete indices
- `lagged_indices`: List of `(position, LaggedSet)` for time-shifted indices

### Coefficient Extraction
When extracting coefficients for constraints/objectives:
- Use full constraint bindings (not just variable sets)
- Handle broadcasting when coef has fewer dims than variable
- Key functions: `_get_coef_for_bindings()`, `_get_coef_for_combo()`

### Lazy Constraint Expansion
Constraints stored symbolically with `free_sets`, expanded at solve time.

## Known Issues

### Direct Solver Extra Non-Zeros
The direct HiGHS solver produces more matrix non-zeros than the LP file path (e.g., 779k vs 744k for full year DIETER). This causes slower solving (~50s vs ~14s). The LP writer correctly deduplicates entries but the direct solver doesn't. Both produce correct results.

## Test Coverage

- **28 tests** passing
- Transport problem, lag/lead constraints, solution extraction
- Both Python and Rust solver paths verified
- GAMS/CPLEX solver verified against HiGHS
