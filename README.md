# nimopt

Algebraic optimization modeling with lazy constraint expansion, powered by **nimblend**.

## Features

- **Expressive syntax**: Sets, Parameters, Variables, Sum, Constraints
- **nimblend backend**: Labeled N-dimensional arrays with automatic broadcasting
- **Lazy constraint expansion**: Constraints stored symbolically, expanded only at LP write
- **Rust acceleration**: Optional fast LP writing via PyO3 (10x+ speedup)
- **Multiple solvers**: HiGHS (open-source), GAMS/CPLEX (commercial)
- **Smart memory management**: Automatic lazy/eager solution storage based on model size
- **Instant model building**: O(1) construction time regardless of model size

## Installation

```bash
pip install nimopt
```

For development:

```bash
git clone https://github.com/yourusername/nimopt.git
cd nimopt
pip install -e ".[dev]"
```

### Optional Rust Acceleration

For 10x+ faster LP writing on large models:

```bash
cd rust
maturin develop --release
```

## Quick Example

```python
import nimopt as no

# Sets define index domains
i = no.Set('i', ['seattle', 'sandiego'])
j = no.Set('j', ['newyork', 'chicago', 'topeka'])

# Parameters are nimblend arrays with labeled coordinates
a = no.Param('a', [i], [350, 600])
b = no.Param('b', [j], [325, 300, 275])
d = no.Param('d', [i, j], [[2.5, 1.7, 1.8],
                           [2.5, 1.8, 1.4]])

# Create model
m = no.Model(sense='minimize')

# Variables indexed by sets
x = m.var('x', [i, j], lb=0)
z = m.var('z')

# Equations with Sum() for aggregation
m.eq('cost', z == no.Sum(i, j, d[i,j] * x[i,j]))
m.eq('supply', no.Sum(j, x[i,j]) <= a[i])  # generates |i| constraints
m.eq('demand', no.Sum(i, x[i,j]) >= b[j])  # generates |j| constraints
m.set_objective(z)

# Export to LP format
m.to_lp('transport.lp')
```

## Solving Models

```python
from nimopt.solvers import HiGHSSolver

# Solve with HiGHS
solver = HiGHSSolver()
solver.solve('transport.lp')

# Extract solution (automatic memory management)
sol = no.extract_solution(solver, m)

print(f"Objective: {sol.objective}")
print(f"Status: {sol.status}")

# Access variable values
print(sol.variables['x'].to_dataframe())

# Export to CSV
no.to_csv(sol, 'solution.csv')
```

## Architecture

### Key Innovation: Lazy Constraint Expansion

Traditional algebraic modeling tools (Pyomo, linopy) expand constraints eagerly:

- `Sum(j, x[i,j]) <= a[i]` immediately creates |i| Constraint objects
- Each constraint built via Python loops over indices
- Memory overhead grows with model size

**nimopt** uses **lazy expansion** with **nimblend arrays**:

- Constraints stored symbolically with coefficient arrays
- nimblend handles broadcasting (e.g., `cost[i,j] * x[i,j]`)
- LP writer expands constraints on-the-fly using vectorized operations
- **Result**: O(1) model building time, regardless of size

### nimblend vs. xarray

| Feature | xarray (linopy) | nimblend (nimopt) |
|---------|-----------------|-------------------|
| Alignment | Inner join (intersection) | Outer join (union) |
| Missing values | NaN | 0 |
| Use case | General data analysis | Optimization modeling |

The outer-join/zero-fill semantics match optimization modeling better: missing coordinate combinations typically mean "zero contribution," not "missing data."

### Performance

For a 200k variable energy model:

| Metric | Eager expansion | nimopt (lazy) |
|--------|-----------------|---------------|
| Model build | 3.0s | ~0s |
| LP write (Python) | 0.7s | 3.5s |
| LP write (Rust) | - | 0.3s |
| **Total (Rust)** | **3.7s** | **0.3s** |

Model building is instant because constraints are stored symbolically.

### Smart Solution Storage

nimopt automatically chooses storage strategy based on model size:

- **< 500k variables**: In-memory (fast access)
- **≥ 500k variables**: Disk-backed lazy storage (low memory)

```python
# Automatic selection
sol = no.extract_solution(solver, m)

# Force lazy storage for large models
sol = no.extract_solution(solver, m, lazy=True, directory='./solution_data')

# Force in-memory for small models
sol = no.extract_solution(solver, m, lazy=False)
```

## Supported Solvers

- **HiGHS** (open-source, recommended)
- **GAMS/CPLEX** (commercial, requires GAMS installation)

## API Reference

### Core Classes

- `Set(name, elements)` - Index domain
- `Param(name, sets, data)` - Constant parameter (nimblend Array)
- `Model(name, sense)` - Optimization model container
- `Variable` - Decision variable (created via `model.var()`)

### Functions

- `Sum(*sets, expr)` - Summation over sets
- `extract_solution(solver, model, **kwargs)` - Extract solution with smart storage
- `to_csv(solution, filename)` - Export solution to CSV

### Model Methods

- `model.var(name, sets, lb, ub, vtype)` - Add variable
- `model.eq(name, constraint)` - Add constraint
- `model.set_objective(expr)` - Set objective function
- `model.to_lp(filename)` - Write LP file

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/ tests/

# Build Rust extension (optional)
cd rust && maturin develop --release
```

## Comparison with Other Libraries

| Library | Backend | Lazy Expansion | Rust Acceleration | Solver Support |
|---------|---------|----------------|-------------------|----------------|
| **nimopt** | nimblend | ✅ | ✅ | HiGHS, GAMS/CPLEX |
| linopy | xarray | ❌ | ❌ | Many (via linopy) |
| Pyomo | Native | ❌ | ❌ | Many (via Pyomo) |
| PuLP | Native | ❌ | ❌ | Many (via PuLP) |

## Citation

If you use nimopt in academic work, please cite:

```bibtex
@software{nimopt2024,
  title = {nimopt: Algebraic Optimization Modeling with Lazy Constraint Expansion},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/nimopt}
}
```
