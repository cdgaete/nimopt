# nimopt Roadmap

This document outlines planned features and enhancements for nimopt.

## Version 0.3.0 - Model Validation & Debugging

**Priority**: HIGH  
**Status**: Planned

### Features

```python
# Model validation
m.validate()                    # Check model consistency
m.stats()                       # Variables, constraints, nonzeros
m.print_constraint('supply')    # Show expanded form
m.explain_infeasible()          # IIS-like analysis (requires solver support)

# Constraint inspection
m.get_constraint('supply')      # Get constraint object
m.list_constraints()            # List all constraint names
```

### Implementation Notes

- `validate()`: Check for undefined variables, dimension mismatches
- `stats()`: Count variables, constraints, nonzeros, sparsity
- `print_constraint()`: Expand symbolic constraint to show actual terms
- `explain_infeasible()`: Requires IIS support from solver (HiGHS, CPLEX)

---

## Version 0.4.0 - Set Operations

**Priority**: MEDIUM  
**Status**: Planned

### Features

```python
# Set algebra
k = i | j               # Union
k = i & j               # Intersection
k = i - j               # Difference
k = i * j               # Cartesian product (tuples)

# Aliasing for same-set indexing
ii = i.alias('ii')
m.eq('no_self', x[i, ii] == 0, where=(i == ii))
```

### Implementation Notes

- Set operations return new `Set` objects
- Cartesian product creates tuple elements
- Aliases enable same-set constraints (e.g., network flows)

---

## Version 0.5.0 - Parameter Operations

**Priority**: MEDIUM  
**Status**: Planned

### Features

```python
# Slicing
cost['seattle', :]      # Get all costs from Seattle
cost[:, 'newyork']      # Get all costs to New York

# Filtering
high_cost = cost.where(cost > 2)

# Aggregation
total = cost.sum()              # Total sum
by_origin = cost.sum('j')       # Sum over destination
```

### Implementation Notes

- Leverage nimblend's existing slicing/filtering capabilities
- Return new `Param` objects for sliced/filtered results
- Aggregation returns scalar or reduced-dimension `Param`

---

## Version 0.6.0 - Data I/O

**Priority**: HIGH  
**Status**: Planned

### Features

```python
# Import from CSV
i = Set.from_csv('cities.csv', column='city')
cost = Param.from_csv('costs.csv', index=['origin', 'dest'], value='cost')

# Import from SQL
import sqlite3
conn = sqlite3.connect('data.db')
demand = Param.from_sql(conn, "SELECT city, demand FROM demands", 
                        index=['city'], value='demand')

# Export to CSV
cost.to_csv('costs_export.csv')

# Export to NetCDF (via nimblend)
cost.to_netcdf('costs.nc')
```

### Implementation Notes

- CSV: Use pandas for reading, convert to nimblend Array
- SQL: Support SQLite, PostgreSQL, MySQL via sqlalchemy
- Export: Leverage nimblend's existing I/O capabilities

---

## Version 0.7.0 - Advanced Constraints

**Priority**: LOW  
**Status**: Planned

### Features

```python
# Special Ordered Sets
m.sos1('choice', [x[i] for i in items])
m.sos2('piecewise', points)

# Ranged constraints
m.eq('range', (10, Sum(i, x[i]), 20))  # 10 <= Sum(i, x[i]) <= 20

# Indicator constraints
m.indicator(y[i], x[i] >= 10)  # If y[i] == 1, then x[i] >= 10
```

### Implementation Notes

- SOS constraints require solver support (HiGHS, CPLEX, Gurobi)
- Ranged constraints: Convert to two separate constraints internally
- Indicator constraints: Use solver-specific syntax in LP file

---

## Version 0.8.0 - Model Management

**Priority**: MEDIUM  
**Status**: Planned

### Features

```python
# Fix variables
m.fix('x', {('seattle', 'newyork'): 50})

# Relax integrality
m.relax('x')            # Integer -> continuous
m.enforce('x')          # Restore integrality

# Disable/enable constraints
m.disable('supply')     # Temporarily disable constraint
m.enable('supply')      # Re-enable constraint

# Clone model
m2 = m.copy()           # Deep copy of entire model
```

### Implementation Notes

- `fix()`: Set variable bounds to fixed value
- `relax()`: Change `vtype` from integer/binary to continuous
- `disable()`/`enable()`: Skip constraint during LP write
- `copy()`: Deep copy all sets, params, variables, constraints

---

## Version 0.9.0 - Performance Enhancements

**Priority**: MEDIUM  
**Status**: Planned

### Features

- **Parallel LP writing**: Use Rust's rayon for multi-threaded constraint expansion
- **Sparse matrix storage**: Use scipy.sparse for large, sparse coefficient matrices
- **Incremental LP updates**: Only rewrite changed constraints
- **Memory profiling**: Track memory usage during model building

### Implementation Notes

- Rayon parallelization in Rust extension
- Detect sparsity threshold and switch to sparse storage
- Track model changes and generate incremental LP diffs

---

## Version 1.0.0 - Production Ready

**Priority**: HIGH  
**Status**: Planned

### Requirements for 1.0 Release

- [ ] All core features implemented (validation, I/O, constraints)
- [ ] Comprehensive test suite (>90% coverage)
- [ ] Full documentation (API reference, tutorials, examples)
- [ ] Benchmarks vs. linopy, Pyomo, PuLP
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Published to PyPI
- [ ] Rust extension available as pre-built wheels (Linux, macOS, Windows)

---

## Future Considerations (Post-1.0)

### Nonlinear Programming

- Support for quadratic objectives (QP, QCQP)
- Nonlinear constraints (requires different solvers: IPOPT, KNITRO)

### Stochastic Programming

- Scenario-based modeling
- Chance constraints
- Two-stage/multi-stage models

### Decomposition Methods

- Benders decomposition
- Dantzig-Wolfe decomposition
- Column generation

### Additional Solvers

- Gurobi (direct Python API)
- CPLEX (direct Python API, not via GAMS)
- SCIP
- Cbc (direct interface)
- GLPK

### Cloud Solving

- Submit models to cloud solvers (Gurobi Cloud, NEOS)
- Async solving with callbacks

---

## Contributing

If you'd like to contribute to any of these features, please:

1. Open an issue to discuss the feature
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.
