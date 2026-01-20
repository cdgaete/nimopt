# nimopt: Rust-Accelerated Direct HiGHS Solver

## Summary

Implemented a high-performance direct HiGHS solver interface for nimopt that bypasses LP file generation entirely. The solver builds the constraint matrix directly in memory using Rust acceleration, achieving **2-4x faster performance** than JuMP, linopy, and pyoptinterface across all model sizes.

---

## Performance Results

### 2 Million Variables (1000×2000 transport problem)

| Library | Build | Solve | **Total** | vs nimopt |
|---------|-------|-------|-----------|-----------|
| **nimopt** | 327 ms | 4,076 ms | **4,404 ms** | 1.0x |
| linopy | 164 ms | 8,745 ms | 8,909 ms | 2.0x slower |
| JuMP (Julia) | 1,671 ms | 9,285 ms | 10,956 ms | 2.5x slower |
| pyoptinterface | 12,243 ms | 4,271 ms | 16,514 ms | 3.7x slower |

### Scaling Comparison

| Size | Vars | JuMP | linopy | pyoptint | **nimopt** |
|------|------|------|--------|----------|------------|
| 30×30 | 900 | 3.7 ms | 64 ms | 7 ms | **1.8 ms** |
| 100×100 | 10k | 33.6 ms | 90 ms | 74 ms | **17 ms** |
| 200×200 | 40k | 141 ms | 186 ms | 288 ms | **62 ms** |
| 300×300 | 90k | 336 ms | 366 ms | 654 ms | **145 ms** |
| 400×400 | 160k | 815 ms | 610 ms | 1,210 ms | **258 ms** |

### nimopt Internal Comparison (90k variables)

| Approach | Time | Speedup |
|----------|------|---------|
| LP file (Python) | 5,866 ms | baseline |
| LP file (Rust) | 396 ms | 14.8x |
| Direct (Python) | 297 ms | 19.7x |
| **Direct (Rust)** | **130 ms** | **45.1x** |

---

## Implementation Details

### New Files

**`src/nimopt/solvers/highs_direct.py`**
- `HiGHSDirectSolver` class - builds HiGHS model in memory
- `_build_matrices_rust_fast()` - Rust-accelerated matrix building
- Lazy variable name generation (only when needed for output)
- Falls back to Python for complex constraint patterns

**`rust/src/lib.rs`** (new functions)
- `build_sum_csr_fast()` - Parallel CSR matrix building using rayon
- `generate_var_names()` - Parallel variable name generation

### Key Optimizations

1. **No file I/O** - Builds HiGHS model directly in memory via CSR matrix
2. **Parallel Rust CSR building** - Uses rayon for parallel constraint generation
3. **Lazy name generation** - Variable names only generated when `get_variable_names()` called
4. **Skip var_idx dict** - Rust path computes indices via strides, no string→index mapping
5. **Direct numpy passing** - Coefficient arrays passed directly from nimblend to Rust

### Data Flow

```
nimblend Array.values (numpy) → Rust build_sum_csr_fast → numpy CSR arrays → HiGHS
```

No intermediate Python loops or string mapping in the hot path.

---

## Usage

```python
import nimopt as no
from nimopt.solvers import HiGHSDirectSolver

# Create model
m = no.Model(sense='minimize')
x = m.var('x', [i, j], lb=0)
m.set_objective(no.Sum(i, j, cost[i,j] * x[i,j]))
m.eq('supply', no.Sum(j, x[i,j]) <= supply[i])
m.eq('demand', no.Sum(i, x[i,j]) >= demand[j])

# Solve directly (no LP file)
solver = HiGHSDirectSolver(use_rust=True)
solver.load_model(m)
result = solver.solve()

print(f"Objective: {result.objective_value}")
print(f"Status: {result.status}")
```

---

## Profile Breakdown (2M variables)

| Component | Time | % |
|-----------|------|---|
| Model definition | 84 ms | 2% |
| Solver load (Rust) | 229 ms | 5% |
| HiGHS solve | 4,076 ms | 93% |
| **Total** | **4,389 ms** | 100% |

The overhead (build + load) is only **7%** of total time - the rest is pure solver time.

---

## Files Changed

```
src/nimopt/solvers/highs_direct.py  (new)
src/nimopt/solvers/__init__.py      (export HiGHSDirectSolver)
rust/src/lib.rs                     (+380 lines)
tests/test_highs_direct.py          (new)
benchmarks/bench_direct.py          (new)
```

## Commit

```
feat: Add Rust-accelerated direct HiGHS solver
commit edd696e
```
