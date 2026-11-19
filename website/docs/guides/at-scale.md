---
title: Declaring a model at scale
description: Read a model's shape as it is declared, build the matrix once, and read a large solution.
---

# Declaring a model at scale

Models with millions of columns are declared the same way as small ones.
A declaration computes shapes, not blocks. A constraint computes its row and
nonzero counts when it is added, and no matrix exists until `assemble()` or
`solve()` is called.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array([f"p{i}" for i in range(200)]))
W = Set("W", np.array([f"w{i}" for i in range(100)]))
cost = Param.from_dense("cost", (P, W), np.ones((200, 100)))

m = Model("transport")
x = m.var("x", (P, W))
m.constraint("supply", Sum(W, x[P, W]) <= 1.0)
m.constraint("demand", Sum(P, x[P, W]) >= 1.0)
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))

print(m.n_columns)
print(m.n_rows)
print(m.nnz)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
20000
300
40000
```

</details>
<!-- /output -->

Twenty thousand columns, three hundred rows and forty thousand nonzeros are
declared, and the model contains no matrix. The parameter is the data of the
caller. The model has added two constraints and an objective, each a symbolic
expression recording the variable, the coefficient and the sets that are
summed.

## Building the matrix once

`assemble()` builds the matrix into one buffer and returns it in CSR form.
Each constraint writes its rows into its own slice. The matrix therefore
exists once, and one expression at a time is materialised.

```python
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array([f"p{i}" for i in range(200)]))
W = Set("W", np.array([f"w{i}" for i in range(100)]))

m = Model("transport")
x = m.var("x", (P, W))
m.constraint("supply", Sum(W, x[P, W]) <= 1.0)
m.constraint("demand", Sum(P, x[P, W]) >= 1.0)

assembled = m.assemble()
print(assembled.n_rows, assembled.n_cols)
print(assembled.values.shape)
print(assembled.row_of("demand"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
300 20000
(40000,)
slice(200, 300, None)
```

</details>
<!-- /output -->

`indices` and `values` are views of that buffer; only `indptr` is built.
`row_of(name)` gives the rows of a constraint as a slice. A caller locates a
named row block in a matrix too large to print.

## A row with one coefficient

A declared row with one coefficient is stored as a bound of its column, and
the matrix passed to the solver has no entry for it. The full-year European
network through the PyPSA backend declares 19,423,878 rows, 9,361,137
columns and 46,354,588 nonzeros, and 8,801,358 of those rows (45.3%) are
[singleton rows](/reference/model#a-singleton-row). The matrix the solver
loads has 10,622,520 rows and 37,553,230 nonzeros.

`benchmarks/bench_pypsa_backend.py` measures both values of `singleton_rows`
on that network. The build is three runs of each value. The solve is Mosek
11.2, barrier without crossover, at 16 threads.

| Full-year European network | `"rows"` | `"bounds"` |
| --- | --- | --- |
| Loaded rows | 19,423,878 | 10,622,520 |
| Matrix | 556.3 MB | 450.6 MB |
| Build | 11.2 s to 11.6 s | 11.5 s to 11.8 s |
| Peak of the process in the build | 3,915 MB to 3,917 MB | 3,636 MB to 3,637 MB |
| Solve | 1,597 s | 1,575 s |
| Peak of the process in the solve | 42.2 GB | 38.5 GB |
| Solution and duals read onto the network | 6.4 s | 6.2 s |

Both solves return status `optimal` and the objective 1,058,936,103,870.507.
Mosek reduces both matrices to the same presolved model of 6,457,581
constraints, and its factorization is the same for both. The build takes
0.2 s to 0.5 s longer under `"bounds"`: each constraint counts the
coefficients of its rows at its declaration, and a constraint with singleton
rows reads them once more at the assembly.

The arrays of both matrices of the same network, passed to each solver
adapter directly:

| Full-year European network, peak of the process | Every row | Singleton rows as bounds |
| --- | --- | --- |
| Mosek 11.2, in its presolve | 19.5 GB | 13.5 GB |
| CPLEX through GAMS 43.4, before its factorization | 21.5 GB | 16.8 GB |
| HiGHS 1.15.1, copy of the model | 1,632 MB | 1,210 MB |
| HiGHS 1.15.1, `presolve()` above the copy | 11,797 MB | 9,259 MB |

CPLEX through GAMS 43.4 loads the GMO in 47 s with every row and in 34 s
without the singleton rows. HiGHS and CPLEX each build the same presolved
model from both matrices.

The duals of a row stored as a bound are computed from the reduced costs. At
24 snapshots with Mosek and crossover, both values return the objective
232,416,019,336.747, and the duals of the 76,866 rows stored as bounds equal
the duals of the solve with every row: the largest difference is 9.3e-9.

A session reads fewer rows as well. `benchmarks/bench_session.py` changes one
column bound of a storage model eight times: a resumed solve takes 70 ms, and
a rebuild with a solve from no basis 1,075 ms.

`singleton_rows="rows"` writes every row into the matrix.

## Reading a large solution

`to_dense()` is for a model small enough to print. At scale, read the CSR
arrays, or read the solution through `primal()` and `dual()`. Both return
labeled arrays over the sets, not offsets into a vector.

A variable over a subset keeps its values sparse. Reading a solution over a
sparse network therefore builds no grid.

## What a variable costs

A member's column is computed from its multi-index and is not stored. A
variable over millions of columns costs its members, not its columns.
`n_columns` above is twenty thousand, and the variable stores a few numbers:
the start of its block and the sizes of its sets.

## Progress

A large model takes seconds to build. `progress=True` reports the build in a
terminal.

```python skip="the report draws to a terminal, and a page is not one"
from nimopt.models import storage

model = storage.definition().build(storage.data(60), progress=True)
solution = model.solve(options={"time_limit": 300.0, "log": True}, progress=True)
```

A solve stopped at `time_limit` reports `feasible` True where the solver
found a point. `objective`, `primal`, `bound` and `gap` then read that
point. See [Solving](/tutorial/solving).
