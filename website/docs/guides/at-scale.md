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
columns and 46,354,588 nonzeros, and 8,801,358 of those rows (45.3%) have one
coefficient. Without those rows the same network has 10,622,520 rows and
37,553,230 nonzeros, and the arrays passed to a solver take 888 MB instead of
1,169 MB.

| Full-year European network, measurement | Every row | One coefficient as a bound |
| --- | --- | --- |
| Mosek 11.2, barrier without crossover at 16 threads, peak of the process | 41.1 GB | 37.8 GB |
| Mosek 11.2, peak of the presolve | 19.5 GB | 13.5 GB |
| CPLEX through GAMS 43.4, peak before the factorization | 21.5 GB | 16.8 GB |
| HiGHS 1.15.1, copy of the model | 1,632 MB | 1,210 MB |
| HiGHS 1.15.1, peak of `presolve()` above the copy | 11,797 MB | 9,259 MB |

Mosek 11.2 solves the full-year European network at 16 threads, barrier
without crossover, to status `optimal` from both forms, with the objective
1,128,619,364,130.2 in both, and its factorization adds 25 GB to 27 GB in
both. CPLEX through GAMS 43.4 loads the GMO in 47 s with every row and in
34 s without the singleton rows. Mosek, CPLEX and HiGHS each build the same
presolved model from both forms.

The duals of a row stored as a bound are computed from the reduced costs. At
24 snapshots with Mosek and crossover, both forms return the objective
302,099,279,596.448, and the duals of the 76,866 rows absent from the matrix
equal the duals of the solve with every row within 1e-6. Every coefficient of
those rows is 1 or -1 at 24 snapshots, and 78,737 of the 79,457 columns have
no bound of their own there.

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
