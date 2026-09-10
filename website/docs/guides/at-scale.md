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
