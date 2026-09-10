---
title: Model
description: The model that contains the columns, the rows and the objective, and the matrix it assembles.
---

# Model

## `Model`

```
Model(name="model", sense="min")
```

A model contains one column space, the constraints declared against it, and
an objective. `name` labels it and is otherwise unused. `sense` is `"min"` or
`"max"`, set once here. Any other value raises `ValueError`.

| Member | Returns |
| --- | --- |
| `var(name, sets, subset=None, lower=0.0, upper=inf, integer=False)` | a `Variable` occupying the next range of columns |
| `constraint(name, relation, where=None, over=None)` | a `Constraint` occupying the next range of rows |
| `set_objective(expression)` | nothing; sets the objective |
| `sense` | `"min"` or `"max"`, as declared |
| `solve(solver="highs", options=None)` | a `Solution` |
| `assemble()` | an `Assembled`: the matrix, with no solver involved |
| `n_columns`, `n_rows`, `nnz` | the shape declared so far |
| `column_bounds()` | the lower and upper bound vectors, in column order |
| `integrality()` | one flag per column |
| `objective_coefficients()` | one coefficient per column |
| `explain()` | an `Explanation` of what the model built |
| `to_yaml(inline=False, instructions=False)` | the text of this model's file, with its data inline where asked and the comment block that describes the format where asked |
| `objective` | the objective expression, or `None` |

Declaring costs shapes, not blocks: `n_rows` and `nnz` are known when a
constraint is added, and no matrix exists until `assemble` or `solve`.

```python
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))
m.constraint("supply", Sum(W, x[P, W]) <= 30.0)
m.constraint("total", Sum(P, W, x[P, W]) <= 100.0)
m.set_objective(Sum(P, W, x[P, W]))

print(m.n_columns, m.n_rows, m.nnz)
print(m.objective_coefficients())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
6 3 12
[1. 1. 1. 1. 1. 1.]
```

</details>
<!-- /output -->

## `Assembled`

The model's matrix in CSR form, returned by `assemble`. `indices` and
`values` are views of the one buffer the model allocated; only `indptr` is
built.

| Member | Returns |
| --- | --- |
| `indptr`, `indices`, `values` | the matrix in CSR form |
| `n_rows`, `n_cols` | its shape |
| `row_lower`, `row_upper` | one bound per row |
| `col_lower`, `col_upper`, `col_cost`, `integrality` | one entry per column |
| `row_of(name)` | a constraint's rows, as a slice |
| `to_dense()` | the matrix as an ndarray |

```python
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))
m.constraint("supply", Sum(W, x[P, W]) <= 30.0)

assembled = m.assemble()
print(assembled.n_rows, assembled.n_cols)
print(assembled.indptr)
print(assembled.row_of("supply"))
print(assembled.to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
2 6
[0 3 6]
slice(0, 2, None)
[[1. 1. 1. 0. 0. 0.]
 [0. 0. 0. 1. 1. 1.]]
```

</details>
<!-- /output -->

`to_dense` is for a small model. A model of any size is read through
`row_of` and the CSR arrays.

## What a model built

`explain()` reports every declaration with the count it built, and has
`built=True`. It returns the record type a `Definition` returns with every
count absent, and one reader covers both.

A model contains variables and constraints. Its sets and parameters are
collected from them, in order of first appearance. A dimension introduced by
a coefficient belongs to no variable and is found through the parameter that
has it.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))

m = Model("transport")
x = m.var("x", (P, W))
m.constraint("supply", Sum(W, cost[P, W] * x[P, W]) <= supply[P])
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))

print(m.explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
transport  min  6 columns · 2 rows · 6 nonzeros
  sets        P 2 · W 3
  parameters  cost (P,W) 6 · supply (P) 2
  variables   x (P×W) 6 cols [0.0, inf]
  constraint  supply (P)  Sum(W, cost[P, W] * x[P, W]) <= supply[P]  2 rows  6 nz
  objective   min  Sum(P, W, cost[P, W] * x[P, W])
```

</details>
<!-- /output -->
