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
| `piecewise(name, x, x_points, y, y_points, sign, method, active=None, relaxed=False, where=None)` | a `Piecewise`; declares the variables and constraints of its method |
| `set_objective(expression)` | nothing; sets the objective |
| `sense` | `"min"` or `"max"`, as declared |
| `solve(solver="highs", options=None)` | a `Solution` |
| `assemble()` | an `Assembled`: the matrix, with no solver involved |
| `n_columns`, `n_rows`, `nnz` | the shape declared so far |
| `column_bounds()` | the lower and upper bound vectors, in column order |
| `integrality()` | one flag per column |
| `objective_coefficients()` | one coefficient per column |
| `explain()` | an `Explanation` of what the model built |
| `to_yaml(inline=False, instructions=False, version=4)` | the text of this model's file, with its data inline where asked and the comment block that describes the format where asked; `version=3` writes the declarations a piecewise declaration generated in its place |
| `piecewise_declarations` | the piecewise declarations, keyed by name |
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

## `Piecewise`

```
Model.piecewise(name, x, x_points, y, y_points, sign, method, active=None, relaxed=False, where=None)
Definition.piecewise(name, x, x_points, y, y_points, sign, method, active=None, relaxed=False, where=None)
```

A piecewise-linear relation of the expression `y` to the expression `x`.
`x` is on the curve through `x_points` and `y_points`. `sign` compares `y`
with the curve: `"=="`, `"<="` or `">="`. The two points are parameters read
at their sets. Each is over some or all of the sets of `x` and over one
breakpoint set, the one set `x` is not over. An entity lists its first
breakpoints, and its last breakpoints may be absent. An entity with no
breakpoint has no generated rows and no generated columns.

`where` restricts the declaration to some entities: a parameter, a tuple of
sets or a domain over the sets of `x_points` other than the breakpoint set.
The breakpoint checks, the generated columns and the generated rows cover
the entities at its coordinates. `x`, `y` and `active` are compared at those
coordinates only.

| `method` | Generates | Requires |
| --- | --- | --- |
| `"incremental"` | per segment, one continuous and one integer column and their rows | breakpoints strictly increasing or strictly decreasing |
| `"tangent"` | one row per segment, and two rows that keep `x` between the first and the last breakpoint | points convex under `>=`, concave under `<=`; no `active`; no `==`; no constant in `x` |

`active` is a binary variable over the sets of `x`, or a sum of them. Where
it is 0, `x` is 0 and `y` is compared with 0. A term that is scaled or
bounded outside 0 and 1 raises ValueError, and so does a continuous term
under the default. `relaxed=True` accepts a continuous `active` between 0
and 1 and scales the curve by its value, which is the linear relaxation of
the switch. `relaxed=True` with no `active` raises ValueError. `Model.piecewise` generates the declarations at
once. `Definition.piecewise` stores the declaration, and `build` generates
them. A generated name is `name`, an underscore and a suffix:

| `method` | Sets | Parameters | Variables | Constraints |
| --- | --- | --- | --- | --- |
| `"incremental"` | `segment` | `members`, `x_step`, `y_step`, `x_first`, `y_first` | `fill`, `order` | `x`, `y`, `order_bound`, `fill_order`, `order_link`, `active` |
| `"tangent"` | `segment` | `slope`, `intercept`, `x_low`, `x_high` | none | `tangent`, `x_min`, `x_max` |

`{name}_active` exists only where `active` is given. The members of
`{name}_segment` are the breakpoint set's members without the first. A
segment is identified by its end breakpoint.

| Member | Contains |
| --- | --- |
| `name`, `x`, `x_points`, `y`, `y_points`, `sign`, `method`, `active`, `relaxed`, `where` | the arguments |
| `breakpoints` | the name of the breakpoint set |
| `names()` | the names the declaration generates, keyed by `"sets"`, `"parameters"`, `"variables"` and `"constraints"` |
| `generated` | the names a model generated, keyed the same way; empty on a definition |
| `generated_names()` | every generated name, as a frozenset |

An argument error raises when the declaration is made. `TypeError` is
raised for an `x`, `y` or `active` that is not an expression, and for points
that are not a parameter read at its sets. `ValueError` is raised for a name
that is not a Python identifier, an unknown `method` or `sign`, expressions
over different sets, points without exactly one breakpoint set, points over
different sets, `"tangent"` with `"=="`, with `active` or with a constant in
`x`, an `active` with a constant, a `where` of another type or over other sets
than the entity sets of `x_points`, and a generated name the model or
definition declares.

A breakpoint error raises `ValueError` when the data is bound, before any
declaration, and identifies the first entity at fault: points with no
breakpoint, points present at different breakpoints, an entity with one
breakpoint, an absent breakpoint before a present one, a value that is not
finite, breakpoints that are not strictly monotonic, and, for `"tangent"`,
points whose curvature does not match `sign`.

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
