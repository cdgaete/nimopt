---
title: Reading the solution
description: Read primal and dual values over their index sets, and the distinction between an absent value and a zero.
sidebar_position: 6
---

# Reading the solution

A solver returns primal and dual values as flat vectors. `nimopt` returns
them as arrays over the sets each variable and constraint was declared on.

## Primals and duals

`solution.primal(name)` returns a variable's values over its sets.
`solution.dual(name)` returns a constraint's dual values over its frame: for
the demand constraint, one value per warehouse.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))

m = Model("transport")
x = m.var("x", (P, W))
m.constraint("supply", Sum(W, x[P, W]) <= supply[P])
m.constraint("demand", Sum(P, x[P, W]) >= demand[W])
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))
solution = m.solve()

shipped = solution.primal("x")
print(shipped.dims)
print(shipped.to_dense())
print(solution.dual("demand").dims)
print(solution.dual("demand").to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W')
[[20.  0. 10.]
 [ 0. 15.  5.]]
('W',)
[3. 1. 6.]
```

</details>
<!-- /output -->

Rows are plants and columns are warehouses: Lisbon ships 20 to Berlin and
10 to Rome, Porto ships 15 to Paris and 5 to Rome. The dual of a demand row
is the increase in total cost per additional unit of demand at that
warehouse: 3 in Berlin, 1 in Paris, 6 in Rome. Lisbon's supply constraint
binds, so each marginal unit is served from Porto at the cost of Porto's
route.

## A variable over a subset

If Porto cannot ship to Rome, the variable is declared over the five
existing routes with `subset=`. The sixth route has no column, and the
solution has values at the five members only.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum, subset

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))

routes = subset(
    (P, W),
    {
        "P": np.array(["lisbon", "lisbon", "lisbon", "porto", "porto"]),
        "W": np.array(["berlin", "paris", "rome", "berlin", "paris"]),
    },
)

m = Model("transport")
x = m.var("x", (P, W), subset=routes)
m.constraint("supply", Sum(W, x[P, W]) <= supply[P])
m.constraint("demand", Sum(P, x[P, W]) >= demand[W])
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))
solution = m.solve()

print(x.n_columns)
print(solution.objective)
print(type(solution.primal("x")).__name__)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
5
135.0
SparseArray
```

</details>
<!-- /output -->

Five columns instead of six. The objective is unchanged at 135, because
Rome is served from Lisbon in both solutions. The array type differs: a
variable over a full product returns a `DenseArray`, a variable over a
subset a `SparseArray` with an entry per member and nothing elsewhere.

## Absence is not zero

The model contains no decision for the route Porto to Rome. Every array a
solution returns declares `absence="unknown"`. `to_dense()` raises
`ValueError` and fills the missing coordinate with no value the model did not
produce.

```python raises=ValueError
import numpy as np
from nimopt import Model, Param, Set, Sum, subset

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))
routes = subset(
    (P, W),
    {
        "P": np.array(["lisbon", "lisbon", "lisbon", "porto", "porto"]),
        "W": np.array(["berlin", "paris", "rome", "berlin", "paris"]),
    },
)

m = Model("transport")
x = m.var("x", (P, W), subset=routes)
m.constraint("supply", Sum(W, x[P, W]) <= supply[P])
m.constraint("demand", Sum(P, x[P, W]) >= demand[W])
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))
solution = m.solve()

solution.primal("x").to_dense()
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: this array declares absence 'unknown' and does not carry every coordinate of its frame, so densifying must state fill=<value> to place at the rest
```

</details>
<!-- /output -->

`to_dense(fill=...)` supplies the value for missing coordinates. `nan`
distinguishes a missing route from a route with zero shipment.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum, subset

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))
routes = subset(
    (P, W),
    {
        "P": np.array(["lisbon", "lisbon", "lisbon", "porto", "porto"]),
        "W": np.array(["berlin", "paris", "rome", "berlin", "paris"]),
    },
)

m = Model("transport")
x = m.var("x", (P, W), subset=routes)
m.constraint("supply", Sum(W, x[P, W]) <= supply[P])
m.constraint("demand", Sum(P, x[P, W]) >= demand[W])
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))
solution = m.solve()

values = solution.primal("x")
print(values.absence, values.nnz)
print(values.to_dense(fill=np.nan))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
unknown 5
[[15.  0. 15.]
 [ 5. 15. nan]]
```

</details>
<!-- /output -->

Lisbon serves Rome alone, and the Porto to Rome cell reads `nan`. A stored
`0.0` would denote a route that exists and ships nothing.

## Summary

The tutorial declared index sets and parameters, a decision variable,
expressions, two constraint families and an objective. It solved the model
and read the solution back over its sets. The [guides](/guides/subsets) cover
variables over sparse networks, conditions on rows, lags, bounds from data,
and models with millions of rows. The [worked models](/models) present ten
complete formulations.
