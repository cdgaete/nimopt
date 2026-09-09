---
title: Get started
description: Install nimopt, build a transport model, solve it, and read the solution back over its index sets.
---

# Get started

## Install

```bash
pip install nimopt
```

`nimblend` is installed as a dependency. HiGHS is the default solver.

## A transport model

Two plants, Lisbon and Porto, ship to three warehouses, Berlin, Paris and
Rome. Plant `p` has supply `s[p]`, warehouse `w` has demand `d[w]`, and one
unit shipped on route `(p, w)` costs `c[p, w]`. The decision variable
`x[p, w]` is the quantity shipped on each route.

```text
minimise    Σ_{p,w} c[p,w] · x[p,w]
subject to  Σ_w x[p,w] ≤ s[p]        for each plant p
            Σ_p x[p,w] ≥ d[w]        for each warehouse w
            x[p,w] ≥ 0
```

In `nimopt`, the sets index every declaration, the parameters hold the data,
`m.var` declares the decision variable, `m.eq` adds each constraint family
under a name, and `set_objective` sets the objective function.

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

m.eq("supply", Sum(W, x[P, W]) <= supply[P])
m.eq("demand", Sum(P, x[P, W]) >= demand[W])
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))

solution = m.solve()
print(solution.status, solution.objective)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
optimal 135.0
```

</details>
<!-- /output -->

The solver reports an optimal solution with objective 135.

## Reading the solution

`primal("x")` returns the shipments as an array indexed over the sets `x`
was declared on. `dual("demand")` returns the dual value of each demand row:
the change in the objective per unit increase in that warehouse's demand.

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
m.eq("supply", Sum(W, x[P, W]) <= supply[P])
m.eq("demand", Sum(P, x[P, W]) >= demand[W])
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))
solution = m.solve()

shipped = solution.primal("x")
print(shipped.dims)
print(shipped.to_dense())
print(solution.dual("demand").to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W')
[[20.  0. 10.]
 [ 0. 15.  5.]]
[3. 1. 6.]
```

</details>
<!-- /output -->

Rows are plants and columns are warehouses. Lisbon ships 20 to Berlin and 10
to Rome; Porto ships 15 to Paris and 5 to Rome. The duals of the demand rows
are 3, 1 and 6: the marginal cost of one additional unit at each warehouse.

Every code block in this documentation is self-contained, which is why the
second block repeats the model. Each block can be pasted into a Python
session as it is, or opened in the playground with "Run this example".

## Next

- [Vocabulary](/vocabulary) defines the terms used throughout: set, member,
  frame, row, absence, and others.
- The [tutorial](/tutorial/sets-and-parameters) builds this model one concept
  per page.
- The [guides](/guides/subsets) cover sparse networks, time lags, conditions
  on rows, and bounds from data.
- [Explanation](/explanation/a-variable-is-a-dimension) covers the design and
  its costs.
