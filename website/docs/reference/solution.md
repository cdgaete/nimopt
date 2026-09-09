---
title: Solution
description: Status, objective, primals and duals, returned over the sets they were declared over.
---

# Solution

## `Solution`

Returned by `Model.solve`. Primal and dual values, returned over the sets
they were declared over.

| Member | Returns |
| --- | --- |
| `status` | the outcome the solver reported |
| `objective` | the optimal objective value |
| `primal(name)` | the named variable's values over its own sets |
| `dual(name)` | the named constraint's duals over its free sets |

`status` is readable whatever the solver reported. `objective`, `primal`
and `dual` are not: a model the solver did not bring to an optimum has no
answer, and a vector it left behind would be indistinguishable from one.
Read `status` first.

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

print(solution.status)
print(solution.objective)
print(solution.primal("x").to_dense())
print(solution.dual("demand").to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
optimal
135.0
[[20.  0. 10.]
 [ 0. 15.  5.]]
[3. 1. 6.]
```

</details>
<!-- /output -->

Reading a value from a model that did not reach an optimum raises
`ValueError`; the message gives the status.

```python raises=ValueError
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("infeasible")
x = m.var("x", (P, W))
m.eq("floor", Sum(W, x[P, W]) >= 10.0)
m.eq("ceiling", Sum(W, x[P, W]) <= 1.0)
m.set_objective(Sum(P, W, x[P, W]))

m.solve().objective
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: the model's status is 'infeasible', so it carries no objective; read `status` before reading values
```

</details>
<!-- /output -->

## The array type of a value

A variable over a full product has a value at every cell of its frame, and
the solver returns them in column order, so they reshape into a
`DenseArray` with no index built at all. A variable over a subset has
values at its members alone, and a dense frame would be the grid it was
declared to avoid, so those stay a `SparseArray`. A dual follows its
constraint's rows by the same rule.

Every array declares `absence="unknown"`: a coordinate the model did not
have has no value, and combining two models' results must not invent a zero
for it.
