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
| `feasible` | whether the solver reports a primal-feasible point |
| `objective` | the objective value of that point |
| `bound` | the bound on the optimal objective the solver proved, or `None` |
| `gap` | the relative distance from the objective to the bound, or `None` |
| `primal(name)` | the named variable's values over its own sets |
| `dual(name)` | the named constraint's duals over its free sets |

`status` and `feasible` are readable whatever the solver reported.
`objective` and `primal` raise `ValueError` where `feasible` is False. They
raise at status `unbounded` and `unbounded_or_infeasible` whatever `feasible`
reports, and `gap` raises there too. An unbounded model has no optimal value.
A solve stopped at a limit reports `feasible` True where the solver found a
point, and those reads then return it. `dual` raises `ValueError` where
`status` is not `optimal`. Read `status` first.

`bound` is a lower bound on the optimal objective under sense `min` and an
upper bound under sense `max`. It is `None` where the solver reports none.
For a model without integer columns it is the objective at status `optimal`
and `None` at any other status. `bound` is readable at every status, and the
solvers report none at status `unbounded` and `unbounded_or_infeasible`.
`gap` is `abs(objective - bound) / abs(objective)`, and is `None` where
`feasible` is False or `bound` is `None`.

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

Reading a value where the solver reports no feasible point raises
`ValueError`; the message gives the status.

```python raises=ValueError
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("infeasible")
x = m.var("x", (P, W))
m.constraint("floor", Sum(W, x[P, W]) >= 10.0)
m.constraint("ceiling", Sum(W, x[P, W]) <= 1.0)
m.set_objective(Sum(P, W, x[P, W]))

m.solve().objective
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: status is 'infeasible' and the solver reports no feasible point; read `status` before reading values
```

</details>
<!-- /output -->

## The array type of a value

A variable over a full product has a value at every cell of its frame. The
solver returns those values in column order, and they reshape into a
`DenseArray` with no index built. A variable over a subset has values at its
members alone. A dense frame would be the grid the declaration avoids, and
those values remain a `SparseArray`. A dual follows the rows of its
constraint by the same rule.

Every array declares `absence="unknown"`. A coordinate the model does not
have has no value, and combining the results of two models adds no zero for
it.
