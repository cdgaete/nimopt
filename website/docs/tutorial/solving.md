---
title: Solving
description: Set the objective function, call the solver, and check the status before reading values.
sidebar_position: 5
---

# Solving

The remaining pieces are the objective function and the solver call.

## The objective

`m.set_objective(expression)` takes an expression with an empty frame. The
model's `sense`, `"min"` by default, sets the direction. `m.solve()`
assembles the matrix, calls HiGHS and returns a `Solution`.

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
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
optimal
135.0
```

</details>
<!-- /output -->

The status is `optimal` and the objective value is 135.

## Status

`status` reports the outcome of the solve. `status` and `feasible` are
readable after any solve. `objective` and `primal` raise `ValueError` where
`feasible` is False. They raise at status `unbounded` and
`unbounded_or_infeasible` whatever `feasible` reports, and `gap` raises
there too. `dual` raises `ValueError` where `status` is not `optimal`.

Raising the demand of Berlin to 40 makes total demand 70 against total
supply 55. The model is infeasible.

```python raises=ValueError
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
demand = Param.from_dense("demand", (W,), np.array([40.0, 15.0, 15.0]))

m = Model("transport")
x = m.var("x", (P, W))
m.constraint("supply", Sum(W, x[P, W]) <= supply[P])
m.constraint("demand", Sum(P, x[P, W]) >= demand[W])
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))

solution = m.solve()
print(solution.status)
solution.objective
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
infeasible
ValueError: status is 'infeasible' and the solver reports no feasible point; read `status` before reading values
```

</details>
<!-- /output -->

For an infeasible model whose cause is not evident, `m.session()` keeps the
solver instance open and `diagnose()` returns the conflicting rows. See
[Solvers](/reference/solvers).

## A solve stopped at a limit

An option in `options()` stops the solver early. A solver stopped at a
limit reports the best point it found, and `feasible` is True for it.
`objective` and `primal` then return that point. `bound` returns what the
solver proved about the optimum, an upper bound under sense `max` and a
lower bound under sense `min`. `gap` returns the relative distance from the
objective to that bound.

A search stopped after one node returns the point the solver found there.
The thread count of the solver determines that point. The example below
reports the properties every such point has.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

rng = np.random.default_rng(1)
ITEM = Set("item", np.array([f"i{t}" for t in range(40)]))
BIN = Set("bin", np.array([f"b{t}" for t in range(5)]))
weight = Param.from_dense("weight", (BIN, ITEM), rng.uniform(1, 50, (5, 40)))
value = Param.from_dense("value", (ITEM,), rng.uniform(1, 100, 40))
capacity = Param.from_dense("capacity", (BIN,), np.full(5, 306.0))

m = Model("knapsack", sense="max")
x = m.var("x", (ITEM,), integer=True, upper=1.0)
m.constraint("capacity", Sum(ITEM, weight[BIN, ITEM] * x[ITEM]) <= capacity[BIN])
m.set_objective(Sum(ITEM, value[ITEM] * x[ITEM]))

solution = m.solve(options={"node_limit": 1})
print(f"status: {solution.status}")
print(f"feasible: {solution.feasible}")
print(f"the bound is above the objective: {solution.bound > solution.objective}")
print(f"the gap is positive: {solution.gap > 0.0}")
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
status: solution_limit
feasible: True
the bound is above the objective: True
the gap is positive: True
```

</details>
<!-- /output -->

The status identifies the limit the solver stopped at. HiGHS reports a stop
at `node_limit` as `solution_limit`. `gap` is `None` where the solver proved
no bound, and `feasible` is False where it found no point.

## The matrix

`m.assemble()` builds the coefficient matrix without a solver call and
returns it in CSR form. `to_dense()` renders it for a model of this size,
and `row_of(name)` gives the row range of a named constraint.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))

m = Model("transport")
x = m.var("x", (P, W))
m.constraint("supply", Sum(W, x[P, W]) <= supply[P])
m.constraint("demand", Sum(P, x[P, W]) >= demand[W])

assembled = m.assemble()
print(m.n_rows, m.n_columns, m.nnz)
print(assembled.row_of("demand"))
print(assembled.to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
5 6 12
slice(2, 5, None)
[[1. 1. 1. 0. 0. 0.]
 [0. 0. 0. 1. 1. 1.]
 [1. 0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 1. 0.]
 [0. 0. 1. 0. 0. 1.]]
```

</details>
<!-- /output -->

The six columns are the routes, Lisbon's three followed by Porto's. Rows 0
and 1 are the supply rows, each with a 1 under its plant's three routes.
Rows 2 to 4 are the demand rows, each with a 1 under the two routes into its
warehouse. `row_of("demand")` returns that range.

Next: [Reading the solution](/tutorial/reading-the-answer).
