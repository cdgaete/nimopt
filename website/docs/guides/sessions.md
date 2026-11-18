---
title: A session that changes a model
description: Fix a column, set a bound or append a row on one open solver model, and solve it again.
---

# A session that changes a model

An iterative method solves one model many times with small changes. A sweep of one bound changes one column and keeps the rest of the model. A Benders loop adds a cut to the master and fixes first-stage values in a subproblem. `Model.session()` assembles the model once and loads it in the solver once. `fix`, `unfix`, `set_bound` and `add_row` change the loaded model, and `solve` solves the model again. With the simplex method HiGHS starts from the basis of the previous solve.

## The sweep

The model below dispatches three plants over twelve hours. The sweep fixes the output of the wind plant at hour 1 at three values, and each solve returns the cost of that point. A right-hand side has the dimensions of its constraint, and `cap` is declared over both sets.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["coal", "gas", "wind"]))
H = Set("H", np.arange(1, 13))
cost = Param.from_dense("cost", (P,), np.array([30.0, 60.0, 0.0]))
cap = Param.from_dense(
    "cap", (P, H), np.repeat(np.array([[140.0], [80.0], [120.0]]), 12, axis=1)
)
need = Param.from_dense(
    "need", (H,), np.array([200.0 + 5.0 * (h % 4) for h in range(1, 13)])
)

m = Model("loop")
g = m.var("g", (P, H), lower=0.0)
m.set_objective(Sum(P, H, cost[P] * g[P, H]))
m.constraint("capacity", g[P, H] <= cap[P, H])
m.constraint("demand", Sum(P, g[P, H]) == need[H])

with m.session() as session:
    print(session.n_columns, "columns,", session.n_rows, "rows")
    for wind in (0.0, 40.0, 80.0):
        session.fix(g["wind", 1], wind)
        solution = session.solve()
        print(f"wind at hour 1 fixed at {wind:g}: {solution.objective:,.2f}")
    print(f"dual of demand at hour 1: {solution.dual('demand').to_dense()[0]:.2f}")
    session.add_row("coal_cap_total", Sum(H, g["coal", H]) <= 1000.0)
    solution = session.solve()
    print(session.n_rows, "rows:", f"{solution.objective:,.2f}")

print(f"the model the session was opened on: {m.solve().objective:,.2f}")
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
36 columns, 48 rows
wind at hour 1 fixed at 0: 37,050.00
wind at hour 1 fixed at 40: 34,650.00
wind at hour 1 fixed at 80: 32,700.00
dual of demand at hour 1: 30.00
49 rows: 35,400.00
the model the session was opened on: 31,500.00
```

</details>
<!-- /output -->

The last line solves the model the session was opened on. The session fixed no column there and appended no row, and that solve returns the optimum of the declared model.

## Targets

A target is a variable or one reference to it. `x` and `x[P, W]` select the whole variable. `x["p0", W]` selects a slice, and `x["p0", "w0"]` selects one column. Any other expression raises `TypeError`. A coefficient, a sum, a lag or a condition does not select columns.

A value for `fix` is a number, or a `Param` for the whole variable. The `Param` pins the members at which it has a value.

## The bound rules

- A fix overrides the bounds the model declares, and the solver does not clamp it into them.
- A later fix replaces an earlier fix on the same column.
- A fix is finite, and a fix on a column of an integer variable is integral.
- `unfix` removes the fix. The columns then have the bounds the model declares, or the bounds a `set_bound` call gave them.
- `unfix` of a variable with no fixed column raises `ValueError`.
- `set_bound` requires at least one of `lower` and `upper`.
- `set_bound` takes a number for a slice, and a number or a `Param` with a value at every member for the whole variable.
- A fix stays in place across `set_bound`.

## Appended rows

`add_row(name, relation, where=None, over=None)` takes the arguments of `Model.constraint`. A relation free over a set appends one row per member. The appended rows follow the rows of the model, in the order of the calls.

`Solution.dual(name)` returns the duals of an appended constraint, and `Diagnosis` reports an appended row by the name of that constraint. A name the model declares raises `ValueError`. A variable of another model raises `ValueError`.

```python
import numpy as np
from nimopt import Model, Set, Sum

i = Set("i", np.array(["a", "b"]))
m = Model("two")
x = m.var("x", (i,), lower=0.0, upper=10.0)
m.constraint("total", Sum(i, x[i]) >= 4.0)
m.constraint("floor_b", x["b"] >= 1.0)
m.set_objective(Sum(i, x[i]))

with m.session() as session:
    print(f"first solve: {session.solve().objective:,.2f}")
    session.add_row("cut", Sum(i, x[i]) >= 7.0)
    solution = session.solve()
    print(session.n_rows, "rows")
    dual = float(solution.dual("cut").to_dense())
    print(f"objective {solution.objective:,.2f}, dual of cut {dual:,.2f}")
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
first solve: 4.00
3 rows
objective 7.00, dual of cut 1.00
```

</details>
<!-- /output -->

## What a session does not change

- The model the session was opened on, and a later solve of that model.
- The columns. A row over a variable the model does not declare raises `ValueError`.
- A right-hand side or a coefficient of a row already loaded.
- The kind of a column.
- The declared rows. No method removes a row.

A session on the Gurobi, Mosek or GAMS adapter raises `ValueError` at `fix`, `unfix`, `set_bound` and `add_row`. `capabilities(solver)` reports `mutation`, which those adapters declare `absent`. After `close()` the next `solve` loads the model again, with the current bounds and the appended rows.

## Options and iterations

The options of the session apply to every solve. Under `barrier`, `hipo` or `pdlp` HiGHS solves the changed model from no basis. The session assembles the matrix once and loads the solver model once under every method. A model with integer columns is accepted, and a resumed solve solves the loaded model again.

`Solution.iterations` returns the simplex iterations of a solve. `benchmarks/bench_session.py` measures a sweep through a session against a rebuild of the model.
