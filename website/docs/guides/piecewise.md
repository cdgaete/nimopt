---
title: Piecewise-linear curves
description: Relate one expression to another through breakpoints, with an exact integer formulation or a tangent formulation, and save the declaration in a model file.
---

# Piecewise-linear curves

A fuel cost that rises in steps, an efficiency that changes with load, and a
revenue that saturates are each a curve through a list of points.
`piecewise` relates an expression `y` to an expression `x` through such
points. `x` is on the curve. `sign` compares `y` with the curve: `==` sets
`y` to the curve, `>=` bounds `y` below by it and `<=` bounds `y` above by it.

The points are two parameters, `x_points` and `y_points`. Each is over some
or all of the sets of `x` and over one breakpoint set. The declaration
generates the variables and the constraints of its method. Each generated
name begins with the declaration's name.

## A cost curve that is not convex

`method="incremental"` is exact for breakpoints that are strictly increasing
or strictly decreasing. It adds one continuous variable and one integer
variable per segment.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

G = Set("G", np.array(["a", "b"]))
B = Set("B", np.array(["b0", "b1", "b2", "b3"]))
power = Param.from_dense("power", (G, B), [[0, 10, 20, 30], [0, 10, 20, 30]])
cost = Param.from_dense("cost", (G, B), [[0, 5, 30, 35], [0, 20, 30, 40]])

m = Model("curve")
p = m.var("p", (G,))
c = m.var("c", (G,))
m.constraint("demand", Sum(G, p[G]) == 25.0)
m.piecewise(
    "fuel",
    x=p[G],
    x_points=power[G, B],
    y=c[G],
    y_points=cost[G, B],
    sign=">=",
    method="incremental",
)
m.set_objective(Sum(G, c[G]))

s = m.solve()
print(s.objective)
print(s.primal("p").to_dense())
print(list(m.variables))
print(list(m.constraints))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
30.0
[10. 15.]
['p', 'c', 'fuel_fill', 'fuel_order']
['demand', 'fuel_x', 'fuel_y', 'fuel_order_bound', 'fuel_fill_order', 'fuel_order_link']
```

</details>
<!-- /output -->

The generated variables and constraints are declarations of the model.
`primal`, `dual`, `row` and `explain` read them by their names.

`x` and `y` are expressions. `p[G] + 5.0` is on the curve five units above
`p`, and the incremental method reads that constant. The tangent method
multiplies `x` by the slope of each segment, and a constant in `x` raises
`ValueError`. Subtract it from `x_points` instead.

## A convex curve

`method="tangent"` adds no variable. It adds one row per segment, and two
rows that keep `x` between the first and the last breakpoint. The rows
describe the curve exactly when the points are convex under `>=`, or
concave under `<=`.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

G = Set("G", np.array(["a"]))
B = Set("B", np.array(["b0", "b1", "b2"]))
power = Param.from_dense("power", (G, B), [[0, 10, 20]])
cost = Param.from_dense("cost", (G, B), [[0, 10, 30]])

m = Model("convex")
p = m.var("p", (G,))
c = m.var("c", (G,))
m.constraint("demand", Sum(G, p[G]) == 15.0)
m.piecewise("fuel", p[G], power[G, B], c[G], cost[G, B], ">=", "tangent")
m.set_objective(Sum(G, c[G]))

s = m.solve()
print(s.objective)
print(m.constraints["fuel_tangent"])
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
20.0
Constraint('fuel_tangent', ('G', 'fuel_segment'), 2 rows, 4 coefficients)
```

</details>
<!-- /output -->

Points that are not convex under `>=` raise `ValueError` and identify the
first entity at fault. The tangent rows of such points describe a different
curve.

```python raises=ValueError
import numpy as np
from nimopt import Model, Param, Set

G = Set("G", np.array(["a"]))
B = Set("B", np.array(["b0", "b1", "b2"]))
power = Param.from_dense("power", (G, B), [[0, 10, 20]])
cost = Param.from_dense("cost", (G, B), [[0, 20, 30]])

m = Model("concave")
p = m.var("p", (G,))
c = m.var("c", (G,))
m.piecewise("fuel", p[G], power[G, B], c[G], cost[G, B], ">=", "tangent")
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: piecewise 'fuel' has points that are not convex, required by sign '>=' at {'G': 'a'}; use method 'incremental'
```

</details>
<!-- /output -->

## One curve for every entity

`x_points` and `y_points` are over the breakpoint set and over as many of the
sets of `x` as the curves differ along. Points over the breakpoint set alone
give every entity the same curve.

## Entities with fewer breakpoints

A parameter built from a table has entries only where the table lists
them. An entity with fewer breakpoints lists its first ones, and the rest
are absent. An entity with no breakpoint has no generated rows and no
generated columns. Its `x` and `y` are not related by the declaration.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

G = Set("G", np.array(["a", "b", "c"]))
B = Set("B", np.array(["b0", "b1", "b2"]))
labels = {
    "G": np.array(["a", "a", "a", "b", "b"]),
    "B": np.array(["b0", "b1", "b2", "b0", "b1"]),
}
power = Param.from_long("power", (G, B), labels, [0.0, 10.0, 20.0, 0.0, 15.0])
cost = Param.from_long("cost", (G, B), labels, [0.0, 10.0, 30.0, 0.0, 30.0])

m = Model("short")
p = m.var("p", (G,))
c = m.var("c", (G,))
m.piecewise("fuel", p[G], power[G, B], c[G], cost[G, B], ">=", "incremental")

print(m.variables["fuel_fill"].n_columns)
print(m.constraints["fuel_x"].n_rows)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
3
2
```

</details>
<!-- /output -->

A breakpoint absent before a present one raises `ValueError`. A table
with an entity of one breakpoint raises `ValueError`.

## A curve that a binary variable switches off

`active=` takes an expression over the sets of `x`, usually a binary
variable. Where it is 1, `x` is on the curve. Where it is 0, `x` is 0 and `y`
is compared with 0. The curve then starts at its first breakpoint, and a
first breakpoint above 0 is a minimum output. `active=` is supported by
`method="incremental"` only.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

G = Set("G", np.array(["a"]))
T = Set("T", np.array(["t0", "t1"]))
B = Set("B", np.array(["b0", "b1", "b2"]))
power = Param.from_dense("power", (G, B), [[10, 20, 30]])
cost = Param.from_dense("cost", (G, B), [[10, 15, 30]])
demand = Param.from_dense("demand", (T,), [5.0, 25.0])

m = Model("committed")
p = m.var("p", (G, T))
c = m.var("c", (G, T))
on = m.var("on", (G, T), upper=1.0, integer=True)
spot = m.var("spot", (T,))
m.constraint("balance", Sum(G, p[G, T]) + spot[T] == demand[T])
m.piecewise(
    "fuel",
    x=p[G, T],
    x_points=power[G, B],
    y=c[G, T],
    y_points=cost[G, B],
    sign=">=",
    method="incremental",
    active=on[G, T],
)
m.set_objective(Sum(G, T, c[G, T]) + 1.2 * Sum(T, spot[T]) + 2.0 * Sum(G, T, on[G, T]))

s = m.solve()
print(s.objective)
print(s.primal("on").to_dense())
print(s.primal("p").to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
29.0
[[0. 1.]]
[[ 0. 20.]]
```

</details>
<!-- /output -->

In `t0` the demand of 5 is below the minimum output of 10, and the unit is
off.

## Saving a piecewise declaration

A model file of version 4 contains the declaration under `piecewise`. The
generated variables, constraints and parameters are not written. Loading
the file generates them again.

```python
from nimopt import Definition, Sum

d = Definition("curve", sense="min")
G, B = d.set("G"), d.set("B")
power, cost = d.param("power", (G, B)), d.param("cost", (G, B))
p = d.var("p", (G,))
c = d.var("c", (G,))
d.constraint("demand", Sum(G, p[G]) == 25.0)
d.piecewise("fuel", p[G], power[G, B], c[G], cost[G, B], ">=", "incremental")
d.set_objective(Sum(G, c[G]))

print(d.to_yaml())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
version: 4
name: curve
sense: min
sets: [G, B]
parameters:
  power: [G, B]
  cost: [G, B]
variables:
  p:
    sets: [G]
  c:
    sets: [G]
constraints:
  demand:
    relation: Sum(G, p[G]) == 25
piecewise:
  fuel:
    x: p[G]
    x_points: power[G, B]
    y: c[G]
    y_points: cost[G, B]
    sign: '>='
    method: incremental
objective: Sum(G, c[G])
```

</details>
<!-- /output -->

`version=3` writes a model in the format a reader of version 3 accepts. The
generated declarations are written in place of the declaration, and the
breakpoints are not written. A model loaded from that file contains the
same rows and no piecewise declaration. A definition has no data to
generate the rows from, and writing it as version 3 raises `ValueError`.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

G = Set("G", np.array(["a"]))
B = Set("B", np.array(["b0", "b1"]))
power = Param.from_dense("power", (G, B), [[0, 10]])
cost = Param.from_dense("cost", (G, B), [[0, 20]])

m = Model("line")
p = m.var("p", (G,))
c = m.var("c", (G,))
m.piecewise("fuel", p[G], power[G, B], c[G], cost[G, B], ">=", "tangent")

print(m.to_yaml(version=3))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
version: 3
name: line
sense: min
sets: [G, fuel_segment]
parameters:
  fuel_slope: [G, fuel_segment]
  fuel_intercept: [G, fuel_segment]
  fuel_x_low: [G]
  fuel_x_high: [G]
variables:
  p:
    sets: [G]
  c:
    sets: [G]
constraints:
  fuel_tangent:
    relation: c[G] - fuel_slope[G, fuel_segment] * p[G] >= fuel_intercept[G, fuel_segment]
  fuel_x_min:
    relation: p[G] >= fuel_x_low[G]
  fuel_x_max:
    relation: p[G] <= fuel_x_high[G]
```

</details>
<!-- /output -->
