---
title: Expressions
description: Write the sums the constraints and the objective are stated over, without computing anything.
sidebar_position: 3
---

# Expressions

Constraints and the objective are stated over sums of variables: the total
shipped from a plant, the total received by a warehouse, the total cost. An
expression is such a sum. It is symbolic: writing one records the variables,
coefficients and sets involved, and computes nothing.

## Referencing a variable

`x[P, W]` references the variable over its sets and returns an expression
with one term. The **frame** of an expression is the tuple of dimensions it
is still indexed over. `x[P, W]` has frame `(P, W)`: one value per route.

```python
import numpy as np
from nimopt import Model, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))

shipped = x[P, W]
print(type(shipped).__name__)
print(shipped.frame)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
Expression
('P', 'W')
```

</details>
<!-- /output -->

## Sum

`Sum(S, expression)` sums over the members of `S` and removes `S` from the
frame. Summing over `W` gives the total shipped from each plant, indexed
over `P`. Summing over both sets gives a scalar.

```python
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))

print(x[P, W].frame)
print(Sum(W, x[P, W]).frame)
print(Sum(P, W, x[P, W]).frame)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W')
('P',)
()
```

</details>
<!-- /output -->

The frame determines the shape of a constraint built on the expression: an
expression with frame `(P,)` produces one row per plant. An expression with
an empty frame is a scalar, which is the form an objective takes.

## Coefficients

The cost of a plan is `Σ_{p,w} c[p, w] · x[p, w]`. Multiplying a reference
by a parameter over the same sets gives the term a coefficient. The frame is
unchanged until the sum is taken.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))

m = Model("transport")
x = m.var("x", (P, W))

per_route = cost[P, W] * x[P, W]
total_cost = Sum(P, W, per_route)
print(per_route.frame)
print(total_cost.frame)
print(len(total_cost.terms))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W')
()
1
```

</details>
<!-- /output -->

`total_cost` is a single term. The same expression over a million routes is
still one term, because it holds references to `cost` and `x` rather than
their values. Values are read when the matrix is assembled.

## Addition and subtraction

Expressions add and subtract, producing one expression over the frame both
share. A balance, inflow minus outflow, is written this way. With a second
variable for returned goods, the net shipment on a route is the outbound
quantity minus the returned quantity.

```python
import numpy as np
from nimopt import Model, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))
returned = m.var("returned", (P, W))

net = x[P, W] - returned[P, W]
print(net.frame)
print(len(net.terms))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W')
2
```

</details>
<!-- /output -->

Two terms, one per variable, over the same frame.

Next: [Constraints](/tutorial/constraints).
