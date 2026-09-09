---
title: Variables
description: Declare the shipment variable over plants and warehouses, and set its bounds and integrality.
sidebar_position: 2
---

# Variables

The decision is the quantity shipped on each route: one variable indexed
over plants and warehouses.

## Declaring a variable

A `Model` holds variables, constraints and the objective. `m.var(name,
sets)` declares a variable indexed over a tuple of sets and returns a handle
for use in expressions.

```python
import numpy as np
from nimopt import Model, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))

print(x.dims)
print(x.n_columns)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W')
6
```

</details>
<!-- /output -->

Two plants by three warehouses gives six members, so `x` occupies six
columns of the coefficient matrix. A column index is computed from a
member's positions in each set; nothing stores a column per member, so a
variable over a million members costs the same to declare as one over six.

## Bounds and integrality

A variable has a lower bound of 0 and no upper bound unless declared
otherwise. `lower=` and `upper=` take a number that applies to every column.
`integer=True` restricts the columns to integer values, which makes the
model a MILP.

```python
import numpy as np
from nimopt import Model, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W), upper=20.0)
trucks = m.var("trucks", (P,), integer=True)

lower, upper = m.column_bounds()
print(lower)
print(upper)
print(m.integrality())
print(m.n_columns)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[0. 0. 0. 0. 0. 0. 0. 0.]
[20. 20. 20. 20. 20. 20. inf inf]
[0 0 0 0 0 0 1 1]
8
```

</details>
<!-- /output -->

A model has one column space shared by all its variables: `x` occupies
columns 0 to 5 and `trucks` columns 6 and 7. `column_bounds()` returns the
lower and upper bound vectors in column order, and `integrality()` returns
one flag per column.

A parameter in place of a number gives each column its own bound; see
[Bounds from a parameter](/guides/bounds-from-parameters).

Next: [Expressions](/tutorial/expressions).
