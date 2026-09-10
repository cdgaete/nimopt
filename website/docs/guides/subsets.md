---
title: A variable over a subset
description: Declare a variable over the members of a set product that exist in the model, so that the others have no column.
---

# A variable over a subset

In a network model, most pairs of a set product are not connected: a plant
serves some warehouses, a line joins two of many buses. A variable declared
over the full product has a column for every pair, including the ones that
do not exist. `subset=` restricts the variable to the members that do, and
the others have no column at all.

## Declaring the subset

`subset(sets, columns)` lists members of a set product by label, one column
per set, read in parallel. `m.var(..., subset=arcs)` declares the variable
over those members.

```python
import numpy as np
from nimopt import Model, Set, Sum, product, subset

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

arcs = subset(
    (P, W),
    {"P": np.array(["p0", "p0", "p1"]), "W": np.array(["w0", "w1", "w2"])},
)

m = Model("network")
x = m.var("x", (P, W), subset=arcs)

print(product((P, W)).size)
print(x.n_columns)
print(m.assemble().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
6
3
[]
```

</details>
<!-- /output -->

The product has six members and the subset three, so `x` has three columns.
The assembled matrix is empty because no constraint has been added. A model
over a sparse network pays for its arcs, not for the grid that contains
them.

## By label or by position

`subset` takes labels and `subset_of` takes integer positions. Both read
their columns in parallel: the k-th entry of each column belongs to the same
member. A subset is a list of members, not a cross product of its columns.

```python
import numpy as np
from nimopt import Set, subset, subset_of

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

by_label = subset(
    (P, W),
    {"P": np.array(["p0", "p1"]), "W": np.array(["w0", "w2"])},
)
by_index = subset_of((P, W), np.array([[0, 1], [0, 2]]))

print(by_label.size, by_index.size)
print(by_label.labels())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
2 2
{'P': array(['p0', 'p1'], dtype='<U2'), 'W': array(['w0', 'w2'], dtype='<U2')}
```

</details>
<!-- /output -->

Both list the same two members, `(p0, w0)` and `(p1, w2)`. `subset_of`
avoids resolving labels when positions are already at hand.

## Constraints over a subset variable

A sum over a subset variable runs over the members the variable has, so a
row contains the arcs at that member and nothing else.

```python
import numpy as np
from nimopt import Model, Set, Sum, subset

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

arcs = subset(
    (P, W),
    {"P": np.array(["p0", "p0", "p1"]), "W": np.array(["w0", "w1", "w2"])},
)

m = Model("network")
x = m.var("x", (P, W), subset=arcs)
rows = m.constraint("capacity", Sum(W, x[P, W]) <= 10.0)

print(rows.n_rows, rows.nnz)
print(m.assemble().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
2 3
[[1. 1. 0.]
 [0. 0. 1.]]
```

</details>
<!-- /output -->

Two rows over three columns: `p0` has two arcs and `p1` one.

## Bounds over a subset

A bound applies to every column of the variable. A parameter indexed over
fewer dimensions than the variable is broadcast over the rest, so a bound
per plant applies to each of that plant's arcs.

```python
import numpy as np
from nimopt import Model, Param, Set, subset

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

arcs = subset(
    (P, W),
    {"P": np.array(["p0", "p0", "p1"]), "W": np.array(["w0", "w1", "w2"])},
)
cap = Param.from_dense("cap", (P,), np.array([4.0, 9.0]))

m = Model("network")
m.var("x", (P, W), subset=arcs, upper=cap)

print(m.column_bounds()[1])
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[4. 4. 9.]
```

</details>
<!-- /output -->

The two `p0` arcs take 4.0 and the `p1` arc 9.0.
