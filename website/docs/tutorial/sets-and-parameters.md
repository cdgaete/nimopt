---
title: Sets and parameters
description: Declare the index sets of the transport model and the data indexed over them.
sidebar_position: 1
---

# Sets and parameters

The tutorial builds one model over six pages, the transport problem from
[Get started](/get-started), one concept per page. This page declares the
index sets and the data.

## The problem

Two plants, Lisbon and Porto, ship to three warehouses, Berlin, Paris and
Rome. Plant `p` has supply `s[p]`, warehouse `w` has demand `d[w]`, and one
unit shipped on route `(p, w)` costs `c[p, w]`. The decision is the quantity
`x[p, w]` shipped on each of the six routes, and the objective is total
cost.

| | Berlin | Paris | Rome | Supply |
| --- | --- | --- | --- | --- |
| Lisbon | 2 | 4 | 5 | 30 |
| Porto | 3 | 1 | 6 | 25 |
| Demand | 20 | 15 | 15 | |

## Sets

A `Set` is a named index dimension with labels. Parameters, variables and
constraints are indexed over sets, and solution values are returned over the
same sets.

```python
import numpy as np
from nimopt import Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

print(P.labels)
print(len(W))
print(P.position_of(np.array(["porto"])))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
['lisbon' 'porto']
3
[1]
```

</details>
<!-- /output -->

`P` has two members and `W` three. `position_of` maps labels to their
integer positions. Those positions are the indices used internally.

## Parameters

A `Param` is data indexed over a set product: one value per combination of
members. `Param.from_dense` takes an array whose shape equals the sizes of
the sets, in order. Cost is indexed over `(P, W)`, supply over `P`, and
demand over `W`.

```python
import numpy as np
from nimopt import Param, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))

print(cost.dims, cost.nnz)
print(cost.materialise().to_dense())
print(supply.dims, demand.dims)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W') 6
[[2. 4. 5.]
 [3. 1. 6.]]
('P',) ('W',)
```

</details>
<!-- /output -->

`cost` has six entries. `materialise()` returns the parameter as a `nimblend`
array, and `to_dense()` renders it as a NumPy array with axes in the
declared set order.

A shape mismatch raises `ValueError`, and the message gives the expected
shape and the actual shape.

```python raises=ValueError
import numpy as np
from nimopt import Param, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

Param.from_dense("cost", (P, W), np.array([[2.0, 4.0], [3.0, 1.0]]))
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: parameter 'cost' is over sets of shape (2, 3); got values of shape (2, 2)
```

</details>
<!-- /output -->

## Sparse data

In a sparse network, a plant serves a subset of the warehouses, and the cost
parameter has entries only on existing routes. `Param.from_long` takes the
entries in long form: one label column per set and one value column, read in
parallel. The k-th entry of each column belongs to the same route.

```python
import numpy as np
from nimopt import Param, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

cost = Param.from_long(
    "cost",
    (P, W),
    {
        "P": np.array(["lisbon", "lisbon", "porto"]),
        "W": np.array(["berlin", "rome", "paris"]),
    },
    np.array([2.0, 5.0, 1.0]),
)

print(cost.nnz)
print(cost.materialise().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
3
[[2. 0. 5.]
 [0. 1. 0.]]
```

</details>
<!-- /output -->

Three routes, three entries. `to_dense()` prints zeros at the three missing
routes, but the parameter stores nothing there: an unlisted route is absent,
not zero. The distinction matters on the last page of the tutorial, where a
variable declared over exactly these routes has no column for the others.

Next: [Variables](/tutorial/variables).
