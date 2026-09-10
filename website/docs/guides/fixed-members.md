---
title: A member fixed at a label
description: Reference one member of a dimension by its label, removing that dimension from the frame.
---

# A member fixed at a label

Initial conditions, terminal conditions and boundary rows reference one
member of a set: the state at the first period, the level at the last. A
label in place of a set in a reference fixes that dimension at one member
and removes it from the frame.

```python
import numpy as np
from nimopt import Model, Set

G = Set("G", np.array(["g0", "g1"]))
T = Set("T", np.array(["t0", "t1", "t2"]))

m = Model("schedule")
x = m.var("x", (G, T))

print(x[G, T].frame)
print(x[G, "t0"].frame)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('G', 'T')
('G',)
```

</details>
<!-- /output -->

`T` is fixed at `t0`, so the reference is indexed over `G` alone. An
initial condition is one row per unit, referencing that unit's column at the
first period.

```python
import numpy as np
from nimopt import Model, Set

G = Set("G", np.array(["g0", "g1"]))
T = Set("T", np.array(["t0", "t1", "t2"]))

m = Model("schedule")
x = m.var("x", (G, T))
rows = m.constraint("start", x[G, "t0"] <= 1.0)

print(rows.n_rows, rows.nnz)
print(m.assemble().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
2 2
[[1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]]
```

</details>
<!-- /output -->

Two rows over six columns, each with one nonzero: the `t0` column of its own
unit.

## A coefficient at a member

A parameter takes a label the same way and yields the coefficients at that
member.

```python
import numpy as np
from nimopt import Model, Param, Set

G = Set("G", np.array(["g0", "g1"]))
T = Set("T", np.array(["t0", "t1", "t2"]))
rate = Param.from_dense("rate", (G, T), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))

m = Model("schedule")
x = m.var("x", (G, T))
rows = m.constraint("start", rate[G, "t0"] * x[G, "t0"] <= 1.0)

print(rate[G, "t0"].dims)
print(m.assemble().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('G',)
[[2. 0. 0. 0. 0. 0.]
 [0. 0. 0. 3. 0. 0.]]
```

</details>
<!-- /output -->

The coefficients are the `t0` column of `rate`: 2.0 and 3.0.

## The label must be a member

A label that is not a member of the set raises `ValueError`; the message
gives the label and the dimension.

```python raises=ValueError
import numpy as np
from nimopt import Model, Set

G = Set("G", np.array(["g0", "g1"]))
T = Set("T", np.array(["t0", "t1", "t2"]))

m = Model("schedule")
x = m.var("x", (G, T))

x[G, "t9"]
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: variable 'x' is read at member 't9' of dimension 'T', which that set does not carry
```

</details>
<!-- /output -->
