---
title: Bounds from a parameter
description: Give each column its own bound from a parameter, broadcast over the dimensions the parameter lacks.
---

# Bounds from a parameter

A capacity per generator, an energy limit per battery, a flow limit per
line: bounds usually come from data and differ by member. `lower=` and
`upper=` accept a number, applied to every column, or a `Param`, giving
each member its own value.

```python
import numpy as np
from nimopt import Model, Param, Set

G = Set("G", np.array(["g0", "g1"]))
cap = Param.from_dense("cap", (G,), np.array([5.0, 7.0]))

m = Model("schedule")
m.var("x", (G,), upper=cap)

lower, upper = m.column_bounds()
print(lower)
print(upper)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[0. 0.]
[5. 7.]
```

</details>
<!-- /output -->

## Broadcasting a narrower parameter

A parameter indexed over fewer dimensions than the variable is broadcast
over the rest. A capacity per unit bounds every period of that unit.

```python
import numpy as np
from nimopt import Model, Param, Set

G = Set("G", np.array(["g0", "g1"]))
T = Set("T", np.array(["t0", "t1", "t2"]))
cap = Param.from_dense("cap", (G,), np.array([5.0, 7.0]))

m = Model("schedule")
m.var("x", (G, T), upper=cap)

print(m.column_bounds()[1])
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[5. 5. 5. 7. 7. 7.]
```

</details>
<!-- /output -->

Six columns; each unit's three periods take that unit's capacity.

## A bound covers every column

A bound with no value for some member of the variable raises `ValueError`;
the message gives the member. A dense parameter covers its product by
construction; a long-form parameter can omit a member.

```python raises=ValueError
import numpy as np
from nimopt import Model, Param, Set

S = Set("S", np.array(["a", "b", "c"]))
cap = Param.from_long("cap", (S,), {"S": np.array(["a", "c"])}, np.array([1.0, 3.0]))

m = Model("bounds")
m.var("x", (S,), upper=cap)

m.column_bounds()
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: the upper bound 'cap' has no value at member ('b',) of variable 'x'; a bound covers every column of the variable it bounds
```

</details>
<!-- /output -->

The check runs when the bound vectors are built. `column_bounds()`,
`assemble()` and `solve()` all build them.

## A bound over a dimension the variable lacks

A parameter indexed over a dimension the variable is not declared on raises
`ValueError`. A bound is per column, and a dimension the variable does not
have selects no column.

```python raises=ValueError
import numpy as np
from nimopt import Model, Param, Set

G = Set("G", np.array(["g0", "g1"]))
W = Set("W", np.array(["w0", "w1"]))
cap = Param.from_dense("cap", (W,), np.array([5.0, 7.0]))

m = Model("schedule")
m.var("x", (G,), upper=cap)
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: variable 'x' is declared over ('G',) and is not over ['W']; its upper bound 'cap' is declared over ('W',)
```

</details>
<!-- /output -->
