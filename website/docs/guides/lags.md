---
title: Lags
description: Reference a variable at the previous or next member of a set, and choose whether the boundary row is dropped or wraps around.
---

# Lags

Time-coupled constraints reference the previous period. A storage balance
relates the state of charge at `t` to that at `t-1`; a ramp limit bounds
the change in output between consecutive periods. `T - 1` is the set `T`
lagged by one member, and `x[T - 1]` references the variable at the
previous member.

## A lag that drops the boundary row

```python
import numpy as np
from nimopt import Model, Set

T = Set("T", np.array(["t0", "t1", "t2"]))

m = Model("schedule")
x = m.var("x", (T,))
rows = m.constraint("carry", x[T] - x[T - 1] <= 0.0)

print(rows.n_rows, rows.nnz)
print(m.assemble().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
2 4
[[-1.  1.  0.]
 [ 0. -1.  1.]]
```

</details>
<!-- /output -->

Each row references its own column and the previous one. The first member
has no predecessor, so its row is not produced: three members give two rows.

`T + 1` references the following member by the same rule.

## A lag that wraps

`T.cyclic` lags with wrap-around: the member before the first is the last.
No row is dropped.

```python
import numpy as np
from nimopt import Model, Set

T = Set("T", np.array(["t0", "t1", "t2"]))

m = Model("schedule")
x = m.var("x", (T,))
rows = m.constraint("carry", x[T] - x[T.cyclic - 1] <= 0.0)

print(rows.n_rows, rows.nnz)
print(m.assemble().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
3 6
[[ 1.  0. -1.]
 [-1.  1.  0.]
 [ 0. -1.  1.]]
```

</details>
<!-- /output -->

Three rows, and the first references the last column: the `-1` in row 0 is
in the final position. A storage balance over a repeating horizon is written
this way, so that the level at the end of the horizon carries into the
beginning.

## A lag applies to a reference, not to a sum

`Sum` runs over the members of a set and takes the set itself. Passing a
lagged set raises `ValueError`.

```python raises=ValueError
import numpy as np
from nimopt import Model, Set, Sum

T = Set("T", np.array(["t0", "t1", "t2"]))

m = Model("schedule")
x = m.var("x", (T,))

Sum(T - 1, x[T])
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: a sum is over the members of ['T'] and takes the set, not a lag of it; write the lag at the variable's reference
```

</details>
<!-- /output -->

The lag belongs on the variable reference: `Sum(T, x[T - 1])`.

## A lag applies to a variable, not to a parameter

Reading a parameter at a lag raises `ValueError`. A coefficient is indexed
by the row it appears in, and a lag selects which column a row references.
`rate[T] * x[T - 1]` applies the rate at `t` to the variable at `t-1`.

```python raises=ValueError
import numpy as np
from nimopt import Param, Set

T = Set("T", np.array(["t0", "t1", "t2"]))
rate = Param.from_dense("rate", (T,), np.array([1.0, 2.0, 3.0]))

rate[T - 1]
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: parameter 'rate' is read at a lag ['T']; write the lag at the variable's reference
```

</details>
<!-- /output -->

A lag is an integer number of members. A fractional lag raises `ValueError`
rather than being truncated to a different lag.

```python raises=ValueError
import numpy as np
from nimopt import Set

T = Set("T", np.arange(3))
T - 1.7
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: a lag is a whole number of members; got 1.7
```

</details>
<!-- /output -->
