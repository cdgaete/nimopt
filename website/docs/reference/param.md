---
title: Param
description: Coefficients over a set product, from a dense array or from label columns.
---

# Param

## `Param`

Coefficients over a set product. A parameter is data, not a model object:
it has no columns and produces no rows. It supplies a term's coefficient
and a constraint's right-hand side.

A parameter's array declares `absence="empty"`. A coordinate it does not
have is a coefficient that is not there, which is the additive identity a
sum needs.

### `Param.from_dense(name, sets, values)`

Every cell of `values` as a coefficient. `values.shape` must equal the
sizes of `sets`, in order; a mismatch raises `ValueError` with both shapes.

```python
import numpy as np
from nimopt import Param, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
print(cost.dims, cost.nnz)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W') 6
```

</details>
<!-- /output -->

### `Param.from_long(name, sets, columns, values)`

Coefficients from one label column per set and one value column. `columns`
is a mapping keyed by set name; each column and `values` are read in
parallel, so all have the same length.

```python
import numpy as np
from nimopt import Param, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

cost = Param.from_long(
    "cost",
    (P, W),
    {"P": np.array(["lisbon", "porto"]), "W": np.array(["berlin", "paris"])},
    np.array([2.0, 1.0]),
)
print(cost.nnz)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
2
```

</details>
<!-- /output -->

A label column of a different length raises `ValueError`; the message
gives the parameter, the column, its length and the value column's.

```python raises=ValueError
import numpy as np
from nimopt import Param, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

Param.from_long(
    "cost",
    (P, W),
    {"P": np.array(["lisbon"]), "W": np.array(["berlin", "paris"])},
    np.array([2.0, 1.0]),
)
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: parameter 'cost': label column 'P' has length 1 and the value column has length 2; they name the same entries
```

</details>
<!-- /output -->

### Members

| Member | Returns |
| --- | --- |
| `dims` | the names of the sets it is indexed over |
| `nnz` | the number of coefficients |
| `materialise()` | the coefficients as a `nimblend` array |
| `param[sets]` | a reference, with the sets given checked against `dims` |

A label in place of a set fixes that dimension at one member: the
coefficients at that member are read and the dimension leaves the
reference.

```python
import numpy as np
from nimopt import Param, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
print(cost[P, W].dims)
print(cost[P, "berlin"].dims)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W')
('P',)
```

</details>
<!-- /output -->

## `Coefficient`

What a term reads as its coefficient: a `name` to report, the `dims` it is
indexed over, the array it `materialise()`s to, and a reading at its sets.
A parameter read at its sets is one, and so is an arithmetic combination of
coefficients, so a function that reports a coefficient handles either
through one interface.

`+`, `-`, `*`, `/` and a power by a number combine coefficients. The
combination is symbolic: it holds references, derives its dimensions from
its operands, and is evaluated once, when the term it multiplies is
materialised. It can therefore be written in a definition before any data
exists.

```python
import numpy as np
from nimopt import Param, Set

G = Set("G", np.array(["base", "peak"]))
T = Set("T", np.arange(3))
price = Param.from_dense("fuel_price", (G, T), np.full((2, 3), 30.0))
eta = Param.from_dense("efficiency", (G, T), np.array([[0.5] * 3, [0.4] * 3]))

unit_cost = price[G, T] / eta[G, T]
print(unit_cost.name, unit_cost.dims)
print(unit_cost[G, T].materialise().to_dense()[:, 0])
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
(fuel_price / efficiency) ('G', 'T')
[60. 75.]
```

</details>
<!-- /output -->

A parameter has no arithmetic of its own. It is read at its sets, and the
references combine.

```python raises=TypeError
import numpy as np
from nimopt import Param, Set

G = Set("G", np.array(["a", "b"]))
price = Param.from_dense("price", (G,), np.array([1.0, 2.0]))
eta = Param.from_dense("eta", (G,), np.array([0.5, 0.4]))
price / eta
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: parameter 'price' is over ('G',) and expresses no coefficient until it is read; read it at its sets as price[G]
```

</details>
<!-- /output -->
