---
title: Constraint
description: The rows an expression produces, the right-hand side that bounds them, and the conditions that shape them.
---

# Constraint

## `Constraint`

Returned by `Model.constraint`. Rows over an expression's frame, bounded by a
right-hand side.

```
Model.constraint(name, relation, where=None, over=None)
```

| Argument | Meaning |
| --- | --- |
| `name` | the name `Solution.dual` reads it back by |
| `relation` | an expression, a sense and a right-hand side |
| `where` | a domain intersecting the rows |
| `over` | the rows, given explicitly |

| Member | Returns |
| --- | --- |
| `n_rows` | the number of rows it produces |
| `nnz` | the number of coefficients they hold |
| `row_of(name)` on the `Assembled` | where those rows sit in the matrix |

A row derived from the terms exists where every term has a value and the
right-hand side has a value. A coefficient absent inside a sum removes a
term and leaves the row standing; a term absent along a free dimension
removes the row, because a row missing one of its terms would express a
constraint that was not written.

`over=` gives the rows explicitly instead, so a term covering some of them
contributes where it has values. A condition given with `where=`
intersects the row domain, so a row outside the condition is not produced.

The expression is symbolic, so the constraint holds the recipe rather than
a block: it is materialised once to compute its shape and once to write it,
and holds nothing in between.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))

m = Model("transport")
x = m.var("x", (P, W))

rows = m.constraint("supply", Sum(W, x[P, W]) <= supply[P])
print(rows.n_rows, rows.nnz)
print(m.assemble().row_of("supply"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
2 6
slice(0, 2, None)
```

</details>
<!-- /output -->

The right-hand side is a number, applied to every row, or a parameter over
exactly the constraint's free dimensions, giving each row its own value. A
parameter over other dimensions raises `ValueError`; the message gives the
constraint's free dimensions and the parameter's.

```python raises=ValueError
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))

m = Model("transport")
x = m.var("x", (P, W))

m.constraint("supply", Sum(W, x[P, W]) <= demand[W])
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: constraint 'supply' has free dimensions ('P',); its right-hand side 'demand' is over ('W',)
```

</details>
<!-- /output -->
