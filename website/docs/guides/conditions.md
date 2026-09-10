---
title: Conditions on a sum and on a constraint
description: Restrict the members a sum runs over, restrict the rows a constraint produces, or state the rows explicitly.
---

# Conditions on a sum and on a constraint

Two situations call for a condition. A constraint may sum over part of a
variable's members, such as the arcs of a network when the variable is
indexed over the full product. And a constraint may apply to some members
of its frame only, such as a capacity limit on one plant. `where=` handles
both. `over=` states a constraint's rows explicitly.

## Restricting a sum

`Sum(..., where=domain)` restricts each term to the members of `domain`
before summing.

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
x = m.var("x", (P, W))
rows = m.constraint("capacity", Sum(W, x[P, W], where=arcs) <= 10.0)

print(x.n_columns)
print(rows.n_rows, rows.nnz)
print(m.assemble().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
6
2 3
[[1. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1.]]
```

</details>
<!-- /output -->

The variable is over the full product and has six columns; the condition
leaves three of them with a coefficient. A subset variable would have three
columns in the first place. Use a condition when the variable is over the
product and one constraint reads part of it; declare a subset when the
model never uses the other members.

## Restricting the rows

`m.constraint(..., where=domain)` takes a domain over the constraint's frame and
keeps the rows in it. A row outside the condition is not produced.

```python
import numpy as np
from nimopt import Model, Set, Sum, subset

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

m = Model("network")
x = m.var("x", (P, W))
only_p0 = subset((P,), {"P": np.array(["p0"])})
rows = m.constraint("capacity", Sum(W, x[P, W]) <= 10.0, where=only_p0)

print(rows.n_rows)
print(m.assemble().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
1
[[1. 1. 1. 0. 0. 0.]]
```

</details>
<!-- /output -->

One row, for `p0`. `p1` has no capacity row.

A condition over dimensions other than the constraint's frame raises
`ValueError`; the message gives both index sets.

```python raises=ValueError
import numpy as np
from nimopt import Model, Set, Sum, subset

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

m = Model("network")
x = m.var("x", (P, W))
by_warehouse = subset((W,), {"W": np.array(["w0"])})

m.constraint("capacity", Sum(W, x[P, W]) <= 10.0, where=by_warehouse)
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: constraint 'capacity' has free dimensions ('P',); its condition is over ('W',)
```

</details>
<!-- /output -->

## Stating the rows explicitly

By default the rows of a constraint are derived from its terms: a row
exists where every term has a value and the right-hand side has a value. A
term with no value along a frame dimension removes the row, because a row
missing one of its terms would express a constraint that was not written.

`over=domain` states the rows instead of deriving them. A term with values
at some of the rows contributes where it has them, and every row in the
domain is produced.

```python
import numpy as np
from nimopt import Model, Set, Sum, product

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

m = Model("network")
x = m.var("x", (P, W))
rows = m.constraint("capacity", Sum(W, x[P, W]) <= 10.0, over=product((P,)))

print(rows.n_rows)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
2
```

</details>
<!-- /output -->

## `over` or `where`, not both

`over=` states the rows and `where=` restricts them, so passing both raises
`ValueError`.

```python raises=ValueError
import numpy as np
from nimopt import Model, Set, Sum, product, subset

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

m = Model("network")
x = m.var("x", (P, W))

m.constraint(
    "capacity",
    Sum(W, x[P, W]) <= 10.0,
    where=subset((P,), {"P": np.array(["p0"])}),
    over=product((P,)),
)
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: constraint 'capacity' states its rows with over= and narrows them with where=; state one
```

</details>
<!-- /output -->

A condition on a sum and a condition on the constraint compose: the first
restricts what is summed, the second which rows exist.
