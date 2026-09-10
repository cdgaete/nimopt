---
title: A variable is a dimension
description: Why the column space is a dimension, and why nothing stores a column index.
---

# A variable is a dimension

A model has one column space. `m.var` does not create an object with its
own numbering; it takes the next range of that space, and each subsequent
variable continues from where the previous one ended.

```python
import numpy as np
from nimopt import Model, Set

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

m = Model("transport")
x = m.var("x", (P, W))
y = m.var("y", (P,))

print(x.n_columns, y.n_columns)
print(m.n_columns)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
6 2
8
```

</details>
<!-- /output -->

The shared column space is what makes the column a dimension. A variable's
coefficients form an array over `(*dims, COLUMN)`, where `COLUMN` is the
model's column space. A variable does not own columns; it occupies a block
of one dimension that all variables share.

## A column is computed, not stored

A member's column is a virtual coordinate, obtained by arithmetic rather
than by lookup.

For a full product, the column is the member's multi-index ravelled against
the set sizes, offset by the start of the variable's block. `ProductCoord`
performs that computation.

```python
import numpy as np
import nimblend as nb

columns = nb.ProductCoord((2, 3))
print(columns.to_position(np.array([[0, 1], [2, 0]])))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[2 3]
```

</details>
<!-- /output -->

Member `(0, 2)` is column 2 and `(1, 0)` is column 3: stride arithmetic and
nothing else.

For a variable over a subset, the column is the member's rank among the
subset's codes. `SubsetCoord` stores the codes in order, and a block already
in canonical order needs no lookup.

```python
import numpy as np
import nimblend as nb

columns = nb.SubsetCoord(np.array([0, 4]), (2, 3))
print(columns.to_position(np.array([[0, 1], [0, 1]])))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[0 1]
```

</details>
<!-- /output -->

Codes `0` and `4` are members `(0, 0)` and `(1, 1)`, with ranks `0` and `1`.

## Consequences

Nothing stores a column index. A variable over a million members stores its
set sizes, the start of its block and, for a subset, the codes of its
members. It stores no integer per member: the column is computed from the
member itself.

The cost of declaring a variable is therefore the cost of its members, not
of its columns. A variable over a full product costs nothing per column:
two set sizes and a start.

A subset variable is not a special case. Both kinds report the position a
member occupies. They differ in whether that position is computed by
arithmetic or by a rank.
