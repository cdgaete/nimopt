---
title: Variable
description: A variable over a set product, the columns it occupies, and the reserved dimension names.
---

# Variable

## `Variable`

Returned by `Model.var`. A variable over a set product, or over a subset
of one.

```
Model.var(name, sets, subset=None, lower=0.0, upper=inf, integer=False)
```

| Argument | Meaning |
| --- | --- |
| `name` | the name `Solution.primal` reads it back by |
| `sets` | the dimensions it is declared over |
| `subset` | the members it has; the full product when omitted |
| `lower`, `upper` | the bound every one of its columns takes |
| `integer` | whether its columns are integral |

The variable's columns are a virtual coordinate: a member's column is
computed from its multi-index by stride arithmetic for a full product, or
is its rank among a subset's codes. Nothing stores a column index, which is
why a variable over millions of columns costs only its members.

| Member | Returns |
| --- | --- |
| `dims` | the names of the sets it is over |
| `n_columns` | the number of columns it occupies |
| `domain()` | the members it has |
| `terms()` | its coefficients over `(*dims, COLUMN)` |
| `variable[sets]` | a one-term expression referencing it |

```python
import numpy as np
from nimopt import Model, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))
open_plant = m.var("open_plant", (P,), lower=0.0, upper=1.0, integer=True)

print(x.dims, x.n_columns)
print(open_plant.n_columns)
print(m.n_columns)
print(m.integrality())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W') 6
2
8
[0 0 0 0 0 0 1 1]
```

</details>
<!-- /output -->

Each variable occupies the next range of the model's one column space, so
`m.n_columns` counts every column declared so far.

## A variable over no dimension

A variable's bracket lists the dimensions it carries, so a variable over
none carries no bracket and enters a row on its own. It is one column: a
value-at-risk level, a budget slack, a bound every row of a family shares.
`theta[()]` is the same term written out.

A variable that does carry dimensions states no term until it is read, and
using one bare raises `TypeError` naming the reading it wants. The same rule
holds for a parameter, which is read `cost[G, T]` and, over no dimension,
`k`.

Comparing a variable states a row, so `==` between two variables states one
too rather than answering true or false. A list of variables therefore cannot
be searched with `in` or `.index`, which compare their items: those raise the
reading refusal, naming whichever variable they reached first. Keep variables
in a dict or a set, which match on identity, or search them by `name`.

```python
import numpy as np
from nimopt import Model, Set, Sum

S = Set("S", np.array(["s1", "s2"]))

m = Model("cvar", sense="min")
theta = m.var("theta", (), lower=-np.inf)
p = m.var("p", (S,))

m.constraint("tail", theta - Sum(S, p[S]) >= 0.0)
m.set_objective(theta)
print(m.n_columns, m.n_rows)
print(m.constraints["tail"].relation)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
3 1
theta - Sum(S, p[S]) >= 0
```

</details>
<!-- /output -->

## `COLUMN` and `ROW`

The dimension names `nimopt` reserves. `COLUMN` is `"__column__"` and `ROW`
is `"__row__"`; both are spelled so that no ordinary set name collides
with them.

A variable's terms are an array over `(*dims, COLUMN)`, and a constraint's
block is one over `(ROW, COLUMN)`. That is the whole of the correspondence
between a model and its matrix: the column space is a dimension, so the
array is the matrix.

```python
import numpy as np
from nimopt import COLUMN, ROW, Model, Set

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))

print(COLUMN, ROW)
print(x.terms().dims)
print(x.domain().dims)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
__column__ __row__
('P', 'W', '__column__')
('P', 'W')
```

</details>
<!-- /output -->

A caller writes neither name. They exist to be recognised when a `nimblend`
array from inside a model is inspected.
