---
title: Expressions are symbolic
description: What an expression stores before it is materialised, and what materialising it does.
---

# Expressions are symbolic

A `Term` describes a block of numbers and stores none. It contains a
reference to a variable, an optional coefficient, the dimensions summed over,
a scale factor, and any lags, conditions or fixed members. An `Expression` is a list
of such terms and the frame they share.

Nothing in that list is an array. `cost[P, W] * x[P, W]` records which
parameter and which variable, and reads neither.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array([f"p{i}" for i in range(200)]))
W = Set("W", np.array([f"w{i}" for i in range(100)]))
cost = Param.from_dense("cost", (P, W), np.ones((200, 100)))

m = Model("transport")
x = m.var("x", (P, W))

expression = Sum(W, cost[P, W] * x[P, W])
print(expression.frame)
print(len(expression.terms))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P',)
1
```

</details>
<!-- /output -->

One term over twenty thousand columns. The same line over twenty million
columns is still one term and costs the same to write.

## Materialisation

An expression becomes matrix entries when it is materialised. That is the
only point at which values are read. Materialisation walks the terms and
issues `nimblend` operations in order. The coefficient is aligned with the
block of the variable, and the summed dimensions are reduced. The scale is
applied, and the terms are combined over the shared frame.

The result is a `nimblend` array over the frame crossed with the column space:
the block of coefficients the constraint contributes to the matrix.

## A hundred constraints cost a hundred shapes

A constraint stores the term list and no block. Adding a constraint
therefore costs the computation of its shape. `n_rows` and `nnz` are known at
declaration, and no coefficient exists yet.

```python
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array([f"p{i}" for i in range(200)]))
W = Set("W", np.array([f"w{i}" for i in range(100)]))

m = Model("transport")
x = m.var("x", (P, W))
for i in range(20):
    m.constraint(f"cap{i}", Sum(W, x[P, W]) <= 1.0)

print(m.n_rows, m.nnz)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
4000 400000
```

</details>
<!-- /output -->

Four thousand rows and four hundred thousand coefficients are declared, and
the model stores twenty term lists.

## The cost

An expression is materialised twice: once to compute its shape when the
constraint is added, and once to write its entries when the matrix is
assembled. Building twice costs build time. One expression is live at a
time, and peak memory is set by the largest constraint, not by the sum of
all of them.

A model that is cheap to declare and more expensive to assemble suits a
builder, because declaration is what a caller iterates on.
