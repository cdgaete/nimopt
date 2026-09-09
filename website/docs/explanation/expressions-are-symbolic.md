---
title: Expressions are symbolic
description: Why writing an expression costs nothing, and what materialising it does.
---

# Expressions are symbolic

A `Term` is a recipe, not a block of numbers. It holds a reference to a
variable, an optional coefficient, the dimensions summed over, a scale
factor, and any lags, conditions or fixed members. An `Expression` is a list
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

An expression becomes matrix entries when it is materialised, which is the
only point at which values are read. Materialisation walks the terms and
issues `nimblend` operations in order: the coefficient is aligned with the
variable's block, the summed dimensions are reduced, the scale is applied,
and the terms are combined over the shared frame.

The result is a `nimblend` array over the frame crossed with the column space:
the block of coefficients the constraint contributes to the matrix.

## A hundred constraints cost a hundred shapes

A constraint holds the recipe rather than the block, so adding one costs
computing its shape. `n_rows` and `nnz` are known at declaration, and no
coefficients exist yet.

```python
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array([f"p{i}" for i in range(200)]))
W = Set("W", np.array([f"w{i}" for i in range(100)]))

m = Model("transport")
x = m.var("x", (P, W))
for i in range(20):
    m.eq(f"cap{i}", Sum(W, x[P, W]) <= 1.0)

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
the model holds twenty term lists.

## The cost

An expression is materialised twice: once to compute its shape when the
constraint is added, and once to write its entries when the matrix is
assembled. Building twice costs build time. In exchange, one expression is
live at a time rather than all of them, so peak memory is set by the largest
constraint rather than by their sum.

A model that is cheap to declare and more expensive to assemble suits a
builder, because declaration is what a caller iterates on.
