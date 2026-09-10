---
title: The array is the matrix
description: Why a constraint needs no assembly step, and what a model's build does.
---

# The array is the matrix

A constraint is a `nimblend` array indexed over its free sets and the column
space, or over `(ROW, COLUMN)` once its frame has been grouped into rows.
The values of that array are the coefficients. No step converts a model into
a matrix: the array is the matrix.

```python
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

m = Model("transport")
x = m.var("x", (P, W))
m.constraint("supply", Sum(W, x[P, W]) <= 1.0)

print(x.terms().dims)
print(m.assemble().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W', '__column__')
[[1. 1. 1. 0. 0. 0.]
 [0. 0. 0. 1. 1. 1.]]
```

</details>
<!-- /output -->

The variable's coefficients are indexed over `('P', 'W', '__column__')`.
Grouping the frame into rows puts the same entries over `('__row__',
'__column__')`. That is a matrix: a row index, a column index and a value.

## One buffer

A model allocates one `nimblend.EntryBuffer` for its whole matrix. Each
constraint reserves the slice its coefficients need and groups its block
directly into that slice. The block never exists as a second object.

The frame precedes the column dimension in canonical order. The grouping
reads a leading prefix, and the result is canonical as written, with no sort
afterwards and no copy into place.

Passing the matrix to a solver returns views. `indices` and `values` are the
buffer, and only `indptr` is built. A model of four million nonzeros passes
its matrix without copying it.

## Why the shape is computed first

Reserving a slice requires its size. A constraint computes its shape when it
is added and writes its entries when the model is assembled. The model checks
that the two agree.

Where the data of a parameter changes between the two, the constraint
computes one number of coefficients and writes another. The model raises, and
it writes no matrix that differs from the shape it reported.

## What the design costs

Materialising the expression twice costs build time. One expression exists at
a time, and the blocks of the constraints never exist together. Peak memory
is set by the largest constraint, not by the sum of all of them.

The declared shape of a model is also exact before anything is built.
`n_rows`, `n_columns` and `nnz` are exact counts, not estimates.
