---
title: The array is the matrix
description: Why a constraint needs no assembly step, and what a model's build does.
---

# The array is the matrix

A constraint is a `nimblend` array indexed over its free sets and the column
space, or over `(ROW, COLUMN)` once its frame has been grouped into rows.
The values of that array are the coefficients. No step converts a model
into a matrix, because the array already is one.

```python
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["p0", "p1"]))
W = Set("W", np.array(["w0", "w1", "w2"]))

m = Model("transport")
x = m.var("x", (P, W))
m.eq("supply", Sum(W, x[P, W]) <= 1.0)

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
'__column__')`, which is a matrix in every sense that matters: a row index,
a column index and a value.

## One buffer

A model allocates one `nimblend.EntryBuffer` for its whole matrix. Each
constraint reserves the slice its coefficients need and groups its block
directly into that slice, so the block never exists as a second object.

The frame precedes the column dimension in canonical order, so the grouping
reads a leading prefix and the result is canonical as written: no sort
afterwards, no copy into place.

Handing the matrix to a solver is then a matter of returning views.
`indices` and `values` are the buffer; only `indptr` is built. A model of
four million nonzeros hands over its matrix without copying it.

## Why the shape is computed first

Reserving a slice requires its size, so a constraint computes its shape when
it is added and writes its entries when the model is assembled. The two must
agree, and the model checks that they do.

If a parameter's data changes between the two, the constraint computed one
number of coefficients and built another. The model raises rather than
writing a matrix that does not match the shape it reported.

## What the design gives up

Materialising the expression twice costs build time. That is the price of
holding one expression at a time rather than every constraint's block at
once, and it is a deliberate trade: peak memory is set by the largest
constraint, not by the sum of all of them.

It also means a model's declared shape is exact before anything is built.
`n_rows`, `n_columns` and `nnz` are facts, not estimates.
