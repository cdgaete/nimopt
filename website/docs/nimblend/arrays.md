---
title: nimblend arrays
description: The array contract, the two implementations behind it, and what an absence declaration means.
---

# nimblend arrays

`nimblend` is the layer below `nimopt`. Its vocabulary is dimensions, labels,
entries and alignment, and it contains no optimization term. A `nimopt` caller
uses these names when reading a solution or inspecting what a model built.

**Import from `nimblend` itself, never from a submodule. Never read an
array's `.index` or `.data` or a domain's `.codes`.** Those are the raw
index matrix, the value buffer and the ravelled members of the layer below
the array layer. Reading them bypasses the contract, and a test in this
repository fails on any of them.

Do not assemble one either. `SparseArray.from_canonical` below takes an index
matrix built by the caller, and that is the work of the array layer. No module
of `nimopt` calls it, and a test enforces that. A `Domain` returns the array
over its own members instead.

## `Array`

A labeled N-dimensional array. `Array` is the contract both implementations
satisfy, and the interface a caller writes against.

`absence` declares the meaning of a coordinate the array does not have:
`"empty"` that it contributes nothing, `"unknown"` that it was not modeled.
Operators and reductions follow from that declaration. The declaration is
part of the definition of the array, not a hint.

| Member | Returns |
| --- | --- |
| `dims`, `shape`, `nnz` | the dimensions, their sizes, and the number of entries |
| `coords` | the coordinate of each dimension, resolving its labels |
| `absence` | the meaning of a coordinate the array does not have |
| `as_empty()`, `as_unknown()` | the array under the other absence declaration |
| `values()`, `coordinates()` | the entries and their multi-indices |
| `domain()` | the coordinates the array has |
| `sum`, `min`, `max`, `mean` | reductions over named dimensions |
| `sel`, `restrict` | a selection by label, and by domain |
| `rename`, `transpose`, `expand`, `conform` | reshaping the dimensions |
| `shift`, `roll` | a lag that drops, and one that wraps |
| `group` | entries combined into a destination |
| `to_dense(fill=None)` | the entries as an ndarray |
| `+`, `-`, `*`, `/`, unary `-` | arithmetic over one frame, and with a scalar |

The arithmetic is part of the contract, not an implementation's own: a
caller writing `coefficient * columns` is writing against `Array`. Both
implementations behave alike, including over frames that differ. One frame
nested inside the other broadcasts over the wider; frames sharing some
dimensions align on those and multiply out the rest. Frames sharing no
dimension raise. A product mixing the two implementations returns a
`SparseArray`: a product intersects presence and has at most the entries of
the sparse operand.

Absence and zero remain distinct. A stored `0.0` is a coordinate present with
the value zero. A coordinate the array does not have is a different case.

```python
import numpy as np
import nimblend as nb

labels = {"A": np.array(["a0", "a1"]), "B": np.array(["b0", "b1"])}
array = nb.SparseArray.from_dense(np.array([[1.0, 0.0], [0.0, 2.0]]), labels)

print(isinstance(array, nb.Array))
print(array.dims, array.shape, array.absence)
print(array.nnz)
print(array.sum("B").to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
True
('A', 'B') (2, 2) empty
4
[1. 2.]
```

</details>
<!-- /output -->

`from_dense` stores every cell it was given, and this array has four entries
and not two. The zeros are stored, and a stored value is present.

## `SparseArray`

Entries in canonical order, under a coordinate per dimension. An entry that
is not stored is absent.

It is what an operation returns when the result is sparse, and what
`Solution.primal` returns for a variable over a subset: a dense frame there
would be the grid the variable was declared to avoid.

| Member | Returns |
| --- | --- |
| `SparseArray.from_dense(values, labels)` | every cell of an ndarray |
| `SparseArray.from_canonical(index, values, coords, dims)` | entries already in order |
| `as_empty()`, `as_unknown()` | the same entries under the other declaration |
| `to_csr()` | the entries as compressed rows |

```python
import numpy as np
import nimblend as nb

labels = {"A": np.array(["a0", "a1"]), "B": np.array(["b0", "b1"])}
array = nb.SparseArray.from_dense(np.array([[1.0, 0.0], [0.0, 2.0]]), labels)

print(array.absence)
print(array.as_unknown().absence)
print(array.to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
empty
unknown
[[1. 0.]
 [0. 2.]]
```

</details>
<!-- /output -->

## `DenseArray`

An ndarray over labeled dimensions, distinguishing absence from zero.

The storage of presence follows the absence declaration: the two
declarations require opposite behavior from an operator. An `"unknown"` array
marks absence with NaN. NaN propagates through arithmetic at no cost and
requires no storage beside the values. An `"empty"` array stores a boolean
mask: absence is the additive identity there, and substituting it costs less
than marking it.

`Solution.primal` returns one of these for a variable over a full product.
The solver returns a value at every cell of the frame in column order, and
the values reshape with no index built.

```python
import numpy as np
import nimblend as nb

coords = {
    "A": nb.StoredCoord(np.array(["a0", "a1"])),
    "B": nb.StoredCoord(np.array(["b0", "b1"])),
}
array = nb.DenseArray(np.array([[1.0, 0.0], [0.0, 2.0]]), coords, ("A", "B"))

print(array.absence, array.nnz)
print(array.present)
print(array.as_unknown().absence)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
empty 4
[[ True  True]
 [ True  True]]
unknown
```

</details>
<!-- /output -->

## Building an array from columns

`nimblend` itself provides the two constructors a caller uses when the data is
not already an ndarray.

| Constructor | Returns |
| --- | --- |
| `from_long(dims, coords, labels, values)` | one label column per dimension and one value column |
| `from_dense(values, labels)` | every cell of an ndarray |
| `is_canonical(index, shape)` | whether buffers are in the order `from_canonical` takes |

`from_long` resolves each label through the coordinate that dimension
already has. A caller with coordinates of its own therefore gives its entries
as labels and resolves no position itself. The columns are read in parallel
and must have equal length.

```python
import numpy as np
import nimblend as nb

coords = {
    "t": nb.StoredCoord(np.array([2030, 2040])),
    "r": nb.StoredCoord(np.array(["DE", "FR"])),
}
arr = nb.from_long(
    ("t", "r"),
    coords,
    {"t": np.array([2030, 2040, 2040]), "r": np.array(["DE", "DE", "FR"])},
    np.array([5.0, 6.0, 7.0]),
)
print(arr.nnz, arr.to_dense()[1, 1])
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
3 7.0
```

</details>
<!-- /output -->

A column of a different length raises `ValueError`; the message gives the
column and both lengths.

```python raises=ValueError
import numpy as np
import nimblend

nimblend.from_long(
    ("t",),
    {"t": nimblend.StoredCoord(np.array([2030, 2040]))},
    {"t": np.array([2030, 2040])},
    np.array([1.0]),
)
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: label column 't' has length 2 and the value column has length 1; they name the same entries
```

</details>
<!-- /output -->

`is_canonical` reports whether buffers meet the precondition of
`SparseArray.from_canonical`: entries sorted by ravel key, with no repeat.
Verifying it inside `from_canonical` would cost the ravel that this path
avoids.

```python
import numpy as np
import nimblend as nb

ordered = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32)
print(nb.is_canonical(ordered, (2, 2)))
print(nb.is_canonical(ordered[:, ::-1].copy(), (2, 2)))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
True
False
```

</details>
<!-- /output -->

## The frame of a binary result

`combined_dims(left, right)` returns the dimensions a binary operator's
result has, from the dimensions of the two operands alone. A caller reads it
before materialising either operand. A combination therefore reports its
frame while its data is unbound.

| Operands | Result |
| --- | --- |
| equal frames | that frame, in its order |
| one frame nested in the other | the wider |
| frames that overlap | the left, then the dimensions only the right has |
| frames sharing no dimension | raises |

```python
import nimblend as nb

print(nb.combined_dims(("P", "Q"), ("Q", "R")))
print(nb.combined_dims(("P",), ("P", "Q")))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'Q', 'R')
('P', 'Q')
```

</details>
<!-- /output -->

Frames sharing no dimension have no common dimension to align on, and
`combined_dims` raises. Their combination would be an outer product.

```python raises=ValueError
import nimblend

nimblend.combined_dims(("P",), ("Q",))
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: frames ('P',) and ('Q',) share no dimension; there is nothing to align them on
```

</details>
<!-- /output -->

## Densifying an unknown array

An array declaring `"unknown"` that does not have every coordinate of its
frame raises on `to_dense()` without a fill. There is no value it can place
at the rest, and choosing one silently would invent a value.

```python raises=ValueError
import numpy as np
import nimblend

labels = {"A": np.array(["a0", "a1"]), "B": np.array(["b0", "b1"])}
partial = nimblend.SparseArray.from_canonical(
    np.array([[0], [0]], dtype=np.int32),
    np.array([1.0]),
    {
        "A": nimblend.StoredCoord(labels["A"]),
        "B": nimblend.StoredCoord(labels["B"]),
    },
    ("A", "B"),
    absence="unknown",
)

partial.to_dense()
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: this array declares absence 'unknown' and does not carry every coordinate of its frame, so densifying must state fill=<value> to place at the rest
```

</details>
<!-- /output -->

With a fill value, the grid is returned.

```python
import numpy as np
import nimblend as nb

labels = {"A": np.array(["a0", "a1"]), "B": np.array(["b0", "b1"])}
partial = nb.SparseArray.from_canonical(
    np.array([[0], [0]], dtype=np.int32),
    np.array([1.0]),
    {
        "A": nb.StoredCoord(labels["A"]),
        "B": nb.StoredCoord(labels["B"]),
    },
    ("A", "B"),
    absence="unknown",
)

print(partial.to_dense(fill=np.nan))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[[ 1. nan]
 [nan nan]]
```

</details>
<!-- /output -->
