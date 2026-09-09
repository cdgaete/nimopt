---
title: nimblend arrays
description: The array contract, the two implementations behind it, and what an absence declaration means.
---

# nimblend arrays

`nimblend` is the layer below `nimopt`. It knows dimensions, labels, entries
and alignment, and nothing about optimization. A `nimopt` caller meets these
names when reading a solution or inspecting what a model built.

**Import from `nimblend` itself, never from a submodule, and never read an
array's `.index` or `.data` or a domain's `.codes`.** Those are the raw
index matrix, value buffer and ravelled members of the layer below the
array layer; reading them bypasses the contract. A test in this repository
fails on any of them.

Nor assemble one. `SparseArray.from_canonical` below takes an index matrix
the caller built, which is the array layer's own work: `nimopt`'s modules
call it nowhere, and a test holds them to that. A `Domain` returns the
array over its own members instead.

## `Array`

A labeled N-dimensional array. `Array` is the contract both
implementations satisfy, and what a caller writes against.

`absence` declares what a coordinate the array does not have means:
`"empty"` that it contributes nothing, `"unknown"` that it was not
modelled. Operators and reductions follow from that declaration, so it is
part of the array's meaning and not a hint.

| Member | Returns |
| --- | --- |
| `dims`, `shape`, `nnz` | the dimensions, their sizes, and the number of entries |
| `coords` | each dimension's coordinate, which resolves its labels |
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
`SparseArray`, because a product intersects presence and so has at most
what the sparse operand has.

Absence and zero stay distinct. A stored `0.0` is a coordinate that is
present with value zero, which is not the same as one the array does not
have.

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

`from_dense` stores every cell it was given, so this array has four entries
and not two: the zeros are stored, and stored means present.

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

How presence is stored follows the absence declaration, because the two
declarations need opposite things from an operator. An `"unknown"` array
tags absence with NaN, which propagates through arithmetic at no cost and
needs no storage beside the values. An `"empty"` array carries a boolean
mask, because absence is the additive identity there and substituting it is
cheaper than tagging.

`Solution.primal` returns one of these for a variable over a full product:
the solver returns a value at every cell of the frame in column order, so
they reshape with no index built at all.

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
already has, so a caller holding coordinates, which is any caller with sets
of its own, gives its entries as labels rather than resolving them to
positions first. The columns are read in parallel, so they must have equal
length.

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

`is_canonical` answers the question `SparseArray.from_canonical` asks a
caller to answer about buffers the caller built: entries sorted by ravel
key with no repeat. Verifying it inside `from_canonical` would cost the
ravel that path exists to avoid.

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
result has, from the two operands' dimensions alone. A caller reads it
before materialising either operand, which is what lets a combination
report its frame while its data is still unbound.

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

Frames sharing no dimension have nothing to align on, so `combined_dims`
raises: their combination would be an outer product no caller asked for.

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
