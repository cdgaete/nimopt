---
title: nimblend domains
description: The coordinates an array has, the three ways a position is computed, and the buffer assembly writes into.
---

# nimblend domains

## `Domain`

A sorted, unique set of multi-indices over named dimensions.

A coordinate resolves labels for one dimension; a domain does so for a
tuple of them: which multi-indices it has, what position each occupies, and
which multi-index sits at a position. It records which coordinates are
present and nothing about where they are numbered from.

| Constructor | Returns |
| --- | --- |
| `Domain.full(dims, coords)` | every coordinate of the product `dims` spans |
| `Domain.from_labels(dims, coords, labels)` | a domain from one label column per dimension |
| `Domain.from_coordinates(dims, coords, index)` | a domain from an index matrix of one row per dimension |

| Member | Returns |
| --- | --- |
| `size`, `dims`, `shape`, `coords` | the number of members, the dimensions, their extents and their coordinates |
| `is_full` | whether every coordinate of the product is present |
| `coordinates()` | the multi-index of each member, as an int32 index matrix |
| `labels()` | each member's label, per dimension |
| `intersect(other)` | the members both have |
| `union(other)` | the members either has |
| `difference(other)` | the members this one has and `other` does not |
| `positions_of(array)` | each entry of `array` as its position here, `-1` where absent |
| `positions_of_coordinates(index)` | each column of an index matrix as its position here, `-1` where absent |
| `expand(dims, coords)` | every member crossed with the full extent of the named dimensions |
| `transpose(*dims)` | the same members, over the dimensions in the order given |
| `as_coord(start=0)` | the domain read as a coordinate, its members numbered from `start` |
| `array(values, absence="empty")` | the members with one value each, as a `SparseArray` |
| `identity(into, coord, start=0)` | each member paired with its own position along `into`, valued 1.0 |

That table is the whole surface. **A domain's `codes` are the raw ravelled
members of the layer below it, as an array's `.index` and `.data` are its
raw buffers; never read them.** `coordinates()` and `labels()` report which
members are present, `positions_of_coordinates` reports where one sits,
`as_coord` numbers them, and `array` and `identity` return an array over
them, so nothing above needs to build an index matrix either. A test in
this repository fails on a read of any of the three.

The label columns of `from_labels` are read in parallel: the k-th entry of
each column belongs to the same member. A domain is a list of members, not
a cross product.

```python
import numpy as np
import nimblend as nb

coords = {
    "P": nb.StoredCoord(np.array(["lisbon", "porto"])),
    "W": nb.StoredCoord(np.array(["berlin", "paris", "rome"])),
}

full = nb.Domain.full(("P", "W"), coords)
pairs = nb.Domain.from_labels(
    ("P", "W"),
    coords,
    {"P": np.array(["lisbon", "porto"]), "W": np.array(["berlin", "paris"])},
)

print(full.size, pairs.size)
print(pairs.coordinates())
print(pairs.labels())
print(full.intersect(pairs).size, full.difference(pairs).size)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
6 2
[[0 1]
 [0 1]]
{'P': array(['lisbon', 'porto'], dtype='<U6'), 'W': array(['berlin', 'paris'], dtype='<U6')}
2 4
```

</details>
<!-- /output -->

The product spans six members; `pairs` has two, `lisbon` with `berlin` and
`porto` with `paris`, because the columns are read in parallel.

`is_full` reports whether a domain has every coordinate its dimensions
span, which is what a caller checks before reading values ordered by member
as a grid: `full.is_full` is `True` and `pairs.is_full` is `False`.

## Querying a domain

`positions_of` takes an array; `positions_of_coordinates` takes the index
matrix behind one, so a caller holding positions queries without building
an array. A member the domain does not have returns `-1`, so a membership
test reads as `>= 0`.

`as_coord` reads the domain as a coordinate: a member's position is its
rank among the members present, numbered from `start`. This is how a
dimension spanning a subset of a product is numbered.

```python
import numpy as np
import nimblend as nb

coords = {
    "P": nb.StoredCoord(np.array(["lisbon", "porto"])),
    "W": nb.StoredCoord(np.array(["berlin", "paris", "rome"])),
}
pairs = nb.Domain.from_labels(
    ("P", "W"),
    coords,
    {"P": np.array(["lisbon", "porto"]), "W": np.array(["berlin", "paris"])},
)

asked = np.array([[0, 1, 1], [0, 1, 2]], dtype=np.int32)
print(pairs.positions_of_coordinates(asked))
print(pairs.positions_of_coordinates(asked) >= 0)

numbered = pairs.as_coord(100)
print(numbered.to_position(asked[:, :2]))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[ 0  1 -1]
[ True  True False]
[100 101]
```

</details>
<!-- /output -->

`("porto", "rome")` is not a member, so it returns `-1`. The other two are
the domain's first and second members, and `as_coord(100)` numbers them
from 100.

## Crossing a domain with further dimensions

`expand` replicates every member across the full extent of the named
dimensions, which is how the coordinates a term *could* have are enumerated
before asking which of them it does. The new dimensions are appended;
`transpose` reads the result in another order. A member's code is its own
scaled by the appended extent, plus each position within it, so the cross
product is arithmetic on the members and no index matrix is built to hold
it.

```python
import numpy as np
import nimblend as nb

coords = {
    "P": nb.StoredCoord(np.array(["lisbon", "porto"])),
    "W": nb.StoredCoord(np.array(["berlin", "paris", "rome"])),
    "H": nb.StoredCoord(np.array([0, 1])),
}
pairs = nb.Domain.from_labels(
    ("P", "W"),
    coords,
    {"P": np.array(["lisbon", "porto"]), "W": np.array(["berlin", "paris"])},
)

hourly = pairs.expand(("H",), coords)
print(hourly.dims, hourly.size)
print(hourly.labels())
print(hourly.transpose("H", "P", "W").dims)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('P', 'W', 'H') 4
{'P': array(['lisbon', 'lisbon', 'porto', 'porto'], dtype='<U6'), 'W': array(['berlin', 'berlin', 'paris', 'paris'], dtype='<U6'), 'H': array([0, 1, 0, 1])}
('H', 'P', 'W')
```

</details>
<!-- /output -->

Two members crossed with two hours are four, and `transpose` presents them
over the dimensions in another order without changing which members are
present.

## A domain returns an array

A caller holding one value per member, or wanting each member paired with
its own position along a new dimension, asks the domain rather than
building an index matrix. Both are readers standing above the raw members,
as `coordinates()` and `as_coord()` are.

`array(values)` assigns one value to each member, in the order they are
held. The members ascend, so the entries are canonical as written and no
sort runs.

```python
import numpy as np
import nimblend as nb

coords = {"t": nb.StoredCoord(np.array([2030, 2040, 2050]))}
members = nb.Domain.full(("t",), coords)
print(members.array(np.array([1.0, 2.0, 3.0])).values())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[1. 2. 3.]
```

</details>
<!-- /output -->

A full domain's members ascend with the ravel key, which is the order
`values.ravel()` reads a grid in, so a whole array is built in one call.

```python
import numpy as np
import nimblend as nb

coords = {
    "x": nb.StoredCoord(np.array(["a", "b"])),
    "y": nb.StoredCoord(np.array([10, 20, 30])),
}
values = np.arange(6, dtype=np.float64).reshape(2, 3)
grid = nb.Domain.full(("x", "y"), coords).array(values.ravel())
print(grid.to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[[0. 1. 2.]
 [3. 4. 5.]]
```

</details>
<!-- /output -->

One value per member is the whole rule. A column of another length raises
`ValueError`.

```python raises=ValueError
import numpy as np
import nimblend

coords = {"t": nimblend.StoredCoord(np.array([2030, 2040, 2050]))}
nimblend.Domain.full(("t",), coords).array(np.array([1.0, 2.0]))
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: a domain of 3 member(s) takes one value each, as a column of that length; got shape (2,)
```

</details>
<!-- /output -->

`identity(into, coord, start)` pairs each member with its own position
along a new dimension, valued 1.0. A member's position is its rank plus
`start`, which is the numbering `as_coord(start)` uses, so an array built
one way and a coordinate built the other place a member alike. `coord` is
the coordinate of the new dimension and spans the whole extent the
positions are numbered into, wider than these members where several
domains share one numbering.

```python
import numpy as np
import nimblend as nb

coords = {"t": nb.StoredCoord(np.array([2030, 2040, 2050]))}
members = nb.Domain.full(("t",), coords)
paired = members.identity("k", nb.ProductCoord((20,)), start=10)
print(paired.dims)
print(paired.coordinates())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('t', 'k')
[[ 0  1  2]
 [10 11 12]]
```

</details>
<!-- /output -->

A destination too short for the members it is asked to number raises
`ValueError` rather than writing a position outside it.

```python raises=ValueError
import numpy as np
import nimblend

coords = {"t": nimblend.StoredCoord(np.array([2030, 2040, 2050]))}
members = nimblend.Domain.full(("t",), coords)
members.identity("k", nimblend.ProductCoord((6,)), start=4)
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: 3 member(s) numbered from 4 reach position 6, and dimension 'k' spans 6
```

</details>
<!-- /output -->

## The three coordinates

A coordinate resolves where a label sits along one dimension. Which of the
three is used follows from what the dimension is.

| Coordinate | Is | Used for |
| --- | --- | --- |
| `StoredCoord(labels)` | labels held as an array | a dimension whose members are named |
| `ProductCoord(sizes, start=0)` | positions of a full product, numbered from `start` | a dimension whose positions are computed, such as a variable's columns |
| `SubsetCoord(codes, sizes, start=0)` | positions of a subset of a product, numbered from `start` in code order | a variable over a subset, where a position is a rank among the codes |

`SubsetCoord` gives an entry's position as its rank among the codes, so a
block already in canonical order needs no lookup at all.

```python
import numpy as np
import nimblend as nb

stored = nb.StoredCoord(np.array(["a", "b", "c"]))
print(stored.to_position(np.array(["c", "a"])))

product = nb.ProductCoord((2, 3))
print(product.to_position(np.array([[0, 1], [2, 0]])))

subset = nb.SubsetCoord(np.array([0, 4]), (2, 3))
print(subset.to_position(np.array([[0, 1], [0, 1]])))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
[2 0]
[2 3]
[0 1]
```

</details>
<!-- /output -->

`StoredCoord` looks a label up among the ones it holds, so `"c"` resolves
to position 2. `ProductCoord` ravels a multi-index against the sizes, so
`(0, 2)` is position 2 and `(1, 0)` is position 3. `SubsetCoord` holds the
codes `0` and `4`, which are those same two members, and returns their
ranks.

`to_position` takes an index matrix of one row per dimension, and each
column is one entry.

## `EntryBuffer`

A fixed index and value buffer handing out successive slices.

A block computed into a reserved slice never exists as a separate object,
so assembling several of them holds one copy of the result rather than one
copy per block plus the result. It is the destination a model's assembly
writes into: each constraint writes its rows into its own slice of one
buffer.

| Member | Returns |
| --- | --- |
| `EntryBuffer(ndim, capacity)` | a buffer for `capacity` entries of `ndim` dimensions |
| `reserve(n)` | the next `n` index and value slices, to write into |
| `written()` | the index and values written so far |
| `array(coords, dims, absence="empty")` | what was written, as an array |

```python
import numpy as np
import nimblend as nb

buffer = nb.EntryBuffer(2, 4)
index, values = buffer.reserve(2)
index[:] = np.array([[0, 1], [0, 1]])
values[:] = np.array([5.0, 6.0])

coords = {
    "A": nb.StoredCoord(np.array(["a0", "a1"])),
    "B": nb.StoredCoord(np.array(["b0", "b1"])),
}
array = buffer.array(coords, ("A", "B"))

print(array.nnz)
print(array.to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
2
[[5. 0.]
 [0. 6.]]
```

</details>
<!-- /output -->

The slices `reserve` hands out are views of the one allocation, so writing
into them is writing into the array that comes out.
