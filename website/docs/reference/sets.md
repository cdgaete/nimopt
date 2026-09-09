---
title: Sets and domains
description: Name a dimension, give it a second name, and list which members of a product a model uses.
---

# Sets and domains

## `Set`

```
Set(name, labels)
```

A named dimension with labels. `labels` is an array; `name` is what every
reference to the dimension uses.

| Member | Returns |
| --- | --- |
| `name`, `labels` | the declared name and labels |
| `len(set)` | the number of members |
| `position_of(labels)` | the position of each label given |
| `coord` | the coordinate the labels resolve through |
| `cyclic` | the same set, with a lag that wraps rather than drops |
| `set - 1` | the set lagged, dropping the members a lag runs off |

```python
import numpy as np
from nimopt import Set

T = Set("T", np.array(["t0", "t1", "t2"]))

print(T.name, len(T))
print(T.labels)
print(T.position_of(np.array(["t2", "t0"])))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
T 3
['t0' 't1' 't2']
[2 0]
```

</details>
<!-- /output -->

## `Alias`

```
Alias(name, set)
```

A second name for a set, sharing its labels and its coordinate. A
parameter over a set and its alias is an ordinary two-dimensional array,
so a model relating a set to itself does so without declaring a second
set. No labels are copied: the alias uses the coordinate the set already
built.

```python
import numpy as np
from nimopt import Alias, Param, Set

N = Set("N", np.array(["a", "b"]))
M = Alias("M", N)

flow = Param.from_dense("flow", (N, M), np.array([[0.0, 1.0], [1.0, 0.0]]))
print(flow.dims)
print(flow.materialise().to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
('N', 'M')
[[0. 1.]
 [1. 0.]]
```

</details>
<!-- /output -->

## `product`

```
product(sets)
```

Every member of a set product, as a domain. Passed to `over=`, it states
the rows of a constraint explicitly, for a constraint whose terms each
cover some of its rows.

## `subset`

```
subset(sets, columns)
```

The members of a set product a model uses, given by label. `columns` holds
one label column per set, keyed by the set's name, read in parallel: the
k-th entry of each column belongs to the same member. It is a list of
members, not a cross product.

## `subset_of`

```
subset_of(sets, index)
```

The same, given by position. Each column of `index` is one member. A
caller holding positions passes them directly rather than building labels
to resolve back.

```python
import numpy as np
from nimopt import Set, product, subset, subset_of

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

print(product((P, W)).size)
print(
    subset(
        (P, W), {"P": np.array(["lisbon", "porto"]), "W": np.array(["berlin", "paris"])}
    ).size
)
print(subset_of((P, W), np.array([[0, 1], [0, 1]])).size)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
6
2
2
```

</details>
<!-- /output -->

The product has six members. Both subsets have two, `lisbon` with `berlin`
and `porto` with `paris`, because the columns are read in parallel.
