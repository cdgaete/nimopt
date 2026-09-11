---
title: Saving and loading a model
description: Write a definition or a built model to a file in the expression syntax of nimopt, read it back, and store the data inline or beside it.
---

# Saving and loading a model

A definition writes a YAML file whose expressions are written as they are
typed in Python. The file is the canonical form of the model: every derived
coefficient written out, every term with its own sum and sign, and the
constant last.

```python
from nimopt import Definition, Sum

d = Definition("dispatch", sense="min")
G, T = d.set("G"), d.set("T")
price, eta = d.param("price", (G, T)), d.param("eta", (G, T))
cap, load = d.param("cap", (G, T)), d.param("load", (T,))
gen = d.var("gen", (G, T), upper=cap)
d.constraint("balance", Sum(G, gen[G, T]) == load[T])
d.set_objective(Sum(G, T, 2 * (price[G, T] / eta[G, T]) * gen[G, T]))

print(d.to_yaml())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
version: 3
name: dispatch
sense: min
sets: [G, T]
parameters:
  price: [G, T]
  eta: [G, T]
  cap: [G, T]
  load: [T]
variables:
  gen:
    sets: [G, T]
    upper: cap
constraints:
  balance:
    relation: Sum(G, gen[G, T]) == load[T]
objective: Sum(G, T, ((price[G, T] / eta[G, T]) * 2) * gen[G, T])
```

</details>
<!-- /output -->

The structure section is the schema of the data: every set, and every
parameter with its dimensions. `build` raises `ValueError` for a mapping that
omits one of them.

## Reading a file back

`loads` reads text and `load` reads a path. A file without data returns a
`Definition`, whose next step is `build(data)`.

```python
import numpy as np
from nimopt import loads

text = """
version: 3
name: dispatch
sense: min
sets: [G, T]
parameters:
  cost: [G]
  load: [T]
variables:
  gen:
    sets: [G, T]
    upper: 10.0
constraints:
  balance:
    relation: Sum(G, gen[G, T]) == load[T]
objective: Sum(G, T, cost[G] * gen[G, T])
"""

d = loads(text)
m = d.build(
    {
        "G": np.array(["a", "b"]),
        "T": np.arange(2),
        "cost": np.array([1.0, 3.0]),
        "load": np.array([12.0, 15.0]),
    }
)
print(m.solve().objective)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
41.0
```

</details>
<!-- /output -->

## Editing by hand

The text is read through the same operators a Python model is built from. An
edit is accepted where Python accepts it, and is normalized the same way.
Adding a scalar, reordering terms and reversing a comparison all parse. The
file written back is the canonical form.

```python
from nimopt import loads

edited = """
version: 3
name: dispatch
sense: min
sets: [G, T]
parameters:
  cost: [G]
  load: [T]
variables:
  gen:
    sets: [G, T]
constraints:
  balance:
    relation: load[T] == Sum(G, gen[G, T]) * 2 + 1 - 1
objective: Sum(G, T, cost[G] * gen[G, T])
"""
print(loads(edited).to_yaml().splitlines()[-2])
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
relation: 2 * Sum(G, gen[G, T]) == load[T]
```

</details>
<!-- /output -->

An edit that raises in Python raises here with the same message. A construct
outside the expression syntax raises and reports it.

```python raises=ValueError
from nimopt import loads

loads(
    """
version: 3
name: dispatch
sense: min
sets: [G, T]
variables:
  gen:
    sets: [G, T]
constraints:
  peak:
    relation: max(gen[G, T]) <= 10
"""
)
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: 'max(gen[G, T]) <= 10': the syntax supports one call; write Sum
```

</details>
<!-- /output -->

## Data inline, for a model small enough to read

A built model writes its data into the file with `inline=True`. A set is a
list, and a dense parameter is nested lists. A parameter with values at some
coordinates of its product is a table of the dimensions then `value`. A file
containing data loads to a built `Model`.

```python
from nimopt import loads
from nimopt.models import transport

m = transport.definition().build(transport.data())
text = m.to_yaml(inline=True)
print(text[text.index("data:") :])
print(loads(text).solve().objective)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
data:
  P: [p0, p1, p2, p3]
  W: [w0, w1, w2, w3, w4, w5]
  cost:
    columns: [P, W, value]
    rows:
    - [p0, w0, 1.1322210842282328]
    - [p0, w1, 7.506161913602179]
    - [p0, w4, 1.3277881914895575]
    - [p1, w0, 6.835972487871987]
    - [p1, w3, 8.302044618221775]
    - [p1, w4, 5.853086206137439]
    - [p2, w2, 5.348999931723383]
    - [p2, w3, 8.480579390302147]
    - [p2, w4, 7.526828432972257]
    - [p3, w1, 1.0219080013611848]
    - [p3, w2, 7.859234212700555]
    - [p3, w3, 1.2686846024437148]
  supply: [60.0, 60.0, 60.0, 60.0]
  demand: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

100.99601811246072
```

</details>
<!-- /output -->

## Datetime members

A set whose members are `datetime64` or `timedelta64` round trips through both
data sources, in every unit. Inline, such a set is written as a mapping of
`dtype` and `members`: a `datetime64` member as its ISO 8601 string, and a
`timedelta64` member as its integer count of the unit in the `dtype`. A member
fixed in a relation is written as quoted text.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum, loads

T = Set("T", np.array(["2030-01-01T00", "2030-01-01T01"], dtype="datetime64[h]"))
cost = Param.from_dense("cost", (T,), np.array([2.0, 5.0]))
m = Model("dispatch", sense="min")
gen = m.var("gen", (T,), upper=10.0)
m.constraint("start", gen["2030-01-01T00"] == 4.0)
m.constraint("total", Sum(T, gen[T]) >= 6.0)
m.set_objective(Sum(T, cost[T] * gen[T]))
text = m.to_yaml(inline=True)
print(text[text.index("constraints:") :])
print(loads(text).solve().objective)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
constraints:
  start:
    relation: gen['2030-01-01T00'] == 4
  total:
    relation: Sum(T, gen[T]) >= 6
objective: Sum(T, cost[T] * gen[T])
data:
  T:
    dtype: datetime64[h]
    members: [2030-01-01T00, 2030-01-01T01]
  cost: [2.0, 5.0]

18.0
```

</details>
<!-- /output -->

The fixed member is written as a string and is read back to the same member.
A string is parsed as ISO 8601, and a `datetime.datetime`, a `datetime.date`
and a `datetime64` of another unit are converted. An integer against a
`timedelta64` set is a count of that set's own unit. A conversion that is not
exact raises `ValueError` instead of truncating.

```python raises=ValueError
import numpy as np
from nimopt import Model, Set

T = Set("T", np.array(["2030-01-01T00", "2030-01-01T01"], dtype="datetime64[h]"))
m = Model("dispatch")
gen = m.var("gen", (T,))
gen["2030-01-01T00:30"]
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: member '2030-01-01T00:30' does not convert exactly to datetime64[h] at dimension 'T' of variable 'gen'; write a member in the unit of that dimension
```

</details>
<!-- /output -->

`Model.row` takes the same text for a datetime dimension, and a `Row` displays
each datetime coordinate in it.

## Data beside the file, for a large model

`save` writes a model's file and its data as an `.npz` beside it, under the
file's stem. The file records the sidecar name. `load` reads both. The file
records no path and no machine name.

```python
import tempfile
from pathlib import Path
from nimopt import load, save
from nimopt.models import storage

m = storage.definition().build(storage.data())
with tempfile.TemporaryDirectory() as held:
    path = Path(held) / "storage.yaml"
    save(m, path)
    print(sorted(p.name for p in Path(held).iterdir()))
    print(path.read_text().splitlines()[-1])
    print(load(path).solve().status)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
['storage.npz', 'storage.yaml']
data: storage.npz
optimal
```

</details>
<!-- /output -->

A definition's file takes data from the caller instead: `load(path, data=...)`
with the mapping `build` takes or the path of an `.npz`. A file that contains
data and a `data=` together raises `ValueError`: one model takes one data
source.

## A file that describes its own format

`instructions=True` on `save`, `Definition.to_yaml` and `Model.to_yaml` writes
a comment block at the top of the file. The block is the same in every file.
It describes the format, not the model: the keys and their order, the
defaults, the rules that determine which rows a constraint has, and the
expression syntax. A reader with one file interprets it without this package.

The block is a YAML comment. A file with it loads to the same model as one
without it, and writing the loaded model with the flag gives the same text.

```python
from nimopt import Definition, loads

d = Definition("dispatch", sense="min")
T = d.set("T")
load, gen = d.param("load", (T,)), d.var("gen", (T,))
d.constraint("balance", gen[T] == load[T])
text = d.to_yaml(instructions=True)
print("\n".join(text.splitlines()[:5]))
print(loads(text).to_yaml() == d.to_yaml())
print(loads(text).to_yaml(instructions=True) == text)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
# --- Reading this file --------------------------------------------------
# A nimopt model file, format version 3. The keys are written in this
# order, and no other key is accepted: version, name, sense, sets,
# aliases, parameters, variables, constraints, objective, data. Only
# version, name and sense are required.
True
True
```

</details>
<!-- /output -->
