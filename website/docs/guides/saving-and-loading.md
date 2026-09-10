---
title: Saving and loading a model
description: Write a definition or a built model to a file in nimopt's own spelling, read it back, and carry the data inline or beside it.
---

# Saving and loading a model

A definition writes itself to a YAML file whose expressions are spelled as
they are typed in Python. The file is the model's final form: every derived
coefficient spelled out, every term with its own sum and sign, the constant
last.

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
version: 2
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

The structure section is the data's schema: every set, and every parameter
with its dimensions. `build` refuses a mapping that misses any of them by name.

## Reading a file back

`loads` reads text and `load` reads a path. A file without data returns a
`Definition`, whose next step is `build(data)`.

```python
import numpy as np
from nimopt import loads

text = """
version: 2
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

The text is read through the same operators a Python model is built from,
so an edit is accepted where Python accepts it and normalised the same way.
Adding a scalar, reordering terms, or writing a comparison the other way
round all read, and the file written back is the canonical form.

```python
from nimopt import loads

edited = """
version: 2
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

An edit Python refuses is refused here with the same message, and a
construct outside the spelling is refused naming it.

```python raises=ValueError
from nimopt import loads

loads(
    """
version: 2
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
ValueError: 'max(gen[G, T]) <= 10': Sum is the one call the syntax supports
```

</details>
<!-- /output -->

## Data inline, for a model small enough to read

A built model writes its data into the file with `inline=True`. A set is a
list, a dense parameter is nested lists, and a parameter carrying some
coordinates of its product is a table of the dimensions then `value`. A
file carrying data loads to a built `Model`.

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

## Data beside the file, for a model of size

`save` writes a model's file and its data as an `.npz` beside it, under the
file's stem, and the file names that sidecar. `load` reads both. The pair
moves together; the file names no path and no machine.

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
with the mapping `build` takes or the path of an `.npz`. A file that carries
data and a `data=` together is refused, because two sources for one model is
a choice the library does not make.

## A file that describes its own format

`instructions=True` on `save`, `Definition.to_yaml` and `Model.to_yaml` writes
a comment block at the top of the file. The block is the same in every file.
It describes the format, not the model: the keys and their order, the
defaults, the rules that decide which rows a constraint has, and the
expression syntax. A reader given one file can interpret it without this
package.

The block is a YAML comment, so a file that carries it loads to the same
model as one that does not, and writing the loaded model with the flag gives
the same text.

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
# A nimopt model file, format version 2. The keys are written in this
# order, and no other key is accepted: version, name, sense, sets,
# aliases, parameters, variables, constraints, objective, data. Only
# version, name and sense are required.
True
True
```

</details>
<!-- /output -->
