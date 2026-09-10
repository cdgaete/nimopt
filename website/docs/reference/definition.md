---
title: Definition
description: The symbols and constraints a model is written from, declared before any data exists.
---

# Definition

## `Definition`

```
Definition(name="definition", sense="min")
```

A definition declares the sets, parameters and variables a model is written
from, and its constraints, in the expression syntax a model uses. It holds
no data: a set declared here names a dimension and has no members, and a
parameter names a shape and has no values.

An expression holds references rather than arrays, so an equation's free
dimensions and its sense are read off the relation rather than declared
beside it. `sense` is `"min"` or `"max"`, set once here.

| Member | Returns |
| --- | --- |
| `set(name)` | a declared `Set`, whose members arrive with the data |
| `alias(name, base)` | a declared `Alias` over one of this definition's sets |
| `param(name, sets)` | a declared `Param`, whose values arrive with the data |
| `var(name, sets, subset=None, lower=0.0, upper=inf, integer=False)` | a declared `Variable` |
| `constraint(name, relation, where=None, over=None)` | nothing; registers the constraint |
| `build(data)` | a `Model` over the declarations, bound to `data` |
| `explain()` | an `Explanation` of what is declared |
| `to_yaml(instructions=False)` | the text of this definition's file, structure and no data; `instructions=True` adds the comment block that describes the format |
| `set_objective(expression)` | nothing; sets the objective |
| `sense` | `"min"` or `"max"`, as declared |
| `sets`, `aliases`, `parameters`, `variables`, `constraints` | the registries, keyed by name |

```python
from nimopt import Definition, Sum

d = Definition("dispatch", sense="min")
snapshot = d.set("snapshot")
generator = d.set("generator")
p_max = d.param("p_max", (generator,))
load = d.param("load", (snapshot,))
cost = d.param("cost", (generator,))
p = d.var("p", (snapshot, generator), lower=0.0, upper=p_max)
d.constraint("balance", Sum(generator, p[snapshot, generator]) == load[snapshot])
d.set_objective(Sum(snapshot, generator, cost[generator] * p[snapshot, generator]))

print(list(d.sets), list(d.parameters))
print(d.constraints["balance"][0].expression.frame)
print(list(d.variables), d)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
['snapshot', 'generator'] ['p_max', 'load', 'cost']
('snapshot',)
['p'] Definition('dispatch', 1 variables, 1 constraints)
```

</details>
<!-- /output -->

## One namespace for sets and parameters

Sets and parameters share one key space, because the data a definition is
built from is keyed by declared name and one would otherwise shadow the
other. Declaring a parameter under a set's name raises `ValueError`.

```python raises=ValueError
from nimopt import Definition

d = Definition("d")
S = d.set("S")
d.param("S", (S,))
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: parameter 'S' is already declared as a set; declare another name
```

</details>
<!-- /output -->

Equations are in no data mapping, so a constraint may take the name of the
parameter that bounds it.

```python
from nimopt import Definition, Sum

d = Definition("d")
S = d.set("S")
supply = d.param("supply", (S,))
one = d.param("one", (S,))
x = d.var("x", (S,))
d.constraint("supply", Sum(S, one[S] * x[S]) <= supply[S])

print(list(d.parameters), list(d.constraints))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
['supply', 'one'] ['supply']
```

</details>
<!-- /output -->

## An alias in a definition

`alias(name, base)` declares a second name for one of the definition's sets,
which is how a model relates a set to itself. The alias carries no data of
its own: it reads the labels its base set binds, so `build` takes members for
the set and none for the alias, and naming the alias in `data` is refused.

```python
import numpy as np
from nimopt import Definition, Sum

d = Definition("network", sense="min")
N = d.set("N")
NP = d.alias("NP", N)
limit = d.param("limit", (N, NP))
flow = d.var("flow", (N, NP), lower=0.0)
d.constraint("cap", flow[N, NP] <= limit[N, NP])
d.set_objective(Sum(N, NP, limit[N, NP] * flow[N, NP]))

m = d.build({"N": np.array(["a", "b"]), "limit": np.ones((2, 2))})
print(m.n_columns, m.n_rows)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
4 4
```

</details>
<!-- /output -->

## Domains in a definition

A `Domain` resolves labels through each set's coordinate, and a declared
set has none. `where=` and `over=` on `constraint`, and `subset=` on `var`,
therefore take a tuple of the definition's sets, meaning their full
product, or one of its parameters, whose coefficients are the coordinates.
Both forms resolve to the same domain, so a model and a definition declare
a sparse variable or an explicit row domain the same way.

## Building

`build(data)` copies the declaration graph, binds the copy, numbers the
columns and returns a `Model`. `data` maps a declared set's name to its
members and a declared parameter's name to its values. The definition is
unchanged, so it builds as many models as it is given datasets.

A parameter's values arrive dense over its product, as an array of one
value per cell, or long over its entries, as a pair of one mapping of label
columns and one value column. The long form is how a parameter with
coefficients at some coordinates and none at the rest is given, and it is
what a variable declared with `subset=` that parameter takes its members
from.

```python
import numpy as np
from nimopt import Definition, Sum

d = Definition("transport", sense="min")
P, W = d.set("P"), d.set("W")
cost = d.param("cost", (P, W))
supply = d.param("supply", (P,))
demand = d.param("demand", (W,))
flow = d.var("flow", (P, W), subset=cost, lower=0.0)
d.constraint("supply", Sum(W, cost[P, W] * flow[P, W]) <= supply[P])
d.constraint("demand", Sum(P, cost[P, W] * flow[P, W]) >= demand[W])
d.set_objective(Sum(P, W, cost[P, W] * flow[P, W]))

m = d.build(
    {
        "P": np.array(["p1", "p2"]),
        "W": np.array(["w1", "w2"]),
        "cost": (
            {"P": np.array(["p1", "p1", "p2"]), "W": np.array(["w1", "w2", "w1"])},
            np.array([1.0, 2.0, 3.0]),
        ),
        "supply": np.array([3.0, 3.0]),
        "demand": np.array([1.0, 1.0]),
    }
)

# three arcs, so three columns rather than the four the product would span
print(m.n_columns, m.n_rows)
print(m.solve().status)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
3 4
optimal
```

</details>
<!-- /output -->

Data that misses a declaration, or names something the definition never
declared, raises `ValueError` before anything is bound.

```python raises=ValueError
from nimopt import Definition

d = Definition("d")
d.set("S")
d.build({})
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: data does not cover ['S']; add an entry for each
```

</details>
<!-- /output -->
