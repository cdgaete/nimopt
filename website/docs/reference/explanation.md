---
title: Explanation
description: The shape of what is declared and, where it is built, of what was built from it.
---

# Explanation

## `Explanation`

Returned by `Definition.explain` and `Model.explain`. A frozen record of
every declaration and what it built. The shapes it is made of are frozen
too. A reader takes a field and parses no rendered text.

| Field | Contains |
| --- | --- |
| `name`, `sense` | the name of the declaration and the direction it optimizes |
| `built` | whether counts are facts about data or absent |
| `sets` | one `SetShape` per dimension |
| `parameters` | one `ParamShape` per parameter |
| `variables` | one `VariableShape` per variable |
| `constraints` | one `ConstraintShape` per equation |
| `piecewise` | one `PiecewiseShape` per piecewise declaration |
| `objective` | the objective expression, or `None` |
| `columns`, `rows`, `nonzeros` | the model's shape, or `None` |
| `singleton_rows` | the number of [singleton rows](/reference/model#a-singleton-row), or `None` |

A count is `None` where nothing is bound. It is never zero. A count of zero
is a value a caller acts on, and a declaration with no data reports no
count.

| Shape | Fields |
| --- | --- |
| `SetShape` | `name`, `size` |
| `ParamShape` | `name`, `dims`, `entries` |
| `VariableShape` | `name`, `dims`, `members`, `columns`, `lower`, `upper`, `integer` |
| `ConstraintShape` | `name`, `free`, `sense`, `rows`, `nonzeros`, `relation`, `singleton_rows` |
| `PiecewiseShape` | `name`, `free`, `method`, `sign`, `breakpoints`, `generated` |

`VariableShape.members` identifies the parameter a sparse variable took its
members from, and is `None` for one over the full product. Columns are absent
until data binds. Without that field a sparse declaration and a dense one
read identically.

`ConstraintShape.free` and `.sense` are read off the relation. Neither is
declared beside it. An expression contains references and reports both.

`Explanation.singleton_rows` and `ConstraintShape.singleton_rows` count the
[singleton rows](/reference/model#a-singleton-row). `rows` and `nonzeros`
report the declared model, and a singleton row is counted in all three
fields. A constraint with none reports zero. The rendered text shows the
count where it is above zero.

```python
from nimopt import Definition, Sum

d = Definition("transport", sense="min")
P, W = d.set("P"), d.set("W")
cost = d.param("cost", (P, W))
supply = d.param("supply", (P,))
flow = d.var("flow", (P, W), subset=cost, lower=0.0)
d.constraint("supply", Sum(W, cost[P, W] * flow[P, W]) <= supply[P])
d.set_objective(Sum(P, W, cost[P, W] * flow[P, W]))

e = d.explain()
print(e.built, e.columns, e.variables[0].members)
print(e.constraints[0].free, e.constraints[0].sense)
print(e)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
False None cost
('P',) <=
transport  min  not built
  sets        P · W
  parameters  cost (P,W) · supply (P)
  variables   flow (P×W) over cost [0.0, inf]
  constraint  supply (P)  Sum(W, cost[P, W] * flow[P, W]) <= supply[P]
  objective   min  Sum(P, W, cost[P, W] * flow[P, W])
```

</details>
<!-- /output -->

`PiecewiseShape.generated` lists the variables and constraints a model
generated for the declaration. A definition generates none, and the tuple is
empty.
