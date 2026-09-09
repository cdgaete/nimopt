---
title: Explanation
description: The shape of what is declared and, where it is built, of what was built from it.
---

# Explanation

## `Explanation`

Returned by `Definition.explain` and `Model.explain`. A frozen record of
every declaration and what it built. The shapes it is made of are frozen
too, so a reader takes a field rather than parsing a rendering.

| Field | Holds |
| --- | --- |
| `name`, `sense` | the declaration's name and the direction it optimises |
| `built` | whether counts are facts about data or absent |
| `sets` | one `SetShape` per dimension |
| `parameters` | one `ParamShape` per parameter |
| `variables` | one `VariableShape` per variable |
| `constraints` | one `ConstraintShape` per equation |
| `objective` | the objective's spelling, or `None` |
| `columns`, `rows`, `nonzeros` | the model's shape, or `None` |

A count is `None` where nothing is bound. It is never zero: a count of zero
is a fact a caller acts on, and reporting one for a declaration would be
false.

| Shape | Fields |
| --- | --- |
| `SetShape` | `name`, `size` |
| `ParamShape` | `name`, `dims`, `entries` |
| `VariableShape` | `name`, `dims`, `members`, `columns`, `lower`, `upper`, `integer` |
| `ConstraintShape` | `name`, `free`, `sense`, `rows`, `nonzeros`, `relation` |

`VariableShape.members` names the parameter a sparse variable took its
members from, and is `None` for one over the full product. Columns are
absent until data binds, so without it a sparse declaration and a dense one
would otherwise read identically.

`ConstraintShape.free` and `.sense` are read off the relation rather than
declared beside it, because an expression holds references and reports
both.

```python
from nimopt import Definition, Sum

d = Definition("transport", sense="min")
P, W = d.set("P"), d.set("W")
cost = d.param("cost", (P, W))
supply = d.param("supply", (P,))
flow = d.var("flow", (P, W), subset=cost, lower=0.0)
d.eq("supply", Sum(W, cost[P, W] * flow[P, W]) <= supply[P])
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
