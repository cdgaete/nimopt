---
title: Constraints
description: Add the supply and demand constraints, and see how many rows each family produces.
sidebar_position: 4
---

# Constraints

The model has two constraint families: a supply limit per plant and a
demand requirement per warehouse.

```text
Σ_w x[p,w] ≤ s[p]        for each plant p
Σ_p x[p,w] ≥ d[w]        for each warehouse w
```

Each family is one line of code and produces one row per member of its
frame.

## Relations

Comparing an expression with `<=`, `>=` or `==` produces a `Relation`: the
expression, the sense, and the right-hand side. A relation is not yet part
of the model.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))

m = Model("transport")
x = m.var("x", (P, W))

rule = Sum(W, x[P, W]) <= supply[P]
print(type(rule).__name__, rule.sense)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
Relation <=
```

</details>
<!-- /output -->

## Adding a constraint

`m.constraint(name, relation)` adds the relation to the model under a name and
returns the `Constraint`. The name identifies the constraint's rows in the
matrix and its dual values in the solution. A constraint produces one row
per member of its expression's frame.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))

m = Model("transport")
x = m.var("x", (P, W))

supply_rows = m.constraint("supply", Sum(W, x[P, W]) <= supply[P])
demand_rows = m.constraint("demand", Sum(P, x[P, W]) >= demand[W])

print(supply_rows.n_rows, supply_rows.nnz)
print(demand_rows.n_rows, demand_rows.nnz)
print(m.n_rows, m.nnz)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
2 6
3 6
5 12
```

</details>
<!-- /output -->

The supply expression has frame `(P,)` and produces two rows; the demand
expression has frame `(W,)` and produces three. Each supply row has three
nonzeros, one per route out of its plant, and each demand row two, one per
route into its warehouse: twelve nonzeros in total.

## The right-hand side

The right-hand side is a scalar, applied to every row, or a parameter read
at exactly the frame of the constraint, giving each row its own value. The
parameter is read at its sets here as it is anywhere else: `supply[P]`, not
`supply`. A parameter without a bracket raises `TypeError` and reports the
reading it requires.

Supply is indexed over `P`, and so are the supply rows. A parameter read over
any other index set raises `ValueError`, and the message gives both index
sets.

```python raises=TypeError
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))

m = Model("transport")
x = m.var("x", (P, W))

m.constraint("supply", Sum(W, x[P, W]) <= supply)
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: parameter 'supply' is over ('P',) and expresses no coefficient until it is read; read it at its sets as supply[P]
```

</details>
<!-- /output -->

```python raises=ValueError
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))

m = Model("transport")
x = m.var("x", (P, W))

m.constraint("supply", Sum(W, x[P, W]) <= demand[W])
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: constraint 'supply' has free dimensions ('P',); its right-hand side 'demand' is over ('W',)
```

</details>
<!-- /output -->

## One bound per constraint

Python evaluates the chained comparison `0 <= expr <= 10` as
`(0 <= expr) and (expr <= 10)` and discards the first relation. `nimopt`
raises `TypeError` on the chained form and drops no bound. Each bound is
written as its own constraint.

```python raises=TypeError
import numpy as np
from nimopt import Model, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

m = Model("transport")
x = m.var("x", (P, W))

0.0 <= Sum(W, x[P, W]) <= 10.0
```

<!-- output -->
<details open>
<summary>Raises TypeError</summary>

```text
TypeError: a relation has no truth value; write each bound in its own constraint
```

</details>
<!-- /output -->

Next: [Solving](/tutorial/solving).
