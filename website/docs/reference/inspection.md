---
title: Row and Absence
description: What one row of a built model contains, and which coordinates were dropped from a constraint.
---

# Inspecting a built model

## `Row`

Returned by `Model.row(name, **coords)`. One row as the assembled matrix
stores it: the columns, the coefficients and the bounds that are passed to
the solver. `Model.row` computes the entries of that row alone. It builds no
matrix and no other row. `coords` gives one label per free dimension. A
constraint with no free dimension has one row, and `Model.row(name)` returns
it.

| Field | Contains |
| --- | --- |
| `constraint` | the equation this row belongs to |
| `coordinate` | the row's own coordinate, per free dimension |
| `index` | the position of the row among the declared rows of the model |
| `terms` | one `RowTerm` per coefficient |
| `sense`, `lower`, `upper` | read from the row's bounds |

| `RowTerm` field | Contains |
| --- | --- |
| `column` | the solver's own column number |
| `variable` | the variable that column belongs to |
| `coordinate` | that column's coordinate, per dimension |
| `coefficient` | the value in the matrix |

A variable occupies a contiguous range of the column space from its `start`.
A column resolves to its variable through that range, and to a coordinate
through the numbering rule of that variable.

`sense` is read from the bounds: equal bounds are `==`, an infinite lower
bound is `<=`, an infinite upper bound is `>=`.

`Row.index` is the position of the row among the declared rows of the model.
It is the same under both values of `singleton_rows`. A row stored as a bound
of its column has no row number in the solver's matrix, and its position
follows the declared rows before it. A row appended in a session follows
every declared row of the model.

`diagnose()` reports a row stored as a column bound as a `Row` as well,
where the solver reports that bound in the conflict. A column of the conflict
is reported as a `ColumnBound` unless every bound of it in the subsystem is
set by a declared row. A bound the model never declared on the variable, such
as a column fixed in a session outside its bounds, is reported as a
`ColumnBound`, beside the row that sets the other side. The `ColumnBound`
reports the bounds the solver has.

The arrays of a loaded matrix are indexed through `Assembled.row_of(name)`,
not through `Row.index`: a row stored as a column bound occupies no entry of
them.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))
cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))

m = Model("transport")
x = m.var("x", (P, W))
m.constraint("supply", Sum(W, cost[P, W] * x[P, W]) <= supply[P])

print(m.row("supply", P="porto"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
supply[P='porto']  row 1
  3·x[porto,berlin] + 1·x[porto,paris] + 6·x[porto,rome] <= 25
```

</details>
<!-- /output -->

A coordinate at which the constraint has no row raises `ValueError`; the
message points to the function that reports why it is missing. `row` and
`absent` raise `KeyError` for a name that is not a declared constraint, and
the message lists the declared constraints. `row` raises `ValueError` for a
label that is not a member of its dimension, and the message identifies the
dimension.

```python raises=ValueError
import numpy as np
from nimopt import Model, Param, Set

P = Set("P", np.array(["p1", "p2", "p3"]))
m = Model("m")
x = m.var("x", (P,), upper=5.0)
one = Param.from_dense("one", (P,), np.ones(3))
rhs = Param.from_long("rhs", (P,), {"P": np.array(["p1", "p2"])}, np.ones(2))
m.constraint("cap", one[P] * x[P] <= rhs[P])

m.row("cap", P="p3")
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: constraint 'cap' has no row at {'P': 'p3'}; read `absent('cap')` for the rule that dropped it
```

</details>
<!-- /output -->

## A status that depends on the matrix

A model that is infeasible and whose objective is also unbounded along a ray
can be reported as `infeasible` or as `unbounded`, and the form of the matrix
can change which one. Gurobi returned `unbounded` under one value of
`singleton_rows` and `infeasible` under the other, on one such model. HiGHS
returned `infeasible` under both values. Both statuses are correct for that
model, and a caller reads `status` and checks the other property separately.

## `Absence`

Returned by `Model.absent(name)`. What a constraint set out to produce,
what it produced, and which coordinates were dropped.

| Field | Contains |
| --- | --- |
| `constraint` | the equation this is about |
| `stated_by` | `"terms"` where the rows are derived, `"over"` where given explicitly |
| `expected`, `standing` | rows expected, rows kept |
| `dropped_rows` | one `DroppedRow(coordinate, rule, detail)` per row lost |
| `dropped_terms` | one `DroppedTerm(coordinate, variable, rule, detail)` per term lost |

`expected - len(dropped_rows) == standing`.

| `dropped_rows` rule | Meaning |
| --- | --- |
| `term-does-not-reach` | a term has no value at that coordinate; the row would express a constraint that was not written |
| `where` | the condition excludes it |
| `absent-rhs` | the right-hand side has no value there |

| `dropped_terms` rule | Meaning |
| --- | --- |
| `absent-coefficient` | a coefficient absent inside a sum; the row is kept with one term fewer |

The two rules differ in what they remove. A coefficient absent inside a sum
removes a **term** and keeps the row. A term absent along a **free**
dimension removes the **row**.

Under `over=` the rows are given explicitly. Nothing is dropped, and a
right-hand side that omits one raises instead. An empty `dropped_rows` beside
`stated_by="over"` follows from that rule.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["p1", "p2"]))
W = Set("W", np.array(["w1", "w2", "w3"]))
m = Model("t")
flow = m.var("flow", (P, W))
cost = Param.from_long(
    "cost",
    (P, W),
    {"P": np.array(["p1", "p1", "p2"]), "W": np.array(["w1", "w2", "w1"])},
    np.array([1.0, 2.0, 3.0]),
)
supply = Param.from_dense("supply", (P,), np.array([3.0, 3.0]))
m.constraint("supply", Sum(W, cost[P, W] * flow[P, W]) <= supply[P])

print(m.absent("supply"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
supply  2 of 2 rows  stated by terms
  term absent P='p1', W='w3'  flow  absent-coefficient (cost)
  term absent P='p2', W='w2'  flow  absent-coefficient (cost)
  term absent P='p2', W='w3'  flow  absent-coefficient (cost)
```

</details>
<!-- /output -->
