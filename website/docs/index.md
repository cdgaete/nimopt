---
title: nimopt
slug: /
description: An LP/MILP builder in which a variable is a dimension. Models are declared over named index sets and solutions are returned over the same sets.
---

# nimopt

`nimopt` is a Python library for building linear and mixed-integer programs. A model is declared symbolically over named index sets — as parameters, variables and constraints — and is expanded into a coefficient matrix only when it is assembled or solved. Solutions are returned as arrays over those same index sets, so a primal value is read by label rather than by column position.

`nimopt` is built on `nimblend`, a labelled sparse N-dimensional array library with no knowledge of optimisation. The dependency runs in one direction, and `nimblend` is documented in [its own section](/nimblend).

## Design

**A variable is a dimension.** A constraint is an array indexed over its free sets crossed with the model's column space, with the coefficients as values. There is no assembly step converting the model into a matrix, because the array is the matrix.

**Absence is distinct from zero.** An entry is either stored or absent, and every array declares what absence means: `"empty"` for a coordinate that contributes nothing, `"unknown"` for one that was never modelled. A missing result is never counted as zero, and division by an absent value raises rather than producing an infinity.

**A subset stays a subset.** A variable declared over a subset of a set product has one column per member of the subset and none for the rest. The full product is never materialised, at declaration or at any point after it.

**Expressions are symbolic.** An expression holds references to variables and parameters rather than their values. `cost[P, W] * x[P, W]` costs the same to write over a million routes as over six; the values are read when the matrix is built.

**Dropped rows are reported.** A row whose terms have no value at some coordinate is dropped rather than written incompletely. `absent()` lists every dropped row with the rule that dropped it, and `row()` returns one row of the assembled matrix as the solver receives it.

## Install

Neither package is published yet, so both are installed from a checkout. `nimblend` is a dependency and is installed first, from wherever it is cloned; `nimopt` then installs from its own root:

```bash
pip install /path/to/nimblend
pip install ".[highs]"
```

HiGHS is the default solver, and `[highs]` installs it. `[gurobi]` and `[mosek]` add those adapters instead, `[bench]` adds the comparison suite and `[dev]` the test and lint tooling. `available()` reports the solvers whose backend can be imported in the current environment, and `capabilities(name)` answers for an adapter whether or not its backend is installed.

## A first model

A transport problem: two plants with limited supply ship to three warehouses with fixed demand, and the objective is total shipping cost.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

P = Set("P", np.array(["lisbon", "porto"]))
W = Set("W", np.array(["berlin", "paris", "rome"]))

cost = Param.from_dense("cost", (P, W), np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]))
supply = Param.from_dense("supply", (P,), np.array([30.0, 25.0]))
demand = Param.from_dense("demand", (W,), np.array([20.0, 15.0, 15.0]))

m = Model("transport")
x = m.var("x", (P, W))

m.constraint("supply", Sum(W, x[P, W]) <= supply[P])
m.constraint("demand", Sum(P, x[P, W]) >= demand[W])
m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))

solution = m.solve()
print(solution.status, solution.objective)
print(solution.primal("x").to_dense())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
optimal 135.0
[[20.  0. 10.]
 [ 0. 15.  5.]]
```

</details>
<!-- /output -->

The primal values are returned as a 2 by 3 array over plants and warehouses, in the order the sets declare their members.

## Declaring before the data exists

A `Definition` states the same model without binding any data. Its sets and parameters are declared by name, its constraints are written in the same expression syntax, and `explain()` reports the whole declaration before a single value has been read.

```python
import numpy as np
from nimopt import Definition, Sum

d = Definition("transport")
P, W = d.set("P"), d.set("W")
cost = d.param("cost", (P, W))
supply = d.param("supply", (P,))
demand = d.param("demand", (W,))
x = d.var("x", (P, W))

d.constraint("supply", Sum(W, x[P, W]) <= supply[P])
d.constraint("demand", Sum(P, x[P, W]) >= demand[W])
d.set_objective(Sum(P, W, cost[P, W] * x[P, W]))

print(d.explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
transport  min  not built
  sets        P · W
  parameters  cost (P,W) · supply (P) · demand (W)
  variables   x (P×W) [0.0, inf]
  constraint  supply (P)  Sum(W, x[P, W]) <= supply[P]
  constraint  demand (W)  Sum(P, x[P, W]) >= demand[W]
  objective   min  Sum(P, W, cost[P, W] * x[P, W])
```

</details>
<!-- /output -->

A definition is copied before it is bound, so one definition builds as many models as it is given datasets for and is unchanged by any of them. The built model is inspected the same way, and `row()` reads one row back out of the assembled matrix as the solver receives it.

```python
import numpy as np
from nimopt import Definition, Sum

d = Definition("transport")
P, W = d.set("P"), d.set("W")
cost = d.param("cost", (P, W))
supply = d.param("supply", (P,))
demand = d.param("demand", (W,))
x = d.var("x", (P, W))
d.constraint("supply", Sum(W, x[P, W]) <= supply[P])
d.constraint("demand", Sum(P, x[P, W]) >= demand[W])
d.set_objective(Sum(P, W, cost[P, W] * x[P, W]))

data = {
    "P": np.array(["lisbon", "porto"]),
    "W": np.array(["berlin", "paris", "rome"]),
    "cost": np.array([[2.0, 4.0, 5.0], [3.0, 1.0, 6.0]]),
    "supply": np.array([30.0, 25.0]),
    "demand": np.array([20.0, 15.0, 15.0]),
}
m = d.build(data)
print(m)
print(m.row("demand", W="paris"))
print(m.absent("demand"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
Model('transport', 1 variables, 6 columns, 5 rows)
demand[W='paris']  row 3
  1·x[lisbon,paris] + 1·x[porto,paris] >= 15
demand  3 of 3 rows  stated by terms
```

</details>
<!-- /output -->

## A variable over a subset

Where a variable spans an arc list rather than a full product, it has a column per arc and the product is never built. A thousand plants each serving three warehouses is three thousand columns, not a million.

```python
import numpy as np
from nimopt import Model, Set, subset

P = Set("P", np.array([f"p{i}" for i in range(1000)]))
W = Set("W", np.array([f"w{i}" for i in range(1000)]))

served = np.array([f"w{(i * 7 + k) % 1000}" for i in range(1000) for k in range(3)])
arcs = subset((P, W), {"P": np.repeat(P.labels, 3), "W": served})

m = Model("transport")
x = m.var("x", (P, W), subset=arcs)
print(f"{m.n_columns} columns over a product of {len(P) * len(W)}")
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
3000 columns over a product of 1000000
```

</details>
<!-- /output -->

## Features

- Sets, aliases, subsets and set products as the index structure of every declaration
- Parameters from dense arrays or long-form columns, broadcast where a parameter is narrower than the variable it multiplies
- Composable coefficients: a parameter read at its sets, or an arithmetic of parameters written before data exists
- Conditions on a sum and on a constraint, lags that drop or wrap at the ends of a set, and members fixed at a label
- Per-column bounds from a parameter, and variables declared over a subset of a set product
- A `Definition` written before data exists and built against any number of datasets
- `explain()` on a definition or a built model, `row()` into the assembled matrix, and `absent()` reporting dropped rows and the rule that dropped each
- A `Session` that keeps the solved instance open, and `diagnose()` reporting the conflicting rows of an infeasible model or the ray of an unbounded one
- Primals and duals returned over their index sets, with absence distinct from zero
- Continuous and integer columns, solved through HiGHS, Gurobi or Mosek behind one adapter contract, with `capabilities()` stating what each adapter does and which capabilities it refuses together
- One option vocabulary translated into each solver's own spelling, so a time limit is stated the same way whatever solves the model
- A model written to and read back from YAML, with its data inline or in a sidecar
- A corpus of worked models under `nimopt.models`, each stating its formulation, inputs at any size, and an objective computed by arithmetic rather than by a solver

## Performance

Where a variable's columns are a subset of a set product, not materialising the product is worth a great deal. On a transport model of 400 000 arcs over a 20 000 000-cell product, `nimopt` builds the same matrix in 72.9 MB of resident memory against a dense rival's 1 682.9 MB, and in 292.9 ms against 844.5 ms.

Where nothing is sparse, the alignment work is a cost with no corresponding saving. On a fully dense temporally-coupled model at 2 111 080 rows, the same comparison reverses: the rival builds the matrix three times faster, for seven per cent more resident memory.

Both rows are in the suite for the same reason. The [benchmark page](/explanation/what-the-numbers-measure) gives each figure, the baseline it is measured against, and what it does not claim.

## Documentation

- [Get started](/get-started): installation, and the transport model above solved and read back.
- [Vocabulary](/vocabulary): the terms used throughout the documentation.
- [Tutorial](/tutorial/sets-and-parameters): the transport model built in six steps, one concept per page.
- [Playground](/playground): every example runs in the browser and can be edited.
- [For agents](/for-agents): the mental model, the public surface and the failure modes on one page.

Every Python example on this site is executed by the test suite and shows the output it produced.

## Licence

MIT. See `LICENSE`.

## Citing

The package carries a `CITATION.cff`. Cite it by author, name and version:

> Gaete-Morales, Carlos. *nimopt* (version 0.1.2). MIT.

## Contributing

Issues and patches are welcome once the repositories are published. Until then, the most useful contribution is a model that does not fit: the formulations that are awkward to write determine the next features.
