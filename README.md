# nimopt

The documentation site is at <https://cdgaete.github.io/nimopt/>.

`nimopt` is a Python library for building linear and mixed-integer programs. A model is declared symbolically over named index sets, in the form of parameters, variables and constraints. The declaration is expanded into a coefficient matrix at assembly or at solve. Solutions are returned as arrays over the same index sets. A primal value is read by label, not by column position.

`nimopt` is built on `nimblend`, a labeled sparse N-dimensional array library. `nimblend` contains no optimization vocabulary and does not import `nimopt`. It is documented in [its own section](website/docs/nimblend/index.md).

## Design

**A variable is a dimension.** A constraint is an array indexed over its free sets crossed with the model's column space, with the coefficients as values. There is no assembly step converting the model into a matrix: the array is the matrix.

**Absence is distinct from zero.** An entry is either stored or absent, and every array declares what absence means: `"empty"` for a coordinate that contributes nothing, `"unknown"` for one that was never modeled. A missing result is never counted as zero. Division by an absent value raises an error; it does not return an infinity.

**A subset determines the columns.** A variable declared over a subset of a set product has one column per member of the subset and none for the rest. The full product is never materialised, at declaration or after it.

**Expressions are symbolic.** An expression contains references to variables and parameters, not their values. `cost[P, W] * x[P, W]` is the same expression over a million routes and over six. The values are read when the matrix is built.

**Dropped rows are reported.** A row whose terms have no value at some coordinate is dropped; it is not written incompletely. `absent()` lists every dropped row with the rule that dropped it. `row()` returns one row of the assembled matrix in the form passed to the solver.

## Install

```bash
pip install "nimopt[highs]"
```

The extra installs `nimopt` and its dependency `nimblend` from PyPI, with HiGHS as the solver backend.

HiGHS is the default solver, and `[highs]` installs it. `[gurobi]` and `[mosek]` add those adapters instead, `[bench]` adds the comparison suite and `[dev]` the test and lint tooling. `available()` reports the solvers whose backend can be imported in the current environment. `capabilities(name)` reports what one adapter supports, whether or not its backend is installed.

Development installs come from a checkout. `nimblend` is a dependency and installs first, from its clone. `nimopt` then installs from its own root:

```bash
pip install /path/to/nimblend
pip install ".[highs]"
```

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

A `Definition` declares the same model without binding data. Its sets and parameters are declared by name and its constraints use the same expression syntax. `explain()` reports the whole declaration before any value is read.

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

A definition is copied before it is bound. One definition builds a model for each dataset it is given, and no build changes the definition. The built model is inspected the same way. `row()` reads one row out of the assembled matrix in the form passed to the solver.

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

Where a variable spans an arc list instead of a full product, it has one column per arc and the product is never built. A thousand plants each serving three warehouses is three thousand columns, not a million.

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
- Continuous and integer columns, solved through HiGHS, Gurobi or Mosek behind one adapter contract, with `capabilities()` reporting what each adapter supports and which capabilities it rejects together
- One option vocabulary translated into each solver's own option names, with a time limit written the same way for every solver
- A model written to and read back from YAML, with its data inline or in a sidecar
- A corpus of worked models under `nimopt.models`, each with its formulation, its inputs at any size, and an objective computed by arithmetic instead of by a solver

## Performance

Where a variable's columns are a subset of a set product, not materialising the product saves memory and time. On a transport model of 400 000 arcs over a 20 000 000-cell product, `nimopt` builds the matrix in 72.9 MB of resident memory against linopy's 1 682.9 MB. The build takes 292.9 ms against 844.5 ms.

Where nothing is sparse, the alignment work costs time and saves nothing. On a fully dense temporally coupled model at 2 111 080 rows, the comparison reverses: linopy builds the matrix three times faster, for seven percent more resident memory.

The benchmark suite measures both models. The [benchmark page](website/docs/explanation/what-the-numbers-measure.md) gives each figure, the baseline it is measured against, and what it does not claim.

## Documentation

- [Get started](website/docs/get-started/index.md): installation, and the transport model above solved and read back.
- [Vocabulary](website/docs/vocabulary/index.md): the terms used throughout the documentation.
- [Tutorial](website/docs/tutorial/sets-and-parameters.md): the transport model built in six steps, one concept per page.
- Playground: every example runs in the browser and can be edited.
- [For agents](website/docs/for-agents.md): the mental model, the public surface and the failure modes on one page.

Every Python example on this site is executed by the test suite and shows the output it produced.

## License

MIT. See `LICENSE`.

## Citing

The package includes a `CITATION.cff`. Cite it by author, name and version:

> Gaete-Morales, Carlos. *nimopt* (version 0.1.2). MIT.

## Contributing

Issues and patches are welcome once the repositories are published. Until then, the most useful contribution is a model that does not fit: the formulations that are awkward to write determine the next features.
