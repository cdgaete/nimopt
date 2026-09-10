---
title: Solvers
description: What each solver adapter can do, the options a caller can set, and a session that keeps one solver open.
---

# Solvers

## `available` and `capabilities`

`available()` lists every adapter whose backend can be imported in the
current environment, with the capabilities each declares. `capabilities(name)`
reports for an adapter whether or not its backend is installed. A descriptor
describes the adapter as shipped. A caller reads one to choose what to
install.

A descriptor describes the adapter, not the library behind it: a solver
feature the adapter does not call is `absent`.

What `available()` lists depends on the machine; what `capabilities(name)`
reports does not.

```python
from nimopt import available, capabilities

print("highs" in available())
print(capabilities("highs"))
print(capabilities("gurobi"))
print(capabilities("mosek"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
True
highs  integrality native · duals native · conflict native · ray native  rejects duals+integrality
gurobi  integrality native · duals native · conflict native · ray native  rejects duals+integrality
mosek  integrality native · duals native · conflict absent · ray native  rejects duals+integrality
```

</details>
<!-- /output -->

## `Capabilities`

| Member | Returns |
| --- | --- |
| `solver` | the adapter's name |
| `support` | one of `"native"` or `"absent"` per capability |
| `rejected` | the pairs this adapter rejects together |
| `supports(capability)` | whether the adapter handles it at all |
| `rejects(one, other)` | whether it rejects the two together |

The capabilities are `integrality`, `duals`, `conflict` and `ray`. A flat
set is insufficient: a solver can support two and reject their
combination. Every adapter rejects `integrality` with `duals`. A
mixed-integer model's duals are not the relaxation's. A model with integer
columns has no duals at all, and `Solution.dual` raises.

```python raises=ValueError
import numpy as np
from nimopt import Model, Param, Set, Sum

T = Set("T", np.arange(2))
one = Param.from_dense("one", (T,), np.ones(2))

m = Model("m")
x = m.var("x", (T,), upper=3.0, integer=True)
m.constraint("cap", one[T] * x[T] <= 2.0)
m.set_objective(Sum(T, one[T] * x[T]))

m.solve().dual("cap")
```

<!-- output -->
<details open>
<summary>Raises ValueError</summary>

```text
ValueError: model 'm' has integer columns and 'highs' reports no duals for it; read primal values only
```

</details>
<!-- /output -->

## `Session`

Returned by `Model.session(solver="highs", options=None)`. The model of one
solver, opened on one assembled model and kept open. A solve passes the
matrix to the solver, and a later query reads the same solved instance.

| Member | Returns |
| --- | --- |
| `assembled` | the matrix the session was opened on |
| `solver`, `capabilities` | which adapter, and what it can do |
| `status` | what the last solve reported, or `None` before one |
| `solve()` | a `Solution` |
| `close()` | releases the solver's model |

`Model.solve()` opens a session, solves and closes it. A caller who needs
only a solution needs no session.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

SNAP = Set("snapshot", np.arange(3))
GEN = Set("generator", np.array(["wind", "gas"]))
p_max = Param.from_dense("p_max", (GEN,), np.array([10.0, 20.0]))
load = Param.from_dense("load", (SNAP,), np.array([25.0, 20.0, 5.0]))
cost = Param.from_dense("cost", (GEN,), np.array([1.0, 5.0]))

m = Model("dispatch", sense="min")
p = m.var("p", (SNAP, GEN), lower=0.0, upper=p_max)
m.constraint("balance", Sum(GEN, p[SNAP, GEN]) == load[SNAP])
m.set_objective(Sum(SNAP, GEN, cost[GEN] * p[SNAP, GEN]))

with m.session() as session:
    solution = session.solve()
    print(solution.status, solution.objective)
    print(session.status)
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
optimal 150.0
optimal
```

</details>
<!-- /output -->

## `Diagnosis`

Returned by `Session.diagnose()`. Why a model did not solve, as the rows
and columns that explain it.

| Member | Returns |
| --- | --- |
| `status`, `solver` | what the solve reported, and which adapter |
| `method` | `"native"` where the solver computed the conflict |
| `conflict` | one `Row` per conflicting row, or `None` where the model is not infeasible |
| `columns` | one `ColumnBound` per column of the conflict, with the bounds the model declares |
| `ray` | one `RayTerm` per column the ray moves, or `None` where the model is not unbounded |

`conflict` contains the same `Row` that `Model.row` returns, and a
conflicting row reads in one format. Two solvers may return different
irreducible sets. Removing either set makes the model feasible. The two
sets need not be equal.

A conflict is computed on the solved backend, and the session keeps that
backend available.

```python
import numpy as np
from nimopt import Model, Param, Set, Sum

SNAP = Set("snapshot", np.arange(3))
GEN = Set("generator", np.array(["wind", "gas"]))
p_max = Param.from_dense("p_max", (GEN,), np.array([10.0, 20.0]))
load = Param.from_dense("load", (SNAP,), np.array([25.0, 100.0, 5.0]))
cost = Param.from_dense("cost", (GEN,), np.array([1.0, 5.0]))

m = Model("dispatch", sense="min")
p = m.var("p", (SNAP, GEN), lower=0.0, upper=p_max)
m.constraint("balance", Sum(GEN, p[SNAP, GEN]) == load[SNAP])
m.set_objective(Sum(SNAP, GEN, cost[GEN] * p[SNAP, GEN]))

with m.session() as session:
    print(session.solve().status)
    print(session.diagnose())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
infeasible
infeasible  highs  conflict native
balance[snapshot=1]  row 1
  1·p[1,wind] + 1·p[1,gas] == 100
  bound  p[snapshot=1, generator='wind']  [0, 10]
  bound  p[snapshot=1, generator='gas']  [0, 20]
```

</details>
<!-- /output -->

HiGHS computes its conflict over the model's linear relaxation. A model
that is feasible as an LP and infeasible only through its integrality
therefore yields no conflict, and the adapter raises. It reports no row it
did not prove. Gurobi's conflict covers the integrality. Mosek's adapter
computes no conflict. A session on it raises, and the message refers to
`capabilities("mosek")`.

## `options` and `Option`

`options()` lists every option a caller can set, under the names of
`nimopt`. An option outside the list raises `ValueError`. No option is passed
to a solver that ignores it, and an unrecognized name stops the solve.

`options(solver)` lists the same options with the name and the values of that
solver. A caller follows those names into the documentation of the solver.

```python
from nimopt import options

for option in options("highs"):
    print(f"{option.name:<16} {option.native}")
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
time_limit       time_limit
iteration_limit  simplex_iteration_limit
node_limit       mip_max_nodes
mip_gap          mip_rel_gap
mip_abs_gap      mip_abs_gap
feasibility_tol  primal_feasibility_tolerance
optimality_tol   dual_feasibility_tolerance
threads          threads
seed             random_seed
log              output_flag
presolve         presolve
method           solver
newton_system    hipo_system
crossover        run_crossover
pdlp_tol         pdlp_optimality_tolerance
```

</details>
<!-- /output -->

An `Option` has a `name`, the `kind` it takes, what it `does`, and its
`choices` where it takes one of a set. Read for a solver it also has
`native` and `native_choices`.

<!-- options -->
| Option | Takes | Does | `highs` | `gurobi` | `mosek` |
| --- | --- | --- | --- | --- | --- |
| `time_limit` | float | seconds the solver may run for | `time_limit` | `TimeLimit` | `optimizer_max_time` |
| `iteration_limit` | int | simplex iterations the solver may take | `simplex_iteration_limit` | `IterationLimit` | `sim_max_iterations` |
| `node_limit` | int | branch-and-bound nodes the solver may explore | `mip_max_nodes` | `NodeLimit` | `mio_max_num_branches` |
| `mip_gap` | float | relative gap at which a mixed-integer solve stops | `mip_rel_gap` | `MIPGap` | `mio_tol_rel_gap` |
| `mip_abs_gap` | float | absolute gap at which a mixed-integer solve stops | `mip_abs_gap` | `MIPGapAbs` | `mio_tol_abs_gap` |
| `feasibility_tol` | float | how far a primal solution may miss a row | `primal_feasibility_tolerance` | `FeasibilityTol` | `basis_tol_x` |
| `optimality_tol` | float | how far a dual solution may miss a bound | `dual_feasibility_tolerance` | `OptimalityTol` | `basis_tol_s` |
| `threads` | int | threads the solver may use; 0 leaves it the choice | `threads` | `Threads` | `num_threads` |
| `seed` | int | the seed the solver randomizes from | `random_seed` | `Seed` | `mio_seed` |
| `log` | bool | whether the solver writes its own iteration log | `output_flag` | `OutputFlag` | `log` |
| `presolve` | `off` / `choose` / `on` | how hard the solver presolves | `presolve` | `Presolve` | `presolve_use` |
| `method` | `choose` / `simplex` / `barrier` / `hipo` / `pdlp` | the algorithm the solver runs | `solver` | `Method` | `optimizer` |
| `newton_system` | `choose` / `augmented` / `normaleq` | the Newton system an interior point method factorizes | `hipo_system` | not supported | not supported |
| `crossover` | `choose` / `off` / `on` | whether an interior point is moved to a vertex after the solve | `run_crossover` | `Crossover` | `intpnt_basis` |
| `pdlp_tol` | float | relative tolerance at which the first-order method stops | `pdlp_optimality_tolerance` | not supported | not supported |
<!-- /options -->

A choice each solver writes differently is given once and translated. The
value a caller writes has one meaning for every solver. No solver
supports every option or every choice: `newton_system` and `pdlp_tol` are
HiGHS's, as are `hipo` and `pdlp` under `method`. Asking Gurobi or Mosek
for one of them raises, and the message identifies it. No solver runs a
different algorithm in its place. Mosek runs only its mixed-integer
optimizer on a model with integer columns. `method` is `choose` there, and
any other choice raises. The guide on [interior point and first-order
methods](/guides/highs-methods) gives the memory each method requires and the
installation of a HiGHS with HiPO and a GPU.

## Progress reporting

`build()`, `assemble()`, `session()` and `solve()` take `progress=`.
`progress=True` draws a report in a terminal and nothing where output is
redirected. A reporter given by the caller is used as passed, and a notebook
or another interface writes through it. A reporter implements
`start(total, what)`, `step(done, what)` and `done()`. That is the whole
contract.

The report covers building. Its resolution is the structure of the model. The
measuring pass counts constraints and the writing pass counts nonzeros. A
model with one constraint of one term reports one step and no fraction.

`log=True` requests the log of the solver. Building finishes before a solver
starts, and the report and the log do not interleave.
