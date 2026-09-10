---
title: Commitment
description: Unit commitment with a binary on-off column per generator and snapshot.
---

# Commitment

`nimopt.models.commitment` is unit commitment. A committed unit runs between
its minimum and its maximum output and incurs a no-load cost. An uncommitted
unit produces nothing. The `capacity` and `minimum` rows are written against
the binary column. This is the only MILP in the corpus.

```text
minimize    Σ_{t,g} cost[g] · gen[t,g] + Σ_{t,g} no_load[g] · on[t,g]
subject to  gen[t,g] − p_max[g] · on[t,g] ≤ 0
            gen[t,g] − p_min[g] · on[t,g] ≥ 0
            Σ_g gen[t,g] == load[t]           for each snapshot t
            on[t,g] ∈ {0, 1},  gen[t,g] ≥ 0
```

```python
from nimopt.models import commitment

print(commitment.definition().explain())
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
commitment  min  not built
  sets        T · G
  parameters  p_max (G) · p_min (G) · cost (G) · no_load (G) · load (T)
  variables   on (T×G) [0.0, 1.0] integer · gen (T×G) [0.0, inf]
  constraint  capacity (T,G)  gen[T, G] - p_max[G] * on[T, G] <= 0
  constraint  minimum (T,G)  gen[T, G] - p_min[G] * on[T, G] >= 0
  constraint  balance (T)  Sum(G, gen[T, G]) == load[T]
  objective   min  Sum(T, G, cost[G] * gen[T, G]) + Sum(T, G, no_load[G] * on[T, G])
```

</details>
<!-- /output -->

Snapshots are uncoupled. `reference` enumerates every on-off subset per
snapshot and takes the cheapest feasible one. The optimum is computed without
a solver.

```python
from nimopt.models import commitment

inputs = commitment.data()
model = commitment.definition().build(inputs)
solution = model.solve()
print(int(model.integrality().sum()), "binary columns of", model.n_columns)
print(solution.objective, commitment.reference(inputs))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
12 binary columns of 24
13800.0 13800.0
```

</details>
<!-- /output -->

Both unit rows are produced for every generator and snapshot.

```python
from nimopt.models import commitment

model = commitment.definition().build(commitment.data())
print(model.absent("capacity"))
```

<!-- output -->
<details open>
<summary>Output</summary>

```text
capacity  12 of 12 rows  stated by terms
```

</details>
<!-- /output -->
